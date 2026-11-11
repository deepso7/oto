//! DPDFNet-2 and FastEnhancer-S through ONNX Runtime (`ort`). Behind the `onnx` feature.
//!
//! Both are streaming models exported with their recurrent state as explicit inputs and
//! outputs, so each frame is one `Session::run` that also returns the state for the next frame.
//! Not allocation-free: ONNX Runtime allocates its outputs on every run, so like DeepFilterNet3
//! these run on the DSP worker thread, never in the audio callback.
//!
//! - **DPDFNet-2 (48 kHz "hr")**, Ceva, Apache-2.0 code and weights (Hugging Face model card
//!   `Ceva-IP/DPDFNet`). Spectrum in, spectrum out: we do its causal STFT here (960-point Vorbis
//!   window, 480 hop), as upstream's `StreamEnhancer` does.
//! - **FastEnhancer-S (48 kHz)**, MIT code; the weights' licence is not stated upstream and the
//!   48 kHz training data includes export-restricted corpora. Wave in, wave out (the export does
//!   its own STFT). Not cleared to ship: see docs/models.md.

use crate::Denoiser;
use ort::session::Session;
use ort::session::builder::{AutoDevicePolicy, GraphOptimizationLevel};
use ort::value::TensorRef;
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

static DPDFNET2: &[u8] = include_bytes!("../models/dpdfnet2_48khz_hr.onnx");
#[cfg(feature = "fastenhancer")]
static FASTENHANCER_S: &[u8] = include_bytes!("../models/fastenhancer_s_48khz.onnx");

/// One single-threaded CPU session: inference shares the DSP worker's core, and spinning
/// intra-op threads would burn a second one.
fn session(model: &[u8]) -> Result<Session, String> {
    // ONNX Runtime allows one environment per process: every session (each model, each reload
    // after a model switch) shares this one.
    static ENV: std::sync::OnceLock<Result<ort::environment::Environment, String>> =
        std::sync::OnceLock::new();
    let env = ENV
        .get_or_init(|| {
            ort::init()
                .with_name("oto")
                .build()
                .map_err(|e| format!("ONNX Runtime: {e}"))
        })
        .as_ref()
        .map_err(Clone::clone)?;
    // CPU only. `ort` defaults to the MaxEfficiency device policy, which on a Mac picks CoreML:
    // it rejects these graphs at run time (shape-rank mismatch) and would wake the ANE for a
    // job one core does.
    Session::builder(env)
        .and_then(|b| Ok(b.with_auto_device(AutoDevicePolicy::PreferCPU)?))
        .and_then(|b| Ok(b.with_no_environment_execution_providers()?))
        .and_then(|b| Ok(b.with_execution_providers([ort::ep::CPU::default().build()])?))
        .and_then(|b| Ok(b.with_optimization_level(GraphOptimizationLevel::All)?))
        .and_then(|b| Ok(b.with_intra_threads(1)?))
        .and_then(|b| Ok(b.with_inter_threads(1)?))
        .and_then(|b| Ok(b.with_config_entry("session.intra_op.allow_spinning", "0")?))
        .and_then(|mut b| b.commit_from_memory(model))
        .map_err(|e| format!("ONNX Runtime: {e}"))
}

fn err(e: ort::Error) -> String {
    format!("ONNX Runtime: {e}")
}

/// DPDFNet-2, 48 kHz. 10 ms frames.
pub struct DpdfNet {
    session: Session,
    state: Vec<f32>,
    init_state: Vec<f32>,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    /// The last `WIN` input samples.
    frame: Vec<f32>,
    time: Vec<f32>,
    spec: Vec<Complex32>,
    /// Interleaved re/im, `[1, 1, BINS, 2]`.
    spec_ri: Vec<f32>,
    ola: Vec<f32>,
    scratch_fwd: Vec<Complex32>,
    scratch_inv: Vec<Complex32>,
}

impl DpdfNet {
    const HOP: usize = 480;
    const WIN: usize = 960;
    const BINS: usize = Self::WIN / 2 + 1;
    /// The STFT's overlap (one hop) plus four hops of model lookahead. Measured against the
    /// input by cross-correlation (50.0 ms) and matches upstream's `ATTN_LIMIT_NOISY_FRAME_OFFSET`.
    const LATENCY: usize = Self::HOP * 5;

    /// Loads the bundled weights. Slow (hundreds of ms); off the RT thread.
    pub fn new() -> Result<Self, String> {
        let session = session(DPDFNET2)?;
        let meta = session.metadata().map_err(err)?;
        let get = |k: &str| meta.custom(k).ok_or_else(|| format!("DPDFNet: no {k} metadata"));
        let num = |k: &str| -> Result<usize, String> {
            get(k)?.parse().map_err(|_| format!("DPDFNet: bad {k} metadata"))
        };
        let floats = |k: &str| -> Result<Vec<f32>, String> {
            get(k)?
                .split(',')
                .map(|v| v.trim().parse().map_err(|_| format!("DPDFNet: bad {k} metadata")))
                .collect()
        };
        let mut init_state = vec![0.0; num("state_size")?];
        let erb = floats("erb_norm_init")?;
        let spec = floats("spec_norm_init")?;
        if erb.len() != num("erb_norm_state_size")? || spec.len() != num("spec_norm_state_size")? {
            return Err("DPDFNet: norm state sizes disagree with metadata".into());
        }
        init_state[..erb.len()].copy_from_slice(&erb);
        init_state[erb.len()..erb.len() + spec.len()].copy_from_slice(&spec);
        drop(meta);

        let window = (0..Self::WIN)
            .map(|i| {
                let h = Self::WIN as f64 / 2.0;
                let s = (0.5 * std::f64::consts::PI * (i as f64 + 0.5) / h).sin();
                (0.5 * std::f64::consts::PI * s * s).sin() as f32
            })
            .collect();
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(Self::WIN);
        let ifft = planner.plan_fft_inverse(Self::WIN);
        Ok(Self {
            session,
            state: init_state.clone(),
            init_state,
            window,
            frame: vec![0.0; Self::WIN],
            time: vec![0.0; Self::WIN],
            spec: fft.make_output_vec(),
            spec_ri: vec![0.0; Self::BINS * 2],
            ola: vec![0.0; Self::WIN],
            scratch_fwd: fft.make_scratch_vec(),
            scratch_inv: ifft.make_scratch_vec(),
            fft,
            ifft,
        })
    }

    fn infer(&mut self) -> Result<(), String> {
        let out = self
            .session
            .run(ort::inputs![
                "spec" => TensorRef::from_array_view(([1usize, 1, Self::BINS, 2], &self.spec_ri[..])).map_err(err)?,
                "state_in" => TensorRef::from_array_view(([self.state.len()], &self.state[..])).map_err(err)?,
            ])
            .map_err(err)?;
        let (_, spec) = out["spec_e"].try_extract_tensor::<f32>().map_err(err)?;
        self.spec_ri.copy_from_slice(spec);
        let (_, state) = out["state_out"].try_extract_tensor::<f32>().map_err(err)?;
        self.state.copy_from_slice(state);
        Ok(())
    }
}

impl Denoiser for DpdfNet {
    fn frame_size(&self) -> usize {
        Self::HOP
    }

    fn latency(&self) -> usize {
        Self::LATENCY
    }

    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        let (h, n) = (Self::HOP, Self::WIN);
        self.frame.copy_within(h.., 0);
        self.frame[n - h..].copy_from_slice(input);
        for ((t, &x), &w) in self.time.iter_mut().zip(&self.frame).zip(&self.window) {
            *t = x * w;
        }
        let ok = self
            .fft
            .process_with_scratch(&mut self.time, &mut self.spec, &mut self.scratch_fwd)
            .is_ok();
        for (ri, c) in self.spec_ri.as_chunks_mut::<2>().0.iter_mut().zip(&self.spec) {
            *ri = [c.re, c.im];
        }
        if !ok || self.infer().is_err() {
            output.fill(0.0);
            return;
        }
        for (c, [re, im]) in self.spec.iter_mut().zip(self.spec_ri.as_chunks::<2>().0) {
            *c = Complex32::new(*re, *im);
        }
        // realfft's inverse needs real DC and Nyquist bins.
        self.spec[0].im = 0.0;
        self.spec[Self::BINS - 1].im = 0.0;
        if self
            .ifft
            .process_with_scratch(&mut self.spec, &mut self.time, &mut self.scratch_inv)
            .is_err()
        {
            output.fill(0.0);
            return;
        }
        // Unnormalised inverse; the Vorbis window is power-complementary at 50% overlap, so
        // windowing again and overlap-adding reconstructs the signal.
        let scale = 1.0 / n as f32;
        for ((o, &t), &w) in self.ola.iter_mut().zip(&self.time).zip(&self.window) {
            *o += t * w * scale;
        }
        output.copy_from_slice(&self.ola[..h]);
        self.ola.copy_within(h.., 0);
        self.ola[n - h..].fill(0.0);
    }

    fn reset(&mut self) {
        self.state.copy_from_slice(&self.init_state);
        self.frame.fill(0.0);
        self.ola.fill(0.0);
    }
}

/// FastEnhancer-S, 48 kHz. 512-sample (10.7 ms) frames.
#[cfg(feature = "fastenhancer")]
pub struct FastEnhancer {
    session: Session,
    /// `cache_in_*` tensors: shape and contents, in input order.
    caches: Vec<(Vec<usize>, Vec<f32>)>,
    names_in: Vec<String>,
    names_out: Vec<String>,
}

#[cfg(feature = "fastenhancer")]
impl FastEnhancer {
    const HOP: usize = 512;
    /// n_fft − hop: the export caches one hop of input and output for its overlap-add.
    const LATENCY: usize = 512;

    /// Loads the bundled weights. Off the RT thread.
    pub fn new() -> Result<Self, String> {
        let session = session(FASTENHANCER_S)?;
        let mut caches = Vec::new();
        let mut names_in = Vec::new();
        for i in session.inputs().iter().filter(|i| i.name().starts_with("cache_in_")) {
            let shape: Vec<usize> = match i.dtype().tensor_shape() {
                Some(s) => s.iter().map(|&d| d.max(1) as usize).collect(),
                None => return Err(format!("FastEnhancer: {} is not a tensor", i.name())),
            };
            let len = shape.iter().product();
            caches.push((shape, vec![0.0; len]));
            names_in.push(i.name().to_string());
        }
        let names_out = names_in.iter().map(|n| n.replace("cache_in_", "cache_out_")).collect();
        Ok(Self {
            session,
            caches,
            names_in,
            names_out,
        })
    }

    fn infer(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        let mut inputs = ort::inputs![
            "wav_in" => TensorRef::from_array_view(([1usize, Self::HOP], input)).map_err(err)?,
        ];
        for (name, (shape, data)) in self.names_in.iter().zip(&self.caches) {
            inputs.push((
                name.clone().into(),
                TensorRef::from_array_view((shape.clone(), &data[..])).map_err(err)?.into(),
            ));
        }
        let out = self.session.run(inputs).map_err(err)?;
        let (_, wav) = out["wav_out"].try_extract_tensor::<f32>().map_err(err)?;
        output.copy_from_slice(wav);
        for (name, (_, data)) in self.names_out.iter().zip(self.caches.iter_mut()) {
            let (_, c) = out[name.as_str()].try_extract_tensor::<f32>().map_err(err)?;
            data.copy_from_slice(c);
        }
        Ok(())
    }
}

#[cfg(feature = "fastenhancer")]
impl Denoiser for FastEnhancer {
    fn frame_size(&self) -> usize {
        Self::HOP
    }

    fn latency(&self) -> usize {
        Self::LATENCY
    }

    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        if self.infer(input, output).is_err() {
            output.fill(0.0);
        }
    }

    fn reset(&mut self) {
        for (_, c) in &mut self.caches {
            c.fill(0.0);
        }
    }
}
