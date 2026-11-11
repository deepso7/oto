//! Oto's signal chain: everything that touches samples and nothing that touches devices.
//!
//! Rules for this crate: no allocation, locking or syscalls on the processing path, so it is
//! safe to call from a realtime audio callback.

mod bench;
#[cfg(feature = "dfn3")]
mod dfn3;
pub mod filters;
#[cfg(feature = "onnx")]
mod onnx;
mod rnnoise;
pub mod signal;

pub use bench::{BENCH_BLOCK, BenchReport, bench, try_bench};
#[cfg(feature = "dfn3")]
pub use dfn3::DeepFilterNet3;
#[cfg(feature = "onnx")]
pub use onnx::DpdfNet;
#[cfg(feature = "fastenhancer")]
pub use onnx::FastEnhancer;
pub use rnnoise::RnNoise;

use filters::{Delay, HighPass, Limiter};

/// The sample rate the whole pipeline runs at. Devices at other rates are resampled at the edges.
pub const SAMPLE_RATE: u32 = 48_000;

/// High-pass corner of the chain, in Hz.
pub const HIGH_PASS_HZ: f32 = 80.0;
/// Limiter ceiling of the chain, in dBFS.
pub const LIMITER_CEILING_DB: f32 = -1.0;

/// Largest input magnitude the chain accepts (+40 dBFS); anything beyond is clamped on entry.
/// Real devices stay within ±1.0, so this only catches garbage before it can overflow a filter
/// or a model.
pub const MAX_INPUT: f32 = 100.0;

/// Makes a block safe to feed to filters and models: NaN and ±infinity become 0.0 and the rest
/// is clamped to ±[`MAX_INPUT`]. One non-finite sample would otherwise stick in recursive state
/// (filters, the model's GRUs) and silence everything after it. Allocation-free.
pub fn sanitize(buf: &mut [f32]) {
    for s in buf {
        *s = if s.is_finite() {
            s.clamp(-MAX_INPUT, MAX_INPUT)
        } else {
            0.0
        };
    }
}

/// A streaming mono denoiser that consumes and produces fixed-size frames.
pub trait Denoiser: Send {
    /// Samples per frame (the model's hop size), e.g. 480 for 10 ms at 48 kHz.
    fn frame_size(&self) -> usize;
    /// Delay the model itself adds, in samples (window overlap plus lookahead).
    fn latency(&self) -> usize;
    /// Process exactly one frame. Both slices are `frame_size()` long.
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]);
    /// Drop internal state, e.g. after a device change or a long gap.
    fn reset(&mut self) {}
    /// Limit how far the model may attenuate, in dB (`f32::INFINITY` = no limit). Return `true`
    /// if the model applies the limit itself; `false` (the default) makes [`Chain`] apply it as
    /// a dry-mix floor instead.
    fn set_max_attenuation_db(&mut self, _db: f32) -> bool {
        false
    }
}

impl<D: Denoiser + ?Sized> Denoiser for Box<D> {
    fn frame_size(&self) -> usize {
        (**self).frame_size()
    }
    fn latency(&self) -> usize {
        (**self).latency()
    }
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        (**self).process_frame(input, output)
    }
    fn reset(&mut self) {
        (**self).reset()
    }
    fn set_max_attenuation_db(&mut self, db: f32) -> bool {
        (**self).set_max_attenuation_db(db)
    }
}

/// Copies input to output. The baseline for latency and CPU measurements.
pub struct Passthrough {
    frame: usize,
}

impl Passthrough {
    pub fn new(frame: usize) -> Self {
        Self { frame }
    }
}

impl Denoiser for Passthrough {
    fn frame_size(&self) -> usize {
        self.frame
    }
    fn latency(&self) -> usize {
        0
    }
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        output.copy_from_slice(input);
    }
}

/// Lets a fixed-frame [`Denoiser`] run inside callbacks of any block size.
///
/// Audio hosts hand us whatever block size they like (PipeWire's quantum can change at runtime),
/// while models need exact frames. This adds exactly one frame of delay and never allocates
/// after construction.
pub struct FrameAdapter<D: Denoiser> {
    denoiser: D,
    inbuf: Box<[f32]>,
    outbuf: Box<[f32]>,
    pos: usize,
}

impl<D: Denoiser> FrameAdapter<D> {
    pub fn new(denoiser: D) -> Self {
        let n = denoiser.frame_size();
        assert!(n > 0, "frame size must be non-zero");
        Self {
            denoiser,
            inbuf: vec![0.0; n].into(),
            outbuf: vec![0.0; n].into(),
            pos: 0,
        }
    }

    /// Total delay in samples: the adapter's one frame plus the model's own.
    pub fn latency(&self) -> usize {
        self.inbuf.len() + self.denoiser.latency()
    }

    /// Process a block of any length. `input` and `output` must be the same length.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        for (x, y) in input.iter().zip(output.iter_mut()) {
            *y = self.tick(*x);
        }
    }

    /// [`process`](Self::process), in place.
    pub fn process_in_place(&mut self, buf: &mut [f32]) {
        for s in buf {
            *s = self.tick(*s);
        }
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.outbuf[self.pos];
        self.inbuf[self.pos] = x;
        self.pos += 1;
        if self.pos == self.inbuf.len() {
            self.denoiser.process_frame(&self.inbuf, &mut self.outbuf);
            self.pos = 0;
        }
        y
    }

    pub fn denoiser(&self) -> &D {
        &self.denoiser
    }

    pub fn denoiser_mut(&mut self) -> &mut D {
        &mut self.denoiser
    }

    pub fn reset(&mut self) {
        self.inbuf.fill(0.0);
        self.outbuf.fill(0.0);
        self.pos = 0;
        self.denoiser.reset();
    }
}

/// Which denoising model a [`Chain`] runs.
///
/// The default is DeepFilterNet3 with the `dfn3` feature (on by default) and RNNoise without. Measured with [`bench`] on an M2 Pro (release, 20 s): RNNoise
/// p50 43 µs / p99 50 µs per 10 ms block, 0.44% of a core, 20 ms chain latency; DeepFilterNet3
/// p50 730 µs / p99 810–900 µs, 6.5% of a core, 40 ms chain latency, and it allocates per
/// frame. With ~11 ms of macOS loopback plus device buffers on top, only RNNoise fits the
/// 50 ms end-to-end budget (and the 35 ms stretch); DFN3 lands near 55–65 ms. Re-run with
/// `cargo test -p oto-dsp --release --features dfn3 -- --ignored --nocapture bench_all`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Model {
    /// RNNoise via `nnnoiseless` (BSD-3-Clause). Always available.
    #[cfg_attr(not(feature = "dfn3"), default)]
    RNNoise,
    /// DeepFilterNet3 via `libDF` + `tract`. Needs the `dfn3` cargo feature.
    #[cfg_attr(feature = "dfn3", default)]
    DeepFilterNet3,
    /// DPDFNet-2 (48 kHz) via ONNX Runtime. Needs the `onnx` cargo feature.
    DpdfNet,
    /// FastEnhancer-S (48 kHz) via ONNX Runtime. Needs the `fastenhancer` cargo feature. Weights
    /// licence unstated upstream: for local evaluation, not cleared to ship.
    FastEnhancer,
}

impl Model {
    pub const ALL: [Model; 4] = [
        Model::RNNoise,
        Model::DeepFilterNet3,
        Model::DpdfNet,
        Model::FastEnhancer,
    ];

    /// Whether this build can run the model.
    pub fn available(self) -> bool {
        match self {
            Model::RNNoise => true,
            Model::DeepFilterNet3 => cfg!(feature = "dfn3"),
            Model::DpdfNet => cfg!(feature = "onnx"),
            Model::FastEnhancer => cfg!(feature = "fastenhancer"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Model::RNNoise => "RNNoise",
            Model::DeepFilterNet3 => "DeepFilterNet3",
            Model::DpdfNet => "DPDFNet-2",
            Model::FastEnhancer => "FastEnhancer-S",
        }
    }

    fn build(self) -> Result<Box<dyn Denoiser>, String> {
        match self {
            Model::RNNoise => Ok(Box::new(RnNoise::new())),
            #[cfg(feature = "dfn3")]
            Model::DeepFilterNet3 => Ok(Box::new(DeepFilterNet3::new()?)),
            #[cfg(not(feature = "dfn3"))]
            Model::DeepFilterNet3 => {
                Err("DeepFilterNet3 is not in this build (enable the `dfn3` feature)".into())
            }
            #[cfg(feature = "onnx")]
            Model::DpdfNet => Ok(Box::new(DpdfNet::new()?)),
            #[cfg(not(feature = "onnx"))]
            Model::DpdfNet => {
                Err("DPDFNet-2 is not in this build (enable the `onnx` feature)".into())
            }
            #[cfg(feature = "fastenhancer")]
            Model::FastEnhancer => Ok(Box::new(FastEnhancer::new()?)),
            #[cfg(not(feature = "fastenhancer"))]
            Model::FastEnhancer => Err(
                "FastEnhancer-S is not in this build (enable the `fastenhancer` feature)".into(),
            ),
        }
    }
}

/// The denoise stage as the [`FrameAdapter`] sees it: the model plus bypass, silence skipping,
/// the strength floor and click-free transitions between them.
///
/// A delay line keeps a dry copy aligned with the model's output, so latency is the same
/// whether the model runs or not, and the dry signal is available for the strength floor.
struct Stage {
    model: Box<dyn Denoiser>,
    dry: Delay,
    dry_buf: Box<[f32]>,
    wet_buf: Box<[f32]>,
    enabled: bool,
    /// Whether the model applies the attenuation limit itself.
    native_limit: bool,
    /// Target and current share of dry signal mixed into the model output (strength floor).
    floor: f32,
    floor_now: f32,
    /// Current share of the denoised path in the output; ramps 0 ↔ 1 over one frame.
    wet_gain: f32,
    /// Frames the model must run before its output is trusted after being idle.
    warmup: usize,
    warmup_frames: usize,
    silent_frames: usize,
    /// Consecutive all-zero frames after which inference stops; by then dry and model tails
    /// are both exactly flushed.
    skip_after: usize,
    inferences: u64,
    /// No frame processed since construction or reset.
    fresh: bool,
}

impl Stage {
    fn new(model: Box<dyn Denoiser>) -> Self {
        let n = model.frame_size();
        let lat = model.latency();
        let settle = lat.div_ceil(n) + 1;
        Self {
            dry: Delay::new(lat),
            dry_buf: vec![0.0; n].into(),
            wet_buf: vec![0.0; n].into(),
            enabled: true,
            native_limit: false,
            floor: 0.0,
            floor_now: 0.0,
            wet_gain: 1.0,
            warmup: 0,
            warmup_frames: settle,
            silent_frames: 0,
            skip_after: settle,
            inferences: 0,
            fresh: true,
            model,
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        if self.fresh {
            // Nothing processed since construction or reset: no transition to smooth.
            self.wet_gain = if enabled { 1.0 } else { 0.0 };
        } else if enabled && !self.enabled && self.wet_gain == 0.0 {
            // The model sat idle; let it re-converge on live audio before we listen to it.
            self.warmup = self.warmup_frames;
        }
        self.enabled = enabled;
    }

    fn set_max_attenuation_db(&mut self, db: f32) {
        let db = if db.is_nan() {
            f32::INFINITY
        } else {
            db.max(0.0)
        };
        self.native_limit = self.model.set_max_attenuation_db(db);
        self.floor = if self.native_limit || db.is_infinite() {
            0.0
        } else {
            10f32.powf(-db / 20.0)
        };
    }
}

impl Denoiser for Stage {
    fn frame_size(&self) -> usize {
        self.dry_buf.len()
    }

    fn latency(&self) -> usize {
        self.model.latency()
    }

    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        let n = input.len();
        self.fresh = false;
        self.dry.process(input, &mut self.dry_buf);

        if input.iter().all(|&x| x == 0.0) {
            self.silent_frames = self.silent_frames.saturating_add(1);
        } else {
            self.silent_frames = 0;
        }
        let skipping = self.silent_frames > self.skip_after;
        let target = if self.enabled && self.warmup == 0 {
            1.0
        } else {
            0.0
        };
        let run = !skipping && (self.enabled || self.wet_gain > 0.0);

        if run {
            self.model.process_frame(input, &mut self.wet_buf);
            self.inferences += 1;
            if self.enabled {
                self.warmup = self.warmup.saturating_sub(1);
            }
        } else if skipping {
            // Input, dry tail and model tail are all zero by now; so is the output.
            output.fill(0.0);
            self.wet_gain = target;
            self.floor_now = self.floor;
            return;
        } else {
            self.wet_buf.copy_from_slice(&self.dry_buf);
        }

        // Ramp wet gain and floor linearly across the frame: no clicks on toggles or strength.
        let step = 1.0 / n as f32;
        let g_step = (target - self.wet_gain) * step;
        let f_step = (self.floor - self.floor_now) * step;
        let (mut g, mut f) = (self.wet_gain, self.floor_now);
        for ((y, &d), &w) in output.iter_mut().zip(&*self.dry_buf).zip(&*self.wet_buf) {
            g += g_step;
            f += f_step;
            let wet = w + (d - w) * f;
            *y = d + (wet - d) * g;
        }
        self.wet_gain = target;
        self.floor_now = self.floor;
    }

    /// Resets the model too, which may allocate.
    fn reset(&mut self) {
        self.model.reset();
        self.dry.reset();
        self.silent_frames = 0;
        self.warmup = 0;
        self.wet_gain = if self.enabled { 1.0 } else { 0.0 };
        self.fresh = true;
        self.floor_now = self.floor;
    }
}

/// The full v1 signal chain: high-pass (80 Hz) → denoise → limiter (−1 dBFS).
///
/// Mono, 48 kHz, any block size, in place. `process`, `set_enabled` and
/// `set_max_attenuation_db` never allocate with RNNoise (DeepFilterNet3's libDF/tract allocate
/// inside inference). Construction and [`reset`](Self::reset) allocate.
///
/// - **High-pass**: 4th-order Butterworth at 80 Hz, zero latency.
/// - **Denoise**: the model behind a [`FrameAdapter`]. Disabled = dry signal through a matching
///   delay, so latency never changes and toggling crossfades over one 10 ms frame. Inference
///   also stops after a few frames of exact digital silence.
/// - **Strength**: [`set_max_attenuation_db`](Self::set_max_attenuation_db). DeepFilterNet3
///   limits natively. RNNoise has no such control, so the chain mixes the aligned dry signal
///   back in at `10^(-dB/20)`: noise is never pulled down by more than the limit, at the cost
///   of also leaving that much of the noise's original colour.
/// - **Limiter**: instant-attack, 50 ms release peak limiter at −1 dBFS, zero latency.
pub struct Chain {
    model: Model,
    hp: HighPass,
    denoise: FrameAdapter<Stage>,
    limiter: Limiter,
    enabled: bool,
    max_attenuation_db: f32,
}

impl Chain {
    /// Build the chain at [`SAMPLE_RATE`] with the default [`Model`]. Allocates; call off the
    /// realtime thread.
    pub fn new() -> Self {
        Self::with_model(Model::default()).expect("the default model is always available")
    }

    /// Build the chain with a specific model. Fails if the model is not compiled in or does not
    /// load. Allocates; call off the realtime thread.
    pub fn with_model(model: Model) -> Result<Self, String> {
        let sr = SAMPLE_RATE as f32;
        let mut chain = Self {
            model,
            hp: HighPass::new(sr as f64, HIGH_PASS_HZ as f64),
            denoise: FrameAdapter::new(Stage::new(model.build()?)),
            limiter: Limiter::new(sr, LIMITER_CEILING_DB, 50.0),
            enabled: true,
            max_attenuation_db: 24.0,
        };
        chain.set_max_attenuation_db(chain.max_attenuation_db);
        Ok(chain)
    }

    /// The model this chain runs.
    pub fn model(&self) -> Model {
        self.model
    }

    /// When `false`, denoising is bypassed (high-pass and limiter still run) and no inference runs.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.denoise.denoiser_mut().set_enabled(enabled);
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Strength: the most the denoiser may attenuate any band, in dB. `f32::INFINITY` = no limit.
    pub fn set_max_attenuation_db(&mut self, db: f32) {
        self.max_attenuation_db = db;
        self.denoise.denoiser_mut().set_max_attenuation_db(db);
    }

    pub fn max_attenuation_db(&self) -> f32 {
        self.max_attenuation_db
    }

    /// Delay the chain adds, in samples at [`SAMPLE_RATE`]. Constant for a given model, whether
    /// enabled or not: one frame of buffering plus the model's own delay.
    pub fn latency(&self) -> usize {
        self.denoise.latency()
    }

    /// Do this thread's one-time setup now, so later `process` calls on it never allocate.
    ///
    /// `nnnoiseless` caches its FFT plans and scratch buffers per thread, so the first model
    /// frame on any new thread allocates (and plans FFTs, ~1 ms). Call this once on the thread
    /// that will run [`process`](Self::process), before audio flows, e.g. at the top of the
    /// first audio callback or when a dedicated DSP thread starts. It runs the model on one
    /// frame of silence, which right after construction or `reset` changes nothing audible;
    /// mid-stream it would drop a frame of the model's history, so do not call it per callback.
    pub fn prime(&mut self) {
        let stage = self.denoise.denoiser_mut();
        let n = stage.wet_buf.len();
        // Borrow the dry buffer as an all-zero input; the next frame overwrites it anyway.
        stage.dry_buf.fill(0.0);
        stage
            .model
            .process_frame(&stage.dry_buf[..n], &mut stage.wet_buf);
    }

    /// Model frames run so far. Lets callers verify that bypass and silence really idle.
    pub fn inference_frames(&self) -> u64 {
        self.denoise.denoiser().inferences
    }

    /// Process one block in place. Non-finite input samples are treated as silence (see
    /// [`sanitize`]), so a glitching device cannot poison the filters or the model.
    pub fn process(&mut self, buf: &mut [f32]) {
        sanitize(buf);
        self.hp.process(buf);
        self.denoise.process_in_place(buf);
        self.limiter.process(buf);
    }

    /// Drop state after a device change or a long gap. Allocates (rebuilds model state); call
    /// off the realtime thread.
    pub fn reset(&mut self) {
        self.hp.reset();
        self.denoise.reset();
        self.limiter.reset();
    }
}

impl Default for Chain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
