//! Oto-1: Oto's own 48 kHz model, run by a small hand-written pure-Rust kernel that builds the
//! same everywhere (native, `wasm32-unknown-unknown` with or without `simd128`): no `ort`, no
//! `tract`, no OS calls, no dependencies, and no heap traffic in [`Oto1::process_frame`].
//!
//! Two candidate architectures share the kernel; `SPEC.md` next to this file is the contract the
//! PyTorch exporter writes against (file layout, tensor names and shapes, features, equations).
//!
//! - STFT: 48 kHz, 1024-point FFT, hop 512, periodic `sqrt(Hann)` analysis and synthesis, no
//!   lookahead. The model adds [`HOP`] samples of delay; `FrameAdapter` adds another, so the
//!   chain delay is 1024 samples (21.33 ms).
//! - Weights: `.otow` v1 ([`format`]), f16 or f32 on disk, f32 in memory.

pub mod arch_a;
pub mod arch_b;
pub mod fft;
pub mod format;
pub mod kernels;
pub mod layers;
mod sha256;
#[cfg(test)]
mod tests;

pub use kernels::Backend;
pub use sha256::sha256;

use arch_a::{ArchA, ConfigA};
use arch_b::{ArchB, ConfigB};
use format::{DTYPE_F16, FLAG_ENROLLMENT, FLAG_ISOLATE, Otow, OtowWriter, Rng};
use layers::Source;

pub const SAMPLE_RATE: u32 = 48_000;
pub const N_FFT: usize = 1024;
pub const HOP: usize = 512;
pub const N_BINS: usize = N_FFT / 2 + 1;

/// Which network a `.otow` file holds (header `arch_id`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {
    /// Candidate A: DeepFilterNet-style ERB gains + deep filter, grouped GRU.
    A = 1,
    /// Candidate B: FastEnhancer-style conv U-Net + bottleneck GRU, complex ratio mask.
    B = 2,
}

impl Arch {
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            1 => Some(Arch::A),
            2 => Some(Arch::B),
            _ => None,
        }
    }
}

/// Band edges (in bins, `n + 1` of them, first 0, last 513) of `n` rectangular ERB bands.
///
/// Glasberg & Moore ERB-rate `erb(f) = 21.4 log10(1 + 0.00437 f)`; edge `k` (`0 < k < n`) is
/// `floor(erb⁻¹(k · erb(24000) / n) / 46.875 + 0.5)`, raised to at least `edge[k-1] + 2` so no
/// band is narrower than two bins. Computed in f64.
pub fn erb_band_edges(n: usize) -> Vec<usize> {
    const MIN_WIDTH: usize = 2;
    let erb = |f: f64| 21.4 * (1.0 + 0.00437 * f).log10();
    let inv = |e: f64| (10f64.powf(e / 21.4) - 1.0) / 0.00437;
    let top = erb(SAMPLE_RATE as f64 / 2.0);
    let bin_hz = SAMPLE_RATE as f64 / N_FFT as f64;
    let mut edges = vec![0usize; n + 1];
    for k in 1..n {
        let bin = (inv(k as f64 * top / n as f64) / bin_hz + 0.5).floor() as usize;
        edges[k] = bin.max(edges[k - 1] + MIN_WIDTH);
    }
    edges[n] = N_BINS;
    assert!(
        edges[n] >= edges[n - 1] + MIN_WIDTH,
        "ERB bands do not fit: {edges:?}"
    );
    edges
}

/// Magnitude-compressed complex spectrum, `X · (|X|² + 1e-12)^(-0.35)` (so `|out| ≈ |X|^0.3`),
/// interleaved in and out.
pub fn compress_into(x: &[f32], out: &mut [f32]) {
    for (o, c) in out
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<2>().0)
    {
        let s = (c[0] * c[0] + c[1] * c[1] + 1e-12).powf(-0.35);
        o[0] = c[0] * s;
        o[1] = c[1] * s;
    }
}

enum Net {
    A(Box<ArchA>),
    B(Box<ArchB>),
}

/// An Oto-1 model ready to stream. Implements [`crate::Denoiser`] with 512-sample frames.
pub struct Oto1 {
    net: Net,
    isolate: bool,
    backend: Backend,
}

impl Oto1 {
    /// Load a `.otow` file, using the fastest kernels this CPU supports.
    pub fn from_otow(bytes: &[u8]) -> Result<Self, String> {
        Self::from_otow_with(bytes, Backend::detect())
    }

    /// Load a `.otow` file with a specific kernel backend.
    pub fn from_otow_with(bytes: &[u8], backend: Backend) -> Result<Self, String> {
        if !backend.is_supported() {
            return Err(format!("Oto-1: CPU does not support backend {backend:?}"));
        }
        let mut o = Otow::parse(bytes)?;
        if o.flags & FLAG_ENROLLMENT != 0 {
            return Err("Oto-1: enrolled models are not supported by this kernel yet".into());
        }
        let arch = Arch::from_id(o.arch_id)
            .ok_or_else(|| format!("Oto-1: unknown arch_id {}", o.arch_id))?;
        let isolate = o.flags & FLAG_ISOLATE != 0;
        let net = match arch {
            Arch::A => {
                let cfg = ConfigA::from_otow(&o)?;
                Net::A(Box::new(ArchA::build(
                    &mut Source::File(&mut o),
                    cfg,
                    backend,
                )?))
            }
            Arch::B => {
                let cfg = ConfigB::from_otow(&o)?;
                Net::B(Box::new(ArchB::build(
                    &mut Source::File(&mut o),
                    cfg,
                    backend,
                )?))
            }
        };
        if let Some(t) = o.tensors.first() {
            return Err(format!(
                "Oto-1: unexpected tensor {} in a {arch:?} file",
                t.name
            ));
        }
        Ok(Self {
            net,
            isolate,
            backend,
        })
    }

    /// A model with seeded random weights (default width), stored as f16 like a real export.
    pub fn random(arch: Arch, seed: u64) -> Self {
        Self::from_otow(&random_otow(arch, seed)).expect("random model loads")
    }

    pub fn arch(&self) -> Arch {
        match self.net {
            Net::A(_) => Arch::A,
            Net::B(_) => Arch::B,
        }
    }

    /// Header flag: an Isolate (background-voice) model, which callers must run fail-closed.
    pub fn is_isolating(&self) -> bool {
        self.isolate
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn set_backend(&mut self, be: Backend) {
        self.backend = be;
        match &mut self.net {
            Net::A(n) => n.set_backend(be),
            Net::B(n) => n.set_backend(be),
        }
    }

    /// The target-presence logit of the most recent frame.
    pub fn p_target(&self) -> f32 {
        match &self.net {
            Net::A(n) => n.p_target(),
            Net::B(n) => n.p_target(),
        }
    }

    /// Multiply-accumulates per frame in the network's layers (excluding FFT and features).
    pub fn macs_per_frame(&self) -> usize {
        match &self.net {
            Net::A(n) => n.macs(),
            Net::B(n) => n.macs(),
        }
    }

    pub fn params(&self) -> usize {
        match &self.net {
            Net::A(n) => n.params(),
            Net::B(n) => n.params(),
        }
    }

    /// Whether any value carried between frames is subnormal (for the silence checks).
    #[doc(hidden)]
    pub fn state_has_subnormals(&self) -> bool {
        let sub = |v: f32| v != 0.0 && v.abs() < f32::MIN_POSITIVE;
        match &self.net {
            Net::A(n) => n.state().any(sub),
            Net::B(n) => n.state().any(sub),
        }
    }
}

impl crate::Denoiser for Oto1 {
    fn frame_size(&self) -> usize {
        HOP
    }
    fn latency(&self) -> usize {
        HOP
    }
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        match &mut self.net {
            Net::A(n) => n.process(input, output),
            Net::B(n) => n.process(input, output),
        }
    }
    fn reset(&mut self) {
        match &mut self.net {
            Net::A(n) => n.reset(),
            Net::B(n) => n.reset(),
        }
    }
}

/// A complete `.otow` (f16) with seeded random weights for `arch` at its default width: the
/// browser benchmark's model and an executable example of the format.
pub fn random_otow(arch: Arch, seed: u64) -> Vec<u8> {
    let mut tensors = Vec::new();
    let mut src = Source::Random(Rng::new(seed), &mut tensors);
    let config = match arch {
        Arch::A => {
            let cfg = ConfigA::default();
            ArchA::build(&mut src, cfg, Backend::Scalar).expect("random A");
            cfg.json()
        }
        Arch::B => {
            let cfg = ConfigB::default();
            ArchB::build(&mut src, cfg, Backend::Scalar).expect("random B");
            cfg.json()
        }
    };
    let mut w = OtowWriter::new(arch as u32, DTYPE_F16, 0, &config);
    for (name, shape, data) in tensors {
        w.add(&name, &shape, data);
    }
    w.finish()
}
