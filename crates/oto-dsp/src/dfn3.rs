//! DeepFilterNet3 via upstream `libDF` (`deep_filter`) and `tract`. Behind the `dfn3` feature.
//!
//! Not allocation-free: `libDF` clones spectrum tensors and `tract` allocates per run. That is
//! one reason it is not the default; see [`crate::Model`].
//!
//! Build notes: pinned to upstream commit 7b14b89 (6 Nov 2026), which builds on Rust 1.99.1
//! only with tract held at exactly 0.21.4 (0.21.5+ renamed `symbol_table`, breaking libDF).
//! Weights licence: that same commit states in the README that the model weights are
//! dual-licensed MIT OR Apache-2.0, and the owner closed issue #700 with it.

use crate::Denoiser;
use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::Array2;

pub struct DeepFilterNet3 {
    model: DfTract,
    /// A freshly initialised copy, so `reset` does not have to re-parse the ONNX graphs.
    fresh: DfTract,
    inb: Array2<f32>,
    outb: Array2<f32>,
    latency: usize,
}

impl DeepFilterNet3 {
    /// Loads the bundled DFN3 weights (MIT OR Apache-2.0). Slow (tens of ms); off the RT thread.
    pub fn new() -> Result<Self, String> {
        let rp = RuntimeParams::default_with_ch(1);
        let model = DfTract::new(DfParams::default(), &rp).map_err(|e| e.to_string())?;
        let hop = model.hop_size;
        // STFT overlap (fft - hop) plus the model's lookahead frames.
        let latency = model.fft_size - hop + model.lookahead * hop;
        Ok(Self {
            fresh: model.clone(),
            model,
            inb: Array2::zeros((1, hop)),
            outb: Array2::zeros((1, hop)),
            latency,
        })
    }
}

// SAFETY: `DfTract` is `!Send` only because tract's `TValue` wraps tensors in `Rc` and its
// plan state holds `Box<dyn OpState>` without a `Send` bound. Every `Rc` reachable from this
// struct is owned by it (`model` and `fresh` may share tensors, but they always move together),
// libDF keeps no global or thread-local handles to them, and none is ever handed out: the
// public API takes and returns plain slices. Moving the whole value to another thread therefore
// cannot race a refcount. It is not `Sync`, and nothing here needs it to be.
unsafe impl Send for DeepFilterNet3 {}

impl Denoiser for DeepFilterNet3 {
    fn frame_size(&self) -> usize {
        self.model.hop_size
    }

    fn latency(&self) -> usize {
        self.latency
    }

    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        self.inb
            .as_slice_mut()
            .expect("contiguous")
            .copy_from_slice(input);
        match self.model.process(self.inb.view(), self.outb.view_mut()) {
            Ok(_) => output.copy_from_slice(self.outb.as_slice().expect("contiguous")),
            Err(_) => output.fill(0.0),
        }
    }

    fn reset(&mut self) {
        let lim = self.model.atten_lim;
        self.model = self.fresh.clone();
        self.model.atten_lim = lim;
    }

    /// libDF limits attenuation natively by mixing the noisy spectrum back in.
    fn set_max_attenuation_db(&mut self, db: f32) -> bool {
        // libDF treats >= 100 dB as "no limit".
        self.model
            .set_atten_lim(if db.is_finite() { db } else { 100.0 });
        true
    }
}
