//! The original RNNoise model (2018 weights) via `nnnoiseless` 0.5, a pure-Rust port
//! (BSD-3-Clause). This is not the newer RNNoise 0.2 model.

use crate::Denoiser;
use nnnoiseless::DenoiseState;

/// `nnnoiseless` wants samples on the i16 scale (±32768) even though they are `f32`.
const SCALE: f32 = 32_768.0;
const FRAME: usize = DenoiseState::FRAME_SIZE;

/// RNNoise: 10 ms frames, one frame of window overlap, no lookahead. Allocation-free per frame.
pub struct RnNoise {
    state: Box<DenoiseState<'static>>,
    inb: [f32; FRAME],
    outb: [f32; FRAME],
}

impl RnNoise {
    pub fn new() -> Self {
        Self {
            state: DenoiseState::new(),
            inb: [0.0; FRAME],
            outb: [0.0; FRAME],
        }
    }
}

impl Default for RnNoise {
    fn default() -> Self {
        Self::new()
    }
}

impl Denoiser for RnNoise {
    fn frame_size(&self) -> usize {
        FRAME
    }

    /// Analysis uses a two-frame window, so each output frame is the previous input frame.
    fn latency(&self) -> usize {
        FRAME
    }

    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        for (d, s) in self.inb.iter_mut().zip(input) {
            *d = *s * SCALE;
        }
        self.state.process_frame(&mut self.outb, &self.inb);
        for (d, s) in output.iter_mut().zip(&self.outb) {
            *d = *s * (1.0 / SCALE);
        }
    }

    /// Rebuilds the model state. Allocates.
    fn reset(&mut self) {
        self.state = DenoiseState::new();
    }
}
