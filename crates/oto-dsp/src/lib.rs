//! Oto's signal chain: everything that touches samples and nothing that touches devices.
//!
//! Rules for this crate: no allocation, locking or syscalls on the processing path, so it is
//! safe to call from a realtime audio callback.

/// The sample rate the whole pipeline runs at. Devices at other rates are resampled at the edges.
pub const SAMPLE_RATE: u32 = 48_000;

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
        Self { denoiser, inbuf: vec![0.0; n].into(), outbuf: vec![0.0; n].into(), pos: 0 }
    }

    /// Total delay in samples: the adapter's one frame plus the model's own.
    pub fn latency(&self) -> usize {
        self.inbuf.len() + self.denoiser.latency()
    }

    /// Process a block of any length. `input` and `output` must be the same length.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        for (x, y) in input.iter().zip(output.iter_mut()) {
            *y = self.outbuf[self.pos];
            self.inbuf[self.pos] = *x;
            self.pos += 1;
            if self.pos == self.inbuf.len() {
                self.denoiser.process_frame(&self.inbuf, &mut self.outbuf);
                self.pos = 0;
            }
        }
    }

    pub fn reset(&mut self) {
        self.inbuf.fill(0.0);
        self.outbuf.fill(0.0);
        self.pos = 0;
        self.denoiser.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_block_sizes_come_out_delayed_by_exactly_one_frame() {
        let mut a = FrameAdapter::new(Passthrough::new(480));
        let input: Vec<f32> = (0..5000).map(|i| i as f32).collect();
        let mut output = vec![0.0; input.len()];
        let mut at = 0;
        for block in [1, 7, 128, 479, 480, 481, 1024, 2300].iter().cycle() {
            if at == input.len() {
                break;
            }
            let end = (at + block).min(input.len());
            a.process(&input[at..end], &mut output[at..end]);
            at = end;
        }
        assert_eq!(a.latency(), 480);
        assert!(output[..480].iter().all(|&s| s == 0.0));
        assert_eq!(&output[480..], &input[..input.len() - 480]);
    }

    #[test]
    fn reset_clears_the_tail() {
        let mut a = FrameAdapter::new(Passthrough::new(4));
        let mut out = [0.0; 6];
        a.process(&[1.0; 6], &mut out);
        a.reset();
        a.process(&[0.0; 6], &mut out);
        assert_eq!(out, [0.0; 6]);
    }
}
