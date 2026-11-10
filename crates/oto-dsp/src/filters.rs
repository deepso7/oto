//! Hand-written filters for the chain: a Butterworth high-pass and a peak limiter.
//!
//! Both are zero-latency, allocation-free and run sample by sample.

use std::f64::consts::PI;

/// Below this (−180 dBFS, far under 24-bit resolution) the filter state is flushed to zero, so
/// digital silence in becomes digital silence out within ~0.1 s (letting the chain skip
/// inference) and the state never decays into denormals (slow on x86).
const DENORMAL_FLOOR: f64 = 1e-9;

/// A second-order section in transposed direct form II, with `f64` state.
///
/// `f64` because an 80 Hz corner at 48 kHz puts the poles very close to the unit circle, where
/// `f32` coefficients and state add audible noise and corner drift.
#[derive(Clone, Debug)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    s1: f64,
    s2: f64,
}

impl Biquad {
    /// RBJ cookbook high-pass at `freq` Hz with quality factor `q`.
    pub fn high_pass(sample_rate: f64, freq: f64, q: f64) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 + cos) / 2.0 / a0,
            b1: -(1.0 + cos) / a0,
            b2: (1.0 + cos) / 2.0 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            s1: 0.0,
            s2: 0.0,
        }
    }

    #[inline]
    pub fn tick(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        // Flush both together and only once both are tiny: zeroing one alone injects energy
        // and sustains a limit cycle.
        if self.s1.abs() < DENORMAL_FLOOR && self.s2.abs() < DENORMAL_FLOOR {
            self.s1 = 0.0;
            self.s2 = 0.0;
        }
        y
    }

    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}

/// Fourth-order (24 dB/octave) Butterworth high-pass: two cascaded biquads.
///
/// Removes rumble, handling noise and DC before the denoiser sees them. At 80 Hz it is
/// −3 dB at the corner, flat (< 0.01 dB) from 200 Hz up, and about −48 dB at 20 Hz.
#[derive(Clone, Debug)]
pub struct HighPass {
    stages: [Biquad; 2],
}

impl HighPass {
    pub fn new(sample_rate: f64, freq: f64) -> Self {
        // Q values of a 4th-order Butterworth: 1 / (2 cos(π/8)) and 1 / (2 cos(3π/8)).
        let q1 = 1.0 / (2.0 * (PI / 8.0).cos());
        let q2 = 1.0 / (2.0 * (3.0 * PI / 8.0).cos());
        Self {
            stages: [
                Biquad::high_pass(sample_rate, freq, q1),
                Biquad::high_pass(sample_rate, freq, q2),
            ],
        }
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        let [a, b] = &mut self.stages;
        for s in buf {
            *s = b.tick(a.tick(*s as f64)) as f32;
        }
    }

    pub fn reset(&mut self) {
        self.stages.iter_mut().for_each(Biquad::reset);
    }
}

/// A lookahead-free peak limiter.
///
/// Instant attack (the gain drops on the very sample that would overshoot), exponential
/// release. The output never exceeds the ceiling; non-finite input becomes silence. Instant
/// attack distorts a little on the transients it catches, which is the right trade for a
/// safety limiter that must add no delay.
#[derive(Clone, Debug)]
pub struct Limiter {
    ceiling: f32,
    release: f32,
    env: f32,
}

impl Limiter {
    /// `ceiling_db` in dBFS (e.g. −1.0), `release_ms` for the gain to recover by ~63%.
    pub fn new(sample_rate: f32, ceiling_db: f32, release_ms: f32) -> Self {
        Self {
            ceiling: 10f32.powf(ceiling_db / 20.0),
            release: (-1.0 / (release_ms * 0.001 * sample_rate)).exp(),
            env: 0.0,
        }
    }

    pub fn ceiling(&self) -> f32 {
        self.ceiling
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        let c = self.ceiling;
        for s in buf {
            let x = if s.is_finite() { *s } else { 0.0 };
            self.env = x.abs().max(self.env * self.release);
            let y = if self.env > c { x * (c / self.env) } else { x };
            // The division can round a hair over; the clamp makes the ceiling exact.
            *s = y.clamp(-c, c);
        }
        if self.env < 1e-20 {
            self.env = 0.0;
        }
    }

    pub fn reset(&mut self) {
        self.env = 0.0;
    }
}

/// A fixed integer delay. Allocates once at construction.
#[derive(Clone, Debug)]
pub struct Delay {
    buf: Box<[f32]>,
    pos: usize,
}

impl Delay {
    pub fn new(samples: usize) -> Self {
        Self {
            buf: vec![0.0; samples].into(),
            pos: 0,
        }
    }

    /// Writes `input` delayed into `output`. Same lengths.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        if self.buf.is_empty() {
            output.copy_from_slice(input);
            return;
        }
        for (x, y) in input.iter().zip(output.iter_mut()) {
            *y = self.buf[self.pos];
            self.buf[self.pos] = *x;
            self.pos += 1;
            if self.pos == self.buf.len() {
                self.pos = 0;
            }
        }
    }

    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.pos = 0;
    }
}
