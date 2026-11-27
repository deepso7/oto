//! The fixed 1024-point real FFT and the streaming `sqrt(Hann)` STFT/iSTFT Oto-1 runs on.
//!
//! One size only, so no planner: a 512-point complex radix-2 FFT plus the standard real-split
//! post/pre-processing. Tables are computed in f64 and rounded once, so they are the same on
//! every target. Scalar on purpose: it is ~2% of a frame and identical on every backend.

use super::layers::FLUSH;
use super::{HOP, N_BINS, N_FFT};

const M: usize = N_FFT / 2; // complex FFT size

/// Real FFT of size 1024. Spectra are interleaved `(re, im)` pairs, `N_BINS` of them.
pub struct Rfft {
    /// `e^{-2πi k / 512}`, k < 256, interleaved.
    tw: Vec<f32>,
    /// `e^{-2πi k / 1024}`, k <= 512, interleaved.
    split: Vec<f32>,
    bitrev: Vec<u16>,
    z: Vec<f32>,
}

impl Default for Rfft {
    fn default() -> Self {
        Self::new()
    }
}

impl Rfft {
    pub fn new() -> Self {
        let tau = core::f64::consts::TAU;
        let mut tw = Vec::with_capacity(M);
        for k in 0..M / 2 {
            let a = -tau * k as f64 / M as f64;
            tw.extend_from_slice(&[a.cos() as f32, a.sin() as f32]);
        }
        let mut split = Vec::with_capacity(2 * (M + 1));
        for k in 0..=M {
            let a = -tau * k as f64 / N_FFT as f64;
            split.extend_from_slice(&[a.cos() as f32, a.sin() as f32]);
        }
        let bits = M.trailing_zeros();
        let bitrev = (0..M)
            .map(|i| ((i as u32).reverse_bits() >> (32 - bits)) as u16)
            .collect();
        Self {
            tw,
            split,
            bitrev,
            z: vec![0.0; 2 * M],
        }
    }

    /// In-place complex FFT of `self.z`; `inverse` conjugates the twiddles (unnormalised).
    fn cfft(&mut self, inverse: bool) {
        let z = &mut self.z;
        for i in 0..M {
            let j = self.bitrev[i] as usize;
            if j > i {
                z.swap(2 * i, 2 * j);
                z.swap(2 * i + 1, 2 * j + 1);
            }
        }
        let sign = if inverse { -1.0 } else { 1.0 };
        let mut len = 2;
        while len <= M {
            let half = len / 2;
            let step = M / len;
            for start in (0..M).step_by(len) {
                for j in 0..half {
                    let wr = self.tw[2 * j * step];
                    let wi = sign * self.tw[2 * j * step + 1];
                    let (a, b) = (2 * (start + j), 2 * (start + j + half));
                    let (br, bi) = (z[b], z[b + 1]);
                    let tr = br * wr - bi * wi;
                    let ti = br * wi + bi * wr;
                    let (ar, ai) = (z[a], z[a + 1]);
                    z[a] = ar + tr;
                    z[a + 1] = ai + ti;
                    z[b] = ar - tr;
                    z[b + 1] = ai - ti;
                }
            }
            len *= 2;
        }
    }

    /// Unnormalised forward transform: `x` (1024) to `spec` (513 interleaved bins).
    pub fn forward(&mut self, x: &[f32], spec: &mut [f32]) {
        assert!(x.len() == N_FFT && spec.len() == 2 * N_BINS);
        self.z.copy_from_slice(x); // z[n] = x[2n] + i x[2n+1]
        self.cfft(false);
        let z = &self.z;
        for k in 0..=M {
            let (a, b) = (2 * (k % M), 2 * ((M - k) % M));
            let (zr, zi) = (z[a], z[a + 1]);
            let (cr, ci) = (z[b], -z[b + 1]); // conj(Z[M-k])
            let (er, ei) = (0.5 * (zr + cr), 0.5 * (zi + ci));
            // fo = -i (Z[k] - conj(Z[M-k])) / 2
            let (dr, di) = (0.5 * (zr - cr), 0.5 * (zi - ci));
            let (or, oi) = (di, -dr);
            let (wr, wi) = (self.split[2 * k], self.split[2 * k + 1]);
            spec[2 * k] = er + (or * wr - oi * wi);
            spec[2 * k + 1] = ei + (or * wi + oi * wr);
        }
    }

    /// Inverse transform scaled by 1/1024 (`numpy.fft.irfft`): `spec` (513 bins) to `x` (1024).
    /// The imaginary parts of the DC and Nyquist bins are ignored, as irfft does.
    pub fn inverse(&mut self, spec: &[f32], x: &mut [f32]) {
        assert!(x.len() == N_FFT && spec.len() == 2 * N_BINS);
        let bin = |k: usize| -> (f32, f32) {
            if k == 0 || k == M {
                (spec[2 * k], 0.0)
            } else {
                (spec[2 * k], spec[2 * k + 1])
            }
        };
        for k in 0..M {
            let (xr, xi) = bin(k);
            let (cr, ci) = {
                let (r, i) = bin(M - k);
                (r, -i)
            };
            let (er, ei) = (0.5 * (xr + cr), 0.5 * (xi + ci));
            let (dr, di) = (0.5 * (xr - cr), 0.5 * (xi - ci));
            // fo = d * conj(W^k)
            let (wr, wi) = (self.split[2 * k], -self.split[2 * k + 1]);
            let (or, oi) = (dr * wr - di * wi, dr * wi + di * wr);
            // Z = fe + i fo
            self.z[2 * k] = er - oi;
            self.z[2 * k + 1] = ei + or;
        }
        self.cfft(true);
        let scale = 1.0 / M as f32;
        for (o, &v) in x.iter_mut().zip(&self.z) {
            *o = v * scale;
        }
    }
}

/// Periodic `sqrt(Hann)`, computed in f64 and rounded once.
pub fn window() -> Vec<f32> {
    (0..N_FFT)
        .map(|n| {
            let c = (core::f64::consts::TAU * n as f64 / N_FFT as f64).cos();
            (0.5 * (1.0 - c)).sqrt() as f32
        })
        .collect()
}

/// Streaming analysis/synthesis around one hop. Input samples and windowed synthesis values with
/// `|v| < 1e-15` are flushed to zero. With an identity spectral model the output is the input
/// delayed by exactly [`HOP`] samples (the overlap); `FrameAdapter` adds the other hop.
pub struct Stft {
    win: Vec<f32>,
    analysis: Vec<f32>,
    ola: Vec<f32>,
    frame: Vec<f32>,
    fft: Rfft,
    /// The current frame's spectrum after [`analyze`](Self::analyze), 513 interleaved bins.
    pub spec: Vec<f32>,
}

impl Default for Stft {
    fn default() -> Self {
        Self::new()
    }
}

impl Stft {
    pub fn new() -> Self {
        Self {
            win: window(),
            analysis: vec![0.0; N_FFT],
            ola: vec![0.0; HOP],
            frame: vec![0.0; N_FFT],
            fft: Rfft::new(),
            spec: vec![0.0; 2 * N_BINS],
        }
    }

    pub fn reset(&mut self) {
        self.analysis.fill(0.0);
        self.ola.fill(0.0);
        self.spec.fill(0.0);
    }

    /// Shift in one hop and transform the windowed frame into [`spec`](Self::spec).
    pub fn analyze(&mut self, input: &[f32]) {
        assert_eq!(input.len(), HOP);
        self.analysis.copy_within(HOP.., 0);
        // Flush tiny input (below -300 dBFS) so subnormals never enter the model or its state.
        for (a, &x) in self.analysis[HOP..].iter_mut().zip(input) {
            *a = if x.abs() < FLUSH { 0.0 } else { x };
        }
        for ((f, &a), &w) in self.frame.iter_mut().zip(&self.analysis).zip(&self.win) {
            *f = a * w;
        }
        self.fft.forward(&self.frame, &mut self.spec);
    }

    /// Inverse-transform `spec`, window, overlap-add and emit one hop.
    pub fn synthesize(&mut self, spec: &[f32], output: &mut [f32]) {
        assert_eq!(output.len(), HOP);
        self.fft.inverse(spec, &mut self.frame);
        for (f, &w) in self.frame.iter_mut().zip(&self.win) {
            let v = *f * w;
            *f = if v.abs() < FLUSH { 0.0 } else { v };
        }
        for ((o, &f), &p) in output.iter_mut().zip(&self.frame[..HOP]).zip(&self.ola) {
            *o = f + p;
        }
        self.ola.copy_from_slice(&self.frame[HOP..]);
    }

    /// Everything that carries state between hops, for the subnormal checks.
    pub fn state(&self) -> impl Iterator<Item = f32> + '_ {
        self.analysis.iter().chain(&self.ola).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; 2 * N_BINS];
        for k in 0..N_BINS {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (n, &v) in x.iter().enumerate() {
                let a = -core::f64::consts::TAU * (k * n % N_FFT) as f64 / N_FFT as f64;
                re += v as f64 * a.cos();
                im += v as f64 * a.sin();
            }
            out[2 * k] = re as f32;
            out[2 * k + 1] = im as f32;
        }
        out
    }

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut r = super::super::format::Rng::new(seed);
        (0..n).map(|_| r.uniform()).collect()
    }

    #[test]
    fn forward_matches_naive_dft() {
        let x = noise(N_FFT, 1);
        let mut fft = Rfft::new();
        let mut spec = vec![0.0; 2 * N_BINS];
        fft.forward(&x, &mut spec);
        let want = naive_dft(&x);
        let err: f32 = spec
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let peak: f32 = want.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(err / peak < 1e-5, "rel err {}", err / peak);
    }

    #[test]
    fn inverse_round_trips() {
        let x = noise(N_FFT, 2);
        let mut fft = Rfft::new();
        let mut spec = vec![0.0; 2 * N_BINS];
        let mut y = vec![0.0; N_FFT];
        fft.forward(&x, &mut spec);
        fft.inverse(&spec, &mut y);
        let err = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 1e-5, "max err {err}");
    }

    #[test]
    fn window_is_cola_squared() {
        let w = window();
        for n in 0..HOP {
            let s = w[n] * w[n] + w[n + HOP] * w[n + HOP];
            assert!((s - 1.0).abs() < 1e-6, "n={n} s={s}");
        }
    }

    #[test]
    fn identity_stft_delays_by_exactly_one_hop() {
        let x = noise(HOP * 20, 3);
        let mut stft = Stft::new();
        let mut y = vec![0.0; x.len()];
        let mut spec = vec![0.0; 2 * N_BINS];
        for (i, o) in x.chunks(HOP).zip(y.chunks_mut(HOP)) {
            stft.analyze(i);
            spec.copy_from_slice(&stft.spec);
            stft.synthesize(&spec, o);
        }
        for n in HOP..x.len() {
            assert!(
                (y[n] - x[n - HOP]).abs() < 2e-6,
                "n={n}: {} vs {}",
                y[n],
                x[n - HOP]
            );
        }
    }
}
