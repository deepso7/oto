//! Nearfield v2.3 `fast` (Ground State Foundry, Apache-2.0) through ONNX Runtime: Voice
//! Isolation. Behind the `nearfield` feature (which needs `onnx`): like FastEnhancer, its
//! weights are not cleared to ship yet (`MODELS.toml`: `ship_in_release = false`), so release
//! builds (`scripts/macos/release.sh`, `dfn3,onnx`) leave it out.
//!
//! Upstream runs at 16 kHz (`src/nearfield/core.py` at tag v2.3.0); this is a sample-for-sample
//! port of its streaming STFT, lookahead and `Post` stage, bridged to 48 kHz by polyphase FIR
//! resamplers that need no buffering beyond Oto's 8 ms frame.

use crate::Denoiser;
use crate::onnx::{err, session};
use ort::session::Session;
use ort::value::TensorRef;
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

static NEARFIELD_FAST: &[u8] = include_bytes!("../models/nearfield_v2_3_fast.onnx");

/// Lowpass taps shared by the 3:1 decimator and the 1:3 interpolator. Odd, so the filter is
/// linear-phase with a whole-sample group delay of `(TAPS - 1) / 2` at 48 kHz.
const RS_TAPS: usize = 95;
const RS_DELAY: usize = (RS_TAPS - 1) / 2;
/// −6 dB point of the lowpass, Hz. With β = 7 (≈70 dB stopband) and 95 taps the transition
/// band is about ±1.1 kHz, so the stopband starts just above 8 kHz, the 16 kHz Nyquist.
const RS_CUTOFF_HZ: f64 = 7_000.0;
const RS_KAISER_BETA: f64 = 7.0;
/// Taps per interpolator phase (`ceil(RS_TAPS / 3)`).
const RS_PHASE_TAPS: usize = RS_TAPS.div_ceil(3);

/// Modified Bessel function of the first kind, order 0 (power series), for the Kaiser window.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
    while term > sum * 1e-12 {
        term *= (x / (2.0 * k)).powi(2);
        sum += term;
        k += 1.0;
    }
    sum
}

/// Kaiser-windowed sinc lowpass at 48 kHz, unity DC gain.
fn lowpass_taps() -> [f64; RS_TAPS] {
    let fc = RS_CUTOFF_HZ / crate::SAMPLE_RATE as f64;
    let c = RS_DELAY as f64;
    let i0b = bessel_i0(RS_KAISER_BETA);
    let mut h = [0.0; RS_TAPS];
    for (k, v) in h.iter_mut().enumerate() {
        let t = k as f64 - c;
        let sinc = if t == 0.0 {
            2.0 * fc
        } else {
            (std::f64::consts::TAU * fc * t).sin() / (std::f64::consts::PI * t)
        };
        let r = t / c;
        *v = sinc * bessel_i0(RS_KAISER_BETA * (1.0 - r * r).max(0.0).sqrt()) / i0b;
    }
    let sum: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= sum);
    h
}

/// 48 → 16 kHz: linear-phase FIR lowpass, evaluated only at every third sample. Allocation-free
/// after construction; delay [`RS_DELAY`] samples at 48 kHz.
struct Decimator {
    h: [f32; RS_TAPS],
    /// The last `RS_TAPS - 1` input samples, then the current block.
    buf: Vec<f32>,
}

impl Decimator {
    fn new(block48: usize) -> Self {
        Self {
            h: lowpass_taps().map(|v| v as f32),
            buf: vec![0.0; RS_TAPS - 1 + block48],
        }
    }

    /// `input.len() == 3 * output.len()`.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let hist = RS_TAPS - 1;
        self.buf[hist..].copy_from_slice(input);
        for (m, y) in output.iter_mut().enumerate() {
            // y[m] = Σ h[k] x[3m - k]; buf index of x[j] is hist + j. Symmetric taps, so the
            // window can run forwards.
            let end = hist + 3 * m + 1;
            *y = self.buf[end - RS_TAPS..end]
                .iter()
                .zip(self.h.iter().rev())
                .map(|(x, h)| x * h)
                .sum();
        }
        let n = self.buf.len();
        self.buf.copy_within(n - hist.., 0);
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
    }
}

/// 16 → 48 kHz: zero-stuffing by 3 and the same lowpass, as three polyphase branches (each
/// normalised to unity DC gain). Allocation-free after construction; delay [`RS_DELAY`] samples
/// at 48 kHz.
struct Interpolator {
    /// `phases[p][j]` multiplies `y[(n - p) / 3 - j]` for output `n ≡ p (mod 3)`.
    phases: [[f32; RS_PHASE_TAPS]; 3],
    /// The last `RS_PHASE_TAPS - 1` input samples, then the current block.
    buf: Vec<f32>,
}

impl Interpolator {
    fn new(block16: usize) -> Self {
        let h = lowpass_taps();
        let mut phases = [[0.0f64; RS_PHASE_TAPS]; 3];
        for (k, &v) in h.iter().enumerate() {
            phases[k % 3][k / 3] = v;
        }
        let phases = phases.map(|p| {
            let s: f64 = p.iter().sum();
            p.map(|v| (v / s) as f32)
        });
        Self {
            phases,
            buf: vec![0.0; RS_PHASE_TAPS - 1 + block16],
        }
    }

    /// `output.len() == 3 * input.len()`.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let hist = RS_PHASE_TAPS - 1;
        self.buf[hist..].copy_from_slice(input);
        for (n, z) in output.iter_mut().enumerate() {
            // z[n] = 3 Σ_k h[k] u[n - k], u[3m] = y[m]: only k ≡ n (mod 3) hit a sample.
            let p = n % 3;
            let m = n / 3; // y index of k = p
            let mut acc = 0.0;
            for (j, &c) in self.phases[p].iter().enumerate() {
                acc += c * self.buf[hist + m - j];
            }
            *z = acc;
        }
        let n = self.buf.len();
        self.buf.copy_within(n - hist.., 0);
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
    }
}

/// Upstream's `Post` spectral stage (`blips` and `pause_db`): works on the model's own output
/// and never mixes the input back in. Zero added delay. Upstream's `level_db` AGC is left off:
/// call apps run their own AGC and the two would fight. Upstream's `tilt_db` is left at 0.
struct Post {
    /// Previous frame's per-bin gain ratio, for blip detection.
    gp: Vec<f32>,
    g: Vec<f32>,
    top: f32,
    gate: f32,
    hold: u32,
    /// Most the pause gate attenuates, dB (the strength).
    pause_db: f32,
}

impl Post {
    const REL: f32 = 0.904_837_4; // exp(-8 / 80): an ~80 ms release at 8 ms hops
    const HOLD: u32 = 12;

    fn new() -> Self {
        Self {
            gp: vec![0.0; Nearfield::BINS],
            g: vec![0.0; Nearfield::BINS],
            top: -60.0,
            gate: 1.0,
            hold: 0,
            pause_db: 12.0,
        }
    }

    fn reset(&mut self) {
        self.gp.fill(0.0);
        self.top = -60.0;
        self.gate = 1.0;
        self.hold = 0;
    }

    /// `y`: the model's spectrum of one frame, `x`: the input spectrum of the same frame, both
    /// interleaved re/im.
    fn apply(&mut self, y: &mut [f32], x: &[f32]) {
        let mag =
            |ri: &[f32], i: usize| (ri[2 * i] * ri[2 * i] + ri[2 * i + 1] * ri[2 * i + 1]).sqrt();
        let n = Nearfield::BINS;
        // Blips: a bin suddenly louder (relative to the input) than its neighbours and its own
        // past is an isolated leftover.
        for i in 0..n {
            self.g[i] = (mag(y, i) / (mag(x, i) + 1e-9)).min(2.0);
        }
        for i in 0..n {
            let (l, r) = (i.saturating_sub(1), (i + 1).min(n - 1));
            let reference = self.gp[l]
                .max(self.gp[i])
                .max(self.gp[r])
                .max(self.g[l])
                .max(self.g[r])
                * 1.5
                + 0.03;
            let g = self.g[i];
            if g > reference {
                let s = reference / g.max(1e-9);
                y[2 * i] *= s;
                y[2 * i + 1] *= s;
            }
        }
        std::mem::swap(&mut self.gp, &mut self.g);
        // Pauses: between the caller's words, ease down by up to `pause_db`.
        let e: f32 = y.iter().map(|v| v * v).sum();
        let n_fft = Nearfield::N_FFT as f32;
        let db = 10.0 * (e / (n_fft * n_fft * 0.375) * 2.0 + 1e-12).log10();
        self.top = if db > self.top { db } else { self.top - 0.004 };
        let thr = (self.top - 32.0).max(-70.0);
        let mut want = 1.0;
        if db >= thr {
            self.hold = Self::HOLD;
        } else if self.hold > 0 {
            self.hold -= 1;
        } else {
            want = 10f32.powf((db - thr).max(-self.pause_db) / 20.0);
        }
        self.gate = if want >= self.gate {
            want
        } else {
            self.gate * Self::REL + want * (1.0 - Self::REL)
        };
        if self.gate < 1.0 {
            y.iter_mut().for_each(|v| *v *= self.gate);
        }
    }
}

/// Nearfield v2.3 `fast` (Ground State Foundry, Apache-2.0): Voice Isolation. Keeps the voice
/// nearest the mic and removes noise *and* other voices. 8 ms frames (384 samples at 48 kHz).
///
/// The model runs at 16 kHz on a 512-point STFT with a 128-sample hop and two frames of
/// lookahead, as upstream's `core.py`. Oto's 48 kHz frame is exactly three of its hops, so a
/// 3:1 polyphase FIR decimator and 1:3 interpolator (linear phase, 95 taps) bridge the rates
/// with no extra buffering. Delay (sample-synchronous): 384 (STFT overlap) + 256 (lookahead) at
/// 16 kHz plus 2 × 47 for the filters = [`Nearfield::LATENCY`] = 2014 samples at 48 kHz.
///
/// Fails closed: an inference error (or a non-finite result) outputs exact silence for that
/// frame, never the input.
pub struct Nearfield {
    session: Session,
    in_names: [String; 3],
    out_names: [String; 2],
    state: Vec<f32>,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    /// The last `N_FFT` samples at 16 kHz.
    frame: Vec<f32>,
    time: Vec<f32>,
    spec: Vec<Complex32>,
    /// The current frame's spectrum and the one `LOOKAHEAD` frames back, interleaved re/im.
    x_ri: Vec<f32>,
    xd_ri: Vec<f32>,
    /// The last `LOOKAHEAD` spectra, oldest at `hist_pos`.
    hist: Vec<f32>,
    hist_pos: usize,
    y_ri: Vec<f32>,
    ola: Vec<f32>,
    scratch_fwd: Vec<Complex32>,
    scratch_inv: Vec<Complex32>,
    frames: u64,
    post: Post,
    dec: Decimator,
    interp: Interpolator,
    in16: Vec<f32>,
    out16: Vec<f32>,
    /// Injected failures still to come ([`Denoiser::inject_fault`]).
    faults: u32,
    /// Frames that failed (inference error or non-finite output), for diagnostics.
    pub errors: u64,
    /// Tests: skip the network and pass the delayed input spectrum through (no post stage),
    /// which leaves exactly the STFT, lookahead and resampler delay.
    #[cfg(test)]
    identity: bool,
}

impl Nearfield {
    const HOP: usize = 128;
    const N_FFT: usize = 512;
    const BINS: usize = Self::N_FFT / 2 + 1;
    const LOOKAHEAD: usize = 2;
    const STATE: [usize; 3] = [4, 64, 128];
    /// Samples at 48 kHz per frame: three 16 kHz hops.
    const FRAME: usize = Self::HOP * 3;
    /// Sample-synchronous delay at 48 kHz: STFT overlap and lookahead at 16 kHz (×3) plus both
    /// resampling filters. Matches the measured delay (`tests::nearfield_*`).
    pub const LATENCY: usize =
        3 * (Self::N_FFT - Self::HOP + Self::LOOKAHEAD * Self::HOP) + 2 * RS_DELAY;

    /// Loads the bundled weights. Off the RT thread.
    pub fn new() -> Result<Self, String> {
        let session = session(NEARFIELD_FAST)?;
        let find = |names: &[&str], want: &str| -> Result<String, String> {
            names
                .iter()
                .find(|n| **n == want)
                .map(|n| n.to_string())
                .ok_or_else(|| format!("Nearfield: no input {want:?} (inputs: {names:?})"))
        };
        let ins: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        let in_names = [find(&ins, "x")?, find(&ins, "xd")?, find(&ins, "state")?];
        let outs: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if outs.len() < 2 {
            return Err(format!("Nearfield: expected 2 outputs, got {outs:?}"));
        }
        let out_names = [outs[0].clone(), outs[1].clone()];
        let n = Self::N_FFT;
        let window = (0..n)
            .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos()) as f32)
            .collect();
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);
        Ok(Self {
            session,
            in_names,
            out_names,
            state: vec![0.0; Self::STATE.iter().product()],
            window,
            frame: vec![0.0; n],
            time: vec![0.0; n],
            spec: fft.make_output_vec(),
            x_ri: vec![0.0; Self::BINS * 2],
            xd_ri: vec![0.0; Self::BINS * 2],
            hist: vec![0.0; Self::BINS * 2 * Self::LOOKAHEAD],
            hist_pos: 0,
            y_ri: vec![0.0; Self::BINS * 2],
            ola: vec![0.0; n],
            scratch_fwd: fft.make_scratch_vec(),
            scratch_inv: ifft.make_scratch_vec(),
            fft,
            ifft,
            frames: 0,
            post: Post::new(),
            dec: Decimator::new(Self::FRAME),
            interp: Interpolator::new(Self::HOP),
            in16: vec![0.0; Self::HOP],
            out16: vec![0.0; Self::HOP],
            faults: 0,
            errors: 0,
            #[cfg(test)]
            identity: false,
        })
    }

    fn infer(&mut self) -> Result<(), String> {
        if self.faults > 0 {
            self.faults -= 1;
            return Err("injected inference error".into());
        }
        let b = Self::BINS;
        let out = self
            .session
            .run(ort::inputs![
                self.in_names[0].as_str() => TensorRef::from_array_view(([1usize, b, 2], &self.x_ri[..])).map_err(err)?,
                self.in_names[1].as_str() => TensorRef::from_array_view(([1usize, b, 2], &self.xd_ri[..])).map_err(err)?,
                self.in_names[2].as_str() => TensorRef::from_array_view((Self::STATE, &self.state[..])).map_err(err)?,
            ])
            .map_err(err)?;
        let (_, y) = out[self.out_names[0].as_str()]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        let (_, st) = out[self.out_names[1].as_str()]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        if y.len() != self.y_ri.len() || st.len() != self.state.len() {
            return Err("Nearfield: unexpected output shape".into());
        }
        if !y.iter().chain(st.iter()).all(|v| v.is_finite()) {
            // A poisoned recurrent state would stick; start the state afresh.
            self.state.fill(0.0);
            return Err("Nearfield: non-finite output".into());
        }
        self.y_ri.copy_from_slice(y);
        self.state.copy_from_slice(st);
        Ok(())
    }

    /// One 128-sample hop at 16 kHz, `in16` → `out16`. Returns `false` (and leaves silence in
    /// `out16`) when the frame failed.
    fn hop(&mut self) -> bool {
        let (h, n, b2) = (Self::HOP, Self::N_FFT, Self::BINS * 2);
        self.frame.copy_within(h.., 0);
        self.frame[n - h..].copy_from_slice(&self.in16);
        for ((t, &x), &w) in self.time.iter_mut().zip(&self.frame).zip(&self.window) {
            *t = x * w;
        }
        let ok = self
            .fft
            .process_with_scratch(&mut self.time, &mut self.spec, &mut self.scratch_fwd)
            .is_ok();
        for (ri, c) in self.x_ri.as_chunks_mut::<2>().0.iter_mut().zip(&self.spec) {
            *ri = [c.re, c.im];
        }
        // xd: the spectrum LOOKAHEAD frames back (zeros until there is one), then remember x.
        let slot = self.hist_pos * b2..(self.hist_pos + 1) * b2;
        self.xd_ri.copy_from_slice(&self.hist[slot.clone()]);
        self.hist[slot].copy_from_slice(&self.x_ri);
        self.hist_pos = (self.hist_pos + 1) % Self::LOOKAHEAD;
        self.frames += 1;

        #[cfg(test)]
        if self.identity {
            self.y_ri.copy_from_slice(&self.xd_ri);
        }
        #[cfg(test)]
        let inferred = ok && (self.identity || self.infer().is_ok());
        #[cfg(not(test))]
        let inferred = ok && self.infer().is_ok();
        // The model's output for the first LOOKAHEAD frames belongs to no input frame yet.
        let emit = inferred && self.frames > Self::LOOKAHEAD as u64;
        let mut good = inferred;
        #[cfg(test)]
        let post = !self.identity;
        #[cfg(not(test))]
        let post = true;
        if emit {
            if post {
                self.post.apply(&mut self.y_ri, &self.xd_ri);
            }
            for (c, [re, im]) in self.spec.iter_mut().zip(self.y_ri.as_chunks::<2>().0) {
                *c = Complex32::new(*re, *im);
            }
            self.spec[0].im = 0.0;
            self.spec[Self::BINS - 1].im = 0.0;
            if self
                .ifft
                .process_with_scratch(&mut self.spec, &mut self.time, &mut self.scratch_inv)
                .is_ok()
            {
                // Unnormalised inverse (× N_FFT); periodic Hann² at 75% overlap sums to 1.5.
                let scale = 1.0 / (n as f32 * 1.5);
                for ((o, &t), &w) in self.ola.iter_mut().zip(&self.time).zip(&self.window) {
                    *o += t * w * scale;
                }
            } else {
                good = false;
            }
        }
        if good {
            self.out16.copy_from_slice(&self.ola[..h]);
        } else {
            self.out16.fill(0.0);
        }
        self.ola.copy_within(h.., 0);
        self.ola[n - h..].fill(0.0);
        good
    }
}

impl Denoiser for Nearfield {
    fn frame_size(&self) -> usize {
        Self::FRAME
    }

    fn latency(&self) -> usize {
        Self::LATENCY
    }

    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        self.dec.process(input, &mut self.in16);
        if self.hop() {
            self.interp.process(&self.out16, output);
        } else {
            // Fail closed: silence for this frame, and the interpolator starts the next one
            // from silence too rather than from the samples before the failure.
            self.errors += 1;
            self.interp.process(&self.out16, output);
            output.fill(0.0);
        }
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
        self.frame.fill(0.0);
        self.hist.fill(0.0);
        self.hist_pos = 0;
        self.ola.fill(0.0);
        self.frames = 0;
        self.post.reset();
        self.dec.reset();
        self.interp.reset();
        self.faults = 0;
    }

    /// Strength maps to how far the pauses between words are pushed down. The mask itself is
    /// the model's; the input is never mixed back in, so every strength removes other voices.
    fn set_max_attenuation_db(&mut self, db: f32) -> bool {
        self.post.pause_db = if db <= 12.0 {
            6.0
        } else if db <= 24.0 {
            12.0
        } else {
            18.0
        };
        true
    }

    fn inject_fault(&mut self, frames: u32) -> bool {
        self.faults = self.faults.saturating_add(frames);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{db, rms, speech_like, white_noise};

    const SR: usize = crate::SAMPLE_RATE as usize;

    fn resample_round_trip(x: &[f32]) -> Vec<f32> {
        let mut d = Decimator::new(384);
        let mut i = Interpolator::new(128);
        let (mut mid, mut y) = (vec![0.0; 128], vec![0.0; x.len()]);
        for (a, b) in x
            .as_chunks::<384>()
            .0
            .iter()
            .zip(y.as_chunks_mut::<384>().0.iter_mut())
        {
            d.process(a, &mut mid);
            i.process(&mid, b);
        }
        y
    }

    fn sine(f: f32, n: usize) -> Vec<f32> {
        // f64 phase: in f32, 2π·f·k/SR loses ~−40 dB of precision over a second of samples.
        (0..n)
            .map(|k| 0.5 * (std::f64::consts::TAU * f as f64 * k as f64 / SR as f64).sin() as f32)
            .collect()
    }

    #[test]
    fn resamplers_are_a_pure_delay_in_the_passband_and_block_aliases() {
        // Impulse: the round trip peaks exactly 2 × 47 samples later.
        let mut imp = vec![0.0; 384 * 4];
        imp[500] = 1.0;
        let y = resample_round_trip(&imp);
        let peak = y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        assert_eq!(peak, 500 + 2 * RS_DELAY);
        // Passband: 200 Hz to 5 kHz kept within 0.5 dB and delayed by exactly 2 × 47.
        let n = 384 * 100;
        for f in [200.0, 1000.0, 3000.0, 5000.0] {
            let x = sine(f, n);
            let y = resample_round_trip(&x);
            let d = 2 * RS_DELAY;
            let g = db(rms(&y[SR / 2..]) / rms(&x[SR / 2 - d..n - d]));
            assert!(g.abs() < 0.5, "{f} Hz: {g:.2} dB");
            let err: Vec<f32> = (SR / 2..n).map(|k| y[k] - x[k - d]).collect();
            assert!(
                db(rms(&err) / rms(&x)) < -30.0,
                "{f} Hz is not a pure delay"
            );
        }
        // Stopband: what would alias into 0–8 kHz at 16 kHz is ≥ 60 dB down.
        for f in [9_000.0, 12_000.0, 20_000.0] {
            let x = sine(f, n);
            let y = resample_round_trip(&x);
            let g = db(rms(&y[SR / 2..]) / rms(&x));
            assert!(g < -60.0, "{f} Hz leaks at {g:.1} dB");
        }
    }

    fn run(m: &mut Nearfield, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; x.len() / 384 * 384];
        for (i, o) in x
            .as_chunks::<384>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<384>().0.iter_mut())
        {
            m.process_frame(i, o);
        }
        out
    }

    #[test]
    fn nearfield_signal_path_delay_is_its_reported_latency() {
        let mut m = Nearfield::new().unwrap();
        m.identity = true;
        assert_eq!(m.frame_size(), 384);
        assert_eq!(Nearfield::LATENCY, 2014);
        // Impulse: peak exactly LATENCY samples later.
        let mut imp = vec![0.0; 384 * 20];
        imp[1000] = 1.0;
        let y = run(&mut m, &imp);
        let peak = y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        assert_eq!(peak, 1000 + m.latency(), "impulse peak");
        // Any audio comes out as the resamplers' round trip of it (the STFT reconstructs
        // exactly), the STFT and lookahead delay later.
        m.reset();
        let x = white_noise(0.3, SR * 2, 5);
        let y = run(&mut m, &x);
        let want = resample_round_trip(&x);
        let d = m.latency() - 2 * RS_DELAY;
        let err: Vec<f32> = (SR / 2..y.len()).map(|k| y[k] - want[k - d]).collect();
        let e = db(rms(&err) / rms(&want[SR / 2 - d..y.len() - d]));
        assert!(e < -40.0, "identity path error {e:.1} dB");
    }

    #[test]
    fn nearfield_runs_resets_bit_identically_and_fails_closed() {
        let mut m = Nearfield::new().unwrap();
        eprintln!(
            "Nearfield inputs {:?}, outputs {:?}",
            m.in_names, m.out_names
        );
        let x = speech_like(SR * 2, 7);
        let first = run(&mut m, &x);
        assert!(first.iter().all(|v| v.is_finite()));
        assert!(
            rms(&first[SR..]) > 1e-3,
            "no output on clean speech-like audio"
        );
        m.reset();
        assert_eq!(
            first,
            run(&mut m, &x),
            "reset does not restore the start state"
        );
        assert_eq!(m.errors, 0);
        // Inference errors: exact silence for those frames, then it recovers.
        m.reset();
        let mut out = vec![0.0; x.len() / 384 * 384];
        for (k, (i, o)) in x
            .as_chunks::<384>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<384>().0.iter_mut())
            .enumerate()
        {
            if k == 100 {
                assert!(m.inject_fault(10));
            }
            m.process_frame(i, o);
        }
        assert_eq!(m.errors, 10);
        assert!(out[100 * 384..110 * 384].iter().all(|&v| v == 0.0));
        assert!(
            rms(&out[150 * 384..]) > 1e-3,
            "does not recover after errors"
        );
        // Strength is native at every setting.
        for db in [12.0, 24.0, f32::INFINITY] {
            assert!(m.set_max_attenuation_db(db));
        }
    }
}
