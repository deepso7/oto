//! Deterministic synthetic test signals at [`crate::SAMPLE_RATE`]. Not for the realtime path.

use std::f32::consts::TAU;

const SR: f32 = crate::SAMPLE_RATE as f32;

/// Small deterministic PRNG (xorshift32), so signals are identical on every run and platform.
#[derive(Clone, Debug)]
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    /// Uniform in [-1, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

/// Sine at `freq` Hz with peak `amp`.
pub fn sine(freq: f32, amp: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (TAU * freq * i as f32 / SR).sin())
        .collect()
}

/// White noise, uniform with peak `amp` (RMS = amp / √3).
pub fn white_noise(amp: f32, n: usize, seed: u32) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| amp * r.uniform()).collect()
}

/// Pink (−3 dB/octave) noise with RMS `rms`, via Paul Kellet's filter. Closer to real room,
/// fan and traffic noise than white noise.
pub fn pink_noise(rms_target: f32, n: usize, seed: u32) -> Vec<f32> {
    let mut r = Rng::new(seed);
    let mut b = [0.0f32; 7];
    let mut out: Vec<f32> = (0..n)
        .map(|_| {
            let w = r.uniform();
            b[0] = 0.99886 * b[0] + w * 0.0555179;
            b[1] = 0.99332 * b[1] + w * 0.0750759;
            b[2] = 0.96900 * b[2] + w * 0.153_852;
            b[3] = 0.86650 * b[3] + w * 0.3104856;
            b[4] = 0.55000 * b[4] + w * 0.5329522;
            b[5] = -0.7616 * b[5] - w * 0.0168980;
            let p = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362;
            b[6] = w * 0.115926;
            p
        })
        .collect();
    let g = rms_target / rms(&out).max(1e-12);
    out.iter_mut().for_each(|x| *x *= g);
    out
}

/// A voiced, speech-like signal: a glottal-ish harmonic series with a gliding pitch (95–210 Hz),
/// three slowly moving formants, and a syllabic envelope with ~4 syllables per second and short
/// pauses. Peak is about 0.5. Not intelligible, but it has speech's spectral and temporal shape,
/// which is what the denoisers key on.
pub fn speech_like(n: usize, seed: u32) -> Vec<f32> {
    let mut r = Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    // Syllable state: length, position, pitch target, formant set.
    let mut syl_len = 0usize;
    let mut syl_pos = 0usize;
    let mut silent = false;
    let mut f0_a = 120.0f32;
    let mut f0_b = 140.0f32;
    let mut formants = [700.0f32, 1200.0, 2600.0];
    for _ in 0..n {
        if syl_pos >= syl_len {
            syl_pos = 0;
            silent = !silent && r.uniform() < -0.4; // occasional pause after a syllable
            syl_len = if silent {
                (SR * (0.15 + 0.07 * r.uniform())) as usize
            } else {
                (SR * (0.16 + 0.06 * r.uniform())) as usize
            };
            f0_a = f0_b;
            f0_b = 150.0 + 55.0 * r.uniform();
            formants = [
                550.0 + 250.0 * r.uniform(),
                1500.0 + 600.0 * r.uniform(),
                2600.0 + 300.0 * r.uniform(),
            ];
        }
        let t = syl_pos as f32 / syl_len as f32;
        syl_pos += 1;
        if silent {
            out.push(0.0);
            continue;
        }
        let f0 = f0_a + (f0_b - f0_a) * t;
        phase = (phase + f0 / SR).fract();
        // Raised-sine syllable envelope.
        let env = (std::f32::consts::PI * t).sin().powf(0.7);
        let mut s = 0.0;
        let mut h = 1;
        while (h as f32) * f0 < 7_000.0 {
            let fh = h as f32 * f0;
            // -6 dB/oct source tilt, then formant resonances.
            let mut g = 1.0 / h as f32;
            for (k, &fc) in formants.iter().enumerate() {
                let bw = 80.0 + 60.0 * k as f32;
                g *= 1.0 + 6.0 / (1.0 + ((fh - fc) / bw).powi(2));
            }
            s += g * (TAU * phase * h as f32).sin();
            h += 1;
        }
        out.push(s * env);
    }
    let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-9);
    out.iter_mut().for_each(|x| *x *= 0.5 / peak);
    out
}

/// The benchmark input: [`speech_like`] plus pink noise at −34 dBFS RMS (about 15 dB SNR).
pub fn noisy_speech(n: usize, seed: u32) -> Vec<f32> {
    let speech = speech_like(n, seed);
    let noise = pink_noise(0.02, n, seed.wrapping_mul(2_654_435_761));
    speech.iter().zip(&noise).map(|(s, n)| s + n).collect()
}

/// Root mean square.
pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt() as f32
}

/// Amplitude ratio in dB.
pub fn db(ratio: f32) -> f32 {
    20.0 * ratio.max(1e-12).log10()
}
