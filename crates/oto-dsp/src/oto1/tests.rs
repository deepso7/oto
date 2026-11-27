use super::fft::Stft;
use super::format::{DTYPE_F32, Otow, OtowWriter, Rng};
use super::*;
use crate::{Denoiser, FrameAdapter};

fn noise(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.uniform() * amp).collect()
}

/// Speech-ish test input: noise bursts with a varying envelope, so features move.
fn input(frames: usize, seed: u64) -> Vec<f32> {
    let mut x = noise(frames * HOP, seed, 0.3);
    for (i, v) in x.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        *v *= 0.5 + 0.5 * (core::f32::consts::TAU * 3.0 * t).sin();
        *v += 0.2 * (core::f32::consts::TAU * 220.0 * t).sin();
    }
    x
}

fn run(m: &mut Oto1, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for (i, o) in x.chunks(HOP).zip(y.chunks_mut(HOP)) {
        m.process_frame(i, o);
    }
    y
}

fn err_db(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    if num == 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * (num / den).log10()
    }
}

#[test]
fn erb_bands_cover_the_spectrum() {
    let e = erb_band_edges(32);
    assert_eq!(e.len(), 33);
    assert_eq!((e[0], e[32]), (0, N_BINS));
    assert!(e.windows(2).all(|w| w[1] >= w[0] + 2), "{e:?}");
    // Pinned so a change here is deliberate (the exporter computes the same list).
    assert_eq!(
        e,
        [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 33, 39, 45, 53, 62, 73, 85, 99, 116,
            134, 156, 182, 211, 245, 284, 329, 381, 442, 513
        ]
    );
}

#[test]
fn random_models_load_and_report_sizes() {
    for arch in [Arch::A, Arch::B] {
        let bytes = random_otow(arch, 7);
        let m = Oto1::from_otow(&bytes).unwrap();
        assert_eq!(m.arch(), arch);
        println!(
            "{arch:?}: {} params, {} MAC/frame ({:.1} MMAC/s), .otow f16 {} bytes",
            m.params(),
            m.macs_per_frame(),
            m.macs_per_frame() as f64 * SAMPLE_RATE as f64 / HOP as f64 / 1e6,
            bytes.len()
        );
        assert!(bytes.len() < 2 * m.params() + 64 * 1024);
    }
}

#[test]
fn simd_backends_match_scalar_end_to_end() {
    let x = input(300, 11);
    for arch in [Arch::A, Arch::B] {
        let bytes = random_otow(arch, 3);
        let mut reference = Oto1::from_otow_with(&bytes, Backend::Scalar).unwrap();
        let want = run(&mut reference, &x);
        assert!(
            want.iter().map(|v| v * v).sum::<f32>() > 1e-3,
            "{arch:?}: silent output"
        );
        for be in Backend::all() {
            let mut m = Oto1::from_otow_with(&bytes, be).unwrap();
            let got = run(&mut m, &x);
            let db = err_db(&got, &want);
            println!("{arch:?} {be:?} vs Scalar: {db:.1} dB");
            assert!(db <= -90.0, "{arch:?} {be:?}: {db:.1} dB");
        }
    }
}

#[test]
fn reset_reproduces_a_fresh_model() {
    let x = input(40, 5);
    for arch in [Arch::A, Arch::B] {
        let mut m = Oto1::random(arch, 1);
        let a = run(&mut m, &x);
        m.reset();
        let b = run(&mut m, &x);
        assert_eq!(a, b, "{arch:?}");
    }
}

/// The STFT with an identity "model", as a Denoiser: what the chain does around any Oto-1 net.
struct Identity(Stft, Vec<f32>);

impl Denoiser for Identity {
    fn frame_size(&self) -> usize {
        HOP
    }
    fn latency(&self) -> usize {
        HOP
    }
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        self.0.analyze(input);
        self.1.copy_from_slice(&self.0.spec);
        self.0.synthesize(&self.1, output);
    }
}

#[test]
fn chain_delay_is_exactly_1024_samples() {
    // Reported.
    for arch in [Arch::A, Arch::B] {
        let fa = FrameAdapter::new(Oto1::random(arch, 2));
        assert_eq!(fa.latency(), 1024);
        let ms = fa.latency() as f64 * 1000.0 / SAMPLE_RATE as f64;
        assert!((ms - 21.333).abs() < 0.001, "{ms}");
    }
    // Measured, through the same STFT and FrameAdapter, at odd block sizes.
    let mut fa = FrameAdapter::new(Identity(Stft::new(), vec![0.0; 2 * N_BINS]));
    let n = 8000;
    let at = 3001;
    let mut x = vec![0.0f32; n];
    x[at] = 1.0;
    let mut y = vec![0.0f32; n];
    let mut i = 0;
    for b in [1usize, 128, 333, 512, 7, 1000].iter().cycle() {
        if i >= n {
            break;
        }
        let e = (i + b).min(n);
        fa.process(&x[i..e], &mut y[i..e]);
        i = e;
    }
    let peak = y
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .unwrap();
    assert_eq!(peak.0, at + 1024);
    assert!((peak.1 - 1.0).abs() < 1e-6);
    let leak: f32 = y
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != at + 1024)
        .map(|(_, v)| v.abs())
        .sum();
    assert!(leak < 1e-5, "{leak}");
}

#[test]
fn digital_silence_stays_silent_and_subnormal_free() {
    for arch in [Arch::A, Arch::B] {
        for be in Backend::all() {
            let mut m = Oto1::from_otow_with(&random_otow(arch, 9), be).unwrap();
            let mut out = vec![1.0f32; HOP];
            let zero = vec![0.0f32; HOP];
            // From reset: exact zeros in, exact zeros out (every path multiplies the input).
            for _ in 0..50 {
                m.process_frame(&zero, &mut out);
                assert!(
                    out.iter().all(|&v| v == 0.0),
                    "{arch:?} {be:?}: silence not silent"
                );
            }
            // Signal, then a decaying tail into tiny and subnormal input, then silence.
            let x = input(30, 4);
            for c in x.chunks(HOP) {
                m.process_frame(c, &mut out);
            }
            let mut tiny = vec![0.0f32; HOP];
            for (i, v) in tiny.iter_mut().enumerate() {
                *v = if i % 2 == 0 { 1e-30 } else { -3e-39 };
            }
            for _ in 0..20 {
                m.process_frame(&tiny, &mut out);
            }
            for f in 0..400 {
                m.process_frame(&zero, &mut out);
                assert!(
                    !m.state_has_subnormals(),
                    "{arch:?} {be:?}: subnormal state at silent frame {f}"
                );
                assert!(
                    out.iter()
                        .all(|v| v.is_finite() && (*v == 0.0 || v.abs() >= f32::MIN_POSITIVE))
                );
            }
            assert!(
                out.iter().all(|&v| v == 0.0),
                "{arch:?} {be:?}: silence tail not silent"
            );
        }
    }
}

#[test]
fn loader_rejects_mismatched_files() {
    // Wrong arch id for the tensors.
    let mut b = random_otow(Arch::A, 1);
    b[8] = 2;
    let digest = sha256(&b[64..]);
    b[32..64].copy_from_slice(&digest);
    let err = Oto1::from_otow(&b).err().unwrap();
    assert!(
        err.contains("missing tensor") || err.contains("shape"),
        "{err}"
    );

    // Extra tensor.
    let o = Otow::parse(&random_otow(Arch::B, 1)).unwrap();
    let mut w = OtowWriter::new(2, DTYPE_F32, 0, &o.config);
    for t in &o.tensors {
        w.add(&t.name, &t.shape, t.data.clone());
    }
    w.add("stray", &[1], vec![0.0]);
    let err = Oto1::from_otow(&w.finish()).err().unwrap();
    assert!(err.contains("stray"), "{err}");

    // f32 file with the same weights loads and matches the f16 one exactly (f16 → f32 is exact).
    let mut w = OtowWriter::new(2, DTYPE_F32, 0, &o.config);
    for t in &o.tensors {
        w.add(&t.name, &t.shape, t.data.clone());
    }
    let x = input(10, 1);
    let a = run(&mut Oto1::from_otow(&w.finish()).unwrap(), &x);
    let b = run(&mut Oto1::from_otow(&random_otow(Arch::B, 1)).unwrap(), &x);
    assert_eq!(a, b);

    // Config that contradicts the kernel.
    let mut w = OtowWriter::new(1, DTYPE_F32, 0, r#"{"arch":"oto1-a","n_erb":24}"#);
    w.add("x", &[1], vec![0.0]);
    assert!(
        Oto1::from_otow(&w.finish())
            .err()
            .unwrap()
            .contains("n_erb")
    );
}
