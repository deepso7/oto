//! Native timing of the random-weight Oto-1 candidates, per 512-sample (10.67 ms) hop.
//!
//! `cargo run --release -p oto-dsp --no-default-features --features oto1 --example oto1_bench`
//! Optional args: `racing_frames paced_seconds` (default `5000 5`). `--write` also writes the
//! random `.otow` files and native reference vectors (input and scalar-backend output, raw f32 LE,
//! 200 hops) to `target/oto1/`, which `web/demo/bench-node.mjs` checks the wasm builds against.
//! `--eval <model.otow> <in.f32> <out-prefix>` runs any `.otow` (e.g. a PyTorch export) instead.

use oto_dsp::Denoiser;
use oto_dsp::oto1::{Arch, Backend, HOP, Oto1, SAMPLE_RATE, random_otow};
use std::time::{Duration, Instant};

fn pct(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn input(n: usize) -> Vec<f32> {
    let mut s = 0x1234_5678u32;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let t = i as f32 / SAMPLE_RATE as f32;
            0.1 * ((s >> 8) as f32 / (1 << 24) as f32 - 0.5)
                + 0.2 * (std::f32::consts::TAU * 220.0 * t).sin()
        })
        .collect()
}

/// `--eval <model.otow> <in.f32> <out-prefix>`: run raw little-endian f32 input (a multiple of
/// 512 samples) through the kernel once per available backend, writing `<out-prefix>.<backend>.f32`
/// (e.g. `.scalar.f32`, `.neon.f32`). Used by `tools/train/tests/test_rust_parity.py`.
fn eval(otow: &str, input: &str, prefix: &str) {
    let model = std::fs::read(otow).unwrap();
    let raw = std::fs::read(input).unwrap();
    let x: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert!(
        x.len().is_multiple_of(HOP),
        "input must be a multiple of {HOP} samples"
    );
    for be in Backend::all() {
        let mut m = Oto1::from_otow_with(&model, be).unwrap_or_else(|e| panic!("{otow}: {e}"));
        let mut y = vec![0.0f32; x.len()];
        for (i, o) in x.chunks(HOP).zip(y.chunks_mut(HOP)) {
            m.process_frame(i, o);
        }
        let p = format!("{prefix}.{}.f32", format!("{be:?}").to_lowercase());
        std::fs::write(
            &p,
            y.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(),
        )
        .unwrap();
        println!("wrote {p}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--eval") {
        let [otow, input, prefix] = &args[i + 1..i + 4] else {
            panic!("--eval <model.otow> <in.f32> <out-prefix>")
        };
        return eval(otow, input, prefix);
    }
    let write = args.iter().any(|a| a == "--write");
    let nums: Vec<f64> = args.iter().filter_map(|a| a.parse().ok()).collect();
    let racing = nums.first().copied().unwrap_or(5000.0) as usize;
    let paced_s = nums.get(1).copied().unwrap_or(5.0);
    let hop = Duration::from_secs_f64(HOP as f64 / SAMPLE_RATE as f64);
    let audio = input(HOP * 1000);
    let mut out = vec![0.0f32; HOP];

    if write {
        std::fs::create_dir_all("target/oto1").unwrap();
        for (arch, name) in [(Arch::A, "a"), (Arch::B, "b")] {
            let p = format!("target/oto1/random-{name}.otow");
            std::fs::write(&p, random_otow(arch, 1)).unwrap();
            println!("wrote {p}");
            let x = &audio[..HOP * 200];
            let mut m = Oto1::from_otow_with(&random_otow(arch, 1), Backend::Scalar).unwrap();
            let mut y = vec![0.0f32; x.len()];
            for (i, o) in x.chunks(HOP).zip(y.chunks_mut(HOP)) {
                m.process_frame(i, o);
            }
            let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::write("target/oto1/ref-input.f32", bytes(x)).unwrap();
            std::fs::write(format!("target/oto1/ref-{name}-out.f32"), bytes(&y)).unwrap();
        }
    }

    println!("hop = {HOP} samples = {:.3} ms", hop.as_secs_f64() * 1e3);
    for arch in [Arch::A, Arch::B] {
        let bytes = random_otow(arch, 1);
        for be in Backend::all() {
            let mut m = Oto1::from_otow_with(&bytes, be).unwrap();
            let frames = audio.as_chunks::<HOP>().0.iter().cycle();
            for f in frames.clone().take(300) {
                m.process_frame(f, &mut out);
            }
            // Racing: back to back.
            let mut t = Vec::with_capacity(racing);
            let start = Instant::now();
            for f in frames.clone().take(racing) {
                let s = Instant::now();
                m.process_frame(f, &mut out);
                t.push(s.elapsed().as_secs_f64() * 1e3);
            }
            let total = start.elapsed().as_secs_f64() * 1e3;
            let mean = total / racing as f64;
            t.sort_by(f64::total_cmp);
            // Paced: one hop per real hop period, like an audio callback.
            let n = (paced_s / hop.as_secs_f64()) as usize;
            let mut p = Vec::with_capacity(n);
            let mut next = Instant::now();
            for f in frames.clone().take(n) {
                next += hop;
                let s = Instant::now();
                m.process_frame(f, &mut out);
                p.push(s.elapsed().as_secs_f64() * 1e3);
                if let Some(d) = next.checked_duration_since(Instant::now()) {
                    std::thread::sleep(d);
                }
            }
            p.sort_by(f64::total_cmp);
            println!(
                "{arch:?} {be:<8?} params {:>7} MAC/frame {:>8} ({:>5.1} MMAC/s) | racing mean {:.4} ms p50 {:.4} p99 {:.4} max {:.4} ({:.2}% core, {:.2} GMAC/s) | paced p50 {:.4} p95 {:.4} p99 {:.4} max {:.4}",
                m.params(),
                m.macs_per_frame(),
                m.macs_per_frame() as f64 / hop.as_secs_f64() / 1e6,
                mean,
                pct(&t, 0.5),
                pct(&t, 0.99),
                t[t.len() - 1],
                100.0 * mean / (hop.as_secs_f64() * 1e3),
                m.macs_per_frame() as f64 / (mean * 1e-3) / 1e9,
                pct(&p, 0.5),
                pct(&p, 0.95),
                pct(&p, 0.99),
                p[p.len() - 1],
            );
        }
    }
}
