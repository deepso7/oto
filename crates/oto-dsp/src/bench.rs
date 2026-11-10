//! Timing harness behind `oto bench`.

use crate::{Chain, Model, SAMPLE_RATE, signal};
use std::time::Instant;

/// Callback size the benchmark feeds the chain: 10 ms, one model frame for both backends.
pub const BENCH_BLOCK: usize = 480;

/// Result of [`bench`]. Times are for one full-chain call on a [`BENCH_BLOCK`]-sample block
/// (high-pass, denoise, limiter), measured back to back on the calling thread.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchReport {
    pub model: Model,
    /// Blocks timed, after warm-up.
    pub frames: usize,
    pub block: usize,
    pub p50_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
    /// Audio time divided by processing time. 10 means ten times faster than realtime.
    pub realtime_factor: f64,
    /// Mean share of one core the chain needs in realtime, in percent (100 / realtime_factor).
    pub cpu_percent: f64,
    /// [`Chain::latency`] in milliseconds: the algorithmic delay the chain adds.
    pub latency_ms: f64,
}

impl std::fmt::Display for BenchReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}: {} x {}-sample blocks, p50 {:.1} us, p99 {:.1} us, max {:.1} us, \
             {:.0}x realtime ({:.2}% of a core), latency {:.1} ms",
            self.model,
            self.frames,
            self.block,
            self.p50_us,
            self.p99_us,
            self.max_us,
            self.realtime_factor,
            self.cpu_percent,
            self.latency_ms
        )
    }
}

/// Benchmark `model` on `seconds` of deterministic synthetic noisy speech.
///
/// Panics if the model is not compiled in; use [`try_bench`] or [`Model::available`] to check.
pub fn bench(model: Model, seconds: f32) -> BenchReport {
    try_bench(model, seconds).unwrap_or_else(|e| panic!("bench {model:?}: {e}"))
}

/// [`bench`], returning an error instead of panicking when the model is unavailable.
pub fn try_bench(model: Model, seconds: f32) -> Result<BenchReport, String> {
    let mut chain = Chain::with_model(model)?;
    chain.set_max_attenuation_db(f32::INFINITY);
    let blocks = ((seconds.max(0.01) * SAMPLE_RATE as f32) as usize).div_ceil(BENCH_BLOCK);
    let input = signal::noisy_speech(blocks * BENCH_BLOCK, 0x070B_E7C4);
    let mut buf = [0.0f32; BENCH_BLOCK];

    // Warm caches and any lazy init with half a second that is not timed.
    for chunk in input.as_chunks::<BENCH_BLOCK>().0.iter().take(50) {
        buf.copy_from_slice(chunk);
        chain.process(&mut buf);
    }
    chain.reset();

    let mut times = Vec::with_capacity(blocks);
    for chunk in input.as_chunks::<BENCH_BLOCK>().0 {
        buf.copy_from_slice(chunk);
        let t = Instant::now();
        chain.process(&mut buf);
        times.push(t.elapsed().as_secs_f64() * 1e6);
        std::hint::black_box(&buf);
    }
    let total: f64 = times.iter().sum();
    times.sort_by(f64::total_cmp);
    let pct = |p: f64| times[((times.len() - 1) as f64 * p).round() as usize];
    let audio_us = (blocks * BENCH_BLOCK) as f64 / SAMPLE_RATE as f64 * 1e6;
    let rtf = audio_us / total.max(1e-9);
    Ok(BenchReport {
        model,
        frames: blocks,
        block: BENCH_BLOCK,
        p50_us: pct(0.50),
        p99_us: pct(0.99),
        max_us: *times.last().unwrap_or(&0.0),
        mean_us: total / blocks as f64,
        realtime_factor: rtf,
        cpu_percent: 100.0 / rtf,
        latency_ms: chain.latency() as f64 * 1000.0 / SAMPLE_RATE as f64,
    })
}
