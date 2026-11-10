//! The realtime path, independent of any audio API: capture stage (downmix → resample to 48 kHz →
//! chain → ring) and output stage (ring → drift-correcting resampler → device).
//!
//! Everything here is called from device callbacks, so after construction nothing allocates,
//! locks or blocks. Cross-thread state is atomics in [`Shared`].
//!
//! # Where the chain runs
//!
//! RNNoise runs inside the capture callback ([`Placement::Inline`]). DeepFilterNet3 allocates on
//! every frame, so it runs on a realtime DSP worker thread ([`Placement::Worker`]): the
//! callback pushes 48 kHz audio into one ring, wakes the worker and pulls processed audio from
//! a second ring, a fixed [`worker_reserve`] (one capture block + 2 ms) behind. When the worker
//! has not delivered in time the callback crossfades to the dry signal at the same delay and
//! counts a starvation; when it catches up, wet audio older than the current position is
//! discarded before fading back, so audio never jumps backwards.
//!
//! # Drift control and latency
//!
//! The mic and the sink run on different clocks, so the ring between them would slowly fill or
//! drain. The output stage measures the ring fill every callback and steers an adaptive resampler
//! (`rubato::Async`, polynomial) with a PI controller ([`DriftController`]) so the *mean* fill
//! sits at a small target:
//!
//! ```text
//! target = out_block + in_block / 2 + OUT_CHUNK + MARGIN_FRAMES      (frames at 48 kHz)
//! ```
//!
//! `out_block` must be available at every output callback; `in_block / 2` covers the producer's
//! sawtooth (fill is sampled at random phases of the input cycle); `OUT_CHUNK` is the resampler's
//! granularity; `MARGIN_FRAMES` (2 ms) absorbs callback jitter. With the default IO buffers
//! (256 frames in, 128 out, 48 kHz) that is 128 + 128 + 32 + 96 = 384 frames = 8 ms of ring
//! latency. Corrections are clamped to ±2000 ppm (≈3.5 cents), far below audibility, and real
//! crystals are within ±200 ppm, so steady-state corrections are tiny.
//!
//! The integrator only runs while the correction is not clamped (anti-windup), so a large
//! transient cannot wind it up and cause an undershoot afterwards.
//!
//! If the ring runs dry the output plays silence, re-primes to the target (counted as an
//! underrun) and fades back in over [`FADE_S`]. Excess latency, e.g. after the output stalled,
//! is handled in three bands, each counted as a resync when frames are dropped:
//!
//! - more than [`MAX_EXCESS`] (100 ms) over target: dropped at once, down to a fill that is safe
//!   at any phase of the input cycle, then faded in;
//! - a mean of more than [`TRIM_EXCESS`] (10 ms) over target across [`TRIM_WINDOW_S`]: the
//!   output fades out, exactly the mean excess is dropped, and it fades back in (a ~6 ms dip
//!   instead of tens of seconds of extra latency);
//! - less than that: bled off by the drift controller at up to 2000 ppm (under ~5 s).

use std::sync::Arc;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicU32, AtomicU64};
use std::time::{Duration, Instant};

use oto_dsp::filters::Delay;
use oto_dsp::{Chain, Model, SAMPLE_RATE};
use rtrb::{Consumer, Producer, RingBuffer};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Adjustable, Async, FixedAsync, PolynomialDegree, Resampler};

use crate::Levels;

/// Ring capacity in frames at 48 kHz (≈341 ms). Only the target fill is normally used.
pub const RING_CAPACITY: usize = 16_384;
/// Largest slab processed in one go; device blocks larger than this are split.
pub const MAX_BLOCK: usize = 4096;
/// Output resampler chunk, in output frames.
pub const OUT_CHUNK: usize = 32;
/// Input resampler chunk (only used when the mic is not at 48 kHz), in input frames.
pub const IN_CHUNK: usize = 64;
/// Safety margin in the ring target, frames at 48 kHz (2 ms).
pub const MARGIN_FRAMES: usize = 96;
/// Fill beyond `target + MAX_EXCESS` is dropped at once (100 ms).
pub const MAX_EXCESS: usize = 4800;
/// A mean fill beyond `target + TRIM_EXCESS` (10 ms) over [`TRIM_WINDOW_S`] is trimmed with a
/// short fade; anything smaller is left to the drift controller.
pub const TRIM_EXCESS: usize = 480;
/// Window over which the mean fill is measured for trimming, seconds. Long enough to average
/// the input block sawtooth of any realistic device.
pub const TRIM_WINDOW_S: f64 = 0.25;
/// Length of the fade applied when audio resumes after priming or a resync, seconds.
pub const FADE_S: f64 = 0.003;
/// Length of the fade-out before the streams stop (a model switch, say), seconds.
pub const STOP_FADE_S: f64 = 0.010;

/// [`Shared::stop_fade`] states.
pub const STOP_FADE_IDLE: u8 = 0;
/// Set by the control thread: the output callback fades to silence and holds it.
pub const STOP_FADE_REQUESTED: u8 = 1;
/// Set by the output callback once its output is silent.
pub const STOP_FADE_DONE: u8 = 2;

const RATE: f64 = SAMPLE_RATE as f64;

/// Mean ring fill to aim for, in frames at 48 kHz. Block sizes are in device frames.
pub fn target_fill(in_block: usize, in_rate: f64, out_block: usize, out_rate: f64) -> usize {
    let in48 = in_block as f64 * RATE / in_rate;
    let out48 = out_block as f64 * RATE / out_rate;
    (out48 + in48 / 2.0).ceil() as usize + OUT_CHUNK + MARGIN_FRAMES
}

/// IO buffer size to request from a device running at `rate`, lasting as long as
/// `frames_at_48k` does at 48 kHz (so a 16 kHz headset is not asked for a 3× longer buffer).
pub fn scaled_io_frames(frames_at_48k: u32, rate: f64) -> u32 {
    if !rate.is_finite() || rate <= 0.0 {
        return frames_at_48k;
    }
    ((frames_at_48k as f64 * rate / RATE).round() as u32).max(1)
}

/// `want` limited to a device's allowed buffer range. Devices occasionally report nonsense
/// (min > max, NaN); such a range is ignored rather than trusted (`f64::clamp` would panic).
pub fn clamp_io_frames(want: u32, min: f64, max: f64) -> u32 {
    if min.is_finite() && max.is_finite() && min <= max && max >= 1.0 {
        (want as f64).clamp(min.max(1.0), max) as u32
    } else {
        want.max(1)
    }
}

/// Estimated mic-to-sink latency in seconds. Latencies and blocks are in frames at their own
/// device's rate; `chain48` (input resampler + chain) and `target48` are at 48 kHz. The ring
/// target already contains the output block, so that is not added again.
pub fn estimate_latency_s(
    in_block: usize,
    in_latency: usize,
    in_rate: f64,
    chain48: usize,
    target48: usize,
    out_latency: usize,
    out_rate: f64,
) -> f64 {
    (in_block + in_latency) as f64 / in_rate
        + (chain48 + target48) as f64 / RATE
        + out_latency as f64 / out_rate
}

/// Callback durations of one stream: maximum and a fixed log-scale histogram, so a worst case
/// and p50/p99 can be read without allocating or locking on the audio thread.
///
/// Buckets: index 0 holds everything under [`CallbackTimes::MIN_NS`]; above that every octave of
/// nanoseconds is split into [`CallbackTimes::SUB`] equal-width buckets, so a bucket's upper
/// edge is at most 25% above its lower edge; the last bucket is open-ended (≥ ~134 ms).
///
/// The first callback after [`CallbackTimes::reset`] is kept apart in `first_ns` rather than in
/// the histogram and maximum: it primes the model on the audio thread (the one documented
/// exception to the no-allocation rule), so it would otherwise always be the maximum.
pub struct CallbackTimes {
    buckets: [AtomicU64; CallbackTimes::BUCKETS],
    max_ns: AtomicU64,
    first_ns: AtomicU64,
    /// Callbacks recorded since the last reset, including the first.
    calls: AtomicU64,
    /// The device IO buffer duration, ns, set by the backend when the stream starts.
    pub period_ns: AtomicU64,
}

impl CallbackTimes {
    /// log2 of the lowest resolved duration (1024 ns ≈ 1 µs).
    const MIN_LOG2: u32 = 10;
    pub const MIN_NS: u64 = 1 << Self::MIN_LOG2;
    /// log2 of the start of the open-ended last bucket (2^27 ns ≈ 134 ms).
    const MAX_LOG2: u32 = 27;
    /// Buckets per octave (a power of two).
    const SUB: usize = 4;
    const SUB_BITS: u32 = Self::SUB.trailing_zeros();
    pub const BUCKETS: usize = 2 + (Self::MAX_LOG2 - Self::MIN_LOG2) as usize * Self::SUB;

    pub fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            max_ns: AtomicU64::new(0),
            first_ns: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            period_ns: AtomicU64::new(0),
        }
    }

    /// The bucket a duration falls into.
    pub fn bucket(ns: u64) -> usize {
        if ns < Self::MIN_NS {
            return 0;
        }
        let e = 63 - ns.leading_zeros();
        if e >= Self::MAX_LOG2 {
            return Self::BUCKETS - 1;
        }
        let sub = ((ns >> (e - Self::SUB_BITS)) as usize) & (Self::SUB - 1);
        1 + (e - Self::MIN_LOG2) as usize * Self::SUB + sub
    }

    /// The exclusive upper edge of a bucket, ns (`u64::MAX` for the last one).
    pub fn upper_ns(bucket: usize) -> u64 {
        if bucket == 0 {
            return Self::MIN_NS;
        }
        if bucket >= Self::BUCKETS - 1 {
            return u64::MAX;
        }
        let i = bucket - 1;
        let e = Self::MIN_LOG2 + (i / Self::SUB) as u32;
        let sub = (i % Self::SUB) as u64;
        (1u64 << e) + ((sub + 1) << (e - Self::SUB_BITS))
    }

    /// Record one callback's duration. Realtime-safe: a few relaxed atomic operations.
    pub fn record(&self, ns: u64) {
        if self.calls.fetch_add(1, Relaxed) == 0 {
            self.first_ns.store(ns, Relaxed);
            return;
        }
        self.buckets[Self::bucket(ns)].fetch_add(1, Relaxed);
        self.max_ns.fetch_max(ns, Relaxed);
    }

    /// Forget everything recorded so far. Call while the stream's callback is not running.
    pub fn reset(&self) {
        for b in &self.buckets {
            b.store(0, Relaxed);
        }
        self.max_ns.store(0, Relaxed);
        self.first_ns.store(0, Relaxed);
        self.calls.store(0, Relaxed);
    }

    /// The `q`-quantile (0.0..=1.0) of the recorded durations, ns: the upper edge of the bucket
    /// it falls in, capped at the maximum, so it errs high by at most one bucket width (25%).
    /// `None` before any callback past the first was recorded.
    pub fn quantile_ns(&self, q: f64) -> Option<u64> {
        let counts: [u64; Self::BUCKETS] = std::array::from_fn(|i| self.buckets[i].load(Relaxed));
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return None;
        }
        // The rank of the sample at quantile q, 1-based.
        let rank = ((q.clamp(0.0, 1.0) * total as f64).ceil() as u64).max(1);
        let max = self.max_ns.load(Relaxed);
        let mut seen = 0;
        for (i, &c) in counts.iter().enumerate() {
            seen += c;
            if seen >= rank {
                return Some(Self::upper_ns(i).min(max));
            }
        }
        Some(max)
    }

    pub fn max_ns(&self) -> u64 {
        self.max_ns.load(Relaxed)
    }

    pub fn first_ns(&self) -> u64 {
        self.first_ns.load(Relaxed)
    }

    pub fn calls(&self) -> u64 {
        self.calls.load(Relaxed)
    }
}

impl Default for CallbackTimes {
    fn default() -> Self {
        Self::new()
    }
}

/// State shared between the control thread and both callbacks. All lock-free.
pub struct Shared {
    /// Denoise on (true) or bypassed (false). The chain still runs HPF and limiter when bypassed.
    pub denoise: AtomicBool,
    pub max_atten_bits: AtomicU32,
    pub levels: Arc<Levels>,
    pub running: AtomicBool,
    pub in_cycles: AtomicU64,
    pub out_cycles: AtomicU64,
    /// Nanoseconds spent inside both callbacks and the DSP worker, cumulative.
    pub busy_ns: AtomicU64,
    /// Per-callback durations of the capture (mic) and output (sink) streams.
    pub capture_times: CallbackTimes,
    pub output_times: CallbackTimes,
    /// Per-block processing time of the DSP worker ([`Placement::Worker`]); the first record is
    /// its priming. Its `period_ns` is the capture period, the time it has per block.
    pub worker_times: CallbackTimes,
    /// The worker runs with realtime (time-constraint) scheduling.
    pub worker_realtime: AtomicBool,
    /// The worker joined the mic's IO-thread audio workgroup.
    pub worker_workgroup: AtomicBool,
    /// Capture blocks the worker had not processed in time, so dry audio filled in.
    pub starvations: AtomicU64,
    /// Times the worker fell a whole ring behind and its backlog was dropped.
    pub worker_overflows: AtomicU64,
    /// The worker reserve of the running streams, samples at 48 kHz (0 when inline).
    pub reserve_frames: AtomicU32,
    /// Fade-out handshake before stopping streams: `STOP_FADE_*`.
    pub stop_fade: AtomicU8,
    /// The model the running streams use ([`encode_model`]), 0 when nothing runs.
    pub model: AtomicU8,
    pub underruns: AtomicU64,
    pub resyncs: AtomicU64,
    pub overflows: AtomicU64,
    pub fill_frames: AtomicU32,
    pub target_frames: AtomicU32,
    /// Current drift correction in ppm (positive = consuming faster than nominal).
    pub correction_ppm: AtomicI32,
    pub in_rate: AtomicU32,
    pub out_rate: AtomicU32,
    /// Estimated input-to-sink latency of the running streams, microseconds.
    pub estimate_us: AtomicU32,
    pub probe: Option<Arc<Probe>>,
    /// Development: mono samples looped in place of the mic signal (see `EngineOptions::source`).
    pub source: Option<Arc<[f32]>>,
}

impl Shared {
    pub fn new(levels: Arc<Levels>, probe: Option<Arc<Probe>>) -> Self {
        Self {
            denoise: AtomicBool::new(true),
            max_atten_bits: AtomicU32::new(24f32.to_bits()),
            levels,
            running: AtomicBool::new(false),
            in_cycles: AtomicU64::new(0),
            out_cycles: AtomicU64::new(0),
            busy_ns: AtomicU64::new(0),
            capture_times: CallbackTimes::new(),
            output_times: CallbackTimes::new(),
            worker_times: CallbackTimes::new(),
            worker_realtime: AtomicBool::new(false),
            worker_workgroup: AtomicBool::new(false),
            starvations: AtomicU64::new(0),
            worker_overflows: AtomicU64::new(0),
            reserve_frames: AtomicU32::new(0),
            stop_fade: AtomicU8::new(STOP_FADE_IDLE),
            model: AtomicU8::new(0),
            underruns: AtomicU64::new(0),
            resyncs: AtomicU64::new(0),
            overflows: AtomicU64::new(0),
            fill_frames: AtomicU32::new(0),
            target_frames: AtomicU32::new(0),
            correction_ppm: AtomicI32::new(0),
            in_rate: AtomicU32::new(0),
            out_rate: AtomicU32::new(0),
            estimate_us: AtomicU32::new(0),
            probe,
            source: None,
        }
    }
}

/// [`Shared::model`]'s encoding: 0 is "nothing running".
pub fn encode_model(m: Option<Model>) -> u8 {
    match m {
        None => 0,
        Some(Model::RNNoise) => 1,
        Some(Model::DeepFilterNet3) => 2,
    }
}

pub fn decode_model(v: u8) -> Option<Model> {
    match v {
        1 => Some(Model::RNNoise),
        2 => Some(Model::DeepFilterNet3),
        _ => None,
    }
}

/// Test hook for `oto latency`: replaces the mic signal with a tone-burst train and records when
/// each burst was captured, in host-clock nanoseconds.
pub struct Probe {
    pub last_inject_ns: AtomicU64,
    pub injections: AtomicU64,
}

impl Probe {
    /// Seconds between bursts.
    pub const PERIOD_S: f64 = 0.5;
    /// Burst length, seconds. A 1 kHz cosine at 0.5 FS, so the onset sample is at full amplitude.
    pub const BURST_S: f64 = 0.010;
    pub const FREQ_HZ: f64 = 1000.0;
    pub const AMPLITUDE: f32 = 0.5;

    pub fn new() -> Self {
        Self {
            last_inject_ns: AtomicU64::new(0),
            injections: AtomicU64::new(0),
        }
    }
}

impl Default for Probe {
    fn default() -> Self {
        Self::new()
    }
}

struct ProbeGen {
    probe: Arc<Probe>,
    rate: f64,
    period: usize,
    burst: usize,
    pos: usize,
}

impl ProbeGen {
    fn fill(&mut self, buf: &mut [f32], first_ns: u64) {
        let w = 2.0 * std::f64::consts::PI * Probe::FREQ_HZ / self.rate;
        for (i, s) in buf.iter_mut().enumerate() {
            if self.pos == 0 && first_ns != 0 {
                let t = first_ns + (i as f64 * 1e9 / self.rate) as u64;
                self.probe.last_inject_ns.store(t, Relaxed);
                self.probe.injections.fetch_add(1, Relaxed);
            }
            *s = if self.pos < self.burst {
                Probe::AMPLITUDE * (w * self.pos as f64).cos() as f32
            } else {
                0.0
            };
            self.pos = (self.pos + 1) % self.period;
        }
    }
}

/// Mixes interleaved `input` (with `channels` channels) into `out`, scaled by `scale`.
/// `out` must hold `input.len() / channels` frames.
fn downmix_add(input: &[f32], channels: usize, out: &mut [f32], scale: f32) {
    if channels == 1 {
        for (o, x) in out.iter_mut().zip(input) {
            *o += x * scale;
        }
        return;
    }
    for (o, frame) in out.iter_mut().zip(input.chunks_exact(channels)) {
        *o += frame.iter().sum::<f32>() * scale;
    }
}

/// Downmixes one device callback's buffers to mono in `out`.
///
/// `buffers` yields each device buffer as `(interleaved samples, channel count)`: an interleaved
/// device has one buffer with all channels, a non-interleaved one has a mono buffer per channel,
/// and mixtures occur. Buffers must cover the same frames; an empty slice stands for a buffer
/// with no data.
///
/// Devices with one or two input channels are averaged (a stereo mic). Devices with more are
/// audio interfaces, where a mic sits on one input and the others carry nothing or something
/// else: only the first channel is used, at full level, instead of an average that would bury
/// it by `20·log10(channels)` dB under the other inputs' noise.
pub fn downmix<'a, I>(buffers: I, out: &mut [f32])
where
    I: Iterator<Item = (&'a [f32], usize)> + Clone,
{
    out.fill(0.0);
    let total: usize = buffers.clone().map(|(_, ch)| ch).sum();
    if total > 2 {
        if let Some((data, ch)) = buffers.into_iter().find(|(d, ch)| *ch > 0 && !d.is_empty()) {
            for (o, frame) in out.iter_mut().zip(data.chunks_exact(ch)) {
                *o = frame[0];
            }
        }
        return;
    }
    if total == 0 {
        return;
    }
    let scale = 1.0 / total as f32;
    for (data, ch) in buffers {
        if ch > 0 {
            downmix_add(data, ch, out, scale);
        }
    }
}

/// Peak level in dBFS, floored at -120.
pub fn peak_dbfs(buf: &[f32]) -> f32 {
    let peak = buf.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if peak <= 1e-6 {
        -120.0
    } else {
        20.0 * peak.log10()
    }
}

/// Where a model's chain runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Inside the capture callback. For models that never allocate once primed (RNNoise).
    Inline,
    /// On a dedicated DSP thread fed through two rings from the capture callback (design B in
    /// docs/plans/0001-deepfilternet3-default.md). For models that allocate per frame
    /// (DeepFilterNet3): the audio callback stays allocation-free, the worker may allocate.
    Worker,
}

/// Where this engine runs `model`'s chain: DeepFilterNet3 on the DSP worker, RNNoise inline.
pub fn placement_for(model: Model) -> Placement {
    match model {
        Model::RNNoise => Placement::Inline,
        Model::DeepFilterNet3 => Placement::Worker,
    }
}

/// What the DSP worker's thread setup produced: whether it got realtime scheduling and whether
/// it joined the capture device's audio workgroup, plus anything to keep alive (and drop on the
/// worker thread when it exits, e.g. to leave the workgroup).
pub struct DspThread {
    pub realtime: bool,
    pub workgroup: bool,
    pub guard: Option<Box<dyn std::any::Any>>,
}

/// Runs first on the DSP worker thread, given the capture period in ns, to set up its
/// scheduling. See [`default_dsp_setup`] and the platform's version.
pub type DspSetup = Box<dyn FnOnce(u64) -> DspThread + Send>;

/// Realtime priority only, no workgroup (no device to take one from).
pub fn default_dsp_setup() -> DspSetup {
    Box::new(|period_ns| DspThread {
        realtime: crate::platform::sys::promote_dsp_thread(period_ns).is_ok(),
        workgroup: false,
        guard: None,
    })
}

/// Crossfade between the worker's (wet) output and the dry signal, samples at 48 kHz (1 ms).
pub const WORKER_FADE: usize = 48;
/// Slack in the worker reserve beyond one capture block and the fade, samples at 48 kHz (1 ms):
/// how much later than the next capture callback the worker may deliver without starving.
pub const WORKER_SLACK: usize = 48;
/// How long the worker sleeps when it hears nothing (it is woken by every capture callback;
/// this only bounds how long it takes to notice a stop request).
const WORKER_IDLE: Duration = Duration::from_millis(20);

/// The reserve between the worker's rings, in samples at 48 kHz, for capture callbacks that
/// deliver at most `max_block48` samples at 48 kHz.
///
/// At each capture callback the worker has normally processed everything pushed before it, so
/// processed audio reaches exactly `reserve` samples past the start of the block the callback
/// must output. The block needs `max_block48` of them, the fade-out on starvation needs
/// [`WORKER_FADE`] more beyond the block (so it can fade on real wet audio), and
/// [`WORKER_SLACK`] covers callback jitter and the worker's wake-up. The worker therefore has
/// one capture period plus the slack to process each block.
pub fn worker_reserve(max_block48: usize) -> usize {
    max_block48 + WORKER_FADE + WORKER_SLACK
}

/// Most samples at 48 kHz one capture callback of `in_block` device frames can produce.
fn max_block48(in_block: usize, rate: f64, resampled: bool) -> usize {
    if !resampled {
        return in_block;
    }
    // The input resampler runs in whole `IN_CHUNK`s, so a callback can complete one chunk more
    // than its own frames fill (the rest of the previous callback's partial chunk).
    let chunks = (in_block + IN_CHUNK - 1).div_ceil(IN_CHUNK);
    chunks * ((IN_CHUNK as f64 * RATE / rate).ceil() as usize + 1)
}

/// The chain running inside the capture callback.
struct InlineChain {
    chain: Chain,
    denoise: bool,
    atten: f32,
    /// Whether `Chain::prime` has run on the audio thread yet.
    primed: bool,
}

impl InlineChain {
    fn process(&mut self, buf: &mut [f32], shared: &Shared) {
        apply_settings(&mut self.chain, &mut self.denoise, &mut self.atten, shared);
        self.chain.process(buf);
    }
}

/// Copies changed bypass/strength settings from `shared` into `chain`.
fn apply_settings(chain: &mut Chain, denoise: &mut bool, atten: &mut f32, shared: &Shared) {
    let d = shared.denoise.load(Relaxed);
    if d != *denoise {
        *denoise = d;
        chain.set_enabled(d);
    }
    let a = f32::from_bits(shared.max_atten_bits.load(Relaxed));
    if a.to_bits() != atten.to_bits() {
        *atten = a;
        chain.set_max_attenuation_db(a);
    }
}

/// Handshake between the capture callback and the worker. All lock-free.
struct WorkerSync {
    stop: AtomicBool,
    /// Bumped by the callback when the input ring overflowed: the worker drops its pending
    /// input and acknowledges by copying the value to `flush_ack`.
    flush_req: AtomicU32,
    flush_ack: AtomicU32,
    /// Tests: the worker leaves its input alone while set, to force starvation and overflow.
    #[cfg(test)]
    pause: AtomicBool,
}

/// The callback's end of the DSP worker: owns the rings, the dry path and the thread.
///
/// Sample accounting: `in_pos` counts samples entering the callback, the output of a call
/// starts at input index `in_pos - delay()` (that is what the dry delay line yields), and wet
/// sample number `w` (counted from `wet_base`, where the current worker stream started) is the
/// chain's output after it consumed input `wet_base + w`, i.e. the processed version of input
/// `wet_base + w - chain_latency`.
struct Worker {
    to: Producer<f32>,
    from: Consumer<f32>,
    sync: Arc<WorkerSync>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// The worker's handle, to wake it. `unpark` does not allocate or block.
    handle: std::thread::Thread,
    flush_gen: u32,
    flushing: bool,
    dry: Delay,
    dry_buf: Box<[f32]>,
    wet_buf: Box<[f32]>,
    chain_latency: i64,
    reserve: usize,
    in_pos: i64,
    wet_base: i64,
    /// Wet samples consumed (used or discarded) since `wet_base`.
    wet_read: i64,
    /// Share of wet signal in the output, ramped per sample.
    mix: f32,
    /// Wet minus dry at the last wet sample used.
    last_diff: f32,
}

impl Worker {
    /// Spawns and primes the worker. Allocates and blocks until the model is primed; call off
    /// the realtime thread.
    fn spawn(
        chain: Chain,
        max_block48: usize,
        period_ns: u64,
        shared: &Arc<Shared>,
        setup: DspSetup,
    ) -> Result<Self, String> {
        let reserve = worker_reserve(max_block48);
        let chain_latency = chain.latency();
        let (to, input) = RingBuffer::new(RING_CAPACITY);
        // Unconsumed wet audio never exceeds the input ring plus the reserve and one block (the
        // callback discards stale wet audio every call), so this ring cannot overflow.
        let (output, from) = RingBuffer::new(2 * RING_CAPACITY);
        let sync = Arc::new(WorkerSync {
            stop: AtomicBool::new(false),
            flush_req: AtomicU32::new(0),
            flush_ack: AtomicU32::new(0),
            #[cfg(test)]
            pause: AtomicBool::new(false),
        });
        shared.worker_times.reset();
        shared.worker_times.period_ns.store(period_ns, Relaxed);
        shared.worker_realtime.store(false, Relaxed);
        shared.worker_workgroup.store(false, Relaxed);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("oto-dsp".into())
            .spawn({
                let (sync, shared) = (sync.clone(), shared.clone());
                move || {
                    let thread = setup(period_ns);
                    shared.worker_realtime.store(thread.realtime, Relaxed);
                    shared.worker_workgroup.store(thread.workgroup, Relaxed);
                    worker_main(chain, input, output, &sync, &shared, ready_tx);
                    // Leaves the workgroup (if any) on this thread, as it must.
                    drop(thread);
                }
            })
            .map_err(|e| format!("cannot start the DSP thread: {e}"))?;
        let handle = thread.thread().clone();
        let delay = chain_latency + reserve;
        let w = Self {
            to,
            from,
            sync,
            thread: Some(thread),
            handle,
            flush_gen: 0,
            flushing: false,
            dry: Delay::new(delay),
            dry_buf: vec![0.0; MAX_BLOCK].into(),
            wet_buf: vec![0.0; MAX_BLOCK].into(),
            chain_latency: chain_latency as i64,
            reserve,
            in_pos: 0,
            wet_base: 0,
            wet_read: 0,
            mix: 0.0,
            last_diff: 0.0,
        };
        // On failure `w` is dropped, which stops and joins the thread.
        match ready_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(()) => Ok(w),
            Err(_) => Err("the DSP thread did not start".into()),
        }
    }

    /// Delay from input to output, samples at 48 kHz: the chain's plus the reserve.
    fn delay(&self) -> usize {
        self.chain_latency as usize + self.reserve
    }

    fn discard_wet(&mut self, n: usize) -> usize {
        let n = n.min(self.from.slots());
        if let Ok(c) = self.from.read_chunk(n) {
            c.commit_all();
        }
        n
    }

    /// Hands `buf` to the worker and replaces it with processed audio from `delay()` samples
    /// ago, or the dry signal at that delay where the worker has not delivered. Realtime-safe.
    fn run(&mut self, buf: &mut [f32], shared: &Shared) {
        let len = buf.len();
        let n = len as i64;
        if self.flushing && self.sync.flush_ack.load(Acquire) == self.flush_gen {
            // The worker dropped its input and everything it pushed before is visible: start a
            // fresh wet stream from this block on.
            self.discard_wet(usize::MAX);
            self.flushing = false;
            self.wet_base = self.in_pos;
            self.wet_read = 0;
        }
        if !self.flushing {
            if self.to.slots() >= len {
                let _ = self.to.push_entire_slice(buf);
            } else {
                // The worker is a ring (≈340 ms) behind. Pushing part of the block would break
                // the sample accounting, so drop its backlog and restart the wet stream.
                shared.worker_overflows.fetch_add(1, Relaxed);
                self.flush_gen = self.flush_gen.wrapping_add(1);
                self.sync.flush_req.store(self.flush_gen, Release);
                self.flushing = true;
            }
            self.handle.unpark();
        }
        self.dry.process(buf, &mut self.dry_buf[..len]);
        let o0 = self.in_pos - self.delay() as i64;
        self.in_pos += n;

        // Block-relative range [first, end) the wet stream covers; `end` may lie past the block
        // (that lookahead lets a fade-out run on real wet audio). Before `wet_base` there is no
        // wet stream: output dry (silence at stream start).
        let first = (self.wet_base - o0).clamp(0, n);
        let mut end = first;
        if !self.flushing && first < n {
            // The wet sample that belongs at block index `first`.
            let need = o0 + first - self.wet_base + self.chain_latency;
            if self.wet_read < need {
                // Stale: the worker caught up after a starvation, or the chain's start-up
                // output. Dropped so the output never jumps backwards.
                let k = self.discard_wet((need - self.wet_read) as usize);
                self.wet_read += k as i64;
            }
            if self.wet_read == need {
                end = first + self.from.slots() as i64;
            }
        }
        let take = (end.min(n) - first).max(0) as usize;
        let f = first as usize;
        if take > 0 {
            let _ = self.from.pop_entire_slice(&mut self.wet_buf[f..f + take]);
            self.wet_read += take as i64;
        }

        let step = 1.0 / WORKER_FADE as f32;
        for (i, (y, &dry)) in buf.iter_mut().zip(&self.dry_buf[..len]).enumerate() {
            let at = i as i64;
            let wet = at >= first && at < end;
            // Normally the fade-out ends exactly where the wet audio runs out (the reserve
            // leaves WORKER_FADE samples of lookahead). The mix never moves by more than one
            // step per sample, so if wet audio ever runs out sooner, the fade continues on the
            // last wet-minus-dry difference instead of cutting to dry.
            let target = if wet {
                ((end - at) as f32 * step).min(1.0)
            } else {
                0.0
            };
            self.mix = (self.mix + step).min(target).max(self.mix - step).max(0.0);
            *y = if !wet {
                dry + self.last_diff * self.mix
            } else if self.mix >= 1.0 {
                self.last_diff = self.wet_buf[i] - dry;
                self.wet_buf[i]
            } else {
                self.last_diff = self.wet_buf[i] - dry;
                dry + self.last_diff * self.mix
            };
        }
        if first < n && end < n {
            shared.starvations.fetch_add(1, Relaxed);
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.sync.stop.store(true, Release);
        self.handle.unpark();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The DSP thread: owns the chain, primes it, then processes whatever the capture callback
/// pushed. May allocate (DeepFilterNet3 does on every frame).
fn worker_main(
    mut chain: Chain,
    mut input: Consumer<f32>,
    mut output: Producer<f32>,
    sync: &WorkerSync,
    shared: &Shared,
    ready: std::sync::mpsc::SyncSender<()>,
) {
    let t0 = Instant::now();
    chain.prime();
    // The first record is kept apart as the priming time.
    shared.worker_times.record(t0.elapsed().as_nanos() as u64);
    let _ = ready.send(());
    let mut buf = vec![0.0f32; MAX_BLOCK];
    let (mut denoise, mut atten) = (chain.enabled(), chain.max_attenuation_db());
    let mut seen = 0u32;
    while !sync.stop.load(Acquire) {
        let g = sync.flush_req.load(Acquire);
        if g != seen {
            let n = input.slots();
            if let Ok(c) = input.read_chunk(n) {
                c.commit_all();
            }
            seen = g;
            sync.flush_ack.store(g, Release);
            continue;
        }
        let n = input.slots().min(MAX_BLOCK);
        #[cfg(test)]
        let n = if sync.pause.load(Acquire) { 0 } else { n };
        if n == 0 {
            std::thread::park_timeout(WORKER_IDLE);
            continue;
        }
        let _ = input.pop_entire_slice(&mut buf[..n]);
        apply_settings(&mut chain, &mut denoise, &mut atten, shared);
        let t = Instant::now();
        chain.process(&mut buf[..n]);
        let dt = t.elapsed().as_nanos() as u64;
        shared.worker_times.record(dt);
        shared.busy_ns.fetch_add(dt, Relaxed);
        if output.push_entire_slice(&buf[..n]).is_err() {
            shared.worker_overflows.fetch_add(1, Relaxed);
        }
    }
}

enum ChainRunner {
    Inline(InlineChain),
    Worker(Worker),
}

/// Mic side: device-rate mono in, 48 kHz processed audio into the ring.
///
/// The chain runs either inline (in [`Self::process`]) or on a DSP worker thread, see
/// [`Placement`]. Either way `process` does not allocate after the first call.
pub struct CaptureStage {
    mono: Box<[f32]>,
    stage: Box<[f32]>,
    stage_len: usize,
    resampler: Option<Async<f32>>,
    out48: Box<[f32]>,
    runner: ChainRunner,
    producer: Producer<f32>,
    shared: Arc<Shared>,
    probe: Option<ProbeGen>,
    source: Option<(Arc<[f32]>, usize)>,
    level_raw: f32,
    level_clean: f32,
}

impl CaptureStage {
    /// Allocates (and for a worker model starts and primes the DSP thread); call off the
    /// realtime thread. `in_block` is the device IO buffer in frames at `rate`. The chain runs
    /// where [`placement_for`] says.
    pub fn new(
        rate: f64,
        in_block: usize,
        chain: Chain,
        producer: Producer<f32>,
        shared: Arc<Shared>,
    ) -> Result<Self, String> {
        let placement = placement_for(chain.model());
        Self::with_placement(rate, in_block, chain, producer, shared, placement)
    }

    /// [`Self::new`] with the chain placed explicitly (tests run RNNoise on the worker too).
    pub fn with_placement(
        rate: f64,
        in_block: usize,
        chain: Chain,
        producer: Producer<f32>,
        shared: Arc<Shared>,
        placement: Placement,
    ) -> Result<Self, String> {
        let setup = default_dsp_setup();
        Self::build(rate, in_block, chain, producer, shared, placement, setup)
    }

    /// The full constructor: `setup` runs first on the DSP worker thread (unused inline). The
    /// macOS backend uses it to join the mic's IO-thread audio workgroup.
    pub fn build(
        rate: f64,
        in_block: usize,
        chain: Chain,
        producer: Producer<f32>,
        shared: Arc<Shared>,
        placement: Placement,
        setup: DspSetup,
    ) -> Result<Self, String> {
        let resampler = if (rate - RATE).abs() > 0.5 {
            let params = rubato::SincInterpolationParameters {
                sinc_len: 64,
                f_cutoff: None,
                oversampling_factor: 128,
                interpolation: rubato::SincInterpolationType::Cubic,
                window: rubato::WindowFunction::BlackmanHarris2,
            };
            let r =
                Async::<f32>::new_sinc(RATE / rate, 1.0, &params, IN_CHUNK, 1, FixedAsync::Input)
                    .map_err(|e| format!("input resampler: {e}"))?;
            Some(r)
        } else {
            None
        };
        let max_out = (MAX_BLOCK as f64 * RATE / rate).ceil() as usize
            + resampler.as_ref().map_or(0, |r| r.output_frames_max())
            + 16;
        let probe = shared.probe.clone().map(|probe| ProbeGen {
            probe,
            rate,
            period: (Probe::PERIOD_S * rate) as usize,
            burst: (Probe::BURST_S * rate) as usize,
            pos: 0,
        });
        let source = shared
            .source
            .clone()
            .filter(|s| !s.is_empty())
            .map(|s| (s, 0));
        let mut chain = chain;
        let denoise = shared.denoise.load(Relaxed);
        let atten = f32::from_bits(shared.max_atten_bits.load(Relaxed));
        chain.set_enabled(denoise);
        chain.set_max_attenuation_db(atten);
        let runner = match placement {
            Placement::Inline => {
                shared.worker_times.reset();
                shared.worker_times.period_ns.store(0, Relaxed);
                shared.worker_realtime.store(false, Relaxed);
                shared.worker_workgroup.store(false, Relaxed);
                ChainRunner::Inline(InlineChain {
                    chain,
                    denoise,
                    atten,
                    primed: false,
                })
            }
            Placement::Worker => {
                let block = in_block.clamp(1, MAX_BLOCK);
                let period_ns = (block as f64 * 1e9 / rate) as u64;
                let max48 = max_block48(block, rate, resampler.is_some()).min(max_out);
                ChainRunner::Worker(Worker::spawn(chain, max48, period_ns, &shared, setup)?)
            }
        };
        let reserve = match &runner {
            ChainRunner::Inline(_) => 0,
            ChainRunner::Worker(w) => w.reserve,
        };
        shared.reserve_frames.store(reserve as u32, Relaxed);
        Ok(Self {
            mono: vec![0.0; MAX_BLOCK].into(),
            stage: vec![0.0; IN_CHUNK].into(),
            stage_len: 0,
            resampler,
            out48: vec![0.0; max_out].into(),
            runner,
            producer,
            shared,
            source,
            probe,
            level_raw: -120.0,
            level_clean: -120.0,
        })
    }

    /// Where the chain runs.
    pub fn placement(&self) -> Placement {
        match self.runner {
            ChainRunner::Inline(_) => Placement::Inline,
            ChainRunner::Worker(_) => Placement::Worker,
        }
    }

    /// The worker reserve in samples at 48 kHz (0 when the chain runs inline).
    pub fn reserve_frames(&self) -> usize {
        match &self.runner {
            ChainRunner::Inline(_) => 0,
            ChainRunner::Worker(w) => w.reserve,
        }
    }

    /// Delay added by the input resampler, the chain and (on the worker) the reserve, in
    /// frames at 48 kHz.
    pub fn latency_frames(&self) -> usize {
        let rs = self
            .resampler
            .as_ref()
            .map_or(0, |r| r.output_delay() + IN_CHUNK);
        rs + match &self.runner {
            ChainRunner::Inline(c) => c.chain.latency(),
            ChainRunner::Worker(w) => w.delay(),
        }
    }

    /// Scratch buffer (`MAX_BLOCK` long) to downmix device audio into before [`Self::process`].
    pub fn mono_mut(&mut self) -> &mut [f32] {
        &mut self.mono
    }

    /// Process the first `n` (≤ `MAX_BLOCK`) samples of [`Self::mono_mut`]. `first_ns` is the
    /// host-clock capture time of the first sample, or 0 if unknown.
    ///
    /// Never allocates, except that the first call primes an inline model on the calling thread
    /// (decision 3 of plan 0001: RNNoise sets up per-thread state on its first frame).
    pub fn process(&mut self, n: usize, first_ns: u64) {
        if let ChainRunner::Inline(c) = &mut self.runner
            && !c.primed
        {
            // One-time per-thread model setup, so later callbacks never allocate.
            c.chain.prime();
            c.primed = true;
        }
        let n = n.min(MAX_BLOCK);
        // A glitching device can hand us NaN or infinity; keep it out of the resampler too.
        oto_dsp::sanitize(&mut self.mono[..n]);
        if let Some(p) = &mut self.probe {
            p.fill(&mut self.mono[..n], first_ns);
        }
        if let Some((src, pos)) = &mut self.source {
            for s in &mut self.mono[..n] {
                *s = src[*pos];
                *pos = (*pos + 1) % src.len();
            }
        }
        let len = match &mut self.resampler {
            None => {
                self.out48[..n].copy_from_slice(&self.mono[..n]);
                n
            }
            Some(rs) => {
                let mut produced = 0;
                let mut at = 0;
                while at < n {
                    let take = (IN_CHUNK - self.stage_len).min(n - at);
                    self.stage[self.stage_len..self.stage_len + take]
                        .copy_from_slice(&self.mono[at..at + take]);
                    self.stage_len += take;
                    at += take;
                    if self.stage_len == IN_CHUNK {
                        self.stage_len = 0;
                        let room = self.out48.len() - produced;
                        let input = InterleavedSlice::new(&self.stage[..], 1, IN_CHUNK);
                        let output =
                            InterleavedSlice::new_mut(&mut self.out48[produced..], 1, room);
                        if let (Ok(i), Ok(mut o)) = (input, output)
                            && let Ok((_, written)) = rs.process_into_buffer(&i, &mut o, None)
                        {
                            produced += written;
                        }
                    }
                }
                produced
            }
        };
        if len == 0 {
            return;
        }
        let buf = &mut self.out48[..len];
        self.level_raw = peak_dbfs(buf);
        match &mut self.runner {
            ChainRunner::Inline(c) => c.process(buf, &self.shared),
            ChainRunner::Worker(w) => {
                // The worker's scratch buffers hold MAX_BLOCK samples.
                for part in buf.chunks_mut(MAX_BLOCK) {
                    w.run(part, &self.shared);
                }
            }
        }
        self.level_clean = peak_dbfs(buf);
        self.shared.levels.set(self.level_raw, self.level_clean);
        let (_, rest) = self.producer.push_partial_slice(buf);
        if !rest.is_empty() {
            self.shared.overflows.fetch_add(1, Relaxed);
        }
    }
}

/// PI controller from ring fill to a relative resample ratio.
pub struct DriftController {
    target: f64,
    ema: Option<f64>,
    integ: f64,
}

impl DriftController {
    /// Fill smoothing time constant, seconds.
    pub const TAU_S: f64 = 0.5;
    /// Proportional gain, per frame of error (100 frames off → 1000 ppm).
    pub const KP: f64 = 1e-5;
    /// Integral gain, per frame·second of error.
    pub const KI: f64 = 1e-6;
    /// Largest correction applied, as a ratio (2000 ppm).
    pub const MAX_CORR: f64 = 0.002;

    pub fn new(target: usize) -> Self {
        Self {
            target: target as f64,
            ema: None,
            integ: 0.0,
        }
    }

    pub fn target(&self) -> f64 {
        self.target
    }

    /// Feed the fill (frames) seen `dt` seconds after the previous update. Returns the
    /// correction: positive means the ring is too full, so consume faster than nominal.
    pub fn update(&mut self, fill: f64, dt: f64) -> f64 {
        let ema = match self.ema {
            None => fill,
            Some(e) => e + (1.0 - (-dt / Self::TAU_S).exp()) * (fill - e),
        };
        self.ema = Some(ema);
        let err = ema - self.target;
        // Anti-windup: while the output is clamped and the error still pushes it further into
        // the clamp, integrating would only store up an overshoot for later.
        let raw = Self::KP * err + Self::KI * self.integ;
        if raw.abs() < Self::MAX_CORR || raw * err < 0.0 {
            let lim = Self::MAX_CORR / Self::KI;
            self.integ = (self.integ + err * dt).clamp(-lim, lim);
        }
        (Self::KP * err + Self::KI * self.integ).clamp(-Self::MAX_CORR, Self::MAX_CORR)
    }

    /// Forget the smoothed fill (after a re-prime), keeping the learned clock offset. The next
    /// [`update`](Self::update) seeds it from the fill it is given.
    pub fn rebase(&mut self) {
        self.ema = None;
    }

    /// Set the smoothed fill to a known mean (after trimming a measured excess), keeping the
    /// learned clock offset.
    pub fn reseed(&mut self, mean_fill: f64) {
        self.ema = Some(mean_fill);
    }
}

/// Sink side: pulls 48 kHz audio from the ring and resamples it to the sink's clock.
pub struct OutputStage {
    consumer: Consumer<f32>,
    resampler: Async<f32>,
    inbuf: Box<[f32]>,
    pend: Box<[f32]>,
    pend_pos: usize,
    pend_len: usize,
    ctl: DriftController,
    priming: bool,
    rate: f64,
    shared: Arc<Shared>,
    /// Output gain and its per-sample step. `gain_step == 0` means steady at 1.0; positive is
    /// a fade-in; negative a fade-out that holds at 0.0 until the pending trim is done.
    gain: f32,
    gain_step: f32,
    fade_step: f32,
    /// Frames to drop from the ring once the fade-out has reached silence.
    trim_pending: usize,
    /// Fill integrated over the current trim window (frame·seconds) and the window's length.
    win_sum: f64,
    win_t: f64,
    /// The previous callback saw more than `target + TRIM_EXCESS`.
    was_over: bool,
    /// Gain of the fade-out before a stop ([`Shared::stop_fade`]) and its per-sample step.
    stop_gain: f32,
    stop_step: f32,
}

impl OutputStage {
    /// Allocates; call off the realtime thread. `target` is the mean ring fill in 48 kHz frames.
    pub fn new(
        consumer: Consumer<f32>,
        rate: f64,
        target: usize,
        shared: Arc<Shared>,
    ) -> Result<Self, String> {
        let resampler = Async::<f32>::new_poly(
            rate / RATE,
            1.01,
            PolynomialDegree::Quintic,
            OUT_CHUNK,
            1,
            FixedAsync::Output,
        )
        .map_err(|e| format!("output resampler: {e}"))?;
        let inbuf = vec![0.0; resampler.input_frames_max()].into();
        let pend = vec![0.0; resampler.output_frames_max()].into();
        shared.target_frames.store(target as u32, Relaxed);
        shared.stop_fade.store(STOP_FADE_IDLE, Relaxed);
        Ok(Self {
            consumer,
            resampler,
            inbuf,
            pend,
            pend_pos: 0,
            pend_len: 0,
            ctl: DriftController::new(target),
            priming: true,
            rate,
            shared,
            gain: 0.0,
            gain_step: 0.0,
            fade_step: (1.0 / (FADE_S * rate).max(1.0)) as f32,
            trim_pending: 0,
            win_sum: 0.0,
            win_t: 0.0,
            was_over: false,
            stop_gain: 1.0,
            stop_step: (1.0 / (STOP_FADE_S * rate).max(1.0)) as f32,
        })
    }

    /// Start from silence and ramp up: audio is (re)entering after silence or a discontinuity.
    fn restart_faded(&mut self) {
        self.gain = 0.0;
        self.gain_step = self.fade_step;
        self.trim_pending = 0;
        self.win_sum = 0.0;
        self.win_t = 0.0;
        self.was_over = false;
    }

    fn discard(&mut self, frames: usize) {
        let n = frames.min(self.consumer.slots());
        if let Ok(chunk) = self.consumer.read_chunk(n) {
            chunk.commit_all();
        }
    }

    fn fill_now(&self) -> f64 {
        self.consumer.slots() as f64 + (self.pend_len - self.pend_pos) as f64 * RATE / self.rate
    }

    #[inline]
    fn apply_gain(&mut self, buf: &mut [f32]) {
        if self.gain_step == 0.0 {
            return;
        }
        for s in buf {
            self.gain = (self.gain + self.gain_step).clamp(0.0, 1.0);
            *s *= self.gain;
        }
        if self.gain_step > 0.0 && self.gain >= 1.0 {
            self.gain_step = 0.0;
        }
    }

    /// Delay of the output resampler, in output frames.
    pub fn latency_frames(&self) -> usize {
        self.resampler.output_delay()
    }

    /// Fill `out` (mono, device rate).
    pub fn render(&mut self, out: &mut [f32]) {
        if self.stop_gain <= 0.0 {
            // Faded out for a stop: hold silence until the stream is stopped.
            out.fill(0.0);
            return;
        }
        self.render_ring(out);
        if self.shared.stop_fade.load(Acquire) == STOP_FADE_REQUESTED {
            for s in out.iter_mut() {
                self.stop_gain = (self.stop_gain - self.stop_step).max(0.0);
                *s *= self.stop_gain;
            }
            if self.stop_gain <= 0.0 {
                self.shared.stop_fade.store(STOP_FADE_DONE, Release);
            }
        }
    }

    fn render_ring(&mut self, out: &mut [f32]) {
        let dt = out.len() as f64 / self.rate;
        let mut fill = self.fill_now();
        let target = self.ctl.target();

        if self.priming {
            if fill < target {
                self.shared.fill_frames.store(fill as u32, Relaxed);
                out.fill(0.0);
                return;
            }
            self.priming = false;
            self.ctl.rebase();
            self.restart_faded();
        }
        if fill > target + MAX_EXCESS as f64 {
            // Far behind (the output stalled): drop the stale audio now. We are at an unknown
            // phase of the input cycle, so keep what priming would leave at its lowest phase
            // plus half an input block; the trim below or the controller removes the rest.
            let out48 = out.len() as f64 * RATE / self.rate;
            let half_in = (target - out48 - (OUT_CHUNK + MARGIN_FRAMES) as f64).max(0.0);
            let keep = (target + half_in).ceil() as usize;
            self.discard(self.consumer.slots().saturating_sub(keep));
            self.shared.resyncs.fetch_add(1, Relaxed);
            self.ctl.rebase();
            self.restart_faded();
            // The controller must see the fill that is left, not the one that was dropped.
            fill = self.fill_now();
        } else if self.trim_pending > 0 {
            if self.gain <= 0.0 {
                // Faded out: drop the measured excess while nothing is audible, then fade in.
                self.discard(self.trim_pending);
                self.shared.resyncs.fetch_add(1, Relaxed);
                self.ctl.reseed(target);
                self.restart_faded();
                fill = self.fill_now();
            }
        } else {
            let over = fill > target + TRIM_EXCESS as f64;
            if over && !self.was_over {
                // Excess just appeared: measure it from here, undiluted by what came before.
                self.win_sum = 0.0;
                self.win_t = 0.0;
            }
            self.was_over = over;
            self.win_sum += fill * dt;
            self.win_t += dt;
            if self.win_t >= TRIM_WINDOW_S {
                let excess = self.win_sum / self.win_t - target;
                if excess > TRIM_EXCESS as f64 {
                    self.trim_pending = excess as usize;
                    self.gain_step = -self.fade_step;
                }
                self.win_sum = 0.0;
                self.win_t = 0.0;
            }
        }
        self.shared.fill_frames.store(fill as u32, Relaxed);

        let corr = self.ctl.update(fill, dt);
        self.shared
            .correction_ppm
            .store((corr * 1e6) as i32, Relaxed);
        let _ = self.resampler.set_resample_ratio_relative(1.0 - corr, true);

        let mut at = 0;
        while at < out.len() {
            if self.pend_pos < self.pend_len {
                let n = (self.pend_len - self.pend_pos).min(out.len() - at);
                out[at..at + n].copy_from_slice(&self.pend[self.pend_pos..self.pend_pos + n]);
                self.apply_gain(&mut out[at..at + n]);
                self.pend_pos += n;
                at += n;
                continue;
            }
            let need = self.resampler.input_frames_next();
            if self
                .consumer
                .pop_entire_slice(&mut self.inbuf[..need])
                .is_err()
            {
                out[at..].fill(0.0);
                self.priming = true;
                self.shared.underruns.fetch_add(1, Relaxed);
                return;
            }
            let input = InterleavedSlice::new(&self.inbuf[..need], 1, need);
            let cap = self.pend.len();
            let output = InterleavedSlice::new_mut(&mut self.pend[..], 1, cap);
            self.pend_pos = 0;
            self.pend_len = match (input, output) {
                (Ok(i), Ok(mut o)) => self
                    .resampler
                    .process_into_buffer(&i, &mut o, None)
                    .map_or(0, |(_, w)| w),
                _ => 0,
            };
            if self.pend_len == 0 {
                out[at..].fill(0.0);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rtrb::RingBuffer;

    #[test]
    fn callback_buckets_are_contiguous_and_narrow() {
        assert_eq!(CallbackTimes::bucket(0), 0);
        assert_eq!(CallbackTimes::bucket(CallbackTimes::MIN_NS - 1), 0);
        assert_eq!(CallbackTimes::bucket(CallbackTimes::MIN_NS), 1);
        assert_eq!(CallbackTimes::bucket(u64::MAX), CallbackTimes::BUCKETS - 1);
        // Every bucket's upper edge is the next bucket's lower edge.
        for b in 0..CallbackTimes::BUCKETS - 1 {
            let hi = CallbackTimes::upper_ns(b);
            assert_eq!(CallbackTimes::bucket(hi - 1), b, "bucket {b}");
            assert_eq!(CallbackTimes::bucket(hi), b + 1, "bucket {b}");
        }
        // Resolution: an upper edge is at most 25% above the lower edge.
        for b in 2..CallbackTimes::BUCKETS - 1 {
            let (lo, hi) = (CallbackTimes::upper_ns(b - 1), CallbackTimes::upper_ns(b));
            assert!(
                hi as f64 <= lo as f64 * 1.25 + 1.0,
                "bucket {b}: {lo}..{hi}"
            );
        }
        // The open-ended bucket starts past 100 ms.
        assert!(CallbackTimes::upper_ns(CallbackTimes::BUCKETS - 2) > 100_000_000);
    }

    #[test]
    fn callback_percentiles_max_and_reset() {
        let t = CallbackTimes::new();
        assert_eq!(t.quantile_ns(0.5), None);
        // The first callback primes the model: kept apart from the statistics.
        t.record(50_000_000);
        assert_eq!(t.first_ns(), 50_000_000);
        assert_eq!(t.quantile_ns(0.5), None);
        assert_eq!(t.max_ns(), 0);
        // 98 callbacks at 100 µs, one at 1 ms, one at 3 ms.
        for _ in 0..98 {
            t.record(100_000);
        }
        t.record(1_000_000);
        t.record(3_000_000);
        assert_eq!(t.calls(), 101);
        assert_eq!(t.max_ns(), 3_000_000);
        let p50 = t.quantile_ns(0.5).unwrap();
        assert!((100_000..=125_000).contains(&p50), "p50 {p50}");
        let p99 = t.quantile_ns(0.99).unwrap();
        assert!((1_000_000..=1_250_000).contains(&p99), "p99 {p99}");
        // The top quantile is capped at the true maximum rather than the bucket edge.
        assert_eq!(t.quantile_ns(1.0), Some(3_000_000));
        t.reset();
        assert_eq!(t.quantile_ns(0.5), None);
        assert_eq!((t.max_ns(), t.first_ns(), t.calls()), (0, 0, 0));
        // After a reset the next callback is a first callback again.
        t.record(7);
        assert_eq!(t.first_ns(), 7);
        assert_eq!(t.quantile_ns(0.5), None);
    }

    #[test]
    fn model_encoding_round_trips() {
        assert_eq!(decode_model(encode_model(None)), None);
        for m in Model::ALL {
            assert_eq!(decode_model(encode_model(Some(m))), Some(m));
        }
    }

    /// RNNoise explicitly: `Chain::new()` is DeepFilterNet3 in `dfn3` builds, which would put
    /// these stages on the worker.
    fn rnnoise() -> Chain {
        Chain::with_model(Model::RNNoise).unwrap()
    }

    fn shared() -> Arc<Shared> {
        let s = Shared::new(Arc::new(Levels::default()), None);
        // Keep the denoiser out of signal-shape assertions once the real chain lands.
        s.denoise.store(false, Relaxed);
        Arc::new(s)
    }

    #[test]
    fn downmix_averages_mono_and_stereo() {
        // Stale contents of `out` must not leak into the mix.
        let mut out = [9.0; 3];
        downmix([(&[0.1f32, 0.2, 0.3][..], 1)].into_iter(), &mut out);
        assert_eq!(out, [0.1, 0.2, 0.3]);
        // Interleaved stereo.
        let input = [1.0f32, 0.0, 0.5, 0.5, -1.0, 1.0];
        downmix([(&input[..], 2)].into_iter(), &mut out);
        assert_eq!(out, [0.5, 0.5, 0.0]);
        // Non-interleaved stereo: one mono buffer per channel.
        let mut out = [9.0; 2];
        let (l, r) = ([1.0f32, 1.0], [0.0f32, -1.0]);
        downmix([(&l[..], 1), (&r[..], 1)].into_iter(), &mut out);
        assert_eq!(out, [0.5, 0.0]);
    }

    #[test]
    fn downmix_takes_first_channel_of_multichannel_devices() {
        // Interleaved 8-input interface: mic on input 1, noise elsewhere.
        let mut inter = [-0.9f32; 8 * 3];
        for f in 0..3 {
            inter[f * 8] = 0.25 * (f + 1) as f32;
        }
        let mut out = [9.0; 3];
        downmix([(&inter[..], 8)].into_iter(), &mut out);
        assert_eq!(out, [0.25, 0.5, 0.75]);
        // Non-interleaved, 3 mono buffers.
        let (a, b, c) = ([0.5f32, -0.5, 0.25], [1.0f32; 3], [1.0f32; 3]);
        downmix(
            [(&a[..], 1), (&b[..], 1), (&c[..], 1)].into_iter(),
            &mut out,
        );
        assert_eq!(out, [0.5, -0.5, 0.25]);
        // Mixed layout (a stereo buffer then mono ones): still channel 0 of the first buffer.
        let st = [0.1f32, 0.9, 0.2, 0.9, 0.3, 0.9];
        downmix([(&st[..], 2), (&b[..], 1)].into_iter(), &mut out);
        assert_eq!(out, [0.1, 0.2, 0.3]);
        // A leading buffer without data is skipped rather than read.
        downmix([(&[][..], 2), (&a[..], 1)].into_iter(), &mut out);
        assert_eq!(out, [0.5, -0.5, 0.25]);
        // Nothing at all is silence.
        downmix(std::iter::empty(), &mut out);
        assert_eq!(out, [0.0; 3]);
    }

    #[test]
    fn io_frames_scale_with_rate_and_survive_bad_ranges() {
        assert_eq!(scaled_io_frames(256, 48_000.0), 256);
        assert_eq!(scaled_io_frames(256, 16_000.0), 85);
        assert_eq!(scaled_io_frames(256, 44_100.0), 235);
        assert_eq!(scaled_io_frames(128, 96_000.0), 256);
        assert_eq!(scaled_io_frames(128, 0.0), 128);
        assert_eq!(scaled_io_frames(128, f64::NAN), 128);
        assert_eq!(clamp_io_frames(85, 16.0, 4096.0), 85);
        assert_eq!(clamp_io_frames(85, 240.0, 4096.0), 240);
        assert_eq!(clamp_io_frames(512, 16.0, 256.0), 256);
        // min > max, NaN or an empty range must not panic.
        assert_eq!(clamp_io_frames(128, 512.0, 64.0), 128);
        assert_eq!(clamp_io_frames(128, f64::NAN, 64.0), 128);
        assert_eq!(clamp_io_frames(128, 0.0, 0.0), 128);
        assert_eq!(clamp_io_frames(0, 0.0, 1e9), 1);
    }

    #[test]
    fn latency_estimate_counts_the_output_block_once() {
        // 256 in, 128 out at 48 kHz, no device latency, 960-frame chain: the ring target (384)
        // already holds the output block.
        let target = target_fill(256, 48_000.0, 128, 48_000.0);
        let s = estimate_latency_s(256, 0, 48_000.0, 960, target, 0, 48_000.0);
        assert!((s - (256 + 960 + 384) as f64 / 48_000.0).abs() < 1e-12);
    }

    #[test]
    fn capture_survives_non_finite_input() {
        let (prod, mut cons) = RingBuffer::new(1 << 16);
        let mut cap = CaptureStage::new(44_100.0, 441, rnnoise(), prod, shared()).unwrap();
        for block in 0..200 {
            for (i, s) in cap.mono_mut()[..441].iter_mut().enumerate() {
                let k = (block * 441 + i) as f64;
                *s = 0.3 * (2.0 * std::f64::consts::PI * 1000.0 * k / 44_100.0).sin() as f32;
            }
            if block == 20 {
                cap.mono_mut()[7] = f32::NAN;
                cap.mono_mut()[8] = f32::INFINITY;
            }
            cap.process(441, 0);
        }
        let mut out = vec![0.0; cons.slots()];
        cons.pop_entire_slice(&mut out).unwrap();
        assert!(out.iter().all(|s| s.is_finite()));
        let (f, rms) = freq_rms(&out, 48_000.0, 48_000);
        assert!((f - 1000.0).abs() < 2.0, "frequency {f}");
        assert!((rms - 0.3 / 2f64.sqrt()).abs() < 0.01, "rms {rms}");
    }

    #[test]
    fn peak_is_in_dbfs() {
        assert_eq!(peak_dbfs(&[0.0; 4]), -120.0);
        assert!((peak_dbfs(&[0.5, -1.0]) - 0.0).abs() < 1e-6);
        assert!((peak_dbfs(&[0.1]) + 20.0).abs() < 1e-4);
    }

    #[test]
    fn target_fill_matches_documented_default() {
        assert_eq!(target_fill(256, 48_000.0, 128, 48_000.0), 384);
    }

    /// Frequency by zero crossings and RMS, skipping the first `skip` samples.
    fn freq_rms(x: &[f32], rate: f64, skip: usize) -> (f64, f64) {
        let x = &x[skip..];
        let crossings = x.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        let rms = (x.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
        (crossings as f64 * rate / x.len() as f64, rms)
    }

    #[test]
    fn capture_resamples_44k1_to_48k_preserving_pitch_and_level() {
        let (prod, mut cons) = RingBuffer::new(1 << 20);
        let mut cap = CaptureStage::new(44_100.0, 441, rnnoise(), prod, shared()).unwrap();
        let total = 44_100 * 2;
        let mut t = 0usize;
        for block in [441usize, 512, 37, 1000].iter().cycle() {
            if t >= total {
                break;
            }
            let n = (*block).min(total - t);
            for (i, s) in cap.mono_mut()[..n].iter_mut().enumerate() {
                let k = (t + i) as f64;
                *s = 0.3 * (2.0 * std::f64::consts::PI * 1000.0 * k / 44_100.0).sin() as f32;
            }
            cap.process(n, 0);
            t += n;
        }
        let got = cons.slots();
        let expect = total as f64 * 48_000.0 / 44_100.0;
        assert!(
            (got as f64 - expect).abs() < 200.0,
            "got {got} frames, expected ≈{expect}"
        );
        let mut out = vec![0.0; got];
        cons.pop_entire_slice(&mut out).unwrap();
        let (f, rms) = freq_rms(&out, 48_000.0, 2000);
        assert!((f - 1000.0).abs() < 2.0, "frequency {f}");
        assert!((rms - 0.3 / 2f64.sqrt()).abs() < 0.01, "rms {rms}");
    }

    #[test]
    fn capture_at_48k_is_bit_transparent_with_stub_chain_bypassed() {
        let (prod, cons) = RingBuffer::new(4096);
        let mut cap = CaptureStage::new(48_000.0, 256, rnnoise(), prod, shared()).unwrap();
        assert_eq!(cap.placement(), Placement::Inline);
        assert_eq!(cap.latency_frames(), rnnoise().latency());
        for (i, s) in cap.mono_mut()[..100].iter_mut().enumerate() {
            *s = i as f32 / 1000.0;
        }
        cap.process(100, 0);
        assert_eq!(cons.slots(), 100);
    }

    fn tone(k: usize, rate: f64) -> f32 {
        0.25 * (2.0 * std::f64::consts::PI * 440.0 * k as f64 / rate).sin() as f32
    }

    fn worker_of(cap: &CaptureStage) -> &Worker {
        match &cap.runner {
            ChainRunner::Worker(w) => w,
            ChainRunner::Inline(_) => panic!("chain runs inline"),
        }
    }

    /// Waits until the worker has processed everything the stage handed it.
    fn settle(cap: &CaptureStage) {
        let w = worker_of(cap);
        let t = Instant::now();
        while w.wet_read + w.from.slots() as i64 != w.in_pos - w.wet_base || w.flushing {
            assert!(
                t.elapsed() < Duration::from_secs(5),
                "worker did not catch up"
            );
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    struct Run {
        out: Vec<f32>,
        shared: Arc<Shared>,
        latency: usize,
        reserve: usize,
    }

    /// Feeds `blocks` device blocks of a 440 Hz tone through a capture stage. With
    /// `Placement::Worker`, the worker is paused during blocks for which `paused` is true and
    /// otherwise waited for after each block (so it never starves by accident).
    fn run_capture(
        rate: f64,
        block: usize,
        placement: Placement,
        blocks: usize,
        sh: Arc<Shared>,
        source: impl Fn(usize) -> f32,
        mut each: impl FnMut(usize, &Shared) -> bool,
    ) -> Run {
        let (prod, mut cons) = RingBuffer::new(1 << 22);
        let mut cap =
            CaptureStage::with_placement(rate, block, rnnoise(), prod, sh.clone(), placement)
                .unwrap();
        assert_eq!(cap.placement(), placement);
        let mut k = 0;
        let mut out = Vec::new();
        for b in 0..blocks {
            let paused = each(b, &sh);
            if placement == Placement::Worker {
                worker_of(&cap).sync.pause.store(paused, Release);
            }
            for s in &mut cap.mono_mut()[..block] {
                *s = source(k);
                k += 1;
            }
            cap.process(block, 0);
            if placement == Placement::Worker && !paused {
                settle(&cap);
            }
            let n = cons.slots();
            let at = out.len();
            out.resize(at + n, 0.0);
            cons.pop_entire_slice(&mut out[at..]).unwrap();
        }
        Run {
            out,
            latency: cap.latency_frames(),
            reserve: cap.reserve_frames(),
            shared: sh,
        }
    }

    fn max_step(x: &[f32]) -> f32 {
        x.windows(2).fold(0.0, |m, w| m.max((w[1] - w[0]).abs()))
    }

    #[test]
    fn worker_output_is_the_inline_output_delayed_by_the_reserve() {
        for (rate, block) in [(48_000.0, 256), (44_100.0, 441), (16_000.0, 85)] {
            let src = |k| tone(k, rate);
            let w = run_capture(
                rate,
                block,
                Placement::Worker,
                300,
                shared(),
                src,
                |_, _| false,
            );
            let r = run_capture(
                rate,
                block,
                Placement::Inline,
                300,
                shared(),
                src,
                |_, _| false,
            );
            assert_eq!(w.shared.starvations.load(Relaxed), 0, "{rate}");
            assert_eq!(w.shared.worker_overflows.load(Relaxed), 0, "{rate}");
            assert_eq!(w.out.len(), r.out.len());
            let res = w.reserve;
            assert_eq!(
                res,
                worker_reserve(max_block48(block, rate, rate != 48_000.0))
            );
            assert_eq!(
                w.latency,
                r.latency + res,
                "{rate}: latency includes the reserve"
            );
            assert_eq!(r.reserve, 0);
            eprintln!(
                "{rate} Hz / {block}: reserve {res} samples ({:.2} ms)",
                res as f64 / 48.0
            );
            // Past the start-up fade the worker path is bit-identical to the inline path, `res`
            // samples later.
            let d = w.latency - r.latency + rnnoise().latency() + WORKER_FADE;
            for k in d..w.out.len() {
                assert_eq!(w.out[k], r.out[k - res], "{rate} Hz, sample {k}");
            }
            assert!(max_step(&w.out) < 0.05, "{rate}: {}", max_step(&w.out));
            // The worker recorded its priming apart and a time for every block.
            assert!(w.shared.worker_times.calls() > 1);
            assert!(w.shared.worker_times.period_ns.load(Relaxed) > 0);
        }
    }

    #[test]
    fn starvation_crossfades_to_dry_and_back_without_going_backwards() {
        let rate = 48_000.0;
        let src = |k| tone(k, rate);
        let paused = |b: usize, _: &Shared| (100..103).contains(&b);
        let w = run_capture(rate, 256, Placement::Worker, 300, shared(), src, paused);
        let r = run_capture(rate, 256, Placement::Inline, 300, shared(), src, |_, _| {
            false
        });
        let starved = w.shared.starvations.load(Relaxed);
        eprintln!("starved blocks: {starved}");
        assert!((2..=4).contains(&starved), "starvations {starved}");
        assert_eq!(w.shared.worker_overflows.load(Relaxed), 0);
        // Dry audio filled the gap: the tone never dropped out.
        let gap = &w.out[101 * 256..104 * 256];
        let rms = (gap.iter().map(|s| s * s).sum::<f32>() / gap.len() as f32).sqrt();
        assert!(
            (rms - 0.25 / 2f32.sqrt()).abs() < 0.03,
            "rms in the gap {rms}"
        );
        // Crossfades, no steps: a 440 Hz tone at 0.25 moves < 0.015 per sample; the wet/dry
        // difference (the high-pass's phase shift) is spread over WORKER_FADE samples.
        let step = max_step(&w.out[1000..]);
        assert!(step < 0.03, "largest step {step}");
        // Once caught up, exactly the inline output at the same fixed delay again: the stale
        // wet audio was discarded rather than played late.
        let res = w.reserve;
        for k in 106 * 256..w.out.len() {
            assert_eq!(w.out[k], r.out[k - res], "sample {k}");
        }
    }

    /// The reserve normally leaves a fade's worth of wet audio beyond every block. If wet audio
    /// ever vanished at a block start anyway, the fade-out continues on the last wet-minus-dry
    /// difference rather than cutting to dry.
    #[test]
    fn wet_running_out_without_lookahead_still_fades() {
        let rate = 48_000.0;
        let src = |k| tone(k, rate);
        let steal = |b: usize, _: &Shared| b >= 100;
        let (prod, mut cons) = RingBuffer::new(1 << 22);
        let sh = shared();
        let mut cap =
            CaptureStage::with_placement(rate, 256, rnnoise(), prod, sh.clone(), Placement::Worker)
                .unwrap();
        let mut out = Vec::new();
        for b in 0..104 {
            worker_of(&cap).sync.pause.store(steal(b, &sh), Release);
            if b == 100 {
                // Everything the worker delivered disappears, as if it had never arrived.
                let ChainRunner::Worker(w) = &mut cap.runner else {
                    unreachable!()
                };
                let k = w.discard_wet(usize::MAX);
                w.wet_read += k as i64;
            }
            for (i, s) in cap.mono_mut()[..256].iter_mut().enumerate() {
                *s = src(b * 256 + i);
            }
            cap.process(256, 0);
            if !steal(b, &sh) {
                settle(&cap);
            }
            let n = cons.slots();
            let at = out.len();
            out.resize(at + n, 0.0);
            cons.pop_entire_slice(&mut out[at..]).unwrap();
        }
        assert!(sh.starvations.load(Relaxed) >= 1);
        let step = max_step(&out[1000..]);
        assert!(step < 0.03, "largest step {step}");
    }

    #[test]
    fn worker_overflow_drops_the_backlog_and_recovers() {
        let rate = 48_000.0;
        let src = |k| tone(k, rate);
        // 80 blocks of 256 overfill the 16384-sample input ring.
        let paused = |b: usize, _: &Shared| (50..130).contains(&b);
        let w = run_capture(rate, 256, Placement::Worker, 400, shared(), src, paused);
        let r = run_capture(rate, 256, Placement::Inline, 400, shared(), src, |_, _| {
            false
        });
        assert_eq!(w.shared.worker_overflows.load(Relaxed), 1);
        assert!(w.shared.starvations.load(Relaxed) > 60);
        assert!(w.out.iter().all(|s| s.is_finite()));
        assert!(max_step(&w.out[1000..]) < 0.03);
        // Back on the fixed delay; the chain's filters have long settled after the jump.
        let res = w.reserve;
        let tail = w.out.len() - 50 * 256;
        for k in tail..w.out.len() {
            assert!((w.out[k] - r.out[k - res]).abs() < 1e-4, "sample {k}");
        }
    }

    #[test]
    fn bypass_and_strength_reach_the_worker_chain() {
        let audio = oto_dsp::signal::noisy_speech(48_000 * 4, 3);
        // Denoise on, then strength changes, bypass and back, at block boundaries.
        let schedule = |b: usize, s: &Shared| {
            let (on, db) = match b {
                0..100 => (true, 24.0),
                100..200 => (true, 12.0),
                200..300 => (false, 12.0),
                300..450 => (true, f32::INFINITY),
                _ => (true, 24.0),
            };
            s.denoise.store(on, Relaxed);
            s.max_atten_bits.store(f32::to_bits(db), Relaxed);
            false
        };
        let src = |k: usize| audio[k];
        let w = run_capture(
            48_000.0,
            256,
            Placement::Worker,
            700,
            shared(),
            src,
            schedule,
        );
        let r = run_capture(
            48_000.0,
            256,
            Placement::Inline,
            700,
            shared(),
            src,
            schedule,
        );
        // The worker saw every change at the same point in the stream as the inline chain.
        let res = w.reserve;
        let d = w.latency - r.latency + rnnoise().latency() + WORKER_FADE;
        for k in d..w.out.len() {
            assert_eq!(w.out[k], r.out[k - res], "sample {k}");
        }
        // And the changes did something: a run that stays on 24 dB differs.
        let fixed = |_: usize, s: &Shared| {
            s.denoise.store(true, Relaxed);
            false
        };
        let f = run_capture(48_000.0, 256, Placement::Worker, 700, shared(), src, fixed);
        let differs = |a: usize, b: usize| (a * 256..b * 256).any(|k| w.out[k] != f.out[k]);
        assert!(differs(110, 200) && differs(210, 300) && differs(310, 450));
    }

    #[test]
    fn stop_fade_ramps_to_silence_and_reports_done() {
        let sh = shared();
        let (mut prod, cons) = RingBuffer::new(RING_CAPACITY);
        let mut out = OutputStage::new(cons, 48_000.0, 300, sh.clone()).unwrap();
        prod.push_entire_slice(&[0.5; 2000]).unwrap();
        let mut buf = [0.0f32; 128];
        for _ in 0..4 {
            out.render(&mut buf);
        }
        assert!(buf.iter().all(|&s| (s - 0.5).abs() < 1e-3));
        sh.stop_fade.store(STOP_FADE_REQUESTED, Release);
        let mut faded = Vec::new();
        for _ in 0..6 {
            out.render(&mut buf);
            faded.extend_from_slice(&buf);
        }
        assert_eq!(sh.stop_fade.load(Acquire), STOP_FADE_DONE);
        // 10 ms at 48 kHz = 480 samples: a ramp, then held silence.
        assert!(faded[479..].iter().all(|&s| s == 0.0));
        assert!(
            max_step(&faded) <= 0.5 / 480.0 + 1e-6,
            "{}",
            max_step(&faded)
        );
        assert!(faded[0] > 0.49);
        // A new stage (the next start) clears the request.
        let (_p, c) = RingBuffer::<f32>::new(16);
        let _next = OutputStage::new(c, 48_000.0, 300, sh.clone()).unwrap();
        assert_eq!(sh.stop_fade.load(Relaxed), STOP_FADE_IDLE);
    }

    struct SimResult {
        mean_fill: f64,
        corr_ppm: f64,
        late_underruns: u64,
        freq: f64,
        rms: f64,
    }

    /// Mic at `48 kHz × (1 + ppm)` in blocks of `in_block`, sink at exactly 48 kHz in blocks
    /// of `out_block`, connected by the real stages and ring. Event-driven simulated time.
    fn simulate(ppm: f64, in_block: usize, out_block: usize, seconds: f64) -> SimResult {
        let sh = shared();
        let (prod, cons) = RingBuffer::new(RING_CAPACITY);
        let mut cap = CaptureStage::new(48_000.0, 256, rnnoise(), prod, sh.clone()).unwrap();
        let target = target_fill(in_block, 48_000.0, out_block, 48_000.0);
        let mut out = OutputStage::new(cons, 48_000.0, target, sh.clone()).unwrap();
        let in_period = in_block as f64 / (48_000.0 * (1.0 + ppm * 1e-6));
        let out_period = out_block as f64 / 48_000.0;
        let (mut tp, mut tc) = (0.0, 0.003);
        let mut phase = 0u64;
        let mut buf = vec![0.0; out_block];
        let (mut fill_sum, mut fill_n, mut corr_sum) = (0.0, 0u64, 0.0);
        let mut underruns_at_settle = None;
        let mut tail = Vec::new();
        while tc < seconds {
            if tp <= tc {
                for s in cap.mono_mut()[..in_block].iter_mut() {
                    // 997 Hz at the *mic's* clock.
                    *s = 0.25
                        * (2.0 * std::f64::consts::PI * 997.0 * phase as f64 / 48_000.0).sin()
                            as f32;
                    phase += 1;
                }
                cap.process(in_block, 0);
                tp += in_period;
            } else {
                out.render(&mut buf);
                if tc > seconds * 0.6 {
                    underruns_at_settle.get_or_insert(sh.underruns.load(Relaxed));
                    fill_sum += sh.fill_frames.load(Relaxed) as f64;
                    corr_sum += sh.correction_ppm.load(Relaxed) as f64;
                    fill_n += 1;
                    if tail.len() < 48_000 * 5 {
                        tail.extend_from_slice(&buf);
                    }
                }
                tc += out_period;
            }
        }
        let (freq, rms) = freq_rms(&tail, 48_000.0, 0);
        SimResult {
            mean_fill: fill_sum / fill_n as f64,
            corr_ppm: corr_sum / fill_n as f64,
            late_underruns: sh.underruns.load(Relaxed) - underruns_at_settle.unwrap(),
            freq,
            rms,
        }
    }

    #[test]
    fn drift_controller_converges_under_clock_mismatch() {
        for (ppm, inb, outb) in [
            (200.0, 256, 128),
            (-200.0, 256, 128),
            (200.0, 480, 512),
            (-200.0, 512, 441),
            (0.0, 256, 256),
        ] {
            let target = target_fill(inb, 48_000.0, outb, 48_000.0) as f64;
            let r = simulate(ppm, inb, outb, 200.0);
            eprintln!(
                "{ppm:+} ppm {inb}/{outb}: fill {:.1}/{target} frames, correction {:.1} ppm, \
                 {:.2} Hz, rms {:.4}",
                r.mean_fill, r.corr_ppm, r.freq, r.rms
            );
            assert!(
                (r.mean_fill - target).abs() < 0.1 * target,
                "{ppm} ppm {inb}/{outb}: mean fill {} vs target {target}",
                r.mean_fill
            );
            // The controller has learned the clock offset (sign: producer fast → consume faster).
            assert!(
                (r.corr_ppm - ppm).abs() < 30.0,
                "{ppm} ppm: correction {}",
                r.corr_ppm
            );
            assert_eq!(
                r.late_underruns, 0,
                "{ppm} ppm {inb}/{outb}: underruns after settling"
            );
            // The tone comes out at its pitch scaled by the sink/mic clock ratio, intact.
            let expect = 997.0 * (1.0 + ppm * 1e-6);
            assert!((r.freq - expect).abs() < 1.0, "{ppm} ppm: freq {}", r.freq);
            assert!(
                (r.rms - 0.25 / 2f64.sqrt()).abs() < 0.01,
                "{ppm} ppm: rms {}",
                r.rms
            );
        }
    }

    #[test]
    fn controller_alone_tracks_offset() {
        // Pure model: fill integrates the rate difference.
        let mut c = DriftController::new(400);
        let mut fill = 400.0;
        let dt = 128.0 / 48_000.0;
        let offset = 200e-6;
        for _ in 0..(120.0 / dt) as usize {
            let corr = c.update(fill, dt);
            fill += 48_000.0 * (offset - corr) * dt;
        }
        assert!((fill - 400.0).abs() < 2.0, "fill {fill}");
    }

    #[test]
    fn output_primes_then_recovers_from_underrun() {
        let sh = shared();
        let (mut prod, cons) = RingBuffer::new(RING_CAPACITY);
        let mut out = OutputStage::new(cons, 48_000.0, 300, sh.clone()).unwrap();
        let mut buf = [1.0f32; 128];
        out.render(&mut buf);
        assert!(buf.iter().all(|&s| s == 0.0), "silence while priming");
        prod.push_entire_slice(&[0.5; 600]).unwrap();
        out.render(&mut buf);
        // Audio re-enters through a short fade, not a step.
        assert!(buf[0].abs() < 0.01, "starts from silence, got {}", buf[0]);
        assert!(
            buf.windows(2)
                .all(|w| w[1] >= w[0] - 1e-6 && w[1] - w[0] < 0.02),
            "fade-in is a smooth ramp"
        );
        out.render(&mut buf);
        assert!(
            buf[32..].iter().all(|&s| (s - 0.5).abs() < 1e-3),
            "plays at full level once faded in"
        );
        for _ in 0..4 {
            out.render(&mut buf);
        }
        assert!(sh.underruns.load(Relaxed) >= 1);
        // Back to priming: silence until the target is reached again.
        prod.push_entire_slice(&[0.5; 100]).unwrap();
        out.render(&mut buf);
        assert!(buf.iter().all(|&s| s == 0.0));
        // And the re-entry fades in again.
        prod.push_entire_slice(&[0.5; 600]).unwrap();
        out.render(&mut buf);
        assert!(
            buf[0].abs() < 0.01 && buf[127] > 0.3,
            "{} {}",
            buf[0],
            buf[127]
        );
    }

    struct StallResult {
        /// Underruns counted after the stall ended.
        underruns: u64,
        resyncs: u64,
        /// Seconds after the stall until the fill stayed within 5 ms of the target.
        cleared_after: f64,
        /// Lowest fill seen at an output callback after the stall, frames.
        min_fill: f64,
        /// Largest sample-to-sample step in the output after the stall.
        max_step: f32,
        /// Mean correction and fill over the last second.
        corr_ppm: f64,
        mean_fill: f64,
        target: f64,
    }

    /// Steady 256/128 streams with the mic 50 ppm fast; at 10 s the output misses callbacks for
    /// `stall_s` while the mic keeps delivering, then `after_s` more seconds of steady running.
    fn simulate_stall(stall_s: f64, after_s: f64) -> StallResult {
        const PPM: f64 = 50.0;
        let (in_block, out_block) = (256usize, 128usize);
        let sh = shared();
        let (prod, cons) = RingBuffer::new(RING_CAPACITY);
        let mut cap = CaptureStage::new(48_000.0, 256, rnnoise(), prod, sh.clone()).unwrap();
        let target = target_fill(in_block, 48_000.0, out_block, 48_000.0);
        let mut out = OutputStage::new(cons, 48_000.0, target, sh.clone()).unwrap();
        let in_period = in_block as f64 / (48_000.0 * (1.0 + PPM * 1e-6));
        let out_period = out_block as f64 / 48_000.0;
        let stall_at = 10.0;
        let resume = stall_at + stall_s;
        let end = resume + after_s;
        let (mut tp, mut tc) = (0.0, 0.003);
        let mut phase = 0u64;
        let mut buf = vec![0.0; out_block];
        let mut stalled = false;
        let mut underruns_at_resume = 0;
        let mut r = StallResult {
            underruns: 0,
            resyncs: 0,
            cleared_after: 0.0,
            min_fill: f64::MAX,
            max_step: 0.0,
            corr_ppm: 0.0,
            mean_fill: 0.0,
            target: target as f64,
        };
        let mut last = 0.0f32;
        let (mut n_tail, mut corr_sum, mut fill_sum) = (0u64, 0.0, 0.0);
        while tc < end {
            if tp <= tc {
                for s in cap.mono_mut()[..in_block].iter_mut() {
                    *s = 0.25
                        * (2.0 * std::f64::consts::PI * 300.0 * phase as f64 / 48_000.0).sin()
                            as f32;
                    phase += 1;
                }
                cap.process(in_block, 0);
                tp += in_period;
                continue;
            }
            if !stalled && tc >= stall_at {
                stalled = true;
                assert_eq!(sh.resyncs.load(Relaxed), 0, "no resync in steady state");
                underruns_at_resume = sh.underruns.load(Relaxed);
                tc += stall_s;
                continue;
            }
            out.render(&mut buf);
            if stalled {
                let fill = sh.fill_frames.load(Relaxed) as f64;
                r.min_fill = r.min_fill.min(fill);
                if (fill - target as f64).abs() > 240.0 {
                    r.cleared_after = tc + out_period - resume;
                }
                for &s in &buf {
                    r.max_step = r.max_step.max((s - last).abs());
                    last = s;
                }
                if tc > end - 1.0 {
                    corr_sum += sh.correction_ppm.load(Relaxed) as f64;
                    fill_sum += fill;
                    n_tail += 1;
                }
            } else {
                last = *buf.last().unwrap();
            }
            tc += out_period;
        }
        r.underruns = sh.underruns.load(Relaxed) - underruns_at_resume;
        r.resyncs = sh.resyncs.load(Relaxed);
        r.corr_ppm = corr_sum / n_tail as f64;
        r.mean_fill = fill_sum / n_tail as f64;
        r
    }

    /// A long output stall overfills the ring past `MAX_EXCESS`. The resync must leave the
    /// controller looking at the fill that remains: seeded with the pre-drop fill it pins at
    /// maximum correction and runs the ring dry a few seconds later.
    #[test]
    fn resync_after_long_stall_does_not_cause_a_follow_on_underrun() {
        let r = simulate_stall(0.3, 10.0);
        eprintln!(
            "0.3 s stall: {} resyncs, cleared after {:.2} s, min fill {}, correction {:.1} ppm",
            r.resyncs, r.cleared_after, r.min_fill, r.corr_ppm
        );
        assert!(r.resyncs >= 1, "the stale audio was dropped");
        assert_eq!(r.underruns, 0, "no underrun or re-prime after the resync");
        assert!(r.min_fill >= 128.0, "min fill {}", r.min_fill);
        assert!(r.cleared_after < 5.0, "cleared after {} s", r.cleared_after);
        assert!(
            (r.mean_fill - r.target).abs() < 0.1 * r.target,
            "fill {} vs target {}",
            r.mean_fill,
            r.target
        );
        // Back to tracking the 50 ppm clock offset, nowhere near the 2000 ppm clamp.
        assert!(
            (r.corr_ppm - 50.0).abs() < 100.0,
            "correction {}",
            r.corr_ppm
        );
    }

    /// A 50 ms stall leaves +50 ms in the ring, below the hard resync threshold. Bleeding that
    /// at 2000 ppm would take ~25 s; it must clear within a few seconds, without swinging below
    /// what one output block needs and without a click.
    #[test]
    fn moderate_excess_clears_quickly_without_undershoot() {
        let r = simulate_stall(0.05, 10.0);
        eprintln!(
            "50 ms stall: {} resyncs, cleared after {:.2} s, min fill {}, correction {:.1} ppm, \
             max step {:.4}",
            r.resyncs, r.cleared_after, r.min_fill, r.corr_ppm, r.max_step
        );
        assert_eq!(r.underruns, 0);
        assert!(r.cleared_after < 3.0, "cleared after {} s", r.cleared_after);
        assert!(r.min_fill >= 128.0, "min fill {}", r.min_fill);
        assert!(
            (r.mean_fill - r.target).abs() < 0.1 * r.target,
            "fill {} vs target {}",
            r.mean_fill,
            r.target
        );
        assert!(
            (r.corr_ppm - 50.0).abs() < 100.0,
            "correction {}",
            r.corr_ppm
        );
        // The trim is faded: a 300 Hz tone at 0.25 moves at most ~0.01 per sample, and the
        // 3 ms fades add little to that. A hard cut would step by up to 0.5.
        assert!(r.max_step < 0.03, "largest output step {}", r.max_step);
    }

    /// A small excess (5 ms) is bled by the controller alone, with no drop.
    #[test]
    fn small_excess_is_bled_without_a_drop_or_windup() {
        let r = simulate_stall(0.005, 12.0);
        eprintln!(
            "5 ms stall: {} resyncs, cleared after {:.2} s, min fill {}, mean fill {:.1}, \
             correction {:.1} ppm",
            r.resyncs, r.cleared_after, r.min_fill, r.mean_fill, r.corr_ppm
        );
        assert_eq!(r.resyncs, 0);
        assert_eq!(r.underruns, 0);
        assert!(r.min_fill >= 128.0, "min fill {}", r.min_fill);
        assert!(r.cleared_after < 6.0, "cleared after {} s", r.cleared_after);
        assert!(
            (r.mean_fill - r.target).abs() < 0.1 * r.target,
            "fill {} vs target {}",
            r.mean_fill,
            r.target
        );
        // Settling on the 50 ppm offset again, far from the 2000 ppm clamp.
        assert!(
            (r.corr_ppm - 50.0).abs() < 200.0,
            "correction {}",
            r.corr_ppm
        );
    }

    #[test]
    fn integrator_does_not_wind_up_while_saturated() {
        // 20 s pinned far above target, then back at target: the correction must drop to
        // (near) zero at once instead of staying high from a wound-up integrator.
        let mut c = DriftController::new(400);
        let dt = 128.0 / 48_000.0;
        c.update(400.0, dt);
        let mut corr = 0.0;
        for _ in 0..(20.0 / dt) as usize {
            corr = c.update(3000.0, dt);
        }
        assert_eq!(corr, DriftController::MAX_CORR);
        c.reseed(400.0);
        let corr = c.update(400.0, dt);
        assert!(corr.abs() < 20e-6, "correction after saturation {corr}");
    }
}
