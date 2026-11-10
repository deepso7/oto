//! The realtime path, independent of any audio API: capture stage (downmix → resample to 48 kHz →
//! chain → ring) and output stage (ring → drift-correcting resampler → device).
//!
//! Everything here is called from device callbacks, so after construction nothing allocates,
//! locks or blocks. Cross-thread state is atomics in [`Shared`].
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
//! If the ring runs dry the output plays silence and re-primes to the target (counted as an
//! underrun). If it ever holds far more than the target (e.g. the output stalled), the excess
//! is dropped in one go (counted as a resync).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering::Relaxed};

use oto_dsp::{Chain, SAMPLE_RATE};
use rtrb::{Consumer, Producer};
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

const RATE: f64 = SAMPLE_RATE as f64;

/// Mean ring fill to aim for, in frames at 48 kHz. Block sizes are in device frames.
pub fn target_fill(in_block: usize, in_rate: f64, out_block: usize, out_rate: f64) -> usize {
    let in48 = in_block as f64 * RATE / in_rate;
    let out48 = out_block as f64 * RATE / out_rate;
    (out48 + in48 / 2.0).ceil() as usize + OUT_CHUNK + MARGIN_FRAMES
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
    /// Nanoseconds spent inside both callbacks, cumulative.
    pub busy_ns: AtomicU64,
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
        }
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
/// `out` must hold `input.len() / channels` frames. Call once per device buffer with
/// `scale = 1 / total_channels` after zeroing `out` to average all channels.
pub fn downmix_add(input: &[f32], channels: usize, out: &mut [f32], scale: f32) {
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

/// Peak level in dBFS, floored at -120.
pub fn peak_dbfs(buf: &[f32]) -> f32 {
    let peak = buf.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if peak <= 1e-6 {
        -120.0
    } else {
        20.0 * peak.log10()
    }
}

/// Mic side: device-rate mono in, 48 kHz processed audio into the ring.
pub struct CaptureStage {
    mono: Box<[f32]>,
    stage: Box<[f32]>,
    stage_len: usize,
    resampler: Option<Async<f32>>,
    out48: Box<[f32]>,
    chain: Chain,
    denoise: bool,
    atten: f32,
    producer: Producer<f32>,
    shared: Arc<Shared>,
    probe: Option<ProbeGen>,
    level_raw: f32,
    level_clean: f32,
    /// Whether `Chain::prime` has run on the audio thread yet.
    primed: bool,
}

impl CaptureStage {
    /// Allocates; call off the realtime thread.
    pub fn new(
        rate: f64,
        chain: Chain,
        producer: Producer<f32>,
        shared: Arc<Shared>,
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
        let mut chain = chain;
        let denoise = shared.denoise.load(Relaxed);
        let atten = f32::from_bits(shared.max_atten_bits.load(Relaxed));
        chain.set_enabled(denoise);
        chain.set_max_attenuation_db(atten);
        Ok(Self {
            mono: vec![0.0; MAX_BLOCK].into(),
            stage: vec![0.0; IN_CHUNK].into(),
            stage_len: 0,
            resampler,
            out48: vec![0.0; max_out].into(),
            chain,
            denoise,
            atten,
            producer,
            shared,
            probe,
            level_raw: -120.0,
            level_clean: -120.0,
            primed: false,
        })
    }

    /// Delay added by the input resampler and the chain, in frames at 48 kHz.
    pub fn latency_frames(&self) -> usize {
        let rs = self
            .resampler
            .as_ref()
            .map_or(0, |r| r.output_delay() + IN_CHUNK);
        rs + self.chain.latency()
    }

    /// Scratch buffer (`MAX_BLOCK` long) to downmix device audio into before [`Self::process`].
    pub fn mono_mut(&mut self) -> &mut [f32] {
        &mut self.mono
    }

    /// Process the first `n` (≤ `MAX_BLOCK`) samples of [`Self::mono_mut`]. `first_ns` is the
    /// host-clock capture time of the first sample, or 0 if unknown.
    pub fn process(&mut self, n: usize, first_ns: u64) {
        if !self.primed {
            // One-time per-thread model setup, so later callbacks never allocate.
            self.chain.prime();
            self.primed = true;
        }
        let n = n.min(MAX_BLOCK);
        if let Some(p) = &mut self.probe {
            p.fill(&mut self.mono[..n], first_ns);
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
        let denoise = self.shared.denoise.load(Relaxed);
        if denoise != self.denoise {
            self.denoise = denoise;
            self.chain.set_enabled(denoise);
        }
        let atten = f32::from_bits(self.shared.max_atten_bits.load(Relaxed));
        if atten.to_bits() != self.atten.to_bits() {
            self.atten = atten;
            self.chain.set_max_attenuation_db(atten);
        }
        self.level_raw = peak_dbfs(buf);
        self.chain.process(buf);
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
        let lim = Self::MAX_CORR / Self::KI;
        self.integ = (self.integ + err * dt).clamp(-lim, lim);
        (Self::KP * err + Self::KI * self.integ).clamp(-Self::MAX_CORR, Self::MAX_CORR)
    }

    /// Forget the smoothed fill (after a re-prime), keeping the learned clock offset.
    pub fn rebase(&mut self) {
        self.ema = None;
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
        })
    }

    /// Delay of the output resampler, in output frames.
    pub fn latency_frames(&self) -> usize {
        self.resampler.output_delay()
    }

    /// Fill `out` (mono, device rate).
    pub fn render(&mut self, out: &mut [f32]) {
        let pending48 = (self.pend_len - self.pend_pos) as f64 * RATE / self.rate;
        let ring = self.consumer.slots();
        let fill = ring as f64 + pending48;
        let target = self.ctl.target();
        self.shared.fill_frames.store(fill as u32, Relaxed);

        if self.priming {
            if fill < target {
                out.fill(0.0);
                return;
            }
            self.priming = false;
            self.ctl.rebase();
        }
        if fill > target + MAX_EXCESS as f64 {
            let drop = ring.saturating_sub(target as usize);
            if let Ok(chunk) = self.consumer.read_chunk(drop) {
                chunk.commit_all();
            }
            self.shared.resyncs.fetch_add(1, Relaxed);
            self.ctl.rebase();
        }

        let corr = self.ctl.update(fill, out.len() as f64 / self.rate);
        self.shared
            .correction_ppm
            .store((corr * 1e6) as i32, Relaxed);
        let _ = self.resampler.set_resample_ratio_relative(1.0 - corr, true);

        let mut at = 0;
        while at < out.len() {
            if self.pend_pos < self.pend_len {
                let n = (self.pend_len - self.pend_pos).min(out.len() - at);
                out[at..at + n].copy_from_slice(&self.pend[self.pend_pos..self.pend_pos + n]);
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

    fn shared() -> Arc<Shared> {
        let s = Shared::new(Arc::new(Levels::default()), None);
        // Keep the denoiser out of signal-shape assertions once the real chain lands.
        s.denoise.store(false, Relaxed);
        Arc::new(s)
    }

    #[test]
    fn downmix_averages_interleaved_channels() {
        let input = [1.0, 0.0, 0.5, 0.5, -1.0, 1.0];
        let mut out = [0.0; 3];
        downmix_add(&input, 2, &mut out, 0.5);
        assert_eq!(out, [0.5, 0.5, 0.0]);
        // A second buffer (e.g. a non-interleaved device) adds into the same mix.
        let mut out = [0.0; 2];
        downmix_add(&[1.0, 1.0], 1, &mut out, 0.5);
        downmix_add(&[0.0, -1.0], 1, &mut out, 0.5);
        assert_eq!(out, [0.5, 0.0]);
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
        let mut cap = CaptureStage::new(44_100.0, Chain::new(), prod, shared()).unwrap();
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
        let mut cap = CaptureStage::new(48_000.0, Chain::new(), prod, shared()).unwrap();
        assert_eq!(cap.latency_frames(), Chain::new().latency());
        for (i, s) in cap.mono_mut()[..100].iter_mut().enumerate() {
            *s = i as f32 / 1000.0;
        }
        cap.process(100, 0);
        assert_eq!(cons.slots(), 100);
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
        let mut cap = CaptureStage::new(48_000.0, Chain::new(), prod, sh.clone()).unwrap();
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
        prod.push_entire_slice(&[0.5; 300]).unwrap();
        out.render(&mut buf);
        assert!(
            buf[100..].iter().all(|&s| (s - 0.5).abs() < 1e-3),
            "plays once primed"
        );
        for _ in 0..3 {
            out.render(&mut buf);
        }
        assert!(sh.underruns.load(Relaxed) >= 1);
        // Back to priming: silence until the target is reached again.
        prod.push_entire_slice(&[0.5; 100]).unwrap();
        out.render(&mut buf);
        assert!(buf.iter().all(|&s| s == 0.0));
    }
}
