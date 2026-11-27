//! End-to-end latency through the real virtual-mic path (`oto latency`).
//!
//! The engine runs with a [`Probe`]: the physical mic still clocks the capture callback, but its
//! samples are replaced by a 1 kHz tone burst every 0.5 s, stamped with the host-clock time at
//! which the mic delivered that sample. We then record from Oto Microphone like any app would
//! (which also makes the engine start), detect each burst onset and stamp it with the recorder's
//! host-clock time. The difference covers: mic IO buffer → input resampler → chain → ring →
//! output resampler → sink IO buffer → driver loopback → recorder. It excludes the physical mic's
//! own converter latency (reported separately by the device, see `Stats::estimated_latency_ms`).
//!
//! A denoiser can suppress the tone, so a run with the model active may detect only some
//! bursts. [`LatencyReport::check`] fails such a run. Measure with `Settings::enabled` false
//! (`oto latency --no-denoise`): the tier's model is still loaded and its delay kept, only its
//! processing is bypassed. Or use [`Stimulus::Speech`] ([`measure_with_stimulus`],
//! `oto latency --stimulus speech`): a voiced, vowel-like burst every 0.8 s that the models
//! keep, so the path is measured with denoising active. Each burst is stamped at its first
//! sample above [`Probe::THRESHOLD`], the same threshold the recorder detects with, so a
//! clean path measures no extra delay from the burst's attack.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::{Acquire, Relaxed};
use std::time::{Duration, Instant};

use crate::platform::sys;
use crate::{Engine, EngineOptions, Event, Probe, Settings, Stats, Stimulus};

#[derive(Clone, Debug)]
pub struct LatencyReport {
    /// The signal injected.
    pub stimulus: Stimulus,
    /// One entry per detected burst (matched 1:1 to individual injections), milliseconds.
    pub measurements_ms: Vec<f64>,
    pub injections: u64,
    /// Bursts that should have been detected: all injections, excluding a trailing undetected
    /// burst only if its detection window was still open when recording stopped.
    pub expected_bursts: u64,
    pub stats: Stats,
    /// Errors the engine reported during the run.
    pub errors: Vec<String>,
    /// Model fallbacks the engine reported during the run, as `requested → effective: reason`.
    pub notices: Vec<String>,
}

/// The share of injected bursts a run must detect to count, see [`LatencyReport::check`].
pub const MIN_DETECTED: f64 = 0.8;

impl LatencyReport {
    /// (min, median, max) in ms, if anything was measured.
    pub fn summary(&self) -> Option<(f64, f64, f64)> {
        let mut v = self.measurements_ms.clone();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        Some((v[0], v[v.len() / 2], v[v.len() - 1]))
    }

    /// The 95th percentile in ms (nearest rank), if anything was measured.
    pub fn p95_ms(&self) -> Option<f64> {
        let mut v = self.measurements_ms.clone();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        let rank = (0.95 * v.len() as f64).ceil() as usize;
        Some(v[rank.clamp(1, v.len()) - 1])
    }

    /// Bursts the run should have detected: all injections, excluding a trailing undetected
    /// burst only if its detection window was still open when recording stopped.
    pub fn expected(&self) -> u64 {
        self.expected_bursts
    }

    /// Expected bursts that were not detected.
    pub fn missed(&self) -> u64 {
        self.expected()
            .saturating_sub(self.measurements_ms.len() as u64)
    }

    /// [`missed`](Self::missed) as a share of [`expected`](Self::expected), 0.0 to 1.0 (0.0
    /// when nothing was expected).
    pub fn miss_rate(&self) -> f64 {
        match self.expected() {
            0 => 0.0,
            e => self.missed() as f64 / e as f64,
        }
    }

    /// Fails unless at least [`MIN_DETECTED`] of the [`expected`](Self::expected) bursts were
    /// detected. Fewer usually means the denoiser suppressed the tone (measure with denoising
    /// bypassed) or the path is dropping audio, and the figures would not be trustworthy.
    pub fn check(&self) -> Result<(), String> {
        let expected = self.expected();
        let detected = self.measurements_ms.len() as u64;
        let what = self.stimulus.name();
        if expected == 0 {
            return Err(format!(
                "no {what} bursts were injected; is the microphone delivering audio?"
            ));
        }
        if (detected as f64) < MIN_DETECTED * expected as f64 {
            let hint = match self.stimulus {
                Stimulus::Tone => "try --no-denoise or --stimulus speech",
                Stimulus::Speech => "try --no-denoise",
            };
            return Err(format!(
                "detected only {detected} of {expected} {what} bursts (need {:.0}%); the \
                 denoiser may be suppressing them ({hint}) or audio is being lost",
                MIN_DETECTED * 100.0
            ));
        }
        Ok(())
    }
}

/// Deduplicates per-injection detections (`(injection_id, latency_ms)`) and computes
/// `expected_bursts`: all injections `1..=injections`, excluding the final injection only when it
/// was NOT detected AND its detection window (`last_inject_ns + window_ns`) was still open when
/// recording stopped (`last_recorded_ns`).
fn reconcile_detections(
    injections: u64,
    last_inject_ns: u64,
    last_recorded_ns: u64,
    window_ns: u64,
    detections: &[(u64, f64)],
) -> (Vec<f64>, u64) {
    if injections == 0 {
        return (Vec::new(), 0);
    }
    let mut seen = BTreeSet::new();
    let mut measurements = Vec::new();
    for &(id, ms) in detections {
        if (1..=injections).contains(&id) && seen.insert(id) {
            measurements.push(ms);
        }
    }
    let last_detected = seen.contains(&injections);
    let trailing_open = !last_detected
        && last_inject_ns != 0
        && last_recorded_ns < last_inject_ns.saturating_add(window_ns);
    let expected = if trailing_open {
        injections - 1
    } else {
        injections
    };
    (measurements, expected)
}

/// [`measure_with_stimulus`] with tone bursts.
pub fn measure(
    settings: Settings,
    opts: EngineOptions,
    seconds: f64,
) -> Result<LatencyReport, String> {
    measure_with_stimulus(settings, opts, seconds, Stimulus::Tone)
}

/// Run the engine for `seconds` with `stimulus` bursts in place of the mic signal and measure
/// each burst's trip to Oto Microphone. Needs the driver.
pub fn measure_with_stimulus(
    settings: Settings,
    mut opts: EngineOptions,
    seconds: f64,
    stimulus: Stimulus,
) -> Result<LatencyReport, String> {
    let uids = opts.uids();
    if !matches!(sys::driver_status(&uids), crate::DriverStatus::Loaded(_)) {
        return Err("Oto driver is not installed (Oto Microphone / Oto Sink not found)".into());
    }
    let period_s = stimulus.period_s();
    let window_ns = (period_s * 0.8 * 1e9) as u64;
    let probe = Arc::new(Probe::with_stimulus(stimulus));
    opts.probe = Some(probe.clone());
    let engine = Engine::start_with(settings, opts)?;

    let rate = sys::device_status(&uids.mic, true)
        .map(|d| d.sample_rate)
        .filter(|r| *r > 0.0)
        .ok_or("cannot read Oto Microphone's sample rate")?;
    let (mut prod, mut cons) = rtrb::RingBuffer::<(u64, f64)>::new(1024);
    let p = probe.clone();
    let last_recorded_ns = Arc::new(AtomicU64::new(0));
    let rec_ns = last_recorded_ns.clone();
    let mut holdoff_until = 0u64;
    let recorder = sys::Recorder::start(
        &uids.mic,
        Box::new(move |data, ch, first_ns| {
            if first_ns == 0 || ch == 0 {
                return;
            }
            let frames = data.len() / ch;
            let end_ns = first_ns + (frames as f64 * 1e9 / rate) as u64;
            rec_ns.store(end_ns, Relaxed);
            for (k, frame) in data.chunks_exact(ch).enumerate() {
                let t = first_ns + (k as f64 * 1e9 / rate) as u64;
                if frame[0].abs() > Probe::THRESHOLD && t > holdoff_until {
                    holdoff_until = t + (period_s * 0.5 * 1e9) as u64;
                    // Read a consistent (id, timestamp) pair: `ProbeGen::fill` stores
                    // `last_inject_ns` before incrementing `injections` with Release ordering.
                    let id = p.injections.load(Acquire);
                    let inj = p.last_inject_ns.load(Acquire);
                    if id != 0 && p.injections.load(Acquire) == id && inj != 0 && t > inj {
                        let dt = t - inj;
                        if dt < window_ns {
                            let _ = prod.push((id, dt as f64 / 1e6));
                        }
                    }
                }
            }
        }),
    )?;

    let mut detections = Vec::new();
    let mut errors = Vec::new();
    let mut notices = Vec::new();
    let end = Instant::now() + Duration::from_secs_f64(seconds);
    while Instant::now() < end {
        std::thread::sleep(Duration::from_millis(50));
        while let Ok(d) = cons.pop() {
            detections.push(d);
        }
        for e in engine.try_events() {
            match e {
                Event::Error(e) => errors.push(e),
                Event::DriverIncompatible { loaded, .. } => {
                    errors.push(crate::incompatible_driver_message(loaded))
                }
                Event::Degraded(e) => errors.push(format!("degraded: {e}")),
                Event::ModelFallback {
                    requested,
                    effective,
                    reason,
                } => notices.push(format!(
                    "{} \u{2192} {}: {reason}",
                    requested.name(),
                    effective.name()
                )),
                _ => {}
            }
        }
    }
    let stats = engine.stats();
    drop(recorder);
    drop(engine);
    while let Ok(d) = cons.pop() {
        detections.push(d);
    }
    let injections = probe.injections.load(Acquire);
    let last_inject_ns = probe.last_inject_ns.load(Acquire);
    let rec_end_ns = last_recorded_ns.load(Relaxed);
    let (measurements_ms, expected_bursts) = reconcile_detections(
        injections,
        last_inject_ns,
        rec_end_ns,
        window_ns,
        &detections,
    );
    Ok(LatencyReport {
        stimulus,
        measurements_ms,
        injections,
        expected_bursts,
        stats,
        errors,
        notices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(injections: u64, expected_bursts: u64, detected: usize) -> LatencyReport {
        LatencyReport {
            stimulus: Stimulus::Tone,
            measurements_ms: vec![40.0; detected],
            injections,
            expected_bursts,
            stats: Stats::default(),
            errors: Vec::new(),
            notices: Vec::new(),
        }
    }

    #[test]
    fn a_run_must_detect_most_bursts() {
        assert!(report(0, 0, 0).check().is_err());
        assert!(
            report(1, 0, 0).check().is_err(),
            "nothing could be expected"
        );
        assert_eq!(report(21, 20, 20).check(), Ok(()));
        assert_eq!(report(21, 20, 20).expected(), 20);
        assert_eq!(report(21, 20, 16).check(), Ok(()));
        let e = report(21, 20, 15).check().unwrap_err();
        assert!(e.contains("15 of 20"), "{e}");
        assert!(report(21, 20, 0).check().is_err());
    }

    #[test]
    fn misses_and_percentiles() {
        let r = report(21, 20, 15);
        assert_eq!(r.missed(), 5);
        assert!((r.miss_rate() - 0.25).abs() < 1e-12);
        assert_eq!(report(0, 0, 0).miss_rate(), 0.0);
        assert_eq!(report(3, 3, 3).missed(), 0);
        assert_eq!(report(3, 3, 0).p95_ms(), None);
        let mut r = report(101, 100, 0);
        r.measurements_ms = (1..=100).map(f64::from).collect();
        r.measurements_ms.reverse();
        assert_eq!(r.p95_ms(), Some(95.0));
        r.measurements_ms = vec![7.0];
        assert_eq!(r.p95_ms(), Some(7.0));
        let mut s = report(21, 20, 15);
        s.stimulus = Stimulus::Speech;
        let err = s.check().unwrap_err();
        assert!(
            err.contains("speech bursts") && !err.contains("--stimulus speech"),
            "{err}"
        );
    }

    #[test]
    fn reconcile_catches_mid_run_drop_even_when_last_burst_is_detected() {
        // 8 injections; burst #4 dropped, final burst #8 WAS detected -> 1 real miss, not 0.
        let det: Vec<(u64, f64)> = [1, 2, 3, 5, 6, 7, 8]
            .into_iter()
            .map(|id| (id, 62.5))
            .collect();
        let (ms, expected) = reconcile_detections(8, 1_000, 1_050, 400, &det);
        assert_eq!((ms.len(), expected), (7, 8));
        let r = report(8, expected, ms.len());
        assert_eq!(r.missed(), 1);
        assert!((r.miss_rate() - 0.125).abs() < 1e-12);

        // 8 injections; final burst #8 NOT detected while its window is still open -> 7 expected.
        let det_open: Vec<(u64, f64)> = (1..=7).map(|id| (id, 62.5)).collect();
        let (ms, expected) = reconcile_detections(8, 1_000, 1_200, 400, &det_open);
        assert_eq!((ms.len(), expected), (7, 7));

        // 8 injections; final burst #8 NOT detected after its window closed -> 8 expected (1 miss).
        let (ms, expected) = reconcile_detections(8, 1_000, 1_500, 400, &det_open);
        assert_eq!((ms.len(), expected), (7, 8));

        // Duplicate detections for the same injection collapse to one.
        let (ms, expected) =
            reconcile_detections(2, 1_000, 1_500, 400, &[(1, 40.0), (1, 41.0), (2, 42.0)]);
        assert_eq!((ms, expected), (vec![40.0, 42.0], 2));
    }
}
