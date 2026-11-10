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
//! processing is bypassed.

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

use crate::platform::sys;
use crate::{Engine, EngineOptions, Event, Probe, Settings, Stats};

#[derive(Clone, Debug)]
pub struct LatencyReport {
    /// One entry per detected burst, milliseconds.
    pub measurements_ms: Vec<f64>,
    pub injections: u64,
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

    /// Bursts the run should have detected: all injected but the last, which may still have
    /// been on its way when the run ended.
    pub fn expected(&self) -> u64 {
        self.injections.saturating_sub(1)
    }

    /// Fails unless at least [`MIN_DETECTED`] of the [`expected`](Self::expected) bursts were
    /// detected. Fewer usually means the denoiser suppressed the tone (measure with denoising
    /// bypassed) or the path is dropping audio, and the figures would not be trustworthy.
    pub fn check(&self) -> Result<(), String> {
        let expected = self.expected();
        let detected = self.measurements_ms.len() as u64;
        if expected == 0 {
            return Err("no tone bursts were injected; is the microphone delivering audio?".into());
        }
        if (detected as f64) < MIN_DETECTED * expected as f64 {
            return Err(format!(
                "detected only {detected} of {expected} tone bursts (need {:.0}%); the \
                 denoiser may be suppressing the tone (try --no-denoise) or audio is being lost",
                MIN_DETECTED * 100.0
            ));
        }
        Ok(())
    }
}

const THRESHOLD: f32 = 0.1;

pub fn measure(
    settings: Settings,
    mut opts: EngineOptions,
    seconds: f64,
) -> Result<LatencyReport, String> {
    let uids = opts.uids();
    if !sys::driver_installed(&uids) {
        return Err("Oto driver is not installed (Oto Microphone / Oto Sink not found)".into());
    }
    let probe = Arc::new(Probe::new());
    opts.probe = Some(probe.clone());
    let engine = Engine::start_with(settings, opts)?;

    let rate = sys::device_status(&uids.mic, true)
        .map(|d| d.sample_rate)
        .filter(|r| *r > 0.0)
        .ok_or("cannot read Oto Microphone's sample rate")?;
    let (mut prod, mut cons) = rtrb::RingBuffer::<f64>::new(1024);
    let p = probe.clone();
    let mut holdoff_until = 0u64;
    let recorder = sys::Recorder::start(
        &uids.mic,
        Box::new(move |data, ch, first_ns| {
            if first_ns == 0 {
                return;
            }
            for (k, frame) in data.chunks_exact(ch).enumerate() {
                let t = first_ns + (k as f64 * 1e9 / rate) as u64;
                if frame[0].abs() > THRESHOLD && t > holdoff_until {
                    holdoff_until = t + (Probe::PERIOD_S * 0.5 * 1e9) as u64;
                    let inj = p.last_inject_ns.load(Relaxed);
                    if inj != 0 && t > inj {
                        let d = (t - inj) as f64 / 1e6;
                        if d < Probe::PERIOD_S * 1000.0 * 0.8 {
                            let _ = prod.push(d);
                        }
                    }
                }
            }
        }),
    )?;

    let mut measurements = Vec::new();
    let mut errors = Vec::new();
    let mut notices = Vec::new();
    let end = Instant::now() + Duration::from_secs_f64(seconds);
    while Instant::now() < end {
        std::thread::sleep(Duration::from_millis(50));
        while let Ok(d) = cons.pop() {
            measurements.push(d);
        }
        for e in engine.try_events() {
            match e {
                Event::Error(e) => errors.push(e),
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
        measurements.push(d);
    }
    Ok(LatencyReport {
        measurements_ms: measurements,
        injections: probe.injections.load(Relaxed),
        stats,
        errors,
        notices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(injections: u64, detected: usize) -> LatencyReport {
        LatencyReport {
            measurements_ms: vec![40.0; detected],
            injections,
            stats: Stats::default(),
            errors: Vec::new(),
            notices: Vec::new(),
        }
    }

    #[test]
    fn a_run_must_detect_most_bursts() {
        assert!(report(0, 0).check().is_err());
        assert!(report(1, 0).check().is_err(), "nothing could be expected");
        assert_eq!(report(21, 20).check(), Ok(()));
        // The last burst may still be in flight.
        assert_eq!(report(21, 20).expected(), 20);
        assert_eq!(report(21, 16).check(), Ok(()));
        let e = report(21, 15).check().unwrap_err();
        assert!(e.contains("15 of 20"), "{e}");
        assert!(report(21, 0).check().is_err());
    }
}
