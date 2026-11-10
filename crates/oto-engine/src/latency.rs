//! End-to-end latency through the real virtual-mic path (`oto latency`).
//!
//! The engine runs with a [`Probe`]: the physical mic still clocks the capture callback, but its
//! samples are replaced by a 1 kHz tone burst every 0.5 s, stamped with the host-clock time at
//! which the mic delivered that sample. We then record from Oto Microphone like any app would
//! (which also makes the engine start), detect each burst onset and stamp it with the recorder's
//! host-clock time. The difference covers: mic IO buffer → input resampler → chain → ring →
//! output resampler → sink IO buffer → driver loopback → recorder. It excludes the physical mic's
//! own converter latency (reported separately by the device, see `Stats::estimated_latency_ms`).

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
}

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
    let end = Instant::now() + Duration::from_secs_f64(seconds);
    while Instant::now() < end {
        std::thread::sleep(Duration::from_millis(50));
        while let Ok(d) = cons.pop() {
            measurements.push(d);
        }
        for e in engine.try_events() {
            if let Event::Error(e) = e {
                errors.push(e);
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
    })
}
