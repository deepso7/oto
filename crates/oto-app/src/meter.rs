//! The raw-vs-cleaned level meter. A metering thread polls the engine's lock-free [`Levels`] at
//! 60 Hz and publishes bar rects through Rune `Live` values, which redraw without re-running the
//! view. It never touches the audio callback, and sleeps while nothing would show.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use oto_engine::Levels;
use rune::Live;

/// Width and height of a meter bar, in points.
pub const BAR_W: f32 = 220.0;
pub const BAR_H: f32 = 6.0;
/// The quietest level the bar shows; anything below reads as empty.
pub const FLOOR_DB: f32 = -60.0;
/// How fast a falling bar drops, in dB per second (attack is instant).
pub const RELEASE_DB_PER_S: f32 = 30.0;

/// The meter's live bar rects plus whether anything is worth metering right now.
#[derive(Clone)]
pub struct Meter {
    pub raw: Live<[f32; 4]>,
    pub clean: Live<[f32; 4]>,
    /// Set by the UI: on while Oto is enabled and an app records from Oto Microphone. The engine
    /// only runs audio then, so levels outside it would be stale.
    pub active: Arc<AtomicBool>,
}

impl Default for Meter {
    fn default() -> Self {
        Meter {
            raw: Live::new(rect(0.0)),
            clean: Live::new(rect(0.0)),
            active: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Meter {
    /// Shows fixed levels (snapshots, tests).
    pub fn show(&self, raw_db: f32, clean_db: f32) {
        self.raw.set(rect(fraction(raw_db)));
        self.clean.set(rect(fraction(clean_db)));
    }
}

/// dBFS to the bar's filled share, 0 to 1. Silence (-inf) and NaN are empty.
pub fn fraction(db: f32) -> f32 {
    if db.is_nan() {
        return 0.0;
    }
    ((db - FLOOR_DB) / -FLOOR_DB).clamp(0.0, 1.0)
}

/// The filled part of a bar for `share`, in the track's points, its end on a half point so it
/// sits on a Retina pixel.
pub fn rect(share: f32) -> [f32; 4] {
    [
        0.0,
        0.0,
        (share.clamp(0.0, 1.0) * BAR_W * 2.0).round() / 2.0,
        BAR_H,
    ]
}

/// Peak-meter ballistics: rises at once, falls at [`RELEASE_DB_PER_S`].
pub fn ballistics(shown_db: f32, new_db: f32, dt: f32) -> f32 {
    let new_db = if new_db.is_finite() {
        new_db.max(FLOOR_DB)
    } else {
        FLOOR_DB
    };
    let fallen = (shown_db - RELEASE_DB_PER_S * dt).max(FLOOR_DB);
    new_db.max(fallen)
}

/// Polls `levels` into `meter` while `visible()` and the meter is active; otherwise empties the
/// bars once and naps.
pub fn spawn(levels: Arc<Levels>, meter: Meter, visible: impl Fn() -> bool + Send + 'static) {
    let _ = std::thread::Builder::new()
        .name("oto-meter".into())
        .spawn(move || {
            const FRAME: Duration = Duration::from_micros(16_667);
            let (mut raw, mut clean) = (FLOOR_DB, FLOOR_DB);
            let mut last = Instant::now();
            loop {
                if !meter.active.load(Ordering::Relaxed) || !visible() {
                    if raw > FLOOR_DB || clean > FLOOR_DB {
                        (raw, clean) = (FLOOR_DB, FLOOR_DB);
                        meter.show(FLOOR_DB, FLOOR_DB);
                    }
                    std::thread::sleep(Duration::from_millis(200));
                    last = Instant::now();
                    continue;
                }
                let now = Instant::now();
                let dt = now.duration_since(last).as_secs_f32();
                last = now;
                raw = ballistics(raw, levels.raw(), dt);
                clean = ballistics(clean, levels.clean(), dt);
                meter.show(raw, clean);
                std::thread::sleep(FRAME.saturating_sub(now.elapsed()));
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_db_to_share() {
        assert_eq!(fraction(0.0), 1.0);
        assert_eq!(fraction(-30.0), 0.5);
        assert_eq!(fraction(-90.0), 0.0);
        assert_eq!(fraction(f32::NEG_INFINITY), 0.0);
        assert_eq!(fraction(f32::NAN), 0.0);
        assert_eq!(fraction(6.0), 1.0);
    }

    #[test]
    fn rect_spans_the_track() {
        assert_eq!(rect(1.0), [0.0, 0.0, BAR_W, BAR_H]);
        assert_eq!(rect(0.0)[2], 0.0);
        assert_eq!(rect(2.0)[2], BAR_W);
    }

    #[test]
    fn rises_at_once_and_falls_slowly() {
        assert_eq!(ballistics(-60.0, -10.0, 0.016), -10.0);
        let fell = ballistics(-10.0, -60.0, 0.1);
        assert!((fell - -13.0).abs() < 1e-4, "{fell}");
        assert_eq!(ballistics(-59.0, f32::NEG_INFINITY, 1.0), FLOOR_DB);
    }
}
