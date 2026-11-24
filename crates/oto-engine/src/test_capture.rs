//! "Hear the Difference": record a few seconds from the mic, clean them offline through the
//! real chain, and play raw and cleaned back to back, level-matched, switching between them
//! without a click (plan 0002, Phase 2, `HEAR_TEST`).
//!
//! The UI should not offer this while an app is recording from Oto Microphone (an active
//! call): it opens the physical mic and the speakers.
//!
//! - [`record_and_process`] records from the physical mic Oto would use, downmixes exactly like
//!   live capture ([`crate::pipeline::downmix`]), runs [`crate::offline::process`] (delay
//!   compensated, so raw and cleaned line up sample for sample) and level-matches the two.
//! - [`TestPlayer`] loops one of them on the default output device; [`TestPlayer::set_cleaned`]
//!   crossfades over 10 ms, lock-free.

use std::sync::Arc;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::time::{Duration, Instant};

use crate::offline::{ProcessOptions, process, resample};
use crate::{EngineOptions, Strength, dsp, platform};

/// Recording length limits, seconds.
pub const MIN_SECONDS: f32 = 1.0;
pub const MAX_SECONDS: f32 = 10.0;
/// Peak ceiling for playback (−1 dBFS).
pub const PEAK_CEILING: f32 = 0.891_250_9;
/// Raw ↔ cleaned crossfade, seconds.
pub const CROSSFADE_S: f64 = 0.010;
/// Fade at each side of the loop point, seconds (20 ms across the boundary).
pub const LOOP_FADE_S: f64 = 0.010;
/// Mic rates at or below this, on a Bluetooth device, mean the hands-free (HFP) profile.
pub const HFP_MAX_RATE: f64 = 24_000.0;

/// A recorded take: the raw mic and the cleaned version, aligned and level-matched.
#[derive(Clone, Debug)]
pub struct TestRecording {
    /// Rate of `raw` and `cleaned` (the mic's), Hz.
    pub sample_rate: u32,
    pub raw: Arc<[f32]>,
    pub cleaned: Arc<[f32]>,
    /// The take was recorded from a Bluetooth headset in its narrowband hands-free profile
    /// (read once the mic was open): the UI should warn that it sounds narrower than normal.
    pub is_bluetooth_hfp: bool,
}

/// Whether a mic is a Bluetooth headset in the hands-free profile.
pub fn is_bluetooth_hfp(bluetooth: bool, rate: f64) -> bool {
    bluetooth && rate > 0.0 && rate <= HFP_MAX_RATE
}

/// Records `seconds` (clamped to 1–10) from `device_uid` (or the input Oto would use), cleans
/// it with `model` at `strength` and level-matches the two. Stops early, keeping what was
/// recorded, when `cancel` is set. Blocks; call off the UI thread.
pub fn record_and_process(
    device_uid: Option<&str>,
    model: dsp::Model,
    strength: Strength,
    seconds: f32,
    cancel: &AtomicBool,
) -> Result<TestRecording, String> {
    if cancel.load(Relaxed) {
        return Err("cancelled".into());
    }
    let seconds = if seconds.is_finite() {
        seconds.clamp(MIN_SECONDS, MAX_SECONDS)
    } else {
        MIN_SECONDS
    };
    let uids = EngineOptions::default().uids();
    // With Oto Microphone as the default input, the engine captures from the physical default
    // it will restore; so does the test.
    let last = crate::state::default_dir().and_then(|d| crate::state::remembered_input(&d));
    let uid = platform::sys::resolve_physical_input(&uids, device_uid, last.as_deref())
        .ok_or("no microphone found")?;
    if cancel.load(Relaxed) {
        return Err("cancelled".into());
    }
    let (raw, rate) = record(&uid, seconds, cancel)?;
    if cancel.load(Relaxed) {
        return Err("cancelled".into());
    }
    // Read with the mic open: that is when macOS moves a Bluetooth headset to hands-free.
    let hfp = is_bluetooth_hfp(platform::sys::is_bluetooth(&uid), rate);
    if cancel.load(Relaxed) {
        return Err("cancelled".into());
    }
    if raw.len() < (0.25 * rate) as usize {
        return Err("too little audio was recorded".into());
    }
    let opts = ProcessOptions {
        model,
        enabled: true,
        max_attenuation_db: strength.max_attenuation_db(),
        ..ProcessOptions::default()
    };
    let rate_u = rate.round() as u32;
    let (cleaned, _) = process(&raw, rate_u, &opts)?;
    let (raw, cleaned) = level_match(raw, cleaned, rate_u);
    Ok(TestRecording {
        sample_rate: rate_u,
        raw: raw.into(),
        cleaned: cleaned.into(),
        is_bluetooth_hfp: hfp,
    })
}

/// Records mono from `uid`, downmixed like live capture, and returns it with the rate it was
/// recorded at: the device's rate once audio flows, since opening a Bluetooth headset's mic
/// switches it to the hands-free profile and its lower rate.
fn record(uid: &str, seconds: f32, cancel: &AtomicBool) -> Result<(Vec<f32>, f64), String> {
    if cancel.load(Relaxed) {
        return Err("cancelled".into());
    }
    let before = platform::sys::nominal_rate(uid).ok_or("cannot read the microphone's rate")?;
    if cancel.load(Relaxed) {
        return Err("cancelled".into());
    }
    // Opening only ever lowers the rate (hands-free); the buffer fits the higher of the two.
    let cap = (seconds as f64 * before.max(48_000.0)).ceil() as usize;
    let (mut prod, mut cons) = rtrb::RingBuffer::<f32>::new(cap + crate::pipeline::MAX_BLOCK);
    let got = Arc::new(AtomicUsize::new(0));
    let got2 = got.clone();
    let callback = Box::new(move |mono: &[f32]| {
        let k = mono.len().min(prod.slots());
        if let Ok(chunk) = prod.write_chunk_uninit(k) {
            chunk.fill_from_iter(mono[..k].iter().copied());
        }
        got2.fetch_add(k, Release);
    });
    if cancel.load(Relaxed) {
        return Err("cancelled".into());
    }
    let recorder = platform::sys::MonoRecorder::start(uid, cancel, callback)?;
    let deadline = Instant::now() + Duration::from_secs_f32(seconds + 3.0);
    let waiting = || !cancel.load(Relaxed) && Instant::now() <= deadline;
    while got.load(Acquire) == 0 && waiting() {
        std::thread::sleep(Duration::from_millis(5));
    }
    let rate = platform::sys::nominal_rate(uid).ok_or("cannot read the microphone's rate")?;
    let want = ((seconds as f64 * rate).ceil() as usize).min(cap);
    while got.load(Acquire) < want && waiting() {
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(recorder);
    let n = cons.slots().min(want);
    let mut out = vec![0.0; n];
    cons.pop_entire_slice(&mut out)
        .map_err(|_| "recording buffer error")?;
    if n == 0 && !cancel.load(Relaxed) {
        return Err("the microphone delivered no audio".into());
    }
    Ok((out, rate))
}

/// Scales `raw` so its level over the cleaned take's active speech matches `cleaned`'s, so
/// neither wins an A/B on loudness. If that would push either peak above −1 dBFS, both are
/// turned down together. Active speech: 10 ms frames of `cleaned` within 30 dB of its loudest
/// frame and above −60 dBFS.
pub fn level_match(mut raw: Vec<f32>, mut cleaned: Vec<f32>, rate: u32) -> (Vec<f32>, Vec<f32>) {
    let n = raw.len().min(cleaned.len());
    let frame = (rate as usize / 100).max(1);
    let energy = |x: &[f32]| x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    let frames: Vec<(usize, f64)> = (0..n / frame)
        .map(|k| {
            (
                k,
                energy(&cleaned[k * frame..(k + 1) * frame]) / frame as f64,
            )
        })
        .collect();
    let loudest = frames.iter().fold(0.0f64, |m, f| m.max(f.1));
    let floor = (loudest * 1e-3).max(1e-6); // −30 dB re the loudest, and −60 dBFS
    let (mut e_raw, mut e_clean) = (0.0, 0.0);
    for &(k, e) in frames.iter().filter(|f| f.1 >= floor) {
        e_clean += e;
        e_raw += energy(&raw[k * frame..(k + 1) * frame]) / frame as f64;
    }
    let mut g_raw = if e_raw > 0.0 && e_clean > 0.0 {
        (e_clean / e_raw).sqrt() as f32
    } else {
        1.0
    };
    let peak = |x: &[f32]| x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (p_raw, p_clean) = (peak(&raw) * g_raw, peak(&cleaned));
    let top = p_raw.max(p_clean);
    let mut g_clean = 1.0;
    if top > PEAK_CEILING {
        let down = PEAK_CEILING / top;
        g_raw *= down;
        g_clean = down;
    }
    raw.iter_mut().for_each(|v| *v *= g_raw);
    cleaned.iter_mut().for_each(|v| *v *= g_clean);
    (raw, cleaned)
}

/// The playback state: loops both takes in step and crossfades between them. Realtime-safe.
struct Looper {
    raw: Arc<[f32]>,
    cleaned: Arc<[f32]>,
    pos: usize,
    /// 0 = raw, 1 = cleaned.
    mix: f32,
    step: f32,
    edge: usize,
    want: Arc<AtomicBool>,
}

impl Looper {
    fn new(raw: Arc<[f32]>, cleaned: Arc<[f32]>, rate: f64, want: Arc<AtomicBool>) -> Self {
        let mix = if want.load(Relaxed) { 1.0 } else { 0.0 };
        Self {
            raw,
            cleaned,
            pos: 0,
            mix,
            step: (1.0 / (CROSSFADE_S * rate).max(1.0)) as f32,
            edge: ((LOOP_FADE_S * rate) as usize).max(1),
            want,
        }
    }

    fn render(&mut self, out: &mut [f32]) {
        let len = self.raw.len().min(self.cleaned.len());
        if len == 0 {
            out.fill(0.0);
            return;
        }
        let target = if self.want.load(Acquire) { 1.0 } else { 0.0 };
        let edge = self.edge.min(len / 2).max(1);
        for o in out {
            self.mix = if target > self.mix {
                (self.mix + self.step).min(target)
            } else {
                (self.mix - self.step).max(target)
            };
            let i = self.pos;
            let s = self.raw[i] + (self.cleaned[i] - self.raw[i]) * self.mix;
            let fade = (i.min(len - 1 - i) as f32 / edge as f32).min(1.0);
            *o = s * fade;
            self.pos = (i + 1) % len;
        }
    }
}

/// Loops a [`TestRecording`] on the default output device until dropped.
pub struct TestPlayer {
    cleaned: Arc<AtomicBool>,
    _player: platform::sys::Player,
}

impl TestPlayer {
    /// Starts looping, raw or cleaned. Resolves the output device and resamples to its rate
    /// atomically so device selection and rate cannot race.
    pub fn start(recording: &TestRecording, play_cleaned: bool) -> Result<Self, String> {
        let from = recording.sample_rate;
        let want = Arc::new(AtomicBool::new(play_cleaned));
        let want2 = want.clone();
        let raw_in = recording.raw.clone();
        let cleaned_in = recording.cleaned.clone();
        let player = platform::sys::Player::start(move |rate| {
            let to = rate.round() as u32;
            let raw: Arc<[f32]> = resample(&raw_in, from, to)?.into();
            let cleaned: Arc<[f32]> = resample(&cleaned_in, from, to)?.into();
            let mut looper = Looper::new(raw, cleaned, rate, want2);
            Ok(Box::new(move |out: &mut [f32]| looper.render(out)))
        })?;
        Ok(Self {
            cleaned: want,
            _player: player,
        })
    }

    /// Switch between raw and cleaned (a 10 ms crossfade, lock-free).
    pub fn set_cleaned(&self, cleaned: bool) {
        self.cleaned.store(cleaned, Release);
    }

    pub fn is_cleaned(&self) -> bool {
        self.cleaned.load(Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::signal::{db, rms, sine, speech_like, white_noise};

    #[test]
    fn hfp_is_bluetooth_at_a_narrowband_rate() {
        assert!(is_bluetooth_hfp(true, 16_000.0));
        assert!(is_bluetooth_hfp(true, 8_000.0));
        assert!(is_bluetooth_hfp(true, 24_000.0));
        assert!(
            !is_bluetooth_hfp(true, 48_000.0),
            "AirPods in A2DP-only use"
        );
        assert!(
            !is_bluetooth_hfp(false, 16_000.0),
            "a USB headset at 16 kHz"
        );
        assert!(!is_bluetooth_hfp(true, 0.0));
    }

    #[test]
    fn level_match_equalises_speech_level_and_respects_the_ceiling() {
        let speech = speech_like(48_000 * 2, 3);
        // Raw: speech plus noise, 6 dB hotter than the cleaned take.
        let noise = white_noise(0.05, speech.len(), 4);
        let raw: Vec<f32> = speech
            .iter()
            .zip(&noise)
            .map(|(s, n)| 2.0 * s + n)
            .collect();
        let cleaned = speech.clone();
        let (r, c) = level_match(raw, cleaned.clone(), 48_000);
        assert_eq!(c, cleaned, "cleaned under the ceiling is untouched");
        // Over active speech frames the two now match.
        let active: Vec<usize> = (0..speech.len() / 480)
            .filter(|k| rms(&speech[k * 480..(k + 1) * 480]) > 0.05)
            .collect();
        let pick = |x: &[f32]| -> Vec<f32> {
            active
                .iter()
                .flat_map(|k| x[k * 480..(k + 1) * 480].to_vec())
                .collect()
        };
        let d = db(rms(&pick(&r)) / rms(&pick(&c)));
        assert!(d.abs() < 0.5, "level mismatch {d:.2} dB");
        // A take that would clip after matching turns both down together.
        let loud: Vec<f32> = speech.iter().map(|s| 1.9 * s).collect();
        let quiet: Vec<f32> = speech.iter().map(|s| 0.5 * s).collect();
        let (r, c) = level_match(quiet, loud, 48_000);
        let peak = |x: &[f32]| x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak(&r) <= PEAK_CEILING + 1e-6 && peak(&c) <= PEAK_CEILING + 1e-6);
        assert!(
            db(rms(&r) / rms(&c)).abs() < 0.1,
            "still matched after turning down"
        );
        // Silence in, silence out, no NaN.
        let (r, c) = level_match(vec![0.0; 1000], vec![0.0; 1000], 48_000);
        assert!(r.iter().chain(&c).all(|v| *v == 0.0));
    }

    #[test]
    fn switching_and_looping_are_click_free() {
        let rate = 48_000.0;
        let n = 48_000;
        // Different tones, so a hard switch would be a step of up to ~0.6.
        let raw: Arc<[f32]> = sine(440.0, 0.3, n).into();
        let cleaned: Arc<[f32]> = sine(660.0, 0.3, n).into();
        let want = Arc::new(AtomicBool::new(false));
        let mut l = Looper::new(raw, cleaned, rate, want.clone());
        let mut out = Vec::new();
        let mut buf = [0.0f32; 128];
        for b in 0..1200 {
            if b % 37 == 5 {
                want.store(!want.load(Relaxed), Relaxed);
            }
            l.render(&mut buf);
            out.extend_from_slice(&buf);
        }
        // Two 0.3 sines at ≤ 660 Hz move < 0.026 per sample; fades add < 0.03.
        let step = out
            .windows(2)
            .fold(0.0f32, |m, w| m.max((w[1] - w[0]).abs()));
        assert!(step < 0.06, "largest step {step}");
        // It loops: 1200 × 128 samples is more than 3 passes, and every loop point is faded.
        for k in 1..3 {
            assert!(out[k * n].abs() < 1e-6, "loop point {k} not faded");
        }
        // Settled on cleaned → exactly cleaned in the middle of the take.
        want.store(true, Relaxed);
        let mut more = vec![0.0f32; n];
        l.render(&mut more);
        l.render(&mut more);
        let pos = l.pos;
        let mid = (n / 2 + n - pos) % n;
        let expect = 0.3 * (std::f32::consts::TAU * 660.0 * (n / 2) as f32 / 48_000.0).sin();
        assert!((more[mid] - expect).abs() < 1e-3);
    }
}
