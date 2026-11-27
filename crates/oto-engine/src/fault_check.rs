//! Fault injection for Voice Isolation's fail-closed design (plan 0002, Phase 2).
//!
//! Every case drives the real realtime path — [`CaptureStage`] (with its DSP worker),
//! [`OutputStage`] and, for mode switches, the real control thread logic — with loud
//! speech-like audio (−12 dBFS peak) standing in for the TV or a colleague, injects one fault
//! and counts what reaches the output:
//!
//! | case | fault |
//! |---|---|
//! | `startup` | streams start: the worker has no wet audio yet |
//! | `starvation` | the DSP worker stalls for a few blocks |
//! | `overflow_flush` | the worker stalls a whole ring, which is flushed and restarted |
//! | `inference_error` | the real Nearfield model's inferences fail |
//! | `load_failure` | Voice Isolation fails to load while a non-isolating model runs |
//! | `mode_switch_preload` | Voice Isolation loads (slowly) while a non-isolating model runs |
//!
//! The two mode-switch cases each run twice: with the old stream's output callback answering
//! the mute late but in time, and with it stalled (the streams are then stopped before the
//! model loads).
//!
//! The isolating model in the plumbing cases is a perfect isolator (its output is exact
//! silence), so *any* output above −60 dBFS can only be the raw mic leaking through a fallback
//! path. `inference_error` runs the real model and counts output during the failed frames,
//! which must be silence. A case whose fault did not actually happen is an error, not a pass.
//!
//! `oto fault-check --json` prints [`Report::to_json`], which `tools/eval/eval.py
//! --fault-report` checks against `[voice_isolation.fault_injection]` in `gates.toml`. It needs
//! the `nearfield` feature: without the real model, `inference_error` would only exercise the
//! stand-in, so [`run_all`] refuses instead of reporting a pass.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use oto_dsp::{Chain, Denoiser, Model};
use rtrb::RingBuffer;

use crate::control::Control;
use crate::pipeline::{
    CaptureStage, OutputStage, RING_CAPACITY, STOP_FADE_S, Shared, decode_model, target_fill,
};
use crate::platform::Platform;
use crate::{Command, Event, InputDevice, Levels, Settings, Strength, Tier};

/// The `MODELS.toml` key the report is for.
pub const MODEL_KEY: &str = "nearfield_v2_3_fast";
/// The cases in `gates.toml`, in its order.
pub const CASES: [&str; 6] = [
    "startup",
    "starvation",
    "overflow_flush",
    "inference_error",
    "load_failure",
    "mode_switch_preload",
];
pub const STRENGTHS: [(&str, Strength); 3] = [
    ("light", Strength::Light),
    ("balanced", Strength::Balanced),
    ("strong", Strength::Strong),
];
/// A sample above this magnitude (−60 dBFS) is a leak.
pub const LEAK_THRESHOLD: f32 = 0.001;
/// Frames for `leaked_frames`: 10 ms at 48 kHz.
pub const LEAK_FRAME: usize = 480;

const RATE: f64 = 48_000.0;
/// Device block for both simulated streams (5.3 ms).
const BLOCK: usize = 256;
const WAIT: Duration = Duration::from_secs(10);

/// One case at one strength.
#[derive(Clone, Debug, PartialEq)]
pub struct CaseResult {
    pub case: &'static str,
    pub strength: &'static str,
    /// Output samples above [`LEAK_THRESHOLD`] where the design promises silence.
    pub leaked_samples: u64,
    /// [`LEAK_FRAME`]-sample frames containing any leaked sample.
    pub leaked_frames: u64,
    /// Times another model ran (or a fallback was announced) instead of Voice Isolation.
    pub fallback_to_other_model: u64,
    /// Output samples checked.
    pub checked_samples: u64,
    /// What stood in for Voice Isolation: `"silencer"` (perfect isolator) or `"nearfield"`.
    pub isolator: &'static str,
}

impl CaseResult {
    pub fn passed(&self) -> bool {
        self.leaked_samples == 0 && self.leaked_frames == 0 && self.fallback_to_other_model == 0
    }
}

/// Every case at every strength.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub model: &'static str,
    pub cases: Vec<CaseResult>,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.cases.len() == CASES.len() * STRENGTHS.len() && self.cases.iter().all(|c| c.passed())
    }

    /// The JSON `eval.py --fault-report` reads.
    pub fn to_json(&self) -> String {
        let cases: Vec<String> = self
            .cases
            .iter()
            .map(|c| {
                format!(
                    "{{\"case\":\"{}\",\"strength\":\"{}\",\"leaked_samples\":{},\
                     \"leaked_frames\":{},\"fallback_to_other_model\":{},\
                     \"checked_samples\":{},\"isolator\":\"{}\"}}",
                    c.case,
                    c.strength,
                    c.leaked_samples,
                    c.leaked_frames,
                    c.fallback_to_other_model,
                    c.checked_samples,
                    c.isolator
                )
            })
            .collect();
        format!(
            "{{\"model\":\"{}\",\"leak_threshold_dbfs\":-60.0,\"passed\":{},\"cases\":[{}]}}",
            self.model,
            self.passed(),
            cases.join(",")
        )
    }
}

/// Runs all cases at all strengths. `Err` if a case could not inject its fault, or if this
/// build has no Nearfield to check.
pub fn run_all() -> Result<Report, String> {
    if !cfg!(feature = "nearfield") {
        return Err(
            "fault-check requires the `nearfield` feature to verify the real Nearfield model"
                .into(),
        );
    }
    run_cases()
}

/// [`run_all`] without the feature check: without `nearfield`, `inference_error` runs the
/// stand-in (unit tests of the plumbing).
fn run_cases() -> Result<Report, String> {
    let mut cases = Vec::new();
    for case in CASES {
        for (name, strength) in STRENGTHS {
            cases.push(run_case(case, name, strength)?);
        }
    }
    Ok(Report {
        model: MODEL_KEY,
        cases,
    })
}

/// Runs one case. `strength_name` is copied into the result.
fn run_case(
    case: &'static str,
    strength_name: &'static str,
    strength: Strength,
) -> Result<CaseResult, String> {
    let mut r = match case {
        "startup" => startup(strength),
        "starvation" => worker_stall(strength, 3, false),
        "overflow_flush" => worker_stall(strength, RING_CAPACITY / BLOCK + 16, true),
        "inference_error" => inference_error(strength),
        "load_failure" => mode_switch(strength, false),
        "mode_switch_preload" => mode_switch(strength, true),
        _ => Err(format!("unknown fault case {case:?}")),
    }
    .map_err(|e| format!("{case}/{strength_name}: {e}"))?;
    r.case = case;
    r.strength = strength_name;
    Ok(r)
}

/// Loud background speech: two talkers' worth of [`oto_dsp::signal::speech_like`], so it has
/// few gaps, at −12 dBFS peak.
fn background(seconds: f32) -> Arc<[f32]> {
    let n = (seconds * RATE as f32) as usize;
    let a = oto_dsp::signal::speech_like(n, 0xB6);
    let b = oto_dsp::signal::speech_like(n, 0x7F1);
    let mut x: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a + 0.8 * b).collect();
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
    let g = 10f32.powf(-12.0 / 20.0) / peak;
    x.iter_mut().for_each(|v| *v *= g);
    x.into()
}

/// A perfect isolator: exact silence out, with Nearfield's frame and delay.
struct Silencer;

impl Denoiser for Silencer {
    fn frame_size(&self) -> usize {
        384
    }
    fn latency(&self) -> usize {
        2014
    }
    fn process_frame(&mut self, _input: &[f32], output: &mut [f32]) {
        output.fill(0.0);
    }
    fn set_max_attenuation_db(&mut self, _db: f32) -> bool {
        true
    }
    /// Fails like Nearfield does: silence (which is what it outputs anyway).
    fn inject_fault(&mut self, _frames: u32) -> bool {
        true
    }
}

pub(crate) fn silencer() -> Chain {
    Chain::with_denoiser(Model::Nearfield, Box::new(Silencer))
}

/// Counts leaked samples and frames in a stream of output.
#[derive(Default)]
struct Leaks {
    samples: u64,
    frames: u64,
    checked: u64,
    pos: usize,
    hit: bool,
}

impl Leaks {
    fn feed(&mut self, x: &[f32]) {
        for &v in x {
            if v.abs() > LEAK_THRESHOLD || !v.is_finite() {
                self.samples += 1;
                self.hit = true;
            }
            self.checked += 1;
            self.pos += 1;
            if self.pos == LEAK_FRAME {
                self.frames += self.hit as u64;
                self.hit = false;
                self.pos = 0;
            }
        }
    }

    fn result(mut self, fallbacks: u64, isolator: &'static str) -> CaseResult {
        self.frames += self.hit as u64;
        CaseResult {
            case: "",
            strength: "",
            leaked_samples: self.samples,
            leaked_frames: self.frames,
            fallback_to_other_model: fallbacks,
            checked_samples: self.checked,
            isolator,
        }
    }
}

fn shared(strength: Strength) -> Arc<Shared> {
    let s = Shared::new(Arc::new(Levels::default()), None);
    s.denoise.store(true, Relaxed);
    s.max_atten_bits
        .store(strength.max_attenuation_db().to_bits(), Relaxed);
    Arc::new(s)
}

/// Capture and output stages at 48 kHz, pumped in lockstep on the calling thread.
struct Rig {
    cap: CaptureStage,
    out: OutputStage,
}

impl Rig {
    fn new(chain: Chain, shared: &Arc<Shared>) -> Result<Self, String> {
        let (prod, cons) = RingBuffer::new(RING_CAPACITY);
        let cap = CaptureStage::new(RATE, BLOCK, chain, prod, shared.clone())?;
        let target = target_fill(BLOCK, RATE, BLOCK, RATE);
        let out = OutputStage::new(cons, RATE, target, shared.clone())?;
        Ok(Self { cap, out })
    }

    /// One capture callback of `src` from `*pos`, then one output callback into `out`. With
    /// `wait`, lets the worker finish first (no accidental starvation).
    fn step(&mut self, src: &[f32], pos: &mut usize, wait: bool, out: &mut [f32]) {
        for s in &mut self.cap.mono_mut()[..BLOCK] {
            *s = src[*pos];
            *pos = (*pos + 1) % src.len();
        }
        self.cap.process(BLOCK, 0);
        if wait {
            self.cap.wait_worker_idle(WAIT);
        }
        self.out.render(out);
    }
}

fn startup(strength: Strength) -> Result<CaseResult, String> {
    let sh = shared(strength);
    let src = background(2.0);
    let mut rig = Rig::new(silencer(), &sh)?;
    let (mut pos, mut buf, mut leaks) = (0, [0.0f32; BLOCK], Leaks::default());
    for _ in 0..400 {
        rig.step(&src, &mut pos, true, &mut buf);
        leaks.feed(&buf);
    }
    if sh.underruns.load(Relaxed) > 1 {
        return Err("the simulated streams underran".into());
    }
    Ok(leaks.result(0, "silencer"))
}

/// Stops the worker for `blocks` capture blocks mid-stream, then lets it catch up.
fn worker_stall(strength: Strength, blocks: usize, overflow: bool) -> Result<CaseResult, String> {
    let sh = shared(strength);
    let src = background(3.0);
    let mut rig = Rig::new(silencer(), &sh)?;
    let (mut pos, mut buf, mut leaks) = (0, [0.0f32; BLOCK], Leaks::default());
    for b in 0..(300 + blocks) {
        let stalled = (150..150 + blocks).contains(&b);
        rig.cap.fault_pause_worker(stalled);
        rig.step(&src, &mut pos, !stalled, &mut buf);
        leaks.feed(&buf);
    }
    let (starved, ovf) = (
        sh.starvations.load(Relaxed),
        sh.worker_overflows.load(Relaxed),
    );
    if starved == 0 || (overflow && ovf == 0) {
        return Err(format!(
            "fault did not happen ({starved} starvations, {ovf} overflows)"
        ));
    }
    if sh.degraded.load(Relaxed) {
        return Err("still degraded after the worker caught up".into());
    }
    Ok(leaks.result(0, "silencer"))
}

/// Real Nearfield (the stand-in without the `nearfield` feature, for unit tests only; see
/// [`run_all`]) through the chain: failed inferences must be exact silence.
/// The model passes a lone voice, so only the failed frames' output is checked.
fn inference_error(strength: Strength) -> Result<CaseResult, String> {
    #[cfg(feature = "nearfield")]
    let (mut chain, isolator) = (Chain::with_model(Model::Nearfield)?, "nearfield");
    #[cfg(not(feature = "nearfield"))]
    let (mut chain, isolator) = (silencer(), "silencer");
    chain.set_max_attenuation_db(strength.max_attenuation_db());
    let src = background(3.0);
    const FRAME: usize = 384;
    const AT: usize = 150;
    const FAILS: usize = 40;
    let mut leaks = Leaks::default();
    let mut buf = [0.0f32; FRAME];
    for (k, block) in src.as_chunks::<FRAME>().0.iter().enumerate() {
        if k == AT && !chain.inject_inference_error(FAILS as u32) {
            return Err("the model has no fault hook".into());
        }
        buf.copy_from_slice(block);
        chain.process(&mut buf);
        // Frame k's output is played during block k + 1.
        if (AT + 1..AT + 1 + FAILS).contains(&k) {
            leaks.feed(&buf);
        }
    }
    Ok(leaks.result(0, isolator))
}

/// A platform whose streams are a [`Rig`], pumped by the case.
struct Sim {
    state: Rc<RefCell<SimState>>,
}

struct SimState {
    rig: Option<Rig>,
    src: Arc<[f32]>,
    pos: usize,
    /// Starts with a non-isolating chain after the switch to Voice Isolation was asked for.
    other_starts: u64,
    switching: bool,
    /// Counting starts when the switch is asked for, after the 10 ms fade-out it is allowed.
    counting: bool,
    skip: usize,
    leaks: Leaks,
    /// Output level before the switch, to show the old stream did pass the background.
    before_rms: f64,
    before_n: u64,
    /// Whether the old streams still ran when Voice Isolation started loading.
    rig_at_load: Option<bool>,
}

impl SimState {
    fn pump(&mut self, blocks: usize) {
        let mut buf = [0.0f32; BLOCK];
        for _ in 0..blocks {
            match &mut self.rig {
                Some(r) => r.step(&self.src, &mut self.pos, true, &mut buf),
                // No streams: the driver plays silence on Oto Microphone.
                None => buf.fill(0.0),
            }
            if !self.counting && self.switching {
                self.counting = true;
                self.skip = (STOP_FADE_S * RATE).ceil() as usize;
            }
            if self.counting {
                let k = self.skip.min(BLOCK);
                self.skip -= k;
                self.leaks.feed(&buf[k..]);
            } else if !self.switching {
                self.before_rms += buf.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
                self.before_n += BLOCK as u64;
            }
        }
    }
}

impl Platform for Sim {
    fn driver_status(&self) -> crate::DriverStatus {
        crate::DriverStatus::Loaded(crate::BUNDLED_DRIVER_VERSION)
    }
    fn clients_active(&self) -> bool {
        true
    }
    fn input_devices(&self) -> Vec<InputDevice> {
        Vec::new()
    }
    fn resolve_input(&self, _: Option<&str>) -> Option<String> {
        Some("sim".into())
    }
    fn start(&mut self, _: &str, chain: Chain, shared: &Arc<Shared>) -> Result<(), String> {
        let mut st = self.state.borrow_mut();
        if st.switching && !chain.model().is_isolating() {
            st.other_starts += 1;
        }
        st.rig = Some(Rig::new(chain, shared)?);
        Ok(())
    }
    fn stop(&mut self) {
        self.state.borrow_mut().rig = None;
    }
    fn refresh(&mut self) {}
}

/// How the old stream's output callback answers the mute before Voice Isolation loads.
#[derive(Clone, Copy, Debug, PartialEq)]
enum MuteAnswer {
    /// A few ms late, but within the wait: the stream fades out and keeps running, muted,
    /// while the model loads.
    Late,
    /// Not within the wait (stalled): the streams are stopped before the model loads, so a
    /// callback arriving during the load has nothing to play.
    Stalled,
}

/// [`mode_switch_once`] with both [`MuteAnswer`]s, results added up.
fn mode_switch(strength: Strength, ok: bool) -> Result<CaseResult, String> {
    let a = mode_switch_once(strength, ok, MuteAnswer::Late)?;
    let b = mode_switch_once(strength, ok, MuteAnswer::Stalled)?;
    Ok(CaseResult {
        leaked_samples: a.leaked_samples + b.leaked_samples,
        leaked_frames: a.leaked_frames + b.leaked_frames,
        fallback_to_other_model: a.fallback_to_other_model + b.fallback_to_other_model,
        checked_samples: a.checked_samples + b.checked_samples,
        ..a
    })
}

/// A non-isolating model runs; the user picks Voice Isolation, which loads slowly (`ok`) or
/// fails to load. Counts output from the end of the old stream's 10 ms mute fade on.
fn mode_switch_once(
    strength: Strength,
    ok: bool,
    answer: MuteAnswer,
) -> Result<CaseResult, String> {
    let sh = shared(strength);
    let state = Rc::new(RefCell::new(SimState {
        rig: None,
        src: background(4.0),
        pos: 0,
        other_starts: 0,
        switching: false,
        counting: false,
        skip: 0,
        leaks: Leaks::default(),
        before_rms: 0.0,
        before_n: 0,
        rig_at_load: None,
    }));
    let loader_state = state.clone();
    let loader = Box::new(move |m: Model| {
        if !m.is_isolating() {
            return Chain::with_model(m);
        }
        // The model takes a while to load; audio keeps flowing meanwhile.
        let mut st = loader_state.borrow_mut();
        st.rig_at_load = Some(st.rig.is_some());
        st.pump(60);
        drop(st);
        if ok {
            Ok(silencer())
        } else {
            Err("injected load failure".into())
        }
    });
    let tier_model: fn(Tier) -> Model = |t| match t {
        Tier::Isolate => Model::Nearfield,
        _ => Model::RNNoise,
    };
    let settings = Settings {
        enabled: true,
        input_device: None,
        strength,
        tier: Tier::Light,
    };
    let (tx, rx) = mpsc::channel();
    let mut c = Control::new(
        Sim {
            state: state.clone(),
        },
        settings,
        true,
        sh.clone(),
        tx,
    )
    .with_models(tier_model, loader);
    if answer == MuteAnswer::Late {
        // The output callback first runs on the third poll of the mute wait.
        let (st, mut polls) = (state.clone(), 0);
        c = c.with_mute_wait(Box::new(move || {
            polls += 1;
            if polls >= 3 {
                st.borrow_mut().pump(1);
            }
        }));
    }
    c.reconcile();
    state.borrow_mut().pump(150);
    {
        let st = state.borrow();
        let rms = (st.before_rms / st.before_n.max(1) as f64).sqrt();
        if rms < 0.01 {
            return Err(format!(
                "the non-isolating stream was not audible before the switch (rms {rms})"
            ));
        }
    }
    state.borrow_mut().switching = true;
    c.handle(crate::control::Msg::Cmd(Command::SetTier(Tier::Isolate)));
    c.reconcile();
    for _ in 0..10 {
        state.borrow_mut().pump(20);
        c.tick(std::time::Instant::now());
        c.reconcile();
    }
    let events: Vec<Event> = rx.try_iter().collect();
    let announced = events
        .iter()
        .filter(|e| matches!(e, Event::ModelFallback { .. }))
        .count() as u64;
    let running = decode_model(sh.model.load(Relaxed));
    let wrong_model = running.is_some_and(|m| !m.is_isolating()) as u64;
    drop(c);
    let st = Rc::try_unwrap(state)
        .map_err(|_| "simulation still borrowed")?
        .into_inner();
    let expect_running = answer == MuteAnswer::Late;
    if st.rig_at_load != Some(expect_running) {
        return Err(format!(
            "{answer:?}: the old streams were {} when Voice Isolation started loading",
            match st.rig_at_load {
                Some(true) => "running",
                Some(false) => "stopped",
                None => "never seen",
            }
        ));
    }
    if ok && running != Some(Model::Nearfield) {
        return Err(format!("Voice Isolation is not running ({running:?})"));
    }
    if !ok && !events.iter().any(|e| matches!(e, Event::Degraded(_))) {
        return Err(format!(
            "no Event::Degraded after the load failure: {events:?}"
        ));
    }
    let fallbacks = announced + wrong_model + st.other_starts;
    Ok(st.leaks.result(fallbacks, "silencer"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Phase 2 hard bar: every case at every strength, zero leak, zero fallback.
    #[test]
    fn every_fault_case_fails_closed() {
        let _rt = crate::REALTIME_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let report = run_cases().unwrap();
        for c in &report.cases {
            eprintln!("{c:?}");
        }
        assert_eq!(report.cases.len(), 18);
        for c in &report.cases {
            assert_eq!(
                (c.leaked_samples, c.leaked_frames, c.fallback_to_other_model),
                (0, 0, 0),
                "{}/{}",
                c.case,
                c.strength
            );
            assert!(
                c.checked_samples > 0,
                "{}/{} checked nothing",
                c.case,
                c.strength
            );
        }
        assert!(report.passed());
        let json = report.to_json();
        assert!(json.starts_with("{\"model\":\"nearfield_v2_3_fast\""));
        assert_eq!(json.matches("\"case\":").count(), 18);
    }

    /// A build without the real model refuses to report on it rather than passing a stand-in.
    #[test]
    fn run_all_needs_the_real_model() {
        if cfg!(feature = "nearfield") {
            return; // `every_fault_case_fails_closed` covers the real run
        }
        let e = run_all().unwrap_err();
        assert!(e.contains("`nearfield` feature"), "{e}");
    }

    /// The harness is sensitive: the same faults with a chain that does not fail closed (the
    /// perfect isolator reported as a noise model) leak the raw mic.
    #[test]
    fn non_fail_closed_chain_is_caught_leaking() {
        let sh = shared(Strength::Balanced);
        let src = background(2.0);
        let chain = Chain::with_denoiser(Model::DeepFilterNet3, Box::new(Silencer));
        let mut rig = Rig::new(chain, &sh).unwrap();
        let (mut pos, mut buf, mut leaks) = (0, [0.0f32; BLOCK], Leaks::default());
        for b in 0..300 {
            let stalled = (150..153).contains(&b);
            rig.cap.fault_pause_worker(stalled);
            rig.step(&src, &mut pos, !stalled, &mut buf);
            leaks.feed(&buf);
        }
        let r = leaks.result(0, "silencer");
        assert!(r.leaked_samples > 100 && r.leaked_frames >= 2, "{r:?}");
    }
}
