//! The control thread: owns settings and the platform backend, and decides which streams run.
//!
//! It is plain synchronous code driven by [`Msg`]s (commands from the UI, notifications from the
//! OS) plus a periodic tick, so the state machine is testable with a fake [`Platform`].

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use oto_dsp::{Chain, Model};

use crate::pipeline::{Shared, encode_model};
use crate::platform::{HalEvent, Platform};
use crate::{Command, Event, Settings, Tier};

pub(crate) enum Msg {
    Cmd(Command),
    Hal(HalEvent),
}

/// What the engine should be doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Nobody records from Oto Microphone (or no driver): no streams, no CPU.
    Idle,
    /// Someone records but denoise is off: mic passes through, denoiser bypassed.
    Bypass,
    /// Someone records and denoise is on.
    Denoise,
}

pub fn decide(enabled: bool, clients_active: bool, driver_installed: bool) -> Mode {
    match (driver_installed && clients_active, enabled) {
        (false, _) => Mode::Idle,
        (true, false) => Mode::Bypass,
        (true, true) => Mode::Denoise,
    }
}

/// The model this build runs for a tier.
///
/// Without the `dfn3` cargo feature every tier is RNNoise (it meets the 50 ms budget, see
/// docs/models.md), so the tier setting changes nothing. With `dfn3`, [`Tier::Light`] is RNNoise and
/// [`Tier::Standard`] and [`Tier::Max`] are DeepFilterNet3, which costs ~15× the CPU, adds
/// 20 ms and allocates inside inference on the audio thread (see `oto_dsp::Model`).
pub fn model_for(tier: Tier) -> Model {
    if cfg!(feature = "dfn3") {
        match tier {
            Tier::Light => Model::RNNoise,
            Tier::Standard | Tier::Max => Model::DeepFilterNet3,
        }
    } else {
        Model::RNNoise
    }
}

/// The tiers worth offering: one per distinct model in this build, in ascending order. Always
/// contains [`Tier::default`]. When it has a single entry the tier setting does nothing and a
/// UI should hide its picker.
pub fn available_tiers() -> &'static [Tier] {
    if cfg!(feature = "dfn3") {
        &[Tier::Light, Tier::Standard]
    } else {
        &[Tier::Standard]
    }
}

/// Called on the control thread when the system default input changed.
pub(crate) type Hook = Box<dyn Fn() + Send>;

/// Builds the chain for a model. [`Chain::with_model`] outside tests.
type Loader = Box<dyn FnMut(Model) -> Result<Chain, String>>;

/// What runs when the requested model cannot be loaded. Always compiled in.
const FALLBACK: Model = Model::RNNoise;

const TICK: Duration = Duration::from_millis(250);
const RETRY: Duration = Duration::from_secs(2);
const STALL: Duration = Duration::from_millis(1500);

/// What the running streams were built for. Keyed on the resolved model, not the tier: a tier
/// change that resolves to the same model must not restart audio.
#[derive(Clone, Debug, PartialEq)]
struct RunKey {
    input: String,
    model: Model,
}

pub(crate) struct Control<P: Platform> {
    p: P,
    settings: Settings,
    force_clients: bool,
    shared: Arc<Shared>,
    events: Sender<Event>,
    clients: bool,
    driver_reported_missing: bool,
    running: Option<RunKey>,
    last_error: Option<String>,
    /// An error was reported and `Event::Recovered` has not been sent since.
    fault: bool,
    retry_at: Option<Instant>,
    watch: (Instant, u64, u64),
    load: (Instant, u64),
    on_default_input: Option<Hook>,
    /// The model a tier asks for. [`model_for`] outside tests.
    tier_model: fn(Tier) -> Model,
    loader: Loader,
    /// A model that failed to load. While set, a request for it resolves to [`FALLBACK`]
    /// without retrying. Cleared when the tier changes; nothing retries on a timer.
    failed: Option<Model>,
}

impl<P: Platform> Control<P> {
    pub fn new(
        p: P,
        settings: Settings,
        force_clients: bool,
        shared: Arc<Shared>,
        events: Sender<Event>,
    ) -> Self {
        let now = Instant::now();
        Self {
            p,
            settings,
            force_clients,
            shared,
            events,
            clients: false,
            driver_reported_missing: false,
            running: None,
            last_error: None,
            fault: false,
            retry_at: None,
            watch: (now, 0, 0),
            load: (now, 0),
            on_default_input: None,
            tier_model: model_for,
            loader: Box::new(Chain::with_model),
            failed: None,
        }
    }

    pub fn on_default_input(mut self, hook: Hook) -> Self {
        self.on_default_input = Some(hook);
        self
    }

    pub fn run(mut self, rx: Receiver<Msg>) {
        self.reconcile();
        loop {
            match rx.recv_timeout(TICK) {
                Ok(Msg::Cmd(Command::Shutdown)) | Err(RecvTimeoutError::Disconnected) => break,
                Ok(msg) => self.handle(msg),
                Err(RecvTimeoutError::Timeout) => self.tick(Instant::now()),
            }
            self.reconcile();
        }
        self.stop();
    }

    fn emit(&self, e: Event) {
        let _ = self.events.send(e);
    }

    fn error(&mut self, e: String) {
        if self.last_error.as_ref() != Some(&e) {
            self.emit(Event::Error(e.clone()));
            self.last_error = Some(e);
            self.fault = true;
        }
    }

    /// The reported error no longer applies. Forgets it, so a recurrence is reported again.
    fn recovered(&mut self) {
        if self.fault {
            self.fault = false;
            self.last_error = None;
            self.emit(Event::Recovered);
        }
    }

    pub fn handle(&mut self, msg: Msg) {
        match msg {
            Msg::Cmd(c) => match c {
                Command::SetEnabled(on) => self.settings.enabled = on,
                Command::SetInputDevice(d) => {
                    self.settings.input_device = d;
                    self.retry_at = None;
                }
                Command::SetStrength(s) => self.settings.strength = s,
                Command::SetTier(t) => {
                    if t != self.settings.tier {
                        // The user asked again: give a model that failed to load another try.
                        self.failed = None;
                    }
                    self.settings.tier = t;
                    self.retry_at = None;
                }
                Command::Shutdown => {}
            },
            Msg::Hal(ev @ (HalEvent::DeviceList | HalEvent::DefaultInput)) => {
                if ev == HalEvent::DefaultInput
                    && let Some(hook) = &self.on_default_input
                {
                    hook();
                }
                self.p.refresh();
                self.retry_at = None;
                self.emit(Event::DevicesChanged);
            }
            Msg::Hal(HalEvent::MicClients) => {}
            Msg::Hal(HalEvent::StreamDevice) => {
                // A running device died or changed format: rebuild on the next reconcile.
                if self.running.is_some() {
                    self.stop();
                    self.retry_at = None;
                }
            }
        }
    }

    pub fn tick(&mut self, now: Instant) {
        if self.running.is_none() {
            return;
        }
        let (since, i0, o0) = self.watch;
        let (i, o) = (
            self.shared.in_cycles.load(Relaxed),
            self.shared.out_cycles.load(Relaxed),
        );
        if i != i0 && o != o0 {
            self.watch = (now, i, o);
            // Both callbacks are being called: whatever went wrong before is over.
            self.recovered();
        } else if now.duration_since(since) > STALL {
            // Typical after sleep/wake or a device reset: restart the streams.
            let which = if i == i0 { "microphone" } else { "Oto sink" };
            self.error(format!("{which} stream stalled; restarting audio"));
            self.stop();
            self.retry_at = None;
            return;
        }
        let (t0, b0) = self.load;
        let dt = now.duration_since(t0);
        if dt >= Duration::from_secs(1) {
            let busy = self.shared.busy_ns.load(Relaxed);
            let share = (busy - b0) as f64 / dt.as_nanos() as f64;
            self.load = (now, busy);
            self.emit(Event::Load(share.clamp(0.0, 1.0) as f32));
        }
    }

    fn stop(&mut self) {
        if self.running.take().is_some() {
            self.p.stop();
            self.shared.running.store(false, Relaxed);
            self.shared.model.store(encode_model(None), Relaxed);
        }
    }

    /// The model to run: what the tier asks for, unless that failed to load.
    fn effective_model(&self) -> Model {
        let requested = (self.tier_model)(self.settings.tier);
        if self.failed == Some(requested) {
            FALLBACK
        } else {
            requested
        }
    }

    /// Builds the chain for `key`. If its model cannot load and is not already the fallback,
    /// remembers the failure, reports it once and switches `key` to the fallback.
    fn load_chain(&mut self, key: &mut RunKey) -> Result<Chain, String> {
        match (self.loader)(key.model) {
            Ok(c) => Ok(c),
            Err(reason) if key.model != FALLBACK => {
                let requested = key.model;
                self.failed = Some(requested);
                key.model = FALLBACK;
                self.emit(Event::ModelFallback {
                    requested,
                    effective: FALLBACK,
                    reason,
                });
                (self.loader)(FALLBACK)
            }
            Err(e) => Err(e),
        }
    }

    pub fn reconcile(&mut self) {
        let driver = self.p.driver_installed();
        if !driver && !self.driver_reported_missing {
            self.emit(Event::Error(
                "Oto driver is not installed: Oto Microphone is not available".into(),
            ));
        }
        if !driver && !self.driver_reported_missing {
            self.fault = true;
        }
        self.driver_reported_missing = !driver;
        let clients = driver && (self.force_clients || self.p.clients_active());
        if clients != self.clients {
            self.clients = clients;
            self.emit(Event::ClientsActive(clients));
        }
        let mode = decide(self.settings.enabled, clients, driver);
        self.shared.denoise.store(mode == Mode::Denoise, Relaxed);
        self.shared.max_atten_bits.store(
            self.settings.strength.max_attenuation_db().to_bits(),
            Relaxed,
        );
        if mode == Mode::Idle {
            self.stop();
            self.retry_at = None;
            if driver {
                // Nothing needs to run and nothing is missing: any earlier error is moot.
                self.recovered();
            }
            return;
        }
        let Some(input) = self.p.resolve_input(self.settings.input_device.as_deref()) else {
            self.stop();
            self.error("no microphone found".into());
            return;
        };
        let mut key = RunKey {
            input,
            model: self.effective_model(),
        };
        if self.running.as_ref() == Some(&key) {
            return;
        }
        if self.retry_at.is_some_and(|t| Instant::now() < t) {
            self.stop();
            return;
        }
        // Allocates and may load a model: fine here, this is the control thread. Loaded before
        // the old streams stop, so a model that fails to load falls back without a restart when
        // the fallback is what already runs.
        let chain = match self.load_chain(&mut key) {
            Ok(c) => c,
            Err(e) => {
                self.stop();
                self.error(e);
                self.retry_at = Some(Instant::now() + RETRY);
                return;
            }
        };
        if self.running.as_ref() == Some(&key) {
            return;
        }
        self.stop();
        let started = self.p.start(&key.input, chain, &self.shared);
        // Opening devices can take a while (Bluetooth: seconds). The stall watchdog and the
        // load meter count from when the streams were actually started, not from before.
        let now = Instant::now();
        match started {
            Ok(()) => {
                self.shared
                    .model
                    .store(encode_model(Some(key.model)), Relaxed);
                self.running = Some(key);
                self.shared.running.store(true, Relaxed);
                self.last_error = None;
                self.retry_at = None;
                let s = &self.shared;
                self.watch = (now, s.in_cycles.load(Relaxed), s.out_cycles.load(Relaxed));
                self.load = (now, s.busy_ns.load(Relaxed));
            }
            Err(e) => {
                self.error(e);
                self.retry_at = Some(now + RETRY);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InputDevice, Levels, Strength};
    use std::sync::mpsc;

    #[derive(Default)]
    struct Fake {
        driver: bool,
        clients: bool,
        inputs: Vec<&'static str>,
        default: Option<&'static str>,
        fail: bool,
        started: Vec<String>,
        stops: usize,
        live: bool,
    }

    impl Platform for Fake {
        fn driver_installed(&self) -> bool {
            self.driver
        }
        fn clients_active(&self) -> bool {
            self.clients
        }
        fn input_devices(&self) -> Vec<InputDevice> {
            Vec::new()
        }
        fn resolve_input(&self, preferred: Option<&str>) -> Option<String> {
            preferred
                .filter(|p| self.inputs.contains(p))
                .or(self.default)
                .map(str::to_owned)
        }
        fn start(&mut self, input: &str, _chain: Chain, _s: &Arc<Shared>) -> Result<(), String> {
            if self.fail {
                return Err("boom".into());
            }
            assert!(!self.live, "start while running");
            self.live = true;
            self.started.push(input.to_owned());
            Ok(())
        }
        fn stop(&mut self) {
            assert!(self.live, "stop while stopped");
            self.live = false;
            self.stops += 1;
        }
        fn refresh(&mut self) {}
    }

    fn ctl(f: Fake, enabled: bool) -> (Control<Fake>, mpsc::Receiver<Event>) {
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Shared::new(Arc::new(Levels::default()), None));
        let s = Settings {
            enabled,
            ..Default::default()
        };
        (Control::new(f, s, false, shared, tx), rx)
    }

    fn fake() -> Fake {
        Fake {
            driver: true,
            inputs: vec!["builtin", "usb"],
            default: Some("builtin"),
            ..Default::default()
        }
    }

    #[test]
    fn decide_table() {
        assert_eq!(decide(true, false, true), Mode::Idle);
        assert_eq!(decide(false, false, true), Mode::Idle);
        assert_eq!(decide(false, true, true), Mode::Bypass);
        assert_eq!(decide(true, true, true), Mode::Denoise);
        assert_eq!(decide(true, true, false), Mode::Idle);
    }

    #[test]
    fn streams_follow_enabled_times_clients() {
        for enabled in [false, true] {
            let (mut c, rx) = ctl(fake(), enabled);
            c.reconcile();
            assert!(!c.p.live, "idle with no clients (enabled={enabled})");
            assert!(rx.try_iter().next().is_none());

            c.p.clients = true;
            c.reconcile();
            assert!(
                c.p.live,
                "streams run when clients are active (enabled={enabled})"
            );
            assert_eq!(c.shared.denoise.load(Relaxed), enabled);
            assert_eq!(
                rx.try_iter().collect::<Vec<_>>(),
                [Event::ClientsActive(true)]
            );

            // Toggling enabled while running only flips denoise, it does not restart streams.
            c.handle(Msg::Cmd(Command::SetEnabled(!enabled)));
            c.reconcile();
            assert_eq!(c.shared.denoise.load(Relaxed), !enabled);
            assert_eq!(c.p.started.len(), 1);

            c.p.clients = false;
            c.reconcile();
            assert!(!c.p.live);
            assert_eq!(
                rx.try_iter().collect::<Vec<_>>(),
                [Event::ClientsActive(false)]
            );
        }
    }

    #[test]
    fn no_driver_means_idle_and_one_error() {
        let (mut c, rx) = ctl(
            Fake {
                driver: false,
                clients: true,
                ..fake()
            },
            true,
        );
        c.reconcile();
        c.reconcile();
        assert!(!c.p.live);
        let ev: Vec<_> = rx.try_iter().collect();
        assert_eq!(ev.len(), 1);
        assert!(matches!(ev[0], Event::Error(_)));
    }

    #[test]
    fn device_choice_and_hotplug_fallback() {
        let (mut c, _rx) = ctl(
            Fake {
                clients: true,
                ..fake()
            },
            true,
        );
        c.handle(Msg::Cmd(Command::SetInputDevice(Some("usb".into()))));
        c.reconcile();
        assert_eq!(c.p.started, ["usb"]);
        // Unplug: the device list changes and the chosen mic is gone, fall back to default.
        c.p.inputs.retain(|d| *d != "usb");
        c.handle(Msg::Hal(HalEvent::StreamDevice));
        c.handle(Msg::Hal(HalEvent::DeviceList));
        c.reconcile();
        assert_eq!(c.p.started, ["usb", "builtin"]);
        // Plug back in: preference wins again.
        c.p.inputs.push("usb");
        c.handle(Msg::Hal(HalEvent::DeviceList));
        c.reconcile();
        assert_eq!(c.p.started, ["usb", "builtin", "usb"]);
    }

    #[test]
    fn strength_reaches_shared_state_and_tier_restarts_only_for_another_model() {
        let (mut c, _rx) = ctl(
            Fake {
                clients: true,
                ..fake()
            },
            true,
        );
        c.handle(Msg::Cmd(Command::SetStrength(Strength::Light)));
        c.reconcile();
        assert_eq!(f32::from_bits(c.shared.max_atten_bits.load(Relaxed)), 12.0);
        // Standard → Max is the same model in every build: audio keeps running.
        assert_eq!(model_for(Tier::Standard), model_for(Tier::Max));
        c.handle(Msg::Cmd(Command::SetTier(Tier::Max)));
        c.reconcile();
        assert_eq!(c.p.started.len(), 1);
        assert_eq!(c.p.stops, 0);
        // Standard/Max → Light restarts only if this build has a different model for it.
        c.handle(Msg::Cmd(Command::SetTier(Tier::Light)));
        c.reconcile();
        let distinct = model_for(Tier::Light) != model_for(Tier::Max);
        assert_eq!(distinct, cfg!(feature = "dfn3"));
        assert_eq!(c.p.started.len(), if distinct { 2 } else { 1 });
    }

    #[test]
    fn available_tiers_resolve_to_distinct_available_models() {
        let tiers = available_tiers();
        assert!(tiers.contains(&Tier::default()));
        let models: Vec<_> = tiers.iter().map(|&t| model_for(t)).collect();
        for (i, m) in models.iter().enumerate() {
            assert!(m.available(), "{m:?} is not in this build");
            assert!(!models[..i].contains(m), "{m:?} offered twice");
        }
        // Nothing reachable is left out.
        for t in [Tier::Light, Tier::Standard, Tier::Max] {
            assert!(models.contains(&model_for(t)));
        }
        assert_eq!(tiers.len(), if cfg!(feature = "dfn3") { 2 } else { 1 });
    }

    fn errors_and_recoveries(rx: &mpsc::Receiver<Event>) -> Vec<Event> {
        rx.try_iter()
            .filter(|e| matches!(e, Event::Error(_) | Event::Recovered))
            .collect()
    }

    fn advance(c: &Control<Fake>) {
        c.shared.in_cycles.fetch_add(10, Relaxed);
        c.shared.out_cycles.fetch_add(10, Relaxed);
    }

    #[test]
    fn recovery_is_reported_once_when_audio_flows_again() {
        let (mut c, rx) = ctl(
            Fake {
                clients: true,
                fail: true,
                ..fake()
            },
            true,
        );
        c.reconcile();
        assert_eq!(errors_and_recoveries(&rx), [Event::Error("boom".into())]);
        // The start succeeds, but nothing has been heard from the callbacks yet.
        c.p.fail = false;
        c.retry_at = None;
        c.reconcile();
        assert!(c.p.live);
        c.tick(Instant::now());
        assert_eq!(errors_and_recoveries(&rx), []);
        // Both callbacks run: recovered, exactly once.
        advance(&c);
        c.tick(Instant::now());
        assert_eq!(errors_and_recoveries(&rx), [Event::Recovered]);
        advance(&c);
        c.tick(Instant::now());
        c.reconcile();
        assert_eq!(errors_and_recoveries(&rx), []);
    }

    #[test]
    fn no_recovery_without_a_prior_error() {
        let (mut c, rx) = ctl(
            Fake {
                clients: true,
                ..fake()
            },
            true,
        );
        c.reconcile();
        advance(&c);
        c.tick(Instant::now());
        c.p.clients = false;
        c.reconcile();
        assert_eq!(errors_and_recoveries(&rx), []);
    }

    #[test]
    fn stall_then_restart_reports_recovery_and_a_repeat_is_reported_again() {
        let (mut c, rx) = ctl(
            Fake {
                clients: true,
                ..fake()
            },
            true,
        );
        c.reconcile();
        for round in 0..2 {
            c.tick(Instant::now() + Duration::from_secs(2));
            let ev = errors_and_recoveries(&rx);
            assert!(
                matches!(&ev[..], [Event::Error(e)] if e.contains("stalled")),
                "round {round}: {ev:?}"
            );
            c.reconcile();
            assert!(c.p.live);
            assert_eq!(errors_and_recoveries(&rx), []);
            advance(&c);
            c.tick(Instant::now());
            assert_eq!(errors_and_recoveries(&rx), [Event::Recovered]);
        }
    }

    #[test]
    fn error_is_cleared_when_nothing_needs_to_run() {
        // No mic while an app records: an error. The app stops recording: nothing is wrong
        // any more. It records again: the error is reported again.
        let (mut c, rx) = ctl(
            Fake {
                clients: true,
                inputs: vec![],
                default: None,
                ..fake()
            },
            true,
        );
        c.reconcile();
        c.reconcile();
        assert_eq!(
            errors_and_recoveries(&rx),
            [Event::Error("no microphone found".into())]
        );
        c.p.clients = false;
        c.reconcile();
        c.reconcile();
        assert_eq!(errors_and_recoveries(&rx), [Event::Recovered]);
        c.p.clients = true;
        c.reconcile();
        assert_eq!(
            errors_and_recoveries(&rx),
            [Event::Error("no microphone found".into())]
        );
    }

    #[test]
    fn driver_coming_back_clears_its_error() {
        let (mut c, rx) = ctl(
            Fake {
                driver: false,
                ..fake()
            },
            true,
        );
        c.reconcile();
        assert!(matches!(&errors_and_recoveries(&rx)[..], [Event::Error(_)]));
        c.p.driver = true;
        c.reconcile();
        c.reconcile();
        assert_eq!(errors_and_recoveries(&rx), [Event::Recovered]);
    }

    #[test]
    fn stall_clock_starts_after_the_streams_started() {
        /// A device that takes longer than the stall limit to open.
        struct Slow(Fake);
        impl Platform for Slow {
            fn driver_installed(&self) -> bool {
                self.0.driver_installed()
            }
            fn clients_active(&self) -> bool {
                self.0.clients_active()
            }
            fn input_devices(&self) -> Vec<InputDevice> {
                Vec::new()
            }
            fn resolve_input(&self, p: Option<&str>) -> Option<String> {
                self.0.resolve_input(p)
            }
            fn start(&mut self, i: &str, c: Chain, s: &Arc<Shared>) -> Result<(), String> {
                std::thread::sleep(STALL + Duration::from_millis(100));
                self.0.start(i, c, s)
            }
            fn stop(&mut self) {
                self.0.stop()
            }
            fn refresh(&mut self) {}
        }
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Shared::new(Arc::new(Levels::default()), None));
        let s = Settings {
            enabled: true,
            ..Default::default()
        };
        let f = Fake {
            clients: true,
            ..fake()
        };
        let mut c = Control::new(Slow(f), s, false, shared, tx);
        c.reconcile();
        // The first tick right after a slow start must not see a stall.
        c.tick(Instant::now());
        assert!(c.p.0.live);
        assert_eq!(c.p.0.stops, 0);
        assert!(!rx.try_iter().any(|e| matches!(e, Event::Error(_))));
    }

    #[test]
    fn default_input_change_runs_the_hook() {
        use std::sync::atomic::AtomicUsize;
        let (c, _rx) = ctl(fake(), true);
        let n = Arc::new(AtomicUsize::new(0));
        let n2 = n.clone();
        let mut c = c.on_default_input(Box::new(move || {
            n2.fetch_add(1, Relaxed);
        }));
        c.handle(Msg::Hal(HalEvent::DeviceList));
        assert_eq!(n.load(Relaxed), 0);
        c.handle(Msg::Hal(HalEvent::DefaultInput));
        assert_eq!(n.load(Relaxed), 1);
    }

    #[test]
    fn start_failure_reports_once_and_backs_off() {
        let (mut c, rx) = ctl(
            Fake {
                clients: true,
                fail: true,
                ..fake()
            },
            true,
        );
        c.reconcile();
        c.reconcile();
        let errs = rx
            .try_iter()
            .filter(|e| matches!(e, Event::Error(_)))
            .count();
        assert_eq!(errs, 1);
        assert!(c.retry_at.is_some());
        c.p.fail = false;
        c.retry_at = Some(Instant::now());
        c.reconcile();
        assert!(c.p.live);
    }

    /// A controller whose Standard/Max tiers ask for DeepFilterNet3 (as with the `dfn3`
    /// feature) and whose DeepFilterNet3 never loads. Returns the models it tried to load.
    fn dfn3_fails(c: &mut Control<Fake>) -> std::rc::Rc<std::cell::RefCell<Vec<Model>>> {
        c.tier_model = |t| match t {
            Tier::Light => Model::RNNoise,
            Tier::Standard | Tier::Max => Model::DeepFilterNet3,
        };
        let loads = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let l = loads.clone();
        c.loader = Box::new(move |m| {
            l.borrow_mut().push(m);
            match m {
                Model::DeepFilterNet3 => Err("cannot load DeepFilterNet3".into()),
                Model::RNNoise => Chain::with_model(m),
            }
        });
        loads
    }

    /// A loader for which every model loads (the Fake ignores the chain, so an RNNoise chain
    /// stands in for DeepFilterNet3 in builds without it).
    fn loads_anything(_: Model) -> Result<Chain, String> {
        Chain::with_model(Model::RNNoise)
    }

    /// What `Stats::model` reports.
    fn running_model(c: &Control<Fake>) -> Option<Model> {
        crate::pipeline::decode_model(c.shared.model.load(Relaxed))
    }

    fn fallbacks(ev: &[Event]) -> usize {
        ev.iter()
            .filter(|e| matches!(e, Event::ModelFallback { .. }))
            .count()
    }

    #[test]
    fn model_load_failure_falls_back_once_without_restarting() {
        use Model::{DeepFilterNet3 as Dfn3, RNNoise};
        let (mut c, rx) = ctl(
            Fake {
                clients: true,
                ..fake()
            },
            true,
        );
        let loads = dfn3_fails(&mut c);
        c.reconcile();
        assert!(c.p.live);
        assert_eq!(*loads.borrow(), [Dfn3, RNNoise]);
        assert_eq!(running_model(&c), Some(RNNoise));
        let ev: Vec<_> = rx.try_iter().collect();
        assert!(
            ev.contains(&Event::ModelFallback {
                requested: Dfn3,
                effective: RNNoise,
                reason: "cannot load DeepFilterNet3".into(),
            }),
            "{ev:?}"
        );
        // A notice, not an error.
        assert!(!ev.iter().any(|e| matches!(e, Event::Error(_))));
        assert!(c.retry_at.is_none());

        // The tier still asks for DeepFilterNet3, but nothing retries or restarts.
        for _ in 0..10 {
            advance(&c);
            c.tick(Instant::now());
            c.reconcile();
        }
        assert_eq!(c.p.started.len(), 1);
        assert_eq!(c.p.stops, 0);
        assert_eq!(loads.borrow().len(), 2);

        // The app stops recording and starts again: RNNoise, without trying DeepFilterNet3.
        c.p.clients = false;
        c.reconcile();
        assert!(!c.p.live);
        assert_eq!(running_model(&c), None);
        c.p.clients = true;
        c.reconcile();
        assert!(c.p.live);
        assert_eq!(*loads.borrow(), [Dfn3, RNNoise, RNNoise]);
        assert_eq!(running_model(&c), Some(RNNoise));
        assert_eq!(fallbacks(&rx.try_iter().collect::<Vec<_>>()), 0);

        // Re-sending the same tier is not a change: still remembered.
        c.handle(Msg::Cmd(Command::SetTier(Tier::Standard)));
        c.reconcile();
        assert_eq!(loads.borrow().len(), 3);
        assert_eq!(c.p.started.len(), 2);
    }

    #[test]
    fn changing_tier_clears_the_remembered_failure() {
        use Model::{DeepFilterNet3 as Dfn3, RNNoise};
        let (mut c, rx) = ctl(
            Fake {
                clients: true,
                ..fake()
            },
            true,
        );
        let loads = dfn3_fails(&mut c);
        c.reconcile();
        assert_eq!(fallbacks(&rx.try_iter().collect::<Vec<_>>()), 1);
        // Standard → Max asks for the same model: tried again, fails again, and since the
        // fallback already runs, audio is not restarted.
        c.handle(Msg::Cmd(Command::SetTier(Tier::Max)));
        c.reconcile();
        assert_eq!(*loads.borrow(), [Dfn3, RNNoise, Dfn3, RNNoise]);
        assert_eq!(c.p.started.len(), 1);
        assert_eq!(c.p.stops, 0);
        assert_eq!(fallbacks(&rx.try_iter().collect::<Vec<_>>()), 1);
        // Light is RNNoise anyway; back to Standard tries DeepFilterNet3 again, which now loads.
        c.handle(Msg::Cmd(Command::SetTier(Tier::Light)));
        c.reconcile();
        assert_eq!(loads.borrow().len(), 4, "Light is already running");
        c.loader = Box::new(loads_anything);
        c.handle(Msg::Cmd(Command::SetTier(Tier::Standard)));
        c.reconcile();
        assert_eq!(running_model(&c), Some(Dfn3));
        assert_eq!(c.p.started.len(), 2);
        assert_eq!(fallbacks(&rx.try_iter().collect::<Vec<_>>()), 0);
    }

    #[test]
    fn device_start_failure_is_an_error_not_a_fallback() {
        let (mut c, rx) = ctl(
            Fake {
                clients: true,
                fail: true,
                ..fake()
            },
            true,
        );
        let loads = dfn3_fails(&mut c);
        // DeepFilterNet3 loads here; it is the device that fails.
        c.loader = Box::new(loads_anything);
        c.reconcile();
        let ev: Vec<_> = rx.try_iter().collect();
        assert_eq!(fallbacks(&ev), 0);
        assert!(ev.contains(&Event::Error("boom".into())));
        assert!(c.retry_at.is_some());
        assert!(c.failed.is_none());
        assert!(loads.borrow().is_empty());
    }

    #[test]
    fn stalled_stream_is_restarted() {
        let (mut c, rx) = ctl(
            Fake {
                clients: true,
                ..fake()
            },
            true,
        );
        c.reconcile();
        let t0 = Instant::now();
        c.tick(t0 + Duration::from_secs(2));
        assert!(!c.p.live);
        assert!(rx.try_iter().any(|e| matches!(e, Event::Error(_))));
        c.reconcile();
        assert!(c.p.live);
        assert_eq!(c.p.started.len(), 2);
    }
}
