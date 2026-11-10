//! The control thread: owns settings and the platform backend, and decides which streams run.
//!
//! It is plain synchronous code driven by [`Msg`]s (commands from the UI, notifications from the
//! OS) plus a periodic tick, so the state machine is testable with a fake [`Platform`].

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use oto_dsp::Chain;

use crate::pipeline::Shared;
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

/// Builds the chain for a tier. Allocates and may load a model: control thread only.
fn make_chain(tier: Tier) -> Result<Chain, String> {
    // TODO(merge): select the model once oto-dsp has it:
    //   let model = match tier { Tier::Light => oto_dsp::Model::RNNoise,
    //                            Tier::Standard | Tier::Max => oto_dsp::Model::DeepFilterNet3 };
    //   Chain::with_model(model)
    let _ = tier;
    Ok(Chain::new())
}

const TICK: Duration = Duration::from_millis(250);
const RETRY: Duration = Duration::from_secs(2);
const STALL: Duration = Duration::from_millis(1500);

#[derive(Clone, Debug, PartialEq)]
struct RunKey {
    input: String,
    tier: Tier,
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
    retry_at: Option<Instant>,
    watch: (Instant, u64, u64),
    load: (Instant, u64),
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
            retry_at: None,
            watch: (now, 0, 0),
            load: (now, 0),
        }
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
                    self.settings.tier = t;
                    self.retry_at = None;
                }
                Command::Shutdown => {}
            },
            Msg::Hal(HalEvent::DeviceList | HalEvent::DefaultInput) => {
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
        }
    }

    pub fn reconcile(&mut self) {
        let driver = self.p.driver_installed();
        if !driver && !self.driver_reported_missing {
            self.emit(Event::Error(
                "Oto driver is not installed: Oto Microphone is not available".into(),
            ));
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
            return;
        }
        let Some(input) = self.p.resolve_input(self.settings.input_device.as_deref()) else {
            self.stop();
            self.error("no microphone found".into());
            return;
        };
        let key = RunKey {
            input,
            tier: self.settings.tier,
        };
        if self.running.as_ref() == Some(&key) {
            return;
        }
        self.stop();
        let now = Instant::now();
        if self.retry_at.is_some_and(|t| now < t) {
            return;
        }
        match make_chain(key.tier).and_then(|c| self.p.start(&key.input, c, &self.shared)) {
            Ok(()) => {
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
    fn strength_reaches_shared_state_and_tier_restarts() {
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
        c.handle(Msg::Cmd(Command::SetTier(Tier::Light)));
        c.reconcile();
        assert_eq!(c.p.started.len(), 2);
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
