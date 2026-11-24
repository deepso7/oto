//! The app's state and everything that changes it. No Rune views and no I/O beyond the hooks
//! [`Hooks`] names, so it unit-tests without a window or an engine.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use oto_engine::dsp::Model;
use oto_engine::test_capture::TestRecording;
use oto_engine::{Command, Event, InputDevice, Settings, Strength, Tier};

use crate::driver::{Outcome, REBOOT_MESSAGE, UninstallOutcome};
use crate::meter::Meter;
use crate::permission::Mic;
use crate::prefs::Prefs;
use crate::tray::Tray;

/// Shows the "Hear the Difference" test: record a few seconds, then compare raw and cleaned.
pub const HEAR_TEST: bool = true;
/// How long "Hear the Difference" records, seconds.
pub const HEAR_SECONDS: f32 = 5.0;
/// Why "Hear the Difference" can't start during a call: it opens the mic and the speakers.
pub const HEAR_IN_CALL: &str = "End your call to run the mic test.";

/// The main page's two modes. Noise Cancellation runs the model picked in Settings
/// ([`Prefs::nc_tier`]); Voice Isolation is [`Tier::Isolate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    NoiseCancellation,
    VoiceIsolation,
}

pub const MODES: [(Mode, &str); 2] = [
    (Mode::NoiseCancellation, "Noise Cancellation"),
    (Mode::VoiceIsolation, "Voice Isolation"),
];

pub const STRENGTHS: [(Strength, &str); 3] = [
    (Strength::Light, "Light"),
    (Strength::Balanced, "Balanced"),
    (Strength::Strong, "Strong"),
];
/// The Noise Cancellation model picker's rows (Settings), in the order offered. Only tiers in
/// [`State::tiers`] show. [`Tier::Isolate`] is a mode, not one of these.
pub const TIERS: [(Tier, &str); 4] = [
    (Tier::Standard, "DeepFilterNet3"),
    (Tier::Max, "DPDFNet-2 (stronger)"),
    (Tier::Fast, "FastEnhancer-S (low delay)"),
    (Tier::Light, "RNNoise (low CPU)"),
];
pub const SYSTEM_DEFAULT: &str = "System default";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    Main,
    Settings,
}

/// The driver install or removal, from the button press to the result.
#[derive(Clone, Debug, PartialEq)]
pub enum Install {
    Idle,
    /// Installing.
    Running,
    /// Uninstalling.
    Removing,
    Failed(String),
    /// It worked, but there's something left for the user to do (restart the Mac).
    Notice(String),
}

impl Install {
    /// The installer or uninstaller is running (and its password prompt may have focus).
    pub fn busy(&self) -> bool {
        matches!(self, Install::Running | Install::Removing)
    }
}

/// The audio engine's life, as the UI sees it.
#[derive(Clone, Debug, PartialEq)]
pub enum EngineState {
    Starting,
    Running,
    /// It didn't start, or it stopped. The controls do nothing until a retry works.
    Failed(String),
}

/// What the system says right now, read off the UI thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Probe {
    /// Core Audio lists Oto Microphone.
    pub driver: bool,
    /// The driver bundle is in the HAL plug-in folder, loaded or not.
    pub on_disk: bool,
    pub mic: Mic,
}

/// Which popup select is open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Open {
    Device,
    Model,
}

/// "Hear the Difference", from the button press to the A/B playback.
#[derive(Clone, Debug, PartialEq)]
pub enum HearTest {
    Idle,
    /// Recording [`HEAR_SECONDS`] from the mic, then (`processing`) cleaning it offline.
    Recording {
        processing: bool,
    },
    /// Looping playback; `cleaned` is the side playing.
    Ready {
        cleaned: bool,
        is_bluetooth_hfp: bool,
    },
    Failed(String),
}

/// One "Hear the Difference" take, for [`Hooks::start_hear_test`]. Setting `cancel` stops the
/// recording and tells the hook its result is no longer wanted.
#[derive(Clone, Debug)]
pub struct HearJob {
    pub device: Option<String>,
    pub model: Model,
    pub strength: Strength,
    pub cancel: Arc<AtomicBool>,
}

/// What the UI asks of the engine's host thread.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    Cmd(Command),
    /// Make Oto Microphone the system default input (`true`) or put the previous one back.
    SystemMic(bool),
}

/// Where requests go.
pub enum Link {
    /// Snapshots: nowhere.
    None,
    /// The engine host thread.
    Engine(Sender<Request>),
    /// Tests: kept for inspection.
    #[cfg_attr(not(test), allow(dead_code))]
    Record(Vec<Request>),
}

impl Link {
    /// False when the host thread is gone.
    fn send(&mut self, req: Request) -> bool {
        match self {
            Link::None => true,
            Link::Engine(tx) => tx.send(req).is_ok(),
            Link::Record(v) => {
                v.push(req);
                true
            }
        }
    }
}

/// Side effects the live app performs and tests and snapshots skip.
#[derive(Clone, Copy)]
pub struct Hooks {
    pub save: fn(&Prefs) -> Result<(), String>,
    pub set_login: fn(bool) -> Result<(), String>,
    /// Starts the install off the UI thread; the result comes back through [`State::installed`].
    pub install: fn(),
    /// Starts the uninstall off the UI thread; the result comes back through
    /// [`State::uninstalled`].
    pub uninstall: fn(),
    /// Starts an engine on a new host thread and returns the link to it. The result comes back
    /// through [`State::engine_started`] or [`State::engine_failed`].
    pub start_engine: fn(Settings, Meter) -> Link,
    /// Shows the system's microphone prompt; the answer comes back through
    /// [`State::mic_answered`].
    pub request_mic: fn(),
    /// Opens System Settings at Privacy, Microphone.
    pub open_mic_settings: fn() -> Result<(), String>,
    /// Records and cleans a "Hear the Difference" take off the UI thread; the result comes back
    /// through [`State::hear_recorded`] (and [`State::hear_processing`] once recording ends).
    pub start_hear_test: fn(HearJob),
    /// Starts looping a take, raw first.
    pub play_hear_test: fn(&TestRecording) -> Result<(), String>,
    /// Stops the playback [`Hooks::play_hear_test`] started.
    pub stop_hear_test: fn(),
    /// Switches the playback between raw (`false`) and cleaned.
    pub set_hear_cleaned: fn(bool),
}

impl Hooks {
    pub const NONE: Hooks = Hooks {
        save: |_| Ok(()),
        set_login: |_| Ok(()),
        install: || {},
        uninstall: || {},
        start_engine: |_, _| Link::None,
        request_mic: || {},
        open_mic_settings: || Ok(()),
        start_hear_test: |_| {},
        play_hear_test: |_| Ok(()),
        stop_hear_test: || {},
        set_hear_cleaned: |_| {},
    };
}

pub struct State {
    pub prefs: Prefs,
    pub driver_installed: bool,
    /// The driver bundle is on disk (it may still need a restart to load, or to unload).
    pub driver_on_disk: bool,
    /// False until the first [`Probe`] arrives, so the app doesn't offer to install a driver
    /// it hasn't looked for yet.
    pub driver_known: bool,
    /// "Not now" on onboarding: show the main screen without a driver.
    pub skipped_onboarding: bool,
    pub install: Install,
    pub engine: EngineState,
    /// Microphone permission, as macOS reports it.
    pub mic: Mic,
    /// The system's microphone prompt is up.
    pub mic_requesting: bool,
    /// The engine was asked to make Oto Microphone the system default input.
    system_mic_applied: bool,
    /// Start at login can be turned on from this copy of Oto.
    pub login_available: bool,
    /// The model tiers this build offers. "Low CPU mode" shows only when there are two or more:
    /// off is [`Tier::Standard`], on is [`Tier::Light`].
    pub tiers: Vec<Tier>,
    /// The model the running streams use (after any fallback) and the estimated mic-to-app
    /// delay in ms, while streams run. From `Engine::stats`.
    pub running: Option<(Model, f32)>,
    /// The requested model couldn't load and another runs instead. Not an error: audio works.
    /// Cleared when the user changes the mode, which makes the engine try again.
    pub model_notice: Option<String>,
    pub devices: Vec<InputDevice>,
    /// An app is recording from Oto Microphone.
    pub clients_active: bool,
    /// Share of one core the audio callback uses, while active.
    pub load: Option<f32>,
    pub error: Option<String>,
    /// Voice Isolation can't run, so Oto Microphone is muted ([`Event::Degraded`]). Cleared by
    /// `Event::DegradedCleared` and by leaving Voice Isolation.
    pub degraded: Option<String>,
    /// "Hear the Difference".
    pub hear: HearTest,
    /// The take being recorded: its cancel flag, which also identifies it.
    hear_job: Option<Arc<AtomicBool>>,
    /// `error` came from the engine, so the engine's `Recovered` clears it.
    error_from_engine: bool,
    pub page: Page,
    pub open: Option<Open>,
    pub meter: Meter,
    pub link: Link,
    pub hooks: Hooks,
    pub tray: Option<Tray>,
    /// The window is a popover under the status item: it hides when it loses focus.
    pub popover: bool,
    /// The window has keyboard focus.
    pub focused: bool,
    /// The window's current content size, for placing it under the status item.
    pub size: (f32, f32),
    /// When the popover last hid for losing focus.
    pub hidden_at: Option<std::time::Instant>,
}

/// What the user reads when the engine can't start.
fn engine_failure(e: &str) -> String {
    if e == oto_engine::ALREADY_RUNNING {
        "Another copy of Oto is already running. Quit it, then try again.".into()
    } else {
        format!("Oto couldn\u{2019}t start its audio engine: {e}")
    }
}

impl State {
    /// A state that already knows the driver's status and has a running engine: what tests and
    /// snapshots want. The live app resets `driver_known` and `engine` (see `main.rs`).
    pub fn new(mut prefs: Prefs, driver_installed: bool) -> State {
        let tiers = oto_engine::available_tiers().to_vec();
        if prefs.nc_tier == Tier::Isolate || !tiers.contains(&prefs.nc_tier) {
            prefs.nc_tier = Tier::default();
        }
        // Never silently rewrite a saved Voice Isolation choice to Noise Cancellation: if this
        // build lacks Nearfield, the engine must fail closed and surface Event::Degraded.
        if prefs.tier != Tier::Isolate && !tiers.contains(&prefs.tier) {
            prefs.tier = prefs.nc_tier;
        }
        State {
            prefs,
            driver_installed,
            driver_on_disk: driver_installed,
            driver_known: true,
            skipped_onboarding: false,
            install: Install::Idle,
            engine: EngineState::Running,
            mic: Mic::Authorized,
            mic_requesting: false,
            system_mic_applied: false,
            login_available: true,
            tiers,
            running: None,
            model_notice: None,
            devices: Vec::new(),
            clients_active: false,
            load: None,
            error: None,
            degraded: None,
            hear: HearTest::Idle,
            hear_job: None,
            error_from_engine: false,
            page: Page::Main,
            open: None,
            meter: Meter::default(),
            link: Link::None,
            hooks: Hooks::NONE,
            tray: None,
            popover: false,
            focused: false,
            size: (crate::view::WIDTH, 400.0),
            hidden_at: None,
        }
    }

    /// Setup shows until the user finishes it, and again whenever the driver is missing (unless
    /// they said "not now").
    pub fn onboarding(&self) -> bool {
        self.driver_known
            && !self.skipped_onboarding
            && (!self.prefs.onboarded || !self.driver_installed)
    }

    /// The popover must stay up although it lost focus: a system prompt Oto raised has it.
    pub fn holding(&self) -> bool {
        self.install.busy() || self.mic_requesting
    }

    fn send(&mut self, req: Request) {
        if !self.link.send(req) && self.engine == EngineState::Running {
            self.engine = EngineState::Failed("Oto's audio engine stopped.".into());
            self.system_mic_applied = false;
        }
    }

    fn command(&mut self, cmd: Command) {
        self.send(Request::Cmd(cmd));
    }

    fn save(&mut self) {
        if let Err(e) = (self.hooks.save)(&self.prefs) {
            self.fail(e);
        }
    }

    /// Shows an error of the app's own (not the engine's).
    fn fail(&mut self, msg: String) {
        self.error = Some(msg);
        self.error_from_engine = false;
    }

    pub fn dismiss_error(&mut self) {
        self.error = None;
    }

    fn meter_gate(&self) {
        self.meter
            .active
            .store(self.prefs.enabled && self.clients_active, Ordering::Relaxed);
    }

    pub fn set_enabled(&mut self, on: bool) {
        if self.prefs.enabled == on {
            return;
        }
        self.prefs.enabled = on;
        self.command(Command::SetEnabled(on));
        self.save();
        self.meter_gate();
        if let Some(t) = &self.tray {
            t.set_enabled(on);
        }
    }

    pub fn toggle(&mut self) {
        self.set_enabled(!self.prefs.enabled);
    }

    pub fn set_strength(&mut self, s: Strength) {
        if self.prefs.strength != s {
            self.prefs.strength = s;
            self.stop_hear_test();
            self.command(Command::SetStrength(s));
            self.save();
        }
    }

    /// Runs tier `t`. A Noise Cancellation tier also becomes the one that mode comes back to.
    pub fn set_tier(&mut self, t: Tier) {
        self.open = None;
        if !self.tiers.contains(&t) {
            return;
        }
        let mut changed = false;
        if t != Tier::Isolate {
            // Leaving Voice Isolation leaves its muted state behind.
            self.degraded = None;
            changed = self.prefs.nc_tier != t;
            self.prefs.nc_tier = t;
        }
        if self.prefs.tier != t {
            self.prefs.tier = t;
            // The engine forgets a failed load on a tier change and tries again; so does this.
            self.model_notice = None;
            self.stop_hear_test();
            self.command(Command::SetTier(t));
            changed = true;
        }
        if changed {
            self.save();
        }
    }

    /// The main page's mode.
    pub fn mode(&self) -> Mode {
        if self.prefs.tier == Tier::Isolate {
            Mode::VoiceIsolation
        } else {
            Mode::NoiseCancellation
        }
    }

    /// Voice Isolation runs.
    pub fn isolating(&self) -> bool {
        self.mode() == Mode::VoiceIsolation
    }

    /// This build has Voice Isolation, so the main page offers the two modes.
    pub fn offers_isolation(&self) -> bool {
        self.tiers.contains(&Tier::Isolate)
    }

    /// Noise Cancellation runs the model picked in Settings; Voice Isolation runs Nearfield.
    pub fn set_mode(&mut self, m: Mode) {
        match m {
            Mode::NoiseCancellation => self.set_tier(self.prefs.nc_tier),
            Mode::VoiceIsolation => self.set_tier(Tier::Isolate),
        }
    }

    /// The degraded banner's way out: Noise Cancellation, which never goes silent.
    pub fn switch_to_noise_cancellation(&mut self) {
        self.set_mode(Mode::NoiseCancellation);
        self.degraded = None;
    }

    /// Picks the Noise Cancellation model (Settings). In Voice Isolation it is only remembered,
    /// for when Noise Cancellation is picked again.
    pub fn set_nc_tier(&mut self, t: Tier) {
        self.open = None;
        if t == Tier::Isolate || !self.tiers.contains(&t) {
            return;
        }
        if self.isolating() {
            if self.prefs.nc_tier != t {
                self.prefs.nc_tier = t;
                self.save();
            }
        } else {
            self.set_tier(t);
        }
    }

    /// The Noise Cancellation model picker's labels: one per such tier this build offers.
    pub fn tier_options(&self) -> Vec<&'static str> {
        TIERS
            .iter()
            .filter(|(t, _)| self.tiers.contains(t))
            .map(|(_, l)| *l)
            .collect()
    }

    /// The Noise Cancellation model's label.
    pub fn tier_label(&self) -> &'static str {
        TIERS
            .iter()
            .find(|(t, _)| *t == self.prefs.nc_tier)
            .map_or("", |(_, l)| *l)
    }

    /// Picks the Noise Cancellation model by its label in [`TIERS`].
    pub fn choose_tier(&mut self, label: &str) {
        self.open = None;
        if let Some((t, _)) = TIERS.iter().find(|(_, l)| *l == label) {
            self.set_nc_tier(*t);
        }
    }

    /// Whether the build offers a choice of Noise Cancellation model, so the picker does
    /// something.
    pub fn offers_tiers(&self) -> bool {
        self.tiers.iter().filter(|t| **t != Tier::Isolate).count() > 1
    }

    pub fn set_start_at_login(&mut self, on: bool) {
        match (self.hooks.set_login)(on) {
            Ok(()) => {
                self.prefs.start_at_login = on;
                self.save();
            }
            Err(e) => self.fail(e),
        }
    }

    /// "Use Oto as the system microphone". The engine puts the previous default back when this
    /// goes off and when Oto quits.
    pub fn set_system_mic(&mut self, on: bool) {
        if self.prefs.system_microphone != on {
            self.prefs.system_microphone = on;
            self.save();
            self.sync_system_mic();
        }
    }

    /// Asks the engine for whatever the setting and the driver's presence call for, once.
    fn sync_system_mic(&mut self) {
        let want = self.prefs.system_microphone
            && self.driver_installed
            && !matches!(self.engine, EngineState::Failed(_));
        if want != self.system_mic_applied {
            self.system_mic_applied = want;
            self.send(Request::SystemMic(want));
        }
    }

    /// The engine couldn't change the default input.
    pub fn system_mic_failed(&mut self, on: bool, e: String) {
        if on {
            self.system_mic_applied = false;
            self.prefs.system_microphone = false;
            self.save();
            self.fail(format!(
                "Couldn't make Oto Microphone the system microphone: {e}"
            ));
        } else {
            self.fail(format!("Couldn't restore your previous microphone: {e}"));
        }
    }

    /// The device picker's rows: `(device id, label)`, the system default first. Labels are
    /// unique, so the select's chosen label maps back to one device.
    pub fn device_options(&self) -> Vec<(Option<String>, String)> {
        // Short enough that a MacBook's own microphone fits the field without truncating.
        let default = match self.devices.iter().find(|d| d.is_default) {
            Some(d) => format!("Default ({})", d.name),
            None => SYSTEM_DEFAULT.to_owned(),
        };
        let mut out = vec![(None, default)];
        let mut push = |id: String, name: &str| {
            let mut label = name.to_owned();
            let mut n = 2;
            while out.iter().any(|(_, l)| *l == label) {
                label = format!("{name} {n}");
                n += 1;
            }
            out.push((Some(id), label));
        };
        for d in &self.devices {
            push(d.id.clone(), &d.name);
        }
        // A picked device that's unplugged stays listed, so the choice still shows.
        if let Some(id) = &self.prefs.input_device
            && !self.devices.iter().any(|d| &d.id == id)
        {
            let name = self
                .prefs
                .input_device_name
                .clone()
                .unwrap_or_else(|| id.clone());
            push(id.clone(), &format!("{name} (not connected)"));
        }
        out
    }

    pub fn device_label(&self) -> String {
        let opts = self.device_options();
        opts.iter()
            .find(|(id, _)| *id == self.prefs.input_device)
            .map_or_else(|| opts[0].1.clone(), |(_, l)| l.clone())
    }

    pub fn choose_device(&mut self, label: &str) {
        self.open = None;
        let Some((id, _)) = self.device_options().into_iter().find(|(_, l)| l == label) else {
            return;
        };
        if id == self.prefs.input_device {
            return;
        }
        self.prefs.input_device_name = id
            .as_ref()
            .and_then(|id| self.devices.iter().find(|d| &d.id == id))
            .map(|d| d.name.clone());
        self.prefs.input_device = id.clone();
        self.stop_hear_test();
        self.command(Command::SetInputDevice(id));
        self.save();
    }

    pub fn toggle_open(&mut self, which: Open) {
        self.open = if self.open == Some(which) {
            None
        } else {
            Some(which)
        };
    }

    /// What `Engine::stats` says runs: the effective model and the estimated delay in ms.
    pub fn set_running(&mut self, running: Option<(Model, f32)>) {
        self.running = running;
    }

    pub fn set_devices(&mut self, devices: Vec<InputDevice>) {
        self.devices = devices;
    }

    pub fn on_event(&mut self, e: Event) {
        match e {
            Event::ClientsActive(on) => {
                self.clients_active = on;
                if on {
                    // A call: the test must not hold the mic or the speakers.
                    self.stop_hear_test();
                }
                if !on {
                    self.load = None;
                }
                self.meter_gate();
            }
            Event::Load(l) => self.load = Some(l.clamp(0.0, 1.0)),
            Event::Error(msg) => {
                self.error = Some(msg);
                self.error_from_engine = true;
            }
            // Only the engine error: Voice Isolation can stay muted (a failed load) while a
            // missing mic or a stall comes and goes; `DegradedCleared` ends that.
            Event::Recovered => self.clear_engine_error(),
            Event::DegradedCleared => self.degraded = None,
            // The host thread re-reads the device list and calls `set_devices`.
            Event::DevicesChanged => {}
            Event::Degraded(msg) => self.degraded = Some(msg),
            Event::ModelFallback {
                requested,
                effective,
                ..
            } => {
                self.model_notice = Some(format!(
                    "{} couldn\u{2019}t load, so Oto is running {}.",
                    requested.name(),
                    effective.name()
                ));
            }
        }
    }

    fn clear_engine_error(&mut self) {
        if self.error_from_engine {
            self.error = None;
            self.error_from_engine = false;
        }
    }

    // ----- the engine's life ------------------------------------------------------------------

    /// Starts an engine through the hook and points the link at it.
    pub fn start_engine(&mut self) {
        self.engine = EngineState::Starting;
        self.system_mic_applied = false;
        self.link = (self.hooks.start_engine)(self.prefs.engine_settings(), self.meter.clone());
        self.sync_system_mic();
    }

    pub fn engine_started(&mut self) {
        self.engine = EngineState::Running;
    }

    /// `e` is `Engine::start`'s error.
    pub fn engine_failed(&mut self, e: &str) {
        self.engine = EngineState::Failed(engine_failure(e));
        self.system_mic_applied = false;
        self.clients_active = false;
        self.load = None;
        self.running = None;
        self.degraded = None;
        self.clear_engine_error();
        self.meter_gate();
        self.stop_hear_test();
    }

    pub fn retry_engine(&mut self) {
        if matches!(self.engine, EngineState::Failed(_)) {
            self.start_engine();
        }
    }

    /// Quitting: the host thread drops the engine, which stops audio and puts back the default
    /// input if "Use Oto as the system microphone" changed it.
    pub fn shutdown(&mut self) {
        self.stop_hear_test();
        self.command(Command::Shutdown);
    }

    // ----- "Hear the Difference" ---------------------------------------------------------------

    /// Why the test can't start now, if it can't. `None` means it can.
    pub fn hear_blocked(&self) -> Option<&'static str> {
        if self.clients_active {
            Some(HEAR_IN_CALL)
        } else if self.engine != EngineState::Running {
            Some("Waits for Oto\u{2019}s audio engine.")
        } else if !self.driver_installed {
            Some("Needs the audio driver.")
        } else if self.mic != Mic::Authorized {
            Some("Needs microphone access.")
        } else {
            None
        }
    }

    /// Goes to Settings. A take recording or playing stops first: its controls live on the
    /// main page, and Settings would hide them while the mic or the speakers stay busy.
    pub fn open_settings(&mut self) {
        self.stop_hear_test();
        self.open = None;
        self.page = Page::Settings;
    }

    /// Records a take (again). Not during a call.
    pub fn start_hear_test(&mut self) {
        if self.hear_blocked().is_some() {
            return;
        }
        self.stop_hear_test();
        let cancel = Arc::new(AtomicBool::new(false));
        self.hear_job = Some(cancel.clone());
        self.hear = HearTest::Recording { processing: false };
        (self.hooks.start_hear_test)(HearJob {
            device: self.prefs.input_device.clone(),
            model: oto_engine::model_for(self.prefs.tier),
            strength: self.prefs.strength,
            cancel,
        });
    }

    /// The take being recorded, for tests that play the hook's part.
    #[cfg(test)]
    pub fn hear_job_for_test(&self) -> Arc<AtomicBool> {
        self.hear_job.clone().expect("a take is recording")
    }

    fn is_hear_job(&self, job: &Arc<AtomicBool>) -> bool {
        self.hear_job.as_ref().is_some_and(|j| Arc::ptr_eq(j, job))
    }

    /// Take `job` is recorded and now being cleaned.
    pub fn hear_processing(&mut self, job: &Arc<AtomicBool>) {
        if self.is_hear_job(job) && matches!(self.hear, HearTest::Recording { .. }) {
            self.hear = HearTest::Recording { processing: true };
        }
    }

    /// Take `job` is done. A cancelled or replaced take is dropped unplayed.
    pub fn hear_recorded(&mut self, job: &Arc<AtomicBool>, r: Result<TestRecording, String>) {
        if !self.is_hear_job(job) || job.load(Ordering::Acquire) {
            return;
        }
        self.hear_job = None;
        self.hear = match r {
            // A call started meanwhile (that cancels the job, so this is belt and braces).
            Ok(_) if self.clients_active => HearTest::Idle,
            Ok(rec) => match (self.hooks.play_hear_test)(&rec) {
                Ok(()) => HearTest::Ready {
                    cleaned: false,
                    is_bluetooth_hfp: rec.is_bluetooth_hfp,
                },
                Err(e) => HearTest::Failed(format!("Couldn\u{2019}t play the test back: {e}")),
            },
            Err(e) => HearTest::Failed(format!("Couldn\u{2019}t record the test: {e}")),
        };
    }

    /// Plays the cleaned take (`true`) or the original.
    pub fn set_hear_cleaned(&mut self, on: bool) {
        if let HearTest::Ready { cleaned, .. } = &mut self.hear
            && *cleaned != on
        {
            *cleaned = on;
            (self.hooks.set_hear_cleaned)(on);
        }
    }

    /// Cancels a recording or stops the playback; back to the button.
    pub fn stop_hear_test(&mut self) {
        if let Some(job) = self.hear_job.take() {
            job.store(true, Ordering::Release);
        }
        if matches!(self.hear, HearTest::Ready { .. }) {
            (self.hooks.stop_hear_test)();
        }
        self.hear = HearTest::Idle;
    }

    // ----- what the system reports -------------------------------------------------------------

    /// A fresh look at the driver and the microphone permission.
    pub fn probed(&mut self, p: Probe) {
        let first = !self.driver_known;
        self.driver_known = true;
        // While the installer runs its own result is the truth; it reports when it ends.
        if !self.install.busy() {
            self.driver_installed = p.driver;
            self.driver_on_disk = p.on_disk;
        }
        self.mic = p.mic;
        // Someone who already has the driver and the permission has nothing to set up.
        if first && p.driver && p.mic == Mic::Authorized && !self.prefs.onboarded {
            self.prefs.onboarded = true;
            self.save();
        }
        self.sync_system_mic();
    }

    /// The step-two button: ask macOS, or, once it has said no, open System Settings (macOS
    /// asks only once).
    pub fn allow_mic(&mut self) {
        match self.mic {
            Mic::NotDetermined if !self.mic_requesting => {
                self.mic_requesting = true;
                (self.hooks.request_mic)();
            }
            Mic::Denied => {
                if let Err(e) = (self.hooks.open_mic_settings)() {
                    self.fail(e);
                }
            }
            _ => {}
        }
    }

    /// The system's microphone prompt was answered; `mic` is the status now.
    pub fn mic_answered(&mut self, mic: Mic) {
        self.mic_requesting = false;
        self.mic = mic;
    }

    /// Setup is over: by "Done", or by "Not Now" once the driver is in (the main screen keeps
    /// saying what's missing). Without a driver "Not Now" only lasts until Oto quits.
    pub fn leave_onboarding(&mut self) {
        if self.driver_installed {
            self.prefs.onboarded = true;
            self.save();
        } else {
            self.skipped_onboarding = true;
        }
    }

    // ----- the driver --------------------------------------------------------------------------

    pub fn start_install(&mut self) {
        if !self.install.busy() {
            self.install = Install::Running;
            (self.hooks.install)();
        }
    }

    /// The install finished; `p` is what the system reports now.
    pub fn installed(&mut self, outcome: Outcome, p: Probe) {
        self.driver_installed = p.driver;
        self.driver_on_disk = p.on_disk;
        self.mic = p.mic;
        self.install = match outcome {
            Outcome::Installed if !p.driver => Install::Failed(
                "The installer finished, but Core Audio doesn't list Oto Microphone yet.".into(),
            ),
            Outcome::Installed => {
                // Only the missing-driver complaint is stale. Any other error stays until the
                // engine reports recovery: it does not repeat an error it has already sent.
                if self
                    .error
                    .as_deref()
                    .is_some_and(|e| e.starts_with("Oto driver is not installed"))
                {
                    self.clear_engine_error();
                }
                Install::Idle
            }
            Outcome::Cancelled => Install::Idle,
            Outcome::InstalledNeedsReboot => {
                self.driver_on_disk = true;
                Install::Notice(format!("Oto's driver is installed. {REBOOT_MESSAGE}"))
            }
            Outcome::Failed(msg) => Install::Failed(msg),
        };
        self.sync_system_mic();
    }

    pub fn start_uninstall(&mut self) {
        if !self.install.busy() {
            self.install = Install::Removing;
            // Stay on the settings page rather than jumping to setup when the driver goes.
            self.skipped_onboarding = true;
            (self.hooks.uninstall)();
        }
    }

    /// The uninstall finished; `p` is what the system reports now.
    pub fn uninstalled(&mut self, outcome: UninstallOutcome, p: Probe) {
        self.driver_installed = p.driver;
        self.driver_on_disk = p.on_disk;
        self.mic = p.mic;
        self.install = match outcome {
            UninstallOutcome::Removed
            | UninstallOutcome::NotInstalled
            | UninstallOutcome::Cancelled => Install::Idle,
            UninstallOutcome::RemovedNeedsReboot => {
                Install::Notice(format!("Oto's driver was removed. {REBOOT_MESSAGE}"))
            }
            UninstallOutcome::Failed(msg) => Install::Failed(msg),
        };
        self.sync_system_mic();
    }

    // ----- what the screens say ----------------------------------------------------------------

    /// The one-line status under the controls.
    pub fn status(&self) -> String {
        match &self.engine {
            EngineState::Failed(_) => return "Audio engine not running".into(),
            EngineState::Starting => return "Starting\u{2026}".into(),
            EngineState::Running => {}
        }
        if !self.driver_installed {
            return if self.driver_known {
                "Driver not installed".into()
            } else {
                "Starting\u{2026}".into()
            };
        }
        if self.mic.blocked() {
            return "No microphone access".into();
        }
        if !self.prefs.enabled {
            return if self.isolating() {
                "Voice Isolation is off".into()
            } else {
                "Noise cancellation is off".into()
            };
        }
        if self.degraded.is_some() {
            return "Voice Isolation can\u{2019}t run: microphone muted".into();
        }
        if self.clients_active {
            "In use by an app".into()
        } else {
            "Waiting for an app to use Oto Microphone".into()
        }
    }

    /// CPU, as "3% CPU", while there's a reading.
    pub fn cpu(&self) -> Option<String> {
        self.load
            .filter(|_| self.clients_active)
            .map(|l| format!("{:.0}% CPU", l * 100.0))
    }

    /// The running model and its delay, as "DeepFilterNet3 · 56 ms", while streams run.
    pub fn running_model(&self) -> Option<String> {
        self.running
            .filter(|_| self.clients_active)
            .map(|(m, ms)| format!("{} \u{b7} {ms:.0} ms", m.name()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> State {
        let mut s = State::new(Prefs::default(), true);
        s.link = Link::Record(Vec::new());
        s.devices = vec![
            InputDevice {
                id: "a".into(),
                name: "MacBook Pro Microphone".into(),
                is_default: true,
            },
            InputDevice {
                id: "b".into(),
                name: "USB Mic".into(),
                is_default: false,
            },
            InputDevice {
                id: "c".into(),
                name: "USB Mic".into(),
                is_default: false,
            },
        ];
        s
    }

    fn requests(s: &State) -> &[Request] {
        match &s.link {
            Link::Record(v) => v,
            _ => &[],
        }
    }

    fn sent(s: &State) -> Vec<Command> {
        requests(s)
            .iter()
            .filter_map(|r| match r {
                Request::Cmd(c) => Some(c.clone()),
                Request::SystemMic(_) => None,
            })
            .collect()
    }

    fn system_mic(s: &State) -> Vec<bool> {
        requests(s)
            .iter()
            .filter_map(|r| match r {
                Request::SystemMic(on) => Some(*on),
                Request::Cmd(_) => None,
            })
            .collect()
    }

    fn probe(driver: bool) -> Probe {
        Probe {
            driver,
            on_disk: driver,
            mic: Mic::Authorized,
        }
    }

    #[test]
    fn toggling_sends_once_and_gates_the_meter() {
        let mut s = state();
        s.on_event(Event::ClientsActive(true));
        assert!(s.meter.active.load(Ordering::Relaxed));
        s.toggle();
        s.set_enabled(false);
        assert_eq!(sent(&s), [Command::SetEnabled(false)]);
        assert!(!s.meter.active.load(Ordering::Relaxed));
    }

    #[test]
    fn device_labels_are_unique_and_map_back() {
        let mut s = state();
        let labels: Vec<_> = s.device_options().into_iter().map(|(_, l)| l).collect();
        assert_eq!(
            labels,
            [
                "Default (MacBook Pro Microphone)",
                "MacBook Pro Microphone",
                "USB Mic",
                "USB Mic 2"
            ]
        );
        s.choose_device("USB Mic 2");
        assert_eq!(s.prefs.input_device.as_deref(), Some("c"));
        assert_eq!(s.device_label(), "USB Mic 2");
        s.choose_device("Default (MacBook Pro Microphone)");
        assert_eq!(
            sent(&s),
            [
                Command::SetInputDevice(Some("c".into())),
                Command::SetInputDevice(None)
            ]
        );
    }

    #[test]
    fn an_unplugged_choice_stays_listed() {
        let mut s = state();
        s.choose_device("USB Mic");
        s.set_devices(Vec::new());
        assert_eq!(s.device_label(), "USB Mic (not connected)");
        assert_eq!(s.device_options()[0].1, SYSTEM_DEFAULT);
    }

    #[test]
    fn events_drive_the_status_line() {
        let mut s = state();
        assert_eq!(s.status(), "Waiting for an app to use Oto Microphone");
        s.on_event(Event::ClientsActive(true));
        s.on_event(Event::Load(0.034));
        assert_eq!(s.status(), "In use by an app");
        assert_eq!(s.cpu().as_deref(), Some("3% CPU"));
        s.on_event(Event::ClientsActive(false));
        assert_eq!(s.cpu(), None);
        s.on_event(Event::Error("Mic unplugged".into()));
        assert_eq!(s.error.as_deref(), Some("Mic unplugged"));
    }

    #[test]
    fn recovered_clears_the_engines_error_and_only_that() {
        let mut s = state();
        s.on_event(Event::Error("Mic unplugged".into()));
        s.on_event(Event::Recovered);
        assert_eq!(s.error, None);

        // The app's own errors aren't the engine's to clear.
        s.hooks.set_login = |_| Err("nope".into());
        s.set_start_at_login(true);
        s.on_event(Event::Recovered);
        assert_eq!(s.error.as_deref(), Some("nope"));
        s.dismiss_error();
        assert_eq!(s.error, None);
    }

    #[test]
    fn a_successful_install_clears_the_missing_driver_error() {
        let mut s = State::new(Prefs::default(), false);
        s.on_event(Event::Error(
            "Oto driver is not installed: Oto Microphone is not available".into(),
        ));
        s.start_install();
        s.installed(Outcome::Installed, probe(true));
        assert_eq!(s.error, None);
        assert_eq!(s.install, Install::Idle);
        assert!(s.driver_installed);
    }

    #[test]
    fn a_successful_install_keeps_other_engine_errors() {
        let mut s = State::new(Prefs::default(), false);
        s.on_event(Event::Error("No microphone found".into()));
        s.start_install();
        s.installed(Outcome::Installed, probe(true));
        assert_eq!(s.error.as_deref(), Some("No microphone found"));
    }

    #[test]
    fn onboarding_until_installed_and_finished() {
        let mut s = State::new(Prefs::default(), false);
        assert!(s.onboarding());
        s.start_install();
        assert_eq!(s.install, Install::Running);
        s.installed(Outcome::Cancelled, probe(false));
        assert_eq!(s.install, Install::Idle);
        assert!(s.onboarding());
        s.installed(Outcome::Installed, probe(false));
        assert!(matches!(s.install, Install::Failed(_)));
        s.installed(Outcome::Installed, probe(true));
        // The driver is in; the permission and system-microphone steps are still on offer.
        assert!(s.onboarding());
        s.leave_onboarding();
        assert!(!s.onboarding());
        assert!(s.prefs.onboarded);
        // It comes back if the driver goes missing.
        s.probed(probe(false));
        assert!(s.onboarding());
    }

    #[test]
    fn not_now_without_a_driver_lasts_one_run() {
        let mut s = State::new(Prefs::default(), false);
        s.leave_onboarding();
        assert!(!s.onboarding());
        assert!(!s.prefs.onboarded);
    }

    #[test]
    fn nothing_is_offered_before_the_first_probe() {
        let mut s = State::new(Prefs::default(), false);
        s.driver_known = false;
        assert!(!s.onboarding());
        assert_eq!(s.status(), "Starting\u{2026}");
        // An existing setup skips onboarding for good.
        s.probed(probe(true));
        assert!(!s.onboarding());
        assert!(s.prefs.onboarded);

        let mut s = State::new(Prefs::default(), false);
        s.driver_known = false;
        s.probed(Probe {
            mic: Mic::NotDetermined,
            ..probe(true)
        });
        assert!(s.onboarding(), "the permission step is still to do");
    }

    #[test]
    fn a_needed_restart_is_a_notice_not_a_failure() {
        let mut s = State::new(Prefs::default(), false);
        s.start_install();
        s.installed(Outcome::InstalledNeedsReboot, probe(false));
        assert_eq!(
            s.install,
            Install::Notice("Oto's driver is installed. Restart your Mac to finish.".into())
        );
        assert!(s.driver_on_disk);
        assert!(!s.driver_installed);
    }

    #[test]
    fn uninstall_outcomes() {
        let mut s = state();
        s.page = Page::Settings;
        s.start_uninstall();
        assert_eq!(s.install, Install::Removing);
        assert!(s.install.busy() && s.holding());
        // A probe during the uninstall doesn't override what it will report.
        s.probed(probe(false));
        assert!(s.driver_installed);
        s.uninstalled(UninstallOutcome::Removed, probe(false));
        assert_eq!(s.install, Install::Idle);
        assert!(!s.driver_installed);
        assert!(!s.onboarding(), "stays on settings");
        assert_eq!(s.status(), "Driver not installed");

        let mut s = state();
        s.start_uninstall();
        s.uninstalled(UninstallOutcome::RemovedNeedsReboot, probe(true));
        assert!(matches!(&s.install, Install::Notice(m) if m.contains("Restart your Mac")));
        s.start_uninstall();
        s.uninstalled(UninstallOutcome::Failed("no".into()), probe(true));
        assert_eq!(s.install, Install::Failed("no".into()));
        s.start_uninstall();
        s.uninstalled(UninstallOutcome::Cancelled, probe(true));
        assert_eq!(s.install, Install::Idle);
        assert!(s.driver_installed);
    }

    #[test]
    fn a_failed_login_change_keeps_the_old_value() {
        let mut s = state();
        s.hooks.set_login = |_| Err("nope".into());
        s.set_start_at_login(true);
        assert!(!s.prefs.start_at_login);
        assert_eq!(s.error.as_deref(), Some("nope"));
    }

    #[test]
    fn strength_and_tier_send_only_changes() {
        let mut s = state();
        s.tiers = vec![Tier::Light, Tier::Standard];
        s.set_strength(Strength::Balanced);
        s.set_strength(Strength::Strong);
        s.set_tier(Tier::Light);
        // Not on offer: ignored.
        s.set_tier(Tier::Max);
        assert_eq!(
            sent(&s),
            [
                Command::SetStrength(Strength::Strong),
                Command::SetTier(Tier::Light)
            ]
        );
    }

    #[test]
    fn the_model_picker_offers_only_this_builds_tiers() {
        let mut s = state();
        s.tiers = vec![Tier::Light, Tier::Standard, Tier::Max, Tier::Fast];
        assert!(s.offers_tiers());
        assert_eq!(s.tier_label(), "DeepFilterNet3");
        assert_eq!(s.tier_options().len(), 4);
        s.choose_tier("DPDFNet-2 (stronger)");
        s.choose_tier("DPDFNet-2 (stronger)");
        s.choose_tier("FastEnhancer-S (low delay)");
        s.choose_tier("RNNoise (low CPU)");
        s.choose_tier("no such model");
        assert_eq!(s.prefs.tier, Tier::Light);
        assert_eq!(
            sent(&s),
            [
                Command::SetTier(Tier::Max),
                Command::SetTier(Tier::Fast),
                Command::SetTier(Tier::Light)
            ]
        );
        s.tiers = vec![Tier::Light, Tier::Standard];
        assert_eq!(s.tier_options(), ["DeepFilterNet3", "RNNoise (low CPU)"]);
        s.tiers = vec![Tier::Standard];
        assert!(!s.offers_tiers());
    }

    #[test]
    fn a_model_fallback_is_a_notice_not_an_error() {
        let mut s = state();
        s.tiers = vec![Tier::Light, Tier::Standard];
        s.on_event(Event::ClientsActive(true));
        s.on_event(Event::ModelFallback {
            requested: Model::DeepFilterNet3,
            effective: Model::RNNoise,
            reason: "running pass codegen".into(),
        });
        s.set_running(Some((Model::RNNoise, 36.0)));
        assert_eq!(s.error, None);
        assert_eq!(
            s.model_notice.as_deref(),
            Some("DeepFilterNet3 couldn\u{2019}t load, so Oto is running RNNoise.")
        );
        assert_eq!(s.running_model().as_deref(), Some("RNNoise \u{b7} 36 ms"));
        // A recovery is about errors, not the model.
        s.on_event(Event::Recovered);
        assert!(s.model_notice.is_some());
        // Changing the model makes the engine try again, so the notice goes.
        s.set_tier(Tier::Light);
        assert_eq!(s.model_notice, None);

        // Nothing runs while no app records.
        s.on_event(Event::ClientsActive(false));
        assert_eq!(s.running_model(), None);
    }

    #[test]
    fn an_unavailable_saved_tier_falls_back() {
        let Some(unavailable) = [Tier::Light, Tier::Standard, Tier::Max, Tier::Fast]
            .into_iter()
            .find(|t| !oto_engine::available_tiers().contains(t))
        else {
            return; // Every tier has its own model in this build.
        };
        let s = State::new(
            Prefs {
                tier: unavailable,
                ..Prefs::default()
            },
            true,
        );
        assert_eq!(s.prefs.tier, Tier::default());
    }

    fn isolation_state() -> State {
        let mut s = state();
        s.tiers = vec![Tier::Light, Tier::Standard, Tier::Max, Tier::Isolate];
        s
    }

    #[test]
    fn modes_keep_the_chosen_noise_cancellation_model() {
        let mut s = isolation_state();
        assert_eq!(s.mode(), Mode::NoiseCancellation);
        assert!(s.offers_isolation());
        s.choose_tier("DPDFNet-2 (stronger)");
        s.set_mode(Mode::VoiceIsolation);
        assert_eq!((s.prefs.tier, s.prefs.nc_tier), (Tier::Isolate, Tier::Max));
        assert_eq!(s.tier_label(), "DPDFNet-2 (stronger)");
        // Picking a model in Voice Isolation only remembers it.
        s.choose_tier("RNNoise (low CPU)");
        assert_eq!(
            (s.prefs.tier, s.prefs.nc_tier),
            (Tier::Isolate, Tier::Light)
        );
        s.set_mode(Mode::NoiseCancellation);
        assert_eq!(s.prefs.tier, Tier::Light);
        s.set_mode(Mode::NoiseCancellation);
        assert_eq!(
            sent(&s),
            [
                Command::SetTier(Tier::Max),
                Command::SetTier(Tier::Isolate),
                Command::SetTier(Tier::Light)
            ]
        );
        // Isolation is a mode, not a model.
        s.set_nc_tier(Tier::Isolate);
        assert_eq!(s.prefs.nc_tier, Tier::Light);
        assert!(!s.tier_options().contains(&""));
        assert_eq!(s.tier_options().len(), 3);

        // Without Nearfield there is no second mode.
        let mut s = state();
        s.tiers = vec![Tier::Light, Tier::Standard];
        assert!(!s.offers_isolation());
        s.set_mode(Mode::VoiceIsolation);
        assert_eq!(s.prefs.tier, Tier::Standard);
    }

    #[test]
    fn a_saved_isolation_without_nearfield_keeps_isolate_to_fail_closed() {
        let s = State::new(
            Prefs {
                tier: Tier::Isolate,
                nc_tier: Tier::Light,
                ..Prefs::default()
            },
            true,
        );
        let tiers = oto_engine::available_tiers();
        // A non-isolating model this build lacks falls back to the default, but a saved
        // Voice Isolation choice stays Isolate so the engine fails closed instead of leaking.
        let nc = if tiers.contains(&Tier::Light) {
            Tier::Light
        } else {
            Tier::default()
        };
        assert_eq!(s.prefs.nc_tier, nc);
        assert_eq!(s.prefs.tier, Tier::Isolate);
    }

    #[test]
    fn degraded_is_its_own_banner_until_recovered_or_switched() {
        let mut s = isolation_state();
        s.set_mode(Mode::VoiceIsolation);
        s.on_event(Event::ClientsActive(true));
        s.on_event(Event::Degraded("Voice Isolation fell behind".into()));
        assert_eq!(s.degraded.as_deref(), Some("Voice Isolation fell behind"));
        assert_eq!(s.error, None, "not an error");
        assert!(s.status().contains("muted"));
        s.on_event(Event::Recovered);
        assert!(
            s.degraded.is_some(),
            "an engine recovery is not the isolation's"
        );
        s.on_event(Event::DegradedCleared);
        assert_eq!(s.degraded, None);
        assert_eq!(s.status(), "In use by an app");

        s.on_event(Event::Degraded("couldn't load".into()));
        s.switch_to_noise_cancellation();
        assert_eq!(s.degraded, None);
        assert_eq!(s.prefs.tier, Tier::Standard);
        assert_eq!(sent(&s).last(), Some(&Command::SetTier(Tier::Standard)));

        // Picking another model from Settings leaves Isolation too.
        s.set_mode(Mode::VoiceIsolation);
        s.on_event(Event::Degraded("couldn't load".into()));
        s.set_tier(Tier::Max);
        assert_eq!(s.degraded, None);
    }

    /// Voice Isolation failed to load; then the mic goes missing and the call ends. The engine
    /// recovers from the missing mic, but the failed load still mutes the next call, so the
    /// banner stays.
    #[test]
    fn degraded_banner_survives_an_unrelated_recovery() {
        let mut s = isolation_state();
        s.set_mode(Mode::VoiceIsolation);
        s.on_event(Event::ClientsActive(true));
        s.on_event(Event::Degraded("Voice Isolation could not load".into()));
        s.on_event(Event::Error("no microphone found".into()));
        s.on_event(Event::ClientsActive(false));
        s.on_event(Event::Recovered);
        assert_eq!(s.error, None, "the missing-mic error is over");
        assert_eq!(
            s.degraded.as_deref(),
            Some("Voice Isolation could not load")
        );
        s.on_event(Event::ClientsActive(true));
        assert!(s.status().contains("muted"), "{}", s.status());
    }

    fn take() -> TestRecording {
        TestRecording {
            sample_rate: 16_000,
            raw: vec![0.0; 1600].into(),
            cleaned: vec![0.0; 1600].into(),
            is_bluetooth_hfp: false,
        }
    }

    static HEAR_JOBS: std::sync::Mutex<Vec<HearJob>> = std::sync::Mutex::new(Vec::new());
    static PLAYING: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);
    static CLEANED: AtomicBool = AtomicBool::new(false);

    fn hear_state() -> State {
        let mut s = state();
        s.hooks.start_hear_test = |job| HEAR_JOBS.lock().unwrap().push(job);
        s.hooks.play_hear_test = |_| {
            PLAYING.fetch_add(1, Ordering::Relaxed);
            Ok(())
        };
        s.hooks.stop_hear_test = || {
            PLAYING.fetch_sub(1, Ordering::Relaxed);
        };
        s.hooks.set_hear_cleaned = |on| CLEANED.store(on, Ordering::Relaxed);
        s
    }

    /// The one test that uses the hear hooks' statics, so parallel tests can't disturb them.
    #[test]
    fn hear_the_difference() {
        let mut s = hear_state();
        s.prefs.strength = Strength::Strong;
        s.start_hear_test();
        assert_eq!(s.hear, HearTest::Recording { processing: false });
        let job = HEAR_JOBS.lock().unwrap().pop().unwrap();
        assert_eq!(job.strength, Strength::Strong);
        assert_eq!(job.model, oto_engine::model_for(s.prefs.tier));
        s.hear_processing(&job.cancel);
        assert_eq!(s.hear, HearTest::Recording { processing: true });
        s.hear_recorded(&job.cancel, Ok(take()));
        assert_eq!(
            s.hear,
            HearTest::Ready {
                cleaned: false,
                is_bluetooth_hfp: false
            }
        );
        assert_eq!(PLAYING.load(Ordering::Relaxed), 1);
        s.set_hear_cleaned(true);
        assert!(CLEANED.load(Ordering::Relaxed));
        s.set_hear_cleaned(false);
        assert!(!CLEANED.load(Ordering::Relaxed));

        // A call stops the playback at once.
        s.on_event(Event::ClientsActive(true));
        assert_eq!(s.hear, HearTest::Idle);
        assert_eq!(PLAYING.load(Ordering::Relaxed), 0);
        // And nothing starts during it.
        s.start_hear_test();
        assert_eq!(s.hear, HearTest::Idle);
        assert!(HEAR_JOBS.lock().unwrap().is_empty());
        assert_eq!(s.hear_blocked(), Some(HEAR_IN_CALL));
        s.on_event(Event::ClientsActive(false));

        // A call during the recording cancels it; its late result is dropped unplayed.
        s.start_hear_test();
        let job = HEAR_JOBS.lock().unwrap().pop().unwrap();
        s.on_event(Event::ClientsActive(true));
        assert!(job.cancel.load(Ordering::Relaxed));
        assert_eq!(s.hear, HearTest::Idle);
        s.hear_recorded(&job.cancel, Ok(take()));
        assert_eq!(s.hear, HearTest::Idle);
        assert_eq!(PLAYING.load(Ordering::Relaxed), 0);
        s.on_event(Event::ClientsActive(false));

        // "Record Again" replaces the take; the old one's result is ignored.
        s.start_hear_test();
        let old = HEAR_JOBS.lock().unwrap().pop().unwrap();
        s.start_hear_test();
        let new = HEAR_JOBS.lock().unwrap().pop().unwrap();
        assert!(old.cancel.load(Ordering::Relaxed));
        s.hear_recorded(&old.cancel, Err("late".into()));
        assert_eq!(s.hear, HearTest::Recording { processing: false });
        s.hear_recorded(&new.cancel, Err("no microphone found".into()));
        assert!(matches!(&s.hear, HearTest::Failed(m) if m.contains("no microphone found")));

        // Opening Settings stops a recording, and a playing take.
        s.start_hear_test();
        let job = HEAR_JOBS.lock().unwrap().pop().unwrap();
        s.open_settings();
        assert_eq!(s.page, Page::Settings);
        assert!(job.cancel.load(Ordering::Relaxed));
        assert_eq!(s.hear, HearTest::Idle);
        s.page = Page::Main;
        s.start_hear_test();
        let job = HEAR_JOBS.lock().unwrap().pop().unwrap();
        s.hear_recorded(&job.cancel, Ok(take()));
        assert_eq!(PLAYING.load(Ordering::Relaxed), 1);
        s.open_settings();
        assert_eq!(PLAYING.load(Ordering::Relaxed), 0);
        assert_eq!(s.hear, HearTest::Idle);
        s.page = Page::Main;

        // Quitting stops a playing test.
        s.start_hear_test();
        let job = HEAR_JOBS.lock().unwrap().pop().unwrap();
        s.hear_recorded(&job.cancel, Ok(take()));
        assert_eq!(PLAYING.load(Ordering::Relaxed), 1);
        s.shutdown();
        assert_eq!(PLAYING.load(Ordering::Relaxed), 0);
        assert_eq!(s.hear, HearTest::Idle);
        assert_eq!(sent(&s).last(), Some(&Command::Shutdown));

        // A playback that can't start is a failure, not a Ready with silence.
        s.hooks.play_hear_test = |_| Err("no output device".into());
        s.start_hear_test();
        let job = HEAR_JOBS.lock().unwrap().pop().unwrap();
        s.hear_recorded(&job.cancel, Ok(take()));
        assert!(matches!(&s.hear, HearTest::Failed(m) if m.contains("no output device")));

        // Not without the mic, the driver or the engine.
        s.mic = Mic::Denied;
        assert!(s.hear_blocked().is_some());
        s.mic = Mic::Authorized;
        s.engine_failed("boom");
        assert!(s.hear_blocked().is_some());
    }

    #[test]
    fn engine_failure_and_retry() {
        static STARTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mut s = state();
        s.hooks.start_engine = |_, _| {
            STARTS.fetch_add(1, Ordering::Relaxed);
            Link::Record(Vec::new())
        };
        s.on_event(Event::ClientsActive(true));
        s.retry_engine();
        assert_eq!(STARTS.load(Ordering::Relaxed), 0, "nothing to retry");

        s.engine_failed("no audio hardware");
        assert_eq!(
            s.engine,
            EngineState::Failed(
                "Oto couldn\u{2019}t start its audio engine: no audio hardware".into()
            )
        );
        assert_eq!(s.status(), "Audio engine not running");
        assert!(!s.clients_active);
        s.retry_engine();
        assert_eq!(STARTS.load(Ordering::Relaxed), 1);
        assert_eq!(s.engine, EngineState::Starting);
        s.engine_started();
        assert_eq!(s.engine, EngineState::Running);

        s.engine_failed(oto_engine::ALREADY_RUNNING);
        assert!(matches!(&s.engine, EngineState::Failed(m) if m.contains("Another copy")));
    }

    #[test]
    fn a_dead_host_thread_shows_as_a_failed_engine() {
        let mut s = state();
        let (tx, rx) = std::sync::mpsc::channel();
        drop(rx);
        s.link = Link::Engine(tx);
        s.set_strength(Strength::Strong);
        assert!(matches!(s.engine, EngineState::Failed(_)));
    }

    #[test]
    fn system_microphone_follows_the_setting_and_the_driver() {
        let mut s = state();
        s.set_system_mic(true);
        s.set_system_mic(true);
        assert!(s.prefs.system_microphone);
        assert_eq!(system_mic(&s), [true]);
        s.set_system_mic(false);
        assert_eq!(system_mic(&s), [true, false]);

        // Saved on, but no driver yet: asked for once the driver appears, undone when it goes.
        let mut s = State::new(
            Prefs {
                system_microphone: true,
                ..Prefs::default()
            },
            false,
        );
        s.link = Link::Record(Vec::new());
        s.probed(probe(false));
        assert!(system_mic(&s).is_empty());
        s.installed(Outcome::Installed, probe(true));
        s.probed(probe(true));
        assert_eq!(system_mic(&s), [true]);
        s.start_uninstall();
        s.uninstalled(UninstallOutcome::Removed, probe(false));
        assert_eq!(system_mic(&s), [true, false]);
    }

    #[test]
    fn a_new_engine_is_asked_again_for_the_system_microphone() {
        let mut s = state();
        s.set_system_mic(true);
        s.hooks.start_engine = |_, _| Link::Record(Vec::new());
        s.engine_failed("boom");
        s.retry_engine();
        assert_eq!(requests(&s), [Request::SystemMic(true)]);
    }

    #[test]
    fn a_failed_system_microphone_switch_turns_the_setting_off() {
        let mut s = state();
        s.set_system_mic(true);
        s.system_mic_failed(true, "no such device".into());
        assert!(!s.prefs.system_microphone);
        assert!(s.error.as_deref().unwrap().contains("no such device"));
        // Nothing to undo, so nothing more is sent.
        s.probed(probe(true));
        assert_eq!(system_mic(&s), [true]);
    }

    #[test]
    fn the_microphone_button_asks_once_then_opens_settings() {
        static ASKED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static OPENED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mut s = state();
        s.hooks.request_mic = || {
            ASKED.fetch_add(1, Ordering::Relaxed);
        };
        s.hooks.open_mic_settings = || {
            OPENED.fetch_add(1, Ordering::Relaxed);
            Ok(())
        };
        s.mic = Mic::NotDetermined;
        s.allow_mic();
        s.allow_mic();
        assert_eq!(ASKED.load(Ordering::Relaxed), 1);
        assert!(s.holding(), "the prompt has focus; the popover stays");
        s.mic_answered(Mic::Denied);
        assert!(!s.holding());
        assert_eq!(s.status(), "No microphone access");
        s.allow_mic();
        assert_eq!(ASKED.load(Ordering::Relaxed), 1);
        assert_eq!(OPENED.load(Ordering::Relaxed), 1);
        s.mic_answered(Mic::Authorized);
        s.allow_mic();
        assert_eq!(OPENED.load(Ordering::Relaxed), 1);
    }
}
