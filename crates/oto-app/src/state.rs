//! The app's state and everything that changes it. No Rune views and no I/O beyond the hooks
//! [`Hooks`] names, so it unit-tests without a window or an engine.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, SyncSender};

use oto_engine::dsp::Model;
use oto_engine::test_capture::TestRecording;
use oto_engine::{
    BUNDLED_DRIVER_VERSION, Command, DriverStatus, Event, InputDevice,
    MIN_COMPATIBLE_DRIVER_VERSION, Settings, Strength, Tier,
};

use crate::driver::{Outcome, REBOOT_MESSAGE, UninstallOutcome};
use crate::meter::Meter;
use crate::permission::Mic;
use crate::prefs::Prefs;
use crate::tray::Tray;
use crate::updater::apply::{InstallMode, NotifyReason, UnsupportedReason};
use crate::updater::keys::RELEASES_URL;
use crate::updater::verify::{CheckOutcome, StagedUpdate, UpdateInfo};

/// Shows the "Hear the Difference" test: record a few seconds, then compare raw and cleaned.
pub const HEAR_TEST: bool = true;
/// How long "Hear the Difference" records, seconds.
pub const HEAR_SECONDS: f32 = 5.0;
/// Why "Hear the Difference" can't start during a call: it opens the mic and the speakers.
pub const HEAR_IN_CALL: &str = "End your call to run the mic test.";
/// Why nothing that touches the audio setup runs while an update candidate waits to commit.
pub const FINISHING_UPDATE: &str = "Finishing update\u{2026}";
/// Why "Restart to Update" and driver changes wait during a call.
pub const UPDATE_IN_CALL: &str = "End your call to finish updating.";
/// What the host thread says when it can't tell whether a call is active.
pub const CALL_CHECK_FAILED: &str =
    "Couldn't verify whether a call is active. Try again in a moment.";
/// What a driver install or update warns about before the password prompt.
pub const DRIVER_PAUSE: &str = "Asks for your password. Audio will pause briefly in all apps (or finish on your next restart if audio is playing).";
/// How long between automatic update checks, seconds.
pub const UPDATE_INTERVAL_SECS: u64 = 24 * 60 * 60;

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
    /// The driver as Core Audio sees it, with its revision.
    pub driver_status: DriverStatus,
    /// `OtoDriverVersion` of the driver bundle in the HAL plug-in folder, loaded or not (0 for
    /// legacy drivers); `None` if absent.
    pub on_disk_version: Option<u32>,
    pub mic: Mic,
}

impl Probe {
    pub fn new(driver_status: DriverStatus, on_disk_version: Option<u32>, mic: Mic) -> Probe {
        Probe {
            driver_status,
            on_disk_version,
            mic,
        }
    }

    /// Core Audio lists Oto Microphone (any revision).
    pub fn is_loaded(&self) -> bool {
        self.driver_status.is_loaded()
    }

    /// The driver bundle is in the HAL plug-in folder, loaded or not.
    pub fn is_on_disk(&self) -> bool {
        self.on_disk_version.is_some()
    }
}

/// The driver as the UI presents it: the state table of docs/plans/0003-app-updates.md,
/// Decision 5.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverUiState {
    /// Not loaded, not on disk: onboarding, **Install…**.
    NotInstalled,
    /// On disk, not loaded: restart the Mac to finish.
    InstalledNeedsReboot,
    /// This build's revision is loaded and the copy on disk is as new.
    UpToDate,
    /// This build's revision is loaded, but the copy on disk is missing or older: **Reinstall…**.
    OnDiskOutdatedOrMissing,
    /// A newer compatible driver is loaded (an older Oto opened): never offer a downgrade.
    NewerLoaded,
    /// A new driver is on disk; Core Audio still holds the old one until a restart.
    UpdateNeedsReboot,
    /// Removed on disk, still loaded until a restart.
    RemovedNeedsReboot,
    /// An older, compatible driver runs: a non-blocking **Update Driver…** banner.
    UpdateAvailable,
    /// The loaded driver is too old for this engine: an urgent **Update Driver…** banner.
    Incompatible,
    /// Core Audio couldn't say; nothing is flagged.
    Unknown,
}

impl DriverUiState {
    /// Places the loaded driver and the one on disk in the state table.
    pub fn of(loaded: DriverStatus, on_disk: Option<u32>, bundled: u32, min_compat: u32) -> Self {
        let rev = match loaded {
            DriverStatus::Unknown => return Self::Unknown,
            DriverStatus::Absent if on_disk.is_some() => return Self::InstalledNeedsReboot,
            DriverStatus::Absent => return Self::NotInstalled,
            DriverStatus::Loaded(rev) => rev,
        };
        if rev > bundled {
            return Self::NewerLoaded;
        }
        if rev == bundled {
            return match on_disk {
                Some(d) if d >= bundled => Self::UpToDate,
                _ => Self::OnDiskOutdatedOrMissing,
            };
        }
        match on_disk {
            Some(d) if d >= bundled => Self::UpdateNeedsReboot,
            None => Self::RemovedNeedsReboot,
            Some(_) if rev < min_compat => Self::Incompatible,
            Some(_) => Self::UpdateAvailable,
        }
    }

    /// The install button offers an update rather than a reinstall.
    pub fn offers_update(self) -> bool {
        matches!(self, Self::UpdateAvailable | Self::Incompatible)
    }
}

/// The app update, from the check to the restart.
#[derive(Clone, Debug, PartialEq)]
pub enum UpdateState {
    Idle,
    /// A check is under way; `manual` ones report every result, background ones only offers.
    Checking {
        manual: bool,
    },
    UpToDate,
    Available(UpdateInfo),
    Downloading {
        info: UpdateInfo,
        pct: u8,
    },
    Verifying(UpdateInfo),
    /// Downloaded and verified: **Restart to Update**.
    Ready(StagedUpdate),
    /// Staging next to the app, shutting the engine down and handing over to the helper.
    Applying,
    /// The manifest isn't signed by a key this build trusts: neutral copy, link to Releases.
    UnrecognizedKey {
        releases_url: String,
    },
    /// A newer version needs a newer macOS.
    IncompatibleOs {
        version: String,
        min_os: String,
    },
    Failed(String),
}

impl UpdateState {
    /// The offer this state is about, if any.
    pub fn info(&self) -> Option<&UpdateInfo> {
        match self {
            UpdateState::Available(i)
            | UpdateState::Downloading { info: i, .. }
            | UpdateState::Verifying(i) => Some(i),
            UpdateState::Ready(staged) => Some(&staged.info),
            _ => None,
        }
    }

    /// A download or verification is running.
    pub fn downloading(&self) -> bool {
        matches!(
            self,
            UpdateState::Downloading { .. } | UpdateState::Verifying(_)
        )
    }
}

/// One update operation (check, download, apply), for the hooks. `gen` identifies it: results
/// of any other generation are stale and dropped. Setting `cancel` asks it to stop.
#[derive(Clone, Debug)]
pub struct UpdateJob {
    pub generation: u64,
    pub cancel: Arc<AtomicBool>,
}

/// The note on the update banner about the audio driver the new version brings, given the
/// loaded driver's revision.
pub fn update_driver_note(info: &UpdateInfo, loaded: Option<u32>) -> Option<&'static str> {
    let loaded = loaded?;
    if loaded < info.min_driver_version {
        Some("Requires an audio driver update after restarting (asks for your password).")
    } else if loaded < info.driver_version {
        Some("An audio driver update may be offered separately after restarting.")
    } else {
        None
    }
}

/// What the install mode says about updating, for captions.
pub fn install_mode_note(mode: &InstallMode) -> Option<&'static str> {
    match mode {
        InstallMode::SelfUpdatable { .. } => None,
        InstallMode::NotifyOnly(NotifyReason::NotInApplications) => {
            Some("Move Oto to Applications for automatic updates.")
        }
        InstallMode::NotifyOnly(_) => Some("Download the installer to update Oto."),
        InstallMode::Unsupported(UnsupportedReason::DevBuild) => {
            Some("This development build doesn\u{2019}t update itself.")
        }
    }
}

/// Only GitHub pages of the project open from the app.
pub fn openable_url(url: &str) -> bool {
    url.starts_with("https://github.com/deepso7/oto/")
        && !url.contains(char::is_whitespace)
        && !url.contains(['"', '\\', '`'])
}

/// An automatic check is due: they're on, this copy can update (or at least be told about
/// one), and the last check finished a day or more ago.
pub fn update_check_due(prefs: &Prefs, mode: &InstallMode, now: u64) -> bool {
    prefs.check_for_updates
        && !matches!(mode, InstallMode::Unsupported(_))
        && prefs
            .last_update_check
            .is_none_or(|t| now.saturating_sub(t) >= UPDATE_INTERVAL_SECS || t > now)
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
#[derive(Clone, Debug)]
pub enum Request {
    Cmd(Command),
    /// Make Oto Microphone the system default input (`true`) or put the previous one back.
    SystemMic(bool),
    /// An update is about to swap the app: unless a call is active, stop the engine, restore
    /// the default input and end the host thread (releasing `engine.lock`). On `Err` the engine
    /// keeps (or goes back to) running.
    CommitShutdown(SyncSender<Result<(), String>>),
    /// The update this candidate belongs to is durable: leave inert mode.
    CommitCandidate(SyncSender<Result<(), String>>),
}

impl PartialEq for Request {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Request::Cmd(a), Request::Cmd(b)) => a == b,
            (Request::SystemMic(a), Request::SystemMic(b)) => a == b,
            (Request::CommitShutdown(_), Request::CommitShutdown(_))
            | (Request::CommitCandidate(_), Request::CommitCandidate(_)) => true,
            _ => false,
        }
    }
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
    /// Checks for an update off the UI thread; the result comes back through
    /// [`State::update_checked`].
    pub check_update: fn(UpdateJob),
    /// Downloads and verifies an update off the UI thread; progress comes back through
    /// [`State::update_progress`] and [`State::update_verifying`], the result through
    /// [`State::update_downloaded`].
    pub download_update: fn(UpdateJob, UpdateInfo),
    /// Stages the update next to the app at `PathBuf`, asks the engine to shut down through
    /// [`State::commit_shutdown`] and hands over to the helper; quits on success, else reports
    /// through [`State::update_apply_failed`].
    pub apply_update: fn(UpdateJob, StagedUpdate, PathBuf),
    /// Opens a validated `https://github.com/` page in the browser.
    pub open_url: fn(&str) -> Result<(), String>,
    /// Seconds since the Unix epoch.
    pub now: fn() -> u64,
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
        check_update: |_| {},
        download_update: |_, _| {},
        apply_update: |_, _, _| {},
        open_url: |_| Ok(()),
        now: || 0,
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
    /// The loaded driver, as last known for certain (an `Unknown` probe keeps the last).
    pub driver_status: DriverStatus,
    /// The revision of the driver on disk; `None` when there is none.
    pub on_disk_version: Option<u32>,
    pub update: UpdateState,
    /// How this copy can update itself; classified off the UI thread once the engine started.
    pub install_mode: InstallMode,
    /// The current update operation's generation; results of others are dropped.
    update_gen: u64,
    /// The current update operation's cancel flag.
    update_cancel: Option<Arc<AtomicBool>>,
    /// This process is an update's new build, waiting for the update to become durable: no
    /// side effects (settings, login item, driver, microphone, system microphone, the mic
    /// test, onboarding) until [`State::commit_candidate`].
    pub candidate_pending: bool,
    /// A save was skipped while `candidate_pending`; done at commit.
    save_deferred: bool,
    /// The engine was told to shut down (quit, or an update's [`Request::CommitShutdown`]).
    shutdown_sent: bool,
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
            driver_status: if driver_installed {
                DriverStatus::Loaded(BUNDLED_DRIVER_VERSION)
            } else {
                DriverStatus::Absent
            },
            on_disk_version: driver_installed.then_some(BUNDLED_DRIVER_VERSION),
            update: UpdateState::Idle,
            install_mode: InstallMode::Unsupported(UnsupportedReason::DevBuild),
            update_gen: 0,
            update_cancel: None,
            candidate_pending: false,
            save_deferred: false,
            shutdown_sent: false,
        }
    }

    /// Setup shows until the user finishes it, and again whenever the driver is missing (unless
    /// they said "not now").
    pub fn onboarding(&self) -> bool {
        self.driver_known
            && !self.candidate_pending
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
        if self.candidate_pending {
            self.save_deferred = true;
            return;
        }
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
        if self.candidate_pending {
            return;
        }
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
        if self.candidate_pending {
            return;
        }
        // A driver known to be too old is refused by the engine: don't ask (and fail) until
        // it is updated.
        let want = self.prefs.system_microphone
            && self.driver_installed
            && (!self.driver_status.is_loaded() || self.driver_status.is_compatible())
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
            // Its own urgent banner (`DriverUiState::Incompatible`), not a dismissible error.
            // A probe that cannot tell yet (or is behind) must not hide it: the engine saw it.
            Event::DriverIncompatible { loaded, .. } => {
                if !self.install.busy() {
                    self.driver_status = DriverStatus::Loaded(loaded);
                    self.driver_installed = true;
                    self.driver_known = true;
                    // A loaded driver came from disk; the next probe corrects this.
                    if self.on_disk_version.is_none() {
                        self.on_disk_version = Some(loaded);
                        self.driver_on_disk = true;
                    }
                }
                // Not missing after all.
                if self
                    .error
                    .as_deref()
                    .is_some_and(|e| e.starts_with("Oto driver is not installed"))
                {
                    self.clear_engine_error();
                }
                self.sync_system_mic();
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

    /// The host thread restarted its engine (a failed restore while an update shut it down):
    /// the new engine needs the system microphone asked for again.
    pub fn engine_restarted(&mut self) {
        self.engine = EngineState::Running;
        self.system_mic_applied = false;
        self.shutdown_sent = false;
        self.sync_system_mic();
    }

    /// How this copy of Oto can update, once the engine knows whether it holds `engine.lock`.
    pub fn set_install_mode(&mut self, mode: InstallMode) {
        self.install_mode = mode;
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
    /// Idempotent: after an update's [`Request::CommitShutdown`] (or a first quit) nothing more
    /// is sent.
    pub fn shutdown(&mut self) {
        self.stop_hear_test();
        if !self.shutdown_sent {
            self.shutdown_sent = true;
            self.command(Command::Shutdown);
        }
    }

    // ----- "Hear the Difference" ---------------------------------------------------------------

    /// Why the test can't start now, if it can't. `None` means it can.
    pub fn hear_blocked(&self) -> Option<&'static str> {
        if self.candidate_pending {
            Some(FINISHING_UPDATE)
        } else if self.clients_active {
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
        self.mic = p.mic;
        // Core Audio couldn't say: keep what was last known for certain, and don't decide
        // anything (setup included) on it.
        if p.driver_status == DriverStatus::Unknown {
            return;
        }
        let first = !self.driver_known;
        self.driver_known = true;
        // While the installer runs its own result is the truth; it reports when it ends.
        if !self.install.busy() {
            self.take_driver(&p);
        }
        // Someone who already has the driver and the permission has nothing to set up.
        if first && p.is_loaded() && p.mic == Mic::Authorized && !self.prefs.onboarded {
            self.prefs.onboarded = true;
            self.save();
        }
        self.sync_system_mic();
    }

    /// The step-two button: ask macOS, or, once it has said no, open System Settings (macOS
    /// asks only once).
    pub fn allow_mic(&mut self) {
        if self.candidate_pending {
            return;
        }
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

    /// The driver's place in the state table.
    pub fn driver_ui(&self) -> DriverUiState {
        DriverUiState::of(
            self.driver_status,
            self.on_disk_version,
            BUNDLED_DRIVER_VERSION,
            MIN_COMPATIBLE_DRIVER_VERSION,
        )
    }

    /// The loaded driver's revision, if one is loaded.
    pub fn loaded_driver(&self) -> Option<u32> {
        match self.driver_status {
            DriverStatus::Loaded(r) => Some(r),
            _ => None,
        }
    }

    fn take_driver(&mut self, p: &Probe) {
        if p.driver_status != DriverStatus::Unknown {
            self.driver_status = p.driver_status;
            self.driver_installed = p.is_loaded();
        }
        self.driver_on_disk = p.is_on_disk();
        self.on_disk_version = p.on_disk_version;
    }

    /// Why the driver can't be installed, updated or removed right now. Serialized with an app
    /// update, and never during a call (the install restarts Core Audio).
    pub fn install_blocked(&self) -> Option<&'static str> {
        if self.candidate_pending {
            Some(FINISHING_UPDATE)
        } else if self.clients_active {
            Some(UPDATE_IN_CALL)
        } else if matches!(self.update, UpdateState::Applying) {
            Some("Oto is updating.")
        } else {
            None
        }
    }

    /// True if a driver newer than this build's bundled revision is either loaded in Core Audio
    /// or already on disk: never offer or allow replacing it with our older bundled driver.
    pub fn newer_driver_present(&self) -> bool {
        matches!(self.driver_status, DriverStatus::Loaded(r) if r > BUNDLED_DRIVER_VERSION)
            || self
                .on_disk_version
                .is_some_and(|v| v > BUNDLED_DRIVER_VERSION)
    }

    pub fn start_install(&mut self) {
        if !self.install.busy() && self.install_blocked().is_none() && !self.newer_driver_present()
        {
            self.install = Install::Running;
            (self.hooks.install)();
        }
    }

    /// The install finished; `p` is what the system reports now.
    pub fn installed(&mut self, outcome: Outcome, p: Probe) {
        self.take_driver(&p);
        self.mic = p.mic;
        self.install = match outcome {
            Outcome::Installed if !p.is_loaded() => Install::Failed(
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
        if !self.install.busy() && self.install_blocked().is_none() {
            self.install = Install::Removing;
            // Stay on the settings page rather than jumping to setup when the driver goes.
            self.skipped_onboarding = true;
            (self.hooks.uninstall)();
        }
    }

    /// The uninstall finished; `p` is what the system reports now.
    pub fn uninstalled(&mut self, outcome: UninstallOutcome, p: Probe) {
        self.take_driver(&p);
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

    // ----- the update candidate ----------------------------------------------------------------

    /// The update this process is the new build of is durable: the engine leaves inert mode,
    /// skipped side effects happen now, and the UI (onboarding included) works normally.
    pub fn commit_candidate(&mut self) {
        if !self.candidate_pending {
            return;
        }
        self.candidate_pending = false;
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        self.send(Request::CommitCandidate(tx));
        if std::mem::take(&mut self.save_deferred) {
            self.save();
        }
        self.sync_system_mic();
    }

    // ----- app updates -------------------------------------------------------------------------

    /// Ends the current update operation, if any, and starts generation `gen + 1`.
    fn next_update_job(&mut self) -> UpdateJob {
        if let Some(c) = self.update_cancel.take() {
            c.store(true, Ordering::Release);
        }
        self.update_gen += 1;
        let cancel = Arc::new(AtomicBool::new(false));
        self.update_cancel = Some(cancel.clone());
        UpdateJob {
            generation: self.update_gen,
            cancel,
        }
    }

    fn is_update_job(&self, generation: u64) -> bool {
        generation == self.update_gen && self.update_cancel.is_some()
    }

    fn finish_update_job(&mut self) {
        self.update_cancel = None;
    }

    fn set_update(&mut self, u: UpdateState) {
        self.update = u;
        let label = self
            .update
            .info()
            .map(|i| format!("Update to v{}\u{2026}", i.version));
        if let Some(t) = &self.tray {
            t.set_update_label(label.as_deref());
        }
    }

    /// The current update operation's generation (tests play the hooks' part with it).
    #[cfg(test)]
    pub fn update_generation(&self) -> u64 {
        self.update_gen
    }

    /// Checks for an update. A manual check reports every result and offers a skipped version
    /// again; a background one only ever shows an offer, and never opens the popover.
    pub fn check_for_updates(&mut self, manual: bool) {
        if self.candidate_pending {
            return;
        }
        match self.update {
            UpdateState::Checking { manual: m } => {
                // Already checking: a manual request makes it report everything.
                self.update = UpdateState::Checking {
                    manual: m || manual,
                };
                return;
            }
            UpdateState::Downloading { .. }
            | UpdateState::Verifying(_)
            | UpdateState::Ready(_)
            | UpdateState::Applying => return,
            // An offer on screen stays unless the user asks again.
            UpdateState::Available(_) | UpdateState::UnrecognizedKey { .. } if !manual => return,
            _ => {}
        }
        let job = self.next_update_job();
        self.set_update(UpdateState::Checking { manual });
        (self.hooks.check_update)(job);
    }

    /// The background schedule's tick: checks if one is due.
    pub fn maybe_check_for_updates(&mut self) {
        let now = (self.hooks.now)();
        if update_check_due(&self.prefs, &self.install_mode, now)
            && matches!(
                self.update,
                UpdateState::Idle | UpdateState::UpToDate | UpdateState::Failed(_)
            )
        {
            self.check_for_updates(false);
        }
    }

    /// Check `gen` finished.
    pub fn update_checked(&mut self, generation: u64, r: Result<CheckOutcome, String>) {
        if !self.is_update_job(generation) {
            return;
        }
        let UpdateState::Checking { manual } = self.update else {
            return;
        };
        self.finish_update_job();
        if r.is_ok() {
            self.prefs.last_update_check = Some((self.hooks.now)());
            self.save();
        }
        let next = match r {
            Ok(CheckOutcome::UpToDate) => UpdateState::UpToDate,
            Ok(CheckOutcome::Available(info)) => {
                if !manual && self.prefs.skipped_version.as_deref() == Some(info.version.as_str()) {
                    UpdateState::Idle
                } else {
                    UpdateState::Available(info)
                }
            }
            Ok(CheckOutcome::UnrecognizedKey { releases_url }) => {
                UpdateState::UnrecognizedKey { releases_url }
            }
            Ok(CheckOutcome::IncompatibleOs { version, min_os }) if manual => {
                UpdateState::IncompatibleOs { version, min_os }
            }
            Ok(CheckOutcome::IncompatibleOs { .. }) => UpdateState::Idle,
            Err(e) if manual => UpdateState::Failed(e),
            // Offline, a captive portal, a server hiccup: the next check will do.
            Err(_) => UpdateState::Idle,
        };
        self.set_update(next);
    }

    /// The offer's primary button: download and verify (when this copy can update itself), or
    /// open the release page.
    pub fn start_update(&mut self) {
        if self.candidate_pending {
            return;
        }
        match self.update.clone() {
            UpdateState::Available(info) => {
                if matches!(self.install_mode, InstallMode::SelfUpdatable { .. }) {
                    let job = self.next_update_job();
                    self.set_update(UpdateState::Downloading {
                        info: info.clone(),
                        pct: 0,
                    });
                    (self.hooks.download_update)(job, info);
                } else {
                    self.open_url(&info.notes_url);
                }
            }
            UpdateState::UnrecognizedKey { .. } => self.open_url(RELEASES_URL),
            _ => {}
        }
    }

    /// Opens a project page on GitHub; anything else is refused.
    pub fn open_url(&mut self, url: &str) {
        let url = if openable_url(url) { url } else { RELEASES_URL };
        if let Err(e) = (self.hooks.open_url)(url) {
            self.fail(e);
        }
    }

    /// The release notes of the version on offer.
    pub fn open_release_notes(&mut self) {
        if let Some(url) = self.update.info().map(|i| i.notes_url.clone()) {
            self.open_url(&url);
        }
    }

    pub fn update_progress(&mut self, generation: u64, pct: u8) {
        if !self.is_update_job(generation) {
            return;
        }
        if let UpdateState::Downloading { pct: p, .. } = &mut self.update {
            *p = pct.min(100);
        }
    }

    /// The download of `gen` is complete; its checks run now.
    pub fn update_verifying(&mut self, generation: u64) {
        if !self.is_update_job(generation) {
            return;
        }
        if let UpdateState::Downloading { info, .. } = &self.update {
            let info = info.clone();
            self.set_update(UpdateState::Verifying(info));
        }
    }

    /// Download `gen` ended. A cancelled one goes back to the offer.
    pub fn update_downloaded(&mut self, generation: u64, r: Result<StagedUpdate, String>) {
        if !self.is_update_job(generation) || !self.update.downloading() {
            return;
        }
        self.finish_update_job();
        let next = match r {
            Ok(staged) => UpdateState::Ready(staged),
            Err(e) => UpdateState::Failed(e),
        };
        self.set_update(next);
    }

    /// Stops a download (back to the offer) or a check (back to idle).
    pub fn cancel_update(&mut self) {
        let back = match &self.update {
            UpdateState::Downloading { info, .. } | UpdateState::Verifying(info) => {
                UpdateState::Available(info.clone())
            }
            UpdateState::Checking { .. } => UpdateState::Idle,
            _ => return,
        };
        if let Some(c) = self.update_cancel.take() {
            c.store(true, Ordering::Release);
        }
        // Whatever the cancelled job still reports is stale.
        self.update_gen += 1;
        self.set_update(back);
    }

    /// "Skip This Version": background checks won't offer it again.
    pub fn skip_update(&mut self) {
        if let UpdateState::Available(info) = &self.update {
            self.prefs.skipped_version = Some(info.version.clone());
            self.save();
            self.set_update(UpdateState::Idle);
        }
    }

    /// Hides the banner until the next check.
    pub fn dismiss_update(&mut self) {
        if matches!(
            self.update,
            UpdateState::Available(_)
                | UpdateState::UnrecognizedKey { .. }
                | UpdateState::IncompatibleOs { .. }
                | UpdateState::Failed(_)
                | UpdateState::UpToDate
        ) {
            self.set_update(UpdateState::Idle);
        }
    }

    pub fn set_check_for_updates(&mut self, on: bool) {
        if self.prefs.check_for_updates != on {
            self.prefs.check_for_updates = on;
            self.save();
        }
    }

    /// Why **Restart to Update** can't run now, if it can't.
    pub fn apply_blocked(&self) -> Option<&'static str> {
        if self.candidate_pending {
            Some(FINISHING_UPDATE)
        } else if self.clients_active {
            Some(UPDATE_IN_CALL)
        } else if self.install.busy() {
            Some("Wait for the audio driver to finish installing.")
        } else if !matches!(self.install_mode, InstallMode::SelfUpdatable { .. }) {
            install_mode_note(&self.install_mode)
        } else {
            None
        }
    }

    /// **Restart to Update**: stages the verified update and hands over to the helper.
    pub fn restart_to_update(&mut self) {
        if self.apply_blocked().is_some() {
            return;
        }
        let InstallMode::SelfUpdatable { bundle } = self.install_mode.clone() else {
            return;
        };
        let UpdateState::Ready(staged) = self.update.clone() else {
            return;
        };
        // The mic test must not hold the mic or the speakers across the restart.
        self.stop_hear_test();
        let job = self.next_update_job();
        self.set_update(UpdateState::Applying);
        (self.hooks.apply_update)(job, staged, bundle);
    }

    /// The apply worker staged the update: ask the engine to shut down. A stale or cancelled
    /// job's request is dropped, which the worker reads as a refusal.
    pub fn commit_shutdown(&mut self, generation: u64, ack: SyncSender<Result<(), String>>) {
        if !self.is_update_job(generation) || self.update != UpdateState::Applying {
            return;
        }
        self.stop_hear_test();
        self.shutdown_sent = true;
        if !self.link.send(Request::CommitShutdown(ack)) {
            self.shutdown_sent = false;
        }
    }

    /// The update couldn't go ahead. `restart_engine` when the engine was already shut down
    /// (the helper died before the commit byte): start a new one and keep running.
    pub fn update_apply_failed(&mut self, generation: u64, msg: String, restart_engine: bool) {
        if !self.is_update_job(generation) {
            return;
        }
        self.finish_update_job();
        self.shutdown_sent = false;
        if restart_engine {
            self.start_engine();
        }
        self.set_update(UpdateState::Failed(msg));
    }

    /// The helper has the commit byte: the engine is down and the app quits now.
    pub fn update_committed(&mut self) {
        self.finish_update_job();
        self.shutdown_sent = true;
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
                _ => None,
            })
            .collect()
    }

    fn system_mic(s: &State) -> Vec<bool> {
        requests(s)
            .iter()
            .filter_map(|r| match r {
                Request::SystemMic(on) => Some(*on),
                _ => None,
            })
            .collect()
    }

    fn probe(driver: bool) -> Probe {
        if driver {
            Probe::new(
                DriverStatus::Loaded(BUNDLED_DRIVER_VERSION),
                Some(BUNDLED_DRIVER_VERSION),
                Mic::Authorized,
            )
        } else {
            Probe::new(DriverStatus::Absent, None, Mic::Authorized)
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
    fn an_incompatible_driver_shows_its_banner_not_an_error() {
        let mut s = State::new(Prefs::default(), false);
        s.link = Link::Record(Vec::new());
        s.prefs.system_microphone = true;
        s.on_event(Event::Error(
            "Oto driver is not installed: Oto Microphone is not available".into(),
        ));
        s.on_event(Event::DriverIncompatible {
            loaded: 0,
            min_required: MIN_COMPATIBLE_DRIVER_VERSION,
        });
        assert_eq!(s.error, None, "no dismissible error banner");
        assert_eq!(s.driver_status, DriverStatus::Loaded(0));
        assert!(s.driver_installed && s.driver_known);
        assert_eq!(s.driver_ui(), DriverUiState::Incompatible);
        // The engine would refuse: Oto Microphone is not made the system input.
        assert!(!system_mic(&s).contains(&true), "{:?}", requests(&s));
        // Updated: the probe sees a compatible driver and the system input follows.
        s.probed(probe(true));
        assert_eq!(s.driver_ui(), DriverUiState::UpToDate);
        assert_eq!(system_mic(&s).last(), Some(&true));
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

    // ----- the driver state table ---------------------------------------------------------------

    #[test]
    fn the_driver_state_table() {
        use DriverStatus::{Absent, Loaded, Unknown};
        use DriverUiState as U;
        // bundled = 3, min_compat = 2
        let of = |l, d| DriverUiState::of(l, d, 3, 2);
        assert_eq!(of(Absent, None), U::NotInstalled);
        assert_eq!(of(Absent, Some(0)), U::InstalledNeedsReboot);
        assert_eq!(of(Absent, Some(3)), U::InstalledNeedsReboot);
        assert_eq!(of(Unknown, None), U::Unknown);
        assert_eq!(of(Unknown, Some(3)), U::Unknown);
        assert_eq!(of(Loaded(3), Some(3)), U::UpToDate);
        assert_eq!(of(Loaded(3), Some(4)), U::UpToDate);
        assert_eq!(of(Loaded(3), None), U::OnDiskOutdatedOrMissing);
        assert_eq!(of(Loaded(3), Some(2)), U::OnDiskOutdatedOrMissing);
        assert_eq!(of(Loaded(4), None), U::NewerLoaded);
        assert_eq!(of(Loaded(4), Some(1)), U::NewerLoaded);
        assert_eq!(of(Loaded(2), Some(3)), U::UpdateNeedsReboot);
        assert_eq!(of(Loaded(1), Some(5)), U::UpdateNeedsReboot);
        assert_eq!(of(Loaded(2), None), U::RemovedNeedsReboot);
        assert_eq!(of(Loaded(2), Some(2)), U::UpdateAvailable);
        assert_eq!(of(Loaded(2), Some(0)), U::UpdateAvailable);
        assert_eq!(of(Loaded(1), Some(1)), U::Incompatible);
        assert_eq!(of(Loaded(0), Some(0)), U::Incompatible);
        assert!(U::UpdateAvailable.offers_update() && U::Incompatible.offers_update());
        assert!(!U::UpToDate.offers_update() && !U::UpdateNeedsReboot.offers_update());
        // Legacy v0.0.1–v0.0.3 drivers (no 'otvr', so revision 0) are incompatible with this
        // build, not missing.
        assert_eq!(MIN_COMPATIBLE_DRIVER_VERSION, 1);
        assert_eq!(
            DriverUiState::of(
                Loaded(0),
                Some(0),
                BUNDLED_DRIVER_VERSION,
                MIN_COMPATIBLE_DRIVER_VERSION
            ),
            U::Incompatible
        );
        assert!(!Loaded(0).is_compatible() && Loaded(0).is_loaded());
    }

    #[test]
    fn an_unknown_driver_status_changes_nothing() {
        let mut s = State::new(Prefs::default(), false);
        s.driver_known = false;
        s.probed(Probe::new(DriverStatus::Unknown, None, Mic::Authorized));
        assert!(!s.driver_known, "nothing is decided on an unknown status");
        assert!(!s.onboarding());

        let mut s = state();
        s.probed(Probe::new(DriverStatus::Unknown, None, Mic::Authorized));
        assert!(s.driver_installed);
        assert_eq!(s.driver_ui(), DriverUiState::UpToDate);
        s.probed(Probe::new(
            DriverStatus::Loaded(BUNDLED_DRIVER_VERSION),
            None,
            Mic::Authorized,
        ));
        assert_eq!(s.driver_ui(), DriverUiState::OnDiskOutdatedOrMissing);
    }

    #[test]
    fn start_install_refuses_to_downgrade_a_newer_driver() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static INSTALLS: AtomicUsize = AtomicUsize::new(0);
        let cases = [
            // Newer loaded, missing on disk.
            (DriverStatus::Loaded(BUNDLED_DRIVER_VERSION + 1), None),
            // Bundled loaded, newer already on disk.
            (
                DriverStatus::Loaded(BUNDLED_DRIVER_VERSION),
                Some(BUNDLED_DRIVER_VERSION + 1),
            ),
            // Older loaded, newer already on disk (waiting for reboot).
            (
                DriverStatus::Loaded(BUNDLED_DRIVER_VERSION.saturating_sub(1)),
                Some(BUNDLED_DRIVER_VERSION + 1),
            ),
            // Absent in Core Audio, newer already on disk.
            (DriverStatus::Absent, Some(BUNDLED_DRIVER_VERSION + 1)),
        ];
        for (loaded, on_disk) in cases {
            let mut s = state();
            s.hooks.install = || {
                INSTALLS.fetch_add(1, Ordering::Relaxed);
            };
            s.probed(Probe::new(loaded, on_disk, Mic::Authorized));
            assert!(s.newer_driver_present(), "{loaded:?} / {on_disk:?}");
            s.start_install();
            assert_eq!(s.install, Install::Idle, "{loaded:?} / {on_disk:?}");
        }
        assert_eq!(INSTALLS.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_cancelled_password_prompt_is_not_an_error() {
        let mut s = state();
        s.start_install();
        assert!(s.install.busy());
        s.installed(Outcome::Cancelled, probe(true));
        assert_eq!(s.install, Install::Idle);
        assert_eq!(s.error, None);
        s.start_uninstall();
        s.uninstalled(UninstallOutcome::Cancelled, probe(true));
        assert_eq!(s.install, Install::Idle);
        assert_eq!(s.error, None);
    }

    // ----- updates -------------------------------------------------------------------------------

    fn info(version: &str) -> UpdateInfo {
        UpdateInfo {
            version: version.into(),
            notes_url: format!("https://github.com/deepso7/oto/releases/tag/v{version}"),
            asset_url: format!(
                "https://github.com/deepso7/oto/releases/download/v{version}/Oto-{version}-macos.zip"
            ),
            sha256: "0".repeat(64),
            size: 1,
            min_os: "13.0".into(),
            driver_version: BUNDLED_DRIVER_VERSION,
            min_driver_version: MIN_COMPATIBLE_DRIVER_VERSION,
            published_at: String::new(),
        }
    }

    fn staged(version: &str) -> StagedUpdate {
        StagedUpdate::unowned(info(version), "/tmp/c/Oto.app".into(), "/tmp/c".into())
    }

    fn updatable() -> State {
        let mut s = state();
        s.install_mode = InstallMode::SelfUpdatable {
            bundle: "/Applications/Oto.app".into(),
        };
        s.hooks.now = || 1_000_000;
        s
    }

    fn shutdowns(s: &State) -> usize {
        sent(s).iter().filter(|c| **c == Command::Shutdown).count()
    }

    #[test]
    fn an_update_from_check_to_restart() {
        static CHECKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static DOWNLOADS: std::sync::Mutex<Vec<UpdateInfo>> = std::sync::Mutex::new(Vec::new());
        static APPLIED: std::sync::Mutex<Vec<(StagedUpdate, PathBuf)>> =
            std::sync::Mutex::new(Vec::new());
        let mut s = updatable();
        s.hooks.check_update = |_| {
            CHECKS.fetch_add(1, Ordering::Relaxed);
        };
        s.hooks.download_update = |_, i| DOWNLOADS.lock().unwrap().push(i);
        s.hooks.apply_update = |_, st, b| APPLIED.lock().unwrap().push((st, b));

        s.check_for_updates(true);
        assert_eq!(s.update, UpdateState::Checking { manual: true });
        // A second request while checking doesn't start another check.
        s.check_for_updates(false);
        assert_eq!(CHECKS.load(Ordering::Relaxed), 1);
        assert_eq!(s.update, UpdateState::Checking { manual: true });
        let g = s.update_generation();
        s.update_checked(g, Ok(CheckOutcome::Available(info("9.0.0"))));
        assert_eq!(s.update, UpdateState::Available(info("9.0.0")));
        assert_eq!(s.prefs.last_update_check, Some(1_000_000));

        s.start_update();
        assert_eq!(DOWNLOADS.lock().unwrap().as_slice(), [info("9.0.0")]);
        let g = s.update_generation();
        s.update_progress(g, 42);
        assert!(matches!(s.update, UpdateState::Downloading { pct: 42, .. }));
        s.update_verifying(g);
        assert_eq!(s.update, UpdateState::Verifying(info("9.0.0")));
        s.update_downloaded(g, Ok(staged("9.0.0")));
        assert_eq!(s.update, UpdateState::Ready(staged("9.0.0")));

        s.restart_to_update();
        assert_eq!(s.update, UpdateState::Applying);
        let applied = APPLIED.lock().unwrap().pop().unwrap();
        assert_eq!(applied.1, PathBuf::from("/Applications/Oto.app"));
        // The worker staged it: the engine is asked to shut down, once.
        let g = s.update_generation();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        s.commit_shutdown(g, tx);
        assert!(matches!(
            requests(&s).last(),
            Some(Request::CommitShutdown(_))
        ));
        s.update_committed();
        // Quitting afterwards doesn't send a second shutdown.
        s.shutdown();
        assert_eq!(shutdowns(&s), 0);
    }

    #[test]
    fn stale_worker_results_are_dropped() {
        let mut s = updatable();
        s.check_for_updates(true);
        let old = s.update_generation();
        s.cancel_update();
        assert_eq!(s.update, UpdateState::Idle);
        s.update_checked(old, Ok(CheckOutcome::Available(info("9.0.0"))));
        assert_eq!(s.update, UpdateState::Idle, "a cancelled check's result");

        s.check_for_updates(true);
        let g = s.update_generation();
        s.update_checked(g, Ok(CheckOutcome::Available(info("9.0.0"))));
        s.start_update();
        let first = s.update_generation();
        s.cancel_update();
        assert_eq!(s.update, UpdateState::Available(info("9.0.0")));
        s.update_progress(first, 80);
        s.update_verifying(first);
        s.update_downloaded(first, Ok(staged("9.0.0")));
        assert_eq!(s.update, UpdateState::Available(info("9.0.0")));

        // A new download replaces the old one, whose cancel flag is set.
        s.start_update();
        let second = s.update_generation();
        assert_ne!(first, second);
        s.update_downloaded(first, Err("late".into()));
        assert!(matches!(s.update, UpdateState::Downloading { pct: 0, .. }));
        s.update_downloaded(second, Err("disk full".into()));
        assert_eq!(s.update, UpdateState::Failed("disk full".into()));
        // A result for the finished job is stale too.
        s.update_downloaded(second, Ok(staged("9.0.0")));
        assert_eq!(s.update, UpdateState::Failed("disk full".into()));
    }

    #[test]
    fn cancelling_sets_the_workers_flag() {
        static JOB: std::sync::Mutex<Option<UpdateJob>> = std::sync::Mutex::new(None);
        let mut s = updatable();
        s.hooks.check_update = |j| *JOB.lock().unwrap() = Some(j);
        s.check_for_updates(true);
        let job = JOB.lock().unwrap().take().unwrap();
        assert!(!job.cancel.load(Ordering::Relaxed));
        s.cancel_update();
        assert!(job.cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn background_checks_stay_quiet() {
        let mut s = updatable();
        for r in [
            Err("offline".to_string()),
            Ok(CheckOutcome::IncompatibleOs {
                version: "9.0.0".into(),
                min_os: "99.0".into(),
            }),
        ] {
            s.check_for_updates(false);
            let g = s.update_generation();
            s.update_checked(g, r);
            assert_eq!(s.update, UpdateState::Idle);
            assert_eq!(s.error, None);
        }
        // A failed check doesn't count as the day's check.
        let mut s = updatable();
        s.check_for_updates(false);
        let g = s.update_generation();
        s.update_checked(g, Err("offline".into()));
        assert_eq!(s.prefs.last_update_check, None);

        // Manual checks say what happened.
        s.check_for_updates(true);
        let g = s.update_generation();
        s.update_checked(g, Err("offline".into()));
        assert_eq!(s.update, UpdateState::Failed("offline".into()));
        s.check_for_updates(true);
        let g = s.update_generation();
        s.update_checked(
            g,
            Ok(CheckOutcome::IncompatibleOs {
                version: "9.0.0".into(),
                min_os: "99.0".into(),
            }),
        );
        assert!(matches!(s.update, UpdateState::IncompatibleOs { .. }));
        s.check_for_updates(true);
        let g = s.update_generation();
        s.update_checked(g, Ok(CheckOutcome::UpToDate));
        assert_eq!(s.update, UpdateState::UpToDate);
    }

    #[test]
    fn a_skipped_version_is_offered_only_on_request() {
        let mut s = updatable();
        s.check_for_updates(true);
        let g = s.update_generation();
        s.update_checked(g, Ok(CheckOutcome::Available(info("9.0.0"))));
        s.skip_update();
        assert_eq!(s.prefs.skipped_version.as_deref(), Some("9.0.0"));
        assert_eq!(s.update, UpdateState::Idle);

        s.check_for_updates(false);
        let g = s.update_generation();
        s.update_checked(g, Ok(CheckOutcome::Available(info("9.0.0"))));
        assert_eq!(s.update, UpdateState::Idle, "skipped");

        s.check_for_updates(false);
        let g = s.update_generation();
        s.update_checked(g, Ok(CheckOutcome::Available(info("9.1.0"))));
        assert_eq!(s.update, UpdateState::Available(info("9.1.0")), "newer");

        s.check_for_updates(true);
        let g = s.update_generation();
        s.update_checked(g, Ok(CheckOutcome::Available(info("9.0.0"))));
        assert_eq!(
            s.update,
            UpdateState::Available(info("9.0.0")),
            "a manual check overrides the skip"
        );
        s.dismiss_update();
        assert_eq!(s.update, UpdateState::Idle);
        assert_eq!(s.prefs.skipped_version.as_deref(), Some("9.0.0"));
    }

    #[test]
    fn automatic_checks_run_once_a_day_when_allowed() {
        let mode = InstallMode::SelfUpdatable {
            bundle: "/Applications/Oto.app".into(),
        };
        let day = UPDATE_INTERVAL_SECS;
        let mut p = Prefs::default();
        assert!(update_check_due(&p, &mode, 10 * day));
        p.last_update_check = Some(10 * day);
        assert!(!update_check_due(&p, &mode, 10 * day + day - 1));
        assert!(update_check_due(&p, &mode, 11 * day));
        // A clock set back doesn't silence checks for good.
        assert!(update_check_due(&p, &mode, 5 * day));
        assert!(update_check_due(
            &p,
            &InstallMode::NotifyOnly(NotifyReason::NotInApplications),
            11 * day
        ));
        assert!(!update_check_due(
            &p,
            &InstallMode::Unsupported(UnsupportedReason::DevBuild),
            11 * day
        ));
        p.check_for_updates = false;
        assert!(!update_check_due(&p, &mode, 11 * day));

        static CHECKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mut s = updatable();
        s.hooks.check_update = |_| {
            CHECKS.fetch_add(1, Ordering::Relaxed);
        };
        s.maybe_check_for_updates();
        assert_eq!(CHECKS.load(Ordering::Relaxed), 1);
        let g = s.update_generation();
        s.update_checked(g, Ok(CheckOutcome::UpToDate));
        s.maybe_check_for_updates();
        assert_eq!(CHECKS.load(Ordering::Relaxed), 1, "checked today");
        s.set_check_for_updates(false);
        s.prefs.last_update_check = None;
        s.maybe_check_for_updates();
        assert_eq!(CHECKS.load(Ordering::Relaxed), 1, "turned off");
    }

    #[test]
    fn notify_only_copies_open_the_release_page() {
        static OPENED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        static DOWNLOADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mut s = updatable();
        s.install_mode = InstallMode::NotifyOnly(NotifyReason::NotInApplications);
        s.hooks.open_url = |u| {
            OPENED.lock().unwrap().push(u.to_owned());
            Ok(())
        };
        s.hooks.download_update = |_, _| {
            DOWNLOADS.fetch_add(1, Ordering::Relaxed);
        };
        s.update = UpdateState::Available(info("9.0.0"));
        s.start_update();
        assert_eq!(DOWNLOADS.load(Ordering::Relaxed), 0);
        assert_eq!(s.update, UpdateState::Available(info("9.0.0")));
        s.restart_to_update();
        assert_eq!(s.update, UpdateState::Available(info("9.0.0")));

        s.update = UpdateState::UnrecognizedKey {
            releases_url: "https://evil.example/".into(),
        };
        s.start_update();
        // Anything but the project's GitHub pages falls back to the fixed Releases page.
        s.open_url("https://evil.example/x");
        assert_eq!(
            OPENED.lock().unwrap().as_slice(),
            [
                "https://github.com/deepso7/oto/releases/tag/v9.0.0",
                RELEASES_URL,
                RELEASES_URL
            ]
        );
        assert!(openable_url(RELEASES_URL));
        assert!(!openable_url("http://github.com/deepso7/oto/releases"));
        assert!(!openable_url("https://github.com/deepso7/oto/releases x"));
        assert!(!openable_url("https://github.com.evil/deepso7/oto/"));
        assert_eq!(
            install_mode_note(&InstallMode::NotifyOnly(NotifyReason::NotInApplications)),
            Some("Move Oto to Applications for automatic updates.")
        );
        assert_eq!(
            install_mode_note(&InstallMode::NotifyOnly(
                NotifyReason::RequiresManualInstall
            )),
            Some("Download the installer to update Oto.")
        );
    }

    #[test]
    fn a_call_blocks_the_restart_and_driver_changes() {
        static APPLIES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static INSTALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mut s = updatable();
        s.hooks.apply_update = |_, _, _| {
            APPLIES.fetch_add(1, Ordering::Relaxed);
        };
        s.hooks.install = || {
            INSTALLS.fetch_add(1, Ordering::Relaxed);
        };
        s.hooks.uninstall = || {
            INSTALLS.fetch_add(1, Ordering::Relaxed);
        };
        s.update = UpdateState::Ready(staged("9.0.0"));
        s.on_event(Event::ClientsActive(true));
        assert_eq!(s.apply_blocked(), Some(UPDATE_IN_CALL));
        assert_eq!(s.install_blocked(), Some(UPDATE_IN_CALL));
        s.restart_to_update();
        s.start_install();
        s.start_uninstall();
        assert_eq!(s.update, UpdateState::Ready(staged("9.0.0")));
        assert_eq!(s.install, Install::Idle);
        assert_eq!(APPLIES.load(Ordering::Relaxed), 0);
        assert_eq!(INSTALLS.load(Ordering::Relaxed), 0);
        s.on_event(Event::ClientsActive(false));

        // A driver install and an update never overlap.
        s.start_install();
        assert_eq!(INSTALLS.load(Ordering::Relaxed), 1);
        assert!(s.apply_blocked().is_some());
        s.restart_to_update();
        assert_eq!(APPLIES.load(Ordering::Relaxed), 0);
        s.installed(Outcome::Installed, probe(true));
        s.restart_to_update();
        assert_eq!(APPLIES.load(Ordering::Relaxed), 1);
        assert_eq!(s.update, UpdateState::Applying);
        assert!(s.install_blocked().is_some());
        s.start_install();
        s.start_uninstall();
        assert_eq!(INSTALLS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_refused_or_failed_hand_over_keeps_oto_running() {
        static STARTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mut s = updatable();
        s.hooks.start_engine = |_, _| {
            STARTS.fetch_add(1, Ordering::Relaxed);
            Link::Record(Vec::new())
        };
        s.update = UpdateState::Ready(staged("9.0.0"));
        s.restart_to_update();
        let g = s.update_generation();
        // The host refused (a call started): the engine kept running.
        s.update_apply_failed(g, UPDATE_IN_CALL.into(), false);
        assert_eq!(s.update, UpdateState::Failed(UPDATE_IN_CALL.into()));
        assert_eq!(STARTS.load(Ordering::Relaxed), 0);
        s.shutdown();
        assert_eq!(shutdowns(&s), 1, "quitting still stops the engine");

        // The helper died before the commit byte (EPIPE): a new engine starts.
        let mut s = updatable();
        s.hooks.start_engine = |_, _| {
            STARTS.fetch_add(1, Ordering::Relaxed);
            Link::Record(Vec::new())
        };
        s.set_system_mic(true);
        s.update = UpdateState::Ready(staged("9.0.0"));
        s.restart_to_update();
        let g = s.update_generation();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        s.commit_shutdown(g, tx);
        s.update_apply_failed(g, "The updater stopped unexpectedly.".into(), true);
        assert_eq!(STARTS.load(Ordering::Relaxed), 1);
        assert_eq!(s.engine, EngineState::Starting);
        assert!(matches!(s.update, UpdateState::Failed(_)));
        assert_eq!(
            requests(&s),
            [Request::SystemMic(true)],
            "the new engine is asked again"
        );
        s.engine_started();
        s.shutdown();
        assert_eq!(shutdowns(&s), 1);
    }

    #[test]
    fn a_stale_hand_over_never_reaches_the_engine() {
        let mut s = updatable();
        s.update = UpdateState::Ready(staged("9.0.0"));
        s.restart_to_update();
        let g = s.update_generation();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        s.commit_shutdown(g + 1, tx);
        assert!(requests(&s).is_empty());
        // The worker reads the dropped ack as a refusal.
        assert!(rx.recv().is_err());
    }

    #[test]
    fn shutting_down_is_idempotent() {
        let mut s = state();
        s.shutdown();
        s.shutdown();
        assert_eq!(shutdowns(&s), 1);
    }

    #[test]
    fn the_banner_says_when_a_driver_update_follows() {
        let mut i = info("9.0.0");
        i.driver_version = 5;
        i.min_driver_version = 4;
        assert_eq!(
            update_driver_note(&i, Some(3)),
            Some("Requires an audio driver update after restarting (asks for your password).")
        );
        assert_eq!(
            update_driver_note(&i, Some(4)),
            Some("An audio driver update may be offered separately after restarting.")
        );
        assert_eq!(update_driver_note(&i, Some(5)), None);
        assert_eq!(update_driver_note(&i, Some(6)), None);
        assert_eq!(update_driver_note(&i, None), None);
    }

    // ----- the update candidate -----------------------------------------------------------------

    #[test]
    fn a_candidate_does_nothing_until_the_update_commits() {
        static SAVES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static SIDE_EFFECTS: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        static HEARS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mut s = State::new(
            Prefs {
                system_microphone: true,
                ..Prefs::default()
            },
            true,
        );
        s.link = Link::Record(Vec::new());
        s.candidate_pending = true;
        s.hooks.save = |_| {
            SAVES.fetch_add(1, Ordering::Relaxed);
            Ok(())
        };
        s.hooks.set_login = |_| {
            SIDE_EFFECTS.fetch_add(1, Ordering::Relaxed);
            Ok(())
        };
        s.hooks.install = || {
            SIDE_EFFECTS.fetch_add(1, Ordering::Relaxed);
        };
        s.hooks.uninstall = || {
            SIDE_EFFECTS.fetch_add(1, Ordering::Relaxed);
        };
        s.hooks.request_mic = || {
            SIDE_EFFECTS.fetch_add(1, Ordering::Relaxed);
        };
        s.hooks.check_update = |_| {
            SIDE_EFFECTS.fetch_add(1, Ordering::Relaxed);
        };
        s.hooks.start_hear_test = |_| {
            HEARS.fetch_add(1, Ordering::Relaxed);
        };

        s.probed(probe(true));
        s.set_strength(Strength::Strong);
        s.set_start_at_login(true);
        s.start_install();
        s.start_uninstall();
        s.mic = Mic::NotDetermined;
        s.allow_mic();
        s.mic = Mic::Authorized;
        s.set_system_mic(false);
        s.set_system_mic(true);
        s.check_for_updates(true);
        assert_eq!(s.hear_blocked(), Some(FINISHING_UPDATE));
        s.start_hear_test();
        assert_eq!(s.hear, HearTest::Idle);
        assert!(!s.onboarding(), "no setup popup while finishing");

        assert_eq!(SAVES.load(Ordering::Relaxed), 0);
        assert_eq!(SIDE_EFFECTS.load(Ordering::Relaxed), 0);
        assert_eq!(HEARS.load(Ordering::Relaxed), 0);
        assert!(!s.prefs.start_at_login);
        assert_eq!(s.install, Install::Idle);
        assert!(system_mic(&s).is_empty(), "the default input is untouched");

        s.commit_candidate();
        assert!(!s.candidate_pending);
        assert!(
            requests(&s)
                .iter()
                .any(|r| matches!(r, Request::CommitCandidate(_)))
        );
        assert_eq!(SAVES.load(Ordering::Relaxed), 1, "the skipped save happens");
        assert_eq!(system_mic(&s), [true]);
        assert_eq!(s.hear_blocked(), None);
        s.start_hear_test();
        assert_eq!(HEARS.load(Ordering::Relaxed), 1);
        assert!(s.onboarding(), "setup resumes after the commit");
        // Committing twice is harmless.
        s.commit_candidate();
        assert_eq!(
            requests(&s)
                .iter()
                .filter(|r| matches!(r, Request::CommitCandidate(_)))
                .count(),
            1
        );
    }
}
