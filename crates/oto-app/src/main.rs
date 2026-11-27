//! Oto's menu-bar app. `cargo run -p oto-app` starts it in the menu bar (the window opens on
//! its own only when the driver still needs installing); a click on the status item opens the
//! popover. `cargo run -p oto-app -- --snap oto.png --scene main-on-with-levels` renders a
//! scene headless; `--list-scenes` lists them.

mod driver;
mod host;
mod login;
mod meter;
mod permission;
mod prefs;
mod state;
mod tray;
mod updater;
mod view;

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use oto_engine::test_capture::{self, TestPlayer, TestRecording};
use oto_engine::{
    BUNDLED_DRIVER_VERSION, CandidateToken, DriverStatus, Engine, EngineOptions, InputDevice,
    Nonce, Settings, Tier, UpdatePaths,
};
use rune::{App, AppHandle, AppSpec, Tx};

use crate::meter::Meter;
use crate::permission::Mic;
use crate::state::{
    EngineState, HEAR_SECONDS, HearJob, HearTest, Hooks, Install, Link, Open, Page, Probe, State,
    UpdateJob, UpdateState,
};
use crate::updater::apply::{self, CliMode, InstallMode, PreparedUpdate};
use crate::updater::verify::{self, StagedUpdate, UpdateInfo};

/// The running app's handle, for hooks that are plain `fn`s.
static HANDLE: OnceLock<AppHandle<State>> = OnceLock::new();
/// The current engine host thread, so quitting can wait for it to put things back.
static HOST: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);
thread_local! {
    /// "Hear the Difference" playback. Started and stopped only from state hooks, which run on
    /// the UI thread.
    static HEAR_PLAYER: RefCell<Option<TestPlayer>> = const { RefCell::new(None) };
}
/// A [`refresh`] is under way.
static REFRESHING: AtomicBool = AtomicBool::new(false);
/// This process is an update's new build (`--update-ack`) that hasn't committed yet: engines it
/// starts are the update's candidate. Taken at commit, so a later restart is an ordinary one.
static CANDIDATE: Mutex<Option<CandidateToken>> = Mutex::new(None);

/// This build's version, which an update must be newer than.
const VERSION: &str = env!("CARGO_PKG_VERSION");
/// When the first automatic update check runs after launch.
const FIRST_UPDATE_CHECK: Duration = Duration::from_secs(20);
/// How often the automatic check looks whether a day has passed since the last check.
const UPDATE_TICK: Duration = Duration::from_secs(60 * 60);
/// How often an update candidate looks for its commit.
const CANDIDATE_POLL: Duration = Duration::from_millis(500);
/// How long an update candidate waits for its engine before giving up (the helper's stage-2
/// deadline is 35 s after launch).
const CANDIDATE_START_WAIT: Duration = Duration::from_secs(25);

/// How long launch waits for the engine before showing the UI anyway. The check for another
/// running Oto is the first thing `Engine::start` does and answers at once; the rest of the
/// start can take as long as Core Audio does.
const START_WAIT: Duration = Duration::from_secs(2);
/// How long a new status item needs to take its place in the menu bar.
const TRAY_SETTLE: Duration = Duration::from_millis(400);

/// What headless snapshots start from: a Mac with the driver in, mic access and two mics.
fn demo() -> State {
    let mut s = State::new(
        prefs::Prefs {
            onboarded: true,
            ..prefs::Prefs::default()
        },
        true,
    );
    s.popover = cfg!(target_os = "macos");
    s.install_mode = InstallMode::SelfUpdatable {
        bundle: PathBuf::from("/Applications/Oto.app"),
    };
    s.devices = vec![
        InputDevice {
            id: "builtin".into(),
            name: "MacBook Pro Microphone".into(),
            is_default: true,
        },
        InputDevice {
            id: "usb".into(),
            name: "Shure MV7".into(),
            is_default: false,
        },
        InputDevice {
            id: "airpods".into(),
            name: "AirPods Pro".into(),
            is_default: false,
        },
    ];
    s
}

fn active(s: &mut State) {
    s.on_event(oto_engine::Event::ClientsActive(true));
    s.on_event(oto_engine::Event::Load(0.042));
    s.set_running(Some((oto_engine::model_for(Tier::Standard), 56.0)));
    s.meter.show(-14.0, -21.0);
}

/// A first launch: no driver, never asked for the microphone.
fn fresh(s: &mut State) {
    s.prefs.onboarded = false;
    s.driver_installed = false;
    s.driver_on_disk = false;
    s.mic = Mic::NotDetermined;
}

/// Setup with the driver in.
fn driver_in(s: &mut State) {
    fresh(s);
    s.driver_installed = true;
    s.driver_on_disk = true;
}

/// An offer for the snapshot scenes.
fn demo_update() -> UpdateInfo {
    UpdateInfo {
        version: "0.1.0".into(),
        notes_url: "https://github.com/deepso7/oto/releases/tag/v0.1.0".into(),
        asset_url: "https://github.com/deepso7/oto/releases/download/v0.1.0/Oto-0.1.0-macos.zip"
            .into(),
        sha256: "0".repeat(64),
        size: 48_000_000,
        min_os: "13.0".into(),
        driver_version: BUNDLED_DRIVER_VERSION,
        min_driver_version: oto_engine::MIN_COMPATIBLE_DRIVER_VERSION,
        published_at: "2026-11-27T00:00:00Z".into(),
    }
}

type Scene = (&'static str, fn(&mut State));

const SCENES: &[Scene] = &[
    ("onboarding", fresh),
    ("onboarding-installing", |s| {
        fresh(s);
        s.install = Install::Running;
    }),
    ("onboarding-failed", |s| {
        fresh(s);
        s.install = Install::Failed("This copy of Oto doesn't include the driver.".into());
    }),
    ("onboarding-restart", |s| {
        fresh(s);
        s.installed(
            driver::Outcome::InstalledNeedsReboot,
            Probe::new(
                DriverStatus::Absent,
                Some(BUNDLED_DRIVER_VERSION),
                Mic::NotDetermined,
            ),
        );
    }),
    ("onboarding-mic", driver_in),
    ("onboarding-mic-denied", |s| {
        driver_in(s);
        s.mic = Mic::Denied;
    }),
    ("onboarding-done", |s| {
        driver_in(s);
        s.mic = Mic::Authorized;
        s.prefs.system_microphone = true;
    }),
    ("main-off", |s| s.prefs.enabled = false),
    ("main-on", |_| {}),
    ("main-on-with-levels", active),
    ("main-error", |s| {
        active(s);
        s.on_event(oto_engine::Event::Error(
            "Your microphone was disconnected. Oto switched to the system default.".into(),
        ));
    }),
    ("main-mic-ask", |s| s.mic = Mic::NotDetermined),
    ("main-mic-denied", |s| s.mic = Mic::Denied),
    ("main-mic-restricted", |s| s.mic = Mic::Restricted),
    ("engine-failed", |s| s.engine_failed("no audio hardware")),
    ("engine-starting", |s| s.engine = EngineState::Starting),
    ("device-picker", |s| s.open = Some(Open::Device)),
    ("settings", |s| {
        s.page = Page::Settings;
        s.prefs.start_at_login = true;
        s.prefs.system_microphone = true;
    }),
    ("settings-no-driver", |s| {
        s.page = Page::Settings;
        s.driver_installed = false;
        s.driver_on_disk = false;
        s.skipped_onboarding = true;
        s.login_available = false;
    }),
    ("settings-uninstalling", |s| {
        s.page = Page::Settings;
        s.install = Install::Removing;
    }),
    ("settings-restart", |s| {
        s.page = Page::Settings;
        s.uninstalled(
            driver::UninstallOutcome::RemovedNeedsReboot,
            Probe::new(
                DriverStatus::Loaded(BUNDLED_DRIVER_VERSION),
                None,
                Mic::Authorized,
            ),
        );
    }),
    ("settings-low-cpu", |s| {
        s.page = Page::Settings;
        s.tiers = vec![Tier::Light, Tier::Standard];
        s.prefs.tier = Tier::Light;
    }),
    ("main-voice-isolation", |s| {
        s.tiers = vec![Tier::Light, Tier::Standard, Tier::Max, Tier::Isolate];
        s.set_tier(Tier::Isolate);
        active(s);
        s.set_running(Some((oto_engine::dsp::Model::Nearfield, 42.0)));
    }),
    ("main-degraded", |s| {
        s.tiers = vec![Tier::Light, Tier::Standard, Tier::Max, Tier::Isolate];
        s.set_tier(Tier::Isolate);
        active(s);
        s.set_running(Some((oto_engine::dsp::Model::Nearfield, 42.0)));
        s.on_event(oto_engine::Event::Degraded(
            "Voice Isolation couldn\u{2019}t load, so Oto Microphone is muted.".into(),
        ));
    }),
    ("main-hear-recording", |s| {
        s.hear = HearTest::Recording { processing: false };
    }),
    ("main-hear-ready", |s| {
        s.hear = HearTest::Ready {
            cleaned: true,
            is_bluetooth_hfp: true,
        };
    }),
    ("settings-models", |s| {
        s.page = Page::Settings;
        s.tiers = vec![
            Tier::Light,
            Tier::Standard,
            Tier::Max,
            Tier::Fast,
            Tier::Isolate,
        ];
        s.set_tier(Tier::Max);
        s.set_tier(Tier::Isolate);
    }),
    ("main-model-fallback", |s| {
        active(s);
        s.on_event(oto_engine::Event::ModelFallback {
            requested: oto_engine::dsp::Model::DeepFilterNet3,
            effective: oto_engine::dsp::Model::RNNoise,
            reason: "running pass codegen".into(),
        });
        s.set_running(Some((oto_engine::dsp::Model::RNNoise, 36.0)));
    }),
    ("update-available", |s| {
        s.update = UpdateState::Available(UpdateInfo {
            min_driver_version: BUNDLED_DRIVER_VERSION + 1,
            driver_version: BUNDLED_DRIVER_VERSION + 1,
            ..demo_update()
        });
    }),
    ("update-downloading", |s| {
        s.update = UpdateState::Downloading {
            info: demo_update(),
            pct: 42,
        };
    }),
    ("update-ready", |s| {
        active(s);
        s.update = UpdateState::Ready(StagedUpdate::unowned(
            demo_update(),
            PathBuf::from("/tmp/Oto.app"),
            PathBuf::from("/tmp"),
        ));
    }),
    ("update-notify-only", |s| {
        s.install_mode = InstallMode::NotifyOnly(apply::NotifyReason::NotInApplications);
        s.update = UpdateState::Available(demo_update());
    }),
    ("update-unrecognized-key", |s| {
        s.update = UpdateState::UnrecognizedKey {
            releases_url: updater::keys::RELEASES_URL.into(),
        };
    }),
    ("settings-updates", |s| {
        s.page = Page::Settings;
        s.update = UpdateState::Downloading {
            info: demo_update(),
            pct: 42,
        };
    }),
    ("driver-update-available", |s| {
        s.driver_status = DriverStatus::Loaded(BUNDLED_DRIVER_VERSION.saturating_sub(1));
        s.on_disk_version = Some(BUNDLED_DRIVER_VERSION.saturating_sub(1));
    }),
    ("driver-update-restart", |s| {
        s.driver_status = DriverStatus::Loaded(BUNDLED_DRIVER_VERSION.saturating_sub(1));
        s.on_disk_version = Some(BUNDLED_DRIVER_VERSION);
    }),
];

/// What the system reports about the driver and the microphone permission. A Core Audio query
/// can stall while coreaudiod restarts: never call this on the UI thread.
pub(crate) fn probe() -> Probe {
    Probe::new(
        Engine::driver_status(),
        driver::on_disk_version(),
        permission::status(),
    )
}

/// Takes a [`probe`] into the state. The first one decides whether setup is due, and opens it.
/// If Core Audio couldn't say whether the driver is loaded, it asks again shortly.
pub(crate) fn apply_probe(s: &mut State, p: Probe, h: &AppHandle<State>) {
    let first = !s.driver_known;
    s.probed(p);
    if p.driver_status == DriverStatus::Unknown {
        std::thread::spawn(|| {
            std::thread::sleep(Duration::from_secs(2));
            refresh();
        });
    } else if first && s.onboarding() {
        open_setup_soon(h);
    }
}

/// Opens the window for setup once the status item has had time to take its place, so the
/// window lands under it. Does nothing if setup is no longer due or the window is up.
fn open_setup_soon(h: &AppHandle<State>) {
    let h = h.clone();
    std::thread::spawn(move || {
        std::thread::sleep(TRAY_SETTLE);
        let h2 = h.clone();
        h.update(move |s, _| {
            if s.onboarding() && s.tray.is_some() && !h2.is_visible() {
                tray::show_window(s, &h2);
            }
        });
    });
}

/// Re-reads the driver and permission status off the UI thread. One at a time: if Core Audio
/// is stuck, later calls wait for the first instead of piling up threads.
fn refresh() {
    let Some(h) = HANDLE.get().cloned() else {
        return;
    };
    if REFRESHING.swap(true, Ordering::AcqRel) {
        return;
    }
    std::thread::spawn(move || {
        let p = probe();
        REFRESHING.store(false, Ordering::Release);
        let h2 = h.clone();
        h.update(move |s, _| apply_probe(s, p, &h2));
    });
}

/// A system prompt Oto raised (the installer's password prompt, the microphone prompt) is
/// gone. While it was up the popover ignored losing focus; now it takes focus back, so the
/// result is seen and the next click elsewhere dismisses it. If focus doesn't come back, it
/// hides: an always-on-top popover must not be left stuck open.
fn after_prompt(s: &mut State, h: &AppHandle<State>) {
    if !s.popover || s.focused || s.holding() || !h.is_visible() {
        return;
    }
    h.open();
    let h = h.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(800));
        let h2 = h.clone();
        h.update(move |s, _| {
            if s.popover && !s.focused && !s.holding() && h2.is_visible() {
                s.open = None;
                s.stop_hear_test();
                h2.hide();
            }
        });
    });
}

/// Starts the driver installer off the UI thread and reports back.
fn install() {
    let Some(h) = HANDLE.get().cloned() else {
        return;
    };
    std::thread::spawn(move || {
        // Checked again by the privileged step right before restarting coreaudiod; this only
        // tells the log why an install may end needing a restart.
        if let Ok(true) = driver::audio_busy() {
            eprintln!("oto: audio is in use; the driver may finish installing on the next restart");
        }
        let outcome = driver::install();
        let p = probe();
        let h2 = h.clone();
        h.update(move |s, _| {
            s.installed(outcome, p);
            after_prompt(s, &h2);
        });
    });
}

/// Starts the driver uninstaller off the UI thread and reports back.
fn uninstall() {
    let Some(h) = HANDLE.get().cloned() else {
        return;
    };
    std::thread::spawn(move || {
        let outcome = driver::uninstall();
        let p = probe();
        let h2 = h.clone();
        h.update(move |s, _| {
            s.uninstalled(outcome, p);
            after_prompt(s, &h2);
        });
    });
}

/// Shows the system's microphone prompt and reports the answer.
fn request_mic() {
    let Some(h) = HANDLE.get().cloned() else {
        return;
    };
    permission::request(move |_| {
        let mic = permission::status();
        let h2 = h.clone();
        h.update(move |s, _| {
            s.mic_answered(mic);
            after_prompt(s, &h2);
        });
    });
}

/// Records and cleans a "Hear the Difference" take off the UI thread and reports back. Marks
/// the switch from recording to cleaning when the recording time is up.
fn start_hear_test(job: HearJob) {
    let Some(h) = HANDLE.get().cloned() else {
        return;
    };
    let (h_rec, h_err, cancel_err, h_timer, cancel_timer) = (
        h.clone(),
        h.clone(),
        job.cancel.clone(),
        h,
        job.cancel.clone(),
    );
    let spawned = std::thread::Builder::new()
        .name("oto-hear-test".into())
        .spawn(move || {
            let r = test_capture::record_and_process(
                job.device.as_deref(),
                job.model,
                job.strength,
                HEAR_SECONDS,
                &job.cancel,
            );
            h_rec.update(move |s, _| s.hear_recorded(&job.cancel, r));
        });
    if let Err(e) = spawned {
        let msg = format!("Couldn\u{2019}t start the mic test: {e}");
        h_err.update(move |s, _| s.hear_recorded(&cancel_err, Err(msg)));
        return;
    }
    let _ = std::thread::Builder::new()
        .name("oto-hear-timer".into())
        .spawn(move || {
            std::thread::sleep(Duration::from_secs_f32(HEAR_SECONDS));
            if !cancel_timer.load(Ordering::Acquire) {
                h_timer.update(move |s, _| s.hear_processing(&cancel_timer));
            }
        });
}

fn play_hear_test(r: &TestRecording) -> Result<(), String> {
    let player = TestPlayer::start(r, false)?;
    HEAR_PLAYER.with(|p| *p.borrow_mut() = Some(player));
    Ok(())
}

fn stop_hear_test() {
    // Dropping the player stops the output.
    HEAR_PLAYER.with(|p| p.borrow_mut().take());
}

fn set_hear_cleaned(on: bool) {
    HEAR_PLAYER.with(|p| {
        if let Some(p) = p.borrow().as_ref() {
            p.set_cleaned(on);
        }
    });
}

/// Starts an engine on a new host thread; `started` hears how `Engine::start_with` went. An
/// uncommitted update candidate's engine starts inert.
fn launch(
    settings: Settings,
    meter: Meter,
    started: Option<mpsc::SyncSender<Result<(), String>>>,
) -> Link {
    let Some(h) = HANDLE.get().cloned() else {
        return Link::None;
    };
    let opts = EngineOptions {
        update_candidate: CANDIDATE.lock().unwrap().clone(),
        ..EngineOptions::default()
    };
    let (tx, rx) = mpsc::channel();
    *HOST.lock().unwrap() = host::spawn(settings, opts, rx, h, meter, started);
    Link::Engine(tx)
}

// ----- updates ----------------------------------------------------------------------------------

/// Checks for an update off the UI thread.
fn check_update(job: UpdateJob) {
    let Some(h) = HANDLE.get().cloned() else {
        return;
    };
    std::thread::spawn(move || {
        let r = match verify::current_os_version() {
            Some(os) => verify::check_manifest(
                &*updater::http::system_transport(),
                VERSION,
                os,
                updater::keys::TRUSTED_KEYS,
                &job.cancel,
            ),
            None => Err("Updates need macOS.".into()),
        };
        h.update(move |s, _| s.update_checked(job.generation, r));
    });
}

/// Downloads and verifies an update off the UI thread, reporting progress as it changes.
fn download_update(job: UpdateJob, info: UpdateInfo) {
    let Some(h) = HANDLE.get().cloned() else {
        return;
    };
    std::thread::spawn(move || {
        let generation = job.generation;
        let (mut last, mut verifying) = (u8::MAX, false);
        let hp = h.clone();
        let r = verify::download_and_verify(
            &*updater::http::system_transport(),
            &info,
            VERSION,
            &job.cancel,
            &mut |got, total| {
                let pct = (got.saturating_mul(100) / total.max(1)).min(100) as u8;
                if pct != last {
                    last = pct;
                    hp.update(move |s, _| s.update_progress(generation, pct));
                }
                if got >= total && !verifying {
                    verifying = true;
                    hp.update(move |s, _| s.update_verifying(generation));
                }
            },
        );
        h.update(move |s, _| s.update_downloaded(generation, r));
    });
}

/// Stages the update, has the engine shut down (unless a call started), and hands over to the
/// helper, which swaps the app once this process is gone. Any failure before the commit byte
/// leaves Oto running: a refused shutdown keeps the engine, a dead helper gets a new engine.
fn apply_update(job: UpdateJob, staged: StagedUpdate, target: PathBuf) {
    let Some(h) = HANDLE.get().cloned() else {
        return;
    };
    std::thread::spawn(move || {
        let generation = job.generation;
        let fail = |msg: String, restart: bool| {
            h.update(move |s, _| s.update_apply_failed(generation, msg, restart));
        };
        let prepared = match PreparedUpdate::prepare(&staged, &target, VERSION) {
            Ok(p) => p,
            Err(e) => return fail(e, false),
        };
        eprintln!(
            "oto: update {} ({}) staged at {} for {}; helper {}",
            prepared.new_version,
            prepared.nonce.as_str(),
            prepared.staging.display(),
            prepared.target.display(),
            prepared.helper_path.display()
        );
        let (ack_tx, ack_rx) = mpsc::sync_channel(1);
        h.update(move |s, _| s.commit_shutdown(generation, ack_tx));
        match ack_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(msg)) => {
                prepared.cancel();
                return fail(msg, false);
            }
            Err(_) => {
                prepared.cancel();
                return fail("The update was cancelled.".into(), false);
            }
        }
        match prepared.commit_and_signal() {
            // The commit pipe stays open until this process exits: the helper's EOF is the
            // proof it's gone.
            Ok(()) => h.update(|s, tx| {
                s.update_committed();
                tx.quit();
            }),
            Err(e) => fail(e, true),
        }
    });
}

/// Opens a project page with `/usr/bin/open`.
fn open_url(url: &str) -> Result<(), String> {
    if !state::openable_url(url) {
        return Err("Oto only opens its GitHub pages.".into());
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("/usr/bin/open")
            .arg(url)
            .spawn()
            .map(|_| ())
            .map_err(|e| format!("Couldn't open {url}: {e}"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(format!("Open {url} in your browser."))
    }
}

/// Automatic checks: the first shortly after launch, then whenever a day has passed.
fn schedule_update_checks(h: AppHandle<State>) {
    std::thread::spawn(move || {
        std::thread::sleep(FIRST_UPDATE_CHECK);
        loop {
            h.update(|s, _| s.maybe_check_for_updates());
            std::thread::sleep(UPDATE_TICK);
        }
    });
}

/// The candidate is up (engine with `engine.lock`, status item or window): writes stage 2 and
/// waits for the helper's commit, then unlocks the app. Revoked or gone: exits at once, with
/// no side effects (the engine is still inert).
fn candidate_ready(nonce: Nonce, h: AppHandle<State>) {
    let Some(paths) = UpdatePaths::default_location() else {
        std::process::exit(0);
    };
    if apply::candidate_mark_ready(&paths, &nonce).is_err() {
        std::process::exit(0);
    }
    std::thread::spawn(move || {
        let stop = AtomicBool::new(false);
        match apply::candidate_wait_for_commit_or_takeover(&paths, &nonce, CANDIDATE_POLL, &stop) {
            apply::CandidateDecision::Committed => {
                CANDIDATE.lock().unwrap().take();
                let h2 = h.clone();
                h.update(move |s, _| {
                    s.commit_candidate();
                    if s.onboarding() {
                        open_setup_soon(&h2);
                    }
                });
            }
            apply::CandidateDecision::Exit => std::process::exit(0),
        }
    });
}

/// The real app's first state: saved preferences, the engine on its own thread, and the status
/// item once the event loop runs. Exits instead if another Oto already runs the engine.
fn live(h: AppHandle<State>) -> State {
    let mut prefs = prefs::load();
    // The system's login items are the truth; the file only mirrors them.
    prefs.start_at_login = login::is_enabled();
    let mut s = State::new(prefs, false);
    // Core Audio and the permission are asked on the engine's thread; until it answers the
    // app neither offers the installer nor claims the driver is there.
    s.driver_known = false;
    s.engine = EngineState::Starting;
    s.login_available = login::is_available();
    s.hooks = Hooks {
        save: prefs::save,
        set_login: login::set,
        install,
        uninstall,
        start_engine: |settings, meter| launch(settings, meter, None),
        request_mic,
        open_mic_settings: permission::open_settings,
        start_hear_test,
        play_hear_test,
        stop_hear_test,
        set_hear_cleaned,
        check_update,
        download_update,
        apply_update,
        open_url,
        now: oto_engine::update_state::now_unix,
    };
    let candidate = CANDIDATE.lock().unwrap().as_ref().map(|t| t.nonce.clone());
    s.candidate_pending = candidate.is_some();
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    s.link = launch(s.prefs.engine_settings(), s.meter.clone(), Some(started_tx));
    let wait = if candidate.is_some() {
        CANDIDATE_START_WAIT
    } else {
        START_WAIT
    };
    match started_rx.recv_timeout(wait) {
        Ok(Err(e)) if e == oto_engine::ALREADY_RUNNING || e == oto_engine::UPDATING => {
            // Before the status item exists: a second Oto (or one started mid-update) leaves no
            // trace.
            if e == oto_engine::ALREADY_RUNNING {
                eprintln!("oto: Oto is already running. Look for it in the menu bar.");
            }
            std::process::exit(0);
        }
        // A candidate acknowledges only with a running engine; without one, the helper rolls
        // back.
        Ok(Err(_)) | Err(_) if candidate.is_some() => std::process::exit(0),
        _ => {}
    }
    let h2 = h.clone();
    h.update(move |s, _| {
        match tray::Tray::new(h2.clone(), s.prefs.enabled) {
            Ok(t) => {
                s.tray = Some(t);
                s.popover = true;
                if s.onboarding() {
                    open_setup_soon(&h2);
                }
            }
            Err(e) => {
                eprintln!("oto: no status item ({e}); opening a window instead");
                h2.open();
            }
        }
        if let Some(nonce) = candidate {
            candidate_ready(nonce, h2.clone());
        }
        schedule_update_checks(h2.clone());
    });
    s
}

/// The popover hides when it loses focus, as a menu-bar popover does, but not while a prompt
/// Oto raised (the installer's password prompt, the microphone prompt) has the focus:
/// [`after_prompt`] deals with it once that's gone.
fn on_focus(s: &mut State, focused: bool, tx: &mut Tx) {
    s.focused = focused;
    if focused {
        refresh();
    } else if s.popover && !s.holding() {
        s.open = None;
        // A hidden popover must not keep the mic or the speakers.
        s.stop_hear_test();
        s.hidden_at = Some(Instant::now());
        tx.close_window();
    }
}

/// Reopening Oto while it already runs in the menu bar (Spotlight, Finder, Applications,
/// `open`) shows the popover under the status item.
fn on_reopen(s: &mut State, _has_visible_windows: bool, _tx: &mut Tx) -> bool {
    let Some(h) = HANDLE.get() else {
        return false;
    };
    tray::show_window(s, h);
    true
}

#[cfg(target_os = "macos")]
fn window_options() -> rune::ConfiguredWindowOptions {
    rune::WindowOptions {
        borderless: true,
        transparent: true,
        always_on_top: true,
        resizable: false,
        min_size: Some((0.0, 0.0)),
        ..Default::default()
    }
    .macos(
        rune::MacWindowOptions::default()
            .nonactivating_panel(true)
            .corner_radius(view::RADIUS)
            .shadow(true),
    )
}

/// Handles the update modes before anything else starts: the helper, manual recovery, the
/// restored app waiting for the helper, the candidate's admission, and normal startup's
/// recovery and admission check. Returns only when the app should start.
fn dispatch_cli() {
    let mode = match apply::parse_cli_mode(std::env::args_os().skip(1)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("oto: {e}");
            std::process::exit(0);
        }
    };
    let exe = || std::env::current_exe().and_then(|p| p.canonicalize()).ok();
    match mode {
        CliMode::FinishUpdate(args) => std::process::exit(apply::run_helper(&args)),
        CliMode::RecoverUpdate => std::process::exit(apply::run_recover_update()),
        CliMode::UpdateAck(nonce) => {
            let token = UpdatePaths::default_location()
                .zip(exe())
                .and_then(|(paths, exe)| {
                    apply::candidate_admit_and_mark_started(&paths, &nonce, &exe).ok()
                });
            match token {
                Some(t) => *CANDIDATE.lock().unwrap() = Some(t),
                // Not (or no longer) this update's candidate: no side effects at all.
                None => std::process::exit(0),
            }
        }
        CliMode::WaitUpdateRelease => {
            if let Some(paths) = UpdatePaths::default_location() {
                apply::wait_for_update_release(&paths, Duration::from_secs(5));
            }
            normal_startup(exe().as_deref());
        }
        CliMode::Normal => {
            // Rune's tooling (`--snap`, `--list-scenes`, …) renders demo state: leave the real
            // update state alone.
            if !std::env::args().skip(1).any(|a| a.starts_with("--")) {
                normal_startup(exe().as_deref());
            }
        }
    }
}

/// Recovers a transaction a dead helper left, cleans up, and exits quietly while an update
/// still blocks the engine.
fn normal_startup(exe: Option<&Path>) {
    if let Some(exe) = exe
        && apply::startup_recovery(exe).startup_should_exit()
    {
        std::process::exit(0);
    }
    if let Some(paths) = UpdatePaths::default_location() {
        apply::cleanup_stale_files(&paths);
        if oto_engine::update_state::update_blocked(&paths).is_some() {
            std::process::exit(0);
        }
    }
}

fn main() {
    dispatch_cli();
    let spec = AppSpec {
        scenes: SCENES,
        ..AppSpec::new("Oto", (view::WIDTH, 560.0), demo, view::view)
    };
    let mut app = App::new(spec)
        .windowless()
        .fit_content_height()
        .on_focus(on_focus)
        .on_reopen(on_reopen)
        .on_quit(|s, _| s.shutdown());
    #[cfg(target_os = "macos")]
    {
        app = app.accessory().configured_options(window_options());
    }
    let h = app.handle();
    let _ = HANDLE.set(h.clone());
    app = app.init(move || {
        if rune::tooling::headless_run().is_some() {
            demo()
        } else {
            live(h)
        }
    });
    app.run();
    // Dropping the engine stops audio and puts back the default input if "Use Oto as the
    // system microphone" changed it: give the host thread a moment to do that.
    if let Some(host) = HOST.lock().unwrap().take() {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !host.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
