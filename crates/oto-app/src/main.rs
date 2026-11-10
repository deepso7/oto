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
mod view;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use oto_engine::{Engine, InputDevice, Settings, Tier};
use rune::{App, AppHandle, AppSpec, Tx};

use crate::meter::Meter;
use crate::permission::Mic;
use crate::state::{EngineState, Hooks, Install, Link, Open, Page, Probe, State};

/// The running app's handle, for hooks that are plain `fn`s.
static HANDLE: OnceLock<AppHandle<State>> = OnceLock::new();
/// The current engine host thread, so quitting can wait for it to put things back.
static HOST: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);
/// A [`refresh`] is under way.
static REFRESHING: AtomicBool = AtomicBool::new(false);

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
            Probe {
                driver: false,
                on_disk: true,
                mic: Mic::NotDetermined,
            },
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
            Probe {
                driver: true,
                on_disk: false,
                mic: Mic::Authorized,
            },
        );
    }),
    ("settings-tiers", |s| {
        s.page = Page::Settings;
        s.tiers = vec![Tier::Light, Tier::Standard];
    }),
    ("settings-tier-picker", |s| {
        s.page = Page::Settings;
        s.tiers = vec![Tier::Light, Tier::Standard];
        s.open = Some(Open::Tier);
    }),
];

/// What the system reports about the driver and the microphone permission. A Core Audio query
/// can stall while coreaudiod restarts: never call this on the UI thread.
pub(crate) fn probe() -> Probe {
    Probe {
        driver: Engine::driver_installed(),
        on_disk: driver::bundle_present(),
        mic: permission::status(),
    }
}

/// Takes a [`probe`] into the state. The first one decides whether setup is due, and opens it.
pub(crate) fn apply_probe(s: &mut State, p: Probe, h: &AppHandle<State>) {
    let first = !s.driver_known;
    s.probed(p);
    if first && s.onboarding() {
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

/// Starts an engine on a new host thread; `started` hears how `Engine::start` went.
fn launch(
    settings: Settings,
    meter: Meter,
    started: Option<mpsc::SyncSender<Result<(), String>>>,
) -> Link {
    let Some(h) = HANDLE.get().cloned() else {
        return Link::None;
    };
    let (tx, rx) = mpsc::channel();
    *HOST.lock().unwrap() = host::spawn(settings, rx, h, meter, started);
    Link::Engine(tx)
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
    };
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    s.link = launch(s.prefs.engine_settings(), s.meter.clone(), Some(started_tx));
    if let Ok(Err(e)) = started_rx.recv_timeout(START_WAIT)
        && e == oto_engine::ALREADY_RUNNING
    {
        // Before the status item exists: a second Oto leaves no trace.
        eprintln!("oto: Oto is already running. Look for it in the menu bar.");
        std::process::exit(0);
    }
    let h2 = h.clone();
    h.update(
        move |s, _| match tray::Tray::new(h2.clone(), s.prefs.enabled) {
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
        },
    );
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
        s.hidden_at = Some(Instant::now());
        tx.close_window();
    }
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
            .corner_radius(12.0)
            .shadow(true),
    )
}

fn main() {
    let spec = AppSpec {
        scenes: SCENES,
        ..AppSpec::new("Oto", (view::WIDTH, 560.0), demo, view::view)
    };
    let mut app = App::new(spec)
        .windowless()
        .fit_content_height()
        .on_focus(on_focus)
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
