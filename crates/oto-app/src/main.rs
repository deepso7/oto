//! Oto's menu-bar app. `cargo run -p oto-app` starts it in the menu bar (the window opens on
//! its own only when the driver still needs installing); a click on the status item opens the
//! popover. `cargo run -p oto-app -- --snap oto.png --scene main-on-with-levels` renders a
//! scene headless; `--list-scenes` lists them.

mod driver;
mod host;
mod login;
mod meter;
mod prefs;
mod state;
mod tray;
mod view;

use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use oto_engine::{Engine, InputDevice, Tier};
use rune::{App, AppHandle, AppSpec, Tx};

use crate::state::{Hooks, Install, Link, Open, Page, State};

/// The running app's handle, for hooks that are plain `fn`s.
static HANDLE: OnceLock<AppHandle<State>> = OnceLock::new();
static HOST: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);

/// What headless snapshots start from: a Mac with the driver in and two mics.
fn demo() -> State {
    let mut s = State::new(prefs::Prefs::default(), true);
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

type Scene = (&'static str, fn(&mut State));

const SCENES: &[Scene] = &[
    ("onboarding", |s| s.driver_installed = false),
    ("onboarding-installing", |s| {
        s.driver_installed = false;
        s.install = Install::Running;
    }),
    ("onboarding-failed", |s| {
        s.driver_installed = false;
        s.install = Install::Failed(
            "The driver installer (install-driver.sh) isn't in this build of Oto.".into(),
        );
    }),
    ("main-off", |s| s.prefs.enabled = false),
    ("main-on", |_| {}),
    ("main-on-with-levels", active),
    ("main-error", |s| {
        active(s);
        s.error =
            Some("Your microphone was disconnected. Oto switched to the system default.".into());
    }),
    ("device-picker", |s| s.open = Some(Open::Device)),
    ("settings", |s| {
        s.page = Page::Settings;
        s.prefs.start_at_login = true;
    }),
    ("settings-max", |s| {
        s.page = Page::Settings;
        s.prefs.tier = Tier::Max;
        s.driver_installed = false;
        s.skipped_onboarding = true;
    }),
];

/// Starts the driver installer off the UI thread and reports back.
fn install() {
    let Some(h) = HANDLE.get().cloned() else {
        return;
    };
    std::thread::spawn(move || {
        let outcome = driver::install();
        let ok = Engine::driver_installed();
        h.update(move |s, _| s.installed(outcome, ok));
    });
}

/// The real app's first state: saved preferences, the engine on its own thread, and the status
/// item once the event loop runs.
fn live(h: AppHandle<State>) -> State {
    let mut prefs = prefs::load();
    // The system's login items are the truth; the file only mirrors them.
    prefs.start_at_login = login::is_enabled();
    let mut s = State::new(prefs, Engine::driver_installed());
    s.hooks = Hooks {
        save: prefs::save,
        set_login: login::set,
        install,
    };
    let (tx, rx) = mpsc::channel();
    s.link = Link::Engine(tx);
    *HOST.lock().unwrap() = host::spawn(s.prefs.engine_settings(), rx, h.clone(), s.meter.clone());
    let h2 = h.clone();
    h.update(
        move |s, _| match tray::Tray::new(h2.clone(), s.prefs.enabled) {
            Ok(t) => {
                s.tray = Some(t);
                s.popover = true;
                if s.onboarding() {
                    // Give the status item a moment to take its place, so the window opens under it.
                    let h3 = h2.clone();
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(400));
                        let h4 = h3.clone();
                        h3.update(move |s, _| tray::show_window(s, &h4));
                    });
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

/// The popover hides when it loses focus, as a menu-bar popover does, but not while the
/// password prompt for the installer has focus.
fn on_focus(s: &mut State, focused: bool, tx: &mut Tx) {
    if focused {
        s.driver_installed = s.driver_installed || Engine::driver_installed();
    } else if s.popover && s.install != Install::Running {
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
    // Give the engine a moment to restore the default input before the process goes.
    if let Some(host) = HOST.lock().unwrap().take() {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !host.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
