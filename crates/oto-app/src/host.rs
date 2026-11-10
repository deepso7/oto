//! The engine's home thread. `Engine` isn't `Sync`, so one thread owns it: it forwards the UI's
//! commands, drains the engine's events into the UI with `AppHandle::update`, re-reads the device
//! list when it changes, and starts the metering thread.

use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::Duration;

use oto_engine::{Command, Engine, Event, Settings};
use rune::AppHandle;

use crate::meter::{self, Meter};
use crate::state::State;

/// How often events are drained when no command arrives. The engine has no blocking event
/// API, so this is a slow poll; 5 Hz is invisible next to a status line.
const EVENT_POLL: Duration = Duration::from_millis(200);

pub fn spawn(
    settings: Settings,
    commands: Receiver<Command>,
    h: AppHandle<State>,
    meter: Meter,
) -> Option<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("oto-engine".into())
        .spawn(move || run(settings, commands, h, meter))
        .ok()
}

fn run(settings: Settings, commands: Receiver<Command>, h: AppHandle<State>, meter: Meter) {
    let engine = match Engine::start(settings) {
        Ok(e) => e,
        Err(e) => {
            h.update(move |s, _| {
                s.error = Some(format!("Oto couldn't start its audio engine: {e}"))
            });
            return;
        }
    };
    let devices = engine.input_devices();
    h.update(move |s, _| s.set_devices(devices));
    let visible = h.clone();
    meter::spawn(engine.levels(), meter, move || visible.is_visible());

    loop {
        match commands.recv_timeout(EVENT_POLL) {
            Ok(Command::Shutdown) => {
                engine.send(Command::Shutdown);
                return;
            }
            Ok(cmd) => engine.send(cmd),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                engine.send(Command::Shutdown);
                return;
            }
        }
        let events = engine.try_events();
        if events.is_empty() {
            continue;
        }
        let devices = events
            .contains(&Event::DevicesChanged)
            .then(|| engine.input_devices());
        h.update(move |s, _| {
            for e in events {
                s.on_event(e);
            }
            if let Some(d) = devices {
                s.set_devices(d);
            }
        });
    }
}
