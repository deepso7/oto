//! The engine's home thread. `Engine` isn't `Sync`, so one thread owns it: it starts the engine,
//! forwards the UI's requests, drains the engine's events into the UI with `AppHandle::update`,
//! re-reads the device list when it changes, and starts the metering thread.

use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::thread::JoinHandle;
use std::time::Duration;

use oto_engine::{Command, Engine, Event, Settings};
use rune::AppHandle;

use crate::meter::{self, Meter};
use crate::state::{Request, State};

/// How often events are drained when no request arrives. The engine has no blocking event
/// API, so this is a slow poll; 5 Hz is invisible next to a status line.
const EVENT_POLL: Duration = Duration::from_millis(200);

/// Starts an engine on a new thread. `Engine::start`'s result goes to `started` first (the
/// app's launch waits on it to see whether another Oto is running) and then to the UI as
/// [`State::engine_started`] or [`State::engine_failed`]. `None` when the thread can't spawn;
/// the dropped `requests` then makes the next request fail, which the state reports.
pub fn spawn(
    settings: Settings,
    requests: Receiver<Request>,
    h: AppHandle<State>,
    meter: Meter,
    started: Option<SyncSender<Result<(), String>>>,
) -> Option<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("oto-engine".into())
        .spawn(move || run(settings, requests, h, meter, started))
        .ok()
}

fn run(
    settings: Settings,
    requests: Receiver<Request>,
    h: AppHandle<State>,
    meter: Meter,
    started: Option<SyncSender<Result<(), String>>>,
) {
    let result = Engine::start(settings);
    if let Some(tx) = started {
        let _ = tx.send(result.as_ref().map(|_| ()).map_err(Clone::clone));
    }
    // The first look at the driver and the permission happens here too, off the UI thread: a
    // Core Audio query can stall while coreaudiod restarts.
    let probe = crate::probe();
    let engine = match result {
        Ok(e) => e,
        Err(e) => {
            let h2 = h.clone();
            h.update(move |s, _| {
                s.engine_failed(&e);
                crate::apply_probe(s, probe, &h2);
            });
            return;
        }
    };
    let devices = engine.input_devices();
    let h2 = h.clone();
    h.update(move |s, _| {
        s.engine_started();
        s.set_devices(devices);
        crate::apply_probe(s, probe, &h2);
    });
    let visible = h.clone();
    meter::spawn(engine.levels(), meter, move || visible.is_visible());

    loop {
        match requests.recv_timeout(EVENT_POLL) {
            // Dropping the engine stops it and restores the default input if Oto changed it.
            Ok(Request::Cmd(Command::Shutdown)) | Err(RecvTimeoutError::Disconnected) => return,
            Ok(Request::Cmd(cmd)) => engine.send(cmd),
            Ok(Request::SystemMic(on)) => {
                let r = if on {
                    engine.set_default_input_to_oto()
                } else {
                    engine.restore_default_input()
                };
                if let Err(e) = r {
                    h.update(move |s, _| s.system_mic_failed(on, e));
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
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
