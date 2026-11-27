//! The engine's home thread. `Engine` isn't `Sync`, so one thread owns it: it starts the engine,
//! forwards the UI's requests, drains the engine's events into the UI with `AppHandle::update`,
//! re-reads the device list when it changes, and starts the metering thread.
//!
//! It also runs the engine's side of an app update: [`Request::CommitShutdown`] stops the engine
//! for the swap unless a call is active (and restarts it if restoring the default input fails),
//! and [`Request::CommitCandidate`] ends an update candidate's inert mode.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use oto_engine::{Command, Engine, EngineOptions, Event, Settings};
use rune::AppHandle;

use crate::meter::{self, Meter};
use crate::state::{CALL_CHECK_FAILED, Request, State, UPDATE_IN_CALL};

/// How often events are drained when no request arrives. The engine has no blocking event
/// API, so this is a slow poll; 5 Hz is invisible next to a status line.
const EVENT_POLL: Duration = Duration::from_millis(200);

/// How long a restart keeps retrying while an update still holds `updating.lock` (its helper
/// releases it once it sees the cancelled or failed hand-over).
const UPDATING_RETRY: Duration = Duration::from_secs(8);

/// Starts an engine on a new thread. `Engine::start_with`'s result goes to `started` first (the
/// app's launch waits on it to see whether another Oto is running) and then to the UI as
/// [`State::engine_started`] or [`State::engine_failed`]. `None` when the thread can't spawn;
/// the dropped `requests` then makes the next request fail, which the state reports.
///
/// Without `started` this is a restart (Try Again, or after an update's helper died): it waits
/// out an update that is just releasing `updating.lock` instead of failing at once.
pub fn spawn(
    settings: Settings,
    opts: EngineOptions,
    requests: Receiver<Request>,
    h: AppHandle<State>,
    meter: Meter,
    started: Option<SyncSender<Result<(), String>>>,
) -> Option<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("oto-engine".into())
        .spawn(move || run(settings, opts, requests, h, meter, started))
        .ok()
}

/// `Engine::start_with`, retried for up to `wait` while it fails with [`oto_engine::UPDATING`].
fn start_retrying(
    settings: &Settings,
    opts: &EngineOptions,
    wait: Duration,
) -> Result<Engine, String> {
    let deadline = Instant::now() + wait;
    loop {
        match Engine::start_with(settings.clone(), opts.clone()) {
            Err(e) if e == oto_engine::UPDATING && Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(250));
            }
            r => return r,
        }
    }
}

/// How this copy can update, now that the engine knows whether it holds `engine.lock`.
fn install_mode(engine: &Engine) -> crate::updater::apply::InstallMode {
    let exe = std::env::current_exe().and_then(|p| p.canonicalize()).ok();
    crate::updater::apply::classify_install(exe.as_deref(), engine.has_instance_lock())
}

/// What [`Request::CommitShutdown`] needs of an engine (faked in tests).
pub(crate) trait Teardown: Sized {
    /// Stops audio, restores the default input and releases `engine.lock`.
    fn shutdown_and_restore(self) -> Result<(), String>;
}

impl Teardown for Engine {
    fn shutdown_and_restore(self) -> Result<(), String> {
        Engine::shutdown_and_restore(self)
    }
}

/// What became of the engine after a [`Request::CommitShutdown`].
pub(crate) enum AfterCommit<E> {
    /// Shut down and restored: the host thread ends.
    Done,
    /// Still running the engine it had: a call is active, or the check failed.
    Running(E),
    /// Restoring failed, so the engine was restarted with the same settings.
    Restarted(E),
    /// Restoring failed and so did the restart.
    Failed(String),
}

/// The engine's side of an update's hand-over. Re-checks Oto Microphone right now (`Ok(None)`,
/// no driver, and `Ok(Some(false))` go ahead; a call or a failed check keeps the engine), then
/// shuts down and restores the default input. The answer goes to `ack` before any restart, so
/// the updater can cancel its helper (which releases `updating.lock`) while `restart` waits for
/// that.
pub(crate) fn commit_shutdown<E: Teardown>(
    engine: E,
    mic_active: impl FnOnce() -> Result<Option<bool>, String>,
    ack: &SyncSender<Result<(), String>>,
    restart: impl FnOnce() -> Result<E, String>,
) -> AfterCommit<E> {
    match mic_active() {
        Ok(None | Some(false)) => {}
        Ok(Some(true)) => {
            let _ = ack.send(Err(UPDATE_IN_CALL.into()));
            return AfterCommit::Running(engine);
        }
        Err(_) => {
            let _ = ack.send(Err(CALL_CHECK_FAILED.into()));
            return AfterCommit::Running(engine);
        }
    }
    match engine.shutdown_and_restore() {
        Ok(()) => {
            let _ = ack.send(Ok(()));
            AfterCommit::Done
        }
        Err(msg) => {
            let _ = ack.send(Err(format!(
                "Couldn\u{2019}t restore your previous microphone, so Oto didn\u{2019}t update: {msg}"
            )));
            match restart() {
                Ok(e) => AfterCommit::Restarted(e),
                Err(e) => AfterCommit::Failed(e),
            }
        }
    }
}

/// Keeps `settings` in step with the commands sent, so a restart resumes where the UI is.
fn track(settings: &mut Settings, cmd: &Command) {
    match cmd {
        Command::SetEnabled(on) => settings.enabled = *on,
        Command::SetInputDevice(d) => settings.input_device = d.clone(),
        Command::SetStrength(s) => settings.strength = *s,
        Command::SetTier(t) => settings.tier = *t,
        Command::Shutdown => {}
    }
}

/// Reports a started engine to the UI and starts its meter.
fn announce(engine: &Engine, h: &AppHandle<State>, meter: &Meter, restarted: bool) {
    let devices = engine.input_devices();
    let mode = install_mode(engine);
    h.update(move |s, _| {
        if restarted {
            s.engine_restarted();
        } else {
            s.engine_started();
        }
        s.set_install_mode(mode);
        s.set_devices(devices);
    });
    let visible = h.clone();
    meter::spawn(Arc::downgrade(&engine.levels()), meter.clone(), move || {
        visible.is_visible()
    });
}

/// Commits an update candidate and clears its one-time token from `opts` so any future engine
/// restart in this process starts as a normal instance rather than retrying a deleted transaction.
fn commit_candidate_opts(
    opts: &mut EngineOptions,
    commit: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    let r = commit();
    if r.is_ok() {
        opts.update_candidate = None;
    }
    r
}

fn run(
    mut settings: Settings,
    mut opts: EngineOptions,
    requests: Receiver<Request>,
    h: AppHandle<State>,
    meter: Meter,
    started: Option<SyncSender<Result<(), String>>>,
) {
    let result = if started.is_some() {
        Engine::start_with(settings.clone(), opts.clone())
    } else {
        start_retrying(&settings, &opts, UPDATING_RETRY)
    };
    if let Some(tx) = started {
        let _ = tx.send(result.as_ref().map(|_| ()).map_err(Clone::clone));
    }
    // The first look at the driver and the permission happens here too, off the UI thread: a
    // Core Audio query can stall while coreaudiod restarts.
    let probe = crate::probe();
    let mut engine = match result {
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
    announce(&engine, &h, &meter, false);
    let h2 = h.clone();
    h.update(move |s, _| crate::apply_probe(s, probe, &h2));
    let mut running = None;

    loop {
        match requests.recv_timeout(EVENT_POLL) {
            // Dropping the engine stops it and restores the default input if Oto changed it.
            Ok(Request::Cmd(Command::Shutdown)) | Err(RecvTimeoutError::Disconnected) => return,
            Ok(Request::Cmd(cmd)) => {
                track(&mut settings, &cmd);
                engine.send(cmd);
            }
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
            Ok(Request::CommitCandidate(ack)) => {
                let r = commit_candidate_opts(&mut opts, || engine.commit_candidate());
                if let Err(e) = &r {
                    let e = e.clone();
                    h.update(move |s, _| s.on_event(Event::Error(e)));
                }
                let _ = ack.send(r);
            }
            Ok(Request::CommitShutdown(ack)) => {
                let (s2, o2) = (settings.clone(), opts.clone());
                match commit_shutdown(engine, Engine::is_oto_mic_active_checked, &ack, || {
                    start_retrying(&s2, &o2, UPDATING_RETRY)
                }) {
                    AfterCommit::Done => return,
                    AfterCommit::Running(e) => engine = e,
                    AfterCommit::Restarted(e) => {
                        engine = e;
                        running = None;
                        announce(&engine, &h, &meter, true);
                    }
                    AfterCommit::Failed(e) => {
                        h.update(move |s, _| s.engine_failed(&e));
                        return;
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
        }
        // The running model and delay change only when streams restart; send them on change.
        let stats = engine.stats();
        let now = stats
            .model
            .filter(|_| stats.running)
            .map(|m| (m, stats.estimated_latency_ms.round()));
        if now != running {
            running = now;
            h.update(move |s, _| s.set_running(now));
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::mpsc::sync_channel;

    #[test]
    fn committing_a_candidate_clears_its_token_for_future_restarts() {
        let token = oto_engine::update_state::CandidateToken::new(
            oto_engine::update_state::Nonce::parse("cand1").unwrap(),
        );
        let mut opts = EngineOptions {
            update_candidate: Some(token.clone()),
            ..EngineOptions::default()
        };
        // A failed commit keeps the token untouched.
        assert!(commit_candidate_opts(&mut opts, || Err("still waiting".into())).is_err());
        assert_eq!(opts.update_candidate, Some(token));
        // A successful commit clears it so later restarts start as a normal engine.
        assert!(commit_candidate_opts(&mut opts, || Ok(())).is_ok());
        assert_eq!(opts.update_candidate, None);
    }

    /// A fake engine: `restore` is what its teardown returns; `id` tells instances apart.
    struct Fake {
        id: u32,
        restore: Result<(), String>,
        torn_down: std::rc::Rc<Cell<u32>>,
    }

    impl Teardown for Fake {
        fn shutdown_and_restore(self) -> Result<(), String> {
            self.torn_down.set(self.torn_down.get() + 1);
            self.restore
        }
    }

    fn fake(restore: Result<(), String>) -> (Fake, std::rc::Rc<Cell<u32>>) {
        let n = std::rc::Rc::new(Cell::new(0));
        (
            Fake {
                id: 1,
                restore,
                torn_down: n.clone(),
            },
            n,
        )
    }

    fn run_commit(
        restore: Result<(), String>,
        mic: Result<Option<bool>, String>,
        restart: Result<u32, String>,
    ) -> (AfterCommit<Fake>, Result<(), String>, u32, bool) {
        let (engine, torn) = fake(restore);
        let (ack, rx) = sync_channel(1);
        let restarted = Cell::new(false);
        let out = commit_shutdown(
            engine,
            || mic,
            &ack,
            || {
                restarted.set(true);
                restart.map(|id| Fake {
                    id,
                    restore: Ok(()),
                    torn_down: torn.clone(),
                })
            },
        );
        let reply = rx.try_recv().expect("the updater always hears back");
        (out, reply, torn.get(), restarted.get())
    }

    #[test]
    fn no_driver_or_an_idle_mic_lets_the_update_go_ahead() {
        for mic in [Ok(None), Ok(Some(false))] {
            let (out, reply, torn, restarted) = run_commit(Ok(()), mic, Ok(2));
            assert!(matches!(out, AfterCommit::Done));
            assert_eq!(reply, Ok(()));
            assert_eq!(torn, 1);
            assert!(!restarted);
        }
    }

    #[test]
    fn an_active_call_keeps_the_engine_running() {
        let (out, reply, torn, restarted) = run_commit(Ok(()), Ok(Some(true)), Ok(2));
        assert!(matches!(out, AfterCommit::Running(Fake { id: 1, .. })));
        assert_eq!(reply, Err(UPDATE_IN_CALL.into()));
        assert_eq!(torn, 0, "never shut down mid-call");
        assert!(!restarted);
    }

    #[test]
    fn a_failed_call_check_keeps_the_engine_running() {
        let (out, reply, torn, _) = run_commit(Ok(()), Err("OSStatus -1".into()), Ok(2));
        assert!(matches!(out, AfterCommit::Running(Fake { id: 1, .. })));
        assert_eq!(reply, Err(CALL_CHECK_FAILED.into()));
        assert_eq!(torn, 0);
    }

    #[test]
    fn a_failed_restore_restarts_the_engine_and_refuses() {
        let (out, reply, torn, restarted) =
            run_commit(Err("device unplugged".into()), Ok(Some(false)), Ok(2));
        assert!(matches!(out, AfterCommit::Restarted(Fake { id: 2, .. })));
        assert!(reply.unwrap_err().contains("device unplugged"));
        assert_eq!(torn, 1);
        assert!(restarted);

        let (out, reply, _, _) = run_commit(
            Err("device unplugged".into()),
            Ok(None),
            Err("no audio hardware".into()),
        );
        assert!(matches!(out, AfterCommit::Failed(e) if e == "no audio hardware"));
        assert!(reply.is_err());
    }

    #[test]
    fn the_answer_is_sent_before_the_restart() {
        // The restart may wait for the helper to release `updating.lock`, which it only does
        // once the updater has the refusal and cancels: the reply must already be there.
        let (engine, _) = fake(Err("nope".into()));
        let (ack, rx) = sync_channel(1);
        let out = commit_shutdown(
            engine,
            || Ok(None),
            &ack,
            || {
                assert!(rx.try_recv().is_ok_and(|r| r.is_err()));
                Err("stop".into())
            },
        );
        assert!(matches!(out, AfterCommit::Failed(_)));
    }

    #[test]
    fn commands_keep_the_restart_settings_current() {
        let mut s = Settings::default();
        track(&mut s, &Command::SetEnabled(true));
        track(&mut s, &Command::SetStrength(oto_engine::Strength::Strong));
        track(&mut s, &Command::SetTier(oto_engine::Tier::Light));
        track(&mut s, &Command::SetInputDevice(Some("usb".into())));
        track(&mut s, &Command::Shutdown);
        assert_eq!(
            s,
            Settings {
                enabled: true,
                input_device: Some("usb".into()),
                strength: oto_engine::Strength::Strong,
                tier: oto_engine::Tier::Light,
            }
        );
    }
}
