//! The audio engine: owns devices, the virtual microphone and the realtime thread.
//!
//! The UI and CLI never touch audio directly. They send [`Command`]s and read [`Event`]s, which
//! keeps the engine free to move into its own process later without changing its callers.
//! See docs/architecture.md for the design. Threads: the caller's, one control thread (`oto-control`) owning
//! devices and state, the two Core Audio IO threads (mic, sink) running the realtime path, and
//! for DeepFilterNet3 a realtime DSP worker (`oto-dsp`) that runs the chain off the callback.

pub use oto_dsp as dsp;

mod control;
/// Fault injection for Voice Isolation's fail-closed paths (`oto fault-check`).
pub mod fault_check;
pub mod latency;
pub mod offline;
/// The realtime path. Public only for the integration tests in `tests/` (allocation counting
/// needs its own test binary); not a stable API.
#[doc(hidden)]
pub mod pipeline;
mod platform;
mod state;
/// "Hear the Difference": record the mic, clean it offline, play raw and cleaned A/B.
pub mod test_capture;
/// Recovery of an app update whose helper died (shared by oto-app and `oto recover-update`).
pub mod update_recovery;
/// On-disk state of an in-progress app update (lock, journal, candidate signals), shared with
/// oto-app and oto-cli.
pub mod update_state;

pub use update_state::{
    AckRecord, CandidateToken, Nonce, TxControlState, TxPhase, UpdatePaths, UpdateTx, UpdatingLock,
};

pub use control::{available_tiers, model_for};

/// Held by tests that pace audio in real time and by CPU-heavy tests, so the latter cannot
/// starve the former's simulated callbacks when the harness runs them in parallel.
#[cfg(test)]
pub(crate) static REALTIME_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
pub use pipeline::{Probe, Stimulus};

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::{Arc, OnceLock, mpsc};

/// The exact error [`Engine::start`] and [`Engine::start_with`] return when another engine
/// (in this or any other process of the same user) is already running. Two engines would both
/// play into the virtual microphone and double the audio. Compare with `==`.
pub const ALREADY_RUNNING: &str = "Oto is already running";

/// The exact error [`Engine::start_with`] returns while an app update is in progress or
/// unresolved (see [`update_state`]), or when an update candidate is no longer admitted.
/// Compare with `==`. `oto-app` exits silently on it; `oto` suggests `oto recover-update`.
pub const UPDATING: &str = "Oto is being updated";

/// The HAL driver revision this build ships (`driver/macos/DRIVER_VERSION`), independent of the
/// app version.
pub const BUNDLED_DRIVER_VERSION: u32 = match u32::from_str_radix(env!("OTO_DRIVER_VERSION"), 10) {
    Ok(v) => v,
    Err(_) => panic!("OTO_DRIVER_VERSION is not a number"),
};

/// The oldest loaded driver revision this engine opens streams with or makes the system default
/// input. Revision 0 is the drivers of v0.0.1 to v0.0.3, which predate the revision property:
/// they are reported as loaded but incompatible, so the app offers to update them.
pub const MIN_COMPATIBLE_DRIVER_VERSION: u32 = 1;

/// The HAL driver as Core Audio sees it right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverStatus {
    /// Neither Oto Microphone nor Oto Sink exists.
    Absent,
    /// Both devices exist; the driver reports this revision (0 for drivers older than the
    /// revision property).
    Loaded(u32),
    /// Could not tell: only one device exists, Core Audio returned an error (e.g. `coreaudiod`
    /// restarting) or the revision did not parse. Transient: ask again later. Never means
    /// "old" or "incompatible".
    Unknown,
}

impl DriverStatus {
    /// Both devices exist (any revision).
    pub fn is_loaded(self) -> bool {
        matches!(self, Self::Loaded(_))
    }

    /// Loaded with a revision this engine works with.
    pub fn is_compatible(self) -> bool {
        matches!(self, Self::Loaded(rev) if driver_compatible(rev, MIN_COMPATIBLE_DRIVER_VERSION))
    }
}

/// `rev >= min`, spelled so a `min` of 0 does not trip "comparison is always true" lints.
pub(crate) fn driver_compatible(rev: u32, min: u32) -> bool {
    rev.checked_sub(min).is_some()
}

/// Parses the driver's `'otvr'` string: a decimal `u32`, nothing else.
pub(crate) fn parse_driver_version(s: &str) -> Option<u32> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// The message for a loaded driver older than [`MIN_COMPATIBLE_DRIVER_VERSION`].
pub fn incompatible_driver_message(rev: u32) -> String {
    format!(
        "Oto's audio driver (revision {rev}) is too old for this version of Oto: update the \
         driver to use Oto Microphone"
    )
}
/// How hard the denoiser works. Maps to an attenuation limit, not to different models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Strength {
    Light,
    #[default]
    Balanced,
    Strong,
}

/// Which model tier runs. See docs/models.md. [`model_for`] maps a tier to the model this
/// build runs for it and [`available_tiers`] lists the tiers that actually differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tier {
    /// RNNoise: the app's Low CPU mode.
    Light,
    /// DeepFilterNet3.
    #[default]
    Standard,
    /// DPDFNet-2: stronger, about 3× DeepFilterNet3's CPU and 20 ms more delay.
    Max,
    /// FastEnhancer-S: the lowest delay of the neural models. Evaluation only (weights
    /// licence unstated upstream).
    Fast,
    /// Voice Isolation (Nearfield v2.3 `fast`): removes background voices as well as noise.
    /// Opt-in, and fails closed: when it cannot run correctly Oto Microphone is silent, never
    /// the raw mic and never another model ([`Event::Degraded`]).
    Isolate,
}

/// Requests from the UI or CLI to the engine.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    SetEnabled(bool),
    /// `None` follows the system default input.
    SetInputDevice(Option<String>),
    SetStrength(Strength),
    SetTier(Tier),
    Shutdown,
}

/// What the engine reports back. Levels do not travel here; they go through lock-free values.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The virtual mic gained or lost its last listener. Inference only runs while `true`.
    ClientsActive(bool),
    DevicesChanged,
    /// Share of one core spent in the audio callbacks and the DSP worker, 0.0 to 1.0.
    Load(f32),
    /// Something is wrong. Stays true until [`Event::Recovered`] (or the next `Error`).
    Error(String),
    /// The loaded driver reports revision `loaded`, older than the `min_required` this engine
    /// runs with: no streams, and Oto Microphone is never made the system default input. Sent
    /// once when the incompatible driver is first seen; [`Event::Recovered`] follows once a
    /// compatible driver loads. See [`incompatible_driver_message`].
    DriverIncompatible {
        loaded: u32,
        min_required: u32,
    },
    /// The last reported [`Event::Error`] no longer applies: audio is flowing normally again,
    /// or nothing needs to run (no app is recording and the driver is present). Sent once per
    /// recovery; UIs clear their error banner on it.
    Recovered,
    /// The tier asks for `requested`, which could not be loaded, so `effective` runs instead.
    /// A notice, not an [`Event::Error`]: audio works. Sent once per failure; the engine does not
    /// retry `requested` until the tier is changed or the engine restarts.
    /// [`Stats::model`] reports what runs.
    ModelFallback {
        requested: dsp::Model,
        effective: dsp::Model,
        reason: String,
    },
    /// Voice Isolation cannot run correctly, so Oto Microphone is silent rather than passing
    /// the raw mic or switching models: the model failed to load (until the tier changes), or
    /// its worker fell behind (until it catches up). Cleared by [`Event::DegradedCleared`],
    /// not by [`Event::Recovered`] (which is about [`Event::Error`]s only).
    Degraded(String),
    /// The last [`Event::Degraded`] no longer applies: the worker caught up, or a mode other
    /// than the one that failed to load was picked. UIs clear their degraded banner on it.
    DegradedCleared,
}

impl Strength {
    /// The denoiser's attenuation limit for this strength, in dB.
    pub fn max_attenuation_db(self) -> f32 {
        match self {
            Strength::Light => 12.0,
            Strength::Balanced => 24.0,
            Strength::Strong => f32::INFINITY,
        }
    }
}

/// Names shared with the macOS HAL plug-in in `driver/macos`. Keep the two in sync.
pub mod names {
    /// The visible virtual input device apps pick.
    pub const MIC_NAME: &str = "Oto Microphone";
    pub const MIC_UID: &str = "dev.oto.microphone";
    /// The hidden output device the engine plays cleaned audio into; the driver loops it to the mic.
    pub const SINK_NAME: &str = "Oto Sink";
    pub const SINK_UID: &str = "dev.oto.sink";
    pub const MANUFACTURER: &str = "Oto";
    /// Bundle name under /Library/Audio/Plug-Ins/HAL.
    pub const DRIVER_BUNDLE: &str = "OtoMicrophone.driver";
}

/// Latest-wins levels the meter reads, in dBFS peak per callback. Written by the engine's
/// metering path, read by the UI at any rate. No locks.
#[derive(Default)]
pub struct Levels {
    raw: std::sync::atomic::AtomicU32,
    clean: std::sync::atomic::AtomicU32,
}

impl Levels {
    pub fn raw(&self) -> f32 {
        f32::from_bits(self.raw.load(std::sync::atomic::Ordering::Relaxed))
    }
    pub fn clean(&self) -> f32 {
        f32::from_bits(self.clean.load(std::sync::atomic::Ordering::Relaxed))
    }
    pub fn set(&self, raw: f32, clean: f32) {
        self.raw
            .store(raw.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.clean
            .store(clean.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// An input device the user can pick.
#[derive(Clone, Debug, PartialEq)]
pub struct InputDevice {
    /// Stable id to pass to [`Command::SetInputDevice`].
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

/// The running engine: a control thread owning devices and state. Dropping it shuts down cleanly
/// (streams stopped, listeners removed) and restores the default input if the engine changed it.
///
/// Only one engine runs per user at a time: starting a second fails with [`ALREADY_RUNNING`].
/// Queries that open no streams ([`list_input_devices`], [`status`], [`Engine::driver_status`],
/// `dsp::bench`) are not affected.
///
/// While an app update is in progress no engine starts ([`UPDATING`]), except the update's
/// designated new build ([`EngineOptions::update_candidate`]), which starts inert until
/// [`Engine::commit_candidate`].
pub struct Engine {
    levels: Arc<Levels>,
    events: mpsc::Receiver<Event>,
    tx: mpsc::Sender<control::Msg>,
    thread: Option<std::thread::JoinHandle<()>>,
    shared: Arc<pipeline::Shared>,
    uids: platform::Uids,
    /// Opened at start, or for an update candidate at [`Engine::commit_candidate`].
    restore: Arc<OnceLock<state::RestoreTarget>>,
    state_dir: Option<PathBuf>,
    /// An uncommitted update candidate: no streams, no change to the default input.
    inert: Arc<AtomicBool>,
    /// The oldest driver revision accepted; 0 when the device UIDs are overridden.
    min_driver: u32,
    /// Released when the engine is dropped (after the streams have stopped).
    lock: Option<state::InstanceLock>,
}

/// Update admission (see [`update_state`]), checked before and again after taking
/// `engine.lock`.
fn admit(paths: Option<&UpdatePaths>, candidate: Option<&CandidateToken>) -> Result<(), String> {
    match (paths, candidate) {
        (Some(p), Some(tok)) if update_state::candidate_admitted(p, tok) => Ok(()),
        (_, Some(_)) => Err(UPDATING.into()),
        (Some(p), None) if update_state::update_blocked(p).is_some() => Err(UPDATING.into()),
        (_, None) => Ok(()),
    }
}

/// Admission, the instance lock, admission again. `between` runs right after the lock is taken
/// (tests use it to change the update state in that window). On a failed re-check the lock is
/// released before returning.
fn admit_and_lock(
    dir: Option<&Path>,
    candidate: Option<&CandidateToken>,
    between: impl FnOnce(),
) -> Result<Option<state::InstanceLock>, String> {
    let paths = dir.map(UpdatePaths::new);
    admit(paths.as_ref(), candidate)?;
    let lock = state::InstanceLock::acquire(dir)?;
    if candidate.is_some() && lock.is_none() {
        return Err("an update candidate needs the instance lock (engine.lock)".into());
    }
    between();
    if let Err(e) = admit(paths.as_ref(), candidate) {
        drop(lock);
        return Err(e);
    }
    Ok(lock)
}

/// Opens the restore target and puts back a default input a crashed run left behind. A failure
/// keeps the target for a later retry.
fn open_restore(dir: Option<&Path>, mic: &str) -> state::RestoreTarget {
    let restore = state::RestoreTarget::open(dir, mic, platform::sys::default_input_uid());
    let _ = restore.restore(platform::sys::default_input_uid, |uid| {
        platform::sys::set_default_input(uid)
    });
    restore
}

impl Engine {
    /// Start the control thread. Audio only runs while enabled and an app records from Oto
    /// Microphone. Fails with exactly [`ALREADY_RUNNING`] if another engine holds the instance
    /// lock.
    ///
    /// If a previous run made Oto Microphone the default input and died without restoring it,
    /// this engine puts the previous microphone back at once. Callers that want Oto as the
    /// default ask for it again.
    pub fn start(initial: Settings) -> Result<Self, String> {
        Self::start_with(initial, EngineOptions::default())
    }

    /// [`Engine::start`] with test and development hooks.
    ///
    /// Update admission: fails with exactly [`UPDATING`] while an update holds `updating.lock`
    /// or its journal is unresolved, or, for an [`EngineOptions::update_candidate`], unless its
    /// `tx-<nonce>-state` is `active`. Checked before and again after taking the instance lock.
    /// A candidate also needs a real instance lock, and starts inert: no streams open and the
    /// default input is not touched until [`Engine::commit_candidate`].
    pub fn start_with(initial: Settings, opts: EngineOptions) -> Result<Self, String> {
        let dir = opts.state_dir.clone().or_else(state::default_dir);
        let candidate = opts.update_candidate.as_ref();
        let lock = admit_and_lock(dir.as_deref(), candidate, || {})?;
        let inert = Arc::new(AtomicBool::new(candidate.is_some()));
        let min_driver = if opts.mic_uid.is_some() || opts.sink_uid.is_some() {
            0
        } else {
            MIN_COMPATIBLE_DRIVER_VERSION
        };
        let levels = Arc::new(Levels::default());
        levels.set(-120.0, -120.0);
        let mut shared = pipeline::Shared::new(levels.clone(), opts.probe.clone());
        shared.source = opts.source.clone();
        let shared = Arc::new(shared);
        let uids = opts.uids();
        let (tx, rx) = mpsc::channel();
        let (etx, events) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let restore = Arc::new(OnceLock::new());
        if candidate.is_none() {
            // Anything pending here is left over from a run that died.
            let _ = restore.set(open_restore(dir.as_deref(), &uids.mic));
        }
        let (sh, u, tx2) = (shared.clone(), uids.clone(), tx.clone());
        let force = opts.force_clients_active;
        let r = restore.clone();
        let inert2 = inert.clone();
        // Runs on the control thread when the system default input changes.
        let on_default: control::Hook = Box::new(move || {
            if let Some(r) = r.get() {
                r.default_changed(platform::sys::default_input_uid)
            }
        });
        let thread = std::thread::Builder::new()
            .name("oto-control".into())
            .spawn(move || match platform::sys::Backend::new(u, tx2) {
                Ok(p) => {
                    let _ = ready_tx.send(Ok(()));
                    control::Control::new(p, initial, force, sh, etx)
                        .on_default_input(on_default)
                        .with_min_driver(min_driver)
                        .with_inert(inert2)
                        .run(rx);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|e| format!("cannot spawn control thread: {e}"))?;
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err("control thread died during start".into()),
        }
        Ok(Self {
            levels,
            events,
            tx,
            thread: Some(thread),
            shared,
            uids,
            restore,
            state_dir: dir,
            inert,
            min_driver,
            lock,
        })
    }

    /// Whether this engine holds the per-user instance lock (`engine.lock`). Without it (no
    /// usable state directory) it runs unguarded, and self-update is not offered.
    pub fn has_instance_lock(&self) -> bool {
        self.lock.is_some()
    }

    /// Whether this is an update candidate that has not committed yet.
    pub fn is_candidate_inert(&self) -> bool {
        self.inert.load(Acquire)
    }

    /// Ends an update candidate's inert phase once the update is durable: adopts the restore
    /// target (and any default input a crashed run left to restore), then lets the control
    /// thread open streams for apps already recording. A no-op for engines that are not
    /// inert, so calling it twice is fine.
    pub fn commit_candidate(&self) -> Result<(), String> {
        if !self.inert.load(Acquire) {
            return Ok(());
        }
        self.restore
            .get_or_init(|| open_restore(self.state_dir.as_deref(), &self.uids.mic));
        self.inert.store(false, Release);
        self.tx
            .send(control::Msg::Reconcile)
            .map_err(|_| "the engine's control thread is gone".to_string())
    }

    /// Stops the engine now, synchronously: streams stopped, control thread joined, the default
    /// input restored if the engine changed it. Unlike dropping, reports whether the restore
    /// worked (on `Err` the target is kept on disk for the next run to retry). The instance lock
    /// is released when this returns.
    pub fn shutdown_and_restore(mut self) -> Result<(), String> {
        self.send(Command::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.restore_default_input()
    }

    pub fn send(&self, cmd: Command) {
        let _ = self.tx.send(control::Msg::Cmd(cmd));
    }
    /// Non-blocking drain of pending events.
    pub fn try_events(&self) -> Vec<Event> {
        self.events.try_iter().collect()
    }
    pub fn levels(&self) -> Arc<Levels> {
        self.levels.clone()
    }
    /// Physical inputs, excluding Oto's own devices.
    pub fn input_devices(&self) -> Vec<InputDevice> {
        platform::sys::input_devices(&self.uids)
    }
    /// The Oto HAL driver as Core Audio sees it right now, with its revision.
    pub fn driver_status() -> DriverStatus {
        platform::sys::driver_status(&EngineOptions::default().uids())
    }

    /// Whether some app records from Oto Microphone: `Ok(None)` if there is no Oto
    /// Microphone, `Err` if Core Audio could not be asked.
    pub fn is_oto_mic_active_checked() -> Result<Option<bool>, String> {
        platform::sys::is_oto_mic_active_checked(&EngineOptions::default().uids())
            .map_err(platform::sys::os_error)
    }

    /// Whether any audio device is in use by any process. `Err` if any device could not be
    /// asked: "idle" is only ever a certain answer.
    pub fn any_audio_active_checked() -> Result<bool, String> {
        platform::sys::any_audio_active_checked().map_err(platform::sys::os_error)
    }

    /// Diagnostics: ring fill, drift correction, xruns, latency estimate.
    pub fn stats(&self) -> Stats {
        Stats::from_shared(&self.shared)
    }

    /// Make Oto Microphone the system default input. The previous default is restored when the
    /// engine is dropped (unless the user picked something else in the meantime). It is also
    /// written to the state directory first, so that if this process dies without restoring,
    /// the next engine to start adopts it. Does file I/O: not for realtime threads.
    ///
    /// Refused while an update candidate is inert ([`UPDATING`]), and unless the loaded driver
    /// is compatible ([`DriverStatus::is_compatible`]; not checked when the device UIDs are
    /// overridden).
    pub fn set_default_input_to_oto(&self) -> Result<(), String> {
        let Some(restore) = self.restore.get().filter(|_| !self.inert.load(Acquire)) else {
            return Err(UPDATING.into());
        };
        match platform::sys::driver_status(&self.uids) {
            DriverStatus::Loaded(rev) if driver_compatible(rev, self.min_driver) => {}
            DriverStatus::Loaded(rev) => return Err(incompatible_driver_message(rev)),
            DriverStatus::Absent => {
                return Err("Oto driver is not installed: Oto Microphone is not available".into());
            }
            DriverStatus::Unknown => {
                return Err("cannot tell whether the Oto driver is loaded; try again".into());
            }
        }
        restore.switch_to_oto(platform::sys::default_input_uid(), |mic| {
            platform::sys::set_default_input(mic)
        })
    }

    /// Undo [`Engine::set_default_input_to_oto`] (from this run or adopted from a crashed one)
    /// now. Also runs on drop. If putting the old default back fails (e.g. it is unplugged),
    /// the target is kept and a later call or run retries.
    pub fn restore_default_input(&self) -> Result<(), String> {
        let Some(restore) = self.restore.get() else {
            return Ok(());
        };
        restore.restore(platform::sys::default_input_uid, |uid| {
            platform::sys::set_default_input(uid)
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.send(Command::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let _ = self.restore_default_input();
    }
}

/// Test and development hooks for [`Engine::start_with`]. The defaults are what the app uses.
#[derive(Clone, Default)]
pub struct EngineOptions {
    /// Replace the mic signal with a tone-burst train and record injection times (`oto latency`).
    /// The real mic is still opened as the clock, so the measurement covers its IO buffer.
    pub probe: Option<Arc<Probe>>,
    /// Use another device as "Oto Microphone" (its clients are watched). Development only.
    pub mic_uid: Option<String>,
    /// Use another output device as "Oto Sink". Development only.
    pub sink_uid: Option<String>,
    /// Behave as if an app is always recording from the mic.
    pub force_clients_active: bool,
    /// Development: loop these mono samples in place of the mic signal, at the mic's rate.
    /// The real mic still clocks the stream. Used to prove the full path with a known signal.
    pub source: Option<Arc<[f32]>>,
    /// Where the instance lock and the pending default-input restore live. `None` (the default)
    /// is `~/Library/Application Support/oto` on macOS and `$XDG_STATE_HOME/oto` (or
    /// `~/.local/state/oto`) elsewhere. Tests point it at a temporary directory.
    pub state_dir: Option<PathBuf>,
    /// This process is the designated new build of an in-progress app update. The engine starts
    /// only while the update's `tx-<nonce>-state` is `active`, needs the instance lock, and
    /// stays inert until [`Engine::commit_candidate`].
    pub update_candidate: Option<CandidateToken>,
}

impl EngineOptions {
    fn uids(&self) -> platform::Uids {
        platform::Uids {
            mic: self
                .mic_uid
                .clone()
                .unwrap_or_else(|| names::MIC_UID.into()),
            sink: self
                .sink_uid
                .clone()
                .unwrap_or_else(|| names::SINK_UID.into()),
        }
    }
}

/// A snapshot of the realtime path, for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    /// Streams are open (an app is recording and the driver is present).
    pub running: bool,
    /// Times streams have been started since the engine launched (> 1 indicates a restart).
    pub stream_starts: u64,
    pub input_rate: u32,
    pub output_rate: u32,
    /// Current ring fill and its target, ms at 48 kHz.
    pub ring_fill_ms: f32,
    pub ring_target_ms: f32,
    /// Drift correction applied by the output resampler, ppm (+ = consuming faster).
    pub drift_ppm: i32,
    pub underruns: u64,
    pub resyncs: u64,
    pub overflows: u64,
    /// Capture blocks the DSP worker had not processed in time, so the dry signal (at the same
    /// delay, crossfaded) filled in. Always 0 for models that run inline.
    pub starvations: u64,
    /// Times the DSP worker fell a whole ring (≈340 ms) behind and its backlog was dropped.
    pub worker_overflows: u64,
    /// Voice Isolation's worker is starved or restarting right now, so the output is silence.
    pub degraded: bool,
    /// Mic-to-sink latency estimated from device-reported latencies, buffers and ring target.
    /// Includes the running model's delay and the worker reserve.
    pub estimated_latency_ms: f32,
    /// The reserve between the capture callback and the DSP worker, ms. 0 when the model runs
    /// inside the capture callback (RNNoise).
    pub worker_reserve_ms: f32,
    /// The DSP worker got realtime (time-constraint) scheduling.
    pub worker_realtime: bool,
    /// The DSP worker joined the mic's IO-thread audio workgroup, so the scheduler runs it on
    /// the audio deadline (on Apple Silicon: performance cores at speed).
    pub worker_workgroup: bool,
    /// The model the running streams use (after any fallback), `None` when not running.
    pub model: Option<dsp::Model>,
    /// Callback timing of the capture (mic) stream, since the streams last started.
    pub capture: CallbackStats,
    /// Callback timing of the output (Oto Sink) stream, since the streams last started.
    pub output: CallbackStats,
    /// Processing time per capture block on the DSP worker, since the streams last started;
    /// `period_us` is the capture period it has per block and `first_us` its priming. All zero
    /// when the model runs inline.
    pub worker: CallbackStats,
}

/// How long one stream's audio callback takes, since the streams last started. A callback
/// that takes longer than `period_us` misses its deadline. Percentiles come from a log-scale
/// histogram and err high by at most 25%. All zero before the stream has run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CallbackStats {
    /// Callbacks seen, including the first.
    pub callbacks: u64,
    /// The device IO buffer duration: the time each callback has.
    pub period_us: f32,
    pub p50_us: f32,
    pub p99_us: f32,
    pub p999_us: f32,
    pub max_us: f32,
    /// Callbacks past the first that took longer than `period_us`.
    pub deadline_misses: u64,
    /// The first callback, which primes the model on the audio thread. Not included in the
    /// percentiles or the maximum.
    pub first_us: f32,
}

impl CallbackStats {
    fn from_times(t: &pipeline::CallbackTimes) -> Self {
        let us = |ns: u64| ns as f32 / 1000.0;
        Self {
            callbacks: t.calls(),
            period_us: us(t.period_ns.load(Relaxed)),
            p50_us: t.quantile_ns(0.5).map_or(0.0, us),
            p99_us: t.quantile_ns(0.99).map_or(0.0, us),
            p999_us: t.quantile_ns(0.999).map_or(0.0, us),
            max_us: us(t.max_ns()),
            deadline_misses: t.deadline_misses(),
            first_us: us(t.first_ns()),
        }
    }
}

impl Stats {
    fn from_shared(s: &pipeline::Shared) -> Self {
        let ms = |frames: u32| frames as f32 * 1000.0 / dsp::SAMPLE_RATE as f32;
        Self {
            running: s.running.load(Relaxed),
            stream_starts: s.stream_starts.load(Relaxed),
            input_rate: s.in_rate.load(Relaxed),
            output_rate: s.out_rate.load(Relaxed),
            ring_fill_ms: ms(s.fill_frames.load(Relaxed)),
            ring_target_ms: ms(s.target_frames.load(Relaxed)),
            drift_ppm: s.correction_ppm.load(Relaxed),
            underruns: s.underruns.load(Relaxed),
            resyncs: s.resyncs.load(Relaxed),
            overflows: s.overflows.load(Relaxed),
            starvations: s.starvations.load(Relaxed),
            worker_overflows: s.worker_overflows.load(Relaxed),
            degraded: s.degraded.load(Relaxed),
            estimated_latency_ms: s.estimate_us.load(Relaxed) as f32 / 1000.0,
            worker_reserve_ms: ms(s.reserve_frames.load(Relaxed)),
            worker_realtime: s.worker_realtime.load(Relaxed),
            worker_workgroup: s.worker_workgroup.load(Relaxed),
            model: pipeline::decode_model(s.model.load(Relaxed)),
            capture: CallbackStats::from_times(&s.capture_times),
            output: CallbackStats::from_times(&s.output_times),
            worker: CallbackStats::from_times(&s.worker_times),
        }
    }
}

/// Physical inputs, excluding Oto's own devices, without starting an engine.
pub fn list_input_devices() -> Vec<InputDevice> {
    platform::sys::input_devices(&EngineOptions::default().uids())
}

/// State of one of Oto's virtual devices.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceStatus {
    pub name: String,
    pub sample_rate: f64,
    pub channels: u32,
    /// Some process is doing IO on it (for the mic: an app is recording).
    pub in_use: bool,
}

/// Driver and device overview for `oto status`.
#[derive(Clone, Debug, PartialEq)]
pub struct Status {
    pub driver: DriverStatus,
    pub microphone: Option<DeviceStatus>,
    pub sink: Option<DeviceStatus>,
    pub default_input: Option<String>,
    pub inputs: Vec<InputDevice>,
}

pub fn status() -> Status {
    let uids = EngineOptions::default().uids();
    let driver = platform::sys::driver_status(&uids);
    Status {
        driver,
        microphone: platform::sys::device_status(&uids.mic, true),
        sink: platform::sys::device_status(&uids.sink, false),
        default_input: platform::sys::default_input_uid().ok().flatten(),
        inputs: platform::sys::input_devices(&uids),
    }
}

/// Initial state for [`Engine::start`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Settings {
    pub enabled: bool,
    pub input_device: Option<String>,
    pub strength: Strength,
    pub tier: Tier,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::tests::TempDir;
    use crate::update_state::*;

    fn tx(nonce: &Nonce, phase: TxPhase) -> UpdateTx {
        UpdateTx {
            nonce: nonce.clone(),
            phase,
            started_at: 0,
            stage1_deadline: None,
            stage2_deadline: None,
            commit_deadline: None,
            target_path: "/Applications/Oto.app".into(),
            staging_path: "/Applications/.Oto-update-n".into(),
            old_version: "0.0.3".into(),
            new_version: "0.0.4".into(),
            helper_path: "/tmp/h".into(),
            candidate_pid: None,
            relaunch_pending: false,
        }
    }

    fn lock_is_free(dir: &Path) -> bool {
        state::InstanceLock::acquire(Some(dir)).is_ok_and(|l| l.is_some())
    }

    #[test]
    fn driver_versions() {
        const { assert!(BUNDLED_DRIVER_VERSION >= 1) };
        assert!(driver_compatible(
            BUNDLED_DRIVER_VERSION,
            MIN_COMPATIBLE_DRIVER_VERSION
        ));
        assert!(driver_compatible(0, 0));
        assert!(driver_compatible(3, 2));
        assert!(!driver_compatible(1, 2));
        assert!(DriverStatus::Loaded(BUNDLED_DRIVER_VERSION).is_compatible());
        // Legacy drivers (no 'otvr') are revision 0: loaded, but too old.
        assert_eq!(MIN_COMPATIBLE_DRIVER_VERSION, 1);
        assert!(DriverStatus::Loaded(0).is_loaded());
        assert!(!DriverStatus::Loaded(0).is_compatible());
        // Unknown and Absent are never loaded or "compatible", and never folded into 0.
        for s in [DriverStatus::Unknown, DriverStatus::Absent] {
            assert!(!s.is_loaded());
            assert!(!s.is_compatible());
        }
        assert_eq!(parse_driver_version("1"), Some(1));
        assert_eq!(parse_driver_version("42"), Some(42));
        for bad in ["", " 1", "1 ", "-1", "+1", "1.0", "x", "99999999999"] {
            assert_eq!(parse_driver_version(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn normal_engines_wait_for_updates() {
        let t = TempDir::new();
        let p = UpdatePaths::new(&t.0);
        let n = Nonce::parse("n").unwrap();
        // Nothing going on.
        let l = admit_and_lock(Some(&t.0), None, || {}).unwrap();
        assert!(l.is_some());
        drop(l);
        // updating.lock held: refused before engine.lock is taken.
        let ul = UpdatingLock::try_acquire(&p).unwrap().unwrap();
        assert_eq!(
            admit_and_lock(Some(&t.0), None, || panic!("must not lock"))
                .err()
                .as_deref(),
            Some(UPDATING)
        );
        drop(ul);
        // Unresolved journal: refused.
        for phase in [TxPhase::Swapping, TxPhase::Verifying, TxPhase::RollingBack] {
            write_tx(&p, &tx(&n, phase)).unwrap();
            assert_eq!(
                admit_and_lock(Some(&t.0), None, || panic!("must not lock"))
                    .err()
                    .as_deref(),
                Some(UPDATING),
                "{phase:?}"
            );
        }
        // Prepared and terminal phases do not block.
        for phase in [TxPhase::Prepared, TxPhase::Committed, TxPhase::RolledBack] {
            write_tx(&p, &tx(&n, phase)).unwrap();
            assert!(admit_and_lock(Some(&t.0), None, || {}).is_ok(), "{phase:?}");
        }
        remove_tx(&p).unwrap();
        assert!(lock_is_free(&t.0));
    }

    #[test]
    fn normal_engines_recheck_after_taking_the_lock() {
        let t = TempDir::new();
        let p = UpdatePaths::new(&t.0);
        let n = Nonce::parse("n").unwrap();
        // An update helper takes updating.lock between the check and engine.lock.
        let held = std::cell::RefCell::new(None);
        let r = admit_and_lock(Some(&t.0), None, || {
            *held.borrow_mut() = UpdatingLock::try_acquire(&p).unwrap();
        });
        assert_eq!(r.err().as_deref(), Some(UPDATING));
        assert!(lock_is_free(&t.0), "engine.lock released on refusal");
        drop(held);
        // Or a journal turns unresolved in that window.
        let r = admit_and_lock(Some(&t.0), None, || {
            write_tx(&p, &tx(&n, TxPhase::Swapping)).unwrap();
        });
        assert_eq!(r.err().as_deref(), Some(UPDATING));
        assert!(lock_is_free(&t.0));
    }

    #[test]
    fn candidates_need_an_active_transaction_and_a_real_lock() {
        let t = TempDir::new();
        let p = UpdatePaths::new(&t.0);
        let tok = CandidateToken::new(Nonce::parse("cand").unwrap());
        // No control file, or not active: refused.
        assert_eq!(
            admit_and_lock(Some(&t.0), Some(&tok), || panic!("must not lock"))
                .err()
                .as_deref(),
            Some(UPDATING)
        );
        for s in [TxControlState::Revoked, TxControlState::Committed] {
            write_tx_state(&p, &tok.nonce, s).unwrap();
            assert_eq!(
                admit_and_lock(Some(&t.0), Some(&tok), || panic!("must not lock"))
                    .err()
                    .as_deref(),
                Some(UPDATING)
            );
        }
        // Active: admitted even though the helper holds updating.lock and the journal is
        // unresolved (that is the transaction it belongs to).
        write_tx_state(&p, &tok.nonce, TxControlState::Active).unwrap();
        write_tx(&p, &tx(&tok.nonce, TxPhase::Verifying)).unwrap();
        let ul = UpdatingLock::try_acquire(&p).unwrap().unwrap();
        let l = admit_and_lock(Some(&t.0), Some(&tok), || {}).unwrap();
        assert!(l.is_some());
        assert_eq!(read_lock_holder_pid(&p), Some(std::process::id()));
        drop(l);
        // Revoked between the check and engine.lock: refused, lock released.
        let r = admit_and_lock(Some(&t.0), Some(&tok), || {
            write_tx_state(&p, &tok.nonce, TxControlState::Revoked).unwrap();
        });
        assert_eq!(r.err().as_deref(), Some(UPDATING));
        assert!(lock_is_free(&t.0));
        drop(ul);
        // No state directory: no lock to hold, so no candidate.
        assert!(admit_and_lock(None, Some(&tok), || {}).is_err());
        // Another engine holds engine.lock: refused as already running.
        write_tx_state(&p, &tok.nonce, TxControlState::Active).unwrap();
        let other = state::InstanceLock::acquire(Some(&t.0)).unwrap();
        assert_eq!(
            admit_and_lock(Some(&t.0), Some(&tok), || {})
                .err()
                .as_deref(),
            Some(ALREADY_RUNNING)
        );
        drop(other);
    }

    /// The real engine on this Mac (Core Audio listeners, no streams): an inert candidate
    /// takes the lock but leaves the default input and the restore target alone.
    #[cfg(target_os = "macos")]
    #[test]
    fn inert_candidate_touches_no_restore_state_until_commit() {
        let t = TempDir::new();
        let p = UpdatePaths::new(&t.0);
        let tok = CandidateToken::new(Nonce::parse("live").unwrap());
        write_tx_state(&p, &tok.nonce, TxControlState::Active).unwrap();
        // A leftover target from a crashed run, pointing at a device that does not exist.
        std::fs::write(p.restore_input(), "dev.oto.test.nonexistent\n").unwrap();
        let opts = EngineOptions {
            state_dir: Some(t.0.clone()),
            update_candidate: Some(tok.clone()),
            ..Default::default()
        };
        let e = Engine::start_with(Settings::default(), opts).unwrap();
        assert!(e.has_instance_lock());
        assert!(e.is_candidate_inert());
        assert_eq!(
            e.set_default_input_to_oto().err().as_deref(),
            Some(UPDATING)
        );
        assert!(e.restore.get().is_none());
        assert!(
            p.restore_input().exists(),
            "restore target untouched while inert"
        );
        // A second, normal engine is refused while the candidate runs.
        let normal = EngineOptions {
            state_dir: Some(t.0.clone()),
            ..Default::default()
        };
        assert_eq!(
            Engine::start_with(Settings::default(), normal)
                .err()
                .as_deref(),
            Some(ALREADY_RUNNING)
        );
        e.commit_candidate().unwrap();
        assert!(!e.is_candidate_inert());
        assert!(e.restore.get().is_some());
        e.commit_candidate().unwrap();
        e.shutdown_and_restore().ok();
        assert!(lock_is_free(&t.0));
    }
}
