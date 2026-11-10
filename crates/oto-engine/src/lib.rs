//! The audio engine: owns devices, the virtual microphone and the realtime thread.
//!
//! The UI and CLI never touch audio directly. They send [`Command`]s and read [`Event`]s, which
//! keeps the engine free to move into its own process later without changing its callers.
//! See PLAN.md for the design. Threads: the caller's, one control thread (`oto-control`) owning
//! devices and state, and the two Core Audio IO threads (mic, sink) running the realtime path.

pub use oto_dsp as dsp;

mod control;
pub mod latency;
mod pipeline;
mod platform;

pub use control::model_for;
pub use pipeline::Probe;

use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex, mpsc};

/// How hard the denoiser works. Maps to an attenuation limit, not to different models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Strength {
    Light,
    #[default]
    Balanced,
    Strong,
}

/// Which model tier runs. See PLAN.md, "Models".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tier {
    Light,
    #[default]
    Standard,
    Max,
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
    /// Share of one core spent in the audio callback, 0.0 to 1.0.
    Load(f32),
    Error(String),
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
pub struct Engine {
    levels: Arc<Levels>,
    events: mpsc::Receiver<Event>,
    tx: mpsc::Sender<control::Msg>,
    thread: Option<std::thread::JoinHandle<()>>,
    shared: Arc<pipeline::Shared>,
    uids: platform::Uids,
    restore_input: Mutex<Option<String>>,
}

impl Engine {
    /// Start the control thread. Audio only runs while enabled and an app records from Oto Microphone.
    pub fn start(initial: Settings) -> Result<Self, String> {
        Self::start_with(initial, EngineOptions::default())
    }

    /// [`Engine::start`] with test and development hooks.
    pub fn start_with(initial: Settings, opts: EngineOptions) -> Result<Self, String> {
        let levels = Arc::new(Levels::default());
        levels.set(-120.0, -120.0);
        let mut shared = pipeline::Shared::new(levels.clone(), opts.probe.clone());
        shared.source = opts.source.clone();
        let shared = Arc::new(shared);
        let uids = opts.uids();
        let (tx, rx) = mpsc::channel();
        let (etx, events) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let (sh, u, tx2) = (shared.clone(), uids.clone(), tx.clone());
        let force = opts.force_clients_active;
        let thread = std::thread::Builder::new()
            .name("oto-control".into())
            .spawn(move || match platform::sys::Backend::new(u, tx2) {
                Ok(p) => {
                    let _ = ready_tx.send(Ok(()));
                    control::Control::new(p, initial, force, sh, etx).run(rx);
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
            restore_input: Mutex::new(None),
        })
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
    /// Whether the Oto HAL driver is installed and visible to Core Audio.
    pub fn driver_installed() -> bool {
        platform::sys::driver_installed(&EngineOptions::default().uids())
    }

    /// Diagnostics: ring fill, drift correction, xruns, latency estimate.
    pub fn stats(&self) -> Stats {
        Stats::from_shared(&self.shared)
    }

    /// Make Oto Microphone the system default input. The previous default is restored when the
    /// engine is dropped (unless the user picked something else in the meantime).
    pub fn set_default_input_to_oto(&self) -> Result<(), String> {
        let prev = platform::sys::default_input_uid();
        if prev.as_deref() == Some(self.uids.mic.as_str()) {
            return Ok(());
        }
        platform::sys::set_default_input(&self.uids.mic)?;
        *self.restore_input.lock().unwrap_or_else(|e| e.into_inner()) = prev;
        Ok(())
    }

    /// Undo [`Engine::set_default_input_to_oto`] now. Also runs on drop.
    pub fn restore_default_input(&self) -> Result<(), String> {
        let prev = self
            .restore_input
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        match prev {
            Some(uid) if platform::sys::default_input_uid().as_deref() == Some(&self.uids.mic) => {
                platform::sys::set_default_input(&uid)
            }
            _ => Ok(()),
        }
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
    /// Mic-to-sink latency estimated from device-reported latencies, buffers and ring target.
    pub estimated_latency_ms: f32,
}

impl Stats {
    fn from_shared(s: &pipeline::Shared) -> Self {
        let ms = |frames: u32| frames as f32 * 1000.0 / dsp::SAMPLE_RATE as f32;
        Self {
            running: s.running.load(Relaxed),
            input_rate: s.in_rate.load(Relaxed),
            output_rate: s.out_rate.load(Relaxed),
            ring_fill_ms: ms(s.fill_frames.load(Relaxed)),
            ring_target_ms: ms(s.target_frames.load(Relaxed)),
            drift_ppm: s.correction_ppm.load(Relaxed),
            underruns: s.underruns.load(Relaxed),
            resyncs: s.resyncs.load(Relaxed),
            overflows: s.overflows.load(Relaxed),
            estimated_latency_ms: s.estimate_us.load(Relaxed) as f32 / 1000.0,
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
    pub driver_installed: bool,
    pub microphone: Option<DeviceStatus>,
    pub sink: Option<DeviceStatus>,
    pub default_input: Option<String>,
    pub inputs: Vec<InputDevice>,
}

pub fn status() -> Status {
    let uids = EngineOptions::default().uids();
    Status {
        driver_installed: platform::sys::driver_installed(&uids),
        microphone: platform::sys::device_status(&uids.mic, true),
        sink: platform::sys::device_status(&uids.sink, false),
        default_input: platform::sys::default_input_uid(),
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
