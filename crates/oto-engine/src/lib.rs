//! The audio engine: owns devices, the virtual microphone and the realtime thread.
//!
//! The UI and CLI never touch audio directly. They send [`Command`]s and read [`Event`]s, which
//! keeps the engine free to move into its own process later without changing its callers.
//! See PLAN.md for the design.

pub use oto_dsp as dsp;

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
        self.raw.store(raw.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.clean.store(clean.to_bits(), std::sync::atomic::Ordering::Relaxed);
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

/// CONTRACT STUB: the running engine. The real implementation replaces the bodies, not the API.
pub struct Engine {
    levels: std::sync::Arc<Levels>,
    events: std::sync::mpsc::Receiver<Event>,
}

impl Engine {
    /// Start the control thread. Audio only runs while enabled and an app records from Oto Microphone.
    pub fn start(_initial: Settings) -> Result<Self, String> {
        let (_tx, events) = std::sync::mpsc::channel();
        Ok(Self { levels: Default::default(), events })
    }
    pub fn send(&self, _cmd: Command) {}
    /// Non-blocking drain of pending events.
    pub fn try_events(&self) -> Vec<Event> {
        self.events.try_iter().collect()
    }
    pub fn levels(&self) -> std::sync::Arc<Levels> {
        self.levels.clone()
    }
    /// Physical inputs, excluding Oto's own devices.
    pub fn input_devices(&self) -> Vec<InputDevice> {
        Vec::new()
    }
    /// Whether the Oto HAL driver is installed and visible to Core Audio.
    pub fn driver_installed() -> bool {
        false
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
