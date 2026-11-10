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
