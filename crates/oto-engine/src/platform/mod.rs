//! The per-OS seam. The control thread only talks to a [`Platform`]; each OS provides one.
//!
//! - macOS: Core Audio, loopback through the Oto HAL driver (`macos`).
//! - Linux: TODO, a PipeWire backend (capture stream + `Audio/Source` stream on one graph clock;
//!   no drift control needed). It implements the same trait.

use std::sync::Arc;

use oto_dsp::Chain;

use crate::InputDevice;
use crate::pipeline::Shared;

/// OS notifications the control thread reacts to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HalEvent {
    /// Devices appeared or disappeared (includes the driver being installed or removed).
    DeviceList,
    DefaultInput,
    /// The number of apps recording from Oto Microphone may have changed.
    MicClients,
    /// A device we stream with died or changed format: restart the streams.
    StreamDevice,
}

/// Device-level operations, owned by the control thread. Not `Send`: create it on that thread.
pub(crate) trait Platform {
    fn driver_installed(&self) -> bool;
    /// Whether any app records from Oto Microphone.
    fn clients_active(&self) -> bool;
    #[allow(dead_code)]
    fn input_devices(&self) -> Vec<InputDevice>;
    /// The physical mic to capture from: `preferred` if present, else the system default input,
    /// else a sensible fallback. Never an Oto device.
    fn resolve_input(&self, preferred: Option<&str>) -> Option<String>;
    /// Open the mic and the sink and start both callbacks.
    fn start(&mut self, input: &str, chain: Chain, shared: &Arc<Shared>) -> Result<(), String>;
    fn stop(&mut self);
    /// Re-register per-device watches after the device list changed.
    fn refresh(&mut self);
}

/// Device UIDs the backend uses. Overridable for development without the driver installed.
#[derive(Clone, Debug)]
pub(crate) struct Uids {
    pub mic: String,
    pub sink: String,
}

#[cfg(target_os = "macos")]
pub(crate) mod macos;
#[cfg(target_os = "macos")]
pub(crate) use macos as sys;

#[cfg(not(target_os = "macos"))]
pub(crate) mod unsupported;
#[cfg(not(target_os = "macos"))]
pub(crate) use unsupported as sys;
