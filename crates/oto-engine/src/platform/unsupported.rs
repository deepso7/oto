//! Placeholder for platforms without a backend yet (Linux: PipeWire backend is TODO).
//! Keeps the crate building and its portable tests running everywhere.

use std::sync::Arc;
use std::sync::mpsc::Sender;

use oto_dsp::Chain;

use super::{Platform, Uids};
use crate::control::Msg;
use crate::pipeline::Shared;
use crate::{DriverStatus, InputDevice};

const UNSUPPORTED: &str = "no audio backend for this platform yet";

pub(crate) struct Unsupported;

impl Unsupported {
    pub fn new(_uids: Uids, _tx: Sender<Msg>) -> Result<Self, String> {
        Err(UNSUPPORTED.into())
    }
}

impl Platform for Unsupported {
    fn driver_status(&self) -> DriverStatus {
        DriverStatus::Absent
    }
    fn clients_active(&self) -> bool {
        false
    }
    fn input_devices(&self) -> Vec<InputDevice> {
        Vec::new()
    }
    fn resolve_input(&self, _preferred: Option<&str>) -> Option<String> {
        None
    }
    fn start(&mut self, _: &str, _: Chain, _: &Arc<Shared>) -> Result<(), String> {
        Err(UNSUPPORTED.into())
    }
    fn stop(&mut self) {}
    fn refresh(&mut self) {}
}

pub(crate) type Backend = Unsupported;

pub fn input_devices(_uids: &Uids) -> Vec<InputDevice> {
    Vec::new()
}
pub fn driver_status(_uids: &Uids) -> DriverStatus {
    DriverStatus::Absent
}
pub fn default_input_uid() -> Result<Option<String>, String> {
    Ok(None)
}
pub fn is_oto_mic_active_checked(_uids: &Uids) -> Result<Option<bool>, String> {
    Ok(None)
}
pub fn any_audio_active_checked() -> Result<bool, String> {
    Ok(false)
}
pub fn set_default_input(_uid: &str) -> Result<(), String> {
    Err(UNSUPPORTED.into())
}
pub fn device_status(_uid: &str, _input: bool) -> Option<crate::DeviceStatus> {
    None
}
/// No realtime scheduling here yet; the DSP worker runs at normal priority.
pub fn promote_dsp_thread(_period_ns: u64) -> Result<&'static str, String> {
    Err(UNSUPPORTED.into())
}
pub struct Recorder;

impl Recorder {
    pub fn start(
        _uid: &str,
        _f: Box<dyn FnMut(&[f32], usize, u64) + Send>,
    ) -> Result<Self, String> {
        Err(UNSUPPORTED.into())
    }
}

pub fn resolve_physical_input(
    _uids: &Uids,
    _preferred: Option<&str>,
    _last: Option<&str>,
) -> Option<String> {
    None
}
pub fn is_bluetooth(_uid: &str) -> bool {
    false
}
pub fn nominal_rate(_uid: &str) -> Option<f64> {
    None
}
pub struct MonoRecorder;

impl MonoRecorder {
    pub fn start(
        _uid: &str,
        _cancel: &std::sync::atomic::AtomicBool,
        _f: Box<dyn FnMut(&[f32]) + Send>,
    ) -> Result<Self, String> {
        Err(UNSUPPORTED.into())
    }
}

pub struct Player;

impl Player {
    pub fn start<B>(_build: B) -> Result<Self, String>
    where
        B: FnOnce(f64) -> Result<Box<dyn FnMut(&mut [f32]) + Send>, String>,
    {
        Err(UNSUPPORTED.into())
    }
}

pub(crate) fn os_error(e: String) -> String {
    e
}
