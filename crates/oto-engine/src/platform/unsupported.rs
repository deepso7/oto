//! Placeholder for platforms without a backend yet (Linux: PipeWire backend is TODO).
//! Keeps the crate building and its portable tests running everywhere.

use std::sync::Arc;
use std::sync::mpsc::Sender;

use oto_dsp::Chain;

use super::{Platform, Uids};
use crate::InputDevice;
use crate::control::Msg;
use crate::pipeline::Shared;

const UNSUPPORTED: &str = "no audio backend for this platform yet";

pub(crate) struct Unsupported;

impl Unsupported {
    pub fn new(_uids: Uids, _tx: Sender<Msg>) -> Result<Self, String> {
        Err(UNSUPPORTED.into())
    }
}

impl Platform for Unsupported {
    fn driver_installed(&self) -> bool {
        false
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
pub fn driver_installed(_uids: &Uids) -> bool {
    false
}
pub fn default_input_uid() -> Option<String> {
    None
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
