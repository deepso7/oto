//! The app's state and everything that changes it. No Rune views and no I/O beyond the hooks
//! [`Hooks`] names, so it unit-tests without a window or an engine.

use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;

use oto_engine::{Command, Event, InputDevice, Strength, Tier};

use crate::meter::Meter;
use crate::prefs::Prefs;
use crate::tray::Tray;

/// Shows the "Hear the difference" test. Off until the engine can record and play back.
pub const HEAR_TEST: bool = false;

pub const STRENGTHS: [(Strength, &str); 3] = [
    (Strength::Light, "Light"),
    (Strength::Balanced, "Balanced"),
    (Strength::Strong, "Strong"),
];
pub const TIERS: [(Tier, &str, &str); 3] = [
    (Tier::Light, "Light", "Lowest CPU, good for older Macs."),
    (
        Tier::Standard,
        "Standard",
        "The best balance of quality and CPU.",
    ),
    (Tier::Max, "Max", "Highest quality, several times the CPU."),
];
pub const SYSTEM_DEFAULT: &str = "System default";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    Main,
    Settings,
}

/// The driver install, from the button press to the result.
#[derive(Clone, Debug, PartialEq)]
pub enum Install {
    Idle,
    Running,
    Failed(String),
}

/// Which popup select is open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Open {
    Device,
    Tier,
}

/// Where commands go.
pub enum Link {
    /// Snapshots: nowhere.
    None,
    /// The engine host thread.
    Engine(Sender<Command>),
    /// Tests: kept for inspection.
    #[cfg_attr(not(test), allow(dead_code))]
    Record(Vec<Command>),
}

impl Link {
    fn send(&mut self, cmd: Command) {
        match self {
            Link::None => {}
            Link::Engine(tx) => {
                let _ = tx.send(cmd);
            }
            Link::Record(v) => v.push(cmd),
        }
    }
}

/// Side effects the live app performs and tests and snapshots skip.
#[derive(Clone, Copy)]
pub struct Hooks {
    pub save: fn(&Prefs) -> Result<(), String>,
    pub set_login: fn(bool) -> Result<(), String>,
    /// Starts the install off the UI thread; the result comes back through [`State::installed`].
    pub install: fn(),
}

impl Hooks {
    pub const NONE: Hooks = Hooks {
        save: |_| Ok(()),
        set_login: |_| Ok(()),
        install: || {},
    };
}

pub struct State {
    pub prefs: Prefs,
    pub driver_installed: bool,
    /// "Not now" on onboarding: show the main screen without a driver.
    pub skipped_onboarding: bool,
    pub install: Install,
    pub devices: Vec<InputDevice>,
    /// An app is recording from Oto Microphone.
    pub clients_active: bool,
    /// Share of one core the audio callback uses, while active.
    pub load: Option<f32>,
    pub error: Option<String>,
    pub page: Page,
    pub open: Option<Open>,
    pub meter: Meter,
    pub link: Link,
    pub hooks: Hooks,
    pub tray: Option<Tray>,
    /// The window is a popover under the status item: it hides when it loses focus.
    pub popover: bool,
    /// The window's current content size, for placing it under the status item.
    pub size: (f32, f32),
    /// When the popover last hid for losing focus.
    pub hidden_at: Option<std::time::Instant>,
}

impl State {
    pub fn new(prefs: Prefs, driver_installed: bool) -> State {
        State {
            prefs,
            driver_installed,
            skipped_onboarding: false,
            install: Install::Idle,
            devices: Vec::new(),
            clients_active: false,
            load: None,
            error: None,
            page: Page::Main,
            open: None,
            meter: Meter::default(),
            link: Link::None,
            hooks: Hooks::NONE,
            tray: None,
            popover: false,
            size: (crate::view::WIDTH, 400.0),
            hidden_at: None,
        }
    }

    /// Onboarding shows until the driver is in (or the user skips it).
    pub fn onboarding(&self) -> bool {
        !self.driver_installed && !self.skipped_onboarding
    }

    fn send(&mut self, cmd: Command) {
        self.link.send(cmd);
    }

    fn save(&mut self) {
        if let Err(e) = (self.hooks.save)(&self.prefs) {
            self.error = Some(e);
        }
    }

    fn meter_gate(&self) {
        self.meter
            .active
            .store(self.prefs.enabled && self.clients_active, Ordering::Relaxed);
    }

    pub fn set_enabled(&mut self, on: bool) {
        if self.prefs.enabled == on {
            return;
        }
        self.prefs.enabled = on;
        self.send(Command::SetEnabled(on));
        self.save();
        self.meter_gate();
        if let Some(t) = &self.tray {
            t.set_enabled(on);
        }
    }

    pub fn toggle(&mut self) {
        self.set_enabled(!self.prefs.enabled);
    }

    pub fn set_strength(&mut self, s: Strength) {
        if self.prefs.strength != s {
            self.prefs.strength = s;
            self.send(Command::SetStrength(s));
            self.save();
        }
    }

    pub fn set_tier(&mut self, t: Tier) {
        self.open = None;
        if self.prefs.tier != t {
            self.prefs.tier = t;
            self.send(Command::SetTier(t));
            self.save();
        }
    }

    pub fn set_start_at_login(&mut self, on: bool) {
        match (self.hooks.set_login)(on) {
            Ok(()) => {
                self.prefs.start_at_login = on;
                self.save();
            }
            Err(e) => self.error = Some(e),
        }
    }

    /// The device picker's rows: `(device id, label)`, "System default" first. Labels are
    /// unique, so the select's chosen label maps back to one device.
    pub fn device_options(&self) -> Vec<(Option<String>, String)> {
        let default = match self.devices.iter().find(|d| d.is_default) {
            Some(d) => format!("{SYSTEM_DEFAULT} ({})", d.name),
            None => SYSTEM_DEFAULT.to_owned(),
        };
        let mut out = vec![(None, default)];
        let mut push = |id: String, name: &str| {
            let mut label = name.to_owned();
            let mut n = 2;
            while out.iter().any(|(_, l)| *l == label) {
                label = format!("{name} {n}");
                n += 1;
            }
            out.push((Some(id), label));
        };
        for d in &self.devices {
            push(d.id.clone(), &d.name);
        }
        // A picked device that's unplugged stays listed, so the choice still shows.
        if let Some(id) = &self.prefs.input_device
            && !self.devices.iter().any(|d| &d.id == id)
        {
            let name = self
                .prefs
                .input_device_name
                .clone()
                .unwrap_or_else(|| id.clone());
            push(id.clone(), &format!("{name} (not connected)"));
        }
        out
    }

    pub fn device_label(&self) -> String {
        let opts = self.device_options();
        opts.iter()
            .find(|(id, _)| *id == self.prefs.input_device)
            .map_or_else(|| opts[0].1.clone(), |(_, l)| l.clone())
    }

    pub fn choose_device(&mut self, label: &str) {
        self.open = None;
        let Some((id, _)) = self.device_options().into_iter().find(|(_, l)| l == label) else {
            return;
        };
        if id == self.prefs.input_device {
            return;
        }
        self.prefs.input_device_name = id
            .as_ref()
            .and_then(|id| self.devices.iter().find(|d| &d.id == id))
            .map(|d| d.name.clone());
        self.prefs.input_device = id.clone();
        self.send(Command::SetInputDevice(id));
        self.save();
    }

    pub fn toggle_open(&mut self, which: Open) {
        self.open = if self.open == Some(which) {
            None
        } else {
            Some(which)
        };
    }

    pub fn set_devices(&mut self, devices: Vec<InputDevice>) {
        self.devices = devices;
    }

    pub fn on_event(&mut self, e: Event) {
        match e {
            Event::ClientsActive(on) => {
                self.clients_active = on;
                if !on {
                    self.load = None;
                }
                self.meter_gate();
            }
            Event::Load(l) => self.load = Some(l.clamp(0.0, 1.0)),
            Event::Error(msg) => self.error = Some(msg),
            // The host thread re-reads the device list and calls `set_devices`.
            Event::DevicesChanged => {}
        }
    }

    /// Quitting: the engine restores the default input and stops.
    pub fn shutdown(&mut self) {
        self.send(Command::Shutdown);
    }

    pub fn start_install(&mut self) {
        if self.install != Install::Running {
            self.install = Install::Running;
            (self.hooks.install)();
        }
    }

    /// The install finished; `driver_installed` is what Core Audio reports now.
    pub fn installed(&mut self, outcome: crate::driver::Outcome, driver_installed: bool) {
        use crate::driver::Outcome;
        self.driver_installed = driver_installed;
        self.install = match outcome {
            Outcome::Installed if !driver_installed => Install::Failed(
                "The installer finished, but Core Audio doesn't list Oto Microphone yet.".into(),
            ),
            Outcome::Installed | Outcome::Cancelled => Install::Idle,
            Outcome::Failed(msg) => Install::Failed(msg),
        };
    }

    /// The one-line status under the controls.
    pub fn status(&self) -> String {
        if !self.driver_installed {
            return "Driver not installed".into();
        }
        if !self.prefs.enabled {
            return "Noise cancellation is off".into();
        }
        if self.clients_active {
            "In use by an app".into()
        } else {
            "Waiting for an app to use Oto Microphone".into()
        }
    }

    /// CPU, as "3% CPU", while there's a reading.
    pub fn cpu(&self) -> Option<String> {
        self.load
            .filter(|_| self.clients_active)
            .map(|l| format!("{:.0}% CPU", l * 100.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::Outcome;

    fn state() -> State {
        let mut s = State::new(Prefs::default(), true);
        s.link = Link::Record(Vec::new());
        s.devices = vec![
            InputDevice {
                id: "a".into(),
                name: "MacBook Pro Microphone".into(),
                is_default: true,
            },
            InputDevice {
                id: "b".into(),
                name: "USB Mic".into(),
                is_default: false,
            },
            InputDevice {
                id: "c".into(),
                name: "USB Mic".into(),
                is_default: false,
            },
        ];
        s
    }

    fn sent(s: &State) -> &[Command] {
        match &s.link {
            Link::Record(v) => v,
            _ => &[],
        }
    }

    #[test]
    fn toggling_sends_once_and_gates_the_meter() {
        let mut s = state();
        s.on_event(Event::ClientsActive(true));
        assert!(s.meter.active.load(Ordering::Relaxed));
        s.toggle();
        s.set_enabled(false);
        assert_eq!(sent(&s), [Command::SetEnabled(false)]);
        assert!(!s.meter.active.load(Ordering::Relaxed));
    }

    #[test]
    fn device_labels_are_unique_and_map_back() {
        let mut s = state();
        let labels: Vec<_> = s.device_options().into_iter().map(|(_, l)| l).collect();
        assert_eq!(
            labels,
            [
                "System default (MacBook Pro Microphone)",
                "MacBook Pro Microphone",
                "USB Mic",
                "USB Mic 2"
            ]
        );
        s.choose_device("USB Mic 2");
        assert_eq!(s.prefs.input_device.as_deref(), Some("c"));
        assert_eq!(s.device_label(), "USB Mic 2");
        s.choose_device("System default (MacBook Pro Microphone)");
        assert_eq!(
            sent(&s),
            [
                Command::SetInputDevice(Some("c".into())),
                Command::SetInputDevice(None)
            ]
        );
    }

    #[test]
    fn an_unplugged_choice_stays_listed() {
        let mut s = state();
        s.choose_device("USB Mic");
        s.set_devices(Vec::new());
        assert_eq!(s.device_label(), "USB Mic (not connected)");
    }

    #[test]
    fn events_drive_the_status_line() {
        let mut s = state();
        assert_eq!(s.status(), "Waiting for an app to use Oto Microphone");
        s.on_event(Event::ClientsActive(true));
        s.on_event(Event::Load(0.034));
        assert_eq!(s.status(), "In use by an app");
        assert_eq!(s.cpu().as_deref(), Some("3% CPU"));
        s.on_event(Event::ClientsActive(false));
        assert_eq!(s.cpu(), None);
        s.on_event(Event::Error("Mic unplugged".into()));
        assert_eq!(s.error.as_deref(), Some("Mic unplugged"));
    }

    #[test]
    fn onboarding_until_installed() {
        let mut s = State::new(Prefs::default(), false);
        assert!(s.onboarding());
        s.start_install();
        assert_eq!(s.install, Install::Running);
        s.installed(Outcome::Cancelled, false);
        assert_eq!(s.install, Install::Idle);
        assert!(s.onboarding());
        s.installed(Outcome::Installed, false);
        assert!(matches!(s.install, Install::Failed(_)));
        s.installed(Outcome::Installed, true);
        assert!(!s.onboarding());
    }

    #[test]
    fn a_failed_login_change_keeps_the_old_value() {
        let mut s = state();
        s.hooks.set_login = |_| Err("nope".into());
        s.set_start_at_login(true);
        assert!(!s.prefs.start_at_login);
        assert_eq!(s.error.as_deref(), Some("nope"));
    }

    #[test]
    fn strength_and_tier_send_only_changes() {
        let mut s = state();
        s.set_strength(Strength::Balanced);
        s.set_strength(Strength::Strong);
        s.set_tier(Tier::Light);
        assert_eq!(
            sent(&s),
            [
                Command::SetStrength(Strength::Strong),
                Command::SetTier(Tier::Light)
            ]
        );
    }
}
