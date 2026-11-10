//! What Oto remembers between launches, as a small TOML file:
//! `~/Library/Application Support/Oto/settings.toml` on macOS, `~/.config/oto/settings.toml` on Linux.

use std::path::PathBuf;

use oto_engine::{Settings, Strength, Tier};
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Prefs {
    pub enabled: bool,
    /// `None` follows the system default input.
    pub input_device: Option<String>,
    /// The chosen device's name when it was picked, so it still reads right while unplugged.
    pub input_device_name: Option<String>,
    #[serde(with = "StrengthDef")]
    pub strength: Strength,
    #[serde(with = "TierDef")]
    /// Saved as "standard" (DeepFilterNet3) or "light" (RNNoise, "Low CPU mode").
    pub tier: Tier,
    pub start_at_login: bool,
    /// "Use Oto as the system microphone": Oto Microphone is the default input while Oto runs.
    pub system_microphone: bool,
    /// Setup was finished (or wasn't needed), so it isn't shown again while the driver is in.
    pub onboarded: bool,
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs {
            enabled: true,
            input_device: None,
            input_device_name: None,
            strength: Strength::default(),
            tier: Tier::default(),
            start_at_login: false,
            system_microphone: false,
            onboarded: false,
        }
    }
}

impl Prefs {
    /// What the engine starts with.
    pub fn engine_settings(&self) -> Settings {
        Settings {
            enabled: self.enabled,
            input_device: self.input_device.clone(),
            strength: self.strength,
            tier: self.tier,
        }
    }

    /// Reads a settings file field by field: a missing, mistyped or unknown value leaves that
    /// one field at its default and is named in the returned warnings; the rest still load.
    /// Only a file that isn't TOML at all gives pure defaults.
    pub fn parse(src: &str) -> (Prefs, Vec<String>) {
        let mut p = Prefs::default();
        let mut warnings = Vec::new();
        let table: toml::Table = match toml::from_str(src) {
            Ok(t) => t,
            Err(e) => return (p, vec![e.to_string()]),
        };
        let mut bad = |key: &str, v: &toml::Value| warnings.push(format!("{key} = {v}"));
        let mut flag = |key: &str, into: &mut bool| match table.get(key) {
            Some(toml::Value::Boolean(b)) => *into = *b,
            Some(v) => bad(key, v),
            None => {}
        };
        flag("enabled", &mut p.enabled);
        flag("start_at_login", &mut p.start_at_login);
        flag("system_microphone", &mut p.system_microphone);
        flag("onboarded", &mut p.onboarded);
        let mut string = |key: &str, into: &mut Option<String>| match table.get(key) {
            Some(toml::Value::String(s)) => *into = Some(s.clone()),
            Some(v) => bad(key, v),
            None => {}
        };
        string("input_device", &mut p.input_device);
        string("input_device_name", &mut p.input_device_name);
        if let Some(v) = table.get("strength") {
            match v.as_str() {
                Some("light") => p.strength = Strength::Light,
                Some("balanced") => p.strength = Strength::Balanced,
                Some("strong") => p.strength = Strength::Strong,
                _ => bad("strength", v),
            }
        }
        if let Some(v) = table.get("tier") {
            match v.as_str() {
                Some("light") => p.tier = Tier::Light,
                // Max isn't offered; a saved one (from before Low CPU mode) means the default.
                Some("standard") | Some("max") => p.tier = Tier::Standard,
                _ => bad("tier", v),
            }
        }
        // A device's name without the device says nothing.
        if p.input_device.is_none() {
            p.input_device_name = None;
        }
        (p, warnings)
    }

    pub fn to_toml(&self) -> String {
        toml::to_string(self).unwrap_or_default()
    }
}

#[derive(Serialize)]
#[serde(remote = "Strength", rename_all = "lowercase")]
enum StrengthDef {
    Light,
    Balanced,
    Strong,
}

#[derive(Serialize)]
#[serde(remote = "Tier", rename_all = "lowercase")]
enum TierDef {
    Light,
    Standard,
    Max,
}

pub fn path() -> Option<PathBuf> {
    let folder = if cfg!(target_os = "macos") {
        "Oto"
    } else {
        "oto"
    };
    dirs::config_dir().map(|d| d.join(folder).join("settings.toml"))
}

/// The saved preferences. Anything missing or unreadable takes its default.
pub fn load() -> Prefs {
    let Some(path) = path() else {
        return Prefs::default();
    };
    let Ok(src) = std::fs::read_to_string(&path) else {
        return Prefs::default();
    };
    let (prefs, warnings) = Prefs::parse(&src);
    for w in warnings {
        eprintln!("oto: ignoring in {}: {w}", path.display());
    }
    prefs
}

pub fn save(prefs: &Prefs) -> Result<(), String> {
    let path = path().ok_or("no config folder")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("Couldn't save settings: {e}"))?;
    }
    // Write then rename, so a crash never leaves half a file.
    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, prefs.to_toml()).map_err(|e| format!("Couldn't save settings: {e}"))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("Couldn't save settings: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let p = Prefs {
            enabled: false,
            input_device: Some("BuiltInMicrophoneDevice".into()),
            input_device_name: Some("MacBook Pro Microphone".into()),
            strength: Strength::Strong,
            tier: Tier::Light,
            start_at_login: true,
            system_microphone: true,
            onboarded: true,
        };
        let src = p.to_toml();
        assert!(src.contains("strength = \"strong\""), "{src}");
        assert_eq!(Prefs::parse(&src), (p, Vec::new()));
        let d = Prefs::default();
        assert_eq!(Prefs::parse(&d.to_toml()), (d, Vec::new()));
    }

    #[test]
    fn saved_tiers_map_to_low_cpu_mode() {
        let tier = |v: &str| Prefs::parse(&format!("tier = \"{v}\"\n")).0.tier;
        assert_eq!(tier("standard"), Tier::Standard);
        assert_eq!(tier("light"), Tier::Light);
        // Max isn't offered any more and loads as the default, without a warning.
        assert_eq!(tier("max"), Tier::Standard);
        assert!(Prefs::parse("tier = \"max\"\n").1.is_empty());
        assert_eq!(Prefs::default().tier, Tier::Standard);
        assert!(Prefs::default().to_toml().contains("tier = \"standard\""));
    }

    #[test]
    fn missing_fields_take_defaults() {
        let (p, warnings) = Prefs::parse("strength = \"light\"\n");
        assert_eq!(
            p,
            Prefs {
                strength: Strength::Light,
                ..Prefs::default()
            }
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn one_bad_value_costs_only_its_own_field() {
        let src = "enabled = false\ntier = \"huge\"\nstrength = \"strong\"\n\
                   start_at_login = \"yes\"\ninput_device = \"usb\"\nonboarded = true\n\
                   colour = \"teal\"\n";
        let (p, warnings) = Prefs::parse(src);
        assert_eq!(
            p,
            Prefs {
                enabled: false,
                strength: Strength::Strong,
                input_device: Some("usb".into()),
                onboarded: true,
                ..Prefs::default()
            }
        );
        assert_eq!(warnings, ["start_at_login = \"yes\"", "tier = \"huge\""]);
    }

    #[test]
    fn a_file_that_is_not_toml_gives_defaults_and_says_why() {
        let (p, warnings) = Prefs::parse("enabled = = false");
        assert_eq!(p, Prefs::default());
        assert_eq!(warnings.len(), 1);
    }
}
