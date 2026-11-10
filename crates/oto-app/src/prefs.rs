//! What Oto remembers between launches, as a small TOML file:
//! `~/Library/Application Support/Oto/settings.toml` on macOS, `~/.config/oto/settings.toml` on Linux.

use std::path::PathBuf;

use oto_engine::{Settings, Strength, Tier};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    pub enabled: bool,
    /// `None` follows the system default input.
    pub input_device: Option<String>,
    /// The chosen device's name when it was picked, so it still reads right while unplugged.
    pub input_device_name: Option<String>,
    #[serde(with = "StrengthDef")]
    pub strength: Strength,
    #[serde(with = "TierDef")]
    pub tier: Tier,
    pub start_at_login: bool,
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

    pub fn from_toml(src: &str) -> Result<Prefs, String> {
        toml::from_str(src).map_err(|e| e.to_string())
    }

    pub fn to_toml(&self) -> String {
        toml::to_string(self).unwrap_or_default()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Strength", rename_all = "lowercase")]
enum StrengthDef {
    Light,
    Balanced,
    Strong,
}

#[derive(Serialize, Deserialize)]
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

/// The saved preferences, or defaults when there are none or they don't parse.
pub fn load() -> Prefs {
    let Some(path) = path() else {
        return Prefs::default();
    };
    match std::fs::read_to_string(&path) {
        Ok(src) => Prefs::from_toml(&src).unwrap_or_else(|e| {
            eprintln!("oto: ignoring {}: {e}", path.display());
            Prefs::default()
        }),
        Err(_) => Prefs::default(),
    }
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
            tier: Tier::Max,
            start_at_login: true,
        };
        let src = p.to_toml();
        assert!(src.contains("strength = \"strong\""), "{src}");
        assert_eq!(Prefs::from_toml(&src).unwrap(), p);
    }

    #[test]
    fn missing_fields_take_defaults() {
        let p = Prefs::from_toml("strength = \"light\"\n").unwrap();
        assert_eq!(
            p,
            Prefs {
                strength: Strength::Light,
                ..Prefs::default()
            }
        );
    }

    #[test]
    fn bad_values_are_errors() {
        assert!(Prefs::from_toml("tier = \"huge\"").is_err());
    }
}
