//! Start at login.
//!
//! On macOS this is a LaunchAgent plist in `~/Library/LaunchAgents` that starts the executable
//! inside `Oto.app` (SMAppService needs a Developer ID-signed bundle). It is only offered when
//! Oto runs from an app bundle: a `cargo run` binary has no Info.plist, so macOS would refuse it
//! the microphone, and its path is a build directory that will not last.

/// Whether Oto starts at login.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Registered, and the registration points at the app that is running now.
    Enabled,
    /// Not registered, or registered for some other copy of Oto.
    Disabled,
    /// This process can't be registered (see [`UNAVAILABLE`]).
    Unavailable,
}

/// Why start at login can't be turned on; what [`set`] returns as its error in that case.
pub const UNAVAILABLE: &str =
    "Start at login needs Oto.app. Move Oto to your Applications folder and open it from there.";

pub fn is_enabled() -> bool {
    status() == Status::Enabled
}

/// False when turning start at login on would fail with [`UNAVAILABLE`].
pub fn is_available() -> bool {
    status() != Status::Unavailable
}

#[cfg(target_os = "macos")]
pub use macos::{set, status};

#[cfg(target_os = "macos")]
mod macos {
    use super::{Status, UNAVAILABLE};
    use std::path::{Path, PathBuf};

    /// The LaunchAgent's label and file name; also the app's bundle identifier.
    const LABEL: &str = "dev.oto.app";

    /// The app bundle `exe` (a resolved path) can be registered as, if any. Gatekeeper runs a
    /// quarantined app from a random read-only "translocated" path that is gone after it
    /// quits, so that doesn't count.
    pub(super) fn registrable_bundle(exe: &Path) -> Option<PathBuf> {
        let translocated = exe
            .components()
            .any(|c| c.as_os_str() == "AppTranslocation");
        crate::driver::app_bundle(exe).filter(|_| !translocated)
    }

    fn xml_escape(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    }

    fn xml_unescape(s: &str) -> String {
        s.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&amp;", "&")
    }

    /// The LaunchAgent that starts `program` at login.
    pub(super) fn agent_plist(program: &Path) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>{LABEL}</string>
	<key>AssociatedBundleIdentifiers</key>
	<array>
		<string>{LABEL}</string>
	</array>
	<key>ProgramArguments</key>
	<array>
		<string>{}</string>
	</array>
	<key>RunAtLoad</key>
	<true/>
</dict>
</plist>
"#,
            xml_escape(&program.to_string_lossy())
        )
    }

    /// The program a LaunchAgent plist starts: the first of its `ProgramArguments`.
    pub(super) fn registered_program(plist: &str) -> Option<PathBuf> {
        let (_, rest) = plist.split_once("<key>ProgramArguments</key>")?;
        let (array, _) = rest.split_once("</array>")?;
        let (_, rest) = array.split_once("<string>")?;
        let (program, _) = rest.split_once("</string>")?;
        Some(PathBuf::from(xml_unescape(program)))
    }

    /// `exe` is the running executable (resolved); `plist` is the LaunchAgent on disk, if any.
    pub(super) fn status_for(exe: &Path, plist: Option<&str>) -> Status {
        let Some(bundle) = registrable_bundle(exe) else {
            return Status::Unavailable;
        };
        let registered = plist
            .and_then(registered_program)
            .and_then(|p| crate::driver::app_bundle(&p));
        if registered == Some(bundle) {
            Status::Enabled
        } else {
            Status::Disabled
        }
    }

    fn agent_file() -> Result<PathBuf, String> {
        let home = dirs::home_dir().ok_or("Couldn't find your home folder.")?;
        Ok(home
            .join("Library/LaunchAgents")
            .join(format!("{LABEL}.plist")))
    }

    fn current_exe() -> Option<PathBuf> {
        std::env::current_exe().and_then(|p| p.canonicalize()).ok()
    }

    pub fn status() -> Status {
        let Some(exe) = current_exe() else {
            return Status::Unavailable;
        };
        let plist = agent_file()
            .ok()
            .and_then(|f| std::fs::read_to_string(f).ok());
        status_for(&exe, plist.as_deref())
    }

    /// Turns start at login on or off. Turning it on fails with [`UNAVAILABLE`] outside an app
    /// bundle; turning it off always removes the registration, whoever it pointed at.
    pub fn set(on: bool) -> Result<(), String> {
        let fail = |e: std::io::Error| format!("Couldn't change start at login: {e}");
        let file = agent_file()?;
        if !on {
            return match std::fs::remove_file(&file) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(fail(e)),
                _ => Ok(()),
            };
        }
        let exe = current_exe()
            .filter(|exe| registrable_bundle(exe).is_some())
            .ok_or(UNAVAILABLE)?;
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).map_err(fail)?;
        }
        // Write beside it and rename, so launchd never reads half a file.
        let tmp = file.with_extension("plist.tmp");
        std::fs::write(&tmp, agent_plist(&exe)).map_err(fail)?;
        std::fs::rename(&tmp, &file).map_err(fail)
    }
}

#[cfg(not(target_os = "macos"))]
mod other {
    use super::Status;
    use auto_launch::{AutoLaunch, AutoLaunchBuilder};

    fn launcher() -> Result<AutoLaunch, String> {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let mut b = AutoLaunchBuilder::new();
        b.set_app_name("dev.oto.app")
            .set_app_path(&exe.to_string_lossy());
        b.build().map_err(|e| e.to_string())
    }

    pub fn status() -> Status {
        match launcher().and_then(|l| l.is_enabled().map_err(|e| e.to_string())) {
            Ok(true) => Status::Enabled,
            _ => Status::Disabled,
        }
    }

    pub fn set(on: bool) -> Result<(), String> {
        let l = launcher()?;
        let r = if on { l.enable() } else { l.disable() };
        r.map_err(|e| format!("Couldn't change start at login: {e}"))
    }
}

#[cfg(not(target_os = "macos"))]
pub use other::{set, status};

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::Status;
    use super::macos::*;
    use std::path::Path;

    const APP_EXE: &str = "/Applications/Oto.app/Contents/MacOS/oto-app";

    #[test]
    fn only_an_app_bundle_can_be_registered() {
        assert_eq!(
            registrable_bundle(Path::new(APP_EXE)).unwrap(),
            Path::new("/Applications/Oto.app")
        );
        assert_eq!(
            registrable_bundle(Path::new("/repo/target/debug/oto-app")),
            None
        );
        let translocated =
            "/private/var/folders/ab/T/AppTranslocation/1234/d/Oto.app/Contents/MacOS/oto-app";
        assert_eq!(registrable_bundle(Path::new(translocated)), None);
    }

    #[test]
    fn a_dev_binary_is_unavailable_even_if_registered() {
        let exe = Path::new("/repo/target/debug/oto-app");
        assert_eq!(status_for(exe, None), Status::Unavailable);
        assert_eq!(
            status_for(exe, Some(&agent_plist(exe))),
            Status::Unavailable
        );
    }

    #[test]
    fn enabled_only_when_the_registration_is_this_app() {
        let exe = Path::new(APP_EXE);
        assert_eq!(status_for(exe, None), Status::Disabled);
        assert_eq!(status_for(exe, Some("not a plist")), Status::Disabled);
        assert_eq!(status_for(exe, Some(&agent_plist(exe))), Status::Enabled);
        // A stale registration: a dev binary, or another copy of the app.
        for other in [
            "/repo/target/debug/oto-app",
            "/Users/me/Downloads/Oto.app/Contents/MacOS/oto-app",
        ] {
            let plist = agent_plist(Path::new(other));
            assert_eq!(status_for(exe, Some(&plist)), Status::Disabled, "{other}");
        }
    }

    #[test]
    fn the_plist_round_trips_awkward_paths() {
        let exe = Path::new("/Users/a & b/<Apps>/Oto.app/Contents/MacOS/oto-app");
        let plist = agent_plist(exe);
        assert!(plist.contains(
            "<string>/Users/a &amp; b/&lt;Apps&gt;/Oto.app/Contents/MacOS/oto-app</string>"
        ));
        assert_eq!(registered_program(&plist).unwrap(), exe);
        assert_eq!(status_for(exe, Some(&plist)), Status::Enabled);
    }

    #[test]
    fn reads_the_plist_older_versions_wrote() {
        // auto-launch's layout: everything in the array on one line.
        let old = "<key>Label</key>\n<string>dev.oto.app</string>\n<key>AssociatedBundleIdentifiers</key>\n<array></array>\n<key>ProgramArguments</key>\n<array><string>/Applications/Oto.app/Contents/MacOS/oto-app</string></array>";
        assert_eq!(registered_program(old).unwrap(), Path::new(APP_EXE));
    }
}
