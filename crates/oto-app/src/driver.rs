//! Installing and removing the macOS HAL driver.
//!
//! The app never runs a script or a build as root. It installs the prebuilt, signed
//! `OtoMicrophone.driver` that ships inside `Oto.app`, with a fixed command sequence passed
//! inline to `osascript … with administrator privileges` (the system's password prompt). The only
//! thing that varies in that command is the source path, which is shell-quoted.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// The driver bundle's file name, in the app's Resources and in the HAL plug-in folder.
pub const BUNDLE_NAME: &str = "OtoMicrophone.driver";
/// `CFBundleIdentifier` of the driver (driver/macos/Info.plist), also its signing identifier.
const BUNDLE_ID: &str = "dev.oto.driver";
/// `CFBundleExecutable` of the driver.
const EXECUTABLE: &str = "OtoMicrophone";
/// Where Core Audio loads HAL plug-ins from.
const HAL_DIR: &str = "/Library/Audio/Plug-Ins/HAL";

/// Shown when the driver is in place but Core Audio hasn't picked it up (or let go of it).
pub const REBOOT_MESSAGE: &str = "Restart your Mac to finish.";

/// The Developer ID team the driver must be signed by, baked in by `scripts/macos/bundle.sh`
/// for Developer ID builds. Without it (ad-hoc and local builds) only the signature's validity
/// and identifier are checked.
const TEAM_ID: Option<&str> = option_env!("OTO_TEAM_ID");
const _: () = {
    if let Some(team) = TEAM_ID {
        let b = team.as_bytes();
        assert!(b.len() == 10, "OTO_TEAM_ID must be a 10-character team id");
        let mut i = 0;
        while i < b.len() {
            assert!(
                b[i].is_ascii_uppercase() || b[i].is_ascii_digit(),
                "OTO_TEAM_ID must be A-Z and 0-9 only"
            );
            i += 1;
        }
    }
};

/// How an install ended.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// Installed, and Core Audio lists Oto Microphone.
    Installed,
    /// Installed, but coreaudiod couldn't be restarted or hasn't loaded the driver: the user has
    /// to restart the Mac ([`REBOOT_MESSAGE`]).
    InstalledNeedsReboot,
    /// The user dismissed the password prompt.
    Cancelled,
    Failed(String),
}

/// How an uninstall ended.
#[derive(Clone, Debug, PartialEq)]
pub enum UninstallOutcome {
    /// Removed, and Core Audio no longer lists Oto Microphone.
    Removed,
    /// Removed from disk, but coreaudiod still has it loaded until the Mac restarts.
    RemovedNeedsReboot,
    /// There was nothing to remove; no password was asked for.
    NotInstalled,
    /// The user dismissed the password prompt.
    Cancelled,
    Failed(String),
}

// ---------------------------------------------------------------------------------------------
// Where the prebuilt driver may come from

/// `…/Oto.app` when `exe` is `…/Oto.app/Contents/MacOS/<binary>`.
pub fn app_bundle(exe: &Path) -> Option<PathBuf> {
    let macos = exe.parent()?;
    let contents = macos.parent()?;
    let app = contents.parent()?;
    let is_app = app
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("app"));
    (macos.file_name()? == "MacOS" && contents.file_name()? == "Contents" && is_app)
        .then(|| app.to_path_buf())
}

/// Every place the prebuilt driver may be taken from. `exe` is the *resolved* path of the
/// running executable. A release build looks only inside its own app bundle; a debug build may
/// also use the driver built in the source tree it was compiled from. Nothing else: no
/// environment override and no search through parent directories.
pub fn candidates(exe: Option<&Path>, debug_build: bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(app) = exe.and_then(app_bundle) {
        out.push(app.join("Contents/Resources").join(BUNDLE_NAME));
    }
    if debug_build {
        out.push(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../driver/macos/build")
                .join(BUNDLE_NAME),
        );
    }
    out
}

/// The driver bundle this build may install, as a resolved path.
pub fn find_source() -> Option<PathBuf> {
    let exe = std::env::current_exe().and_then(|p| p.canonicalize()).ok();
    candidates(exe.as_deref(), cfg!(debug_assertions))
        .into_iter()
        .find_map(|p| p.canonicalize().ok().filter(|p| p.is_dir()))
}

// ---------------------------------------------------------------------------------------------
// Checking the source before asking for a password

/// The code requirement the driver has to satisfy: our identifier and, when the build knows its
/// team, a Developer ID signature from that team.
fn requirement(team: Option<&str>) -> String {
    match team {
        Some(team) => format!(
            "identifier \"{BUNDLE_ID}\" and anchor apple generic and certificate leaf[subject.OU] = \"{team}\""
        ),
        None => format!("identifier \"{BUNDLE_ID}\""),
    }
}

fn plist_value(plist: &Path, key: &str) -> Option<String> {
    let out = Command::new("/usr/bin/plutil")
        .args(["-extract", key, "raw", "-o", "-"])
        .arg(plist)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// Checks that `source` is our driver bundle with a valid signature. This runs unprivileged and
/// is for a clear error before the password prompt; the privileged step checks the signature
/// again on its own root-owned copy, which is the check that counts.
fn validate_with(source: &Path, requirement: &str) -> Result<(), String> {
    let broken = |what: &str| format!("The driver in this copy of Oto is damaged ({what}).");
    let meta = std::fs::symlink_metadata(source)
        .map_err(|_| "This copy of Oto doesn't include the driver.".to_owned())?;
    if !meta.is_dir() {
        return Err(broken("not a bundle"));
    }
    let plist = source.join("Contents/Info.plist");
    if !plist.is_file() {
        return Err(broken("no Info.plist"));
    }
    if plist_value(&plist, "CFBundleIdentifier").as_deref() != Some(BUNDLE_ID) {
        return Err(broken("unexpected bundle identifier"));
    }
    if plist_value(&plist, "CFBundleExecutable").as_deref() != Some(EXECUTABLE) {
        return Err(broken("unexpected executable name"));
    }
    let exe = source.join("Contents/MacOS").join(EXECUTABLE);
    let is_file = std::fs::symlink_metadata(&exe).is_ok_and(|m| m.is_file());
    if !is_file {
        return Err(broken("executable missing"));
    }
    let verified = Command::new("/usr/bin/codesign")
        .args(["--verify", "--strict"])
        .arg(format!("-R={requirement}"))
        .arg(source)
        .output()
        .map_err(|e| format!("Couldn't run codesign: {e}"))?;
    if !verified.status.success() {
        return Err(broken("its code signature isn't valid"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The privileged commands

/// Stage a root-owned copy next to the destination, verify it, then swap it in with renames so
/// coreaudiod never sees a half-copied bundle; on failure the old driver is put back. Reads
/// `SRC`, `DEST_DIR`, `NAME` and `REQ`. A failed coreaudiod restart is not an error: it prints
/// `OTO_RESULT=reboot`. (`launchctl kickstart` is refused under SIP; `killall` is what works.)
const INSTALL_BODY: &str = r#"set -u
PATH=/usr/bin:/bin:/usr/sbin:/sbin
umask 022
DEST="$DEST_DIR/$NAME"
STAGE="$DEST_DIR/.$NAME.staging-$$"
OLD="$DEST_DIR/.$NAME.old-$$"
fail() { rm -rf "$STAGE"; echo "$1" >&2; exit 1; }
mkdir -p "$DEST_DIR" || fail "Couldn't create $DEST_DIR."
rm -rf "$DEST_DIR/.$NAME".staging-* "$DEST_DIR/.$NAME".old-*
cp -R "$SRC" "$STAGE" || fail "Couldn't copy the driver into $DEST_DIR."
chown -R root:wheel "$STAGE" || fail "Couldn't set the driver's owner."
chmod -R ug-s,go-w "$STAGE" || fail "Couldn't set the driver's permissions."
xattr -dr com.apple.quarantine "$STAGE" 2>/dev/null
codesign --verify --strict -R="$REQ" "$STAGE" 2>/dev/null || fail "The driver's code signature isn't valid, so it was not installed."
HAD_OLD=0
if [ -e "$DEST" ] || [ -L "$DEST" ]; then
  mv "$DEST" "$OLD" || fail "Couldn't replace the installed driver."
  HAD_OLD=1
fi
if ! mv "$STAGE" "$DEST"; then
  if [ "$HAD_OLD" = 1 ]; then mv "$OLD" "$DEST"; fi
  fail "Couldn't move the driver into place."
fi
rm -rf "$OLD"
if killall coreaudiod 2>/dev/null; then echo OTO_RESULT=restarted; else echo OTO_RESULT=reboot; fi
"#;

/// Remove the bundle (renamed away first, so it leaves in one step) and restart coreaudiod.
/// Reads `DEST_DIR` and `NAME`.
const UNINSTALL_BODY: &str = r#"set -u
PATH=/usr/bin:/bin:/usr/sbin:/sbin
DEST="$DEST_DIR/$NAME"
OLD="$DEST_DIR/.$NAME.old-$$"
if [ ! -e "$DEST" ] && [ ! -L "$DEST" ]; then echo OTO_RESULT=absent; exit 0; fi
mv "$DEST" "$OLD" || { echo "Couldn't remove the driver." >&2; exit 1; }
rm -rf "$OLD" "$DEST_DIR/.$NAME".staging-* "$DEST_DIR/.$NAME".old-*
if killall coreaudiod 2>/dev/null; then echo OTO_RESULT=restarted; else echo OTO_RESULT=reboot; fi
"#;

/// `s` as one POSIX shell word.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// The shell command that installs `source` into `dest_dir`: variable assignments, then the
/// fixed [`INSTALL_BODY`].
fn install_command(source: &Path, dest_dir: &str, requirement: &str) -> String {
    format!(
        "SRC={}; DEST_DIR={}; NAME={}; REQ={}\n{INSTALL_BODY}",
        shell_quote(&source.to_string_lossy()),
        shell_quote(dest_dir),
        shell_quote(BUNDLE_NAME),
        shell_quote(requirement),
    )
}

fn uninstall_command(dest_dir: &str) -> String {
    format!(
        "DEST_DIR={}; NAME={}\n{UNINSTALL_BODY}",
        shell_quote(dest_dir),
        shell_quote(BUNDLE_NAME),
    )
}

/// The AppleScript that runs `command` as root. Only the AppleScript string literal needs
/// escaping here; the shell quoting is already in `command`.
pub fn applescript(command: &str) -> String {
    let literal = command
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!("do shell script \"{literal}\" with administrator privileges")
}

enum Privileged {
    /// The command's standard output.
    Done(String),
    Cancelled,
    Failed(String),
}

fn run_privileged(command: &str) -> Privileged {
    let out = match Command::new("/usr/bin/osascript")
        .arg("-e")
        .arg(applescript(command))
        .output()
    {
        Ok(out) => out,
        Err(e) => return Privileged::Failed(format!("Couldn't run osascript: {e}")),
    };
    if out.status.success() {
        return Privileged::Done(String::from_utf8_lossy(&out.stdout).into_owned());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    // osascript reports a dismissed prompt as error -128.
    if err.contains("(-128)") {
        Privileged::Cancelled
    } else {
        Privileged::Failed(last_line(&err))
    }
}

/// osascript's error, without its "0:123: execution error:" prefix.
fn last_line(err: &str) -> String {
    let line = err.trim().lines().last().unwrap_or("The installer failed.");
    let line = line
        .split_once("execution error: ")
        .map_or(line, |(_, rest)| rest);
    line.trim().to_owned()
}

/// What the privileged command reported about coreaudiod.
#[derive(Debug, PartialEq)]
enum Restart {
    Restarted,
    NeedsReboot,
    /// Uninstall only: the bundle was already gone.
    Absent,
}

fn parse_result(stdout: &str) -> Restart {
    if stdout.contains("OTO_RESULT=restarted") {
        Restart::Restarted
    } else if stdout.contains("OTO_RESULT=absent") {
        Restart::Absent
    } else {
        Restart::NeedsReboot
    }
}

/// Polls `listed` until it returns `want`; false if it never does within `timeout`.
fn wait_until(listed: impl Fn() -> bool, want: bool, settle: Duration, timeout: Duration) -> bool {
    // coreaudiod was just killed: give it a moment so the old instance's answer isn't taken.
    std::thread::sleep(settle);
    let deadline = Instant::now() + timeout;
    loop {
        if listed() == want {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

const SETTLE: Duration = Duration::from_secs(1);
const APPEAR_TIMEOUT: Duration = Duration::from_secs(20);

/// Installs the bundled driver and waits for Core Audio to list it. Shows the system's password
/// prompt. Blocks for up to ~20 s after the prompt: call it off the UI thread.
pub fn install() -> Outcome {
    let Some(source) = find_source() else {
        return Outcome::Failed("This copy of Oto doesn't include the driver.".into());
    };
    let requirement = requirement(TEAM_ID);
    if let Err(e) = validate_with(&source, &requirement) {
        return Outcome::Failed(e);
    }
    match run_privileged(&install_command(&source, HAL_DIR, &requirement)) {
        Privileged::Cancelled => Outcome::Cancelled,
        Privileged::Failed(e) => Outcome::Failed(e),
        Privileged::Done(stdout) => {
            let loaded = parse_result(&stdout) == Restart::Restarted
                && wait_until(
                    oto_engine::Engine::driver_installed,
                    true,
                    SETTLE,
                    APPEAR_TIMEOUT,
                );
            if loaded {
                Outcome::Installed
            } else {
                Outcome::InstalledNeedsReboot
            }
        }
    }
}

/// Whether the driver bundle is in the HAL plug-in folder (whether or not Core Audio loaded it).
pub fn bundle_present() -> bool {
    Path::new(HAL_DIR)
        .join(BUNDLE_NAME)
        .symlink_metadata()
        .is_ok()
}

/// Removes the installed driver and waits for Core Audio to drop it. Shows the system's password
/// prompt unless there is nothing to remove. Blocks: call it off the UI thread.
pub fn uninstall() -> UninstallOutcome {
    if !bundle_present() {
        return UninstallOutcome::NotInstalled;
    }
    match run_privileged(&uninstall_command(HAL_DIR)) {
        Privileged::Cancelled => UninstallOutcome::Cancelled,
        Privileged::Failed(e) => UninstallOutcome::Failed(e),
        Privileged::Done(stdout) => match parse_result(&stdout) {
            Restart::Absent => UninstallOutcome::NotInstalled,
            Restart::Restarted
                if wait_until(
                    oto_engine::Engine::driver_installed,
                    false,
                    SETTLE,
                    APPEAR_TIMEOUT,
                ) =>
            {
                UninstallOutcome::Removed
            }
            _ => UninstallOutcome::RemovedNeedsReboot,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(paths: &[PathBuf]) -> Vec<String> {
        paths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn a_release_build_looks_only_inside_its_bundle() {
        let exe = Path::new("/Applications/Oto.app/Contents/MacOS/oto-app");
        assert_eq!(
            strings(&candidates(Some(exe), false)),
            ["/Applications/Oto.app/Contents/Resources/OtoMicrophone.driver"]
        );
    }

    #[test]
    fn a_release_build_outside_a_bundle_has_no_source() {
        // A bare binary next to a planted Resources or scripts directory finds nothing.
        for exe in [
            "/Users/me/Downloads/oto-app",
            "/Users/me/Downloads/MacOS/oto-app",
            "/Users/me/Downloads/Oto/Contents/MacOS/oto-app",
        ] {
            assert!(candidates(Some(Path::new(exe)), false).is_empty(), "{exe}");
        }
        assert!(candidates(None, false).is_empty());
    }

    #[test]
    fn a_debug_build_may_also_use_the_source_tree() {
        let exe = Path::new("/repo/target/debug/oto-app");
        let c = strings(&candidates(Some(exe), true));
        assert_eq!(c.len(), 1, "{c:?}");
        assert!(
            c[0].ends_with("oto-app/../../driver/macos/build/OtoMicrophone.driver"),
            "{c:?}"
        );
        // Bundled debug builds prefer the bundle.
        let exe = Path::new("/tmp/Oto.app/Contents/MacOS/oto-app");
        let c = strings(&candidates(Some(exe), true));
        assert_eq!(c.len(), 2);
        assert_eq!(c[0], "/tmp/Oto.app/Contents/Resources/OtoMicrophone.driver");
    }

    #[test]
    fn quotes_shell_words() {
        assert_eq!(shell_quote("plain"), "'plain'");
        assert_eq!(shell_quote("it's $HOME `x`"), r#"'it'\''s $HOME `x`'"#);
    }

    #[test]
    fn the_command_quotes_an_awkward_path() {
        let src = Path::new("/Users/a b/it's \"Oto\".app/Contents/Resources/OtoMicrophone.driver");
        let cmd = install_command(src, HAL_DIR, &requirement(None));
        let (first, body) = cmd.split_once('\n').unwrap();
        assert_eq!(
            first,
            r#"SRC='/Users/a b/it'\''s "Oto".app/Contents/Resources/OtoMicrophone.driver'; DEST_DIR='/Library/Audio/Plug-Ins/HAL'; NAME='OtoMicrophone.driver'; REQ='identifier "dev.oto.driver"'"#
        );
        // The rest is fixed text: nothing from the path reaches it.
        assert_eq!(body, INSTALL_BODY);
        assert!(!body.contains("Users"));

        let script = applescript(&cmd);
        assert!(script.starts_with(
            r#"do shell script "SRC='/Users/a b/it'\\''s \"Oto\".app/Contents/Resources/OtoMicrophone.driver'; "#
        ));
        assert!(script.ends_with("\" with administrator privileges"));
        assert!(!script.contains('\n'));
    }

    #[test]
    fn the_requirement_pins_the_team_when_known() {
        assert_eq!(requirement(None), r#"identifier "dev.oto.driver""#);
        assert_eq!(
            requirement(Some("ABCDE12345")),
            r#"identifier "dev.oto.driver" and anchor apple generic and certificate leaf[subject.OU] = "ABCDE12345""#
        );
    }

    #[test]
    fn a_missing_bundle_fails_validation() {
        let missing = Path::new("/nonexistent/OtoMicrophone.driver");
        let err = validate_with(missing, &requirement(None)).unwrap_err();
        assert_eq!(err, "This copy of Oto doesn't include the driver.");
    }

    #[test]
    fn reads_the_result_marker() {
        assert_eq!(parse_result("OTO_RESULT=restarted\n"), Restart::Restarted);
        assert_eq!(parse_result("OTO_RESULT=reboot\n"), Restart::NeedsReboot);
        assert_eq!(parse_result("OTO_RESULT=absent\n"), Restart::Absent);
        assert_eq!(parse_result(""), Restart::NeedsReboot);
    }

    #[test]
    fn trims_osascript_errors() {
        assert_eq!(
            last_line("0:120: execution error: Driver missing (1)\n"),
            "Driver missing (1)"
        );
    }

    #[test]
    fn waits_for_the_wanted_answer() {
        let none = Duration::ZERO;
        assert!(wait_until(|| true, true, none, none));
        assert!(!wait_until(|| true, false, none, none));
    }

    /// The command sequences, run for real as the current user against a scratch directory,
    /// with the root-only tools replaced by shell functions.
    #[cfg(unix)]
    mod sequence {
        use super::*;

        struct Scratch(PathBuf);

        impl Scratch {
            fn new(name: &str) -> Self {
                let dir = std::env::temp_dir()
                    .join(format!("oto driver's test-{}-{name}", std::process::id()));
                let _ = std::fs::remove_dir_all(&dir);
                std::fs::create_dir_all(&dir).unwrap();
                Self(dir.canonicalize().unwrap())
            }

            /// A stand-in driver bundle whose executable holds `marker`.
            fn source(&self, marker: &str) -> PathBuf {
                let src = self.0.join("src").join(BUNDLE_NAME);
                std::fs::create_dir_all(src.join("Contents/MacOS")).unwrap();
                std::fs::write(src.join("Contents/MacOS").join(EXECUTABLE), marker).unwrap();
                src
            }

            fn hal(&self) -> String {
                self.0.join("HAL").to_string_lossy().into_owned()
            }

            fn installed(&self) -> Option<String> {
                let exe = Path::new(&self.hal())
                    .join(BUNDLE_NAME)
                    .join("Contents/MacOS")
                    .join(EXECUTABLE);
                std::fs::read_to_string(exe).ok()
            }

            /// Names left in the HAL directory.
            fn listing(&self) -> Vec<String> {
                let mut names: Vec<_> = std::fs::read_dir(self.hal())
                    .map(|d| {
                        d.map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                            .collect()
                    })
                    .unwrap_or_default();
                names.sort();
                names
            }
        }

        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        /// Runs `command` with chown/xattr as no-ops and the given codesign/killall results.
        fn run(command: &str, codesign: i32, killall: i32) -> (bool, String, String) {
            let stubs = format!(
                "chown() {{ :; }}; xattr() {{ :; }}; codesign() {{ return {codesign}; }}; killall() {{ return {killall}; }}\n"
            );
            let out = Command::new("/bin/sh")
                .arg("-c")
                .arg(format!("{stubs}{command}"))
                .output()
                .unwrap();
            (
                out.status.success(),
                String::from_utf8_lossy(&out.stdout).into_owned(),
                String::from_utf8_lossy(&out.stderr).into_owned(),
            )
        }

        #[test]
        fn installs_fresh_then_replaces() {
            let t = Scratch::new("fresh");
            let src = t.source("one");
            let req = requirement(None);
            let (ok, out, err) = run(&install_command(&src, &t.hal(), &req), 0, 0);
            assert!(ok, "{err}");
            assert_eq!(parse_result(&out), Restart::Restarted);
            assert_eq!(t.installed().as_deref(), Some("one"));

            let src = t.source("two");
            let (ok, _, err) = run(&install_command(&src, &t.hal(), &req), 0, 0);
            assert!(ok, "{err}");
            assert_eq!(t.installed().as_deref(), Some("two"));
            // No staging or set-aside copies are left behind.
            assert_eq!(t.listing(), [BUNDLE_NAME]);
        }

        #[test]
        fn a_failed_restart_asks_for_a_reboot() {
            let t = Scratch::new("reboot");
            let src = t.source("one");
            let (ok, out, _) = run(&install_command(&src, &t.hal(), &requirement(None)), 0, 1);
            assert!(ok);
            assert_eq!(parse_result(&out), Restart::NeedsReboot);
            assert_eq!(t.installed().as_deref(), Some("one"));
        }

        #[test]
        fn a_bad_signature_leaves_the_old_driver() {
            let t = Scratch::new("badsig");
            let req = requirement(None);
            let src = t.source("old");
            assert!(run(&install_command(&src, &t.hal(), &req), 0, 0).0);

            let src = t.source("evil");
            let (ok, out, err) = run(&install_command(&src, &t.hal(), &req), 1, 0);
            assert!(!ok);
            assert!(out.is_empty(), "{out}");
            assert!(err.contains("signature isn't valid"), "{err}");
            assert_eq!(t.installed().as_deref(), Some("old"));
            assert_eq!(t.listing(), [BUNDLE_NAME]);
        }

        #[test]
        fn a_missing_source_fails_cleanly() {
            let t = Scratch::new("missing");
            let src = t.0.join("nope").join(BUNDLE_NAME);
            let (ok, _, err) = run(&install_command(&src, &t.hal(), &requirement(None)), 0, 0);
            assert!(!ok);
            assert!(err.contains("Couldn't copy the driver"), "{err}");
            assert!(t.listing().is_empty(), "{:?}", t.listing());
        }

        #[test]
        fn uninstalls() {
            let t = Scratch::new("uninstall");
            let src = t.source("one");
            assert!(run(&install_command(&src, &t.hal(), &requirement(None)), 0, 0).0);

            let (ok, out, err) = run(&uninstall_command(&t.hal()), 0, 1);
            assert!(ok, "{err}");
            assert_eq!(parse_result(&out), Restart::NeedsReboot);
            assert!(t.listing().is_empty(), "{:?}", t.listing());

            let (ok, out, _) = run(&uninstall_command(&t.hal()), 0, 0);
            assert!(ok);
            assert_eq!(parse_result(&out), Restart::Absent);
        }
    }
}
