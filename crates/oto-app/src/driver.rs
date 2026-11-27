//! Installing and removing the macOS HAL driver.
//!
//! The app never runs a script or a build as root. It installs the prebuilt, signed
//! `OtoMicrophone.driver` that ships inside `Oto.app`, with a fixed command sequence passed
//! inline to `osascript … with administrator privileges` (the system's password prompt). The
//! only things that vary in that command are paths, code requirements and the invoking user id,
//! all shell-quoted.
//!
//! The privileged step copies the bundled `oto-hal-helper` into a root-owned staging folder,
//! verifies its signature there and runs that copy, which takes the kernel lock on the HAL
//! folder, swaps the driver in atomically and restarts `coreaudiod` only if no audio is playing
//! (asking as the invoking user); otherwise it reports `OTO_RESULT=reboot`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// The driver bundle's file name, in the app's Resources and in the HAL plug-in folder.
pub const BUNDLE_NAME: &str = "OtoMicrophone.driver";
/// `CFBundleIdentifier` of the driver (driver/macos/Info.plist), also its signing identifier.
const BUNDLE_ID: &str = "dev.oto.driver";
/// The signing identifier of `oto-hal-helper`.
pub(crate) const HELPER_ID: &str = "dev.oto.hal-helper";
/// The helper's file name, in `Oto.app/Contents/Helpers` and in `driver/macos/build`.
pub(crate) const HELPER_NAME: &str = "oto-hal-helper";
/// `CFBundleIdentifier` of the app.
pub(crate) const APP_ID: &str = "dev.oto.app";
/// `CFBundleExecutable` of the driver.
const EXECUTABLE: &str = "OtoMicrophone";
/// Where Core Audio loads HAL plug-ins from.
const HAL_DIR: &str = "/Library/Audio/Plug-Ins/HAL";

/// Shown when the driver is in place but Core Audio hasn't picked it up (or let go of it).
pub const REBOOT_MESSAGE: &str = "Restart your Mac to finish.";

/// The Developer ID team the driver must be signed by, baked in by `scripts/macos/bundle.sh`
/// for Developer ID builds. Without it (ad-hoc and local builds) only the signature's validity
/// and identifier are checked.
pub(crate) const TEAM_ID: Option<&str> = option_env!("OTO_TEAM_ID");
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

/// Every place `oto-hal-helper` may be taken from, like [`candidates`]:
/// `Oto.app/Contents/Helpers/oto-hal-helper`, and in debug builds the one built in the source
/// tree.
pub fn helper_candidates(exe: Option<&Path>, debug_build: bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(app) = exe.and_then(app_bundle) {
        out.push(app.join("Contents/Helpers").join(HELPER_NAME));
    }
    if debug_build {
        out.push(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../driver/macos/build")
                .join(HELPER_NAME),
        );
    }
    out
}

/// The `oto-hal-helper` this build runs as root (after the root step re-verifies it).
pub fn find_helper() -> Option<PathBuf> {
    let exe = std::env::current_exe().and_then(|p| p.canonicalize()).ok();
    helper_candidates(exe.as_deref(), cfg!(debug_assertions))
        .into_iter()
        .find_map(|p| p.canonicalize().ok().filter(|p| p.is_file()))
}

// ---------------------------------------------------------------------------------------------
// Checking the source before asking for a password

/// The code requirement code signed as `bundle_id` has to satisfy: that identifier and, when the
/// build knows its team, a **Developer ID Application** signature from that team (the Developer
/// ID intermediate `1.2.840.113635.100.6.2.6` and leaf `1.2.840.113635.100.6.1.13` OIDs exclude
/// Apple Development and Mac App Store certificates of the same team). Ad-hoc and local builds
/// only pin the identifier.
pub(crate) fn requirement_for(bundle_id: &str, team: Option<&str>) -> String {
    match team {
        Some(team) => format!(
            "identifier \"{bundle_id}\" and anchor apple generic and \
             certificate 1[field.1.2.840.113635.100.6.2.6] exists and \
             certificate leaf[field.1.2.840.113635.100.6.1.13] exists and \
             certificate leaf[subject.OU] = \"{team}\""
        ),
        None => format!("identifier \"{bundle_id}\""),
    }
}

/// The requirement for the driver bundle.
fn requirement(team: Option<&str>) -> String {
    requirement_for(BUNDLE_ID, team)
}

/// The requirement for the driver bundle this build trusts.
pub(crate) fn driver_requirement() -> String {
    requirement(TEAM_ID)
}

/// The requirement every copy of `Oto.app` (and its update) must meet.
pub(crate) fn app_requirement() -> String {
    requirement_for(APP_ID, TEAM_ID)
}

/// The requirement for `oto-hal-helper`.
pub(crate) fn helper_requirement() -> String {
    requirement_for(HELPER_ID, TEAM_ID)
}

/// One string value from a property list (any format).
pub(crate) fn plist_value(plist: &Path, key: &str) -> Option<String> {
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
pub(crate) fn validate_with(source: &Path, requirement: &str) -> Result<(), String> {
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

/// Checks that `helper` is a regular file validly signed with `requirement`. Unprivileged, for a
/// clear error before the prompt; the root step checks its own root-owned copy again.
pub(crate) fn validate_helper(helper: &Path, requirement: &str) -> Result<(), String> {
    let broken = "The installer helper in this copy of Oto is damaged.".to_owned();
    let is_file = std::fs::symlink_metadata(helper).is_ok_and(|m| m.is_file());
    if !is_file {
        return Err("This copy of Oto doesn't include the installer helper.".into());
    }
    let verified = Command::new("/usr/bin/codesign")
        .args(["--verify", "--strict"])
        .arg(format!("-R={requirement}"))
        .arg(helper)
        .output()
        .map_err(|e| format!("Couldn't run codesign: {e}"))?;
    if !verified.status.success() {
        return Err(broken);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Driver revisions on disk

/// The installed driver's `Info.plist`.
fn installed_plist() -> PathBuf {
    Path::new(HAL_DIR)
        .join(BUNDLE_NAME)
        .join("Contents/Info.plist")
}

/// `OtoDriverVersion` from a driver's `Info.plist`: `None` if the file can't be read, `Some(0)`
/// for drivers from before the key existed (v0.0.1 to v0.0.3).
pub fn bundle_version(plist: &Path) -> Option<u32> {
    if !plist.is_file() {
        return None;
    }
    Some(
        plist_value(plist, "OtoDriverVersion")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    )
}

/// The revision of the driver in the HAL plug-in folder (whether or not Core Audio loaded it):
/// `None` if no driver is installed, `Some(0)` for a legacy or unreadable one.
pub fn on_disk_version() -> Option<u32> {
    if !bundle_present() {
        return None;
    }
    Some(bundle_version(&installed_plist()).unwrap_or(0))
}

/// Whether any audio device is in use right now (`Err` if Core Audio couldn't say). Callers
/// warn before the password prompt that audio will pause, or that the install will finish on
/// the next restart; the privileged step checks again right before restarting `coreaudiod`.
/// Blocks briefly: call it off the UI thread.
pub fn audio_busy() -> Result<bool, String> {
    oto_engine::Engine::any_audio_active_checked()
}

/// The real user id of this process, which the privileged step runs `--check-audio-idle` as.
fn invoking_uid() -> u32 {
    #[cfg(unix)]
    {
        // SAFETY: getuid has no preconditions and cannot fail.
        unsafe { libc::getuid() }
    }
    #[cfg(not(unix))]
    {
        0
    }
}

// ---------------------------------------------------------------------------------------------
// The privileged commands

/// Copy `oto-hal-helper` and the driver into a fresh root-owned staging folder, verify both
/// signatures there (as root, on the copies that will be used), then let the staged helper
/// install: it takes the kernel lock on the HAL folder, swaps the driver in with
/// `renamex_np(RENAME_SWAP)` (so it is never missing), and restarts coreaudiod only if the
/// invoking user's audio is idle, printing `OTO_RESULT=restarted` or `OTO_RESULT=reboot`. Reads
/// `SRC`, `HELPER`, `DEST_DIR`, `NAME`, `REQ`, `HELPER_REQ` and `INVOKING_UID`.
const INSTALL_BODY: &str = r#"set -u
PATH=/usr/bin:/bin:/usr/sbin:/sbin
umask 022
STAGE_DIR=""
cleanup() { if [ -n "$STAGE_DIR" ]; then rm -rf "$STAGE_DIR"; fi; }
fail() { cleanup; echo "$1" >&2; exit 1; }
STAGE_DIR="$(mktemp -d /private/tmp/oto-driver-install.XXXXXX)" || fail "Couldn't create a staging folder."
STAGE_HELPER="$STAGE_DIR/oto-hal-helper"
STAGE_DRIVER="$STAGE_DIR/$NAME"
cp "$HELPER" "$STAGE_HELPER" || fail "Couldn't copy the installer helper."
cp -R "$SRC" "$STAGE_DRIVER" || fail "Couldn't copy the driver into $STAGE_DIR."
chown -R root:wheel "$STAGE_DIR" || fail "Couldn't set the driver's owner."
chmod -R ug-s,go-w "$STAGE_DIR" || fail "Couldn't set the driver's permissions."
chmod 0755 "$STAGE_DIR" "$STAGE_HELPER" || fail "Couldn't set the installer helper's permissions."
xattr -dr com.apple.quarantine "$STAGE_DIR" 2>/dev/null
codesign --verify --strict -R="$HELPER_REQ" "$STAGE_HELPER" 2>/dev/null || fail "The installer helper's code signature isn't valid, so nothing was installed."
codesign --verify --strict -R="$REQ" "$STAGE_DRIVER" 2>/dev/null || fail "The driver's code signature isn't valid, so it was not installed."
mkdir -p "$DEST_DIR" || fail "Couldn't create $DEST_DIR."
"$STAGE_HELPER" install "$STAGE_DRIVER" "$DEST_DIR" "$NAME" "$INVOKING_UID"
RC=$?
cleanup
case "$RC" in
  0) ;;
  3) echo "A newer Oto audio driver is already installed." >&2; exit 1 ;;
  *) echo "Couldn't install the driver; the installed one (if any) is unchanged." >&2; exit 1 ;;
esac
"#;

/// Stage and verify `oto-hal-helper` as for install, then let it remove the driver (renamed
/// away first, so it leaves in one step) and restart coreaudiod if audio is idle. Prints
/// `OTO_RESULT=absent` if there was nothing to remove. Reads `HELPER`, `DEST_DIR`, `NAME`,
/// `HELPER_REQ` and `INVOKING_UID`.
const UNINSTALL_BODY: &str = r#"set -u
PATH=/usr/bin:/bin:/usr/sbin:/sbin
umask 022
STAGE_DIR=""
cleanup() { if [ -n "$STAGE_DIR" ]; then rm -rf "$STAGE_DIR"; fi; }
fail() { cleanup; echo "$1" >&2; exit 1; }
STAGE_DIR="$(mktemp -d /private/tmp/oto-driver-uninstall.XXXXXX)" || fail "Couldn't create a staging folder."
STAGE_HELPER="$STAGE_DIR/oto-hal-helper"
cp "$HELPER" "$STAGE_HELPER" || fail "Couldn't copy the installer helper."
chown -R root:wheel "$STAGE_DIR" || fail "Couldn't set the installer helper's owner."
chmod 0755 "$STAGE_DIR" "$STAGE_HELPER" || fail "Couldn't set the installer helper's permissions."
xattr -dr com.apple.quarantine "$STAGE_DIR" 2>/dev/null
codesign --verify --strict -R="$HELPER_REQ" "$STAGE_HELPER" 2>/dev/null || fail "The installer helper's code signature isn't valid, so nothing was removed."
"$STAGE_HELPER" uninstall "$DEST_DIR" "$NAME" "$INVOKING_UID"
RC=$?
cleanup
[ "$RC" = 0 ] || { echo "Couldn't remove the driver." >&2; exit 1; }
"#;

/// `s` as one POSIX shell word.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// What the privileged step needs besides the fixed body: everything is shell-quoted.
struct Params<'a> {
    helper: &'a Path,
    dest_dir: &'a str,
    requirement: &'a str,
    helper_requirement: &'a str,
    uid: u32,
}

/// The shell command that installs `source` into `dest_dir`: variable assignments, then the
/// fixed [`INSTALL_BODY`].
fn install_command(source: &Path, p: &Params) -> String {
    format!(
        "SRC={}; HELPER={}; DEST_DIR={}; NAME={}; REQ={}; HELPER_REQ={}; INVOKING_UID={}\n{INSTALL_BODY}",
        shell_quote(&source.to_string_lossy()),
        shell_quote(&p.helper.to_string_lossy()),
        shell_quote(p.dest_dir),
        shell_quote(BUNDLE_NAME),
        shell_quote(p.requirement),
        shell_quote(p.helper_requirement),
        p.uid,
    )
}

fn uninstall_command(p: &Params) -> String {
    format!(
        "HELPER={}; DEST_DIR={}; NAME={}; HELPER_REQ={}; INVOKING_UID={}\n{UNINSTALL_BODY}",
        shell_quote(&p.helper.to_string_lossy()),
        shell_quote(p.dest_dir),
        shell_quote(BUNDLE_NAME),
        shell_quote(p.helper_requirement),
        p.uid,
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

/// Polls `check` until it returns true; false if it never does within `timeout`.
fn wait_until(check: impl Fn() -> bool, settle: Duration, timeout: Duration) -> bool {
    // coreaudiod was just killed: give it a moment so the old instance's answer isn't taken.
    std::thread::sleep(settle);
    let deadline = Instant::now() + timeout;
    loop {
        if check() {
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

/// Core Audio has loaded exactly the driver this build ships.
fn bundled_driver_loaded() -> bool {
    oto_engine::Engine::driver_status()
        == oto_engine::DriverStatus::Loaded(oto_engine::BUNDLED_DRIVER_VERSION)
}

/// Core Audio has positively dropped the driver (`Unknown` is "not yet").
fn driver_gone() -> bool {
    oto_engine::Engine::driver_status() == oto_engine::DriverStatus::Absent
}

/// How a successful privileged install ends, given its output and a check for the loaded
/// revision: a reboot is never waited for.
fn install_outcome(stdout: &str, loaded: impl Fn() -> bool, settle: Duration) -> Outcome {
    match parse_result(stdout) {
        Restart::Restarted if wait_until(loaded, settle, APPEAR_TIMEOUT) => Outcome::Installed,
        _ => Outcome::InstalledNeedsReboot,
    }
}

fn uninstall_outcome(stdout: &str, gone: impl Fn() -> bool, settle: Duration) -> UninstallOutcome {
    match parse_result(stdout) {
        Restart::Absent => UninstallOutcome::NotInstalled,
        Restart::Restarted if wait_until(gone, settle, APPEAR_TIMEOUT) => UninstallOutcome::Removed,
        _ => UninstallOutcome::RemovedNeedsReboot,
    }
}

/// Refuses to overwrite a newer driver already loaded or on disk with this build's older copy.
/// Rechecked live before prompting for administrator privileges (and again under the HAL lock by
/// `oto-hal-helper`).
fn check_no_downgrade(
    loaded: oto_engine::DriverStatus,
    on_disk: Option<u32>,
    bundled: u32,
) -> Result<(), String> {
    let newer_loaded = matches!(loaded, oto_engine::DriverStatus::Loaded(v) if v > bundled);
    let newer_on_disk = on_disk.is_some_and(|v| v > bundled);
    if newer_loaded || newer_on_disk {
        return Err("A newer Oto audio driver is already installed.".into());
    }
    Ok(())
}

/// Installs the bundled driver and waits for Core Audio to load this build's revision. Shows
/// the system's password prompt. Blocks for up to ~20 s after the prompt: call it off the UI
/// thread. Ends with [`Outcome::InstalledNeedsReboot`] at once, without waiting, when audio was
/// busy and coreaudiod was left alone.
pub fn install() -> Outcome {
    if let Err(e) = check_no_downgrade(
        oto_engine::Engine::driver_status(),
        on_disk_version(),
        oto_engine::BUNDLED_DRIVER_VERSION,
    ) {
        return Outcome::Failed(e);
    }
    let Some(source) = find_source() else {
        return Outcome::Failed("This copy of Oto doesn't include the driver.".into());
    };
    let Some(helper) = find_helper() else {
        return Outcome::Failed("This copy of Oto doesn't include the installer helper.".into());
    };
    let requirement = requirement(TEAM_ID);
    let helper_requirement = helper_requirement();
    if let Err(e) = validate_with(&source, &requirement) {
        return Outcome::Failed(e);
    }
    if let Err(e) = validate_helper(&helper, &helper_requirement) {
        return Outcome::Failed(e);
    }
    let params = Params {
        helper: &helper,
        dest_dir: HAL_DIR,
        requirement: &requirement,
        helper_requirement: &helper_requirement,
        uid: invoking_uid(),
    };
    match run_privileged(&install_command(&source, &params)) {
        Privileged::Cancelled => Outcome::Cancelled,
        Privileged::Failed(e) => Outcome::Failed(e),
        Privileged::Done(stdout) => install_outcome(&stdout, bundled_driver_loaded, SETTLE),
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
    let Some(helper) = find_helper() else {
        return UninstallOutcome::Failed(
            "This copy of Oto doesn't include the installer helper.".into(),
        );
    };
    let helper_requirement = helper_requirement();
    if let Err(e) = validate_helper(&helper, &helper_requirement) {
        return UninstallOutcome::Failed(e);
    }
    let params = Params {
        helper: &helper,
        dest_dir: HAL_DIR,
        requirement: "",
        helper_requirement: &helper_requirement,
        uid: invoking_uid(),
    };
    match run_privileged(&uninstall_command(&params)) {
        Privileged::Cancelled => UninstallOutcome::Cancelled,
        Privileged::Failed(e) => UninstallOutcome::Failed(e),
        Privileged::Done(stdout) => uninstall_outcome(&stdout, driver_gone, SETTLE),
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

    fn params<'a>(helper: &'a Path, dest_dir: &'a str, req: &'a str, hreq: &'a str) -> Params<'a> {
        Params {
            helper,
            dest_dir,
            requirement: req,
            helper_requirement: hreq,
            uid: 501,
        }
    }

    #[test]
    fn a_release_build_looks_only_inside_its_bundle() {
        let exe = Path::new("/Applications/Oto.app/Contents/MacOS/oto-app");
        assert_eq!(
            strings(&candidates(Some(exe), false)),
            ["/Applications/Oto.app/Contents/Resources/OtoMicrophone.driver"]
        );
        assert_eq!(
            strings(&helper_candidates(Some(exe), false)),
            ["/Applications/Oto.app/Contents/Helpers/oto-hal-helper"]
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
            assert!(
                helper_candidates(Some(Path::new(exe)), false).is_empty(),
                "{exe}"
            );
        }
        assert!(candidates(None, false).is_empty());
        assert!(helper_candidates(None, false).is_empty());
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
        let h = strings(&helper_candidates(Some(exe), true));
        assert_eq!(h.len(), 1, "{h:?}");
        assert!(
            h[0].ends_with("oto-app/../../driver/macos/build/oto-hal-helper"),
            "{h:?}"
        );
        // Bundled debug builds prefer the bundle.
        let exe = Path::new("/tmp/Oto.app/Contents/MacOS/oto-app");
        let c = strings(&candidates(Some(exe), true));
        assert_eq!(c.len(), 2);
        assert_eq!(c[0], "/tmp/Oto.app/Contents/Resources/OtoMicrophone.driver");
        let h = strings(&helper_candidates(Some(exe), true));
        assert_eq!(h[0], "/tmp/Oto.app/Contents/Helpers/oto-hal-helper");
    }

    #[test]
    fn quotes_shell_words() {
        assert_eq!(shell_quote("plain"), "'plain'");
        assert_eq!(shell_quote("it's $HOME `x`"), r#"'it'\''s $HOME `x`'"#);
    }

    #[test]
    fn the_command_quotes_an_awkward_path() {
        let src = Path::new("/Users/a b/it's \"Oto\".app/Contents/Resources/OtoMicrophone.driver");
        let helper = Path::new("/Users/a b/it's \"Oto\".app/Contents/Helpers/oto-hal-helper");
        let req = requirement(None);
        let hreq = requirement_for(HELPER_ID, None);
        let cmd = install_command(src, &params(helper, HAL_DIR, &req, &hreq));
        let (first, body) = cmd.split_once('\n').unwrap();
        assert_eq!(
            first,
            r#"SRC='/Users/a b/it'\''s "Oto".app/Contents/Resources/OtoMicrophone.driver'; HELPER='/Users/a b/it'\''s "Oto".app/Contents/Helpers/oto-hal-helper'; DEST_DIR='/Library/Audio/Plug-Ins/HAL'; NAME='OtoMicrophone.driver'; REQ='identifier "dev.oto.driver"'; HELPER_REQ='identifier "dev.oto.hal-helper"'; INVOKING_UID=501"#
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

        let cmd = uninstall_command(&params(helper, HAL_DIR, "", &hreq));
        let (first, body) = cmd.split_once('\n').unwrap();
        assert_eq!(
            first,
            r#"HELPER='/Users/a b/it'\''s "Oto".app/Contents/Helpers/oto-hal-helper'; DEST_DIR='/Library/Audio/Plug-Ins/HAL'; NAME='OtoMicrophone.driver'; HELPER_REQ='identifier "dev.oto.hal-helper"'; INVOKING_UID=501"#
        );
        assert_eq!(body, UNINSTALL_BODY);
    }

    #[test]
    fn the_requirement_pins_the_team_when_known() {
        assert_eq!(requirement(None), r#"identifier "dev.oto.driver""#);
        assert_eq!(
            requirement(Some("ABCDE12345")),
            r#"identifier "dev.oto.driver" and anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU] = "ABCDE12345""#
        );
        assert_eq!(
            requirement_for(HELPER_ID, Some("ABCDE12345")),
            requirement(Some("ABCDE12345")).replace("dev.oto.driver", "dev.oto.hal-helper")
        );
        assert_eq!(app_requirement(), requirement_for("dev.oto.app", TEAM_ID));
        assert_eq!(helper_requirement(), requirement_for(HELPER_ID, TEAM_ID));
    }

    #[test]
    fn a_missing_bundle_fails_validation() {
        let missing = Path::new("/nonexistent/OtoMicrophone.driver");
        let err = validate_with(missing, &requirement(None)).unwrap_err();
        assert_eq!(err, "This copy of Oto doesn't include the driver.");
        let err = validate_helper(Path::new("/nonexistent/oto-hal-helper"), "x").unwrap_err();
        assert!(err.contains("installer helper"), "{err}");
    }

    #[test]
    fn reads_the_result_marker() {
        assert_eq!(parse_result("OTO_RESULT=restarted\n"), Restart::Restarted);
        assert_eq!(parse_result("OTO_RESULT=reboot\n"), Restart::NeedsReboot);
        assert_eq!(parse_result("OTO_RESULT=absent\n"), Restart::Absent);
        assert_eq!(parse_result(""), Restart::NeedsReboot);
        // The helper's other lines don't confuse it.
        assert_eq!(
            parse_result("OTO_FROM=0\nOTO_TO=1\nInstalled x\nOTO_RESULT=restarted\n"),
            Restart::Restarted
        );
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
        assert!(wait_until(|| true, none, none));
        assert!(!wait_until(|| false, none, none));
    }

    #[test]
    fn a_reboot_result_is_never_waited_for() {
        use std::cell::Cell;
        let asked = Cell::new(0);
        let check = || {
            asked.set(asked.get() + 1);
            true
        };
        let out = install_outcome("OTO_RESULT=reboot\n", check, Duration::ZERO);
        assert_eq!(out, Outcome::InstalledNeedsReboot);
        assert_eq!(asked.get(), 0);
        assert_eq!(
            install_outcome("OTO_RESULT=restarted\n", check, Duration::ZERO),
            Outcome::Installed
        );
        assert_eq!(asked.get(), 1);
        assert_eq!(
            uninstall_outcome("OTO_RESULT=reboot\n", check, Duration::ZERO),
            UninstallOutcome::RemovedNeedsReboot
        );
        assert_eq!(
            uninstall_outcome("OTO_RESULT=absent\n", check, Duration::ZERO),
            UninstallOutcome::NotInstalled
        );
        assert_eq!(
            uninstall_outcome("OTO_RESULT=restarted\n", check, Duration::ZERO),
            UninstallOutcome::Removed
        );
    }

    #[test]
    fn reads_driver_revisions_from_info_plists() {
        let dir = std::env::temp_dir().join(format!("oto-driver-plist-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let plist = |name: &str, extra: &str| {
            let p = dir.join(name);
            std::fs::write(
                &p,
                format!(
                    r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>CFBundleVersion</key><string>0.0.3</string>{extra}</dict></plist>"#
                ),
            )
            .unwrap();
            p
        };
        let current = plist(
            "current.plist",
            "<key>OtoDriverVersion</key><integer>7</integer>",
        );
        let legacy = plist("legacy.plist", "");
        assert_eq!(bundle_version(&current), Some(7));
        assert_eq!(bundle_version(&legacy), Some(0));
        assert_eq!(bundle_version(&dir.join("missing.plist")), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The command sequences, run for real as the current user against a scratch directory,
    /// with the root-only tools replaced by shell functions and `oto-hal-helper` by a script
    /// that installs with plain renames and reports what `FAKE_RESULT` says.
    #[cfg(unix)]
    mod sequence {
        use super::*;

        struct Scratch(PathBuf);

        const FAKE_HELPER: &str = r#"#!/bin/sh
echo "$0 $*" >> "$(dirname "$FAKE_LOG")/args"
case "$1" in
  install)
    [ "$5" = "$FAKE_UID" ] || exit 2
    D="$3/$4"
    [ -n "${FAKE_RC:-}" ] && exit "$FAKE_RC"
    rm -rf "$3/.$4.swap"
    cp -R "$2" "$3/.$4.swap" || exit 1
    rm -rf "$D"; mv "$3/.$4.swap" "$D" || exit 1
    echo "OTO_RESULT=${FAKE_RESULT:-restarted}" ;;
  uninstall)
    D="$2/$3"
    if [ ! -e "$D" ]; then echo OTO_RESULT=absent; exit 0; fi
    rm -rf "$D" || exit 1
    echo "OTO_RESULT=${FAKE_RESULT:-restarted}" ;;
  *) exit 2 ;;
esac
"#;

        impl Scratch {
            fn new(name: &str) -> Self {
                let dir = std::env::temp_dir()
                    .join(format!("oto driver's test-{}-{name}", std::process::id()));
                let _ = std::fs::remove_dir_all(&dir);
                std::fs::create_dir_all(&dir).unwrap();
                let s = Self(dir.canonicalize().unwrap());
                let helper = s.helper();
                std::fs::create_dir_all(helper.parent().unwrap()).unwrap();
                std::fs::write(&helper, FAKE_HELPER).unwrap();
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755)).unwrap();
                s
            }

            fn helper(&self) -> PathBuf {
                self.0.join("bundle helper").join(HELPER_NAME)
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

            /// Staging folders left behind.
            fn stages(&self) -> usize {
                std::fs::read_dir(&self.0)
                    .unwrap()
                    .filter(|e| {
                        e.as_ref()
                            .unwrap()
                            .file_name()
                            .to_string_lossy()
                            .starts_with("stage.")
                    })
                    .count()
            }

            fn install(&self, src: &Path) -> String {
                let (req, hreq) = (requirement(None), requirement_for(HELPER_ID, None));
                let helper = self.helper();
                install_command(src, &params(&helper, &self.hal(), &req, &hreq))
            }

            fn uninstall(&self) -> String {
                let hreq = requirement_for(HELPER_ID, None);
                let helper = self.helper();
                uninstall_command(&params(&helper, &self.hal(), "", &hreq))
            }

            /// Runs `command` with chown/xattr as no-ops, mktemp inside the scratch dir, the given
            /// codesign results (helper, driver) and the helper's reported result.
            fn run(
                &self,
                command: &str,
                codesign: (i32, i32),
                env: &[(&str, &str)],
            ) -> (bool, String, String) {
                let (h, d) = codesign;
                let stubs = format!(
                    "chown() {{ :; }}; xattr() {{ :; }}; \
                     mktemp() {{ command mktemp -d {}/stage.XXXXXX; }}; \
                     codesign() {{ case \"$*\" in *oto-hal-helper*) return {h};; *) return {d};; esac; }}\n",
                    shell_quote(&self.0.to_string_lossy())
                );
                std::fs::create_dir_all(self.0.join("log")).unwrap();
                let out = Command::new("/bin/sh")
                    .arg("-c")
                    .arg(format!("{stubs}{command}"))
                    .env("FAKE_LOG", self.0.join("log/x"))
                    .env("FAKE_UID", "501")
                    .envs(env.iter().copied())
                    .output()
                    .unwrap();
                (
                    out.status.success(),
                    String::from_utf8_lossy(&out.stdout).into_owned(),
                    String::from_utf8_lossy(&out.stderr).into_owned(),
                )
            }

            /// What the staged helper was run as.
            fn helper_calls(&self) -> String {
                std::fs::read_to_string(self.0.join("log/args")).unwrap_or_default()
            }
        }

        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        #[test]
        fn installs_fresh_then_replaces_through_the_staged_helper() {
            let t = Scratch::new("fresh");

            let src = t.source("one");
            let (ok, out, err) = t.run(&t.install(&src), (0, 0), &[]);
            assert!(ok, "{err}");
            assert_eq!(parse_result(&out), Restart::Restarted);
            assert_eq!(t.installed().as_deref(), Some("one"));
            // The copy in the staging folder ran, not the one in the bundle.
            let calls = t.helper_calls();
            assert!(calls.contains("/stage."), "{calls}");
            assert!(calls.contains(" install "), "{calls}");
            assert!(calls.trim_end().ends_with(" 501"), "{calls}");

            let src = t.source("two");
            let (ok, _, err) = t.run(&t.install(&src), (0, 0), &[]);
            assert!(ok, "{err}");
            assert_eq!(t.installed().as_deref(), Some("two"));
            assert_eq!(t.listing(), [BUNDLE_NAME]);
            assert_eq!(t.stages(), 0);
        }

        fn ready(name: &str) -> Scratch {
            Scratch::new(name)
        }

        #[test]
        fn busy_audio_asks_for_a_reboot() {
            let t = ready("reboot");
            let src = t.source("one");
            let (ok, out, _) = t.run(&t.install(&src), (0, 0), &[("FAKE_RESULT", "reboot")]);
            assert!(ok);
            assert_eq!(parse_result(&out), Restart::NeedsReboot);
            assert_eq!(t.installed().as_deref(), Some("one"));
            assert_eq!(t.stages(), 0);
        }

        #[test]
        fn a_bad_driver_signature_leaves_the_old_driver() {
            let t = ready("badsig");
            let src = t.source("old");
            assert!(t.run(&t.install(&src), (0, 0), &[]).0);

            let src = t.source("evil");
            let (ok, out, err) = t.run(&t.install(&src), (0, 1), &[]);
            assert!(!ok);
            assert!(out.is_empty(), "{out}");
            assert!(err.contains("driver's code signature isn't valid"), "{err}");
            assert_eq!(t.installed().as_deref(), Some("old"));
            assert_eq!(t.listing(), [BUNDLE_NAME]);
            assert_eq!(t.stages(), 0);
        }

        #[test]
        fn a_bad_helper_signature_runs_nothing() {
            let t = ready("badhelper");
            let src = t.source("one");
            let (ok, _, err) = t.run(&t.install(&src), (1, 0), &[]);
            assert!(!ok);
            assert!(err.contains("installer helper's code signature"), "{err}");
            assert!(t.helper_calls().is_empty());
            assert!(t.installed().is_none());
            let (ok, _, err) = t.run(&t.uninstall(), (1, 0), &[]);
            assert!(!ok);
            assert!(err.contains("nothing was removed"), "{err}");
            assert!(t.helper_calls().is_empty());
            assert_eq!(t.stages(), 0);
        }

        #[test]
        fn a_failing_helper_is_reported() {
            let t = ready("helperfail");
            let src = t.source("one");
            let (ok, _, err) = t.run(&t.install(&src), (0, 0), &[("FAKE_RC", "1")]);
            assert!(!ok);
            assert!(err.contains("Couldn't install the driver"), "{err}");
            let (ok, _, err) = t.run(&t.install(&src), (0, 0), &[("FAKE_RC", "3")]);
            assert!(!ok);
            assert!(err.contains("newer"), "{err}");
            assert_eq!(t.stages(), 0);
        }

        #[test]
        fn a_missing_source_fails_cleanly() {
            let t = ready("missing");
            let src = t.0.join("nope").join(BUNDLE_NAME);
            let (ok, _, err) = t.run(&t.install(&src), (0, 0), &[]);
            assert!(!ok);
            assert!(err.contains("Couldn't copy the driver"), "{err}");
            assert!(t.listing().is_empty(), "{:?}", t.listing());
            assert_eq!(t.stages(), 0);
        }

        #[test]
        fn uninstalls() {
            let t = ready("uninstall");
            let src = t.source("one");
            assert!(t.run(&t.install(&src), (0, 0), &[]).0);

            let (ok, out, err) = t.run(&t.uninstall(), (0, 0), &[("FAKE_RESULT", "reboot")]);
            assert!(ok, "{err}");
            assert_eq!(parse_result(&out), Restart::NeedsReboot);
            assert!(t.listing().is_empty(), "{:?}", t.listing());

            let (ok, out, _) = t.run(&t.uninstall(), (0, 0), &[]);
            assert!(ok);
            assert_eq!(parse_result(&out), Restart::Absent);
            assert!(t.helper_calls().contains(" uninstall "));
            assert_eq!(t.stages(), 0);
        }
    }

    #[test]
    fn check_no_downgrade_refuses_newer_loaded_or_on_disk() {
        use oto_engine::DriverStatus::{Absent, Loaded, Unknown};
        assert!(check_no_downgrade(Loaded(2), None, 1).is_err());
        assert!(check_no_downgrade(Loaded(1), Some(2), 1).is_err());
        assert!(check_no_downgrade(Loaded(0), Some(2), 1).is_err());
        assert!(check_no_downgrade(Absent, Some(2), 1).is_err());
        assert!(check_no_downgrade(Unknown, Some(2), 1).is_err());
        assert!(check_no_downgrade(Loaded(1), Some(1), 1).is_ok());
        assert!(check_no_downgrade(Loaded(0), Some(0), 1).is_ok());
        assert!(check_no_downgrade(Absent, None, 1).is_ok());
    }
}
