//! Applying an update: where Oto may replace itself, the staged bundle swap through a detached
//! helper, the new build's two-stage acknowledgment, rollback and recovery.
//!
//! See docs/plans/0003-app-updates.md, Decision 6. In short:
//!
//! 1. [`classify_install`] decides whether this copy may replace itself.
//! 2. [`PreparedUpdate::prepare`] (old app still running): clones the verified update next to
//!    the app as `.Oto-update-<nonce>` and verifies it there, copies the running executable to
//!    `update-helper-<nonce>` in the state directory, and starts it with `--finish-update` in its
//!    own session (so a LaunchAgent's teardown doesn't kill it). The helper takes
//!    `updating.lock`, writes `update-tx.json` (`prepared`) and says "ready".
//! 3. The old app shuts its engine down (restoring the default input) and calls
//!    [`PreparedUpdate::commit_and_signal`], then exits. The helper ([`run_helper`]) sees the
//!    commit byte and then EOF (the old app is gone), takes `engine.lock`, swaps the bundles with
//!    `renamex_np(RENAME_SWAP)`, marks `tx-<nonce>-state = active`, and launches the new build
//!    with `--update-ack <nonce>`.
//! 4. The new build ([`candidate_admit_and_mark_started`], [`candidate_mark_ready`],
//!    [`candidate_wait_for_commit_or_takeover`]) acknowledges twice and stays inert until the
//!    helper publishes `committed`; on `revoked` or a missing state it exits untouched.
//! 5. Without a timely `ready`, the helper revokes, stops the candidate, swaps the old build
//!    back and relaunches it with `--wait-update-release`. A helper that dies at any point is
//!    covered by [`recover_update`] (shared with `oto recover-update`).

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use oto_engine::update_recovery::{
    self as rec, EngineLockGuard, RecoveryMode, RecoveryOps, RecoveryOutcome, STAGING_PREFIX,
    WAIT_UPDATE_RELEASE_ARG,
};
use oto_engine::update_state::{
    self as us, AckRecord, CandidateToken, Nonce, TxControlState, TxPhase, UpdatePaths, UpdateTx,
    UpdatingLock,
};

use super::verify::{self, StagedUpdate};
use crate::driver;

/// The argument the helper sends the new build.
pub const UPDATE_ACK_ARG: &str = "--update-ack";
/// The argument the old app starts its helper copy with.
pub const FINISH_UPDATE_ARG: &str = "--finish-update";
/// `update-helper-<nonce> --recover-update`: manual recovery with the preserved old binary.
pub const RECOVER_UPDATE_ARG: &str = "--recover-update";

/// How long the old app waits for its helper's "ready".
const HELPER_READY_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a launched instance (restored app) gets to report that it started.
const LAUNCH_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------------------------
// Install-mode classification

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallMode {
    /// This bundle may be replaced in place.
    SelfUpdatable { bundle: PathBuf },
    /// Offer "Download…" instead.
    NotifyOnly(NotifyReason),
    /// No updates at all (dev and ad-hoc builds).
    Unsupported(UnsupportedReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotifyReason {
    /// The engine runs without `engine.lock`, so the swap couldn't be serialized.
    NoInstanceLock,
    /// Translocated or on a read-only volume (a mounted DMG): *"Move Oto to Applications for
    /// automatic updates."*
    NotInApplications,
    /// Owned by another user or root, not writable, or the volume can't swap atomically:
    /// *"Download the installer to update Oto."*
    RequiresManualInstall,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsupportedReason {
    /// Not inside an `.app`, or built without a Developer ID team.
    DevBuild,
}

/// How this copy of Oto can be updated. `exe` is the canonical path of the running executable.
/// Touches the file system: call it off the UI thread.
pub fn classify_install(exe: Option<&Path>, has_instance_lock: bool) -> InstallMode {
    classify_with(exe, has_instance_lock, driver::TEAM_ID.is_some())
}

fn classify_with(exe: Option<&Path>, has_instance_lock: bool, team_signed: bool) -> InstallMode {
    let Some(exe) = exe else {
        return InstallMode::Unsupported(UnsupportedReason::DevBuild);
    };
    let Some(bundle) = driver::app_bundle(exe) else {
        return InstallMode::Unsupported(UnsupportedReason::DevBuild);
    };
    if !team_signed {
        return InstallMode::Unsupported(UnsupportedReason::DevBuild);
    }
    if !has_instance_lock {
        return InstallMode::NotifyOnly(NotifyReason::NoInstanceLock);
    }
    let vol = volume_info(&bundle);
    if registrable(exe).is_none() || vol.as_ref().is_none_or(|v| v.read_only) {
        return InstallMode::NotifyOnly(NotifyReason::NotInApplications);
    }
    let vol = vol.unwrap_or_default();
    let Some(parent) = bundle.parent() else {
        return InstallMode::NotifyOnly(NotifyReason::RequiresManualInstall);
    };
    let real_dir = fs::symlink_metadata(&bundle).is_ok_and(|m| m.file_type().is_dir());
    if real_dir
        && owned_by_me(&bundle)
        && writable(parent)
        && writable(&bundle)
        && (vol.rename_swap || vol.fs_type == "apfs")
    {
        InstallMode::SelfUpdatable { bundle }
    } else {
        InstallMode::NotifyOnly(NotifyReason::RequiresManualInstall)
    }
}

#[cfg(target_os = "macos")]
fn registrable(exe: &Path) -> Option<PathBuf> {
    crate::login::registrable_bundle(exe)
}

#[cfg(not(target_os = "macos"))]
fn registrable(_exe: &Path) -> Option<PathBuf> {
    None
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct VolumeInfo {
    read_only: bool,
    fs_type: String,
    rename_swap: bool,
}

fn cpath(p: &Path) -> io::Result<std::ffi::CString> {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::CString::new(p.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))
}

#[cfg(target_os = "macos")]
fn volume_info(path: &Path) -> Option<VolumeInfo> {
    let c = cpath(path).ok()?;
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    // SAFETY: valid path and out-pointer.
    if unsafe { libc::statfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    let cstr = |a: &[libc::c_char]| {
        let bytes: Vec<u8> = a
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    let mount = cstr(&st.f_mntonname);
    Some(VolumeInfo {
        read_only: st.f_flags & (libc::MNT_RDONLY as u32) != 0,
        fs_type: cstr(&st.f_fstypename),
        rename_swap: supports_rename_swap(Path::new(&mount)),
    })
}

#[cfg(not(target_os = "macos"))]
fn volume_info(_path: &Path) -> Option<VolumeInfo> {
    None
}

/// `VOL_CAP_INT_RENAME_SWAP` of the volume mounted at `mount`.
#[cfg(target_os = "macos")]
fn supports_rename_swap(mount: &Path) -> bool {
    #[repr(C, packed(4))]
    struct Buf {
        len: u32,
        caps: libc::vol_capabilities_attr_t,
    }
    let Ok(c) = cpath(mount) else { return false };
    let mut list = libc::attrlist {
        bitmapcount: libc::ATTR_BIT_MAP_COUNT,
        reserved: 0,
        commonattr: 0,
        volattr: libc::ATTR_VOL_INFO | libc::ATTR_VOL_CAPABILITIES,
        dirattr: 0,
        fileattr: 0,
        forkattr: 0,
    };
    let mut buf: Buf = unsafe { std::mem::zeroed() };
    // SAFETY: buffers are as large as we say.
    let r = unsafe {
        libc::getattrlist(
            c.as_ptr(),
            (&mut list as *mut libc::attrlist).cast(),
            (&mut buf as *mut Buf).cast(),
            std::mem::size_of::<Buf>(),
            0,
        )
    };
    if r != 0 {
        return false;
    }
    let caps = buf.caps;
    let i = libc::VOL_CAPABILITIES_INTERFACES;
    let bit = libc::VOL_CAP_INT_RENAME_SWAP;
    caps.valid[i] & bit != 0 && caps.capabilities[i] & bit != 0
}

fn owned_by_me(path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    // SAFETY: geteuid cannot fail.
    let me = unsafe { libc::geteuid() };
    fs::symlink_metadata(path).is_ok_and(|m| m.uid() == me)
}

fn writable(path: &Path) -> bool {
    cpath(path).is_ok_and(|c| unsafe { libc::access(c.as_ptr(), libc::W_OK) } == 0)
}

// ---------------------------------------------------------------------------------------------
// File-system primitives

/// `renamex_np(a, b, RENAME_SWAP)`.
pub fn rename_swap(a: &Path, b: &Path) -> io::Result<()> {
    rec::sys::rename_swap(a, b)
}

/// Copies the directory tree `src` to the new path `dst`: an APFS clone (`clonefile`), or a
/// metadata-preserving recursive copy (`copyfile`) where cloning isn't possible (`EXDEV`,
/// `ENOTSUP`). Symlinks are copied as symlinks.
pub fn clone_tree(src: &Path, dst: &Path) -> io::Result<()> {
    clone_tree_with(src, dst, false)
}

fn clone_tree_with(src: &Path, dst: &Path, force_copy: bool) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        const CLONE_NOFOLLOW: u32 = 0x0001;
        // COPYFILE_ACL | STAT | XATTR | DATA
        const COPYFILE_ALL: u32 = 0x0F;
        let (s, d) = (cpath(src)?, cpath(dst)?);
        if fs::symlink_metadata(dst).is_ok() {
            return Err(io::Error::from(io::ErrorKind::AlreadyExists));
        }
        if !force_copy {
            // SAFETY: valid paths.
            if unsafe { libc::clonefile(s.as_ptr(), d.as_ptr(), CLONE_NOFOLLOW) } == 0 {
                return Ok(());
            }
            let e = io::Error::last_os_error();
            if !matches!(e.raw_os_error(), Some(libc::EXDEV) | Some(libc::ENOTSUP)) {
                return Err(e);
            }
        }
        // SAFETY: valid paths, no state object.
        let r = unsafe {
            libc::copyfile(
                s.as_ptr(),
                d.as_ptr(),
                std::ptr::null_mut(),
                COPYFILE_ALL | libc::COPYFILE_RECURSIVE | libc::COPYFILE_NOFOLLOW,
            )
        };
        if r == 0 {
            Ok(())
        } else {
            let e = io::Error::last_os_error();
            let _ = rec::remove_path(dst);
            Err(e)
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (src, dst, force_copy);
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}

fn set_mode(path: &Path, mode: u32) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
}

/// A pipe whose ends are both close-on-exec: `(read, write)`.
fn pipe() -> io::Result<(OwnedFd, OwnedFd)> {
    let mut fds = [0; 2];
    // SAFETY: two-element out array.
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fresh descriptors we own.
    let (r, w) = unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) };
    set_cloexec(r.as_raw_fd(), true)?;
    set_cloexec(w.as_raw_fd(), true)?;
    Ok((r, w))
}

fn set_cloexec(fd: RawFd, on: bool) -> io::Result<()> {
    // SAFETY: fcntl on a descriptor; async-signal-safe (also used in pre_exec).
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFD);
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        let flags = if on {
            flags | libc::FD_CLOEXEC
        } else {
            flags & !libc::FD_CLOEXEC
        };
        if libc::fcntl(fd, libc::F_SETFD, flags) < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Waits until `fd` is readable (data or EOF). False on timeout.
fn wait_readable(fd: RawFd, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        let mut p = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let ms = left.as_millis().min(i32::MAX as u128) as i32;
        // SAFETY: one pollfd.
        let r = unsafe { libc::poll(&mut p, 1, ms) };
        if r > 0 {
            return true;
        }
        if r == 0 || Instant::now() >= deadline {
            return false;
        }
        // EINTR: retry.
    }
}

/// Reads one byte: `Ok(Some(b))`, `Ok(None)` at EOF, `Err` on timeout or error.
fn read_byte(f: &mut File, timeout: Option<Duration>) -> io::Result<Option<u8>> {
    if let Some(t) = timeout
        && !wait_readable(f.as_raw_fd(), t)
    {
        return Err(io::Error::from(io::ErrorKind::TimedOut));
    }
    let mut b = [0u8; 1];
    loop {
        match f.read(&mut b) {
            Ok(0) => return Ok(None),
            Ok(_) => return Ok(Some(b[0])),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Side effects (real and fake)

/// What the helper does beyond recovery's operations.
pub trait UpdateOps: RecoveryOps + Sync {
    /// `Info.plist` identity and version (`version`, newer than `current`) plus the Developer ID
    /// signatures of the bundle, its driver and its helper.
    fn verify_staged(&self, app: &Path, version: &str, current: &str) -> Result<(), String>;
    /// Launches a new instance of `target` with `--update-ack <nonce>`.
    fn launch_candidate(&self, target: &Path, nonce: &Nonce) -> Result<Box<dyn Launched>, String>;
}

/// A launch in progress (`NSWorkspace` answers asynchronously).
pub trait Launched: Send {
    /// Waits up to `timeout` for the launch to be confirmed: `Some(Ok(pid))`, `Some(Err)` if it
    /// failed, `None` if still pending.
    fn wait(&self, timeout: Duration) -> Option<Result<Option<u32>, String>>;
    /// Terminates the launched instance (`terminate`, then `forceTerminate` after `grace`).
    fn terminate(&self, grace: Duration);
}

/// The real operations: `renamex_np`, `LSRegisterURL`, `proc_pidpath`, `NSWorkspace`.
pub struct AppOps;

impl RecoveryOps for AppOps {
    fn swap(&self, a: &Path, b: &Path) -> io::Result<()> {
        rename_swap(a, b)
    }
    fn register(&self, app: &Path) {
        let st = rec::sys::ls_register(app);
        if st != 0 {
            self.log(&format!("LSRegisterURL({}) failed: {st}", app.display()));
        }
    }
    fn bundle_version(&self, app: &Path) -> Option<String> {
        driver::plist_value(
            &app.join("Contents/Info.plist"),
            "CFBundleShortVersionString",
        )
    }
    fn pid_exe(&self, pid: u32) -> Option<PathBuf> {
        rec::sys::pid_path(pid)
    }
    fn pid_alive(&self, pid: u32) -> bool {
        rec::sys::pid_alive(pid)
    }
    fn terminate(&self, pid: u32, grace: Duration) -> bool {
        rec::sys::terminate(pid, grace)
    }
    fn launch_restored(&self, target: &Path) -> Result<(), String> {
        let l = workspace::launch(target, &[WAIT_UPDATE_RELEASE_ARG])?;
        match l.wait(LAUNCH_TIMEOUT) {
            Some(Ok(_)) => Ok(()),
            Some(Err(e)) => Err(e),
            None => Err("the launch did not start in time".into()),
        }
    }
    fn log(&self, msg: &str) {
        eprintln!(
            "[{}] oto-update[{}]: {msg}",
            us::now_unix(),
            std::process::id()
        );
    }
}

impl UpdateOps for AppOps {
    fn verify_staged(&self, app: &Path, version: &str, current: &str) -> Result<(), String> {
        verify::check_bundle_info(app, version, current)?;
        verify::BundleChecks::signatures(&verify::SystemChecks, app)
    }
    fn launch_candidate(&self, target: &Path, nonce: &Nonce) -> Result<Box<dyn Launched>, String> {
        Ok(Box::new(workspace::launch(
            target,
            &[UPDATE_ACK_ARG, nonce.as_str()],
        )?))
    }
}

/// `NSWorkspace.openApplicationAtURL:configuration:completionHandler:` with
/// `createsNewApplicationInstance = true`, keeping the `NSRunningApplication`.
mod workspace {
    use super::Launched;
    use std::path::Path;
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    #[derive(Default)]
    struct Inner {
        done: bool,
        pid: Option<u32>,
        error: Option<String>,
        #[cfg(target_os = "macos")]
        app: Option<SendApp>,
    }

    #[cfg(target_os = "macos")]
    struct SendApp(objc2::rc::Retained<objc2_app_kit::NSRunningApplication>);
    // SAFETY: NSRunningApplication's terminate/forceTerminate/processIdentifier are thread-safe.
    #[cfg(target_os = "macos")]
    unsafe impl Send for SendApp {}

    pub struct Launch {
        state: Arc<(Mutex<Inner>, Condvar)>,
    }

    impl Launched for Launch {
        fn wait(&self, timeout: Duration) -> Option<Result<Option<u32>, String>> {
            let (m, cv) = &*self.state;
            let deadline = Instant::now() + timeout;
            let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
            while !g.done {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return None;
                }
                g = cv
                    .wait_timeout(g, left)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
            Some(match &g.error {
                Some(e) => Err(e.clone()),
                None => Ok(g.pid),
            })
        }

        fn terminate(&self, grace: Duration) {
            let (m, _) = &*self.state;
            let pid = m.lock().unwrap_or_else(|e| e.into_inner()).pid;
            #[cfg(target_os = "macos")]
            {
                let g = m.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(app) = &g.app {
                    app.0.terminate();
                }
            }
            if let Some(pid) = pid
                && !oto_engine::update_recovery::sys::wait_gone(pid, grace)
            {
                #[cfg(target_os = "macos")]
                {
                    let g = m.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(app) = &g.app {
                        app.0.forceTerminate();
                    }
                }
                oto_engine::update_recovery::sys::terminate(pid, Duration::ZERO);
            }
        }
    }

    #[cfg(target_os = "macos")]
    pub fn launch(app: &Path, args: &[&str]) -> Result<Launch, String> {
        use block2::RcBlock;
        use objc2_app_kit::{NSRunningApplication, NSWorkspace, NSWorkspaceOpenConfiguration};
        use objc2_foundation::{NSArray, NSError, NSString, NSURL};

        let state = Arc::new((Mutex::new(Inner::default()), Condvar::new()));
        let url =
            NSURL::fileURLWithPath_isDirectory(&NSString::from_str(&app.to_string_lossy()), true);
        let config = NSWorkspaceOpenConfiguration::configuration();
        config.setCreatesNewApplicationInstance(true);
        let args: Vec<_> = args.iter().map(|a| NSString::from_str(a)).collect();
        config.setArguments(&NSArray::from_retained_slice(&args));
        let s2 = state.clone();
        let block = RcBlock::new(move |app: *mut NSRunningApplication, err: *mut NSError| {
            let (m, cv) = &*s2;
            let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: the handler's arguments are valid (or null) for this call.
            if let Some(e) = unsafe { err.as_ref() } {
                g.error = Some(e.localizedDescription().to_string());
            }
            if let Some(app) = unsafe { objc2::rc::Retained::retain(app) } {
                g.pid = u32::try_from(app.processIdentifier()).ok();
                g.app = Some(SendApp(app));
            } else if g.error.is_none() {
                g.error = Some("the app did not start".into());
            }
            g.done = true;
            cv.notify_all();
        });
        NSWorkspace::sharedWorkspace().openApplicationAtURL_configuration_completionHandler(
            &url,
            &config,
            Some(&block),
        );
        Ok(Launch { state })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn launch(_app: &Path, _args: &[&str]) -> Result<Launch, String> {
        Err("launching apps is only supported on macOS".into())
    }
}

// ---------------------------------------------------------------------------------------------
// The old app: staging and handing off to the helper

/// The helper's arguments (`--finish-update …`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HelperArgs {
    pub nonce: Nonce,
    pub target: PathBuf,
    pub staging: PathBuf,
    pub old_version: String,
    pub new_version: String,
    pub old_pid: u32,
    pub commit_fd: RawFd,
    pub ready_fd: RawFd,
}

impl HelperArgs {
    pub fn to_args(&self) -> Vec<OsString> {
        let mut v: Vec<OsString> = vec![FINISH_UPDATE_ARG.into()];
        let mut kv = |k: &str, val: OsString| {
            v.push(k.into());
            v.push(val);
        };
        kv("--nonce", self.nonce.as_str().into());
        kv("--target", self.target.clone().into());
        kv("--staging", self.staging.clone().into());
        kv("--old-ver", self.old_version.clone().into());
        kv("--new-ver", self.new_version.clone().into());
        kv("--old-pid", self.old_pid.to_string().into());
        kv("--commit-fd", self.commit_fd.to_string().into());
        kv("--ready-fd", self.ready_fd.to_string().into());
        v
    }
}

/// How this process was started.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CliMode {
    Normal,
    /// The new build of transaction `nonce`.
    UpdateAck(Nonce),
    /// The restored old build after a rollback: wait for `updating.lock` first.
    WaitUpdateRelease,
    /// The detached update helper.
    FinishUpdate(HelperArgs),
    /// `update-helper-<nonce> --recover-update`.
    RecoverUpdate,
}

/// Parses the arguments after the program name. Unknown arguments (e.g. `-psn_…` from old
/// launchers) leave the mode `Normal`.
pub fn parse_cli_mode(args: impl IntoIterator<Item = OsString>) -> Result<CliMode, String> {
    let args: Vec<OsString> = args.into_iter().collect();
    let s = |i: usize| args.get(i).and_then(|a| a.to_str());
    match s(0) {
        Some(UPDATE_ACK_ARG) => {
            let n = s(1).ok_or("--update-ack needs a nonce")?;
            Ok(CliMode::UpdateAck(Nonce::parse(n)?))
        }
        Some(WAIT_UPDATE_RELEASE_ARG) => Ok(CliMode::WaitUpdateRelease),
        Some(RECOVER_UPDATE_ARG) => Ok(CliMode::RecoverUpdate),
        Some(FINISH_UPDATE_ARG) => {
            let get = |k: &str| -> Result<OsString, String> {
                let i = args
                    .iter()
                    .position(|a| a == k)
                    .ok_or(format!("{FINISH_UPDATE_ARG} needs {k}"))?;
                args.get(i + 1).cloned().ok_or(format!("{k} needs a value"))
            };
            let text = |v: OsString| {
                v.into_string()
                    .map_err(|_| "argument is not UTF-8".to_string())
            };
            let num = |v: OsString, k: &str| -> Result<i64, String> {
                text(v)?.parse().map_err(|_| format!("{k} is not a number"))
            };
            Ok(CliMode::FinishUpdate(HelperArgs {
                nonce: Nonce::parse(&text(get("--nonce")?)?)?,
                target: get("--target")?.into(),
                staging: get("--staging")?.into(),
                old_version: text(get("--old-ver")?)?,
                new_version: text(get("--new-ver")?)?,
                old_pid: u32::try_from(num(get("--old-pid")?, "--old-pid")?)
                    .map_err(|_| "--old-pid is out of range")?,
                commit_fd: RawFd::try_from(num(get("--commit-fd")?, "--commit-fd")?)
                    .map_err(|_| "--commit-fd is out of range")?,
                ready_fd: RawFd::try_from(num(get("--ready-fd")?, "--ready-fd")?)
                    .map_err(|_| "--ready-fd is out of range")?,
            }))
        }
        _ => Ok(CliMode::Normal),
    }
}

/// `~/Library/Logs/Oto/updater.log`.
pub fn helper_log_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join("Library/Logs/Oto/updater.log"))
}

/// Starts the helper. Receives the child's ends of the pipes, which it must close in this
/// process once the helper has them.
pub type SpawnHelper<'a> = dyn Fn(&Path, &HelperArgs, OwnedFd, OwnedFd) -> Result<(), String> + 'a;

/// The real spawn: `helper --finish-update …` in its own session (`setsid`), inheriting only the
/// two pipe ends, logging to [`helper_log_path`].
pub fn spawn_detached(
    helper: &Path,
    args: &HelperArgs,
    commit_rx: OwnedFd,
    ready_tx: OwnedFd,
) -> Result<(), String> {
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};
    let log = helper_log_path().ok_or("Couldn't find your home folder.")?;
    if let Some(d) = log.parent() {
        let _ = fs::create_dir_all(d);
    }
    let mut o = fs::OpenOptions::new();
    o.create(true).append(true);
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    let out = o
        .open(&log)
        .map_err(|e| format!("Couldn't open {}: {e}", log.display()))?;
    let err = out.try_clone().map_err(|e| e.to_string())?;
    let (c, r) = (commit_rx.as_raw_fd(), ready_tx.as_raw_fd());
    let mut cmd = Command::new(helper);
    cmd.args(args.to_args())
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err);
    // SAFETY: only async-signal-safe calls (setsid, fcntl) between fork and exec.
    unsafe {
        cmd.pre_exec(move || {
            if libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            set_cloexec(c, false)?;
            set_cloexec(r, false)
        });
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Couldn't start the updater: {e}"))?;
    drop(commit_rx);
    drop(ready_tx);
    // Reap it if it exits while this process still runs.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

/// An update staged next to the app, with its helper waiting for the commit byte. Dropping it
/// without [`Self::commit_and_signal`] cancels: the helper sees EOF with no byte and cleans up.
pub struct PreparedUpdate {
    pub nonce: Nonce,
    pub target: PathBuf,
    pub staging: PathBuf,
    pub helper_path: PathBuf,
    pub new_version: String,
    commit_tx: Option<File>,
    /// Keep the commit pipe open until this process exits, so the helper's EOF proves exit.
    hold_until_exit: bool,
}

impl PreparedUpdate {
    /// Stages `staged` next to `target` (the running bundle, [`InstallMode::SelfUpdatable`]),
    /// copies the running executable to `update-helper-<nonce>` and starts it. Blocks for a few
    /// seconds: call it off the UI thread, while the engine still runs.
    pub fn prepare(
        staged: &StagedUpdate,
        target: &Path,
        current_version: &str,
    ) -> Result<Self, String> {
        let paths = UpdatePaths::default_location().ok_or("Couldn't find your home folder.")?;
        let exe = std::env::current_exe()
            .and_then(|p| p.canonicalize())
            .map_err(|e| format!("Couldn't find Oto's executable: {e}"))?;
        let mut p = Self::prepare_with(
            staged,
            target,
            current_version,
            &paths,
            &AppOps,
            &exe,
            &spawn_detached,
        )?;
        p.hold_until_exit = true;
        Ok(p)
    }

    /// [`Self::prepare`] with explicit locations, operations and spawner.
    pub fn prepare_with(
        staged: &StagedUpdate,
        target: &Path,
        current_version: &str,
        paths: &UpdatePaths,
        ops: &dyn UpdateOps,
        helper_exe: &Path,
        spawn: &SpawnHelper,
    ) -> Result<Self, String> {
        let new_version = staged.info.version.clone();
        let parent = target.parent().ok_or("Oto isn't in a folder.")?;
        let nonce = Nonce::random().map_err(|e| e.to_string())?;
        let staging = parent.join(format!("{STAGING_PREFIX}{nonce}"));
        let helper_path = paths.update_helper(&nonce);
        let helper_tmp = paths.dir.join(format!(".update-helper-{nonce}.tmp"));
        let cleanup = || {
            let _ = rec::remove_path(&staging);
            let _ = rec::remove_path(&helper_tmp);
            let _ = rec::remove_path(&helper_path);
        };

        clone_tree(&staged.cached_app_path, &staging)
            .map_err(|e| format!("Couldn't copy the update next to Oto: {e}"))?;
        if let Err(e) = ops.verify_staged(&staging, &new_version, current_version) {
            cleanup();
            return Err(e);
        }
        let _ = rec::remove_path(&staged.cache_dir);

        // Copy to a hidden temp path first, set mode and refresh mtime (macOS fs::copy preserves
        // the source binary's old mtime), then rename into place. Serialized under a persistent
        // `.helper-prep.lock` with startup cleanup so an active temp is never reclaimed mid-copy,
        // while a crash leftover temp is reclaimed once the lock releases.
        let copied = fs::create_dir_all(&paths.dir).and_then(|()| {
            let prep_lock = open_helper_prep_lock(paths)?;
            prep_lock.lock()?;
            let _ = rec::remove_path(&helper_tmp);
            let _ = rec::remove_path(&helper_path);
            fs::copy(helper_exe, &helper_tmp)?;
            set_mode(&helper_tmp, 0o700)?;
            fs::OpenOptions::new()
                .write(true)
                .open(&helper_tmp)
                .and_then(|f| f.set_modified(std::time::SystemTime::now()))?;
            fs::rename(&helper_tmp, &helper_path)
        });
        if let Err(e) = copied {
            cleanup();
            return Err(format!("Couldn't prepare the updater: {e}"));
        }

        let (commit_rx, commit_tx) = pipe().map_err(|e| e.to_string())?;
        let (ready_rx, ready_tx) = pipe().map_err(|e| e.to_string())?;
        let args = HelperArgs {
            nonce: nonce.clone(),
            target: target.to_path_buf(),
            staging: staging.clone(),
            old_version: current_version.to_owned(),
            new_version: new_version.clone(),
            old_pid: std::process::id(),
            commit_fd: commit_rx.as_raw_fd(),
            ready_fd: ready_tx.as_raw_fd(),
        };
        if let Err(e) = spawn(&helper_path, &args, commit_rx, ready_tx) {
            cleanup();
            return Err(e);
        }
        let mut ready = File::from(ready_rx);
        match read_byte(&mut ready, Some(HELPER_READY_TIMEOUT)) {
            Ok(Some(b'R')) => {}
            _ => {
                // The helper failed (another update runs, or a stale one couldn't be cleared).
                drop(commit_tx);
                cleanup();
                return Err("The updater couldn't start. Try again later.".into());
            }
        }
        Ok(Self {
            nonce,
            target: target.to_path_buf(),
            staging,
            helper_path,
            new_version,
            commit_tx: Some(File::from(commit_tx)),
            hold_until_exit: false,
        })
    }

    /// Tells the helper to swap once this process exits. Call after the engine shut down and
    /// restored the default input, then quit promptly. `Err` (the helper died: `EPIPE`) means
    /// nothing will happen: restart the engine and keep running.
    pub fn commit_and_signal(mut self) -> Result<(), String> {
        let mut f = self.commit_tx.take().ok_or("already committed")?;
        f.write_all(&[1])
            .map_err(|e| format!("The updater stopped unexpectedly ({e})."))?;
        if self.hold_until_exit {
            // The helper waits for EOF, i.e. for this process to be gone.
            std::mem::forget(f);
        }
        Ok(())
    }

    /// Abandons the update: the helper sees EOF without a byte, removes the staged copy, the
    /// journal and itself, and exits without touching Oto.
    pub fn cancel(mut self) {
        self.commit_tx.take();
    }
}

// ---------------------------------------------------------------------------------------------
// The helper

/// Timeouts of the helper protocol.
#[derive(Clone, Copy, Debug)]
pub struct HelperTimings {
    pub lock_wait: Duration,
    pub eof_wait: Duration,
    pub stage1: Duration,
    pub stage2: Duration,
    pub commit: Duration,
    pub launch_settle: Duration,
    pub terminate_grace: Duration,
    pub poll: Duration,
}

impl Default for HelperTimings {
    fn default() -> Self {
        Self {
            lock_wait: Duration::from_secs(5),
            eof_wait: Duration::from_secs(10),
            stage1: Duration::from_secs(rec::STAGE1_SECS),
            stage2: Duration::from_secs(rec::STAGE2_SECS),
            commit: Duration::from_secs(rec::COMMIT_SECS),
            launch_settle: Duration::from_secs(3),
            terminate_grace: rec::TERMINATE_GRACE,
            poll: Duration::from_millis(100),
        }
    }
}

/// How a helper run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HelperOutcome {
    /// Nothing was swapped (cancelled, another update running, checks failed).
    Aborted(String),
    /// The new version is in place and committed.
    Committed,
    /// The new version failed; the old one is back (and was relaunched if `relaunched`).
    RolledBack { why: String, relaunched: bool },
    /// Rolling back couldn't finish; the journal stays for recovery.
    RecoveryNeeded(String),
}

/// `--finish-update`: the real helper. Returns the exit status.
pub fn run_helper(args: &HelperArgs) -> i32 {
    let ops = AppOps;
    let Some(paths) = UpdatePaths::default_location() else {
        ops.log("no state directory");
        return 1;
    };
    // SAFETY: the old app passed these descriptors and nothing else in this process owns them.
    let (commit_rx, ready_tx) = unsafe {
        (
            File::from_raw_fd(args.commit_fd),
            File::from_raw_fd(args.ready_fd),
        )
    };
    let _ = set_cloexec(args.commit_fd, true);
    let _ = set_cloexec(args.ready_fd, true);
    let out = run_helper_with(
        args,
        &paths,
        &ops,
        &HelperTimings::default(),
        commit_rx,
        ready_tx,
    );
    ops.log(&format!("update {}: {out:?}", args.nonce));
    match out {
        HelperOutcome::Committed | HelperOutcome::Aborted(_) => 0,
        HelperOutcome::RolledBack { .. } => 0,
        HelperOutcome::RecoveryNeeded(_) => 1,
    }
}

/// The helper protocol over explicit paths, operations and pipe ends.
pub fn run_helper_with(
    args: &HelperArgs,
    paths: &UpdatePaths,
    ops: &dyn UpdateOps,
    t: &HelperTimings,
    mut commit_rx: File,
    mut ready_tx: File,
) -> HelperOutcome {
    // 1. updating.lock first; a stale journal is recovered before anything is written.
    let lock = match UpdatingLock::acquire_timeout(paths, t.lock_wait) {
        Ok(Some(l)) => l,
        Ok(None) => return HelperOutcome::Aborted("another update is running".into()),
        Err(e) => return HelperOutcome::Aborted(format!("updating.lock: {e}")),
    };
    if paths.tx_journal().exists() {
        let r = rec::recover_locked(paths, &lock, ops, &RecoveryMode::Manual);
        ops.log(&format!("recovered an earlier update first: {r:?}"));
        if paths.tx_journal().exists() {
            return HelperOutcome::Aborted("an earlier update is still unresolved".into());
        }
    }
    let mut tx = UpdateTx {
        nonce: args.nonce.clone(),
        phase: TxPhase::Prepared,
        started_at: ops.now(),
        stage1_deadline: None,
        stage2_deadline: None,
        commit_deadline: None,
        target_path: args.target.clone(),
        staging_path: args.staging.clone(),
        old_version: args.old_version.clone(),
        new_version: args.new_version.clone(),
        helper_path: paths.update_helper(&args.nonce),
        candidate_pid: None,
        relaunch_pending: false,
    };
    if let Err(e) = rec::validate_tx_paths(paths, &tx) {
        return HelperOutcome::Aborted(e);
    }
    if let Err(e) = us::write_tx(paths, &tx) {
        return HelperOutcome::Aborted(format!("couldn't write the journal: {e}"));
    }

    // Undoes `prepared` under updating.lock, releases it, then relaunches the old app if asked
    // and it has really exited.
    let abort_prepared = |lock: UpdatingLock, tx: &UpdateTx, why: String, relaunch_old: bool| {
        let _ = rec::remove_path(&tx.staging_path);
        let _ = us::remove_tx(paths);
        let _ = rec::remove_path(&tx.helper_path);
        drop(lock);
        if relaunch_old && !ops.pid_alive(args.old_pid) {
            let _ = ops.launch_restored(&tx.target_path);
        }
        HelperOutcome::Aborted(why)
    };

    // 2. Ready; then the commit byte, then EOF (the old app has exited).
    if ready_tx.write_all(b"R").is_err() {
        return abort_prepared(lock, &tx, "the app went away".into(), false);
    }
    drop(ready_tx);
    match read_byte(&mut commit_rx, None) {
        Ok(Some(_)) => {}
        _ => return abort_prepared(lock, &tx, "cancelled".into(), false),
    }
    match read_byte(&mut commit_rx, Some(t.eof_wait)) {
        Ok(None) => {}
        _ => return abort_prepared(lock, &tx, "the app did not exit".into(), true),
    }

    // 3. engine.lock, then swap.
    let engine = match EngineLockGuard::acquire_timeout(paths, t.lock_wait) {
        Ok(Some(g)) => g,
        _ => return abort_prepared(lock, &tx, "engine.lock stayed busy".into(), true),
    };
    tx.phase = TxPhase::Swapping;
    if let Err(e) = us::write_tx(paths, &tx) {
        drop(engine);
        return abort_prepared(lock, &tx, format!("couldn't write the journal: {e}"), true);
    }
    let checked = match ops.bundle_version(&tx.target_path) {
        Some(v) if v == tx.old_version => {
            ops.verify_staged(&tx.staging_path, &tx.new_version, &tx.old_version)
        }
        other => Err(format!("the installed app changed (version {other:?})")),
    };
    if let Err(e) = checked {
        drop(engine);
        return abort_prepared(lock, &tx, e, true);
    }
    if let Err(e) = ops.swap(&tx.staging_path, &tx.target_path) {
        drop(engine);
        return abort_prepared(lock, &tx, format!("couldn't swap: {e}"), true);
    }
    ops.register(&tx.target_path);

    // 4. Admit the candidate, persist deadlines, launch.
    if let Err(e) = us::write_tx_state(paths, &tx.nonce, TxControlState::Active) {
        return roll_back(
            paths,
            &lock,
            ops,
            t,
            tx,
            None,
            format!("tx state: {e}"),
            Some(engine),
        );
    }
    let now = ops.now();
    tx.phase = TxPhase::Verifying;
    tx.stage1_deadline = Some(now.saturating_add(ceil_secs(t.stage1)));
    tx.stage2_deadline = Some(now.saturating_add(ceil_secs(t.stage2)));
    tx.commit_deadline = Some(now.saturating_add(ceil_secs(t.commit)));
    if let Err(e) = us::write_tx(paths, &tx) {
        return roll_back(
            paths,
            &lock,
            ops,
            t,
            tx,
            None,
            format!("journal: {e}"),
            Some(engine),
        );
    }
    drop(engine);
    let start = Instant::now();
    let launched = match ops.launch_candidate(&tx.target_path, &tx.nonce) {
        Ok(l) => l,
        Err(e) => return roll_back(paths, &lock, ops, t, tx, None, format!("launch: {e}"), None),
    };

    // 5. Wait for the two acks.
    let mut launch_pid: Option<u32> = None;
    let failure = loop {
        match launched.wait(Duration::ZERO) {
            Some(Err(e)) => break format!("launch failed: {e}"),
            Some(Ok(pid)) => launch_pid = launch_pid.or(pid),
            None => {}
        }
        if tx.candidate_pid.is_none()
            && let Some(a) = us::read_ack(paths, &tx.nonce, false)
        {
            tx.candidate_pid = Some(a.pid);
            let _ = us::write_tx(paths, &tx);
        }
        let expected = tx.candidate_pid.or(launch_pid);
        if let Some(pid) = expected
            && !ops.pid_alive(pid)
        {
            break format!("the new version (pid {pid}) exited before it was ready");
        }
        if let Some(a) = us::read_ack(paths, &tx.nonce, true)
            && expected.is_none_or(|p| p == a.pid)
        {
            let elapsed = start.elapsed();
            if elapsed > t.stage2 || elapsed > t.commit || !ready_ack_timely(&tx, &a, ops.now()) {
                break "the new version became ready too late".into();
            }
            if !ops.pid_alive(a.pid) {
                break format!(
                    "the new version (pid {}) exited after reporting ready",
                    a.pid
                );
            }
            if EngineLockGuard::is_held(paths) && us::read_lock_holder_pid(paths) == Some(a.pid) {
                // Re-check deadlines right before committing so a slow check never commits late.
                let elapsed = start.elapsed();
                if elapsed > t.stage2 || elapsed > t.commit || !ready_ack_timely(&tx, &a, ops.now())
                {
                    break "the new version missed its commit deadline".into();
                }
                return commit(paths, lock, ops, tx);
            }
        }
        let elapsed = start.elapsed();
        if tx.candidate_pid.is_none() && elapsed > t.stage1 {
            break "the new version did not start".into();
        }
        if elapsed > t.stage2 {
            break "the new version did not become ready".into();
        }
        if elapsed > t.commit {
            break "the new version missed its commit deadline".into();
        }
        std::thread::sleep(t.poll);
    };
    roll_back(paths, &lock, ops, t, tx, Some(launched), failure, None)
}

fn ceil_secs(d: Duration) -> u64 {
    d.as_secs().saturating_add(u64::from(d.subsec_nanos() > 0))
}

fn effective_deadlines(tx: &UpdateTx) -> (u64, u64) {
    (
        tx.stage2_deadline
            .unwrap_or_else(|| tx.started_at.saturating_add(rec::STAGE2_SECS)),
        tx.commit_deadline
            .unwrap_or_else(|| tx.started_at.saturating_add(rec::COMMIT_SECS)),
    )
}

/// Checks that a ready ack carries a timestamp within the persisted stage-2 deadline and that the
/// persisted commit deadline has not passed.
fn ready_ack_timely(tx: &UpdateTx, ack: &AckRecord, now: u64) -> bool {
    let Some(ready_at) = ack.ready_at else {
        return false;
    };
    let (stage2, commit) = effective_deadlines(tx);
    ready_at <= stage2 && now <= commit
}

/// Strict commit order, all under `updating.lock`: journal `committed`, then the candidate's
/// `committed`, cleanup, journal removed, lock released. `tx-<nonce>-state` and the acks stay
/// for the candidate to delete.
fn commit(
    paths: &UpdatePaths,
    lock: UpdatingLock,
    ops: &dyn UpdateOps,
    mut tx: UpdateTx,
) -> HelperOutcome {
    tx.phase = TxPhase::Committed;
    if let Err(e) = us::write_tx(paths, &tx) {
        ops.log(&format!("couldn't record the commit: {e}"));
    }
    if let Err(e) = us::write_tx_state(paths, &tx.nonce, TxControlState::Committed) {
        // The journal says committed: recovery publishes it later.
        ops.log(&format!("couldn't publish the commit: {e}"));
        return HelperOutcome::RecoveryNeeded(e.to_string());
    }
    if let Err(e) = rec::remove_path(&tx.staging_path) {
        ops.log(&format!("couldn't remove the old version: {e}"));
    }
    let _ = us::remove_tx(paths);
    drop(lock);
    HelperOutcome::Committed
}

/// Revoke, stop the candidate, take `engine.lock`, swap back, relaunch the old version.
#[allow(clippy::too_many_arguments)]
fn roll_back(
    paths: &UpdatePaths,
    lock: &UpdatingLock,
    ops: &dyn UpdateOps,
    t: &HelperTimings,
    tx: UpdateTx,
    launched: Option<Box<dyn Launched>>,
    why: String,
    engine: Option<EngineLockGuard>,
) -> HelperOutcome {
    ops.log(&format!("rolling back: {why}"));
    let _ = us::write_tx_state(paths, &tx.nonce, TxControlState::Revoked);
    if let Some(l) = &launched {
        // Catch a launch that is still being answered, then stop exactly that instance.
        let _ = l.wait(t.launch_settle);
        l.terminate(t.terminate_grace);
    }
    let cand = tx
        .candidate_pid
        .or_else(|| us::read_ack(paths, &tx.nonce, false).map(|a| a.pid));
    if let Some(pid) = cand
        && ops.pid_alive(pid)
        && ops
            .pid_exe(pid)
            .is_some_and(|p| p.starts_with(tx.target_path.join("Contents/MacOS")))
    {
        ops.terminate(pid, t.terminate_grace);
    }
    let engine = match engine {
        Some(e) => e,
        None => match EngineLockGuard::acquire_timeout(paths, t.lock_wait * 2) {
            Ok(Some(g)) => g,
            _ => {
                return HelperOutcome::RecoveryNeeded(format!(
                    "{why}; engine.lock stayed busy, so the old version is restored on the next \
                     launch"
                ));
            }
        },
    };
    match rec::roll_back_locked(paths, lock, ops, &RecoveryMode::Manual, tx, engine) {
        RecoveryOutcome::Relaunched => HelperOutcome::RolledBack {
            why,
            relaunched: true,
        },
        RecoveryOutcome::RelaunchFailed(_) | RecoveryOutcome::Resolved => {
            HelperOutcome::RolledBack {
                why,
                relaunched: false,
            }
        }
        other => HelperOutcome::RecoveryNeeded(format!("{why}; {other:?}")),
    }
}

// ---------------------------------------------------------------------------------------------
// The candidate (new build started with --update-ack)

/// Admission of the new build: its `tx-<nonce>-state` is `active`, the journal is this
/// transaction in `verifying`, this executable is inside its target and this build is its new
/// version. Then writes `ack-<nonce>-started`. On `Err` the process must exit at once, with no
/// side effects. Call before starting Rune or the engine.
pub fn candidate_admit_and_mark_started(
    paths: &UpdatePaths,
    nonce: &Nonce,
    exe: &Path,
) -> Result<CandidateToken, String> {
    candidate_admit_with(paths, nonce, exe, env!("CARGO_PKG_VERSION"))
}

fn candidate_admit_with(
    paths: &UpdatePaths,
    nonce: &Nonce,
    exe: &Path,
    version: &str,
) -> Result<CandidateToken, String> {
    if us::read_tx_state(paths, nonce) != Some(TxControlState::Active) {
        return Err("this update is not active".into());
    }
    let tx = us::read_tx(paths)
        .ok()
        .flatten()
        .filter(|tx| tx.nonce == *nonce && tx.phase == TxPhase::Verifying)
        .ok_or("no matching update transaction")?;
    if !exe.starts_with(&tx.target_path) {
        return Err("not running from the updated app".into());
    }
    if tx.new_version != version {
        return Err(format!(
            "this is {version}, the update is {}",
            tx.new_version
        ));
    }
    us::write_ack(
        paths,
        &AckRecord {
            nonce: nonce.clone(),
            pid: std::process::id(),
            ready_at: None,
        },
    )
    .map_err(|e| e.to_string())?;
    Ok(CandidateToken::new(nonce.clone()))
}

/// Stage 2: the engine holds `engine.lock` (admitted again after taking it) and the tray (or
/// fallback window) is up. Independent of the driver and permissions.
pub fn candidate_mark_ready(paths: &UpdatePaths, nonce: &Nonce) -> Result<(), String> {
    us::write_ack(
        paths,
        &AckRecord {
            nonce: nonce.clone(),
            pid: std::process::id(),
            ready_at: Some(us::now_unix()),
        },
    )
    .map_err(|e| e.to_string())
}

/// What the candidate does after waiting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateDecision {
    /// Call `Engine::commit_candidate()` and unlock the UI.
    Committed,
    /// Revoked or gone: exit now with no side effects.
    Exit,
}

/// Polls `tx-<nonce>-state` until it is `committed` (then deletes the candidate's state and ack
/// files) or `revoked`/missing. If `updating.lock` comes free while still `active` (the helper
/// died after `ready`), takes it and commits on its own. Call after [`candidate_mark_ready`],
/// on a worker thread; `stop` ends the wait with [`CandidateDecision::Exit`].
pub fn candidate_wait_for_commit_or_takeover(
    paths: &UpdatePaths,
    nonce: &Nonce,
    poll: Duration,
    stop: &AtomicBool,
) -> CandidateDecision {
    loop {
        if stop.load(Ordering::Relaxed) {
            return CandidateDecision::Exit;
        }
        match us::read_tx_state(paths, nonce) {
            Some(TxControlState::Committed) => {
                rec::remove_state_and_acks(paths, nonce);
                return CandidateDecision::Committed;
            }
            Some(TxControlState::Revoked) | None => return CandidateDecision::Exit,
            Some(TxControlState::Active) => {}
        }
        if !UpdatingLock::is_held(paths)
            && let Ok(Some(lock)) = UpdatingLock::try_acquire(paths)
        {
            let d = self_commit(paths, nonce);
            drop(lock);
            if d == CandidateDecision::Committed {
                rec::remove_state_and_acks(paths, nonce);
            }
            return d;
        }
        std::thread::sleep(poll);
    }
}

/// Under `updating.lock`: re-read both files and finish the commit the dead helper owed us. If
/// the journal already reached `committed` before the helper died, publish that terminal state
/// unconditionally; if still `verifying`, validate the candidate's ready ack and sample the clock
/// immediately before transitioning the journal.
fn self_commit(paths: &UpdatePaths, nonce: &Nonce) -> CandidateDecision {
    self_commit_with(paths, nonce, us::now_unix)
}

fn self_commit_with(
    paths: &UpdatePaths,
    nonce: &Nonce,
    mut now: impl FnMut() -> u64,
) -> CandidateDecision {
    let tx = us::read_tx(paths)
        .ok()
        .flatten()
        .filter(|tx| tx.nonce == *nonce);
    match us::read_tx_state(paths, nonce) {
        Some(TxControlState::Committed) => {
            if let Some(tx) = tx {
                let _ = rec::remove_path(&tx.staging_path);
                let _ = us::remove_tx(paths);
            }
            CandidateDecision::Committed
        }
        Some(TxControlState::Active) => {
            let Some(mut tx) = tx else {
                return CandidateDecision::Exit;
            };
            match tx.phase {
                TxPhase::Committed => {
                    if us::write_tx_state(paths, nonce, TxControlState::Committed).is_err() {
                        return CandidateDecision::Exit;
                    }
                    let _ = rec::remove_path(&tx.staging_path);
                    let _ = us::remove_tx(paths);
                    CandidateDecision::Committed
                }
                TxPhase::Verifying => {
                    let me = std::process::id();
                    let Some(ack) = us::read_ack(paths, nonce, true).filter(|a| a.pid == me) else {
                        return CandidateDecision::Exit;
                    };
                    if !ready_ack_timely(&tx, &ack, now()) {
                        return CandidateDecision::Exit;
                    }
                    tx.phase = TxPhase::Committed;
                    if us::write_tx(paths, &tx).is_err() {
                        return CandidateDecision::Exit;
                    }
                    if us::write_tx_state(paths, nonce, TxControlState::Committed).is_err() {
                        return CandidateDecision::Exit;
                    }
                    let _ = rec::remove_path(&tx.staging_path);
                    let _ = us::remove_tx(paths);
                    CandidateDecision::Committed
                }
                _ => CandidateDecision::Exit,
            }
        }
        Some(TxControlState::Revoked) | None => CandidateDecision::Exit,
    }
}

/// `--wait-update-release`: waits up to `timeout` for `updating.lock` to be released (the
/// helper releases it right after launching the restored app). True once free.
pub fn wait_for_update_release(paths: &UpdatePaths, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if !UpdatingLock::is_held(paths) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

// ---------------------------------------------------------------------------------------------
// Recovery and cleanup at startup

/// Dead-helper recovery with the real operations ([`rec::recover`]).
pub fn recover_update(paths: &UpdatePaths, mode: RecoveryMode) -> RecoveryOutcome {
    rec::recover(paths, &AppOps, &mode)
}

/// Recovery as a normally starting `oto-app` runs it, before `Engine::start_with`. If
/// [`RecoveryOutcome::startup_should_exit`], exit quietly (the restored app was launched).
pub fn startup_recovery(exe: &Path) -> RecoveryOutcome {
    let Some(paths) = UpdatePaths::default_location() else {
        return RecoveryOutcome::NoTransaction;
    };
    let out = recover_update(
        &paths,
        RecoveryMode::Startup {
            exe: exe.to_path_buf(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
        },
    );
    if out == RecoveryOutcome::NoTransaction || out == RecoveryOutcome::Resolved {
        cleanup_stale_files(&paths);
        if let Some(cache) = verify::updates_cache_dir() {
            verify::reclaim_abandoned_work_dirs(&cache);
        }
    }
    out
}

/// `update-helper-<nonce> --recover-update`. Returns the exit status.
pub fn run_recover_update() -> i32 {
    let Some(paths) = UpdatePaths::default_location() else {
        eprintln!("oto: no state directory");
        return 1;
    };
    let out = recover_update(&paths, RecoveryMode::Manual);
    eprintln!("oto: update recovery: {out:?}");
    match out {
        RecoveryOutcome::Deferred(_) | RecoveryOutcome::RelaunchFailed(_) => 1,
        _ => 0,
    }
}

/// Files of finished transactions older than this are removed at startup.
const STALE_AGE: Duration = Duration::from_secs(10 * 60);
const HELPER_PREP_LOCK: &str = ".helper-prep.lock";

fn open_helper_prep_lock(paths: &UpdatePaths) -> io::Result<File> {
    fs::create_dir_all(&paths.dir)?;
    let mut o = fs::OpenOptions::new();
    o.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(paths.dir.join(HELPER_PREP_LOCK))
}

/// Removes leftovers of finished transactions: helper copies, state and ack files of other
/// transactions older than ten minutes (so a starting helper or a candidate reading its
/// `committed` is never raced), plus any crashed `.update-helper-<nonce>.tmp` files when no
/// helper copy is in flight. Only while holding `updating.lock`.
pub fn cleanup_stale_files(paths: &UpdatePaths) {
    cleanup_stale_files_with(paths, STALE_AGE);
}

fn cleanup_stale_files_with(paths: &UpdatePaths, max_age: Duration) {
    let Ok(Some(_lock)) = UpdatingLock::try_acquire(paths) else {
        return;
    };
    let prep_lock = open_helper_prep_lock(paths)
        .ok()
        .and_then(|f| f.try_lock().ok().map(|()| f));
    let current = us::read_tx(paths).ok().flatten().map(|tx| tx.nonce);
    let Ok(rd) = fs::read_dir(&paths.dir) else {
        return;
    };
    for e in rd.flatten() {
        let Some(name) = e.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if let Some(n) = name
            .strip_prefix(".update-helper-")
            .and_then(|r| r.strip_suffix(".tmp"))
        {
            if prep_lock.is_some() && Nonce::parse(n).is_ok() {
                let _ = rec::remove_path(&e.path());
            }
            continue;
        }
        let nonce = name
            .strip_prefix("update-helper-")
            .or_else(|| {
                name.strip_prefix("tx-")
                    .and_then(|r| r.strip_suffix("-state"))
            })
            .or_else(|| {
                name.strip_prefix("ack-")
                    .and_then(|r| r.strip_suffix("-started").or(r.strip_suffix("-ready")))
            });
        let Some(n) = nonce else {
            continue;
        };
        let Ok(n) = Nonce::parse(n) else { continue };
        if current.as_ref() == Some(&n) {
            continue;
        }
        let old = e
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|m| m.elapsed().ok())
            .is_some_and(|age| age > max_age);
        if old {
            let _ = rec::remove_path(&e.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::updater::verify::UpdateInfo;
    use crate::updater::verify::tests::Tmp;
    use std::sync::{Arc, Mutex};

    // ------------------------------------------------------------------ classification

    fn fake_bundle(root: &Path) -> PathBuf {
        let app = root.join("Applications/Oto.app");
        fs::create_dir_all(app.join("Contents/MacOS")).unwrap();
        fs::write(app.join("Contents/MacOS/oto-app"), "").unwrap();
        app.join("Contents/MacOS/oto-app")
    }

    #[test]
    fn dev_builds_and_bare_binaries_are_unsupported() {
        let dev = InstallMode::Unsupported(UnsupportedReason::DevBuild);
        assert_eq!(classify_with(None, true, true), dev);
        assert_eq!(
            classify_with(Some(Path::new("/repo/target/debug/oto-app")), true, true),
            dev
        );
        let t = Tmp::new("cls");
        let exe = fake_bundle(&t.0);
        assert_eq!(classify_with(Some(&exe), true, false), dev);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn an_owned_writable_bundle_on_apfs_updates_itself() {
        let t = Tmp::new("cls");
        let exe = fake_bundle(&t.0);
        assert_eq!(
            classify_with(Some(&exe), true, true),
            InstallMode::SelfUpdatable {
                bundle: t.0.join("Applications/Oto.app")
            }
        );
        assert_eq!(
            classify_with(Some(&exe), false, true),
            InstallMode::NotifyOnly(NotifyReason::NoInstanceLock)
        );
        let v = volume_info(&t.0).unwrap();
        assert!(!v.read_only);
        assert!(v.rename_swap || v.fs_type == "apfs", "{v:?}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn translocated_read_only_and_unwritable_bundles_only_notify() {
        let t = Tmp::new("cls");
        let tr = t.0.join("AppTranslocation/ABC/d");
        fs::create_dir_all(tr.join("Oto.app/Contents/MacOS")).unwrap();
        assert_eq!(
            classify_with(Some(&tr.join("Oto.app/Contents/MacOS/oto-app")), true, true),
            InstallMode::NotifyOnly(NotifyReason::NotInApplications)
        );
        // A missing path has no volume to ask: treated like a read-only mount.
        assert_eq!(
            classify_with(
                Some(Path::new("/nonexistent/Oto.app/Contents/MacOS/oto-app")),
                true,
                true
            ),
            InstallMode::NotifyOnly(NotifyReason::NotInApplications)
        );
        // The system volume is read-only.
        assert_eq!(
            classify_with(
                Some(Path::new(
                    "/System/Applications/Oto.app/Contents/MacOS/oto-app"
                )),
                true,
                true
            ),
            InstallMode::NotifyOnly(NotifyReason::NotInApplications)
        );

        let exe = fake_bundle(&t.0);
        let apps = t.0.join("Applications");
        set_mode(&apps, 0o555).unwrap();
        let mode = classify_with(Some(&exe), true, true);
        set_mode(&apps, 0o755).unwrap();
        assert_eq!(
            mode,
            InstallMode::NotifyOnly(NotifyReason::RequiresManualInstall)
        );

        // A symlinked bundle is not swapped.
        let real = t.0.join("elsewhere/Oto.app");
        fs::create_dir_all(real.join("Contents/MacOS")).unwrap();
        let link_dir = t.0.join("Linked");
        fs::create_dir_all(&link_dir).unwrap();
        std::os::unix::fs::symlink(&real, link_dir.join("Oto.app")).unwrap();
        assert_eq!(
            classify_with(
                Some(&link_dir.join("Oto.app/Contents/MacOS/oto-app")),
                true,
                true
            ),
            InstallMode::NotifyOnly(NotifyReason::RequiresManualInstall)
        );
    }

    // ------------------------------------------------------------------ primitives

    #[cfg(target_os = "macos")]
    #[test]
    fn clone_and_copy_fallback_preserve_the_tree() {
        use std::os::unix::fs::PermissionsExt;
        let t = Tmp::new("clone");
        let src = t.0.join("Oto.app");
        fs::create_dir_all(src.join("Contents/MacOS")).unwrap();
        fs::write(src.join("Contents/MacOS/oto-app"), "bin").unwrap();
        set_mode(&src.join("Contents/MacOS/oto-app"), 0o755).unwrap();
        for (i, force) in [false, true].into_iter().enumerate() {
            let dst = t.0.join(format!(".Oto-update-{i}"));
            clone_tree_with(&src, &dst, force).unwrap();
            let exe = dst.join("Contents/MacOS/oto-app");
            assert_eq!(fs::read_to_string(&exe).unwrap(), "bin");
            assert_eq!(
                fs::metadata(&exe).unwrap().permissions().mode() & 0o777,
                0o755
            );
            // An existing destination is an error, not a merge.
            assert!(clone_tree_with(&src, &dst, force).is_err());
        }
        assert!(clone_tree(&t.0.join("missing"), &t.0.join("x")).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rename_swap_exchanges_and_swaps_back() {
        let t = Tmp::new("swap");
        let (a, b) = (t.0.join("Oto.app"), t.0.join(".Oto-update-n"));
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        fs::write(a.join("v"), "old").unwrap();
        fs::write(b.join("v"), "new").unwrap();
        rename_swap(&b, &a).unwrap();
        assert_eq!(fs::read_to_string(a.join("v")).unwrap(), "new");
        rename_swap(&b, &a).unwrap();
        assert_eq!(fs::read_to_string(a.join("v")).unwrap(), "old");
    }

    #[test]
    fn helper_arguments_round_trip() {
        let a = HelperArgs {
            nonce: Nonce::parse("n1").unwrap(),
            target: "/Applications/Oto.app".into(),
            staging: "/Applications/.Oto-update-n1".into(),
            old_version: "0.1.0".into(),
            new_version: "0.2.0".into(),
            old_pid: 42,
            commit_fd: 5,
            ready_fd: 7,
        };
        assert_eq!(
            parse_cli_mode(a.to_args()).unwrap(),
            CliMode::FinishUpdate(a.clone())
        );
        let os = |v: &[&str]| v.iter().map(OsString::from).collect::<Vec<_>>();
        assert_eq!(
            parse_cli_mode(os(&["--update-ack", "abc"])).unwrap(),
            CliMode::UpdateAck(Nonce::parse("abc").unwrap())
        );
        assert!(parse_cli_mode(os(&["--update-ack", "../x"])).is_err());
        assert!(parse_cli_mode(os(&["--update-ack"])).is_err());
        assert_eq!(
            parse_cli_mode(os(&["--wait-update-release"])).unwrap(),
            CliMode::WaitUpdateRelease
        );
        assert_eq!(
            parse_cli_mode(os(&["--recover-update"])).unwrap(),
            CliMode::RecoverUpdate
        );
        assert_eq!(parse_cli_mode(os(&[])).unwrap(), CliMode::Normal);
        assert_eq!(parse_cli_mode(os(&["-psn_0_1"])).unwrap(), CliMode::Normal);
        let mut bad = a.to_args();
        bad.truncate(bad.len() - 2);
        assert!(parse_cli_mode(bad).is_err());
    }

    #[test]
    fn a_dropped_commit_pipe_reads_as_eof_and_a_dead_reader_as_epipe() {
        let (r, w) = pipe().unwrap();
        let (mut r, w) = (File::from(r), File::from(w));
        drop(w);
        assert_eq!(
            read_byte(&mut r, Some(Duration::from_secs(1))).unwrap(),
            None
        );

        let (r, w) = pipe().unwrap();
        let (r, mut w) = (File::from(r), File::from(w));
        drop(r);
        let e = w.write_all(&[1]).unwrap_err();
        assert_eq!(e.raw_os_error(), Some(libc::EPIPE));

        let (r, _w) = pipe().unwrap();
        let mut r = File::from(r);
        let e = read_byte(&mut r, Some(Duration::from_millis(50))).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::TimedOut);
    }

    // ------------------------------------------------------------------ the protocol

    /// Bundles are directories with a `version` file; processes are a table; the "candidate"
    /// is a closure run on a thread when launched.
    #[derive(Default)]
    struct Fake {
        procs: Mutex<Vec<(u32, PathBuf)>>,
        launched_restored: Mutex<Vec<PathBuf>>,
        candidate: Mutex<Option<Box<dyn FnOnce() + Send>>>,
        launch_fails: AtomicBool,
        verify_fails: AtomicBool,
        terminated: Mutex<Vec<u32>>,
    }

    struct FakeLaunch(Option<u32>);
    impl Launched for FakeLaunch {
        fn wait(&self, _: Duration) -> Option<Result<Option<u32>, String>> {
            Some(Ok(self.0))
        }
        fn terminate(&self, _: Duration) {}
    }

    impl RecoveryOps for Fake {
        fn swap(&self, a: &Path, b: &Path) -> io::Result<()> {
            rename_swap(a, b)
        }
        fn register(&self, _: &Path) {}
        fn bundle_version(&self, app: &Path) -> Option<String> {
            fs::read_to_string(app.join("version")).ok()
        }
        fn pid_exe(&self, pid: u32) -> Option<PathBuf> {
            let p = self.procs.lock().unwrap();
            p.iter().find(|(x, _)| *x == pid).map(|(_, e)| e.clone())
        }
        fn pid_alive(&self, pid: u32) -> bool {
            // This test process stands in for the old app and the candidate.
            let me = std::process::id();
            self.procs.lock().unwrap().iter().any(|(x, _)| *x == pid)
                || (pid == me && !self.terminated.lock().unwrap().contains(&me))
        }
        fn terminate(&self, pid: u32, _: Duration) -> bool {
            self.procs.lock().unwrap().retain(|(x, _)| *x != pid);
            self.terminated.lock().unwrap().push(pid);
            true
        }
        fn launch_restored(&self, target: &Path) -> Result<(), String> {
            self.launched_restored.lock().unwrap().push(target.into());
            Ok(())
        }
        fn log(&self, _: &str) {}
    }

    impl UpdateOps for Fake {
        fn verify_staged(&self, app: &Path, version: &str, _: &str) -> Result<(), String> {
            if self.verify_fails.load(Ordering::Relaxed) {
                return Err("bad signature".into());
            }
            match self.bundle_version(app) {
                Some(v) if v == version => Ok(()),
                v => Err(format!("version {v:?}")),
            }
        }
        fn launch_candidate(&self, target: &Path, _: &Nonce) -> Result<Box<dyn Launched>, String> {
            if self.launch_fails.load(Ordering::Relaxed) {
                return Err("no".into());
            }
            let pid = std::process::id();
            self.procs
                .lock()
                .unwrap()
                .push((pid, target.join("Contents/MacOS/oto-app")));
            if let Some(c) = self.candidate.lock().unwrap().take() {
                std::thread::spawn(c);
            }
            Ok(Box::new(FakeLaunch(Some(pid))))
        }
    }

    fn fast() -> HelperTimings {
        HelperTimings {
            lock_wait: Duration::from_millis(300),
            eof_wait: Duration::from_millis(500),
            stage1: Duration::from_millis(400),
            stage2: Duration::from_millis(800),
            commit: Duration::from_secs(2),
            launch_settle: Duration::ZERO,
            terminate_grace: Duration::ZERO,
            poll: Duration::from_millis(10),
        }
    }

    struct World {
        t: Tmp,
        paths: UpdatePaths,
        target: PathBuf,
        staged: StagedUpdate,
        helper_exe: PathBuf,
    }

    fn bundle(p: &Path, v: &str) {
        fs::create_dir_all(p.join("Contents/MacOS")).unwrap();
        fs::write(p.join("version"), v).unwrap();
    }

    impl World {
        fn new() -> Self {
            let t = Tmp::new("apply");
            let paths = UpdatePaths::new(t.0.join("state"));
            let target = t.0.join("Applications/Oto.app");
            bundle(&target, "1.0.0");
            let cache = t.0.join("cache/1.1.0-app");
            bundle(&cache.join("Oto.app"), "1.1.0");
            let helper_exe = t.0.join("oto-app-binary");
            fs::write(&helper_exe, "exe").unwrap();
            let staged = StagedUpdate::unowned(
                UpdateInfo {
                    version: "1.1.0".into(),
                    notes_url: String::new(),
                    asset_url: String::new(),
                    sha256: String::new(),
                    size: 1,
                    min_os: "12.0".into(),
                    driver_version: 1,
                    min_driver_version: 0,
                    published_at: String::new(),
                },
                cache.join("Oto.app"),
                cache,
            );
            Self {
                t,
                paths,
                target,
                staged,
                helper_exe,
            }
        }
        /// Puts the downloaded copy back in the cache (preparing consumes it).
        fn restage(&self) {
            if !self.staged.cached_app_path.exists() {
                bundle(&self.staged.cached_app_path, "1.1.0");
            }
        }
        fn version(&self) -> String {
            fs::read_to_string(self.target.join("version")).unwrap()
        }
        fn leftovers(&self) -> Vec<String> {
            let mut v: Vec<String> = fs::read_dir(self.target.parent().unwrap())
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|n| n != "Oto.app")
                .collect();
            v.sort();
            v
        }
    }

    /// Prepares with the helper protocol on a thread; returns the prepared update and the
    /// helper's outcome receiver.
    fn prepare(
        w: &World,
        ops: Arc<Fake>,
        timings: HelperTimings,
    ) -> (
        Result<PreparedUpdate, String>,
        std::sync::mpsc::Receiver<HelperOutcome>,
    ) {
        w.restage();
        let (tx, rx) = std::sync::mpsc::channel();
        let paths = w.paths.clone();
        let spawn = move |_: &Path, a: &HelperArgs, c: OwnedFd, r: OwnedFd| {
            let (a, paths, ops, tx) = (a.clone(), paths.clone(), ops.clone(), tx.clone());
            std::thread::spawn(move || {
                let out =
                    run_helper_with(&a, &paths, &*ops, &timings, File::from(c), File::from(r));
                let _ = tx.send(out);
            });
            Ok(())
        };
        let verify_ops = Fake::default();
        let p = PreparedUpdate::prepare_with(
            &w.staged,
            &w.target,
            "1.0.0",
            &w.paths,
            &verify_ops,
            &w.helper_exe,
            &spawn,
        );
        (p, rx)
    }

    fn outcome(rx: &std::sync::mpsc::Receiver<HelperOutcome>) -> HelperOutcome {
        rx.recv_timeout(Duration::from_secs(10)).unwrap()
    }

    fn hold_engine_lock(paths: &UpdatePaths, pid: u32) -> File {
        fs::create_dir_all(&paths.dir).unwrap();
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(paths.engine_lock())
            .unwrap();
        file.try_lock().unwrap();
        fs::write(paths.engine_lock(), format!("{pid}\n")).unwrap();
        file
    }

    /// A well-behaved candidate: admitted, started, holds `engine.lock`, ready, then waits for
    /// commit or revocation.
    fn good_candidate(
        paths: UpdatePaths,
        target: PathBuf,
        nonce: Arc<Mutex<Option<Nonce>>>,
    ) -> Box<dyn FnOnce() + Send> {
        Box::new(move || {
            let n = loop {
                if let Some(n) = nonce.lock().unwrap().clone() {
                    break n;
                }
                std::thread::sleep(Duration::from_millis(5));
            };
            let exe = target.join("Contents/MacOS/oto-app");
            candidate_admit_with(&paths, &n, &exe, "1.1.0").unwrap();
            let _engine = hold_engine_lock(&paths, std::process::id());
            candidate_mark_ready(&paths, &n).unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                if matches!(
                    us::read_tx_state(&paths, &n),
                    Some(TxControlState::Committed | TxControlState::Revoked) | None
                ) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        })
    }

    fn nonce_of(p: &PreparedUpdate) -> Nonce {
        p.nonce.clone()
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_healthy_update_commits_and_cleans_up() {
        let w = World::new();
        let ops = Arc::new(Fake::default());
        let slot = Arc::new(Mutex::new(None));
        *ops.candidate.lock().unwrap() = Some(good_candidate(
            w.paths.clone(),
            w.target.clone(),
            slot.clone(),
        ));
        let (p, rx) = prepare(&w, ops.clone(), fast());
        let p = p.unwrap();
        let n = nonce_of(&p);
        *slot.lock().unwrap() = Some(n.clone());
        // Prepared: the journal exists, the helper holds updating.lock, the cache is gone.
        assert_eq!(
            us::read_tx(&w.paths).unwrap().unwrap().phase,
            TxPhase::Prepared
        );
        assert!(UpdatingLock::is_held(&w.paths));
        assert!(!w.staged.cache_dir.exists());
        assert!(p.helper_path.is_file());
        p.commit_and_signal().unwrap();
        assert_eq!(outcome(&rx), HelperOutcome::Committed);
        assert_eq!(w.version(), "1.1.0");
        assert!(w.leftovers().is_empty(), "{:?}", w.leftovers());
        assert!(us::read_tx(&w.paths).unwrap().is_none());
        assert!(!UpdatingLock::is_held(&w.paths));
        // The candidate's files are left for the candidate.
        assert_eq!(
            us::read_tx_state(&w.paths, &n),
            Some(TxControlState::Committed)
        );
        let stop = AtomicBool::new(false);
        assert_eq!(
            candidate_wait_for_commit_or_takeover(&w.paths, &n, Duration::from_millis(5), &stop),
            CandidateDecision::Committed
        );
        assert_eq!(us::read_tx_state(&w.paths, &n), None);
        assert!(us::read_ack(&w.paths, &n, true).is_none());
        let _ = &w.t;
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn cancelling_before_the_commit_byte_changes_nothing() {
        let w = World::new();
        let ops = Arc::new(Fake::default());
        let (p, rx) = prepare(&w, ops, fast());
        let p = p.unwrap();
        let helper = p.helper_path.clone();
        p.cancel();
        assert!(matches!(outcome(&rx), HelperOutcome::Aborted(_)));
        assert_eq!(w.version(), "1.0.0");
        assert!(w.leftovers().is_empty(), "{:?}", w.leftovers());
        assert!(us::read_tx(&w.paths).unwrap().is_none());
        assert!(!helper.exists());
        assert!(!UpdatingLock::is_held(&w.paths));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_dead_helper_makes_the_commit_fail_with_epipe() {
        let w = World::new();
        let p = {
            // A "helper" that says ready and dies.
            let spawn = |_: &Path, _: &HelperArgs, c: OwnedFd, r: OwnedFd| {
                let mut r = File::from(r);
                r.write_all(b"R").unwrap();
                drop(c);
                Ok(())
            };
            w.restage();
            PreparedUpdate::prepare_with(
                &w.staged,
                &w.target,
                "1.0.0",
                &w.paths,
                &Fake::default(),
                &w.helper_exe,
                &spawn,
            )
            .unwrap()
        };
        let e = p.commit_and_signal().unwrap_err();
        assert!(e.contains("stopped unexpectedly"), "{e}");
        assert_eq!(w.version(), "1.0.0");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_helper_that_never_gets_ready_is_cleaned_up() {
        let w = World::new();
        let spawn = |_: &Path, _: &HelperArgs, c: OwnedFd, r: OwnedFd| {
            drop((c, r));
            Ok(())
        };
        w.restage();
        let e = PreparedUpdate::prepare_with(
            &w.staged,
            &w.target,
            "1.0.0",
            &w.paths,
            &Fake::default(),
            &w.helper_exe,
            &spawn,
        )
        .err()
        .unwrap();
        assert!(e.contains("couldn't start"), "{e}");
        assert!(w.leftovers().is_empty(), "{:?}", w.leftovers());
        // A staged copy that fails verification is removed too.
        let bad = Fake::default();
        bad.verify_fails.store(true, Ordering::Relaxed);
        w.restage();
        let e = PreparedUpdate::prepare_with(
            &w.staged,
            &w.target,
            "1.0.0",
            &w.paths,
            &bad,
            &w.helper_exe,
            &spawn,
        )
        .err()
        .unwrap();
        assert_eq!(e, "bad signature");
        assert!(w.leftovers().is_empty(), "{:?}", w.leftovers());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_busy_updating_lock_or_stale_journal_blocks_a_new_update() {
        let w = World::new();
        let held = UpdatingLock::try_acquire(&w.paths).unwrap().unwrap();
        let (p, rx) = prepare(&w, Arc::new(Fake::default()), fast());
        assert!(p.is_err());
        assert!(matches!(outcome(&rx), HelperOutcome::Aborted(_)));
        drop(held);
        assert!(w.leftovers().is_empty(), "{:?}", w.leftovers());

        // A stale `prepared` journal is recovered first, then the update proceeds.
        let old = Nonce::parse("old").unwrap();
        let stale_staging = w.target.parent().unwrap().join(".Oto-update-old");
        bundle(&stale_staging, "0.9.0");
        us::write_tx(
            &w.paths,
            &UpdateTx {
                nonce: old.clone(),
                phase: TxPhase::Prepared,
                started_at: 1,
                stage1_deadline: None,
                stage2_deadline: None,
                commit_deadline: None,
                target_path: w.target.clone(),
                staging_path: stale_staging.clone(),
                old_version: "0.8.0".into(),
                new_version: "0.9.0".into(),
                helper_path: w.paths.update_helper(&old),
                candidate_pid: None,
                relaunch_pending: false,
            },
        )
        .unwrap();
        let (p, rx) = prepare(&w, Arc::new(Fake::default()), fast());
        let p = p.unwrap();
        assert!(!stale_staging.exists());
        assert_eq!(us::read_tx(&w.paths).unwrap().unwrap().nonce, p.nonce);
        p.cancel();
        assert!(matches!(outcome(&rx), HelperOutcome::Aborted(_)));

        // An unresolved journal whose engine.lock is held by someone else can't be cleared:
        // the new update is refused and the old journal kept.
        let mut tx = us::read_tx(&w.paths).unwrap();
        assert!(tx.is_none());
        let n = Nonce::parse("busy").unwrap();
        let busy = UpdateTx {
            nonce: n.clone(),
            phase: TxPhase::Swapping,
            started_at: 1,
            stage1_deadline: None,
            stage2_deadline: None,
            commit_deadline: None,
            target_path: w.target.clone(),
            staging_path: w.target.parent().unwrap().join(".Oto-update-busy"),
            old_version: "0.8.0".into(),
            new_version: "0.9.0".into(),
            helper_path: w.paths.update_helper(&n),
            candidate_pid: None,
            relaunch_pending: false,
        };
        us::write_tx(&w.paths, &busy).unwrap();
        tx = Some(busy);
        let engine = EngineLockGuard::try_acquire(&w.paths).unwrap().unwrap();
        let (p, rx) = prepare(&w, Arc::new(Fake::default()), fast());
        assert!(p.is_err());
        assert!(matches!(outcome(&rx), HelperOutcome::Aborted(_)));
        assert_eq!(us::read_tx(&w.paths).unwrap(), tx);
        drop(engine);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_candidate_that_never_starts_is_rolled_back() {
        let w = World::new();
        let ops = Arc::new(Fake::default());
        let (p, rx) = prepare(&w, ops.clone(), fast());
        p.unwrap().commit_and_signal().unwrap();
        let out = outcome(&rx);
        assert!(
            matches!(
                out,
                HelperOutcome::RolledBack {
                    relaunched: true,
                    ..
                }
            ),
            "{out:?}"
        );
        assert_eq!(w.version(), "1.0.0");
        assert!(w.leftovers().is_empty(), "{:?}", w.leftovers());
        assert_eq!(
            *ops.launched_restored.lock().unwrap(),
            std::slice::from_ref(&w.target)
        );
        assert!(us::read_tx(&w.paths).unwrap().is_none());
        assert!(!UpdatingLock::is_held(&w.paths));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_candidate_that_starts_but_never_gets_ready_is_revoked_and_stopped() {
        let w = World::new();
        let ops = Arc::new(Fake::default());
        let slot = Arc::new(Mutex::new(None::<Nonce>));
        let (paths, target, s2) = (w.paths.clone(), w.target.clone(), slot.clone());
        let seen_revoked = Arc::new(AtomicBool::new(false));
        let seen = seen_revoked.clone();
        *ops.candidate.lock().unwrap() = Some(Box::new(move || {
            let n = loop {
                if let Some(n) = s2.lock().unwrap().clone() {
                    break n;
                }
                std::thread::sleep(Duration::from_millis(5));
            };
            candidate_admit_with(&paths, &n, &target.join("Contents/MacOS/oto-app"), "1.1.0")
                .unwrap();
            // Hangs; then sees revocation (or its files gone) and exits.
            let stop = AtomicBool::new(false);
            let d =
                candidate_wait_for_commit_or_takeover(&paths, &n, Duration::from_millis(5), &stop);
            seen.store(d == CandidateDecision::Exit, Ordering::Relaxed);
        }));
        let (p, rx) = prepare(&w, ops.clone(), fast());
        let p = p.unwrap();
        *slot.lock().unwrap() = Some(p.nonce.clone());
        p.commit_and_signal().unwrap();
        let out = outcome(&rx);
        assert!(matches!(out, HelperOutcome::RolledBack { .. }), "{out:?}");
        assert_eq!(w.version(), "1.0.0");
        assert!(ops.terminated.lock().unwrap().contains(&std::process::id()));
        std::thread::sleep(Duration::from_millis(100));
        assert!(seen_revoked.load(Ordering::Relaxed));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_candidate_that_exits_after_writing_ready_is_rolled_back() {
        let w = World::new();
        let ops = Arc::new(Fake::default());
        let slot = Arc::new(Mutex::new(None::<Nonce>));
        let (paths, target, s2, ops2) =
            (w.paths.clone(), w.target.clone(), slot.clone(), ops.clone());
        *ops.candidate.lock().unwrap() = Some(Box::new(move || {
            let n = loop {
                if let Some(n) = s2.lock().unwrap().clone() {
                    break n;
                }
                std::thread::sleep(Duration::from_millis(5));
            };
            candidate_admit_with(&paths, &n, &target.join("Contents/MacOS/oto-app"), "1.1.0")
                .unwrap();
            // Writes ready and immediately crashes (engine.lock released, PID dead).
            let engine = hold_engine_lock(&paths, std::process::id());
            candidate_mark_ready(&paths, &n).unwrap();
            drop(engine);
            ops2.terminate(std::process::id(), Duration::ZERO);
        }));
        let mut t = fast();
        t.poll = Duration::from_millis(60);
        let (p, rx) = prepare(&w, ops.clone(), t);
        let p = p.unwrap();
        *slot.lock().unwrap() = Some(p.nonce.clone());
        p.commit_and_signal().unwrap();
        let out = outcome(&rx);
        assert!(matches!(out, HelperOutcome::RolledBack { .. }), "{out:?}");
        assert_eq!(w.version(), "1.0.0");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_ready_ack_after_stage2_deadline_does_not_commit() {
        let w = World::new();
        let ops = Arc::new(Fake::default());
        let slot = Arc::new(Mutex::new(None::<Nonce>));
        let (paths, target, s2) = (w.paths.clone(), w.target.clone(), slot.clone());
        *ops.candidate.lock().unwrap() = Some(Box::new(move || {
            let n = loop {
                if let Some(n) = s2.lock().unwrap().clone() {
                    break n;
                }
                std::thread::sleep(Duration::from_millis(5));
            };
            candidate_admit_with(&paths, &n, &target.join("Contents/MacOS/oto-app"), "1.1.0")
                .unwrap();
            let _engine = hold_engine_lock(&paths, std::process::id());
            // Establish a late ready ack (recorded ready_at past the persisted stage2_deadline)
            // before the helper checks it.
            let tx = us::read_tx(&paths).unwrap().unwrap();
            let late = tx.stage2_deadline.unwrap() + 1;
            us::write_ack(
                &paths,
                &AckRecord {
                    nonce: n.clone(),
                    pid: std::process::id(),
                    ready_at: Some(late),
                },
            )
            .unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                if matches!(
                    us::read_tx_state(&paths, &n),
                    Some(TxControlState::Committed | TxControlState::Revoked) | None
                ) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }));
        let (p, rx) = prepare(&w, ops.clone(), fast());
        let p = p.unwrap();
        *slot.lock().unwrap() = Some(p.nonce.clone());
        p.commit_and_signal().unwrap();
        let out = outcome(&rx);
        assert!(matches!(out, HelperOutcome::RolledBack { .. }), "{out:?}");
        assert_eq!(w.version(), "1.0.0");

        // Timely ready_at, but commit deadline already passed before commit decision.
        let ops = Arc::new(Fake::default());
        let slot = Arc::new(Mutex::new(None::<Nonce>));
        *ops.candidate.lock().unwrap() = Some(good_candidate(
            w.paths.clone(),
            w.target.clone(),
            slot.clone(),
        ));
        let mut t = fast();
        t.commit = Duration::ZERO;
        let (p, rx) = prepare(&w, ops.clone(), t);
        let p = p.unwrap();
        *slot.lock().unwrap() = Some(p.nonce.clone());
        p.commit_and_signal().unwrap();
        let out = outcome(&rx);
        assert!(matches!(out, HelperOutcome::RolledBack { .. }), "{out:?}");
        assert_eq!(w.version(), "1.0.0");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn prepare_refreshes_helper_mtime_from_an_aged_source_executable() {
        let w = World::new();
        let old_time = std::time::SystemTime::now() - Duration::from_secs(3600);
        fs::OpenOptions::new()
            .write(true)
            .open(&w.helper_exe)
            .unwrap()
            .set_modified(old_time)
            .unwrap();
        // Capture the helper path as soon as spawn is invoked (before any journal is written)
        // and verify startup cleanup does not delete it.
        let spawn = |helper: &Path, _: &HelperArgs, _c: OwnedFd, r: OwnedFd| {
            cleanup_stale_files(&w.paths);
            assert!(
                helper.is_file(),
                "freshly copied helper from an aged binary must survive startup cleanup"
            );
            let age = fs::metadata(helper)
                .unwrap()
                .modified()
                .unwrap()
                .elapsed()
                .unwrap_or_default();
            assert!(age < Duration::from_secs(60), "helper age was {age:?}");
            let mut r = File::from(r);
            r.write_all(b"R").unwrap();
            Ok(())
        };
        w.restage();
        let p = PreparedUpdate::prepare_with(
            &w.staged,
            &w.target,
            "1.0.0",
            &w.paths,
            &Fake::default(),
            &w.helper_exe,
            &spawn,
        )
        .unwrap();
        p.cancel();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_failed_launch_or_a_changed_target_rolls_back_or_aborts() {
        let w = World::new();
        let ops = Arc::new(Fake::default());
        ops.launch_fails.store(true, Ordering::Relaxed);
        let (p, rx) = prepare(&w, ops.clone(), fast());
        p.unwrap().commit_and_signal().unwrap();
        assert!(matches!(outcome(&rx), HelperOutcome::RolledBack { .. }));
        assert_eq!(w.version(), "1.0.0");

        // The installed app changed between staging and the swap: no swap at all.
        let ops = Arc::new(Fake::default());
        let (p, rx) = prepare(&w, ops.clone(), fast());
        let p = p.unwrap();
        fs::write(w.target.join("version"), "1.0.5").unwrap();
        p.commit_and_signal().unwrap();
        assert!(matches!(outcome(&rx), HelperOutcome::Aborted(_)));
        assert_eq!(w.version(), "1.0.5");
        assert!(w.leftovers().is_empty(), "{:?}", w.leftovers());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_held_commit_pipe_aborts_without_swapping() {
        // The old app writes the byte but never exits (EOF never comes).
        let w = World::new();
        let ops = Arc::new(Fake::default());
        let (p, rx) = prepare(&w, ops.clone(), fast());
        let mut p = p.unwrap();
        let f = p.commit_tx.take().unwrap();
        (&f).write_all(&[1]).unwrap();
        assert!(matches!(outcome(&rx), HelperOutcome::Aborted(_)));
        drop(f);
        assert_eq!(w.version(), "1.0.0");
        assert!(w.leftovers().is_empty(), "{:?}", w.leftovers());
        // The old app is still alive (this test), so nothing is relaunched.
        assert!(ops.launched_restored.lock().unwrap().is_empty());
    }

    // ------------------------------------------------------------------ candidate

    fn verifying(w: &World, n: &Nonce) -> UpdateTx {
        let staging = w.target.parent().unwrap().join(format!(".Oto-update-{n}"));
        bundle(&staging, "1.0.0");
        let now = us::now_unix();
        let tx = UpdateTx {
            nonce: n.clone(),
            phase: TxPhase::Verifying,
            started_at: now,
            stage1_deadline: Some(now + rec::STAGE1_SECS),
            stage2_deadline: Some(now + rec::STAGE2_SECS),
            commit_deadline: Some(now + rec::COMMIT_SECS),
            target_path: w.target.clone(),
            staging_path: staging,
            old_version: "1.0.0".into(),
            new_version: "1.1.0".into(),
            helper_path: w.paths.update_helper(n),
            candidate_pid: None,
            relaunch_pending: false,
        };
        us::write_tx(&w.paths, &tx).unwrap();
        us::write_tx_state(&w.paths, n, TxControlState::Active).unwrap();
        tx
    }

    #[test]
    fn candidate_admission_rules() {
        let w = World::new();
        let n = Nonce::parse("c1").unwrap();
        let exe = w.target.join("Contents/MacOS/oto-app");
        assert!(candidate_admit_with(&w.paths, &n, &exe, "1.1.0").is_err());
        verifying(&w, &n);
        assert!(candidate_admit_with(&w.paths, &n, &exe, "1.2.0").is_err());
        assert!(
            candidate_admit_with(&w.paths, &n, Path::new("/elsewhere/oto-app"), "1.1.0").is_err()
        );
        assert!(
            candidate_admit_with(&w.paths, &Nonce::parse("c2").unwrap(), &exe, "1.1.0").is_err()
        );
        let tok = candidate_admit_with(&w.paths, &n, &exe, "1.1.0").unwrap();
        assert_eq!(tok.nonce, n);
        assert_eq!(
            us::read_ack(&w.paths, &n, false).unwrap().pid,
            std::process::id()
        );
        us::write_tx_state(&w.paths, &n, TxControlState::Revoked).unwrap();
        assert!(candidate_admit_with(&w.paths, &n, &exe, "1.1.0").is_err());
    }

    #[test]
    fn a_stuck_candidate_commits_itself_when_the_helper_is_gone() {
        let w = World::new();
        let n = Nonce::parse("c3").unwrap();
        let tx = verifying(&w, &n);
        candidate_mark_ready(&w.paths, &n).unwrap();
        // Nobody holds updating.lock: the helper died after `ready`.
        let stop = AtomicBool::new(false);
        let d =
            candidate_wait_for_commit_or_takeover(&w.paths, &n, Duration::from_millis(5), &stop);
        assert_eq!(d, CandidateDecision::Committed);
        assert!(!tx.staging_path.exists());
        assert!(us::read_tx(&w.paths).unwrap().is_none());
        assert_eq!(us::read_tx_state(&w.paths, &n), None);
        assert!(us::read_ack(&w.paths, &n, true).is_none());
    }

    #[test]
    fn self_commit_enforces_ready_ack_and_deadlines() {
        let w = World::new();
        let n = Nonce::parse("c3late").unwrap();
        let tx = verifying(&w, &n);
        let stage2 = tx.stage2_deadline.unwrap();
        let commit_dl = tx.commit_deadline.unwrap();

        // Missing ready ack: never self-commits, and never even samples the clock.
        let mut clock_calls = 0usize;
        assert_eq!(
            self_commit_with(&w.paths, &n, || {
                clock_calls += 1;
                stage2
            }),
            CandidateDecision::Exit
        );
        assert_eq!(clock_calls, 0);
        assert!(tx.staging_path.exists());
        assert_eq!(
            us::read_tx_state(&w.paths, &n),
            Some(TxControlState::Active)
        );

        // Late ready ack (ready_at > stage2_deadline), even while now <= commit_deadline: refused.
        us::write_ack(
            &w.paths,
            &AckRecord {
                nonce: n.clone(),
                pid: std::process::id(),
                ready_at: Some(stage2 + 1),
            },
        )
        .unwrap();
        assert_eq!(
            self_commit_with(&w.paths, &n, || commit_dl),
            CandidateDecision::Exit
        );
        assert!(tx.staging_path.exists());
        assert_eq!(
            us::read_tx_state(&w.paths, &n),
            Some(TxControlState::Active)
        );

        // Timely ready ack, but the clock crosses commit_deadline by the time the files have been
        // read and validated: refused because the clock is sampled after the reads.
        us::write_ack(
            &w.paths,
            &AckRecord {
                nonce: n.clone(),
                pid: std::process::id(),
                ready_at: Some(stage2),
            },
        )
        .unwrap();
        let mut now_at_decision = commit_dl;
        assert_eq!(
            self_commit_with(&w.paths, &n, || {
                now_at_decision += 1;
                now_at_decision
            }),
            CandidateDecision::Exit
        );
        assert_eq!(now_at_decision, commit_dl + 1);
        assert!(tx.staging_path.exists());

        // Timely ready ack and within commit_deadline: commits and removes staging + journal.
        assert_eq!(
            self_commit_with(&w.paths, &n, || commit_dl),
            CandidateDecision::Committed
        );
        assert!(!tx.staging_path.exists());
        assert!(us::read_tx(&w.paths).unwrap().is_none());

        // Helper died after writing phase=committed to the journal but before publishing
        // tx-<nonce>-state=committed: self_commit publishes the terminal state unconditionally
        // (even past commit_deadline and without checking verifying deadlines).
        let n2 = Nonce::parse("c3durable").unwrap();
        let mut tx2 = verifying(&w, &n2);
        tx2.phase = TxPhase::Committed;
        us::write_tx(&w.paths, &tx2).unwrap();
        let mut durable_clock_calls = 0usize;
        assert_eq!(
            self_commit_with(&w.paths, &n2, || {
                durable_clock_calls += 1;
                u64::MAX
            }),
            CandidateDecision::Committed
        );
        assert_eq!(durable_clock_calls, 0);
        assert_eq!(
            us::read_tx_state(&w.paths, &n2),
            Some(TxControlState::Committed)
        );
        assert!(!tx2.staging_path.exists());
        assert!(us::read_tx(&w.paths).unwrap().is_none());
    }

    #[test]
    fn stale_helper_prep_temps_are_reclaimed_only_when_unlocked() {
        let w = World::new();
        fs::create_dir_all(&w.paths.dir).unwrap();
        let tmp = w.paths.dir.join(".update-helper-crashed1.tmp");
        fs::write(&tmp, "partial").unwrap();
        // While a helper copy holds .helper-prep.lock, the active temp must survive cleanup.
        let prep_lock = open_helper_prep_lock(&w.paths).unwrap();
        prep_lock.lock().unwrap();
        cleanup_stale_files(&w.paths);
        assert!(tmp.exists(), "active prep temp survived while locked");
        drop(prep_lock);
        // Once released (or after a crash), startup cleanup reclaims the leftover temp.
        cleanup_stale_files(&w.paths);
        assert!(!tmp.exists(), "abandoned prep temp was reclaimed");
    }

    #[test]
    fn a_candidate_waits_while_the_helper_lives_and_obeys_it() {
        let w = World::new();
        let n = Nonce::parse("c4").unwrap();
        verifying(&w, &n);
        let held = UpdatingLock::try_acquire(&w.paths).unwrap().unwrap();
        let (paths, n2) = (w.paths.clone(), n.clone());
        let h = std::thread::spawn(move || {
            let stop = AtomicBool::new(false);
            candidate_wait_for_commit_or_takeover(&paths, &n2, Duration::from_millis(5), &stop)
        });
        std::thread::sleep(Duration::from_millis(50));
        assert!(!h.is_finished());
        us::write_tx_state(&w.paths, &n, TxControlState::Revoked).unwrap();
        assert_eq!(h.join().unwrap(), CandidateDecision::Exit);
        drop(held);

        // Committed by recovery while the journal still exists: staging cleaned first.
        let n = Nonce::parse("c5").unwrap();
        let tx = verifying(&w, &n);
        us::write_tx_state(&w.paths, &n, TxControlState::Committed).unwrap();
        assert_eq!(self_commit(&w.paths, &n), CandidateDecision::Committed);
        assert!(!tx.staging_path.exists());
        assert!(us::read_tx(&w.paths).unwrap().is_none());

        // Missing state: exit with no side effects.
        let n = Nonce::parse("c6").unwrap();
        let tx = verifying(&w, &n);
        fs::remove_file(w.paths.tx_state(&n)).unwrap();
        let stop = AtomicBool::new(false);
        assert_eq!(
            candidate_wait_for_commit_or_takeover(&w.paths, &n, Duration::from_millis(5), &stop),
            CandidateDecision::Exit
        );
        assert!(tx.staging_path.exists());
        assert!(us::read_tx(&w.paths).unwrap().is_some());
    }

    #[test]
    fn wait_update_release_waits_for_the_lock() {
        let w = World::new();
        assert!(wait_for_update_release(&w.paths, Duration::ZERO));
        let held = UpdatingLock::try_acquire(&w.paths).unwrap().unwrap();
        assert!(!wait_for_update_release(
            &w.paths,
            Duration::from_millis(60)
        ));
        let h = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            drop(held);
        });
        assert!(wait_for_update_release(&w.paths, Duration::from_secs(5)));
        h.join().unwrap();
    }

    #[test]
    fn stale_files_are_cleaned_but_the_current_transaction_is_kept() {
        let w = World::new();
        fs::create_dir_all(&w.paths.dir).unwrap();
        let cur = Nonce::parse("cur").unwrap();
        verifying(&w, &cur);
        fs::write(w.paths.update_helper(&cur), "h").unwrap();
        let old = Nonce::parse("old").unwrap();
        fs::write(w.paths.update_helper(&old), "h").unwrap();
        us::write_tx_state(&w.paths, &old, TxControlState::Committed).unwrap();
        // Fresh unreferenced helper and state files are kept so a starting helper or reading
        // candidate is never raced.
        cleanup_stale_files(&w.paths);
        assert!(w.paths.update_helper(&cur).exists());
        assert!(w.paths.tx_state(&cur).exists());
        assert!(w.paths.update_helper(&old).exists());
        assert!(w.paths.tx_state(&old).exists());
        // Once past the age threshold, unreferenced helper and state files are removed while the
        // current transaction's files stay.
        std::thread::sleep(Duration::from_millis(10));
        cleanup_stale_files_with(&w.paths, Duration::ZERO);
        assert!(w.paths.update_helper(&cur).exists());
        assert!(w.paths.tx_state(&cur).exists());
        assert!(!w.paths.update_helper(&old).exists());
        assert!(!w.paths.tx_state(&old).exists());
        // Not while another process holds updating.lock.
        fs::write(w.paths.update_helper(&old), "h").unwrap();
        std::thread::sleep(Duration::from_millis(10));
        let held = UpdatingLock::try_acquire(&w.paths).unwrap().unwrap();
        cleanup_stale_files_with(&w.paths, Duration::ZERO);
        assert!(w.paths.update_helper(&old).exists());
        drop(held);
    }
}
