//! Recovery of an app update whose helper died, shared by `oto-app` (at startup and as
//! `update-helper-<nonce> --recover-update`) and `oto recover-update`, so every entry point runs
//! exactly the same state machine over the files in [`crate::update_state`].
//!
//! See docs/plans/0003-app-updates.md, Decision 6 ("Complete dead-helper, hung-candidate &
//! independent recovery state machine"). Recovery only ever runs while holding `updating.lock`,
//! so it never races a live helper (which holds that lock for the whole transaction), and only
//! touches the installed bundle while also holding `engine.lock`, so it never swaps a bundle
//! under a running engine.
//!
//! Side effects outside the state directory (bundle swap, LaunchServices registration, process
//! inspection, relaunch) go through [`RecoveryOps`], so the logic is tested without touching
//! `/Applications` or real processes. [`SystemOps`] is the real implementation.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::update_state::{
    self as us, Nonce, TxControlState, TxPhase, UpdatePaths, UpdateTx, UpdatingLock,
};

/// Seconds after the swap by which the candidate must have written `ack-<nonce>-started`.
pub const STAGE1_SECS: u64 = 15;
/// Seconds after the swap by which the candidate must have written `ack-<nonce>-ready`.
pub const STAGE2_SECS: u64 = 35;
/// Seconds after the swap by which a ready candidate must have been committed.
pub const COMMIT_SECS: u64 = 50;
/// How long a candidate gets to exit after `SIGTERM` before `SIGKILL`.
pub const TERMINATE_GRACE: Duration = Duration::from_secs(3);
/// Bounded retry for `engine.lock` once the holder is known to be gone.
pub const ENGINE_LOCK_WAIT: Duration = Duration::from_secs(5);
/// The prefix of the staged sibling bundle (`<target dir>/.Oto-update-<nonce>`); never `.app`,
/// so LaunchServices never indexes a second copy.
pub const STAGING_PREFIX: &str = ".Oto-update-";
/// The argument a restored app is launched with after a rollback: it waits for `updating.lock`
/// to clear before starting its engine.
pub const WAIT_UPDATE_RELEASE_ARG: &str = "--wait-update-release";

/// Everything recovery does outside the state directory.
pub trait RecoveryOps {
    /// Atomically exchanges two directory entries (`renamex_np(a, b, RENAME_SWAP)`).
    fn swap(&self, a: &Path, b: &Path) -> io::Result<()>;
    /// Tells LaunchServices the bundle at `app` changed (`LSRegisterURL(app, true)`).
    fn register(&self, app: &Path);
    /// `CFBundleShortVersionString` of the bundle at `app`, `None` if missing or unreadable.
    fn bundle_version(&self, app: &Path) -> Option<String>;
    /// The executable path of a running process (`proc_pidpath`).
    fn pid_exe(&self, pid: u32) -> Option<PathBuf>;
    /// Whether the process exists.
    fn pid_alive(&self, pid: u32) -> bool;
    /// `SIGTERM`, up to `grace`, then `SIGKILL`; true once the process is gone.
    fn terminate(&self, pid: u32, grace: Duration) -> bool;
    /// Launches a new instance of the app at `target` with [`WAIT_UPDATE_RELEASE_ARG`]. `Ok`
    /// once the launch has started.
    fn launch_restored(&self, target: &Path) -> Result<(), String>;
    /// Unix seconds.
    fn now(&self) -> u64 {
        us::now_unix()
    }
    fn log(&self, msg: &str) {
        eprintln!("oto-update: {msg}");
    }
}

/// Who is recovering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecoveryMode {
    /// `oto-app` starting normally (not as an update candidate): its canonical executable path
    /// and version, so recovery can tell whether this process *is* the restored app.
    Startup { exe: PathBuf, version: String },
    /// `update-helper-<nonce> --recover-update`, `oto recover-update`, or a helper clearing a
    /// stale journal before starting a new transaction.
    Manual,
}

/// What recovery did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecoveryOutcome {
    /// No journal.
    NoTransaction,
    /// `updating.lock` is held: a live helper (or a candidate finishing its own commit) owns the
    /// transaction.
    HelperRunning,
    /// Left alone for now (a candidate still starting, `engine.lock` held by some other
    /// process, an unexpected bundle state): the journal is kept.
    Deferred(String),
    /// The transaction is resolved; carry on.
    Resolved,
    /// The old app was put back and launched. A starting `oto-app` that is not that restored
    /// app exits.
    Relaunched,
    /// The old app was put back, but launching it failed; the journal keeps
    /// `relaunch_pending` so a later recovery launches it.
    RelaunchFailed(String),
}

impl RecoveryOutcome {
    /// Whether a normally starting `oto-app` should exit instead of starting its engine.
    pub fn startup_should_exit(&self) -> bool {
        matches!(self, Self::Relaunched | Self::RelaunchFailed(_))
    }
}

// ---------------------------------------------------------------------------------------------
// engine.lock from outside the engine

/// An exclusive `flock` on `engine.lock`, as the update helper and recovery take it to keep any
/// engine from starting while they swap bundles. Unlike the engine's own lock it writes no PID
/// (the file keeps whatever the last engine left, which the engine clears on exit).
#[derive(Debug)]
pub struct EngineLockGuard {
    _file: File,
}

impl EngineLockGuard {
    /// `Ok(None)` if another process holds it.
    pub fn try_acquire(paths: &UpdatePaths) -> io::Result<Option<Self>> {
        fs::create_dir_all(&paths.dir)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(paths.engine_lock())?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { _file: file })),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(e)) => Err(e),
        }
    }

    /// Retries [`Self::try_acquire`] until `timeout`.
    pub fn acquire_timeout(paths: &UpdatePaths, timeout: Duration) -> io::Result<Option<Self>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(l) = Self::try_acquire(paths)? {
                return Ok(Some(l));
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Whether some process holds `engine.lock` right now (shared probe, never blocks).
    pub fn is_held(paths: &UpdatePaths) -> bool {
        let Ok(file) = OpenOptions::new().read(true).open(paths.engine_lock()) else {
            return false;
        };
        match file.try_lock_shared() {
            Ok(()) => false,
            Err(TryLockError::WouldBlock) => true,
            Err(TryLockError::Error(_)) => false,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Small file helpers (all best-effort where the plan says so)

/// Removes a file, directory tree or symlink; missing is fine.
pub fn remove_path(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
        Ok(m) if m.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
    }
}

fn best_effort_remove(ops: &dyn RecoveryOps, path: &Path) {
    if let Err(e) = remove_path(path) {
        ops.log(&format!("couldn't remove {}: {e}", path.display()));
    }
}

/// Deletes `tx-<nonce>-state`, `ack-<nonce>-started` and `ack-<nonce>-ready`.
pub fn remove_state_and_acks(paths: &UpdatePaths, nonce: &Nonce) {
    for p in [
        paths.tx_state(nonce),
        paths.ack_started(nonce),
        paths.ack_ready(nonce),
    ] {
        let _ = remove_path(&p);
    }
}

/// The journal's paths must be exactly what a helper writes: an absolute `.app` target, the
/// staging sibling for this nonce, and the helper copy in the state directory. Anything else is
/// treated as a corrupt journal, so a tampered journal can never point recovery at other files.
pub fn validate_tx_paths(paths: &UpdatePaths, tx: &UpdateTx) -> Result<(), String> {
    let t = &tx.target_path;
    let is_app = t.extension().is_some_and(|e| e == "app");
    if !t.is_absolute() || !is_app || t.parent().is_none() {
        return Err(format!(
            "target {} is not an absolute .app path",
            t.display()
        ));
    }
    let expected_staging = t
        .parent()
        .map(|d| d.join(format!("{STAGING_PREFIX}{}", tx.nonce)));
    if expected_staging.as_deref() != Some(tx.staging_path.as_path()) {
        return Err(format!(
            "staging {} is not next to the target",
            tx.staging_path.display()
        ));
    }
    if tx.helper_path != paths.update_helper(&tx.nonce) {
        return Err(format!(
            "helper {} is not in the state directory",
            tx.helper_path.display()
        ));
    }
    Ok(())
}

fn exe_dir(target: &Path) -> PathBuf {
    target.join("Contents/MacOS")
}

/// `pid` is alive and runs an executable inside `target`.
fn pid_runs_from(ops: &dyn RecoveryOps, pid: u32, target: &Path) -> bool {
    pid != 0
        && pid != std::process::id()
        && ops
            .pid_exe(pid)
            .is_some_and(|p| p.starts_with(exe_dir(target)))
}

fn is_restored_target(mode: &RecoveryMode, tx: &UpdateTx) -> bool {
    match mode {
        RecoveryMode::Startup { exe, version } => {
            exe.starts_with(&tx.target_path) && *version == tx.old_version
        }
        RecoveryMode::Manual => false,
    }
}

/// The deadlines persisted in a `verifying` journal (derived from `started_at` if a writer left
/// them out).
fn deadlines(tx: &UpdateTx) -> (u64, u64) {
    (
        tx.stage2_deadline.unwrap_or(tx.started_at + STAGE2_SECS),
        tx.commit_deadline.unwrap_or(tx.started_at + COMMIT_SECS),
    )
}

// ---------------------------------------------------------------------------------------------
// Entry points

/// Takes `updating.lock` (without waiting: a held lock means a live helper owns the update) and
/// runs [`recover_locked`].
pub fn recover(paths: &UpdatePaths, ops: &dyn RecoveryOps, mode: &RecoveryMode) -> RecoveryOutcome {
    if !paths.tx_journal().exists() {
        return RecoveryOutcome::NoTransaction;
    }
    let lock = match UpdatingLock::try_acquire(paths) {
        Ok(Some(l)) => l,
        Ok(None) => return RecoveryOutcome::HelperRunning,
        Err(e) => return RecoveryOutcome::Deferred(format!("updating.lock: {e}")),
    };
    recover_locked(paths, &lock, ops, mode)
}

/// Dead-helper recovery, dispatched on the journal's phase. The caller holds `updating.lock`
/// (the `_lock` argument proves it) and keeps holding it until this returns.
pub fn recover_locked(
    paths: &UpdatePaths,
    _lock: &UpdatingLock,
    ops: &dyn RecoveryOps,
    mode: &RecoveryMode,
) -> RecoveryOutcome {
    let tx = match us::read_tx(paths) {
        Ok(None) => return RecoveryOutcome::NoTransaction,
        Ok(Some(tx)) => tx,
        Err(e) => return quarantine_journal(paths, ops, &e.to_string()),
    };
    if let Err(e) = validate_tx_paths(paths, &tx) {
        return quarantine_journal(paths, ops, &e);
    }
    ops.log(&format!(
        "recovering update {} in phase {:?} ({} -> {})",
        tx.nonce, tx.phase, tx.old_version, tx.new_version
    ));
    match tx.phase {
        TxPhase::Prepared => {
            // No swap happened; the old app may legitimately still be running, so engine.lock is
            // not needed.
            best_effort_remove(ops, &tx.staging_path);
            remove_state_and_acks(paths, &tx.nonce);
            let _ = us::remove_tx(paths);
            let _ = remove_path(&tx.helper_path);
            RecoveryOutcome::Resolved
        }
        TxPhase::Committed => {
            // The helper may have died between the journal's `committed` and publishing it to
            // the candidate: publish it now. Never delete the candidate's state or acks here.
            if us::read_tx_state(paths, &tx.nonce) == Some(TxControlState::Active)
                && let Err(e) = us::write_tx_state(paths, &tx.nonce, TxControlState::Committed)
            {
                return RecoveryOutcome::Deferred(format!("couldn't publish the commit: {e}"));
            }
            best_effort_remove(ops, &tx.staging_path);
            if let Err(e) = us::remove_tx(paths) {
                return RecoveryOutcome::Deferred(format!("couldn't remove the journal: {e}"));
            }
            let _ = remove_path(&tx.helper_path);
            RecoveryOutcome::Resolved
        }
        TxPhase::RolledBack => recover_rolled_back(paths, ops, mode, tx),
        TxPhase::Swapping | TxPhase::Verifying | TxPhase::RollingBack => {
            recover_unresolved(paths, ops, mode, tx)
        }
    }
}

/// An unreadable or tampered journal: keep it aside for inspection, revoke every candidate that
/// might be waiting for a commit (they exit with no side effects; whatever bundle is installed
/// starts normally next time), and carry on.
fn quarantine_journal(paths: &UpdatePaths, ops: &dyn RecoveryOps, why: &str) -> RecoveryOutcome {
    ops.log(&format!(
        "update journal is unusable ({why}); setting it aside"
    ));
    let aside = paths
        .dir
        .join(format!("update-tx.corrupt-{}.json", ops.now()));
    if let Err(e) = fs::rename(paths.tx_journal(), &aside) {
        return RecoveryOutcome::Deferred(format!("couldn't set the journal aside: {e}"));
    }
    for nonce in state_nonces(paths) {
        if us::read_tx_state(paths, &nonce) == Some(TxControlState::Active) {
            let _ = us::write_tx_state(paths, &nonce, TxControlState::Revoked);
        }
    }
    RecoveryOutcome::Resolved
}

/// Nonces with a `tx-<nonce>-state` file.
pub fn state_nonces(paths: &UpdatePaths) -> Vec<Nonce> {
    let Ok(rd) = fs::read_dir(&paths.dir) else {
        return Vec::new();
    };
    rd.filter_map(|e| {
        let name = e.ok()?.file_name().into_string().ok()?;
        let n = name.strip_prefix("tx-")?.strip_suffix("-state")?;
        Nonce::parse(n).ok()
    })
    .collect()
}

fn recover_rolled_back(
    paths: &UpdatePaths,
    ops: &dyn RecoveryOps,
    mode: &RecoveryMode,
    tx: UpdateTx,
) -> RecoveryOutcome {
    best_effort_remove(ops, &tx.staging_path);
    remove_state_and_acks(paths, &tx.nonce);
    let finish = || {
        let _ = us::remove_tx(paths);
        let _ = remove_path(&tx.helper_path);
        RecoveryOutcome::Resolved
    };
    if is_restored_target(mode, &tx) || !tx.relaunch_pending {
        return finish();
    }
    match EngineLockGuard::try_acquire(paths) {
        Ok(None) => {
            // Someone runs an engine: only if it is the restored app is the relaunch done.
            let restored = us::read_lock_holder_pid(paths)
                .is_some_and(|pid| pid_runs_from(ops, pid, &tx.target_path));
            if restored {
                finish()
            } else {
                RecoveryOutcome::Deferred(
                    "engine.lock is held by another process; the restored app still has to be \
                     launched"
                        .into(),
                )
            }
        }
        Ok(Some(engine)) => {
            drop(engine);
            relaunch(paths, ops, &tx)
        }
        Err(e) => RecoveryOutcome::Deferred(format!("engine.lock: {e}")),
    }
}

/// Launches the restored app and, once the launch started, drops the journal (keeping the
/// helper copy for the restored app's startup cleanup).
fn relaunch(paths: &UpdatePaths, ops: &dyn RecoveryOps, tx: &UpdateTx) -> RecoveryOutcome {
    match ops.launch_restored(&tx.target_path) {
        Ok(()) => {
            let _ = us::remove_tx(paths);
            RecoveryOutcome::Relaunched
        }
        Err(e) => {
            ops.log(&format!(
                "couldn't relaunch {}: {e}",
                tx.target_path.display()
            ));
            RecoveryOutcome::RelaunchFailed(e)
        }
    }
}

fn recover_unresolved(
    paths: &UpdatePaths,
    ops: &dyn RecoveryOps,
    mode: &RecoveryMode,
    tx: UpdateTx,
) -> RecoveryOutcome {
    let engine = match EngineLockGuard::try_acquire(paths) {
        Ok(Some(g)) => g,
        Ok(None) => match take_over_from_candidate(paths, ops, &tx) {
            Ok(g) => g,
            Err(outcome) => return outcome,
        },
        Err(e) => return RecoveryOutcome::Deferred(format!("engine.lock: {e}")),
    };
    // Both locks held from here.
    let tv = ops.bundle_version(&tx.target_path);
    let sv = ops.bundle_version(&tx.staging_path);
    let old = Some(tx.old_version.as_str());
    let new = Some(tx.new_version.as_str());
    match tx.phase {
        TxPhase::Swapping | TxPhase::Verifying => {
            if tv.as_deref() == old {
                // The swap never happened.
                best_effort_remove(ops, &tx.staging_path);
                remove_state_and_acks(paths, &tx.nonce);
                let _ = us::remove_tx(paths);
                let _ = remove_path(&tx.helper_path);
                return RecoveryOutcome::Resolved;
            }
            if tv.as_deref() == new && sv.as_deref() == old {
                if us::read_tx_state(paths, &tx.nonce) == Some(TxControlState::Committed) {
                    best_effort_remove(ops, &tx.staging_path);
                    remove_state_and_acks(paths, &tx.nonce);
                    let _ = us::remove_tx(paths);
                    let _ = remove_path(&tx.helper_path);
                    return RecoveryOutcome::Resolved;
                }
                return roll_back(paths, ops, mode, tx, engine);
            }
            if tv.as_deref() == new && !tx.staging_path.exists() {
                // Nothing to roll back to: keep the (verified) new build and stop any
                // candidate from waiting on a commit that will never come.
                ops.log("the old app is gone; keeping the new version");
                if us::read_tx_state(paths, &tx.nonce) == Some(TxControlState::Active) {
                    let _ = us::write_tx_state(paths, &tx.nonce, TxControlState::Revoked);
                }
                remove_state_and_acks(paths, &tx.nonce);
                let _ = us::remove_tx(paths);
                let _ = remove_path(&tx.helper_path);
                return RecoveryOutcome::Resolved;
            }
            RecoveryOutcome::Deferred(format!(
                "unexpected bundle versions (target {tv:?}, staging {sv:?})"
            ))
        }
        TxPhase::RollingBack => {
            if tv.as_deref() == new && sv.as_deref() == old {
                if let Err(e) = ops.swap(&tx.staging_path, &tx.target_path) {
                    return RecoveryOutcome::Deferred(format!("couldn't swap back: {e}"));
                }
                ops.register(&tx.target_path);
            } else if tv.as_deref() != old {
                return RecoveryOutcome::Deferred(format!(
                    "unexpected bundle versions while rolling back (target {tv:?}, staging {sv:?})"
                ));
            }
            finish_rollback(paths, ops, mode, tx, engine)
        }
        _ => RecoveryOutcome::Deferred("not an unresolved phase".into()),
    }
}

/// `engine.lock` is held. Only a verified candidate of this transaction in `verifying` may be
/// overridden, and only once it missed its persisted deadline. `Ok` carries `engine.lock`, taken
/// after the candidate was stopped.
fn take_over_from_candidate(
    paths: &UpdatePaths,
    ops: &dyn RecoveryOps,
    tx: &UpdateTx,
) -> Result<EngineLockGuard, RecoveryOutcome> {
    let holder = us::read_lock_holder_pid(paths);
    let cand = tx
        .candidate_pid
        .or_else(|| us::read_ack(paths, &tx.nonce, false).map(|a| a.pid));
    let verified = tx.phase == TxPhase::Verifying
        && holder.is_some()
        && holder == cand
        && holder.is_some_and(|pid| pid_runs_from(ops, pid, &tx.target_path));
    let Some(pid) = holder.filter(|_| verified) else {
        return Err(RecoveryOutcome::Deferred(
            "engine.lock is held by a process that is not this update's candidate".into(),
        ));
    };
    if us::read_tx_state(paths, &tx.nonce) == Some(TxControlState::Committed) {
        // Durable already: leave the live candidate alone, finish the helper's cleanup.
        best_effort_remove(ops, &tx.staging_path);
        let _ = us::remove_tx(paths);
        return Err(RecoveryOutcome::Resolved);
    }
    let now = ops.now();
    let (stage2, commit) = deadlines(tx);
    let ready = us::read_ack(paths, &tx.nonce, true).is_some_and(|a| a.pid == pid);
    let hung = (ready && now > commit) || (!ready && now > stage2);
    if !hung {
        return Err(RecoveryOutcome::Deferred(
            "the new version is still starting".into(),
        ));
    }
    ops.log(&format!(
        "candidate {pid} missed its deadline ({}); rolling back",
        if ready { "commit" } else { "ready" }
    ));
    if let Err(e) = us::write_tx_state(paths, &tx.nonce, TxControlState::Revoked) {
        return Err(RecoveryOutcome::Deferred(format!("couldn't revoke: {e}")));
    }
    ops.terminate(pid, TERMINATE_GRACE);
    match EngineLockGuard::acquire_timeout(paths, ENGINE_LOCK_WAIT) {
        Ok(Some(g)) => Ok(g),
        Ok(None) => Err(RecoveryOutcome::Deferred(
            "the candidate did not release engine.lock".into(),
        )),
        Err(e) => Err(RecoveryOutcome::Deferred(format!("engine.lock: {e}"))),
    }
}

/// Rolls back a swapped transaction: revokes the candidate, stops `tx.candidate_pid` if it still
/// runs from the target, records `rolling_back`, swaps the old build back (`target` new,
/// `staging` old), records `rolled_back` with `relaunch_pending`, cleans up, releases
/// `engine.lock` and launches the restored app (unless `mode` says this process is it). The
/// caller holds `updating.lock` and passes `engine.lock`. Used by the live helper and recovery.
pub fn roll_back_locked(
    paths: &UpdatePaths,
    _lock: &UpdatingLock,
    ops: &dyn RecoveryOps,
    mode: &RecoveryMode,
    tx: UpdateTx,
    engine: EngineLockGuard,
) -> RecoveryOutcome {
    roll_back(paths, ops, mode, tx, engine)
}

/// Swaps the old build back (target is new, staging is old), under both locks.
fn roll_back(
    paths: &UpdatePaths,
    ops: &dyn RecoveryOps,
    mode: &RecoveryMode,
    mut tx: UpdateTx,
    engine: EngineLockGuard,
) -> RecoveryOutcome {
    if let Err(e) = us::write_tx_state(paths, &tx.nonce, TxControlState::Revoked) {
        return RecoveryOutcome::Deferred(format!("couldn't revoke: {e}"));
    }
    if let Some(pid) = tx.candidate_pid
        && ops.pid_alive(pid)
        && pid_runs_from(ops, pid, &tx.target_path)
    {
        ops.terminate(pid, TERMINATE_GRACE);
    }
    tx.phase = TxPhase::RollingBack;
    if let Err(e) = us::write_tx(paths, &tx) {
        return RecoveryOutcome::Deferred(format!("couldn't write the journal: {e}"));
    }
    if let Err(e) = ops.swap(&tx.staging_path, &tx.target_path) {
        return RecoveryOutcome::Deferred(format!("couldn't swap back: {e}"));
    }
    ops.register(&tx.target_path);
    finish_rollback(paths, ops, mode, tx, engine)
}

/// The old build is at the target: record `rolled_back`, clean up, release `engine.lock`, then
/// either carry on (this process is the restored app) or launch it.
fn finish_rollback(
    paths: &UpdatePaths,
    ops: &dyn RecoveryOps,
    mode: &RecoveryMode,
    mut tx: UpdateTx,
    engine: EngineLockGuard,
) -> RecoveryOutcome {
    tx.phase = TxPhase::RolledBack;
    tx.relaunch_pending = true;
    if let Err(e) = us::write_tx(paths, &tx) {
        return RecoveryOutcome::Deferred(format!("couldn't write the journal: {e}"));
    }
    best_effort_remove(ops, &tx.staging_path);
    remove_state_and_acks(paths, &tx.nonce);
    drop(engine);
    if is_restored_target(mode, &tx) {
        let _ = us::remove_tx(paths);
        let _ = remove_path(&tx.helper_path);
        return RecoveryOutcome::Resolved;
    }
    relaunch(paths, ops, &tx)
}

// ---------------------------------------------------------------------------------------------
// The real side effects

/// The real [`RecoveryOps`]: `renamex_np`, `LSRegisterURL`, `plutil`, `proc_pidpath`, signals
/// and `/usr/bin/open -n`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemOps;

impl RecoveryOps for SystemOps {
    fn swap(&self, a: &Path, b: &Path) -> io::Result<()> {
        sys::rename_swap(a, b)
    }
    fn register(&self, app: &Path) {
        sys::ls_register(app);
    }
    fn bundle_version(&self, app: &Path) -> Option<String> {
        plist_string(
            &app.join("Contents/Info.plist"),
            "CFBundleShortVersionString",
        )
    }
    fn pid_exe(&self, pid: u32) -> Option<PathBuf> {
        sys::pid_path(pid)
    }
    fn pid_alive(&self, pid: u32) -> bool {
        sys::pid_alive(pid)
    }
    fn terminate(&self, pid: u32, grace: Duration) -> bool {
        sys::terminate(pid, grace)
    }
    fn launch_restored(&self, target: &Path) -> Result<(), String> {
        let out = std::process::Command::new("/usr/bin/open")
            .arg("-n")
            .arg(target)
            .args(["--args", WAIT_UPDATE_RELEASE_ARG])
            .output()
            .map_err(|e| format!("couldn't run open: {e}"))?;
        if out.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&out.stderr).trim().to_owned())
        }
    }
}

/// One string value from a property list (any format), through `plutil`.
pub fn plist_string(plist: &Path, key: &str) -> Option<String> {
    let out = std::process::Command::new("/usr/bin/plutil")
        .args(["-extract", key, "raw", "-o", "-"])
        .arg(plist)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// Thin FFI for the update code: atomic exchange, LaunchServices, process inspection.
pub mod sys {
    use std::io;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    #[cfg(target_os = "macos")]
    mod ffi {
        use std::ffi::{c_char, c_int, c_uint, c_void};

        pub const RENAME_SWAP: c_uint = 0x0000_0002;
        pub const PROC_PIDPATHINFO_MAXSIZE: u32 = 4096;
        pub const SIGKILL: c_int = 9;
        pub const SIGTERM: c_int = 15;
        pub const EPERM: i32 = 1;

        unsafe extern "C" {
            pub fn renamex_np(from: *const c_char, to: *const c_char, flags: c_uint) -> c_int;
            pub fn proc_pidpath(pid: c_int, buffer: *mut c_void, size: u32) -> c_int;
            pub fn kill(pid: c_int, sig: c_int) -> c_int;
        }

        pub type CFTypeRef = *const c_void;
        pub type CFURLRef = *const c_void;

        #[link(name = "CoreFoundation", kind = "framework")]
        unsafe extern "C" {
            pub fn CFURLCreateFromFileSystemRepresentation(
                allocator: CFTypeRef,
                buffer: *const u8,
                len: isize,
                is_directory: u8,
            ) -> CFURLRef;
            pub fn CFRelease(cf: CFTypeRef);
        }

        #[link(name = "CoreServices", kind = "framework")]
        unsafe extern "C" {
            pub fn LSRegisterURL(url: CFURLRef, update: u8) -> i32;
        }
    }

    #[cfg(target_os = "macos")]
    fn cpath(p: &Path) -> io::Result<std::ffi::CString> {
        use std::os::unix::ffi::OsStrExt;
        std::ffi::CString::new(p.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))
    }

    /// `renamex_np(a, b, RENAME_SWAP)`: both must exist; they trade places in one step.
    pub fn rename_swap(a: &Path, b: &Path) -> io::Result<()> {
        #[cfg(target_os = "macos")]
        {
            let (ca, cb) = (cpath(a)?, cpath(b)?);
            // SAFETY: both are valid NUL-terminated paths.
            if unsafe { ffi::renamex_np(ca.as_ptr(), cb.as_ptr(), ffi::RENAME_SWAP) } == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (a, b);
            Err(io::Error::from(io::ErrorKind::Unsupported))
        }
    }

    /// `LSRegisterURL(app, true)`; returns the `OSStatus` (0 on success).
    pub fn ls_register(app: &Path) -> i32 {
        #[cfg(target_os = "macos")]
        {
            use std::os::unix::ffi::OsStrExt;
            let bytes = app.as_os_str().as_bytes();
            // SAFETY: plain CoreFoundation calls; the URL is released after use.
            unsafe {
                let url = ffi::CFURLCreateFromFileSystemRepresentation(
                    std::ptr::null(),
                    bytes.as_ptr(),
                    bytes.len() as isize,
                    1,
                );
                if url.is_null() {
                    return -1;
                }
                let st = ffi::LSRegisterURL(url, 1);
                ffi::CFRelease(url);
                st
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = app;
            -1
        }
    }

    /// The executable of a running process.
    pub fn pid_path(pid: u32) -> Option<PathBuf> {
        #[cfg(target_os = "macos")]
        {
            use std::os::unix::ffi::OsStrExt;
            let pid = i32::try_from(pid).ok()?;
            let mut buf = vec![0u8; ffi::PROC_PIDPATHINFO_MAXSIZE as usize];
            // SAFETY: the buffer is as large as we say.
            let n = unsafe {
                ffi::proc_pidpath(pid, buf.as_mut_ptr().cast(), ffi::PROC_PIDPATHINFO_MAXSIZE)
            };
            (n > 0)
                .then(|| PathBuf::from(std::ffi::OsStr::from_bytes(&buf[..n as usize]).to_owned()))
        }
        #[cfg(not(target_os = "macos"))]
        {
            std::fs::read_link(format!("/proc/{pid}/exe")).ok()
        }
    }

    /// Whether `pid` exists (ours or not).
    pub fn pid_alive(pid: u32) -> bool {
        #[cfg(target_os = "macos")]
        {
            let Ok(p) = i32::try_from(pid) else {
                return false;
            };
            if p <= 0 {
                return false;
            }
            // SAFETY: signal 0 only checks.
            let ok = unsafe { ffi::kill(p, 0) == 0 };
            ok || io::Error::last_os_error().raw_os_error() == Some(ffi::EPERM)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Path::new(&format!("/proc/{pid}")).exists()
        }
    }

    /// Sends `sig` (`SIGTERM` = 15, `SIGKILL` = 9).
    pub fn signal(pid: u32, sig: i32) -> bool {
        #[cfg(target_os = "macos")]
        {
            let Ok(p) = i32::try_from(pid) else {
                return false;
            };
            // SAFETY: plain kill(2) on a positive pid.
            p > 0 && unsafe { ffi::kill(p, sig) == 0 }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (pid, sig);
            false
        }
    }

    /// `SIGTERM`, wait up to `grace`, then `SIGKILL` and wait briefly. True once gone.
    pub fn terminate(pid: u32, grace: Duration) -> bool {
        #[cfg(target_os = "macos")]
        let (term, kill) = (ffi::SIGTERM, ffi::SIGKILL);
        #[cfg(not(target_os = "macos"))]
        let (term, kill) = (15, 9);
        if !pid_alive(pid) {
            return true;
        }
        signal(pid, term);
        if wait_gone(pid, grace) {
            return true;
        }
        signal(pid, kill);
        wait_gone(pid, Duration::from_secs(2))
    }

    /// Polls until `pid` is gone or `timeout` passes.
    pub fn wait_gone(pid: u32, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if !pid_alive(pid) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::state::tests::TempDir;
    use crate::update_state::AckRecord;
    use std::cell::{Cell, RefCell};

    /// Bundles are directories holding a `version` file; swaps are three renames; processes
    /// are a table.
    #[derive(Default)]
    pub(crate) struct FakeOps {
        pub now: Cell<u64>,
        pub procs: RefCell<Vec<(u32, PathBuf)>>,
        pub launched: RefCell<Vec<PathBuf>>,
        pub terminated: RefCell<Vec<u32>>,
        pub launch_fails: Cell<bool>,
        pub registered: RefCell<Vec<PathBuf>>,
        /// Releases this lock when a process is terminated (the candidate's engine.lock).
        pub on_terminate: RefCell<Option<File>>,
    }

    impl RecoveryOps for FakeOps {
        fn swap(&self, a: &Path, b: &Path) -> io::Result<()> {
            let tmp = a.with_extension("swaptmp");
            fs::rename(a, &tmp)?;
            fs::rename(b, a)?;
            fs::rename(&tmp, b)
        }
        fn register(&self, app: &Path) {
            self.registered.borrow_mut().push(app.into());
        }
        fn bundle_version(&self, app: &Path) -> Option<String> {
            fs::read_to_string(app.join("version")).ok()
        }
        fn pid_exe(&self, pid: u32) -> Option<PathBuf> {
            let procs = self.procs.borrow();
            procs
                .iter()
                .find(|(p, _)| *p == pid)
                .map(|(_, e)| e.clone())
        }
        fn pid_alive(&self, pid: u32) -> bool {
            self.procs.borrow().iter().any(|(p, _)| *p == pid)
        }
        fn terminate(&self, pid: u32, _: Duration) -> bool {
            self.procs.borrow_mut().retain(|(p, _)| *p != pid);
            self.terminated.borrow_mut().push(pid);
            self.on_terminate.borrow_mut().take();
            true
        }
        fn launch_restored(&self, target: &Path) -> Result<(), String> {
            if self.launch_fails.get() {
                return Err("no".into());
            }
            self.launched.borrow_mut().push(target.into());
            Ok(())
        }
        fn now(&self) -> u64 {
            self.now.get()
        }
        fn log(&self, _: &str) {}
    }

    pub(crate) struct World {
        pub _t: TempDir,
        pub paths: UpdatePaths,
        pub apps: PathBuf,
        pub nonce: Nonce,
        pub tx: UpdateTx,
    }

    pub(crate) fn bundle(path: &Path, version: &str) {
        fs::create_dir_all(path.join("Contents/MacOS")).unwrap();
        fs::write(path.join("version"), version).unwrap();
    }

    impl World {
        pub fn new(phase: TxPhase) -> Self {
            let t = TempDir::new();
            let paths = UpdatePaths::new(t.0.join("state"));
            let apps = t.0.join("Applications");
            fs::create_dir_all(&paths.dir).unwrap();
            fs::create_dir_all(&apps).unwrap();
            let nonce = Nonce::parse("abc").unwrap();
            let target = apps.join("Oto.app");
            let tx = UpdateTx {
                nonce: nonce.clone(),
                phase,
                started_at: 1000,
                stage1_deadline: Some(1015),
                stage2_deadline: Some(1035),
                commit_deadline: Some(1050),
                staging_path: apps.join(format!("{STAGING_PREFIX}abc")),
                target_path: target,
                old_version: "1.0.0".into(),
                new_version: "1.1.0".into(),
                helper_path: paths.update_helper(&nonce),
                candidate_pid: None,
                relaunch_pending: false,
            };
            fs::write(&tx.helper_path, "helper").unwrap();
            us::write_tx(&paths, &tx).unwrap();
            Self {
                _t: t,
                paths,
                apps,
                nonce,
                tx,
            }
        }
        /// Target and staging holding these versions (`None`: absent).
        pub fn bundles(&self, target: Option<&str>, staging: Option<&str>) {
            if let Some(v) = target {
                bundle(&self.tx.target_path, v);
            }
            if let Some(v) = staging {
                bundle(&self.tx.staging_path, v);
            }
        }
        pub fn write(&mut self) {
            us::write_tx(&self.paths, &self.tx).unwrap();
        }
        pub fn target_version(&self) -> Option<String> {
            fs::read_to_string(self.tx.target_path.join("version")).ok()
        }
        pub fn state(&self) -> Option<TxControlState> {
            us::read_tx_state(&self.paths, &self.nonce)
        }
        pub fn journal(&self) -> Option<UpdateTx> {
            us::read_tx(&self.paths).unwrap()
        }
        pub fn candidate_exe(&self) -> PathBuf {
            self.tx.target_path.join("Contents/MacOS/oto-app")
        }
        /// Holds engine.lock in the way a running engine does: lock plus PID in the file.
        pub fn hold_engine_lock(&self, pid: u32) -> File {
            let f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(self.paths.engine_lock())
                .unwrap();
            f.try_lock().unwrap();
            fs::write(self.paths.engine_lock(), format!("{pid}\n")).unwrap();
            f
        }
        pub fn ack(&self, pid: u32, ready: bool) {
            us::write_ack(
                &self.paths,
                &AckRecord {
                    nonce: self.nonce.clone(),
                    pid,
                    ready_at: ready.then_some(1),
                },
            )
            .unwrap();
        }
    }

    fn manual(w: &World, ops: &FakeOps) -> RecoveryOutcome {
        recover(&w.paths, ops, &RecoveryMode::Manual)
    }

    #[test]
    fn nothing_to_do_and_a_live_helper_is_left_alone() {
        let w = World::new(TxPhase::Verifying);
        let ops = FakeOps::default();
        let held = UpdatingLock::try_acquire(&w.paths).unwrap().unwrap();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::HelperRunning);
        drop(held);
        us::remove_tx(&w.paths).unwrap();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::NoTransaction);
    }

    #[test]
    fn prepared_cleans_up_without_engine_lock() {
        let w = World::new(TxPhase::Prepared);
        w.bundles(Some("1.0.0"), Some("1.1.0"));
        // The old app may still run and hold engine.lock: prepared needs no engine.lock.
        let _engine = w.hold_engine_lock(4242);
        let ops = FakeOps::default();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Resolved);
        assert!(w.journal().is_none());
        assert!(!w.tx.staging_path.exists());
        assert!(!w.tx.helper_path.exists());
        assert_eq!(w.target_version().as_deref(), Some("1.0.0"));
    }

    #[test]
    fn swapping_before_the_exchange_keeps_the_old_app() {
        let w = World::new(TxPhase::Swapping);
        w.bundles(Some("1.0.0"), Some("1.1.0"));
        let ops = FakeOps::default();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Resolved);
        assert_eq!(w.target_version().as_deref(), Some("1.0.0"));
        assert!(!w.tx.staging_path.exists());
        assert!(w.journal().is_none());
        assert!(ops.launched.borrow().is_empty());
    }

    #[test]
    fn swapping_after_the_exchange_rolls_back_and_relaunches() {
        let w = World::new(TxPhase::Swapping);
        w.bundles(Some("1.1.0"), Some("1.0.0"));
        let ops = FakeOps::default();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Relaunched);
        assert_eq!(w.target_version().as_deref(), Some("1.0.0"));
        assert!(!w.tx.staging_path.exists());
        assert_eq!(
            *ops.launched.borrow(),
            std::slice::from_ref(&w.tx.target_path)
        );
        assert_eq!(
            *ops.registered.borrow(),
            std::slice::from_ref(&w.tx.target_path)
        );
        assert!(w.journal().is_none());
        assert_eq!(w.state(), None);
        // The helper copy stays until the restored app cleans it up.
        assert!(w.tx.helper_path.exists());
    }

    #[test]
    fn a_failed_relaunch_keeps_the_obligation() {
        let w = World::new(TxPhase::Swapping);
        w.bundles(Some("1.1.0"), Some("1.0.0"));
        let ops = FakeOps::default();
        ops.launch_fails.set(true);
        assert!(matches!(
            manual(&w, &ops),
            RecoveryOutcome::RelaunchFailed(_)
        ));
        let j = w.journal().unwrap();
        assert_eq!(j.phase, TxPhase::RolledBack);
        assert!(j.relaunch_pending);
        // Next time it launches.
        ops.launch_fails.set(false);
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Relaunched);
        assert!(w.journal().is_none());
    }

    #[test]
    fn verifying_committed_but_dead_candidate_finalizes() {
        let w = World::new(TxPhase::Verifying);
        w.bundles(Some("1.1.0"), Some("1.0.0"));
        us::write_tx_state(&w.paths, &w.nonce, TxControlState::Committed).unwrap();
        let ops = FakeOps::default();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Resolved);
        assert_eq!(w.target_version().as_deref(), Some("1.1.0"));
        assert!(!w.tx.staging_path.exists());
        assert!(w.journal().is_none());
        assert!(!w.tx.helper_path.exists());
    }

    #[test]
    fn verifying_with_a_starting_candidate_is_deferred() {
        let mut w = World::new(TxPhase::Verifying);
        w.bundles(Some("1.1.0"), Some("1.0.0"));
        w.tx.candidate_pid = Some(77);
        w.write();
        us::write_tx_state(&w.paths, &w.nonce, TxControlState::Active).unwrap();
        let _engine = w.hold_engine_lock(77);
        let ops = FakeOps::default();
        ops.procs.borrow_mut().push((77, w.candidate_exe()));
        ops.now.set(1020);
        assert!(matches!(manual(&w, &ops), RecoveryOutcome::Deferred(_)));
        // Ready and within the commit deadline: still left to self-commit.
        w.ack(77, false);
        w.ack(77, true);
        ops.now.set(1045);
        assert!(matches!(manual(&w, &ops), RecoveryOutcome::Deferred(_)));
        assert_eq!(w.state(), Some(TxControlState::Active));
        assert_eq!(w.target_version().as_deref(), Some("1.1.0"));
    }

    #[test]
    fn a_candidate_hung_before_ready_is_rolled_back() {
        let mut w = World::new(TxPhase::Verifying);
        w.bundles(Some("1.1.0"), Some("1.0.0"));
        w.tx.candidate_pid = Some(77);
        w.write();
        us::write_tx_state(&w.paths, &w.nonce, TxControlState::Active).unwrap();
        w.ack(77, false);
        let engine = w.hold_engine_lock(77);
        let ops = FakeOps::default();
        ops.procs.borrow_mut().push((77, w.candidate_exe()));
        *ops.on_terminate.borrow_mut() = Some(engine);
        ops.now.set(1036);
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Relaunched);
        assert_eq!(*ops.terminated.borrow(), [77]);
        assert_eq!(w.target_version().as_deref(), Some("1.0.0"));
        assert!(w.journal().is_none());
    }

    #[test]
    fn a_candidate_hung_after_ready_is_rolled_back_after_the_commit_deadline() {
        let w = World::new(TxPhase::Verifying);
        w.bundles(Some("1.1.0"), Some("1.0.0"));
        // No candidate_pid recorded (helper died first): the started ack identifies it.
        us::write_tx_state(&w.paths, &w.nonce, TxControlState::Active).unwrap();
        w.ack(78, false);
        w.ack(78, true);
        let engine = w.hold_engine_lock(78);
        let ops = FakeOps::default();
        ops.procs.borrow_mut().push((78, w.candidate_exe()));
        *ops.on_terminate.borrow_mut() = Some(engine);
        ops.now.set(1051);
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Relaunched);
        assert_eq!(*ops.terminated.borrow(), [78]);
        assert_eq!(w.target_version().as_deref(), Some("1.0.0"));
        assert!(w.journal().is_none());
    }

    #[test]
    fn a_committed_live_candidate_is_left_untouched() {
        let mut w = World::new(TxPhase::Verifying);
        w.bundles(Some("1.1.0"), Some("1.0.0"));
        w.tx.candidate_pid = Some(77);
        w.write();
        us::write_tx_state(&w.paths, &w.nonce, TxControlState::Committed).unwrap();
        let _engine = w.hold_engine_lock(77);
        let ops = FakeOps::default();
        ops.procs.borrow_mut().push((77, w.candidate_exe()));
        ops.now.set(5000);
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Resolved);
        assert!(ops.terminated.borrow().is_empty());
        assert!(!w.tx.staging_path.exists());
        assert!(w.journal().is_none());
        // The candidate still reads its commit.
        assert_eq!(w.state(), Some(TxControlState::Committed));
    }

    #[test]
    fn an_unverified_lock_holder_blocks_recovery() {
        let mut w = World::new(TxPhase::Verifying);
        w.bundles(Some("1.1.0"), Some("1.0.0"));
        w.tx.candidate_pid = Some(77);
        w.write();
        // Some other engine (e.g. `oto run` from elsewhere) holds engine.lock.
        let _engine = w.hold_engine_lock(90);
        let ops = FakeOps::default();
        ops.procs
            .borrow_mut()
            .push((90, PathBuf::from("/usr/local/bin/oto")));
        ops.now.set(9999);
        assert!(matches!(manual(&w, &ops), RecoveryOutcome::Deferred(_)));
        assert!(w.journal().is_some());
        // Same PID as recorded but running from elsewhere: still not verified.
        let mut w2 = World::new(TxPhase::Swapping);
        w2.tx.candidate_pid = Some(90);
        w2.write();
        let _e = w2.hold_engine_lock(90);
        assert!(matches!(manual(&w2, &ops), RecoveryOutcome::Deferred(_)));
    }

    #[test]
    fn rolling_back_before_and_after_the_exchange() {
        // Before: target still new.
        let w = World::new(TxPhase::RollingBack);
        w.bundles(Some("1.1.0"), Some("1.0.0"));
        let ops = FakeOps::default();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Relaunched);
        assert_eq!(w.target_version().as_deref(), Some("1.0.0"));
        assert!(w.journal().is_none());
        // After: target already old, staging holds the failed build.
        let w = World::new(TxPhase::RollingBack);
        w.bundles(Some("1.0.0"), Some("1.1.0"));
        let ops = FakeOps::default();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Relaunched);
        assert_eq!(w.target_version().as_deref(), Some("1.0.0"));
        assert!(!w.tx.staging_path.exists());
        assert_eq!(
            *ops.launched.borrow(),
            std::slice::from_ref(&w.tx.target_path)
        );
    }

    #[test]
    fn the_restored_app_itself_continues_startup() {
        let w = World::new(TxPhase::RollingBack);
        w.bundles(Some("1.0.0"), Some("1.1.0"));
        let ops = FakeOps::default();
        let mode = RecoveryMode::Startup {
            exe: w.candidate_exe(),
            version: "1.0.0".into(),
        };
        assert_eq!(recover(&w.paths, &ops, &mode), RecoveryOutcome::Resolved);
        assert!(ops.launched.borrow().is_empty());
        assert!(w.journal().is_none());
        assert!(!w.tx.helper_path.exists());
    }

    #[test]
    fn the_new_build_at_startup_rolls_back_and_exits() {
        let w = World::new(TxPhase::Verifying);
        w.bundles(Some("1.1.0"), Some("1.0.0"));
        let ops = FakeOps::default();
        let mode = RecoveryMode::Startup {
            exe: w.candidate_exe(),
            version: "1.1.0".into(),
        };
        let out = recover(&w.paths, &ops, &mode);
        assert_eq!(out, RecoveryOutcome::Relaunched);
        assert!(out.startup_should_exit());
        assert_eq!(w.target_version().as_deref(), Some("1.0.0"));
    }

    #[test]
    fn committed_publishes_the_commit_and_keeps_candidate_files() {
        let w = World::new(TxPhase::Committed);
        w.bundles(Some("1.1.0"), Some("1.0.0"));
        us::write_tx_state(&w.paths, &w.nonce, TxControlState::Active).unwrap();
        w.ack(5, true);
        let ops = FakeOps::default();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Resolved);
        assert_eq!(w.state(), Some(TxControlState::Committed));
        assert!(us::read_ack(&w.paths, &w.nonce, true).is_some());
        assert!(!w.tx.staging_path.exists());
        assert!(w.journal().is_none());
        // Idempotent.
        assert_eq!(manual(&w, &ops), RecoveryOutcome::NoTransaction);
    }

    #[test]
    fn rolled_back_relaunch_rules() {
        // Not pending: just clean up.
        let w = World::new(TxPhase::RolledBack);
        let ops = FakeOps::default();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Resolved);
        assert!(!w.tx.helper_path.exists());

        // Pending, engine.lock free: launch.
        let mut w = World::new(TxPhase::RolledBack);
        w.tx.relaunch_pending = true;
        w.write();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Relaunched);
        assert!(w.tx.helper_path.exists());

        // Pending, the restored app already holds engine.lock: done.
        let mut w = World::new(TxPhase::RolledBack);
        w.tx.relaunch_pending = true;
        w.write();
        let _e = w.hold_engine_lock(55);
        ops.procs.borrow_mut().push((55, w.candidate_exe()));
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Resolved);
        assert!(w.journal().is_none());

        // Pending, an unrelated engine holds it: keep the obligation.
        let mut w = World::new(TxPhase::RolledBack);
        w.tx.relaunch_pending = true;
        w.write();
        let _e = w.hold_engine_lock(56);
        assert!(matches!(manual(&w, &ops), RecoveryOutcome::Deferred(_)));
        assert!(w.journal().is_some());
        assert!(w.tx.helper_path.exists());
    }

    #[test]
    fn a_corrupt_or_tampered_journal_is_set_aside_and_candidates_revoked() {
        let w = World::new(TxPhase::Verifying);
        us::write_tx_state(&w.paths, &w.nonce, TxControlState::Active).unwrap();
        fs::write(w.paths.tx_journal(), "{garbage").unwrap();
        let ops = FakeOps::default();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Resolved);
        assert!(!w.paths.tx_journal().exists());
        assert_eq!(w.state(), Some(TxControlState::Revoked));
        assert!(
            fs::read_dir(&w.paths.dir).unwrap().any(|e| e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("corrupt"))
        );

        // Valid JSON pointing at somewhere else entirely.
        let mut w = World::new(TxPhase::Prepared);
        w.tx.staging_path = PathBuf::from("/Users/someone/Documents");
        w.write();
        assert_eq!(manual(&w, &ops), RecoveryOutcome::Resolved);
        assert!(!w.paths.tx_journal().exists());
    }

    #[test]
    fn engine_lock_guard_excludes_engines() {
        let w = World::new(TxPhase::Prepared);
        let g = EngineLockGuard::try_acquire(&w.paths).unwrap().unwrap();
        assert!(EngineLockGuard::try_acquire(&w.paths).unwrap().is_none());
        assert!(crate::state::InstanceLock::acquire(Some(&w.paths.dir)).is_err());
        drop(g);
        assert!(EngineLockGuard::try_acquire(&w.paths).unwrap().is_some());
        let _ = w.apps;
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn real_rename_swap_exchanges_directories() {
        let t = TempDir::new();
        fs::create_dir_all(&t.0).unwrap();
        let (a, b) = (t.0.join("a.app"), t.0.join(".b"));
        bundle(&a, "1");
        bundle(&b, "2");
        sys::rename_swap(&a, &b).unwrap();
        assert_eq!(fs::read_to_string(a.join("version")).unwrap(), "2");
        assert_eq!(fs::read_to_string(b.join("version")).unwrap(), "1");
        assert!(sys::rename_swap(&a, &t.0.join("missing")).is_err());
        assert!(sys::pid_alive(std::process::id()));
        let me = sys::pid_path(std::process::id()).unwrap();
        assert_eq!(me, std::env::current_exe().unwrap().canonicalize().unwrap());
    }
}
