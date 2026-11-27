//! On-disk state of an in-progress app update, shared by every Oto process of one user.
//!
//! See docs/plans/0003-app-updates.md (Decision 6). Everything lives in the same non-purgeable
//! directory as `engine.lock` and `restore-input` ([`state_dir`]):
//!
//! | File | Writer | Content |
//! |---|---|---|
//! | `updating.lock` | update helper / recovery | held (`flock`) for a whole transaction |
//! | `update-tx.json` | helper / recovery, under `updating.lock` | [`UpdateTx`] |
//! | `tx-<nonce>-state` | helper / recovery, under `updating.lock` | [`TxControlState`] |
//! | `ack-<nonce>-started` | the candidate only | [`AckRecord`] |
//! | `ack-<nonce>-ready` | the candidate only | [`AckRecord`] with `ready_at` |
//! | `update-helper-<nonce>` | the old app | the helper executable |
//!
//! [`crate::Engine::start_with`] enforces admission with these: a normal engine does not start
//! while an update holds `updating.lock` or an unresolved journal exists ([`update_blocked`]),
//! and the designated new build (the candidate, [`CandidateToken`]) starts only while its
//! `tx-<nonce>-state` is `active`, inert until committed. Both are checked before and again
//! after taking `engine.lock`.
//!
//! All writes are atomic (temp file, fsync, rename), so readers see the old content or the new.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// The update journal.
pub const TX_JOURNAL: &str = "update-tx.json";
/// Held exclusively for the whole of an update transaction.
pub const UPDATING_LOCK: &str = "updating.lock";
/// Held by the running engine (see [`crate::Engine`]); holds that engine's PID.
pub const ENGINE_LOCK: &str = "engine.lock";
/// The default input to put back after a crash.
pub const RESTORE_INPUT: &str = "restore-input";

/// The per-user state directory (`~/Library/Application Support/oto` on macOS), or `None` if
/// the home directory is unknown. Never a cache directory: nothing here may be purged while a
/// transaction is unresolved.
pub fn state_dir() -> Option<PathBuf> {
    crate::state::default_dir()
}

/// The paths of the update files in one state directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpdatePaths {
    pub dir: PathBuf,
}

impl UpdatePaths {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
    /// In [`state_dir`].
    pub fn default_location() -> Option<Self> {
        state_dir().map(Self::new)
    }
    pub fn updating_lock(&self) -> PathBuf {
        self.dir.join(UPDATING_LOCK)
    }
    pub fn tx_journal(&self) -> PathBuf {
        self.dir.join(TX_JOURNAL)
    }
    pub fn engine_lock(&self) -> PathBuf {
        self.dir.join(ENGINE_LOCK)
    }
    pub fn restore_input(&self) -> PathBuf {
        self.dir.join(RESTORE_INPUT)
    }
    pub fn tx_state(&self, nonce: &Nonce) -> PathBuf {
        self.dir.join(format!("tx-{}-state", nonce.0))
    }
    pub fn ack_started(&self, nonce: &Nonce) -> PathBuf {
        self.dir.join(format!("ack-{}-started", nonce.0))
    }
    pub fn ack_ready(&self, nonce: &Nonce) -> PathBuf {
        self.dir.join(format!("ack-{}-ready", nonce.0))
    }
    pub fn update_helper(&self, nonce: &Nonce) -> PathBuf {
        self.dir.join(format!("update-helper-{}", nonce.0))
    }
}

/// A transaction id, safe to embed in file names: 1 to 64 ASCII letters, digits, `-` or `_`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Nonce(String);

impl Nonce {
    pub fn parse(s: &str) -> Result<Self, String> {
        let ok = !s.is_empty()
            && s.len() <= 64
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
        if ok {
            Ok(Self(s.to_owned()))
        } else {
            Err(format!("invalid update nonce {s:?}"))
        }
    }
    /// A fresh random nonce (128 bits from the OS, hex).
    pub fn random() -> io::Result<Self> {
        let mut b = [0u8; 16];
        File::open("/dev/urandom").and_then(|mut f| io::Read::read_exact(&mut f, &mut b))?;
        Ok(Self(b.iter().map(|x| format!("{x:02x}")).collect()))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Nonce {
    type Error = String;
    fn try_from(s: String) -> Result<Self, String> {
        Self::parse(&s)
    }
}

impl From<Nonce> for String {
    fn from(n: Nonce) -> String {
        n.0
    }
}

impl std::fmt::Display for Nonce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where an update transaction is. Written to [`UpdateTx::phase`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TxPhase {
    /// Staged; the installed app is untouched.
    Prepared,
    /// The bundle swap is in progress.
    Swapping,
    /// Swapped; the new build is starting and has not committed.
    Verifying,
    /// Terminal: the new build is in place for good.
    Committed,
    /// The old build is being swapped back.
    RollingBack,
    /// Terminal: the old build is back.
    RolledBack,
}

impl TxPhase {
    /// The installed app may be mid-swap or unverified: no engine but the candidate may run.
    pub fn is_unresolved(self) -> bool {
        matches!(self, Self::Swapping | Self::Verifying | Self::RollingBack)
    }
}

/// `update-tx.json`. Times are Unix seconds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateTx {
    pub nonce: Nonce,
    pub phase: TxPhase,
    pub started_at: u64,
    #[serde(default)]
    pub stage1_deadline: Option<u64>,
    #[serde(default)]
    pub stage2_deadline: Option<u64>,
    #[serde(default)]
    pub commit_deadline: Option<u64>,
    pub target_path: PathBuf,
    pub staging_path: PathBuf,
    pub old_version: String,
    pub new_version: String,
    pub helper_path: PathBuf,
    #[serde(default)]
    pub candidate_pid: Option<u32>,
    /// After a rollback: the restored app still has to be launched.
    #[serde(default)]
    pub relaunch_pending: bool,
}

/// `tx-<nonce>-state`: the helper's signal to the candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxControlState {
    /// The candidate may start (inert) and acknowledge.
    Active,
    /// The update is durable: the candidate commits ([`crate::Engine::commit_candidate`]).
    Committed,
    /// Rolled back or abandoned: the candidate exits with no side effects.
    Revoked,
}

impl TxControlState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Committed => "committed",
            Self::Revoked => "revoked",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "active" => Some(Self::Active),
            "committed" => Some(Self::Committed),
            "revoked" => Some(Self::Revoked),
            _ => None,
        }
    }
}

/// `ack-<nonce>-started` (no `ready_at`) and `ack-<nonce>-ready`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AckRecord {
    pub nonce: Nonce,
    pub pid: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ready_at: Option<u64>,
}

/// Identifies the process as the designated new build of transaction `nonce`
/// ([`crate::EngineOptions::update_candidate`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateToken {
    pub nonce: Nonce,
}

impl CandidateToken {
    pub fn new(nonce: Nonce) -> Self {
        Self { nonce }
    }
}

/// Seconds since the Unix epoch.
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// ---------------------------------------------------------------------------------------------
// updating.lock

/// An exclusive `flock` on `updating.lock`. The kernel drops it when the file closes, including
/// on a crash, so it can never go stale. The file itself is never deleted.
#[derive(Debug)]
pub struct UpdatingLock {
    _file: File,
}

impl UpdatingLock {
    /// `Ok(None)` if another process holds it.
    pub fn try_acquire(paths: &UpdatePaths) -> io::Result<Option<Self>> {
        let file = open_lock_file(&paths.updating_lock())?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { _file: file })),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(e)) => Err(e),
        }
    }

    /// Retries [`Self::try_acquire`] until `timeout`, so a momentary probe by another process
    /// never fails an update.
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

    /// Whether some process holds `updating.lock` right now. Probes with a shared lock, so
    /// concurrent probes never block each other (only an exclusive holder makes it `true`).
    /// A lock file that cannot be opened counts as not held.
    pub fn is_held(paths: &UpdatePaths) -> bool {
        let Ok(file) = OpenOptions::new().read(true).open(paths.updating_lock()) else {
            return false;
        };
        match file.try_lock_shared() {
            Ok(()) => false,
            Err(TryLockError::WouldBlock) => true,
            Err(TryLockError::Error(_)) => false,
        }
    }
}

fn open_lock_file(path: &Path) -> io::Result<File> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut o = OpenOptions::new();
    o.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(path)
}

// ---------------------------------------------------------------------------------------------
// Journal, control state, acks

/// `Ok(None)` if there is no journal. A journal that exists but does not parse is an `Err`.
pub fn read_tx(paths: &UpdatePaths) -> io::Result<Option<UpdateTx>> {
    let text = match fs::read_to_string(paths.tx_journal()) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Atomically replaces `update-tx.json`. Call only while holding [`UpdatingLock`].
pub fn write_tx(paths: &UpdatePaths, tx: &UpdateTx) -> io::Result<()> {
    let json = serde_json::to_vec_pretty(tx).map_err(io::Error::other)?;
    write_atomic(&paths.tx_journal(), &json)
}

pub fn remove_tx(paths: &UpdatePaths) -> io::Result<()> {
    remove_if_exists(&paths.tx_journal())
}

/// `None` if missing or unreadable (both mean "not active" to a candidate).
pub fn read_tx_state(paths: &UpdatePaths, nonce: &Nonce) -> Option<TxControlState> {
    TxControlState::parse(&fs::read_to_string(paths.tx_state(nonce)).ok()?)
}

/// Atomically writes `tx-<nonce>-state`. Helper or recovery only, under [`UpdatingLock`].
pub fn write_tx_state(paths: &UpdatePaths, nonce: &Nonce, s: TxControlState) -> io::Result<()> {
    write_atomic(
        &paths.tx_state(nonce),
        format!("{}\n", s.as_str()).as_bytes(),
    )
}

/// Atomically writes `ack-<nonce>-started` (`ready_at: None`) or `ack-<nonce>-ready`. The
/// candidate only.
pub fn write_ack(paths: &UpdatePaths, ack: &AckRecord) -> io::Result<()> {
    let path = if ack.ready_at.is_some() {
        paths.ack_ready(&ack.nonce)
    } else {
        paths.ack_started(&ack.nonce)
    };
    let json = serde_json::to_vec(ack).map_err(io::Error::other)?;
    write_atomic(&path, &json)
}

/// Reads `ack-<nonce>-ready` (`ready = true`) or `ack-<nonce>-started`. `None` if missing,
/// unreadable or for another nonce.
pub fn read_ack(paths: &UpdatePaths, nonce: &Nonce, ready: bool) -> Option<AckRecord> {
    let path = if ready {
        paths.ack_ready(nonce)
    } else {
        paths.ack_started(nonce)
    };
    let ack: AckRecord = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    (ack.nonce == *nonce).then_some(ack)
}

/// The PID the running engine wrote into `engine.lock`, if any. Recovery compares it with the
/// candidate's PID before touching a running app.
pub fn read_lock_holder_pid(paths: &UpdatePaths) -> Option<u32> {
    fs::read_to_string(paths.engine_lock())
        .ok()?
        .trim()
        .parse()
        .ok()
}

// ---------------------------------------------------------------------------------------------
// Admission

/// Why an engine may not start now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Blocked {
    /// An update holds `updating.lock`.
    UpdateRunning,
    /// `update-tx.json` is in an unresolved phase (a helper died mid-update): recovery must run.
    Unresolved(TxPhase),
    /// `update-tx.json` exists but cannot be read: recovery must look at it.
    UnreadableJournal,
}

/// Whether a normal (non-candidate) engine must not start: an update holds `updating.lock`,
/// or the journal shows an unresolved transaction. `prepared` and terminal phases do not block.
pub fn update_blocked(paths: &UpdatePaths) -> Option<Blocked> {
    if UpdatingLock::is_held(paths) {
        return Some(Blocked::UpdateRunning);
    }
    match read_tx(paths) {
        Ok(Some(tx)) if tx.phase.is_unresolved() => Some(Blocked::Unresolved(tx.phase)),
        Ok(_) => None,
        Err(_) => Some(Blocked::UnreadableJournal),
    }
}

/// Whether the candidate `token` may start its engine: its `tx-<nonce>-state` is `active`.
/// (The commit can only follow the candidate's `ready` ack, which follows engine start.)
pub fn candidate_admitted(paths: &UpdatePaths, token: &CandidateToken) -> bool {
    read_tx_state(paths, &token.nonce) == Some(TxControlState::Active)
}

// ---------------------------------------------------------------------------------------------

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Temp file (0600) next to `file`, fsync, rename over it, fsync the directory.
pub(crate) fn write_atomic(file: &Path, content: &[u8]) -> io::Result<()> {
    let dir = file.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir)?;
    let name = file.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = dir.join(format!(".{name}.tmp{}", std::process::id()));
    let res = (|| {
        let mut o = OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        let mut f = o.open(&tmp)?;
        f.write_all(content)?;
        f.sync_all()?;
        fs::rename(&tmp, file)?;
        if let Ok(d) = File::open(dir) {
            let _ = d.sync_all();
        }
        Ok(())
    })();
    if res.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::tests::TempDir;

    pub(crate) fn sample_tx(nonce: &Nonce, phase: TxPhase) -> UpdateTx {
        UpdateTx {
            nonce: nonce.clone(),
            phase,
            started_at: 1,
            stage1_deadline: None,
            stage2_deadline: None,
            commit_deadline: None,
            target_path: "/Applications/Oto.app".into(),
            staging_path: "/Applications/.Oto-update-x".into(),
            old_version: "0.0.3".into(),
            new_version: "0.0.4".into(),
            helper_path: "/tmp/update-helper-x".into(),
            candidate_pid: None,
            relaunch_pending: false,
        }
    }

    #[test]
    fn nonce_rejects_path_tricks() {
        assert!(Nonce::parse("abc-123_X").is_ok());
        for bad in ["", "../x", "a/b", "a.b", "a b", &"x".repeat(65)] {
            assert!(Nonce::parse(bad).is_err(), "{bad:?}");
        }
        let r = Nonce::random().unwrap();
        assert_eq!(r.as_str().len(), 32);
        assert!(serde_json::from_str::<Nonce>("\"../evil\"").is_err());
    }

    #[test]
    fn journal_round_trips_with_snake_case_phases() {
        let t = TempDir::new();
        let p = UpdatePaths::new(&t.0);
        assert!(read_tx(&p).unwrap().is_none());
        let n = Nonce::parse("n1").unwrap();
        let tx = sample_tx(&n, TxPhase::RollingBack);
        write_tx(&p, &tx).unwrap();
        assert_eq!(read_tx(&p).unwrap(), Some(tx));
        let raw = fs::read_to_string(p.tx_journal()).unwrap();
        assert!(raw.contains("\"rolling_back\""), "{raw}");
        // Only the journal is left: no temp file.
        assert_eq!(fs::read_dir(&t.0).unwrap().count(), 1);
        // Optional fields may be absent (older writers).
        fs::write(
            p.tx_journal(),
            r#"{"nonce":"n1","phase":"prepared","started_at":5,"target_path":"/a",
                "staging_path":"/b","old_version":"1","new_version":"2","helper_path":"/h"}"#,
        )
        .unwrap();
        let tx = read_tx(&p).unwrap().unwrap();
        assert_eq!(tx.phase, TxPhase::Prepared);
        assert_eq!(tx.candidate_pid, None);
        assert!(!tx.relaunch_pending);
        remove_tx(&p).unwrap();
        remove_tx(&p).unwrap();
    }

    #[test]
    fn control_state_and_acks() {
        let t = TempDir::new();
        let p = UpdatePaths::new(&t.0);
        let n = Nonce::parse("n2").unwrap();
        assert_eq!(read_tx_state(&p, &n), None);
        for s in [
            TxControlState::Active,
            TxControlState::Committed,
            TxControlState::Revoked,
        ] {
            write_tx_state(&p, &n, s).unwrap();
            assert_eq!(read_tx_state(&p, &n), Some(s));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(p.tx_state(&n)).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let started = AckRecord {
            nonce: n.clone(),
            pid: 42,
            ready_at: None,
        };
        let ready = AckRecord {
            ready_at: Some(7),
            ..started.clone()
        };
        write_ack(&p, &started).unwrap();
        assert_eq!(read_ack(&p, &n, false), Some(started));
        assert_eq!(read_ack(&p, &n, true), None);
        write_ack(&p, &ready).unwrap();
        assert_eq!(read_ack(&p, &n, true), Some(ready));
        // An ack for another nonce under this name is ignored.
        let other = Nonce::parse("other").unwrap();
        fs::copy(p.ack_ready(&n), p.ack_ready(&other)).unwrap();
        assert_eq!(read_ack(&p, &other, true), None);
    }

    #[test]
    fn updating_lock_is_exclusive_and_probe_sees_it() {
        let t = TempDir::new();
        let p = UpdatePaths::new(&t.0);
        assert!(!UpdatingLock::is_held(&p));
        let held = UpdatingLock::try_acquire(&p).unwrap().expect("free");
        assert!(UpdatingLock::is_held(&p));
        assert!(UpdatingLock::try_acquire(&p).unwrap().is_none());
        let start = Instant::now();
        assert!(
            UpdatingLock::acquire_timeout(&p, Duration::from_millis(120))
                .unwrap()
                .is_none()
        );
        assert!(start.elapsed() >= Duration::from_millis(120));
        drop(held);
        assert!(!UpdatingLock::is_held(&p));
        // The file stays (never deleted).
        assert!(p.updating_lock().exists());
        assert!(UpdatingLock::try_acquire(&p).unwrap().is_some());
    }

    #[test]
    fn blocked_by_lock_or_unresolved_journal_only() {
        let t = TempDir::new();
        let p = UpdatePaths::new(&t.0);
        let n = Nonce::parse("n3").unwrap();
        assert_eq!(update_blocked(&p), None);
        for phase in [TxPhase::Prepared, TxPhase::Committed, TxPhase::RolledBack] {
            write_tx(&p, &sample_tx(&n, phase)).unwrap();
            assert_eq!(update_blocked(&p), None, "{phase:?}");
        }
        for phase in [TxPhase::Swapping, TxPhase::Verifying, TxPhase::RollingBack] {
            write_tx(&p, &sample_tx(&n, phase)).unwrap();
            assert_eq!(update_blocked(&p), Some(Blocked::Unresolved(phase)));
        }
        fs::write(p.tx_journal(), "{not json").unwrap();
        assert_eq!(update_blocked(&p), Some(Blocked::UnreadableJournal));
        remove_tx(&p).unwrap();
        let l = UpdatingLock::try_acquire(&p).unwrap().unwrap();
        assert_eq!(update_blocked(&p), Some(Blocked::UpdateRunning));
        drop(l);
        assert_eq!(update_blocked(&p), None);
    }

    #[test]
    fn candidate_admitted_only_while_active() {
        let t = TempDir::new();
        let p = UpdatePaths::new(&t.0);
        let tok = CandidateToken::new(Nonce::parse("n4").unwrap());
        assert!(!candidate_admitted(&p, &tok));
        write_tx_state(&p, &tok.nonce, TxControlState::Active).unwrap();
        assert!(candidate_admitted(&p, &tok));
        write_tx_state(&p, &tok.nonce, TxControlState::Committed).unwrap();
        assert!(!candidate_admitted(&p, &tok));
        write_tx_state(&p, &tok.nonce, TxControlState::Revoked).unwrap();
        assert!(!candidate_admitted(&p, &tok));
        fs::write(p.tx_state(&tok.nonce), "garbage").unwrap();
        assert!(!candidate_admitted(&p, &tok));
    }
}
