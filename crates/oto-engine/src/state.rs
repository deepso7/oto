//! Small on-disk state shared by every Oto process of one user: the single-instance lock and
//! the default input to restore after a crash.
//!
//! Lives in `~/Library/Application Support/oto/` on macOS and `$XDG_STATE_HOME/oto/` (default
//! `~/.local/state/oto/`) elsewhere. All of it is best-effort: without a usable directory the
//! engine still runs, just without crash recovery or the instance check. Nothing here is ever
//! called from an audio callback.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const LOCK_FILE: &str = "engine.lock";
const RESTORE_FILE: &str = "restore-input";

/// The per-user state directory, or `None` if the home directory is unknown.
pub(crate) fn default_dir() -> Option<PathBuf> {
    let home = || {
        std::env::var_os("HOME")
            .filter(|h| !h.is_empty())
            .map(PathBuf::from)
    };
    if cfg!(target_os = "macos") {
        return Some(home()?.join("Library/Application Support/oto"));
    }
    match std::env::var_os("XDG_STATE_HOME").filter(|v| !v.is_empty()) {
        Some(x) => Some(PathBuf::from(x).join("oto")),
        None => Some(home()?.join(".local/state/oto")),
    }
}

/// Held for the lifetime of an [`crate::Engine`]: an exclusive advisory lock on a file in the
/// state directory. The OS releases it when the file closes, including when the process dies,
/// so a crash never leaves a stale lock.
pub(crate) struct InstanceLock {
    _file: File,
}

impl InstanceLock {
    /// `Ok(None)` when the lock file cannot be used at all (no directory, read-only home): the
    /// engine then runs unguarded rather than refusing to start. `Err` only when another engine
    /// holds the lock.
    pub fn acquire(dir: Option<&Path>) -> Result<Option<Self>, String> {
        let Some(dir) = dir else { return Ok(None) };
        if fs::create_dir_all(dir).is_err() {
            return Ok(None);
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(LOCK_FILE));
        let Ok(file) = file else { return Ok(None) };
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { _file: file })),
            Err(TryLockError::WouldBlock) => Err(crate::ALREADY_RUNNING.into()),
            // Locks unsupported on this filesystem: nothing to enforce with.
            Err(TryLockError::Error(_)) => Ok(None),
        }
    }
}

/// The default input to put back once Oto Microphone stops being it, kept in memory and
/// mirrored to disk so it survives a crash.
///
/// The system default itself is passed in by the caller (as values or closures), which keeps
/// this logic testable without Core Audio.
pub(crate) struct RestoreTarget {
    /// UID of Oto Microphone.
    mic: String,
    file: Option<PathBuf>,
    /// Also serialises changes of the system default made through this type.
    prev: Mutex<Option<String>>,
}

impl RestoreTarget {
    /// Picks up what a previous run left behind. A leftover target is adopted only while Oto
    /// Microphone is still the default (`current`); otherwise the user moved on and it is
    /// discarded.
    pub fn open(dir: Option<&Path>, mic: &str, current: Option<&str>) -> Self {
        let file = dir.map(|d| d.join(RESTORE_FILE));
        let mut prev = file.as_deref().and_then(read);
        if prev.is_some() && current != Some(mic) {
            prev = None;
            if let Some(f) = &file {
                remove(f);
            }
        }
        Self {
            mic: mic.to_owned(),
            file,
            prev: Mutex::new(prev),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<String>> {
        self.prev.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[cfg(test)]
    pub fn pending(&self) -> Option<String> {
        self.lock().clone()
    }

    /// Make Oto Microphone the default. `current` is the default right now; `set` changes it.
    /// The target is written to disk *before* the default changes, so a crash at any point
    /// leaves either nothing to undo or a record of how to undo it.
    pub fn switch_to_oto(
        &self,
        current: Option<String>,
        set: impl FnOnce(&str) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut prev = self.lock();
        if current.as_deref() == Some(self.mic.as_str()) {
            // Already the default: whatever target we hold (possibly adopted) stays valid.
            return Ok(());
        }
        let before = prev.clone();
        self.store(current.as_deref());
        *prev = current;
        if let Err(e) = set(&self.mic) {
            self.store(before.as_deref());
            *prev = before;
            return Err(e);
        }
        Ok(())
    }

    /// Put the remembered default back, if Oto Microphone is still the default. Clears the
    /// target either way, except when setting it fails (so a later attempt can retry).
    pub fn restore(
        &self,
        current: impl FnOnce() -> Option<String>,
        set: impl FnOnce(&str) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut prev = self.lock();
        let Some(uid) = prev.take() else {
            return Ok(());
        };
        if current().as_deref() == Some(self.mic.as_str())
            && let Err(e) = set(&uid)
        {
            *prev = Some(uid);
            return Err(e);
        }
        self.store(None);
        Ok(())
    }

    /// The system default input changed. If it is no longer Oto Microphone, the user (or we)
    /// picked something else and there is nothing left to restore.
    pub fn default_changed(&self, current: impl FnOnce() -> Option<String>) {
        let mut prev = self.lock();
        if prev.is_some() && current().as_deref() != Some(self.mic.as_str()) {
            *prev = None;
            self.store(None);
        }
    }

    fn store(&self, uid: Option<&str>) {
        let Some(file) = &self.file else { return };
        match uid {
            Some(uid) => {
                let _ = write_atomic(file, uid);
            }
            None => remove(file),
        }
    }
}

fn read(file: &Path) -> Option<String> {
    let s = fs::read_to_string(file).ok()?;
    let s = s.trim();
    (!s.is_empty()).then(|| s.to_owned())
}

fn remove(file: &Path) {
    let _ = fs::remove_file(file);
}

/// Write to a sibling temp file, flush it to disk, then rename over the target: readers see
/// the old content or the new, never a partial file.
fn write_atomic(file: &Path, content: &str) -> std::io::Result<()> {
    if let Some(dir) = file.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = file.with_extension(format!("tmp{}", std::process::id()));
    let res = (|| {
        let mut f = File::create(&tmp)?;
        f.write_all(content.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all()?;
        fs::rename(&tmp, file)
    })();
    if res.is_err() {
        remove(&tmp);
    }
    res
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

    /// A fresh directory under the system temp dir, removed on drop.
    pub(crate) struct TempDir(pub PathBuf);

    impl TempDir {
        pub fn new() -> Self {
            static N: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!(
                "oto-test-{}-{}",
                std::process::id(),
                N.fetch_add(1, Relaxed)
            ));
            let _ = fs::remove_dir_all(&dir);
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const MIC: &str = "dev.oto.microphone";

    /// A stand-in for the system default input.
    struct Sys(RefCell<Option<String>>);

    impl Sys {
        fn new(d: &str) -> Self {
            Self(RefCell::new(Some(d.to_owned())))
        }
        fn get(&self) -> Option<String> {
            self.0.borrow().clone()
        }
        fn set(&self, uid: &str) -> Result<(), String> {
            *self.0.borrow_mut() = Some(uid.to_owned());
            Ok(())
        }
    }

    fn on_disk(t: &TempDir) -> Option<String> {
        read(&t.0.join(RESTORE_FILE))
    }

    #[test]
    fn switch_persists_and_restore_clears() {
        let t = TempDir::new();
        let sys = Sys::new("builtin");
        let r = RestoreTarget::open(Some(&t.0), MIC, sys.get().as_deref());
        assert_eq!(r.pending(), None);
        r.switch_to_oto(sys.get(), |u| sys.set(u)).unwrap();
        assert_eq!(sys.get().as_deref(), Some(MIC));
        assert_eq!(r.pending().as_deref(), Some("builtin"));
        assert_eq!(on_disk(&t).as_deref(), Some("builtin"));
        // No temp file is left next to it.
        assert_eq!(fs::read_dir(&t.0).unwrap().count(), 1);
        // Asking again changes nothing.
        r.switch_to_oto(sys.get(), |_| panic!("already default"))
            .unwrap();
        assert_eq!(on_disk(&t).as_deref(), Some("builtin"));

        r.restore(|| sys.get(), |u| sys.set(u)).unwrap();
        assert_eq!(sys.get().as_deref(), Some("builtin"));
        assert_eq!(r.pending(), None);
        assert_eq!(on_disk(&t), None);
        // A second restore (e.g. on drop) is a no-op.
        r.restore(|| sys.get(), |_| panic!("nothing to restore"))
            .unwrap();
    }

    #[test]
    fn crash_leaves_a_target_the_next_run_adopts() {
        let t = TempDir::new();
        let sys = Sys::new("usb");
        {
            let r = RestoreTarget::open(Some(&t.0), MIC, sys.get().as_deref());
            r.switch_to_oto(sys.get(), |u| sys.set(u)).unwrap();
            // Dropped without `restore`: SIGKILL, panic=abort, power loss.
        }
        assert_eq!(on_disk(&t).as_deref(), Some("usb"));
        let r = RestoreTarget::open(Some(&t.0), MIC, sys.get().as_deref());
        assert_eq!(r.pending().as_deref(), Some("usb"));
        // Oto Microphone is still the default, so `switch_to_oto` keeps the adopted target.
        r.switch_to_oto(sys.get(), |_| panic!("already default"))
            .unwrap();
        r.restore(|| sys.get(), |u| sys.set(u)).unwrap();
        assert_eq!(sys.get().as_deref(), Some("usb"));
        assert_eq!(on_disk(&t), None);
    }

    #[test]
    fn leftover_target_is_dropped_if_the_user_moved_on() {
        let t = TempDir::new();
        let sys = Sys::new("usb");
        {
            let r = RestoreTarget::open(Some(&t.0), MIC, sys.get().as_deref());
            r.switch_to_oto(sys.get(), |u| sys.set(u)).unwrap();
        }
        // After the crash the user picked another mic in System Settings.
        sys.set("airpods").unwrap();
        let r = RestoreTarget::open(Some(&t.0), MIC, sys.get().as_deref());
        assert_eq!(r.pending(), None);
        assert_eq!(on_disk(&t), None);
    }

    #[test]
    fn picking_another_default_while_running_clears_the_target() {
        let t = TempDir::new();
        let sys = Sys::new("builtin");
        let r = RestoreTarget::open(Some(&t.0), MIC, sys.get().as_deref());
        r.switch_to_oto(sys.get(), |u| sys.set(u)).unwrap();
        // A notification while Oto Microphone is still the default keeps it.
        r.default_changed(|| sys.get());
        assert_eq!(on_disk(&t).as_deref(), Some("builtin"));
        sys.set("airpods").unwrap();
        r.default_changed(|| sys.get());
        assert_eq!(r.pending(), None);
        assert_eq!(on_disk(&t), None);
        // Restoring now must not override the user's choice.
        r.restore(|| sys.get(), |_| panic!("must not set")).unwrap();
        assert_eq!(sys.get().as_deref(), Some("airpods"));
    }

    #[test]
    fn restore_skips_setting_when_default_already_changed_but_still_clears() {
        let t = TempDir::new();
        let sys = Sys::new("builtin");
        let r = RestoreTarget::open(Some(&t.0), MIC, sys.get().as_deref());
        r.switch_to_oto(sys.get(), |u| sys.set(u)).unwrap();
        sys.set("airpods").unwrap();
        r.restore(|| sys.get(), |_| panic!("must not set")).unwrap();
        assert_eq!(on_disk(&t), None);
    }

    #[test]
    fn failures_keep_state_consistent() {
        let t = TempDir::new();
        let sys = Sys::new("builtin");
        let r = RestoreTarget::open(Some(&t.0), MIC, sys.get().as_deref());
        // Switching fails: nothing is left to restore.
        assert!(r.switch_to_oto(sys.get(), |_| Err("nope".into())).is_err());
        assert_eq!(r.pending(), None);
        assert_eq!(on_disk(&t), None);
        // Restoring fails (device unplugged): the target is kept for a retry.
        r.switch_to_oto(sys.get(), |u| sys.set(u)).unwrap();
        assert!(r.restore(|| sys.get(), |_| Err("gone".into())).is_err());
        assert_eq!(r.pending().as_deref(), Some("builtin"));
        assert_eq!(on_disk(&t).as_deref(), Some("builtin"));
    }

    #[test]
    fn works_in_memory_without_a_directory() {
        let sys = Sys::new("builtin");
        let r = RestoreTarget::open(None, MIC, sys.get().as_deref());
        r.switch_to_oto(sys.get(), |u| sys.set(u)).unwrap();
        r.restore(|| sys.get(), |u| sys.set(u)).unwrap();
        assert_eq!(sys.get().as_deref(), Some("builtin"));
    }

    #[test]
    fn second_instance_is_refused_until_the_first_is_gone() {
        let t = TempDir::new();
        let first = InstanceLock::acquire(Some(&t.0)).unwrap();
        assert!(first.is_some());
        let second = InstanceLock::acquire(Some(&t.0));
        assert_eq!(second.err().as_deref(), Some(crate::ALREADY_RUNNING));
        drop(first);
        assert!(InstanceLock::acquire(Some(&t.0)).unwrap().is_some());
        // No directory: nothing to lock, but not an error.
        assert!(InstanceLock::acquire(None).unwrap().is_none());
    }
}
