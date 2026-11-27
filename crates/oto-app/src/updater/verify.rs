//! Checking for an update, and downloading, extracting and verifying one.
//!
//! An update is only offered if `update.json` carries a valid Ed25519 signature from a key in
//! [`keys::TRUSTED_KEYS`], its manifest passes every URL allowlist, this Mac meets `min_os`,
//! and the version is strictly newer. It is only staged once the archive has the exact size
//! and SHA-256 from the signed manifest, unpacks with the bounded extractor, and the unpacked
//! `Oto.app` has the right identity and version, a Developer ID signature from our team on the
//! app, its driver and its helper, and passes Gatekeeper.
//!
//! [`keys::TRUSTED_KEYS`]: super::keys::TRUSTED_KEYS

use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use oto_sign::manifest::{MAX_ARCHIVE_BYTES, MAX_ENVELOPE_BYTES, PLATFORM_MACOS};
use oto_sign::{OsVersion, SemVer, TrustedKey, UpdateManifest, VerifyError};

use super::http::Transport;
use super::keys;
use crate::driver;

/// The app's bundle identifier and executable, which an update must keep.
pub const APP_BUNDLE_ID: &str = "dev.oto.app";
pub const APP_EXECUTABLE: &str = "oto-app";

/// Neutral copy for an envelope no trusted key signed (key rotated past this build, or forged).
pub const UNRECOGNIZED_KEY_MESSAGE: &str = "Oto couldn't verify this update's signature — check \
                                            GitHub Releases for the latest version.";

/// A verified update offer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpdateInfo {
    pub version: String,
    /// Always `https://github.com/deepso7/oto/releases/tag/v<version>`.
    pub notes_url: String,
    pub asset_url: String,
    pub sha256: String,
    pub size: u64,
    pub min_os: String,
    /// The HAL driver revision the release bundles.
    pub driver_version: u32,
    /// The oldest loaded driver revision the release's engine accepts.
    pub min_driver_version: u32,
    pub published_at: String,
}

/// What a check found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckOutcome {
    UpToDate,
    Available(UpdateInfo),
    /// No signature from a key this build trusts: show [`UNRECOGNIZED_KEY_MESSAGE`] with a link
    /// to `releases_url` (always [`keys::RELEASES_URL`]).
    UnrecognizedKey {
        releases_url: String,
    },
    /// A newer version exists but needs a newer macOS.
    IncompatibleOs {
        version: String,
        min_os: String,
    },
}

const WORK_CACHE_LOCK: &str = ".work-cache.lock";

/// Acquires the persistent cache-level lock (never unlinked) used to serialize work-directory
/// creation against orphan reclamation so a reclaimer can never unlink a sibling lock file
/// between `open` and `flock`.
fn acquire_cache_lock(cache: &Path) -> Result<File, String> {
    private_dir(cache)?;
    let lock = open_work_lock(&cache.join(WORK_CACHE_LOCK))
        .map_err(|e| format!("Couldn't lock the download folder: {e}"))?;
    lock.lock()
        .map_err(|e| format!("Couldn't lock the download folder: {e}"))?;
    Ok(lock)
}

/// Owns a private `work-<nonce>` cache directory and its sibling exclusive lock file. Dropping the
/// last handle deletes both so abandoned, cancelled or quit downloads do not accumulate app
/// bundles, while a live holder keeps the lock so crash-recovery skips it.
#[derive(Debug)]
struct WorkDirGuard {
    work_dir: PathBuf,
    lock_path: PathBuf,
    _lock: File,
}

impl WorkDirGuard {
    fn create(cache: &Path) -> Result<Self, String> {
        Self::create_with_hook(cache, || {})
    }

    fn create_with_hook(cache: &Path, after_lock_open: impl FnOnce()) -> Result<Self, String> {
        let _cache_lock = acquire_cache_lock(cache)?;
        let nonce = oto_engine::update_state::Nonce::random()
            .map_err(|e| format!("Couldn't prepare the download folder: {e}"))?;
        let lock_path = cache.join(format!(".work-{nonce}.lock"));
        let work_dir = cache.join(format!("work-{nonce}"));
        let lock = open_work_lock(&lock_path)
            .map_err(|e| format!("Couldn't lock the download folder: {e}"))?;
        after_lock_open();
        lock.try_lock()
            .map_err(|e| format!("Couldn't lock the download folder: {e}"))?;
        if let Err(e) = private_dir(&work_dir) {
            let _ = std::fs::remove_file(&lock_path);
            return Err(e);
        }
        Ok(Self {
            work_dir,
            lock_path,
            _lock: lock,
        })
    }
}

impl Drop for WorkDirGuard {
    fn drop(&mut self) {
        let _ = oto_engine::update_recovery::remove_path(&self.work_dir);
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

fn open_work_lock(path: &Path) -> std::io::Result<File> {
    let mut o = OpenOptions::new();
    o.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(path)
}

/// Removes any abandoned `work-<nonce>` directories (and orphaned `.work-<nonce>.lock` files) in
/// `cache` whose per-directory lock is not held by a live download or [`StagedUpdate`].
pub fn reclaim_abandoned_work_dirs(cache: &Path) {
    if !cache.is_dir() {
        return;
    }
    let Ok(_cache_lock) = acquire_cache_lock(cache) else {
        return;
    };
    let Ok(rd) = std::fs::read_dir(cache) else {
        return;
    };
    for e in rd.flatten() {
        let Some(name) = e.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if name == WORK_CACHE_LOCK {
            continue;
        }
        if let Some(nonce) = name.strip_prefix("work-") {
            if oto_engine::update_state::Nonce::parse(nonce).is_err() {
                continue;
            }
            let lock_path = cache.join(format!(".work-{nonce}.lock"));
            let Ok(lock) = open_work_lock(&lock_path) else {
                continue;
            };
            match lock.try_lock() {
                Ok(()) => {
                    let _ = oto_engine::update_recovery::remove_path(&e.path());
                    let _ = std::fs::remove_file(&lock_path);
                }
                Err(TryLockError::WouldBlock) | Err(TryLockError::Error(_)) => {}
            }
        } else if let Some(nonce) = name
            .strip_prefix(".work-")
            .and_then(|r| r.strip_suffix(".lock"))
        {
            if oto_engine::update_state::Nonce::parse(nonce).is_err() {
                continue;
            }
            if cache.join(format!("work-{nonce}")).exists() {
                continue;
            }
            let Ok(lock) = open_work_lock(&e.path()) else {
                continue;
            };
            if lock.try_lock().is_ok() && !cache.join(format!("work-{nonce}")).exists() {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}

/// A downloaded, verified update in the cache, ready for [`super::apply`].
#[derive(Clone, Debug)]
pub struct StagedUpdate {
    pub info: UpdateInfo,
    /// The verified `Oto.app` (in `cache_dir`).
    pub cached_app_path: PathBuf,
    /// The private (0700) directory holding it; deleted once staged next to the app or when
    /// dropped.
    pub cache_dir: PathBuf,
    /// Owns `cache_dir` when created by [`download_and_verify_in`]; `None` for unowned fixtures.
    _work_guard: Option<Arc<WorkDirGuard>>,
}

impl StagedUpdate {
    /// Constructs a [`StagedUpdate`] without taking ownership of `cache_dir` (for demo scenes and
    /// tests whose paths are fixtures).
    pub fn unowned(info: UpdateInfo, cached_app_path: PathBuf, cache_dir: PathBuf) -> Self {
        Self {
            info,
            cached_app_path,
            cache_dir,
            _work_guard: None,
        }
    }
}

impl PartialEq for StagedUpdate {
    fn eq(&self, other: &Self) -> bool {
        self.info == other.info
            && self.cached_app_path == other.cached_app_path
            && self.cache_dir == other.cache_dir
    }
}

impl Eq for StagedUpdate {}

/// This Mac's OS version (`NSProcessInfo.operatingSystemVersion`); `None` off macOS.
pub fn current_os_version() -> Option<OsVersion> {
    #[cfg(target_os = "macos")]
    {
        let v = objc2_foundation::NSProcessInfo::processInfo().operatingSystemVersion();
        Some(OsVersion {
            major: v.majorVersion.max(0) as u64,
            minor: v.minorVersion.max(0) as u64,
            patch: v.patchVersion.max(0) as u64,
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Fetches and verifies `update.json` from [`keys::manifest_url`].
pub fn check_manifest(
    transport: &dyn Transport,
    current_version: &str,
    current_os: OsVersion,
    trusted_keys: &[TrustedKey],
    cancel: &AtomicBool,
) -> Result<CheckOutcome, String> {
    let current: SemVer = current_version.parse()?;
    let bytes = transport.get_bytes(&keys::manifest_url(), MAX_ENVELOPE_BYTES, cancel)?;
    let payload = match oto_sign::verify_envelope(&bytes, trusted_keys) {
        Ok((payload, _key_id)) => payload,
        Err(VerifyError::UnrecognizedKey) => {
            return Ok(CheckOutcome::UnrecognizedKey {
                releases_url: keys::RELEASES_URL.to_owned(),
            });
        }
        Err(e) => return Err(format!("Couldn't verify the update: {e}")),
    };
    let m = UpdateManifest::parse(&payload).map_err(|e| format!("Invalid update manifest: {e}"))?;
    let info = update_info(&m)?;
    if m.semver()? <= current {
        return Ok(CheckOutcome::UpToDate);
    }
    let asset = &m.platforms[PLATFORM_MACOS];
    if !asset.supports_os(current_os)? {
        return Ok(CheckOutcome::IncompatibleOs {
            version: m.version.clone(),
            min_os: asset.min_os.clone(),
        });
    }
    Ok(CheckOutcome::Available(info))
}

/// The macOS offer in a validated manifest, its URLs checked once more.
fn update_info(m: &UpdateManifest) -> Result<UpdateInfo, String> {
    let asset = m
        .platforms
        .get(PLATFORM_MACOS)
        .ok_or("This release has no macOS download.")?;
    oto_sign::manifest::validate_notes_url(&m.notes_url, &m.version)?;
    oto_sign::manifest::validate_asset_url(&asset.url, &m.version)?;
    Ok(UpdateInfo {
        version: m.version.clone(),
        notes_url: m.notes_url.clone(),
        asset_url: asset.url.clone(),
        sha256: asset.sha256.clone(),
        size: asset.size,
        min_os: asset.min_os.clone(),
        driver_version: m.driver_version,
        min_driver_version: m.min_driver_version,
        published_at: m.published_at.clone(),
    })
}

// ---------------------------------------------------------------------------------------------
// Download, extract, verify

/// Signature and Gatekeeper checks on an unpacked `Oto.app` (stubbed in tests).
pub trait BundleChecks: Send + Sync {
    /// Developer ID `codesign --verify --deep --strict` on the app, its embedded driver and
    /// its embedded `oto-hal-helper`.
    fn signatures(&self, app: &Path) -> Result<(), String>;
    /// `spctl --assess --type execute` on an `.app`-named copy (offline Gatekeeper check
    /// against the stapled notarization ticket). Only meaningful for Developer ID builds.
    fn gatekeeper(&self, app: &Path) -> Result<(), String>;
}

/// The real checks, pinned to this build's team ([`driver::TEAM_ID`]).
pub struct SystemChecks;

impl BundleChecks for SystemChecks {
    fn signatures(&self, app: &Path) -> Result<(), String> {
        codesign(app, &driver::app_requirement(), true)
            .map_err(|e| format!("The update's code signature isn't valid: {e}"))?;
        driver::validate_with(
            &app.join("Contents/Resources").join(driver::BUNDLE_NAME),
            &driver::driver_requirement(),
        )?;
        codesign(
            &app.join("Contents/Helpers").join(driver::HELPER_NAME),
            &driver::helper_requirement(),
            false,
        )
        .map_err(|e| format!("The update's installer helper isn't validly signed: {e}"))
    }

    fn gatekeeper(&self, app: &Path) -> Result<(), String> {
        if driver::TEAM_ID.is_none() {
            return Ok(());
        }
        let out = Command::new("/usr/sbin/spctl")
            .args(["--assess", "--type", "execute", "--verbose"])
            .arg(app)
            .output()
            .map_err(|e| format!("Couldn't run spctl: {e}"))?;
        if out.status.success() {
            Ok(())
        } else {
            Err(format!(
                "Gatekeeper rejected the update: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ))
        }
    }
}

/// `codesign --verify --strict [--deep] -R=<requirement> <path>`.
pub fn codesign(path: &Path, requirement: &str, deep: bool) -> Result<(), String> {
    let mut cmd = Command::new("/usr/bin/codesign");
    cmd.arg("--verify").arg("--strict");
    if deep {
        cmd.arg("--deep");
    }
    let out = cmd
        .arg(format!("-R={requirement}"))
        .arg(path)
        .output()
        .map_err(|e| format!("couldn't run codesign: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_owned())
    }
}

/// Checks `Contents/Info.plist` of `app`: our identifier and executable, version exactly
/// `expected` and newer than `current`.
pub fn check_bundle_info(app: &Path, expected: &str, current: &str) -> Result<(), String> {
    let plist = app.join("Contents/Info.plist");
    let get = |k| driver::plist_value(&plist, k);
    if get("CFBundleIdentifier").as_deref() != Some(APP_BUNDLE_ID) {
        return Err("The update isn't Oto (unexpected bundle identifier).".into());
    }
    if get("CFBundleExecutable").as_deref() != Some(APP_EXECUTABLE) {
        return Err("The update has an unexpected executable.".into());
    }
    let version = get("CFBundleShortVersionString").unwrap_or_default();
    if version != expected {
        return Err(format!(
            "The update is version {version:?}, not {expected} as announced."
        ));
    }
    if expected.parse::<SemVer>()? <= current.parse::<SemVer>()? {
        return Err(format!("{expected} is not newer than {current}."));
    }
    Ok(())
}

/// Free bytes for an unprivileged user on the volume holding `path`.
pub fn free_bytes(path: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
        // SAFETY: valid path and out-pointer.
        if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
            return None;
        }
        Some((st.f_bavail as u64).saturating_mul(st.f_frsize as u64))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

fn need_space(path: &Path, bytes: u64, what: &str) -> Result<(), String> {
    match free_bytes(path) {
        Some(free) if free >= bytes => Ok(()),
        Some(free) => Err(format!(
            "Not enough free disk space {what} ({} MB needed, {} MB free).",
            bytes.div_ceil(1 << 20),
            free >> 20
        )),
        None => Err(format!("Couldn't check the free disk space {what}.")),
    }
}

/// `~/Library/Caches/dev.oto.app/updates` (purgeable: only downloads and unpacked copies).
pub fn updates_cache_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|c| c.join(APP_BUNDLE_ID).join("updates"))
}

/// A private (0700) directory, created if missing.
pub fn private_dir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("Couldn't create {}: {e}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
            .map_err(|e| format!("Couldn't protect {}: {e}", dir.display()))?;
    }
    Ok(())
}

/// Downloads the archive of `info` into the cache, then extracts and verifies it. Blocks: call
/// it off the UI thread. `on_progress(received, total)`.
pub fn download_and_verify(
    transport: &dyn Transport,
    info: &UpdateInfo,
    current_version: &str,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<StagedUpdate, String> {
    let cache = updates_cache_dir().ok_or("Couldn't find your Caches folder.")?;
    let exe = std::env::current_exe().and_then(|p| p.canonicalize()).ok();
    let target_dir = exe
        .as_deref()
        .and_then(driver::app_bundle)
        .and_then(|b| b.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| cache.clone());
    download_and_verify_in(
        transport,
        info,
        current_version,
        &cache,
        &target_dir,
        &SystemChecks,
        cancel,
        on_progress,
    )
}

/// [`download_and_verify`] with explicit locations and checks.
#[allow(clippy::too_many_arguments)]
pub fn download_and_verify_in(
    transport: &dyn Transport,
    info: &UpdateInfo,
    current_version: &str,
    cache: &Path,
    target_dir: &Path,
    checks: &dyn BundleChecks,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<StagedUpdate, String> {
    // The offer must still be one we would make.
    oto_sign::manifest::validate_asset_url(&info.asset_url, &info.version)?;
    if info.version.parse::<SemVer>()? <= current_version.parse::<SemVer>()? {
        return Err(format!(
            "{} is not newer than {current_version}.",
            info.version
        ));
    }
    if info.size == 0 || info.size > MAX_ARCHIVE_BYTES {
        return Err("The update has an implausible size.".into());
    }
    private_dir(cache)?;
    reclaim_abandoned_work_dirs(cache);
    let guard = WorkDirGuard::create(cache)?;
    let work = guard.work_dir.clone();
    need_space(&work, info.size.saturating_mul(3), "for the download")?;
    let part = work.join(format!("{}.zip.part", info.version));
    transport.download_file(
        &info.asset_url,
        &part,
        info.size,
        info.size.min(MAX_ARCHIVE_BYTES),
        cancel,
        on_progress,
    )?;
    let check_cancel = || {
        if cancel.load(std::sync::atomic::Ordering::Acquire) {
            Err(super::http::CANCELLED.to_owned())
        } else {
            Ok(())
        }
    };
    let map_archive_err = |e: String| {
        if e == oto_sign::extract::CANCELLED {
            super::http::CANCELLED.to_owned()
        } else {
            format!("The update archive is invalid: {e}")
        }
    };
    check_cancel()?;
    let total = oto_sign::inspect_verified_zip_uncompressed_total_cancellable(
        &part,
        &info.sha256,
        info.size,
        cancel,
    )
    .map_err(map_archive_err)?;
    check_cancel()?;
    need_space(&work, total, "to unpack the update")?;
    need_space(target_dir, total, "next to Oto")?;
    let dir = work.join(format!("{}-app", info.version));
    private_dir(&dir)?;
    let app =
        oto_sign::extract_verified_zip_cancellable(&part, &dir, &info.sha256, info.size, cancel)
            .map_err(map_archive_err)?;
    let _ = std::fs::remove_file(&part);
    check_cancel()?;
    check_bundle_info(&app, &info.version, current_version)?;
    check_cancel()?;
    checks.signatures(&app)?;
    check_cancel()?;
    checks.gatekeeper(&app)?;
    check_cancel()?;
    Ok(StagedUpdate {
        info: info.clone(),
        cached_app_path: app,
        cache_dir: work,
        _work_guard: Some(Arc::new(guard)),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::updater::http::tests::FakeTransport;
    use oto_sign::manifest::{REPO, asset_url_prefix, notes_url};
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::sync::atomic::AtomicU32;

    const KEY_A: [u8; 32] = [3; 32];
    const KEY_B: [u8; 32] = [9; 32];

    pub(crate) struct Tmp(pub PathBuf);
    impl Tmp {
        pub fn new(tag: &str) -> Self {
            static N: AtomicU32 = AtomicU32::new(0);
            let d = std::env::temp_dir().join(format!(
                "oto-upd-{tag}-{}-{}",
                std::process::id(),
                N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Self(d.canonicalize().unwrap())
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn trusted(id: &'static str, seed: &[u8; 32]) -> TrustedKey {
        TrustedKey {
            key_id: id,
            pubkey: oto_sign::envelope::public_key(seed),
        }
    }

    pub(crate) fn manifest(version: &str, sha: &str, size: u64, min_os: &str) -> UpdateManifest {
        UpdateManifest {
            schema: 1,
            version: version.into(),
            published_at: "2026-11-27T00:00:00Z".into(),
            notes_url: notes_url(REPO, version),
            driver_version: 1,
            min_driver_version: 0,
            platforms: BTreeMap::from([(
                PLATFORM_MACOS.to_string(),
                oto_sign::PlatformAsset {
                    url: format!("{}Oto-{version}-macos.zip", asset_url_prefix(REPO, version)),
                    sha256: sha.into(),
                    size,
                    min_os: min_os.into(),
                },
            )]),
        }
    }

    fn signed(m: &UpdateManifest, keys: &[(&str, &[u8; 32])]) -> Vec<u8> {
        let payload = serde_json::to_vec(m).unwrap();
        serde_json::to_vec(&oto_sign::sign_envelope(&payload, keys).unwrap()).unwrap()
    }

    fn check(
        t: &FakeTransport,
        current: &str,
        os: &str,
        keys: &[TrustedKey],
    ) -> Result<CheckOutcome, String> {
        check_manifest(
            t,
            current,
            os.parse().unwrap(),
            keys,
            &AtomicBool::new(false),
        )
    }

    fn serve(t: &FakeTransport, body: Vec<u8>) {
        t.serve(&keys::manifest_url(), body);
    }

    #[test]
    fn offers_a_newer_signed_release() {
        let t = FakeTransport::default();
        let m = manifest("0.9.0", &"ab".repeat(32), 1000, "12.0");
        serve(&t, signed(&m, &[("a", &KEY_A)]));
        let out = check(&t, "0.8.9", "14.1", &[trusted("a", &KEY_A)]).unwrap();
        let CheckOutcome::Available(info) = out else {
            panic!("{out:?}")
        };
        assert_eq!(info.version, "0.9.0");
        assert_eq!(
            info.notes_url,
            "https://github.com/deepso7/oto/releases/tag/v0.9.0"
        );
        assert_eq!(info.size, 1000);
        assert_eq!(info.driver_version, 1);
        assert_eq!(*t.requests.lock().unwrap(), [keys::manifest_url()]);
    }

    #[test]
    fn equal_or_older_versions_are_up_to_date() {
        let t = FakeTransport::default();
        serve(
            &t,
            signed(
                &manifest("0.9.0", &"ab".repeat(32), 1, "12.0"),
                &[("a", &KEY_A)],
            ),
        );
        let k = [trusted("a", &KEY_A)];
        assert_eq!(
            check(&t, "0.9.0", "14.0", &k).unwrap(),
            CheckOutcome::UpToDate
        );
        assert_eq!(
            check(&t, "0.10.0", "14.0", &k).unwrap(),
            CheckOutcome::UpToDate
        );
    }

    #[test]
    fn a_newer_macos_requirement_is_reported() {
        let t = FakeTransport::default();
        serve(
            &t,
            signed(
                &manifest("0.9.0", &"ab".repeat(32), 1, "15.2"),
                &[("a", &KEY_A)],
            ),
        );
        assert_eq!(
            check(&t, "0.8.0", "15.1.9", &[trusted("a", &KEY_A)]).unwrap(),
            CheckOutcome::IncompatibleOs {
                version: "0.9.0".into(),
                min_os: "15.2".into()
            }
        );
    }

    #[test]
    fn signatures_unknown_keys_and_rotation() {
        let t = FakeTransport::default();
        let m = manifest("0.9.0", &"ab".repeat(32), 1, "12.0");
        // Signed only by a key this build doesn't know: neutral copy, fixed link.
        serve(&t, signed(&m, &[("new", &KEY_B)]));
        assert_eq!(
            check(&t, "0.8.0", "14.0", &[trusted("old", &KEY_A)]).unwrap(),
            CheckOutcome::UnrecognizedKey {
                releases_url: "https://github.com/deepso7/oto/releases".into()
            }
        );
        // No trusted keys compiled in at all: the same.
        assert!(matches!(
            check(&t, "0.8.0", "14.0", &[]).unwrap(),
            CheckOutcome::UnrecognizedKey { .. }
        ));
        // Dual-signed during rotation: an old build still verifies.
        serve(&t, signed(&m, &[("old", &KEY_A), ("new", &KEY_B)]));
        assert!(matches!(
            check(&t, "0.8.0", "14.0", &[trusted("old", &KEY_A)]).unwrap(),
            CheckOutcome::Available(_)
        ));
        // A trusted key id with a bad signature (tampered payload) is an error.
        let mut env: oto_sign::SignedEnvelope =
            serde_json::from_slice(&signed(&m, &[("old", &KEY_A)])).unwrap();
        let other = manifest("9.9.9", &"ab".repeat(32), 1, "12.0");
        env.payload = base64_encode(&serde_json::to_vec(&other).unwrap());
        serve(&t, serde_json::to_vec(&env).unwrap());
        let e = check(&t, "0.8.0", "14.0", &[trusted("old", &KEY_A)]).unwrap_err();
        assert!(e.contains("signature"), "{e}");
    }

    fn base64_encode(b: &[u8]) -> String {
        // Tiny standard base64 (tests only).
        const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::new();
        for c in b.chunks(3) {
            let n = (c[0] as u32) << 16
                | (*c.get(1).unwrap_or(&0) as u32) << 8
                | *c.get(2).unwrap_or(&0) as u32;
            for i in 0..4 {
                if i <= c.len() {
                    s.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
                } else {
                    s.push('=');
                }
            }
        }
        s
    }

    #[test]
    fn disallowed_urls_and_garbage_are_errors() {
        let t = FakeTransport::default();
        let k = [trusted("a", &KEY_A)];
        let mut m = manifest("0.9.0", &"ab".repeat(32), 1, "12.0");
        m.notes_url = "https://evil.example/notes".into();
        serve(&t, signed(&m, &[("a", &KEY_A)]));
        assert!(check(&t, "0.8.0", "14.0", &k).is_err());

        let mut m = manifest("0.9.0", &"ab".repeat(32), 1, "12.0");
        m.platforms.get_mut(PLATFORM_MACOS).unwrap().url =
            "https://github.com/deepso7/oto/releases/download/v0.8.0/Oto.zip".into();
        serve(&t, signed(&m, &[("a", &KEY_A)]));
        assert!(check(&t, "0.8.0", "14.0", &k).is_err());

        let mut m = manifest("0.9.0", &"ab".repeat(32), 1, "12.0");
        let asset = m.platforms.remove(PLATFORM_MACOS).unwrap();
        m.platforms.insert("linux-aarch64-appimage".into(), asset);
        m.platforms.get_mut("linux-aarch64-appimage").unwrap().url =
            "https://github.com/deepso7/oto/releases/download/v0.9.0/Oto.AppImage".into();
        serve(&t, signed(&m, &[("a", &KEY_A)]));
        let e = check(&t, "0.8.0", "14.0", &k).unwrap_err();
        assert!(e.contains("no macOS"), "{e}");

        serve(&t, b"not json".to_vec());
        assert!(check(&t, "0.8.0", "14.0", &k).is_err());
        serve(&t, vec![b' '; MAX_ENVELOPE_BYTES + 1]);
        assert!(check(&t, "0.8.0", "14.0", &k).is_err());
    }

    // -----------------------------------------------------------------------------------------

    pub(crate) fn info_plist(id: &str, exe: &str, version: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>{id}</string>
<key>CFBundleExecutable</key><string>{exe}</string>
<key>CFBundleShortVersionString</key><string>{version}</string>
</dict></plist>"#
        )
    }

    /// A zip of `Oto.app` with the given Info.plist.
    pub(crate) fn app_zip(plist: &str) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut z = zip_writer(&mut buf);
            z.add_dir("Oto.app/");
            z.add_dir("Oto.app/Contents/");
            z.add_file("Oto.app/Contents/Info.plist", plist.as_bytes(), false);
            z.add_dir("Oto.app/Contents/MacOS/");
            z.add_file("Oto.app/Contents/MacOS/oto-app", b"#!/bin/sh\n", true);
            z.finish();
        }
        buf.into_inner()
    }

    /// Minimal stored-only ZIP writer (tests only; oto-app does not link a zip writer).
    struct ZipW<'a> {
        out: &'a mut std::io::Cursor<Vec<u8>>,
        central: Vec<u8>,
        n: u16,
    }
    fn zip_writer(out: &mut std::io::Cursor<Vec<u8>>) -> ZipW<'_> {
        ZipW {
            out,
            central: Vec::new(),
            n: 0,
        }
    }
    fn crc32(data: &[u8]) -> u32 {
        let mut c = !0u32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
        }
        !c
    }
    impl ZipW<'_> {
        fn entry(&mut self, name: &str, data: &[u8], mode: u32) {
            let off = self.out.get_ref().len() as u32;
            let crc = crc32(data);
            let len = data.len() as u32;
            let mut h = Vec::new();
            h.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            h.extend_from_slice(&20u16.to_le_bytes());
            h.extend_from_slice(&0u16.to_le_bytes());
            h.extend_from_slice(&0u16.to_le_bytes()); // stored
            h.extend_from_slice(&[0; 4]);
            h.extend_from_slice(&crc.to_le_bytes());
            h.extend_from_slice(&len.to_le_bytes());
            h.extend_from_slice(&len.to_le_bytes());
            h.extend_from_slice(&(name.len() as u16).to_le_bytes());
            h.extend_from_slice(&0u16.to_le_bytes());
            h.extend_from_slice(name.as_bytes());
            self.out.write_all(&h).unwrap();
            self.out.write_all(data).unwrap();
            let c = &mut self.central;
            c.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            c.extend_from_slice(&(3u16 << 8 | 20).to_le_bytes()); // made by unix
            c.extend_from_slice(&20u16.to_le_bytes());
            c.extend_from_slice(&0u16.to_le_bytes());
            c.extend_from_slice(&0u16.to_le_bytes());
            c.extend_from_slice(&[0; 4]);
            c.extend_from_slice(&crc.to_le_bytes());
            c.extend_from_slice(&len.to_le_bytes());
            c.extend_from_slice(&len.to_le_bytes());
            c.extend_from_slice(&(name.len() as u16).to_le_bytes());
            c.extend_from_slice(&[0; 8]);
            c.extend_from_slice(&(mode << 16).to_le_bytes());
            c.extend_from_slice(&off.to_le_bytes());
            c.extend_from_slice(name.as_bytes());
            self.n += 1;
        }
        fn add_dir(&mut self, name: &str) {
            self.entry(name, b"", 0o040_755);
        }
        fn add_file(&mut self, name: &str, data: &[u8], exec: bool) {
            self.entry(name, data, if exec { 0o100_755 } else { 0o100_644 });
        }
        fn finish(&mut self) {
            let off = self.out.get_ref().len() as u32;
            let central = std::mem::take(&mut self.central);
            self.out.write_all(&central).unwrap();
            let mut e = Vec::new();
            e.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
            e.extend_from_slice(&[0; 4]);
            e.extend_from_slice(&self.n.to_le_bytes());
            e.extend_from_slice(&self.n.to_le_bytes());
            e.extend_from_slice(&(central.len() as u32).to_le_bytes());
            e.extend_from_slice(&off.to_le_bytes());
            e.extend_from_slice(&0u16.to_le_bytes());
            self.out.write_all(&e).unwrap();
        }
    }

    #[derive(Default)]
    pub(crate) struct FakeChecks {
        pub bad_signature: bool,
        pub bad_gatekeeper: bool,
    }
    impl BundleChecks for FakeChecks {
        fn signatures(&self, _: &Path) -> Result<(), String> {
            if self.bad_signature {
                Err("bad signature".into())
            } else {
                Ok(())
            }
        }
        fn gatekeeper(&self, _: &Path) -> Result<(), String> {
            if self.bad_gatekeeper {
                Err("rejected".into())
            } else {
                Ok(())
            }
        }
    }

    fn info_for(zip: &[u8], version: &str) -> UpdateInfo {
        let sha = {
            use sha2_like::digest;
            digest(zip)
        };
        update_info(&manifest(version, &sha, zip.len() as u64, "12.0")).unwrap()
    }

    /// SHA-256 through oto-sign's file hasher.
    mod sha2_like {
        pub fn digest(b: &[u8]) -> String {
            let p = std::env::temp_dir().join(format!(
                "oto-sha-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            std::fs::write(&p, b).unwrap();
            let (h, _) = oto_sign::sha256_file(&p).unwrap();
            let _ = std::fs::remove_file(&p);
            h
        }
    }

    fn download(
        t: &FakeTransport,
        info: &UpdateInfo,
        checks: &FakeChecks,
        cache: &Path,
    ) -> Result<StagedUpdate, String> {
        let mut seen = 0;
        let r = download_and_verify_in(
            t,
            info,
            "0.8.0",
            cache,
            cache,
            checks,
            &AtomicBool::new(false),
            &mut |n, total| {
                assert!(n <= total);
                seen = n;
            },
        );
        if r.is_ok() {
            assert_eq!(seen, info.size);
        }
        r
    }

    #[test]
    fn downloads_extracts_and_verifies() {
        let tmp = Tmp::new("dl");
        let cache = tmp.0.join("updates");
        let zip = app_zip(&info_plist("dev.oto.app", "oto-app", "0.9.0"));
        let info = info_for(&zip, "0.9.0");
        let t = FakeTransport::default();
        t.serve(&info.asset_url, zip);
        let staged = download(&t, &info, &FakeChecks::default(), &cache).unwrap();
        assert_eq!(staged.info, info);
        assert!(staged.cached_app_path.ends_with("Oto.app"));
        assert!(
            staged
                .cached_app_path
                .join("Contents/MacOS/oto-app")
                .is_file()
        );
        // The archive is gone, the unpacked copy is private.
        assert!(!staged.cache_dir.join("0.9.0.zip.part").exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&staged.cache_dir)
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o700);
            let mode = std::fs::metadata(&cache).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o700);
        }
    }

    fn work_entries(cache: &Path) -> Vec<String> {
        std::fs::read_dir(cache)
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|name| name != WORK_CACHE_LOCK)
            .collect()
    }

    #[test]
    fn a_tampered_archive_or_bundle_is_refused_and_cleaned_up() {
        let tmp = Tmp::new("bad");
        let cache = tmp.0.join("updates");
        let good = app_zip(&info_plist("dev.oto.app", "oto-app", "0.9.0"));
        let info = info_for(&good, "0.9.0");
        let t = FakeTransport::default();

        // Different bytes, same size: SHA-256 mismatch.
        let mut evil = good.clone();
        let i = evil.len() / 2;
        evil[i] ^= 0xff;
        t.serve(&info.asset_url, evil);
        assert!(download(&t, &info, &FakeChecks::default(), &cache).is_err());
        assert!(
            work_entries(&cache).is_empty(),
            "{:?}",
            work_entries(&cache)
        );

        // Longer than announced: cut off by the byte cap.
        let mut long = good.clone();
        long.push(0);
        t.serve(&info.asset_url, long);
        let e = download(&t, &info, &FakeChecks::default(), &cache).unwrap_err();
        assert!(e.contains("larger"), "{e}");

        // Valid archive, wrong identity / version inside.
        for plist in [
            info_plist("dev.evil.app", "oto-app", "0.9.0"),
            info_plist("dev.oto.app", "other", "0.9.0"),
            info_plist("dev.oto.app", "oto-app", "0.9.1"),
        ] {
            let zip = app_zip(&plist);
            let info = info_for(&zip, "0.9.0");
            t.serve(&info.asset_url, zip);
            assert!(download(&t, &info, &FakeChecks::default(), &cache).is_err());
            assert!(
                work_entries(&cache).is_empty(),
                "{:?}",
                work_entries(&cache)
            );
        }

        // Signature or Gatekeeper failure.
        t.serve(&info.asset_url, good.clone());
        let bad_sig = FakeChecks {
            bad_signature: true,
            ..Default::default()
        };
        assert!(download(&t, &info, &bad_sig, &cache).is_err());
        let bad_gk = FakeChecks {
            bad_gatekeeper: true,
            ..Default::default()
        };
        assert!(download(&t, &info, &bad_gk, &cache).is_err());
        assert!(
            work_entries(&cache).is_empty(),
            "{:?}",
            work_entries(&cache)
        );

        // Not newer than what runs.
        let zip = app_zip(&info_plist("dev.oto.app", "oto-app", "0.8.0"));
        let old = info_for(&zip, "0.8.0");
        t.serve(&old.asset_url, zip);
        assert!(download(&t, &old, &FakeChecks::default(), &cache).is_err());
    }

    #[test]
    fn cancelling_stops_the_download() {
        let tmp = Tmp::new("cancel");
        let zip = app_zip(&info_plist("dev.oto.app", "oto-app", "0.9.0"));
        let info = info_for(&zip, "0.9.0");
        let t = FakeTransport::default();
        t.serve(&info.asset_url, zip);
        let e = download_and_verify_in(
            &t,
            &info,
            "0.8.0",
            &tmp.0,
            &tmp.0,
            &FakeChecks::default(),
            &AtomicBool::new(true),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert_eq!(e, crate::updater::http::CANCELLED);
    }

    #[test]
    fn cancelling_during_verification_stops_before_extract_and_checks() {
        struct PanicChecks;
        impl BundleChecks for PanicChecks {
            fn signatures(&self, _: &Path) -> Result<(), String> {
                panic!("signature check must not run after cancel");
            }
            fn gatekeeper(&self, _: &Path) -> Result<(), String> {
                panic!("gatekeeper check must not run after cancel");
            }
        }
        let tmp = Tmp::new("cancel-verify");
        let cache = tmp.0.join("updates");
        let zip = app_zip(&info_plist("dev.oto.app", "oto-app", "0.9.0"));
        let info = info_for(&zip, "0.9.0");
        let t = FakeTransport::default();
        t.serve(&info.asset_url, zip);
        let cancel = AtomicBool::new(false);
        let e = download_and_verify_in(
            &t,
            &info,
            "0.8.0",
            &cache,
            &cache,
            &PanicChecks,
            &cancel,
            &mut |got, total| {
                // Cancel as soon as the download completes and transitions into verification.
                if got >= total {
                    cancel.store(true, std::sync::atomic::Ordering::Release);
                }
            },
        )
        .unwrap_err();
        assert_eq!(e, crate::updater::http::CANCELLED);
        assert!(
            work_entries(&cache).is_empty(),
            "{:?}",
            work_entries(&cache)
        );
    }

    #[test]
    fn a_tampered_archive_fails_digest_before_zip_parsing() {
        let tmp = Tmp::new("digest-first");
        let cache = tmp.0.join("updates");
        let good = app_zip(&info_plist("dev.oto.app", "oto-app", "0.9.0"));
        let info = info_for(&good, "0.9.0");
        let t = FakeTransport::default();
        // Same byte length as the signed archive, but not a ZIP at all: must reject on SHA-256
        // before attempting to parse ZIP headers.
        t.serve(&info.asset_url, vec![0x42; good.len()]);
        let e = download(&t, &info, &FakeChecks::default(), &cache).unwrap_err();
        assert!(e.contains("SHA-256"), "{e}");
        assert!(!e.contains("readable ZIP"), "{e}");
    }

    #[test]
    fn a_failing_download_does_not_delete_a_sibling_download() {
        let tmp = Tmp::new("sibling");
        let cache = tmp.0.join("updates");
        let zip = app_zip(&info_plist("dev.oto.app", "oto-app", "0.9.0"));
        let info = info_for(&zip, "0.9.0");
        let t = FakeTransport::default();
        t.serve(&info.asset_url, zip);

        // First download succeeds into its own private work dir.
        let retry = download(&t, &info, &FakeChecks::default(), &cache).unwrap();
        assert!(retry.cached_app_path.is_dir());

        // A cancelled/failing download into the same cache root only cleans up its own work dir.
        let e = download_and_verify_in(
            &t,
            &info,
            "0.8.0",
            &cache,
            &cache,
            &FakeChecks::default(),
            &AtomicBool::new(true),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert_eq!(e, crate::updater::http::CANCELLED);
        assert!(retry.cached_app_path.is_dir(), "sibling download survived");
        // Dropping the staged update removes its work directory and lock file.
        let work_dir = retry.cache_dir.clone();
        drop(retry);
        assert!(!work_dir.exists());
        assert!(
            work_entries(&cache).is_empty(),
            "{:?}",
            work_entries(&cache)
        );
    }

    #[test]
    fn reclaim_removes_abandoned_crash_dirs_without_touching_live_downloads() {
        let tmp = Tmp::new("reclaim");
        let cache = tmp.0.join("updates");
        let zip = app_zip(&info_plist("dev.oto.app", "oto-app", "0.9.0"));
        let info = info_for(&zip, "0.9.0");
        let t = FakeTransport::default();
        t.serve(&info.asset_url, zip);

        // Simulate a crashed process that left an unlocked work dir behind.
        let crashed = cache.join("work-crashed1");
        std::fs::create_dir_all(&crashed).unwrap();
        std::fs::write(crashed.join("leftover"), b"bytes").unwrap();

        // A live download reclaims the crash leftover while keeping its own locked dir.
        let live = download(&t, &info, &FakeChecks::default(), &cache).unwrap();
        assert!(!crashed.exists(), "unlocked crash leftover was reclaimed");
        assert!(live.cached_app_path.is_dir());

        // Reclaiming again while `live` holds its lock leaves `live` intact.
        reclaim_abandoned_work_dirs(&cache);
        assert!(live.cached_app_path.is_dir());

        // Dropping `live` cleans up its work dir and lock file completely.
        drop(live);
        assert!(
            work_entries(&cache).is_empty(),
            "{:?}",
            work_entries(&cache)
        );
    }

    #[test]
    fn open_before_flock_interleaving_is_serialized_against_reclaim() {
        let tmp = Tmp::new("interleave");
        let cache = tmp.0.join("updates");
        let reclaimer = std::sync::Mutex::new(None);
        let cache_for_thread = cache.clone();
        // Spawn a concurrent reclaimer right after opening `.work-<nonce>.lock` and before
        // taking its per-directory flock; `.work-cache.lock` must hold it off until creation
        // finishes so the live lock inode is never unlinked.
        let guard = WorkDirGuard::create_with_hook(&cache, || {
            let (tx, rx) = std::sync::mpsc::channel();
            let c = cache_for_thread.clone();
            let h = std::thread::spawn(move || {
                // Deterministically observe contention on `.work-cache.lock` before allowing
                // creation to proceed, then run the blocking reclaimer.
                let probe = open_work_lock(&c.join(WORK_CACHE_LOCK)).unwrap();
                assert!(matches!(probe.try_lock(), Err(TryLockError::WouldBlock)));
                drop(probe);
                tx.send(()).unwrap();
                reclaim_abandoned_work_dirs(&c);
            });
            rx.recv().unwrap();
            *reclaimer.lock().unwrap() = Some(h);
        })
        .unwrap();
        reclaimer.into_inner().unwrap().unwrap().join().unwrap();
        assert!(
            guard.work_dir.is_dir(),
            "live work dir survived concurrent sweep"
        );
        assert!(
            guard.lock_path.is_file(),
            "live lock inode was not unlinked"
        );
        // A second sweep afterwards must still see the live flock and leave the work dir intact.
        reclaim_abandoned_work_dirs(&cache);
        assert!(guard.work_dir.is_dir());
        drop(guard);
        assert!(
            work_entries(&cache).is_empty(),
            "{:?}",
            work_entries(&cache)
        );
    }

    #[test]
    fn bundle_info_rules() {
        let tmp = Tmp::new("plist");
        let app = tmp.0.join("Oto.app");
        std::fs::create_dir_all(app.join("Contents")).unwrap();
        let write = |p: &str| std::fs::write(app.join("Contents/Info.plist"), p).unwrap();
        write(&info_plist("dev.oto.app", "oto-app", "1.2.3"));
        check_bundle_info(&app, "1.2.3", "1.2.2").unwrap();
        assert!(check_bundle_info(&app, "1.2.3", "1.2.3").is_err());
        assert!(check_bundle_info(&app, "1.2.4", "1.0.0").is_err());
        assert!(check_bundle_info(&tmp.0.join("missing.app"), "1.2.3", "1.0.0").is_err());
    }

    #[test]
    fn free_space_is_measured() {
        let tmp = Tmp::new("space");
        assert!(free_bytes(&tmp.0).unwrap() > 0);
        assert!(need_space(&tmp.0, u64::MAX, "here").is_err());
        assert!(need_space(&tmp.0, 1, "here").is_ok());
        assert!(free_bytes(Path::new("/nonexistent/oto")).is_none());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn reads_the_os_version() {
        let v = current_os_version().unwrap();
        assert!(v.major >= 11, "{v}");
    }
}
