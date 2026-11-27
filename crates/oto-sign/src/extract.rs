// SPDX-License-Identifier: MIT OR Apache-2.0
//! The one bounded ZIP extractor for update archives (`Oto-<version>-macos.zip`), used by the
//! updater and by `scripts/macos/release.sh` (via `oto-sign extract-verify`), so a release is
//! only published if this exact code can unpack it.
//!
//! The archive must hold nothing but regular files and directories under a single top-level
//! `Oto.app/`. Everything else is refused: `..`, absolute paths, NUL or backslash in names,
//! duplicate paths after case folding (APFS and HFS+ are case-insensitive by default),
//! symlinks, device nodes, FIFOs, sockets, encryption, and compression other than Stored and
//! Deflate. Sizes in the ZIP headers are only used for the up-front estimate; the caps are
//! enforced on the bytes actually written. Modes are rewritten: `0755` for directories and
//! executables, `0644` for everything else, so no setuid, setgid, sticky or world-write bit
//! survives.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};

use sha2::{Digest as _, Sha256};
use zip::{CompressionMethod, ZipArchive};

use crate::manifest::{MAX_ARCHIVE_BYTES, MAX_EXTRACTED_BYTES, MAX_ZIP_ENTRIES};
use crate::{hex, is_sha256_hex};

/// The single top-level directory an update archive holds.
pub const APP_DIR: &str = "Oto.app";

const S_IFMT: u32 = 0o170_000;
const S_IFREG: u32 = 0o100_000;
const S_IFDIR: u32 = 0o040_000;
const S_IFLNK: u32 = 0o120_000;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    pub entries: usize,
    pub extracted_bytes: u64,
}

impl Limits {
    pub(crate) const DEFAULT: Self = Self {
        entries: MAX_ZIP_ENTRIES,
        extracted_bytes: MAX_EXTRACTED_BYTES,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Dir,
    File { executable: bool },
}

#[derive(Debug)]
struct Planned {
    index: usize,
    /// Validated components, the first being [`APP_DIR`].
    components: Vec<String>,
    kind: Kind,
    declared_size: u64,
}

/// Streams `path` through SHA-256. Returns the lowercase hex digest and the byte count.
pub fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    sha256_reader(&mut f, u64::MAX).map_err(|e| format!("{}: {e}", path.display()))
}

/// Hashes at most `cap + 1` bytes of `r`.
fn sha256_reader(r: &mut impl Read, cap: u64) -> io::Result<(String, u64)> {
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut total = 0u64;
    let mut r = r.take(cap.saturating_add(1));
    loop {
        let n = match r.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex(&hasher.finalize()), total))
}

/// Opens `zip_path`, verifies it is a regular file of exactly `expected_size` bytes (1..=150 MiB)
/// with SHA-256 `expected_sha256`, and rewinds it to byte 0 so the verified descriptor can be read
/// directly without re-opening the path.
fn open_verified_archive(
    zip_path: &Path,
    expected_sha256: &str,
    expected_size: u64,
) -> Result<File, String> {
    let expected_sha256 = expected_sha256.to_ascii_lowercase();
    if !is_sha256_hex(&expected_sha256) {
        return Err(format!(
            "expected SHA-256 '{expected_sha256}' is not 64 hex digits"
        ));
    }
    if expected_size == 0 || expected_size > MAX_ARCHIVE_BYTES {
        return Err(format!(
            "expected size {expected_size} is outside 1..={MAX_ARCHIVE_BYTES}"
        ));
    }
    let mut file = File::open(zip_path).map_err(|e| format!("{}: {e}", zip_path.display()))?;
    let meta = file
        .metadata()
        .map_err(|e| format!("{}: {e}", zip_path.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a regular file", zip_path.display()));
    }
    if meta.len() != expected_size {
        return Err(format!(
            "archive is {} bytes, expected {expected_size}",
            meta.len()
        ));
    }
    let (digest, read) =
        sha256_reader(&mut file, expected_size).map_err(|e| format!("reading the archive: {e}"))?;
    if read != expected_size {
        return Err(format!("archive is {read} bytes, expected {expected_size}"));
    }
    if digest != expected_sha256 {
        return Err(format!(
            "archive SHA-256 is {digest}, expected {expected_sha256}"
        ));
    }
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    Ok(file)
}

/// Scans the central directory of the archive at `zip_path` without extracting it: checks
/// the entry count, every path and file type, and returns the declared uncompressed total
/// (≤ 400 MiB), for disk-space checks before extracting.
pub fn inspect_zip_uncompressed_total(zip_path: &Path) -> Result<u64, String> {
    let file = File::open(zip_path).map_err(|e| format!("{}: {e}", zip_path.display()))?;
    let mut archive =
        ZipArchive::new(file).map_err(|e| format!("not a readable ZIP archive: {e}"))?;
    let plan = scan(&mut archive, Limits::DEFAULT)?;
    Ok(plan.iter().map(|p| p.declared_size).sum())
}

/// Verifies `zip_path`'s exact byte size and SHA-256 first, then scans its central directory on
/// the same open file descriptor and returns the declared uncompressed total (≤ 400 MiB).
pub fn inspect_verified_zip_uncompressed_total(
    zip_path: &Path,
    expected_sha256: &str,
    expected_size: u64,
) -> Result<u64, String> {
    let file = open_verified_archive(zip_path, expected_sha256, expected_size)?;
    let mut archive =
        ZipArchive::new(file).map_err(|e| format!("not a readable ZIP archive: {e}"))?;
    let plan = scan(&mut archive, Limits::DEFAULT)?;
    Ok(plan.iter().map(|p| p.declared_size).sum())
}

/// Checks that `zip_path` is exactly `expected_size` bytes (≤ 150 MiB) with SHA-256
/// `expected_sha256`, then extracts it into `dest_dir` (an existing, empty directory) and
/// returns the path of the extracted `Oto.app`.
pub fn extract_verified_zip(
    zip_path: &Path,
    dest_dir: &Path,
    expected_sha256: &str,
    expected_size: u64,
) -> Result<PathBuf, String> {
    extract_verified_zip_with(
        zip_path,
        dest_dir,
        expected_sha256,
        expected_size,
        Limits::DEFAULT,
    )
}

pub(crate) fn extract_verified_zip_with(
    zip_path: &Path,
    dest_dir: &Path,
    expected_sha256: &str,
    expected_size: u64,
    limits: Limits,
) -> Result<PathBuf, String> {
    check_empty_dir(dest_dir)?;
    // One open file for hashing and extracting, so the bytes checked are the bytes unpacked.
    let file = open_verified_archive(zip_path, expected_sha256, expected_size)?;
    let mut archive =
        ZipArchive::new(file).map_err(|e| format!("not a readable ZIP archive: {e}"))?;
    extract_archive(&mut archive, dest_dir, limits)
}

fn check_empty_dir(dir: &Path) -> Result<(), String> {
    let meta = fs::symlink_metadata(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if !meta.file_type().is_dir() {
        return Err(format!("{} is not a directory", dir.display()));
    }
    let mut entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if entries.next().is_some() {
        return Err(format!("{} is not empty", dir.display()));
    }
    Ok(())
}

/// Validates one entry name and splits it into components. Returns whether the name ends in
/// `/` (a directory entry).
fn split_name(raw: &[u8]) -> Result<(Vec<String>, bool), String> {
    let name = std::str::from_utf8(raw)
        .map_err(|_| format!("entry name {:?} is not UTF-8", String::from_utf8_lossy(raw)))?;
    let shown = name.escape_debug();
    if name.is_empty() || name.len() > 1024 {
        return Err(format!("entry name '{shown}' is empty or too long"));
    }
    if name.contains('\0') || name.contains('\\') || name.chars().any(char::is_control) {
        return Err(format!(
            "entry name '{shown}' contains NUL, a backslash or a control character"
        ));
    }
    if name.starts_with('/') {
        return Err(format!("entry '{shown}' is an absolute path"));
    }
    let (body, is_dir) = match name.strip_suffix('/') {
        Some(body) => (body, true),
        None => (name, false),
    };
    let components: Vec<String> = body.split('/').map(str::to_owned).collect();
    for c in &components {
        if c.is_empty() || c == "." || c == ".." {
            return Err(format!(
                "entry '{shown}' has an empty, '.' or '..' path component"
            ));
        }
        if c.len() > 255 {
            return Err(format!("entry '{shown}' has a component over 255 bytes"));
        }
    }
    if components[0] != APP_DIR {
        return Err(format!("entry '{shown}' is outside {APP_DIR}/"));
    }
    Ok((components, is_dir))
}

/// Case-folded path key (APFS/HFS+ default volumes are case-insensitive).
fn fold(components: &[String]) -> String {
    components.join("/").to_lowercase()
}

/// Checks every entry's metadata and returns the extraction plan.
fn scan<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    limits: Limits,
) -> Result<Vec<Planned>, String> {
    let count = archive.len();
    if count == 0 {
        return Err("the archive is empty".into());
    }
    if count > limits.entries {
        return Err(format!(
            "the archive has {count} entries, over the {}-entry limit",
            limits.entries
        ));
    }
    // The reader keys entries by name, so exact duplicate names would otherwise be hidden from
    // the loop below. This also rejects local headers that disagree with the central directory
    // and entries whose data overlaps (another zip-bomb trick).
    archive.validate_header_consistency().map_err(|e| {
        format!("inconsistent ZIP archive (duplicates paths or mismatched headers): {e}")
    })?;

    // Folded path → (exact path, is_dir, listed explicitly).
    let mut seen: HashMap<String, (String, bool, bool)> = HashMap::new();
    let mut plan = Vec::with_capacity(count);
    let mut declared_total = 0u64;
    for index in 0..count {
        let entry = archive
            .by_index_raw(index)
            .map_err(|e| format!("entry {index}: {e}"))?;
        let (components, named_dir) = split_name(entry.name_raw())?;
        let shown = components.join("/");
        if entry.encrypted() {
            return Err(format!("entry '{shown}' is encrypted"));
        }
        if !matches!(
            entry.compression(),
            CompressionMethod::Stored | CompressionMethod::Deflated
        ) {
            return Err(format!(
                "entry '{shown}' uses unsupported compression {:?}",
                entry.compression()
            ));
        }

        let mode = entry.unix_mode().unwrap_or(0);
        let kind = match mode & S_IFMT {
            S_IFLNK => return Err(format!("entry '{shown}' is a symbolic link")),
            0 if named_dir => Kind::Dir,
            0 => Kind::File {
                executable: mode & 0o111 != 0,
            },
            S_IFDIR if named_dir => Kind::Dir,
            S_IFREG if !named_dir => Kind::File {
                executable: mode & 0o111 != 0,
            },
            S_IFDIR | S_IFREG => {
                return Err(format!(
                    "entry '{shown}' has a file type that contradicts its name"
                ));
            }
            other => {
                return Err(format!(
                    "entry '{shown}' is a special file (mode {other:o})"
                ));
            }
        };
        if entry.is_symlink() {
            return Err(format!("entry '{shown}' is a symbolic link"));
        }
        let declared_size = entry.size();
        if kind == Kind::Dir && declared_size != 0 {
            return Err(format!("directory entry '{shown}' has data"));
        }

        // Every ancestor is an implied directory; the entry itself is explicit.
        for depth in 1..components.len() {
            let parent = &components[..depth];
            let key = fold(parent);
            let exact = parent.join("/");
            match seen.get(&key) {
                None => {
                    seen.insert(key, (exact, true, false));
                }
                Some((other, true, _)) if *other == exact => {}
                Some((other, _, _)) => {
                    return Err(format!(
                        "entry '{shown}' collides with '{other}' (paths differ only in case, or a file is used as a directory)"
                    ));
                }
            }
        }
        let key = fold(&components);
        let is_dir = kind == Kind::Dir;
        match seen.get_mut(&key) {
            None => {
                seen.insert(key, (shown.clone(), is_dir, true));
            }
            Some((other, true, explicit @ false)) if is_dir && *other == shown => *explicit = true,
            Some((other, _, _)) => {
                return Err(format!(
                    "entry '{shown}' duplicates '{other}' (paths compared case-insensitively)"
                ));
            }
        }

        declared_total = declared_total.saturating_add(declared_size);
        if declared_total > limits.extracted_bytes {
            return Err(format!(
                "the archive would unpack to over {} bytes",
                limits.extracted_bytes
            ));
        }
        plan.push(Planned {
            index,
            components,
            kind,
            declared_size,
        });
    }
    match seen.get(&APP_DIR.to_lowercase()) {
        Some((exact, true, _)) if exact == APP_DIR => Ok(plan),
        _ => Err(format!("the archive has no {APP_DIR}/ directory")),
    }
}

fn mkdir(path: &Path) -> io::Result<()> {
    match fs::DirBuilder::new().mode(0o755).create(path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            // Only directories this extractor made can exist here (dest started empty, and
            // nothing else is ever created but plain directories and files).
            if !fs::symlink_metadata(path)?.file_type().is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "a file is in the way",
                ));
            }
        }
        Err(e) => return Err(e),
    }
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
}

/// Copies at most `remaining` bytes from `r` into `w`, counting what is actually read rather
/// than trusting any header. Fails as soon as `remaining` would be exceeded.
pub(crate) fn copy_capped(
    r: &mut impl Read,
    w: &mut impl Write,
    remaining: u64,
) -> Result<u64, String> {
    let mut buf = vec![0u8; 64 * 1024];
    let mut written = 0u64;
    loop {
        let n = match r.read(&mut buf) {
            Ok(0) => return Ok(written),
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("reading: {e}")),
        };
        written += n as u64;
        if written > remaining {
            return Err(format!(
                "the archive unpacks to more than the {}-byte limit",
                remaining
            ));
        }
        w.write_all(&buf[..n])
            .map_err(|e| format!("writing: {e}"))?;
    }
}

fn extract_archive<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    dest: &Path,
    limits: Limits,
) -> Result<PathBuf, String> {
    let plan = scan(archive, limits)?;
    let mut total = 0u64;
    for p in &plan {
        let shown = p.components.join("/");
        let mut path = dest.to_path_buf();
        for (i, c) in p.components.iter().enumerate() {
            path.push(c);
            if i + 1 < p.components.len() {
                mkdir(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
        match p.kind {
            Kind::Dir => mkdir(&path).map_err(|e| format!("{}: {e}", path.display()))?,
            Kind::File { executable } => {
                let mode = if executable { 0o755 } else { 0o644 };
                // create_new (O_CREAT | O_EXCL) never follows a symlink or overwrites anything,
                // including a name a normalization-insensitive filesystem considers the same.
                let mut out = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(mode)
                    .open(&path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let mut entry = archive
                    .by_index(p.index)
                    .map_err(|e| format!("entry '{shown}': {e}"))?;
                let budget = (limits.extracted_bytes - total).min(p.declared_size);
                let n = copy_capped(&mut entry, &mut out, budget)
                    .map_err(|e| format!("entry '{shown}': {e}"))?;
                if n != p.declared_size {
                    return Err(format!(
                        "entry '{shown}' unpacked to {n} bytes, but its header says {}",
                        p.declared_size
                    ));
                }
                total += n;
                out.flush()
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                out.set_permissions(fs::Permissions::from_mode(mode))
                    .map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
    }
    let app = dest.join(APP_DIR);
    match fs::symlink_metadata(&app) {
        Ok(m) if m.file_type().is_dir() => Ok(app),
        _ => Err(format!("the archive has no {APP_DIR}/ directory")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static N: AtomicUsize = AtomicUsize::new(0);
            let p = std::env::temp_dir().join(format!(
                "oto-sign-test-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    enum E<'a> {
        Dir(&'a str),
        File(&'a str, &'a [u8], u32),
        Symlink(&'a str, &'a str),
    }

    fn opts(mode: u32) -> SimpleFileOptions {
        SimpleFileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .unix_permissions(mode)
    }

    fn build(entries: &[E]) -> Vec<u8> {
        let mut w = ZipWriter::new(Cursor::new(Vec::new()));
        for e in entries {
            match e {
                E::Dir(n) => w.add_directory(*n, opts(0o755)).unwrap(),
                E::File(n, data, mode) => {
                    w.start_file(*n, opts(*mode)).unwrap();
                    w.write_all(data).unwrap();
                }
                E::Symlink(n, t) => w.add_symlink(*n, *t, opts(0o755)).unwrap(),
            }
        }
        w.finish().unwrap().into_inner()
    }

    fn app() -> Vec<E<'static>> {
        vec![
            E::Dir("Oto.app/"),
            E::Dir("Oto.app/Contents/"),
            E::Dir("Oto.app/Contents/MacOS/"),
            E::File(
                "Oto.app/Contents/MacOS/oto-app",
                b"\xcf\xfa\xed\xfe binary",
                0o4777,
            ),
            E::File("Oto.app/Contents/Info.plist", b"<plist/>", 0o666),
            E::File(
                "Oto.app/Contents/Resources/OtoMicrophone.driver/Contents/MacOS/OtoMicrophone",
                b"drv",
                0o755,
            ),
        ]
    }

    /// Writes `bytes` to a zip file and extracts it with real hash/size checks.
    fn extract_bytes(bytes: &[u8], limits: Limits) -> (TempDir, Result<PathBuf, String>) {
        let t = TempDir::new();
        let zip = t.0.join("u.zip");
        fs::write(&zip, bytes).unwrap();
        let dest = t.0.join("out");
        fs::create_dir(&dest).unwrap();
        let (sha, size) = sha256_file(&zip).unwrap();
        let r = extract_verified_zip_with(&zip, &dest, &sha, size, limits);
        (t, r)
    }

    fn extract(entries: &[E]) -> Result<PathBuf, String> {
        extract_bytes(&build(entries), Limits::DEFAULT).1
    }

    fn mode_of(p: &Path) -> u32 {
        fs::symlink_metadata(p).unwrap().permissions().mode() & 0o7777
    }

    /// Rewrites fields of the central-directory (and local) header for `name`.
    fn patch(bytes: &mut [u8], name: &str, central: impl Fn(&mut [u8]), local: impl Fn(&mut [u8])) {
        let n = name.as_bytes();
        let mut hit = false;
        for i in 0..bytes.len().saturating_sub(46) {
            if bytes[i..i + 4] == [0x50, 0x4b, 0x01, 0x02] {
                let len = u16::from_le_bytes([bytes[i + 28], bytes[i + 29]]) as usize;
                if bytes.get(i + 46..i + 46 + len) == Some(n) {
                    central(&mut bytes[i..i + 46]);
                    hit = true;
                }
            } else if bytes[i..i + 4] == [0x50, 0x4b, 0x03, 0x04] {
                let len = u16::from_le_bytes([bytes[i + 26], bytes[i + 27]]) as usize;
                if bytes.get(i + 30..i + 30 + len) == Some(n) {
                    local(&mut bytes[i..i + 30]);
                }
            }
        }
        assert!(hit, "no central header for {name}");
    }

    fn set_mode(bytes: &mut [u8], name: &str, mode: u32) {
        patch(
            bytes,
            name,
            |h| h[38..42].copy_from_slice(&(mode << 16).to_le_bytes()),
            |_| {},
        );
    }

    #[test]
    fn extracts_a_clean_app_with_clean_modes() {
        let (t, r) = extract_bytes(&build(&app()), Limits::DEFAULT);
        let app = r.unwrap();
        assert_eq!(app, t.0.join("out/Oto.app"));
        let bin = app.join("Contents/MacOS/oto-app");
        assert_eq!(fs::read(&bin).unwrap(), b"\xcf\xfa\xed\xfe binary");
        // setuid and world-write stripped; executables 0755, the rest 0644, dirs 0755.
        assert_eq!(mode_of(&bin), 0o755);
        assert_eq!(mode_of(&app.join("Contents/Info.plist")), 0o644);
        assert_eq!(mode_of(&app), 0o755);
        // Implied directories (no entry of their own) are made too.
        assert_eq!(
            mode_of(&app.join("Contents/Resources/OtoMicrophone.driver")),
            0o755
        );
    }

    #[test]
    fn inspect_sums_declared_sizes() {
        let t = TempDir::new();
        let zip = t.0.join("u.zip");
        fs::write(&zip, build(&app())).unwrap();
        let (sha, size) = sha256_file(&zip).unwrap();
        let expected = (b"\xcf\xfa\xed\xfe binary".len() + b"<plist/>".len() + 3) as u64;
        assert_eq!(inspect_zip_uncompressed_total(&zip).unwrap(), expected);
        assert_eq!(
            inspect_verified_zip_uncompressed_total(&zip, &sha, size).unwrap(),
            expected
        );
        // Non-ZIP bytes of the same length fail digest verification before ZIP parsing.
        fs::write(&zip, vec![0x42; size as usize]).unwrap();
        let err = inspect_verified_zip_uncompressed_total(&zip, &sha, size).unwrap_err();
        assert!(err.contains("SHA-256"), "{err}");
        fs::write(&zip, build(&[E::File("x", b"", 0o644)])).unwrap();
        assert!(inspect_zip_uncompressed_total(&zip).is_err());
    }

    #[test]
    fn checks_size_and_hash_first() {
        let t = TempDir::new();
        let zip = t.0.join("u.zip");
        fs::write(&zip, build(&app())).unwrap();
        let dest = t.0.join("out");
        fs::create_dir(&dest).unwrap();
        let (sha, size) = sha256_file(&zip).unwrap();
        assert!(
            extract_verified_zip(&zip, &dest, &sha, size + 1)
                .unwrap_err()
                .contains("bytes")
        );
        assert!(
            extract_verified_zip(&zip, &dest, &"0".repeat(64), size)
                .unwrap_err()
                .contains("SHA-256")
        );
        assert!(extract_verified_zip(&zip, &dest, "abc", size).is_err());
        assert!(extract_verified_zip(&zip, &dest, &sha, MAX_ARCHIVE_BYTES + 1).is_err());
        assert!(extract_verified_zip(&zip, &dest, &sha, 0).is_err());
        assert_eq!(
            fs::read_dir(&dest).unwrap().count(),
            0,
            "nothing extracted on a bad hash"
        );
        // Uppercase expected hashes are fine.
        extract_verified_zip(&zip, &dest, &sha.to_uppercase(), size).unwrap();
    }

    #[test]
    fn dest_must_be_an_existing_empty_dir() {
        let t = TempDir::new();
        let zip = t.0.join("u.zip");
        fs::write(&zip, build(&app())).unwrap();
        let (sha, size) = sha256_file(&zip).unwrap();
        assert!(extract_verified_zip(&zip, &t.0.join("missing"), &sha, size).is_err());
        let full = t.0.join("full");
        fs::create_dir(&full).unwrap();
        fs::write(full.join("x"), b"").unwrap();
        assert!(
            extract_verified_zip(&zip, &full, &sha, size)
                .unwrap_err()
                .contains("not empty")
        );
        std::os::unix::fs::symlink(t.0.join("missing"), t.0.join("link")).unwrap();
        assert!(extract_verified_zip(&zip, &t.0.join("link"), &sha, size).is_err());
        assert!(extract_verified_zip(&zip, &zip, &sha, size).is_err());
    }

    #[test]
    fn rejects_path_traversal() {
        for bad in [
            "Oto.app/../evil",
            "Oto.app/Contents/../../evil",
            "../Oto.app/x",
            "/Oto.app/x",
            "Oto.app/./x",
            "Oto.app//x",
            "Oto.app/a\\..\\..\\evil",
            "Oto.app/a\0b",
            "Oto.app/a\nb",
        ] {
            let mut e = app();
            e.push(E::File(bad, b"x", 0o644));
            assert!(extract(&e).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn requires_everything_under_oto_app() {
        for stray in [
            "evil",
            "__MACOSX/Oto.app/._x",
            "Other.app/x",
            "oto.app/x",
            "Oto.app.evil/x",
        ] {
            let mut e = app();
            e.push(E::File(stray, b"x", 0o644));
            assert!(extract(&e).is_err(), "{stray:?} should be rejected");
        }
        // No Oto.app at all, or Oto.app as a file.
        assert!(extract(&[E::File("Oto.app", b"x", 0o644)]).is_err());
        assert!(extract(&[E::Dir("Oto.app/")]).is_ok());
    }

    #[test]
    fn rejects_case_folded_duplicates() {
        for dup in [
            E::File("Oto.app/Contents/info.plist", b"x", 0o644),
            E::File("Oto.app/Contents/INFO.PLIST", b"x", 0o644),
            E::File("Oto.app/contents/MacOS/x", b"x", 0o644),
            E::Dir("Oto.app/Contents/macos/"),
            // A file where a directory already is, and the other way round.
            E::File("Oto.app/Contents/MacOS", b"x", 0o644),
            E::File("Oto.app/Contents/Info.plist/x", b"x", 0o644),
            // Unicode case folding, not just ASCII.
            E::File("Oto.app/Contents/É", b"x", 0o644),
        ] {
            let mut e = app();
            e.push(E::File("Oto.app/Contents/é", b"x", 0o644));
            e.push(dup);
            assert!(extract(&e).is_err());
        }
        // Exact duplicates (the ZIP writer refuses them, so rename one afterwards).
        for (placeholder, real) in [
            ("Oto.app/Contents/Info.pliZZ", "Oto.app/Contents/Info.plist"),
            ("Oto.app/Contents/MacOZ/", "Oto.app/Contents/MacOS/"),
        ] {
            let mut e = app();
            e.push(if real.ends_with('/') {
                E::Dir(placeholder)
            } else {
                E::File(placeholder, b"x", 0o644)
            });
            let mut bytes = build(&e);
            let (from, to) = (placeholder.as_bytes(), real.as_bytes());
            for i in 0..bytes.len() - from.len() {
                if &bytes[i..i + from.len()] == from {
                    bytes[i..i + to.len()].copy_from_slice(to);
                }
            }
            assert!(
                extract_bytes(&bytes, Limits::DEFAULT)
                    .1
                    .unwrap_err()
                    .contains("duplicate"),
                "{real}"
            );
        }
    }

    #[test]
    fn explicit_dir_after_implied_one_is_fine() {
        let e = [
            E::File("Oto.app/Contents/MacOS/oto-app", b"x", 0o755),
            E::Dir("Oto.app/Contents/"),
            E::Dir("Oto.app/"),
        ];
        extract(&e).unwrap();
    }

    #[test]
    fn rejects_symlinks() {
        let mut e = app();
        e.push(E::Symlink("Oto.app/Contents/link", "/etc/passwd"));
        assert!(extract(&e).unwrap_err().contains("symbolic link"));
        // A symlink as the top-level directory.
        assert!(extract(&[E::Symlink("Oto.app", "/Applications")]).is_err());
    }

    #[test]
    fn rejects_special_files() {
        for (kind, mode) in [
            ("fifo", 0o010_644),
            ("char", 0o020_644),
            ("block", 0o060_644),
            ("socket", 0o140_644),
            ("link", 0o120_777),
        ] {
            let mut bytes = build(&app());
            set_mode(&mut bytes, "Oto.app/Contents/Info.plist", mode);
            let (_t, r) = extract_bytes(&bytes, Limits::DEFAULT);
            assert!(r.is_err(), "{kind} should be rejected");
        }
        // A directory-typed entry whose name doesn't end in '/' (and vice versa).
        let mut bytes = build(&app());
        set_mode(&mut bytes, "Oto.app/Contents/Info.plist", 0o040_755);
        assert!(extract_bytes(&bytes, Limits::DEFAULT).1.is_err());
        let mut bytes = build(&app());
        set_mode(&mut bytes, "Oto.app/Contents/", 0o100_644);
        assert!(extract_bytes(&bytes, Limits::DEFAULT).1.is_err());
    }

    #[test]
    fn entry_count_limit() {
        let names: Vec<String> = (0..20).map(|i| format!("Oto.app/f{i}")).collect();
        let mut e = vec![E::Dir("Oto.app/")];
        e.extend(names.iter().map(|n| E::File(n, b"x", 0o644)));
        let bytes = build(&e);
        assert!(
            extract_bytes(
                &bytes,
                Limits {
                    entries: 21,
                    ..Limits::DEFAULT
                }
            )
            .1
            .is_ok()
        );
        assert!(
            extract_bytes(
                &bytes,
                Limits {
                    entries: 20,
                    ..Limits::DEFAULT
                }
            )
            .1
            .unwrap_err()
            .contains("entries")
        );
    }

    #[test]
    fn declared_size_limit() {
        let big = vec![0u8; 256 * 1024];
        let bytes = build(&[E::Dir("Oto.app/"), E::File("Oto.app/big", &big, 0o644)]);
        let limits = Limits {
            extracted_bytes: 100 * 1024,
            ..Limits::DEFAULT
        };
        assert!(extract_bytes(&bytes, limits).1.is_err());
    }

    #[test]
    fn zip_bomb_with_lying_headers_is_capped_on_actual_bytes() {
        // 4 MiB of zeros deflates to a few KiB; the headers then claim it unpacks to 16 bytes,
        // so the up-front scan passes and only counting real output can stop it.
        let big = vec![0u8; 4 * 1024 * 1024];
        let mut bytes = build(&[E::Dir("Oto.app/"), E::File("Oto.app/bomb", &big, 0o644)]);
        let lie = 16u32.to_le_bytes();
        patch(
            &mut bytes,
            "Oto.app/bomb",
            |h| h[24..28].copy_from_slice(&lie),
            |h| h[22..26].copy_from_slice(&lie),
        );
        let limits = Limits {
            extracted_bytes: 64 * 1024,
            ..Limits::DEFAULT
        };
        let (t, r) = extract_bytes(&bytes, limits);
        assert!(r.is_err());
        let written = fs::metadata(t.0.join("out/Oto.app/bomb"))
            .map(|m| m.len())
            .unwrap_or(0);
        assert!(written <= 64 * 1024, "wrote {written} bytes past the cap");
    }

    #[test]
    fn copy_capped_counts_real_bytes() {
        let mut sink = Vec::new();
        let mut endless = io::repeat(0).take(10 * 1024 * 1024);
        assert!(copy_capped(&mut endless, &mut sink, 1024 * 1024).is_err());
        assert!(sink.len() <= 1024 * 1024);
        let mut sink = Vec::new();
        assert_eq!(
            copy_capped(&mut io::repeat(1).take(1000), &mut sink, 1000).unwrap(),
            1000
        );
    }

    #[test]
    fn rejects_unsupported_compression_and_non_utf8_names() {
        let mut bytes = build(&app());
        // Method 12 (bzip2), not compiled in.
        patch(
            &mut bytes,
            "Oto.app/Contents/Info.plist",
            |h| h[10..12].copy_from_slice(&12u16.to_le_bytes()),
            |h| h[8..10].copy_from_slice(&12u16.to_le_bytes()),
        );
        assert!(extract_bytes(&bytes, Limits::DEFAULT).1.is_err());
        assert!(split_name(b"Oto.app/\xff").is_err());
    }
}
