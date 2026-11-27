// SPDX-License-Identifier: MIT OR Apache-2.0
//! Oto's update trust format and archive extractor, shared by `oto-app` (the updater) and the
//! release pipeline (`oto-sign` CLI), so both sides use exactly the same code.
//!
//! - [`envelope`]: `update.json`, a bounded JSON envelope carrying the raw manifest bytes and
//!   one or more Ed25519 signatures over them (several during key rotation).
//! - [`manifest`]: the decoded manifest, its size limits and strict URL allowlists.
//! - [`extract`]: the single bounded ZIP extractor for `Oto-<version>-macos.zip`.
//! - [`semver`]: strict `major.minor.patch` and OS version comparison.
//!
//! See docs/plans/0003-app-updates.md, Decision 2.

pub mod envelope;
pub mod extract;
pub mod manifest;
pub mod semver;

pub use envelope::{
    SignatureEntry, SignedEnvelope, TrustedKey, VerifyError, sign_envelope, verify_envelope,
};
pub use extract::{
    extract_verified_zip, inspect_verified_zip_uncompressed_total, inspect_zip_uncompressed_total,
    sha256_file,
};
pub use manifest::{PlatformAsset, UpdateManifest};
pub use semver::{OsVersion, SemVer};

/// Lowercase hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 0xf) as usize] as char);
    }
    s
}

/// True for exactly 64 lowercase hex digits (a SHA-256 as written in manifests).
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
