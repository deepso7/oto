// SPDX-License-Identifier: MIT OR Apache-2.0
//! The update manifest (the envelope's decoded `payload`), its size limits, and the URL
//! allowlists every manifest and redirect hop must pass.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::is_sha256_hex;
use crate::semver::{OsVersion, SemVer};

/// The GitHub repository releases come from.
pub const REPO: &str = "deepso7/oto";
/// Where the updater fetches the signed envelope.
pub const MANIFEST_URL: &str =
    "https://github.com/deepso7/oto/releases/latest/download/update.json";
/// The fixed page shown when an update can't be verified.
pub const RELEASES_URL: &str = "https://github.com/deepso7/oto/releases";
/// The manifest schema this crate reads and writes.
pub const SCHEMA: u32 = 1;
/// The macOS platform key.
pub const PLATFORM_MACOS: &str = "macos-universal";

/// Largest `update.json` accepted.
pub const MAX_ENVELOPE_BYTES: usize = 64 * 1024;
/// Largest update archive accepted (compressed).
pub const MAX_ARCHIVE_BYTES: u64 = 150 * 1024 * 1024;
/// Most bytes the extractor will write for one archive.
pub const MAX_EXTRACTED_BYTES: u64 = 400 * 1024 * 1024;
/// Most entries (files and directories) one archive may hold.
pub const MAX_ZIP_ENTRIES: usize = 5_000;
/// Most redirect hops the updater follows.
pub const MAX_REDIRECTS: usize = 5;

/// The decoded manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateManifest {
    pub schema: u32,
    pub version: String,
    /// UTC, `YYYY-MM-DDTHH:MM:SSZ`.
    pub published_at: String,
    pub notes_url: String,
    /// The HAL driver revision bundled in this release (`driver/macos/DRIVER_VERSION`).
    pub driver_version: u32,
    /// The oldest loaded driver revision this release's engine accepts.
    pub min_driver_version: u32,
    pub platforms: BTreeMap<String, PlatformAsset>,
}

/// One downloadable archive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlatformAsset {
    pub url: String,
    pub sha256: String,
    pub size: u64,
    pub min_os: String,
}

impl UpdateManifest {
    /// Parses and fully validates manifest bytes against the pinned repository.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let m: Self =
            serde_json::from_slice(bytes).map_err(|e| format!("invalid manifest JSON: {e}"))?;
        m.validate()?;
        Ok(m)
    }

    /// The release version.
    pub fn semver(&self) -> Result<SemVer, String> {
        self.version.parse()
    }

    /// Checks every field against the pinned repository ([`REPO`]).
    pub fn validate(&self) -> Result<(), String> {
        self.validate_for_repo(REPO)
    }

    /// Checks every field, with URLs pinned to `repo` (`owner/name`).
    pub fn validate_for_repo(&self, repo: &str) -> Result<(), String> {
        if self.schema != SCHEMA {
            return Err(format!(
                "unsupported manifest schema {} (expected {SCHEMA})",
                self.schema
            ));
        }
        self.semver()?;
        validate_timestamp(&self.published_at)?;
        validate_notes_url_for(repo, &self.notes_url, &self.version)?;
        if self.min_driver_version > self.driver_version {
            return Err(format!(
                "min_driver_version {} is above driver_version {}",
                self.min_driver_version, self.driver_version
            ));
        }
        if self.platforms.is_empty() {
            return Err("the manifest lists no platforms".into());
        }
        for (key, asset) in &self.platforms {
            if key.is_empty()
                || key.len() > 64
                || !key.bytes().all(|b| {
                    b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_')
                })
            {
                return Err(format!("invalid platform key '{key}'"));
            }
            asset
                .validate_for_repo(repo, &self.version)
                .map_err(|e| format!("platform {key}: {e}"))?;
        }
        Ok(())
    }
}

impl PlatformAsset {
    fn validate_for_repo(&self, repo: &str, version: &str) -> Result<(), String> {
        validate_asset_url_for(repo, &self.url, version)?;
        if !is_sha256_hex(&self.sha256) {
            return Err(format!(
                "sha256 '{}' is not 64 lowercase hex digits",
                self.sha256
            ));
        }
        if self.size == 0 || self.size > MAX_ARCHIVE_BYTES {
            return Err(format!(
                "size {} is outside 1..={MAX_ARCHIVE_BYTES}",
                self.size
            ));
        }
        self.min_os.parse::<OsVersion>()?;
        Ok(())
    }

    /// True if `os` meets this asset's `min_os`.
    pub fn supports_os(&self, os: OsVersion) -> Result<bool, String> {
        Ok(os >= self.min_os.parse::<OsVersion>()?)
    }
}

/// `YYYY-MM-DDTHH:MM:SSZ` with plausible field ranges.
fn validate_timestamp(s: &str) -> Result<(), String> {
    let b = s.as_bytes();
    let shape = b.len() == 20
        && b.iter().enumerate().all(|(i, &c)| match i {
            4 | 7 => c == b'-',
            10 => c == b'T',
            13 | 16 => c == b':',
            19 => c == b'Z',
            _ => c.is_ascii_digit(),
        });
    let num = |r: std::ops::Range<usize>| s[r].parse::<u32>().unwrap_or(u32::MAX);
    if !shape
        || !(1..=12).contains(&num(5..7))
        || !(1..=31).contains(&num(8..10))
        || num(11..13) > 23
        || num(14..16) > 59
        || num(17..19) > 60
    {
        return Err(format!(
            "published_at '{s}' is not a UTC timestamp YYYY-MM-DDTHH:MM:SSZ"
        ));
    }
    Ok(())
}

/// Validates one URL the updater may fetch, initial or after a redirect: `https://` only,
/// port 443 (or none), no userinfo, a plain DNS host, and no whitespace, control characters
/// or backslashes anywhere.
pub fn validate_redirect_url(url: &str) -> Result<(), String> {
    if url.len() > 4096 {
        return Err("URL is too long".into());
    }
    if url
        .chars()
        .any(|c| c.is_control() || c.is_whitespace() || c == '\\')
    {
        return Err(format!(
            "URL '{}' contains whitespace, control characters or backslashes",
            url.escape_debug()
        ));
    }
    let rest = url
        .strip_prefix("https://")
        .ok_or_else(|| format!("URL '{url}' is not https://"))?;
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..end];
    if authority.contains('@') {
        return Err(format!("URL '{url}' carries credentials"));
    }
    let host = match authority.rsplit_once(':') {
        Some((host, "443")) => host,
        Some(_) => return Err(format!("URL '{url}' uses a port other than 443")),
        None => authority,
    };
    let host_ok = !host.is_empty()
        && host.len() <= 253
        && host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        });
    if !host_ok {
        return Err(format!("URL '{url}' has an invalid host"));
    }
    Ok(())
}

/// Validates the URL the envelope is fetched from (the pinned [`MANIFEST_URL`], or a debug
/// override): the redirect rules, plus a real path.
pub fn validate_manifest_url(url: &str) -> Result<(), String> {
    validate_redirect_url(url)?;
    let rest = &url["https://".len()..];
    match rest.find('/') {
        Some(i) if rest.len() > i + 1 => Ok(()),
        _ => Err(format!("manifest URL '{url}' has no path")),
    }
}

/// The prefix every asset of `version` must start with.
pub fn asset_url_prefix(repo: &str, version: &str) -> String {
    format!("https://github.com/{repo}/releases/download/v{version}/")
}

/// The only release-notes URL accepted for `version`.
pub fn notes_url(repo: &str, version: &str) -> String {
    format!("https://github.com/{repo}/releases/tag/v{version}")
}

/// An archive URL: `https://github.com/deepso7/oto/releases/download/v<version>/<file>`, with
/// a single plain file name (no further path, query or fragment).
pub fn validate_asset_url(url: &str, version: &str) -> Result<(), String> {
    validate_asset_url_for(REPO, url, version)
}

pub fn validate_asset_url_for(repo: &str, url: &str, version: &str) -> Result<(), String> {
    validate_redirect_url(url)?;
    let prefix = asset_url_prefix(repo, version);
    let file = url
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("asset URL '{url}' is not under {prefix}"))?;
    let file_ok = !file.is_empty()
        && !file.starts_with('.')
        && file
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b'+'));
    if !file_ok {
        return Err(format!(
            "asset URL '{url}' does not end in a plain file name"
        ));
    }
    Ok(())
}

/// The release notes link must be exactly the release page for `version`.
pub fn validate_notes_url(url: &str, version: &str) -> Result<(), String> {
    validate_notes_url_for(REPO, url, version)
}

fn validate_notes_url_for(repo: &str, url: &str, version: &str) -> Result<(), String> {
    let expected = notes_url(repo, version);
    if url == expected {
        Ok(())
    } else {
        Err(format!("notes_url '{url}' is not {expected}"))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn sample(version: &str) -> UpdateManifest {
        UpdateManifest {
            schema: 1,
            version: version.into(),
            published_at: "2026-11-27T00:00:00Z".into(),
            notes_url: notes_url(REPO, version),
            driver_version: 1,
            min_driver_version: 0,
            platforms: BTreeMap::from([(
                PLATFORM_MACOS.to_string(),
                PlatformAsset {
                    url: format!("{}Oto-{version}-macos.zip", asset_url_prefix(REPO, version)),
                    sha256: "ab".repeat(32),
                    size: 45_123_456,
                    min_os: "12.0".into(),
                },
            )]),
        }
    }

    #[test]
    fn sample_validates_and_round_trips() {
        let m = sample("0.1.0");
        m.validate().unwrap();
        let bytes = serde_json::to_vec(&m).unwrap();
        assert_eq!(UpdateManifest::parse(&bytes).unwrap(), m);
    }

    #[test]
    fn rejects_bad_manifests() {
        let check = |f: &dyn Fn(&mut UpdateManifest)| {
            let mut m = sample("0.1.0");
            f(&mut m);
            m.validate()
        };
        let check_asset = |f: &dyn Fn(&mut PlatformAsset)| {
            check(&|m| f(m.platforms.get_mut(PLATFORM_MACOS).unwrap()))
        };
        assert!(check(&|m| m.schema = 2).is_err());
        assert!(check(&|m| m.version = "0.1".into()).is_err());
        assert!(check(&|m| m.published_at = "2026-11-27".into()).is_err());
        assert!(check(&|m| m.published_at = "2026-13-27T00:00:00Z".into()).is_err());
        assert!(check(&|m| m.notes_url = notes_url(REPO, "0.0.9")).is_err());
        assert!(check(&|m| m.min_driver_version = 2).is_err());
        assert!(check(&|m| m.platforms.clear()).is_err());
        assert!(check_asset(&|a| a.sha256 = "AB".repeat(32)).is_err());
        assert!(check_asset(&|a| a.sha256 = "ab".repeat(31)).is_err());
        assert!(check_asset(&|a| a.size = 0).is_err());
        assert!(check_asset(&|a| a.size = MAX_ARCHIVE_BYTES + 1).is_err());
        assert!(check_asset(&|a| a.min_os = "twelve".into()).is_err());
        assert!(
            check_asset(&|a| {
                a.url =
                    "https://github.com/deepso7/oto/releases/download/v0.0.9/Oto-0.1.0-macos.zip"
                        .into()
            })
            .is_err()
        );
        assert!(
            check(&|m| {
                let a = m.platforms.remove(PLATFORM_MACOS).unwrap();
                m.platforms.insert("macOS Universal".into(), a);
            })
            .is_err()
        );
        // Wrong repository.
        assert!(sample("0.1.0").validate_for_repo("someone/else").is_err());
    }

    #[test]
    fn min_os() {
        let a = &sample("0.1.0").platforms[PLATFORM_MACOS];
        assert!(a.supports_os("12.0".parse().unwrap()).unwrap());
        assert!(a.supports_os("27.1.2".parse().unwrap()).unwrap());
        assert!(!a.supports_os("11.7.10".parse().unwrap()).unwrap());
    }

    #[test]
    fn redirect_urls() {
        for ok in [
            MANIFEST_URL,
            "https://release-assets.githubusercontent.com/github-production-release-asset/1?sp=r&sig=x%2By",
            "https://objects.githubusercontent.com:443/a/b",
            "https://example.com",
            "https://example.com?x=1",
        ] {
            validate_redirect_url(ok).unwrap_or_else(|e| panic!("{ok}: {e}"));
        }
        for bad in [
            "http://github.com/deepso7/oto",
            "HTTPS://github.com/x",
            "ftp://github.com/x",
            "https://user:pass@github.com/x",
            "https://user@github.com/x",
            "https://github.com:8443/x",
            "https://github.com:/x",
            "https://github.com:443:443/x",
            "https:///x",
            "https://[::1]/x",
            "https://git hub.com/x",
            "https://github.com/x y",
            "https://github.com/x\n",
            "https://github.com\\@evil.com/x",
            "https://-github.com/x",
            "https://github..com/x",
            "//github.com/x",
            "github.com/x",
        ] {
            assert!(
                validate_redirect_url(bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn manifest_urls() {
        validate_manifest_url(MANIFEST_URL).unwrap();
        assert!(validate_manifest_url("https://github.com").is_err());
        assert!(validate_manifest_url("https://github.com/").is_err());
        assert!(
            validate_manifest_url(
                "http://github.com/deepso7/oto/releases/latest/download/update.json"
            )
            .is_err()
        );
    }

    #[test]
    fn asset_urls() {
        let good = "https://github.com/deepso7/oto/releases/download/v0.1.0/Oto-0.1.0-macos.zip";
        validate_asset_url(good, "0.1.0").unwrap();
        for bad in [
            "https://github.com/deepso7/oto/releases/download/v0.1.1/Oto-0.1.0-macos.zip",
            "https://github.com/deepso7/oto/releases/download/v0.1.0/",
            "https://github.com/deepso7/oto/releases/download/v0.1.0/../v0.0.1/Oto.zip",
            "https://github.com/deepso7/oto/releases/download/v0.1.0/a/b.zip",
            "https://github.com/deepso7/oto/releases/download/v0.1.0/Oto.zip?x=1",
            "https://github.com/deepso7/oto/releases/download/v0.1.0/Oto.zip#x",
            "https://github.com/deepso7/oto/releases/download/v0.1.0/.hidden",
            "https://github.com/evil/oto/releases/download/v0.1.0/Oto-0.1.0-macos.zip",
            "https://github.com.evil.com/deepso7/oto/releases/download/v0.1.0/Oto.zip",
            "https://evil.com/https://github.com/deepso7/oto/releases/download/v0.1.0/Oto.zip",
            "http://github.com/deepso7/oto/releases/download/v0.1.0/Oto-0.1.0-macos.zip",
            "https://github.com:444/deepso7/oto/releases/download/v0.1.0/Oto-0.1.0-macos.zip",
        ] {
            assert!(
                validate_asset_url(bad, "0.1.0").is_err(),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn notes_urls() {
        validate_notes_url(
            "https://github.com/deepso7/oto/releases/tag/v0.1.0",
            "0.1.0",
        )
        .unwrap();
        for bad in [
            "https://github.com/deepso7/oto/releases/tag/v0.1.1",
            "https://github.com/deepso7/oto/releases/tag/v0.1.0/",
            "https://github.com/deepso7/oto/releases/tag/v0.1.0?x",
            "http://github.com/deepso7/oto/releases/tag/v0.1.0",
            "https://evil.com/deepso7/oto/releases/tag/v0.1.0",
            "javascript:alert(1)",
        ] {
            assert!(
                validate_notes_url(bad, "0.1.0").is_err(),
                "{bad:?} should be rejected"
            );
        }
    }
}
