//! Where updates come from and who may sign them.

use oto_sign::TrustedKey;

/// The signed envelope of the latest release. Pinned: see [`manifest_url`].
pub const MANIFEST_URL: &str = oto_sign::manifest::MANIFEST_URL;

/// Shown (as a fixed link) when an update can't be verified.
pub const RELEASES_URL: &str = oto_sign::manifest::RELEASES_URL;

/// The Ed25519 public keys an update envelope must be signed with (any one suffices).
///
/// Generate a key **offline** with `oto-sign keygen --key-id <YYYY-MM>`, store the secret
/// value only in the `release` environment's `OTO_UPDATE_SIGNING_KEY` and an offline backup,
/// and paste the printed `TrustedKey { … }` line below, before the first signed release
/// (packaging/macos/README.md, *Update signing key*). During a rotation both the old and the
/// new key are listed here for at least 6 months / 3 releases.
///
/// While this list is empty every envelope is reported as signed by an unrecognized key, so
/// the app never offers an update it cannot verify; it only points at [`RELEASES_URL`].
pub const TRUSTED_KEYS: &[TrustedKey] = &[
    // TrustedKey { key_id: "2026-11", pubkey: [/* 32 bytes from `oto-sign keygen` */] },
];

/// The manifest URL to fetch. `OTO_UPDATE_URL` overrides it only in debug builds and in builds
/// without a Developer ID team (dev and ad-hoc builds), never in a release build signed for
/// distribution. An override that is not a valid `https://` URL is ignored.
pub fn manifest_url() -> String {
    url_with_override(
        override_allowed(),
        std::env::var("OTO_UPDATE_URL").ok().as_deref(),
    )
}

fn override_allowed() -> bool {
    cfg!(debug_assertions) || crate::driver::TEAM_ID.is_none()
}

fn url_with_override(allowed: bool, value: Option<&str>) -> String {
    match value {
        Some(v) if allowed && oto_sign::manifest::validate_manifest_url(v).is_ok() => v.to_owned(),
        _ => MANIFEST_URL.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pinned_urls_are_the_repository_s() {
        assert_eq!(
            MANIFEST_URL,
            "https://github.com/deepso7/oto/releases/latest/download/update.json"
        );
        assert_eq!(RELEASES_URL, "https://github.com/deepso7/oto/releases");
        oto_sign::manifest::validate_manifest_url(MANIFEST_URL).unwrap();
    }

    #[test]
    fn the_override_applies_only_where_allowed_and_valid() {
        let custom = "https://example.com/oto/update.json";
        assert_eq!(url_with_override(true, Some(custom)), custom);
        assert_eq!(url_with_override(false, Some(custom)), MANIFEST_URL);
        assert_eq!(url_with_override(true, None), MANIFEST_URL);
        for bad in [
            "http://example.com/u.json",
            "https://user@x.com/u",
            "file:///tmp/u",
        ] {
            assert_eq!(url_with_override(true, Some(bad)), MANIFEST_URL, "{bad}");
        }
        // A release build signed with a team never honours it.
        if !cfg!(debug_assertions) && crate::driver::TEAM_ID.is_some() {
            assert!(!override_allowed());
        }
    }

    #[test]
    fn trusted_keys_are_well_formed_and_unique() {
        for (i, k) in TRUSTED_KEYS.iter().enumerate() {
            assert!(oto_sign::envelope::valid_key_id(k.key_id), "{}", k.key_id);
            assert!(
                TRUSTED_KEYS[..i].iter().all(|o| o.key_id != k.key_id),
                "duplicate {}",
                k.key_id
            );
            assert_ne!(k.pubkey, [0; 32]);
        }
    }
}
