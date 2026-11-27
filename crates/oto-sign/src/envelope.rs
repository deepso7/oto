// SPDX-License-Identifier: MIT OR Apache-2.0
//! `update.json`: one bounded JSON file carrying the raw manifest bytes (base64) and one or
//! more Ed25519 signatures over exactly those bytes. Several signatures appear during key
//! rotation, so clients that only know the old key still verify.

use std::fmt;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use ed25519_dalek::{Signature, Signer as _, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::manifest::MAX_ENVELOPE_BYTES;

/// The envelope format this crate reads and writes.
pub const FORMAT: u32 = 1;
/// Most signatures one envelope may carry.
pub const MAX_SIGNATURES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedEnvelope {
    pub format: u32,
    pub signatures: Vec<SignatureEntry>,
    /// Base64 (standard, padded) of the UTF-8 manifest JSON.
    pub payload: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignatureEntry {
    pub key_id: String,
    /// Base64 of the 64-byte Ed25519 signature over the decoded payload bytes.
    pub sig: String,
}

/// A public key compiled into the updater.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustedKey {
    pub key_id: &'static str,
    pub pubkey: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// Larger than [`MAX_ENVELOPE_BYTES`].
    TooLarge,
    /// Not a well-formed format-1 envelope.
    InvalidFormat(String),
    /// No signature is from a trusted key (rotated past this build, or forged). The updater
    /// shows neutral copy and the fixed releases URL.
    UnrecognizedKey,
    /// A signature names a trusted key but doesn't verify (tampered payload or signature).
    InvalidSignature,
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge => write!(
                f,
                "update envelope is larger than {MAX_ENVELOPE_BYTES} bytes"
            ),
            Self::InvalidFormat(e) => write!(f, "malformed update envelope: {e}"),
            Self::UnrecognizedKey => write!(f, "update envelope is not signed by a recognized key"),
            Self::InvalidSignature => write!(f, "update envelope signature is invalid"),
        }
    }
}

impl std::error::Error for VerifyError {}

/// `[A-Za-z0-9._-]{1,64}`.
pub fn valid_key_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// The public key for a 32-byte Ed25519 seed.
pub fn public_key(seed: &[u8; 32]) -> [u8; 32] {
    SigningKey::from_bytes(seed).verifying_key().to_bytes()
}

/// Signs `payload` with every `(key_id, seed)` in `keys`.
pub fn sign_envelope(payload: &[u8], keys: &[(&str, &[u8; 32])]) -> Result<SignedEnvelope, String> {
    if keys.is_empty() {
        return Err("no signing keys".into());
    }
    if keys.len() > MAX_SIGNATURES {
        return Err(format!("at most {MAX_SIGNATURES} signing keys"));
    }
    let mut signatures = Vec::with_capacity(keys.len());
    for (i, (key_id, seed)) in keys.iter().enumerate() {
        if !valid_key_id(key_id) {
            return Err(format!(
                "invalid key id '{key_id}' (want [A-Za-z0-9._-]{{1,64}})"
            ));
        }
        if keys[..i].iter().any(|(other, _)| other == key_id) {
            return Err(format!("key id '{key_id}' appears twice"));
        }
        let sig = SigningKey::from_bytes(seed).sign(payload);
        signatures.push(SignatureEntry {
            key_id: key_id.to_string(),
            sig: B64.encode(sig.to_bytes()),
        });
    }
    let envelope = SignedEnvelope {
        format: FORMAT,
        signatures,
        payload: B64.encode(payload),
    };
    let size = serde_json::to_vec(&envelope)
        .map_err(|e| e.to_string())?
        .len();
    if size > MAX_ENVELOPE_BYTES {
        return Err(format!(
            "signed envelope would be {size} bytes, over the {MAX_ENVELOPE_BYTES}-byte limit"
        ));
    }
    Ok(envelope)
}

/// Verifies an envelope and returns the payload bytes with the `key_id` that verified them.
/// Succeeds if any signature from a trusted key verifies (strict Ed25519 verification).
pub fn verify_envelope(
    envelope: &[u8],
    trusted: &[TrustedKey],
) -> Result<(Vec<u8>, String), VerifyError> {
    if envelope.len() > MAX_ENVELOPE_BYTES {
        return Err(VerifyError::TooLarge);
    }
    let env: SignedEnvelope =
        serde_json::from_slice(envelope).map_err(|e| VerifyError::InvalidFormat(e.to_string()))?;
    if env.format != FORMAT {
        return Err(VerifyError::InvalidFormat(format!(
            "unsupported format {}",
            env.format
        )));
    }
    if env.signatures.is_empty() || env.signatures.len() > MAX_SIGNATURES {
        return Err(VerifyError::InvalidFormat(format!(
            "{} signatures (want 1..={MAX_SIGNATURES})",
            env.signatures.len()
        )));
    }
    if let Some(bad) = env.signatures.iter().find(|s| !valid_key_id(&s.key_id)) {
        return Err(VerifyError::InvalidFormat(format!(
            "invalid key id '{}'",
            bad.key_id.escape_debug()
        )));
    }
    let payload = B64
        .decode(&env.payload)
        .map_err(|e| VerifyError::InvalidFormat(format!("payload: {e}")))?;

    let mut named_trusted_key = false;
    for entry in &env.signatures {
        for key in trusted.iter().filter(|k| k.key_id == entry.key_id) {
            named_trusted_key = true;
            let Ok(sig) = B64.decode(&entry.sig) else {
                continue;
            };
            let Ok(sig) = <[u8; 64]>::try_from(sig.as_slice()) else {
                continue;
            };
            let Ok(vk) = VerifyingKey::from_bytes(&key.pubkey) else {
                continue;
            };
            if vk
                .verify_strict(&payload, &Signature::from_bytes(&sig))
                .is_ok()
            {
                return Ok((payload, entry.key_id.clone()));
            }
        }
    }
    Err(if named_trusted_key {
        VerifyError::InvalidSignature
    } else {
        VerifyError::UnrecognizedKey
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::UpdateManifest;
    use crate::manifest::tests::sample;

    const OLD: [u8; 32] = [7; 32];
    const NEW: [u8; 32] = [42; 32];

    fn trusted(id: &'static str, seed: &[u8; 32]) -> TrustedKey {
        TrustedKey {
            key_id: id,
            pubkey: public_key(seed),
        }
    }

    fn payload() -> Vec<u8> {
        serde_json::to_vec(&sample("0.1.0")).unwrap()
    }

    fn envelope_bytes(env: &SignedEnvelope) -> Vec<u8> {
        serde_json::to_vec(env).unwrap()
    }

    #[test]
    fn round_trip() {
        let env = sign_envelope(&payload(), &[("2026-11", &OLD)]).unwrap();
        let (bytes, key_id) =
            verify_envelope(&envelope_bytes(&env), &[trusted("2026-11", &OLD)]).unwrap();
        assert_eq!(bytes, payload());
        assert_eq!(key_id, "2026-11");
        UpdateManifest::parse(&bytes).unwrap();
    }

    #[test]
    fn dual_key_rotation() {
        let env = sign_envelope(&payload(), &[("2026-11", &OLD), ("2027-05", &NEW)]).unwrap();
        let bytes = envelope_bytes(&env);
        // An old build knows only the old key; a new build only the new one; both verify.
        assert_eq!(
            verify_envelope(&bytes, &[trusted("2026-11", &OLD)])
                .unwrap()
                .1,
            "2026-11"
        );
        assert_eq!(
            verify_envelope(&bytes, &[trusted("2027-05", &NEW)])
                .unwrap()
                .1,
            "2027-05"
        );
        // A bridge build trusting both is fine too.
        verify_envelope(
            &bytes,
            &[trusted("2027-05", &NEW), trusted("2026-11", &OLD)],
        )
        .unwrap();
        // After the old key is retired, releases signed only with it are unrecognized.
        let old_only = envelope_bytes(&sign_envelope(&payload(), &[("2026-11", &OLD)]).unwrap());
        assert_eq!(
            verify_envelope(&old_only, &[trusted("2027-05", &NEW)]),
            Err(VerifyError::UnrecognizedKey)
        );
    }

    #[test]
    fn unrecognized_key() {
        let env = sign_envelope(&payload(), &[("attacker", &NEW)]).unwrap();
        assert_eq!(
            verify_envelope(&envelope_bytes(&env), &[trusted("2026-11", &OLD)]),
            Err(VerifyError::UnrecognizedKey)
        );
        assert_eq!(
            verify_envelope(&envelope_bytes(&env), &[]),
            Err(VerifyError::UnrecognizedKey)
        );
    }

    #[test]
    fn bad_signature_under_trusted_key_id() {
        // Signed with another key but claiming the trusted id.
        let env = sign_envelope(&payload(), &[("2026-11", &NEW)]).unwrap();
        assert_eq!(
            verify_envelope(&envelope_bytes(&env), &[trusted("2026-11", &OLD)]),
            Err(VerifyError::InvalidSignature)
        );
        // Garbage signature bytes.
        let mut env = sign_envelope(&payload(), &[("2026-11", &OLD)]).unwrap();
        env.signatures[0].sig = B64.encode([0u8; 64]);
        assert_eq!(
            verify_envelope(&envelope_bytes(&env), &[trusted("2026-11", &OLD)]),
            Err(VerifyError::InvalidSignature)
        );
        env.signatures[0].sig = "not base64!".into();
        assert_eq!(
            verify_envelope(&envelope_bytes(&env), &[trusted("2026-11", &OLD)]),
            Err(VerifyError::InvalidSignature)
        );
        env.signatures[0].sig = B64.encode([1u8; 63]);
        assert_eq!(
            verify_envelope(&envelope_bytes(&env), &[trusted("2026-11", &OLD)]),
            Err(VerifyError::InvalidSignature)
        );
    }

    #[test]
    fn tampered_payload() {
        let mut env = sign_envelope(&payload(), &[("2026-11", &OLD)]).unwrap();
        let mut m = sample("0.1.0");
        m.platforms.get_mut("macos-universal").unwrap().sha256 = "cd".repeat(32);
        env.payload = B64.encode(serde_json::to_vec(&m).unwrap());
        assert_eq!(
            verify_envelope(&envelope_bytes(&env), &[trusted("2026-11", &OLD)]),
            Err(VerifyError::InvalidSignature)
        );
        // A valid signature from the trusted key alongside a forged one still verifies only
        // the bytes it signed: swapping the payload breaks both.
        let mut env = sign_envelope(&payload(), &[("2026-11", &OLD), ("evil", &NEW)]).unwrap();
        env.payload = B64.encode(b"{}");
        assert_eq!(
            verify_envelope(&envelope_bytes(&env), &[trusted("2026-11", &OLD)]),
            Err(VerifyError::InvalidSignature)
        );
    }

    #[test]
    fn malformed_envelopes() {
        let t = [trusted("2026-11", &OLD)];
        let good = sign_envelope(&payload(), &[("2026-11", &OLD)]).unwrap();
        let fmt_err =
            |b: &[u8]| matches!(verify_envelope(b, &t), Err(VerifyError::InvalidFormat(_)));
        assert!(fmt_err(b"not json"));
        assert!(fmt_err(b"{}"));
        let mut e = good.clone();
        e.format = 2;
        assert!(fmt_err(&envelope_bytes(&e)));
        let mut e = good.clone();
        e.signatures.clear();
        assert!(fmt_err(&envelope_bytes(&e)));
        let mut e = good.clone();
        e.payload = "%%%".into();
        assert!(fmt_err(&envelope_bytes(&e)));
        let mut e = good.clone();
        e.signatures[0].key_id = "../x".into();
        assert!(fmt_err(&envelope_bytes(&e)));
        let mut v: serde_json::Value = serde_json::to_value(&good).unwrap();
        v["extra"] = 1.into();
        assert!(fmt_err(&serde_json::to_vec(&v).unwrap()));
        let mut e = good.clone();
        e.signatures = vec![e.signatures[0].clone(); MAX_SIGNATURES + 1];
        assert!(fmt_err(&envelope_bytes(&e)));
    }

    #[test]
    fn too_large() {
        let mut big = envelope_bytes(&sign_envelope(&payload(), &[("2026-11", &OLD)]).unwrap());
        big.resize(MAX_ENVELOPE_BYTES + 1, b' ');
        assert_eq!(
            verify_envelope(&big, &[trusted("2026-11", &OLD)]),
            Err(VerifyError::TooLarge)
        );
        assert!(sign_envelope(&vec![b'x'; MAX_ENVELOPE_BYTES], &[("2026-11", &OLD)]).is_err());
    }

    #[test]
    fn signing_rejects_bad_key_sets() {
        assert!(sign_envelope(b"x", &[]).is_err());
        assert!(sign_envelope(b"x", &[("a", &OLD), ("a", &NEW)]).is_err());
        assert!(sign_envelope(b"x", &[("bad id", &OLD)]).is_err());
    }
}
