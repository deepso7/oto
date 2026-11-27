// SPDX-License-Identifier: MIT OR Apache-2.0
//! `oto-sign`: release-side tool for Oto's signed update envelopes. See
//! packaging/macos/README.md ("Update signing") and docs/plans/0003-app-updates.md.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use lexopt::prelude::*;
use oto_sign::envelope::{self, TrustedKey};
use oto_sign::manifest::{self, MAX_ARCHIVE_BYTES, PLATFORM_MACOS, PlatformAsset, UpdateManifest};
use oto_sign::{SemVer, extract_verified_zip, sha256_file, sign_envelope, verify_envelope};

const USAGE: &str = "\
usage: oto-sign <command> [options]

commands:
  keygen [--key-id <id>]
      New Ed25519 keypair. Prints the key id, the OTO_UPDATE_SIGNING_KEY secret value and
      the public key (base64 and as a Rust TrustedKey literal).
  make-payload --version <X.Y.Z> --zip <path> --driver-version <n> --min-driver-version <n>
               --min-os <X.Y> [--repo <owner/name>] [--platform <key>] [--published-at <ts>]
               --out <path>
      Writes the unsigned manifest for a release archive.
  sign --payload <path> --out <path> [--key-env <VAR>]
      Validates the manifest and signs it with every key in $VAR (default
      OTO_UPDATE_SIGNING_KEY): comma-separated <key_id>:<base64 seed> entries.
  verify --envelope <path> (--pubkey <key_id>:<base64 public key>... | --key-env <VAR>)
         [--repo <owner/name>]
      Verifies an envelope and its manifest; prints the decoded manifest to stdout. With
      --key-env, trusts the public halves of the signing keys in $VAR (as `sign` reads them).
  extract-verify --zip <path> --dest <dir> --expected-sha256 <hex> --expected-size <bytes>
      Checks and unpacks an update archive exactly as the updater does; prints the Oto.app path.
  check-version-gt --candidate <X.Y.Z> --current <X.Y.Z>
      Exits 0 if candidate > current, 1 if not, 2 on bad input (a leading 'v' is allowed).
";

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("oto-sign: {e}");
            ExitCode::from(2)
        }
    }
}

type Opts = BTreeMap<String, Vec<String>>;

/// Parses `--name value` pairs (repeatable) after the command.
fn parse_opts(parser: &mut lexopt::Parser, allowed: &[&str]) -> Result<Opts, String> {
    let mut opts = Opts::new();
    while let Some(arg) = parser.next().map_err(|e| e.to_string())? {
        match arg {
            Long(name) if allowed.contains(&name) => {
                let name = name.to_string();
                let value: OsString = parser.value().map_err(|e| e.to_string())?;
                let value = value
                    .into_string()
                    .map_err(|_| format!("--{name}: not UTF-8"))?;
                opts.entry(name).or_default().push(value);
            }
            Short('h') | Long("help") => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(other.unexpected().to_string()),
        }
    }
    Ok(opts)
}

fn one<'a>(opts: &'a Opts, name: &str) -> Result<&'a str, String> {
    match opts.get(name).map(Vec::as_slice) {
        Some([v]) => Ok(v),
        Some(_) => Err(format!("--{name} given more than once")),
        None => Err(format!("--{name} is required")),
    }
}

fn opt<'a>(opts: &'a Opts, name: &str) -> Result<Option<&'a str>, String> {
    if opts.contains_key(name) {
        one(opts, name).map(Some)
    } else {
        Ok(None)
    }
}

fn num<T: std::str::FromStr>(opts: &Opts, name: &str) -> Result<T, String> {
    let v = one(opts, name)?;
    v.parse()
        .map_err(|_| format!("--{name}: '{v}' is not a non-negative integer"))
}

fn run() -> Result<ExitCode, String> {
    let mut parser = lexopt::Parser::from_env();
    let cmd = match parser.next().map_err(|e| e.to_string())? {
        Some(Value(v)) => v.into_string().map_err(|_| "bad command".to_string())?,
        Some(Short('h') | Long("help")) | None => {
            print!("{USAGE}");
            return Ok(ExitCode::SUCCESS);
        }
        Some(other) => return Err(other.unexpected().to_string()),
    };
    match cmd.as_str() {
        "keygen" => keygen(&parse_opts(&mut parser, &["key-id"])?),
        "make-payload" => make_payload(&parse_opts(
            &mut parser,
            &[
                "version",
                "zip",
                "driver-version",
                "min-driver-version",
                "min-os",
                "repo",
                "platform",
                "published-at",
                "out",
            ],
        )?),
        "sign" => sign(&parse_opts(&mut parser, &["payload", "out", "key-env"])?),
        "verify" => verify(&parse_opts(
            &mut parser,
            &["envelope", "pubkey", "key-env", "repo"],
        )?),
        "extract-verify" => extract(&parse_opts(
            &mut parser,
            &["zip", "dest", "expected-sha256", "expected-size"],
        )?),
        "check-version-gt" => {
            check_version_gt(&parse_opts(&mut parser, &["candidate", "current"])?)
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    }
}

/// UTC (year, month, day, hour, minute, second) for a Unix time.
fn utc(secs: u64) -> (i64, u32, u32, u64, u64, u64) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d, rem / 3600, rem % 3600 / 60, rem % 60)
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn keygen(opts: &Opts) -> Result<ExitCode, String> {
    let key_id = match opt(opts, "key-id")? {
        Some(id) => id.to_string(),
        None => {
            let (y, m, ..) = utc(now());
            format!("{y:04}-{m:02}")
        }
    };
    if !envelope::valid_key_id(&key_id) {
        return Err(format!(
            "invalid key id '{key_id}' (want [A-Za-z0-9._-]{{1,64}})"
        ));
    }
    let mut seed = [0u8; 32];
    getrandom::fill(&mut seed).map_err(|e| format!("no system randomness: {e}"))?;
    let public = envelope::public_key(&seed);
    let literal = public
        .iter()
        .map(|b| format!("0x{b:02x}"))
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!(
        "Keep the secret value offline and in the `release` environment only; never commit it."
    );
    println!("key_id: {key_id}");
    println!(
        "OTO_UPDATE_SIGNING_KEY (secret): {key_id}:{}",
        B64.encode(seed)
    );
    println!("public key (base64): {}", B64.encode(public));
    println!("verify argument: --pubkey {key_id}:{}", B64.encode(public));
    println!("Rust (crates/oto-app/src/updater/keys.rs):");
    println!("    TrustedKey {{ key_id: \"{key_id}\", pubkey: [{literal}] }},");
    Ok(ExitCode::SUCCESS)
}

fn make_payload(opts: &Opts) -> Result<ExitCode, String> {
    let version = one(opts, "version")?;
    version.parse::<SemVer>()?;
    let zip = PathBuf::from(one(opts, "zip")?);
    let repo = opt(opts, "repo")?.unwrap_or(manifest::REPO);
    let platform = opt(opts, "platform")?.unwrap_or(PLATFORM_MACOS);
    let file_name = zip
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("{}: no file name", zip.display()))?;
    let (sha256, size) = sha256_file(&zip)?;
    if size > MAX_ARCHIVE_BYTES {
        return Err(format!(
            "{} is {size} bytes, over the {MAX_ARCHIVE_BYTES}-byte limit",
            zip.display()
        ));
    }
    let published_at = match opt(opts, "published-at")? {
        Some(ts) => ts.to_string(),
        None => {
            let (y, mo, d, h, mi, s) = utc(now());
            format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
        }
    };
    let m = UpdateManifest {
        schema: manifest::SCHEMA,
        version: version.to_string(),
        published_at,
        notes_url: manifest::notes_url(repo, version),
        driver_version: num(opts, "driver-version")?,
        min_driver_version: num(opts, "min-driver-version")?,
        platforms: BTreeMap::from([(
            platform.to_string(),
            PlatformAsset {
                url: format!("{}{file_name}", manifest::asset_url_prefix(repo, version)),
                sha256,
                size,
                min_os: one(opts, "min-os")?.to_string(),
            },
        )]),
    };
    m.validate_for_repo(repo)?;
    let mut json = serde_json::to_string_pretty(&m).map_err(|e| e.to_string())?;
    json.push('\n');
    let out = one(opts, "out")?;
    fs::write(out, &json).map_err(|e| format!("{out}: {e}"))?;
    eprintln!("wrote {out}");
    print!("{json}");
    Ok(ExitCode::SUCCESS)
}

/// Parses `key_id:base64seed[,key_id:base64seed...]`.
fn parse_signing_keys(value: &str) -> Result<Vec<(String, [u8; 32])>, String> {
    let mut keys = Vec::new();
    for (i, item) in value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .enumerate()
    {
        let (id, seed) = item
            .split_once(':')
            .ok_or_else(|| format!("signing key #{} is not <key_id>:<base64 seed>", i + 1))?;
        let seed = B64
            .decode(seed.trim())
            .map_err(|_| format!("signing key '{id}': seed is not base64"))?;
        let seed: [u8; 32] = seed
            .try_into()
            .map_err(|_| format!("signing key '{id}': seed is not 32 bytes"))?;
        keys.push((id.trim().to_string(), seed));
    }
    if keys.is_empty() {
        return Err("no signing keys given".into());
    }
    Ok(keys)
}

fn parse_pubkey(value: &str) -> Result<TrustedKey, String> {
    let (id, key) = value
        .split_once(':')
        .ok_or_else(|| format!("--pubkey '{value}' is not <key_id>:<base64>"))?;
    let key = B64
        .decode(key)
        .map_err(|_| format!("--pubkey '{id}': not base64"))?;
    let pubkey: [u8; 32] = key
        .try_into()
        .map_err(|_| format!("--pubkey '{id}': not 32 bytes"))?;
    // The CLI runs once; leaking the id gives TrustedKey its 'static str.
    Ok(TrustedKey {
        key_id: Box::leak(id.to_string().into_boxed_str()),
        pubkey,
    })
}

fn sign(opts: &Opts) -> Result<ExitCode, String> {
    let payload_path = one(opts, "payload")?;
    let out = one(opts, "out")?;
    let var = opt(opts, "key-env")?.unwrap_or("OTO_UPDATE_SIGNING_KEY");
    let payload = fs::read(payload_path).map_err(|e| format!("{payload_path}: {e}"))?;
    // Refuse to sign anything the updater would reject.
    let m = UpdateManifest::parse(&payload)?;
    let secret = std::env::var(var).map_err(|_| format!("${var} is not set"))?;
    let keys = parse_signing_keys(&secret)?;
    let refs: Vec<(&str, &[u8; 32])> = keys.iter().map(|(id, seed)| (id.as_str(), seed)).collect();
    let env = sign_envelope(&payload, &refs)?;
    let mut json = serde_json::to_string_pretty(&env).map_err(|e| e.to_string())?;
    json.push('\n');
    // Self-check with the derived public keys before writing anything.
    for (id, seed) in &keys {
        let trusted = [TrustedKey {
            key_id: Box::leak(id.clone().into_boxed_str()),
            pubkey: envelope::public_key(seed),
        }];
        let (bytes, _) = verify_envelope(json.as_bytes(), &trusted)
            .map_err(|e| format!("self-check with '{id}': {e}"))?;
        if bytes != payload {
            return Err("self-check: payload changed".into());
        }
    }
    fs::write(out, &json).map_err(|e| format!("{out}: {e}"))?;
    let ids: Vec<&str> = keys.iter().map(|(id, _)| id.as_str()).collect();
    eprintln!(
        "signed {} {} with {} → {out} ({} bytes)",
        m.version,
        payload_path,
        ids.join(", "),
        json.len()
    );
    Ok(ExitCode::SUCCESS)
}

fn verify(opts: &Opts) -> Result<ExitCode, String> {
    let path = one(opts, "envelope")?;
    let repo = opt(opts, "repo")?.unwrap_or(manifest::REPO);
    let mut trusted: Vec<TrustedKey> = opts
        .get("pubkey")
        .into_iter()
        .flatten()
        .map(|v| parse_pubkey(v))
        .collect::<Result<_, _>>()?;
    if let Some(var) = opt(opts, "key-env")? {
        let secret = std::env::var(var).map_err(|_| format!("${var} is not set"))?;
        for (id, seed) in parse_signing_keys(&secret)? {
            trusted.push(TrustedKey {
                key_id: Box::leak(id.into_boxed_str()),
                pubkey: envelope::public_key(&seed),
            });
        }
    }
    if trusted.is_empty() {
        return Err("give --pubkey or --key-env".into());
    }
    let bytes = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let (payload, key_id) = verify_envelope(&bytes, &trusted).map_err(|e| e.to_string())?;
    let m: UpdateManifest =
        serde_json::from_slice(&payload).map_err(|e| format!("invalid manifest JSON: {e}"))?;
    m.validate_for_repo(repo)?;
    eprintln!("{path}: Oto {} signed by '{key_id}'", m.version);
    print!("{}", String::from_utf8_lossy(&payload));
    Ok(ExitCode::SUCCESS)
}

fn extract(opts: &Opts) -> Result<ExitCode, String> {
    let zip = PathBuf::from(one(opts, "zip")?);
    let dest = PathBuf::from(one(opts, "dest")?);
    let sha = one(opts, "expected-sha256")?;
    let size: u64 = num(opts, "expected-size")?;
    if !dest.exists() {
        use std::os::unix::fs::DirBuilderExt as _;
        fs::DirBuilder::new()
            .mode(0o700)
            .recursive(true)
            .create(&dest)
            .map_err(|e| format!("{}: {e}", dest.display()))?;
    }
    let app = extract_verified_zip(&zip, &dest, sha, size)?;
    println!("{}", app.display());
    Ok(ExitCode::SUCCESS)
}

fn check_version_gt(opts: &Opts) -> Result<ExitCode, String> {
    let parse = |name: &str| -> Result<SemVer, String> {
        let v = one(opts, name)?;
        v.strip_prefix('v').unwrap_or(v).parse()
    };
    let (candidate, current) = (parse("candidate")?, parse("current")?);
    if candidate > current {
        println!("{candidate} > {current}");
        Ok(ExitCode::SUCCESS)
    } else {
        println!("{candidate} <= {current}");
        Ok(ExitCode::from(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_dates() {
        assert_eq!(utc(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(utc(951_782_400), (2000, 2, 29, 0, 0, 0));
        assert_eq!(utc(1_795_740_245), (2026, 11, 27, 0, 44, 5));
    }

    #[test]
    fn signing_key_lists() {
        let a = B64.encode([1u8; 32]);
        let b = B64.encode([2u8; 32]);
        let keys = parse_signing_keys(&format!("old:{a}, new:{b}")).unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[1].0, "new");
        assert!(parse_signing_keys("").is_err());
        assert!(parse_signing_keys("nocolon").is_err());
        assert!(parse_signing_keys(&format!("x:{}", B64.encode([1u8; 31]))).is_err());
    }
}
