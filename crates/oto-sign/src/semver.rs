// SPDX-License-Identifier: MIT OR Apache-2.0
//! Strict release versions (`major.minor.patch`, no pre-release or build metadata: Oto has a
//! single stable channel) and OS versions (`major.minor[.patch]`) for `min_os`.

use std::fmt;
use std::str::FromStr;

/// A release version. Ordering is numeric, field by field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemVer {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

impl SemVer {
    /// Parses a release tag, `v` followed by a version (`v0.1.0`).
    pub fn parse_tag(tag: &str) -> Result<Self, String> {
        tag.strip_prefix('v')
            .ok_or_else(|| format!("tag '{tag}' does not start with 'v'"))?
            .parse()
    }
}

impl FromStr for SemVer {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let parts = numeric_parts(s, 3, 3)
            .ok_or_else(|| format!("'{s}' is not a version of the form X.Y.Z"))?;
        Ok(Self {
            major: parts[0],
            minor: parts[1],
            patch: parts[2],
        })
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// An OS version (`12.0`, `14.2.1`). A missing patch is 0, so `12.0 == 12.0.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OsVersion {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

impl FromStr for OsVersion {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let parts = numeric_parts(s, 2, 3)
            .ok_or_else(|| format!("'{s}' is not an OS version of the form X.Y or X.Y.Z"))?;
        Ok(Self {
            major: parts[0],
            minor: parts[1],
            patch: parts.get(2).copied().unwrap_or(0),
        })
    }
}

impl fmt::Display for OsVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Splits on `.` into `min..=max` plain decimal numbers: ASCII digits only, no sign, no
/// whitespace, no leading zeros (except `0` itself), each fitting a `u64`.
fn numeric_parts(s: &str, min: usize, max: usize) -> Option<Vec<u64>> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() < min || parts.len() > max {
        return None;
    }
    parts
        .into_iter()
        .map(|p| {
            let ok = !p.is_empty()
                && p.bytes().all(|b| b.is_ascii_digit())
                && (p == "0" || !p.starts_with('0'));
            if ok { p.parse().ok() } else { None }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> SemVer {
        s.parse().unwrap()
    }

    #[test]
    fn parses_strict_versions() {
        assert_eq!(
            v("0.1.0"),
            SemVer {
                major: 0,
                minor: 1,
                patch: 0
            }
        );
        assert_eq!(v("10.20.30").to_string(), "10.20.30");
        for bad in [
            "",
            "1",
            "1.2",
            "1.2.3.4",
            "v1.2.3",
            "01.2.3",
            "1.02.3",
            "1.2.03",
            "1.2.3-rc.1",
            "1.2.3+b",
            " 1.2.3",
            "1.2.3 ",
            "1..3",
            "-1.2.3",
            "+1.2.3",
            "1.2.x",
            "99999999999999999999.0.0",
        ] {
            assert!(bad.parse::<SemVer>().is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn orders_numerically() {
        assert!(v("0.0.10") > v("0.0.9"));
        assert!(v("0.10.0") > v("0.9.99"));
        assert!(v("1.0.0") > v("0.99.99"));
        assert!(v("1.2.3") == v("1.2.3"));
        assert!(!(v("1.2.3") > v("1.2.3")));
    }

    #[test]
    fn parses_tags() {
        assert_eq!(SemVer::parse_tag("v0.0.4").unwrap(), v("0.0.4"));
        assert!(SemVer::parse_tag("0.0.4").is_err());
        assert!(SemVer::parse_tag("vv0.0.4").is_err());
    }

    #[test]
    fn os_versions() {
        let os = |s: &str| s.parse::<OsVersion>().unwrap();
        assert_eq!(os("12.0"), os("12.0.0"));
        assert!(os("13.0") > os("12.7.6"));
        assert!(os("12.10") > os("12.9"));
        assert!(os("14.2.1") > os("14.2"));
        assert!(os("11.7") < os("12.0"));
        for bad in ["", "12", "12.", "12.0.0.0", "12.a", "12.00", " 12.0"] {
            assert!(
                bad.parse::<OsVersion>().is_err(),
                "{bad:?} should not parse"
            );
        }
    }
}
