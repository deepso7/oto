//! Installing the macOS HAL driver: find `install-driver.sh` and run it with admin rights through
//! `osascript`, which shows the system's password prompt.

use std::path::{Path, PathBuf};

const SCRIPT: &str = "install-driver.sh";

/// Where the installer script may live, most specific first: an explicit override, an app
/// bundle's Resources, then the repo (walking up from the executable, then the source tree this
/// binary was built from).
pub fn candidates(exe: Option<&Path>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("OTO_INSTALL_SCRIPT") {
        out.push(PathBuf::from(p));
    }
    if let Some(dir) = exe.and_then(Path::parent) {
        // Oto.app/Contents/MacOS/oto-app → Oto.app/Contents/Resources.
        let resources = dir.join("../Resources");
        out.push(resources.join(SCRIPT));
        out.push(resources.join("scripts/macos").join(SCRIPT));
        // target/debug/oto-app → <repo>/scripts/macos.
        for up in dir.ancestors() {
            out.push(up.join("scripts/macos").join(SCRIPT));
        }
    }
    out.push(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/macos")
            .join(SCRIPT),
    );
    out
}

pub fn find_script() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    candidates(exe.as_deref())
        .into_iter()
        .find(|p| p.is_file())
        .map(|p| p.canonicalize().unwrap_or(p))
}

/// The AppleScript that runs `script` as root. The path goes through `quoted form of`, so
/// spaces and quotes in it are safe; only the AppleScript string literal needs escaping.
pub fn applescript(script: &Path) -> String {
    let literal = script
        .to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    format!(
        "do shell script \"/bin/bash \" & quoted form of \"{literal}\" with administrator privileges"
    )
}

/// How an install ended.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Installed,
    /// The user dismissed the password prompt.
    Cancelled,
    Failed(String),
}

/// Runs the installer and waits for it. Blocks: call it off the UI thread.
pub fn install() -> Outcome {
    let Some(script) = find_script() else {
        return Outcome::Failed(
            "The driver installer (install-driver.sh) isn't in this build of Oto.".into(),
        );
    };
    let out = match std::process::Command::new("/usr/bin/osascript")
        .arg("-e")
        .arg(applescript(&script))
        .output()
    {
        Ok(out) => out,
        Err(e) => return Outcome::Failed(format!("Couldn't run osascript: {e}")),
    };
    if out.status.success() {
        return Outcome::Installed;
    }
    let err = String::from_utf8_lossy(&out.stderr);
    // osascript reports a dismissed prompt as error -128.
    if err.contains("(-128)") {
        Outcome::Cancelled
    } else {
        Outcome::Failed(last_line(&err))
    }
}

/// osascript's error, without its "0:123: execution error:" prefix.
fn last_line(err: &str) -> String {
    let line = err.trim().lines().last().unwrap_or("The installer failed.");
    let line = line
        .split_once("execution error: ")
        .map_or(line, |(_, rest)| rest);
    line.trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_in_the_bundle_then_the_repo() {
        let c = candidates(Some(Path::new(
            "/Applications/Oto.app/Contents/MacOS/oto-app",
        )));
        let c: Vec<_> = c.iter().map(|p| p.to_string_lossy().into_owned()).collect();
        let bundle = c.iter().position(|p| {
            p == "/Applications/Oto.app/Contents/MacOS/../Resources/install-driver.sh"
        });
        let repo = c
            .iter()
            .position(|p| p.ends_with("oto-app/../../scripts/macos/install-driver.sh"));
        assert!(bundle.unwrap() < repo.unwrap(), "{c:?}");
    }

    #[test]
    fn escapes_the_path() {
        let s = applescript(Path::new("/tmp/a \"b\"/install-driver.sh"));
        assert_eq!(
            s,
            r#"do shell script "/bin/bash " & quoted form of "/tmp/a \"b\"/install-driver.sh" with administrator privileges"#
        );
    }

    #[test]
    fn trims_osascript_errors() {
        assert_eq!(
            last_line("0:120: execution error: Driver missing (1)\n"),
            "Driver missing (1)"
        );
    }
}
