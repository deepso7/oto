//! Microphone permission. macOS hands a process without it silence rather than an error, so the
//! app asks AVFoundation where it stands instead of guessing from flat meters.
//!
//! The status belongs to the "responsible" process: Oto.app when launched from Finder, the
//! terminal when a bare binary is run from one.

/// `AVAuthorizationStatus` for audio capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mic {
    Authorized,
    /// Never asked: the system prompt can still be shown.
    NotDetermined,
    /// The user said no. Only System Settings can change it.
    Denied,
    /// Blocked by a policy the user can't change (Screen Time, a device profile).
    Restricted,
}

impl Mic {
    /// Oto gets silence and no prompt will come.
    pub fn blocked(self) -> bool {
        matches!(self, Mic::Denied | Mic::Restricted)
    }
}

/// System Settings, Privacy & Security, Microphone.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub const SETTINGS_URL: &str =
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone";

/// Maps `AVAuthorizationStatus` raw values. Unknown values read as not asked yet.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn from_raw(raw: isize) -> Mic {
    match raw {
        1 => Mic::Restricted,
        2 => Mic::Denied,
        3 => Mic::Authorized,
        _ => Mic::NotDetermined,
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use block2::RcBlock;
    use objc2::runtime::Bool;
    use objc2::{class, msg_send};
    use objc2_foundation::NSString;

    use super::Mic;

    #[link(name = "AVFoundation", kind = "framework")]
    unsafe extern "C" {
        static AVMediaTypeAudio: &'static NSString;
    }

    /// Where the permission stands. No prompt; safe on any thread. The first call talks to the
    /// system's permission daemon, so the app reads it off the UI thread.
    pub fn status() -> Mic {
        // SAFETY: a class method taking an NSString constant and returning an NSInteger enum.
        let raw: isize = unsafe {
            msg_send![
                class!(AVCaptureDevice),
                authorizationStatusForMediaType: AVMediaTypeAudio
            ]
        };
        super::from_raw(raw)
    }

    /// Shows the system's microphone prompt if the user was never asked, and calls `done` with
    /// the answer on a queue of the system's choosing. Already answered: `done` is called with
    /// that answer and nothing is shown.
    pub fn request(done: impl Fn(bool) + Send + 'static) {
        let block = RcBlock::new(move |granted: Bool| done(granted.as_bool()));
        // SAFETY: the completion handler is `void (^)(BOOL)`; AVFoundation copies the block.
        let _: () = unsafe {
            msg_send![
                class!(AVCaptureDevice),
                requestAccessForMediaType: AVMediaTypeAudio,
                completionHandler: &*block
            ]
        };
    }

    pub fn open_settings() -> Result<(), String> {
        let failed = |why: String| format!("Couldn't open System Settings: {why}");
        match std::process::Command::new("/usr/bin/open")
            .arg(super::SETTINGS_URL)
            .status()
        {
            Ok(s) if s.success() => Ok(()),
            Ok(s) => Err(failed(s.to_string())),
            Err(e) => Err(failed(e.to_string())),
        }
    }
}

#[cfg(target_os = "macos")]
pub use macos::{open_settings, request, status};

/// Elsewhere there's no such permission: always allowed.
#[cfg(not(target_os = "macos"))]
mod other {
    use super::Mic;

    pub fn status() -> Mic {
        Mic::Authorized
    }

    pub fn request(done: impl Fn(bool) + Send + 'static) {
        done(true);
    }

    pub fn open_settings() -> Result<(), String> {
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
pub use other::{open_settings, request, status};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_values_map_to_avauthorizationstatus() {
        assert_eq!(from_raw(0), Mic::NotDetermined);
        assert_eq!(from_raw(1), Mic::Restricted);
        assert_eq!(from_raw(2), Mic::Denied);
        assert_eq!(from_raw(3), Mic::Authorized);
        assert_eq!(from_raw(99), Mic::NotDetermined);
        assert!(Mic::Denied.blocked() && Mic::Restricted.blocked());
        assert!(!Mic::NotDetermined.blocked() && !Mic::Authorized.blocked());
    }

    /// Reading the status shows no prompt, so it runs headless; it proves the AVFoundation
    /// class, selector and constant resolve.
    #[cfg(target_os = "macos")]
    #[test]
    fn status_reads_without_a_prompt() {
        let first = status();
        assert_eq!(status(), first);
    }
}
