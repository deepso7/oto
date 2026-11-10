//! Start at login. On macOS a LaunchAgent plist in `~/Library/LaunchAgents`, which works for a
//! bare `cargo run` binary and for a bundle alike (SMAppService needs a signed bundle).

use auto_launch::{AutoLaunch, AutoLaunchBuilder};

fn launcher() -> Result<AutoLaunch, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut b = AutoLaunchBuilder::new();
    b.set_app_name("dev.oto.app")
        .set_app_path(&exe.to_string_lossy());
    #[cfg(target_os = "macos")]
    b.set_macos_launch_mode(auto_launch::MacOSLaunchMode::LaunchAgent);
    b.build().map_err(|e| e.to_string())
}

pub fn is_enabled() -> bool {
    launcher()
        .and_then(|l| l.is_enabled().map_err(|e| e.to_string()))
        .unwrap_or(false)
}

pub fn set(on: bool) -> Result<(), String> {
    let l = launcher()?;
    let r = if on { l.enable() } else { l.disable() };
    r.map_err(|e| format!("Couldn't change start at login: {e}"))
}
