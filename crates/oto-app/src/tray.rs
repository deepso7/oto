//! The menu-bar status item. On macOS it's `tray-icon`'s `NSStatusItem`, made on Rune's main
//! thread from an `AppHandle::update` once the event loop runs (Rune's own `Tray` has no macOS
//! backend yet). A left click toggles the popover under the item; a right click opens the menu:
//! the switch, Open Oto, Quit. Elsewhere there's no tray yet and the app opens a normal window.

use std::time::{Duration, Instant};

use rune::AppHandle;

use crate::state::State;

/// The status item's glyph: five rounded level bars, a template image so macOS tints it for
/// the menu bar. Off draws the bars faint. `px` square, straight RGBA.
pub fn glyph(px: u32, on: bool) -> Vec<u8> {
    let heights = [0.30, 0.62, 0.92, 0.62, 0.30];
    let s = px as f32 / 22.0; // pixels per point; the menu bar draws the image 22 pt tall
    let (bar_w, gap) = (2.4 * s, 1.6 * s);
    let total = heights.len() as f32 * bar_w + (heights.len() - 1) as f32 * gap;
    let left = (px as f32 - total) / 2.0;
    let alpha = if on { 1.0 } else { 0.4 };
    let max_h = 15.0 * s;
    let mid = px as f32 / 2.0;
    let mut rgba = vec![0u8; (px * px * 4) as usize];
    const SS: usize = 4;
    for y in 0..px {
        for x in 0..px {
            let mut cover = 0usize;
            for sy in 0..SS {
                for sx in 0..SS {
                    let fx = x as f32 + (sx as f32 + 0.5) / SS as f32;
                    let fy = y as f32 + (sy as f32 + 0.5) / SS as f32;
                    let inside = heights.iter().enumerate().any(|(i, h)| {
                        let x0 = left + i as f32 * (bar_w + gap);
                        let half = (h * max_h / 2.0).max(bar_w / 2.0);
                        let r = bar_w / 2.0;
                        let cx = x0 + r;
                        // A vertical capsule centred on the middle line.
                        let dy = ((fy - mid).abs() - (half - r)).max(0.0);
                        let dx = fx - cx;
                        dx * dx + dy * dy <= r * r
                    });
                    cover += inside as usize;
                }
            }
            let a = cover as f32 / (SS * SS) as f32 * alpha;
            rgba[((y * px + x) * 4 + 3) as usize] = (a * 255.0).round() as u8;
        }
    }
    rgba
}

/// A press on the status item right after the popover hid for losing focus is the same click
/// that hid it: don't reopen.
const REOPEN_GUARD: Duration = Duration::from_millis(400);

pub fn hid_recently(hidden_at: Option<Instant>) -> bool {
    hidden_at.is_some_and(|t| t.elapsed() < REOPEN_GUARD)
}

/// Shows the window, under the status item when there is one, else where it was.
pub fn show_window(s: &mut State, h: &AppHandle<State>) {
    if let Some(rect) = s.tray.as_ref().and_then(|t| t.popover_rect(s.size)) {
        h.place(rune::DisplayTarget::Primary, rect);
    }
    h.open();
}

pub fn toggle_window(s: &mut State, h: &AppHandle<State>) {
    if h.is_visible() {
        h.hide();
    } else if !hid_recently(s.hidden_at) {
        show_window(s, h);
    }
}

#[cfg(target_os = "macos")]
pub use mac::Tray;

#[cfg(target_os = "macos")]
mod mac {
    use objc2::MainThreadMarker;
    use objc2_app_kit::NSScreen;
    use rune::{AppHandle, OverlayPlacement};
    use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem};
    use tray_icon::{
        Icon, MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent,
    };

    use crate::state::State;

    pub struct Tray {
        icon: TrayIcon,
        toggle: CheckMenuItem,
    }

    fn icon(on: bool) -> Icon {
        Icon::from_rgba(super::glyph(44, on), 44, 44).expect("glyph size matches")
    }

    impl Tray {
        /// Must run on the main thread with the event loop running.
        pub fn new(h: AppHandle<State>, enabled: bool) -> Result<Tray, String> {
            let toggle =
                CheckMenuItem::with_id("toggle", "Noise Cancellation", true, enabled, None);
            let open = MenuItem::with_id("open", "Open Oto…", true, None);
            let quit = MenuItem::with_id("quit", "Quit Oto", true, None);
            let menu = Menu::new();
            menu.append_items(&[
                &toggle,
                &PredefinedMenuItem::separator(),
                &open,
                &PredefinedMenuItem::separator(),
                &quit,
            ])
            .map_err(|e| e.to_string())?;
            let icon = TrayIconBuilder::new()
                .with_id("oto")
                .with_icon_templated(icon(enabled))
                .with_tooltip("Oto")
                .with_menu(Box::new(menu))
                .with_menu_on_left_click(false)
                .build()
                .map_err(|e| e.to_string())?;

            let menu_h = h.clone();
            MenuEvent::set_event_handler(Some(move |e: MenuEvent| {
                let h = menu_h.clone();
                match e.id().as_ref() {
                    "toggle" => h.update(|s, _| s.toggle()),
                    "open" => {
                        let h2 = h.clone();
                        h.update(move |s, _| super::show_window(s, &h2));
                    }
                    "quit" => h.update(|s, tx| {
                        s.shutdown();
                        tx.quit();
                    }),
                    _ => {}
                }
            }));
            TrayIconEvent::set_event_handler(Some(move |e: TrayIconEvent| {
                if let TrayIconEvent::Click {
                    button: MouseButton::Left,
                    button_state: MouseButtonState::Up,
                    ..
                } = e
                {
                    let h2 = h.clone();
                    h.update(move |s, _| super::toggle_window(s, &h2));
                }
            }));
            Ok(Tray { icon, toggle })
        }

        pub fn set_enabled(&self, on: bool) {
            self.toggle.set_checked(on);
            let _ = self.icon.set_icon_templated(Some(icon(on)));
        }

        /// Where a `size` window goes under the status item: centred on it, 6 pt below the menu
        /// bar, kept 8 pt inside the screen. Top-down global points (Rune's `Rect` placement).
        pub fn popover_rect(&self, size: (f32, f32)) -> Option<OverlayPlacement> {
            let mtm = MainThreadMarker::new()?;
            let window = self.icon.ns_status_item()?.button(mtm)?.window()?;
            let primary = NSScreen::screens(mtm).firstObject()?.frame().size.height;
            let item = window.frame();
            let screen = window.screen().map(|s| s.visibleFrame());
            let item_top = primary - item.origin.y - item.size.height;
            // Right after creation the item has no place in the menu bar yet.
            if item.size.width <= 0.0 || item_top > 64.0 {
                return None;
            }
            let mut x = item.origin.x + item.size.width / 2.0 - f64::from(size.0) / 2.0;
            if let Some(v) = screen {
                x = x
                    .min(v.origin.x + v.size.width - f64::from(size.0) - 8.0)
                    .max(v.origin.x + 8.0);
            }
            let y = item_top + item.size.height + 6.0;
            Some(OverlayPlacement::Rect {
                x: x as f32,
                y: y as f32,
                width: size.0,
                height: size.1,
            })
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use other::Tray;

#[cfg(not(target_os = "macos"))]
mod other {
    use rune::{AppHandle, OverlayPlacement};

    use crate::state::State;

    /// TODO(linux): Rune's own `rune::Tray` (StatusNotifierItem) once Linux is in scope.
    pub struct Tray;

    impl Tray {
        pub fn new(_: AppHandle<State>, _: bool) -> Result<Tray, String> {
            Err("no tray on this platform yet".into())
        }
        pub fn set_enabled(&self, _: bool) {}
        pub fn popover_rect(&self, _: (f32, f32)) -> Option<OverlayPlacement> {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_is_centred_and_dims_when_off() {
        let on = glyph(44, true);
        let off = glyph(44, false);
        assert_eq!(on.len(), 44 * 44 * 4);
        let alpha = |v: &[u8], x: usize, y: usize| v[(y * 44 + x) * 4 + 3];
        // The middle bar is solid through the centre; corners are empty.
        assert_eq!(alpha(&on, 22, 22), 255);
        assert_eq!(alpha(&on, 0, 0), 0);
        assert!(alpha(&off, 22, 22) < 128);
        // Symmetric left to right.
        for y in 0..44 {
            for x in 0..22 {
                assert_eq!(alpha(&on, x, y), alpha(&on, 43 - x, y), "({x}, {y})");
            }
        }
    }
}
