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
    if let Some((x, y)) = s.tray.as_ref().and_then(|t| t.popover_origin(s.size)) {
        // On `DisplayTarget::Primary` Rune's `Rect` is a global rect, in points from the primary
        // display's top-left, that may sit on any display (rune `OverlayPlacement::Rect`; its
        // macOS `overlay_rect_on` offsets the rect from the first screen's frame and doesn't
        // clamp it). So the status item's own display needs no separate target.
        h.place(
            rune::DisplayTarget::Primary,
            rune::OverlayPlacement::Rect {
                x,
                y,
                width: s.size.0,
                height: s.size.1,
            },
        );
    }
    h.open();
}

/// How far below its display's top edge a status item can sit and still be in the menu bar
/// (which is 24 to 40 pt tall). A new item's window sits elsewhere until it's given a place.
const MENU_BAR_MAX: f64 = 64.0;
/// The gap under the menu bar, and the margin kept to the display's side edges.
const GAP: f64 = 6.0;
const MARGIN: f64 = 8.0;

/// Where a popover of `size` goes under a status item: its top-left in points from the primary
/// display's top-left, y down (negative on displays left of or above the primary). Centred on
/// the item, [`GAP`] below it, kept [`MARGIN`] inside the side edges of the item's display.
///
/// Inputs are AppKit rects, `[x, y, width, height]` with y up from the primary display's
/// bottom-left: the status item's window, and its display's frame and visible frame when
/// known. `None` when the item isn't in a menu bar yet.
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
pub fn popover_origin(
    primary_height: f64,
    item: [f64; 4],
    display: Option<([f64; 4], [f64; 4])>,
    size: (f32, f32),
) -> Option<(f32, f32)> {
    let top_down = |r: [f64; 4]| primary_height - r[1] - r[3];
    let item_top = top_down(item);
    // Measured from the top of the item's own display: other displays' tops needn't be level
    // with the primary's.
    let display_top = display.map_or(0.0, |(frame, _)| top_down(frame));
    let below = item_top - display_top;
    if item[2] <= 0.0 || !(-1.0..=MENU_BAR_MAX).contains(&below) {
        return None;
    }
    let width = f64::from(size.0);
    let mut x = item[0] + item[2] / 2.0 - width / 2.0;
    if let Some((_, visible)) = display {
        x = x
            .min(visible[0] + visible[2] - width - MARGIN)
            .max(visible[0] + MARGIN);
    }
    Some((x as f32, (item_top + item[3] + GAP) as f32))
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
    use rune::AppHandle;
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

        /// Where a `size` window goes under the status item; see [`super::popover_origin`].
        pub fn popover_origin(&self, size: (f32, f32)) -> Option<(f32, f32)> {
            let mtm = MainThreadMarker::new()?;
            let window = self.icon.ns_status_item()?.button(mtm)?.window()?;
            let primary = NSScreen::screens(mtm).firstObject()?.frame().size.height;
            let rect =
                |r: objc2_foundation::NSRect| [r.origin.x, r.origin.y, r.size.width, r.size.height];
            let display = window
                .screen()
                .map(|s| (rect(s.frame()), rect(s.visibleFrame())));
            super::popover_origin(primary, rect(window.frame()), display, size)
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use other::Tray;

#[cfg(not(target_os = "macos"))]
mod other {
    use rune::AppHandle;

    use crate::state::State;

    /// TODO(linux): Rune's own `rune::Tray` (StatusNotifierItem) once Linux is in scope.
    pub struct Tray;

    impl Tray {
        pub fn new(_: AppHandle<State>, _: bool) -> Result<Tray, String> {
            Err("no tray on this platform yet".into())
        }
        pub fn set_enabled(&self, _: bool) {}
        pub fn popover_origin(&self, _: (f32, f32)) -> Option<(f32, f32)> {
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

    const PRIMARY: [f64; 4] = [0.0, 0.0, 1512.0, 982.0];
    const SIZE: (f32, f32) = (340.0, 400.0);

    /// A display's visible frame: its frame less a 24 pt menu bar.
    fn visible(frame: [f64; 4]) -> [f64; 4] {
        [frame[0], frame[1], frame[2], frame[3] - 24.0]
    }

    /// A 30 pt status item in the menu bar of `frame`, its left edge at `x`.
    fn item(frame: [f64; 4], x: f64) -> [f64; 4] {
        [x, frame[1] + frame[3] - 24.0, 30.0, 24.0]
    }

    #[test]
    fn popover_sits_under_the_item_on_the_primary_display() {
        let at = popover_origin(
            PRIMARY[3],
            item(PRIMARY, 1000.0),
            Some((PRIMARY, visible(PRIMARY))),
            SIZE,
        );
        assert_eq!(at, Some((1015.0 - 170.0, 30.0)));
        // Near the right edge it stays 8 pt inside the display.
        let at = popover_origin(
            PRIMARY[3],
            item(PRIMARY, 1470.0),
            Some((PRIMARY, visible(PRIMARY))),
            SIZE,
        );
        assert_eq!(at, Some((1512.0 - 340.0 - 8.0, 30.0)));
    }

    #[test]
    fn popover_follows_the_item_to_a_display_left_of_and_above_the_primary() {
        // 1920x1080, left of the primary, its top 498 pt above the primary's.
        let left = [-1920.0, 400.0, 1920.0, 1080.0];
        let at = popover_origin(
            PRIMARY[3],
            item(left, -300.0),
            Some((left, visible(left))),
            SIZE,
        );
        // Global points from the primary's top-left: both negative here.
        assert_eq!(at, Some((-285.0 - 170.0, -498.0 + 24.0 + 6.0)));
        // Clamped to that display's edges, not the primary's.
        let at = popover_origin(
            PRIMARY[3],
            item(left, -1915.0),
            Some((left, visible(left))),
            SIZE,
        );
        assert_eq!(at, Some((-1912.0, -468.0)));
        let at = popover_origin(
            PRIMARY[3],
            item(left, -40.0),
            Some((left, visible(left))),
            SIZE,
        );
        assert_eq!(at, Some((-348.0, -468.0)));
    }

    #[test]
    fn popover_follows_the_item_to_a_display_whose_top_is_lower() {
        // Right of the primary, its top 402 pt below the primary's: further down than any
        // menu bar is tall, measured from the primary's top.
        let right = [1512.0, -500.0, 1920.0, 1080.0];
        let at = popover_origin(
            PRIMARY[3],
            item(right, 3000.0),
            Some((right, visible(right))),
            SIZE,
        );
        assert_eq!(at, Some((3015.0 - 170.0, 402.0 + 30.0)));
    }

    #[test]
    fn an_item_without_a_place_gives_no_position() {
        // Zero-sized, or parked away from its display's menu bar.
        let display = Some((PRIMARY, visible(PRIMARY)));
        assert_eq!(
            popover_origin(982.0, [10.0, 958.0, 0.0, 24.0], display, SIZE),
            None
        );
        assert_eq!(
            popover_origin(982.0, [10.0, 0.0, 30.0, 24.0], display, SIZE),
            None
        );
        // No display known: judged against the primary's top, and not clamped.
        assert_eq!(
            popover_origin(982.0, item(PRIMARY, 1500.0), None, SIZE),
            Some((1345.0, 30.0))
        );
        assert_eq!(
            popover_origin(982.0, [10.0, 0.0, 30.0, 24.0], None, SIZE),
            None
        );
    }
}
