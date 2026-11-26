//! The menu-bar status item. On macOS it's `tray-icon`'s `NSStatusItem`, made on Rune's main
//! thread from an `AppHandle::update` once the event loop runs (Rune's own `Tray` has no macOS
//! backend yet). A left click toggles the popover under the item; a right click opens the menu:
//! the switch, Open Oto, Quit. Elsewhere there's no tray yet and the app opens a normal window.

use std::time::{Duration, Instant};

use rune::AppHandle;

use crate::state::State;

/// The status item's glyph, in points: the menu bar draws images 22 pt tall.
pub const GLYPH_PT: (u32, u32) = (26, 22);

/// The status item's glyph: the Oto mark (packaging/brand), simplified for the menu bar. Two
/// faint, uneven noise bars go into the o and two even voice bars come out. A template image,
/// so macOS tints it; off draws it all faint. `scale` pixels per point, straight RGBA, returned
/// with its pixel width and height.
pub fn glyph(scale: u32, on: bool) -> (u32, u32, Vec<u8>) {
    let (w, h) = (GLYPH_PT.0 * scale, GLYPH_PT.1 * scale);
    let s = scale as f32;
    let (cx, mid) = (w as f32 / 2.0, h as f32 / 2.0);
    let r = s; // bar half-width: 2 pt bars
    // Bars as (centre offset from the o, half-height, opacity), in points.
    let bars = [
        (-10.75, 3.0, 0.45),
        (-7.25, 6.5, 0.45),
        (7.25, 6.0, 1.0),
        (10.75, 3.0, 1.0),
    ];
    let (ring_r, ring_half_w) = (3.25 * s, 1.0 * s);
    let dim = if on { 1.0 } else { 0.4 };
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    const SS: usize = 4;
    for y in 0..h {
        for x in 0..w {
            let mut cover = 0.0f32;
            for sy in 0..SS {
                for sx in 0..SS {
                    let fx = x as f32 + (sx as f32 + 0.5) / SS as f32;
                    let fy = y as f32 + (sy as f32 + 0.5) / SS as f32;
                    let ring = ((fx - cx).hypot(fy - mid) - ring_r).abs() <= ring_half_w;
                    // Bars are vertical capsules centred on the middle line.
                    let bar = bars.iter().find(|(off, half, _)| {
                        let dy = ((fy - mid).abs() - (half * s - r)).max(0.0);
                        (fx - (cx + off * s)).hypot(dy) <= r
                    });
                    cover += if ring { 1.0 } else { bar.map_or(0.0, |b| b.2) };
                }
            }
            let a = cover / (SS * SS) as f32 * dim;
            rgba[((y * w + x) * 4 + 3) as usize] = (a * 255.0).round() as u8;
        }
    }
    (w, h, rgba)
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

/// How far below its display's top edge a status item's bottom can sit and still be in the
/// menu bar (which is 24 to 40 pt tall). A new item's window sits elsewhere until it's given a
/// place.
const MENU_BAR_MAX: f64 = 64.0;
/// The gap under the menu bar, and the margin kept to the display's side edges.
const GAP: f64 = 6.0;
const MARGIN: f64 = 8.0;

/// Where a popover of `size` goes under a status item: its top-left in points from the primary
/// display's top-left, y down (negative on displays left of or above the primary). Centred on
/// the item, [`GAP`] below the item (and below the display's menu bar, when taller than the
/// button), kept [`MARGIN`] inside the side edges of the item's display.
///
/// Inputs are AppKit rects, `[x, y, width, height]` with y up from the primary display's
/// bottom-left: the status item's button (or window), and its display's frame and visible
/// frame when known. `None` when the item isn't in a menu bar yet.
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
pub fn popover_origin(
    primary_height: f64,
    item: [f64; 4],
    display: Option<([f64; 4], [f64; 4])>,
    size: (f32, f32),
) -> Option<(f32, f32)> {
    let top_down = |r: [f64; 4]| primary_height - r[1] - r[3];
    let item_top = top_down(item);
    let item_bottom = item_top + item[3];
    // Measured from the top of the item's own display: other displays' tops needn't be level
    // with the primary's. Checking the bottom tolerates status item windows on macOS 26/27
    // whose frame extends a few points above the display's top edge.
    let display_top = display.map_or(0.0, |(frame, _)| top_down(frame));
    let bottom_below = item_bottom - display_top;
    if item[2] <= 0.0 || item[3] <= 0.0 || bottom_below <= 0.0 || bottom_below > MENU_BAR_MAX {
        return None;
    }
    let width = f64::from(size.0);
    let mut x = item[0] + item[2] / 2.0 - width / 2.0;
    let mut bar_bottom = item_bottom;
    if let Some((_, visible)) = display {
        x = x
            .min(visible[0] + visible[2] - width - MARGIN)
            .max(visible[0] + MARGIN);
        // The button (22 pt) sits inside a taller menu bar (30 pt on macOS 27, 37 pt on
        // notched displays): keep the gap below the menu bar itself, not just the button.
        let visible_top = top_down(visible);
        if visible_top - display_top <= MENU_BAR_MAX {
            bar_bottom = bar_bottom.max(visible_top);
        }
    }
    Some((x as f32, (bar_bottom + GAP) as f32))
}

pub fn toggle_window(s: &mut State, h: &AppHandle<State>) {
    if h.is_visible() {
        // A hidden popover must not keep the mic or the speakers.
        s.stop_hear_test();
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
        let (w, h, rgba) = super::glyph(2, on);
        Icon::from_rgba(rgba, w, h).expect("glyph size matches")
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
            let button = self.icon.ns_status_item()?.button(mtm)?;
            let window = button.window()?;
            let primary = NSScreen::screens(mtm).firstObject()?.frame().size.height;
            let rect =
                |r: objc2_foundation::NSRect| [r.origin.x, r.origin.y, r.size.width, r.size.height];
            // Use the button's screen rect rather than `window.frame()`: on macOS 26/27
            // `NSStatusBarWindow` extends above and below the menu bar.
            let button_rect =
                window.convertRectToScreen(button.convertRect_toView(button.bounds(), None));
            let display = window
                .screen()
                .map(|s| (rect(s.frame()), rect(s.visibleFrame())));
            super::popover_origin(primary, rect(button_rect), display, size)
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
    fn glyph_is_the_mark_and_dims_when_off() {
        let (w, h, on) = glyph(2, true);
        let (_, _, off) = glyph(2, false);
        assert_eq!((w, h), (52, 44));
        assert_eq!(on.len(), (w * h * 4) as usize);
        let alpha = |v: &[u8], x: u32, y: u32| v[((y * w + x) * 4 + 3) as usize];
        // The o is centred, solid on its rim and empty in the middle; the corners are empty.
        assert_eq!(alpha(&on, 26, 22), 0);
        assert_eq!(alpha(&on, 26 + 6, 22), 255);
        assert_eq!(alpha(&on, 0, 0), 0);
        // Voice bars (right) are solid, noise bars (left) faint, about the o.
        let (voice, noise) = (alpha(&on, 26 + 14, 22), alpha(&on, 26 - 14, 22));
        assert_eq!(voice, 255);
        assert!((100..140).contains(&noise), "{noise}");
        // Off dims everything.
        assert!(alpha(&off, 26 + 6, 22) < 128);
        // The glyph stays within the canvas, with no clipped ink at the edges.
        for y in 0..h {
            assert_eq!(alpha(&on, 0, y), 0);
            assert_eq!(alpha(&on, w - 1, y), 0);
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
    fn popover_sits_under_taller_macos27_menu_bar_and_overhanging_window() {
        // 2560x1440 with a 30 pt menu bar and 67 pt bottom dock.
        let frame = [0.0, 0.0, 2560.0, 1440.0];
        let visible = [0.0, 67.0, 2560.0, 1343.0];
        // The 22 pt status bar button sits inside the 30 pt menu bar: the popover keeps GAP
        // below the menu bar itself (30 + 6 = 36), not just below the button (25 + 6 = 31).
        let button = [2095.0, 1415.0, 46.0, 22.0];
        assert_eq!(
            popover_origin(frame[3], button, Some((frame, visible)), SIZE),
            Some((2095.0 + 23.0 - 170.0, 36.0))
        );
        // Even if given an NSStatusBarWindow whose top extends above the display (1406 + 39 =
        // 1445 > 1440), the item is still recognized as in the menu bar.
        let overhanging_win = [2095.0, 1406.0, 46.0, 39.0];
        assert_eq!(
            popover_origin(frame[3], overhanging_win, Some((frame, visible)), SIZE),
            Some((2095.0 + 23.0 - 170.0, 40.0))
        );
        // Auto-hidden menu bar: visibleFrame covers the full height, so y follows the button.
        assert_eq!(
            popover_origin(frame[3], button, Some((frame, frame)), SIZE),
            Some((2095.0 + 23.0 - 170.0, 31.0))
        );
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
            popover_origin(982.0, [10.0, 958.0, 30.0, 0.0], display, SIZE),
            None
        );
        assert_eq!(
            popover_origin(982.0, [10.0, 0.0, 30.0, 24.0], display, SIZE),
            None
        );
        assert_eq!(
            popover_origin(982.0, [10.0, 1000.0, 30.0, 24.0], display, SIZE),
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
