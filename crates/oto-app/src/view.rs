//! The popover's screens: onboarding, main and settings. Built from Rune's neutral tokens and
//! rune-kit's Select, SegmentedControl, Switch and Button; the big switch and the meter are
//! drawn here.

use rune::tok::{self, Theme, motion, type_};
use rune::{Color, Cx, Cycle, El, Icon, SystemMaterial, TextStyle, col, el, icon, row, text};
use rune_kit::{
    Button, ButtonVariant, ControlSize, IconButton, IconButtonVariant, Kit, Placement, Select,
    SelectAppearance, StateColors,
};

use crate::meter::{BAR_H, BAR_W};
use crate::permission::Mic;
use crate::state::{
    DRIVER_PAUSE, DriverUiState, EngineState, HEAR_TEST, HearTest, Install, MODES, Open, Page,
    STRENGTHS, State, UpdateState, install_mode_note, openable_url, update_driver_note,
};
use crate::updater::apply::InstallMode;
use crate::updater::verify::UNRECOGNIZED_KEY_MESSAGE;

pub const WIDTH: f32 = 340.0;
/// The popover's corner radius; main.rs gives the window the same.
pub const RADIUS: f32 = 22.0;
const PAD: f32 = 12.0;
/// A tile's side padding.
const TILE_PX: f32 = 12.0;
/// A tile's content width.
const CARD_INNER: f32 = WIDTH - 2.0 * PAD - 2.0 * TILE_PX;
const TILE_RADIUS: f32 = 16.0;
/// A dropdown list: its 24 pt capsule rows sit 4 pt in, so this keeps the corners concentric.
const MENU_RADIUS: f32 = 16.0;

const H1: TextStyle = TextStyle::new(19.0, 680.0).lh(24.0).tracking(-0.2);
const BRAND: TextStyle = TextStyle::new(15.0, 700.0).lh(20.0).tracking(-0.1);
const TITLE: TextStyle = TextStyle::new(14.0, 620.0).lh(18.0);
const HERO: TextStyle = TextStyle::new(16.0, 680.0).lh(20.0).tracking(-0.15);
const BODY: TextStyle = type_::BODY.lh(18.0);
const LABEL: TextStyle = TextStyle::new(10.5, 600.0).lh(13.0).tracking(0.7);
const CAPTION: TextStyle = TextStyle::new(11.5, 440.0).lh(15.0).tracking(0.05);
const CAPTION_NUM: TextStyle = TextStyle::new(11.0, 500.0).lh(15.0).features("'tnum' 1");
const PILL: TextStyle = TextStyle::new(12.0, 560.0).lh(14.0);

/// The brand mark from packaging/brand/oto-mark.svg on a 24 grid: noise in (faint, ragged),
/// the o, and voice out (even).
static MARK: Icon = Icon {
    name: "oto-mark",
    view: 24.0,
    body: r#"<g stroke="currentColor" stroke-width="1.25" stroke-linecap="round" fill="none"><path d="M0.81 10.54V13.46M2.51 8.59V15.41M4.22 11.03V12.97M5.92 7.62V16.38" stroke-opacity=".45"/><circle cx="12" cy="12" r="3.16" stroke-width="1.75"/><path d="M18.08 9.81V14.19M19.78 8.11V15.89M21.49 9.81V14.19M23.19 11.27V12.73"/></g>"#,
};
static POWER: Icon = Icon {
    name: "oto-power",
    view: 16.0,
    body: r#"<path d="M8 2.25V7.25" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><path d="M4.6 4.35A5.1 5.1 0 1 0 11.4 4.35" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" fill="none"/>"#,
};
static SLIDERS: Icon = Icon {
    name: "oto-sliders",
    view: 16.0,
    body: r#"<path d="M2.5 5H8.5M12.5 5H13.5M2.5 11H3.5M7.5 11H13.5" stroke="currentColor" stroke-width="1.25" stroke-linecap="round"/><circle cx="10.5" cy="5" r="1.75" stroke="currentColor" stroke-width="1.25"/><circle cx="5.5" cy="11" r="1.75" stroke="currentColor" stroke-width="1.25"/>"#,
};
static CHEVRON_LEFT: Icon = Icon {
    name: "oto-chevron-left",
    view: 16.0,
    body: r#"<path d="M9.5 4L5.5 8L9.5 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>"#,
};

/// The brand colours, swappable while we try palettes. `OTO_SCHEME=ocean` picks Summer Ocean
/// Breeze; anything else is the app icon's blue.
#[derive(Clone, Copy)]
struct Scheme {
    /// The lit hero, top to bottom; also the logo tile.
    brand: (&'static str, &'static str),
    /// Switches and focus rings, light and dark.
    accent: (&'static str, &'static str),
    /// What sits on the lit hero: the power button, the Out bar and the rings.
    on_hero: &'static str,
    /// The power glyph on `on_hero`.
    glyph: &'static str,
    /// The In bar on the lit hero.
    hero_in: &'static str,
    /// Muted: the state line when Oto is on but silencing the mic.
    alert: Option<(&'static str, &'static str)>,
    /// The window behind the glass (and in renders), light and dark; None keeps neutral grey.
    backdrop: Option<(&'static str, &'static str)>,
}

const ICON_BLUE: Scheme = Scheme {
    brand: ("#4392F7", "#285CC2"),
    accent: ("#2F6FDB", "#4A97F8"),
    on_hero: "#FFFFFF",
    glyph: "#285CC2",
    hero_in: "#FFFFFF80",
    alert: None,
    backdrop: None,
};

/// Summer Ocean Breeze: Strawberry Red, Honeydew, Frosted Blue, Steel Blue, Deep Space Blue.
const OCEAN: Scheme = Scheme {
    brand: ("#457B9D", "#1D3557"),
    accent: ("#457B9D", "#6FA3C2"),
    on_hero: "#F1FAEE",
    glyph: "#1D3557",
    hero_in: "#A8DADCB0",
    alert: Some(("#C42634", "#F0606B")),
    backdrop: Some(("#ECF5EC", "#13233A")),
};

fn scheme() -> Scheme {
    static S: std::sync::OnceLock<Scheme> = std::sync::OnceLock::new();
    *S.get_or_init(|| match std::env::var("OTO_SCHEME").as_deref() {
        Ok("ocean") => OCEAN,
        _ => ICON_BLUE,
    })
}

/// Everything a piece needs to paint itself. The popover is Liquid Glass; what sits on it is
/// translucent so the glass shows through.
struct Paint {
    t: Theme,
    k: Kit,
    dark: bool,
    accent: Color,
    green: Color,
    amber: Color,
    hair: f32,
    /// What the window shows where the native glass can't draw (headless renders, older macOS).
    backdrop: Color,
    /// A tile on the glass, its top-lit bevel, and a stronger fill for wells and chips. Tiles
    /// are a thin tint, as Control Center's modules are, so the glass shows through them.
    platter: Color,
    bevel: (Color, Color),
    well: Color,
    /// The lit hero, top to bottom.
    brand: (Color, Color),
    /// On the lit hero: the power button, Out bar and rings; the glyph; the In bar.
    on_hero: Color,
    glyph: Color,
    hero_in: Color,
}

impl Paint {
    fn new(dark: bool, scale: f32) -> Paint {
        let hex = |s: &str| rune::color::hex(s).unwrap_or(Color::BLACK);
        let w = Color::WHITE;
        let sc = scheme();
        let pick = |(l, d): (&str, &str)| hex(if dark { d } else { l });
        let brand = (hex(sc.brand.0), hex(sc.brand.1));
        let accent = pick(sc.accent);
        let t = tok::theme(dark);
        let (platter, bevel, well, field, backdrop, menu) = if dark {
            (
                w.alpha(0.045),
                (w.alpha(0.14), w.alpha(0.02)),
                w.alpha(0.08),
                w.alpha(0.08),
                rune::oklch(0.215, 0.012, 262.0),
                rune::oklch(0.25, 0.01, 262.0),
            )
        } else {
            (
                w.alpha(0.3),
                (w.alpha(0.85), w.alpha(0.1)),
                rune::oklcha(0.2, 0.01, 262.0, 0.07),
                w.alpha(0.6),
                rune::oklch(0.935, 0.008, 262.0),
                w,
            )
        };
        let backdrop = sc.backdrop.map_or(backdrop, pick);
        let mut k = Kit::neutral(dark);
        // Fields and neutral buttons sit on tiles, so they are translucent too; lists stay solid.
        k.c.bg_base = field;
        k.c.bg_button_neutral = field;
        k.c.layer_01 = menu;
        k.c.border_focus = accent;
        // A scheme with its own alert colour tints warnings with it, so a muted mic reads as one
        // colour from the hero line to the banner.
        if let Some(a) = sc.alert {
            let c = pick(a);
            k.c.warning.fg = c;
            k.c.warning.icon = c;
            k.c.warning.bg = c.alpha(if dark { 0.22 } else { 0.12 });
            k.c.warning.border = c.alpha(if dark { 0.4 } else { 0.3 });
        }
        k.c.bg_accent = accent;
        // Controls are capsules, as macOS 26 draws them: the 28 pt field and buttons round to
        // half their height, like the pills beside them, and list rows round with them.
        k.radius.sm = 12.0;
        k.radius.md = 14.0;
        k.radius.lg = 16.0;
        Paint {
            t,
            k,
            dark,
            accent,
            green: rune::oklch(if dark { 0.74 } else { 0.62 }, 0.17, 152.0),
            amber: sc.alert.map_or(
                // Light mode is darker so the small muted line keeps its contrast on a pale tile.
                rune::oklch(if dark { 0.8 } else { 0.5 }, 0.13, 65.0),
                pick,
            ),
            hair: 1.0 / scale,
            backdrop,
            platter,
            bevel,
            well,
            brand,
            on_hero: hex(sc.on_hero),
            glyph: hex(sc.glyph),
            hero_in: hex(sc.hero_in),
        }
    }
}

pub fn view(s: &State, cx: &mut Cx<State>) -> El {
    let p = Paint::new(cx.system_dark(), cx.env().scale);
    let mut layer = Vec::new();
    let body = if s.onboarding() {
        onboarding(s, &p)
    } else {
        match s.page {
            Page::Main => main_page(s, &p, &mut layer),
            Page::Settings => settings_page(s, &p, &mut layer),
        }
    };
    let mut window = col()
        .key("window")
        .w_full()
        .fill(p.backdrop)
        .padding(PAD)
        .gap(8.0)
        .child(body)
        .on_layout(|s: &mut State, r: [f32; 4]| s.size = (r[2], r[3]));
    if s.popover {
        // Liquid Glass behind the whole popover; the window clips itself to the same radius.
        window = window.radius(RADIUS).system_material(SystemMaterial::Glass);
    }
    el().key("root").w_full().child(window).children(layer)
}

/// A tile on the glass: translucent, continuous corners, lit along its top edge.
fn tile(p: &Paint, key: &str) -> El {
    col()
        .key(key)
        .w_full()
        .radius(TILE_RADIUS)
        .continuous_corners()
        .fill(p.platter)
        .border_gradient(p.hair, p.bevel.0, p.bevel.1)
}

fn card(p: &Paint, key: &str) -> El {
    tile(p, key).px(TILE_PX).py(6.0)
}

fn separator(p: &Paint, key: &str) -> El {
    el().key(key)
        .w_full()
        .h(p.hair)
        .fill(p.t.separator)
        .a11y_hidden()
}

fn caption(p: &Paint, key: &str, s: impl Into<String>) -> El {
    text(s.into())
        .key(key)
        .font(CAPTION)
        .color(p.t.text_secondary)
        .wrap()
        .w_full()
}

/// The app icon in small: the mark, white, on the icon's blue.
fn logo(p: &Paint, size: f32) -> El {
    el().key("logo")
        .size(size, size)
        .center()
        .radius(size * 0.26)
        .continuous_corners()
        .gradient(p.brand.0, p.brand.1)
        .border_gradient(p.hair, Color::WHITE.alpha(0.3), Color::WHITE.alpha(0.06))
        .a11y_hidden()
        .child(icon(&MARK, size * 0.78).color(Color::WHITE))
}

fn header(p: &Paint, title: &str, leading: Option<El>, trailing: Option<El>) -> El {
    let brand = leading.is_none();
    let mut r = row()
        .key("header")
        .w_full()
        .h(30.0)
        .px(2.0)
        .center_y()
        .gap(9.0);
    r = r.child(leading.unwrap_or_else(|| logo(p, 24.0)));
    r = r.child(
        text(title)
            .key("title")
            .font(if brand { BRAND } else { TITLE })
            .color(p.t.text)
            .grow(1.0)
            .shrink()
            .role("heading"),
    );
    if let Some(t) = trailing {
        r = r.child(t);
    }
    r
}

/// A capsule segmented control: a sliding raised pill over a recessed track. Segments are keyed
/// `segment{i}`, as rune-kit's are.
fn pills(
    p: &Paint,
    label: &str,
    labels: &[&str],
    selected: Option<usize>,
    width: f32,
    pick: fn(&mut State, usize),
) -> El {
    let h = 28.0;
    let inset = 2.0;
    let n = labels.len().max(1);
    let seg = width / n as f32;
    let mut r = row()
        .role("radiogroup")
        .label(label)
        .focus_group()
        .w(width)
        .h(h)
        .radius(h / 2.0)
        .fill(if p.dark {
            Color::BLACK.alpha(0.22)
        } else {
            p.well
        });
    if let Some(i) = selected {
        r = r.child(
            el().key("indicator")
                .absolute(seg * i as f32 + inset, inset)
                .size(seg - 2.0 * inset, h - 2.0 * inset)
                .radius((h - 2.0 * inset) / 2.0)
                .fill(if p.dark {
                    Color::WHITE.alpha(0.16)
                } else {
                    Color::WHITE
                })
                .border_gradient(
                    p.hair,
                    Color::WHITE.alpha(if p.dark { 0.22 } else { 1.0 }),
                    Color::WHITE.alpha(0.0),
                )
                .shadow(
                    1.0,
                    3.0,
                    Color::BLACK.alpha(if p.dark { 0.25 } else { 0.1 }),
                )
                .motion(motion::SNAPPY)
                .a11y_hidden(),
        );
    }
    for (i, l) in labels.iter().enumerate() {
        let on = selected == Some(i);
        r = r.child(
            row()
                .key(format!("segment{i}"))
                .role("radio")
                .label(*l)
                .checked(on)
                .focusable(true)
                .focus_selected(on)
                .position_in_set(i + 1, n)
                .focus_ring(2.0, 1.0, p.accent)
                .radius(h / 2.0)
                .hover(if on { Color::TRANSPARENT } else { p.t.hover })
                .w(seg)
                .h(h)
                .center()
                .px(6.0)
                .press_scale(0.97)
                .on_press(move |s: &mut State| pick(s, i))
                .child(
                    text(*l)
                        .key("label")
                        .font(PILL)
                        .color(if on { p.t.text } else { p.t.text_secondary })
                        .truncate()
                        .a11y_hidden(),
                ),
        );
    }
    r
}

/// A switch drawn here rather than rune-kit's: the kit's off track is the card's own colour in
/// the light theme, which leaves an off switch all but invisible on a card.
fn switch(
    p: &Paint,
    key: &str,
    label: &str,
    on: bool,
    (w, h): (f32, f32),
    toggle: Option<fn(&mut State)>,
) -> El {
    let inset = 2.0;
    let k = h - 2.0 * inset;
    let mut e = el()
        .key(key)
        .role("switch")
        .label(label)
        .checked(on)
        .focusable(toggle.is_some())
        .disabled(toggle.is_none())
        .focus_ring(2.0, 2.0, p.accent)
        .size(w, h)
        .radius(h / 2.0)
        .fill(if on {
            p.accent
        } else if p.dark {
            Color::WHITE.alpha(0.14)
        } else {
            p.t.control_off
        })
        .motion(motion::SNAPPY)
        .child(
            el().key("knob")
                .size(k, k)
                .absolute(if on { w - k - inset } else { inset }, inset)
                .radius(k / 2.0)
                .fill(p.t.knob)
                .shadow(1.0, 3.0, p.t.knob_shadow)
                .shadow(0.0, 0.5, p.t.edge)
                .a11y_hidden(),
        );
    e = match toggle {
        Some(f) => e.on_press(f),
        // Can't be changed right now: dimmed, and a press does nothing.
        None => e.opacity(0.45),
    };
    e
}

/// One ripple ring's brightness over a breath: swells in, then fades as the next ring takes it.
static BREATH: [(f32, f32); 4] = [(0.0, 0.0), (0.22, 1.0), (0.6, 0.25), (1.0, 0.0)];
/// Seconds per breath of the ripple.
const BREATH_S: f32 = 2.6;

/// The main on/off: a round power button. White on the lit hero, accent on a plain one (on but
/// muted), a thin outline when off. While Oto is cleaning a call, rings breathe out from it.
fn power_button(p: &Paint, label: &str, on: bool, lit: bool, breathing: bool) -> El {
    let d = 44.0;
    let button = el()
        .key("switch")
        .role("switch")
        .label(label)
        .checked(on)
        .focusable(true)
        .focus_ring(2.0, 2.0, if lit { Color::WHITE } else { p.accent })
        .size(d, d)
        .center()
        .radius(d / 2.0)
        .fill(if lit {
            p.on_hero
        } else if on {
            p.accent
        } else {
            Color::TRANSPARENT
        })
        .border(if on { 0.0 } else { 1.25 }, p.t.text_secondary.alpha(0.55))
        .shadow(
            if on { 3.0 } else { 0.0 },
            if on { 10.0 } else { 0.0 },
            Color::BLACK.alpha(if on { 0.22 } else { 0.0 }),
        )
        .hover(if on { Color::TRANSPARENT } else { p.t.hover })
        .press_scale(0.96)
        .motion(motion::SNAPPY)
        .on_press(|s: &mut State| s.toggle())
        .child(
            icon(&POWER, 20.0)
                .color(if lit {
                    p.glyph
                } else if on {
                    Color::WHITE
                } else {
                    p.t.text
                })
                .a11y_hidden(),
        );
    let mut wrap = el().key("power").size(d, d);
    if breathing {
        for (i, ring) in [54.0f32, 64.0, 74.0].into_iter().enumerate() {
            let at = (d - ring) / 2.0;
            wrap = wrap.child(
                el().key(format!("ring{i}"))
                    .absolute(at, at)
                    .size(ring, ring)
                    .radius(ring / 2.0)
                    .border(1.0, p.on_hero.alpha(0.55 - 0.12 * i as f32))
                    .cycle(Cycle {
                        period: BREATH_S,
                        phase: 0.16 * i as f32,
                        stops: &BREATH,
                        // Reduce Motion: the rings hold still, but still show Oto is working.
                        still: 0.6 - 0.15 * i as f32,
                        fps: 30.0,
                    })
                    .a11y_hidden(),
            );
        }
    }
    wrap.child(button)
}

/// A settings-sized switch; `enabled` false shows it dimmed and inert.
fn small_switch(
    p: &Paint,
    key: &str,
    label: &str,
    on: bool,
    enabled: bool,
    toggle: fn(&mut State),
) -> El {
    switch(p, key, label, on, (38.0, 22.0), enabled.then_some(toggle))
}

fn field_row(p: &Paint, key: &str, label: &str, control: El) -> El {
    row()
        .key(key)
        .w_full()
        .h(36.0)
        .center_y()
        .justify_between()
        .gap(12.0)
        .child(text(label).key("label").font(BODY).color(p.t.text).shrink())
        .child(control)
}

/// A small label over a full-width control.
fn labelled(p: &Paint, key: &str, label: &str, control: El) -> El {
    col()
        .key(key)
        .w_full()
        .gap(5.0)
        .child(
            text(label.to_uppercase())
                .key("label")
                .font(LABEL)
                .color(p.t.text_secondary)
                .a11y_hidden(),
        )
        .child(control)
}

/// A status colour's background, thinned so the glass reads through it.
fn tint(c: &StateColors) -> Color {
    c.bg.alpha(c.bg.a * 0.7)
}

fn error_banner(p: &Paint, msg: &str, dismiss: fn(&mut State)) -> El {
    let c = &p.k.c;
    row()
        .key("error")
        .role("alert")
        .w_full()
        .pl(12.0)
        .pr(6.0)
        .py(6.0)
        .gap(8.0)
        .center_y()
        .radius(12.0)
        .continuous_corners()
        .fill(tint(&c.danger))
        .border(p.hair, c.danger.border)
        .child(
            text(msg.to_owned())
                .key("message")
                .font(CAPTION)
                .color(c.danger.fg)
                .wrap()
                .grow(1.0)
                .shrink(),
        )
        .child(
            IconButton::new(p.k.glyph.close)
                .variant(IconButtonVariant::GhostMuted)
                .control_size(ControlSize::Small)
                .label("Dismiss")
                .on_press(dismiss)
                .build(&p.k)
                .key("dismiss"),
        )
}

/// A notice's button: its label and what it does.
type Action<'a> = (&'a str, fn(&mut State));

/// A message that isn't an error of the moment: what's missing, and the button that fixes it.
fn notice(p: &Paint, key: &str, c: &StateColors, msg: &str, action: Option<Action>) -> El {
    let mut r = row()
        .key(key)
        .role("status")
        .w_full()
        .pl(12.0)
        .pr(8.0)
        .py(8.0)
        .gap(10.0)
        .center_y()
        .radius(12.0)
        .continuous_corners()
        .fill(tint(c))
        .border(p.hair, c.border)
        .child(
            text(msg.to_owned())
                .key("message")
                .font(CAPTION)
                .color(c.fg)
                .wrap()
                .grow(1.0)
                .shrink(),
        );
    if let Some((label, f)) = action {
        r = r.child(
            Button::new(label)
                .control_size(ControlSize::Small)
                .on_press(f)
                .build(&p.k)
                .key(format!("{key}-action")),
        );
    }
    r
}

/// A banner button: its key, label, action and whether it can be pressed now.
type BannerButton = (&'static str, String, fn(&mut State), bool);

/// A notice with a title line, an optional detail, a row of buttons and an optional dismiss:
/// the update and driver banners.
fn banner(
    p: &Paint,
    (key, c, urgent): (&str, &StateColors, bool),
    msg: &str,
    detail: Option<String>,
    buttons: Vec<BannerButton>,
    dismiss: Option<fn(&mut State)>,
) -> El {
    let mut head = row().w_full().gap(8.0).items_start().child(
        text(msg.to_owned())
            .key(format!("{key}-message"))
            .font(LABEL)
            .color(c.fg)
            .wrap()
            .grow(1.0)
            .shrink(),
    );
    if let Some(f) = dismiss {
        head = head.child(
            IconButton::new(p.k.glyph.close)
                .variant(IconButtonVariant::GhostMuted)
                .control_size(ControlSize::Small)
                .label("Dismiss")
                .on_press(f)
                .build(&p.k)
                .key(format!("{key}-dismiss")),
        );
    }
    let mut b = col()
        .key(key)
        .role(if urgent { "alert" } else { "status" })
        .w_full()
        .pl(12.0)
        .pr(8.0)
        .py(8.0)
        .gap(6.0)
        .radius(12.0)
        .continuous_corners()
        .fill(tint(c))
        .border(p.hair, c.border)
        .child(head);
    if let Some(d) = detail {
        b = b.child(
            text(d)
                .key(format!("{key}-detail"))
                .font(CAPTION)
                .color(c.fg)
                .wrap()
                .w_full(),
        );
    }
    if !buttons.is_empty() {
        let mut r = row().key(format!("{key}-actions")).w_full().gap(8.0);
        for (k, label, f, enabled) in buttons {
            r = r.child(
                Button::new(label)
                    .control_size(ControlSize::Small)
                    .disabled(!enabled)
                    .on_press(f)
                    .build(&p.k)
                    .key(k),
            );
        }
        b = b.child(r);
    }
    b
}

/// Notes joined into one detail line; `None` when there are none.
fn notes<'a>(parts: impl IntoIterator<Item = Option<&'a str>>) -> Option<String> {
    let v: Vec<&str> = parts.into_iter().flatten().collect();
    (!v.is_empty()).then(|| v.join(" "))
}

/// The app update banner on the main page: an offer, its download, the restart, or a neutral
/// note when the update couldn't be verified.
fn update_banner(s: &State, p: &Paint) -> Option<El> {
    let c = &p.k.c.info;
    let self_updatable = matches!(s.install_mode, InstallMode::SelfUpdatable { .. });
    let start: fn(&mut State) = |s| s.start_update();
    let cancel: fn(&mut State) = |s| s.cancel_update();
    Some(match &s.update {
        UpdateState::Available(info) => {
            let mode_note = (!self_updatable)
                .then(|| install_mode_note(&s.install_mode))
                .flatten();
            let label = if self_updatable {
                "Update\u{2026}"
            } else {
                "Download\u{2026}"
            };
            banner(
                p,
                ("update", c, false),
                &format!("Oto {} is available.", info.version),
                notes([update_driver_note(info, s.loaded_driver()), mode_note]),
                vec![
                    ("update-action", label.into(), start, !s.candidate_pending),
                    (
                        "update-skip",
                        "Skip This Version".into(),
                        |s| s.skip_update(),
                        true,
                    ),
                ],
                Some(|s| s.dismiss_update()),
            )
        }
        UpdateState::Downloading { info, pct } => banner(
            p,
            ("update", c, false),
            &format!("Downloading Oto {}\u{2026} {pct}%", info.version),
            None,
            vec![("update-cancel", "Cancel".into(), cancel, true)],
            None,
        ),
        UpdateState::Verifying(info) => banner(
            p,
            ("update", c, false),
            &format!("Verifying Oto {}\u{2026}", info.version),
            None,
            vec![("update-cancel", "Cancel".into(), cancel, true)],
            None,
        ),
        UpdateState::Ready(staged) => {
            let blocked = s.apply_blocked();
            banner(
                p,
                ("update", c, false),
                &format!("Oto {} is ready to install.", staged.info.version),
                notes([blocked, update_driver_note(&staged.info, s.loaded_driver())]),
                vec![(
                    "update-action",
                    "Restart to Update".into(),
                    |s| s.restart_to_update(),
                    blocked.is_none(),
                )],
                None,
            )
        }
        UpdateState::Applying => banner(
            p,
            ("update", c, false),
            "Restarting to update\u{2026}",
            None,
            Vec::new(),
            None,
        ),
        UpdateState::UnrecognizedKey { .. } => banner(
            p,
            ("update", &p.k.c.info, false),
            UNRECOGNIZED_KEY_MESSAGE,
            None,
            vec![("update-action", "Download\u{2026}".into(), start, true)],
            Some(|s| s.dismiss_update()),
        ),
        UpdateState::Failed(msg) => banner(
            p,
            ("update", &p.k.c.danger, false),
            "Couldn\u{2019}t update Oto.",
            Some(msg.clone()),
            vec![(
                "update-retry",
                "Try Again".into(),
                |s| s.check_for_updates(true),
                true,
            )],
            Some(|s| s.dismiss_update()),
        ),
        _ => return None,
    })
}

/// The driver banners of the state table: an update to offer (urgent when the loaded driver is
/// too old), or a restart to finish one.
fn driver_banner(s: &State, p: &Paint) -> Option<El> {
    let ui = s.driver_ui();
    if ui == DriverUiState::UpdateNeedsReboot {
        return Some(notice(
            p,
            "driver-restart",
            &p.k.c.info,
            "Restart your Mac to finish updating the audio driver.",
            None,
        ));
    }
    if !ui.offers_update() {
        return None;
    }
    let blocked = s.install_blocked();
    let outcome = match &s.install {
        Install::Failed(m) | Install::Notice(m) => Some(m.as_str()),
        _ => None,
    };
    let (key, c, urgent, msg) = if ui == DriverUiState::Incompatible {
        (
            "driver-incompatible",
            &p.k.c.warning,
            true,
            "Oto\u{2019}s audio driver is too old for this version of Oto, so Oto Microphone is off until you update it.",
        )
    } else {
        (
            "driver-update",
            &p.k.c.info,
            false,
            "Audio driver update available. Asks for your password; audio will pause briefly in all apps.",
        )
    };
    let label = if s.install == Install::Running {
        "Updating\u{2026}"
    } else {
        "Update Driver\u{2026}"
    };
    Some(banner(
        p,
        (key, c, urgent),
        msg,
        notes([outcome, blocked]),
        vec![(
            "driver-update-action",
            label.into(),
            |s| s.start_install(),
            !s.install.busy() && blocked.is_none(),
        )],
        None,
    ))
}

/// Microphone access, when Oto doesn't have it: without it macOS feeds Oto silence.
fn mic_notice(s: &State, p: &Paint) -> Option<El> {
    let c = &p.k.c;
    match s.mic {
        Mic::Authorized => None,
        Mic::NotDetermined => Some(notice(
            p,
            "mic",
            &c.info,
            "Oto needs your permission to hear the microphone.",
            Some(("Allow\u{2026}", |s| s.allow_mic())),
        )),
        Mic::Denied => Some(notice(
            p,
            "mic",
            &c.warning,
            "Oto can\u{2019}t hear your microphone. Allow it in System Settings, under Privacy & Security.",
            Some(("Open Settings", |s| s.allow_mic())),
        )),
        Mic::Restricted => Some(notice(
            p,
            "mic",
            &c.warning,
            "Microphone access is restricted on this Mac, so Oto can\u{2019}t hear it.",
            None,
        )),
    }
}

/// Voice Isolation can't run, so Oto Microphone is muted: what happened, and the way out.
fn degraded_banner(p: &Paint, title: &str, msg: &str) -> El {
    let c = &p.k.c.warning;
    col()
        .key("degraded")
        .role("alert")
        .w_full()
        .px(12.0)
        .py(10.0)
        .gap(8.0)
        .radius(12.0)
        .continuous_corners()
        .fill(tint(c))
        .border(p.hair, c.border)
        .child(
            text(title.to_owned())
                .key("degraded-title")
                .font(LABEL)
                .color(c.fg)
                .wrap()
                .w_full(),
        )
        .child(
            text(msg.to_owned())
                .key("degraded-message")
                .font(CAPTION)
                .color(c.fg)
                .wrap()
                .w_full(),
        )
        .child(
            row().w_full().child(
                Button::new("Switch to Noise Cancellation")
                    .control_size(ControlSize::Small)
                    .on_press(|s: &mut State| s.switch_to_noise_cancellation())
                    .build(&p.k)
                    .key("degraded-action"),
            ),
        )
}

/// "Hear the Difference": a button, then the recording, then raw-or-cleaned playback.
fn hear_card(s: &State, p: &Paint) -> El {
    let title = text("Hear the difference")
        .key("hear-title")
        .font(TITLE)
        .color(p.t.text)
        .a11y_hidden();
    let mut c = card(p, "hear").py(TILE_PX).gap(6.0).child(title);
    match &s.hear {
        HearTest::Idle | HearTest::Failed(_) => {
            let blocked = s.hear_blocked();
            let retry = matches!(s.hear, HearTest::Failed(_));
            if let HearTest::Failed(msg) = &s.hear {
                c = c.child(
                    text(msg.clone())
                        .key("hear-error")
                        .font(CAPTION)
                        .color(p.k.c.danger.fg)
                        .wrap()
                        .w_full(),
                );
            }
            c = c.child(caption(
                p,
                "hear-about",
                blocked.unwrap_or(
                    "Record 5 seconds of your voice, then compare the original with Oto\u{2019}s version.",
                ),
            ));
            c.child(
                row().w_full().child(
                    Button::new(if retry { "Try Again" } else { "Record 5 s" })
                        .variant(ButtonVariant::Outline)
                        .disabled(blocked.is_some())
                        .on_press(|s: &mut State| s.start_hear_test())
                        .build(&p.k)
                        .key("hear-start"),
                ),
            )
        }
        HearTest::Recording { processing } => c
            .child(caption(
                p,
                "hear-progress",
                if *processing {
                    "Processing\u{2026}"
                } else {
                    "Recording 5 s\u{2026} Speak normally, with your usual background."
                },
            ))
            .child(
                row().w_full().child(
                    Button::new(if *processing { "Cancel" } else { "Stop" })
                        .variant(ButtonVariant::Outline)
                        .on_press(|s: &mut State| s.stop_hear_test())
                        .build(&p.k)
                        .key("hear-stop"),
                ),
            ),
        HearTest::Ready {
            cleaned,
            is_bluetooth_hfp,
        } => {
            let ab = pills(
                p,
                "Playback",
                &["Original", "Cleaned"],
                Some(usize::from(*cleaned)),
                CARD_INNER,
                |s, i| s.set_hear_cleaned(i == 1),
            );
            c = c.child(el().key("hear-ab").w_full().child(ab));
            if *is_bluetooth_hfp {
                c = c.child(caption(
                    p,
                    "hear-hfp",
                    "Recorded in Bluetooth hands-free mode, so this take sounds narrower than your headset does for music.",
                ));
            }
            c.child(
                row()
                    .key("hear-actions")
                    .w_full()
                    .gap(8.0)
                    .justify_between()
                    .child(
                        Button::new("Record Again")
                            .variant(ButtonVariant::Ghost)
                            .disabled(s.hear_blocked().is_some())
                            .on_press(|s: &mut State| s.start_hear_test())
                            .build(&p.k)
                            .key("hear-again"),
                    )
                    .child(
                        Button::new("Done")
                            .variant(ButtonVariant::Outline)
                            .on_press(|s: &mut State| s.stop_hear_test())
                            .build(&p.k)
                            .key("hear-done"),
                    ),
            )
        }
    }
}

/// Stands in for the controls while there's no engine for them to control.
fn engine_card(p: &Paint, msg: &str) -> El {
    card(p, "engine")
        .py(14.0)
        .gap(6.0)
        .child(
            text("Oto isn\u{2019}t running")
                .key("engine-title")
                .font(TITLE)
                .color(p.t.text),
        )
        .child(caption(p, "engine-message", msg))
        .child(
            row().w_full().pt(6.0).child(
                Button::new("Try Again")
                    .variant(ButtonVariant::Contrast)
                    .on_press(|s: &mut State| s.retry_engine())
                    .build(&p.k)
                    .key("retry"),
            ),
        )
}

fn main_page(s: &State, p: &Paint, layer: &mut Vec<El>) -> El {
    let on = s.prefs.enabled;
    let gear = IconButton::new(&SLIDERS)
        .variant(IconButtonVariant::GhostMuted)
        .label("Settings")
        .on_press(State::open_settings)
        .build(&p.k)
        .key("settings");

    if let EngineState::Failed(msg) = &s.engine {
        let mut page = col()
            .key("main")
            .w_full()
            .gap(12.0)
            .child(header(p, "Oto", None, Some(gear)))
            .child(engine_card(p, msg));
        if let Some(e) = &s.error {
            page = page.child(error_banner(p, e, |s| s.dismiss_error()));
        }
        return page.child(status_line(s, p));
    }

    let isolating = s.isolating();
    let hero_title = if isolating {
        "Voice Isolation"
    } else {
        "Noise Cancellation"
    };
    let state_line = if !on {
        "Off \u{00b7} your microphone passes through"
    } else if s.degraded.is_some() {
        "Muted until it can run again"
    } else if isolating && !s.offers_isolation() {
        "Muted \u{00b7} switch to Noise Cancellation"
    } else if s.clients_active && isolating {
        "Keeping only your voice"
    } else if s.clients_active {
        "Cleaning your voice"
    } else {
        "On \u{00b7} ready for your next call"
    };
    let hero = hero(s, p, hero_title, state_line);

    let options: Vec<String> = s.device_options().into_iter().map(|(_, l)| l).collect();
    let current = s.device_label();
    let device = Select::new("device", &options)
        .appearance(SelectAppearance::Field)
        .width(CARD_INNER)
        .label("Microphone")
        .current(Some(&current))
        .open(s.open == Some(Open::Device))
        .on_toggle(|s: &mut State| s.toggle_open(Open::Device))
        .on_close(|s: &mut State| s.open = None)
        .on_choose(|s: &mut State, v: &str, _| s.choose_device(v))
        .placement(Placement::BottomStart)
        .build(&p.k, layer);
    if s.open == Some(Open::Device)
        && let Some(list) = layer.pop()
    {
        // As wide as the field, so device names don't truncate in a narrow list.
        layer.push(list.w(CARD_INNER).radius(MENU_RADIUS).continuous_corners());
    }
    let mic_row = labelled(p, "mic-row", "Microphone", device);
    let labels: Vec<&str> = STRENGTHS.iter().map(|(_, l)| *l).collect();
    let strength = pills(
        p,
        "Strength",
        &labels,
        STRENGTHS.iter().position(|(v, _)| *v == s.prefs.strength),
        CARD_INNER,
        |s, i| s.set_strength(STRENGTHS[i].0),
    );
    let mut controls = card(p, "controls").py(TILE_PX).gap(10.0).child(mic_row);
    // Two modes only when this build has Voice Isolation. The Noise Cancellation model is a
    // setting.
    if s.offers_isolation() {
        let labels: Vec<&str> = MODES.iter().map(|(_, l)| *l).collect();
        let mode = pills(
            p,
            "Mode",
            &labels,
            MODES.iter().position(|(m, _)| *m == s.mode()),
            CARD_INNER,
            |s, i| s.set_mode(MODES[i].0),
        );
        let mut row = labelled(p, "mode-row", "Mode", mode);
        if isolating {
            row = row.child(caption(
                p,
                "mode-about",
                "Removes background noise and nearby voices. If it can\u{2019}t run, your microphone goes silent instead of passing them through.",
            ));
        }
        controls = controls.child(row);
    }
    controls = controls.child(labelled(p, "strength-row", "Strength", strength));

    let mut page = col()
        .key("main")
        .w_full()
        .gap(12.0)
        .child(header(p, "Oto", None, Some(gear)))
        .child(hero);
    if !s.driver_installed && s.driver_known {
        page = page.child(notice(
            p,
            "driver-missing",
            &p.k.c.warning,
            "The audio driver isn\u{2019}t installed, so there is no Oto Microphone yet.",
            Some(("Set Up\u{2026}", |s| s.skipped_onboarding = false)),
        ));
    } else if let Some(n) = mic_notice(s, p) {
        page = page.child(n);
    }
    if let Some(b) = driver_banner(s, p) {
        page = page.child(b);
    }
    if let Some(b) = update_banner(s, p) {
        page = page.child(b);
    }
    if on && let Some(msg) = &s.degraded {
        page = page.child(degraded_banner(
            p,
            "Voice Isolation is muting your microphone",
            msg,
        ));
    } else if s.isolating() && !s.offers_isolation() {
        let (title, msg) = if on {
            (
                "Voice Isolation is muting your microphone",
                "Voice Isolation isn\u{2019}t included in this copy of Oto, so Oto Microphone stays muted until you switch modes.",
            )
        } else {
            (
                "Voice Isolation is unavailable",
                "Voice Isolation isn\u{2019}t included in this copy of Oto. Processing is off (bypass); turning it on will mute Oto Microphone until you switch modes.",
            )
        };
        page = page.child(degraded_banner(p, title, msg));
    }
    if let Some(msg) = &s.model_notice {
        page = page.child(notice(
            p,
            "model-notice",
            &p.k.c.info,
            msg,
            Some(("OK", |s| s.model_notice = None)),
        ));
    }
    page = page.child(controls);
    if let Some(e) = &s.error {
        page = page.child(error_banner(p, e, |s| s.dismiss_error()));
    }
    if s.driver_installed && !s.clients_active {
        page = page.child(caption(
            p,
            "hint",
            if s.prefs.system_microphone {
                "Oto Microphone is the system microphone. Apps that pick their own need \u{201c}Oto Microphone\u{201d} chosen as the microphone."
            } else {
                "In your call app, choose \u{201c}Oto Microphone\u{201d} as the microphone."
            },
        ));
    }
    if HEAR_TEST {
        page = page.child(hear_card(s, p));
    }
    page.child(status_line(s, p))
}

/// The on/off tile. Lit in the app icon's blue while Oto runs, with the noise-in and
/// voice-out levels; a plain tile while it's off.
fn hero(s: &State, p: &Paint, title: &str, state_line: &str) -> El {
    let on = s.prefs.enabled;
    // On but muted (Voice Isolation can't run): not lit, and the state line says so in amber.
    let muted = on && (s.degraded.is_some() || (s.isolating() && !s.offers_isolation()));
    let lit = on && !muted;
    let ink = if lit { p.on_hero } else { p.t.text };
    let ink2 = if lit {
        p.on_hero
    } else if muted {
        p.amber
    } else {
        p.t.text_secondary
    };
    let mut t = tile(p, "hero").padding(14.0).gap(12.0);
    if lit {
        t = t
            .linear_gradient(
                std::f32::consts::PI * 0.88,
                // Tinted glass, not paint: the material shows through the hero's colour.
                [
                    (0.0, p.brand.0.mix(p.brand.1, 0.7).alpha(0.94)),
                    (1.0, p.brand.1.alpha(0.94)),
                ],
            )
            .border_gradient(p.hair, Color::WHITE.alpha(0.42), Color::WHITE.alpha(0.06))
            .shadow(8.0, 22.0, p.brand.1.alpha(if p.dark { 0.45 } else { 0.32 }));
    }
    t = t.child(
        row()
            .key("hero-top")
            .w_full()
            .center_y()
            .gap(12.0)
            .child(
                col()
                    .grow(1.0)
                    .shrink()
                    .gap(3.0)
                    .child(text(title).key("hero-title").font(HERO).color(ink))
                    .child(
                        text(state_line)
                            .key("hero-sub")
                            .font(CAPTION)
                            .color(ink2)
                            .wrap()
                            .w_full(),
                    ),
            )
            .child(power_button(p, title, on, lit, lit && s.clients_active)),
    );
    // Muted means no stream feeds the meter, which would hold its last samples, so no levels.
    if !lit {
        return t;
    }
    let (raw_c, clean_c, label_c) = (p.hero_in, p.on_hero, p.on_hero.alpha(0.72));
    let mut levels = col()
        .key("meter")
        .w_full()
        .gap(6.0)
        .child(meter_row(
            p,
            "raw",
            "In",
            "Input level",
            &s.meter.raw,
            (raw_c, label_c),
        ))
        .child(meter_row(
            p,
            "clean",
            "Out",
            "Cleaned level",
            &s.meter.clean,
            (clean_c, label_c),
        ));
    if !s.clients_active {
        levels = levels.child(
            text("Levels show while an app is using Oto Microphone.")
                .key("meter-hint")
                .font(CAPTION)
                .color(ink2)
                .wrap()
                .w_full(),
        );
    }
    t.child(levels)
}

fn meter_row(
    _p: &Paint,
    key: &str,
    label: &str,
    a11y: &str,
    live: &rune::Live<[f32; 4]>,
    (color, ink): (Color, Color),
) -> El {
    row()
        .key(key)
        .w_full()
        .gap(10.0)
        .center_y()
        .child(
            text(label)
                .key("label")
                .font(CAPTION_NUM)
                .color(ink)
                .grow(1.0)
                .a11y_hidden(),
        )
        .child(
            el().key("track")
                .role("meter")
                .label(a11y)
                .size(BAR_W, BAR_H)
                .radius(BAR_H / 2.0)
                .fill(Color::BLACK.alpha(0.18))
                .child(
                    el().key("fill")
                        .absolute(0.0, 0.0)
                        .size(0.0, BAR_H)
                        .radius(BAR_H / 2.0)
                        .fill(color)
                        .live_rect(live),
                ),
        )
}

fn status_line(s: &State, p: &Paint) -> El {
    let dot = if !matches!(s.engine, EngineState::Running) || !s.driver_known {
        p.t.text_secondary.alpha(0.5)
    } else if !s.driver_installed
        || s.mic.blocked()
        || (s.prefs.enabled && (s.degraded.is_some() || (s.isolating() && !s.offers_isolation())))
    {
        p.amber
    } else if s.prefs.enabled && s.clients_active {
        p.green
    } else {
        p.t.text_secondary.alpha(0.5)
    };
    let mut r = row()
        .key("status")
        .w_full()
        .px(6.0)
        .pt(2.0)
        .gap(7.0)
        .center_y()
        .child(
            el().key("dot")
                .size(6.0, 6.0)
                .radius(3.0)
                .fill(dot)
                .shadow(0.0, 4.0, dot.alpha(0.6))
                .a11y_hidden(),
        )
        .child(
            text(s.status())
                .key("status-text")
                .font(CAPTION)
                .color(p.t.text_secondary)
                .grow(1.0)
                .shrink()
                .truncate(),
        );
    let meta: Vec<String> = s.running_model().into_iter().chain(s.cpu()).collect();
    if !meta.is_empty() {
        r = r.child(
            text(meta.join(" \u{00b7} "))
                .key("model")
                .font(CAPTION_NUM)
                .color(p.t.text_secondary.alpha(0.85)),
        );
    }
    r
}

/// The Noise Cancellation model picker, when this build has more than one.
fn models_card(s: &State, p: &Paint, layer: &mut Vec<El>) -> Option<El> {
    if !s.offers_tiers() {
        return None;
    }
    let options = s.tier_options();
    let model = Select::new("model-picker", &options)
        .appearance(SelectAppearance::Field)
        .width(CARD_INNER)
        .label("Noise Cancellation model")
        .current(Some(s.tier_label()))
        .open(s.open == Some(Open::Model))
        .on_toggle(|s: &mut State| s.toggle_open(Open::Model))
        .on_close(|s: &mut State| s.open = None)
        .on_choose(|s: &mut State, v: &str, _| s.choose_tier(v))
        .placement(Placement::BottomStart)
        .build(&p.k, layer);
    if s.open == Some(Open::Model)
        && let Some(list) = layer.pop()
    {
        layer.push(list.w(CARD_INNER).radius(MENU_RADIUS).continuous_corners());
    }
    Some(
        card(p, "models")
            .py(12.0)
            .gap(8.0)
            .child(labelled(p, "model-row", "Noise Cancellation model", model))
            .child(caption(
                p,
                "model-about",
                if s.isolating() {
                    "Used when you switch back to Noise Cancellation."
                } else {
                    "DeepFilterNet3 suits most calls. RNNoise uses the least CPU."
                },
            )),
    )
}

fn settings_page(s: &State, p: &Paint, layer: &mut Vec<El>) -> El {
    let back = IconButton::new(&CHEVRON_LEFT)
        .variant(IconButtonVariant::GhostMuted)
        .label("Back")
        .on_press(|s: &mut State| {
            s.open = None;
            s.page = Page::Main;
        })
        .build(&p.k)
        .key("back");
    let login = small_switch(
        p,
        "login",
        "Start at login",
        s.prefs.start_at_login,
        s.login_available,
        |s| s.set_start_at_login(!s.prefs.start_at_login),
    );
    let mut general = card(p, "general").child(field_row(p, "login-row", "Start at login", login));
    if !s.login_available {
        general = general.child(caption(p, "login-about", crate::login::UNAVAILABLE).pb(8.0));
    }
    let system_mic = small_switch(
        p,
        "system-mic",
        "Use Oto as the system microphone",
        s.prefs.system_microphone,
        s.driver_installed,
        |s| s.set_system_mic(!s.prefs.system_microphone),
    );
    general = general
        .child(separator(p, "sep-login"))
        .child(field_row(
            p,
            "system-mic-row",
            "Use Oto as the system microphone",
            system_mic,
        ))
        .child(
            caption(
                p,
                "system-mic-about",
                if s.driver_installed {
                    "Apps that follow the system setting get Oto Microphone. Your previous microphone comes back when you turn this off or quit Oto."
                } else {
                    "Needs the audio driver."
                },
            )
            .pb(8.0),
        );

    let busy = s.install.busy();
    let blocked = s.install_blocked();
    let ui = s.driver_ui();
    let present = s.driver_installed || s.driver_on_disk;
    let (driver_label, driver_about) = match ui {
        DriverUiState::NotInstalled => ("Audio driver not installed", None),
        DriverUiState::InstalledNeedsReboot => ("Audio driver needs a restart", None),
        DriverUiState::UpToDate | DriverUiState::NewerLoaded => ("Audio driver installed", None),
        DriverUiState::OnDiskOutdatedOrMissing => (
            "Audio driver needs repair",
            Some(
                "The copy on disk is missing or outdated. Reinstall to keep Oto Microphone after a restart.",
            ),
        ),
        DriverUiState::UpdateAvailable => ("Audio driver update available", None),
        DriverUiState::Incompatible => ("Audio driver needs an update", None),
        DriverUiState::UpdateNeedsReboot => (
            "Audio driver update needs a restart",
            Some("Restart your Mac to finish updating the audio driver."),
        ),
        DriverUiState::RemovedNeedsReboot => (
            "Audio driver removed",
            Some("Restart your Mac to finish removing the driver."),
        ),
        DriverUiState::Unknown if present => ("Audio driver installed", None),
        DriverUiState::Unknown => ("Audio driver not installed", None),
    };
    let driver_row = if s.newer_driver_present() {
        row().key("driver-row").w_full().h(40.0).center_y().child(
            text(driver_label)
                .key("label")
                .font(BODY)
                .color(p.t.text)
                .shrink(),
        )
    } else {
        let install = Button::new(match &s.install {
            Install::Running if ui.offers_update() => "Updating\u{2026}",
            Install::Running => "Installing\u{2026}",
            _ if ui.offers_update() => "Update\u{2026}",
            _ if present => "Reinstall\u{2026}",
            _ => "Install\u{2026}",
        })
        .disabled(busy || blocked.is_some())
        .on_press(|s: &mut State| s.start_install())
        .build(&p.k)
        .key("install");
        field_row(p, "driver-row", driver_label, install)
    };
    let mut driver = card(p, "driver").child(driver_row);
    // An install's own result says what's left to do.
    let about = if matches!(s.install, Install::Notice(_)) {
        None
    } else {
        driver_about
    };
    driver = driver.child(
        caption(
            p,
            "driver-about",
            notes([about, blocked.or(Some(DRIVER_PAUSE))]).unwrap_or_default(),
        )
        .pb(8.0),
    );
    if s.driver_installed || s.driver_on_disk {
        let uninstall = Button::new(if s.install == Install::Removing {
            "Removing\u{2026}"
        } else {
            "Uninstall\u{2026}"
        })
        .disabled(busy || blocked.is_some())
        .on_press(|s: &mut State| s.start_uninstall())
        .build(&p.k)
        .key("uninstall");
        driver = driver.child(separator(p, "sep-uninstall")).child(field_row(
            p,
            "uninstall-row",
            "Remove the driver",
            uninstall,
        ));
    }
    match &s.install {
        Install::Failed(msg) => {
            driver = driver.child(
                text(msg.clone())
                    .key("install-error")
                    .font(CAPTION)
                    .color(p.k.c.danger.fg)
                    .wrap()
                    .w_full()
                    .pb(8.0),
            );
        }
        Install::Notice(msg) => {
            driver = driver.child(
                text(msg.clone())
                    .key("install-notice")
                    .font(CAPTION)
                    .color(p.t.text)
                    .wrap()
                    .w_full()
                    .pb(8.0),
            );
        }
        _ => {}
    }

    let mut page = col()
        .key("settings-page")
        .w_full()
        .gap(12.0)
        .child(header(p, "Settings", Some(back), None))
        .children(models_card(s, p, layer))
        .child(general)
        .child(updates_card(s, p))
        .child(driver);
    if let Some(e) = &s.error {
        page = page.child(error_banner(p, e, |s| s.dismiss_error()));
    }
    page.child(
        row()
            .key("about")
            .w_full()
            .pl(4.0)
            .center_y()
            .justify_end()
            .child(
                Button::new("Quit Oto")
                    .variant(ButtonVariant::Ghost)
                    .on_press_tx(|s: &mut State, tx| {
                        s.shutdown();
                        tx.quit();
                    })
                    .build(&p.k)
                    .key("quit"),
            ),
    )
}

/// Settings' Updates card: automatic checks, the version with the one button that moves the
/// update on, what's happening, and the release notes.
fn updates_card(s: &State, p: &Paint) -> El {
    let auto = small_switch(
        p,
        "auto-update",
        "Check for updates automatically",
        s.prefs.check_for_updates,
        !s.candidate_pending,
        |s| s.set_check_for_updates(!s.prefs.check_for_updates),
    );
    let self_updatable = matches!(s.install_mode, InstallMode::SelfUpdatable { .. });
    let noop: fn(&mut State) = |_| {};
    let (label, action, enabled): (&str, fn(&mut State), bool) = match &s.update {
        UpdateState::Checking { .. } => ("Checking\u{2026}", noop, false),
        UpdateState::Available(_) if self_updatable => {
            ("Update\u{2026}", |s| s.start_update(), !s.candidate_pending)
        }
        UpdateState::Available(_) | UpdateState::UnrecognizedKey { .. } => {
            ("Download\u{2026}", |s| s.start_update(), true)
        }
        UpdateState::Downloading { .. } | UpdateState::Verifying(_) => {
            ("Cancel", |s| s.cancel_update(), true)
        }
        UpdateState::Ready(_) => (
            "Restart to Update",
            |s| s.restart_to_update(),
            s.apply_blocked().is_none(),
        ),
        UpdateState::Applying => ("Restarting\u{2026}", noop, false),
        _ => (
            "Check for Updates",
            |s| s.check_for_updates(true),
            !s.candidate_pending,
        ),
    };
    let button = Button::new(label)
        .control_size(ControlSize::Small)
        .disabled(!enabled)
        .on_press(action)
        .build(&p.k)
        .key("update-button");
    let mode_note = install_mode_note(&s.install_mode);
    let status: Option<String> = match &s.update {
        UpdateState::Idle => mode_note.map(str::to_owned),
        UpdateState::Checking { .. } => Some("Checking\u{2026}".into()),
        UpdateState::UpToDate => Some("Up to date".into()),
        UpdateState::Available(i) => notes([
            Some(format!("Oto {} is available.", i.version).as_str()),
            (!self_updatable).then_some(mode_note).flatten(),
        ]),
        UpdateState::Downloading { pct, .. } => Some(format!("Downloading {pct}%\u{2026}")),
        UpdateState::Verifying(_) => Some("Verifying\u{2026}".into()),
        UpdateState::Ready(_) => notes([Some("Ready to install."), s.apply_blocked()]),
        UpdateState::Applying => Some("Restarting\u{2026}".into()),
        UpdateState::UnrecognizedKey { .. } => Some(UNRECOGNIZED_KEY_MESSAGE.into()),
        UpdateState::IncompatibleOs { version, min_os } => {
            Some(format!("Oto {version} needs macOS {min_os} or later."))
        }
        UpdateState::Failed(e) => Some(e.clone()),
    };
    let mut c = card(p, "updates")
        .child(field_row(
            p,
            "auto-update-row",
            "Check for updates automatically",
            auto,
        ))
        .child(separator(p, "sep-updates"))
        .child(field_row(
            p,
            "update-row",
            concat!("Oto ", env!("CARGO_PKG_VERSION")),
            button,
        ));
    if let Some(msg) = status {
        let failed = matches!(s.update, UpdateState::Failed(_));
        c = c.child(
            text(msg)
                .key("update-status")
                .font(CAPTION)
                .color(if failed {
                    p.k.c.danger.fg
                } else {
                    p.t.text_secondary
                })
                .wrap()
                .w_full()
                .pb(8.0),
        );
    }
    if let Some(info) = s.update.info().filter(|i| openable_url(&i.notes_url)) {
        c = c.child(
            row().w_full().pb(8.0).child(
                Button::new(format!("Release Notes for {}", info.version))
                    .variant(ButtonVariant::Ghost)
                    .control_size(ControlSize::Small)
                    .on_press(|s: &mut State| s.open_release_notes())
                    .build(&p.k)
                    .key("release-notes"),
            ),
        );
    }
    c
}

/// One setup step: a numbered badge that turns into a tick when the step is really done.
fn step(p: &Paint, n: usize, title: &str, detail: &str, done: bool, trailing: Option<El>) -> El {
    let badge = el()
        .key("badge")
        .size(20.0, 20.0)
        .center()
        .radius(10.0)
        .fill(if done { p.green } else { p.t.control_off })
        .a11y_hidden()
        .child(if done {
            icon(p.k.glyph.check, 12.0).color(Color::WHITE)
        } else {
            text(n.to_string())
                .font(TextStyle::new(11.0, 600.0).lh(13.0))
                .color(p.t.text)
        });
    let mut r = row()
        .key(format!("step{n}"))
        .w_full()
        .py(8.0)
        .gap(10.0)
        .items_start()
        .child(badge)
        .child(
            col()
                .grow(1.0)
                .shrink()
                .gap(2.0)
                .child(
                    text(title.to_owned())
                        .key("title")
                        .font(TextStyle::new(13.0, 600.0).lh(20.0))
                        .color(p.t.text)
                        .wrap()
                        .w_full(),
                )
                .child(caption(p, "detail", detail.to_owned())),
        );
    if let Some(t) = trailing {
        r = r.child(t);
    }
    r
}

fn onboarding(s: &State, p: &Paint) -> El {
    let restart = !s.driver_installed && s.driver_on_disk;
    let system_mic = small_switch(
        p,
        "system-mic",
        "Use Oto as the system microphone",
        s.prefs.system_microphone,
        s.driver_installed,
        |s| s.set_system_mic(!s.prefs.system_microphone),
    );
    let steps = card(p, "steps")
        .py(4.0)
        .child(step(
            p,
            1,
            "Install the audio driver",
            if s.driver_installed {
                "\u{201c}Oto Microphone\u{201d} is installed."
            } else if restart {
                "Installed. Restart your Mac to finish."
            } else {
                "Adds \u{201c}Oto Microphone\u{201d}. Asks for your password. Audio will pause briefly in all apps (or finish on your next restart if audio is playing)."
            },
            s.driver_installed,
            None,
        ))
        .child(separator(p, "sep1"))
        .child(step(
            p,
            2,
            "Allow microphone access",
            match s.mic {
                Mic::Authorized => "Oto can hear your microphone.",
                Mic::NotDetermined => {
                    "macOS asks once. Oto only listens while an app is using Oto Microphone."
                }
                Mic::Denied => {
                    "Turned off. Allow Oto in System Settings, under Privacy & Security, Microphone."
                }
                Mic::Restricted => "Microphone access is restricted on this Mac.",
            },
            s.mic == Mic::Authorized,
            None,
        ))
        .child(separator(p, "sep2"))
        .child(step(
            p,
            3,
            "Use Oto as the system microphone",
            "Optional. Or choose \u{201c}Oto Microphone\u{201d} in Zoom, Meet, Discord or any app.",
            s.prefs.system_microphone && s.driver_installed,
            Some(system_mic),
        ));

    type NextAction = (&'static str, bool, fn(&mut State));
    let newer_on_disk = !s.driver_installed && s.newer_driver_present();
    // The one button that moves setup on: whichever step is next.
    let next_action: Option<NextAction> = if newer_on_disk {
        None
    } else if !s.driver_installed {
        Some((
            match (&s.install, restart) {
                (Install::Running, _) => "Installing\u{2026}",
                (_, true) => "Install Again",
                (_, false) => "Install Driver",
            },
            !s.install.busy(),
            |s| s.start_install(),
        ))
    } else {
        Some(match s.mic {
            Mic::NotDetermined => ("Allow Microphone", !s.mic_requesting, |s| s.allow_mic()),
            Mic::Denied => ("Open Settings", true, |s| s.allow_mic()),
            Mic::Authorized | Mic::Restricted => ("Done", true, |s| s.leave_onboarding()),
        })
    };

    let mut page = col()
        .key("onboarding")
        .w_full()
        .gap(12.0)
        .child(header(p, "Oto", None, None))
        .child(
            col()
                .key("intro")
                .w_full()
                .gap(6.0)
                .pt(4.0)
                .child(text("Set up Oto Microphone").key("h1").font(H1).color(p.t.text).role("heading"))
                .child(
                    text("Oto adds a virtual microphone that removes background noise. Any app that can pick a mic can use it.")
                        .key("lede")
                        .font(BODY)
                        .color(p.t.text_secondary)
                        .wrap()
                        .w_full(),
                ),
        )
        .child(steps);
    match &s.install {
        Install::Failed(msg) => {
            page = page.child(error_banner(p, msg, |s| s.install = Install::Idle));
        }
        Install::Notice(msg) => page = page.child(notice(p, "restart", &p.k.c.info, msg, None)),
        _ => {}
    }
    if newer_on_disk && !matches!(s.install, Install::Notice(_)) {
        let msg = format!(
            "A newer Oto audio driver is installed. {}",
            crate::driver::REBOOT_MESSAGE
        );
        page = page.child(notice(p, "restart", &p.k.c.info, &msg, None));
    }
    if let EngineState::Failed(msg) = &s.engine {
        page = page.child(engine_card(p, msg));
    }
    if let Some(e) = &s.error {
        page = page.child(error_banner(p, e, |s| s.dismiss_error()));
    }
    let mut actions = row()
        .key("actions")
        .w_full()
        .gap(8.0)
        .pt(4.0)
        .center_y()
        .justify_between();
    actions = if matches!(next_action, Some(("Done", _, _))) {
        // Nothing left to put off.
        actions.child(el().key("no-skip").a11y_hidden())
    } else {
        actions.child(
            Button::new("Not Now")
                .variant(ButtonVariant::Ghost)
                .control_size(ControlSize::Large)
                .on_press(|s: &mut State| s.leave_onboarding())
                .build(&p.k)
                .key("skip"),
        )
    };
    if let Some((label, enabled, next)) = next_action {
        actions = actions.child(
            Button::new(label)
                .variant(ButtonVariant::Contrast)
                .control_size(ControlSize::Large)
                .disabled(!enabled)
                .on_press(next)
                .build(&p.k)
                .key("next"),
        );
    }
    page.child(actions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefs::Prefs;
    use crate::state::Request;
    use oto_engine::{Command, Strength, Tier};
    use rune::{Env, Runtime};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn runtime(s: State) -> Runtime<State> {
        let mut rt = Runtime::new(s, view, (WIDTH, 900.0), Env::new(true, 2.0));
        rt.frame(0.0);
        rt
    }

    fn installed() -> State {
        let prefs = Prefs {
            onboarded: true,
            ..Prefs::default()
        };
        let mut s = State::new(prefs, true);
        s.link = crate::state::Link::Record(Vec::new());
        s
    }

    fn click(rt: &mut Runtime<State>, key: &str) {
        let (x, y) = rt.find(key).unwrap_or_else(|| panic!("no element {key}"));
        rt.pointer_down(x, y);
        rt.pointer_up(x, y);
        rt.frame(1.0 / 60.0);
    }

    fn requests(rt: &Runtime<State>) -> Vec<Request> {
        match &rt.state.link {
            crate::state::Link::Record(v) => v.clone(),
            _ => Vec::new(),
        }
    }

    fn sent(rt: &Runtime<State>) -> Vec<Command> {
        requests(rt)
            .into_iter()
            .filter_map(|r| match r {
                Request::Cmd(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_big_switch_toggles_the_engine() {
        let mut rt = runtime(installed());
        click(&mut rt, "switch");
        assert!(!rt.state.prefs.enabled);
        click(&mut rt, "switch");
        assert_eq!(
            sent(&rt),
            [Command::SetEnabled(false), Command::SetEnabled(true)]
        );
    }

    #[test]
    fn strength_segments_send_strength() {
        let mut rt = runtime(installed());
        click(&mut rt, "strength-row/segment2");
        assert_eq!(rt.state.prefs.strength, Strength::Strong);
        assert_eq!(sent(&rt), [Command::SetStrength(Strength::Strong)]);
    }

    #[test]
    fn onboarding_shows_without_the_driver_and_skips() {
        let mut rt = runtime(State::new(Prefs::default(), false));
        assert!(rt.find("onboarding").is_some());
        assert!(rt.find("switch").is_none());
        click(&mut rt, "skip");
        assert!(rt.find("switch").is_some());
    }

    #[test]
    fn settings_and_back() {
        let mut rt = runtime(installed());
        click(&mut rt, "settings");
        assert_eq!(rt.state.page, Page::Settings);
        assert!(rt.find("login").is_some());
        click(&mut rt, "back");
        assert_eq!(rt.state.page, Page::Main);
    }

    #[test]
    fn install_failures_dismiss() {
        let mut s = State::new(Prefs::default(), false);
        s.install = Install::Failed("missing".into());
        let mut rt = runtime(s);
        click(&mut rt, "dismiss");
        assert_eq!(rt.state.install, Install::Idle);
    }

    #[test]
    fn errors_dismiss() {
        let mut s = installed();
        s.error = Some("Mic unplugged".into());
        let mut rt = runtime(s);
        click(&mut rt, "dismiss");
        assert_eq!(rt.state.error, None);
    }

    #[test]
    fn setup_walks_through_its_real_steps() {
        static INSTALLS: AtomicUsize = AtomicUsize::new(0);
        static ASKS: AtomicUsize = AtomicUsize::new(0);
        let mut s = State::new(Prefs::default(), false);
        s.link = crate::state::Link::Record(Vec::new());
        s.mic = Mic::NotDetermined;
        s.hooks.install = || {
            INSTALLS.fetch_add(1, Ordering::Relaxed);
        };
        s.hooks.request_mic = || {
            ASKS.fetch_add(1, Ordering::Relaxed);
        };
        let mut rt = runtime(s);
        // Step one: the driver. The system-microphone switch waits for it.
        click(&mut rt, "system-mic");
        assert!(!rt.state.prefs.system_microphone);
        click(&mut rt, "next");
        assert_eq!(INSTALLS.load(Ordering::Relaxed), 1);
        assert_eq!(rt.state.install, Install::Running);
        let probe = crate::state::Probe::new(
            oto_engine::DriverStatus::Loaded(oto_engine::BUNDLED_DRIVER_VERSION),
            Some(oto_engine::BUNDLED_DRIVER_VERSION),
            Mic::NotDetermined,
        );
        rt.update(|s, _| s.installed(crate::driver::Outcome::Installed, probe));
        rt.frame(1.0 / 60.0);
        // Step two: the permission. Setup is still up.
        assert!(rt.find("onboarding").is_some());
        click(&mut rt, "next");
        assert_eq!(ASKS.load(Ordering::Relaxed), 1);
        assert_eq!(INSTALLS.load(Ordering::Relaxed), 1);
        rt.update(|s, _| s.mic_answered(Mic::Authorized));
        rt.frame(1.0 / 60.0);
        // Step three: optional.
        click(&mut rt, "system-mic");
        assert!(rt.state.prefs.system_microphone);
        assert_eq!(requests(&rt), [Request::SystemMic(true)]);
        assert!(rt.find("skip").is_none(), "nothing left to put off");
        click(&mut rt, "next");
        assert!(rt.state.prefs.onboarded);
        assert!(rt.find("switch").is_some());
    }

    #[test]
    fn a_failed_engine_replaces_the_controls_with_a_retry() {
        static STARTS: AtomicUsize = AtomicUsize::new(0);
        let mut s = installed();
        s.hooks.start_engine = |_, _| {
            STARTS.fetch_add(1, Ordering::Relaxed);
            crate::state::Link::Record(Vec::new())
        };
        s.engine_failed("no audio hardware");
        let mut rt = runtime(s);
        assert!(rt.find("switch").is_none());
        assert!(rt.find("segment0").is_none());
        assert!(rt.find("settings").is_some(), "Quit stays reachable");
        click(&mut rt, "retry");
        assert_eq!(STARTS.load(Ordering::Relaxed), 1);
        assert_eq!(rt.state.engine, EngineState::Starting);
        assert!(rt.find("switch").is_some());
    }

    #[test]
    fn a_denied_microphone_says_so_and_opens_settings() {
        static OPENED: AtomicUsize = AtomicUsize::new(0);
        let mut s = installed();
        assert!(runtime(installed()).find("mic").is_none());
        s.mic = Mic::Denied;
        s.hooks.open_mic_settings = || {
            OPENED.fetch_add(1, Ordering::Relaxed);
            Ok(())
        };
        let mut rt = runtime(s);
        assert!(rt.find("mic").is_some());
        click(&mut rt, "mic-action");
        assert_eq!(OPENED.load(Ordering::Relaxed), 1);
    }

    fn settings() -> State {
        let mut s = installed();
        s.page = Page::Settings;
        s
    }

    #[test]
    fn the_model_picker_is_in_settings_only_with_a_choice() {
        let mut s = settings();
        s.tiers = vec![Tier::Standard, Tier::Isolate];
        assert!(runtime(s).find("model-row").is_none());

        let mut s = installed();
        s.tiers = vec![Tier::Light, Tier::Standard, Tier::Max, Tier::Fast];
        assert!(
            runtime(s).find("model-row").is_none(),
            "not on the main page"
        );

        let mut s = settings();
        s.tiers = vec![Tier::Light, Tier::Standard, Tier::Max, Tier::Fast];
        let mut rt = runtime(s);
        assert!(rt.find("model-row").is_some());
        // Options follow TIERS: DeepFilterNet3, DPDFNet-2, FastEnhancer-S, RNNoise.
        click(&mut rt, "model-picker-trigger");
        assert_eq!(rt.state.open, Some(Open::Model));
        click(&mut rt, "option1");
        assert_eq!(rt.state.prefs.tier, Tier::Max);
        assert_eq!(rt.state.open, None, "choosing closes the menu");
        assert_eq!(sent(&rt), [Command::SetTier(Tier::Max)]);

        // What this build offers decides the default state.
        assert_eq!(
            runtime(settings()).find("model-row").is_some(),
            oto_engine::available_tiers()
                .iter()
                .filter(|t| **t != Tier::Isolate)
                .count()
                > 1
        );
    }

    #[test]
    fn two_modes_show_only_with_voice_isolation() {
        let mut s = installed();
        s.tiers = vec![Tier::Light, Tier::Standard, Tier::Max];
        assert!(runtime(s).find("mode-row").is_none());

        let mut s = installed();
        s.tiers = vec![Tier::Light, Tier::Standard, Tier::Max, Tier::Isolate];
        s.set_tier(Tier::Max);
        let mut rt = runtime(s);
        click(&mut rt, "mode-row/segment1");
        assert_eq!(rt.state.prefs.tier, Tier::Isolate);
        assert!(rt.find("mode-about").is_some());
        click(&mut rt, "mode-row/segment0");
        assert_eq!(rt.state.prefs.tier, Tier::Max, "back to the chosen model");
        assert_eq!(
            sent(&rt),
            [
                Command::SetTier(Tier::Max),
                Command::SetTier(Tier::Isolate),
                Command::SetTier(Tier::Max)
            ]
        );
    }

    #[test]
    fn the_degraded_banner_switches_to_noise_cancellation() {
        use oto_engine::Event;
        let mut s = installed();
        s.tiers = vec![Tier::Light, Tier::Standard, Tier::Isolate];
        s.set_tier(Tier::Light);
        s.set_tier(Tier::Isolate);
        s.on_event(Event::Degraded("muted".into()));
        let mut rt = runtime(s);
        assert!(rt.find("degraded").is_some());
        assert!(rt.find("error").is_none(), "not an error banner");
        click(&mut rt, "degraded-action");
        assert_eq!(rt.state.prefs.tier, Tier::Light);
        assert_eq!(rt.state.degraded, None);
        assert!(rt.find("degraded").is_none());
    }

    #[test]
    fn hear_the_difference_records_compares_and_stops() {
        use crate::state::HearTest;
        use oto_engine::test_capture::TestRecording;
        static STARTS: AtomicUsize = AtomicUsize::new(0);
        static STOPS: AtomicUsize = AtomicUsize::new(0);
        static CLEANED: AtomicUsize = AtomicUsize::new(0);
        let mut s = installed();
        s.hooks.start_hear_test = |_| {
            STARTS.fetch_add(1, Ordering::Relaxed);
        };
        s.hooks.stop_hear_test = || {
            STOPS.fetch_add(1, Ordering::Relaxed);
        };
        s.hooks.set_hear_cleaned = |on| {
            CLEANED.fetch_add(usize::from(on), Ordering::Relaxed);
        };
        let mut rt = runtime(s);
        click(&mut rt, "hear-start");
        assert_eq!(STARTS.load(Ordering::Relaxed), 1);
        assert!(rt.find("hear-stop").is_some());
        let rec = TestRecording {
            sample_rate: 16_000,
            raw: vec![0.0; 16_000].into(),
            cleaned: vec![0.0; 16_000].into(),
            is_bluetooth_hfp: true,
        };
        rt.update(move |s, _| {
            let job = s.hear_job_for_test();
            s.hear_recorded(&job, Ok(rec));
        });
        rt.frame(1.0 / 60.0);
        assert!(rt.find("hear-hfp").is_some());
        click(&mut rt, "hear-ab/segment1");
        assert_eq!(CLEANED.load(Ordering::Relaxed), 1);
        click(&mut rt, "hear-done");
        assert_eq!(STOPS.load(Ordering::Relaxed), 1);
        assert_eq!(rt.state.hear, HearTest::Idle);
    }

    #[test]
    fn hear_the_difference_waits_for_the_call_to_end() {
        let mut s = installed();
        s.on_event(oto_engine::Event::ClientsActive(true));
        s.hooks.start_hear_test = |_| panic!("a disabled button must not start the test");
        let mut rt = runtime(s);
        click(&mut rt, "hear-start");
        assert!(rt.find("hear-about").is_some());
        assert_eq!(rt.state.hear_blocked(), Some(crate::state::HEAR_IN_CALL));
    }

    #[test]
    fn a_model_fallback_is_a_notice_with_the_running_model() {
        use oto_engine::Event;
        use oto_engine::dsp::Model;
        let mut s = installed();
        s.on_event(Event::ClientsActive(true));
        s.on_event(Event::ModelFallback {
            requested: Model::DeepFilterNet3,
            effective: Model::RNNoise,
            reason: "no".into(),
        });
        s.set_running(Some((Model::RNNoise, 36.0)));
        let mut rt = runtime(s);
        assert!(rt.find("error").is_none(), "not an error banner");
        assert!(rt.find("model").is_some());
        click(&mut rt, "model-notice-action");
        assert!(rt.state.model_notice.is_none());
        assert!(rt.find("model-notice").is_none());
    }

    #[test]
    fn settings_switches_and_uninstall() {
        static UNINSTALLS: AtomicUsize = AtomicUsize::new(0);
        let mut s = installed();
        s.page = Page::Settings;
        s.hooks.uninstall = || {
            UNINSTALLS.fetch_add(1, Ordering::Relaxed);
        };
        let mut rt = runtime(s);
        click(&mut rt, "system-mic");
        assert_eq!(requests(&rt), [Request::SystemMic(true)]);
        click(&mut rt, "login");
        assert!(rt.state.prefs.start_at_login);
        click(&mut rt, "uninstall");
        assert_eq!(UNINSTALLS.load(Ordering::Relaxed), 1);
        assert_eq!(rt.state.install, Install::Removing);
        // Busy: neither button starts anything more.
        click(&mut rt, "uninstall");
        click(&mut rt, "install");
        assert_eq!(UNINSTALLS.load(Ordering::Relaxed), 1);
        assert_eq!(rt.state.install, Install::Removing);
    }

    #[test]
    fn start_at_login_is_disabled_with_its_reason_when_unavailable() {
        let mut s = installed();
        s.page = Page::Settings;
        s.login_available = false;
        s.hooks.set_login = |_| panic!("a disabled switch must not act");
        let mut rt = runtime(s);
        assert!(rt.find("login-about").is_some());
        click(&mut rt, "login");
        assert!(!rt.state.prefs.start_at_login);
    }

    #[test]
    fn without_a_driver_the_main_page_leads_back_to_setup() {
        let mut s = State::new(Prefs::default(), false);
        s.leave_onboarding();
        let mut rt = runtime(s);
        assert!(rt.find("uninstall").is_none());
        click(&mut rt, "driver-missing-action");
        assert!(rt.find("onboarding").is_some());
    }

    fn offer(version: &str) -> crate::updater::verify::UpdateInfo {
        crate::updater::verify::UpdateInfo {
            version: version.into(),
            notes_url: format!("https://github.com/deepso7/oto/releases/tag/v{version}"),
            asset_url: String::new(),
            sha256: String::new(),
            size: 1,
            min_os: "13.0".into(),
            driver_version: oto_engine::BUNDLED_DRIVER_VERSION,
            min_driver_version: oto_engine::MIN_COMPATIBLE_DRIVER_VERSION,
            published_at: String::new(),
        }
    }

    fn updatable() -> State {
        let mut s = installed();
        s.install_mode = InstallMode::SelfUpdatable {
            bundle: "/Applications/Oto.app".into(),
        };
        s
    }

    #[test]
    fn the_update_banner_downloads_skips_and_dismisses() {
        static DOWNLOADS: AtomicUsize = AtomicUsize::new(0);
        let mut s = updatable();
        s.hooks.download_update = |_, _| {
            DOWNLOADS.fetch_add(1, Ordering::Relaxed);
        };
        s.update = UpdateState::Available(offer("9.0.0"));
        let mut rt = runtime(s);
        assert!(rt.find("update").is_some());
        click(&mut rt, "update-action");
        assert_eq!(DOWNLOADS.load(Ordering::Relaxed), 1);
        assert!(rt.find("update-cancel").is_some());
        click(&mut rt, "update-cancel");
        assert_eq!(rt.state.update, UpdateState::Available(offer("9.0.0")));
        click(&mut rt, "update-skip");
        assert_eq!(rt.state.prefs.skipped_version.as_deref(), Some("9.0.0"));
        assert!(rt.find("update").is_none());

        let mut s = updatable();
        s.update = UpdateState::Available(offer("9.0.0"));
        let mut rt = runtime(s);
        click(&mut rt, "update-dismiss");
        assert_eq!(rt.state.update, UpdateState::Idle);
        assert_eq!(rt.state.prefs.skipped_version, None);
    }

    #[test]
    fn restart_to_update_waits_for_the_call_to_end() {
        static APPLIES: AtomicUsize = AtomicUsize::new(0);
        let mut s = updatable();
        s.hooks.apply_update = |_, _, _| {
            APPLIES.fetch_add(1, Ordering::Relaxed);
        };
        s.update = UpdateState::Ready(crate::updater::verify::StagedUpdate::unowned(
            offer("9.0.0"),
            "/tmp/Oto.app".into(),
            "/tmp".into(),
        ));
        s.on_event(oto_engine::Event::ClientsActive(true));
        let mut rt = runtime(s);
        assert!(rt.find("update-detail").is_some());
        click(&mut rt, "update-action");
        assert_eq!(APPLIES.load(Ordering::Relaxed), 0);
        rt.update(|s, _| s.on_event(oto_engine::Event::ClientsActive(false)));
        rt.frame(1.0 / 60.0);
        click(&mut rt, "update-action");
        assert_eq!(APPLIES.load(Ordering::Relaxed), 1);
        assert_eq!(rt.state.update, UpdateState::Applying);
    }

    #[test]
    fn copies_that_cant_update_themselves_offer_the_download() {
        static OPENED: AtomicUsize = AtomicUsize::new(0);
        let mut s = installed();
        s.install_mode =
            InstallMode::NotifyOnly(crate::updater::apply::NotifyReason::NotInApplications);
        s.hooks.open_url = |u| {
            assert!(u.starts_with("https://github.com/deepso7/oto/releases"));
            OPENED.fetch_add(1, Ordering::Relaxed);
            Ok(())
        };
        s.update = UpdateState::Available(offer("9.0.0"));
        let mut rt = runtime(s);
        assert!(rt.find("update-detail").is_some(), "the install-mode note");
        click(&mut rt, "update-action");
        assert_eq!(OPENED.load(Ordering::Relaxed), 1);

        rt.update(|s, _| {
            s.update = UpdateState::UnrecognizedKey {
                releases_url: crate::updater::keys::RELEASES_URL.into(),
            }
        });
        rt.frame(1.0 / 60.0);
        click(&mut rt, "update-action");
        assert_eq!(OPENED.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn the_driver_update_banner_installs_but_not_mid_call() {
        static INSTALLS: AtomicUsize = AtomicUsize::new(0);
        let mut s = installed();
        s.hooks.install = || {
            INSTALLS.fetch_add(1, Ordering::Relaxed);
        };
        // A legacy (pre-'otvr') driver: loaded, but too old for this build.
        s.on_event(oto_engine::Event::DriverIncompatible {
            loaded: 0,
            min_required: oto_engine::MIN_COMPATIBLE_DRIVER_VERSION,
        });
        s.on_disk_version = Some(0);
        assert_eq!(s.driver_ui(), crate::state::DriverUiState::Incompatible);
        assert_eq!(s.error, None);
        s.on_event(oto_engine::Event::ClientsActive(true));
        let mut rt = runtime(s);
        assert!(rt.find("driver-incompatible").is_some());
        click(&mut rt, "driver-update-action");
        assert_eq!(INSTALLS.load(Ordering::Relaxed), 0);
        rt.update(|s, _| s.on_event(oto_engine::Event::ClientsActive(false)));
        rt.frame(1.0 / 60.0);
        click(&mut rt, "driver-update-action");
        assert_eq!(INSTALLS.load(Ordering::Relaxed), 1);
        assert_eq!(rt.state.install, Install::Running);
        // Installed, but finishing on the next restart: the banner says so.
        let p = crate::state::Probe::new(
            oto_engine::DriverStatus::Loaded(0),
            Some(oto_engine::BUNDLED_DRIVER_VERSION),
            Mic::Authorized,
        );
        rt.update(move |s, _| s.installed(crate::driver::Outcome::InstalledNeedsReboot, p));
        rt.frame(1.0 / 60.0);
        assert!(rt.find("driver-restart").is_some());
        assert!(rt.find("driver-update-action").is_none());
    }

    #[test]
    fn the_updates_card_checks_and_shows_release_notes() {
        static CHECKS: AtomicUsize = AtomicUsize::new(0);
        let mut s = updatable();
        s.page = Page::Settings;
        s.hooks.check_update = |_| {
            CHECKS.fetch_add(1, Ordering::Relaxed);
        };
        let mut rt = runtime(s);
        assert!(rt.find("release-notes").is_none());
        click(&mut rt, "auto-update");
        assert!(!rt.state.prefs.check_for_updates);
        click(&mut rt, "update-button");
        assert_eq!(CHECKS.load(Ordering::Relaxed), 1);
        assert!(rt.find("update-status").is_some(), "Checking…");
        click(&mut rt, "update-button");
        assert_eq!(CHECKS.load(Ordering::Relaxed), 1, "disabled while checking");
        let g = rt.state.update_generation();
        rt.update(move |s, _| {
            s.update_checked(
                g,
                Ok(crate::updater::verify::CheckOutcome::Available(offer(
                    "9.0.0",
                ))),
            )
        });
        rt.frame(1.0 / 60.0);
        assert!(rt.find("release-notes").is_some());
    }

    #[test]
    fn settings_hides_reinstall_when_a_newer_driver_is_loaded() {
        let cases = [
            (
                oto_engine::DriverStatus::Loaded(oto_engine::BUNDLED_DRIVER_VERSION + 1),
                None,
            ),
            (
                oto_engine::DriverStatus::Loaded(oto_engine::BUNDLED_DRIVER_VERSION),
                Some(oto_engine::BUNDLED_DRIVER_VERSION + 1),
            ),
            (
                oto_engine::DriverStatus::Loaded(
                    oto_engine::BUNDLED_DRIVER_VERSION.saturating_sub(1),
                ),
                Some(oto_engine::BUNDLED_DRIVER_VERSION + 1),
            ),
        ];
        for (loaded, on_disk) in cases {
            let mut s = installed();
            s.page = Page::Settings;
            s.probed(crate::state::Probe::new(loaded, on_disk, Mic::Authorized));
            let rt = runtime(s);
            assert!(rt.find("driver-row").is_some(), "{loaded:?} / {on_disk:?}");
            assert!(
                rt.find("install").is_none(),
                "never offer a downgrade for {loaded:?} / {on_disk:?}"
            );
            assert!(rt.find("uninstall").is_some(), "{loaded:?} / {on_disk:?}");
        }
    }

    #[test]
    fn onboarding_shows_restart_guidance_instead_of_install_when_newer_driver_is_on_disk() {
        let mut s = State::new(Prefs::default(), false);
        s.probed(crate::state::Probe::new(
            oto_engine::DriverStatus::Absent,
            Some(oto_engine::BUNDLED_DRIVER_VERSION + 1),
            Mic::Authorized,
        ));
        assert!(s.onboarding());
        assert!(s.newer_driver_present());
        let rt = runtime(s);
        assert!(rt.find("onboarding").is_some());
        assert!(rt.find("restart").is_some());
        assert!(
            rt.find("next").is_none(),
            "no Install Again downgrade button"
        );
        assert!(rt.find("skip").is_some());

        // Even when an install error arrived before the follow-up probe confirmed the newer driver
        // on disk, both the error banner and the restart guidance show while the install button
        // stays hidden.
        let mut s2 = State::new(Prefs::default(), false);
        s2.installed(
            crate::driver::Outcome::Failed("A newer Oto audio driver is already installed.".into()),
            crate::state::Probe::new(
                oto_engine::DriverStatus::Absent,
                Some(oto_engine::BUNDLED_DRIVER_VERSION + 1),
                Mic::Authorized,
            ),
        );
        let rt2 = runtime(s2);
        assert!(rt2.find("error").is_some());
        assert!(rt2.find("restart").is_some());
        assert!(rt2.find("next").is_none());
    }

    #[test]
    fn update_failures_show_retry_and_dismiss_on_the_main_page() {
        static CHECKS: AtomicUsize = AtomicUsize::new(0);
        let mut s = updatable();
        s.hooks.check_update = |_| {
            CHECKS.fetch_add(1, Ordering::Relaxed);
        };
        s.update = UpdateState::Failed("Download interrupted.".into());
        let mut rt = runtime(s);
        assert!(rt.find("update").is_some());
        assert!(rt.find("update-detail").is_some());
        assert!(rt.find("update-retry").is_some());
        assert!(rt.find("update-dismiss").is_some());
        click(&mut rt, "update-retry");
        assert_eq!(CHECKS.load(Ordering::Relaxed), 1);

        rt.update(|s, _| s.update = UpdateState::Failed("Download interrupted.".into()));
        rt.frame(1.0 / 60.0);
        click(&mut rt, "update-dismiss");
        assert_eq!(rt.state.update, UpdateState::Idle);
        assert!(rt.find("update").is_none());
    }

    #[test]
    fn hear_the_difference_waits_for_the_update_to_commit() {
        let mut s = installed();
        s.candidate_pending = true;
        s.hooks.start_hear_test = |_| panic!("blocked until the update commits");
        let mut rt = runtime(s);
        click(&mut rt, "hear-start");
        assert_eq!(
            rt.state.hear_blocked(),
            Some(crate::state::FINISHING_UPDATE)
        );
        assert_eq!(rt.state.hear, HearTest::Idle);
    }
}
