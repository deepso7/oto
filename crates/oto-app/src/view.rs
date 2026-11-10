//! The popover's screens: onboarding, main and settings. Built from Rune's neutral tokens and
//! rune-kit's Select, SegmentedControl, Switch and Button; the big switch and the meter are
//! drawn here.

use rune::tok::{self, Theme, motion, type_};
use rune::{Color, Cx, El, Icon, TextStyle, col, el, icon, row, text};
use rune_kit::{
    Button, ButtonVariant, ControlSize, IconButton, IconButtonVariant, Kit, Placement,
    SegmentedControl, Select, SelectAppearance, StateColors,
};

use crate::meter::{BAR_H, BAR_W};
use crate::permission::Mic;
use crate::state::{EngineState, HEAR_TEST, Install, Open, Page, STRENGTHS, State, TIERS};

pub const WIDTH: f32 = 340.0;
const PAD: f32 = 16.0;
/// A card's content width.
const CARD_INNER: f32 = WIDTH - 2.0 * PAD - 24.0;
const RADIUS: f32 = 12.0;

const H1: TextStyle = TextStyle::new(17.0, 650.0).lh(22.0);
const TITLE: TextStyle = TextStyle::new(14.0, 600.0).lh(18.0);
const BODY: TextStyle = type_::BODY.lh(18.0);
const LABEL: TextStyle = TextStyle::new(11.5, 500.0).lh(14.0);
const CAPTION: TextStyle = TextStyle::new(11.5, 400.0).lh(15.0);
const CAPTION_NUM: TextStyle = TextStyle::new(11.5, 500.0).lh(15.0).features("'tnum' 1");

static WAVE: Icon = Icon {
    name: "oto-wave",
    view: 16.0,
    body: r#"<path d="M2.6 6.6V9.4M5.3 4.6V11.4M8 2.6V13.4M10.7 4.6V11.4M13.4 6.6V9.4" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>"#,
};
static SLIDERS: Icon = Icon {
    name: "oto-sliders",
    view: 16.0,
    body: r#"<path d="M2.5 5H8.5M12.5 5H13.5M2.5 11H3.5M7.5 11H13.5" stroke="currentColor" stroke-linecap="round"/><circle cx="10.5" cy="5" r="1.75" stroke="currentColor"/><circle cx="5.5" cy="11" r="1.75" stroke="currentColor"/>"#,
};
static CHEVRON_LEFT: Icon = Icon {
    name: "oto-chevron-left",
    view: 16.0,
    body: r#"<path d="M9.5 4L5.5 8L9.5 12" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>"#,
};

/// Everything a piece needs to paint itself.
struct Paint {
    t: Theme,
    k: Kit,
    accent: Color,
    green: Color,
    amber: Color,
    hair: f32,
}

pub fn view(s: &State, cx: &mut Cx<State>) -> El {
    let dark = cx.system_dark();
    let p = Paint {
        t: tok::theme(dark),
        k: Kit::neutral(dark),
        accent: tok::accent(0, dark),
        green: rune::oklch(if dark { 0.72 } else { 0.62 }, 0.17, 150.0),
        amber: rune::oklch(if dark { 0.78 } else { 0.7 }, 0.15, 75.0),
        hair: 1.0 / cx.env().scale,
    };
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
        .fill(p.t.bg)
        .padding(PAD)
        .gap(12.0)
        .child(body)
        .on_layout(|s: &mut State, r: [f32; 4]| s.size = (r[2], r[3]));
    if s.popover {
        // The popover window clips itself to this radius; the hairline gives it an edge on
        // backgrounds of the same lightness.
        window = window.radius(RADIUS).border(p.hair, p.t.card_border);
    }
    el().key("root").w_full().child(window).children(layer)
}

fn card(p: &Paint, key: &str) -> El {
    col()
        .key(key)
        .w_full()
        .px(12.0)
        .py(6.0)
        .radius(10.0)
        .fill(p.t.card)
        .border(p.hair, p.t.card_border)
        .shadow(1.0, 2.0, p.t.shadow.alpha(0.08))
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

fn logo(p: &Paint, size: f32) -> El {
    el().key("logo")
        .size(size, size)
        .center()
        .radius(size * 0.28)
        .gradient(p.accent, p.accent.mix(Color::BLACK, 0.18))
        .a11y_hidden()
        .child(icon(&WAVE, size * 0.62).color(Color::WHITE))
}

fn header(p: &Paint, title: &str, leading: Option<El>, trailing: Option<El>) -> El {
    let mut r = row().key("header").w_full().h(28.0).center_y().gap(8.0);
    r = r.child(leading.unwrap_or_else(|| logo(p, 22.0)));
    r = r.child(
        text(title)
            .key("title")
            .font(TITLE)
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
        .fill(if on { p.accent } else { p.t.control_off })
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

/// The main on/off: bigger than a settings toggle.
fn big_switch(p: &Paint, on: bool) -> El {
    switch(
        p,
        "switch",
        "Noise cancellation",
        on,
        (46.0, 28.0),
        Some(|s| s.toggle()),
    )
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
        .h(40.0)
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
        .gap(6.0)
        .child(
            text(label)
                .key("label")
                .font(LABEL)
                .color(p.t.text_secondary)
                .a11y_hidden(),
        )
        .child(control)
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
        .radius(8.0)
        .fill(c.danger.bg)
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
        .radius(8.0)
        .fill(c.bg)
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
        .on_press(|s: &mut State| {
            s.open = None;
            s.page = Page::Settings;
        })
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

    let hero = card(p, "hero").py(14.0).child(
        row()
            .w_full()
            .center_y()
            .gap(12.0)
            .child(
                col()
                    .grow(1.0)
                    .shrink()
                    .gap(3.0)
                    .child(
                        text("Noise cancellation")
                            .key("hero-title")
                            .font(TITLE)
                            .color(p.t.text),
                    )
                    .child(
                        text(if !on {
                            "Turn on to remove background noise"
                        } else if s.clients_active {
                            "Cleaning your voice"
                        } else {
                            "On, ready for your next call"
                        })
                        .key("hero-sub")
                        .font(CAPTION)
                        .color(p.t.text_secondary),
                    ),
            )
            .child(big_switch(p, on)),
    );

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
        layer.push(list.w(CARD_INNER));
    }
    let labels: Vec<&str> = STRENGTHS.iter().map(|(_, l)| *l).collect();
    let strength = SegmentedControl::new(
        &labels,
        STRENGTHS.iter().position(|(v, _)| *v == s.prefs.strength),
    )
    .width(Some(CARD_INNER))
    .label("Strength")
    .on_pick(|s: &mut State, i: usize, _| s.set_strength(STRENGTHS[i].0))
    .build(&p.k);
    let controls = card(p, "controls")
        .py(12.0)
        .gap(12.0)
        .child(labelled(p, "mic-row", "Microphone", device))
        .child(labelled(p, "strength-row", "Strength", strength));

    let metering = on && s.clients_active;
    let mut meter = card(p, "meter")
        .py(12.0)
        .gap(10.0)
        .child(meter_row(
            p,
            "raw",
            "Input",
            &s.meter.raw,
            p.t.text_secondary.alpha(0.75),
        ))
        .child(meter_row(p, "clean", "Cleaned", &s.meter.clean, p.accent));
    if !metering {
        meter = meter.child(caption(
            p,
            "meter-hint",
            "Levels show while an app is using Oto Microphone.",
        ));
    }

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
    page = page.child(controls).child(meter);
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
        // TODO(engine): record 5 s from the real mic, then play it back raw and cleaned. Needs an
        // engine API for a test capture; hidden behind HEAR_TEST until then.
        page = page.child(
            Button::new("Hear the Difference")
                .variant(ButtonVariant::Outline)
                .on_press(|_: &mut State| {})
                .build(&p.k)
                .key("hear"),
        );
    }
    page.child(status_line(s, p))
}

fn meter_row(p: &Paint, key: &str, label: &str, live: &rune::Live<[f32; 4]>, color: Color) -> El {
    row()
        .key(key)
        .w_full()
        .gap(8.0)
        .center_y()
        .child(
            text(label)
                .key("label")
                .font(CAPTION)
                .color(p.t.text_secondary)
                .grow(1.0),
        )
        .child(
            el().key("track")
                .role("meter")
                .label(label)
                .size(BAR_W, BAR_H)
                .radius(BAR_H / 2.0)
                .fill(p.t.control_off)
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
    } else if !s.driver_installed || s.mic.blocked() {
        p.amber
    } else if s.prefs.enabled && s.clients_active {
        p.green
    } else {
        p.t.text_secondary.alpha(0.5)
    };
    let mut r = row()
        .key("status")
        .w_full()
        .px(4.0)
        .gap(6.0)
        .center_y()
        .child(
            el().key("dot")
                .size(7.0, 7.0)
                .radius(3.5)
                .fill(dot)
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
    if let Some(cpu) = s.cpu() {
        r = r.child(
            text(cpu)
                .key("cpu")
                .font(CAPTION_NUM)
                .color(p.t.text_secondary),
        );
    }
    r
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
    // Only when the choice changes something: with one model in the build there's no picker.
    let tiers: Vec<_> = TIERS
        .iter()
        .filter(|(t, _, _)| s.tiers.contains(t))
        .collect();
    if tiers.len() > 1 {
        let tier_labels: Vec<&str> = tiers.iter().map(|(_, l, _)| *l).collect();
        let (_, tier_label, tier_about) = tiers
            .iter()
            .find(|(t, _, _)| *t == s.prefs.tier)
            .copied()
            .unwrap_or(tiers[0]);
        let tier = Select::new("tier", &tier_labels)
            .appearance(SelectAppearance::Field)
            .width(132.0)
            .label("Model")
            .current(Some(tier_label))
            .open(s.open == Some(Open::Tier))
            .on_toggle(|s: &mut State| s.toggle_open(Open::Tier))
            .on_close(|s: &mut State| s.open = None)
            .on_choose(|s: &mut State, v: &str, _| {
                if let Some((t, _, _)) = TIERS.iter().find(|(_, l, _)| *l == v) {
                    s.set_tier(*t);
                }
            })
            .build(&p.k, layer);
        general = general
            .child(separator(p, "sep-tier"))
            .child(field_row(p, "tier-row", "Model", tier))
            .child(caption(p, "tier-about", *tier_about).pb(8.0));
    }

    let busy = s.install.busy();
    let install = Button::new(match (&s.install, s.driver_installed || s.driver_on_disk) {
        (Install::Running, _) => "Installing\u{2026}",
        (_, true) => "Reinstall\u{2026}",
        (_, false) => "Install\u{2026}",
    })
    .disabled(busy)
    .on_press(|s: &mut State| s.start_install())
    .build(&p.k)
    .key("install");
    let mut driver = card(p, "driver").child(field_row(
        p,
        "driver-row",
        match (s.driver_installed, s.driver_on_disk) {
            (true, _) => "Audio driver installed",
            (false, true) => "Audio driver needs a restart",
            (false, false) => "Audio driver not installed",
        },
        install,
    ));
    if s.driver_installed || s.driver_on_disk {
        let uninstall = Button::new(if s.install == Install::Removing {
            "Removing\u{2026}"
        } else {
            "Uninstall\u{2026}"
        })
        .disabled(busy)
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
        .child(general)
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
            .justify_between()
            .child(
                text(concat!("Oto ", env!("CARGO_PKG_VERSION")))
                    .key("version")
                    .font(CAPTION)
                    .color(p.t.text_secondary),
            )
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
                "Adds \u{201c}Oto Microphone\u{201d}. Asks for your password; sound pauses for a second."
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

    // The one button that moves setup on: whichever step is next.
    let (label, enabled, next): (&str, bool, fn(&mut State)) = if !s.driver_installed {
        (
            match (&s.install, restart) {
                (Install::Running, _) => "Installing\u{2026}",
                (_, true) => "Install Again",
                (_, false) => "Install Driver",
            },
            !s.install.busy(),
            |s| s.start_install(),
        )
    } else {
        match s.mic {
            Mic::NotDetermined => ("Allow Microphone", !s.mic_requesting, |s| s.allow_mic()),
            Mic::Denied => ("Open Settings", true, |s| s.allow_mic()),
            Mic::Authorized | Mic::Restricted => ("Done", true, |s| s.leave_onboarding()),
        }
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
    actions = if label == "Done" {
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
    page.child(
        actions.child(
            Button::new(label)
                .variant(ButtonVariant::Contrast)
                .control_size(ControlSize::Large)
                .disabled(!enabled)
                .on_press(next)
                .build(&p.k)
                .key("next"),
        ),
    )
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
                Request::SystemMic(_) => None,
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
        click(&mut rt, "segment2");
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
        let probe = crate::state::Probe {
            driver: true,
            on_disk: true,
            mic: Mic::NotDetermined,
        };
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

    #[test]
    fn the_model_picker_shows_only_with_a_choice() {
        let mut s = installed();
        s.page = Page::Settings;
        s.tiers = vec![Tier::Standard];
        assert!(runtime(s).find("tier-row").is_none());

        let mut s = installed();
        s.page = Page::Settings;
        s.tiers = vec![Tier::Light, Tier::Standard];
        assert!(runtime(s).find("tier-row").is_some());

        // What this build offers decides the default state.
        let mut s = installed();
        s.page = Page::Settings;
        assert_eq!(
            runtime(s).find("tier-row").is_some(),
            oto_engine::available_tiers().len() > 1
        );
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
}
