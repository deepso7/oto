//! The Oto window: one switch for now. `cargo run -p oto-app` opens it;
//! `cargo run -p oto-app -- --snap oto.png --scene on` renders it headless.

use rune::tok::{self, motion, type_};
use rune::{AppSpec, Cx, El, col, el, text};

#[derive(Default)]
struct State {
    enabled: bool,
}

fn view(s: &State, _: &mut Cx<State>) -> El {
    let t = tok::theme(true);
    let label = text(if s.enabled { "Noise cancellation on" } else { "Noise cancellation off" })
        .font(type_::TITLE)
        .color(t.text);
    let knob = el().key("knob").size(18.0, 18.0).radius(9.0).fill(t.knob)
        .absolute(if s.enabled { 18.0 } else { 2.0 }, 2.0);
    let switch = el().key("switch").role("switch").label("Noise cancellation").checked(s.enabled)
        .focusable(true).size(38.0, 22.0).radius(11.0)
        .fill(if s.enabled { tok::accent(0, true) } else { t.control_off })
        .motion(motion::SNAPPY)
        .on_press(|s: &mut State| s.enabled = !s.enabled)
        .child(knob);
    col().w_full().h_full().center().gap(16.0).fill(t.bg).child(label).child(switch)
}

fn main() {
    rune::run(AppSpec::<State> {
        scenes: &[("on", |s| s.enabled = true)],
        ..AppSpec::new("Oto", (320.0, 200.0), State::default, view)
    });
}
