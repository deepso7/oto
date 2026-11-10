//! `oto`: the headless front end. Runs the engine without a window, lists devices, reports
//! driver status, and hosts the latency and benchmark tools (see docs/roadmap.md).

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use lexopt::prelude::*;
use oto_engine::{Engine, EngineOptions, Event, Settings, Strength, Tier, dsp};

const USAGE: &str = "\
oto: noise-cancelling virtual microphone (headless)

USAGE:
    oto run [--device ID] [--strength light|balanced|strong] [--tier TIER]
            [--no-denoise] [--set-default]
        Run until Ctrl-C, printing events and a level line. Audio only flows while an app
        records from Oto Microphone. --set-default makes Oto Microphone the default input
        until exit.
    oto devices
        List physical input devices (IDs for --device).
    oto status
        Driver and device status. Exits 1 when the driver is not installed.
    oto latency [--seconds N] [--device ID] [--strength S] [--tier TIER] [--no-denoise]
        Measure end-to-end latency through the real Oto Microphone path (needs the driver).
    oto bench [--model rnnoise|dfn3] [--tier TIER] [--seconds N]
        Time the DSP chain on synthetic audio.
    oto --version | --help

--tier takes light, standard or max, limited to the tiers this build has a distinct model for.
--seconds takes a number above 0, up to 86400.

Development flags (run, latency): --mic-uid UID, --sink-uid UID stand in for Oto's devices;
--force-clients runs as if an app were recording; --input-raw FILE loops mono f32le samples
in place of the mic signal (the mic still clocks the stream).
";

/// The longest `--seconds` accepted: a day. Keeps the value well inside what `Duration` and
/// the sample counters hold.
const MAX_SECONDS: f64 = 86_400.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cmd {
    Run,
    Devices,
    Status,
    Latency,
    Bench,
}

impl Cmd {
    fn parse(s: &str) -> Option<Cmd> {
        Some(match s {
            "run" => Cmd::Run,
            "devices" => Cmd::Devices,
            "status" => Cmd::Status,
            "latency" => Cmd::Latency,
            "bench" => Cmd::Bench,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Cmd::Run => "run",
            Cmd::Devices => "devices",
            Cmd::Status => "status",
            Cmd::Latency => "latency",
            Cmd::Bench => "bench",
        }
    }

    /// The flags this subcommand reads. Anything else is an error, not silently ignored.
    fn flags(self) -> &'static [&'static str] {
        const ENGINE: [&str; 4] = ["mic-uid", "sink-uid", "force-clients", "input-raw"];
        const RUN: [&str; 9] = [
            "device",
            "strength",
            "tier",
            "no-denoise",
            "set-default",
            ENGINE[0],
            ENGINE[1],
            ENGINE[2],
            ENGINE[3],
        ];
        const LATENCY: [&str; 9] = [
            "seconds",
            "device",
            "strength",
            "tier",
            "no-denoise",
            ENGINE[0],
            ENGINE[1],
            ENGINE[2],
            ENGINE[3],
        ];
        match self {
            Cmd::Run => &RUN,
            Cmd::Latency => &LATENCY,
            Cmd::Bench => &["model", "tier", "seconds"],
            Cmd::Devices | Cmd::Status => &[],
        }
    }
}

#[derive(Default)]
struct Args {
    device: Option<String>,
    strength: Strength,
    tier: Tier,
    no_denoise: bool,
    set_default: bool,
    seconds: Option<f64>,
    model: Option<dsp::Model>,
    /// `--input-raw`'s file, read by [`engine_options`].
    input_raw: Option<String>,
    opts: EngineOptions,
}

/// What the command line asks for.
enum Invocation {
    Help,
    Version,
    Run(Cmd, Box<Args>),
}

fn tier_name(t: Tier) -> &'static str {
    match t {
        Tier::Light => "light",
        Tier::Standard => "standard",
        Tier::Max => "max",
    }
}

/// Parses `--tier`, refusing tiers this build has no distinct model for: accepting them would
/// silently run another tier's model.
fn parse_tier(s: &str, available: &[Tier]) -> Result<Tier, String> {
    let tier = match s {
        "light" => Tier::Light,
        "standard" => Tier::Standard,
        "max" => Tier::Max,
        _ => return Err(format!("unknown tier {s:?} (light, standard or max)")),
    };
    if available.contains(&tier) {
        return Ok(tier);
    }
    let names: Vec<_> = available.iter().map(|t| tier_name(*t)).collect();
    Err(format!(
        "tier {s:?} is not available in this build (available: {})",
        names.join(", ")
    ))
}

fn parse_seconds(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 && v <= MAX_SECONDS => Ok(v),
        _ => Err(format!(
            "--seconds takes a number above 0, up to {MAX_SECONDS}, not {s:?}"
        )),
    }
}

/// Parses the arguments after the program name. Touches no files and no devices.
fn parse<I>(args: I) -> Result<Invocation, String>
where
    I: IntoIterator,
    I::Item: Into<std::ffi::OsString>,
{
    let mut p = lexopt::Parser::from_args(args);
    let mut a = Args::default();
    let mut cmd = None;
    // Flags are checked against the subcommand at the end, so they may come before it.
    let mut seen: Vec<&'static str> = Vec::new();
    let text = |p: &mut lexopt::Parser| -> Result<String, String> {
        p.value()
            .and_then(|v| v.string())
            .map_err(|e| e.to_string())
    };
    while let Some(arg) = p.next().map_err(|e| e.to_string())? {
        let flag = match arg {
            Short('h') | Long("help") => return Ok(Invocation::Help),
            Short('V') | Long("version") => return Ok(Invocation::Version),
            Value(v) if cmd.is_none() => {
                let v = v.string().map_err(|e| e.to_string())?;
                cmd = Some(Cmd::parse(&v).ok_or_else(|| format!("unknown command {v:?}"))?);
                continue;
            }
            Long("device") => {
                a.device = Some(text(&mut p)?);
                "device"
            }
            Long("strength") => {
                a.strength = match text(&mut p)?.as_str() {
                    "light" => Strength::Light,
                    "balanced" => Strength::Balanced,
                    "strong" => Strength::Strong,
                    s => return Err(format!("unknown strength {s:?}")),
                };
                "strength"
            }
            Long("tier") => {
                a.tier = parse_tier(&text(&mut p)?, oto_engine::available_tiers())?;
                "tier"
            }
            Long("model") => {
                a.model = Some(match text(&mut p)?.as_str() {
                    "rnnoise" => dsp::Model::RNNoise,
                    "dfn3" => dsp::Model::DeepFilterNet3,
                    s => return Err(format!("unknown model {s:?}")),
                });
                "model"
            }
            Long("no-denoise") => {
                a.no_denoise = true;
                "no-denoise"
            }
            Long("set-default") => {
                a.set_default = true;
                "set-default"
            }
            Long("seconds") => {
                a.seconds = Some(parse_seconds(&text(&mut p)?)?);
                "seconds"
            }
            Long("mic-uid") => {
                a.opts.mic_uid = Some(text(&mut p)?);
                "mic-uid"
            }
            Long("sink-uid") => {
                a.opts.sink_uid = Some(text(&mut p)?);
                "sink-uid"
            }
            Long("force-clients") => {
                a.opts.force_clients_active = true;
                "force-clients"
            }
            Long("input-raw") => {
                a.input_raw = Some(text(&mut p)?);
                "input-raw"
            }
            _ => return Err(arg.unexpected().to_string()),
        };
        seen.push(flag);
    }
    let Some(cmd) = cmd else {
        return match seen.first() {
            None => Ok(Invocation::Help),
            Some(f) => Err(format!("--{f} needs a command")),
        };
    };
    if let Some(f) = seen.iter().find(|f| !cmd.flags().contains(f)) {
        return Err(format!("--{f} does not apply to `oto {}`", cmd.name()));
    }
    Ok(Invocation::Run(cmd, Box::new(a)))
}

fn main() {
    let (cmd, args) = match parse(std::env::args_os().skip(1)) {
        Ok(Invocation::Help) => {
            print!("{USAGE}");
            return;
        }
        Ok(Invocation::Version) => {
            println!("oto {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        Ok(Invocation::Run(cmd, args)) => (cmd, *args),
        Err(e) => {
            eprintln!("oto: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let result = match cmd {
        Cmd::Run => run(args),
        Cmd::Devices => devices(),
        Cmd::Status => status(),
        Cmd::Latency => latency(args),
        Cmd::Bench => bench(args),
    };
    if let Err(e) = result {
        eprintln!("oto: {e}");
        std::process::exit(1);
    }
}

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: i32) {
    STOP.store(true, Ordering::Relaxed);
}

unsafe extern "C" {
    fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
}

/// SIGINT (Ctrl-C), SIGTERM and SIGHUP (the terminal closed) all end `oto run` through its
/// normal exit, where dropping the engine restores the default input.
fn install_signal_handlers() {
    const SIGHUP: i32 = 1;
    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;
    unsafe {
        signal(SIGHUP, on_signal);
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
    }
}

/// `Engine::start`'s error, as the CLI says it.
fn start_error(e: String) -> String {
    if e == oto_engine::ALREADY_RUNNING {
        "Oto is already running (the app, or another `oto run`). Quit it first: two engines \
         would both feed Oto Microphone."
            .into()
    } else {
        e
    }
}

/// The engine options, with `--input-raw`'s file read in.
fn engine_options(a: &Args) -> Result<EngineOptions, String> {
    let mut opts = a.opts.clone();
    if let Some(path) = &a.input_raw {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let samples: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        opts.source = Some(samples.into());
    }
    Ok(opts)
}

fn settings(a: &Args) -> Settings {
    Settings {
        enabled: !a.no_denoise,
        input_device: a.device.clone(),
        strength: a.strength,
        tier: a.tier,
    }
}

fn run(a: Args) -> Result<(), String> {
    let opts = engine_options(&a)?;
    install_signal_handlers();
    let engine = Engine::start_with(settings(&a), opts).map_err(start_error)?;
    if a.set_default {
        engine.set_default_input_to_oto()?;
        println!("Oto Microphone is the default input until exit");
    }
    println!(
        "oto running (denoise {}, strength {:?}, tier {:?}); Ctrl-C to stop",
        if a.no_denoise { "off" } else { "on" },
        a.strength,
        a.tier
    );
    let levels = engine.levels();
    let mut last_line = Instant::now();
    let mut line_shown = false;
    let mut load = 0.0f32;
    while !STOP.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(50));
        for e in engine.try_events() {
            if let Event::Load(l) = e {
                load = l;
                continue;
            }
            if line_shown {
                println!();
                line_shown = false;
            }
            match e {
                Event::ClientsActive(true) => println!("an app is recording from Oto Microphone"),
                Event::ClientsActive(false) => println!("no app is recording; idle"),
                Event::DevicesChanged => println!("devices changed"),
                Event::Error(e) => println!("error: {e}"),
                Event::Recovered => println!("recovered"),
                Event::Load(_) => {}
            }
        }
        if last_line.elapsed() >= Duration::from_millis(250) {
            last_line = Instant::now();
            let s = engine.stats();
            if s.running {
                print!(
                    "\rraw {:6.1} dBFS  clean {:6.1} dBFS  ring {:4.1}/{:4.1} ms  drift {:+5} ppm  \
                     xruns {}/{}  est {:4.1} ms  load {:4.1}% ",
                    levels.raw(),
                    levels.clean(),
                    s.ring_fill_ms,
                    s.ring_target_ms,
                    s.drift_ppm,
                    s.underruns,
                    s.resyncs,
                    s.estimated_latency_ms,
                    load * 100.0
                );
                line_shown = true;
                let _ = std::io::stdout().flush();
            }
        }
    }
    println!("\nstopping");
    drop(engine);
    Ok(())
}

fn devices() -> Result<(), String> {
    let list = oto_engine::list_input_devices();
    if list.is_empty() {
        println!("no input devices");
    }
    for d in list {
        println!(
            "{} {}  ({})",
            if d.is_default { "*" } else { " " },
            d.name,
            d.id
        );
    }
    Ok(())
}

fn status() -> Result<(), String> {
    let s = oto_engine::status();
    println!(
        "driver installed: {}",
        if s.driver_installed { "yes" } else { "no" }
    );
    let show = |label: &str, d: &Option<oto_engine::DeviceStatus>| match d {
        Some(d) => println!(
            "{label}: {} ({} Hz, {} ch){}",
            d.name,
            d.sample_rate,
            d.channels,
            if d.in_use { ", in use" } else { "" }
        ),
        None => println!("{label}: not found"),
    };
    show("microphone", &s.microphone);
    show("sink", &s.sink);
    println!(
        "default input: {}",
        s.default_input.as_deref().unwrap_or("none")
    );
    println!("inputs:");
    for d in s.inputs {
        println!(
            "  {} {}  ({})",
            if d.is_default { "*" } else { " " },
            d.name,
            d.id
        );
    }
    if !s.driver_installed {
        return Err("the Oto driver is not installed".into());
    }
    Ok(())
}

fn latency(a: Args) -> Result<(), String> {
    let seconds = a.seconds.unwrap_or(10.0);
    let opts = engine_options(&a)?;
    println!(
        "measuring for {seconds} s: tone bursts injected at the engine input, recorded from Oto Microphone"
    );
    let r = oto_engine::latency::measure(settings(&a), opts, seconds).map_err(start_error)?;
    for e in &r.errors {
        println!("engine error: {e}");
    }
    match r.summary() {
        Some((min, med, max)) => {
            println!(
                "end-to-end (mic IO buffer → chain → ring → sink → driver → recorder): \
                 median {med:.1} ms, min {min:.1}, max {max:.1} ({} of {} bursts)",
                r.measurements_ms.len(),
                r.injections
            );
        }
        None => println!("no bursts detected ({} injected)", r.injections),
    }
    let s = r.stats;
    println!(
        "ring {:.1}/{:.1} ms, drift {:+} ppm, underruns {}, resyncs {}, estimate {:.1} ms \
         (includes device-reported mic latency)",
        s.ring_fill_ms,
        s.ring_target_ms,
        s.drift_ppm,
        s.underruns,
        s.resyncs,
        s.estimated_latency_ms
    );
    Ok(())
}

fn bench(a: Args) -> Result<(), String> {
    let seconds = a.seconds.unwrap_or(10.0);
    let model = a.model.unwrap_or_else(|| oto_engine::model_for(a.tier));
    let report = dsp::try_bench(model, seconds as f32)?;
    println!("{report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(line: &str) -> Result<(Cmd, Args), String> {
        match parse(line.split_whitespace())? {
            Invocation::Run(cmd, a) => Ok((cmd, *a)),
            Invocation::Help => Err("help".into()),
            Invocation::Version => Err("version".into()),
        }
    }

    fn error(line: &str) -> String {
        match args(line) {
            Ok((cmd, _)) => panic!("`oto {line}` parsed as {cmd:?}"),
            Err(e) => e,
        }
    }

    #[test]
    fn version_and_help() {
        assert!(matches!(parse(["--version"]), Ok(Invocation::Version)));
        assert!(matches!(parse(["-V"]), Ok(Invocation::Version)));
        assert!(matches!(
            parse(["status", "--version"]),
            Ok(Invocation::Version)
        ));
        assert!(matches!(parse(["--help"]), Ok(Invocation::Help)));
        assert!(matches!(parse(["-h"]), Ok(Invocation::Help)));
        assert!(matches!(parse(Vec::<&str>::new()), Ok(Invocation::Help)));
    }

    #[test]
    fn subcommands_take_their_own_flags() {
        let (cmd, a) =
            args("run --device usb --strength strong --no-denoise --set-default").unwrap();
        assert_eq!(cmd, Cmd::Run);
        assert_eq!(a.device.as_deref(), Some("usb"));
        assert_eq!(a.strength, Strength::Strong);
        assert!(a.no_denoise && a.set_default);

        let (cmd, a) = args("latency --seconds 2.5 --mic-uid x --force-clients").unwrap();
        assert_eq!(cmd, Cmd::Latency);
        assert_eq!(a.seconds, Some(2.5));
        assert_eq!(a.opts.mic_uid.as_deref(), Some("x"));
        assert!(a.opts.force_clients_active);

        let (cmd, a) = args("bench --model rnnoise --seconds 1").unwrap();
        assert_eq!(cmd, Cmd::Bench);
        assert_eq!(a.model, Some(dsp::Model::RNNoise));

        assert_eq!(args("devices").unwrap().0, Cmd::Devices);
        assert_eq!(args("status").unwrap().0, Cmd::Status);
        // A flag may come before its subcommand.
        assert!(args("--set-default run").unwrap().1.set_default);
        // Parsing reads no files.
        let (_, a) = args("run --input-raw /nonexistent/oto.raw").unwrap();
        assert_eq!(a.input_raw.as_deref(), Some("/nonexistent/oto.raw"));
        assert!(engine_options(&a).is_err());
    }

    #[test]
    fn flags_that_do_not_apply_are_rejected() {
        assert_eq!(
            error("devices --set-default"),
            "--set-default does not apply to `oto devices`"
        );
        assert_eq!(
            error("--set-default devices"),
            "--set-default does not apply to `oto devices`"
        );
        assert_eq!(
            error("status --device x"),
            "--device does not apply to `oto status`"
        );
        assert_eq!(
            error("run --seconds 3"),
            "--seconds does not apply to `oto run`"
        );
        assert_eq!(
            error("run --model rnnoise"),
            "--model does not apply to `oto run`"
        );
        assert_eq!(
            error("latency --set-default"),
            "--set-default does not apply to `oto latency`"
        );
        assert_eq!(
            error("bench --device x"),
            "--device does not apply to `oto bench`"
        );
        assert_eq!(
            error("bench --force-clients"),
            "--force-clients does not apply to `oto bench`"
        );
        assert_eq!(error("--no-denoise"), "--no-denoise needs a command");
    }

    #[test]
    fn unknown_things_are_errors() {
        assert_eq!(error("frobnicate"), "unknown command \"frobnicate\"");
        assert!(error("run --frobnicate").contains("--frobnicate"));
        assert!(error("run extra").contains("extra"));
        assert!(error("run --strength loud").contains("unknown strength"));
        assert!(error("bench --model big").contains("unknown model"));
        assert!(error("run --device").contains("--device"));
    }

    #[test]
    fn seconds_must_be_finite_and_positive() {
        for bad in [
            "0", "-1", "nan", "NaN", "inf", "-inf", "1e300", "abc", "86401",
        ] {
            for cmd in ["bench", "latency"] {
                let e = error(&format!("{cmd} --seconds {bad}"));
                assert!(e.contains("--seconds"), "{cmd} {bad}: {e}");
            }
        }
        assert_eq!(parse_seconds("0.25"), Ok(0.25));
        assert_eq!(parse_seconds("86400"), Ok(86_400.0));
        // lexopt's `--seconds=N` form goes the same way.
        assert_eq!(args("bench --seconds=3").unwrap().1.seconds, Some(3.0));
        assert!(error("bench --seconds=-3").contains("--seconds"));
    }

    #[test]
    fn tiers_without_their_own_model_are_rejected() {
        let one = [Tier::Standard];
        assert_eq!(parse_tier("standard", &one), Ok(Tier::Standard));
        assert_eq!(
            parse_tier("max", &one).unwrap_err(),
            "tier \"max\" is not available in this build (available: standard)"
        );
        assert_eq!(
            parse_tier("light", &[Tier::Light, Tier::Standard]),
            Ok(Tier::Light)
        );
        assert!(
            parse_tier("max", &[Tier::Light, Tier::Standard])
                .unwrap_err()
                .contains("available: light, standard")
        );
        assert!(
            parse_tier("huge", &one)
                .unwrap_err()
                .contains("unknown tier")
        );

        // The real flag follows this build's tiers.
        for t in [Tier::Light, Tier::Standard, Tier::Max] {
            let parsed = args(&format!("run --tier {}", tier_name(t)));
            if oto_engine::available_tiers().contains(&t) {
                assert_eq!(parsed.unwrap().1.tier, t);
            } else {
                assert!(parsed.err().unwrap().contains("not available"));
            }
        }
    }

    #[test]
    fn already_running_gets_its_own_message() {
        assert!(start_error(oto_engine::ALREADY_RUNNING.into()).contains("Quit it first"));
        assert_eq!(start_error("boom".into()), "boom");
    }
}
