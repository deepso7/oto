//! `oto`: the headless front end. Runs the engine without a window, lists devices, reports
//! driver status, and hosts the latency and benchmark tools (see PLAN.md, phase 0).

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use lexopt::prelude::*;
use oto_engine::{Engine, EngineOptions, Event, Settings, Strength, Tier, dsp};

const USAGE: &str = "\
oto: noise-cancelling virtual microphone (headless)

USAGE:
    oto run [--device ID] [--strength light|balanced|strong] [--tier light|standard|max]
            [--no-denoise] [--set-default]
        Run until Ctrl-C, printing events and a level line. Audio only flows while an app
        records from Oto Microphone. --set-default makes Oto Microphone the default input
        until exit.
    oto devices
        List physical input devices (IDs for --device).
    oto status
        Driver and device status.
    oto latency [--seconds N] [--device ID] [--no-denoise]
        Measure end-to-end latency through the real Oto Microphone path (needs the driver).
    oto bench [--tier light|standard|max] [--seconds N]
        Time the DSP chain on synthetic audio.

Development flags (run, latency): --mic-uid UID, --sink-uid UID stand in for Oto's devices;
--force-clients runs as if an app were recording.
";

#[derive(Default)]
struct Args {
    cmd: String,
    device: Option<String>,
    strength: Strength,
    tier: Tier,
    no_denoise: bool,
    set_default: bool,
    seconds: Option<f64>,
    opts: EngineOptions,
}

fn parse() -> Result<Args, lexopt::Error> {
    let mut p = lexopt::Parser::from_env();
    let mut a = Args::default();
    while let Some(arg) = p.next()? {
        match arg {
            Value(v) if a.cmd.is_empty() => a.cmd = v.string()?,
            Long("device") => a.device = Some(p.value()?.string()?),
            Long("strength") => {
                a.strength = match p.value()?.string()?.as_str() {
                    "light" => Strength::Light,
                    "balanced" => Strength::Balanced,
                    "strong" => Strength::Strong,
                    s => return Err(format!("unknown strength {s:?}").into()),
                }
            }
            Long("tier") => {
                a.tier = match p.value()?.string()?.as_str() {
                    "light" => Tier::Light,
                    "standard" => Tier::Standard,
                    "max" => Tier::Max,
                    s => return Err(format!("unknown tier {s:?}").into()),
                }
            }
            Long("no-denoise") => a.no_denoise = true,
            Long("set-default") => a.set_default = true,
            Long("seconds") => a.seconds = Some(p.value()?.parse()?),
            Long("mic-uid") => a.opts.mic_uid = Some(p.value()?.string()?),
            Long("sink-uid") => a.opts.sink_uid = Some(p.value()?.string()?),
            Long("force-clients") => a.opts.force_clients_active = true,
            Short('h') | Long("help") => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }
    Ok(a)
}

fn main() {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("oto: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let result = match args.cmd.as_str() {
        "run" => run(args),
        "devices" => devices(),
        "status" => status(),
        "latency" => latency(args),
        "bench" => bench(args),
        "" => {
            print!("{USAGE}");
            Ok(())
        }
        c => Err(format!("unknown command {c:?} (try --help)")),
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

fn install_ctrl_c() {
    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;
    unsafe {
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
    }
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
    install_ctrl_c();
    let engine = Engine::start_with(settings(&a), a.opts.clone())?;
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
    Ok(())
}

fn latency(a: Args) -> Result<(), String> {
    let seconds = a.seconds.unwrap_or(10.0);
    println!(
        "measuring for {seconds} s: tone bursts injected at the engine input, recorded from Oto Microphone"
    );
    let r = oto_engine::latency::measure(settings(&a), a.opts.clone(), seconds)?;
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
    let model = oto_engine::model_for(a.tier);
    let report = dsp::try_bench(model, seconds as f32)?;
    println!("{}: {report}", model.name());
    Ok(())
}
