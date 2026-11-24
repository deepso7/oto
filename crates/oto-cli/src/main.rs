//! `oto`: the headless front end. Runs the engine without a window, lists devices, reports
//! driver status, and hosts the latency and benchmark tools (see docs/roadmap.md).

use std::fmt::Write as _;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use lexopt::prelude::*;
use oto_engine::offline::{self, ProcessOptions, ProcessReport};
use oto_engine::{
    CallbackStats, Engine, EngineOptions, Event, Settings, Stimulus, Strength, Tier, dsp,
};

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
                [--stimulus tone|speech] [--json]
        Measure end-to-end latency through the real Oto Microphone path (needs the driver).
        --no-denoise keeps the tier's model and its delay but bypasses its processing, so it
        cannot suppress the test tone. --stimulus speech injects voiced bursts the models
        keep, to measure with denoising on. Exits 1 if fewer than 80% of the bursts arrive.
    oto bench [--model rnnoise|dfn3|dpdfnet|fastenhancer] [--tier TIER] [--seconds N] [--json]
        Time the DSP chain on synthetic audio.
    oto bench --live [--tier TIER] [--seconds N] [--device ID] [--strength S] [--no-denoise]
                     [--json]
        Run the real engine as if an app were recording and report callback timing
        (p50/p99/p99.9/max against the period, deadline misses), xruns, CPU and latency.
    oto process INPUT.wav OUTPUT.wav [--model M] [--tier TIER] [--strength S] [--no-denoise]
                [--chunk-size N] [--no-delay-comp] [--json]
        Run a WAV file (16/24/32-bit PCM or 32-bit float, any rate and channel count) through
        the live chain offline. Writes mono at the input's rate, aligned with the input and of
        the same length unless --no-delay-comp keeps the chain's delay. --chunk-size is the
        block size fed to the chain at 48 kHz (default 480).
    oto --version | --help

--tier takes standard (DeepFilterNet3, the default), light (RNNoise, the app's Low CPU
mode), max (DPDFNet-2) or fast (FastEnhancer-S), limited to the tiers this build has a
distinct model for.
--seconds takes a number above 0, up to 86400.

--json prints one JSON object on stdout instead of text.

Development flags (run, latency, bench --live): --mic-uid UID, --sink-uid UID stand in for Oto's devices;
--force-clients runs as if an app were recording; --input-raw FILE loops mono f32le samples
in place of the mic signal (the mic still clocks the stream).
";

/// The longest `--seconds` accepted: a day. Keeps the value well inside what `Duration` and
/// the sample counters hold.
const MAX_SECONDS: f64 = 86_400.0;

/// The largest `--chunk-size` accepted (about 1.4 s at 48 kHz).
const MAX_CHUNK: usize = 65_536;

/// `oto bench` flags that only mean something with `--live`.
const LIVE_ONLY: [&str; 6] = [
    "device",
    "strength",
    "no-denoise",
    "mic-uid",
    "sink-uid",
    "input-raw",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cmd {
    Run,
    Devices,
    Status,
    Latency,
    Bench,
    Process,
}

impl Cmd {
    fn parse(s: &str) -> Option<Cmd> {
        Some(match s {
            "run" => Cmd::Run,
            "devices" => Cmd::Devices,
            "status" => Cmd::Status,
            "latency" => Cmd::Latency,
            "bench" => Cmd::Bench,
            "process" => Cmd::Process,
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
            Cmd::Process => "process",
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
        const LATENCY: [&str; 11] = [
            "seconds",
            "device",
            "strength",
            "tier",
            "no-denoise",
            "stimulus",
            "json",
            ENGINE[0],
            ENGINE[1],
            ENGINE[2],
            ENGINE[3],
        ];
        const BENCH: [&str; 11] = [
            "model",
            "tier",
            "seconds",
            "json",
            "live",
            LIVE_ONLY[0],
            LIVE_ONLY[1],
            LIVE_ONLY[2],
            LIVE_ONLY[3],
            LIVE_ONLY[4],
            LIVE_ONLY[5],
        ];
        const PROCESS: [&str; 7] = [
            "model",
            "tier",
            "strength",
            "no-denoise",
            "chunk-size",
            "no-delay-comp",
            "json",
        ];
        match self {
            Cmd::Run => &RUN,
            Cmd::Latency => &LATENCY,
            Cmd::Bench => &BENCH,
            Cmd::Process => &PROCESS,
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
    json: bool,
    live: bool,
    stimulus: Stimulus,
    chunk_size: Option<usize>,
    no_delay_comp: bool,
    /// `oto process`'s input and output files.
    paths: Vec<String>,
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
        Tier::Fast => "fast",
    }
}

/// Parses `--tier`, refusing tiers this build has no distinct model for: accepting them would
/// silently run another tier's model.
fn parse_tier(s: &str, available: &[Tier]) -> Result<Tier, String> {
    let tier = match s {
        "light" => Tier::Light,
        "standard" => Tier::Standard,
        "max" => Tier::Max,
        "fast" => Tier::Fast,
        _ => return Err(format!("unknown tier {s:?} (light, standard, max or fast)")),
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

fn parse_chunk_size(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(v) if (1..=MAX_CHUNK).contains(&v) => Ok(v),
        _ => Err(format!(
            "--chunk-size takes a whole number from 1 to {MAX_CHUNK}, not {s:?}"
        )),
    }
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
            Value(v) if cmd == Some(Cmd::Process) && a.paths.len() < 2 => {
                a.paths.push(v.string().map_err(|e| e.to_string())?);
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
                    "dpdfnet" => dsp::Model::DpdfNet,
                    "fastenhancer" => dsp::Model::FastEnhancer,
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
            Long("json") => {
                a.json = true;
                "json"
            }
            Long("live") => {
                a.live = true;
                "live"
            }
            Long("stimulus") => {
                a.stimulus = match text(&mut p)?.as_str() {
                    "tone" => Stimulus::Tone,
                    "speech" => Stimulus::Speech,
                    s => return Err(format!("unknown stimulus {s:?} (tone or speech)")),
                };
                "stimulus"
            }
            Long("chunk-size") => {
                a.chunk_size = Some(parse_chunk_size(&text(&mut p)?)?);
                "chunk-size"
            }
            Long("no-delay-comp") => {
                a.no_delay_comp = true;
                "no-delay-comp"
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
    if cmd == Cmd::Bench {
        if !a.live
            && let Some(f) = seen.iter().find(|f| LIVE_ONLY.contains(f))
        {
            return Err(format!("--{f} needs --live (`oto bench --live`)"));
        }
        if a.live && a.model.is_some() {
            return Err("--model does not apply to `oto bench --live`; it runs a --tier".into());
        }
    }
    if cmd == Cmd::Process && a.paths.len() != 2 {
        return Err("`oto process` needs an input and an output WAV file".into());
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
        Cmd::Bench if args.live => bench_live(args),
        Cmd::Bench => bench(args),
        Cmd::Process => process(args),
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
    let mut model = None;
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
                Event::ModelFallback {
                    requested,
                    effective,
                    reason,
                } => println!(
                    "notice: {} could not be loaded, running {} instead ({reason})",
                    requested.name(),
                    effective.name()
                ),
                Event::Load(_) => {}
            }
        }
        if last_line.elapsed() >= Duration::from_millis(250) {
            last_line = Instant::now();
            let s = engine.stats();
            if s.model != model {
                model = s.model;
                if let Some(m) = model {
                    if line_shown {
                        println!();
                        line_shown = false;
                    }
                    println!(
                        "model: {} (est. latency {:.1} ms)",
                        m.name(),
                        s.estimated_latency_ms
                    );
                }
            }
            if s.running {
                print!(
                    "\rraw {:6.1} dBFS  clean {:6.1} dBFS  ring {:4.1}/{:4.1} ms  drift {:+5} ppm  \
                     xruns {}/{}  est {:4.1} ms  load {:4.1}%  cb in {} out {} ",
                    levels.raw(),
                    levels.clean(),
                    s.ring_fill_ms,
                    s.ring_target_ms,
                    s.drift_ppm,
                    s.underruns,
                    s.resyncs,
                    s.estimated_latency_ms,
                    load * 100.0,
                    callback_times(&s.capture),
                    callback_times(&s.output),
                );
                if s.worker.callbacks > 0 {
                    // DeepFilterNet3 runs on the worker thread (design B in docs/plans/0001).
                    print!(
                        "worker {} starved {} ovf {} ",
                        callback_times(&s.worker),
                        s.starvations,
                        s.worker_overflows,
                    );
                }
                line_shown = true;
                let _ = std::io::stdout().flush();
            }
        }
    }
    println!("\nstopping");
    drop(engine);
    Ok(())
}

/// One stream's callback time as `p50/p99/max of period` in µs, e.g. `120/310/900 of 5333µs`.
fn callback_times(c: &oto_engine::CallbackStats) -> String {
    format!(
        "{:.0}/{:.0}/{:.0} of {:.0}µs",
        c.p50_us, c.p99_us, c.max_us, c.period_us
    )
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

/// A JSON object written by hand: the output is flat numbers and strings, not worth a serde
/// dependency. Non-finite numbers become `null`.
#[derive(Default)]
struct Json(String);

impl Json {
    fn key(&mut self, k: &str) -> &mut String {
        self.0.push(if self.0.is_empty() { '{' } else { ',' });
        let _ = write!(self.0, "{}:", json_str(k));
        &mut self.0
    }

    fn num(mut self, k: &str, v: f64) -> Self {
        let out = self.key(k);
        if v.is_finite() {
            let _ = write!(out, "{v}");
        } else {
            out.push_str("null");
        }
        self
    }

    /// An `f32` in its shortest form (`0.1`, not `0.10000000149011612`).
    fn num32(mut self, k: &str, v: f32) -> Self {
        let out = self.key(k);
        if v.is_finite() {
            let _ = write!(out, "{v}");
        } else {
            out.push_str("null");
        }
        self
    }

    fn opt(self, k: &str, v: Option<f64>) -> Self {
        self.num(k, v.unwrap_or(f64::NAN))
    }

    fn int(mut self, k: &str, v: u64) -> Self {
        let _ = write!(self.key(k), "{v}");
        self
    }

    fn sint(mut self, k: &str, v: i64) -> Self {
        let _ = write!(self.key(k), "{v}");
        self
    }

    fn bool(mut self, k: &str, v: bool) -> Self {
        let _ = write!(self.key(k), "{v}");
        self
    }

    fn str(mut self, k: &str, v: &str) -> Self {
        let v = json_str(v);
        self.key(k).push_str(&v);
        self
    }

    fn obj(mut self, k: &str, v: Json) -> Self {
        let v = v.finish();
        self.key(k).push_str(&v);
        self
    }

    fn strs(mut self, k: &str, v: &[String]) -> Self {
        let items: Vec<_> = v.iter().map(|s| json_str(s)).collect();
        let _ = write!(self.key(k), "[{}]", items.join(","));
        self
    }

    fn nums(mut self, k: &str, v: &[f64]) -> Self {
        let items: Vec<_> = v
            .iter()
            .map(|x| {
                if x.is_finite() {
                    x.to_string()
                } else {
                    "null".into()
                }
            })
            .collect();
        let _ = write!(self.key(k), "[{}]", items.join(","));
        self
    }

    fn finish(mut self) -> String {
        if self.0.is_empty() {
            self.0.push('{');
        }
        self.0.push('}');
        self.0
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn callback_json(c: &CallbackStats) -> Json {
    Json::default()
        .int("callbacks", c.callbacks)
        .num32("period_us", c.period_us)
        .num32("p50_us", c.p50_us)
        .num32("p99_us", c.p99_us)
        .num32("p999_us", c.p999_us)
        .num32("max_us", c.max_us)
        .num32("first_us", c.first_us)
        .int("deadline_misses", c.deadline_misses)
}

fn stats_json(s: &oto_engine::Stats) -> Json {
    Json::default()
        .bool("running", s.running)
        .str("model", s.model.map_or("", |m| m.name()))
        .int("input_rate", s.input_rate as u64)
        .int("output_rate", s.output_rate as u64)
        .num32("estimated_latency_ms", s.estimated_latency_ms)
        .num32("ring_fill_ms", s.ring_fill_ms)
        .num32("ring_target_ms", s.ring_target_ms)
        .num32("worker_reserve_ms", s.worker_reserve_ms)
        .bool("worker_realtime", s.worker_realtime)
        .bool("worker_workgroup", s.worker_workgroup)
        .sint("drift_ppm", s.drift_ppm as i64)
        .int("underruns", s.underruns)
        .int("resyncs", s.resyncs)
        .int("overflows", s.overflows)
        .int("starvations", s.starvations)
        .int("worker_overflows", s.worker_overflows)
        .obj("capture", callback_json(&s.capture))
        .obj("output", callback_json(&s.output))
        .obj("worker", callback_json(&s.worker))
}

/// Engine errors and model fallbacks seen while a measurement ran.
#[derive(Default)]
struct Incidents {
    errors: Vec<String>,
    notices: Vec<String>,
    loads: Vec<f32>,
}

impl Incidents {
    fn note(&mut self, e: Event) {
        match e {
            Event::Error(e) => self.errors.push(e),
            Event::ModelFallback {
                requested,
                effective,
                reason,
            } => self.notices.push(format!(
                "{} \u{2192} {}: {reason}",
                requested.name(),
                effective.name()
            )),
            Event::Load(l) => self.loads.push(l),
            _ => {}
        }
    }

    /// Mean share of one core over the run, percent.
    fn cpu_percent(&self) -> Option<f64> {
        (!self.loads.is_empty()).then(|| {
            self.loads.iter().map(|&l| l as f64).sum::<f64>() * 100.0 / self.loads.len() as f64
        })
    }
}

/// The tier and what it runs, for progress lines.
fn describe_run(a: &Args) -> String {
    format!(
        "tier {}, {} {}",
        tier_name(a.tier),
        oto_engine::model_for(a.tier).name(),
        if a.no_denoise {
            "bypassed, delay kept"
        } else {
            "denoising"
        },
    )
}

fn latency(a: Args) -> Result<(), String> {
    let seconds = a.seconds.unwrap_or(10.0);
    let opts = engine_options(&a)?;
    let intro = format!(
        "measuring for {seconds} s ({}): {} bursts injected at the engine input, recorded \
         from Oto Microphone",
        describe_run(&a),
        a.stimulus.name(),
    );
    // Keep stdout pure JSON with --json.
    if a.json {
        eprintln!("{intro}");
    } else {
        println!("{intro}");
    }
    let r = oto_engine::latency::measure_with_stimulus(settings(&a), opts, seconds, a.stimulus)
        .map_err(start_error)?;
    let s = r.stats;
    if a.json {
        let summary = r.summary();
        let json = Json::default()
            .str("platform", std::env::consts::OS)
            .str("stimulus", r.stimulus.name())
            .str("tier", tier_name(a.tier))
            .bool("denoise", !a.no_denoise)
            .num("seconds", seconds)
            .int("injections", r.injections)
            .int("expected", r.expected())
            .int("detected", r.measurements_ms.len() as u64)
            .int("missed", r.missed())
            .num("miss_rate", r.miss_rate())
            .opt("min_ms", summary.map(|s| s.0))
            .opt("median_ms", summary.map(|s| s.1))
            .opt("p95_ms", r.p95_ms())
            .opt("max_ms", summary.map(|s| s.2))
            .nums("measurements_ms", &r.measurements_ms)
            .bool("ok", r.check().is_ok())
            .strs("errors", &r.errors)
            .strs("notices", &r.notices)
            .obj("stats", stats_json(&s));
        println!("{}", json.finish());
        return r.check();
    }
    for e in &r.errors {
        println!("engine error: {e}");
    }
    for n in &r.notices {
        println!("notice: model fallback, {n}");
    }
    if let Some(m) = s.model {
        println!("running model: {}", m.name());
    }
    match r.summary() {
        Some((min, med, max)) => {
            println!(
                "end-to-end (mic IO buffer → chain → ring → sink → driver → recorder): \
                 median {med:.1} ms, p95 {:.1}, min {min:.1}, max {max:.1} ({} of {} bursts, \
                 {} missed)",
                r.p95_ms().unwrap_or(f64::NAN),
                r.measurements_ms.len(),
                r.expected(),
                r.missed(),
            );
        }
        None => println!("no bursts detected ({} expected)", r.expected()),
    }
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
    if s.worker.callbacks > 0 {
        println!(
            "worker {}, starved {}",
            callback_detail(&s.worker),
            s.starvations
        );
    }
    r.check()
}

fn bench(a: Args) -> Result<(), String> {
    let seconds = a.seconds.unwrap_or(10.0);
    let model = a.model.unwrap_or_else(|| oto_engine::model_for(a.tier));
    let r = dsp::try_bench(model, seconds as f32)?;
    if a.json {
        let json = Json::default()
            .str("model", r.model.name())
            .int("frames", r.frames as u64)
            .int("block", r.block as u64)
            .num("p50_us", r.p50_us)
            .num("p99_us", r.p99_us)
            .num("p999_us", r.p999_us)
            .num("max_us", r.max_us)
            .num("mean_us", r.mean_us)
            .num("realtime_factor", r.realtime_factor)
            .num("cpu_percent", r.cpu_percent)
            .num("latency_ms", r.latency_ms)
            .int("latency_samples", r.latency_samples as u64);
        println!("{}", json.finish());
    } else {
        println!("{r}");
    }
    Ok(())
}

/// `oto bench --live`: the real engine, forced active, for `--seconds`.
fn bench_live(a: Args) -> Result<(), String> {
    let seconds = a.seconds.unwrap_or(10.0);
    let mut opts = engine_options(&a)?;
    opts.force_clients_active = true;
    let intro = format!("live bench for {seconds} s ({})", describe_run(&a));
    if a.json {
        eprintln!("{intro}");
    } else {
        println!("{intro}");
    }
    install_signal_handlers();
    let engine = Engine::start_with(settings(&a), opts).map_err(start_error)?;
    let mut seen = Incidents::default();
    let end = Instant::now() + Duration::from_secs_f64(seconds);
    while Instant::now() < end && !STOP.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(50));
        engine.try_events().into_iter().for_each(|e| seen.note(e));
    }
    let s = engine.stats();
    engine.try_events().into_iter().for_each(|e| seen.note(e));
    drop(engine);
    let cpu = seen.cpu_percent();
    if a.json {
        let json = Json::default()
            .str("platform", std::env::consts::OS)
            .str("tier", tier_name(a.tier))
            .bool("denoise", !a.no_denoise)
            .num("seconds", seconds)
            .opt("cpu_percent", cpu)
            .strs("errors", &seen.errors)
            .strs("notices", &seen.notices)
            .obj("stats", stats_json(&s));
        println!("{}", json.finish());
    } else {
        for e in &seen.errors {
            println!("engine error: {e}");
        }
        for n in &seen.notices {
            println!("notice: model fallback, {n}");
        }
        println!(
            "model {}, est. latency {:.1} ms, cpu {}",
            s.model.map_or("none", |m| m.name()),
            s.estimated_latency_ms,
            cpu.map_or("n/a".into(), |c| format!("{c:.1}%")),
        );
        println!("capture {}", callback_detail(&s.capture));
        println!("output  {}", callback_detail(&s.output));
        if s.worker.callbacks > 0 {
            println!("worker  {}", callback_detail(&s.worker));
        }
        println!(
            "starvations {}, worker overflows {}, underruns {}, resyncs {}",
            s.starvations, s.worker_overflows, s.underruns, s.resyncs
        );
    }
    if s.capture.callbacks == 0 {
        return Err(
            "no audio callbacks ran; is the Oto driver installed and a mic present?".into(),
        );
    }
    Ok(())
}

/// One stream's callback timing in full, e.g.
/// `p50 120 / p99 310 / p99.9 600 / max 900 of 5333 µs, 1875 callbacks, 0 deadline misses`.
fn callback_detail(c: &CallbackStats) -> String {
    format!(
        "p50 {:.0} / p99 {:.0} / p99.9 {:.0} / max {:.0} of {:.0} µs, {} callbacks, {} deadline \
         misses",
        c.p50_us, c.p99_us, c.p999_us, c.max_us, c.period_us, c.callbacks, c.deadline_misses
    )
}

/// The offline options `oto process` asks for.
fn process_options(a: &Args) -> ProcessOptions {
    ProcessOptions {
        model: a.model.unwrap_or_else(|| oto_engine::model_for(a.tier)),
        enabled: !a.no_denoise,
        max_attenuation_db: a.strength.max_attenuation_db(),
        chunk_size: a.chunk_size.unwrap_or(offline::DEFAULT_CHUNK),
        delay_compensate: !a.no_delay_comp,
    }
}

fn process(a: Args) -> Result<(), String> {
    let (input, output) = (Path::new(&a.paths[0]), Path::new(&a.paths[1]));
    let r = offline::process_file(input, output, &process_options(&a), None)?;
    if a.json {
        println!("{}", process_json(&r, &a.paths[0], &a.paths[1]).finish());
    } else {
        println!(
            "{} -> {}: {} ({}, max attenuation {} dB), {} Hz, {} samples, chain latency {} \
             samples at 48 kHz ({}), {:.0} ms ({:.1}x realtime), rms {:.1} -> {:.1} dBFS, \
             peak {:.1} dBFS",
            a.paths[0],
            a.paths[1],
            r.model.name(),
            if r.enabled { "denoising" } else { "bypassed" },
            r.max_attenuation_db,
            r.sample_rate,
            r.output_samples,
            r.chain_latency_samples,
            if r.delay_compensated {
                "compensated".to_string()
            } else {
                format!("{} samples kept in front", r.total_delay_samples)
            },
            r.elapsed_ms,
            r.realtime_factor,
            r.input_rms_dbfs,
            r.output_rms_dbfs,
            r.peak_dbfs,
        );
    }
    if r.non_finite_outputs > 0 {
        return Err(format!(
            "{} non-finite output samples",
            r.non_finite_outputs
        ));
    }
    Ok(())
}

fn process_json(r: &ProcessReport, input: &str, output: &str) -> Json {
    Json::default()
        .str("input", input)
        .str("output", output)
        .str("model", r.model.name())
        .bool("denoise", r.enabled)
        .num32("max_attenuation_db", r.max_attenuation_db)
        .int("chunk_size", r.chunk_size as u64)
        .int("sample_rate", r.sample_rate as u64)
        .int("input_samples", r.input_samples as u64)
        .int("output_samples", r.output_samples as u64)
        .int("chain_latency_samples", r.chain_latency_samples as u64)
        .int("total_delay_samples", r.total_delay_samples as u64)
        .bool("delay_compensated", r.delay_compensated)
        .num("elapsed_ms", r.elapsed_ms)
        .num("realtime_factor", r.realtime_factor)
        .num32("input_rms_dbfs", r.input_rms_dbfs)
        .num32("output_rms_dbfs", r.output_rms_dbfs)
        .num32("peak_dbfs", r.peak_dbfs)
        .int("non_finite_outputs", r.non_finite_outputs as u64)
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

        let (cmd, a) = args("latency --no-denoise --tier standard").unwrap();
        assert_eq!(cmd, Cmd::Latency);
        assert!(a.no_denoise);
        assert_eq!(a.tier, Tier::Standard);
        // Bypassed, not idle: the tier's model still loads and keeps its delay.
        let s = settings(&a);
        assert!(!s.enabled);
        assert_eq!(s.tier, Tier::Standard);

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
            "--device needs --live (`oto bench --live`)"
        );
        assert_eq!(
            error("process a.wav b.wav --device x"),
            "--device does not apply to `oto process`"
        );
        assert_eq!(
            error("run --stimulus speech"),
            "--stimulus does not apply to `oto run`"
        );
        assert_eq!(error("run --json"), "--json does not apply to `oto run`");
        assert_eq!(
            error("bench --live --mic-uid x --force-clients"),
            "--force-clients does not apply to `oto bench`"
        );
        assert_eq!(
            error("latency --chunk-size 256"),
            "--chunk-size does not apply to `oto latency`"
        );
        assert_eq!(
            error("bench --force-clients"),
            "--force-clients does not apply to `oto bench`"
        );
        assert_eq!(error("--no-denoise"), "--no-denoise needs a command");
    }

    #[test]
    fn bench_live_and_json() {
        let (cmd, a) = args("bench --json").unwrap();
        assert_eq!(cmd, Cmd::Bench);
        assert!(a.json && !a.live);
        let (_, a) = args(
            "bench --live --seconds 3 --device usb --strength light --no-denoise --mic-uid m \
             --sink-uid s --input-raw x.raw --json",
        )
        .unwrap();
        assert!(a.live && a.json && a.no_denoise);
        assert_eq!(a.device.as_deref(), Some("usb"));
        assert_eq!(a.strength, Strength::Light);
        assert_eq!(a.opts.sink_uid.as_deref(), Some("s"));
        for f in [
            "--strength strong",
            "--no-denoise",
            "--mic-uid m",
            "--input-raw f",
        ] {
            let e = error(&format!("bench {f}"));
            assert!(e.contains("needs --live"), "{f}: {e}");
        }
        assert!(error("bench --live --model rnnoise").contains("--tier"));
        // `--model` stays fine offline.
        assert!(args("bench --model rnnoise --json").is_ok());
    }

    #[test]
    fn latency_stimulus() {
        let (_, a) = args("latency").unwrap();
        assert_eq!(a.stimulus, Stimulus::Tone);
        let (_, a) = args("latency --stimulus speech --json").unwrap();
        assert_eq!(a.stimulus, Stimulus::Speech);
        assert!(a.json);
        assert!(error("latency --stimulus noise").contains("unknown stimulus"));
    }

    #[test]
    fn process_takes_two_files_and_its_flags() {
        let (cmd, a) = args(
            "process in.wav out.wav --model rnnoise --strength strong --no-denoise \
             --chunk-size 256 --no-delay-comp --json",
        )
        .unwrap();
        assert_eq!(cmd, Cmd::Process);
        assert_eq!(a.paths, ["in.wav", "out.wav"]);
        let o = process_options(&a);
        assert_eq!(o.model, dsp::Model::RNNoise);
        assert!(!o.enabled && !o.delay_compensate);
        assert_eq!(o.chunk_size, 256);
        assert_eq!(o.max_attenuation_db, f32::INFINITY);
        assert!(a.json);
        // Defaults: the tier's model, 480-sample blocks, aligned output.
        let (_, a) = args("process --tier standard in.wav out.wav").unwrap();
        let o = process_options(&a);
        assert_eq!(o.model, oto_engine::model_for(Tier::Standard));
        assert!(o.enabled && o.delay_compensate);
        assert_eq!(o.chunk_size, 480);
        assert_eq!(o.max_attenuation_db, 24.0);

        let need = "`oto process` needs an input and an output WAV file";
        assert_eq!(error("process"), need);
        assert_eq!(error("process in.wav"), need);
        assert!(error("process a b c").contains("c"));
        // Only `process` takes file arguments.
        assert!(error("bench in.wav").contains("in.wav"));
        for bad in ["0", "-1", "65537", "abc", "1.5"] {
            let e = error(&format!("process a b --chunk-size {bad}"));
            assert!(e.contains("--chunk-size"), "{bad}: {e}");
        }
        assert_eq!(parse_chunk_size("65536"), Ok(65_536));
    }

    #[test]
    fn json_is_well_formed() {
        assert_eq!(Json::default().finish(), "{}");
        let j = Json::default()
            .str("s", "a\"b\\c\n\u{1}")
            .num("x", 1.5)
            .num("nan", f64::NAN)
            .num32("f", 0.1)
            .int("n", 7)
            .sint("d", -3)
            .bool("b", true)
            .opt("none", None)
            .strs("l", &["x".into(), "y".into()])
            .nums("v", &[1.0, f64::INFINITY])
            .obj("o", Json::default().int("k", 1))
            .finish();
        assert_eq!(
            j,
            r#"{"s":"a\"b\\c\n\u0001","x":1.5,"nan":null,"f":0.1,"n":7,"d":-3,"b":true,"none":null,"l":["x","y"],"v":[1,null],"o":{"k":1}}"#
        );
    }

    #[test]
    fn process_writes_an_aligned_file() {
        let dir = std::env::temp_dir().join(format!("oto-cli-process-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (inp, outp) = (dir.join("in.wav"), dir.join("out.wav"));
        let x = dsp::signal::noisy_speech(4_800, 1);
        offline::Wav::mono(48_000, offline::SampleFormat::Float32, x)
            .write(&inp)
            .unwrap();
        let line = format!(
            "process {} {} --model rnnoise --chunk-size 256 --json",
            inp.display(),
            outp.display()
        );
        let (_, a) = args(&line).unwrap();
        process(a).unwrap();
        let out = offline::Wav::read(&outp).unwrap();
        assert_eq!((out.frames(), out.sample_rate), (4_800, 48_000));
        let _ = std::fs::remove_dir_all(&dir);
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

        // With DeepFilterNet3 in the build, light is RNNoise and standard is DeepFilterNet3.
        if oto_engine::available_tiers().len() > 1 {
            let tier = |line: &str| args(line).unwrap().1.tier;
            assert_eq!(
                oto_engine::model_for(tier("run --tier light")),
                dsp::Model::RNNoise
            );
            assert_eq!(
                oto_engine::model_for(tier("run --tier standard")),
                dsp::Model::DeepFilterNet3
            );
            assert_eq!(
                oto_engine::model_for(tier("latency --tier light --no-denoise")),
                dsp::Model::RNNoise
            );
        }

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
