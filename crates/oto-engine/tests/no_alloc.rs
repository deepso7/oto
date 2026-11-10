//! The realtime contract of plan 0001, decision 3, checked on the engine's real capture path:
//! once a stream is running, the audio callbacks never touch the heap.
//!
//! - RNNoise runs inline: `CaptureStage::process` (resampling, chain, bypass and strength
//!   changes through `Shared`) and `OutputStage::render` allocate nothing from the second
//!   callback on. The first callback is the one documented exception: it primes the model on
//!   the audio thread.
//! - DeepFilterNet3 runs on the DSP worker: the callback side (rings, dry path, crossfades)
//!   allocates nothing from the second callback on, while the worker thread may.
//!
//! A counting global allocator counts only on the thread that opted in (the "audio thread"
//! here), so the DSP worker and the test harness do not count. Set `OTO_ALLOC_TRACE=N` to
//! print a backtrace for each of the first N counted heap calls.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use oto_engine::Levels;
use oto_engine::dsp::{Chain, Model, signal};
use oto_engine::pipeline::{
    CaptureStage, OutputStage, Placement, RING_CAPACITY, STOP_FADE_REQUESTED, Shared,
    placement_for, target_fill,
};
use rtrb::RingBuffer;

struct Counting;

static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static TRACE_LEFT: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

fn note(bytes: usize) {
    if !COUNTING.with(Cell::get) {
        return;
    }
    CALLS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(bytes, Ordering::Relaxed);
    if TRACE_LEFT
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
        .is_ok()
    {
        COUNTING.with(|c| c.set(false));
        eprintln!(
            "--- heap call ({bytes} bytes)\n{}",
            std::backtrace::Backtrace::force_capture()
        );
        COUNTING.with(|c| c.set(true));
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note(0);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The counters are global, so tests that count take turns.
static SERIAL: Mutex<()> = Mutex::new(());

/// Heap calls (allocations, reallocations and frees) and bytes requested on this thread by `f`.
fn count<R>(f: impl FnOnce() -> R) -> (usize, usize, R) {
    let trace = std::env::var("OTO_ALLOC_TRACE")
        .ok()
        .map_or(0, |v| v.parse().unwrap_or(1));
    TRACE_LEFT.store(trace, Ordering::Relaxed);
    CALLS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    COUNTING.with(|c| c.set(true));
    let r = f();
    COUNTING.with(|c| c.set(false));
    (
        CALLS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
        r,
    )
}

/// One mic and one sink, wired like the macOS backend (minus the devices).
struct Path {
    shared: Arc<Shared>,
    cap: CaptureStage,
    out: OutputStage,
    block: usize,
    audio: Vec<f32>,
    at: usize,
    sink: Vec<f32>,
    rate: f64,
    /// 48 kHz frames captured but not yet rendered by the sink.
    owed: f64,
}

impl Path {
    fn new(rate: f64, block: usize, chain: Chain, placement: Placement) -> Self {
        let shared = Arc::new(Shared::new(Arc::new(Levels::default()), None));
        let (prod, cons) = RingBuffer::new(RING_CAPACITY);
        let cap = CaptureStage::with_placement(rate, block, chain, prod, shared.clone(), placement)
            .unwrap();
        assert_eq!(cap.placement(), placement);
        let target = target_fill(block, rate, 128, 48_000.0);
        let out = OutputStage::new(cons, 48_000.0, target, shared.clone()).unwrap();
        // Noisy speech with a stretch of digital silence (the chain's skip path).
        let mut audio = signal::noisy_speech(rate as usize * 2, 1);
        audio.extend(std::iter::repeat_n(0.0, rate as usize / 2));
        audio.extend(signal::noisy_speech(rate as usize, 2));
        Self {
            shared,
            cap,
            out,
            block,
            audio,
            at: 0,
            sink: vec![0.0; 128],
            rate,
            owed: 0.0,
        }
    }

    /// One capture callback and the sink callbacks that fall in its period.
    fn callback(&mut self, i: usize) {
        let s = &self.shared;
        // The setters the control thread uses, at arbitrary callbacks.
        match i % 50 {
            10 => s.denoise.store(false, Relaxed),
            20 => s.denoise.store(true, Relaxed),
            25 => s.max_atten_bits.store(12f32.to_bits(), Relaxed),
            30 => s.max_atten_bits.store(f32::INFINITY.to_bits(), Relaxed),
            35 => s.max_atten_bits.store(24f32.to_bits(), Relaxed),
            _ => {}
        }
        let n = self.block;
        for x in &mut self.cap.mono_mut()[..n] {
            *x = self.audio[self.at];
            self.at = (self.at + 1) % self.audio.len();
        }
        self.cap.process(n, 0);
        // The sink consumes at 48 kHz what the mic delivered at its rate.
        self.owed += n as f64 * 48_000.0 / self.rate;
        while self.owed >= 128.0 {
            self.out.render(&mut self.sink);
            self.owed -= 128.0;
        }
    }
}

#[test]
fn inline_capture_path_never_allocates_after_the_first_callback() {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(placement_for(Model::RNNoise), Placement::Inline);
    for (rate, block) in [(48_000.0, 256), (44_100.0, 441), (16_000.0, 85)] {
        let chain = Chain::with_model(Model::RNNoise).unwrap();
        let mut p = Path::new(rate, block, chain, Placement::Inline);
        // The first callback primes RNNoise on this thread: the documented exception.
        p.callback(0);
        let (calls, bytes, ()) = count(|| {
            for i in 1..1500 {
                p.callback(i);
            }
            // A stop fade-out on the way out.
            p.shared.stop_fade.store(STOP_FADE_REQUESTED, Relaxed);
            for i in 1500..1510 {
                p.callback(i);
            }
        });
        assert_eq!(
            calls, 0,
            "{rate} Hz: the capture path made {calls} heap calls ({bytes} bytes)"
        );
        assert_eq!(p.shared.overflows.load(Relaxed), 0, "{rate} Hz");
        assert_eq!(p.shared.starvations.load(Relaxed), 0, "{rate} Hz");
    }
}

/// Runs a worker-placed chain in real time: pacing gives the worker its capture period, and a
/// few unpaced callbacks starve it on purpose so the dry fallback is counted too.
fn worker_callback_side_never_allocates(chain: Chain) {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let model = chain.model();
    let mut p = Path::new(48_000.0, 256, chain, Placement::Worker);
    let period = Duration::from_secs_f64(256.0 / 48_000.0);
    p.callback(0);
    std::thread::sleep(period);
    let (calls, bytes, ()) = count(|| {
        for i in 1..400 {
            p.callback(i);
            if !(200..204).contains(&i) {
                std::thread::sleep(period);
            }
        }
    });
    let s = &p.shared;
    let us = |ns: u64| ns as f64 / 1e3;
    let t = &s.worker_times;
    eprintln!(
        "{}: callback side {calls} heap calls; worker {} blocks, p50 {:.0} µs, p99 {:.0} µs, \
         max {:.0} µs of {:.0} µs, priming {:.0} µs, realtime {}; starved {} (4 forced), \
         reserve {} samples",
        model.name(),
        t.calls(),
        us(t.quantile_ns(0.5).unwrap_or(0)),
        us(t.quantile_ns(0.99).unwrap_or(0)),
        us(t.max_ns()),
        us(t.period_ns.load(Relaxed)),
        us(t.first_ns()),
        s.worker_realtime.load(Relaxed),
        s.starvations.load(Relaxed),
        s.reserve_frames.load(Relaxed),
    );
    assert_eq!(
        calls,
        0,
        "{}: the capture callback made {calls} heap calls ({bytes} bytes)",
        model.name()
    );
    assert!(
        s.worker_times.calls() > 300,
        "the worker processed the audio"
    );
    assert_eq!(s.worker_overflows.load(Relaxed), 0);
}

#[test]
fn rnnoise_on_the_worker_keeps_the_callback_allocation_free() {
    worker_callback_side_never_allocates(Chain::with_model(Model::RNNoise).unwrap());
}

/// DeepFilterNet3 allocates ~1,700 times per frame inside tract; on the worker that is allowed,
/// in the callback it is not.
#[cfg(feature = "dfn3")]
#[test]
fn dfn3_runs_on_the_worker_and_the_callback_stays_allocation_free() {
    assert_eq!(placement_for(Model::DeepFilterNet3), Placement::Worker);
    worker_callback_side_never_allocates(Chain::with_model(Model::DeepFilterNet3).unwrap());
}

#[test]
fn the_counter_works() {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (calls, bytes, v) = count(|| vec![1u8; 16]);
    assert!(calls >= 1 && bytes >= 16);
    drop(v);
}
