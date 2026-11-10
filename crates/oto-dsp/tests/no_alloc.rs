//! Proves `Chain::process` and its realtime-safe setters never touch the heap with RNNoise, and
//! measures how far DeepFilterNet3 is from that (`dfn3` feature). The engine's own
//! `crates/oto-engine/tests/no_alloc.rs` checks the real capture path, counting from the
//! second callback as decision 3 of plan 0001 states the contract.
//!
//! A counting global allocator that only counts on threads that opted in, so the test harness's
//! own allocations on other threads do not interfere.
//!
//! Set `OTO_ALLOC_TRACE=N` to print a backtrace for each of the first N counted heap calls
//! (`OTO_ALLOC_TRACE=1` for just the first).

use oto_dsp::{Chain, Model, signal};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

/// Allocations, including `realloc` (which may move the block).
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);
/// Bytes requested by `ALLOCS` (the new size, for `realloc`).
static BYTES: AtomicUsize = AtomicUsize::new(0);
/// Backtraces still to print, from `OTO_ALLOC_TRACE`. Read once per `count` call.
static TRACE_LEFT: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

fn note(counter: &AtomicUsize, bytes: usize) {
    if !COUNTING.with(Cell::get) {
        return;
    }
    counter.fetch_add(1, Ordering::Relaxed);
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
        note(&ALLOCS, layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(&ALLOCS, layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(&ALLOCS, new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note(&FREES, 0);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[derive(Debug, Clone, Copy)]
struct Heap {
    allocs: usize,
    frees: usize,
    bytes: usize,
}

impl Heap {
    /// Any heap call at all, which is what the realtime contract forbids.
    fn calls(self) -> usize {
        self.allocs + self.frees
    }
}

/// The counters are global, so tests that count take turns (the harness runs them in parallel).
static SERIAL: Mutex<()> = Mutex::new(());

fn count<R>(f: impl FnOnce() -> R) -> (Heap, R) {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let trace = std::env::var("OTO_ALLOC_TRACE")
        .ok()
        .map_or(0, |v| v.parse().unwrap_or(1));
    TRACE_LEFT.store(trace, Ordering::Relaxed);
    for c in [&ALLOCS, &FREES, &BYTES] {
        c.store(0, Ordering::Relaxed);
    }
    COUNTING.with(|c| c.set(true));
    let r = f();
    COUNTING.with(|c| c.set(false));
    let heap = Heap {
        allocs: ALLOCS.load(Ordering::Relaxed),
        frees: FREES.load(Ordering::Relaxed),
        bytes: BYTES.load(Ordering::Relaxed),
    };
    (heap, r)
}

#[test]
fn rnnoise_chain_process_never_allocates() {
    let sr = oto_dsp::SAMPLE_RATE as usize;
    let mut chain = Chain::with_model(Model::RNNoise).unwrap();
    chain.prime();
    let mut audio = signal::noisy_speech(sr * 3, 1);
    audio.extend(std::iter::repeat_n(0.0, sr)); // digital silence: the skip path
    audio.extend(signal::noisy_speech(sr, 2));
    let blocks = [1usize, 64, 128, 333, 480, 512, 1024, 2048];

    let (heap, ()) = count(|| {
        let mut at = 0;
        for (i, &n) in blocks.iter().cycle().enumerate() {
            if at >= audio.len() {
                break;
            }
            let end = (at + n).min(audio.len());
            match i % 40 {
                10 => chain.set_enabled(false),
                20 => chain.set_enabled(true),
                25 => chain.set_max_attenuation_db(12.0),
                30 => chain.set_max_attenuation_db(f32::INFINITY),
                35 => chain.set_max_attenuation_db(24.0),
                _ => {}
            }
            chain.process(&mut audio[at..end]);
            at = end;
        }
    });
    assert_eq!(heap.calls(), 0, "Chain::process touched the heap: {heap:?}");
    assert!(chain.inference_frames() > 0);
}

/// Measurement, not a contract check: DeepFilterNet3 allocates on every frame (libDF clones its
/// spectrum tensors and tract allocates a fresh tensor for every node output on every run, about
/// 1,700 heap calls and 0.95 MB per 10 ms frame, ~95% inside tract). That is why the engine runs
/// it on a DSP worker thread instead of in the audio callback (design B of plan 0001); the
/// engine's `tests/no_alloc.rs` checks that the callback side of that stays allocation-free.
/// This prints the per-frame numbers so the gap stays visible. It is `#[ignore]` because it
/// asserts nothing; run it with
/// `cargo test -p oto-dsp --features dfn3 --test no_alloc -- --ignored --nocapture`
/// (add `--release` for release numbers, `OTO_ALLOC_TRACE=N` to see where they come from).
#[cfg(feature = "dfn3")]
#[test]
#[ignore = "measurement: DFN3 is known to allocate per frame"]
fn dfn3_allocations_per_frame() {
    const FRAME: usize = 480; // 10 ms, one DFN3 hop: one inference per `process` call
    const FRAMES: usize = 100;
    let mut chain = Chain::with_model(Model::DeepFilterNet3).unwrap();
    chain.prime();
    let audio = signal::noisy_speech(FRAME * (FRAMES + 1), 1);
    let (warm, rest) = audio.split_at(FRAME);
    chain.process(&mut warm.to_vec());
    let mut buf = rest.to_vec();
    let before = chain.inference_frames();

    let (heap, ()) = count(|| {
        for block in buf.as_chunks_mut::<FRAME>().0 {
            chain.process(block);
        }
    });
    let ran = (chain.inference_frames() - before) as usize;
    assert_eq!(ran, FRAMES, "every frame should have run the model");
    println!(
        "DeepFilterNet3: {:.1} allocs, {:.1} frees, {:.0} bytes allocated per 10 ms frame \
         ({FRAMES} frames: {heap:?})",
        heap.allocs as f64 / ran as f64,
        heap.frees as f64 / ran as f64,
        heap.bytes as f64 / ran as f64,
    );
}

#[test]
fn the_counter_works() {
    let (heap, v) = count(|| vec![1u8; 16]);
    assert!(heap.allocs >= 1 && heap.bytes >= 16);
    drop(v);
}
