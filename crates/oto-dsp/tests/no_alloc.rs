//! Proves `Chain::process` and its realtime-safe setters never touch the heap (RNNoise build).
//!
//! A counting global allocator that only counts on threads that opted in, so the test harness's
//! own allocations on other threads do not interfere.

use oto_dsp::{Chain, Model, signal};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

fn note() {
    // Set OTO_ALLOC_TRACE=1 to print where the first counted allocation came from.
    if COUNTING.with(Cell::get)
        && ALLOCS.fetch_add(1, Ordering::Relaxed) == 0
        && std::env::var_os("OTO_ALLOC_TRACE").is_some()
    {
        COUNTING.with(|c| c.set(false));
        eprintln!("{}", std::backtrace::Backtrace::force_capture());
        COUNTING.with(|c| c.set(true));
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note();
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn count<R>(f: impl FnOnce() -> R) -> (usize, R) {
    ALLOCS.store(0, Ordering::Relaxed);
    COUNTING.with(|c| c.set(true));
    let r = f();
    COUNTING.with(|c| c.set(false));
    (ALLOCS.load(Ordering::Relaxed), r)
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

    let (allocs, ()) = count(|| {
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
    assert_eq!(allocs, 0, "Chain::process allocated {allocs} times");
    assert!(chain.inference_frames() > 0);
}

#[test]
fn the_counter_works() {
    let (allocs, v) = count(|| vec![1u8; 16]);
    assert!(allocs >= 1);
    drop(v);
}
