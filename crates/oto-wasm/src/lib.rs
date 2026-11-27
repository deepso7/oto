//! Oto-1 for a browser `AudioWorkletProcessor`.
//!
//! A plain `cdylib` for `wasm32-unknown-unknown`: no `wasm-bindgen` (its glue needs
//! `TextDecoder`, which `AudioWorkletGlobalScope` lacks), no imports at all, `panic = "abort"`
//! (from the workspace release profile), and a flat `extern "C"` ABI over one static model:
//!
//! 1. `oto_alloc(len)` → copy the `.otow` bytes there → `oto_init(ptr, len)` → `oto_free(ptr, len)`.
//!    Or `oto_init_random(arch_id, seed)` for the benchmark models. Non-zero return = error;
//!    read `oto_error_ptr()` / `oto_error_len()` (UTF-8).
//! 2. Make `Float32Array` views on `oto_input_ptr()` / `oto_output_ptr()` (512 floats each)
//!    *after* init: init grows memory, processing never does.
//! 3. Per hop: write 512 samples to the input, `oto_process_hop()`, read 512 samples of output.
//!
//! The model itself delays by 512 samples; buffering 128-sample quanta into 512-sample hops adds
//! another 512 (1024 total, 21.33 ms), the same as `FrameAdapter` natively.
//!
//! Build: `crates/oto-wasm/build-wasm.sh` (SIMD and scalar binaries into `web/demo/wasm/`).

use oto_dsp::Denoiser;
use oto_dsp::oto1::{Arch, HOP, Oto1, random_otow};

/// Self-contained state backing the single-threaded WASM C ABI (and isolated native tests).
pub struct AbiState {
    model: Option<Oto1>,
    input: [f32; HOP],
    output: [f32; HOP],
    error: [u8; 256],
    error_len: usize,
}

impl AbiState {
    pub const fn new() -> Self {
        Self {
            model: None,
            input: [0.0; HOP],
            output: [0.0; HOP],
            error: [0; 256],
            error_len: 0,
        }
    }

    fn set_error(&mut self, msg: &str) {
        let n = msg.len().min(self.error.len());
        self.error[..n].copy_from_slice(&msg.as_bytes()[..n]);
        self.error_len = n;
    }

    fn install(&mut self, r: Result<Oto1, String>) -> i32 {
        match r {
            Ok(m) => {
                self.model = Some(m);
                self.input.fill(0.0);
                self.output.fill(0.0);
                self.error_len = 0;
                0
            }
            Err(e) => {
                self.model = None;
                self.set_error(&e);
                1
            }
        }
    }

    pub fn init(&mut self, bytes: &[u8]) -> i32 {
        self.install(Oto1::from_otow(bytes))
    }

    pub fn init_random(&mut self, arch_id: u32, seed: u32) -> i32 {
        match Arch::from_id(arch_id) {
            Some(a) => self.install(Oto1::from_otow(&random_otow(a, seed as u64))),
            None => self.install(Err(format!("unknown arch_id {arch_id}"))),
        }
    }

    pub fn input_ptr(&mut self) -> *mut f32 {
        self.input.as_mut_ptr()
    }

    pub fn output_ptr(&mut self) -> *mut f32 {
        self.output.as_mut_ptr()
    }

    pub fn process_hop(&mut self) -> i32 {
        match &mut self.model {
            Some(m) => {
                oto_dsp::sanitize(&mut self.input);
                m.process_frame(&self.input, &mut self.output);
                0
            }
            None => {
                self.output.fill(0.0);
                1
            }
        }
    }

    pub fn reset(&mut self) {
        if let Some(m) = &mut self.model {
            m.reset();
        }
        self.output.fill(0.0);
    }

    pub fn unload(&mut self) {
        self.model = None;
    }

    pub fn latency(&self) -> u32 {
        self.model.as_ref().map_or(0, |m| m.latency() as u32)
    }

    pub fn p_target(&self) -> f32 {
        self.model.as_ref().map_or(0.0, |m| m.p_target())
    }

    pub fn params(&self) -> u32 {
        self.model.as_ref().map_or(0, |m| m.params() as u32)
    }

    pub fn macs_per_hop(&self) -> u32 {
        self.model.as_ref().map_or(0, |m| m.macs_per_frame() as u32)
    }

    pub fn is_isolating(&self) -> u32 {
        self.model.as_ref().is_some_and(|m| m.is_isolating()) as u32
    }

    pub fn error_bytes(&self) -> &[u8] {
        &self.error[..self.error_len]
    }
}

impl Default for AbiState {
    fn default() -> Self {
        Self::new()
    }
}

/// Unsynchronized single-threaded global ABI, compiled only on non-atomic `wasm32`.
/// On native targets `AbiState` is used directly without global mutable aliasing.
#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
mod wasm_abi {
    use super::*;
    use core::cell::UnsafeCell;
    use oto_dsp::oto1::Backend;

    struct Global(UnsafeCell<AbiState>);
    unsafe impl Sync for Global {}

    static STATE: Global = Global(UnsafeCell::new(AbiState::new()));

    #[allow(clippy::mut_from_ref)]
    fn state() -> &'static mut AbiState {
        // SAFETY: non-atomic wasm32 is single-threaded, and no export re-enters another.
        unsafe { &mut *STATE.0.get() }
    }

    /// Allocate `len` bytes (for the weights upload). Free with [`oto_free`].
    #[unsafe(no_mangle)]
    pub extern "C" fn oto_alloc(len: usize) -> *mut u8 {
        let mut v = Vec::<u8>::with_capacity(len.max(1));
        let p = v.as_mut_ptr();
        core::mem::forget(v);
        p
    }

    /// Free a block from [`oto_alloc`].
    ///
    /// # Safety
    /// `ptr` and `len` must come from one `oto_alloc` call, freed once.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn oto_free(ptr: *mut u8, len: usize) {
        drop(unsafe { Vec::from_raw_parts(ptr, 0, len.max(1)) });
    }

    /// Load a `.otow` file from linear memory. 0 = ok.
    ///
    /// # Safety
    /// `ptr..ptr+len` must be readable linear memory.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn oto_init(ptr: *const u8, len: usize) -> i32 {
        let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
        state().init(bytes)
    }

    /// Build a seeded random-weight model (`arch_id` 1 = Candidate A, 2 = B). 0 = ok.
    #[unsafe(no_mangle)]
    pub extern "C" fn oto_init_random(arch_id: u32, seed: u32) -> i32 {
        state().init_random(arch_id, seed)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn oto_input_ptr() -> *mut f32 {
        state().input_ptr()
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn oto_output_ptr() -> *mut f32 {
        state().output_ptr()
    }

    /// Run one 512-sample hop from the input buffer into the output buffer. 0 = ok; 1 = no model
    /// loaded (output is zeroed; callers bypass with their own delay line).
    #[unsafe(no_mangle)]
    pub extern "C" fn oto_process_hop() -> i32 {
        state().process_hop()
    }

    /// Clear the model's streaming state (no allocation).
    #[unsafe(no_mangle)]
    pub extern "C" fn oto_reset() {
        state().reset();
    }

    /// Drop the model and its memory (linear memory itself never shrinks).
    #[unsafe(no_mangle)]
    pub extern "C" fn oto_unload() {
        state().unload();
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn oto_hop_size() -> u32 {
        HOP as u32
    }

    /// The model's own delay in samples (the 128 → 512 ring adds another hop).
    #[unsafe(no_mangle)]
    pub extern "C" fn oto_latency() -> u32 {
        state().latency()
    }

    /// Target-presence logit of the last hop.
    #[unsafe(no_mangle)]
    pub extern "C" fn oto_p_target() -> f32 {
        state().p_target()
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn oto_params() -> u32 {
        state().params()
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn oto_macs_per_hop() -> u32 {
        state().macs_per_hop()
    }

    /// 1 if this binary was built with `simd128` kernels, else 0.
    #[unsafe(no_mangle)]
    pub extern "C" fn oto_simd() -> u32 {
        (Backend::detect() != Backend::Scalar) as u32
    }

    /// 1 if the loaded model is an Isolate model (run it fail-closed).
    #[unsafe(no_mangle)]
    pub extern "C" fn oto_is_isolating() -> u32 {
        state().is_isolating()
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn oto_error_ptr() -> *const u8 {
        state().error.as_ptr()
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn oto_error_len() -> u32 {
        state().error_len as u32
    }
}

#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
pub use wasm_abi::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_round_trip() {
        let mut st = AbiState::new();
        assert_eq!(st.process_hop(), 1);
        assert_ne!(st.init_random(9, 1), 0);
        assert!(
            core::str::from_utf8(st.error_bytes())
                .unwrap()
                .contains("arch_id")
        );

        let bytes = random_otow(Arch::B, 4);
        assert_eq!(st.init(&bytes), 0);
        assert_eq!(st.latency(), 512);
        let (inp, out) = (st.input_ptr(), st.output_ptr());
        let mut energy = 0.0;
        for h in 0..20 {
            for i in 0..HOP {
                unsafe { *inp.add(i) = ((h * HOP + i) as f32 * 0.05).sin() * 0.3 };
            }
            assert_eq!(st.process_hop(), 0);
            energy += (0..HOP)
                .map(|i| unsafe { *out.add(i) }.powi(2))
                .sum::<f32>();
        }
        assert!(energy > 0.0 && energy.is_finite());
        assert_eq!(st.init_random(1, 7), 0);
        assert!(st.params() > 500_000);
        assert!(oto_dsp::oto1::Backend::detect().is_supported());
    }
}
