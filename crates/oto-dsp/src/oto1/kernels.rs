//! The two hot kernels every Oto-1 layer reduces to, with a scalar reference and `core::arch`
//! SIMD per target (no `std::simd`, no auto-vectorisation trusted for parity):
//!
//! - [`gemv`]: `y = init + W x` on a [`Matrix`] packed in 16-output column panels. Every backend
//!   sums each output in the same order (bias first, then `i = 0, 1, …`), so the only difference
//!   between backends is fused multiply-add: NEON and AVX2 use FMA, scalar, SSE2 and wasm
//!   `simd128` do not (and are bit-identical to each other).
//! - [`activate`]: sigmoid / tanh / SiLU through one Cephes-style `exp` polynomial, evaluated with
//!   the same operation sequence on every backend (no FMA), so activations are bit-identical
//!   everywhere.

// Explicit index loops over accumulator registers read better than iterator chains here.
#![allow(clippy::needless_range_loop)]

/// Outputs per packed panel.
pub const PACK: usize = 16;
/// Most panels a kernel accumulates at once (16 NEON / 8 AVX2 accumulator registers).
const MAX_NP: usize = 4;

/// Which kernel implementation runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Portable scalar code: the reference, and the non-SIMD wasm build.
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128,
}

impl Backend {
    /// The fastest backend this CPU / build supports.
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Backend::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                Backend::Avx2Fma
            } else {
                Backend::Sse2
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            Backend::Simd128
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        {
            Backend::Scalar
        }
    }

    /// Every backend that can run here, scalar first (for parity tests and benchmarks).
    pub fn all() -> Vec<Self> {
        #[allow(unused_mut)]
        let mut v = vec![Backend::Scalar];
        #[cfg(target_arch = "aarch64")]
        v.push(Backend::Neon);
        #[cfg(target_arch = "x86_64")]
        {
            v.push(Backend::Sse2);
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                v.push(Backend::Avx2Fma);
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        v.push(Backend::Simd128);
        v
    }

    /// Whether this CPU / build can execute this backend's instructions.
    pub fn is_supported(self) -> bool {
        match self {
            Backend::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => true,
            #[cfg(target_arch = "x86_64")]
            Backend::Sse2 => true,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2Fma => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Backend::Simd128 => true,
        }
    }

    fn max_panels(self) -> usize {
        match self {
            Backend::Scalar => 1,
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => 4,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2Fma => 4,
            #[cfg(target_arch = "x86_64")]
            Backend::Sse2 => 2,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Backend::Simd128 => 2,
        }
    }
}

/// A dense `n_out × n_in` weight matrix, packed for [`gemv`]: panel `p` holds outputs
/// `16p..16p+16` as `n_in` rows of 16 contiguous floats (zero-padded past `n_out`).
#[derive(Clone, Debug)]
pub struct Matrix {
    n_out: usize,
    n_in: usize,
    panels: usize,
    data: Vec<f32>,
}

impl Matrix {
    /// From a row-major `[n_out][n_in]` matrix (PyTorch `Linear.weight` layout).
    pub fn from_rows(w: &[f32], n_out: usize, n_in: usize) -> Self {
        assert_eq!(w.len(), n_out * n_in);
        let panels = n_out.div_ceil(PACK);
        let mut data = vec![0.0; panels * n_in * PACK];
        for o in 0..n_out {
            let (p, j) = (o / PACK, o % PACK);
            for i in 0..n_in {
                data[(p * n_in + i) * PACK + j] = w[o * n_in + i];
            }
        }
        Self {
            n_out,
            n_in,
            panels,
            data,
        }
    }

    pub fn n_out(&self) -> usize {
        self.n_out
    }

    pub fn n_in(&self) -> usize {
        self.n_in
    }

    /// Element `(o, i)`, for tests.
    pub fn get(&self, o: usize, i: usize) -> f32 {
        self.data[((o / PACK) * self.n_in + i) * PACK + o % PACK]
    }

    pub fn macs(&self) -> usize {
        self.n_out * self.n_in
    }
}

/// `y = bias + W x`.
pub fn gemv(be: Backend, m: &Matrix, x: &[f32], bias: &[f32], y: &mut [f32]) {
    assert!(be.is_supported(), "CPU does not support backend {be:?}");
    assert!(x.len() >= m.n_in && bias.len() >= m.n_out && y.len() >= m.n_out);
    // SAFETY: lengths and CPU features checked above; `init` and `y` may alias in `run`, which
    // reads a panel's init values before it writes that panel.
    unsafe { run(be, m, x.as_ptr(), bias.as_ptr(), y.as_mut_ptr()) }
}

/// `y += W x`.
pub fn gemv_acc(be: Backend, m: &Matrix, x: &[f32], y: &mut [f32]) {
    assert!(be.is_supported(), "CPU does not support backend {be:?}");
    assert!(x.len() >= m.n_in && y.len() >= m.n_out);
    let p = y.as_mut_ptr();
    // SAFETY: as in `gemv`.
    unsafe { run(be, m, x.as_ptr(), p, p) }
}

unsafe fn run(be: Backend, m: &Matrix, x: *const f32, init: *const f32, y: *mut f32) {
    let max = be.max_panels();
    let mut p = 0;
    while p < m.panels {
        let np = (m.panels - p).min(max);
        let mut acc = [[0.0f32; PACK]; MAX_NP];
        for (q, a) in acc.iter_mut().enumerate().take(np) {
            for (j, v) in a.iter_mut().enumerate() {
                let o = (p + q) * PACK + j;
                if o < m.n_out {
                    *v = unsafe { *init.add(o) };
                }
            }
        }
        let w = unsafe { m.data.as_ptr().add(p * m.n_in * PACK) };
        unsafe { panels(be, np, w, m.n_in, x, &mut acc) };
        for (q, a) in acc.iter().enumerate().take(np) {
            for (j, &v) in a.iter().enumerate() {
                let o = (p + q) * PACK + j;
                if o < m.n_out {
                    unsafe { *y.add(o) = v };
                }
            }
        }
        p += np;
    }
}

type Acc = [[f32; PACK]; MAX_NP];

unsafe fn panels(be: Backend, np: usize, w: *const f32, n_in: usize, x: *const f32, acc: &mut Acc) {
    macro_rules! by_np {
        ($f:ident) => {
            match np {
                1 => $f::<1>(w, n_in, x, acc),
                2 => $f::<2>(w, n_in, x, acc),
                3 => $f::<3>(w, n_in, x, acc),
                _ => $f::<4>(w, n_in, x, acc),
            }
        };
    }
    unsafe {
        match be {
            Backend::Scalar => by_np!(scalar_panels),
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => by_np!(neon_panels),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2Fma => by_np!(avx2_panels),
            #[cfg(target_arch = "x86_64")]
            Backend::Sse2 => by_np!(sse2_panels),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Backend::Simd128 => by_np!(simd128_panels),
        }
    }
}

unsafe fn scalar_panels<const NP: usize>(w: *const f32, n_in: usize, x: *const f32, acc: &mut Acc) {
    for i in 0..n_in {
        let xi = unsafe { *x.add(i) };
        for (p, a) in acc.iter_mut().enumerate().take(NP) {
            let wp = unsafe { w.add((p * n_in + i) * PACK) };
            for (j, v) in a.iter_mut().enumerate() {
                *v += xi * unsafe { *wp.add(j) };
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_panels<const NP: usize>(w: *const f32, n_in: usize, x: *const f32, acc: &mut Acc) {
    use core::arch::aarch64::*;
    unsafe {
        let mut a = [[vdupq_n_f32(0.0); 4]; NP];
        for p in 0..NP {
            for k in 0..4 {
                a[p][k] = vld1q_f32(acc[p].as_ptr().add(4 * k));
            }
        }
        for i in 0..n_in {
            let xv = vdupq_n_f32(*x.add(i));
            for p in 0..NP {
                let wp = w.add((p * n_in + i) * PACK);
                for k in 0..4 {
                    a[p][k] = vfmaq_f32(a[p][k], vld1q_f32(wp.add(4 * k)), xv);
                }
            }
        }
        for p in 0..NP {
            for k in 0..4 {
                vst1q_f32(acc[p].as_mut_ptr().add(4 * k), a[p][k]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_panels<const NP: usize>(w: *const f32, n_in: usize, x: *const f32, acc: &mut Acc) {
    use core::arch::x86_64::*;
    unsafe {
        let mut a = [[_mm256_setzero_ps(); 2]; NP];
        for p in 0..NP {
            for k in 0..2 {
                a[p][k] = _mm256_loadu_ps(acc[p].as_ptr().add(8 * k));
            }
        }
        for i in 0..n_in {
            let xv = _mm256_set1_ps(*x.add(i));
            for p in 0..NP {
                let wp = w.add((p * n_in + i) * PACK);
                for k in 0..2 {
                    a[p][k] = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(8 * k)), xv, a[p][k]);
                }
            }
        }
        for p in 0..NP {
            for k in 0..2 {
                _mm256_storeu_ps(acc[p].as_mut_ptr().add(8 * k), a[p][k]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_panels<const NP: usize>(w: *const f32, n_in: usize, x: *const f32, acc: &mut Acc) {
    use core::arch::x86_64::*;
    unsafe {
        let mut a = [[_mm_setzero_ps(); 4]; NP];
        for p in 0..NP {
            for k in 0..4 {
                a[p][k] = _mm_loadu_ps(acc[p].as_ptr().add(4 * k));
            }
        }
        for i in 0..n_in {
            let xv = _mm_set1_ps(*x.add(i));
            for p in 0..NP {
                let wp = w.add((p * n_in + i) * PACK);
                for k in 0..4 {
                    a[p][k] = _mm_add_ps(a[p][k], _mm_mul_ps(_mm_loadu_ps(wp.add(4 * k)), xv));
                }
            }
        }
        for p in 0..NP {
            for k in 0..4 {
                _mm_storeu_ps(acc[p].as_mut_ptr().add(4 * k), a[p][k]);
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn simd128_panels<const NP: usize>(
    w: *const f32,
    n_in: usize,
    x: *const f32,
    acc: &mut Acc,
) {
    use core::arch::wasm32::*;
    unsafe {
        let mut a = [[f32x4_splat(0.0); 4]; NP];
        for p in 0..NP {
            for k in 0..4 {
                a[p][k] = v128_load(acc[p].as_ptr().add(4 * k) as *const v128);
            }
        }
        for i in 0..n_in {
            let xv = f32x4_splat(*x.add(i));
            for p in 0..NP {
                let wp = w.add((p * n_in + i) * PACK);
                for k in 0..4 {
                    let wv = v128_load(wp.add(4 * k) as *const v128);
                    a[p][k] = f32x4_add(a[p][k], f32x4_mul(wv, xv));
                }
            }
        }
        for p in 0..NP {
            for k in 0..4 {
                v128_store(acc[p].as_mut_ptr().add(4 * k) as *mut v128, a[p][k]);
            }
        }
    }
}

/// Elementwise nonlinearity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Sigmoid,
    Tanh,
    /// `x * sigmoid(x)`, evaluated as `x / (1 + exp(-x))`.
    Silu,
}

// Cephes `expf` constants.
// Clamped so neither `exp` nor `1 / (1 + exp(x))` ever produces a subnormal (slow on x86, and
// what the GRU flush exists to avoid); sigmoid(-80) = 1.8e-35 is still a normal float.
const EXP_HI: f32 = 80.0;
const EXP_LO: f32 = -87.0;
const LOG2E: f32 = core::f32::consts::LOG2_E;
const C1: f32 = 0.693_359_4;
const C2: f32 = -2.121_944_4e-4;
const P0: f32 = 1.987_569_1e-4;
const P1: f32 = 1.398_199_9e-3;
const P2: f32 = 8.333_452e-3;
const P3: f32 = 4.166_579_6e-2;
const P4: f32 = 1.666_666_5e-1;
const P5: f32 = 0.5; // Cephes 5.0000001201e-1, which rounds to exactly 0.5 in f32

/// `exp(x)` to ~1 ulp on `[-87, 80]`, clamped outside. The reference every SIMD path matches
/// bit for bit (same operations, same order, no FMA).
#[inline]
pub fn exp_ref(x: f32) -> f32 {
    let x = x.clamp(EXP_LO, EXP_HI);
    let fx = x * LOG2E + 0.5;
    let t = fx as i32;
    let n = if (t as f32) > fx { t - 1 } else { t };
    let nf = n as f32;
    let r = x - nf * C1;
    let r = r - nf * C2;
    let z = r * r;
    let mut y = P0;
    y = y * r + P1;
    y = y * r + P2;
    y = y * r + P3;
    y = y * r + P4;
    y = y * r + P5;
    y *= z;
    y += r;
    y += 1.0;
    y * f32::from_bits(((n + 127) as u32) << 23)
}

#[inline]
fn act_ref(act: Act, x: f32) -> f32 {
    match act {
        Act::Sigmoid => 1.0 / (1.0 + exp_ref(-x)),
        Act::Tanh => 2.0 / (1.0 + exp_ref(-2.0 * x)) - 1.0,
        Act::Silu => x / (1.0 + exp_ref(-x)),
    }
}

/// Apply `act` in place.
pub fn activate(be: Backend, act: Act, v: &mut [f32]) {
    assert!(be.is_supported(), "CPU does not support backend {be:?}");
    let done = match be {
        Backend::Scalar => 0,
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { neon_act(act, v) },
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2Fma | Backend::Sse2 => unsafe { sse2_act(act, v) },
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Backend::Simd128 => unsafe { simd128_act(act, v) },
    };
    for x in &mut v[done..] {
        *x = act_ref(act, *x);
    }
}

/// Writes the 4-lane SIMD version of `act_ref` for one target, from the target's primitive ops.
/// Returns how many leading elements it processed (a multiple of 4).
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! simd_act {
    ($name:ident, $feat:literal, $arch:path, $ty:ty,
     splat = $splat:expr, load = $load:expr, store = $store:expr,
     add = $add:expr, sub = $sub:expr, mul = $mul:expr, div = $div:expr,
     min = $min:expr, max = $max:expr, floor_i = $floor_i:expr, i2f = $i2f:expr,
     pow2 = $pow2:expr) => {
        #[target_feature(enable = $feat)]
        unsafe fn $name(act: Act, v: &mut [f32]) -> usize {
            use $arch::*;
            let n = v.len() / 4 * 4;
            unsafe {
                let one: $ty = $splat(1.0);
                let two: $ty = $splat(2.0);
                let exp = |x: $ty| -> $ty {
                    let x = $min($max(x, $splat(EXP_LO)), $splat(EXP_HI));
                    let fx = $add($mul(x, $splat(LOG2E)), $splat(0.5));
                    let ni = $floor_i(fx);
                    let nf = $i2f(ni);
                    let r = $sub(x, $mul(nf, $splat(C1)));
                    let r = $sub(r, $mul(nf, $splat(C2)));
                    let z = $mul(r, r);
                    let mut y = $splat(P0);
                    y = $add($mul(y, r), $splat(P1));
                    y = $add($mul(y, r), $splat(P2));
                    y = $add($mul(y, r), $splat(P3));
                    y = $add($mul(y, r), $splat(P4));
                    y = $add($mul(y, r), $splat(P5));
                    y = $mul(y, z);
                    y = $add(y, r);
                    y = $add(y, one);
                    $mul(y, $pow2(ni))
                };
                let neg = |x: $ty| $sub($splat(0.0), x);
                let p = v.as_mut_ptr();
                let mut i = 0;
                while i < n {
                    let x: $ty = $load(p.add(i));
                    let y = match act {
                        Act::Sigmoid => $div(one, $add(one, exp(neg(x)))),
                        Act::Tanh => $sub($div(two, $add(one, exp(neg($mul(two, x))))), one),
                        Act::Silu => $div(x, $add(one, exp(neg(x)))),
                    };
                    $store(p.add(i), y);
                    i += 4;
                }
            }
            n
        }
    };
}

#[cfg(target_arch = "aarch64")]
simd_act!(
    neon_act,
    "neon",
    core::arch::aarch64,
    float32x4_t,
    splat = |c: f32| vdupq_n_f32(c),
    load = |p: *const f32| vld1q_f32(p),
    store = |p: *mut f32, v| vst1q_f32(p, v),
    add = |a, b| vaddq_f32(a, b),
    sub = |a, b| vsubq_f32(a, b),
    mul = |a, b| vmulq_f32(a, b),
    div = |a, b| vdivq_f32(a, b),
    min = |a, b| vminq_f32(a, b),
    max = |a, b| vmaxq_f32(a, b),
    floor_i = |x| vcvtmq_s32_f32(x),
    i2f = |n| vcvtq_f32_s32(n),
    pow2 = |n| vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n, vdupq_n_s32(127))))
);

#[cfg(target_arch = "x86_64")]
simd_act!(
    sse2_act,
    "sse2",
    core::arch::x86_64,
    __m128,
    splat = |c: f32| _mm_set1_ps(c),
    load = |p: *const f32| _mm_loadu_ps(p),
    store = |p: *mut f32, v| _mm_storeu_ps(p, v),
    add = |a, b| _mm_add_ps(a, b),
    sub = |a, b| _mm_sub_ps(a, b),
    mul = |a, b| _mm_mul_ps(a, b),
    div = |a, b| _mm_div_ps(a, b),
    min = |a, b| _mm_min_ps(a, b),
    max = |a, b| _mm_max_ps(a, b),
    floor_i = |x| {
        // truncate, then step down where truncation rounded up (negative non-integers)
        let t = _mm_cvttps_epi32(x);
        let up = _mm_castps_si128(_mm_cmpgt_ps(_mm_cvtepi32_ps(t), x));
        _mm_add_epi32(t, up)
    },
    i2f = |n| _mm_cvtepi32_ps(n),
    pow2 = |n| _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(n, _mm_set1_epi32(127))))
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
simd_act!(
    simd128_act,
    "simd128",
    core::arch::wasm32,
    v128,
    splat = |c: f32| f32x4_splat(c),
    load = |p: *const f32| v128_load(p as *const v128),
    store = |p: *mut f32, v| v128_store(p as *mut v128, v),
    add = |a, b| f32x4_add(a, b),
    sub = |a, b| f32x4_sub(a, b),
    mul = |a, b| f32x4_mul(a, b),
    div = |a, b| f32x4_div(a, b),
    // pmin/pmax are the cheap x86-friendly forms; identical to min/max for non-NaN input.
    min = |a, b| f32x4_pmin(a, b),
    max = |a, b| f32x4_pmax(a, b),
    floor_i = |x| i32x4_trunc_sat_f32x4(f32x4_floor(x)),
    i2f = |n| f32x4_convert_i32x4(n),
    pow2 = |n| i32x4_shl(i32x4_add(n, i32x4_splat(127)), 23)
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oto1::format::Rng;

    fn rand_vec(r: &mut Rng, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| r.uniform() * s).collect()
    }

    /// Relative RMS error in dB.
    pub(crate) fn err_db(a: &[f32], b: &[f32]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
        if num == 0.0 {
            f64::NEG_INFINITY
        } else {
            10.0 * (num / den.max(1e-30)).log10()
        }
    }

    #[test]
    fn packing_round_trips() {
        let mut r = Rng::new(1);
        let (o, i) = (37, 11);
        let w = rand_vec(&mut r, o * i, 1.0);
        let m = Matrix::from_rows(&w, o, i);
        for a in 0..o {
            for b in 0..i {
                assert_eq!(m.get(a, b), w[a * i + b]);
            }
        }
    }

    #[test]
    fn gemv_backends_match_naive_reference() {
        let mut r = Rng::new(2);
        for &(o, i) in &[
            (1, 1),
            (6, 48),
            (16, 3),
            (17, 5),
            (48, 144),
            (100, 70),
            (384, 512),
            (576, 384),
        ] {
            let w = rand_vec(&mut r, o * i, 1.0 / (i as f32).sqrt());
            let x = rand_vec(&mut r, i, 1.0);
            let b = rand_vec(&mut r, o, 0.1);
            let m = Matrix::from_rows(&w, o, i);
            // Naive row-major reference, same summation order (bias, then i ascending).
            let want: Vec<f32> = (0..o)
                .map(|a| (0..i).fold(b[a], |s, k| s + w[a * i + k] * x[k]))
                .collect();
            for be in Backend::all() {
                let mut y = vec![f32::NAN; o];
                gemv(be, &m, &x, &b, &mut y);
                let db = err_db(&y, &want);
                assert!(db <= -90.0, "{be:?} {o}x{i}: {db:.1} dB");
                // Only the FMA backends may differ from the reference at all.
                if !matches!(format!("{be:?}").as_str(), "Neon" | "Avx2Fma") {
                    assert_eq!(y, want, "{be:?} {o}x{i} should be bit-identical");
                }
                // Accumulate form: y += W x on top of the bias.
                let mut y2 = b.clone();
                gemv_acc(be, &m, &x, &mut y2);
                assert_eq!(y2, y, "{be:?}: gemv_acc != gemv");
            }
        }
    }

    #[test]
    fn exp_is_accurate() {
        let mut worst = 0.0f64;
        let mut x = -86.9f32;
        while x < 80.0 {
            let rel = ((exp_ref(x) as f64 - (x as f64).exp()) / (x as f64).exp()).abs();
            worst = worst.max(rel);
            x += 0.0137;
        }
        assert!(worst < 3e-7, "worst rel err {worst:e}");
        assert_eq!(exp_ref(-1000.0), exp_ref(EXP_LO));
        assert!(exp_ref(1000.0).is_finite());
    }

    #[test]
    fn activations_are_bit_identical_across_backends_and_close_to_libm() {
        let mut r = Rng::new(3);
        let mut x = rand_vec(&mut r, 1003, 12.0);
        x.extend_from_slice(&[0.0, -0.0, 1e-20, -1e-20, 90.0, -90.0, 1e6, -1e6]);
        for act in [Act::Sigmoid, Act::Tanh, Act::Silu] {
            let mut want = x.clone();
            activate(Backend::Scalar, act, &mut want);
            for (&xi, &yi) in x.iter().zip(&want) {
                let exact = match act {
                    Act::Sigmoid => 1.0 / (1.0 + (-xi as f64).exp()),
                    Act::Tanh => (xi as f64).tanh(),
                    Act::Silu => xi as f64 / (1.0 + (-xi as f64).exp()),
                };
                let tol = 4e-7 * exact.abs().max(1.0);
                assert!(
                    (yi as f64 - exact).abs() <= tol,
                    "{act:?}({xi}) = {yi}, want {exact}"
                );
            }
            for be in Backend::all() {
                let mut y = x.clone();
                activate(be, act, &mut y);
                assert_eq!(
                    y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{be:?} {act:?}"
                );
            }
        }
    }
}
