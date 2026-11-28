//! The two hot kernels every Oto-1 layer reduces to, with a scalar reference and `core::arch`
//! SIMD per target (no `std::simd`, no auto-vectorisation trusted for parity):
//!
//! - [`gemv`]: `y = init + W x` on a [`Matrix`] packed in 16-output column panels. Every backend
//!   sums each output in the same order (bias first, then `i = 0, 1, …`), so the only difference
//!   between backends is fused multiply-add: NEON and AVX2 use FMA, scalar, SSE2 and wasm
//!   `simd128` do not (and are bit-identical to each other). [`gemv_rows`] runs one matrix over
//!   several input rows (a convolution's interior), sharing each weight load between rows; each
//!   row's result is exactly [`gemv`]'s. Narrow matrices (`n_out <= 8`) skip the zero padding.
//! - [`activate`]: sigmoid / tanh / SiLU through one Cephes-style `exp` polynomial, evaluated with
//!   the same operation sequence on every backend (no FMA), so activations are bit-identical
//!   everywhere.

// Explicit index loops over accumulator registers read better than iterator chains here.
#![allow(clippy::needless_range_loop)]

/// Outputs per packed panel.
pub const PACK: usize = 16;
/// Most panels a kernel accumulates at once (see [`Backend::tile`]).
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

    /// How [`gemv_rows`] tiles a matrix with `n_out` outputs over `rows` inputs: `(l, r, np)`,
    /// computing `4 l` lanes of each 16-output panel (`l < 4` only when `n_out <= 8`, skipping
    /// the zero padding), up to `r` input rows sharing each weight load, `np` panels at a time.
    ///
    /// Accumulators live in registers: `r · np · (4 l / lanes)` of them, plus `r` broadcasts of
    /// `x` and one weight vector. x86_64 (and so V8 / SpiderMonkey wasm on x86_64) has 16 vector
    /// registers, so SSE2 and `simd128` stay at 12 accumulators or fewer; NEON has 32.
    // Only the SIMD backends fuse rows; scalar-only builds (e.g. plain wasm32) have no other.
    #[cfg_attr(
        not(any(
            target_arch = "aarch64",
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        allow(unused_variables)
    )]
    fn tile(self, n_out: usize, rows: usize) -> (usize, usize, usize) {
        let l = match n_out {
            0..=4 => 1,
            5..=8 => 2,
            _ => 4,
        };
        match self {
            Backend::Scalar => (l, 1, 1),
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => match (l, rows > 1) {
                (4, false) => (4, 1, 4),
                (4, true) => (4, 2, 2),
                _ => (l, 4, 1),
            },
            // 8 lanes per register, so a narrow panel is at least one full register.
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2Fma => match (l, rows > 1) {
                (4, false) => (4, 1, 4),
                (4, true) => (4, 2, 2),
                _ => (2, 4, 1),
            },
            #[cfg(target_arch = "x86_64")]
            Backend::Sse2 => match (l, rows > 1) {
                (4, false) => (4, 1, 3),
                (4, true) => (4, 2, 1),
                _ => (l, 4, 1),
            },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Backend::Simd128 => match (l, rows > 1) {
                (4, false) => (4, 1, 3),
                (4, true) => (4, 2, 1),
                _ => (l, 4, 1),
            },
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
    // SAFETY: lengths and CPU features checked above.
    unsafe { run(be, m, 1, x.as_ptr(), 0, bias.as_ptr(), y.as_mut_ptr(), 0) }
}

/// `y += W x`.
pub fn gemv_acc(be: Backend, m: &Matrix, x: &[f32], y: &mut [f32]) {
    assert!(be.is_supported(), "CPU does not support backend {be:?}");
    assert!(x.len() >= m.n_in && y.len() >= m.n_out);
    let p = y.as_mut_ptr();
    // SAFETY: as in `gemv`; `init` and `y` alias, which `run` allows.
    unsafe { run(be, m, 1, x.as_ptr(), 0, p, p, 0) }
}

/// `rows` GEMVs against one matrix: `y[r·y_step..][..n_out] = bias + W x[r·x_step..][..n_in]`
/// for `r = 0..rows` (input rows may overlap, as a convolution's do). Each output is exactly
/// what [`gemv`] computes for that row; sharing the weight loads across rows is what is faster.
#[allow(clippy::too_many_arguments)]
pub fn gemv_rows(
    be: Backend,
    m: &Matrix,
    x: &[f32],
    x_step: usize,
    rows: usize,
    bias: &[f32],
    y: &mut [f32],
    y_step: usize,
) {
    assert!(be.is_supported(), "CPU does not support backend {be:?}");
    if rows == 0 {
        return;
    }
    assert!(rows == 1 || y_step >= m.n_out, "output rows overlap");
    assert!(bias.len() >= m.n_out);
    let x_need = (rows - 1)
        .checked_mul(x_step)
        .and_then(|v| v.checked_add(m.n_in))
        .expect("gemv_rows input span overflow");
    let y_need = (rows - 1)
        .checked_mul(y_step)
        .and_then(|v| v.checked_add(m.n_out))
        .expect("gemv_rows output span overflow");
    assert!(x.len() >= x_need);
    assert!(y.len() >= y_need);
    // SAFETY: lengths and CPU features checked above; output rows are disjoint.
    unsafe {
        run(
            be,
            m,
            rows,
            x.as_ptr(),
            x_step,
            bias.as_ptr(),
            y.as_mut_ptr(),
            y_step,
        )
    }
}

/// Most input rows a kernel accumulates at once.
const MAX_R: usize = 4;
type Ptrs<T> = [T; MAX_R];

/// Row `r` reads `x + r·xs` and writes `y + r·ys`; every row starts from `init` (shared), which
/// may alias `y` only when `rows == 1` (a kernel reads all of its `init` before writing `y`).
#[allow(clippy::too_many_arguments)]
unsafe fn run(
    be: Backend,
    m: &Matrix,
    rows: usize,
    x: *const f32,
    xs: usize,
    init: *const f32,
    y: *mut f32,
    ys: usize,
) {
    let (l, rmax, npmax) = be.tile(m.n_out, rows);
    let mut p = 0;
    while p < m.panels {
        let np = (m.panels - p).min(npmax);
        let w = unsafe { m.data.as_ptr().add(p * m.n_in * PACK) };
        let o0 = p * PACK;
        // Lanes the kernel touches: panels p..p+np, the last one only up to 4 l.
        let end = o0 + (np - 1) * PACK + 4 * l;
        let mut r = 0;
        while r < rows {
            let rr = [4, 2, 1]
                .into_iter()
                .find(|&n| n <= rmax && n <= rows - r)
                .unwrap();
            let xp: Ptrs<*const f32> = core::array::from_fn(|q| {
                if q < rr {
                    unsafe { x.add((r + q) * xs) }
                } else {
                    x
                }
            });
            if end <= m.n_out {
                // Whole tile inside the output: the kernel reads init and writes y directly.
                let ip: Ptrs<*const f32> = [unsafe { init.add(o0) }; MAX_R];
                let yp: Ptrs<*mut f32> = core::array::from_fn(|q| {
                    if q < rr {
                        unsafe { y.add((r + q) * ys + o0) }
                    } else {
                        y
                    }
                });
                unsafe { kernel(be, rr, l, np, w, m.n_in, &xp, &ip, &yp) };
            } else {
                // The tile overhangs n_out: go through a zero-padded copy.
                let mut tmp = [[0.0f32; PACK * MAX_NP]; MAX_R];
                let valid = m.n_out - o0;
                for t in tmp.iter_mut().take(rr) {
                    for (j, v) in t.iter_mut().enumerate().take(valid) {
                        *v = unsafe { *init.add(o0 + j) };
                    }
                }
                // Derive ip from the same raw mutable pointer as yp (no overlapping &/&mut).
                let base = tmp.as_mut_ptr() as *mut f32;
                let stride = PACK * MAX_NP;
                let yp: Ptrs<*mut f32> = core::array::from_fn(|q| unsafe { base.add(q * stride) });
                let ip: Ptrs<*const f32> = core::array::from_fn(|q| yp[q] as *const f32);
                unsafe { kernel(be, rr, l, np, w, m.n_in, &xp, &ip, &yp) };
                for (q, t) in tmp.iter().enumerate().take(rr) {
                    for (j, &v) in t.iter().enumerate().take(valid) {
                        unsafe { *y.add((r + q) * ys + o0 + j) = v };
                    }
                }
            }
            r += rr;
        }
        p += np;
    }
}

/// Runs the `(rows, l, panels)` kernel instance of `be`, for each tile [`Backend::tile`] picks.
#[allow(clippy::too_many_arguments)]
unsafe fn kernel(
    be: Backend,
    r: usize,
    l: usize,
    np: usize,
    w: *const f32,
    n_in: usize,
    x: &Ptrs<*const f32>,
    init: &Ptrs<*const f32>,
    y: &Ptrs<*mut f32>,
) {
    macro_rules! arms {
        ($f:ident: $(($r:literal, $l:literal, $np:literal)),*) => {
            match (r, l, np) {
                $(($r, $l, $np) => $f::<$np, $r, $l>(w, n_in, x, init, y),)*
                _ => unreachable!("no {} kernel for rows {r}, l {l}, panels {np}", stringify!($f)),
            }
        };
    }
    unsafe {
        match be {
            Backend::Scalar => arms!(scalar_tile: (1, 1, 1), (1, 2, 1), (1, 4, 1)),
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => arms!(neon_tile:
                (1, 4, 1), (1, 4, 2), (1, 4, 3), (1, 4, 4), (2, 4, 1), (2, 4, 2),
                (1, 2, 1), (2, 2, 1), (4, 2, 1), (1, 1, 1), (2, 1, 1), (4, 1, 1)),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2Fma => arms!(avx2_tile:
                (1, 4, 1), (1, 4, 2), (1, 4, 3), (1, 4, 4), (2, 4, 1), (2, 4, 2),
                (1, 2, 1), (2, 2, 1), (4, 2, 1)),
            #[cfg(target_arch = "x86_64")]
            Backend::Sse2 => arms!(sse2_tile:
                (1, 4, 1), (1, 4, 2), (1, 4, 3), (2, 4, 1),
                (1, 2, 1), (2, 2, 1), (4, 2, 1), (1, 1, 1), (2, 1, 1), (4, 1, 1)),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Backend::Simd128 => arms!(simd128_tile:
                (1, 4, 1), (1, 4, 2), (1, 4, 3), (2, 4, 1),
                (1, 2, 1), (2, 2, 1), (4, 2, 1), (1, 1, 1), (2, 1, 1), (4, 1, 1)),
        }
    }
}

/// The reference tile: for each of `R` rows, lanes `0..4 L` of panels `0..NP` (relative to `w`,
/// `init` and `y`), `acc = init`, then `acc += x[i] * w[i]` for `i = 0..n_in`, no FMA.
unsafe fn scalar_tile<const NP: usize, const R: usize, const L: usize>(
    w: *const f32,
    n_in: usize,
    x: &Ptrs<*const f32>,
    init: &Ptrs<*const f32>,
    y: &Ptrs<*mut f32>,
) {
    let width = 4 * L;
    let mut a = [[[0.0f32; PACK]; NP]; R];
    for r in 0..R {
        for p in 0..NP {
            for j in 0..width {
                a[r][p][j] = unsafe { *init[r].add(p * PACK + j) };
            }
        }
    }
    for i in 0..n_in {
        for r in 0..R {
            let xi = unsafe { *x[r].add(i) };
            for p in 0..NP {
                let wp = unsafe { w.add((p * n_in + i) * PACK) };
                for j in 0..width {
                    a[r][p][j] += xi * unsafe { *wp.add(j) };
                }
            }
        }
    }
    for r in 0..R {
        for p in 0..NP {
            for j in 0..width {
                unsafe { *y[r].add(p * PACK + j) = a[r][p][j] };
            }
        }
    }
}

/// [`scalar_tile`] in one target's vector registers (`$lanes` floats each). Each weight vector is
/// loaded once per `i` and used by all `R` rows; per lane the operation order is the reference's.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! simd_tile {
    ($name:ident, $feat:literal, $arch:path, $ty:ty, $lanes:literal,
     zero = $zero:expr, splat = $splat:expr, load = $load:expr, store = $store:expr,
     madd = $madd:expr) => {
        #[target_feature(enable = $feat)]
        unsafe fn $name<const NP: usize, const R: usize, const L: usize>(
            w: *const f32,
            n_in: usize,
            x: &Ptrs<*const f32>,
            init: &Ptrs<*const f32>,
            y: &Ptrs<*mut f32>,
        ) {
            use $arch::*;
            const VPP: usize = PACK / $lanes;
            let nv = (4 * L).div_ceil($lanes);
            let x = *x;
            unsafe {
                let mut a: [[[$ty; VPP]; NP]; R] = [[[$zero; VPP]; NP]; R];
                for r in 0..R {
                    for p in 0..NP {
                        for k in 0..nv {
                            a[r][p][k] = $load(init[r].add(p * PACK + k * $lanes));
                        }
                    }
                }
                for i in 0..n_in {
                    let mut xv: [$ty; R] = [$zero; R];
                    for r in 0..R {
                        xv[r] = $splat(*x[r].add(i));
                    }
                    for p in 0..NP {
                        let wp = w.add((p * n_in + i) * PACK);
                        for k in 0..nv {
                            let wv = $load(wp.add(k * $lanes));
                            for r in 0..R {
                                a[r][p][k] = $madd(a[r][p][k], wv, xv[r]);
                            }
                        }
                    }
                }
                for r in 0..R {
                    for p in 0..NP {
                        for k in 0..nv {
                            $store(y[r].add(p * PACK + k * $lanes), a[r][p][k]);
                        }
                    }
                }
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
simd_tile!(
    neon_tile,
    "neon",
    core::arch::aarch64,
    float32x4_t,
    4,
    zero = vdupq_n_f32(0.0),
    splat = |c: f32| vdupq_n_f32(c),
    load = |p: *const f32| vld1q_f32(p),
    store = |p: *mut f32, v| vst1q_f32(p, v),
    madd = |a, w, x| vfmaq_f32(a, w, x)
);

#[cfg(target_arch = "x86_64")]
simd_tile!(
    avx2_tile,
    "avx2,fma",
    core::arch::x86_64,
    __m256,
    8,
    zero = _mm256_setzero_ps(),
    splat = |c: f32| _mm256_set1_ps(c),
    load = |p: *const f32| _mm256_loadu_ps(p),
    store = |p: *mut f32, v| _mm256_storeu_ps(p, v),
    madd = |a, w, x| _mm256_fmadd_ps(w, x, a)
);

#[cfg(target_arch = "x86_64")]
simd_tile!(
    sse2_tile,
    "sse2",
    core::arch::x86_64,
    __m128,
    4,
    zero = _mm_setzero_ps(),
    splat = |c: f32| _mm_set1_ps(c),
    load = |p: *const f32| _mm_loadu_ps(p),
    store = |p: *mut f32, v| _mm_storeu_ps(p, v),
    madd = |a, w, x| _mm_add_ps(a, _mm_mul_ps(w, x))
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
simd_tile!(
    simd128_tile,
    "simd128",
    core::arch::wasm32,
    v128,
    4,
    zero = f32x4_splat(0.0),
    splat = |c: f32| f32x4_splat(c),
    load = |p: *const f32| v128_load(p as *const v128),
    store = |p: *mut f32, v| v128_store(p as *mut v128, v),
    madd = |a, w, x| f32x4_add(a, f32x4_mul(w, x))
);

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
            (1, 128),
            (4, 7),
            (6, 48),
            (8, 192),
            (9, 20),
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
    fn gemv_rows_matches_gemv_per_row() {
        let mut r = Rng::new(4);
        // Narrow (1 / 4 / 6 / 8 outputs), partial and whole panels, odd row counts, overlapping
        // input rows (step < n_in, like a strided conv) and padded output rows.
        for &(o, i, rows, x_step, y_step) in &[
            (1, 96, 32, 32, 1),
            (4, 9, 5, 3, 4),
            (6, 48, 96, 48, 6),
            (8, 192, 7, 64, 8),
            (16, 3, 3, 1, 17),
            (40, 30, 6, 20, 40),
            (48, 144, 47, 96, 48),
            (64, 192, 9, 64, 64),
            (100, 70, 4, 70, 100),
        ] {
            let w = rand_vec(&mut r, o * i, 1.0 / (i as f32).sqrt());
            let x = rand_vec(&mut r, (rows - 1) * x_step + i, 1.0);
            let b = rand_vec(&mut r, o, 0.1);
            let m = Matrix::from_rows(&w, o, i);
            for be in Backend::all() {
                let mut y = vec![f32::NAN; (rows - 1) * y_step + o];
                gemv_rows(be, &m, &x, x_step, rows, &b, &mut y, y_step);
                for row in 0..rows {
                    let xr = &x[row * x_step..row * x_step + i];
                    let mut want = vec![0.0; o];
                    gemv(be, &m, xr, &b, &mut want);
                    let got = &y[row * y_step..row * y_step + o];
                    assert_eq!(got, &want[..], "{be:?} {o}x{i} row {row}/{rows}");
                    // Gaps between output rows are left alone.
                    assert!(
                        y[row * y_step + o..((row + 1) * y_step).min(y.len())]
                            .iter()
                            .all(|v| v.is_nan())
                    );
                    if be == Backend::Scalar {
                        let naive: Vec<f32> = (0..o)
                            .map(|a| (0..i).fold(b[a], |s, k| s + w[a * i + k] * xr[k]))
                            .collect();
                        assert_eq!(got, &naive[..], "scalar {o}x{i} row {row}");
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn gemv_rows_rejects_overflowing_span() {
        let m = Matrix::from_rows(&[1.0], 1, 1);
        let x = [1.0];
        let b = [0.0];
        let mut y = [0.0; 3];
        gemv_rows(
            Backend::Scalar,
            &m,
            &x,
            1usize << (usize::BITS - 1),
            3,
            &b,
            &mut y,
            1,
        );
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
