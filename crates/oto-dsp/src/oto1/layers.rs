//! Oto-1's layer set, all built on [`gemv`] and [`activate`]. Activations are laid out
//! frequency-major, `[F][C]` (index `f * C + c`); see `SPEC.md` for the PyTorch equivalents.

use super::format::{Otow, Rng};
use super::kernels::{Act, Backend, Matrix, activate, gemv, gemv_rows};

/// Below this a recurrent state value is flushed to zero (no subnormals in silence).
pub const FLUSH: f32 = 1e-15;

/// Gathers tensors for one model, either from a parsed file or as seeded random weights (with
/// the shapes the file would have), so both paths build the exact same layers.
pub enum Source<'a> {
    File(&'a mut Otow),
    Random(Rng, &'a mut Vec<(String, Vec<usize>, Vec<f32>)>),
}

impl Source<'_> {
    /// Tensor `name` with `shape`; random tensors are uniform in `±1/sqrt(fan_in)` (PyTorch's
    /// default init), biases included.
    pub fn take(&mut self, name: &str, shape: &[usize], fan_in: usize) -> Result<Vec<f32>, String> {
        match self {
            Source::File(o) => o.take(name, shape),
            Source::Random(rng, out) => {
                let s = 1.0 / (fan_in as f32).sqrt();
                let v: Vec<f32> = (0..shape.iter().product())
                    .map(|_| rng.uniform() * s)
                    .collect();
                out.push((name.to_owned(), shape.to_vec(), v.clone()));
                Ok(v)
            }
        }
    }
}

/// `y = W x + b` (PyTorch `nn.Linear`: `weight [out, in]`, `bias [out]`).
pub struct Linear {
    m: Matrix,
    b: Vec<f32>,
}

impl Linear {
    pub fn load(src: &mut Source, name: &str, n_in: usize, n_out: usize) -> Result<Self, String> {
        let w = src.take(&format!("{name}.weight"), &[n_out, n_in], n_in)?;
        let b = src.take(&format!("{name}.bias"), &[n_out], n_in)?;
        Ok(Self {
            m: Matrix::from_rows(&w, n_out, n_in),
            b,
        })
    }

    pub fn forward(&self, be: Backend, x: &[f32], y: &mut [f32]) {
        gemv(be, &self.m, x, &self.b, &mut y[..self.m.n_out()]);
    }

    pub fn n_out(&self) -> usize {
        self.m.n_out()
    }

    pub fn macs(&self) -> usize {
        self.m.macs()
    }

    pub fn params(&self) -> usize {
        self.m.macs() + self.b.len()
    }
}

/// Frequency-only convolution (time kernel 1) with BatchNorm folded in: `[F_in][C_in]` to
/// `[F_out][C_out]`, `F_out = (F_in + 2 pad - K) / stride + 1`, zero padding.
///
/// PyTorch: `nn.Conv2d(C_in, C_out, (1, K), stride=(1, stride), padding=(0, pad))` on
/// `[B, C, T, F]` (weight `[C_out, C_in, 1, K]`) or `nn.Conv1d` on `[B*T, C, F]` (weight
/// `[C_out, C_in, K]`); the file may store either rank.
pub struct Conv {
    ci: usize,
    k: usize,
    stride: usize,
    pad: usize,
    f_in: usize,
    f_out: usize,
    /// Rows of the GEMV input are `k * C_in + ci` (one input frequency row after another).
    lin: Linear,
}

/// Shape of one [`Conv`].
#[derive(Clone, Copy, Debug)]
pub struct ConvSpec {
    pub ci: usize,
    pub co: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub f_in: usize,
}

impl ConvSpec {
    pub fn f_out(&self) -> usize {
        (self.f_in + 2 * self.pad - self.k) / self.stride + 1
    }
}

impl Conv {
    pub fn load(src: &mut Source, name: &str, s: ConvSpec) -> Result<Self, String> {
        let fan_in = s.ci * s.k;
        let wname = format!("{name}.weight");
        // Accept [Co, Ci, K] (Conv1d) or [Co, Ci, 1, K] (Conv2d with a (1, K) kernel).
        let w = match src {
            Source::File(o)
                if o.tensors
                    .iter()
                    .any(|t| t.name == wname && t.shape.len() == 4) =>
            {
                o.take(&wname, &[s.co, s.ci, 1, s.k])?
            }
            _ => src.take(&wname, &[s.co, s.ci, s.k], fan_in)?,
        };
        let b = src.take(&format!("{name}.bias"), &[s.co], fan_in)?;
        // [co][ci][k] -> rows (co), columns (k * ci + c).
        let mut rows = vec![0.0; s.co * s.k * s.ci];
        for o in 0..s.co {
            for c in 0..s.ci {
                for k in 0..s.k {
                    rows[o * s.k * s.ci + k * s.ci + c] = w[(o * s.ci + c) * s.k + k];
                }
            }
        }
        Ok(Self {
            ci: s.ci,
            k: s.k,
            stride: s.stride,
            pad: s.pad,
            f_in: s.f_in,
            f_out: s.f_out(),
            lin: Linear {
                m: Matrix::from_rows(&rows, s.co, s.k * s.ci),
                b,
            },
        })
    }

    pub fn f_out(&self) -> usize {
        self.f_out
    }

    pub fn out_len(&self) -> usize {
        self.f_out * self.lin.n_out()
    }

    /// Scratch `patch` needs `K * C_in` floats.
    pub fn patch_len(&self) -> usize {
        self.k * self.ci
    }

    pub fn forward(&self, be: Backend, x: &[f32], patch: &mut [f32], y: &mut [f32]) {
        let (ci, co, k, stride) = (self.ci, self.lin.n_out(), self.k, self.stride);
        assert!(x.len() >= self.f_in * ci && y.len() >= self.out_len());
        // Output row f reads input rows first(f)..first(f) + K; the interior rows (all K inside
        // the input) are one contiguous range, read in place from [F][C] with no copy.
        let first = |f: usize| (f * stride) as isize - self.pad as isize;
        let inside = |f: usize| first(f) >= 0 && first(f) as usize + k <= self.f_in;
        let lo = (0..self.f_out).find(|&f| inside(f)).unwrap_or(self.f_out);
        let hi = (lo..self.f_out).find(|&f| !inside(f)).unwrap_or(self.f_out);
        if lo < hi {
            let s = first(lo) as usize * ci;
            let n_rows = hi - lo;
            gemv_rows(
                be,
                &self.lin.m,
                &x[s..],
                if n_rows > 1 { stride * ci } else { 0 },
                n_rows,
                &self.lin.b,
                &mut y[lo * co..],
                co,
            );
        }
        // Edge rows: gather the K input rows into `patch`, zero-padding past either end.
        for f in (0..lo).chain(hi..self.f_out) {
            for t in 0..k {
                let row = first(f) + t as isize;
                let dst = &mut patch[t * ci..(t + 1) * ci];
                if row >= 0 && (row as usize) < self.f_in {
                    let r = row as usize;
                    dst.copy_from_slice(&x[r * ci..(r + 1) * ci]);
                } else {
                    dst.fill(0.0);
                }
            }
            self.lin
                .forward(be, &patch[..k * ci], &mut y[f * co..(f + 1) * co]);
        }
    }

    pub fn macs(&self) -> usize {
        self.f_out * self.lin.macs()
    }

    pub fn params(&self) -> usize {
        self.lin.params()
    }
}

/// [`Conv`] followed by an activation over its whole output.
pub fn conv_act(
    be: Backend,
    c: &Conv,
    act: Option<Act>,
    x: &[f32],
    patch: &mut [f32],
    y: &mut [f32],
) {
    c.forward(be, x, patch, y);
    if let Some(a) = act {
        activate(be, a, &mut y[..c.out_len()]);
    }
}

/// One GRU layer, optionally split into `G` independent groups (block-diagonal weights).
/// PyTorch `nn.GRU` equations and gate order `(r, z, n)`:
///
/// ```text
/// r = σ(W_ir x + b_ir + W_hr h + b_hr)
/// z = σ(W_iz x + b_iz + W_hz h + b_hz)
/// n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))
/// h = (1 − z) ⊙ n + z ⊙ h,   then |h| < 1e-15 → 0
/// ```
///
/// Group `g` reads input `x[g*I..(g+1)*I]` and owns hidden `h[g*H..(g+1)*H]`.
pub struct Gru {
    groups: usize,
    i: usize,
    h: usize,
    w_ih: Vec<Matrix>,
    w_hh: Vec<Matrix>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
}

impl Gru {
    /// Tensors `{name}.weight_ih [G, 3H, I]`, `.weight_hh [G, 3H, H]`, `.bias_ih [G, 3H]`,
    /// `.bias_hh [G, 3H]` (per group sizes), or plain `nn.GRU` shapes without `G` when
    /// `groups == 1`.
    pub fn load(
        src: &mut Source,
        name: &str,
        groups: usize,
        i: usize,
        h: usize,
    ) -> Result<Self, String> {
        let g3 = |rest: &[usize]| -> Vec<usize> {
            if groups == 1 {
                rest.to_vec()
            } else {
                [&[groups][..], rest].concat()
            }
        };
        let fan = h; // PyTorch inits all GRU tensors with 1/sqrt(hidden)
        let wih = src.take(&format!("{name}.weight_ih"), &g3(&[3 * h, i]), fan)?;
        let whh = src.take(&format!("{name}.weight_hh"), &g3(&[3 * h, h]), fan)?;
        let b_ih = src.take(&format!("{name}.bias_ih"), &g3(&[3 * h]), fan)?;
        let b_hh = src.take(&format!("{name}.bias_hh"), &g3(&[3 * h]), fan)?;
        let w_ih = (0..groups)
            .map(|g| Matrix::from_rows(&wih[g * 3 * h * i..(g + 1) * 3 * h * i], 3 * h, i))
            .collect();
        let w_hh = (0..groups)
            .map(|g| Matrix::from_rows(&whh[g * 3 * h * h..(g + 1) * 3 * h * h], 3 * h, h))
            .collect();
        Ok(Self {
            groups,
            i,
            h,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    pub fn hidden(&self) -> usize {
        self.groups * self.h
    }

    /// Scratch `gi` and `gh` each need `3 * H` floats (per group H).
    pub fn scratch_len(&self) -> usize {
        3 * self.h
    }

    pub fn step(&self, be: Backend, x: &[f32], state: &mut [f32], gi: &mut [f32], gh: &mut [f32]) {
        let (i, h) = (self.i, self.h);
        assert!(x.len() >= self.groups * i && state.len() >= self.groups * h);
        for g in 0..self.groups {
            let hs = &mut state[g * h..(g + 1) * h];
            let b = g * 3 * h..(g + 1) * 3 * h;
            gemv(
                be,
                &self.w_ih[g],
                &x[g * i..(g + 1) * i],
                &self.b_ih[b.clone()],
                gi,
            );
            gemv(be, &self.w_hh[g], hs, &self.b_hh[b], gh);
            for j in 0..2 * h {
                gi[j] += gh[j];
            }
            activate(be, Act::Sigmoid, &mut gi[..2 * h]);
            for j in 0..h {
                gi[2 * h + j] += gi[j] * gh[2 * h + j];
            }
            activate(be, Act::Tanh, &mut gi[2 * h..3 * h]);
            for j in 0..h {
                let (z, n) = (gi[h + j], gi[2 * h + j]);
                let v = (1.0 - z) * n + z * hs[j];
                hs[j] = if v.abs() < FLUSH { 0.0 } else { v };
            }
        }
    }

    pub fn macs(&self) -> usize {
        self.groups * (self.w_ih[0].macs() + self.w_hh[0].macs())
    }

    pub fn params(&self) -> usize {
        self.macs() + self.b_ih.len() + self.b_hh.len()
    }
}

/// DFN's grouped-GRU shuffle between layers: `h.view(H/G, G).transpose().reshape(H)`, i.e.
/// `out[b * (H/G) + a] = h[a * G + b]`.
pub fn group_shuffle(h: &[f32], groups: usize, out: &mut [f32]) {
    let per = h.len() / groups;
    for a in 0..per {
        for b in 0..groups {
            out[b * per + a] = h[a * groups + b];
        }
    }
}

/// `out[f] = [a[f] (C_a channels), b[f] (C_b channels)]`: `torch.cat([a, b], dim=1)`.
pub fn concat_channels(a: &[f32], ca: usize, b: &[f32], cb: usize, f: usize, out: &mut [f32]) {
    for i in 0..f {
        out[i * (ca + cb)..i * (ca + cb) + ca].copy_from_slice(&a[i * ca..(i + 1) * ca]);
        out[i * (ca + cb) + ca..(i + 1) * (ca + cb)].copy_from_slice(&b[i * cb..(i + 1) * cb]);
    }
}

/// `x += y`, elementwise.
pub fn add_into(x: &mut [f32], y: &[f32]) {
    for (a, &b) in x.iter_mut().zip(y) {
        *a += b;
    }
}
