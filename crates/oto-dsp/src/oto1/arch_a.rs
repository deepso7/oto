//! Candidate A (`arch_id = 1`): DeepFilterNet-style hybrid. 32 ERB gains over the full band plus
//! a causal 3-tap complex deep filter on bins `0..96`, around a grouped GRU. See `SPEC.md`.

use super::fft::Stft;
use super::format::Otow;
use super::kernels::{Act, Backend, activate};
use super::layers::{Conv, ConvSpec, Gru, Linear, Source, add_into, conv_act, group_shuffle};
use super::{HOP, N_BINS, compress_into, erb_band_edges};

pub const N_ERB: usize = 32;
pub const N_DF: usize = 96;
pub const DF_ORDER: usize = 3;
/// Frequency rows of the embedding both encoder paths end at.
const F_EMB: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfigA {
    pub channels: usize,
    pub gru_hidden: usize,
    pub gru_groups: usize,
}

impl Default for ConfigA {
    fn default() -> Self {
        Self {
            channels: 48,
            gru_hidden: 192,
            gru_groups: 4,
        }
    }
}

impl ConfigA {
    pub fn from_otow(o: &Otow) -> Result<Self, String> {
        for (key, want) in [
            ("n_erb", N_ERB),
            ("n_df_bins", N_DF),
            ("df_order", DF_ORDER),
        ] {
            if let Some(v) = o.config_u32(key)
                && v as usize != want
            {
                return Err(format!(
                    "Oto-1 A: {key} = {v}, this kernel implements {want}"
                ));
            }
        }
        let d = Self::default();
        let get = |k: &str, d: usize| o.config_u32(k).map_or(d, |v| v as usize);
        let c = Self {
            channels: get("channels", d.channels),
            gru_hidden: get("gru_hidden", d.gru_hidden),
            gru_groups: get("gru_groups", d.gru_groups),
        };
        let ok = c.channels > 0
            && c.gru_groups > 0
            && c.gru_hidden.is_multiple_of(c.gru_groups)
            && (F_EMB * c.channels).is_multiple_of(c.gru_groups);
        if !ok {
            return Err(format!("Oto-1 A: inconsistent config {c:?}"));
        }
        Ok(c)
    }

    pub fn json(&self) -> String {
        format!(
            r#"{{"arch":"oto1-a","channels":{},"df_order":{DF_ORDER},"gru_groups":{},"gru_hidden":{},"n_df_bins":{N_DF},"n_erb":{N_ERB}}}"#,
            self.channels, self.gru_groups, self.gru_hidden
        )
    }
}

pub struct ArchA {
    be: Backend,
    cfg: ConfigA,
    stft: Stft,
    band_of: Vec<u8>,
    band_width: Vec<f32>,
    edges: Vec<usize>,

    erb0: Conv,
    erb1: Conv,
    erb2: Conv,
    df0: Conv,
    df1: Conv,
    df2: Conv,
    df3: Conv,
    gru0: Gru,
    gru1: Gru,
    erb_fc: Linear,
    erb_up2: Conv,
    erb_up1: Conv,
    erb_out: Conv,
    df_fc0: Linear,
    df_fc1: Linear,
    df_convp: Conv,
    p_head: Linear,

    feat_erb: Vec<f32>,
    feat_df: Vec<f32>,
    e0: Vec<f32>,
    e1: Vec<f32>,
    e2: Vec<f32>,
    d0: Vec<f32>,
    d1: Vec<f32>,
    d2: Vec<f32>,
    d3: Vec<f32>,
    h0: Vec<f32>,
    h0s: Vec<f32>,
    h1: Vec<f32>,
    gi: Vec<f32>,
    gh: Vec<f32>,
    t: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    gains: Vec<f32>,
    q: Vec<f32>,
    coef: Vec<f32>,
    cp: Vec<f32>,
    ptg: Vec<f32>,
    patch: Vec<f32>,
    yspec: Vec<f32>,
    /// Bins `0..N_DF` of the previous `DF_ORDER - 1` spectra, newest first.
    hist: Vec<f32>,
}

impl ArchA {
    pub fn build(src: &mut Source, cfg: ConfigA, be: Backend) -> Result<Self, String> {
        let c = cfg.channels;
        let (h, g) = (cfg.gru_hidden, cfg.gru_groups);
        let cs = |ci, co, k, stride, pad, f_in| ConvSpec {
            ci,
            co,
            k,
            stride,
            pad,
            f_in,
        };
        let erb0 = Conv::load(src, "enc.erb0", cs(1, c, 3, 1, 1, N_ERB))?;
        let erb1 = Conv::load(src, "enc.erb1", cs(c, c, 3, 2, 1, 32))?;
        let erb2 = Conv::load(src, "enc.erb2", cs(c, c, 3, 2, 1, 16))?;
        let df0 = Conv::load(src, "enc.df0", cs(2, c, 3, 1, 1, N_DF))?;
        let df1 = Conv::load(src, "enc.df1", cs(c, c, 3, 2, 1, 96))?;
        let df2 = Conv::load(src, "enc.df2", cs(c, c, 3, 2, 1, 48))?;
        let df3 = Conv::load(src, "enc.df3", cs(c, c, 3, 3, 0, 24))?;
        debug_assert_eq!((erb2.f_out(), df3.f_out()), (F_EMB, F_EMB));
        let emb = F_EMB * c;
        let gru0 = Gru::load(src, "gru0", g, emb / g, h / g)?;
        let gru1 = Gru::load(src, "gru1", g, h / g, h / g)?;
        let erb_fc = Linear::load(src, "erb_dec.fc", h, emb)?;
        let erb_up2 = Conv::load(src, "erb_dec.up2", cs(c, 2 * c, 3, 1, 1, 8))?;
        let erb_up1 = Conv::load(src, "erb_dec.up1", cs(c, 2 * c, 3, 1, 1, 16))?;
        let erb_out = Conv::load(src, "erb_dec.out", cs(c, 1, 3, 1, 1, 32))?;
        let df_fc0 = Linear::load(src, "df_dec.fc0", h, emb)?;
        let df_fc1 = Linear::load(src, "df_dec.fc1", emb, N_DF * DF_ORDER * 2)?;
        let df_convp = Conv::load(src, "df_dec.convp", cs(c, DF_ORDER * 2, 1, 1, 0, N_DF))?;
        let p_head = Linear::load(src, "p_head", h, 1)?;

        let convs = [
            &erb0, &erb1, &erb2, &df0, &df1, &df2, &df3, &erb_up2, &erb_up1, &erb_out, &df_convp,
        ];
        let patch = vec![0.0; convs.iter().map(|c| c.patch_len()).max().unwrap()];
        let edges = erb_band_edges(N_ERB);
        let mut band_of = vec![0u8; N_BINS];
        let mut band_width = vec![0.0; N_ERB];
        for b in 0..N_ERB {
            band_of[edges[b]..edges[b + 1]].fill(b as u8);
            band_width[b] = (edges[b + 1] - edges[b]) as f32;
        }
        let z = |n: usize| vec![0.0f32; n];
        Ok(Self {
            be,
            cfg,
            stft: Stft::new(),
            band_of,
            band_width,
            edges,
            feat_erb: z(N_ERB),
            feat_df: z(2 * N_DF),
            e0: z(erb0.out_len()),
            e1: z(erb1.out_len()),
            e2: z(erb2.out_len()),
            d0: z(df0.out_len()),
            d1: z(df1.out_len()),
            d2: z(df2.out_len()),
            d3: z(df3.out_len()),
            h0: z(h),
            h0s: z(h),
            h1: z(h),
            gi: z(gru0.scratch_len().max(gru1.scratch_len())),
            gh: z(gru0.scratch_len().max(gru1.scratch_len())),
            t: z(emb),
            u: z(erb_up2.out_len()),
            v: z(erb_up1.out_len()),
            gains: z(N_ERB),
            q: z(emb),
            coef: z(N_DF * DF_ORDER * 2),
            cp: z(df_convp.out_len()),
            ptg: z(1),
            patch,
            yspec: z(2 * N_BINS),
            hist: z((DF_ORDER - 1) * 2 * N_DF),
            erb0,
            erb1,
            erb2,
            df0,
            df1,
            df2,
            df3,
            gru0,
            gru1,
            erb_fc,
            erb_up2,
            erb_up1,
            erb_out,
            df_fc0,
            df_fc1,
            df_convp,
            p_head,
        })
    }

    pub fn config(&self) -> ConfigA {
        self.cfg
    }

    pub fn p_target(&self) -> f32 {
        self.ptg[0]
    }

    pub fn set_backend(&mut self, be: Backend) {
        self.be = be;
    }

    pub fn macs(&self) -> usize {
        let convs = [
            &self.erb0,
            &self.erb1,
            &self.erb2,
            &self.df0,
            &self.df1,
            &self.df2,
            &self.df3,
            &self.erb_up2,
            &self.erb_up1,
            &self.erb_out,
            &self.df_convp,
        ];
        let lins = [&self.erb_fc, &self.df_fc0, &self.df_fc1, &self.p_head];
        convs.iter().map(|c| c.macs()).sum::<usize>()
            + lins.iter().map(|l| l.macs()).sum::<usize>()
            + self.gru0.macs()
            + self.gru1.macs()
    }

    pub fn params(&self) -> usize {
        let convs = [
            &self.erb0,
            &self.erb1,
            &self.erb2,
            &self.df0,
            &self.df1,
            &self.df2,
            &self.df3,
            &self.erb_up2,
            &self.erb_up1,
            &self.erb_out,
            &self.df_convp,
        ];
        let lins = [&self.erb_fc, &self.df_fc0, &self.df_fc1, &self.p_head];
        convs.iter().map(|c| c.params()).sum::<usize>()
            + lins.iter().map(|l| l.params()).sum::<usize>()
            + self.gru0.params()
            + self.gru1.params()
    }

    pub fn band_edges(&self) -> &[usize] {
        &self.edges
    }

    pub fn reset(&mut self) {
        self.stft.reset();
        for b in [&mut self.h0, &mut self.h1, &mut self.hist, &mut self.ptg] {
            b.fill(0.0);
        }
    }

    pub fn state(&self) -> impl Iterator<Item = f32> + '_ {
        self.stft
            .state()
            .chain(self.h0.iter().chain(&self.h1).chain(&self.hist).copied())
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert!(input.len() == HOP && output.len() == HOP);
        let be = self.be;
        self.stft.analyze(input);
        let x = &self.stft.spec;

        // Features.
        for b in 0..N_ERB {
            let mut s = 0.0f32;
            for k in self.edges[b]..self.edges[b + 1] {
                s += x[2 * k] * x[2 * k] + x[2 * k + 1] * x[2 * k + 1];
            }
            self.feat_erb[b] = (s / self.band_width[b] + 1e-10).log10();
        }
        compress_into(&x[..2 * N_DF], &mut self.feat_df);

        // Encoders.
        let p = &mut self.patch;
        conv_act(
            be,
            &self.erb0,
            Some(Act::Silu),
            &self.feat_erb,
            p,
            &mut self.e0,
        );
        conv_act(be, &self.erb1, Some(Act::Silu), &self.e0, p, &mut self.e1);
        conv_act(be, &self.erb2, Some(Act::Silu), &self.e1, p, &mut self.e2);
        conv_act(
            be,
            &self.df0,
            Some(Act::Silu),
            &self.feat_df,
            p,
            &mut self.d0,
        );
        conv_act(be, &self.df1, Some(Act::Silu), &self.d0, p, &mut self.d1);
        conv_act(be, &self.df2, Some(Act::Silu), &self.d1, p, &mut self.d2);
        conv_act(be, &self.df3, Some(Act::Silu), &self.d2, p, &mut self.d3);
        // emb = e2 + d3, kept in d3.
        add_into(&mut self.d3, &self.e2);

        // Grouped GRUs.
        self.gru0
            .step(be, &self.d3, &mut self.h0, &mut self.gi, &mut self.gh);
        group_shuffle(&self.h0, self.cfg.gru_groups, &mut self.h0s);
        self.gru1
            .step(be, &self.h0s, &mut self.h1, &mut self.gi, &mut self.gh);

        // ERB gain decoder.
        self.erb_fc.forward(be, &self.h1, &mut self.t);
        activate(be, Act::Silu, &mut self.t);
        add_into(&mut self.t, &self.e2);
        conv_act(be, &self.erb_up2, Some(Act::Silu), &self.t, p, &mut self.u);
        add_into(&mut self.u, &self.e1);
        conv_act(be, &self.erb_up1, Some(Act::Silu), &self.u, p, &mut self.v);
        add_into(&mut self.v, &self.e0);
        conv_act(
            be,
            &self.erb_out,
            Some(Act::Sigmoid),
            &self.v,
            p,
            &mut self.gains,
        );

        // Deep-filter coefficient decoder.
        self.df_fc0.forward(be, &self.h1, &mut self.q);
        activate(be, Act::Silu, &mut self.q);
        self.df_fc1.forward(be, &self.q, &mut self.coef);
        self.df_convp.forward(be, &self.d0, p, &mut self.cp);
        add_into(&mut self.coef, &self.cp);
        activate(be, Act::Tanh, &mut self.coef);

        self.p_head.forward(be, &self.h1, &mut self.ptg);

        // Apply: ERB gains above the DF band, deep filter below it (on the noisy spectrum).
        let y = &mut self.yspec;
        for k in N_DF..N_BINS {
            let g = self.gains[self.band_of[k] as usize];
            y[2 * k] = x[2 * k] * g;
            y[2 * k + 1] = x[2 * k + 1] * g;
        }
        for k in 0..N_DF {
            let (mut yr, mut yi) = (0.0f32, 0.0f32);
            for i in 0..DF_ORDER {
                let (cr, ci) = (
                    self.coef[k * 2 * DF_ORDER + 2 * i],
                    self.coef[k * 2 * DF_ORDER + 2 * i + 1],
                );
                let (xr, xi) = if i == 0 {
                    (x[2 * k], x[2 * k + 1])
                } else {
                    let h = &self.hist[(i - 1) * 2 * N_DF..];
                    (h[2 * k], h[2 * k + 1])
                };
                yr += cr * xr - ci * xi;
                yi += cr * xi + ci * xr;
            }
            y[2 * k] = yr;
            y[2 * k + 1] = yi;
        }
        self.hist.copy_within(..(DF_ORDER - 2) * 2 * N_DF, 2 * N_DF);
        self.hist[..2 * N_DF].copy_from_slice(&x[..2 * N_DF]);

        self.stft.synthesize(&self.yspec, output);
    }
}
