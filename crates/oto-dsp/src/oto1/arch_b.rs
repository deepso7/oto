//! Candidate B (`arch_id = 2`): FastEnhancer-style U-Net. Frequency-only conv encoder
//! (512 → 128 → 64 → 32 → 16 rows), a 2-layer GRU over time at the bottleneck, and a
//! skip-connected sub-pixel conv decoder that outputs a bounded complex ratio mask on bins
//! `0..512` (Nyquist is zeroed). See `SPEC.md`.

use super::fft::Stft;
use super::format::Otow;
use super::kernels::{Act, Backend, activate};
use super::layers::{Conv, ConvSpec, Gru, Linear, Source, concat_channels, conv_act};
use super::{HOP, N_BINS, compress_into};

/// Bins the network sees and masks (the Nyquist bin 512 is dropped).
pub const F_IN: usize = 512;
const F_BOTT: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfigB {
    pub channels: usize,
    pub gru_hidden: usize,
}

/// Legacy v1 fallback width when an older `.otow` file omits `"channels"`.
const LEGACY_V1_CHANNELS: usize = 32;

impl Default for ConfigB {
    fn default() -> Self {
        Self {
            channels: 24,
            gru_hidden: 128,
        }
    }
}

impl ConfigB {
    pub fn from_otow(o: &Otow) -> Result<Self, String> {
        let d = Self::default();
        let get = |k: &str, d: usize| o.config_u32(k).map_or(d, |v| v as usize);
        let c = Self {
            channels: get("channels", LEGACY_V1_CHANNELS),
            gru_hidden: get("gru_hidden", d.gru_hidden),
        };
        if c.channels == 0 || c.gru_hidden == 0 {
            return Err(format!("Oto-1 B: inconsistent config {c:?}"));
        }
        Ok(c)
    }

    pub fn json(&self) -> String {
        format!(
            r#"{{"arch":"oto1-b","channels":{},"gru_hidden":{}}}"#,
            self.channels, self.gru_hidden
        )
    }
}

pub struct ArchB {
    be: Backend,
    cfg: ConfigB,
    stft: Stft,

    enc0: Conv,
    enc1: Conv,
    enc2: Conv,
    enc3: Conv,
    gru0: Gru,
    gru1: Gru,
    bott: Linear,
    dec3: Conv,
    dec2: Conv,
    dec1: Conv,
    dec0: Conv,
    p_head: Linear,

    feat: Vec<f32>,
    e0: Vec<f32>,
    e1: Vec<f32>,
    e2: Vec<f32>,
    e3: Vec<f32>,
    h0: Vec<f32>,
    h1: Vec<f32>,
    gi: Vec<f32>,
    gh: Vec<f32>,
    b: Vec<f32>,
    cat: Vec<f32>,
    d3: Vec<f32>,
    d2: Vec<f32>,
    d1: Vec<f32>,
    m: Vec<f32>,
    r: Vec<f32>,
    ptg: Vec<f32>,
    patch: Vec<f32>,
    yspec: Vec<f32>,
}

impl ArchB {
    pub fn build(src: &mut Source, cfg: ConfigB, be: Backend) -> Result<Self, String> {
        let (c, h) = (cfg.channels, cfg.gru_hidden);
        let cs = |ci, co, k, stride, pad, f_in| ConvSpec {
            ci,
            co,
            k,
            stride,
            pad,
            f_in,
        };
        let enc0 = Conv::load(src, "enc0", cs(2, c, 8, 4, 2, F_IN))?;
        let enc1 = Conv::load(src, "enc1", cs(c, c, 3, 2, 1, 128))?;
        let enc2 = Conv::load(src, "enc2", cs(c, c, 3, 2, 1, 64))?;
        let enc3 = Conv::load(src, "enc3", cs(c, c, 3, 2, 1, 32))?;
        debug_assert_eq!(enc3.f_out(), F_BOTT);
        let gru0 = Gru::load(src, "gru0", 1, F_BOTT * c, h)?;
        let gru1 = Gru::load(src, "gru1", 1, h, h)?;
        let bott = Linear::load(src, "bott.fc", h, F_BOTT * c)?;
        let dec3 = Conv::load(src, "dec3", cs(2 * c, 2 * c, 3, 1, 1, 16))?;
        let dec2 = Conv::load(src, "dec2", cs(2 * c, 2 * c, 3, 1, 1, 32))?;
        let dec1 = Conv::load(src, "dec1", cs(2 * c, 2 * c, 3, 1, 1, 64))?;
        let dec0 = Conv::load(src, "dec0", cs(2 * c, 8, 3, 1, 1, 128))?;
        let p_head = Linear::load(src, "p_head", h, 1)?;
        let convs = [&enc0, &enc1, &enc2, &enc3, &dec3, &dec2, &dec1, &dec0];
        let patch = vec![0.0; convs.iter().map(|c| c.patch_len()).max().unwrap()];
        let z = |n: usize| vec![0.0f32; n];
        Ok(Self {
            be,
            cfg,
            stft: Stft::new(),
            feat: z(2 * F_IN),
            e0: z(enc0.out_len()),
            e1: z(enc1.out_len()),
            e2: z(enc2.out_len()),
            e3: z(enc3.out_len()),
            h0: z(h),
            h1: z(h),
            gi: z(3 * h),
            gh: z(3 * h),
            b: z(F_BOTT * c),
            cat: z(128 * 2 * c),
            d3: z(dec3.out_len()),
            d2: z(dec2.out_len()),
            d1: z(dec1.out_len()),
            m: z(dec0.out_len()),
            r: z(F_IN),
            ptg: z(1),
            patch,
            yspec: z(2 * N_BINS),
            enc0,
            enc1,
            enc2,
            enc3,
            gru0,
            gru1,
            bott,
            dec3,
            dec2,
            dec1,
            dec0,
            p_head,
        })
    }

    pub fn config(&self) -> ConfigB {
        self.cfg
    }

    pub fn p_target(&self) -> f32 {
        self.ptg[0]
    }

    pub fn set_backend(&mut self, be: Backend) {
        self.be = be;
    }

    fn convs(&self) -> [&Conv; 8] {
        [
            &self.enc0, &self.enc1, &self.enc2, &self.enc3, &self.dec3, &self.dec2, &self.dec1,
            &self.dec0,
        ]
    }

    pub fn macs(&self) -> usize {
        self.convs().iter().map(|c| c.macs()).sum::<usize>()
            + self.gru0.macs()
            + self.gru1.macs()
            + self.bott.macs()
            + self.p_head.macs()
    }

    pub fn params(&self) -> usize {
        self.convs().iter().map(|c| c.params()).sum::<usize>()
            + self.gru0.params()
            + self.gru1.params()
            + self.bott.params()
            + self.p_head.params()
    }

    pub fn reset(&mut self) {
        self.stft.reset();
        self.h0.fill(0.0);
        self.h1.fill(0.0);
        self.ptg.fill(0.0);
    }

    pub fn state(&self) -> impl Iterator<Item = f32> + '_ {
        self.stft
            .state()
            .chain(self.h0.iter().chain(&self.h1).copied())
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert!(input.len() == HOP && output.len() == HOP);
        let (be, c) = (self.be, self.cfg.channels);
        self.stft.analyze(input);
        let x = &self.stft.spec;
        compress_into(&x[..2 * F_IN], &mut self.feat);

        let p = &mut self.patch;
        conv_act(be, &self.enc0, Some(Act::Silu), &self.feat, p, &mut self.e0);
        conv_act(be, &self.enc1, Some(Act::Silu), &self.e0, p, &mut self.e1);
        conv_act(be, &self.enc2, Some(Act::Silu), &self.e1, p, &mut self.e2);
        conv_act(be, &self.enc3, Some(Act::Silu), &self.e2, p, &mut self.e3);

        self.gru0
            .step(be, &self.e3, &mut self.h0, &mut self.gi, &mut self.gh);
        self.gru1
            .step(be, &self.h0, &mut self.h1, &mut self.gi, &mut self.gh);
        self.bott.forward(be, &self.h1, &mut self.b);
        activate(be, Act::Silu, &mut self.b);

        // Decoder: concat skip, conv to 2C channels, sub-pixel shuffle (a no-op in [F][C]).
        concat_channels(&self.b, c, &self.e3, c, 16, &mut self.cat);
        conv_act(be, &self.dec3, Some(Act::Silu), &self.cat, p, &mut self.d3);
        concat_channels(&self.d3, c, &self.e2, c, 32, &mut self.cat);
        conv_act(be, &self.dec2, Some(Act::Silu), &self.cat, p, &mut self.d2);
        concat_channels(&self.d2, c, &self.e1, c, 64, &mut self.cat);
        conv_act(be, &self.dec1, Some(Act::Silu), &self.cat, p, &mut self.d1);
        concat_channels(&self.d1, c, &self.e0, c, 128, &mut self.cat);
        conv_act(be, &self.dec0, None, &self.cat, p, &mut self.m);

        self.p_head.forward(be, &self.h1, &mut self.ptg);

        // Bounded complex ratio mask: m * tanh(|m|) / |m|.
        for k in 0..F_IN {
            let (mr, mi) = (self.m[2 * k], self.m[2 * k + 1]);
            self.r[k] = (mr * mr + mi * mi).sqrt();
        }
        activate(be, Act::Tanh, &mut self.r);
        let y = &mut self.yspec;
        for k in 0..F_IN {
            let (mr, mi) = (self.m[2 * k], self.m[2 * k + 1]);
            let s = self.r[k] / (mr * mr + mi * mi).sqrt().max(1e-8);
            let (gr, gi) = (mr * s, mi * s);
            let (xr, xi) = (x[2 * k], x[2 * k + 1]);
            y[2 * k] = xr * gr - xi * gi;
            y[2 * k + 1] = xr * gi + xi * gr;
        }
        y[2 * F_IN] = 0.0;
        y[2 * F_IN + 1] = 0.0;

        self.stft.synthesize(&self.yspec, output);
    }
}
