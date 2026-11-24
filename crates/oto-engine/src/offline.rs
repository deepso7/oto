//! Offline processing of WAV files through the live chain (`oto process`).
//!
//! Evaluation needs the exact audio the virtual microphone would produce, without devices. This
//! runs the same [`Chain`] the capture path runs, set up the same way (`set_enabled`,
//! `set_max_attenuation_db`, then `prime`), fed in fixed blocks like a device callback:
//!
//! 1. Downmix to mono exactly like a live device ([`downmix`]: one or two channels averaged,
//!    more than two take channel 0).
//! 2. Resample to 48 kHz if needed (`rubato`'s synchronous FFT resampler, whose startup delay
//!    `process_all` trims, so resampling adds no delay).
//! 3. Run the chain in `chunk_size` blocks, then enough trailing zeros to flush
//!    [`Chain::latency`].
//! 4. Drop the chain's leading delay (unless [`ProcessOptions::delay_compensate`] is off),
//!    resample back to the input rate and trim to length.
//!
//! With delay compensation (the default), output sample `i` lines up with input sample `i` and
//! both have the same length, which is what objective metrics (PESQ, DNSMOS, SI-SDR) expect.
//! Without it the output keeps the chain's delay in front and the flushed tail at the end
//! (`input_samples + total_delay_samples` long), for impulse and latency checks.
//!
//! Also holds a small RIFF/WAVE reader and writer: PCM 16/24/32-bit and IEEE float 32-bit, plain
//! or `WAVE_FORMAT_EXTENSIBLE`, any channel count. Not for the realtime path.

use std::path::Path;
use std::time::Instant;

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use crate::dsp::{Chain, Model, SAMPLE_RATE};
use crate::pipeline::downmix;

/// Block size [`ProcessOptions::default`] feeds the chain: 10 ms at 48 kHz.
pub const DEFAULT_CHUNK: usize = 480;

/// How samples are stored in a WAV file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    Pcm16,
    Pcm24,
    Pcm32,
    Float32,
}

impl SampleFormat {
    fn bytes(self) -> usize {
        match self {
            SampleFormat::Pcm16 => 2,
            SampleFormat::Pcm24 => 3,
            SampleFormat::Pcm32 | SampleFormat::Float32 => 4,
        }
    }

    fn tag(self) -> u16 {
        match self {
            SampleFormat::Float32 => FORMAT_FLOAT,
            _ => FORMAT_PCM,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SampleFormat::Pcm16 => "pcm16",
            SampleFormat::Pcm24 => "pcm24",
            SampleFormat::Pcm32 => "pcm32",
            SampleFormat::Float32 => "float32",
        }
    }

    fn decode(self, b: &[u8]) -> f32 {
        match self {
            SampleFormat::Pcm16 => i16::from_le_bytes([b[0], b[1]]) as f32 / 32_768.0,
            // Into the top three bytes of an i32, so the shift sign-extends.
            SampleFormat::Pcm24 => {
                (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0
            }
            SampleFormat::Pcm32 => {
                (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64 / 2_147_483_648.0) as f32
            }
            SampleFormat::Float32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        }
    }

    /// Appends `x`. Integer formats clamp to ±1.0 and store non-finite samples as 0.
    fn encode(self, x: f32, out: &mut Vec<u8>) {
        let pcm = |full: f64| {
            if x.is_finite() {
                (x.clamp(-1.0, 1.0) as f64 * full).round() as i32
            } else {
                0
            }
        };
        match self {
            SampleFormat::Pcm16 => out.extend_from_slice(&(pcm(32_767.0) as i16).to_le_bytes()),
            SampleFormat::Pcm24 => out.extend_from_slice(&pcm(8_388_607.0).to_le_bytes()[..3]),
            SampleFormat::Pcm32 => out.extend_from_slice(&pcm(2_147_483_647.0).to_le_bytes()),
            SampleFormat::Float32 => out.extend_from_slice(&x.to_le_bytes()),
        }
    }
}

const FORMAT_PCM: u16 = 1;
const FORMAT_FLOAT: u16 = 3;
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// A decoded WAV file: interleaved samples as `f32` (integer formats scaled to ±1.0).
#[derive(Clone, Debug, PartialEq)]
pub struct Wav {
    pub sample_rate: u32,
    pub channels: u16,
    pub format: SampleFormat,
    pub samples: Vec<f32>,
}

impl Wav {
    /// A mono file.
    pub fn mono(sample_rate: u32, format: SampleFormat, samples: Vec<f32>) -> Self {
        Self {
            sample_rate,
            channels: 1,
            format,
            samples,
        }
    }

    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }

    /// Mono, mixed down the way the live capture path mixes a device with this many channels.
    pub fn downmixed(&self) -> Vec<f32> {
        let ch = self.channels.max(1) as usize;
        if ch == 1 {
            return self.samples.clone();
        }
        let mut out = vec![0.0; self.frames()];
        downmix(
            [(&self.samples[..out.len() * ch], ch)].into_iter(),
            &mut out,
        );
        out
    }

    /// Parses a RIFF/WAVE file. Unknown chunks are skipped; a data chunk whose size overruns
    /// the file (as streaming writers leave it) is cut to what is there.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err("not a RIFF/WAVE file".into());
        }
        let mut fmt: Option<(u32, u16, SampleFormat)> = None;
        let mut at = 12;
        while at + 8 <= bytes.len() {
            let id = &bytes[at..at + 4];
            let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
            let body = &bytes[at + 8..(at + 8).saturating_add(size).min(bytes.len())];
            match id {
                b"fmt " => fmt = Some(parse_fmt(body)?),
                b"data" => {
                    let (sample_rate, channels, format) =
                        fmt.ok_or("the data chunk comes before the fmt chunk")?;
                    let frame = format.bytes() * channels as usize;
                    let whole = body.len() / frame * frame;
                    let samples = body[..whole]
                        .chunks_exact(format.bytes())
                        .map(|b| format.decode(b))
                        .collect();
                    return Ok(Self {
                        sample_rate,
                        channels,
                        format,
                        samples,
                    });
                }
                _ => {}
            }
            // Chunks are padded to an even length.
            at = at
                .saturating_add(8)
                .saturating_add(size)
                .saturating_add(size & 1);
        }
        Err(if fmt.is_none() {
            "no fmt chunk".into()
        } else {
            "no data chunk".into()
        })
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The file as bytes: a plain `WAVEFORMAT` header (integer PCM) or `WAVEFORMATEX` plus a
    /// `fact` chunk (float), which every reader accepts.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        let ch = self.channels.max(1);
        let width = self.format.bytes();
        let data_len = self.samples.len() * width;
        let float = self.format == SampleFormat::Float32;
        let fmt_len: usize = if float { 18 } else { 16 };
        let fact_len: usize = if float { 12 } else { 0 };
        let riff_len = 4 + (8 + fmt_len) + fact_len + 8 + data_len + (data_len & 1);
        let riff_len = u32::try_from(riff_len).map_err(|_| "too long for a WAV file")?;
        let mut out = Vec::with_capacity(riff_len as usize + 8);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&riff_len.to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&(fmt_len as u32).to_le_bytes());
        out.extend_from_slice(&self.format.tag().to_le_bytes());
        out.extend_from_slice(&ch.to_le_bytes());
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        let block = ch as u32 * width as u32;
        out.extend_from_slice(&(self.sample_rate * block).to_le_bytes());
        out.extend_from_slice(&(block as u16).to_le_bytes());
        out.extend_from_slice(&(width as u16 * 8).to_le_bytes());
        if float {
            out.extend_from_slice(&0u16.to_le_bytes()); // cbSize
            out.extend_from_slice(b"fact");
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&(self.frames() as u32).to_le_bytes());
        }
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data_len as u32).to_le_bytes());
        for &x in &self.samples {
            self.format.encode(x, &mut out);
        }
        if data_len & 1 == 1 {
            out.push(0);
        }
        Ok(out)
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.encode()?).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// `(sample rate, channels, format)` from a fmt chunk body.
fn parse_fmt(b: &[u8]) -> Result<(u32, u16, SampleFormat), String> {
    if b.len() < 16 {
        return Err("fmt chunk too short".into());
    }
    let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let mut tag = u16_at(0);
    let channels = u16_at(2);
    let rate = u32::from_le_bytes(b[4..8].try_into().unwrap());
    let block_align = u16_at(12);
    let bits = u16_at(14);
    if tag == FORMAT_EXTENSIBLE {
        // cbSize (16), valid bits (18), channel mask (20), then the sub-format GUID (24..40),
        // whose first two bytes are the actual format tag.
        if b.len() < 40 {
            return Err("WAVE_FORMAT_EXTENSIBLE fmt chunk too short".into());
        }
        tag = u16_at(24);
    }
    let format = match (tag, bits) {
        (FORMAT_PCM, 16) => SampleFormat::Pcm16,
        (FORMAT_PCM, 24) => SampleFormat::Pcm24,
        (FORMAT_PCM, 32) => SampleFormat::Pcm32,
        (FORMAT_FLOAT, 32) => SampleFormat::Float32,
        _ => {
            return Err(format!(
                "unsupported sample format (tag {tag}, {bits} bits); use 16/24/32-bit PCM or \
                 32-bit float"
            ));
        }
    };
    if channels == 0 || rate == 0 {
        return Err("fmt chunk has no channels or a zero sample rate".into());
    }
    if block_align as usize != channels as usize * format.bytes() {
        return Err(format!(
            "block align {block_align} does not match {channels} x {bits}-bit"
        ));
    }
    Ok((rate, channels, format))
}

/// How [`process`] runs the chain.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessOptions {
    pub model: Model,
    /// `false` bypasses the denoiser like `Settings::enabled`: the model's delay stays, high-pass
    /// and limiter still run.
    pub enabled: bool,
    /// [`Chain::set_max_attenuation_db`], e.g. from `Strength::max_attenuation_db`.
    pub max_attenuation_db: f32,
    /// Samples per [`Chain::process`] call, at 48 kHz. Live callbacks are typically 256–512.
    pub chunk_size: usize,
    /// Remove the chain's delay so the output lines up with the input (see the module docs).
    pub delay_compensate: bool,
}

impl Default for ProcessOptions {
    fn default() -> Self {
        Self {
            model: Model::default(),
            enabled: true,
            max_attenuation_db: crate::Strength::default().max_attenuation_db(),
            chunk_size: DEFAULT_CHUNK,
            delay_compensate: true,
        }
    }
}

/// What [`process`] did. Levels are over the whole mono signal; dBFS values floor at −120.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessReport {
    /// The model that ran.
    pub model: Model,
    pub enabled: bool,
    pub max_attenuation_db: f32,
    pub chunk_size: usize,
    /// Rate of the input and of the output, Hz.
    pub sample_rate: u32,
    pub input_samples: usize,
    pub output_samples: usize,
    /// [`Chain::latency`], samples at 48 kHz.
    pub chain_latency_samples: usize,
    /// The chain's delay at `sample_rate`, rounded: what was removed with `delay_compensated`,
    /// or what leads the output without it. Resampling adds none.
    pub total_delay_samples: usize,
    pub delay_compensated: bool,
    /// Wall time for resampling and the chain (not file IO), ms.
    pub elapsed_ms: f64,
    /// Audio duration divided by `elapsed_ms`.
    pub realtime_factor: f64,
    pub input_rms_dbfs: f32,
    pub output_rms_dbfs: f32,
    /// Output peak.
    pub peak_dbfs: f32,
    /// NaN or infinite output samples. The chain sanitises and limits, so anything but 0 is a bug.
    pub non_finite_outputs: usize,
}

/// Run mono `input` at `sample_rate` through the chain. Returns the output at `sample_rate`.
pub fn process(
    input: &[f32],
    sample_rate: u32,
    opts: &ProcessOptions,
) -> Result<(Vec<f32>, ProcessReport), String> {
    if sample_rate == 0 {
        return Err("sample rate must be above 0".into());
    }
    if opts.chunk_size == 0 {
        return Err("chunk size must be above 0".into());
    }
    // Set up like `CaptureStage::build` and its first callback.
    let mut chain = Chain::with_model(opts.model)?;
    chain.set_enabled(opts.enabled);
    chain.set_max_attenuation_db(opts.max_attenuation_db);
    chain.prime();
    let lat = chain.latency();
    let delay = (lat as f64 * sample_rate as f64 / SAMPLE_RATE as f64).round() as usize;

    let started = Instant::now();
    // Sanitize before resampling just as the live capture path does: rubato can abort on NaN/inf.
    let mut clean_in = input.to_vec();
    crate::dsp::sanitize(&mut clean_in);
    let mut buf = resample(&clean_in, sample_rate, SAMPLE_RATE)?;
    let len48 = buf.len();
    // Flush the chain's delay with silence, in whole blocks like a device would deliver.
    buf.resize((len48 + lat).next_multiple_of(opts.chunk_size), 0.0);
    for block in buf.chunks_mut(opts.chunk_size) {
        chain.process(block);
    }
    let end = len48 + lat;
    let (y48, want) = if opts.delay_compensate {
        (&buf[lat..end], input.len())
    } else {
        (&buf[..end], input.len() + delay)
    };
    let mut out = resample(y48, SAMPLE_RATE, sample_rate)?;
    out.resize(want, 0.0);
    let elapsed = started.elapsed().as_secs_f64();

    let audio_s = input.len() as f64 / sample_rate as f64;
    let report = ProcessReport {
        model: chain.model(),
        enabled: opts.enabled,
        max_attenuation_db: opts.max_attenuation_db,
        chunk_size: opts.chunk_size,
        sample_rate,
        input_samples: input.len(),
        output_samples: out.len(),
        chain_latency_samples: lat,
        total_delay_samples: delay,
        delay_compensated: opts.delay_compensate,
        elapsed_ms: elapsed * 1e3,
        realtime_factor: audio_s / elapsed.max(1e-9),
        input_rms_dbfs: rms_dbfs(&clean_in),
        output_rms_dbfs: rms_dbfs(&out),
        peak_dbfs: dbfs(out.iter().fold(0.0f32, |m, x| m.max(x.abs()))),
        non_finite_outputs: out.iter().filter(|x| !x.is_finite()).count(),
    };
    Ok((out, report))
}

/// [`process`] a WAV file into a mono WAV at the input's rate. `format` defaults to 16-bit PCM
/// for 16-bit input and 32-bit float otherwise.
pub fn process_file(
    input: &Path,
    output: &Path,
    opts: &ProcessOptions,
    format: Option<SampleFormat>,
) -> Result<ProcessReport, String> {
    let wav = Wav::read(input)?;
    let (out, report) = process(&wav.downmixed(), wav.sample_rate, opts)?;
    let format = format.unwrap_or(match wav.format {
        SampleFormat::Pcm16 => SampleFormat::Pcm16,
        _ => SampleFormat::Float32,
    });
    Wav::mono(wav.sample_rate, format, out).write(output)?;
    Ok(report)
}

/// Resamples a whole mono clip with its startup delay trimmed, so `out[i]` lines up with
/// `x[i * to / from]`. `ceil(len * to / from)` samples long. A copy when the rates match.
fn resample(x: &[f32], from: u32, to: u32) -> Result<Vec<f32>, String> {
    if from == to || x.is_empty() {
        return Ok(x.to_vec());
    }
    let mut r = Fft::<f32>::new(from as usize, to as usize, 1024, 1, FixedSync::Input)
        .map_err(|e| format!("resampler {from} -> {to} Hz: {e}"))?;
    let input = InterleavedSlice::new(x, 1, x.len()).map_err(|e| e.to_string())?;
    let out = r
        .process_all(&input, x.len(), None)
        .map_err(|e| format!("resampling {from} -> {to} Hz: {e}"))?;
    Ok(out.take_data())
}

fn dbfs(amplitude: f32) -> f32 {
    if amplitude <= 1e-6 {
        -120.0
    } else {
        20.0 * amplitude.log10()
    }
}

fn rms_dbfs(x: &[f32]) -> f32 {
    dbfs(crate::dsp::signal::rms(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::signal;

    fn rnnoise() -> ProcessOptions {
        ProcessOptions {
            model: Model::RNNoise,
            ..ProcessOptions::default()
        }
    }

    fn bypass() -> ProcessOptions {
        ProcessOptions {
            enabled: false,
            ..rnnoise()
        }
    }

    fn argmax(x: &[f32]) -> usize {
        (0..x.len())
            .max_by(|&a, &b| x[a].abs().total_cmp(&x[b].abs()))
            .unwrap()
    }

    #[test]
    fn wav_round_trips_every_format() {
        let samples: Vec<f32> = vec![0.0, 0.5, -0.5, 0.25, -1.0, 0.999, 0.125, -0.0625];
        for format in [
            SampleFormat::Pcm16,
            SampleFormat::Pcm24,
            SampleFormat::Pcm32,
            SampleFormat::Float32,
        ] {
            for channels in [1u16, 2, 4] {
                let w = Wav {
                    sample_rate: 16_000,
                    channels,
                    format,
                    samples: samples.clone(),
                };
                let back = Wav::parse(&w.encode().unwrap()).unwrap();
                assert_eq!(back.sample_rate, 16_000);
                assert_eq!((back.channels, back.format), (channels, format));
                assert_eq!(back.frames(), samples.len() / channels as usize);
                let tol = match format {
                    SampleFormat::Pcm16 => 1.0 / 16_000.0,
                    SampleFormat::Pcm24 => 1e-6,
                    _ => 1e-7,
                };
                for (a, b) in samples.iter().zip(&back.samples) {
                    assert!((a - b).abs() <= tol, "{format:?}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn integer_formats_clamp_and_zero_non_finite() {
        let w = Wav::mono(
            48_000,
            SampleFormat::Pcm16,
            vec![2.0, -2.0, f32::NAN, f32::INFINITY],
        );
        let back = Wav::parse(&w.encode().unwrap()).unwrap();
        assert_eq!(
            back.samples,
            [32_767.0 / 32_768.0, -32_767.0 / 32_768.0, 0.0, 0.0]
        );
    }

    #[test]
    fn wav_parser_handles_extensible_odd_chunks_and_short_data() {
        // WAVE_FORMAT_EXTENSIBLE, 24-bit PCM sub-format, stereo, after an odd-sized LIST chunk.
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF\0\0\0\0WAVE");
        b.extend_from_slice(b"LIST");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(b"abc\0"); // 3 bytes + pad
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&40u32.to_le_bytes());
        b.extend_from_slice(&FORMAT_EXTENSIBLE.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&44_100u32.to_le_bytes());
        b.extend_from_slice(&(44_100u32 * 6).to_le_bytes());
        b.extend_from_slice(&6u16.to_le_bytes());
        b.extend_from_slice(&24u16.to_le_bytes());
        b.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        b.extend_from_slice(&24u16.to_le_bytes()); // valid bits
        b.extend_from_slice(&3u32.to_le_bytes()); // channel mask
        b.extend_from_slice(&FORMAT_PCM.to_le_bytes());
        b.extend_from_slice(&[0; 14]); // rest of the GUID
        b.extend_from_slice(b"data");
        // Claims more than is there, like an unfinished stream; plus half a frame at the end.
        b.extend_from_slice(&u32::MAX.to_le_bytes());
        for v in [0x40_0000i32, -0x40_0000, 0x7F_FFFF, -0x80_0000] {
            b.extend_from_slice(&v.to_le_bytes()[..3]);
        }
        b.extend_from_slice(&[1, 2, 3]);
        let w = Wav::parse(&b).unwrap();
        assert_eq!((w.sample_rate, w.channels), (44_100, 2));
        assert_eq!(w.format, SampleFormat::Pcm24);
        assert_eq!(w.samples, [0.5, -0.5, 8_388_607.0 / 8_388_608.0, -1.0]);
        // Stereo averages, like a live stereo mic.
        assert_eq!(
            w.downmixed(),
            [0.0, (8_388_607.0 / 8_388_608.0 - 1.0) / 2.0]
        );
    }

    #[test]
    fn downmix_matches_the_live_path() {
        let w = Wav {
            sample_rate: 48_000,
            channels: 4,
            format: SampleFormat::Float32,
            samples: vec![0.25, 1.0, 1.0, 1.0, -0.5, 1.0, 1.0, 1.0],
        };
        // More than two channels: an interface, channel 0 at full level.
        assert_eq!(w.downmixed(), [0.25, -0.5]);
    }

    #[test]
    fn wav_parser_rejects_what_it_cannot_read() {
        assert!(Wav::parse(b"").is_err());
        assert!(Wav::parse(b"RIFF\0\0\0\0AVI ").is_err());
        assert_eq!(Wav::parse(b"RIFF\0\0\0\0WAVE").unwrap_err(), "no fmt chunk");
        let mut w = Wav::mono(8_000, SampleFormat::Pcm16, vec![0.0; 4])
            .encode()
            .unwrap();
        // 8-bit PCM.
        w[34] = 8;
        w[32] = 1;
        assert!(Wav::parse(&w).unwrap_err().contains("unsupported"));
        // A-law (tag 6).
        let mut w = Wav::mono(8_000, SampleFormat::Pcm16, vec![0.0; 4])
            .encode()
            .unwrap();
        w[20] = 6;
        assert!(Wav::parse(&w).unwrap_err().contains("tag 6"));
    }

    #[test]
    fn delay_compensation_aligns_output_with_input() {
        for rate in [48_000u32, 16_000, 44_100] {
            let n = rate as usize / 2;
            let at = n / 3;
            let mut x = vec![0.0f32; n];
            x[at] = 0.5;
            let (y, r) = process(&x, rate, &bypass()).unwrap();
            assert_eq!(
                r.chain_latency_samples, 960,
                "RNNoise: 10 ms frame + 10 ms model"
            );
            assert_eq!(y.len(), n, "{rate}");
            assert_eq!(r.output_samples, n);
            assert_eq!(argmax(&y), at, "{rate}: compensated impulse");

            let raw = ProcessOptions {
                delay_compensate: false,
                ..bypass()
            };
            let (y, r) = process(&x, rate, &raw).unwrap();
            let delay = 960 * rate as usize / 48_000;
            assert_eq!(r.total_delay_samples, delay, "{rate}");
            assert_eq!(y.len(), n + delay, "{rate}: the tail is flushed, not cut");
            assert_eq!(argmax(&y), at + delay, "{rate}: raw delay");
            assert!(!r.delay_compensated);
        }
    }

    #[test]
    fn block_size_does_not_change_the_output() {
        let x = signal::noisy_speech(24_000, 3);
        let (a, _) = process(&x, 48_000, &rnnoise()).unwrap();
        for chunk in [256, 37, 4096] {
            let opts = ProcessOptions {
                chunk_size: chunk,
                ..rnnoise()
            };
            let (b, r) = process(&x, 48_000, &opts).unwrap();
            assert_eq!(r.chunk_size, chunk);
            assert_eq!(a, b, "chunk {chunk}");
        }
        assert!(
            process(
                &x,
                48_000,
                &ProcessOptions {
                    chunk_size: 0,
                    ..rnnoise()
                }
            )
            .is_err()
        );
        assert!(process(&x, 0, &rnnoise()).is_err());
    }

    #[test]
    fn the_denoiser_runs_and_the_report_adds_up() {
        // Pink noise only: RNNoise at Strong pulls it well down, bypass leaves it.
        let x = signal::pink_noise(0.05, 48_000, 9);
        let strong = ProcessOptions {
            max_attenuation_db: f32::INFINITY,
            ..rnnoise()
        };
        let (_, on) = process(&x, 48_000, &strong).unwrap();
        let (_, off) = process(&x, 48_000, &bypass()).unwrap();
        assert!(
            on.output_rms_dbfs < off.output_rms_dbfs - 6.0,
            "on {} off {}",
            on.output_rms_dbfs,
            off.output_rms_dbfs
        );
        assert!((off.input_rms_dbfs - 20.0 * 0.05f32.log10()).abs() < 0.1);
        assert_eq!(on.model, Model::RNNoise);
        assert_eq!((on.input_samples, on.sample_rate), (48_000, 48_000));
        assert_eq!(on.non_finite_outputs, 0);
        assert!(on.peak_dbfs <= -1.0 + 1e-3, "the limiter holds");
        assert!(on.elapsed_ms > 0.0 && on.realtime_factor > 0.0);
        // Garbage in still comes out finite at both native and resampled rates.
        for rate in [48_000, 16_000] {
            let (y, r) = process(
                &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, 0.0],
                rate,
                &rnnoise(),
            )
            .unwrap();
            assert_eq!(r.non_finite_outputs, 0, "rate {rate}");
            assert!(r.input_rms_dbfs.is_finite(), "rate {rate}");
            assert_eq!(y.len(), 5, "rate {rate}");
        }
        let (y, _) = process(&[], 16_000, &rnnoise()).unwrap();
        assert!(y.is_empty());
    }

    #[test]
    fn files_process_to_mono_at_the_input_rate() {
        let dir = std::env::temp_dir().join(format!("oto-offline-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (inp, outp) = (dir.join("in.wav"), dir.join("out.wav"));
        let mono = signal::noisy_speech(8_000, 5);
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();
        Wav {
            sample_rate: 16_000,
            channels: 2,
            format: SampleFormat::Pcm16,
            samples: stereo,
        }
        .write(&inp)
        .unwrap();
        let r = process_file(&inp, &outp, &rnnoise(), None).unwrap();
        let out = Wav::read(&outp).unwrap();
        assert_eq!((out.sample_rate, out.channels), (16_000, 1));
        assert_eq!(out.format, SampleFormat::Pcm16, "16-bit in, 16-bit out");
        assert_eq!(out.frames(), 8_000);
        assert_eq!(r.output_samples, 8_000);
        process_file(&inp, &outp, &rnnoise(), Some(SampleFormat::Float32)).unwrap();
        assert_eq!(Wav::read(&outp).unwrap().format, SampleFormat::Float32);
        assert!(Wav::read(&dir.join("missing.wav")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
