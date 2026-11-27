//! `.otow` v1: the flat, 64-byte-aligned weight container Oto-1 loads (layout in `SPEC.md`).
//!
//! Parsing copies every tensor out as `f32` (f16 is widened here, once), so the model never
//! borrows the file and nothing on the processing path decodes anything.

use super::sha256::sha256;

pub const MAGIC: [u8; 4] = *b"OTOW";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: usize = 64;
pub const ENTRY_LEN: usize = 96;
pub const ALIGN: usize = 64;
pub const NAME_LEN: usize = 64;

pub const DTYPE_F16: u32 = 1;
pub const DTYPE_F32: u32 = 2;

pub const FLAG_ISOLATE: u32 = 1 << 0;
pub const FLAG_ENROLLMENT: u32 = 1 << 1;

/// IEEE-754 binary16 bits to f32, exact (every f16 is representable as f32).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal f16: renormalise into an f32 normal.
            let shift = man.leading_zeros() - 21; // bring the top set bit to bit 10
            let man = (man << shift) & 0x3ff;
            let exp = 127 - 15 + 1 - shift;
            sign | (exp << 23) | (man << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// f32 to IEEE-754 binary16 bits, round to nearest, ties to even (what `numpy.float16` and
/// `torch.half` do). Used by the writer and tests; the runtime only reads.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign; // rounds to ±0
        }
        let man = man | 0x80_0000;
        let shift = (14 - e) as u32; // 24-bit mantissa down to (10 + e) bits
        let half = 1u32 << (shift - 1);
        let rest = man & ((1 << shift) - 1);
        let mut v = man >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1; // may carry into the exponent, which is still correct (up to inf)
    }
    sign | v as u16
}

/// One tensor, widened to f32.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: u32,
    pub data: Vec<f32>,
}

/// A parsed, checksum-verified `.otow` file.
#[derive(Clone, Debug)]
pub struct Otow {
    pub arch_id: u32,
    pub dtype: u32,
    pub flags: u32,
    pub config: String,
    pub tensors: Vec<Tensor>,
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

impl Otow {
    /// Parse and verify. Every structural rule in `SPEC.md` is checked: magic, version,
    /// alignment, bounds, SHA-256, dtype, rank, unique names, and payloads that neither overlap
    /// the header/config/directory nor each other.
    pub fn parse(b: &[u8]) -> Result<Self, String> {
        if b.len() < HEADER_LEN {
            return Err(format!(
                ".otow: {} bytes is shorter than the header",
                b.len()
            ));
        }
        if b[0..4] != MAGIC {
            return Err(".otow: bad magic (expected OTOW)".into());
        }
        let version = u32_at(b, 4);
        if version != VERSION {
            return Err(format!(
                ".otow: version {version}, this build reads {VERSION}"
            ));
        }
        let arch_id = u32_at(b, 8);
        let dtype = u32_at(b, 12);
        let flags = u32_at(b, 16);
        let config_len = u32_at(b, 20) as usize;
        let num_tensors = u32_at(b, 24) as usize;
        let table_offset = u32_at(b, 28) as usize;
        if dtype != DTYPE_F16 && dtype != DTYPE_F32 {
            return Err(format!(".otow: unknown file dtype {dtype}"));
        }
        if flags & !(FLAG_ISOLATE | FLAG_ENROLLMENT) != 0 {
            return Err(format!(".otow: unknown flags {flags:#x}"));
        }
        let digest = sha256(&b[HEADER_LEN..]);
        if digest[..] != b[32..64] {
            return Err(".otow: SHA-256 mismatch (file corrupt or truncated)".into());
        }
        let config_end = HEADER_LEN
            .checked_add(config_len)
            .ok_or(".otow: config length overflows")?;
        if !table_offset.is_multiple_of(ALIGN)
            || table_offset < config_end
            || table_offset > b.len()
        {
            return Err(format!(".otow: bad table offset {table_offset}"));
        }
        let table_end = num_tensors
            .checked_mul(ENTRY_LEN)
            .and_then(|n| n.checked_add(table_offset))
            .filter(|&e| e <= b.len())
            .ok_or(".otow: tensor directory runs past the end of the file")?;
        let config = core::str::from_utf8(&b[HEADER_LEN..config_end])
            .map_err(|_| ".otow: config is not UTF-8")?
            .to_owned();
        if b[config_end..table_offset].iter().any(|&x| x != 0) {
            return Err(".otow: config padding is not zero".into());
        }

        let mut tensors: Vec<Tensor> = Vec::with_capacity(num_tensors);
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(num_tensors);
        for t in 0..num_tensors {
            let e = &b[table_offset + t * ENTRY_LEN..table_offset + (t + 1) * ENTRY_LEN];
            let name_end = e[..NAME_LEN]
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(NAME_LEN);
            if name_end == 0 || e[name_end..NAME_LEN].iter().any(|&c| c != 0) {
                return Err(format!(".otow: tensor {t}: empty or badly padded name"));
            }
            let name = core::str::from_utf8(&e[..name_end])
                .map_err(|_| format!(".otow: tensor {t}: name is not UTF-8"))?
                .to_owned();
            let ndim = u32_at(e, 64) as usize;
            if !(1..=4).contains(&ndim) {
                return Err(format!(".otow: {name}: ndim {ndim} not in 1..=4"));
            }
            let dims: Vec<usize> = (0..4).map(|i| u32_at(e, 68 + 4 * i) as usize).collect();
            if dims[..ndim].contains(&0) || dims[ndim..].iter().any(|&d| d != 0) {
                return Err(format!(".otow: {name}: bad shape {dims:?} for ndim {ndim}"));
            }
            let shape = dims[..ndim].to_vec();
            let tdtype = u32_at(e, 84);
            let width = match tdtype {
                DTYPE_F16 => 2,
                DTYPE_F32 => 4,
                _ => return Err(format!(".otow: {name}: unknown dtype {tdtype}")),
            };
            let offset = usize::try_from(u64_at(e, 88))
                .map_err(|_| format!(".otow: {name}: offset out of range"))?;
            let count = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
            let bytes = count.and_then(|c| c.checked_mul(width));
            let end = bytes
                .and_then(|n| n.checked_add(offset))
                .filter(|&e| e <= b.len());
            let (Some(count), Some(end)) = (count, end) else {
                return Err(format!(
                    ".otow: {name}: payload runs past the end of the file"
                ));
            };
            if !offset.is_multiple_of(ALIGN) || offset < table_end {
                return Err(format!(
                    ".otow: {name}: payload offset {offset} misaligned or inside the directory"
                ));
            }
            if tensors.iter().any(|x| x.name == name) {
                return Err(format!(".otow: duplicate tensor {name}"));
            }
            if spans.iter().any(|&(s, e)| offset < e && s < end) {
                return Err(format!(".otow: {name}: payload overlaps another tensor"));
            }
            spans.push((offset, end));
            let raw = &b[offset..end];
            let data: Vec<f32> = if tdtype == DTYPE_F16 {
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            } else {
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            debug_assert_eq!(data.len(), count);
            if data.iter().any(|v| !v.is_finite()) {
                return Err(format!(".otow: {name}: non-finite weight"));
            }
            tensors.push(Tensor {
                name,
                shape,
                dtype: tdtype,
                data,
            });
        }
        Ok(Self {
            arch_id,
            dtype,
            flags,
            config,
            tensors,
        })
    }

    /// Removes and returns the tensor called `name`, checking its shape.
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let i = self
            .tensors
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| format!(".otow: missing tensor {name}"))?;
        let t = self.tensors.swap_remove(i);
        if t.shape != shape {
            return Err(format!(
                ".otow: {name}: shape {:?}, expected {shape:?}",
                t.shape
            ));
        }
        Ok(t.data)
    }

    /// An unsigned integer from the canonical JSON config (`"key":123`). The config is flat and
    /// written by our own exporter with `sort_keys=True, separators=(",", ":")`, so a scan is
    /// enough and keeps a JSON parser out of the wasm build.
    pub fn config_u32(&self, key: &str) -> Option<u32> {
        let pat = format!("\"{key}\":");
        let at = self.config.find(&pat)? + pat.len();
        let digits: &str = &self.config[at..];
        let end = digits
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(digits.len());
        digits[..end].parse().ok()
    }
}

fn pad_to(out: &mut Vec<u8>, align: usize) {
    while !out.len().is_multiple_of(align) {
        out.push(0);
    }
}

/// Builds `.otow` files. Used by tests, the random benchmark models and as an executable
/// reference for the PyTorch exporter.
pub struct OtowWriter {
    arch_id: u32,
    dtype: u32,
    flags: u32,
    config: String,
    tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl OtowWriter {
    pub fn new(arch_id: u32, dtype: u32, flags: u32, config: &str) -> Self {
        assert!(dtype == DTYPE_F16 || dtype == DTYPE_F32);
        Self {
            arch_id,
            dtype,
            flags,
            config: config.to_owned(),
            tensors: Vec::new(),
        }
    }

    pub fn add(&mut self, name: &str, shape: &[usize], data: Vec<f32>) {
        assert!(name.len() < NAME_LEN && !name.is_empty());
        assert!((1..=4).contains(&shape.len()));
        assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}");
        self.tensors.push((name.to_owned(), shape.to_vec(), data));
    }

    pub fn finish(&self) -> Vec<u8> {
        let mut out = vec![0u8; HEADER_LEN];
        out.extend_from_slice(self.config.as_bytes());
        pad_to(&mut out, ALIGN);
        let table_offset = out.len();
        out.resize(table_offset + self.tensors.len() * ENTRY_LEN, 0);
        pad_to(&mut out, ALIGN);
        for (i, (name, shape, data)) in self.tensors.iter().enumerate() {
            let offset = out.len();
            match self.dtype {
                DTYPE_F16 => data
                    .iter()
                    .for_each(|&v| out.extend_from_slice(&f32_to_f16(v).to_le_bytes())),
                _ => data
                    .iter()
                    .for_each(|&v| out.extend_from_slice(&v.to_le_bytes())),
            }
            pad_to(&mut out, ALIGN);
            let e = &mut out[table_offset + i * ENTRY_LEN..table_offset + (i + 1) * ENTRY_LEN];
            e[..name.len()].copy_from_slice(name.as_bytes());
            e[64..68].copy_from_slice(&(shape.len() as u32).to_le_bytes());
            for (j, &d) in shape.iter().enumerate() {
                e[68 + 4 * j..72 + 4 * j].copy_from_slice(&(d as u32).to_le_bytes());
            }
            e[84..88].copy_from_slice(&self.dtype.to_le_bytes());
            e[88..96].copy_from_slice(&(offset as u64).to_le_bytes());
        }
        out[0..4].copy_from_slice(&MAGIC);
        out[4..8].copy_from_slice(&VERSION.to_le_bytes());
        out[8..12].copy_from_slice(&self.arch_id.to_le_bytes());
        out[12..16].copy_from_slice(&self.dtype.to_le_bytes());
        out[16..20].copy_from_slice(&self.flags.to_le_bytes());
        out[20..24].copy_from_slice(&(self.config.len() as u32).to_le_bytes());
        out[24..28].copy_from_slice(&(self.tensors.len() as u32).to_le_bytes());
        out[28..32].copy_from_slice(&(table_offset as u32).to_le_bytes());
        let digest = sha256(&out[HEADER_LEN..]);
        out[32..64].copy_from_slice(&digest);
        out
    }
}

/// Deterministic xorshift64* generator for random benchmark weights (no `rand` dependency, so
/// the wasm build can build random models too).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-1, 1).
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_every_finite_value() {
        for h in 0u16..=0xffff {
            let x = f16_to_f32(h);
            if x.is_nan() {
                assert!(f16_to_f32(f32_to_f16(x)).is_nan());
                continue;
            }
            assert_eq!(f32_to_f16(x), h, "h={h:#06x} x={x}");
        }
    }

    #[test]
    fn f16_rounds_to_nearest_even() {
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // rounds up to inf
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00); // tie, even stays
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02); // tie, odd rounds up
        assert_eq!(f32_to_f16(5.960_464_5e-8), 0x0001); // smallest subnormal
        assert_eq!(f32_to_f16(2.98e-8), 0x0000); // below half of it
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0xc000), -2.0);
    }

    fn sample() -> Vec<u8> {
        let mut w = OtowWriter::new(2, DTYPE_F16, 0, r#"{"arch":"x","channels":32}"#);
        w.add("a.weight", &[3, 2], vec![0.5, -1.0, 2.0, 0.25, 0.0, 3.0]);
        w.add("a.bias", &[3], vec![1.0, 2.0, 3.0]);
        w.finish()
    }

    #[test]
    fn writer_and_parser_agree() {
        let b = sample();
        assert_eq!(b.len() % ALIGN, 0);
        let mut o = Otow::parse(&b).unwrap();
        assert_eq!(o.arch_id, 2);
        assert_eq!(o.config_u32("channels"), Some(32));
        assert_eq!(o.config_u32("missing"), None);
        assert_eq!(o.take("a.bias", &[3]).unwrap(), vec![1.0, 2.0, 3.0]);
        assert!(o.take("a.weight", &[2, 3]).is_err());
    }

    #[test]
    fn corruption_is_rejected() {
        let good = sample();
        for at in [0usize, 4, 70, good.len() - 1] {
            let mut b = good.clone();
            b[at] ^= 1;
            assert!(Otow::parse(&b).is_err(), "flip at {at} accepted");
        }
        // Crafted 64-byte header with config_len = u32::MAX and matching SHA-256 must return Err,
        // never wrap HEADER_LEN + config_len on 32-bit targets (wasm32).
        let mut crafted = vec![0u8; HEADER_LEN];
        crafted[0..4].copy_from_slice(&MAGIC);
        crafted[4..8].copy_from_slice(&VERSION.to_le_bytes());
        crafted[8..12].copy_from_slice(&1u32.to_le_bytes());
        crafted[12..16].copy_from_slice(&DTYPE_F16.to_le_bytes());
        crafted[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
        crafted[28..32].copy_from_slice(&(HEADER_LEN as u32).to_le_bytes());
        let d = sha256(&crafted[HEADER_LEN..]);
        crafted[32..64].copy_from_slice(&d);
        assert!(Otow::parse(&crafted).is_err());
        assert!(Otow::parse(&good[..good.len() - 64]).is_err());
    }
}
