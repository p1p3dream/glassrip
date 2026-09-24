//! Baseline JPEG decoder that reproduces libjpeg-turbo's default decompression output.
//!
//! OpenCV's `imread` decodes through libjpeg-turbo with its defaults: the accurate integer
//! IDCT (`JDCT_ISLOW`), fancy (triangle) chroma upsampling, and the table-driven YCbCr to RGB
//! conversion in `jdcolor.c`. Generic Rust decoders use different IDCTs and upsamplers and
//! differ by a few levels on a noticeable share of pixels, so this module ports those three
//! pieces exactly. Only sequential Huffman (SOF0/SOF1) 8-bit files with 1 or 3 components
//! and h1v1, h2v1 or h2v2 chroma sampling are handled; anything else reports
//! [`Unsupported`] so the caller can fall back to a general decoder.

use std::fmt;

/// The file uses a JPEG feature this decoder does not implement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported(pub String);

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unsupported JPEG: {}", self.0)
    }
}

/// Decoder failure: either malformed data or an unsupported feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JpegError {
    /// The stream is malformed.
    Corrupt(String),
    /// A valid stream that uses a feature outside this decoder's scope.
    Unsupported(Unsupported),
}

impl fmt::Display for JpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JpegError::Corrupt(m) => write!(f, "corrupt JPEG: {m}"),
            JpegError::Unsupported(u) => u.fmt(f),
        }
    }
}

fn corrupt<T>(msg: &str) -> Result<T, JpegError> {
    Err(JpegError::Corrupt(msg.to_string()))
}

fn unsupported<T>(msg: &str) -> Result<T, JpegError> {
    Err(JpegError::Unsupported(Unsupported(msg.to_string())))
}

/// Natural-order index for each zigzag position.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[derive(Clone)]
struct Huffman {
    /// (code length, symbol) for codes up to `LOOKAHEAD` bits, indexed by the next bits.
    fast: Vec<(u8, u8)>,
    maxcode: [i32; 18],
    valoffset: [i32; 18],
    values: Vec<u8>,
}

const LOOKAHEAD: u32 = 9;

impl Huffman {
    fn new(counts: &[u8; 16], values: Vec<u8>) -> Result<Self, JpegError> {
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if total != values.len() || total > 256 {
            return corrupt("bad Huffman table");
        }
        let mut maxcode = [-1i32; 18];
        let mut valoffset = [0i32; 18];
        let mut fast = vec![(0u8, 0u8); 1 << LOOKAHEAD];
        let mut code: i32 = 0;
        let mut k: i32 = 0;
        for len in 1..=16usize {
            let n = i32::from(counts[len - 1]);
            if n > 0 {
                valoffset[len] = k - code;
                for i in 0..n {
                    let c = code + i;
                    if len as u32 <= LOOKAHEAD {
                        let shift = LOOKAHEAD - len as u32;
                        let base = (c as usize) << shift;
                        let sym = values[(k + i) as usize];
                        for slot in fast.iter_mut().skip(base).take(1 << shift) {
                            *slot = (len as u8, sym);
                        }
                    }
                }
                code += n;
                k += n;
                maxcode[len] = code - 1;
            }
            code <<= 1;
        }
        maxcode[17] = i32::MAX;
        Ok(Self {
            fast,
            maxcode,
            valoffset,
            values,
        })
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
    /// Set once a marker is reached; further reads return zero bits, as libjpeg does.
    hit_marker: bool,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            acc: 0,
            nbits: 0,
            hit_marker: false,
        }
    }

    #[inline]
    fn fill(&mut self) {
        while self.nbits <= 56 {
            let mut byte = 0u8;
            if !self.hit_marker && self.pos < self.data.len() {
                let b = self.data[self.pos];
                if b == 0xFF {
                    let next = self.data.get(self.pos + 1).copied().unwrap_or(0xD9);
                    if next == 0x00 {
                        byte = 0xFF;
                        self.pos += 2;
                    } else {
                        self.hit_marker = true;
                    }
                } else {
                    byte = b;
                    self.pos += 1;
                }
            }
            self.acc |= u64::from(byte) << (56 - self.nbits);
            self.nbits += 8;
        }
    }

    #[inline]
    fn peek(&mut self, n: u32) -> u32 {
        if self.nbits < n {
            self.fill();
        }
        (self.acc >> (64 - n)) as u32
    }

    #[inline]
    fn consume(&mut self, n: u32) {
        self.acc <<= n;
        self.nbits -= n;
    }

    #[inline]
    fn bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let v = self.peek(n);
        self.consume(n);
        v
    }

    #[inline]
    fn decode(&mut self, h: &Huffman) -> Result<u8, JpegError> {
        let look = self.peek(LOOKAHEAD) as usize;
        let (len, sym) = h.fast[look];
        if len > 0 {
            self.consume(u32::from(len));
            return Ok(sym);
        }
        let mut len = LOOKAHEAD as usize + 1;
        let mut code = self.peek(len as u32) as i32;
        while code > h.maxcode[len] {
            len += 1;
            if len > 16 {
                return corrupt("bad Huffman code");
            }
            code = self.peek(len as u32) as i32;
        }
        self.consume(len as u32);
        let idx = (code + h.valoffset[len]) as usize;
        h.values
            .get(idx)
            .copied()
            .ok_or_else(|| JpegError::Corrupt("Huffman index out of range".into()))
    }

    #[inline]
    fn receive_extend(&mut self, s: u32) -> i32 {
        if s == 0 {
            return 0;
        }
        let v = self.bits(s) as i32;
        if v < (1 << (s - 1)) {
            v - (1 << s) + 1
        } else {
            v
        }
    }

    /// Discards buffered bits and consumes an RSTn marker if one is next.
    fn restart(&mut self) {
        self.acc = 0;
        self.nbits = 0;
        self.hit_marker = false;
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF {
                let m = self.data[self.pos + 1];
                if (0xD0..=0xD7).contains(&m) {
                    self.pos += 2;
                    return;
                }
                if m == 0xFF {
                    self.pos += 1;
                    continue;
                }
                return;
            }
            self.pos += 1;
        }
    }
}

#[derive(Clone)]
struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    td: usize,
    ta: usize,
    /// Width and height of the coded plane, a multiple of 8.
    pw: usize,
    ph: usize,
    /// `ceil(image_width * h / hmax)` as libjpeg computes `downsampled_width`.
    dw: usize,
    dh: usize,
    plane: Vec<u8>,
}

/// A decoded JPEG before color conversion: component planes at their coded resolution.
pub struct Decoded {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    comps: Vec<Component>,
    hmax: usize,
    vmax: usize,
}

// ISLOW IDCT constants (jidctint.c).
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;
const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

#[inline]
fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// libjpeg's post-IDCT range limit: `range_limit[x & 1023]` with `range_limit` centred on 128.
#[inline]
fn idct_limit(x: i32) -> u8 {
    let i = x & 1023;
    if i < 128 {
        (i + 128) as u8
    } else if i < 512 {
        255
    } else if i < 896 {
        0
    } else {
        (i - 896) as u8
    }
}

/// Port of libjpeg-turbo `jpeg_idct_islow`. `coef` is in natural order, already multiplied
/// by the quantization table.
fn idct_islow(coef: &[i32; 64], out: &mut [u8], stride: usize) {
    let mut ws = [0i32; 64];
    for col in 0..8 {
        let c = |r: usize| coef[r * 8 + col];
        if (1..8).all(|r| c(r) == 0) {
            let dc = c(0) << PASS1_BITS;
            for r in 0..8 {
                ws[r * 8 + col] = dc;
            }
            continue;
        }
        let (z2, z3) = (c(2), c(6));
        let z1 = (z2 + z3) * FIX_0_541196100;
        let tmp2 = z1 + z3 * -FIX_1_847759065;
        let tmp3 = z1 + z2 * FIX_0_765366865;
        let (z2, z3) = (c(0), c(4));
        let tmp0 = (z2 + z3) << CONST_BITS;
        let tmp1 = (z2 - z3) << CONST_BITS;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        let (mut t0, mut t1, mut t2, mut t3) = (c(7), c(5), c(3), c(1));
        let z1 = t0 + t3;
        let z2 = t1 + t2;
        let z3 = t0 + t2;
        let z4 = t1 + t3;
        let z5 = (z3 + z4) * FIX_1_175875602;
        t0 *= FIX_0_298631336;
        t1 *= FIX_2_053119869;
        t2 *= FIX_3_072711026;
        t3 *= FIX_1_501321110;
        let z1 = z1 * -FIX_0_899976223;
        let z2 = z2 * -FIX_2_562915447;
        let z3 = z3 * -FIX_1_961570560 + z5;
        let z4 = z4 * -FIX_0_390180644 + z5;
        t0 += z1 + z3;
        t1 += z2 + z4;
        t2 += z2 + z3;
        t3 += z1 + z4;
        let sh = CONST_BITS - PASS1_BITS;
        ws[col] = descale(tmp10 + t3, sh);
        ws[7 * 8 + col] = descale(tmp10 - t3, sh);
        ws[8 + col] = descale(tmp11 + t2, sh);
        ws[6 * 8 + col] = descale(tmp11 - t2, sh);
        ws[2 * 8 + col] = descale(tmp12 + t1, sh);
        ws[5 * 8 + col] = descale(tmp12 - t1, sh);
        ws[3 * 8 + col] = descale(tmp13 + t0, sh);
        ws[4 * 8 + col] = descale(tmp13 - t0, sh);
    }
    let sh = CONST_BITS + PASS1_BITS + 3;
    for row in 0..8 {
        let w = &ws[row * 8..row * 8 + 8];
        let o = &mut out[row * stride..row * stride + 8];
        let (z2, z3) = (w[2], w[6]);
        let z1 = (z2 + z3) * FIX_0_541196100;
        let tmp2 = z1 + z3 * -FIX_1_847759065;
        let tmp3 = z1 + z2 * FIX_0_765366865;
        let tmp0 = (w[0] + w[4]) << CONST_BITS;
        let tmp1 = (w[0] - w[4]) << CONST_BITS;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        let (mut t0, mut t1, mut t2, mut t3) = (w[7], w[5], w[3], w[1]);
        let z1 = t0 + t3;
        let z2 = t1 + t2;
        let z3 = t0 + t2;
        let z4 = t1 + t3;
        let z5 = (z3 + z4) * FIX_1_175875602;
        t0 *= FIX_0_298631336;
        t1 *= FIX_2_053119869;
        t2 *= FIX_3_072711026;
        t3 *= FIX_1_501321110;
        let z1 = z1 * -FIX_0_899976223;
        let z2 = z2 * -FIX_2_562915447;
        let z3 = z3 * -FIX_1_961570560 + z5;
        let z4 = z4 * -FIX_0_390180644 + z5;
        t0 += z1 + z3;
        t1 += z2 + z4;
        t2 += z2 + z3;
        t3 += z1 + z4;
        o[0] = idct_limit(descale(tmp10 + t3, sh));
        o[7] = idct_limit(descale(tmp10 - t3, sh));
        o[1] = idct_limit(descale(tmp11 + t2, sh));
        o[6] = idct_limit(descale(tmp11 - t2, sh));
        o[2] = idct_limit(descale(tmp12 + t1, sh));
        o[5] = idct_limit(descale(tmp12 - t1, sh));
        o[3] = idct_limit(descale(tmp13 + t0, sh));
        o[4] = idct_limit(descale(tmp13 - t0, sh));
    }
}

fn be16(d: &[u8], p: usize) -> Result<usize, JpegError> {
    match (d.get(p), d.get(p + 1)) {
        (Some(&a), Some(&b)) => Ok((usize::from(a) << 8) | usize::from(b)),
        _ => corrupt("truncated segment"),
    }
}

/// Decodes a baseline JPEG into component planes.
pub fn decode(data: &[u8]) -> Result<Decoded, JpegError> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return corrupt("missing SOI");
    }
    let mut pos = 2;
    let mut qt: [Option<[i32; 64]>; 4] = [None, None, None, None];
    let mut dc: [Option<Huffman>; 4] = [None, None, None, None];
    let mut ac: [Option<Huffman>; 4] = [None, None, None, None];
    let mut frame: Option<(usize, usize, Vec<Component>)> = None;
    let mut restart_interval = 0usize;
    let mut scanned = false;

    loop {
        while pos < data.len() && data[pos] != 0xFF {
            pos += 1;
        }
        while pos < data.len() && data[pos] == 0xFF {
            pos += 1;
        }
        let Some(&marker) = data.get(pos) else {
            break;
        };
        pos += 1;
        match marker {
            0xD8 | 0x01 | 0xD0..=0xD7 => continue,
            0xD9 => break,
            _ => {}
        }
        let len = be16(data, pos)?;
        if len < 2 || pos + len > data.len() {
            return corrupt("bad segment length");
        }
        let seg = &data[pos + 2..pos + len];
        let seg_end = pos + len;
        match marker {
            0xDB => {
                let mut p = 0;
                while p < seg.len() {
                    let pq = seg[p] >> 4;
                    let tq = usize::from(seg[p] & 15);
                    p += 1;
                    if tq > 3 {
                        return corrupt("bad DQT id");
                    }
                    let mut t = [0i32; 64];
                    for &zz in ZIGZAG.iter() {
                        let v = if pq == 0 {
                            let v = *seg.get(p).ok_or(JpegError::Corrupt("short DQT".into()))?;
                            p += 1;
                            i32::from(v)
                        } else {
                            let v = be16(seg, p)? as i32;
                            p += 2;
                            v
                        };
                        t[zz] = v;
                    }
                    qt[tq] = Some(t);
                }
            }
            0xC4 => {
                let mut p = 0;
                while p < seg.len() {
                    let tc = seg[p] >> 4;
                    let th = usize::from(seg[p] & 15);
                    p += 1;
                    if th > 3 || tc > 1 || p + 16 > seg.len() {
                        return corrupt("bad DHT");
                    }
                    let mut counts = [0u8; 16];
                    counts.copy_from_slice(&seg[p..p + 16]);
                    p += 16;
                    let total: usize = counts.iter().map(|&c| usize::from(c)).sum();
                    if p + total > seg.len() {
                        return corrupt("short DHT");
                    }
                    let table = Huffman::new(&counts, seg[p..p + total].to_vec())?;
                    p += total;
                    if tc == 0 {
                        dc[th] = Some(table);
                    } else {
                        ac[th] = Some(table);
                    }
                }
            }
            0xDD => {
                restart_interval = be16(seg, 0)?;
            }
            0xC0 | 0xC1 => {
                if seg.len() < 6 || seg[0] != 8 {
                    return unsupported("only 8-bit precision");
                }
                let height = be16(seg, 1)?;
                let width = be16(seg, 3)?;
                let nf = usize::from(seg[5]);
                if width == 0 || height == 0 || !(nf == 1 || nf == 3) || seg.len() < 6 + 3 * nf {
                    return unsupported("component count or size");
                }
                let mut comps = Vec::with_capacity(nf);
                for i in 0..nf {
                    let b = &seg[6 + 3 * i..9 + 3 * i];
                    comps.push(Component {
                        id: b[0],
                        h: usize::from(b[1] >> 4),
                        v: usize::from(b[1] & 15),
                        tq: usize::from(b[2]),
                        td: 0,
                        ta: 0,
                        pw: 0,
                        ph: 0,
                        dw: 0,
                        dh: 0,
                        plane: Vec::new(),
                    });
                }
                frame = Some((width, height, comps));
            }
            0xC2 | 0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return unsupported("non-baseline frame type");
            }
            0xDA => {
                let Some((width, height, comps)) = frame.as_mut() else {
                    return corrupt("SOS before SOF");
                };
                if scanned {
                    return unsupported("multiple scans");
                }
                let ns = usize::from(*seg.first().ok_or(JpegError::Corrupt("empty SOS".into()))?);
                if ns != comps.len() || seg.len() < 1 + 2 * ns + 3 {
                    return unsupported("non-interleaved scan");
                }
                let mut order = Vec::with_capacity(ns);
                for i in 0..ns {
                    let id = seg[1 + 2 * i];
                    let t = seg[2 + 2 * i];
                    let ci = comps
                        .iter()
                        .position(|c| c.id == id)
                        .ok_or(JpegError::Corrupt("unknown component in SOS".into()))?;
                    comps[ci].td = usize::from(t >> 4);
                    comps[ci].ta = usize::from(t & 15);
                    order.push(ci);
                }
                pos = seg_end;
                pos = decode_scan(
                    data,
                    pos,
                    *width,
                    *height,
                    comps,
                    &order,
                    &qt,
                    &dc,
                    &ac,
                    restart_interval,
                )?;
                scanned = true;
                continue;
            }
            _ => {}
        }
        pos = seg_end;
    }

    let Some((width, height, comps)) = frame else {
        return corrupt("no frame");
    };
    if !scanned {
        return corrupt("no scan");
    }
    let hmax = comps.iter().map(|c| c.h).max().unwrap_or(1);
    let vmax = comps.iter().map(|c| c.v).max().unwrap_or(1);
    Ok(Decoded {
        width,
        height,
        comps,
        hmax,
        vmax,
    })
}

#[allow(clippy::too_many_arguments)]
fn decode_scan(
    data: &[u8],
    pos: usize,
    width: usize,
    height: usize,
    comps: &mut [Component],
    order: &[usize],
    qt: &[Option<[i32; 64]>; 4],
    dc: &[Option<Huffman>; 4],
    ac: &[Option<Huffman>; 4],
    restart_interval: usize,
) -> Result<usize, JpegError> {
    let hmax = comps.iter().map(|c| c.h).max().unwrap_or(1);
    let vmax = comps.iter().map(|c| c.v).max().unwrap_or(1);
    if comps
        .iter()
        .any(|c| c.h == 0 || c.v == 0 || c.h > 2 || c.v > 2)
    {
        return unsupported("sampling factors");
    }
    let mcux = width.div_ceil(8 * hmax);
    let mcuy = height.div_ceil(8 * vmax);
    for c in comps.iter_mut() {
        c.pw = mcux * c.h * 8;
        c.ph = mcuy * c.v * 8;
        c.dw = (width * c.h).div_ceil(hmax);
        c.dh = (height * c.v).div_ceil(vmax);
        c.plane = vec![0u8; c.pw * c.ph];
        if c.tq > 3 || qt[c.tq].is_none() {
            return corrupt("missing quantization table");
        }
        if c.td > 3 || c.ta > 3 || dc[c.td].is_none() || ac[c.ta].is_none() {
            return corrupt("missing Huffman table");
        }
    }
    let single = comps.len() == 1;
    // A single-component scan is non-interleaved: one block per MCU over the component's
    // own block grid (which for a 1-component frame is simply ceil(w/8) x ceil(h/8)).
    let (mx, my) = if single {
        (width.div_ceil(8), height.div_ceil(8))
    } else {
        (mcux, mcuy)
    };
    let mut br = BitReader::new(data, pos);
    let mut pred = vec![0i32; comps.len()];
    let mut coef = [0i32; 64];
    let mut mcus_left = restart_interval;
    for my_i in 0..my {
        for mx_i in 0..mx {
            if restart_interval > 0 {
                if mcus_left == 0 {
                    br.restart();
                    pred.iter_mut().for_each(|p| *p = 0);
                    mcus_left = restart_interval;
                }
                mcus_left -= 1;
            }
            for &ci in order {
                let (bh, bv) = if single {
                    (1, 1)
                } else {
                    (comps[ci].h, comps[ci].v)
                };
                for v in 0..bv {
                    for h in 0..bh {
                        let c = &comps[ci];
                        let (Some(q), Some(dct), Some(act)) = (&qt[c.tq], &dc[c.td], &ac[c.ta])
                        else {
                            return corrupt("missing table");
                        };
                        coef.iter_mut().for_each(|x| *x = 0);
                        let t = br.decode(dct)?;
                        if t > 15 {
                            return corrupt("bad DC magnitude");
                        }
                        pred[ci] += br.receive_extend(u32::from(t));
                        coef[0] = pred[ci] * q[0];
                        let mut k = 1usize;
                        while k < 64 {
                            let rs = br.decode(act)?;
                            let r = usize::from(rs >> 4);
                            let s = u32::from(rs & 15);
                            if s == 0 {
                                if r == 15 {
                                    k += 16;
                                    continue;
                                }
                                break;
                            }
                            k += r;
                            if k > 63 {
                                return corrupt("AC index overflow");
                            }
                            let z = ZIGZAG[k];
                            coef[z] = br.receive_extend(s) * q[z];
                            k += 1;
                        }
                        let bx = mx_i * bh + h;
                        let by = my_i * bv + v;
                        let c = &mut comps[ci];
                        if bx * 8 < c.pw && by * 8 < c.ph {
                            let off = by * 8 * c.pw + bx * 8;
                            let pw = c.pw;
                            idct_islow(&coef, &mut c.plane[off..], pw);
                        }
                    }
                }
            }
        }
    }
    // Continue after the entropy-coded segment.
    let mut p = br.pos;
    while p + 1 < data.len()
        && !(data[p] == 0xFF && data[p + 1] != 0x00 && !(0xD0..=0xD7).contains(&data[p + 1]))
    {
        p += 1;
    }
    Ok(p)
}

/// libjpeg `sample_range_limit` for color conversion: clamps to 0..=255.
#[inline]
fn clamp_u8(x: i32) -> u8 {
    x.clamp(0, 255) as u8
}

struct YccTables {
    cr_r: [i32; 256],
    cb_b: [i32; 256],
    cr_g: [i32; 256],
    cb_g: [i32; 256],
}

impl YccTables {
    /// `build_ycc_rgb_table` from libjpeg-turbo `jdcolor.c`.
    fn new() -> Self {
        const SCALEBITS: i32 = 16;
        const ONE_HALF: i64 = 1 << (SCALEBITS - 1);
        let fix = |x: f64| (x * f64::from(1 << SCALEBITS) + 0.5) as i64;
        let mut t = Self {
            cr_r: [0; 256],
            cb_b: [0; 256],
            cr_g: [0; 256],
            cb_g: [0; 256],
        };
        for i in 0..256usize {
            let x = i as i64 - 128;
            t.cr_r[i] = ((fix(1.40200) * x + ONE_HALF) >> SCALEBITS) as i32;
            t.cb_b[i] = ((fix(1.77200) * x + ONE_HALF) >> SCALEBITS) as i32;
            t.cr_g[i] = (-fix(0.71414) * x) as i32;
            t.cb_g[i] = (-fix(0.34414) * x + ONE_HALF) as i32;
        }
        t
    }
}

/// Upsamples one chroma component to full resolution rows `0..height` using
/// libjpeg-turbo's fancy upsampling. Returns a `width x height` plane.
///
/// The index loops mirror `jdsample.c` so each output sample's formula is easy to audit.
#[allow(clippy::needless_range_loop)]
fn upsample(c: &Component, hmax: usize, vmax: usize, width: usize, height: usize) -> Vec<u8> {
    let hf = hmax / c.h;
    let vf = vmax / c.v;
    let src = |x: usize, y: usize| i32::from(c.plane[y * c.pw + x]);
    let ow = c.dw * hf;
    let mut out = vec![0u8; width * height];
    let dw = c.dw;
    let last_row = c.dh.saturating_sub(1);
    let mut row = vec![0u8; ow.max(width)];
    for y in 0..height {
        match (hf, vf) {
            (1, 1) => {
                for x in 0..dw.min(width) {
                    row[x] = src(x, y) as u8;
                }
            }
            (2, 1) => {
                // h2v1_fancy_upsample
                if dw == 1 {
                    row[0] = src(0, y) as u8;
                    row[1] = row[0];
                } else {
                    let v0 = src(0, y);
                    row[0] = v0 as u8;
                    row[1] = ((v0 * 3 + src(1, y) + 2) >> 2) as u8;
                    for i in 1..dw - 1 {
                        let v = src(i, y) * 3;
                        row[2 * i] = ((v + src(i - 1, y) + 1) >> 2) as u8;
                        row[2 * i + 1] = ((v + src(i + 1, y) + 2) >> 2) as u8;
                    }
                    let vl = src(dw - 1, y);
                    row[2 * dw - 2] = ((vl * 3 + src(dw - 2, y) + 1) >> 2) as u8;
                    row[2 * dw - 1] = vl as u8;
                }
            }
            (2, 2) => {
                // h2v2_fancy_upsample: nearest input row plus the row above (even output
                // rows) or below (odd output rows), edge rows duplicated.
                let inrow = y / 2;
                let near = inrow.min(last_row);
                let far = if y % 2 == 0 {
                    near.saturating_sub(1)
                } else {
                    (near + 1).min(last_row)
                };
                let colsum = |x: usize| src(x, near) * 3 + src(x, far);
                if dw == 1 {
                    let t = colsum(0);
                    row[0] = ((t * 4 + 8) >> 4) as u8;
                    row[1] = ((t * 4 + 7) >> 4) as u8;
                } else {
                    let mut this = colsum(0);
                    let mut next = colsum(1);
                    row[0] = ((this * 4 + 8) >> 4) as u8;
                    row[1] = ((this * 3 + next + 7) >> 4) as u8;
                    let mut last = this;
                    this = next;
                    for i in 1..dw - 1 {
                        next = colsum(i + 1);
                        row[2 * i] = ((this * 3 + last + 8) >> 4) as u8;
                        row[2 * i + 1] = ((this * 3 + next + 7) >> 4) as u8;
                        last = this;
                        this = next;
                    }
                    row[2 * dw - 2] = ((this * 3 + last + 8) >> 4) as u8;
                    row[2 * dw - 1] = ((this * 4 + 7) >> 4) as u8;
                }
            }
            _ => {
                // h1v2 and other ratios are rejected earlier; replicate as a safe default.
                let sy = (y / vf).min(last_row);
                for x in 0..width {
                    row[x] = src((x / hf).min(dw - 1), sy) as u8;
                }
            }
        }
        out[y * width..(y + 1) * width].copy_from_slice(&row[..width]);
    }
    out
}

impl Decoded {
    /// The luma plane, as libjpeg returns it for `JCS_GRAYSCALE` output.
    pub fn gray(&self) -> Vec<u8> {
        let c = &self.comps[0];
        let mut out = vec![0u8; self.width * self.height];
        for y in 0..self.height {
            out[y * self.width..(y + 1) * self.width]
                .copy_from_slice(&c.plane[y * c.pw..y * c.pw + self.width]);
        }
        out
    }

    /// Interleaved BGR, as libjpeg returns it for `JCS_EXT_BGR` output with default settings.
    pub fn bgr(&self) -> Result<Vec<u8>, JpegError> {
        let (w, h) = (self.width, self.height);
        let mut out = vec![0u8; w * h * 3];
        if self.comps.len() == 1 {
            let g = self.gray();
            for (i, &v) in g.iter().enumerate() {
                out[3 * i..3 * i + 3].copy_from_slice(&[v, v, v]);
            }
            return Ok(out);
        }
        let y = &self.comps[0];
        if y.h != self.hmax || y.v != self.vmax {
            return unsupported("luma is not the full-resolution component");
        }
        for c in &self.comps[1..] {
            let ok = matches!((self.hmax / c.h, self.vmax / c.v), (1, 1) | (2, 1) | (2, 2))
                && self.hmax.is_multiple_of(c.h)
                && self.vmax.is_multiple_of(c.v);
            if !ok {
                return unsupported("chroma sampling ratio");
            }
        }
        let cb = upsample(&self.comps[1], self.hmax, self.vmax, w, h);
        let cr = upsample(&self.comps[2], self.hmax, self.vmax, w, h);
        let t = YccTables::new();
        for yy in 0..h {
            for x in 0..w {
                let i = yy * w + x;
                let yv = i32::from(y.plane[yy * y.pw + x]);
                let cbv = usize::from(cb[i]);
                let crv = usize::from(cr[i]);
                let r = clamp_u8(yv + t.cr_r[crv]);
                let g = clamp_u8(yv + ((t.cb_g[cbv] + t.cr_g[crv]) >> 16));
                let b = clamp_u8(yv + t.cb_b[cbv]);
                out[3 * i] = b;
                out[3 * i + 1] = g;
                out[3 * i + 2] = r;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_of_dc_only_block_is_flat() {
        let mut coef = [0i32; 64];
        // DC of 80 after dequantization: 80 / 8 = 10 above mid-grey.
        coef[0] = 80;
        let mut out = [0u8; 64];
        idct_islow(&coef, &mut out, 8);
        assert!(out.iter().all(|&v| v == 138));
    }

    #[test]
    fn idct_limit_matches_libjpeg_table() {
        assert_eq!(idct_limit(0), 128);
        assert_eq!(idct_limit(127), 255);
        assert_eq!(idct_limit(300), 255);
        assert_eq!(idct_limit(-1), 127);
        assert_eq!(idct_limit(-128), 0);
        assert_eq!(idct_limit(-300), 0);
    }

    #[test]
    fn ycc_tables_match_known_values() {
        let t = YccTables::new();
        // FIX(1.402) = 91881; for x = 127: (91881*127 + 32768) >> 16 = 178.
        assert_eq!(t.cr_r[255], 178);
        assert_eq!(t.cr_r[128], 0);
        assert_eq!(t.cb_b[0], -227);
    }

    #[test]
    fn rejects_non_jpeg() {
        assert!(matches!(decode(b"nope"), Err(JpegError::Corrupt(_))));
    }
}
