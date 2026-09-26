//! `production` mode alignment and pair scores (spec deviations from `prototype_compat`).
//!
//! 1. ECC starts from a phase-correlation translation instead of the identity.
//! 2. When ECC fails (or converges to an implausible warp), the phase-correlation translation
//!    is used on its own; when that is not trustworthy either, alignment has failed and the
//!    caller treats the pair as changed (`align_failed`), never as aligned by identity.
//! 3. SSIM is averaged over the warp's valid-pixel mask instead of a fixed inset;
//!    `changed_frac` counts the pixels the warp cannot cover as changed (so two different
//!    screens aligned over a small overlap do not score as "same"), and the ink change
//!    counts reference ink in the uncovered region as gone.
//! 4. Frames of any size are accepted: [`frame_features_any`] area-resamples to a 1920 px
//!    long edge instead of requiring exactly 1920x1080.
//!
//! These change the numbers, so nothing here is covered by the parity gate.

use std::cell::RefCell;
use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::ecc::{find_transform_ecc_affine, EccParams};
use crate::ink::{dilate, InkFrame, BORDER, DILATE};
use crate::plane::Plane;
use crate::ssim::ssim_map;
use crate::warp::{affine_linear_f32, affine_nearest_u8, Affine, IDENTITY};

/// Translation found by phase correlation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseShift {
    /// Horizontal shift: content at `x` in the template appears at `x + dx` in the input.
    pub dx: f64,
    /// Vertical shift, same convention.
    pub dy: f64,
    /// Normalized peak response in `[0, 1]` (1 for a pure circular shift).
    pub response: f64,
}

thread_local! {
    static PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
}

fn plans(w: usize, h: usize) -> [Arc<dyn Fft<f32>>; 4] {
    PLANNER.with(|p| {
        let mut p = p.borrow_mut();
        [
            p.plan_fft_forward(w),
            p.plan_fft_forward(h),
            p.plan_fft_inverse(w),
            p.plan_fft_inverse(h),
        ]
    })
}

fn fft2(
    data: &mut [Complex32],
    w: usize,
    h: usize,
    row: &Arc<dyn Fft<f32>>,
    col: &Arc<dyn Fft<f32>>,
) {
    for r in data.chunks_mut(w) {
        row.process(r);
    }
    let mut column = vec![Complex32::new(0.0, 0.0); h];
    for x in 0..w {
        for (y, c) in column.iter_mut().enumerate() {
            *c = data[y * w + x];
        }
        col.process(&mut column);
        for (y, c) in column.iter().enumerate() {
            data[y * w + x] = *c;
        }
    }
}

fn hann(n: usize) -> Vec<f32> {
    if n < 2 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
            (0.5 - 0.5 * t.cos()) as f32
        })
        .collect()
}

/// Phase correlation of two same-size images with a Hann window. Returns `None` when the
/// sizes differ or an image is smaller than 4x4.
pub fn phase_correlate(template: &Plane<f32>, input: &Plane<f32>) -> Option<PhaseShift> {
    let (w, h) = (template.width, template.height);
    if (input.width, input.height) != (w, h) || w < 4 || h < 4 {
        return None;
    }
    let (wx, wy) = (hann(w), hann(h));
    let windowed = |p: &Plane<f32>| -> Vec<Complex32> {
        let mean = p.data.iter().map(|&v| f64::from(v)).sum::<f64>() / p.data.len() as f64;
        let mean = mean as f32;
        p.data
            .iter()
            .enumerate()
            .map(|(i, &v)| Complex32::new((v - mean) * wx[i % w] * wy[i / w], 0.0))
            .collect()
    };
    let [fr, fc, ir, ic] = plans(w, h);
    let mut a = windowed(template);
    let mut b = windowed(input);
    fft2(&mut a, w, h, &fr, &fc);
    fft2(&mut b, w, h, &fr, &fc);
    let mut cross: Vec<Complex32> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| {
            let c = x.conj() * y;
            let n = c.norm();
            if n > 1e-12 {
                c / n
            } else {
                Complex32::new(0.0, 0.0)
            }
        })
        .collect();
    fft2(&mut cross, w, h, &ir, &ic);
    let scale = 1.0 / (w * h) as f32;
    let r: Vec<f32> = cross.iter().map(|c| c.re * scale).collect();
    let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
    for (i, &v) in r.iter().enumerate() {
        if v > best {
            best = v;
            bi = i;
        }
    }
    let (px, py) = ((bi % w) as isize, (bi / w) as isize);
    // 3x3 weighted centroid around the peak (circular indexing) and summed response.
    let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            let x = (px + dx).rem_euclid(w as isize) as usize;
            let y = (py + dy).rem_euclid(h as isize) as usize;
            let v = f64::from(r[y * w + x]).max(0.0);
            sx += v * dx as f64;
            sy += v * dy as f64;
            sw += v;
        }
    }
    let (ox, oy) = if sw > 0.0 {
        (sx / sw, sy / sw)
    } else {
        (0.0, 0.0)
    };
    let wrap = |p: isize, n: usize| -> f64 {
        if p > (n / 2) as isize {
            (p - n as isize) as f64
        } else {
            p as f64
        }
    };
    Some(PhaseShift {
        dx: wrap(px, w) + ox,
        dy: wrap(py, h) + oy,
        response: sw.clamp(0.0, 1.0),
    })
}

/// How a pair was aligned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignMethod {
    /// ECC (affine) converged from the phase-correlation start.
    Ecc,
    /// ECC failed; the phase-correlation translation alone was used.
    PhaseTranslation,
    /// Neither method produced a trustworthy alignment.
    Failed,
}

/// Parameters for [`align`].
#[derive(Debug, Clone, Copy)]
pub struct AlignParams {
    /// ECC termination and prefilter.
    pub ecc: EccParams,
    /// Minimum phase-correlation response to trust its translation.
    pub min_phase_response: f64,
    /// Largest accepted translation as a fraction of the image width or height.
    pub max_shift_frac: f64,
    /// Largest accepted deviation of the linear part from the identity (scale, shear).
    pub max_linear_dev: f64,
}

impl Default for AlignParams {
    fn default() -> Self {
        Self {
            ecc: EccParams::default(),
            min_phase_response: 0.05,
            max_shift_frac: 0.5,
            max_linear_dev: 0.15,
        }
    }
}

/// Result of [`align`].
#[derive(Debug, Clone, Copy)]
pub struct Alignment {
    /// Warp mapping template coordinates into the input (identity when failed).
    pub warp: Affine,
    /// How the warp was obtained.
    pub method: AlignMethod,
    /// Phase-correlation estimate, when computed.
    pub phase: Option<PhaseShift>,
    /// Whether ECC itself converged.
    pub ecc_ok: bool,
    /// Both images are one flat color ([`is_flat`]): there is nothing to align on,
    /// and every warp gives the same comparison, so the identity is exact even
    /// though neither method ran (`method` stays [`AlignMethod::Failed`]).
    pub flat: bool,
}

impl Alignment {
    /// True unless alignment failed; two flat images count as aligned.
    pub fn ok(&self) -> bool {
        self.method != AlignMethod::Failed || self.flat
    }
}

/// Largest standard deviation (gray levels) of a blurred alignment image that
/// counts as one flat color. Compression noise on a flat fill stays far under it
/// after the prefilter; drawn content of any size raises it.
pub const FLAT_MAX_STD: f64 = 1.0;

/// True when `p` is one flat color: no structure for phase correlation or ECC.
/// A pair where only one side is flat still fails alignment: a view that went
/// blank cannot be told from a pan or cut to empty canvas.
pub fn is_flat(p: &Plane<f32>) -> bool {
    if p.data.is_empty() {
        return false;
    }
    let n = p.data.len() as f64;
    let mean = p.data.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var = p
        .data
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n;
    var.is_finite() && var.sqrt() <= FLAT_MAX_STD
}

fn plausible(m: &Affine, w: usize, h: usize, p: &AlignParams) -> bool {
    if m.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let dev = [m[0] - 1.0, m[1], m[3], m[4] - 1.0]
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()));
    f64::from(dev) <= p.max_linear_dev
        && f64::from(m[2].abs()) <= p.max_shift_frac * w as f64
        && f64::from(m[5].abs()) <= p.max_shift_frac * h as f64
}

/// Aligns `input` to `template`: phase correlation, then ECC from that start, then the
/// translation alone, else failure. Two same-size flat images ([`is_flat`]) align as
/// the identity without either method (both would fail on them); a pair where only
/// one image is flat fails.
pub fn align(template: &Plane<f32>, input: &Plane<f32>, p: &AlignParams) -> Alignment {
    let (w, h) = (template.width, template.height);
    let (flat_t, flat_i) = (is_flat(template), is_flat(input));
    if flat_t || flat_i {
        // Two flat images: nothing to align, the identity is exact. One flat image:
        // any fit to it is spurious (noise the phase normalization amplifies).
        return Alignment {
            warp: IDENTITY,
            method: AlignMethod::Failed,
            phase: None,
            ecc_ok: false,
            flat: flat_t && flat_i && (input.width, input.height) == (w, h),
        };
    }
    let phase = phase_correlate(template, input);
    let trusted = phase.filter(|s| {
        s.response >= p.min_phase_response
            && s.dx.abs() <= p.max_shift_frac * w as f64
            && s.dy.abs() <= p.max_shift_frac * h as f64
    });
    let init = trusted.map_or(IDENTITY, |s| [1.0, 0.0, s.dx as f32, 0.0, 1.0, s.dy as f32]);
    let ecc = find_transform_ecc_affine(template, input, init, p.ecc);
    if ecc.ok() && plausible(&ecc.warp, w, h, p) {
        return Alignment {
            warp: ecc.warp,
            method: AlignMethod::Ecc,
            phase,
            ecc_ok: true,
            flat: false,
        };
    }
    match trusted {
        Some(_) => Alignment {
            warp: init,
            method: AlignMethod::PhaseTranslation,
            phase,
            ecc_ok: ecc.ok(),
            flat: false,
        },
        None => Alignment {
            warp: IDENTITY,
            method: AlignMethod::Failed,
            phase,
            ecc_ok: ecc.ok(),
            flat: false,
        },
    }
}

/// Pixels of a `w x h` output whose inverse-mapped source position under `m` lies inside a
/// `sw x sh` source (so bilinear sampling never reads the zero border).
pub fn valid_mask(m: &Affine, w: usize, h: usize, sw: usize, sh: usize) -> Plane<u8> {
    let mut out = Plane::new(w, h);
    let (maxx, maxy) = (sw.saturating_sub(1) as f32, sh.saturating_sub(1) as f32);
    for y in 0..h {
        let yf = y as f32;
        let (mx, my) = (yf.mul_add(m[1], m[2]), yf.mul_add(m[4], m[5]));
        for x in 0..w {
            let xf = x as f32;
            let sx = m[0].mul_add(xf, mx);
            let sy = m[3].mul_add(xf, my);
            out.data[y * w + x] =
                u8::from((0.0..=maxx).contains(&sx) && (0.0..=maxy).contains(&sy));
        }
    }
    out
}

/// Erodes a 0/1 mask with a `(2r+1)` square; pixels outside the image count as valid, so the
/// image border itself is not eroded.
pub fn erode_mask(m: &Plane<u8>, r: usize) -> Plane<u8> {
    if r == 0 {
        return m.clone();
    }
    let (w, h) = (m.width, m.height);
    let mut tmp = Plane::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            tmp.data[y * w + x] = (lo..=hi).map(|xx| m.data[y * w + xx]).min().unwrap_or(0);
        }
    }
    let mut out = Plane::new(w, h);
    for y in 0..h {
        let lo = y.saturating_sub(r);
        let hi = (y + r).min(h - 1);
        for x in 0..w {
            out.data[y * w + x] = (lo..=hi).map(|yy| tmp.data[yy * w + x]).min().unwrap_or(0);
        }
    }
    out
}

/// Parameters for [`pair_score`].
#[derive(Debug, Clone, Copy)]
pub struct ScoreParams {
    /// Alignment.
    pub align: AlignParams,
    /// Absolute gray difference counted as a changed pixel.
    pub changed_delta: f32,
    /// Pairs whose valid region is smaller than this share of the image count as failed.
    pub min_valid_frac: f64,
}

impl Default for ScoreParams {
    fn default() -> Self {
        Self {
            align: AlignParams::default(),
            changed_delta: 25.0,
            min_valid_frac: 0.5,
        }
    }
}

/// Pair score in `production` mode.
#[derive(Debug, Clone, Copy)]
pub struct PairScore {
    /// Mean SSIM over the eroded valid mask (0 when alignment failed).
    pub ssim: f64,
    /// Share of valid pixels whose aligned difference exceeds the delta (1 when failed).
    pub changed_frac: f64,
    /// Length of the translation.
    pub shift: f64,
    /// Share of the image inside the valid mask.
    pub valid_frac: f64,
    /// Alignment (method `Failed` also covers a too-small valid region).
    pub alignment: Alignment,
}

impl PairScore {
    /// True unless alignment failed.
    pub fn align_ok(&self) -> bool {
        self.alignment.ok()
    }
}

/// SSIM window radius (7x7 window).
const SSIM_RADIUS: usize = 3;

/// Aligns `b` to `a` and scores the pair over the valid-pixel mask.
pub fn pair_score(a: &Plane<f32>, b: &Plane<f32>, p: &ScoreParams) -> PairScore {
    let mut alignment = align(a, b, &p.align);
    let (w, h) = (a.width, a.height);
    let failed = |alignment: Alignment, valid_frac: f64| PairScore {
        ssim: 0.0,
        changed_frac: 1.0,
        shift: 0.0,
        valid_frac,
        alignment: Alignment {
            method: AlignMethod::Failed,
            ..alignment
        },
    };
    if !alignment.ok() {
        return failed(alignment, 0.0);
    }
    let wm = alignment.warp;
    let bw = affine_linear_f32(b, &wm, w, h);
    let valid = valid_mask(&wm, w, h, b.width, b.height);
    let n_valid = valid.data.iter().filter(|&&v| v > 0).count();
    let valid_frac = n_valid as f64 / (w * h).max(1) as f64;
    let inner = erode_mask(&valid, SSIM_RADIUS);
    let n_inner = inner.data.iter().filter(|&&v| v > 0).count();
    if valid_frac < p.min_valid_frac || n_inner == 0 {
        alignment.method = AlignMethod::Failed;
        return failed(alignment, valid_frac);
    }
    let map = ssim_map(a, &bw);
    let ssim = map
        .data
        .iter()
        .zip(&inner.data)
        .filter(|(_, &m)| m > 0)
        .map(|(&v, _)| f64::from(v))
        .sum::<f64>()
        / n_inner as f64;
    let changed = a
        .data
        .iter()
        .zip(&bw.data)
        .zip(&valid.data)
        .filter(|(_, &m)| m > 0)
        .filter(|((&x, &y), _)| (x - y).abs() > p.changed_delta)
        .count();
    // Pixels the warp cannot cover count as changed.
    let total = (w * h).max(1);
    PairScore {
        ssim,
        changed_frac: (changed + (total - n_valid)) as f64 / total as f64,
        shift: f64::from(wm[2].hypot(wm[5])),
        valid_frac,
        alignment,
    }
}

/// Ink change in `production` mode.
#[derive(Debug, Clone, Copy)]
pub struct InkChange {
    /// Change value (1 when alignment failed: segmentation must cut there, but the
    /// value then measures nothing; check [`InkChange::measured`] before using it
    /// as evidence of drawn or erased content).
    pub value: f64,
    /// Alignment on the ink path's small gray.
    pub alignment: Alignment,
}

impl InkChange {
    /// True when the value was measured on aligned ink masks (not the failure 1.0).
    pub fn measured(&self) -> bool {
        self.alignment.ok()
    }
}

/// `ink_change(A, B)` with production alignment: phase-initialized ECC on the ink alignment
/// images, translation doubled for the ink masks. A's ink where B's warp cannot cover counts
/// as gone (B's warped mask is empty there). Failure counts as a full change.
pub fn ink_change(a: &InkFrame, b: &InkFrame, p: &AlignParams) -> InkChange {
    let alignment = align(&a.align, &b.align, p);
    if !alignment.ok() {
        return InkChange {
            value: 1.0,
            alignment,
        };
    }
    let mut wm = alignment.warp;
    wm[2] *= 2.0;
    wm[5] *= 2.0;
    let (w, h) = (a.mask.width, a.mask.height);
    let ib = affine_nearest_u8(&b.mask, &wm, w, h);
    let keep = |m: &Plane<u8>| -> Plane<u8> {
        let mut out = m.clone();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let border = y < BORDER
                    || y >= h.saturating_sub(BORDER)
                    || x < BORDER
                    || x >= w.saturating_sub(BORDER);
                if border {
                    out.data[i] = 0;
                }
            }
        }
        out
    };
    let (ia, ib) = (keep(&a.mask), keep(&ib));
    let (da, db) = rayon::join(|| dilate(&ia, DILATE), || dilate(&ib, DILATE));
    let (mut new, mut gone, mut na, mut nb) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..ia.data.len() {
        let (pa, pb) = (ia.data[i] > 0, ib.data[i] > 0);
        na += usize::from(pa);
        nb += usize::from(pb);
        new += usize::from(pb && da.data[i] == 0);
        gone += usize::from(pa && db.data[i] == 0);
    }
    InkChange {
        value: (new + gone) as f64 / (na + nb).max(1) as f64,
        alignment,
    }
}

/// Long edge frames are resampled to before production features.
pub const PRODUCTION_LONG_EDGE: usize = 1920;

/// Fractional area (box) resampling of interleaved 8-bit data, like `INTER_AREA` for
/// downscaling (and bilinear-like box sampling when enlarging).
fn area_resize(src: &[u8], w: usize, h: usize, cn: usize, dw: usize, dh: usize) -> Vec<u8> {
    let weights = |n: usize, dn: usize| -> Vec<Vec<(usize, f64)>> {
        let scale = n as f64 / dn as f64;
        (0..dn)
            .map(|d| {
                let (lo, hi) = (d as f64 * scale, (d + 1) as f64 * scale);
                let mut v = Vec::new();
                let mut s = lo.floor() as usize;
                while (s as f64) < hi && s < n {
                    let a = (s as f64).max(lo);
                    let b = ((s + 1) as f64).min(hi);
                    if b > a {
                        v.push((s, (b - a) / (hi - lo)));
                    }
                    s += 1;
                }
                v
            })
            .collect()
    };
    let (wx, wy) = (weights(w, dw), weights(h, dh));
    let mut tmp = vec![0f64; dw * h * cn];
    for y in 0..h {
        for (dx, ws) in wx.iter().enumerate() {
            for c in 0..cn {
                tmp[(y * dw + dx) * cn + c] = ws
                    .iter()
                    .map(|&(x, k)| k * f64::from(src[(y * w + x) * cn + c]))
                    .sum();
            }
        }
    }
    let mut out = vec![0u8; dw * dh * cn];
    for (dy, ws) in wy.iter().enumerate() {
        for dx in 0..dw {
            for c in 0..cn {
                let v: f64 = ws
                    .iter()
                    .map(|&(y, k)| k * tmp[(y * dw + dx) * cn + c])
                    .sum();
                out[(dy * dw + dx) * cn + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Target size for a frame: long edge [`PRODUCTION_LONG_EDGE`], aspect kept, both sides
/// rounded down to multiples of 6 (the pair-score and ink grids are 6x and 3x downsamples).
pub fn production_size(w: usize, h: usize) -> (usize, usize) {
    let long = w.max(h).max(1) as f64;
    let s = PRODUCTION_LONG_EDGE as f64 / long;
    let r6 = |v: f64| ((v.round() as usize) / 6 * 6).max(6);
    (r6(w as f64 * s), r6(h as f64 * s))
}

/// Per-frame features for any frame size (`production` mode): area-resample to a 1920 px
/// long edge when needed, then the same sharpness, 6x small gray, and ink computations as
/// `prototype_compat`. A 1920x1080 frame gives exactly the compat features.
pub fn frame_features_any(
    gray: &Plane<u8>,
    bgr: &crate::plane::Bgr,
) -> crate::error::Result<crate::features::FrameFeatures> {
    use crate::gaussian::{blur_f32, blur_u8, KernelSize};
    use crate::ink::{adaptive_threshold_mean_inv, bgr_to_gray, color_mask, BLOCK, OFFSET};
    use crate::resize::{area_bgr, area_gray, half_linear_gray};
    if gray.width < 12 || gray.height < 12 || (gray.width, gray.height) != (bgr.width, bgr.height) {
        return Err(crate::error::MediaError::Size {
            width: gray.width,
            height: gray.height,
            reason: "frame too small, or gray and color sizes differ",
        });
    }
    let (tw, th) = production_size(gray.width, gray.height);
    let (gray, bgr) = if (tw, th) == (gray.width, gray.height) {
        (gray.clone(), bgr.clone())
    } else {
        (
            Plane {
                width: tw,
                height: th,
                data: area_resize(&gray.data, gray.width, gray.height, 1, tw, th),
            },
            crate::plane::Bgr {
                width: tw,
                height: th,
                data: area_resize(&bgr.data, bgr.width, bgr.height, 3, tw, th),
            },
        )
    };
    let (sharpness, small) = rayon::join(
        || crate::sharpness::laplacian_variance(&gray),
        || blur_u8(&area_gray(&gray, 6), KernelSize::K5, 1.2).map(f32::from),
    );
    let im = area_bgr(&bgr, 3);
    let g3 = blur_u8(&bgr_to_gray(&im), KernelSize::K3, 0.0);
    let dark = adaptive_threshold_mean_inv(&g3, BLOCK, OFFSET);
    let col = color_mask(&im);
    let mask = Plane {
        width: dark.width,
        height: dark.height,
        data: dark
            .data
            .iter()
            .zip(&col.data)
            .map(|(a, b)| a | b)
            .collect(),
    };
    let align = blur_f32(&half_linear_gray(&g3).map(f32::from), KernelSize::K5, 1.2);
    Ok(crate::features::FrameFeatures {
        sharpness,
        small,
        ink: InkFrame { mask, align },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smooth random-ish texture (sum of sinusoids), 0..255.
    fn texture(w: usize, h: usize, ox: f64, oy: f64) -> Plane<f32> {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (xf, yf) = (x as f64 - ox, y as f64 - oy);
                let v = 128.0
                    + 40.0 * (xf * 0.21).sin()
                    + 35.0 * (yf * 0.17).cos()
                    + 30.0 * ((xf + yf) * 0.09).sin()
                    + 20.0 * ((xf * 0.05) * (yf * 0.07)).sin();
                *p.get_mut(x, y) = v as f32;
            }
        }
        p
    }

    /// A flat fill with faint deterministic noise (well under a gray level).
    fn flat(w: usize, h: usize, v: f32) -> Plane<f32> {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                *p.get_mut(x, y) = v + 0.3 * (((x * 7 + y * 13) % 5) as f32 - 2.0) / 2.0;
            }
        }
        p
    }

    #[test]
    fn two_flat_frames_align_as_the_identity_and_measure_no_change() {
        let (a, b) = (flat(160, 96, 240.0), flat(160, 96, 240.0));
        let p = ScoreParams::default();
        let al = align(&a, &b, &p.align);
        assert!(al.ok() && al.flat && al.warp == IDENTITY, "{al:?}");
        let s = pair_score(&a, &b, &p);
        assert!(s.ssim > 0.99 && s.changed_frac == 0.0, "{s:?}");
        let ink = |align: Plane<f32>| InkFrame {
            mask: Plane::new(320, 192),
            align,
        };
        let c = ink_change(&ink(a), &ink(b), &p.align);
        assert!(c.measured(), "{c:?}");
        assert_eq!(c.value, 0.0);
    }

    #[test]
    fn a_view_that_goes_flat_stays_unmeasured() {
        // Content to an empty canvas: an erasure or a pan to empty space, and two
        // frames cannot say which. The ink change stays the failure value.
        let (a, b) = (texture(160, 96, 0.0, 0.0), flat(160, 96, 240.0));
        assert!(!is_flat(&a) && is_flat(&b));
        let p = AlignParams::default();
        assert!(!align(&a, &b, &p).ok());
        assert!(!align(&b, &a, &p).ok());
        let mut mask = Plane::new(320, 192);
        for x in 100..200 {
            *mask.get_mut(x, 90) = 255;
        }
        let c = ink_change(
            &InkFrame { mask, align: a },
            &InkFrame {
                mask: Plane::new(320, 192),
                align: b,
            },
            &p,
        );
        assert!(!c.measured(), "{c:?}");
        assert_eq!(c.value, 1.0);
    }

    #[test]
    fn phase_correlation_finds_translation_sign() {
        let a = texture(160, 96, 0.0, 0.0);
        // Content moved right by 6 and down by 3 in the input.
        let b = texture(160, 96, 6.0, 3.0);
        let s = phase_correlate(&a, &b).unwrap();
        assert!((s.dx - 6.0).abs() < 0.6, "{s:?}");
        assert!((s.dy - 3.0).abs() < 0.6, "{s:?}");
        assert!(s.response > 0.1, "{s:?}");
    }

    #[test]
    fn align_recovers_shift_and_scores_high() {
        let a = texture(160, 96, 0.0, 0.0);
        let b = texture(160, 96, 9.0, -4.0);
        let s = pair_score(&a, &b, &ScoreParams::default());
        assert!(s.align_ok());
        assert!(
            (f64::from(s.alignment.warp[2]) - 9.0).abs() < 0.5,
            "{:?}",
            s.alignment
        );
        assert!(
            (f64::from(s.alignment.warp[5]) + 4.0).abs() < 0.5,
            "{:?}",
            s.alignment
        );
        assert!(s.ssim > 0.95, "{}", s.ssim);
        // Only the uncovered border counts as changed.
        assert!(
            (s.changed_frac - (1.0 - s.valid_frac)).abs() < 0.01,
            "{} {}",
            s.changed_frac,
            s.valid_frac
        );
        assert!(
            s.valid_frac < 0.95 && s.valid_frac > 0.8,
            "{}",
            s.valid_frac
        );
    }

    #[test]
    fn uncovered_pixels_count_as_changed() {
        let a = texture(160, 96, 0.0, 0.0);
        let b = texture(160, 96, 30.0, 0.0);
        let s = pair_score(&a, &b, &ScoreParams::default());
        assert!(s.align_ok(), "{:?}", s.alignment);
        // About 30 of 160 columns are uncovered: they count as changed.
        assert!(
            (s.changed_frac - (1.0 - s.valid_frac)).abs() < 0.01,
            "{} vs {}",
            s.changed_frac,
            s.valid_frac
        );
        assert!(s.changed_frac > 0.15);
    }

    #[test]
    fn scaled_warps_are_rejected() {
        // A 1.3x zoom exceeds the 0.15 linear-deviation limit.
        let a = texture(160, 96, 0.0, 0.0);
        let mut b = Plane::new(160, 96);
        for y in 0..96 {
            for x in 0..160 {
                *b.get_mut(x, y) = a.get(
                    ((x as f64 / 1.3) as usize).min(159),
                    ((y as f64 / 1.3) as usize).min(95),
                );
            }
        }
        let p = AlignParams::default();
        let al = align(&a, &b, &p);
        if al.method == AlignMethod::Ecc {
            assert!(plausible(&al.warp, 160, 96, &p));
        }
        assert!(!plausible(&[1.3, 0.0, 0.0, 0.0, 1.3, 0.0], 160, 96, &p));
        assert!(plausible(&[1.1, 0.0, 0.0, 0.0, 1.1, 0.0], 160, 96, &p));
    }

    fn frame(w: usize, h: usize) -> (Plane<u8>, crate::plane::Bgr) {
        let g = Plane::from_vec(
            w,
            h,
            (0..w * h)
                .map(|i| ((i % w) * 7 + (i / w) * 3) as u8)
                .collect(),
        )
        .unwrap();
        let bgr = crate::plane::Bgr {
            width: w,
            height: h,
            data: g.data.iter().flat_map(|&v| [v, v, v]).collect(),
        };
        (g, bgr)
    }

    #[test]
    fn any_size_frames_are_resampled() {
        for (w, h, sw, sh) in [
            (1920, 1080, 320, 180),
            (1920, 1440, 320, 240),
            (1080, 1920, 180, 320),
            (3840, 2160, 320, 180),
            (640, 480, 320, 240),
        ] {
            let (g, c) = frame(w, h);
            let f = frame_features_any(&g, &c).unwrap();
            assert_eq!((f.small.width, f.small.height), (sw, sh), "{w}x{h}");
            assert_eq!(
                (f.ink.mask.width, f.ink.mask.height),
                (2 * sw, 2 * sh),
                "{w}x{h}"
            );
            assert_eq!((f.ink.align.width, f.ink.align.height), (sw, sh), "{w}x{h}");
        }
    }

    #[test]
    fn full_hd_matches_compat_features() {
        let (g, c) = frame(1920, 1080);
        let a = frame_features_any(&g, &c).unwrap();
        let s = crate::features::small_gray(&g);
        let i = crate::ink::ink_frame(&c).unwrap();
        assert_eq!(a.small, s);
        assert_eq!(a.ink.mask, i.mask);
        assert_eq!(a.ink.align, i.align);
        assert_eq!(a.sharpness, crate::sharpness::laplacian_variance(&g));
    }

    #[test]
    fn area_resize_averages() {
        let src = vec![0u8, 100, 200, 100];
        assert_eq!(area_resize(&src, 4, 1, 1, 2, 1), vec![50, 150]);
        assert_eq!(area_resize(&src, 4, 1, 1, 4, 1), src);
    }

    #[test]
    fn flat_images_of_different_colors_count_as_changed() {
        // Nothing to align on: the identity stands in (no method ran), and the
        // pair still differs everywhere.
        let a = Plane::filled(64, 48, 10.0f32);
        let b = Plane::filled(64, 48, 200.0f32);
        let s = pair_score(&a, &b, &ScoreParams::default());
        assert_eq!(s.alignment.method, AlignMethod::Failed);
        assert!(s.alignment.ok() && s.alignment.flat);
        assert!(s.ssim < 0.1 && s.changed_frac == 1.0, "{s:?}");
        // Against a textured frame a flat one does not align at all.
        let s = pair_score(&texture(64, 48, 0.0, 0.0), &b, &ScoreParams::default());
        assert!(!s.alignment.ok());
        assert_eq!((s.ssim, s.changed_frac), (0.0, 1.0));
    }

    #[test]
    fn erosion_keeps_image_border() {
        let mut m = Plane::filled(10, 6, 1u8);
        assert_eq!(erode_mask(&m, 2), m);
        *m.get_mut(5, 3) = 0;
        let e = erode_mask(&m, 1);
        assert_eq!(e.get(4, 2), 0);
        assert_eq!(e.get(7, 3), 1);
    }

    #[test]
    fn valid_mask_marks_uncovered_columns() {
        let m = [1.0, 0.0, 3.0, 0.0, 1.0, 0.0];
        let v = valid_mask(&m, 10, 2, 10, 2);
        assert_eq!(v.row(0), &[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]);
    }
}
