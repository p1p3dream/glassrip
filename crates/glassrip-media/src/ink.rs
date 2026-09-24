//! The prototype's ink metric (`ink_metric.py`): a binary mask of dark strokes and saturated
//! color at 640x360, and a symmetric "ink change" score between two aligned frames.

use crate::ecc::{find_transform_ecc_affine, EccOutcome, EccParams};
use crate::error::{MediaError, Result};
use crate::gaussian::{blur_f32, blur_u8, KernelSize};
use crate::plane::{Bgr, Plane};
use crate::resize::{area_bgr, half_linear_gray};
use crate::util::replicate;
use crate::warp::{affine_nearest_u8, IDENTITY};

/// Adaptive threshold block size.
pub const BLOCK: usize = 31;
/// Adaptive threshold offset `C`.
pub const OFFSET: i32 = 12;
/// Minimum HSV saturation for colored ink.
pub const MIN_SAT: u8 = 70;
/// Minimum HSV value for colored ink.
pub const MIN_VAL: u8 = 80;
/// Border excluded from the ink comparison, in 640x360 pixels.
pub const BORDER: usize = 20;
/// Side of the square dilation used for matching strokes.
pub const DILATE: usize = 5;

/// `cv2.cvtColor(bgr, COLOR_BGR2GRAY)` as the prototype's OpenCV build computed it.
///
/// OpenCV's generic code uses 14-bit coefficients `(4899 R + 9617 G + 1868 B + 2^13) >> 14`,
/// but the arm64 wheels that produced the reference dispatch this call to the Carotene HAL,
/// which uses 15-bit BT.601 coefficients with a rounding shift. The two differ by one level
/// on a small share of pixels; this function reproduces the Carotene result.
pub fn bgr_to_gray(img: &Bgr) -> Plane<u8> {
    let mut out = Plane::new(img.width, img.height);
    for (o, px) in out.data.iter_mut().zip(img.data.as_chunks::<3>().0) {
        let v = u32::from(px[0]) * 3735 + u32::from(px[1]) * 19235 + u32::from(px[2]) * 9798;
        *o = ((v + (1 << 14)) >> 15).min(255) as u8;
    }
    out
}

/// OpenCV's 8-bit HSV saturation for a BGR pixel (`RGB2HSV_b`, `hsv_shift = 12`).
#[inline]
fn hsv_s_v(b: u8, g: u8, r: u8, sdiv: &[i32; 256]) -> (u8, u8) {
    let v = b.max(g).max(r);
    let vmin = b.min(g).min(r);
    let diff = i32::from(v - vmin);
    let s = (diff * sdiv[usize::from(v)] + (1 << 11)) >> 12;
    (s as u8, v)
}

fn sdiv_table() -> [i32; 256] {
    let mut t = [0i32; 256];
    for (i, v) in t.iter_mut().enumerate().skip(1) {
        *v = (f64::from(255u32 << 12) / i as f64).round_ties_even() as i32;
    }
    t
}

/// Mask of pixels with HSV `S > 70` and `V > 80`, as 0/255.
pub fn color_mask(img: &Bgr) -> Plane<u8> {
    let sdiv = sdiv_table();
    let mut out = Plane::new(img.width, img.height);
    for (o, px) in out.data.iter_mut().zip(img.data.as_chunks::<3>().0) {
        let (s, v) = hsv_s_v(px[0], px[1], px[2], &sdiv);
        *o = if s > MIN_SAT && v > MIN_VAL { 255 } else { 0 };
    }
    out
}

/// `boxFilter(src, CV_8U, (k, k), normalize=True, BORDER_REPLICATE)`: rounded block mean.
pub fn box_mean_replicate(src: &Plane<u8>, k: usize) -> Plane<u8> {
    let (w, h) = (src.width, src.height);
    let r = (k / 2) as isize;
    // Horizontal sums.
    let mut hs = vec![0u32; w * h];
    for y in 0..h {
        let row = src.row(y);
        for x in 0..w {
            let mut s = 0u32;
            for d in -r..=r {
                s += u32::from(row[replicate(x as isize + d, w)]);
            }
            hs[y * w + x] = s;
        }
    }
    let area = (k * k) as f64;
    let mut out = Plane::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut s = 0u32;
            for d in -r..=r {
                s += hs[replicate(y as isize + d, h) * w + x];
            }
            out.data[y * w + x] = (f64::from(s) / area).round_ties_even().min(255.0) as u8;
        }
    }
    out
}

/// `adaptiveThreshold(src, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, block, c)`.
pub fn adaptive_threshold_mean_inv(src: &Plane<u8>, block: usize, c: i32) -> Plane<u8> {
    let mean = box_mean_replicate(src, block);
    let mut out = Plane::new(src.width, src.height);
    for ((o, &s), &m) in out.data.iter_mut().zip(&src.data).zip(&mean.data) {
        *o = if i32::from(s) - i32::from(m) <= -c {
            255
        } else {
            0
        };
    }
    out
}

/// `cv2.dilate(src, ones((k, k)))` with the default border (pixels outside never win).
pub fn dilate(src: &Plane<u8>, k: usize) -> Plane<u8> {
    let (w, h) = (src.width, src.height);
    let r = (k / 2) as isize;
    let mut tmp = Plane::new(w, h);
    for y in 0..h {
        let row = src.row(y);
        for x in 0..w {
            let lo = (x as isize - r).max(0) as usize;
            let hi = ((x as isize + r) as usize).min(w - 1);
            tmp.data[y * w + x] = row[lo..=hi].iter().copied().max().unwrap_or(0);
        }
    }
    let mut out = Plane::new(w, h);
    for y in 0..h {
        let lo = (y as isize - r).max(0) as usize;
        let hi = ((y as isize + r) as usize).min(h - 1);
        for x in 0..w {
            let mut m = 0u8;
            for yy in lo..=hi {
                m = m.max(tmp.data[yy * w + x]);
            }
            out.data[y * w + x] = m;
        }
    }
    out
}

/// Per-frame inputs of the ink metric.
#[derive(Debug, Clone)]
pub struct InkFrame {
    /// Ink mask, 0 or 255 (the prototype's `ink_m`).
    pub mask: Plane<u8>,
    /// Alignment image: the 640x360 blurred gray (the prototype's `ink_g`) halved with
    /// `INTER_LINEAR`, cast to f32, Gaussian 5x5 sigma 1.2 in f32. Precomputed because every
    /// comparison needs it.
    pub align: Plane<f32>,
}

/// Frame size the prototype's ink path was run on and that `prototype_compat` reproduces:
/// `cv2.resize(..., (640, 360), INTER_AREA)` is then an exact 3x3 block mean.
pub const INK_INPUT: (usize, usize) = (1920, 1080);

/// `ink(f)` from `ink_metric.py`, from a full-resolution 1920x1080 BGR frame. Other sizes are
/// rejected: OpenCV's general `INTER_AREA` path for non-integer factors is not ported.
pub fn ink_frame(full: &Bgr) -> Result<InkFrame> {
    if (full.width, full.height) != INK_INPUT {
        return Err(MediaError::Size {
            width: full.width,
            height: full.height,
            reason: "prototype_compat ink metric needs a 1920x1080 frame",
        });
    }
    let im = area_bgr(full, 3);
    let gray = blur_u8(&bgr_to_gray(&im), KernelSize::K3, 0.0);
    let dark = adaptive_threshold_mean_inv(&gray, BLOCK, OFFSET);
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
    let align = blur_f32(&half_linear_gray(&gray).map(f32::from), KernelSize::K5, 1.2);
    Ok(InkFrame { mask, align })
}

/// Result of [`ink_change`].
#[derive(Debug, Clone, Copy)]
pub struct InkChange {
    /// `(|B \ dilate(A)| + |A \ dilate(B)|) / max(1, |A| + |B|)` inside the border mask.
    pub value: f64,
    /// ECC outcome on the 320x180 alignment images (translation not yet doubled).
    pub ecc: EccOutcome,
}

fn border_masked(m: &Plane<u8>, border: usize) -> Plane<u8> {
    let mut out = m.clone();
    let (w, h) = (m.width, m.height);
    for y in 0..h {
        for x in 0..w {
            if y < border
                || y >= h.saturating_sub(border)
                || x < border
                || x >= w.saturating_sub(border)
            {
                out.data[y * w + x] = 0;
            }
        }
    }
    out
}

/// `ink_change(A, B)` from `ink_metric.py`: align B to A on the halved gray, double the
/// translation, warp B's mask (nearest), then compare dilated masks.
pub fn ink_change(a: &InkFrame, b: &InkFrame) -> InkChange {
    let ecc = find_transform_ecc_affine(&a.align, &b.align, IDENTITY, EccParams::default());
    let mut wm = ecc.warp;
    wm[2] *= 2.0;
    wm[5] *= 2.0;
    let (w, h) = (a.mask.width, a.mask.height);
    let ib = affine_nearest_u8(&b.mask, &wm, w, h);
    let ia = border_masked(&a.mask, BORDER);
    let ib = border_masked(&ib, BORDER);
    let (da, db) = rayon::join(|| dilate(&ia, DILATE), || dilate(&ib, DILATE));
    let mut new = 0usize;
    let mut gone = 0usize;
    let mut na = 0usize;
    let mut nb = 0usize;
    for i in 0..ia.data.len() {
        let (pa, pb) = (ia.data[i] > 0, ib.data[i] > 0);
        na += usize::from(pa);
        nb += usize::from(pb);
        if pb && da.data[i] == 0 {
            new += 1;
        }
        if pa && db.data[i] == 0 {
            gone += 1;
        }
    }
    InkChange {
        value: (new + gone) as f64 / (na + nb).max(1) as f64,
        ecc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_matches_carotene_coefficients() {
        let img = Bgr {
            width: 3,
            height: 1,
            data: vec![255, 255, 255, 0, 0, 255, 10, 200, 30],
        };
        let g = bgr_to_gray(&img);
        assert_eq!(g.data[0], 255);
        // Pure red: (255*9798 + 16384) >> 15 = 76.
        assert_eq!(g.data[1], 76);
        assert_eq!(
            g.data[2],
            ((10 * 3735 + 200 * 19235 + 30 * 9798 + 16384) >> 15) as u8
        );
    }

    #[test]
    fn saturation_formula() {
        let t = sdiv_table();
        assert_eq!(hsv_s_v(0, 0, 255, &t), (255, 255));
        assert_eq!(hsv_s_v(100, 100, 100, &t), (0, 100));
        // diff 60 at v 200: 60 * round(255*4096/200) = 60 * 5222 -> (313320 + 2048) >> 12 = 76.
        assert_eq!(hsv_s_v(140, 170, 200, &t).0, 76);
    }

    #[test]
    fn box_mean_rounds_to_nearest() {
        let mut p = Plane::filled(5, 5, 0u8);
        *p.get_mut(2, 2) = 9 * 3;
        // 3x3 mean at the centre: 27 / 9 = 3.
        assert_eq!(box_mean_replicate(&p, 3).get(2, 2), 3);
        // Replicated border at the corner: 4 copies of 10 in a 3x3 window -> 40/9 = 4.44 -> 4.
        let mut q = Plane::filled(4, 4, 0u8);
        *q.get_mut(0, 0) = 10;
        assert_eq!(box_mean_replicate(&q, 3).get(0, 0), 4);
    }

    #[test]
    fn adaptive_threshold_marks_dark_strokes() {
        let mut p = Plane::filled(40, 40, 200u8);
        for y in 10..30 {
            *p.get_mut(20, y) = 20;
        }
        let t = adaptive_threshold_mean_inv(&p, 31, 12);
        assert_eq!(t.get(20, 15), 255);
        assert_eq!(t.get(5, 5), 0);
        assert_eq!(t.get(22, 15), 0);
    }

    #[test]
    fn dilate_is_square_max_filter() {
        let mut p = Plane::filled(9, 9, 0u8);
        *p.get_mut(4, 4) = 255;
        let d = dilate(&p, 5);
        for y in 0..9 {
            for x in 0..9 {
                let inside = (2..=6).contains(&x) && (2..=6).contains(&y);
                assert_eq!(d.get(x, y) == 255, inside, "({x},{y})");
            }
        }
        let mut e = Plane::filled(6, 6, 0u8);
        *e.get_mut(0, 0) = 7;
        assert_eq!(dilate(&e, 5).get(2, 2), 7);
        assert_eq!(dilate(&e, 5).get(3, 3), 0);
    }

    #[test]
    fn border_mask_zeroes_edges() {
        let p = Plane::filled(50, 50, 255u8);
        let m = border_masked(&p, 20);
        assert_eq!(m.data.iter().filter(|&&v| v > 0).count(), 100);
    }

    #[test]
    fn ink_frame_requires_prototype_size() {
        let small = Bgr {
            width: 640,
            height: 360,
            data: vec![0; 640 * 360 * 3],
        };
        assert!(ink_frame(&small).is_err());
        let full = Bgr {
            width: 1920,
            height: 1080,
            data: vec![200; 1920 * 1080 * 3],
        };
        let f = ink_frame(&full).unwrap();
        assert_eq!((f.mask.width, f.mask.height), (640, 360));
        assert_eq!((f.align.width, f.align.height), (320, 180));
        assert!(f.mask.data.iter().all(|&v| v == 0));
    }
}
