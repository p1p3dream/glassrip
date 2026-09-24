//! Pixel measurements: element fill and outline (list membership), temporal
//! variance (tile tiebreak), chrome masks, and the screen-quad warp.

use glassrip_vision::board::StickyColor;
use glassrip_vision::BBox;
use image::{GrayImage, Rgb, RgbImage};

use crate::artifacts::{Point, ShapeClass};

/// Fill and outline measured for one element box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementPixels {
    pub fill_rgb: [u8; 3],
    pub hue: f64,
    pub saturation: f64,
    pub value: f64,
    /// Share of edge samples where an outline differs from the fill.
    pub outline_fraction: f64,
}

/// Thresholds for [`classify_shape`].
#[derive(
    Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct ShapeThresholds {
    /// Fill saturation at or above which a box is a filled note.
    pub filled_min_saturation: f64,
    /// Fill value at or above which a filled note is bright enough to be a note.
    pub filled_min_value: f64,
    /// Fill saturation below which a box counts as light (unfilled).
    pub light_max_saturation: f64,
    pub light_min_value: f64,
    /// Outline fraction needed for an outlined box.
    pub outline_min_fraction: f64,
    /// Luma difference that counts as an outline.
    pub outline_min_contrast: f64,
    /// Background saturation behind OCR text at or above which the text sits on a card.
    pub text_bg_min_saturation: f64,
}

impl Default for ShapeThresholds {
    fn default() -> Self {
        Self {
            filled_min_saturation: 0.18,
            filled_min_value: 0.45,
            light_max_saturation: 0.10,
            light_min_value: 0.70,
            outline_min_fraction: 0.6,
            outline_min_contrast: 35.0,
            text_bg_min_saturation: 0.08,
        }
    }
}

fn luma(p: &Rgb<u8>) -> f64 {
    0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2])
}

/// HSV of an RGB color: hue in degrees, saturation and value in [0, 1].
pub fn hsv(rgb: [u8; 3]) -> (f64, f64, f64) {
    let [r, g, b] = rgb.map(|c| f64::from(c) / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / d).rem_euclid(6.0))
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    let s = if max == 0.0 { 0.0 } else { d / max };
    (h, s, max)
}

fn median_u8(v: &mut [u8]) -> u8 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

fn clamp_box(b: &BBox, w: u32, h: u32) -> Option<(u32, u32, u32, u32)> {
    let x0 = b.x1.max(0.0).floor() as u32;
    let y0 = b.y1.max(0.0).floor() as u32;
    let x1 = (b.x2.ceil().max(0.0) as u32).min(w);
    let y1 = (b.y2.ceil().max(0.0) as u32).min(h);
    (x1 > x0 + 2 && y1 > y0 + 2).then_some((x0, y0, x1, y1))
}

/// Measure fill (median of the brighter interior pixels, so text is ignored)
/// and outline (darkest pixel across each edge vs the fill) inside `bbox`.
pub fn measure(img: &RgbImage, bbox: &BBox, t: &ShapeThresholds) -> Option<ElementPixels> {
    let (x0, y0, x1, y1) = clamp_box(bbox, img.width(), img.height())?;
    let (w, h) = (x1 - x0, y1 - y0);
    // Interior: inner 70%.
    let (ix0, iy0) = (x0 + w * 15 / 100, y0 + h * 15 / 100);
    let (ix1, iy1) = (x1 - w * 15 / 100, y1 - h * 15 / 100);
    let mut px: Vec<Rgb<u8>> = Vec::new();
    for y in iy0..iy1.max(iy0 + 1) {
        for x in ix0..ix1.max(ix0 + 1) {
            if x < img.width() && y < img.height() {
                px.push(*img.get_pixel(x, y));
            }
        }
    }
    if px.is_empty() {
        return None;
    }
    // Background is the brighter majority; drop the darkest 35% (text strokes).
    px.sort_by(|a, b| luma(a).total_cmp(&luma(b)));
    let keep = &px[px.len() * 35 / 100..];
    let mut ch: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for p in keep {
        for c in 0..3 {
            ch[c].push(p[c]);
        }
    }
    let fill = [
        median_u8(&mut ch[0]),
        median_u8(&mut ch[1]),
        median_u8(&mut ch[2]),
    ];
    let fill_luma = luma(&Rgb(fill));
    let (hue, saturation, value) = hsv(fill);

    // Outline: across each edge, scan a band of +-6% of the box size (at least
    // 3 px) and compare the most different pixel to the fill.
    let band_x = (w * 6 / 100).max(3);
    let band_y = (h * 6 / 100).max(3);
    let mut hits = 0usize;
    let mut total = 0usize;
    let differs = |p: &Rgb<u8>| -> bool {
        let dl = (luma(p) - fill_luma).abs();
        let dc: f64 = (0..3)
            .map(|c| (f64::from(p[c]) - f64::from(fill[c])).abs())
            .fold(0.0, f64::max);
        dl >= t.outline_min_contrast || dc >= t.outline_min_contrast * 1.5
    };
    let samples = 16u32;
    for k in 1..samples {
        let sx = x0 + w * k / samples;
        let sy = y0 + h * k / samples;
        for edge_y in [y0, y1.saturating_sub(1)] {
            total += 1;
            let lo = edge_y.saturating_sub(band_y);
            let hi = (edge_y + band_y).min(img.height().saturating_sub(1));
            if (lo..=hi).any(|y| differs(img.get_pixel(sx.min(img.width() - 1), y))) {
                hits += 1;
            }
        }
        for edge_x in [x0, x1.saturating_sub(1)] {
            total += 1;
            let lo = edge_x.saturating_sub(band_x);
            let hi = (edge_x + band_x).min(img.width().saturating_sub(1));
            if (lo..=hi).any(|x| differs(img.get_pixel(x, sy.min(img.height() - 1)))) {
                hits += 1;
            }
        }
    }
    Some(ElementPixels {
        fill_rgb: fill,
        hue,
        saturation,
        value,
        outline_fraction: if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        },
    })
}

/// Background behind a text box: the box padded by a third of its height,
/// median of the brighter 65% of pixels (text strokes are dropped).
pub fn text_background(img: &RgbImage, text_box: &BBox) -> Option<ElementPixels> {
    let pad = text_box.height().max(1.0) * 0.35;
    let b = BBox::new(
        text_box.x1 - pad,
        text_box.y1 - pad,
        text_box.x2 + pad,
        text_box.y2 + pad,
    );
    let (x0, y0, x1, y1) = clamp_box(&b, img.width(), img.height())?;
    let mut px: Vec<Rgb<u8>> = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            px.push(*img.get_pixel(x, y));
        }
    }
    px.sort_by(|a, b| luma(a).total_cmp(&luma(b)));
    let keep = &px[px.len() * 35 / 100..];
    let mut ch: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for p in keep {
        for c in 0..3 {
            ch[c].push(p[c]);
        }
    }
    let fill = [
        median_u8(&mut ch[0]),
        median_u8(&mut ch[1]),
        median_u8(&mut ch[2]),
    ];
    let (hue, saturation, value) = hsv(fill);
    Some(ElementPixels {
        fill_rgb: fill,
        hue,
        saturation,
        value,
        outline_fraction: 0.0,
    })
}

/// Card class from the background behind an element's OCR text: colored
/// (green or other) or plain.
pub fn classify_text_background(m: &ElementPixels, t: &ShapeThresholds) -> ShapeClass {
    if m.saturation >= t.text_bg_min_saturation && m.value >= t.filled_min_value {
        if (95.0..=170.0).contains(&m.hue) {
            ShapeClass::GreenTag
        } else {
            ShapeClass::FilledSticky
        }
    } else {
        ShapeClass::Unclear
    }
}

/// Shape class from measured pixels.
pub fn classify_shape(m: &ElementPixels, t: &ShapeThresholds) -> ShapeClass {
    if m.saturation >= t.filled_min_saturation && m.value >= t.filled_min_value {
        if (75.0..=165.0).contains(&m.hue) {
            ShapeClass::GreenTag
        } else {
            ShapeClass::FilledSticky
        }
    } else if m.saturation <= t.light_max_saturation
        && m.value >= t.light_min_value
        && m.outline_fraction >= t.outline_min_fraction
    {
        ShapeClass::OutlinedBox
    } else {
        ShapeClass::Unclear
    }
}

/// Sticky color name from a measured fill.
pub fn sticky_color(m: &ElementPixels) -> StickyColor {
    if m.saturation < 0.12 {
        return StickyColor::White;
    }
    match m.hue {
        h if !(15.0..345.0).contains(&h) => StickyColor::Pink,
        h if h < 40.0 => StickyColor::Orange,
        h if h < 75.0 => StickyColor::Yellow,
        h if h < 165.0 => StickyColor::Green,
        h if h < 255.0 => StickyColor::Blue,
        h if h < 300.0 => StickyColor::Purple,
        _ => StickyColor::Pink,
    }
}

/// Paint masks with `fill` (source pixels relative to the crop origin).
pub fn paint_masks(img: &mut RgbImage, boxes: &[BBox], fill: Rgb<u8>) {
    for b in boxes {
        if let Some((x0, y0, x1, y1)) = clamp_box(b, img.width(), img.height()) {
            for y in y0..y1 {
                for x in x0..x1 {
                    img.put_pixel(x, y, fill);
                }
            }
        }
    }
}

/// Median color of an image's border (a background estimate for masks).
pub fn border_median(img: &RgbImage) -> Rgb<u8> {
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return Rgb([255, 255, 255]);
    }
    let mut ch: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut push = |p: &Rgb<u8>| {
        for c in 0..3 {
            ch[c].push(p[c]);
        }
    };
    for x in (0..w).step_by(4) {
        push(img.get_pixel(x, 0));
        push(img.get_pixel(x, h - 1));
    }
    for y in (0..h).step_by(4) {
        push(img.get_pixel(0, y));
        push(img.get_pixel(w - 1, y));
    }
    Rgb([
        median_u8(&mut ch[0]),
        median_u8(&mut ch[1]),
        median_u8(&mut ch[2]),
    ])
}

/// Solve the homography mapping `src` corners to `dst` corners (4 points, DLT).
fn homography(src: &[Point; 4], dst: &[Point; 4]) -> Option<[f64; 9]> {
    let mut a = [[0f64; 9]; 8];
    for i in 0..4 {
        let ([x, y], [u, v]) = (src[i], dst[i]);
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    // Gaussian elimination with partial pivoting on the 8x8 system.
    for col in 0..8 {
        let piv = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        let pivot_row = a[col];
        for (r, row) in a.iter_mut().enumerate() {
            if r != col {
                let f = row[col] / pivot_row[col];
                for (v, p) in row.iter_mut().zip(pivot_row.iter()).skip(col) {
                    *v -= f * p;
                }
            }
        }
    }
    let mut h = [0f64; 9];
    for i in 0..8 {
        h[i] = a[i][8] / a[i][i];
    }
    h[8] = 1.0;
    Some(h)
}

/// Warp the quadrilateral `quad` (image pixels, TL TR BR BL) of `img` to an
/// `out_w` x `out_h` rectangle with bilinear sampling.
pub fn warp_quad(img: &GrayImage, quad: &[Point; 4], out_w: u32, out_h: u32) -> Option<GrayImage> {
    let (w, h) = (f64::from(out_w), f64::from(out_h));
    let rect = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    // Map output pixels back to the source.
    let m = homography(&rect, quad)?;
    let mut out = GrayImage::new(out_w, out_h);
    let (iw, ih) = (img.width() as f64, img.height() as f64);
    for y in 0..out_h {
        for x in 0..out_w {
            let (fx, fy) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            let d = m[6] * fx + m[7] * fy + m[8];
            if d.abs() < 1e-12 {
                continue;
            }
            let sx = (m[0] * fx + m[1] * fy + m[2]) / d - 0.5;
            let sy = (m[3] * fx + m[4] * fy + m[5]) / d - 0.5;
            if sx < 0.0 || sy < 0.0 || sx >= iw - 1.0 || sy >= ih - 1.0 {
                continue;
            }
            let (x0, y0) = (sx.floor() as u32, sy.floor() as u32);
            let (ax, ay) = (sx - f64::from(x0), sy - f64::from(y0));
            let p = |xx: u32, yy: u32| f64::from(img.get_pixel(xx, yy)[0]);
            let v = p(x0, y0) * (1.0 - ax) * (1.0 - ay)
                + p(x0 + 1, y0) * ax * (1.0 - ay)
                + p(x0, y0 + 1) * (1.0 - ax) * ay
                + p(x0 + 1, y0 + 1) * ax * ay;
            out.put_pixel(x, y, image::Luma([v.round().clamp(0.0, 255.0) as u8]));
        }
    }
    Some(out)
}

/// Per-pixel temporal standard deviation over equally sized gray frames.
pub fn temporal_std(frames: &[GrayImage]) -> Option<(Vec<f32>, u32, u32)> {
    let first = frames.first()?;
    let (w, h) = (first.width(), first.height());
    if frames.len() < 2 || frames.iter().any(|f| f.width() != w || f.height() != h) {
        return None;
    }
    let n = frames.len() as f32;
    let mut out = vec![0f32; (w * h) as usize];
    for (i, o) in out.iter_mut().enumerate() {
        let (mut s, mut s2) = (0f32, 0f32);
        for f in frames {
            let v = f32::from(f.as_raw()[i]);
            s += v;
            s2 += v * v;
        }
        let mean = s / n;
        *o = (s2 / n - mean * mean).max(0.0).sqrt();
    }
    Some((out, w, h))
}

/// Mean of `map` inside `b` (map pixels) and outside it.
pub fn inside_outside_mean(map: &[f32], w: u32, h: u32, b: &BBox) -> Option<(f64, f64)> {
    let (x0, y0, x1, y1) = clamp_box(b, w, h)?;
    let (mut si, mut ni, mut so, mut no) = (0f64, 0usize, 0f64, 0usize);
    for y in 0..h {
        for x in 0..w {
            let v = f64::from(map[(y * w + x) as usize]);
            if x >= x0 && x < x1 && y >= y0 && y < y1 {
                si += v;
                ni += 1;
            } else {
                so += v;
                no += 1;
            }
        }
    }
    (ni > 0 && no > 0).then(|| (si / ni as f64, so / no as f64))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn canvas() -> RgbImage {
        RgbImage::from_pixel(300, 200, Rgb([250, 250, 250]))
    }

    fn fill(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, c: [u8; 3]) {
        for y in y0..y1 {
            for x in x0..x1 {
                img.put_pixel(x, y, Rgb(c));
            }
        }
    }

    #[test]
    fn outlined_box_vs_filled_sticky_vs_green_tag() {
        let t = ShapeThresholds::default();
        let mut img = canvas();
        // Outlined node with dark text.
        fill(&mut img, 20, 20, 120, 80, [30, 30, 30]);
        fill(&mut img, 23, 23, 117, 77, [255, 255, 255]);
        fill(&mut img, 50, 45, 90, 52, [20, 20, 20]);
        // Yellow sticky.
        fill(&mut img, 150, 20, 230, 90, [250, 235, 120]);
        fill(&mut img, 165, 50, 215, 56, [30, 30, 30]);
        // Green tag.
        fill(&mut img, 150, 120, 210, 160, [110, 200, 120]);

        let node = measure(&img, &BBox::new(20.0, 20.0, 120.0, 80.0), &t).unwrap();
        assert_eq!(
            classify_shape(&node, &t),
            ShapeClass::OutlinedBox,
            "{node:?}"
        );
        let sticky = measure(&img, &BBox::new(150.0, 20.0, 230.0, 90.0), &t).unwrap();
        assert_eq!(
            classify_shape(&sticky, &t),
            ShapeClass::FilledSticky,
            "{sticky:?}"
        );
        assert_eq!(sticky_color(&sticky), StickyColor::Yellow);
        let tag = measure(&img, &BBox::new(150.0, 120.0, 210.0, 160.0), &t).unwrap();
        assert_eq!(classify_shape(&tag, &t), ShapeClass::GreenTag);
        // Plain background with no outline is unclear.
        let bg = measure(&img, &BBox::new(20.0, 120.0, 100.0, 180.0), &t).unwrap();
        assert_eq!(classify_shape(&bg, &t), ShapeClass::Unclear);
        assert!(measure(&img, &BBox::new(500.0, 0.0, 600.0, 10.0), &t).is_none());
    }

    #[test]
    fn masks_and_border() {
        let mut img = canvas();
        paint_masks(
            &mut img,
            &[BBox::new(10.0, 10.0, 20.0, 20.0)],
            Rgb([0, 0, 0]),
        );
        assert_eq!(img.get_pixel(15, 15), &Rgb([0, 0, 0]));
        assert_eq!(border_median(&img), Rgb([250, 250, 250]));
    }

    #[test]
    fn warp_identity_quad_keeps_pixels() {
        let mut g = GrayImage::new(40, 30);
        for (x, y, p) in g.enumerate_pixels_mut() {
            *p = image::Luma([((x * 5 + y * 3) % 256) as u8]);
        }
        let quad = [[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0]];
        let out = warp_quad(&g, &quad, 40, 30).unwrap();
        assert_eq!(out.get_pixel(10, 10), g.get_pixel(10, 10));
    }

    #[test]
    fn variance_finds_moving_region() {
        let mut frames = Vec::new();
        for k in 0..4u8 {
            let mut g = GrayImage::from_pixel(50, 40, image::Luma([200]));
            for y in 5..15 {
                for x in 30..45 {
                    g.put_pixel(x, y, image::Luma([k * 60]));
                }
            }
            frames.push(g);
        }
        let (map, w, h) = temporal_std(&frames).unwrap();
        let (inside, outside) =
            inside_outside_mean(&map, w, h, &BBox::new(30.0, 5.0, 45.0, 15.0)).unwrap();
        assert!(inside > 40.0 && outside < 1.0);
        assert!(temporal_std(&frames[..1]).is_none());
    }
}
