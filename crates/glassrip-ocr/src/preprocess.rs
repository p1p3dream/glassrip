//! Input tensors for the detector and the recognizer.
//!
//! PaddleOCR models are trained on BGR images, so channels are written in BGR
//! order. Detection uses ImageNet mean and std; recognition maps pixels to
//! `[-1, 1]`.

use image::imageops::FilterType;
use image::RgbImage;

use crate::PixelBox;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Detector input: NCHW data plus the scales back to source pixels.
#[derive(Debug, Clone)]
pub struct DetInput {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub scale_x: f64,
    pub scale_y: f64,
}

fn round32(v: f64) -> u32 {
    let r = ((v / 32.0).round() * 32.0).max(32.0);
    if r > f64::from(u32::MAX) {
        u32::MAX
    } else {
        r as u32
    }
}

/// Detector inputs are padded so their short side is a multiple of this.
pub const DET_BUCKET: u32 = 128;

fn ceil_to(v: u32, step: u32) -> u32 {
    v.div_ceil(step).saturating_mul(step)
}

/// Resize so the long side is at most `max_side` (both sides multiples of
/// 32), then pad at the bottom and right to multiples of [`DET_BUCKET`]. Few
/// distinct shapes keep ONNX Runtime's CUDA arena from growing with every new
/// keyframe size; padding does not move boxes (the scales refer to the
/// resized content, and padded pixels are the normalized zero).
pub fn det_input(image: &RgbImage, max_side: u32) -> DetInput {
    let (w, h) = (image.width().max(1), image.height().max(1));
    let ratio = (f64::from(max_side) / f64::from(w.max(h))).min(1.0);
    let rw = round32(f64::from(w) * ratio);
    let rh = round32(f64::from(h) * ratio);
    let (pw, ph) = (ceil_to(rw, DET_BUCKET), ceil_to(rh, DET_BUCKET));
    let resized = image::imageops::resize(image, rw, rh, FilterType::Triangle);
    let (pw_us, ph_us) = (pw as usize, ph as usize);
    let plane = pw_us * ph_us;
    let mut data = vec![0f32; 3 * plane];
    for (x, y, px) in resized.enumerate_pixels() {
        let i = y as usize * pw_us + x as usize;
        // BGR order: channel 0 is blue.
        for (c, src) in [2usize, 1, 0].into_iter().enumerate() {
            data[c * plane + i] = (f32::from(px[src]) / 255.0 - MEAN[c]) / STD[c];
        }
    }
    DetInput {
        data,
        width: pw_us,
        height: ph_us,
        scale_x: f64::from(w) / f64::from(rw),
        scale_y: f64::from(h) / f64::from(rh),
    }
}

/// Width a crop gets at the recognizer's input height.
pub fn rec_width(crop_w: u32, crop_h: u32, rec_height: u32, max_width: u32) -> u32 {
    let w = (f64::from(crop_w) * f64::from(rec_height) / f64::from(crop_h.max(1))).ceil();
    (w as u32).clamp(8, max_width.max(8))
}

/// Crop `bbox` (source pixels) out of `image`, rounding outward.
pub fn crop(image: &RgbImage, bbox: &PixelBox) -> Option<RgbImage> {
    let x0 = bbox.x1.floor().max(0.0) as u32;
    let y0 = bbox.y1.floor().max(0.0) as u32;
    let x1 = (bbox.x2.ceil() as u32).min(image.width());
    let y1 = (bbox.y2.ceil() as u32).min(image.height());
    (x1 > x0 && y1 > y0)
        .then(|| image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image())
}

/// Recognizer batch: crops resized to `rec_height`, padded with zeros to the
/// widest crop. Returns NCHW data and the padded width.
pub fn rec_batch(crops: &[&RgbImage], rec_height: u32, max_width: u32) -> (Vec<f32>, usize) {
    let widths: Vec<u32> = crops
        .iter()
        .map(|c| rec_width(c.width(), c.height(), rec_height, max_width))
        .collect();
    let padded = widths.iter().copied().max().unwrap_or(8) as usize;
    let h = rec_height as usize;
    let plane = h * padded;
    let mut data = vec![0f32; crops.len() * 3 * plane];
    for (n, (crop, &w)) in crops.iter().zip(&widths).enumerate() {
        let resized = image::imageops::resize(*crop, w, rec_height, FilterType::Triangle);
        let base = n * 3 * plane;
        for (x, y, px) in resized.enumerate_pixels() {
            let i = y as usize * padded + x as usize;
            for (c, src) in [2usize, 1, 0].into_iter().enumerate() {
                data[base + c * plane + i] = f32::from(px[src]) / 127.5 - 1.0;
            }
        }
    }
    (data, padded)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn det_input_is_multiple_of_32_and_bgr() {
        let img = RgbImage::from_pixel(100, 50, Rgb([255, 0, 0]));
        let d = det_input(&img, 1920);
        assert_eq!((d.width % 32, d.height % 32), (0, 0));
        assert_eq!(d.data.len(), 3 * d.width * d.height);
        // Channel 0 is blue (0), channel 2 is red (255).
        let plane = d.width * d.height;
        assert!((d.data[0] - (0.0 - MEAN[0]) / STD[0]).abs() < 1e-5);
        assert!((d.data[2 * plane] - (1.0 - MEAN[2]) / STD[2]).abs() < 1e-5);
        // Content 96 x 64 (multiples of 32), padded to 128 x 128.
        assert_eq!((d.width, d.height), (128, 128));
        assert!((d.scale_x * 96.0 - 100.0).abs() < 1e-9);
    }

    #[test]
    fn det_inputs_fall_into_few_buckets() {
        let shapes: std::collections::BTreeSet<(usize, usize)> = [
            (1920, 1080),
            (1680, 832),
            (1676, 796),
            (1680, 830),
            (1594, 810),
            (1664, 832),
            (1642, 832),
            (1610, 828),
        ]
        .iter()
        .map(|&(w, h)| {
            let d = det_input(&RgbImage::new(w, h), 1920);
            assert_eq!((d.width % 128, d.height % 128), (0, 0));
            (d.width, d.height)
        })
        .collect();
        assert!(shapes.len() <= 3, "{shapes:?}");
        // Padding keeps the content scale.
        let d = det_input(&RgbImage::new(1680, 832), 1920);
        assert!((d.scale_x - 1680.0 / 1696.0).abs() < 1e-12);
        assert_eq!((d.width, d.height), (1792, 896));
    }

    #[test]
    fn det_input_limits_long_side() {
        let img = RgbImage::new(4000, 1000);
        let d = det_input(&img, 1920);
        assert!(d.width <= 1920 + 16);
    }

    #[test]
    fn rec_batch_pads_to_widest() {
        let a = RgbImage::from_pixel(40, 10, Rgb([255, 255, 255]));
        let b = RgbImage::from_pixel(10, 10, Rgb([0, 0, 0]));
        let (data, padded) = rec_batch(&[&a, &b], 48, 3200);
        assert_eq!(padded, 192);
        assert_eq!(data.len(), 2 * 3 * 48 * 192);
        assert!((data[0] - 1.0).abs() < 1e-6);
        // Padding of the narrow crop stays 0.
        let base = 3 * 48 * 192;
        assert_eq!(data[base + 100], 0.0);
        assert!((data[base] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn crop_rounds_outward_and_rejects_empty() {
        let img = RgbImage::new(20, 20);
        let c = crop(
            &img,
            &PixelBox {
                x1: 1.5,
                y1: 1.5,
                x2: 5.2,
                y2: 4.1,
            },
        );
        assert_eq!(c.map(|c| (c.width(), c.height())), Some((5, 4)));
        let none = crop(
            &img,
            &PixelBox {
                x1: 30.0,
                y1: 0.0,
                x2: 40.0,
                y2: 5.0,
            },
        );
        assert!(none.is_none());
    }
}
