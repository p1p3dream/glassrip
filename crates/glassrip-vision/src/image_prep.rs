//! Image sizing, token budgeting, and encoding for vision requests.
//!
//! Sizing rule (board reads): the canvas crop is taken from the native frame.
//! If its long edge is at least [`LOW_RES_THRESHOLD`] px it is resized to
//! [`BOARD_LONG_EDGE`] px long edge. Otherwise it is upscaled by
//! [`LOW_RES_UPSCALE`] and flagged `low_res`. Either way the result must fit the
//! encoder's image-token cap; images that would exceed it are shrunk further
//! and flagged `token_capped`.

use std::io::Cursor;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};

use crate::error::{Result, VisionError};

/// Side of one vision-encoder patch group in pixels (qwen2.5vl merges 14 px patches 2x2).
pub const PATCH_PX: u32 = 28;
/// Ollama's image token cap for qwen2.5vl.
pub const MAX_IMAGE_TOKENS: u32 = 4096;
/// Target long edge for board reads.
pub const BOARD_LONG_EDGE: u32 = 1920;
/// Crops whose long edge is below this are treated as low resolution.
pub const LOW_RES_THRESHOLD: u32 = 1280;
/// Upscale factor applied to low resolution crops.
pub const LOW_RES_UPSCALE: f64 = 1.5;
/// Long edge of the classification thumbnail.
pub const THUMBNAIL_LONG_EDGE: u32 = 768;
/// JPEG quality used for every request image.
pub const JPEG_QUALITY: u8 = 90;

/// Image tokens the encoder will spend on a `width` x `height` image.
pub fn image_tokens(width: u32, height: u32) -> u32 {
    width.div_ceil(PATCH_PX) * height.div_ceil(PATCH_PX)
}

/// A single JPEG image, base64 encoded, that is guaranteed to fit the token cap.
///
/// The only constructors check the cap, so a value of this type is always sendable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedImage {
    base64_jpeg: String,
    width: u32,
    height: u32,
}

impl EncodedImage {
    /// Encode an image as JPEG and base64. Fails if it exceeds [`MAX_IMAGE_TOKENS`].
    pub fn encode(image: &DynamicImage) -> Result<Self> {
        let (width, height) = image.dimensions();
        check_token_cap(width, height)?;
        let rgb = image.to_rgb8();
        let mut buf = Cursor::new(Vec::new());
        let encoder = JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);
        rgb.write_with_encoder(encoder)?;
        Ok(Self {
            base64_jpeg: STANDARD.encode(buf.into_inner()),
            width,
            height,
        })
    }

    /// Base64 JPEG payload as sent to the server.
    pub fn base64(&self) -> &str {
        &self.base64_jpeg
    }

    /// Width in pixels of the encoded image.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels of the encoded image.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Image tokens this image costs.
    pub fn tokens(&self) -> u32 {
        image_tokens(self.width, self.height)
    }
}

fn check_token_cap(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(VisionError::Config(format!(
            "image has zero size ({width}x{height})"
        )));
    }
    let tokens = image_tokens(width, height);
    if tokens > MAX_IMAGE_TOKENS {
        return Err(VisionError::ImageTooManyTokens {
            width,
            height,
            tokens,
            max: MAX_IMAGE_TOKENS,
        });
    }
    Ok(())
}

/// Planned output size for a source image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizePlan {
    pub width: u32,
    pub height: u32,
    /// Source was below [`LOW_RES_THRESHOLD`] and was upscaled.
    pub low_res: bool,
    /// The target size was reduced further to respect [`MAX_IMAGE_TOKENS`].
    pub token_capped: bool,
}

impl SizePlan {
    /// Sent-image pixels per source pixel along x.
    pub fn scale_x(&self, source_width: u32) -> f64 {
        f64::from(self.width) / f64::from(source_width.max(1))
    }

    /// Sent-image pixels per source pixel along y.
    pub fn scale_y(&self, source_height: u32) -> f64 {
        f64::from(self.height) / f64::from(source_height.max(1))
    }
}

fn scaled(value: u32, factor: f64) -> u32 {
    // Rounded and clamped to at least 1 px; inputs are image sizes, far below u32::MAX.
    let v = (f64::from(value) * factor).round();
    if v < 1.0 {
        1
    } else if v > f64::from(u32::MAX) {
        u32::MAX
    } else {
        v as u32
    }
}

/// Shrink `(w, h)` by the largest factor <= 1 that keeps the token cost within the cap.
fn fit_token_cap(width: u32, height: u32) -> (u32, u32, bool) {
    if image_tokens(width, height) <= MAX_IMAGE_TOKENS {
        return (width, height, false);
    }
    // Start from the area estimate, then step down until the ceil-rounded cost fits.
    let budget_px = f64::from(MAX_IMAGE_TOKENS) * f64::from(PATCH_PX * PATCH_PX);
    let mut factor = (budget_px / (f64::from(width) * f64::from(height))).sqrt();
    loop {
        let (w, h) = (scaled(width, factor), scaled(height, factor));
        if image_tokens(w, h) <= MAX_IMAGE_TOKENS || factor <= 0.01 {
            return (w, h, true);
        }
        factor *= 0.99;
    }
}

/// Apply the board-read sizing rule to a crop of `width` x `height`.
pub fn plan_board_size(width: u32, height: u32) -> SizePlan {
    let long = width.max(height);
    let (w, h, low_res) = if long >= LOW_RES_THRESHOLD {
        let factor = f64::from(BOARD_LONG_EDGE) / f64::from(long);
        (scaled(width, factor), scaled(height, factor), false)
    } else {
        (
            scaled(width, LOW_RES_UPSCALE),
            scaled(height, LOW_RES_UPSCALE),
            true,
        )
    };
    let (width, height, token_capped) = fit_token_cap(w, h);
    SizePlan {
        width,
        height,
        low_res,
        token_capped,
    }
}

/// Plan a thumbnail with the given long edge (always resized to exactly that long edge).
pub fn plan_thumbnail_size(width: u32, height: u32, long_edge: u32) -> SizePlan {
    let long = width.max(height).max(1);
    let factor = f64::from(long_edge) / f64::from(long);
    let (w, h, token_capped) = fit_token_cap(scaled(width, factor), scaled(height, factor));
    SizePlan {
        width: w,
        height: h,
        low_res: false,
        token_capped,
    }
}

/// An image ready to send plus the geometry needed to map model coordinates back.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub image: EncodedImage,
    pub plan: SizePlan,
    /// Size of the source crop the plan was computed from.
    pub source_width: u32,
    pub source_height: u32,
}

impl PreparedImage {
    /// Map a point in sent-image pixels back to source (canvas) pixels.
    pub fn to_source(&self, x: f64, y: f64) -> (f64, f64) {
        (
            x / self.plan.scale_x(self.source_width),
            y / self.plan.scale_y(self.source_height),
        )
    }
}

fn prepare_with(image: &DynamicImage, plan: SizePlan) -> Result<PreparedImage> {
    let (sw, sh) = image.dimensions();
    let resized = if (plan.width, plan.height) == (sw, sh) {
        image.clone()
    } else {
        image.resize_exact(plan.width, plan.height, FilterType::Lanczos3)
    };
    Ok(PreparedImage {
        image: EncodedImage::encode(&resized)?,
        plan,
        source_width: sw,
        source_height: sh,
    })
}

/// Resize a canvas crop per the board-read rule and encode it.
pub fn prepare_board_image(canvas_crop: &DynamicImage) -> Result<PreparedImage> {
    let (w, h) = canvas_crop.dimensions();
    prepare_with(canvas_crop, plan_board_size(w, h))
}

/// Resize to exactly `long_edge` px on the long side (still within the token
/// cap) and encode. Used for board tiles, which are always sent at full size.
pub fn prepare_long_edge(image: &DynamicImage, long_edge: u32) -> Result<PreparedImage> {
    let (w, h) = image.dimensions();
    prepare_with(image, plan_thumbnail_size(w, h, long_edge))
}

/// Resize a frame to the classification thumbnail and encode it.
pub fn prepare_thumbnail(frame: &DynamicImage) -> Result<PreparedImage> {
    let (w, h) = frame.dimensions();
    prepare_with(frame, plan_thumbnail_size(w, h, THUMBNAIL_LONG_EDGE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn token_formula_matches_spec_examples() {
        assert_eq!(image_tokens(28, 28), 1);
        assert_eq!(image_tokens(29, 28), 2);
        // 1920x1080: 69 x 39 = 2691, "about 2,700".
        assert_eq!(image_tokens(1920, 1080), 2691);
    }

    #[test]
    fn large_crop_resizes_to_1920_long_edge() {
        let p = plan_board_size(3840, 2160);
        assert_eq!((p.width, p.height), (1920, 1080));
        assert!(!p.low_res && !p.token_capped);
    }

    #[test]
    fn crop_at_threshold_upscales_at_most_1_5x() {
        let p = plan_board_size(1280, 720);
        assert_eq!((p.width, p.height), (1920, 1080));
        assert!(!p.low_res);
    }

    #[test]
    fn small_crop_upscales_1_5x_and_flags_low_res() {
        let p = plan_board_size(1000, 600);
        assert_eq!((p.width, p.height), (1500, 900));
        assert!(p.low_res);
        let p = plan_board_size(1279, 400);
        assert_eq!(p.width, 1919);
        assert!(p.low_res);
    }

    #[test]
    fn portrait_uses_long_edge() {
        let p = plan_board_size(1080, 1920);
        assert_eq!((p.width, p.height), (1080, 1920));
    }

    #[test]
    fn square_crop_is_shrunk_to_token_cap() {
        // 1920x1920 would cost 69*69 = 4761 tokens.
        let p = plan_board_size(2400, 2400);
        assert!(p.token_capped);
        assert!(image_tokens(p.width, p.height) <= MAX_IMAGE_TOKENS);
        assert!(p.width >= 1700, "shrunk too far: {}", p.width);
    }

    #[test]
    fn thumbnail_is_768_long_edge() {
        let p = plan_thumbnail_size(3840, 2160, THUMBNAIL_LONG_EDGE);
        assert_eq!((p.width, p.height), (768, 432));
        let p = plan_thumbnail_size(400, 300, THUMBNAIL_LONG_EDGE);
        assert_eq!((p.width, p.height), (768, 576));
    }

    #[test]
    fn encode_rejects_over_cap_and_zero_size() {
        let big = DynamicImage::ImageRgb8(RgbImage::new(1920, 1920));
        assert!(matches!(
            EncodedImage::encode(&big),
            Err(VisionError::ImageTooManyTokens { tokens: 4761, .. })
        ));
        let empty = DynamicImage::ImageRgb8(RgbImage::new(0, 10));
        assert!(matches!(
            EncodedImage::encode(&empty),
            Err(VisionError::Config(_))
        ));
    }

    #[test]
    fn prepare_board_image_maps_coordinates_back() -> Result<()> {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(640, 360, Rgb([200, 30, 30])));
        let prepared = prepare_board_image(&img)?;
        assert!(prepared.plan.low_res);
        assert_eq!(prepared.image.width(), 960);
        assert_eq!(prepared.image.height(), 540);
        assert!(!prepared.image.base64().is_empty());
        let (x, y) = prepared.to_source(960.0, 540.0);
        assert!((x - 640.0).abs() < 1e-6 && (y - 360.0).abs() < 1e-6);
        Ok(())
    }
}
