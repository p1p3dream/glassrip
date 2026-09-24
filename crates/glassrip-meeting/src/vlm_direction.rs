//! VLM fallback for edge direction (spec 6.11, edge direction step 2).
//!
//! Used only for edges whose pixel vote is inconclusive across all keyframes. One crop
//! shows the whole connector: both node boxes plus padding, with box A outlined in red
//! and box B in blue so the model knows which line is meant. The model reports where
//! the arrowhead end of that line is in image space (`left`, `right`, `up`, `down`,
//! `none`, `unclear`). The answer is mapped to an endpoint through the connector
//! geometry: the end positions come from the pixel check's termini when traced, else
//! from the facing border points of the two boxes. An answer along an axis on which
//! the two ends are not clearly separated is not mapped.

use glassrip_vision::backend::{GenerationOptions, VisionRequest};
use glassrip_vision::image_prep::EncodedImage;
use glassrip_vision::{BBox, VisionClient};
use image::imageops::FilterType;
use image::{DynamicImage, Rgb, RgbImage};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::direction::EndVerdict;
use crate::pixel_direction::{exit_point, EndEvidence};

/// Prompt for one connector crop. `{a}` and `{b}` are replaced by the box texts.
pub const CONNECTOR_PROMPT: &str = "\
The image is a crop of a whiteboard diagram. Box A (text: \"{a}\") is outlined in red and box B \
(text: \"{b}\") is outlined in blue. Look only at the connector line that joins box A and box B.

Where is the arrowhead on that line, in image directions? Answer with exactly one value:
- left: the arrowhead is at the left end of the line.
- right: the arrowhead is at the right end of the line.
- up: the arrowhead is at the top end of the line.
- down: the arrowhead is at the bottom end of the line.
- none: the line has no arrowhead.
- unclear: the line or its ends cannot be seen clearly.";

/// Image-space position of the arrowhead end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ArrowheadSide {
    /// Left end.
    Left,
    /// Right end.
    Right,
    /// Top end.
    Up,
    /// Bottom end.
    Down,
    /// No arrowhead.
    None,
    /// Cannot tell.
    Unclear,
}

/// Model reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArrowheadReply {
    /// The single answer.
    pub answer: ArrowheadSide,
}

/// VLM fallback settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VlmCheckParams {
    /// Padding around the two boxes, in canvas pixels.
    pub pad_px: f64,
    /// Long edge of the image sent to the model.
    pub send_long_edge: u32,
    /// Generation seed.
    pub seed: u64,
    /// Output token cap.
    pub num_predict: u32,
    /// Keyframes asked per inconclusive edge (highest vote weight first).
    pub frames_per_edge: usize,
    /// Minimum separation of the two ends along the answered axis, as a share of the
    /// distance between the ends (at least 8 px).
    pub min_axis_separation: f64,
}

impl Default for VlmCheckParams {
    fn default() -> Self {
        Self {
            pad_px: 24.0,
            send_long_edge: 1024,
            seed: 7,
            num_predict: 32,
            frames_per_edge: 1,
            min_axis_separation: 0.2,
        }
    }
}

/// VLM fallback result for one edge in one keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VlmEvidence {
    /// The model's answer.
    pub answer: ArrowheadSide,
    /// End positions used for the mapping (`src`, `dst`), in canvas pixels.
    pub src_point: (f64, f64),
    /// See `src_point`.
    pub dst_point: (f64, f64),
    /// Mapped verdict relative to `src -> dst` as read.
    pub verdict: EndVerdict,
}

/// Map an image-space answer to a verdict through the two end positions.
pub fn map_side(
    side: ArrowheadSide,
    src: (f64, f64),
    dst: (f64, f64),
    min_axis_separation: f64,
) -> EndVerdict {
    let (dx, dy) = (dst.0 - src.0, dst.1 - src.1);
    let min_sep = (min_axis_separation * (dx * dx + dy * dy).sqrt()).max(8.0);
    // `delta` is dst minus src along the axis; `dst_is_head` when dst is on the side.
    let pick = |delta: f64, dst_on_positive_side_is_head: bool| {
        if delta.abs() < min_sep {
            EndVerdict::Unknown
        } else if (delta > 0.0) == dst_on_positive_side_is_head {
            EndVerdict::Forward
        } else {
            EndVerdict::Reverse
        }
    };
    match side {
        ArrowheadSide::Right => pick(dx, true),
        ArrowheadSide::Left => pick(dx, false),
        ArrowheadSide::Down => pick(dy, true),
        ArrowheadSide::Up => pick(dy, false),
        ArrowheadSide::None => EndVerdict::NoArrowhead,
        ArrowheadSide::Unclear => EndVerdict::Unknown,
    }
}

/// One edge to ask about, in canvas pixels.
#[derive(Debug, Clone, Copy)]
pub struct EdgeEnds<'a> {
    /// Tail box as read.
    pub src: &'a BBox,
    /// Head box as read.
    pub dst: &'a BBox,
    /// Tail text.
    pub src_text: &'a str,
    /// Head text.
    pub dst_text: &'a str,
    /// Pixel terminus at the tail, when found.
    pub src_end: Option<&'a EndEvidence>,
    /// Pixel terminus at the head, when found.
    pub dst_end: Option<&'a EndEvidence>,
}

fn center(b: &BBox) -> (f64, f64) {
    ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
}

/// End positions: pixel termini when traced, else the facing border points.
pub fn end_points(e: &EdgeEnds<'_>) -> ((f64, f64), (f64, f64)) {
    let s = e
        .src_end
        .map(|t| (t.x, t.y))
        .unwrap_or_else(|| exit_point(e.src, center(e.dst)));
    let d = e
        .dst_end
        .map(|t| (t.x, t.y))
        .unwrap_or_else(|| exit_point(e.dst, center(e.src)));
    (s, d)
}

fn outline(img: &mut RgbImage, b: &BBox, ox: f64, oy: f64, color: Rgb<u8>) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let x1 = (b.x1 - ox).round() as i64;
    let x2 = (b.x2 - ox).round() as i64;
    let y1 = (b.y1 - oy).round() as i64;
    let y2 = (b.y2 - oy).round() as i64;
    let mut put = |x: i64, y: i64| {
        if x >= 0 && y >= 0 && x < w && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
    };
    for t in -3..=-1 {
        for x in x1 + t..=x2 - t {
            put(x, y1 + t);
            put(x, y2 - t);
        }
        for y in y1 + t..=y2 - t {
            put(x1 + t, y);
            put(x2 - t, y);
        }
    }
}

/// Crop covering both boxes (and so the connector between them), with the boxes
/// outlined, resized to at most `send_long_edge`.
pub fn connector_crop(
    image: &RgbImage,
    e: &EdgeEnds<'_>,
    params: &VlmCheckParams,
) -> Option<DynamicImage> {
    let (w, h) = (f64::from(image.width()), f64::from(image.height()));
    let (sp, dp) = end_points(e);
    let p = params.pad_px;
    let region = BBox::new(
        e.src.x1.min(e.dst.x1).min(sp.0).min(dp.0) - p,
        e.src.y1.min(e.dst.y1).min(sp.1).min(dp.1) - p,
        e.src.x2.max(e.dst.x2).max(sp.0).max(dp.0) + p,
        e.src.y2.max(e.dst.y2).max(sp.1).max(dp.1) + p,
    )
    .clamped(w, h);
    if !region.is_well_formed() {
        return None;
    }
    let (x0, y0) = (region.x1.floor(), region.y1.floor());
    let cw = (region.x2.ceil() - x0).max(1.0) as u32;
    let ch = (region.y2.ceil() - y0).max(1.0) as u32;
    let mut crop = image::imageops::crop_imm(image, x0 as u32, y0 as u32, cw, ch).to_image();
    outline(&mut crop, e.src, x0, y0, Rgb([230, 20, 20]));
    outline(&mut crop, e.dst, x0, y0, Rgb([20, 60, 230]));
    let long = cw.max(ch);
    let img = DynamicImage::ImageRgb8(crop);
    Some(if long > params.send_long_edge {
        let s = f64::from(params.send_long_edge) / f64::from(long);
        let nw = ((f64::from(cw) * s).round() as u32).max(1);
        let nh = ((f64::from(ch) * s).round() as u32).max(1);
        img.resize_exact(nw, nh, FilterType::Lanczos3)
    } else {
        img
    })
}

/// Build the request for one connector crop.
pub fn connector_request(
    crop: &DynamicImage,
    src_text: &str,
    dst_text: &str,
    params: &VlmCheckParams,
) -> glassrip_vision::Result<VisionRequest> {
    let prompt = CONNECTOR_PROMPT
        .replace("{a}", src_text)
        .replace("{b}", dst_text);
    VisionRequest::for_output::<ArrowheadReply>(
        &prompt,
        EncodedImage::encode(crop)?,
        GenerationOptions {
            seed: params.seed,
            num_predict: params.num_predict,
        },
    )
}

/// Ask the model about one edge (box A = `src`, box B = `dst`).
pub async fn check_connector(
    client: &VisionClient,
    image: &RgbImage,
    ends: EdgeEnds<'_>,
    params: &VlmCheckParams,
    cancel: CancellationToken,
) -> glassrip_vision::Result<Option<VlmEvidence>> {
    let Some(crop) = connector_crop(image, &ends, params) else {
        return Ok(None);
    };
    let req = connector_request(&crop, ends.src_text, ends.dst_text, params)?;
    let (reply, _raw): (ArrowheadReply, _) = client.infer_typed(req, cancel).await?;
    let (sp, dp) = end_points(&ends);
    Ok(Some(VlmEvidence {
        answer: reply.answer,
        src_point: sp,
        dst_point: dp,
        verdict: map_side(reply.answer, sp, dp, params.min_axis_separation),
    }))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn schema_is_the_image_space_enum() {
        let s = glassrip_vision::OutputSchema::for_type::<ArrowheadReply>().unwrap();
        let text = s.text();
        for v in ["left", "right", "up", "down", "none", "unclear"] {
            assert!(text.contains(&format!("\"{v}\"")), "{text}");
        }
        assert!(!text.contains("points_to"));
        assert!(serde_json::from_str::<ArrowheadReply>(r#"{"answer":"points_to_A"}"#).is_err());
    }

    #[test]
    fn sides_map_through_geometry() {
        use ArrowheadSide::*;
        // src at left, dst at right.
        let (s, d) = ((100.0, 200.0), (400.0, 210.0));
        assert_eq!(map_side(Right, s, d, 0.2), EndVerdict::Forward);
        assert_eq!(map_side(Left, s, d, 0.2), EndVerdict::Reverse);
        // Ends are not separated vertically: up/down cannot be mapped.
        assert_eq!(map_side(Up, s, d, 0.2), EndVerdict::Unknown);
        // Vertical edge, src below dst.
        let (s, d) = ((300.0, 500.0), (302.0, 200.0));
        assert_eq!(map_side(Up, s, d, 0.2), EndVerdict::Forward);
        assert_eq!(map_side(Down, s, d, 0.2), EndVerdict::Reverse);
        assert_eq!(map_side(Left, s, d, 0.2), EndVerdict::Unknown);
        // Elbow: src top-left, dst bottom-right; both axes separate.
        let (s, d) = ((100.0, 100.0), (500.0, 400.0));
        assert_eq!(map_side(Down, s, d, 0.2), EndVerdict::Forward);
        assert_eq!(map_side(Left, s, d, 0.2), EndVerdict::Reverse);
        assert_eq!(map_side(None, s, d, 0.2), EndVerdict::NoArrowhead);
        assert_eq!(map_side(Unclear, s, d, 0.2), EndVerdict::Unknown);
    }

    #[test]
    fn crop_covers_both_boxes_and_is_bounded() {
        let img = RgbImage::new(2000, 1200);
        let a = BBox::new(100.0, 100.0, 300.0, 200.0);
        let b = BBox::new(1500.0, 900.0, 1700.0, 1000.0);
        let e = EdgeEnds {
            src: &a,
            dst: &b,
            src_text: "A",
            dst_text: "B",
            src_end: None,
            dst_end: None,
        };
        let c = connector_crop(&img, &e, &VlmCheckParams::default()).unwrap();
        assert_eq!(c.width().max(c.height()), 1024);
        let (sp, dp) = end_points(&e);
        // Each end point lies on its own box's border.
        let on = |p: (f64, f64), b: &BBox| {
            p.0 >= b.x1 - 1e-9 && p.0 <= b.x2 + 1e-9 && p.1 >= b.y1 - 1e-9 && p.1 <= b.y2 + 1e-9
        };
        assert!(on(sp, &a) && on(dp, &b), "{sp:?} {dp:?}");
    }
}
