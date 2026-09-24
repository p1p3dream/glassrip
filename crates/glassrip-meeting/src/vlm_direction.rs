//! Binary VLM check on endpoint crops (spec 6.11, edge direction step 2).
//!
//! For each end of an edge, a square crop centered on the pixel check's terminus (or,
//! when the pixel check did not reach that node, on the point where the line between
//! the two box centers leaves the box) is upscaled and sent with a schema that allows
//! exactly one of `points_to_A`, `points_to_B`, `no_arrowhead`, `unclear`.

use glassrip_vision::backend::{GenerationOptions, VisionRequest};
use glassrip_vision::image_prep::EncodedImage;
use glassrip_vision::{BBox, VisionClient};
use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::direction::{vlm_verdict, EndVerdict, EndpointAnswer};
use crate::pixel_direction::{exit_point, EndEvidence};

/// Prompt for one endpoint crop. `{a}` and `{b}` are replaced by the box texts.
pub const ENDPOINT_PROMPT: &str = "\
The image is a close-up crop of a whiteboard diagram. It shows one end of a connector line \
where it meets box A (text: \"{a}\"). The other end of the same line connects to box B \
(text: \"{b}\"), which may be outside this crop.

Look only at the end of the line next to box A and answer with exactly one value:
- points_to_A: a filled or open arrowhead at this end touches box A.
- points_to_B: the arrowhead at this end points away from box A.
- no_arrowhead: the line simply ends at box A with no arrowhead.
- unclear: the end of the line is not visible or cannot be judged.";

/// Model reply for one endpoint crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EndpointReply {
    /// The single answer.
    pub answer: EndpointAnswer,
}

/// VLM check settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VlmCheckParams {
    /// Side of the square crop in canvas pixels.
    pub crop_px: u32,
    /// Side of the image sent to the model.
    pub send_px: u32,
    /// Generation seed.
    pub seed: u64,
    /// Output token cap.
    pub num_predict: u32,
}

impl Default for VlmCheckParams {
    fn default() -> Self {
        Self {
            crop_px: 112,
            send_px: 448,
            seed: 7,
            num_predict: 32,
        }
    }
}

/// VLM check result for one edge in one keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VlmEvidence {
    /// Answer for the crop at the `src` end (A = src).
    pub src_end: EndpointAnswer,
    /// Answer for the crop at the `dst` end (A = dst).
    pub dst_end: EndpointAnswer,
    /// Combined verdict relative to `src -> dst` as read.
    pub verdict: EndVerdict,
}

/// Crop center for one end.
pub fn endpoint_center(pixel_end: Option<&EndEvidence>, this: &BBox, other: &BBox) -> (f64, f64) {
    if let Some(e) = pixel_end {
        return (e.x, e.y);
    }
    exit_point(
        this,
        ((other.x1 + other.x2) / 2.0, (other.y1 + other.y2) / 2.0),
    )
}

/// Square crop around `center`, clamped to the image and resized to `send_px`.
pub fn endpoint_crop(
    image: &RgbImage,
    center: (f64, f64),
    params: &VlmCheckParams,
) -> Option<DynamicImage> {
    let (w, h) = image.dimensions();
    let side = params.crop_px.min(w).min(h);
    if side == 0 {
        return None;
    }
    let half = f64::from(side) / 2.0;
    let x0 = (center.0 - half).round().clamp(0.0, f64::from(w - side)) as u32;
    let y0 = (center.1 - half).round().clamp(0.0, f64::from(h - side)) as u32;
    let crop = image::imageops::crop_imm(image, x0, y0, side, side).to_image();
    Some(DynamicImage::ImageRgb8(crop).resize_exact(
        params.send_px,
        params.send_px,
        FilterType::Lanczos3,
    ))
}

/// Build the request for one endpoint crop.
pub fn endpoint_request(
    crop: &DynamicImage,
    this_text: &str,
    other_text: &str,
    params: &VlmCheckParams,
) -> glassrip_vision::Result<VisionRequest> {
    let prompt = ENDPOINT_PROMPT
        .replace("{a}", this_text)
        .replace("{b}", other_text);
    VisionRequest::for_output::<EndpointReply>(
        &prompt,
        EncodedImage::encode(crop)?,
        GenerationOptions {
            seed: params.seed,
            num_predict: params.num_predict,
        },
    )
}

/// Inputs for checking one edge.
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

/// Ask the model about both ends of one edge.
pub async fn check_edge(
    client: &VisionClient,
    image: &RgbImage,
    ends: EdgeEnds<'_>,
    params: &VlmCheckParams,
    cancel: CancellationToken,
) -> glassrip_vision::Result<VlmEvidence> {
    let mut answers = [EndpointAnswer::Unclear; 2];
    let sides = [
        (
            ends.src,
            ends.dst,
            ends.src_text,
            ends.dst_text,
            ends.src_end,
        ),
        (
            ends.dst,
            ends.src,
            ends.dst_text,
            ends.src_text,
            ends.dst_end,
        ),
    ];
    for (slot, (this, other, this_text, other_text, pix)) in answers.iter_mut().zip(sides) {
        let center = endpoint_center(pix, this, other);
        let Some(crop) = endpoint_crop(image, center, params) else {
            continue;
        };
        let req = endpoint_request(&crop, this_text, other_text, params)?;
        let (reply, _raw): (EndpointReply, _) = client.infer_typed(req, cancel.clone()).await?;
        *slot = reply.answer;
    }
    Ok(VlmEvidence {
        src_end: answers[0],
        dst_end: answers[1],
        verdict: vlm_verdict(answers[0], answers[1]),
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn schema_is_a_closed_enum() {
        let s = glassrip_vision::OutputSchema::for_type::<EndpointReply>().unwrap();
        let text = s.text();
        for v in ["points_to_A", "points_to_B", "no_arrowhead", "unclear"] {
            assert!(text.contains(v), "{text}");
        }
        let ok: EndpointReply = serde_json::from_str(r#"{"answer":"points_to_B"}"#).unwrap();
        assert_eq!(ok.answer, EndpointAnswer::PointsToB);
        assert!(serde_json::from_str::<EndpointReply>(r#"{"answer":"left"}"#).is_err());
    }

    #[test]
    fn crop_is_clamped_and_resized() {
        let img = RgbImage::new(200, 100);
        let p = VlmCheckParams::default();
        let c = endpoint_crop(&img, (5.0, 95.0), &p).unwrap();
        assert_eq!((c.width(), c.height()), (p.send_px, p.send_px));
        let a = BBox::new(10.0, 10.0, 50.0, 40.0);
        let b = BBox::new(150.0, 10.0, 190.0, 40.0);
        let (x, y) = endpoint_center(None, &a, &b);
        assert!((x - 50.0).abs() < 1e-9 && (y - 25.0).abs() < 1e-9);
    }
}
