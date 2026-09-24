//! Artifact names, consumer views of upstream artifacts, and this crate's outputs.
//!
//! Upstream item types are read through small views that name only the fields used
//! here and ignore everything else, so producers may add fields freely. The fields
//! follow the data contracts (spec section 7); where the contract leaves the shape
//! open (the validated board wrapper, the crop path field name), the views accept the
//! plausible spellings through aliases.

use glassrip_vision::board::{EdgeStyle, ValidatedBoard};
use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::direction::EndVerdict;
use crate::pixel_direction::PixelEvidence;
use crate::vlm_direction::VlmEvidence;

/// Validated board readings (input).
pub const BOARD_VALIDATE: &str = "glassrip.board_validate";
/// Canvas crops (input).
pub const CANVAS_CROP: &str = "glassrip.canvas_crop";
/// Keyframes (input).
pub const KEYFRAMES: &str = "glassrip.keyframes";
/// OCR spans (input, registration anchors).
pub const OCR: &str = "glassrip.ocr";
/// Edge direction evidence (output of `edge_direction`).
pub const EDGE_DIRECTION: &str = "glassrip.edge_direction";
/// Consolidated board state (output of `board_state`).
pub const BOARD_STATE: &str = "glassrip.board_state";

/// Keyframe boundary scores (consumer view).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct BoundaryView {
    /// Aligned ink change against the previous run's anchor.
    #[serde(default)]
    pub ink_change: Option<f64>,
}

/// One keyframe (consumer view of `glassrip.keyframes`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct KeyframeView {
    /// Stable keyframe id.
    pub keyframe_id: String,
    /// Run start.
    pub t_start_s: f64,
    /// Run end.
    pub t_end_s: f64,
    /// Representative frame time.
    pub t_rep_s: f64,
    /// Boundary that opened this run.
    #[serde(default)]
    pub boundary: Option<BoundaryView>,
}

/// Canvas size in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanvasDims {
    /// Width.
    pub width: f64,
    /// Height.
    pub height: f64,
}

impl CanvasDims {
    /// Diagonal length.
    pub fn diagonal(&self) -> f64 {
        (self.width * self.width + self.height * self.height).sqrt()
    }
}

/// A validated board item (consumer view of `glassrip.board_validate`): either the
/// bare [`ValidatedBoard`] (record id = keyframe id) or a wrapper that names the
/// keyframe and canvas.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum ValidateItemView {
    /// `{keyframe_id, canvas?, board_title?, board | result}`.
    Wrapped {
        /// Keyframe id.
        keyframe_id: String,
        /// Canvas size, when recorded.
        #[serde(default, alias = "canvas_size")]
        canvas: Option<CanvasDims>,
        /// Board title, when a producer extracted one.
        #[serde(default, alias = "title")]
        board_title: Option<String>,
        /// The validated board.
        #[serde(alias = "result")]
        board: ValidatedBoard,
    },
    /// The bare validated board.
    Bare(ValidatedBoard),
}

/// One validated board with its context.
#[derive(Debug, Clone, PartialEq)]
pub struct BoardItem {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Canvas size of the reading's coordinates, when recorded.
    pub canvas: Option<CanvasDims>,
    /// Board title, when known.
    pub board_title: Option<String>,
    /// The validated board.
    pub board: ValidatedBoard,
}

impl ValidateItemView {
    /// The item's parts, taking the record id when the item is bare.
    pub fn into_item(self, record_id: &str) -> BoardItem {
        match self {
            Self::Wrapped {
                keyframe_id,
                canvas,
                board_title,
                board,
            } => BoardItem {
                keyframe_id,
                canvas,
                board_title,
                board,
            },
            Self::Bare(board) => BoardItem {
                keyframe_id: record_id.to_string(),
                canvas: None,
                board_title: None,
                board,
            },
        }
    }
}

/// One canvas crop (consumer view of `glassrip.canvas_crop`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CanvasCropView {
    /// Keyframe id (defaults to the record id).
    #[serde(default)]
    pub keyframe_id: Option<String>,
    /// Crop image path, absolute or relative to the run directory. It may name
    /// the whole frame (`source_image_path`); `crop` then cuts the canvas out.
    #[serde(alias = "crop_path", alias = "image_path", alias = "source_image_path")]
    pub path: String,
    /// Crop box in frame pixels, when recorded.
    #[serde(
        default,
        alias = "crop_bbox",
        alias = "crop_box",
        alias = "bbox",
        alias = "canvas_bbox"
    )]
    pub crop: Option<BBox>,
}

/// One OCR span (consumer view).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct OcrSpanView {
    /// Text.
    pub text: String,
    /// Box in frame pixels.
    #[serde(alias = "box")]
    pub bbox: BBox,
    /// Recognition confidence.
    #[serde(default, alias = "conf")]
    pub confidence: Option<f64>,
    /// `canvas`, `chrome`, or `tile`.
    #[serde(default)]
    pub region: Option<String>,
}

/// OCR spans of one keyframe (consumer view of `glassrip.ocr`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct OcrView {
    /// Keyframe id (defaults to the record id).
    #[serde(default)]
    pub keyframe_id: Option<String>,
    /// Spans.
    #[serde(alias = "lines", alias = "items")]
    pub spans: Vec<OcrSpanView>,
}

/// Direction evidence for one edge reading.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeEvidence {
    /// Tail `local_id` as read.
    pub src: String,
    /// Head `local_id` as read.
    pub dst: String,
    /// Tail text.
    pub src_text: String,
    /// Head text.
    pub dst_text: String,
    /// Label as read (empty when none).
    pub label: String,
    /// Line style.
    pub style: EdgeStyle,
    /// Pixel check.
    pub pixel: PixelEvidence,
    /// VLM endpoint check, when run.
    pub vlm: Option<VlmEvidence>,
}

impl EdgeEvidence {
    /// Pixel and VLM verdicts relative to `src -> dst` as read.
    pub fn verdicts(&self) -> (EndVerdict, Option<EndVerdict>) {
        (self.pixel.verdict, self.vlm.as_ref().map(|v| v.verdict))
    }
}

/// How the reading's coordinates relate to the crop image the pixel check ran on.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateCheck {
    /// Canvas size recorded with the reading equals the crop size.
    Verified1to1,
    /// Canvas size differs; boxes were scaled into crop pixels by `board_to_image`.
    Scaled,
    /// No canvas size recorded; every box lies inside the crop, so 1:1 is assumed.
    Assumed1to1,
    /// No canvas size recorded and boxes fall outside the crop; evidence is unreliable.
    Inconsistent,
}

/// Per-axis scale from reading coordinates to crop pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AxisScale {
    /// x scale.
    pub x: f64,
    /// y scale.
    pub y: f64,
}

/// Evidence for every edge of one keyframe. Termini and end points are in the
/// reading's canvas coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeDirectionItem {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Size of the reading's canvas (the crop size divided by `board_to_image`).
    pub canvas: CanvasDims,
    /// Reading coordinates to crop pixels.
    pub board_to_image: AxisScale,
    /// How `board_to_image` was established.
    pub coordinates: CoordinateCheck,
    /// Crop box in frame pixels, when the crop stage recorded it.
    pub crop_in_frame: Option<BBox>,
    /// Laplacian variance of the crop (vote weight input).
    pub sharpness: f64,
    /// Median node box height in pixels (zoom; vote weight input).
    pub zoom: f64,
    /// Per-edge evidence.
    pub edges: Vec<EdgeEvidence>,
    /// Why the keyframe has no evidence (for example an unreadable crop).
    pub error: Option<String>,
}

/// An edge whose pixel vote was inconclusive, and what the VLM fallback said.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VlmFallback {
    /// Texts of the two ends as grouped across keyframes.
    pub ends: (String, String),
    /// Keyframes the VLM was asked on.
    pub keyframes: Vec<String>,
    /// Requests that failed.
    pub errors: Vec<String>,
}

/// `glassrip.edge_direction` item: evidence for all board keyframes (one item, since
/// the VLM fallback depends on the cross-keyframe pixel vote).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeDirectionBatch {
    /// Per-keyframe evidence.
    pub keyframes: Vec<EdgeDirectionItem>,
    /// Edges sent to the VLM fallback.
    pub vlm_fallback: Vec<VlmFallback>,
    /// Whether a vision client was available.
    pub vlm_available: bool,
}
