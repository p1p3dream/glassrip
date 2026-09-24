//! Artifact names, consumer views of upstream artifacts, and this crate's outputs.
//!
//! Upstream item types are read through small views that name only the fields used
//! here and ignore everything else, so producers may add fields freely. The fields
//! follow the data contracts (spec section 7); where the contract leaves the shape
//! open (the validated board wrapper, the crop path field name), the views accept the
//! plausible spellings through aliases.

use glassrip_vision::board::{EdgeStyle, ValidatedBoard};
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
    /// `{keyframe_id, canvas?, board | result}`.
    Wrapped {
        /// Keyframe id.
        keyframe_id: String,
        /// Canvas size, when recorded.
        #[serde(default, alias = "canvas_size")]
        canvas: Option<CanvasDims>,
        /// The validated board.
        #[serde(alias = "result")]
        board: ValidatedBoard,
    },
    /// The bare validated board.
    Bare(ValidatedBoard),
}

impl ValidateItemView {
    /// `(keyframe_id, canvas, board)`, taking the record id when the item is bare.
    pub fn into_parts(self, record_id: &str) -> (String, Option<CanvasDims>, ValidatedBoard) {
        match self {
            Self::Wrapped {
                keyframe_id,
                canvas,
                board,
            } => (keyframe_id, canvas, board),
            Self::Bare(board) => (record_id.to_string(), None, board),
        }
    }
}

/// One canvas crop (consumer view of `glassrip.canvas_crop`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CanvasCropView {
    /// Keyframe id (defaults to the record id).
    #[serde(default)]
    pub keyframe_id: Option<String>,
    /// Crop image path, absolute or relative to the run directory.
    #[serde(alias = "crop_path", alias = "image_path")]
    pub path: String,
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

/// `glassrip.edge_direction` item: evidence for every edge of one keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeDirectionItem {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Canvas size of the crop the evidence was computed on.
    pub canvas: CanvasDims,
    /// Laplacian variance of the crop (vote weight input).
    pub sharpness: f64,
    /// Median node box height in pixels (zoom; vote weight input).
    pub zoom: f64,
    /// Per-edge evidence.
    pub edges: Vec<EdgeEvidence>,
}
