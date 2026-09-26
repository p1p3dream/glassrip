//! Artifact names, versions, and item types for the vision branch.
//!
//! Input views (`*View`) are what these stages read from upstream artifacts.
//! They accept unknown fields so upstream producers can add fields in minor
//! versions; outputs are strict (`deny_unknown_fields`).

use glassrip_vision::board::{BoardReading, CanvasSize, ValidatedBoard};
use glassrip_vision::classify::{ClassifyMethod, RuleHit, ScreenType};
use glassrip_vision::BBox;
use schemars::JsonSchema;
use semver::Version;
use serde::{Deserialize, Serialize};

/// `glassrip.frames` (input).
pub const FRAMES: &str = "glassrip.frames";
/// `glassrip.screen_quads` (input).
pub const SCREEN_QUADS: &str = "glassrip.screen_quads";
/// `glassrip.keyframes` (input).
pub const KEYFRAMES: &str = "glassrip.keyframes";
/// `glassrip.rectified_keyframes` (input).
pub const RECTIFIED_KEYFRAMES: &str = "glassrip.rectified_keyframes";
/// `ocr_harvest` output.
pub const OCR: &str = "glassrip.ocr";
/// `ocr_vocabulary` output: the ASR vocabulary list.
pub const ASR_VOCABULARY: &str = "glassrip.asr_vocabulary";
/// `classify` output.
pub const SCREEN_CLASS: &str = "glassrip.screen_class";
/// `canvas_crop` output.
pub const CANVAS_CROP: &str = "glassrip.canvas_crop";
/// `board_read` output.
pub const BOARD_READING: &str = "glassrip.board_reading";
/// `board_validate` output.
pub const BOARD_VALIDATE: &str = "glassrip.board_validate";

/// Major version every input view understands.
pub const INPUT_MAJOR: u64 = 1;

/// Version of every artifact produced here.
pub fn output_version() -> Version {
    Version::new(1, 0, 0)
}

/// A corner point in pixels.
pub type Point = [f64; 2];

/// `glassrip.frames` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FrameView {
    pub frame_id: String,
    pub pts_s: f64,
    /// Image path, relative to the run directory or absolute.
    pub path: String,
    #[serde(default)]
    pub blake3: Option<String>,
}

/// `glassrip.screen_quads` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScreenQuadView {
    pub frame_id: String,
    /// Corners top-left, top-right, bottom-right, bottom-left; `None` when no
    /// monitor was found (screen recordings).
    #[serde(default)]
    pub quad: Option<[Point; 4]>,
    #[serde(default)]
    pub confidence: Option<f64>,
}

/// `glassrip.keyframes` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct KeyframeView {
    pub keyframe_id: String,
    pub rep_frame_id: String,
    pub t_start_s: f64,
    pub t_end_s: f64,
    pub t_rep_s: f64,
    #[serde(default)]
    pub n_frames: Option<u32>,
}

/// `glassrip.rectified_keyframes` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RectifiedKeyframeView {
    pub keyframe_id: String,
    /// Representative image, relative to the run directory or absolute.
    #[serde(alias = "path")]
    pub image_path: String,
    #[serde(default, alias = "blake3")]
    pub image_blake3: Option<String>,
    #[serde(default)]
    pub median_quad: Option<[Point; 4]>,
    #[serde(default)]
    pub n_frames_stacked: Option<u32>,
    /// `warp_median`, `warp_single`, or `passthrough`.
    #[serde(default)]
    pub method: Option<String>,
}

/// Where a text span sits on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TextRegion {
    /// Inside the board canvas (decided by `canvas_crop`).
    Canvas,
    /// Application or conferencing UI.
    Chrome,
    /// A participant video tile or its name label.
    Tile,
    /// Inside the shared area but not yet assigned (before `canvas_crop`).
    Unassigned,
}

/// Why a span counts as chrome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ChromeReason {
    /// Matches the chrome denylist.
    Denylist,
    /// Conferencing banner or title bar (for example `(Presenting`).
    Banner,
    /// Participant tile name label.
    TileName,
    /// Outside the shared screen area.
    OutsideShare,
    /// Whiteboard application sidebar or top bar.
    AppPanel,
    /// Outside the board canvas.
    OutsideCanvas,
}

/// One OCR text span.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OcrSpan {
    pub text: String,
    /// Rectified keyframe pixels.
    pub bbox: BBox,
    pub confidence: f64,
    pub region: TextRegion,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chrome_reason: Option<ChromeReason>,
    /// Median luma (0 to 255) inside the span box: conferencing name labels are
    /// light text on a dark overlay, board text is dark on light.
    pub bg_luma: f64,
}

/// `glassrip.ocr` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OcrKeyframe {
    pub keyframe_id: String,
    pub image_width: u32,
    pub image_height: u32,
    pub execution_provider: String,
    pub spans: Vec<OcrSpan>,
    /// Participant names read from tile labels and banners in this keyframe.
    pub tile_names: Vec<String>,
    /// Shared screen area from layout analysis, when found.
    pub share_area: Option<BBox>,
}

/// Vocabulary term kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TermKind {
    Participant,
    BoardLabel,
    Identifier,
}

/// One ASR vocabulary term.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VocabularyTerm {
    pub text: String,
    pub kind: TermKind,
    /// Keyframes the term was seen in.
    pub keyframes: u32,
    pub first_keyframe_id: String,
}

/// `glassrip.asr_vocabulary` item (a single item with id `vocabulary`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Vocabulary {
    /// Participants first, then by keyframe count.
    pub terms: Vec<VocabularyTerm>,
}

/// How a classification was reached after smoothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClassSource {
    /// The per-keyframe combiner decided.
    Combined,
    /// Temporal smoothing replaced a weak or isolated answer with the type of both neighbors.
    Smoothed,
}

/// The model's own answer, kept for audit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelAnswer {
    pub screen_type: ScreenType,
    pub app_hint: String,
    pub confidence: f64,
    /// Source pixels.
    pub canvas_bbox: BBox,
    /// Key of the raw response in the raw store.
    pub request_key: String,
}

/// `glassrip.screen_class` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScreenClassItem {
    pub keyframe_id: String,
    pub screen_type: ScreenType,
    pub app_hint: Option<String>,
    pub confidence: f64,
    pub method: ClassifyMethod,
    pub source: ClassSource,
    /// Source pixels; present only when the final type matches the model.
    pub canvas_bbox: Option<BBox>,
    /// True when the keyframe proceeds to board reading.
    pub reads_board: bool,
    pub model: Option<ModelAnswer>,
    /// Model request failure, when the model could not answer.
    pub model_error: Option<String>,
    pub rule_hits: Vec<RuleHit>,
    /// Type before smoothing, when smoothing changed it.
    pub smoothed_from: Option<ScreenType>,
}

/// How the canvas box was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CanvasMethod {
    /// Conferencing layout and application panels from OCR.
    Layout,
    /// Layout, with a tile decided by the temporal-variance tiebreak.
    LayoutVariance,
    /// The classifier's canvas box (no layout evidence).
    ModelBox,
    /// Nothing found; the whole image.
    FullFrame,
}

/// A participant tile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TileBox {
    pub name: String,
    pub bbox: BBox,
}

/// A region painted over before the crop is sent to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChromeMask {
    /// Source pixels.
    pub bbox: BBox,
    pub reason: ChromeReason,
    pub text: String,
}

/// `glassrip.canvas_crop` item (whiteboard keyframes only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanvasCropItem {
    pub keyframe_id: String,
    pub source_frame_id: String,
    pub source_image_path: String,
    pub source_image_blake3: Option<String>,
    pub image_width: u32,
    pub image_height: u32,
    /// Final crop box in source pixels (whole pixels).
    pub canvas_bbox: BBox,
    /// Box before per-segment stabilization.
    pub raw_canvas_bbox: BBox,
    pub method: CanvasMethod,
    /// Index of the stabilization segment (consecutive board keyframes with one layout).
    pub segment: u32,
    pub stabilized: bool,
    pub share_area: Option<BBox>,
    pub tiles: Vec<TileBox>,
    pub masks: Vec<ChromeMask>,
    /// Final region per OCR span (same order as the `glassrip.ocr` spans).
    pub span_regions: Vec<TextRegion>,
    /// Median height of canvas text spans, source pixels.
    pub canvas_text_height_px: Option<f64>,
    pub participants: Vec<String>,
}

/// Which part of the canvas a request covered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RequestRole {
    Overview,
    Tile,
}

/// One model request made for a board reading.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RequestLog {
    pub role: RequestRole,
    /// Canvas pixels covered by the request.
    pub region: BBox,
    pub sent_width: u32,
    pub sent_height: u32,
    /// Raw store key (replay key).
    pub request_key: String,
    pub latency_s: f64,
    pub attempts: u32,
    pub repaired: bool,
    pub eval_count: Option<u32>,
    /// Informational: prompt tokens the server evaluated for this request. It
    /// excludes a cached prefix reused from an earlier request, so it can be
    /// far below the prompt's size.
    pub prompt_eval_count: Option<u32>,
    pub done_reason: Option<String>,
    /// The first reply stopped at the output limit and this is the compact
    /// retry (smaller list budgets); `request_key` is the retry's key.
    #[serde(default)]
    pub compact_retry: bool,
    /// The first reply fell into a repetition loop (stopped while streaming, or
    /// found in the returned text) and this is the retry with a repeat penalty;
    /// `request_key` is the retry's key. Replay rebuilds the same retry from this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition: Option<glassrip_vision::repetition::RepetitionFinding>,
    /// Sampling overrides of the request that produced the reading (set on the
    /// repetition retry; part of its request key).
    #[serde(
        default,
        skip_serializing_if = "glassrip_vision::SamplingOverrides::is_empty"
    )]
    pub sampling: glassrip_vision::SamplingOverrides,
    /// Output budget of the request that produced the reading.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    /// A complete reply listed one text at many places (see
    /// `glassrip_vision::degenerate`). Replay rebuilds the same retry and the
    /// same collapse from this and the recorded replies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degenerate: Option<DegenerateLog>,
}

/// What the degenerate-reading rule did for one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DegenerateLog {
    /// The repeated texts of the first complete reply.
    pub finding: glassrip_vision::degenerate::DegenerateFinding,
    /// The repeat-penalty retry answered, and the reading is its reply;
    /// `request_key`, `sampling`, and `num_predict` are the retry's. False when
    /// no retry was left (the reply already was a retry) or the retry failed.
    pub retried: bool,
    /// How the retry failed (`output limit`, `repetition loop`, or `invalid
    /// reply`: answers the raw store records); the first reply was kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_error: Option<String>,
    /// Texts of the kept reply reduced to their best-supported copies (a warning:
    /// the reading was still degenerate). Empty when the retry came back sound.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collapsed: Vec<glassrip_vision::degenerate::CollapsedText>,
}

/// Model identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelRef {
    pub name: String,
    pub digest: Option<String>,
}

/// `glassrip.board_reading` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardReadingItem {
    pub keyframe_id: String,
    pub source_frame_id: String,
    pub source_image_path: String,
    pub source_image_blake3: Option<String>,
    /// Crop box in source pixels; element boxes in `result` are relative to it.
    pub crop_box: BBox,
    pub masks: Vec<ChromeMask>,
    /// Participant tiles in source pixels (from `canvas_crop`).
    pub tiles: Vec<TileBox>,
    pub model: ModelRef,
    /// Wall time for all requests of this keyframe.
    pub latency_s: f64,
    pub low_res: bool,
    pub token_capped: bool,
    pub tiled: bool,
    pub requests: Vec<RequestLog>,
    pub participants: Vec<String>,
    /// Reading in canvas pixels (merged over tiles when tiled).
    pub result: BoardReading,
}

/// Shape and color measured inside an element's box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ShapeClass {
    /// Light fill with a visible outline: a node.
    OutlinedBox,
    /// Saturated fill: a sticky note.
    FilledSticky,
    /// Green fill: an owner tag when the text is a short name.
    GreenTag,
    /// Not decidable from pixels.
    Unclear,
}

/// Which list an element belongs to after the pixel check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MemberList {
    Nodes,
    Stickies,
    OwnerTags,
}

/// Pixel measurement and list decision for one element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MembershipDecision {
    pub text: String,
    pub bbox: BBox,
    pub from: MemberList,
    pub to: MemberList,
    pub shape: ShapeClass,
    /// Median fill color (RGB).
    pub fill_rgb: [u8; 3],
    pub fill_saturation: f64,
    /// Share of edge samples with a visible outline.
    pub outline_fraction: f64,
    /// OCR spans whose text background decided the class (0: the element box did).
    pub anchored_spans: u32,
}

/// A rule applied after the shared validator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExtraIssue {
    pub kind: String,
    pub detail: String,
}

/// `glassrip.board_validate` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardValidateItem {
    pub keyframe_id: String,
    pub source_frame_id: String,
    /// Rectified keyframe image; crop it with `crop_box` (and paint `masks`)
    /// to get the canvas the element boxes refer to.
    pub source_image_path: String,
    /// Canvas crop in source pixels. Every box in `board` (nodes, stickies,
    /// owner tags, other text, edge `label_bbox_2d`) is in canvas pixels,
    /// relative to this crop's top-left corner.
    pub crop_box: BBox,
    pub masks: Vec<ChromeMask>,
    /// Canvas size in pixels (equals the crop box size).
    pub canvas: CanvasSize,
    pub board: ValidatedBoard,
    pub membership: Vec<MembershipDecision>,
    pub extra_issues: Vec<ExtraIssue>,
    /// `confidence == 0` or no content: the keyframe should be classified again.
    pub needs_reclassification: bool,
}
