//! Artifact names and item types produced by the media stages (data contracts, spec 7).
//!
//! Every artifact is a `glassrip-core` envelope whose items are `Record<T>` with the `T`
//! below. Times are seconds on the video PTS timeline in fields ending in `_s`. Joins use
//! the stable ids (`frame_id`, `keyframe_id`); paths are relative to the run directory.

use schemars::JsonSchema;
use semver::Version;
use serde::{Deserialize, Serialize};

/// `glassrip.media_probe`.
pub const MEDIA_PROBE: &str = "glassrip.media_probe";
/// `glassrip.orientation`.
pub const ORIENTATION: &str = "glassrip.orientation";
/// `glassrip.frames`.
pub const FRAMES: &str = "glassrip.frames";
/// `glassrip.screen_quads`.
pub const SCREEN_QUADS: &str = "glassrip.screen_quads";
/// `glassrip.features`.
pub const FEATURES: &str = "glassrip.features";
/// `glassrip.keyframes`.
pub const KEYFRAMES: &str = "glassrip.keyframes";
/// `glassrip.rectified_keyframes`.
pub const RECTIFIED_KEYFRAMES: &str = "glassrip.rectified_keyframes";

/// Schema version of every artifact in this crate.
pub fn v1() -> Version {
    Version::new(1, 0, 0)
}

/// Directory (relative to the run root) holding sampled frames.
pub const FRAMES_DIR: &str = "frames/sampled";
/// Directory (relative to the run root) holding representative keyframe images.
pub const KEYFRAMES_DIR: &str = "frames/keyframes";

/// A point `[x, y]` in pixels of the sampled frame.
pub type Point = [f64; 2];

/// A quadrilateral: top-left, top-right, bottom-right, bottom-left.
pub type Quad = [Point; 4];

// ---------------------------------------------------------------- probe

/// Video stream facts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VideoStream {
    /// Stream index in the container.
    pub index: u32,
    /// Codec name (`hevc`, `h264`, ...).
    pub codec: String,
    /// Codec profile, when reported.
    pub profile: Option<String>,
    /// Coded width (before any rotation).
    pub width: u32,
    /// Coded height.
    pub height: u32,
    /// Pixel format.
    pub pix_fmt: Option<String>,
    /// `avg_frame_rate` as reported.
    pub avg_frame_rate: String,
    /// `r_frame_rate` as reported.
    pub r_frame_rate: String,
    /// `avg_frame_rate` as a number.
    pub avg_fps: Option<f64>,
    /// `r_frame_rate` as a number.
    pub r_fps: Option<f64>,
    /// Time base, for example `1/90000`.
    pub time_base: String,
    /// Stream start time.
    pub start_time_s: f64,
    /// Stream start time in time-base units.
    pub start_pts: i64,
    /// Stream duration, when reported.
    pub duration_s: Option<f64>,
    /// Frame count, when reported.
    pub nb_frames: Option<u64>,
}

/// Audio stream facts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AudioStream {
    /// Stream index.
    pub index: u32,
    /// Codec name.
    pub codec: String,
    /// Sample rate in Hz.
    pub sample_rate: Option<u32>,
    /// Channel count.
    pub channels: Option<u32>,
    /// Stream start time (edit lists can offset it from the video).
    pub start_time_s: Option<f64>,
    /// Stream duration.
    pub duration_s: Option<f64>,
}

/// `glassrip.media_probe` item (id `video`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MediaProbe {
    /// Video path as given on the command line.
    pub video_path: String,
    /// blake3 of the file.
    pub file_blake3: String,
    /// File size.
    pub file_size_bytes: u64,
    /// Container format names.
    pub format_name: String,
    /// Container start time.
    pub start_time_s: f64,
    /// Container duration (never defaulted: a missing duration is an error).
    pub duration_s: f64,
    /// End of the media on the PTS timeline (`start_time_s + duration_s`).
    pub end_s: f64,
    /// The first video stream.
    pub video: VideoStream,
    /// The first audio stream, if any.
    pub audio: Option<AudioStream>,
    /// True when an audio stream exists.
    pub audio_present: bool,
    /// Variable frame rate: `avg_frame_rate != r_frame_rate`.
    pub vfr: bool,
    /// Average frame rate.
    pub avg_fps: Option<f64>,
    /// Display-matrix rotation from the container, a hint only (never applied).
    pub container_rotation_deg: Option<f64>,
}

// ---------------------------------------------------------------- orient

/// Vote counts per clockwise correction.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RotationVotes {
    /// Frames voting for no rotation.
    pub deg_0: u32,
    /// Frames voting for a 90 degree clockwise correction.
    pub deg_90: u32,
    /// Frames voting for 180 degrees.
    pub deg_180: u32,
    /// Frames voting for a 270 degree clockwise correction.
    pub deg_270: u32,
}

/// How the orientation was decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OrientMethod {
    /// PP-LCNet document orientation vote.
    PpLcnetVote,
    /// Set by configuration (`--orient-override`).
    Override,
    /// `ocrs` fallback (no ONNX Runtime): dictionary hits per rotation. `votes` holds the
    /// hit counts.
    OcrsDictionary,
}

/// Outcome of the 180 degree text-recognition check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConfirmStatus {
    /// The chosen rotation reads clearly better than its 180 degree flip.
    Confirmed,
    /// Too few text lines to compare; the vote stands.
    InsufficientText,
}

/// PP-OCRv5 recognition confidence at the chosen rotation versus its 180 degree flip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OrientConfirmation {
    /// Result.
    pub status: ConfirmStatus,
    /// Mean line confidence at the chosen rotation.
    pub chosen_confidence: f64,
    /// Mean line confidence turned 180 degrees.
    pub flipped_confidence: f64,
    /// Lines compared.
    pub lines: u32,
    /// Detector and recognizer used.
    pub models: Vec<ModelRef>,
}

/// One sampled frame's classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OrientSample {
    /// Sample time.
    pub t_s: f64,
    /// Softmax probabilities per clockwise correction `[0, 90, 180, 270]`.
    pub probs: [f64; 4],
    /// Correction with the highest probability.
    pub predicted_deg: u32,
    /// Whether the top probability passed the confidence floor (only those vote).
    pub confident: bool,
}

/// A pinned model file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelRef {
    /// Model name from the models manifest.
    pub name: String,
    /// SHA-256 of the file.
    pub sha256: String,
}

/// `glassrip.orientation` item (id `orientation`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Orientation {
    /// Clockwise rotation applied to decoded (`-noautorotate`) frames: 0, 90, 180 or 270.
    pub applied_rotation_deg: u32,
    /// How it was decided.
    pub method: OrientMethod,
    /// Votes of confident samples.
    pub votes: RotationVotes,
    /// Per-sample results.
    pub samples: Vec<OrientSample>,
    /// Container rotation hint from probe.
    pub container_rotation_deg: Option<f64>,
    /// Always false: container rotation is never trusted.
    pub container_rotation_trusted: bool,
    /// Model used, when any.
    pub model: Option<ModelRef>,
    /// 180 degree confirmation (ONNX builds).
    pub confirmation: Option<OrientConfirmation>,
    /// ONNX Runtime execution provider (`cpu`, `cuda`), when a model ran.
    pub execution_provider: Option<String>,
}

// ---------------------------------------------------------------- frames

/// How a sampled frame was chosen within its interval bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Sampling {
    /// Frame nearest the bucket's center (every frame decoded).
    Grid,
    /// Sync (key) frame nearest the bucket's center (only sync frames decoded).
    Sync,
}

/// `glassrip.frames` item (id = `frame_id`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FrameRecord {
    /// Stable id `f<bucket:06>`.
    pub frame_id: String,
    /// Interval bucket index: the frame's PTS lies in `[k, k+1) * interval` after the
    /// stream start.
    pub bucket: u64,
    /// Presentation timestamp in time-base units (from `showinfo`).
    pub pts: i64,
    /// Presentation time.
    pub pts_s: f64,
    /// JPEG path relative to the run directory.
    pub path: String,
    /// blake3 of the JPEG.
    pub blake3: String,
    /// Width after rotation and scaling.
    pub width: u32,
    /// Height after rotation and scaling.
    pub height: u32,
    /// Selection rule used for this frame's chunk.
    pub sampling: Sampling,
    /// Decoder that produced the frame (`software`, `cuda`, `videotoolbox`).
    pub decoder: String,
}

// ---------------------------------------------------------------- screen_quad

/// `glassrip.screen_quads` item (id = `frame_id`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScreenQuad {
    /// Frame id.
    pub frame_id: String,
    /// Monitor quadrilateral in frame pixels, or none.
    pub quad: Option<Quad>,
    /// Confidence in `[0, 1]` (edge support along the quad); for `none`, the best rejected
    /// candidate's confidence or 0.
    pub confidence: f64,
    /// Quad area as a share of the frame.
    pub area_frac: Option<f64>,
}

// ---------------------------------------------------------------- features

/// How a pair was aligned (`production` mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AlignMethod {
    /// ECC from the phase-correlation start.
    Ecc,
    /// Phase-correlation translation after ECC failed.
    PhaseTranslation,
    /// No trustworthy alignment: the pair counts as changed.
    Failed,
    /// `prototype_compat`: identity-initialized ECC (partial warp kept on failure).
    CompatEcc,
}

impl From<glassrip_media::production::AlignMethod> for AlignMethod {
    fn from(m: glassrip_media::production::AlignMethod) -> Self {
        use glassrip_media::production::AlignMethod as M;
        match m {
            M::Ecc => Self::Ecc,
            M::PhaseTranslation => Self::PhaseTranslation,
            M::Failed => Self::Failed,
        }
    }
}

/// Scores of a frame against the previous frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PairToPrev {
    /// Previous frame id (the ECC template).
    pub prev_frame_id: String,
    /// SSIM after alignment.
    pub ssim: f64,
    /// Share of changed pixels after alignment.
    pub changed_frac: f64,
    /// Ink change.
    pub ink_change: f64,
    /// Pair-score alignment usable (`production`: not `failed`; compat: ECC converged).
    pub align_ok: bool,
    /// Pair-score alignment method.
    pub align_method: AlignMethod,
    /// Ink-path alignment usable.
    pub ink_align_ok: bool,
    /// Length of the pair-score translation, pixels of the 320-wide image.
    pub shift_px: f64,
    /// Share of pixels inside the warp's valid mask (`production` only).
    pub valid_frac: Option<f64>,
}

/// `glassrip.features` item (id = `frame_id`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FrameFeatures {
    /// Frame id.
    pub frame_id: String,
    /// Position in PTS order (0-based).
    pub index: u64,
    /// Presentation time.
    pub pts_s: f64,
    /// Frame JPEG path (relative to the run directory), carried for downstream scoring.
    pub frame_path: String,
    /// Frame JPEG blake3.
    pub frame_blake3: String,
    /// Population variance of the 3x3 Laplacian on full-resolution gray.
    pub sharpness_lapvar: f64,
    /// Scores against the previous frame (`None` for the first frame).
    pub prev: Option<PairToPrev>,
}

// ---------------------------------------------------------------- keyframes

/// Why a keyframe starts where it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryReason {
    /// First keyframe of the video.
    Start,
    /// SSIM below threshold.
    Ssim,
    /// Changed fraction above threshold.
    Frac,
    /// Ink change above threshold.
    Ink,
    /// Alignment failed (`production`).
    AlignFailed,
}

/// Boundary decision that started a keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Boundary {
    /// Reason.
    pub reason: BoundaryReason,
    /// Frame compared with: the previous run's first frame (segmentation), or the
    /// preceding frame (production island runs and the runs around them).
    pub anchor_frame_id: Option<String>,
    /// SSIM of anchor vs first frame.
    pub ssim: Option<f64>,
    /// Changed fraction.
    pub changed_frac: Option<f64>,
    /// Ink change (not computed when SSIM or frac already decided).
    pub ink_change: Option<f64>,
    /// Pair-score alignment usable.
    pub align_ok: Option<bool>,
}

/// A singleton merged into this keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MergedSingleton {
    /// The singleton frame.
    pub frame_id: String,
    /// SSIM against this run's representative at merge time.
    pub ssim: f64,
    /// Changed fraction against it.
    pub changed_frac: f64,
    /// Merged only because the frame was blurry.
    pub blurry: bool,
}

/// `glassrip.keyframes` item (id = `keyframe_id`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Keyframe {
    /// Stable id `k<index:04>`.
    pub keyframe_id: String,
    /// Position in time order.
    pub index: u32,
    /// Representative (sharpest) frame.
    pub rep_frame_id: String,
    /// All frames of the run, in time order.
    pub frame_ids: Vec<String>,
    /// First frame's time.
    pub t_start_s: f64,
    /// Next keyframe's start, or the probed end for the last one.
    pub t_end_s: f64,
    /// Representative frame's time.
    pub t_rep_s: f64,
    /// Frames in the run.
    pub n_frames: u32,
    /// Representative's sharpness.
    pub sharpness_lapvar: f64,
    /// Boundary that started the run.
    pub boundary: Boundary,
    /// Singletons merged into this run.
    pub merged: Vec<MergedSingleton>,
}

// ---------------------------------------------------------------- rectify

/// How the representative image was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RectifyMethod {
    /// Median-quad warp of several aligned frames, temporal median.
    WarpMedian,
    /// Median-quad warp of the representative frame alone.
    WarpSingle,
    /// No monitor quad: the representative frame unchanged.
    Passthrough,
}

/// `glassrip.rectified_keyframes` item (id = `keyframe_id`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RectifiedKeyframe {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Representative frame id (the stack reference).
    pub rep_frame_id: String,
    /// Representative image path, relative to the run directory.
    pub path: String,
    /// blake3 of the image.
    pub blake3: String,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Per-corner median quad over the run, in frame pixels.
    pub median_quad: Option<Quad>,
    /// Frames of the run with a usable quad.
    pub n_frames_with_quad: u32,
    /// Frames in the temporal median (1 for `warp_single` and `passthrough`).
    pub n_frames_stacked: u32,
    /// Frames in the stack.
    pub stacked_frame_ids: Vec<String>,
    /// Method.
    pub method: RectifyMethod,
}
