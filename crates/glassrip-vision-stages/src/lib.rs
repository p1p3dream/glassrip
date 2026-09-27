//! Vision-branch stages of glassrip meeting mode, as `glassrip-core` stages:
//!
//! `ocr_harvest` -> (`ocr_vocabulary`) -> `classify` -> `canvas_crop` ->
//! `board_read` -> `board_validate`.
//!
//! - [`artifacts`]: artifact names and item types (spec section 7).
//! - [`layout`]: tiles, banner, shared area, and whiteboard panels from OCR text.
//! - [`pixels`]: shape and color measurement, temporal variance, masks, quad warp.
//! - [`placement`]: preflight and mid-run GPU placement checks with pause and resume.
//! - [`consensus`]: the vote over several readings of one keyframe.
//! - [`raw_store`]: raw model responses by request key, recording and replay backends.
//! - [`adapter`]: upstream artifacts from prototype keyframes, for running before
//!   the media stages exist.
//! - [`stages`]: the stages.

pub mod adapter;
pub mod artifacts;
pub mod consensus;
pub mod layout;
pub mod pipeline;
pub mod pixels;
pub mod placement;
pub mod raw_store;
pub mod stages;

pub use stages::board_read::{BoardReadParams, BoardReadStage, ConsensusParams};
pub use stages::board_validate::{BoardValidateParams, BoardValidateStage};
pub use stages::canvas_crop::{CanvasCropParams, CanvasCropStage};
pub use stages::classify::{ClassifyParams, ClassifyStage};
pub use stages::ocr_harvest::OcrHarvestStage;
pub use stages::vocabulary::{VocabularyParams, VocabularyStage};
