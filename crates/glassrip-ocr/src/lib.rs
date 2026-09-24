//! PP-OCRv5 text detection and recognition for glassrip.
//!
//! The pure parts (input tensors, DB box extraction from the detection
//! probability map, CTC decoding) are always compiled and unit tested with
//! synthetic data. The ONNX Runtime engine sits behind the `onnx` feature; the
//! `cuda` feature loads ONNX Runtime dynamically from `ORT_DYLIB_PATH` and
//! registers the CUDA execution provider with `error_on_failure`.
//!
//! Stages depend on the [`TextRecognizer`] trait, so they can be tested with a
//! fake recognizer and run on hosts without ONNX Runtime.

pub mod ctc;
pub mod db;
pub mod models;
pub mod preprocess;

#[cfg(feature = "onnx")]
pub mod engine;

use serde::{Deserialize, Serialize};

/// Axis-aligned pixel box: top-left `(x1, y1)`, bottom-right `(x2, y2)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PixelBox {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl PixelBox {
    pub fn width(&self) -> f64 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f64 {
        self.y2 - self.y1
    }
}

/// One recognized text span in source image pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecognizedSpan {
    pub text: String,
    pub bbox: PixelBox,
    /// Mean per-character probability of the recognized text, in [0, 1].
    pub confidence: f64,
    /// Mean detection probability inside the text region, in [0, 1].
    pub det_score: f64,
}

/// Detection and recognition parameters (PaddleOCR defaults unless noted).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OcrConfig {
    /// Longest side fed to the detector. Phone keyframes are read at native
    /// 1920 px so small canvas text is still detected.
    pub det_max_side: u32,
    /// Probability threshold for the binary text mask.
    pub det_threshold: f32,
    /// Minimum mean probability inside a candidate box.
    pub box_threshold: f32,
    /// Box expansion ratio (DB "unclip").
    pub unclip_ratio: f64,
    /// Boxes with a shorter side (detector pixels) are dropped.
    pub min_box_side: u32,
    /// Recognition input height.
    pub rec_height: u32,
    /// Maximum recognition input width (longer crops are squeezed).
    pub rec_max_width: u32,
    /// Crops per recognition batch.
    pub rec_batch: usize,
    /// Spans whose recognition confidence is below this are dropped.
    pub drop_score: f64,
    /// Total CUDA memory arena limit for OCR, MiB, split between the two
    /// sessions by [`OcrConfig::session_limits_mib`]. The server detector at
    /// 1920 px needs about 3 GiB of working memory (spec 5.3 plans 1 to 2 GB).
    pub cuda_mem_limit_mib: u32,
}

impl OcrConfig {
    /// Arena limits `(detector, recognizer)` in MiB: the recognizer gets a
    /// quarter of the total, at most 1024 MiB, and the detector the rest, so
    /// the two never exceed `cuda_mem_limit_mib` together.
    pub fn session_limits_mib(&self) -> (u32, u32) {
        let rec = (self.cuda_mem_limit_mib / 4).min(1024);
        (self.cuda_mem_limit_mib - rec, rec)
    }
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            det_max_side: 1920,
            det_threshold: 0.3,
            box_threshold: 0.6,
            unclip_ratio: 1.5,
            min_box_side: 3,
            rec_height: 48,
            rec_max_width: 3200,
            rec_batch: 8,
            drop_score: 0.5,
            cuda_mem_limit_mib: 4096,
        }
    }
}

/// Errors from OCR.
#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    #[error("OCR model files missing in {dir}: {missing}. {hint}")]
    ModelsMissing {
        dir: String,
        missing: String,
        hint: &'static str,
    },
    #[error("OCR model {file} has sha256 {actual}, expected {expected}")]
    ModelHashMismatch {
        file: String,
        actual: String,
        expected: String,
    },
    #[error("I/O on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("ONNX Runtime: {0}")]
    Runtime(String),
    #[error("unexpected model output: {0}")]
    Output(String),
    #[error("invalid input: {0}")]
    Input(String),
}

/// Anything that turns an RGB image into text spans.
pub trait TextRecognizer: Send + Sync {
    /// Detect and recognize text. Blocking; call from a blocking context.
    fn recognize(&self, image: &image::RgbImage) -> Result<Vec<RecognizedSpan>, OcrError>;

    /// Execution provider actually in use (for example `CUDA` or `CPU`).
    fn execution_provider(&self) -> String;

    /// Identity of the models (file hashes), recorded in cache keys.
    fn model_fingerprint(&self) -> String;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn memory_budget_is_a_total() {
        let c = OcrConfig::default();
        let (det, rec) = c.session_limits_mib();
        assert_eq!(det + rec, c.cuda_mem_limit_mib);
        assert_eq!((det, rec), (3072, 1024));
        let small = OcrConfig {
            cuda_mem_limit_mib: 2048,
            ..OcrConfig::default()
        };
        assert_eq!(small.session_limits_mib(), (1536, 512));
    }
}
