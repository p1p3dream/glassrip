//! Error type for the media crate.

use std::path::PathBuf;

/// Errors produced by decoding, feature extraction and segmentation.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// Reading a file failed.
    #[error("failed to read {path}: {source}")]
    Io {
        /// File that could not be read.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// JPEG decoding failed.
    #[error("failed to decode JPEG {path}: {message}")]
    Decode {
        /// File that could not be decoded.
        path: PathBuf,
        /// Decoder message.
        message: String,
    },
    /// Image dimensions do not match what an operation requires.
    #[error("unexpected image size {width}x{height}: {reason}")]
    Size {
        /// Actual width.
        width: usize,
        /// Actual height.
        height: usize,
        /// Why the size is not acceptable.
        reason: &'static str,
    },
    /// Invalid input to segmentation or scoring.
    #[error("invalid input: {0}")]
    Invalid(String),
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, MediaError>;
