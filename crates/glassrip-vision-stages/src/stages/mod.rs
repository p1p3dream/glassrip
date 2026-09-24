//! The vision-branch stages.

pub mod board_read;
pub mod board_validate;
pub mod canvas_crop;
pub mod classify;
pub mod ocr_harvest;
pub mod vocabulary;

use std::path::{Path, PathBuf};

use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::{InputDecl, StageError, StageInputs};

use crate::artifacts::INPUT_MAJOR;

/// Declare an input artifact at the major version these stages understand.
pub(crate) fn input(schema: &'static str) -> InputDecl {
    InputDecl {
        schema,
        major: INPUT_MAJOR,
    }
}

/// Run directory of the artifacts a stage reads (`<run>/artifacts/<x>.jsonl`).
pub(crate) fn run_root(inputs: &StageInputs, schema: &str) -> Result<PathBuf, StageError> {
    let p = inputs.path(schema)?;
    p.parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            StageError::Invalid(format!("artifact path {} has no run root", p.display()))
        })
}

/// Resolve an artifact-relative path against the run directory.
pub(crate) fn resolve(root: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> ErrorInfo {
    ErrorInfo::new(ErrorCode::InvalidInput, message)
}

pub(crate) fn internal(message: impl Into<String>) -> ErrorInfo {
    ErrorInfo::new(ErrorCode::Internal, message)
}

/// Decode an image file on a blocking thread.
pub(crate) async fn load_rgb(path: PathBuf) -> Result<image::RgbImage, ErrorInfo> {
    tokio::task::spawn_blocking(move || {
        image::open(&path)
            .map(|i| i.to_rgb8())
            .map_err(|e| invalid(format!("cannot read image {}: {e}", path.display())))
    })
    .await
    .map_err(|e| internal(format!("image task failed: {e}")))?
}

/// blake3 of a string (prompt hashes for cache keys).
pub(crate) fn text_hash(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}
