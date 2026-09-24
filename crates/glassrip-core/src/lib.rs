//! Core building blocks for glassrip pipelines.
//!
//! This crate owns the pieces every stage shares:
//!
//! - [`envelope`]: versioned artifact envelopes, per-item outcomes, and schema checks.
//! - [`versioning`]: the pattern for `schema_version`-tagged enums that migrate into
//!   the current type.
//! - [`canonical`]: deterministic (canonical) JSON used for hashing.
//! - [`cache`]: the content-addressed stage cache and its key derivation.
//! - [`atomic`]: crash-safe whole-file writes.
//! - [`jsonl`]: the crash-safe JSONL artifact store with resume semantics.
//! - [`manifest`]: the `run.lock.json` manifest and the run directory lock.
//! - [`graph`]: stage graph validation (DAG, inputs) and stage selection.
//! - [`runner`]: the [`runner::Stage`] trait and the stage runner.
//! - [`config`]: `glassrip.toml` configuration with defaults and validation.
//! - [`error`]: the crate-wide error type.

#![deny(clippy::unwrap_used, clippy::expect_used)]
#![warn(missing_docs)]

pub mod atomic;
pub mod cache;
pub mod canonical;
pub mod envelope;
pub mod graph;
pub mod jsonl;
pub mod manifest;
pub mod versioning;


/// Returns the blake3 hash of `bytes` as lowercase hex.
pub fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Returns the blake3 hash of a file's contents as lowercase hex, streaming the file.
pub fn blake3_file(path: &std::path::Path) -> std::io::Result<String> {
    let mut hasher = blake3::Hasher::new();
    let mut file = fs_err::File::open(path)?;
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hasher.finalize().to_hex().to_string())
}
