//! Crate-wide error type.
//!
//! Each module has its own `thiserror` enum; [`Error`] wraps them for callers that
//! want a single type.

use crate::atomic::AtomicWriteError;
use crate::cache::CacheError;
use crate::canonical::CanonicalJsonError;
use crate::config::ConfigError;
use crate::envelope::{EnvelopeReadError, SchemaError};
use crate::graph::GraphError;
use crate::jsonl::JsonlError;
use crate::manifest::ManifestError;
use crate::runner::RunnerError;
use crate::versioning::VersionedDecodeError;

/// Any error from this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Schema name or version mismatch.
    #[error(transparent)]
    Schema(#[from] SchemaError),
    /// Envelope read failure.
    #[error(transparent)]
    EnvelopeRead(#[from] EnvelopeReadError),
    /// Versioned value decode or migration failure.
    #[error(transparent)]
    Versioned(#[from] VersionedDecodeError),
    /// Canonical JSON failure.
    #[error(transparent)]
    Canonical(#[from] CanonicalJsonError),
    /// Atomic write failure.
    #[error(transparent)]
    AtomicWrite(#[from] AtomicWriteError),
    /// Cache failure.
    #[error(transparent)]
    Cache(#[from] CacheError),
    /// JSONL store failure.
    #[error(transparent)]
    Jsonl(#[from] JsonlError),
    /// Manifest or lock failure.
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    /// Stage graph failure.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Stage runner failure.
    #[error(transparent)]
    Runner(#[from] RunnerError),
    /// Configuration failure.
    #[error(transparent)]
    Config(#[from] ConfigError),
}

/// Result alias using [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
