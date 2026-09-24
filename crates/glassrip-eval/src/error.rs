//! Error type for the eval crate.

use std::path::PathBuf;

/// Errors raised while loading fixtures, replaying responses, or scoring.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// Filesystem error with the path involved.
    #[error("{path}: {source}")]
    Io {
        /// Path that failed.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// JSON that could not be parsed or does not match the expected shape.
    #[error("{path}: invalid JSON: {message}")]
    Json {
        /// File that failed.
        path: PathBuf,
        /// Parser message.
        message: String,
    },
    /// TOML that could not be parsed.
    #[error("{path}: invalid TOML: {message}")]
    Toml {
        /// File that failed.
        path: PathBuf,
        /// Parser message.
        message: String,
    },
    /// Image decode or encode failure.
    #[error("{path}: image error: {message}")]
    Image {
        /// File that failed.
        path: PathBuf,
        /// Message.
        message: String,
    },
    /// A fixture is structurally invalid (missing files, bad references).
    #[error("fixture {case}: {message}")]
    Fixture {
        /// Case name or path.
        case: String,
        /// What is wrong.
        message: String,
    },
    /// Replay mode found no recorded response for a request.
    #[error(
        "no recorded response for {kind} request of case {case} (key {key}); \
         run with --rerecord against a live server to record it"
    )]
    MissingResponse {
        /// Request kind (for example `classify`).
        kind: String,
        /// Case name.
        case: String,
        /// Replay key.
        key: String,
    },
    /// Error from the vision crate (live backend, schema validation, decoding).
    #[error("vision: {0}")]
    Vision(#[from] glassrip_vision::VisionError),
    /// Configuration problem.
    #[error("config: {0}")]
    Config(String),
    /// Anything else, with context.
    #[error("{0}")]
    Other(String),
}

/// Result alias for the eval crate.
pub type Result<T, E = EvalError> = std::result::Result<T, E>;

impl EvalError {
    /// Wraps an IO error with its path.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Wraps a JSON error with its path.
    pub fn json(path: impl Into<PathBuf>, message: impl ToString) -> Self {
        Self::Json {
            path: path.into(),
            message: message.to_string(),
        }
    }
}

/// Reads a file to a string, attaching the path to errors.
pub(crate) fn read_to_string(path: &std::path::Path) -> Result<String> {
    fs_err::read_to_string(path).map_err(|e| EvalError::io(path, e))
}

/// Reads and parses a JSON file.
pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    let text = read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| EvalError::json(path, e))
}

/// Reads and parses a TOML file.
pub(crate) fn read_toml<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    let text = read_to_string(path)?;
    toml::from_str(&text).map_err(|e| EvalError::Toml {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

/// Writes pretty JSON with a trailing newline, creating parent directories.
pub(crate) fn write_json<T: serde::Serialize>(path: &std::path::Path, value: &T) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value).map_err(|e| EvalError::json(path, e))?;
    text.push('\n');
    write_text(path, &text)
}

/// Writes text, creating parent directories.
pub(crate) fn write_text(path: &std::path::Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs_err::create_dir_all(parent).map_err(|e| EvalError::io(parent, e))?;
    }
    fs_err::write(path, text).map_err(|e| EvalError::io(path, e))
}
