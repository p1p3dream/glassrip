//! Typed errors for the audio branch.

use std::path::PathBuf;

/// Result alias used across the crate.
pub type Result<T, E = AudioError> = std::result::Result<T, E>;

/// Errors produced by the audio branch.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// Filesystem or process IO failed.
    #[error("io error on {path}: {source}")]
    Io {
        /// Path or command involved.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// An external command (ffmpeg, ffprobe) exited unsuccessfully.
    #[error("{command} failed with status {status}: {stderr_tail}")]
    Command {
        /// Program name.
        command: String,
        /// Exit status as text.
        status: String,
        /// Last part of stderr.
        stderr_tail: String,
    },
    /// The input has no audio stream.
    #[error("no audio stream in {0}")]
    NoAudioStream(PathBuf),
    /// Output from a tool could not be parsed.
    #[error("could not parse {what}: {message}")]
    Parse {
        /// What was being parsed.
        what: String,
        /// Parser message.
        message: String,
    },
    /// whisper.cpp reported an error.
    #[error("whisper: {0}")]
    Whisper(String),
    /// The diarization backend reported an error.
    #[error("diarization: {0}")]
    Diarization(String),
    /// A model file is missing or does not match its manifest hash.
    #[error("model {name}: {message}")]
    Model {
        /// Model name or path.
        name: String,
        /// What is wrong.
        message: String,
    },
    /// Invalid configuration or input.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// A capability was requested that this build does not include.
    #[error("this build was compiled without the `{0}` feature")]
    FeatureDisabled(&'static str),
    /// A blocking task panicked or was cancelled.
    #[error("background task failed: {0}")]
    Task(String),
}

impl AudioError {
    /// Wrap an IO error with the path it concerns.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

impl From<whisper_rs::WhisperError> for AudioError {
    fn from(value: whisper_rs::WhisperError) -> Self {
        Self::Whisper(value.to_string())
    }
}
