//! Typed errors for the vision crate.

use std::fmt;
use std::time::Duration;

/// One validation failure, located by a JSON path (for example `/nodes/2/bbox/x1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldError {
    /// JSON pointer style path to the offending value (`""` means the document root).
    pub path: String,
    /// Human readable description of the failure.
    pub message: String,
}

impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = if self.path.is_empty() {
            "/"
        } else {
            &self.path
        };
        write!(f, "{path}: {}", self.message)
    }
}

/// Render a list of field errors as one line per error.
pub fn format_field_errors(errors: &[FieldError]) -> String {
    errors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every failure the vision layer can report.
#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    /// The caller's cancellation token fired.
    #[error("request cancelled")]
    Cancelled,

    /// Every attempt timed out.
    #[error("request timed out after {attempts} attempt(s) (per-request timeout {timeout:?})")]
    Timeout { attempts: u32, timeout: Duration },

    /// Retryable failures (connect errors, 5xx, 429) persisted through every attempt.
    #[error("retries exhausted after {attempts} attempt(s); last error: {last}")]
    RetriesExhausted { attempts: u32, last: String },

    /// The server answered with a non-retryable status.
    #[error("{path} returned HTTP {status}: {body}")]
    Http {
        path: String,
        status: u16,
        body: String,
    },

    /// A transport error that is not worth retrying (for example an invalid URL).
    #[error("transport error: {0}")]
    Transport(String),

    /// The server response did not have the expected shape.
    #[error("unexpected server response: {0}")]
    Protocol(String),

    /// The model output failed schema or type validation, including after the repair retry.
    #[error("model output failed validation after {attempts} attempt(s):\n{}", format_field_errors(.errors))]
    SchemaInvalid {
        attempts: u32,
        errors: Vec<FieldError>,
        raw_text: String,
    },

    /// Generation stopped at the output limit (`done_reason: length`), so the reply
    /// is incomplete. A schema repair would echo the cut-off text back, so the
    /// caller decides instead (for example a retry with a smaller list budget).
    #[error(
        "generation stopped at the output limit of {num_predict} tokens (done_reason length) \
         after {} bytes of output",
        .raw_text.len()
    )]
    Truncated {
        num_predict: u32,
        eval_count: Option<u32>,
        raw_text: String,
    },

    /// Decoding an already-validated value into the caller's type failed.
    #[error("could not decode model output:\n{}", format_field_errors(.errors))]
    Decode { errors: Vec<FieldError> },

    /// The model is not available on the server.
    #[error("model {model} is not available on the server; run `ollama pull {model}`")]
    ModelNotFound { model: String },

    /// Preflight found the model partly in system memory.
    #[error(
        "model {model} is spilling out of VRAM: size {size} bytes, size_vram {size_vram} bytes \
         (pass allow_spill to run anyway)"
    )]
    Spill {
        model: String,
        size: u64,
        size_vram: u64,
    },

    /// Preflight could not find the model among loaded models after loading it.
    #[error("model {model} is not loaded on the server after a load request")]
    ModelNotLoaded { model: String },

    /// The model digest changed between preflights within one run.
    #[error("model {model} digest changed mid-run: {previous} -> {current}")]
    DigestChanged {
        model: String,
        previous: String,
        current: String,
    },

    /// The image would exceed the vision encoder's token budget.
    #[error("image {width}x{height} needs {tokens} image tokens; the cap is {max}")]
    ImageTooManyTokens {
        width: u32,
        height: u32,
        tokens: u32,
        max: u32,
    },

    /// The request would not fit in the fixed context window.
    #[error("estimated {estimated} tokens exceed num_ctx {num_ctx}")]
    ContextOverflow { estimated: u32, num_ctx: u32 },

    /// Image decode or encode failed.
    #[error("image processing failed: {0}")]
    Image(#[from] image::ImageError),

    /// Invalid configuration or request construction.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// The startup self-test did not produce a parseable reply.
    #[error("self-test failed: {0}")]
    SelfTestFailed(String),
}

impl VisionError {
    /// True when retrying the same request later could succeed.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            VisionError::Timeout { .. } | VisionError::RetriesExhausted { .. }
        )
    }

    /// True when the reply was cut off at the output limit.
    pub fn is_truncated(&self) -> bool {
        matches!(self, VisionError::Truncated { .. })
    }
}

/// Convenience alias.
pub type Result<T, E = VisionError> = std::result::Result<T, E>;
