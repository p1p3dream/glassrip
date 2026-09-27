//! The backend-neutral vision interface.

use std::time::Duration;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::error::{Result, VisionError};
use crate::image_prep::EncodedImage;
use crate::schema::{decode_value, prompt_with_schema, OutputSchema};

/// Identity of a backend, recorded in run manifests and cache keys.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendId {
    /// Backend implementation name, for example `ollama`.
    pub backend: String,
    /// Model tag, for example `qwen2.5vl:7b`.
    pub model: String,
    /// Model digest, known after `preflight` (or `resolve_digest`) has run.
    pub digest: Option<String>,
    /// Server version string when the backend exposes one.
    pub server_version: Option<String>,
}

/// Where the model sits after preflight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    pub model: String,
    /// Total bytes the loaded model occupies.
    pub size_bytes: u64,
    /// Bytes resident in VRAM.
    pub size_vram_bytes: u64,
    /// True when `size_vram_bytes >= size_bytes`.
    pub fully_on_gpu: bool,
    /// Context length the server loaded the model with, when reported.
    pub context_length: Option<u32>,
    /// Recommended number of in-flight requests. Equals the configured slot
    /// count when fully on GPU; reduced when spilling was allowed.
    pub concurrency_hint: usize,
}

/// Per-request generation options that may vary between requests.
///
/// `num_ctx` is deliberately absent: it is fixed per backend instance, because
/// changing it makes Ollama reload the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub seed: u64,
    pub num_predict: u32,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            num_predict: 2048,
        }
    }
}

/// Sampling settings a retry or a sampled consensus read may change. Unset
/// fields leave the server's defaults (temperature 0), and are left out of
/// request keys, so requests without overrides keep the keys (and recorded
/// responses) they always had.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SamplingOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Ollama `repeat_penalty` (the server default is 1.1).
    pub repeat_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Ollama `repeat_last_n`: how many recent tokens the penalty looks back over
    /// (the server default is 64, shorter than one board list item).
    pub repeat_last_n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Sampling temperature (unset: 0, greedy). A consensus board read samples
    /// its independent reads above 0.
    pub temperature: Option<f32>,
}

impl SamplingOverrides {
    /// No override set.
    pub fn is_empty(&self) -> bool {
        self.repeat_penalty.is_none() && self.repeat_last_n.is_none() && self.temperature.is_none()
    }
}

/// One schema-constrained request carrying exactly one image.
#[derive(Debug, Clone)]
pub struct VisionRequest {
    /// Prompt text, already including the schema instructions.
    pub prompt: String,
    /// The one image. A single field (not a list) makes multi-image requests unrepresentable.
    pub image: EncodedImage,
    /// Output schema; also sent as the server-side `format` constraint.
    pub schema: OutputSchema,
    pub options: GenerationOptions,
    /// Sampling overrides (retries only).
    pub sampling: SamplingOverrides,
    /// Stop the reply as soon as it falls into a repetition loop
    /// ([`crate::repetition`]); a streaming backend checks while it generates, and
    /// every backend checks the returned text. The server sees the same request
    /// either way, so the guard is not part of the request key.
    pub repetition_guard: Option<crate::repetition::RepetitionParams>,
}

impl VisionRequest {
    /// Build a request whose output must decode as `T`. The schema text is appended to `prompt`.
    pub fn for_output<T>(
        prompt: &str,
        image: EncodedImage,
        options: GenerationOptions,
    ) -> Result<Self>
    where
        T: schemars::JsonSchema + DeserializeOwned + 'static,
    {
        let schema = OutputSchema::for_type::<T>()?;
        Ok(Self {
            prompt: prompt_with_schema(prompt, &schema),
            image,
            schema,
            options,
            sampling: SamplingOverrides::default(),
            repetition_guard: None,
        })
    }
}

/// Timing information reported by the server (all optional; not every backend reports them).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Durations {
    pub total: Option<Duration>,
    pub load: Option<Duration>,
    pub prompt_eval: Option<Duration>,
    pub eval: Option<Duration>,
    /// Client-measured wall time for the successful attempt.
    pub wall: Duration,
}

/// A validated model reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawResponse {
    /// Text exactly as the model produced it.
    pub raw_text: String,
    /// Parsed JSON, already validated against the request schema and type.
    pub json: Value,
    pub prompt_eval_count: Option<u32>,
    pub eval_count: Option<u32>,
    pub durations: Durations,
    /// Transport attempts spent on the final (successful) generation.
    pub attempts: u32,
    /// True when the answer came from the schema repair retry.
    pub repaired: bool,
    /// Server `done_reason` when reported (for example `stop` or `length`).
    pub done_reason: Option<String>,
}

impl RawResponse {
    /// Decode the validated JSON into the caller's type.
    pub fn decode<T: DeserializeOwned>(&self) -> Result<T> {
        decode_value(&self.json).map_err(|errors| VisionError::Decode { errors })
    }
}

/// A vision model server.
#[async_trait]
pub trait VisionBackend: Send + Sync {
    /// Backend name, model, and digest (digest present once resolved).
    fn id(&self) -> BackendId;

    /// Make sure the model is available and loaded, and report where it sits.
    async fn preflight(&self) -> Result<Placement>;

    /// Run one request. Retries, timeouts, and the schema repair retry are handled inside.
    async fn infer(&self, request: VisionRequest, cancel: CancellationToken)
        -> Result<RawResponse>;
}
