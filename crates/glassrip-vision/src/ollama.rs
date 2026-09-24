//! Ollama backend: `/api/chat` with schema-constrained output, `/api/show` and
//! `/api/tags` for digests, `/api/ps` for GPU placement.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use backon::{ExponentialBuilder, Retryable};
use image::{DynamicImage, Rgb, RgbImage};
use reqwest::header::{HeaderMap, RETRY_AFTER};
use reqwest::{Client, StatusCode};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::backend::{
    BackendId, Durations, GenerationOptions, Placement, RawResponse, VisionBackend, VisionRequest,
};
use crate::error::{format_field_errors, FieldError, Result, VisionError};
use crate::image_prep::EncodedImage;
use crate::schema::parse_output_text;

/// Default per-request timeout for 7b-class models.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// Default context window, used for every request in a run.
pub const DEFAULT_NUM_CTX: u32 = 8192;
/// Default server slot count for qwen2.5vl:7b with `OLLAMA_NUM_PARALLEL=4`.
pub const DEFAULT_SLOTS: usize = 4;

/// Backend configuration. `num_ctx` is fixed for the life of the backend.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Server base URL, for example `http://localhost:11434`.
    pub base_url: String,
    pub model: String,
    num_ctx: u32,
    /// `keep_alive` sent with every request (default `30m`).
    pub keep_alive: String,
    /// Value for the top-level `think` field (default `Some(false)`; `None` omits it).
    pub think: Option<bool>,
    pub request_timeout: Duration,
    pub connect_timeout: Duration,
    /// Timeout for metadata calls (`/api/show`, `/api/ps`, `/api/tags`, `/api/version`).
    pub metadata_timeout: Duration,
    /// Total transport attempts per generation, including the first.
    pub max_attempts: u32,
    pub backoff_min: Duration,
    pub backoff_max: Duration,
    /// Upper bound on a server-provided `Retry-After`.
    pub max_retry_after: Duration,
    /// Server slot count (client concurrency when the model is fully on GPU).
    pub slots: usize,
    /// Tolerate a model that is partly in system memory.
    pub allow_spill: bool,
}

impl OllamaConfig {
    /// Defaults for `model` at `base_url` with the given fixed context size.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>, num_ctx: u32) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            num_ctx,
            keep_alive: "30m".to_string(),
            think: Some(false),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: Duration::from_secs(10),
            metadata_timeout: Duration::from_secs(30),
            max_attempts: 5,
            backoff_min: Duration::from_millis(500),
            backoff_max: Duration::from_secs(30),
            max_retry_after: Duration::from_secs(120),
            slots: DEFAULT_SLOTS,
            allow_spill: false,
        }
    }

    /// The fixed context size.
    pub fn num_ctx(&self) -> u32 {
        self.num_ctx
    }
}

/// Answer schema for the startup self-test.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SelfTestAnswer {
    pub dominant_color: SelfTestColor,
    pub contains_text: YesNo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SelfTestColor {
    Red,
    Green,
    Blue,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum YesNo {
    Yes,
    No,
}

/// Result of [`OllamaBackend::self_test`].
#[derive(Debug, Clone)]
pub struct SelfTestReport {
    pub answer: SelfTestAnswer,
    pub response: RawResponse,
    pub latency: Duration,
}

const SELF_TEST_PROMPT: &str =
    "Look at this small synthetic test image. Report its dominant color and whether it contains any text.";

/// Generate the synthetic self-test image: a solid red square.
pub fn self_test_image() -> Result<EncodedImage> {
    let img = RgbImage::from_pixel(112, 112, Rgb([220, 20, 20]));
    EncodedImage::encode(&DynamicImage::ImageRgb8(img))
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: Option<ChatMessage>,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_duration: Option<u64>,
    eval_duration: Option<u64>,
    done_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PsResponse {
    #[serde(default)]
    models: Vec<PsModel>,
}

#[derive(Debug, Deserialize)]
struct PsModel {
    #[serde(default)]
    name: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    size_vram: u64,
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagModel>,
}

#[derive(Debug, Deserialize)]
struct TagModel {
    #[serde(default)]
    name: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    digest: String,
}

/// Outcome of one HTTP attempt, before retry policy is applied.
#[derive(Debug)]
enum AttemptError {
    Retryable {
        message: String,
        retry_after: Option<Duration>,
        timed_out: bool,
    },
    Fatal(VisionError),
}

impl AttemptError {
    fn is_retryable(&self) -> bool {
        matches!(self, AttemptError::Retryable { .. })
    }

    fn retry_after(&self) -> Option<Duration> {
        match self {
            AttemptError::Retryable { retry_after, .. } => *retry_after,
            AttemptError::Fatal(_) => None,
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn nanos(v: Option<u64>) -> Option<Duration> {
    v.map(Duration::from_nanos)
}

/// Parse `Retry-After` as delta seconds or an HTTP date.
pub fn parse_retry_after(headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
    let raw = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    if let Ok(secs) = raw.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    let when = httpdate::parse_http_date(raw).ok()?;
    Some(when.duration_since(now).unwrap_or(Duration::ZERO))
}

fn same_model(candidate_name: &str, candidate_model: &str, wanted: &str) -> bool {
    let norm = |s: &str| {
        if s.contains(':') {
            s.to_string()
        } else {
            format!("{s}:latest")
        }
    };
    let w = norm(wanted);
    (!candidate_name.is_empty() && norm(candidate_name) == w)
        || (!candidate_model.is_empty() && norm(candidate_model) == w)
}

/// Ollama implementation of [`VisionBackend`].
#[derive(Debug)]
pub struct OllamaBackend {
    config: OllamaConfig,
    http: Client,
    digest: Mutex<Option<String>>,
    server_version: Mutex<Option<String>>,
}

impl OllamaBackend {
    /// Validate the configuration and build the HTTP client.
    pub fn new(config: OllamaConfig) -> Result<Self> {
        if !(2048..=131_072).contains(&config.num_ctx) {
            return Err(VisionError::Config(format!(
                "num_ctx {} is outside 2048..=131072",
                config.num_ctx
            )));
        }
        if config.slots == 0 {
            return Err(VisionError::Config("slots must be at least 1".into()));
        }
        if config.max_attempts == 0 {
            return Err(VisionError::Config("max_attempts must be at least 1".into()));
        }
        if config.model.trim().is_empty() {
            return Err(VisionError::Config("model must not be empty".into()));
        }
        if !(config.base_url.starts_with("http://") || config.base_url.starts_with("https://")) {
            return Err(VisionError::Config(format!(
                "base_url must start with http:// or https://, got {}",
                config.base_url
            )));
        }
        let http = Client::builder()
            .connect_timeout(config.connect_timeout)
            .build()
            .map_err(|e| VisionError::Config(format!("cannot build HTTP client: {e}")))?;
        Ok(Self {
            config,
            http,
            digest: Mutex::new(None),
            server_version: Mutex::new(None),
        })
    }

    pub fn config(&self) -> &OllamaConfig {
        &self.config
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.config.base_url)
    }

    fn backoff(&self) -> ExponentialBuilder {
        ExponentialBuilder::default()
            .with_min_delay(self.config.backoff_min)
            .with_max_delay(self.config.backoff_max)
            .with_max_times(self.config.max_attempts.saturating_sub(1) as usize)
            .with_jitter()
    }

    /// One POST, classified into success, retryable failure, or fatal failure.
    async fn post_once(&self, path: &str, body: &Value, timeout: Duration) -> Result<Value, AttemptError> {
        let resp = self
            .http
            .post(self.url(path))
            .timeout(timeout)
            .json(body)
            .send()
            .await
            .map_err(|e| self.classify_reqwest(e))?;
        let status = resp.status();
        if status.is_success() {
            return resp.json::<Value>().await.map_err(|e| {
                if e.is_timeout() {
                    self.classify_reqwest(e)
                } else {
                    AttemptError::Fatal(VisionError::Protocol(format!(
                        "{path} returned undecodable JSON: {e}"
                    )))
                }
            });
        }
        let retry_after = parse_retry_after(resp.headers(), SystemTime::now())
            .map(|d| d.min(self.config.max_retry_after));
        let text = resp.text().await.unwrap_or_default();
        Err(self.classify_status(status, text, retry_after))
    }

    async fn get_once(&self, path: &str) -> Result<Value, AttemptError> {
        let resp = self
            .http
            .get(self.url(path))
            .timeout(self.config.metadata_timeout)
            .send()
            .await
            .map_err(|e| self.classify_reqwest(e))?;
        let status = resp.status();
        if status.is_success() {
            return resp.json::<Value>().await.map_err(|e| {
                AttemptError::Fatal(VisionError::Protocol(format!(
                    "{path} returned undecodable JSON: {e}"
                )))
            });
        }
        let retry_after = parse_retry_after(resp.headers(), SystemTime::now())
            .map(|d| d.min(self.config.max_retry_after));
        let text = resp.text().await.unwrap_or_default();
        Err(self.classify_status(status, text, retry_after))
    }

    fn classify_reqwest(&self, e: reqwest::Error) -> AttemptError {
        if e.is_builder() {
            return AttemptError::Fatal(VisionError::Transport(e.to_string()));
        }
        AttemptError::Retryable {
            timed_out: e.is_timeout(),
            message: e.to_string(),
            retry_after: None,
        }
    }

    fn classify_status(&self, status: StatusCode, body: String, retry_after: Option<Duration>) -> AttemptError {
        if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
            return AttemptError::Retryable {
                message: format!("HTTP {}: {}", status.as_u16(), body.trim()),
                retry_after,
                timed_out: false,
            };
        }
        if status == StatusCode::NOT_FOUND && body.contains("not found") {
            return AttemptError::Fatal(VisionError::ModelNotFound {
                model: self.config.model.clone(),
            });
        }
        AttemptError::Fatal(VisionError::Http {
            status: status.as_u16(),
            body: body.trim().to_string(),
        })
    }

    /// Run `op` with exponential backoff, jitter, `Retry-After`, and cancellation.
    async fn with_retries<F, Fut>(&self, cancel: &CancellationToken, op: F) -> Result<(Value, u32)>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<Value, AttemptError>> + Send,
    {
        let attempts = AtomicU32::new(0);
        let run = || async {
            attempts.fetch_add(1, Ordering::SeqCst);
            op().await
        };
        let retrying = run
            .retry(self.backoff())
            .sleep(tokio::time::sleep)
            .when(AttemptError::is_retryable)
            .adjust(|e: &AttemptError, next: Option<Duration>| match (next, e.retry_after()) {
                (Some(d), Some(ra)) => Some(d.max(ra)),
                (next, _) => next,
            })
            .notify(|e: &AttemptError, wait: Duration| {
                tracing::warn!(model = %self.config.model, ?wait, error = ?e, "retrying Ollama request");
            });
        let result = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(VisionError::Cancelled),
            r = retrying => r,
        };
        let n = attempts.load(Ordering::SeqCst);
        match result {
            Ok(v) => Ok((v, n)),
            Err(AttemptError::Fatal(e)) => Err(e),
            Err(AttemptError::Retryable {
                timed_out: true, ..
            }) => Err(VisionError::Timeout {
                attempts: n,
                timeout: self.config.request_timeout,
            }),
            Err(AttemptError::Retryable { message, .. }) => Err(VisionError::RetriesExhausted {
                attempts: n,
                last: message,
            }),
        }
    }

    fn chat_body(&self, request: &VisionRequest, repair: Option<(&str, &[FieldError])>) -> Value {
        let mut messages = vec![json!({
            "role": "user",
            "content": request.prompt,
            "images": [request.image.base64()],
        })];
        if let Some((bad_text, errors)) = repair {
            messages.push(json!({"role": "assistant", "content": bad_text}));
            messages.push(json!({
                "role": "user",
                "content": format!(
                    "Your previous reply failed validation against the required JSON schema:\n{}\n\
                     Reply again with only a corrected JSON object that satisfies the schema.",
                    format_field_errors(errors)
                ),
            }));
        }
        let GenerationOptions { seed, num_predict } = request.options;
        let mut body = json!({
            "model": self.config.model,
            "messages": messages,
            "stream": false,
            "keep_alive": self.config.keep_alive,
            "format": request.schema.format_value(),
            "options": {
                "temperature": 0,
                "seed": seed,
                "num_ctx": self.config.num_ctx,
                "num_predict": num_predict,
            },
        });
        if let (Some(think), Value::Object(map)) = (self.config.think, &mut body) {
            map.insert("think".into(), Value::Bool(think));
        }
        body
    }

    /// Rough token estimate: image tokens + prompt (about 3 chars per token) + `num_predict`.
    fn check_context(&self, request: &VisionRequest) -> Result<()> {
        let prompt_tokens = u32::try_from(request.prompt.len() / 3).unwrap_or(u32::MAX);
        let estimated = request
            .image
            .tokens()
            .saturating_add(prompt_tokens)
            .saturating_add(request.options.num_predict);
        if estimated > self.config.num_ctx {
            return Err(VisionError::ContextOverflow {
                estimated,
                num_ctx: self.config.num_ctx,
            });
        }
        Ok(())
    }

    async fn generate(&self, body: &Value, cancel: &CancellationToken) -> Result<(ChatResponse, u32, Duration)> {
        let start = Instant::now();
        let (value, attempts) = self
            .with_retries(cancel, || self.post_once("/api/chat", body, self.config.request_timeout))
            .await?;
        let wall = start.elapsed();
        let resp: ChatResponse = serde_json::from_value(value)
            .map_err(|e| VisionError::Protocol(format!("/api/chat response: {e}")))?;
        Ok((resp, attempts, wall))
    }

    fn to_raw(resp: ChatResponse, json: Value, text: String, attempts: u32, wall: Duration, repaired: bool) -> RawResponse {
        RawResponse {
            raw_text: text,
            json,
            prompt_eval_count: resp.prompt_eval_count,
            eval_count: resp.eval_count,
            durations: Durations {
                total: nanos(resp.total_duration),
                load: nanos(resp.load_duration),
                prompt_eval: nanos(resp.prompt_eval_duration),
                eval: nanos(resp.eval_duration),
                wall,
            },
            attempts,
            repaired,
            done_reason: resp.done_reason,
        }
    }

    async fn infer_inner(
        &self,
        request: &VisionRequest,
        cancel: &CancellationToken,
        allow_repair: bool,
    ) -> Result<RawResponse> {
        self.check_context(request)?;
        let body = self.chat_body(request, None);
        let (resp, attempts, wall) = self.generate(&body, cancel).await?;
        let text = resp.message.as_ref().map(|m| m.content.clone()).unwrap_or_default();
        let first_errors = match validate_text(request, &text) {
            Ok(json) => return Ok(Self::to_raw(resp, json, text, attempts, wall, false)),
            Err(errors) => errors,
        };
        if !allow_repair {
            return Err(VisionError::SchemaInvalid {
                attempts: 1,
                errors: first_errors,
                raw_text: text,
            });
        }
        tracing::warn!(
            model = %self.config.model,
            errors = %format_field_errors(&first_errors),
            "model output failed validation; sending one repair request"
        );
        let body = self.chat_body(request, Some((&text, &first_errors)));
        let (resp, attempts, wall) = self.generate(&body, cancel).await?;
        let text = resp.message.as_ref().map(|m| m.content.clone()).unwrap_or_default();
        match validate_text(request, &text) {
            Ok(json) => Ok(Self::to_raw(resp, json, text, attempts, wall, true)),
            Err(errors) => Err(VisionError::SchemaInvalid {
                attempts: 2,
                errors,
                raw_text: text,
            }),
        }
    }

    /// Server version from `/api/version`.
    pub async fn server_version(&self) -> Result<String> {
        let cancel = CancellationToken::new();
        let (v, _) = self.with_retries(&cancel, || self.get_once("/api/version")).await?;
        let version = v
            .get("version")
            .and_then(Value::as_str)
            .ok_or_else(|| VisionError::Protocol("/api/version has no version".into()))?
            .to_string();
        *lock(&self.server_version) = Some(version.clone());
        Ok(version)
    }

    /// Resolve the model digest from `/api/show`, falling back to `/api/tags`.
    ///
    /// Errors with [`VisionError::DigestChanged`] if a different digest was seen earlier.
    pub async fn resolve_digest(&self) -> Result<String> {
        let cancel = CancellationToken::new();
        let body = json!({"model": self.config.model});
        let (show, _) = self
            .with_retries(&cancel, || self.post_once("/api/show", &body, self.config.metadata_timeout))
            .await?;
        let digest = match show.get("digest").and_then(Value::as_str) {
            Some(d) if !d.is_empty() => d.to_string(),
            _ => {
                let (tags, _) = self.with_retries(&cancel, || self.get_once("/api/tags")).await?;
                let tags: TagsResponse = serde_json::from_value(tags)
                    .map_err(|e| VisionError::Protocol(format!("/api/tags response: {e}")))?;
                tags.models
                    .into_iter()
                    .find(|m| same_model(&m.name, &m.model, &self.config.model))
                    .map(|m| m.digest)
                    .filter(|d| !d.is_empty())
                    .ok_or_else(|| VisionError::ModelNotFound {
                        model: self.config.model.clone(),
                    })?
            }
        };
        let mut stored = lock(&self.digest);
        if let Some(previous) = stored.as_ref() {
            if previous != &digest {
                return Err(VisionError::DigestChanged {
                    model: self.config.model.clone(),
                    previous: previous.clone(),
                    current: digest,
                });
            }
        }
        *stored = Some(digest.clone());
        Ok(digest)
    }

    async fn loaded_model(&self) -> Result<Option<PsModel>> {
        let cancel = CancellationToken::new();
        let (ps, _) = self.with_retries(&cancel, || self.get_once("/api/ps")).await?;
        let ps: PsResponse = serde_json::from_value(ps)
            .map_err(|e| VisionError::Protocol(format!("/api/ps response: {e}")))?;
        Ok(ps
            .models
            .into_iter()
            .find(|m| same_model(&m.name, &m.model, &self.config.model)))
    }

    /// Current placement from `/api/ps` without loading or failing on spill.
    /// Returns `None` when the model is not loaded. Use this for periodic mid-run checks.
    pub async fn placement(&self) -> Result<Option<Placement>> {
        Ok(self.loaded_model().await?.map(|m| self.placement_from(&m)))
    }

    fn placement_from(&self, m: &PsModel) -> Placement {
        let fully_on_gpu = m.size_vram >= m.size;
        Placement {
            model: self.config.model.clone(),
            size_bytes: m.size,
            size_vram_bytes: m.size_vram,
            fully_on_gpu,
            context_length: m.context_length,
            concurrency_hint: if fully_on_gpu { self.config.slots } else { 1 },
        }
    }

    /// Ask the server to load the model with this backend's `num_ctx` and `keep_alive`.
    async fn load(&self) -> Result<()> {
        let cancel = CancellationToken::new();
        let body = json!({
            "model": self.config.model,
            "messages": [],
            "keep_alive": self.config.keep_alive,
            "options": {"num_ctx": self.config.num_ctx},
        });
        self.with_retries(&cancel, || self.post_once("/api/chat", &body, self.config.request_timeout))
            .await?;
        Ok(())
    }

    /// Unload the model now (`keep_alive: 0`).
    pub async fn unload(&self) -> Result<()> {
        let cancel = CancellationToken::new();
        let body = json!({"model": self.config.model, "messages": [], "keep_alive": 0});
        self.with_retries(&cancel, || self.post_once("/api/chat", &body, self.config.metadata_timeout))
            .await?;
        Ok(())
    }

    /// Startup self-test: one tiny synthetic image, a two-field enum schema, no repair retry.
    pub async fn self_test(&self, cancel: CancellationToken) -> Result<SelfTestReport> {
        let request = VisionRequest::for_output::<SelfTestAnswer>(
            SELF_TEST_PROMPT,
            self_test_image()?,
            GenerationOptions {
                seed: 1,
                num_predict: 64,
            },
        )?;
        let start = Instant::now();
        let response = match self.infer_inner(&request, &cancel, false).await {
            Ok(r) => r,
            Err(e @ (VisionError::SchemaInvalid { .. } | VisionError::Decode { .. })) => {
                return Err(VisionError::SelfTestFailed(e.to_string()));
            }
            Err(e) => return Err(e),
        };
        let answer: SelfTestAnswer = response
            .decode()
            .map_err(|e| VisionError::SelfTestFailed(e.to_string()))?;
        Ok(SelfTestReport {
            answer,
            response,
            latency: start.elapsed(),
        })
    }
}

fn validate_text(request: &VisionRequest, text: &str) -> std::result::Result<Value, Vec<FieldError>> {
    let json = parse_output_text(text).map_err(|e| vec![e])?;
    request.schema.validate(&json)?;
    Ok(json)
}

#[async_trait]
impl VisionBackend for OllamaBackend {
    fn id(&self) -> BackendId {
        BackendId {
            backend: "ollama".to_string(),
            model: self.config.model.clone(),
            digest: lock(&self.digest).clone(),
            server_version: lock(&self.server_version).clone(),
        }
    }

    /// Resolve version and digest, load the model if needed, and check GPU placement.
    async fn preflight(&self) -> Result<Placement> {
        if let Err(e) = self.server_version().await {
            tracing::warn!(error = %e, "could not read Ollama server version");
        }
        self.resolve_digest().await?;
        let loaded = match self.loaded_model().await? {
            Some(m) => m,
            None => {
                self.load().await?;
                self.loaded_model()
                    .await?
                    .ok_or_else(|| VisionError::ModelNotLoaded {
                        model: self.config.model.clone(),
                    })?
            }
        };
        if let (Some(loaded_digest), Some(known)) = (loaded.digest.as_deref(), lock(&self.digest).as_deref()) {
            if !loaded_digest.is_empty() && loaded_digest != known {
                return Err(VisionError::DigestChanged {
                    model: self.config.model.clone(),
                    previous: known.to_string(),
                    current: loaded_digest.to_string(),
                });
            }
        }
        let placement = self.placement_from(&loaded);
        if !placement.fully_on_gpu {
            if !self.config.allow_spill {
                return Err(VisionError::Spill {
                    model: self.config.model.clone(),
                    size: loaded.size,
                    size_vram: loaded.size_vram,
                });
            }
            tracing::warn!(
                model = %self.config.model,
                size = loaded.size,
                size_vram = loaded.size_vram,
                concurrency_hint = placement.concurrency_hint,
                "model is spilling out of VRAM; continuing because allow_spill is set"
            );
        }
        Ok(placement)
    }

    async fn infer(&self, request: VisionRequest, cancel: CancellationToken) -> Result<RawResponse> {
        self.infer_inner(&request, &cancel, true).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    #[test]
    fn retry_after_seconds_and_date() {
        let mut h = HeaderMap::new();
        h.insert(RETRY_AFTER, HeaderValue::from_static("7"));
        assert_eq!(parse_retry_after(&h, SystemTime::now()), Some(Duration::from_secs(7)));

        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        let later = httpdate::fmt_http_date(now + Duration::from_secs(30));
        let mut h = HeaderMap::new();
        if let Ok(v) = HeaderValue::from_str(&later) {
            h.insert(RETRY_AFTER, v);
        }
        assert_eq!(parse_retry_after(&h, now), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after(&HeaderMap::new(), now), None);
    }

    #[test]
    fn model_names_match_with_implicit_latest() {
        assert!(same_model("qwen2.5vl:7b", "", "qwen2.5vl:7b"));
        assert!(same_model("llava", "", "llava:latest"));
        assert!(!same_model("qwen2.5vl:32b", "qwen2.5vl:32b", "qwen2.5vl:7b"));
    }

    #[test]
    fn constructor_rejects_bad_config() {
        let bad_ctx = OllamaConfig::new("http://localhost:11434", "m", 512);
        assert!(matches!(OllamaBackend::new(bad_ctx), Err(VisionError::Config(_))));
        let mut no_slots = OllamaConfig::new("http://localhost:11434", "m", 8192);
        no_slots.slots = 0;
        assert!(matches!(OllamaBackend::new(no_slots), Err(VisionError::Config(_))));
        let bad_url = OllamaConfig::new("localhost:11434", "m", 8192);
        assert!(matches!(OllamaBackend::new(bad_url), Err(VisionError::Config(_))));
    }
}
