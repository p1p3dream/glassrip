//! Local text model access (Ollama `/api/chat` with a JSON schema `format`).

use std::time::Duration;

use async_trait::async_trait;
use backon::{ExponentialBuilder, Retryable};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A chat message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// `system` or `user`.
    pub role: String,
    /// Content.
    pub content: String,
}

/// One schema-constrained chat request.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    /// What the call is for (`map 1/3`, `reduce`, `repair`), for logs and replay.
    pub purpose: String,
    /// Messages.
    pub messages: Vec<Message>,
    /// JSON schema sent as `format`.
    pub format: Value,
    /// Maximum tokens generated.
    pub num_predict: u32,
}

/// A chat reply.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ChatResponse {
    /// Message content (JSON text when a format was given).
    pub content: String,
    /// Prompt tokens evaluated.
    pub prompt_eval_count: Option<u64>,
    /// Tokens generated.
    pub eval_count: Option<u64>,
    /// Server-side total duration, seconds.
    pub total_duration_s: Option<f64>,
    /// Why generation stopped (`stop`, `length`).
    pub done_reason: Option<String>,
}

/// A model resident on the server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LoadedModel {
    /// Model tag.
    pub name: String,
    /// Bytes in total.
    pub size: u64,
    /// Bytes in VRAM.
    pub size_vram: u64,
    /// Context length, when reported.
    pub context_length: Option<u64>,
}

/// Errors from the text backend.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LlmError {
    /// Transport failure or retryable HTTP status.
    #[error("request failed: {0}")]
    Transport(String),
    /// Non-retryable HTTP status.
    #[error("HTTP {status}: {body}")]
    Http {
        /// Status code.
        status: u16,
        /// Body text.
        body: String,
    },
    /// Unexpected response shape.
    #[error("protocol: {0}")]
    Protocol(String),
}

/// A local text model server.
#[async_trait]
pub trait TextBackend: Send + Sync {
    /// Runs one chat request against `model`.
    async fn chat(&self, model: &str, req: &ChatRequest) -> Result<ChatResponse, LlmError>;
    /// Loads `model` with a fixed context size.
    async fn load(&self, model: &str) -> Result<(), LlmError>;
    /// Unloads `model` now (`keep_alive: 0`).
    async fn unload(&self, model: &str) -> Result<(), LlmError>;
    /// Models currently loaded.
    async fn loaded(&self) -> Result<Vec<LoadedModel>, LlmError>;
    /// Digest of `model`, when the server reports one.
    async fn digest(&self, model: &str) -> Result<Option<String>, LlmError>;
}

/// Ollama settings for the text model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OllamaTextConfig {
    /// Base URL, for example `http://localhost:11434`.
    pub base_url: String,
    /// Fixed context size (changing it reloads the model).
    pub num_ctx: u32,
    /// `keep_alive` for requests.
    pub keep_alive: String,
    /// Sampling temperature.
    pub temperature: f64,
    /// Seed.
    pub seed: u64,
    /// Per-request timeout, seconds.
    pub request_timeout_s: u64,
    /// Attempts per request, including the first.
    pub max_attempts: u32,
}

impl Default for OllamaTextConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".into(),
            num_ctx: 16384,
            keep_alive: "10m".into(),
            temperature: 0.0,
            seed: 42,
            request_timeout_s: 900,
            max_attempts: 4,
        }
    }
}

/// Ollama implementation of [`TextBackend`].
#[derive(Debug, Clone)]
pub struct OllamaText {
    cfg: OllamaTextConfig,
    http: reqwest::Client,
}

enum Attempt {
    Retry(LlmError),
    Fatal(LlmError),
}

impl OllamaText {
    /// Builds a client.
    pub fn new(cfg: OllamaTextConfig) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| LlmError::Transport(e.to_string()))?;
        Ok(Self { cfg, http })
    }

    async fn call(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
        timeout: Duration,
    ) -> Result<Value, LlmError> {
        let url = format!("{}{path}", self.cfg.base_url.trim_end_matches('/'));
        let once = || async {
            let mut rb = self.http.request(method.clone(), &url).timeout(timeout);
            if let Some(b) = body {
                rb = rb.json(b);
            }
            let resp = rb
                .send()
                .await
                .map_err(|e| Attempt::Retry(LlmError::Transport(e.to_string())))?;
            let status = resp.status();
            let text = resp
                .text()
                .await
                .map_err(|e| Attempt::Retry(LlmError::Transport(e.to_string())))?;
            if status.is_success() {
                return serde_json::from_str::<Value>(&text)
                    .map_err(|e| Attempt::Fatal(LlmError::Protocol(format!("{path}: {e}"))));
            }
            let err = LlmError::Http {
                status: status.as_u16(),
                body: text.chars().take(500).collect(),
            };
            if status.is_server_error() || status.as_u16() == 429 {
                Err(Attempt::Retry(err))
            } else {
                Err(Attempt::Fatal(err))
            }
        };
        let backoff = ExponentialBuilder::default()
            .with_min_delay(Duration::from_millis(500))
            .with_max_delay(Duration::from_secs(20))
            .with_max_times(self.cfg.max_attempts.saturating_sub(1) as usize)
            .with_jitter();
        once.retry(backoff)
            .when(|e| matches!(e, Attempt::Retry(_)))
            .await
            .map_err(|e| match e {
                Attempt::Retry(e) | Attempt::Fatal(e) => e,
            })
    }
}

#[async_trait]
impl TextBackend for OllamaText {
    async fn chat(&self, model: &str, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let body = json!({
            "model": model,
            "messages": req.messages,
            "stream": false,
            "think": false,
            "format": req.format,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "num_ctx": self.cfg.num_ctx,
                "num_predict": req.num_predict,
                "temperature": self.cfg.temperature,
                "seed": self.cfg.seed,
            },
        });
        let v = self
            .call(
                reqwest::Method::POST,
                "/api/chat",
                Some(&body),
                Duration::from_secs(self.cfg.request_timeout_s),
            )
            .await?;
        let content = v
            .pointer("/message/content")
            .and_then(Value::as_str)
            .ok_or_else(|| LlmError::Protocol("/api/chat reply has no message content".into()))?
            .to_string();
        Ok(ChatResponse {
            content,
            prompt_eval_count: v.get("prompt_eval_count").and_then(Value::as_u64),
            eval_count: v.get("eval_count").and_then(Value::as_u64),
            total_duration_s: v
                .get("total_duration")
                .and_then(Value::as_u64)
                .map(|ns| ns as f64 / 1e9),
            done_reason: v
                .get("done_reason")
                .and_then(Value::as_str)
                .map(str::to_string),
        })
    }

    async fn load(&self, model: &str) -> Result<(), LlmError> {
        let body = json!({
            "model": model, "messages": [], "stream": false,
            "keep_alive": self.cfg.keep_alive,
            "options": {"num_ctx": self.cfg.num_ctx},
        });
        self.call(
            reqwest::Method::POST,
            "/api/chat",
            Some(&body),
            Duration::from_secs(self.cfg.request_timeout_s),
        )
        .await
        .map(|_| ())
    }

    async fn unload(&self, model: &str) -> Result<(), LlmError> {
        let body = json!({"model": model, "messages": [], "stream": false, "keep_alive": 0});
        self.call(
            reqwest::Method::POST,
            "/api/chat",
            Some(&body),
            Duration::from_secs(60),
        )
        .await
        .map(|_| ())
    }

    async fn loaded(&self) -> Result<Vec<LoadedModel>, LlmError> {
        let v = self
            .call(
                reqwest::Method::GET,
                "/api/ps",
                None,
                Duration::from_secs(30),
            )
            .await?;
        Ok(v.get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|m| LoadedModel {
                name: m
                    .get("name")
                    .or_else(|| m.get("model"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                size: m.get("size").and_then(Value::as_u64).unwrap_or(0),
                size_vram: m.get("size_vram").and_then(Value::as_u64).unwrap_or(0),
                context_length: m.get("context_length").and_then(Value::as_u64),
            })
            .collect())
    }

    async fn digest(&self, model: &str) -> Result<Option<String>, LlmError> {
        let v = self
            .call(
                reqwest::Method::POST,
                "/api/show",
                Some(&json!({"model": model})),
                Duration::from_secs(30),
            )
            .await?;
        Ok(v.get("digest")
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|s| !s.is_empty()))
    }
}

/// True when two model tags name the same model (`name` vs `name:latest`).
pub fn same_model(a: &str, b: &str) -> bool {
    let norm = |s: &str| {
        if s.contains(':') {
            s.to_string()
        } else {
            format!("{s}:latest")
        }
    };
    norm(a) == norm(b)
}
