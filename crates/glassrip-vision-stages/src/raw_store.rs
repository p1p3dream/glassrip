//! Raw model responses stored by request key, for offline replay.
//!
//! The key is derived from everything the server sees (model name, prompt,
//! schema, image bytes, seed, `num_predict`), so a replay run that builds the
//! same requests finds the same responses. [`RecordingBackend`] writes every
//! successful response; [`ReplayBackend`] serves them without a server.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use glassrip_vision::{
    BackendId, Placement, RawResponse, VisionBackend, VisionError, VisionRequest,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// Replay key for a request.
pub fn request_key(model: &str, request: &VisionRequest) -> String {
    let body = json!({
        "domain": "glassrip.vision_request.v1",
        "model": model,
        "prompt": request.prompt,
        "schema": request.schema.json(),
        "image_blake3": blake3::hash(request.image.base64().as_bytes()).to_hex().to_string(),
        "image_size": [request.image.width(), request.image.height()],
        "seed": request.options.seed,
        "num_predict": request.options.num_predict,
    });
    let text = glassrip_core::canonical::canonical_value_string(&body);
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

/// One stored response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedResponse {
    pub key: String,
    pub model: String,
    pub digest: Option<String>,
    pub response: RawResponse,
    /// The reply stopped at the output limit: `response` holds the cut-off text
    /// (`json` is null) and replay answers with [`VisionError::Truncated`].
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// The stored form of a reply cut off at the output limit.
fn truncated_response(raw_text: &str, eval_count: Option<u32>) -> RawResponse {
    RawResponse {
        raw_text: raw_text.to_string(),
        json: serde_json::Value::Null,
        prompt_eval_count: None,
        eval_count,
        durations: glassrip_vision::Durations::default(),
        attempts: 1,
        repaired: false,
        done_reason: Some("length".into()),
    }
}

/// Directory of recorded responses: `<dir>/<key[0..2]>/<key>.json`.
#[derive(Debug, Clone)]
pub struct RawStore {
    dir: PathBuf,
}

/// Raw store failure.
#[derive(Debug, thiserror::Error)]
pub enum RawStoreError {
    #[error("raw store I/O on {path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("raw store entry {path} is not valid: {message}")]
    Parse { path: PathBuf, message: String },
}

impl RawStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn path(&self, key: &str) -> PathBuf {
        let prefix = key.get(..2).unwrap_or("xx");
        self.dir.join(prefix).join(format!("{key}.json"))
    }

    pub fn get(&self, key: &str) -> Result<Option<RecordedResponse>, RawStoreError> {
        let path = self.path(key);
        let bytes = match fs_err::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(RawStoreError::Io {
                    path,
                    message: e.to_string(),
                })
            }
        };
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| RawStoreError::Parse {
                path,
                message: e.to_string(),
            })
    }

    pub fn put(&self, entry: &RecordedResponse) -> Result<(), RawStoreError> {
        let path = self.path(&entry.key);
        let io = |message: String| RawStoreError::Io {
            path: path.clone(),
            message,
        };
        if let Some(parent) = path.parent() {
            fs_err::create_dir_all(parent).map_err(|e| io(e.to_string()))?;
        }
        let bytes = serde_json::to_vec_pretty(entry).map_err(|e| io(e.to_string()))?;
        glassrip_core::atomic::write_atomic(&path, &bytes).map_err(|e| io(e.to_string()))
    }
}

/// Wraps a live backend and records every successful response.
pub struct RecordingBackend {
    inner: Arc<dyn VisionBackend>,
    store: RawStore,
}

impl RecordingBackend {
    pub fn new(inner: Arc<dyn VisionBackend>, store: RawStore) -> Self {
        Self { inner, store }
    }
}

#[async_trait]
impl VisionBackend for RecordingBackend {
    fn id(&self) -> BackendId {
        self.inner.id()
    }

    async fn preflight(&self) -> glassrip_vision::Result<Placement> {
        self.inner.preflight().await
    }

    async fn infer(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> glassrip_vision::Result<RawResponse> {
        let id = self.inner.id();
        let key = request_key(&id.model, &request);
        let response = match self.inner.infer(request, cancel).await {
            Ok(r) => r,
            Err(e) => {
                // A truncation is a deterministic answer to this request; record
                // it so replay takes the same (retry) path.
                if let VisionError::Truncated {
                    raw_text,
                    eval_count,
                    ..
                } = &e
                {
                    let entry = RecordedResponse {
                        key,
                        model: id.model,
                        digest: id.digest,
                        response: truncated_response(raw_text, *eval_count),
                        truncated: true,
                    };
                    if let Err(w) = self.store.put(&entry) {
                        tracing::warn!(error = %w, "could not record truncated response");
                    }
                }
                return Err(e);
            }
        };
        let entry = RecordedResponse {
            key,
            model: id.model,
            digest: id.digest,
            response: response.clone(),
            truncated: false,
        };
        // A response that cannot be recorded is still a valid answer; replay
        // will report the missing key.
        if let Err(e) = self.store.put(&entry) {
            tracing::warn!(error = %e, "could not record raw response");
        }
        Ok(response)
    }
}

/// Serves recorded responses; never contacts a server.
pub struct ReplayBackend {
    model: String,
    store: RawStore,
}

impl ReplayBackend {
    pub fn new(model: impl Into<String>, store: RawStore) -> Self {
        Self {
            model: model.into(),
            store,
        }
    }
}

#[async_trait]
impl VisionBackend for ReplayBackend {
    fn id(&self) -> BackendId {
        BackendId {
            backend: "replay".into(),
            model: self.model.clone(),
            digest: None,
            server_version: None,
        }
    }

    async fn preflight(&self) -> glassrip_vision::Result<Placement> {
        Ok(Placement {
            model: self.model.clone(),
            size_bytes: 0,
            size_vram_bytes: 0,
            fully_on_gpu: true,
            context_length: None,
            concurrency_hint: 1,
        })
    }

    async fn infer(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> glassrip_vision::Result<RawResponse> {
        if cancel.is_cancelled() {
            return Err(VisionError::Cancelled);
        }
        let key = request_key(&self.model, &request);
        match self.store.get(&key) {
            Ok(Some(entry)) if entry.truncated => Err(VisionError::Truncated {
                num_predict: request.options.num_predict,
                eval_count: entry.response.eval_count,
                raw_text: entry.response.raw_text,
            }),
            Ok(Some(entry)) => {
                // Validate against the request schema, as a live reply would be.
                request
                    .schema
                    .validate(&entry.response.json)
                    .map_err(|errors| VisionError::SchemaInvalid {
                        attempts: 1,
                        errors,
                        raw_text: entry.response.raw_text.clone(),
                    })?;
                let mut r = entry.response;
                r.durations.wall = Duration::ZERO;
                Ok(r)
            }
            Ok(None) => Err(VisionError::Config(format!(
                "no recorded response for request key {key}"
            ))),
            Err(e) => Err(VisionError::Protocol(e.to_string())),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use glassrip_vision::{Durations, EncodedImage, GenerationOptions};
    use image::{DynamicImage, RgbImage};
    use schemars::JsonSchema;

    #[derive(Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    #[allow(dead_code)]
    struct Answer {
        n: u32,
    }

    fn request(seed: u64) -> VisionRequest {
        let img = DynamicImage::ImageRgb8(RgbImage::new(28, 28));
        let image = EncodedImage::encode(&img).unwrap();
        VisionRequest::for_output::<Answer>(
            "count",
            image,
            GenerationOptions {
                seed,
                num_predict: 16,
            },
        )
        .unwrap()
    }

    struct Fixed;

    #[async_trait]
    impl VisionBackend for Fixed {
        fn id(&self) -> BackendId {
            BackendId {
                backend: "fixed".into(),
                model: "m".into(),
                digest: Some("d".into()),
                server_version: None,
            }
        }
        async fn preflight(&self) -> glassrip_vision::Result<Placement> {
            Err(VisionError::Config("unused".into()))
        }
        async fn infer(
            &self,
            _r: VisionRequest,
            _c: CancellationToken,
        ) -> glassrip_vision::Result<RawResponse> {
            Ok(RawResponse {
                raw_text: "{\"n\": 3}".into(),
                json: json!({"n": 3}),
                prompt_eval_count: Some(10),
                eval_count: Some(5),
                durations: Durations::default(),
                attempts: 1,
                repaired: false,
                done_reason: Some("stop".into()),
            })
        }
    }

    #[test]
    fn key_depends_on_every_input() {
        let a = request_key("m", &request(1));
        assert_eq!(a, request_key("m", &request(1)));
        assert_ne!(a, request_key("m", &request(2)));
        assert_ne!(a, request_key("other", &request(1)));
    }

    #[tokio::test]
    async fn record_then_replay() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        let rec = RecordingBackend::new(Arc::new(Fixed), store.clone());
        let live = rec
            .infer(request(7), CancellationToken::new())
            .await
            .unwrap();
        let replay = ReplayBackend::new("m", store);
        let back = replay
            .infer(request(7), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(back.json, live.json);
        let missing = replay.infer(request(8), CancellationToken::new()).await;
        assert!(matches!(missing, Err(VisionError::Config(m)) if m.contains("no recorded")));
    }
}
