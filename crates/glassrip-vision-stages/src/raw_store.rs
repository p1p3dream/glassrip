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
use glassrip_vision::repetition::RepetitionFinding;
use glassrip_vision::{
    BackendId, Placement, RawResponse, VisionBackend, VisionError, VisionRequest,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// Replay key for a request. Sampling overrides are part of the key only when set,
/// so requests without them keep their original keys. The repetition guard is not:
/// the server sees the same request with or without it.
pub fn request_key(model: &str, request: &VisionRequest) -> String {
    let mut body = json!({
        "domain": "glassrip.vision_request.v1",
        "model": model,
        "prompt": request.prompt,
        "schema": request.schema.json(),
        "image_blake3": blake3::hash(request.image.base64().as_bytes()).to_hex().to_string(),
        "image_size": [request.image.width(), request.image.height()],
        "seed": request.options.seed,
        "num_predict": request.options.num_predict,
    });
    if !request.sampling.is_empty() {
        if let serde_json::Value::Object(map) = &mut body {
            map.insert("sampling".into(), json!(request.sampling));
        }
    }
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
    /// The reply was stopped for a repetition loop: `response` holds the text up to
    /// that point (`json` is null) and replay answers with
    /// [`VisionError::Repetition`] and this finding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition: Option<RepetitionFinding>,
}

/// The stored form of a reply cut off at the output limit (`done_reason`
/// `length`) or stopped for repetition (`repetition`).
fn stopped_response(raw_text: &str, eval_count: Option<u32>, done_reason: &str) -> RawResponse {
    RawResponse {
        raw_text: raw_text.to_string(),
        json: serde_json::Value::Null,
        prompt_eval_count: None,
        eval_count,
        durations: glassrip_vision::Durations::default(),
        attempts: 1,
        repaired: false,
        done_reason: Some(done_reason.into()),
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
                // A truncation or a repetition stop is a deterministic answer to
                // this request; record it so replay takes the same (retry) path.
                let stopped = match &e {
                    VisionError::Truncated {
                        raw_text,
                        eval_count,
                        ..
                    } => Some((stopped_response(raw_text, *eval_count, "length"), None)),
                    VisionError::Repetition {
                        raw_text, finding, ..
                    } => Some((
                        stopped_response(raw_text, None, "repetition"),
                        Some(finding.clone()),
                    )),
                    _ => None,
                };
                if let Some((response, repetition)) = stopped {
                    let entry = RecordedResponse {
                        key,
                        model: id.model,
                        digest: id.digest,
                        truncated: repetition.is_none(),
                        response,
                        repetition,
                    };
                    if let Err(w) = self.store.put(&entry) {
                        tracing::warn!(error = %w, "could not record stopped response");
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
            repetition: None,
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
            Ok(Some(RecordedResponse {
                repetition: Some(finding),
                response,
                ..
            })) => Err(VisionError::Repetition {
                num_predict: request.options.num_predict,
                finding,
                raw_text: response.raw_text,
            }),
            Ok(Some(entry)) if entry.truncated => {
                // The live client checks a guarded reply for loops before calling
                // it truncated; replay does the same.
                glassrip_vision::ollama::repetition(&request, &entry.response.raw_text)?;
                Err(VisionError::Truncated {
                    num_predict: request.options.num_predict,
                    eval_count: entry.response.eval_count,
                    raw_text: entry.response.raw_text,
                })
            }
            Ok(Some(entry)) => {
                glassrip_vision::ollama::repetition(&request, &entry.response.raw_text)?;
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

    #[test]
    fn sampling_overrides_change_the_key_and_the_guard_does_not() {
        let plain = request(1);
        let base = request_key("m", &plain);
        let mut guarded = plain.clone();
        guarded.repetition_guard = Some(Default::default());
        assert_eq!(
            base,
            request_key("m", &guarded),
            "the server sees the same request"
        );
        let mut penalized = guarded.clone();
        penalized.sampling.repeat_penalty = Some(1.3);
        assert_ne!(base, request_key("m", &penalized));
        let mut windowed = plain;
        windowed.sampling.repeat_last_n = Some(512);
        assert_ne!(base, request_key("m", &windowed));
        assert_ne!(request_key("m", &penalized), request_key("m", &windowed));
    }

    /// Answers every request with a repetition stop.
    struct Looping;

    fn finding() -> RepetitionFinding {
        RepetitionFinding {
            kind: glassrip_vision::repetition::RepetitionKind::Templated,
            repeats: 6,
            pattern: "{\"src\":\"n#\"}".into(),
            at_bytes: 900,
        }
    }

    #[async_trait]
    impl VisionBackend for Looping {
        fn id(&self) -> BackendId {
            Fixed.id()
        }
        async fn preflight(&self) -> glassrip_vision::Result<Placement> {
            Err(VisionError::Config("unused".into()))
        }
        async fn infer(
            &self,
            r: VisionRequest,
            _c: CancellationToken,
        ) -> glassrip_vision::Result<RawResponse> {
            Err(VisionError::Repetition {
                num_predict: r.options.num_predict,
                finding: finding(),
                raw_text: "{\"edges\": [".into(),
            })
        }
    }

    #[tokio::test]
    async fn repetition_stops_are_recorded_and_replayed() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        let mut req = request(7);
        req.repetition_guard = Some(Default::default());
        let rec = RecordingBackend::new(Arc::new(Looping), store.clone());
        let live = rec.infer(req.clone(), CancellationToken::new()).await;
        assert!(matches!(live, Err(VisionError::Repetition { .. })));
        let entry = store.get(&request_key("m", &req)).unwrap().unwrap();
        assert_eq!(entry.repetition, Some(finding()));
        assert!(!entry.truncated);
        let replay = ReplayBackend::new("m", store);
        match replay.infer(req, CancellationToken::new()).await {
            Err(VisionError::Repetition {
                finding: f,
                raw_text,
                ..
            }) => {
                assert_eq!(f, finding());
                assert_eq!(raw_text, "{\"edges\": [");
            }
            other => panic!("expected a replayed repetition stop, got {other:?}"),
        }
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
