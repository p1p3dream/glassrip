//! Raw model responses stored by request key, for offline replay.
//!
//! The key is derived from everything the server sees (model name, prompt,
//! schema, image bytes, seed, `num_predict`, sampling overrides) and, for a
//! guarded request, the repetition guard's policy: a reply the guard stopped is an
//! answer to the request *and* the guard, so a changed guard never follows a retry
//! path an older guard chose. [`RecordingBackend`] writes every successful
//! response and every truncation or repetition stop; [`ReplayBackend`] serves them
//! without a server, and refuses records made under another model digest or
//! another guard policy instead of mixing them into one run.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
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
/// so requests without them keep their original keys. The repetition guard's
/// policy ([`RepetitionParams::policy_id`]) is part of the key when the request is
/// guarded.
///
/// [`RepetitionParams::policy_id`]: glassrip_vision::repetition::RepetitionParams::policy_id
pub fn request_key(model: &str, request: &VisionRequest) -> String {
    key_of(model, request, true)
}

/// The key a request had before the guard policy joined it: records stored under
/// it are recognized and refused explicitly.
fn legacy_request_key(model: &str, request: &VisionRequest) -> String {
    key_of(model, request, false)
}

fn key_of(model: &str, request: &VisionRequest, with_guard: bool) -> String {
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
    if let serde_json::Value::Object(map) = &mut body {
        if !request.sampling.is_empty() {
            map.insert("sampling".into(), json!(request.sampling));
        }
        if let Some(guard) = request.repetition_guard.as_ref().filter(|_| with_guard) {
            map.insert("repetition_guard".into(), json!(guard.policy_id()));
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
    /// Policy of the repetition guard the request carried
    /// ([`RepetitionParams::policy_id`]), if any.
    ///
    /// [`RepetitionParams::policy_id`]: glassrip_vision::repetition::RepetitionParams::policy_id
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_policy: Option<String>,
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
        let guard_policy = request.repetition_guard.as_ref().map(|g| g.policy_id());
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
                        guard_policy,
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
            guard_policy,
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
///
/// Every record served in one replay must come from one model digest (no digest
/// counts as one value): the one given with [`ReplayBackend::with_digest`], else
/// the store's only digest, else the first one served. A record from another
/// digest, from another repetition guard policy, or a stored repetition stop the
/// current guard would not make, is refused as incompatible (re-record it) rather
/// than replayed. [`VisionBackend::id`] reports the store's digest (or every digest
/// of a mixed store), so stage cache keys built from it tell stores apart.
pub struct ReplayBackend {
    model: String,
    store: RawStore,
    /// Reported by `id`.
    id_digest: Option<String>,
    /// The digest every served record must carry, once known.
    pinned: Mutex<Option<Option<String>>>,
}

/// Distinct digests of the records in `store` (unreadable entries are skipped:
/// serving them reports the error).
fn store_digests(store: &RawStore) -> std::collections::BTreeSet<Option<String>> {
    let mut out = std::collections::BTreeSet::new();
    let Ok(prefixes) = fs_err::read_dir(store.dir()) else {
        return out;
    };
    for prefix in prefixes.flatten() {
        let Ok(files) = fs_err::read_dir(prefix.path()) else {
            continue;
        };
        for file in files.flatten() {
            let path = file.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Some(entry) = fs_err::read(&path)
                .ok()
                .and_then(|b| serde_json::from_slice::<RecordedResponse>(&b).ok())
            {
                out.insert(entry.digest);
            }
        }
    }
    out
}

impl ReplayBackend {
    /// Replay `store`, learning its model digest from its records.
    pub fn new(model: impl Into<String>, store: RawStore) -> Self {
        let digests = store_digests(&store);
        let (id_digest, pinned) = match digests.len() {
            0 => (None, None),
            1 => {
                let only = digests.into_iter().next().flatten();
                (only.clone(), Some(only))
            }
            _ => (
                Some(format!(
                    "mixed:{}",
                    digests
                        .iter()
                        .map(|d| d.as_deref().unwrap_or("none"))
                        .collect::<Vec<_>>()
                        .join(",")
                )),
                None,
            ),
        };
        Self {
            model: model.into(),
            store,
            id_digest,
            pinned: Mutex::new(pinned),
        }
    }

    /// Refuse records made under any other model digest.
    #[must_use]
    pub fn with_digest(mut self, digest: impl Into<String>) -> Self {
        let digest = digest.into();
        *self.pinned.lock().unwrap_or_else(PoisonError::into_inner) = Some(Some(digest.clone()));
        self.id_digest = Some(digest);
        self
    }

    fn incompatible(key: &str, why: impl std::fmt::Display) -> VisionError {
        VisionError::Config(format!(
            "recorded response {key} is incompatible with this replay: {why}; re-record it"
        ))
    }

    /// Refuse a record from another digest or guard policy.
    fn check(
        &self,
        key: &str,
        entry: &RecordedResponse,
        request: &VisionRequest,
    ) -> glassrip_vision::Result<()> {
        let policy = request.repetition_guard.as_ref().map(|g| g.policy_id());
        if entry.guard_policy != policy {
            return Err(Self::incompatible(
                key,
                format!(
                    "recorded under repetition guard {:?}, the request carries {:?}",
                    entry.guard_policy, policy
                ),
            ));
        }
        let mut pinned = self.pinned.lock().unwrap_or_else(PoisonError::into_inner);
        match pinned.as_ref() {
            Some(want) if *want != entry.digest => Err(Self::incompatible(
                key,
                format!(
                    "recorded with model digest {}, this replay uses {}",
                    entry.digest.as_deref().unwrap_or("(none)"),
                    want.as_deref().unwrap_or("(none)")
                ),
            )),
            Some(_) => Ok(()),
            None => {
                *pinned = Some(entry.digest.clone());
                Ok(())
            }
        }
    }
}

#[async_trait]
impl VisionBackend for ReplayBackend {
    fn id(&self) -> BackendId {
        BackendId {
            backend: "replay".into(),
            model: self.model.clone(),
            digest: self.id_digest.clone(),
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
        let entry = match self.store.get(&key) {
            Ok(Some(entry)) => entry,
            Ok(None) => {
                let legacy = legacy_request_key(&self.model, &request);
                if legacy != key && matches!(self.store.get(&legacy), Ok(Some(_))) {
                    return Err(Self::incompatible(
                        &legacy,
                        "it was recorded before the repetition guard policy joined the request key",
                    ));
                }
                return Err(VisionError::Config(format!(
                    "no recorded response for request key {key}"
                )));
            }
            Err(e) => return Err(VisionError::Protocol(e.to_string())),
        };
        self.check(&key, &entry, &request)?;
        if let Some(finding) = entry.repetition {
            // The stop must still be one the current guard makes on this text.
            let again = request
                .repetition_guard
                .as_ref()
                .and_then(|g| glassrip_vision::repetition::detect(&entry.response.raw_text, g));
            if again.is_none() {
                return Err(Self::incompatible(
                    &key,
                    format!("the current repetition guard does not stop its text ({finding})"),
                ));
            }
            return Err(VisionError::Repetition {
                num_predict: request.options.num_predict,
                finding,
                raw_text: entry.response.raw_text,
            });
        }
        if entry.truncated {
            // The live client checks a reply cut off at the limit for loops before
            // calling it truncated; replay does the same.
            glassrip_vision::ollama::repetition(&request, &entry.response.raw_text)?;
            return Err(VisionError::Truncated {
                num_predict: request.options.num_predict,
                eval_count: entry.response.eval_count,
                raw_text: entry.response.raw_text,
            });
        }
        // Validate against the request schema, as a live reply would be; a reply
        // that validates is accepted whatever the guard says, as live.
        if let Err(errors) = request.schema.validate(&entry.response.json) {
            glassrip_vision::ollama::repetition(&request, &entry.response.raw_text)?;
            return Err(VisionError::SchemaInvalid {
                attempts: 1,
                errors,
                raw_text: entry.response.raw_text.clone(),
            });
        }
        let mut r = entry.response;
        r.durations.wall = Duration::ZERO;
        Ok(r)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use glassrip_vision::repetition::RepetitionParams;
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
    fn sampling_overrides_and_the_guard_policy_change_the_key() {
        let plain = request(1);
        let base = request_key("m", &plain);
        let mut guarded = plain.clone();
        guarded.repetition_guard = Some(Default::default());
        let with_guard = request_key("m", &guarded);
        assert_ne!(base, with_guard, "a guarded reply answers the guard too");
        assert_eq!(legacy_request_key("m", &guarded), base);
        let mut stricter = guarded.clone();
        stricter.repetition_guard = Some(RepetitionParams {
            min_templated_run: 8,
            ..RepetitionParams::default()
        });
        assert_ne!(with_guard, request_key("m", &stricter));
        let mut penalized = guarded.clone();
        penalized.sampling.repeat_penalty = Some(1.3);
        assert_ne!(with_guard, request_key("m", &penalized));
        let mut windowed = plain;
        windowed.sampling.repeat_last_n = Some(512);
        assert_ne!(base, request_key("m", &windowed));
        assert_ne!(request_key("m", &penalized), request_key("m", &windowed));
    }

    /// Answers every request with a repetition stop.
    struct Looping;

    /// One edge restated under new ids until stopped.
    fn looping_text() -> String {
        let mut s = String::from("{\"nodes\": [{\"local_id\": \"n1\"}], \"edges\": [");
        for k in 15..25 {
            s.push_str(&format!(
                "{{\"src\": \"n{k}\", \"dst\": \"n{}\", \"label\": \"Relay\"}}, ",
                k + 1
            ));
        }
        s
    }

    fn finding() -> RepetitionFinding {
        glassrip_vision::repetition::detect(&looping_text(), &RepetitionParams::default()).unwrap()
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
                raw_text: looping_text(),
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
                assert_eq!(raw_text, looping_text());
            }
            other => panic!("expected a replayed repetition stop, got {other:?}"),
        }
    }

    /// `Fixed` with no model digest.
    struct NoDigest;

    #[async_trait]
    impl VisionBackend for NoDigest {
        fn id(&self) -> BackendId {
            BackendId {
                digest: None,
                ..Fixed.id()
            }
        }
        async fn preflight(&self) -> glassrip_vision::Result<Placement> {
            Err(VisionError::Config("unused".into()))
        }
        async fn infer(
            &self,
            r: VisionRequest,
            c: CancellationToken,
        ) -> glassrip_vision::Result<RawResponse> {
            Fixed.infer(r, c).await
        }
    }

    /// Codex review 3: a record without a digest is a digest of its own, not a
    /// wildcard.
    #[tokio::test]
    async fn records_without_a_digest_do_not_mix_with_digested_ones() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        RecordingBackend::new(Arc::new(NoDigest), store.clone())
            .infer(request(1), CancellationToken::new())
            .await
            .unwrap();
        RecordingBackend::new(Arc::new(Fixed), store.clone())
            .infer(request(2), CancellationToken::new())
            .await
            .unwrap();
        let replay = ReplayBackend::new("m", store.clone());
        replay
            .infer(request(1), CancellationToken::new())
            .await
            .unwrap();
        expect_incompatible(
            replay.infer(request(2), CancellationToken::new()).await,
            "this replay uses (none)",
        );
        expect_incompatible(
            ReplayBackend::new("m", store)
                .with_digest("d")
                .infer(request(1), CancellationToken::new())
                .await,
            "model digest (none)",
        );
    }

    /// Codex review 4: the replay backend reports the store's digest, so stage
    /// cache keys built from `id()` tell two stores apart.
    #[tokio::test]
    async fn the_replay_backend_reports_the_store_digest() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        assert_eq!(ReplayBackend::new("m", store.clone()).id().digest, None);
        RecordingBackend::new(Arc::new(Fixed), store.clone())
            .infer(request(1), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(
            ReplayBackend::new("m", store).id().digest.as_deref(),
            Some("d")
        );
    }

    fn guarded(seed: u64) -> VisionRequest {
        let mut r = request(seed);
        r.repetition_guard = Some(RepetitionParams::default());
        r
    }

    fn expect_incompatible(r: glassrip_vision::Result<RawResponse>, what: &str) {
        match r {
            Err(VisionError::Config(m)) if m.contains("incompatible") && m.contains(what) => {}
            other => panic!("expected an incompatible record ({what}), got {other:?}"),
        }
    }

    /// Codex 10 / GLM M2: a stop recorded under an older guard is not replayed
    /// as if the current guard had made it.
    #[tokio::test]
    async fn a_stop_the_current_guard_would_not_make_is_incompatible() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        let req = guarded(7);
        // A regular card row an older guard stopped as a "loop".
        let row: String = (0..6)
            .map(|c| {
                format!(
                    "{{\"local_id\": \"n{}\", \"text\": \"Card\", \"bbox_2d\": [{}, 900, {}, 930]}}, ",
                    30 + c,
                    100 + c * 120,
                    180 + c * 120
                )
            })
            .collect();
        store
            .put(&RecordedResponse {
                key: request_key("m", &req),
                model: "m".into(),
                digest: Some("d".into()),
                response: stopped_response(&format!("{{\"nodes\": [{row}"), None, "repetition"),
                truncated: false,
                repetition: Some(finding()),
                guard_policy: Some(RepetitionParams::default().policy_id()),
            })
            .unwrap();
        let replay = ReplayBackend::new("m", store);
        expect_incompatible(
            replay.infer(req, CancellationToken::new()).await,
            "does not stop its text",
        );
    }

    #[tokio::test]
    async fn records_under_the_old_key_are_refused_explicitly() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        let req = guarded(7);
        // A pre-policy store: the stop sits under the key without the guard.
        store
            .put(&RecordedResponse {
                key: legacy_request_key("m", &req),
                model: "m".into(),
                digest: Some("d".into()),
                response: stopped_response(&looping_text(), None, "repetition"),
                truncated: false,
                repetition: Some(finding()),
                guard_policy: None,
            })
            .unwrap();
        let replay = ReplayBackend::new("m", store);
        expect_incompatible(
            replay.infer(req, CancellationToken::new()).await,
            "before the repetition guard policy",
        );
    }

    #[tokio::test]
    async fn one_replay_never_mixes_model_digests() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        let rec = RecordingBackend::new(Arc::new(Fixed), store.clone());
        for seed in [1, 2] {
            rec.infer(request(seed), CancellationToken::new())
                .await
                .unwrap();
        }
        // The model was retagged between recordings of two requests.
        let key = request_key("m", &request(2));
        let mut entry = store.get(&key).unwrap().unwrap();
        entry.digest = Some("d-retagged".into());
        store.put(&entry).unwrap();

        let replay = ReplayBackend::new("m", store.clone());
        assert_eq!(replay.id().digest.as_deref(), Some("mixed:d,d-retagged"));
        replay
            .infer(request(1), CancellationToken::new())
            .await
            .unwrap();
        expect_incompatible(
            replay.infer(request(2), CancellationToken::new()).await,
            "model digest d-retagged",
        );
        let pinned = ReplayBackend::new("m", store).with_digest("d-retagged");
        assert_eq!(pinned.id().digest.as_deref(), Some("d-retagged"));
        expect_incompatible(
            pinned.infer(request(1), CancellationToken::new()).await,
            "model digest d,",
        );
        pinned
            .infer(request(2), CancellationToken::new())
            .await
            .unwrap();
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
