//! Recorded-response replay (spec 9.1).
//!
//! Every model request eval makes is identified by a replay key: a glassrip-core
//! [`CacheKeyParts`] key over the request kind, model name, generation options,
//! request-specific parameters (for example the canvas crop), the blake3 of the
//! source image file, and the hashes of the prompt and output schema. Hashing the
//! source file rather than the encoded JPEG keeps keys stable across image
//! encoder versions. The model digest is deliberately left out so CI can replay
//! without a server; the digest is recorded next to the response instead.
//!
//! Layout: `<responses>/<kind>/<key>.json`, one [`ResponseRecord`] each.
//!
//! - [`Responder::Replay`] loads the record and re-validates the raw text
//!   against the request's schema and type, exactly as a live reply would be.
//! - [`Responder::Live`] sends the request through a [`VisionClient`] (any
//!   [`glassrip_vision::VisionBackend`], normally `OllamaBackend`) and, when
//!   recording (`--rerecord`), writes the raw text. [`Responder::prune`] then
//!   removes records the run did not touch.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use glassrip_core::cache::CacheKeyParts;
use glassrip_vision::backend::Durations;
use glassrip_vision::schema::parse_output_text;
use glassrip_vision::{RawResponse, VisionClient, VisionError, VisionRequest};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::error::{read_json, write_json, EvalError, Result};

/// Bump when the key derivation changes.
pub const REPLAY_KEY_VERSION: u32 = 1;

/// Identity of one request beyond the request itself.
#[derive(Debug, Clone)]
pub struct RequestId<'a> {
    /// Request kind (`classify`, `board_read`).
    pub kind: &'a str,
    /// Case name (recorded for humans; not part of the key).
    pub case: &'a str,
    /// Model name.
    pub model: &'a str,
    /// blake3 of the source image file.
    pub source_blake3: &'a str,
    /// Request-specific parameters (crop box and similar).
    pub extra: Value,
}

/// Computes the replay key of a request.
pub fn replay_key(id: &RequestId<'_>, request: &VisionRequest) -> Result<String> {
    let parts = CacheKeyParts {
        stage: format!("eval.{}", id.kind),
        stage_version: REPLAY_KEY_VERSION,
        params: json!({
            "model": id.model,
            "seed": request.options.seed,
            "num_predict": request.options.num_predict,
            "extra": id.extra,
        }),
        input_hashes: vec![id.source_blake3.to_string()],
        prompt_hash: Some(glassrip_core::blake3_hex(request.prompt.as_bytes())),
        schema_hash: Some(glassrip_core::blake3_hex(request.schema.text().as_bytes())),
        ..Default::default()
    };
    parts
        .key()
        .map(|k| k.as_str().to_string())
        .map_err(|e| EvalError::Other(format!("replay key: {e}")))
}

/// One recorded model response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseRecord {
    /// Record format version.
    pub record_version: u32,
    /// Replay key.
    pub key: String,
    /// Request kind.
    pub kind: String,
    /// Case name.
    pub case: String,
    /// Model name.
    pub model: String,
    /// Model digest when the server reported one.
    #[serde(default)]
    pub model_digest: Option<String>,
    /// Raw model text, exactly as returned.
    pub raw_text: String,
    /// Client wall time of the recorded request, seconds.
    #[serde(default)]
    pub wall_s: Option<f64>,
}

/// Directory of recorded responses.
#[derive(Debug, Clone)]
pub struct ResponseStore {
    root: PathBuf,
}

impl ResponseStore {
    /// A store rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path of a record.
    pub fn path(&self, kind: &str, key: &str) -> PathBuf {
        self.root.join(kind).join(format!("{key}.json"))
    }

    /// Loads a record; `Ok(None)` when absent.
    pub fn load(&self, kind: &str, key: &str) -> Result<Option<ResponseRecord>> {
        let p = self.path(kind, key);
        if !p.is_file() {
            return Ok(None);
        }
        read_json(&p).map(Some)
    }

    /// Writes a record.
    pub fn save(&self, record: &ResponseRecord) -> Result<()> {
        write_json(&self.path(&record.kind, &record.key), record)
    }

    /// Every `(kind, key)` in the store.
    pub fn list(&self) -> Result<Vec<(String, String)>> {
        let mut out = Vec::new();
        if !self.root.is_dir() {
            return Ok(out);
        }
        for kind in crate::fixture::case_dirs(&self.root)? {
            let kind_name = kind
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let entries = fs_err::read_dir(&kind).map_err(|e| EvalError::io(&kind, e))?;
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("json") {
                    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                        out.push((kind_name.clone(), stem.to_string()));
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }
}

/// Converts recorded raw text into a validated response for `request`.
pub fn response_from_record(
    record: &ResponseRecord,
    request: &VisionRequest,
) -> Result<RawResponse> {
    let value = parse_output_text(&record.raw_text).map_err(|e| {
        EvalError::Vision(VisionError::SchemaInvalid {
            attempts: 1,
            errors: vec![e],
            raw_text: record.raw_text.clone(),
        })
    })?;
    request.schema.validate(&value).map_err(|errors| {
        EvalError::Vision(VisionError::SchemaInvalid {
            attempts: 1,
            errors,
            raw_text: record.raw_text.clone(),
        })
    })?;
    Ok(RawResponse {
        raw_text: record.raw_text.clone(),
        json: value,
        prompt_eval_count: None,
        eval_count: None,
        durations: Durations {
            wall: Duration::from_secs_f64(record.wall_s.unwrap_or(0.0).max(0.0)),
            ..Default::default()
        },
        attempts: 0,
        repaired: false,
        done_reason: None,
    })
}

/// Where responses come from.
pub enum Responder {
    /// Recorded responses only.
    Replay {
        /// Store to read.
        store: ResponseStore,
    },
    /// A live backend, optionally recording.
    Live {
        /// Client (concurrency-limited).
        client: VisionClient,
        /// Store to write when recording.
        record: Option<ResponseStore>,
        /// Keys written in this run (for pruning).
        written: Mutex<BTreeSet<(String, String)>>,
    },
}

/// A response with its measured or recorded wall time.
#[derive(Debug, Clone)]
pub struct Timed {
    /// The validated response.
    pub response: RawResponse,
    /// Wall time, seconds (recorded time in replay mode).
    pub wall_s: f64,
}

impl Responder {
    /// Replay from `store`.
    pub fn replay(store: ResponseStore) -> Self {
        Self::Replay { store }
    }

    /// Live against `client`, recording to `record` when given.
    pub fn live(client: VisionClient, record: Option<ResponseStore>) -> Self {
        Self::Live {
            client,
            record,
            written: Mutex::new(BTreeSet::new()),
        }
    }

    /// True for live mode.
    pub fn is_live(&self) -> bool {
        matches!(self, Self::Live { .. })
    }

    /// Answers one request.
    pub async fn respond(
        &self,
        id: &RequestId<'_>,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> Result<Timed> {
        let key = replay_key(id, &request)?;
        match self {
            Self::Replay { store } => {
                let record =
                    store
                        .load(id.kind, &key)?
                        .ok_or_else(|| EvalError::MissingResponse {
                            kind: id.kind.to_string(),
                            case: id.case.to_string(),
                            key: key.clone(),
                        })?;
                let response = response_from_record(&record, &request)?;
                Ok(Timed {
                    response,
                    wall_s: record.wall_s.unwrap_or(0.0),
                })
            }
            Self::Live {
                client,
                record,
                written,
            } => {
                let start = Instant::now();
                let response = client.infer(request, cancel).await?;
                let wall_s = start.elapsed().as_secs_f64();
                if let Some(store) = record {
                    let rec = ResponseRecord {
                        record_version: 1,
                        key: key.clone(),
                        kind: id.kind.to_string(),
                        case: id.case.to_string(),
                        model: id.model.to_string(),
                        model_digest: client.backend().id().digest,
                        raw_text: response.raw_text.clone(),
                        wall_s: Some((wall_s * 1000.0).round() / 1000.0),
                    };
                    store.save(&rec)?;
                    if let Ok(mut w) = written.lock() {
                        w.insert((id.kind.to_string(), key));
                    }
                }
                Ok(Timed { response, wall_s })
            }
        }
    }

    /// After a recording run, deletes records this run did not write. Returns the
    /// number removed. No-op unless recording.
    pub fn prune(&self) -> Result<usize> {
        let Self::Live {
            record: Some(store),
            written,
            ..
        } = self
        else {
            return Ok(0);
        };
        let keep = written
            .lock()
            .map(|w| w.clone())
            .map_err(|_| EvalError::Other("recording set poisoned".into()))?;
        let mut removed = 0;
        for (kind, key) in store.list()? {
            if !keep.contains(&(kind.clone(), key.clone())) {
                let p = store.path(&kind, &key);
                fs_err::remove_file(&p).map_err(|e| EvalError::io(&p, e))?;
                removed += 1;
            }
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use glassrip_vision::classify::ScreenClassOutput;
    use glassrip_vision::{BackendId, EncodedImage, GenerationOptions, Placement, VisionBackend};
    use image::{DynamicImage, RgbImage};
    use std::sync::Arc;

    struct Fixed(&'static str);

    #[async_trait]
    impl VisionBackend for Fixed {
        fn id(&self) -> BackendId {
            BackendId {
                backend: "fixed".into(),
                model: "m".into(),
                digest: Some("sha256:abc".into()),
                server_version: None,
            }
        }
        async fn preflight(&self) -> glassrip_vision::Result<Placement> {
            Err(VisionError::Config("unused".into()))
        }
        async fn infer(
            &self,
            request: VisionRequest,
            _cancel: CancellationToken,
        ) -> glassrip_vision::Result<RawResponse> {
            let json =
                parse_output_text(self.0).map_err(|e| VisionError::Decode { errors: vec![e] })?;
            request
                .schema
                .validate(&json)
                .map_err(|errors| VisionError::Decode { errors })?;
            Ok(RawResponse {
                raw_text: self.0.to_string(),
                json,
                prompt_eval_count: Some(1),
                eval_count: Some(1),
                durations: Durations::default(),
                attempts: 1,
                repaired: false,
                done_reason: Some("stop".into()),
            })
        }
    }

    const ANSWER: &str = r#"{"screen_type":"whiteboard","app_hint":"","canvas_bbox":{"x1":0,"y1":0,"x2":10,"y2":10},"confidence":0.9}"#;

    fn request() -> VisionRequest {
        let img = DynamicImage::ImageRgb8(RgbImage::new(16, 16));
        VisionRequest::for_output::<ScreenClassOutput>(
            "classify",
            EncodedImage::encode(&img).unwrap(),
            GenerationOptions::default(),
        )
        .unwrap()
    }

    fn id(extra: Value) -> RequestId<'static> {
        RequestId {
            kind: "classify",
            case: "c1",
            model: "m",
            source_blake3: "00ff",
            extra,
        }
    }

    #[test]
    fn key_depends_on_inputs() {
        let r = request();
        let a = replay_key(&id(json!({})), &r).unwrap();
        assert_eq!(a, replay_key(&id(json!({})), &r).unwrap());
        assert_ne!(
            a,
            replay_key(&id(json!({"crop": [1, 2, 3, 4]})), &r).unwrap()
        );
        let mut other = id(json!({}));
        other.model = "m2";
        assert_ne!(a, replay_key(&other, &r).unwrap());
        let mut r2 = request();
        r2.options.seed += 1;
        assert_ne!(a, replay_key(&id(json!({})), &r2).unwrap());
        assert_eq!(a.len(), 64);
    }

    #[tokio::test]
    async fn record_then_replay_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = ResponseStore::new(dir.path());
        // A stale record that pruning must remove.
        store
            .save(&ResponseRecord {
                record_version: 1,
                key: "stale".into(),
                kind: "classify".into(),
                case: "old".into(),
                model: "m".into(),
                model_digest: None,
                raw_text: "{}".into(),
                wall_s: None,
            })
            .unwrap();
        let client = VisionClient::new(Arc::new(Fixed(ANSWER)), 2).unwrap();
        let live = Responder::live(client, Some(store.clone()));
        let got = live
            .respond(&id(json!({})), request(), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(live.prune().unwrap(), 1);
        assert_eq!(store.list().unwrap().len(), 1);

        let replay = Responder::replay(store.clone());
        let back = replay
            .respond(&id(json!({})), request(), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(back.response.raw_text, got.response.raw_text);
        assert_eq!(back.response.json, got.response.json);
        let decoded: ScreenClassOutput = back.response.decode().unwrap();
        assert!((decoded.confidence - 0.9).abs() < 1e-12);
        let key = replay_key(&id(json!({})), &request()).unwrap();
        let rec = store.load("classify", &key).unwrap().unwrap();
        assert_eq!(rec.model_digest.as_deref(), Some("sha256:abc"));

        // A different request is missing in replay mode.
        let missing = replay
            .respond(&id(json!({"crop": 1})), request(), CancellationToken::new())
            .await;
        assert!(matches!(missing, Err(EvalError::MissingResponse { .. })));
    }

    #[test]
    fn replay_revalidates_against_schema() {
        let rec = ResponseRecord {
            record_version: 1,
            key: "k".into(),
            kind: "classify".into(),
            case: "c".into(),
            model: "m".into(),
            model_digest: None,
            raw_text: r#"{"screen_type":"spaceship"}"#.into(),
            wall_s: None,
        };
        assert!(response_from_record(&rec, &request()).is_err());
        let ok = ResponseRecord {
            raw_text: format!("```json\n{ANSWER}\n```"),
            ..rec
        };
        assert!(response_from_record(&ok, &request()).is_ok());
    }
}
