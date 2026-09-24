//! Model backends for `glassrip meeting`: OCR, vision model, text model, ASR,
//! and diarization. Each is either available or carries the reason it is not
//! (a missing build feature, an unpulled model, a missing model file); preflight
//! turns the reasons of every backend the selected stages need into one error.
//!
//! [`Backends::connect`] builds the real backends for this build; tests build
//! [`Backends`] from scripted implementations.
//!
//! Offline reruns: when the model server cannot be reached but the output
//! directory's `run.lock.json` records the model's digest, the vision and text
//! backends are created offline with that digest ([`VisionBackends::offline`],
//! [`Backends::text_offline`]). The digest keys the stage caches, so a model
//! stage whose output is cached restores without contacting the server; one that
//! is not cached fails with the recorded reason. OCR, ASR, and diarization are
//! local and must still be present when their stages are selected.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use glassrip_audio::asr::{AsrConfig, VadSettings};
use glassrip_audio::stages::{AsrEngine, DiarizeEngine, SingleSpeakerEngine, WhisperEngine};
use glassrip_core::config::Config;
use glassrip_notes::notes::llm::{LlmError, OllamaText, OllamaTextConfig, TextBackend};
use glassrip_ocr::TextRecognizer;
use glassrip_vision::{
    BackendId, OllamaBackend, OllamaConfig, Placement, RawResponse, VisionBackend, VisionError,
    VisionRequest,
};
use glassrip_vision_stages::placement::PlacementProbe;
use glassrip_vision_stages::raw_store::{RawStore, RecordingBackend};
use tokio_util::sync::CancellationToken;

use super::preflight::Needs;

/// A backend, or why it is unavailable.
pub type Avail<T> = Result<T, String>;

/// The vision model: requests, placement checks, and identity.
#[derive(Clone)]
pub struct VisionBackends {
    /// Request backend (recording raw responses in a real run).
    pub backend: Arc<dyn VisionBackend>,
    /// Placement and digest checks.
    pub probe: Arc<dyn PlacementProbe>,
    /// Model tag.
    pub model: String,
    /// Model digest, when known before the run.
    pub digest: Option<String>,
    /// Server version, when known.
    pub server_version: Option<String>,
    /// Client concurrency (server slots).
    pub concurrency: usize,
    /// Bytes of the model files, when the server reported them.
    pub size_bytes: Option<u64>,
    /// Parameters in billions, when the server reported them.
    pub parameter_size_b: Option<f64>,
    /// Why the server is not used (offline rerun), when it is not.
    pub offline: Option<String>,
}

/// Loaded bytes per parameter used to estimate a quantized model's size when
/// the server does not report the file size (Q4-class weights plus vision
/// projector overhead).
pub const EST_GB_PER_B_PARAMS: f64 = 0.7;

impl VisionBackends {
    /// Estimated model size in GB: the file size, else the reported parameter
    /// count, else the parameter count in the tag (`qwen2.5vl:32b`). `None` when
    /// nothing is known.
    pub fn estimated_gb(&self) -> Option<f64> {
        #[allow(clippy::cast_precision_loss)]
        let file = self.size_bytes.map(|b| b as f64 / 1e9);
        file.or_else(|| self.parameter_size_b.map(|b| b * EST_GB_PER_B_PARAMS))
            .or_else(|| size_b(&self.model).map(|b| b * EST_GB_PER_B_PARAMS))
    }

    /// Client concurrency: one slot for models above the sequential threshold
    /// (27B/32B class, spec 5.3), else the server's slot count.
    pub fn slots(&self, sequential_threshold_gb: f64) -> usize {
        if self
            .estimated_gb()
            .is_some_and(|gb| gb > sequential_threshold_gb)
        {
            1
        } else {
            self.concurrency.max(1)
        }
    }

    /// Backends for a model whose server is unreachable, identified by the
    /// digest recorded in a previous run's manifest. Every request fails with
    /// `reason`; cached stages still restore.
    pub fn offline(model: &str, digest: &str, reason: &str) -> Self {
        let off = Arc::new(Offline {
            model: model.to_string(),
            digest: digest.to_string(),
            reason: reason.to_string(),
        });
        Self {
            backend: off.clone(),
            probe: off,
            model: model.to_string(),
            digest: Some(digest.to_string()),
            server_version: None,
            concurrency: 1,
            size_bytes: None,
            parameter_size_b: None,
            offline: Some(reason.to_string()),
        }
    }
}

/// A vision backend and probe that only carry an identity.
struct Offline {
    model: String,
    digest: String,
    reason: String,
}

impl Offline {
    fn err(&self) -> VisionError {
        VisionError::Transport(format!("model server not used: {}", self.reason))
    }
}

#[async_trait]
impl VisionBackend for Offline {
    fn id(&self) -> BackendId {
        BackendId {
            backend: "offline".into(),
            model: self.model.clone(),
            digest: Some(self.digest.clone()),
            server_version: None,
        }
    }
    async fn preflight(&self) -> glassrip_vision::Result<Placement> {
        Err(self.err())
    }
    async fn infer(
        &self,
        _request: VisionRequest,
        _cancel: CancellationToken,
    ) -> glassrip_vision::Result<RawResponse> {
        Err(self.err())
    }
}

#[async_trait]
impl PlacementProbe for Offline {
    async fn preflight(&self) -> Result<Placement, VisionError> {
        Err(self.err())
    }
    async fn placement(&self) -> Result<Option<Placement>, VisionError> {
        Err(self.err())
    }
    async fn digest(&self) -> Result<String, VisionError> {
        Ok(self.digest.clone())
    }
    async fn server_up(&self) -> bool {
        false
    }
}

/// Model digests recorded in `<out_dir>/run.lock.json` (empty when absent or unreadable).
pub fn recorded_digests(out_dir: &Path) -> BTreeMap<String, String> {
    let path = out_dir.join(glassrip_core::manifest::MANIFEST_FILE);
    fs_err::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice::<glassrip_core::manifest::RunManifest>(&b).ok())
        .map(|m| m.model_digests)
        .unwrap_or_default()
}

/// True for errors that mean the server could not be reached (as opposed to a
/// missing model or a bad reply).
fn unreachable(e: &VisionError) -> bool {
    matches!(
        e,
        VisionError::RetriesExhausted { .. }
            | VisionError::Timeout { .. }
            | VisionError::Transport(_)
    )
}

/// Every model backend of a run.
pub struct Backends {
    /// PP-OCRv5 for `ocr_harvest` (dropped after GPU phase A).
    pub ocr: Avail<Arc<dyn TextRecognizer>>,
    /// Vision model for `classify`, `board_read`, and the edge-direction fallback.
    pub vision: Avail<VisionBackends>,
    /// Text model for `notes`.
    pub text: Avail<Arc<dyn TextBackend>>,
    /// Speech recognizer for `asr`.
    pub asr: Avail<Arc<dyn AsrEngine>>,
    /// Diarizer for `diarize` (and the gap-fill embedder).
    pub diarize: Avail<Arc<dyn DiarizeEngine>>,
    /// Why the text model's server is not used (offline rerun), when it is not.
    pub text_offline: Option<String>,
    /// Model name to digest, recorded in `run.lock.json`.
    pub model_digests: BTreeMap<String, String>,
}

impl Backends {
    /// Nothing available (each field says `reason`).
    pub fn none(reason: &str) -> Self {
        Self {
            ocr: Err(reason.to_string()),
            vision: Err(reason.to_string()),
            text: Err(reason.to_string()),
            asr: Err(reason.to_string()),
            diarize: Err(reason.to_string()),
            text_offline: None,
            model_digests: BTreeMap::new(),
        }
    }

    /// Builds the real backends the selected stages need; unneeded ones are left
    /// unavailable without contacting anything. `out_dir` is the run directory
    /// (read only, for digests recorded by an earlier run; see the module docs).
    pub async fn connect(config: &Config, needs: &Needs, out_dir: &Path) -> Self {
        let raw_dir = out_dir.join(super::RAW_RESPONSES_DIR);
        let recorded = recorded_digests(out_dir);
        let mut b = Self::none("not needed by the selected stages");
        if needs.ocr {
            b.ocr = ocr_engine();
        }
        // edge_direction uses the vision model only as an optional fallback; it
        // is still connected so the stage keys and behaves as in a full run.
        if needs.vision || needs.vision_optional {
            b.vision = vision(config, &raw_dir, &recorded).await;
            if let Ok(v) = &b.vision {
                if let Some(d) = &v.digest {
                    b.model_digests.insert(v.model.clone(), d.clone());
                }
            }
        }
        if needs.text {
            match text(config, &recorded).await {
                Ok((backend, digest, offline)) => {
                    if let Some(d) = digest {
                        b.model_digests.insert(config.models.text.clone(), d);
                    }
                    b.text_offline = offline;
                    b.text = Ok(backend);
                }
                Err(e) => b.text = Err(e),
            }
        }
        if needs.asr {
            b.asr = asr(config);
        }
        if needs.diarize || needs.asr {
            b.diarize = diarizer(config);
        }
        b
    }
}

#[cfg(feature = "onnx")]
fn ocr_engine() -> Avail<Arc<dyn TextRecognizer>> {
    let dir = glassrip_ocr::models::default_dir();
    glassrip_ocr::engine::PpOcrEngine::new(&dir, glassrip_ocr::OcrConfig::default())
        .map(|e| Arc::new(e) as Arc<dyn TextRecognizer>)
        .map_err(|e| format!("PP-OCRv5 could not start: {e}"))
}

#[cfg(not(feature = "onnx"))]
fn ocr_engine() -> Avail<Arc<dyn TextRecognizer>> {
    Err(
        "ocr_harvest needs PP-OCRv5 on ONNX Runtime: build with `--features onnx` (CPU) \
         or `--features cuda`"
            .into(),
    )
}

/// Parameter count in billions from a tag such as `qwen2.5vl:32b`.
fn size_b(model: &str) -> Option<f64> {
    let tag = model.rsplit(':').next()?;
    let digits: String = tag
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    tag[digits.len()..]
        .starts_with('b')
        .then(|| digits.parse().ok())
        .flatten()
}

/// Request timeout: the 30B-class value for models of 14B parameters and up.
pub fn request_timeout(config: &Config, model: &str) -> Duration {
    let large = size_b(model).is_some_and(|b| b >= 14.0);
    Duration::from_secs(if large {
        config.ollama.large_model_request_timeout_s
    } else {
        config.ollama.request_timeout_s
    })
}

async fn vision(
    config: &Config,
    raw_dir: &Path,
    recorded: &BTreeMap<String, String>,
) -> Avail<VisionBackends> {
    let model = config.models.vision.clone();
    let mut oc = OllamaConfig::new(&config.ollama.host, &model, config.ollama.num_ctx);
    oc.keep_alive = config.ollama.keep_alive.clone();
    oc.slots = config.ollama.num_parallel as usize;
    oc.allow_spill = config.gpu.allow_spill;
    oc.request_timeout = request_timeout(config, &model);
    let ollama = Arc::new(OllamaBackend::new(oc).map_err(|e| e.to_string())?);
    let host = config.ollama.host.trim_end_matches('/').to_string();
    let digest = match ollama.resolve_digest().await {
        Ok(d) => d,
        Err(e) if unreachable(&e) => {
            let Some(d) = recorded.get(&model) else {
                return Err(format!("vision model {model} on {host}: {e}"));
            };
            let reason = format!("{host} unreachable ({e})");
            tracing::warn!(%model, digest = %d, "{reason}; model stages must restore from cache");
            return Ok(VisionBackends::offline(&model, d, &reason));
        }
        Err(e) => return Err(format!("vision model {model} on {host}: {e}")),
    };
    let server_version = ollama.server_version().await.ok();
    let size = match ollama.model_size().await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(%model, error = %e, "model size unknown");
            glassrip_vision::ModelSize::default()
        }
    };
    Ok(VisionBackends {
        // Raw replies are kept next to the run for offline replay.
        backend: Arc::new(RecordingBackend::new(
            Arc::clone(&ollama) as Arc<dyn VisionBackend>,
            RawStore::new(raw_dir),
        )),
        probe: ollama,
        model,
        digest: Some(digest),
        server_version,
        concurrency: config.ollama.num_parallel as usize,
        size_bytes: size.file_bytes,
        parameter_size_b: size.parameter_size_b,
        offline: None,
    })
}

/// The text backend, its digest, and the offline reason when the server is
/// unreachable but a digest was recorded.
type TextSetup = (Arc<dyn TextBackend>, Option<String>, Option<String>);

async fn text(config: &Config, recorded: &BTreeMap<String, String>) -> Avail<TextSetup> {
    let model = &config.models.text;
    let backend = OllamaText::new(OllamaTextConfig {
        base_url: config.ollama.host.clone(),
        ..OllamaTextConfig::default()
    })
    .map_err(|e| e.to_string())?;
    let host = config.ollama.host.trim_end_matches('/');
    match backend.digest(model).await {
        Ok(d) => Ok((Arc::new(backend), d, None)),
        Err(e @ LlmError::Transport(_)) if recorded.contains_key(model) => {
            let reason = format!("{host} unreachable ({e})");
            tracing::warn!(%model, "{reason}; notes must restore from cache");
            let digest = recorded.get(model).cloned();
            Ok((Arc::new(backend), digest, Some(reason)))
        }
        Err(LlmError::Http { status: 404, .. }) => Err(format!(
            "text model {model} is not available on {}; run `ollama pull {model}`",
            host
        )),
        Err(e) => Err(format!("cannot check text model {model} on {host}: {e}")),
    }
}

/// `$GLASSRIP_MODELS_DIR`, else `~/.glassrip/models`.
pub fn models_dir() -> PathBuf {
    glassrip_audio::models::models_dir().unwrap_or_else(|| PathBuf::from(".glassrip/models"))
}

/// Resolves `--asr-model`: an existing path, else `ggml-<name>.bin` in the models directory.
pub fn asr_model_path(name: &str) -> PathBuf {
    let p = PathBuf::from(name);
    if p.is_file() {
        return p;
    }
    let file = if name.ends_with(".bin") {
        name.to_string()
    } else {
        format!("ggml-{name}.bin")
    };
    models_dir().join(file)
}

fn asr(config: &Config) -> Avail<Arc<dyn AsrEngine>> {
    let path = asr_model_path(&config.models.asr);
    if !path.is_file() {
        return Err(format!(
            "whisper model `{}` not found at {}; download it (URLs in crates/glassrip-audio/models.toml) \
             or pass --asr-model PATH",
            config.models.asr,
            path.display()
        ));
    }
    let mut cfg = AsrConfig::new(path);
    let vad = models_dir().join(&config.audio.vad_model);
    if vad.is_file() {
        cfg.vad = Some(VadSettings::new(vad));
    } else {
        tracing::warn!(vad = %vad.display(), "VAD model missing; decoding fixed-length chunks");
    }
    cfg.language = config.audio.language.clone();
    cfg.beam_size = config.audio.beam_size;
    cfg.max_prompt_tokens = config.audio.vocabulary_max_tokens as usize;
    Ok(Arc::new(WhisperEngine::new(cfg)))
}

fn diarizer(config: &Config) -> Avail<Arc<dyn DiarizeEngine>> {
    if config.audio.speakers == Some(1) {
        return Ok(Arc::new(SingleSpeakerEngine));
    }
    speakrs(config)
}

#[cfg(feature = "diarize")]
fn speakrs(config: &Config) -> Avail<Arc<dyn DiarizeEngine>> {
    use glassrip_audio::diarize::{DiarizeConfig, DiarizeMode};
    let dir = models_dir().join("speakrs");
    if !dir.is_dir() {
        return Err(format!(
            "speakrs models not found in {}; download them (crates/glassrip-audio/models.toml)",
            dir.display()
        ));
    }
    Ok(Arc::new(glassrip_audio::stages::SpeakrsEngine::new(
        DiarizeConfig {
            models_dir: dir,
            mode: if cfg!(feature = "cuda") {
                DiarizeMode::Cuda
            } else {
                DiarizeMode::Cpu
            },
            num_speakers: config.audio.speakers.map(|n| n as usize),
            verify_models: true,
        },
    )))
}

#[cfg(not(feature = "diarize"))]
fn speakrs(_config: &Config) -> Avail<Arc<dyn DiarizeEngine>> {
    Err(
        "diarize needs speakrs: build with `--features diarize` (CPU) or `--features cuda`, \
         or pass `--speakers 1`"
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_sizes_pick_timeouts() {
        assert_eq!(size_b("qwen2.5vl:7b"), Some(7.0));
        assert_eq!(size_b("qwen2.5vl:32b"), Some(32.0));
        assert_eq!(size_b("gemma4:12b"), Some(12.0));
        assert_eq!(size_b("llava:latest"), None);
        let c = Config::default();
        assert_eq!(
            request_timeout(&c, "qwen2.5vl:7b"),
            Duration::from_secs(120)
        );
        assert_eq!(request_timeout(&c, "qwen3.6:27b"), Duration::from_secs(400));
    }

    #[test]
    fn asr_model_names_resolve_in_the_models_dir() {
        let p = asr_model_path("large-v3-turbo");
        assert!(p.ends_with("ggml-large-v3-turbo.bin"), "{}", p.display());
        assert!(asr_model_path("custom.bin").ends_with("custom.bin"));
    }

    fn vb(model: &str) -> VisionBackends {
        VisionBackends::offline(model, "sha256:d", "test")
    }

    #[test]
    fn model_size_estimates_and_slots() {
        let mut v = vb("custom-vl");
        assert_eq!(v.estimated_gb(), None);
        v.concurrency = 4;
        assert_eq!(v.slots(14.0), 4, "unknown size keeps the server slots");
        let big = VisionBackends {
            size_bytes: Some(21_000_000_000),
            ..v.clone()
        };
        assert_eq!(big.estimated_gb(), Some(21.0));
        assert_eq!(big.slots(14.0), 1);
        let by_params = VisionBackends {
            parameter_size_b: Some(32.8),
            ..v.clone()
        };
        assert!(by_params.estimated_gb().is_some_and(|g| g > 14.0));
        let by_tag = VisionBackends {
            model: "qwen2.5vl:32b".into(),
            ..v.clone()
        };
        assert_eq!(by_tag.slots(14.0), 1);
        let small = VisionBackends {
            model: "qwen2.5vl:7b".into(),
            ..v
        };
        assert!(small.estimated_gb().is_some_and(|g| g < 14.0));
    }

    #[tokio::test]
    async fn offline_backends_carry_the_recorded_digest_and_refuse_requests() {
        let v = vb("m:1b");
        assert_eq!(v.backend.id().digest.as_deref(), Some("sha256:d"));
        assert_eq!(v.probe.digest().await.unwrap(), "sha256:d");
        assert!(v.probe.placement().await.is_err());
        assert!(v.offline.is_some());
    }

    #[test]
    fn digests_are_read_from_an_earlier_manifest() {
        let dir = tempfile::tempdir().unwrap();
        assert!(recorded_digests(dir.path()).is_empty());
        let mut run = glassrip_core::manifest::RunDir::open(
            dir.path(),
            "r",
            glassrip_core::envelope::Producer::glassrip("0.1.0", None),
        )
        .unwrap();
        run.update(|m| {
            m.model_digests.insert("m:1b".into(), "sha256:abc".into());
        })
        .unwrap();
        drop(run);
        assert_eq!(recorded_digests(dir.path())["m:1b"], "sha256:abc");
        assert!(unreachable(&VisionError::Transport("x".into())));
        assert!(!unreachable(&VisionError::ModelNotFound {
            model: "m".into()
        }));
    }

    #[test]
    fn single_speaker_needs_no_feature() {
        let mut c = Config::default();
        c.audio.speakers = Some(1);
        assert!(diarizer(&c).is_ok());
        c.audio.speakers = Some(3);
        assert_eq!(
            diarizer(&c).is_ok(),
            cfg!(feature = "diarize") && models_dir().join("speakrs").is_dir()
        );
    }
}
