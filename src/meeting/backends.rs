//! Model backends for `glassrip meeting`: OCR, vision model, text model, ASR,
//! and diarization. Each is either available or carries the reason it is not
//! (a missing build feature, an unpulled model, a missing model file); preflight
//! turns the reasons of every backend the selected stages need into one error.
//!
//! [`Backends::connect`] builds the real backends for this build; tests build
//! [`Backends`] from scripted implementations.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use glassrip_audio::asr::{AsrConfig, VadSettings};
use glassrip_audio::stages::{AsrEngine, DiarizeEngine, SingleSpeakerEngine, WhisperEngine};
use glassrip_core::config::Config;
use glassrip_notes::notes::llm::{LlmError, OllamaText, OllamaTextConfig, TextBackend};
use glassrip_ocr::TextRecognizer;
use glassrip_vision::{OllamaBackend, OllamaConfig, VisionBackend};
use glassrip_vision_stages::placement::PlacementProbe;
use glassrip_vision_stages::raw_store::{RawStore, RecordingBackend};

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
            model_digests: BTreeMap::new(),
        }
    }

    /// Builds the real backends the selected stages need; unneeded ones are left
    /// unavailable without contacting anything.
    pub async fn connect(config: &Config, needs: &Needs, raw_dir: &Path) -> Self {
        let mut b = Self::none("not needed by the selected stages");
        if needs.ocr {
            b.ocr = ocr_engine();
        }
        // edge_direction uses the vision model only as an optional fallback; it
        // is still connected so the stage keys and behaves as in a full run.
        if needs.vision || needs.vision_optional {
            b.vision = vision(config, raw_dir).await;
            if let Ok(v) = &b.vision {
                if let Some(d) = &v.digest {
                    b.model_digests.insert(v.model.clone(), d.clone());
                }
            }
        }
        if needs.text {
            b.text = text(config).await.map(|(backend, digest)| {
                if let Some(d) = digest {
                    b.model_digests.insert(config.models.text.clone(), d);
                }
                backend
            });
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

async fn vision(config: &Config, raw_dir: &Path) -> Avail<VisionBackends> {
    let model = config.models.vision.clone();
    let mut oc = OllamaConfig::new(&config.ollama.host, &model, config.ollama.num_ctx);
    oc.keep_alive = config.ollama.keep_alive.clone();
    oc.slots = config.ollama.num_parallel as usize;
    oc.allow_spill = config.gpu.allow_spill;
    oc.request_timeout = request_timeout(config, &model);
    let ollama = Arc::new(OllamaBackend::new(oc).map_err(|e| e.to_string())?);
    let digest = ollama.resolve_digest().await.map_err(|e| {
        format!(
            "vision model {model} on {}: {e}",
            config.ollama.host.trim_end_matches('/')
        )
    })?;
    let server_version = ollama.server_version().await.ok();
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
    })
}

async fn text(config: &Config) -> Avail<(Arc<dyn TextBackend>, Option<String>)> {
    let model = &config.models.text;
    let backend = OllamaText::new(OllamaTextConfig {
        base_url: config.ollama.host.clone(),
        ..OllamaTextConfig::default()
    })
    .map_err(|e| e.to_string())?;
    match backend.digest(model).await {
        Ok(Some(d)) => Ok((Arc::new(backend), Some(d))),
        Ok(None) => Ok((Arc::new(backend), None)),
        Err(LlmError::Http { status: 404, .. }) => Err(format!(
            "text model {model} is not available on {}; run `ollama pull {model}`",
            config.ollama.host.trim_end_matches('/')
        )),
        Err(e) => Err(format!(
            "cannot check text model {model} on {}: {e}",
            config.ollama.host.trim_end_matches('/')
        )),
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
