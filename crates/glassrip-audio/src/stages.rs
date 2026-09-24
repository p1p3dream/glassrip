//! The audio branch as [`glassrip_core::runner::Stage`]s (spec 5.2, 6.12):
//!
//! | Stage | Output | Inputs |
//! |---|---|---|
//! | [`AudioExtractStage`] | `glassrip.audio` | `glassrip.media_probe`, the source video |
//! | [`AsrStage`] | `glassrip.asr` | `glassrip.audio`, `glassrip.asr_vocabulary` |
//! | [`DiarizeStage`] | `glassrip.diarization` | `glassrip.audio` |
//! | [`GapFillStage`] | `glassrip.gap_fill` | `glassrip.asr`, `glassrip.diarization` |
//! | [`AssignWordsStage`] | `glassrip.transcript` | `glassrip.asr`, `glassrip.gap_fill` |
//!
//! The recognizer and the diarizer sit behind [`AsrEngine`] and [`DiarizeEngine`],
//! so the stages run with whisper.cpp and speakrs in a real run and with scripted
//! engines in tests. Decoded samples are stored content-addressed
//! (`<store>/<blake3>.f32le`, 16 kHz mono little-endian `f32`) outside the run
//! directory, so a stage restored from cache in a new run still finds them;
//! readers verify the hash.
//!
//! A video without an audio stream yields one `glassrip.audio` item with
//! `has_audio: false`; ASR and diarization then plan no work and `assign_words`
//! writes an empty `glassrip.transcript` (spec 8.1).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, KeyExtras, Stage, StageError, StageInputs, WorkItem,
};
use schemars::JsonSchema;
use semver::Version;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::asr::{AsrConfig, AsrOutput, AsrSegment, Transcriber};
use crate::assign::AssignConfig;
use crate::diarize::{Diarization, DiarizeConfig};
use crate::error::AudioError;
use crate::extract::{extract_audio, f32le_to_samples, ExtractOptions, SAMPLE_RATE};
use crate::gapfill::{GapFillConfig, SpanEmbedder};
use crate::recluster::Turn;
use crate::types::{TranscriptSegment, TRANSCRIPT_SCHEMA};
use crate::vocab::CorrectionConfig;
use crate::words::AsrWord;

/// `glassrip.media_probe` (input of `audio_extract`).
pub const MEDIA_PROBE: &str = "glassrip.media_probe";
/// `audio_extract` output.
pub const AUDIO: &str = "glassrip.audio";
/// `ocr_vocabulary` output (input of `asr`).
pub const ASR_VOCABULARY: &str = "glassrip.asr_vocabulary";
/// `asr` output.
pub const ASR: &str = "glassrip.asr";
/// `diarize` output.
pub const DIARIZATION: &str = "glassrip.diarization";
/// `gap_fill` output.
pub const GAP_FILL: &str = "glassrip.gap_fill";
/// `assign_words` output.
pub const TRANSCRIPT: &str = TRANSCRIPT_SCHEMA;

fn v1() -> Version {
    Version::new(1, 0, 0)
}

fn input(schema: &'static str) -> InputDecl {
    InputDecl { schema, major: 1 }
}

fn audio_err(e: &AudioError) -> ErrorInfo {
    let code = match e {
        AudioError::Io { .. } => ErrorCode::Io,
        AudioError::Command { .. } => ErrorCode::ExternalCommand,
        AudioError::Model { .. } | AudioError::InvalidInput(_) | AudioError::FeatureDisabled(_) => {
            ErrorCode::InvalidInput
        }
        AudioError::Whisper(_) | AudioError::Diarization(_) => ErrorCode::ModelRequest,
        _ => ErrorCode::Internal,
    };
    ErrorInfo::new(code, e.to_string())
}

fn task_err(e: tokio::task::JoinError) -> ErrorInfo {
    ErrorInfo::new(ErrorCode::Internal, format!("audio task failed: {e}"))
}

// ------------------------------------------------------------------ extract

/// `glassrip.audio` item (one item, id `audio`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AudioItem {
    /// False when the video has no audio stream (the audio branch is skipped).
    pub has_audio: bool,
    /// Decoded samples (16 kHz mono `f32` little endian), when there is audio.
    pub path: Option<String>,
    /// blake3 of the sample file.
    pub blake3: Option<String>,
    /// Sample rate of the file.
    pub sample_rate: u32,
    /// Samples in the file.
    pub n_samples: u64,
    /// Decoded length, seconds.
    pub duration_s: f64,
    /// `start_time` of the audio stream, seconds.
    pub audio_start_s: Option<f64>,
    /// `start_time` of the first video stream, seconds.
    pub video_start_s: Option<f64>,
    /// Seconds added to audio times to reach the video timeline (spec 8).
    pub timeline_offset_s: f64,
}

/// `audio_extract` parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AudioExtractParams {
    /// Output sample rate (fixed by the models).
    pub sample_rate_hz: u32,
    /// Content-addressed sample store.
    pub store_dir: PathBuf,
}

/// Decodes the first audio stream of the source video with ffmpeg.
pub struct AudioExtractStage {
    params: AudioExtractParams,
    video: PathBuf,
    tools: ExtractOptions,
    tool_versions: std::collections::BTreeMap<String, String>,
}

impl AudioExtractStage {
    /// A stage for `video`, storing samples under `store_dir`. `tool_versions`
    /// (ffmpeg and ffprobe version strings) join the cache key.
    pub fn new(
        video: PathBuf,
        store_dir: PathBuf,
        tools: ExtractOptions,
        tool_versions: std::collections::BTreeMap<String, String>,
    ) -> Self {
        Self {
            params: AudioExtractParams {
                sample_rate_hz: SAMPLE_RATE,
                store_dir,
            },
            video,
            tools,
            tool_versions,
        }
    }
}

/// Writes `samples` to `<dir>/<blake3>.f32le` (atomically; an existing file with
/// the same hash is kept) and returns `(path, blake3)`.
pub fn store_samples(dir: &Path, samples: &[f32]) -> Result<(PathBuf, String), ErrorInfo> {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let hash = glassrip_core::blake3_hex(&bytes);
    let path = dir.join(format!("{hash}.f32le"));
    let io = |e: &dyn std::fmt::Display| {
        ErrorInfo::new(
            ErrorCode::Io,
            format!("cannot store audio samples at {}: {e}", path.display()),
        )
    };
    if !path.is_file() {
        fs_err::create_dir_all(dir).map_err(|e| io(&e))?;
        glassrip_core::atomic::write_atomic(&path, &bytes).map_err(|e| io(&e))?;
    }
    Ok((path, hash))
}

/// Reads a sample file and checks its hash.
pub fn load_samples(path: &Path, blake3: &str) -> Result<Vec<f32>, ErrorInfo> {
    let bytes = fs_err::read(path).map_err(|e| {
        ErrorInfo::new(
            ErrorCode::Io,
            format!("audio samples unreadable ({e}); rerun with --force-stage audio_extract"),
        )
    })?;
    let got = glassrip_core::blake3_hex(&bytes);
    if got != blake3 {
        return Err(ErrorInfo::new(
            ErrorCode::InvalidInput,
            format!(
                "audio samples at {} have blake3 {got}, expected {blake3}; rerun with --force-stage audio_extract",
                path.display()
            ),
        ));
    }
    if bytes.len() % 4 != 0 {
        return Err(ErrorInfo::new(
            ErrorCode::InvalidInput,
            format!("{} is not a whole number of f32 samples", path.display()),
        ));
    }
    Ok(f32le_to_samples(&bytes))
}

impl Stage for AudioExtractStage {
    type Params = AudioExtractParams;
    type Work = ();
    type Output = AudioItem;

    fn name(&self) -> &'static str {
        "audio_extract"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: AUDIO,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![input(MEDIA_PROBE)]
    }
    fn params(&self) -> &AudioExtractParams {
        &self.params
    }
    fn external_inputs(&self) -> Vec<PathBuf> {
        vec![self.video.clone()]
    }
    fn key_extras(&self) -> KeyExtras {
        KeyExtras {
            decoder: Some("ffmpeg-f32le-16k-mono".into()),
            tool_versions: self.tool_versions.clone(),
            ..KeyExtras::default()
        }
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<()>>, StageError> {
        // The probe is declared for ordering (timeline offsets come from ffprobe).
        let _ = inputs.header(MEDIA_PROBE)?;
        Ok(vec![WorkItem {
            id: "audio".into(),
            work: (),
        }])
    }
    async fn process(&self, ctx: &ItemContext, _work: ()) -> Result<AudioItem, ErrorInfo> {
        let started = Instant::now();
        let audio = match extract_audio(&self.tools, &self.video).await {
            Ok(a) => a,
            Err(AudioError::NoAudioStream(_)) => {
                tracing::warn!(video = %self.video.display(), "no audio stream; the audio branch is skipped");
                return Ok(AudioItem {
                    has_audio: false,
                    path: None,
                    blake3: None,
                    sample_rate: SAMPLE_RATE,
                    n_samples: 0,
                    duration_s: 0.0,
                    audio_start_s: None,
                    video_start_s: None,
                    timeline_offset_s: 0.0,
                });
            }
            Err(e) => return Err(audio_err(&e)),
        };
        ctx.record_command(
            vec![
                self.tools.ffmpeg.display().to_string(),
                "-i".into(),
                self.video.display().to_string(),
                "-map".into(),
                "0:a:0".into(),
                "-ac".into(),
                "1".into(),
                "-ar".into(),
                SAMPLE_RATE.to_string(),
                "-f".into(),
                "f32le".into(),
                "pipe:1".into(),
            ],
            Some(0),
            Some(started.elapsed().as_secs_f64()),
        );
        let duration_s = audio.duration_s();
        let n_samples = audio.samples.len() as u64;
        let dir = self.params.store_dir.clone();
        let samples = audio.samples;
        let (path, hash) = tokio::task::spawn_blocking(move || store_samples(&dir, &samples))
            .await
            .map_err(task_err)??;
        Ok(AudioItem {
            has_audio: true,
            path: Some(path.display().to_string()),
            blake3: Some(hash),
            sample_rate: audio.sample_rate,
            n_samples,
            duration_s,
            audio_start_s: Some(audio.audio_start_s),
            video_start_s: audio.video_start_s,
            timeline_offset_s: audio.timeline_offset_s,
        })
    }
}

/// The audio item, when the artifact has one with audio.
fn audio_input(inputs: &StageInputs) -> Result<Option<AudioItem>, StageError> {
    Ok(inputs
        .read_ok::<AudioItem>(AUDIO)?
        .into_iter()
        .map(|(_, a)| a)
        .find(|a| a.has_audio))
}

fn samples_of(a: &AudioItem) -> Result<(PathBuf, String), ErrorInfo> {
    match (&a.path, &a.blake3) {
        (Some(p), Some(h)) => Ok((PathBuf::from(p), h.clone())),
        _ => Err(ErrorInfo::new(
            ErrorCode::InvalidInput,
            "audio item has no sample file",
        )),
    }
}

// ---------------------------------------------------------------------- asr

/// A speech recognizer for [`AsrStage`] (blocking; called on a blocking thread).
pub trait AsrEngine: Send + Sync {
    /// Settings and model identity, recorded in the stage params and cache key.
    fn describe(&self) -> Value;
    /// Transcribes 16 kHz mono samples; `vocabulary` (names first) conditions
    /// the decoder and the correction pass.
    fn transcribe(&self, samples: &[f32], vocabulary: &[String]) -> crate::Result<AsrOutput>;
}

/// whisper.cpp through `whisper-rs` (Metal on macOS, CUDA with the `cuda` feature).
pub struct WhisperEngine {
    cfg: AsrConfig,
}

impl WhisperEngine {
    /// An engine for `cfg` (its `vocabulary` is replaced per call).
    pub fn new(cfg: AsrConfig) -> Self {
        Self { cfg }
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

impl AsrEngine for WhisperEngine {
    fn describe(&self) -> Value {
        let c = &self.cfg;
        serde_json::json!({
            "engine": "whisper.cpp",
            "model": file_name(&c.model_path),
            "vad_model": c.vad.as_ref().map(|v| file_name(&v.model_path)),
            "language": c.language,
            "beam_size": c.beam_size,
            "max_prompt_tokens": c.max_prompt_tokens,
            "carry_previous_words": c.carry_previous_words,
            "dtw": c.dtw,
            "max_chunk_s": c.max_chunk_s,
            "max_merge_gap_s": c.max_merge_gap_s,
            "use_gpu": c.use_gpu,
        })
    }
    fn transcribe(&self, samples: &[f32], vocabulary: &[String]) -> crate::Result<AsrOutput> {
        let mut cfg = self.cfg.clone();
        cfg.vocabulary = vocabulary.to_vec();
        Transcriber::new(cfg)?.transcribe(samples)
    }
}

/// One recognized word.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AsrWordRecord {
    /// Word with attached punctuation.
    pub w: String,
    /// Start, seconds (audio timeline).
    pub start_s: f64,
    /// End, seconds (audio timeline).
    pub end_s: f64,
    /// Mean token probability.
    pub p: f32,
}

/// One decoded segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AsrSegmentRecord {
    /// Start, seconds (audio timeline).
    pub start_s: f64,
    /// End, seconds (audio timeline).
    pub end_s: f64,
    /// Words.
    pub words: Vec<AsrWordRecord>,
}

/// `glassrip.asr` item (one item, id `asr`; none without audio).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AsrRecord {
    /// Sample file the words come from (also read by gap filling).
    pub audio_path: String,
    /// Its blake3.
    pub audio_blake3: String,
    /// Seconds added to audio times to reach the video timeline.
    pub timeline_offset_s: f64,
    /// Vocabulary given to the recognizer (names first).
    pub vocabulary: Vec<String>,
    /// Segments on the audio timeline.
    pub segments: Vec<AsrSegmentRecord>,
    /// Decode chunks `(start_s, end_s)`.
    pub chunk_spans: Vec<(f64, f64)>,
    /// Speech regions found by the detector.
    pub speech_regions: usize,
    /// Prompt tokens used.
    pub prompt_tokens: usize,
    /// Vocabulary terms that fit in the prompt.
    pub prompt_terms: Vec<String>,
    /// Device the decoder ran on.
    pub backend: String,
    /// Decode wall time, seconds.
    pub wall_s: f64,
}

impl AsrRecord {
    /// The library form for assembly.
    pub fn to_output(&self) -> AsrOutput {
        AsrOutput {
            segments: self
                .segments
                .iter()
                .map(|s| AsrSegment {
                    start_s: s.start_s,
                    end_s: s.end_s,
                    words: s
                        .words
                        .iter()
                        .map(|w| AsrWord {
                            w: w.w.clone(),
                            start_s: w.start_s,
                            end_s: w.end_s,
                            p: w.p,
                        })
                        .collect(),
                })
                .collect(),
            chunk_spans: self.chunk_spans.clone(),
            speech_regions: self.speech_regions,
            prompt_tokens: self.prompt_tokens,
            prompt_terms: self.prompt_terms.clone(),
            backend: self.backend.clone(),
        }
    }
}

/// `asr` parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AsrParams {
    /// [`AsrEngine::describe`].
    pub engine: Value,
    /// Vocabulary terms taken from `glassrip.asr_vocabulary`, at most.
    pub max_vocabulary_terms: usize,
}

/// Work of [`AsrStage`].
#[derive(Debug, Clone)]
pub struct AsrWork {
    audio: AudioItem,
    vocabulary: Vec<String>,
}

/// `glassrip.asr_vocabulary` item as read here (terms in priority order).
#[derive(Debug, Clone, Deserialize)]
struct VocabularyView {
    #[serde(default)]
    terms: Vec<TermView>,
}

#[derive(Debug, Clone, Deserialize)]
struct TermView {
    text: String,
}

/// Speech recognition with the on-screen vocabulary.
pub struct AsrStage {
    params: AsrParams,
    engine: Arc<dyn AsrEngine>,
}

impl AsrStage {
    /// A stage around `engine`.
    pub fn new(engine: Arc<dyn AsrEngine>, max_vocabulary_terms: usize) -> Self {
        Self {
            params: AsrParams {
                engine: engine.describe(),
                max_vocabulary_terms,
            },
            engine,
        }
    }
}

impl Stage for AsrStage {
    type Params = AsrParams;
    type Work = AsrWork;
    type Output = AsrRecord;

    fn name(&self) -> &'static str {
        "asr"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: ASR,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![input(AUDIO), input(ASR_VOCABULARY)]
    }
    fn params(&self) -> &AsrParams {
        &self.params
    }
    fn item_timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(4 * 3600))
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<AsrWork>>, StageError> {
        let Some(audio) = audio_input(inputs)? else {
            return Ok(Vec::new());
        };
        let mut vocabulary: Vec<String> = Vec::new();
        for (_, v) in inputs.read_ok::<VocabularyView>(ASR_VOCABULARY)? {
            for t in v.terms {
                let t = t.text.trim().to_string();
                if !t.is_empty() && !vocabulary.contains(&t) {
                    vocabulary.push(t);
                }
            }
        }
        vocabulary.truncate(self.params.max_vocabulary_terms);
        Ok(vec![WorkItem {
            id: "asr".into(),
            work: AsrWork { audio, vocabulary },
        }])
    }
    async fn process(&self, _ctx: &ItemContext, work: AsrWork) -> Result<AsrRecord, ErrorInfo> {
        let (path, hash) = samples_of(&work.audio)?;
        let engine = Arc::clone(&self.engine);
        let vocab = work.vocabulary.clone();
        let (p, h) = (path.clone(), hash.clone());
        let started = Instant::now();
        let out = tokio::task::spawn_blocking(move || -> Result<AsrOutput, ErrorInfo> {
            let samples = load_samples(&p, &h)?;
            engine
                .transcribe(&samples, &vocab)
                .map_err(|e| audio_err(&e))
        })
        .await
        .map_err(task_err)??;
        Ok(AsrRecord {
            audio_path: path.display().to_string(),
            audio_blake3: hash,
            timeline_offset_s: work.audio.timeline_offset_s,
            vocabulary: work.vocabulary,
            segments: out
                .segments
                .iter()
                .map(|s| AsrSegmentRecord {
                    start_s: s.start_s,
                    end_s: s.end_s,
                    words: s
                        .words
                        .iter()
                        .map(|w| AsrWordRecord {
                            w: w.w.clone(),
                            start_s: w.start_s,
                            end_s: w.end_s,
                            p: w.p,
                        })
                        .collect(),
                })
                .collect(),
            chunk_spans: out.chunk_spans,
            speech_regions: out.speech_regions,
            prompt_tokens: out.prompt_tokens,
            prompt_terms: out.prompt_terms,
            backend: out.backend,
            wall_s: started.elapsed().as_secs_f64(),
        })
    }
}

// ------------------------------------------------------------------ diarize

/// A diarizer for [`DiarizeStage`] (blocking; called on a blocking thread).
pub trait DiarizeEngine: Send + Sync {
    /// Settings and model identity, recorded in the stage params and cache key.
    fn describe(&self) -> Value;
    /// Diarizes 16 kHz mono samples into exclusive turns (audio timeline).
    fn diarize(&self, samples: &[f32]) -> crate::Result<Diarization>;
    /// Embedder for gap filling (speech the diarizer left uncovered), when the
    /// engine has one.
    fn embedder(&self) -> crate::Result<Option<Box<dyn SpanEmbedder>>> {
        Ok(None)
    }
}

/// speakrs (pyannote community-1 port). Needs the `diarize` feature (or `cuda`);
/// without it every call fails with a message naming the feature.
pub struct SpeakrsEngine {
    cfg: DiarizeConfig,
}

impl SpeakrsEngine {
    /// An engine for `cfg`.
    pub fn new(cfg: DiarizeConfig) -> Self {
        Self { cfg }
    }

    /// True when this build can diarize.
    pub fn available() -> bool {
        cfg!(feature = "diarize")
    }
}

impl DiarizeEngine for SpeakrsEngine {
    fn describe(&self) -> Value {
        serde_json::json!({
            "engine": "speakrs-0.5",
            "mode": self.cfg.mode.as_str(),
            "num_speakers": self.cfg.num_speakers,
        })
    }
    fn diarize(&self, samples: &[f32]) -> crate::Result<Diarization> {
        crate::diarize::diarize(samples, &self.cfg)
    }
    #[cfg(feature = "diarize")]
    fn embedder(&self) -> crate::Result<Option<Box<dyn SpanEmbedder>>> {
        Ok(Some(Box::new(crate::gapfill::SpeakrsEmbedder::new(
            &self.cfg,
        )?)))
    }
}

/// One speaker for the whole recording (`--speakers 1`): a single turn over the
/// audio, no model. Always available.
pub struct SingleSpeakerEngine;

impl DiarizeEngine for SingleSpeakerEngine {
    fn describe(&self) -> Value {
        serde_json::json!({"engine": "single-speaker"})
    }
    fn diarize(&self, samples: &[f32]) -> crate::Result<Diarization> {
        let duration = samples.len() as f64 / f64::from(SAMPLE_RATE);
        Ok(Diarization {
            turns: vec![Turn::new(0.0, duration, 0)],
            labels: vec![crate::diarize::speaker_label(0)],
            talk_time_s: vec![duration],
            num_clusters_raw: 1,
            active_s: duration,
            centroids: vec![None],
        })
    }
}

/// `glassrip.diarization` item (one item, id `diarization`; none without audio).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DiarizationRecord {
    /// Exclusive turns (audio timeline).
    pub turns: Vec<Turn>,
    /// Label per speaker index.
    pub labels: Vec<String>,
    /// Talk time per speaker index, seconds.
    pub talk_time_s: Vec<f64>,
    /// Clusters before re-clustering.
    pub num_clusters_raw: usize,
    /// Seconds with any speaker active.
    pub active_s: f64,
    /// Embedding centroid per speaker index.
    pub centroids: Vec<Option<Vec<f32>>>,
    /// Wall time, seconds.
    pub wall_s: f64,
}

impl DiarizationRecord {
    /// The library form.
    pub fn to_diarization(&self) -> Diarization {
        Diarization {
            turns: self.turns.clone(),
            labels: self.labels.clone(),
            talk_time_s: self.talk_time_s.clone(),
            num_clusters_raw: self.num_clusters_raw,
            active_s: self.active_s,
            centroids: self.centroids.clone(),
        }
    }
}

/// `diarize` parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DiarizeParams {
    /// [`DiarizeEngine::describe`].
    pub engine: Value,
}

/// Speaker diarization (reads only the audio, so it runs next to ASR).
pub struct DiarizeStage {
    params: DiarizeParams,
    engine: Arc<dyn DiarizeEngine>,
}

impl DiarizeStage {
    /// A stage around `engine`.
    pub fn new(engine: Arc<dyn DiarizeEngine>) -> Self {
        Self {
            params: DiarizeParams {
                engine: engine.describe(),
            },
            engine,
        }
    }
}

impl Stage for DiarizeStage {
    type Params = DiarizeParams;
    type Work = AudioItem;
    type Output = DiarizationRecord;

    fn name(&self) -> &'static str {
        "diarize"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: DIARIZATION,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![input(AUDIO)]
    }
    fn params(&self) -> &DiarizeParams {
        &self.params
    }
    fn item_timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(4 * 3600))
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<AudioItem>>, StageError> {
        Ok(audio_input(inputs)?
            .map(|audio| WorkItem {
                id: "diarization".into(),
                work: audio,
            })
            .into_iter()
            .collect())
    }
    async fn process(
        &self,
        _ctx: &ItemContext,
        audio: AudioItem,
    ) -> Result<DiarizationRecord, ErrorInfo> {
        let (path, hash) = samples_of(&audio)?;
        let engine = Arc::clone(&self.engine);
        let started = Instant::now();
        let d = tokio::task::spawn_blocking(move || -> Result<Diarization, ErrorInfo> {
            let samples = load_samples(&path, &hash)?;
            engine.diarize(&samples).map_err(|e| audio_err(&e))
        })
        .await
        .map_err(task_err)??;
        Ok(DiarizationRecord {
            turns: d.turns,
            labels: d.labels,
            talk_time_s: d.talk_time_s,
            num_clusters_raw: d.num_clusters_raw,
            active_s: d.active_s,
            centroids: d.centroids,
            wall_s: started.elapsed().as_secs_f64(),
        })
    }
}

// ----------------------------------------------------------------- gap fill

/// `gap_fill` parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GapFillStageParams {
    /// Gap filling gates; `None` disables gap filling.
    pub gap_fill: Option<GapFillConfig>,
    /// [`DiarizeEngine::describe`] of the embedder source, when there is one.
    pub embedder: Option<Value>,
}

/// `glassrip.gap_fill` item (one item, id `gap_fill`; none without audio).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GapFillRecord {
    /// Diarization with the gap-fill turns added (audio timeline); `None`
    /// without diarization.
    pub diarization: Option<DiarizationRecord>,
    /// Counters.
    pub stats: crate::gapfill::GapFillStats,
    /// One record per uncovered span (audio timeline).
    pub spans: Vec<crate::gapfill::GapSpan>,
    /// The embedder (and the audio) were loaded: at least one span was long
    /// enough to embed.
    pub embedder_loaded: bool,
}

/// Work of [`GapFillStage`].
#[derive(Debug, Clone)]
pub struct GapFillWork {
    asr: AsrRecord,
    diarization: Option<DiarizationRecord>,
}

/// Labels ASR words that no diarization turn covers by embedding their audio
/// and matching the speaker centroids. Planning only reads the two input
/// artifacts; the embedder and the audio are loaded in `process`, on a blocking
/// thread, and only when some uncovered span is long enough to embed. The
/// output is a cached artifact, so a resumed run does not redo the inference.
pub struct GapFillStage {
    params: GapFillStageParams,
    embedder_source: Option<Arc<dyn DiarizeEngine>>,
}

impl GapFillStage {
    /// A stage; without `embedder_source` (or with `gap_fill: None`) the
    /// diarization passes through unchanged.
    pub fn new(
        gap_fill: Option<GapFillConfig>,
        embedder_source: Option<Arc<dyn DiarizeEngine>>,
    ) -> Self {
        Self {
            params: GapFillStageParams {
                gap_fill,
                embedder: embedder_source.as_ref().map(|e| e.describe()),
            },
            embedder_source,
        }
    }
}

/// Loads the embedder on the first embedding request.
struct LazyEmbedder<'a> {
    source: &'a dyn DiarizeEngine,
    inner: Option<Box<dyn SpanEmbedder>>,
    unavailable: bool,
}

impl SpanEmbedder for LazyEmbedder<'_> {
    fn embed(&mut self, audio: &[f32]) -> crate::Result<Option<Vec<f32>>> {
        if self.inner.is_none() && !self.unavailable {
            match self.source.embedder()? {
                Some(e) => self.inner = Some(e),
                None => self.unavailable = true,
            }
        }
        match self.inner.as_mut() {
            Some(e) => e.embed(audio),
            None => Ok(None),
        }
    }
}

/// The gap-fill computation (blocking: may load a model and the audio).
pub fn gap_fill_blocking(
    work: GapFillWork,
    cfg: Option<GapFillConfig>,
    source: Option<&dyn DiarizeEngine>,
) -> Result<GapFillRecord, ErrorInfo> {
    let unchanged = |diarization| GapFillRecord {
        diarization,
        stats: crate::gapfill::GapFillStats::default(),
        spans: Vec::new(),
        embedder_loaded: false,
    };
    let Some(record) = work.diarization else {
        return Ok(unchanged(None));
    };
    let (Some(cfg), Some(source)) = (cfg, source) else {
        return Ok(unchanged(Some(record)));
    };
    let mut words: Vec<(f64, f64)> = work
        .asr
        .segments
        .iter()
        .flat_map(|s| s.words.iter().map(|w| (w.start_s, w.end_s)))
        .collect();
    words.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut diar = record.to_diarization();
    let spans = crate::gapfill::uncovered_spans(&words, &diar.turns, &cfg);
    if !spans.iter().any(|(s, e)| e - s >= cfg.min_span_s) {
        // Nothing long enough to embed: no model, no audio.
        let stats = crate::gapfill::GapFillStats {
            spans: spans.len(),
            rejected_short: spans.len(),
            ..Default::default()
        };
        let spans = spans
            .into_iter()
            .map(|(start_s, end_s)| crate::gapfill::GapSpan {
                start_s,
                end_s,
                status: crate::gapfill::SpanStatus::TooShort,
                speaker: None,
                similarity: None,
                margin: None,
            })
            .collect();
        return Ok(GapFillRecord {
            diarization: Some(record),
            stats,
            spans,
            embedder_loaded: false,
        });
    }
    let samples = load_samples(Path::new(&work.asr.audio_path), &work.asr.audio_blake3)?;
    let mut embedder = LazyEmbedder {
        source,
        inner: None,
        unavailable: false,
    };
    let (stats, spans) = crate::gapfill::fill_gaps_with(
        &mut embedder,
        &samples,
        SAMPLE_RATE,
        &mut diar,
        &words,
        &cfg,
    )
    .map_err(|e| audio_err(&e))?;
    let embedder_loaded = embedder.inner.is_some();
    Ok(GapFillRecord {
        diarization: Some(DiarizationRecord {
            turns: diar.turns,
            labels: diar.labels,
            talk_time_s: diar.talk_time_s,
            num_clusters_raw: diar.num_clusters_raw,
            active_s: diar.active_s,
            centroids: diar.centroids,
            wall_s: record.wall_s,
        }),
        stats,
        spans,
        embedder_loaded,
    })
}

impl Stage for GapFillStage {
    type Params = GapFillStageParams;
    type Work = GapFillWork;
    type Output = GapFillRecord;

    fn name(&self) -> &'static str {
        "gap_fill"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: GAP_FILL,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![input(ASR), input(DIARIZATION)]
    }
    fn params(&self) -> &GapFillStageParams {
        &self.params
    }
    fn item_timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(4 * 3600))
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<GapFillWork>>, StageError> {
        let Some((_, asr)) = inputs.read_ok::<AsrRecord>(ASR)?.into_iter().next() else {
            return Ok(Vec::new());
        };
        let diarization = inputs
            .read_ok::<DiarizationRecord>(DIARIZATION)?
            .into_iter()
            .next()
            .map(|(_, d)| d);
        Ok(vec![WorkItem {
            id: "gap_fill".into(),
            work: GapFillWork { asr, diarization },
        }])
    }
    async fn process(
        &self,
        _ctx: &ItemContext,
        work: GapFillWork,
    ) -> Result<GapFillRecord, ErrorInfo> {
        let cfg = self.params.gap_fill;
        let source = self.embedder_source.clone();
        tokio::task::spawn_blocking(move || gap_fill_blocking(work, cfg, source.as_deref()))
            .await
            .map_err(task_err)?
    }
}

// ------------------------------------------------------------- assign words

/// `assign_words` parameters.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AssignWordsParams {
    /// Vocabulary correction thresholds.
    pub correction: CorrectionConfig,
    /// Word assignment settings.
    pub assign: AssignConfig,
}

/// Vocabulary correction and word-to-speaker assignment into
/// `glassrip.transcript`, one item per segment (id `segment_id`). Planning reads
/// the ASR and gap-fill artifacts and groups words into segments (text work
/// only; no model and no audio).
pub struct AssignWordsStage {
    params: AssignWordsParams,
}

impl AssignWordsStage {
    /// A stage.
    pub fn new(params: AssignWordsParams) -> Self {
        Self { params }
    }
}

/// Builds the transcript segments from the ASR record and the (gap-filled)
/// diarization.
pub fn assemble_transcript(
    asr: &AsrRecord,
    diar: Option<&DiarizationRecord>,
    params: &AssignWordsParams,
) -> Vec<TranscriptSegment> {
    let output = asr.to_output();
    let diarization = diar.map(DiarizationRecord::to_diarization);
    crate::transcript::build_segments(
        &output.segments,
        diarization.as_ref(),
        &crate::vocab::Vocabulary::new(&asr.vocabulary),
        &params.correction,
        &params.assign,
        asr.timeline_offset_s,
    )
}

impl Stage for AssignWordsStage {
    type Params = AssignWordsParams;
    type Work = TranscriptSegment;
    type Output = TranscriptSegment;

    fn name(&self) -> &'static str {
        "assign_words"
    }
    fn version(&self) -> u32 {
        2
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: TRANSCRIPT,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![input(ASR), input(GAP_FILL)]
    }
    fn params(&self) -> &AssignWordsParams {
        &self.params
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<TranscriptSegment>>, StageError> {
        let Some((_, asr)) = inputs.read_ok::<AsrRecord>(ASR)?.into_iter().next() else {
            // No audio stream: an empty transcript (spec 8.1).
            return Ok(Vec::new());
        };
        let diar = inputs
            .read_ok::<GapFillRecord>(GAP_FILL)?
            .into_iter()
            .next()
            .and_then(|(_, g)| g.diarization);
        Ok(assemble_transcript(&asr, diar.as_ref(), &self.params)
            .into_iter()
            .map(|s| WorkItem {
                id: s.segment_id.clone(),
                work: s,
            })
            .collect())
    }
    async fn process(
        &self,
        _ctx: &ItemContext,
        segment: TranscriptSegment,
    ) -> Result<TranscriptSegment, ErrorInfo> {
        Ok(segment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_round_trip_and_hash_check() {
        let dir = tempfile::tempdir().unwrap();
        let samples = vec![0.0f32, 0.5, -0.25, 1.0];
        let (path, hash) = store_samples(dir.path(), &samples).unwrap();
        assert!(path.ends_with(format!("{hash}.f32le")));
        assert_eq!(load_samples(&path, &hash).unwrap(), samples);
        // Storing again keeps the same file.
        assert_eq!(store_samples(dir.path(), &samples).unwrap().0, path);
        assert!(load_samples(&path, "0000").is_err());
        assert!(load_samples(&dir.path().join("missing.f32le"), &hash).is_err());
    }

    fn asr_record(audio_path: &str, audio_blake3: &str) -> AsrRecord {
        let word = |w: &str, s: f64, p: f32| AsrWordRecord {
            w: w.into(),
            start_s: s,
            end_s: s + 0.4,
            p,
        };
        AsrRecord {
            audio_path: audio_path.into(),
            audio_blake3: audio_blake3.into(),
            timeline_offset_s: 0.5,
            vocabulary: vec!["Quorra".into()],
            segments: vec![AsrSegmentRecord {
                start_s: 1.0,
                end_s: 4.0,
                words: vec![
                    word("Hello", 1.0, 0.9),
                    word("Quora,", 1.5, 0.3),
                    word("ship", 2.5, 0.9),
                    word("it.", 3.0, 0.9),
                ],
            }],
            chunk_spans: vec![(0.0, 5.0)],
            speech_regions: 1,
            prompt_tokens: 3,
            prompt_terms: vec!["Quorra".into()],
            backend: "scripted".into(),
            wall_s: 0.0,
        }
    }

    #[test]
    fn assembly_without_diarization_labels_one_speaker() {
        let segs = assemble_transcript(&asr_record("x", "y"), None, &AssignWordsParams::default());
        assert_eq!(segs.len(), 1);
        let s = &segs[0];
        assert_eq!(s.speaker_label, "SPEAKER_00");
        // Times move onto the video timeline.
        assert!((s.start_s - 1.5).abs() < 1e-9, "{}", s.start_s);
        // The low-probability word near a vocabulary term is corrected, raw kept.
        assert!(s.text.contains("Quorra"), "{}", s.text);
        assert!(s.text_raw.contains("Quora,"), "{}", s.text_raw);
    }

    #[test]
    fn assembly_with_turns_splits_by_speaker() {
        let diar = DiarizationRecord {
            turns: vec![Turn::new(0.0, 2.2, 0), Turn::new(2.2, 5.0, 1)],
            labels: vec!["SPEAKER_00".into(), "SPEAKER_01".into()],
            talk_time_s: vec![2.2, 2.8],
            num_clusters_raw: 2,
            active_s: 5.0,
            centroids: vec![None, None],
            wall_s: 0.0,
        };
        let segs = assemble_transcript(
            &asr_record("x", "y"),
            Some(&diar),
            &AssignWordsParams::default(),
        );
        let labels: Vec<&str> = segs.iter().map(|s| s.speaker_label.as_str()).collect();
        assert_eq!(labels, vec!["SPEAKER_00", "SPEAKER_01"]);
    }

    // ---- gap_fill and assign_words on the core runner.

    use std::sync::atomic::{AtomicUsize, Ordering};

    use glassrip_core::cache::Cache;
    use glassrip_core::envelope::{EnvelopeHeader, Outcome, Producer, Record, SchemaReq};
    use glassrip_core::graph::{Selection, StageDecl, StageGraph};
    use glassrip_core::manifest::RunDir;
    use glassrip_core::runner::{stage_decl, Runner, RunnerError, RunnerOptions};

    /// Counts embedder loads; fails the load when `fail` is set.
    struct CountingSource {
        loads: Arc<AtomicUsize>,
        fail: bool,
    }

    struct FixedEmbedder;

    impl SpanEmbedder for FixedEmbedder {
        fn embed(&mut self, _audio: &[f32]) -> crate::Result<Option<Vec<f32>>> {
            Ok(Some(vec![1.0, 0.0]))
        }
    }

    impl DiarizeEngine for CountingSource {
        fn describe(&self) -> Value {
            serde_json::json!({"engine": "counting"})
        }
        fn diarize(&self, _samples: &[f32]) -> crate::Result<Diarization> {
            Err(AudioError::InvalidInput("unused".into()))
        }
        fn embedder(&self) -> crate::Result<Option<Box<dyn SpanEmbedder>>> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(AudioError::InvalidInput("embedder load failed".into()));
            }
            Ok(Some(Box::new(FixedEmbedder)))
        }
    }

    fn write<T: Serialize>(run: &Path, schema: &str, id: &str, item: T) {
        let header = EnvelopeHeader {
            schema: schema.into(),
            schema_version: v1(),
            run_id: "t".into(),
            producer: Producer::glassrip("0.1.0", None),
            inputs: vec![],
            params: serde_json::json!({}),
            content_hash: None,
            restored_from: None,
        };
        let records = vec![Record {
            id: id.to_string(),
            outcome: Outcome::ok(item),
        }];
        let path = run.join("artifacts").join(format!("{schema}.jsonl"));
        glassrip_core::jsonl::write_atomic(&path, &header, &records).unwrap();
    }

    /// Words at 1.0 to 2.2 s (covered by speaker 0) and 6.0 to 7.2 s (uncovered).
    fn inputs(run: &Path, audio: &Path, gap: bool) {
        let (path, hash) = store_samples(audio, &vec![0.1f32; 16_000 * 10]).unwrap();
        let word = |w: &str, s: f64| AsrWordRecord {
            w: w.into(),
            start_s: s,
            end_s: s + 0.4,
            p: 0.9,
        };
        let mut asr = asr_record(&path.display().to_string(), &hash);
        asr.segments = vec![
            AsrSegmentRecord {
                start_s: 1.0,
                end_s: 2.2,
                words: vec![word("one", 1.0), word("two", 1.4), word("three", 1.8)],
            },
            AsrSegmentRecord {
                start_s: 6.0,
                end_s: 7.2,
                words: vec![word("four", 6.0), word("five", 6.4), word("six", 6.8)],
            },
        ];
        write(run, ASR, "asr", asr);
        let end = if gap { 3.0 } else { 9.0 };
        let diar = DiarizationRecord {
            turns: vec![Turn::new(0.0, end, 0)],
            labels: vec!["SPEAKER_00".into()],
            talk_time_s: vec![end],
            num_clusters_raw: 1,
            active_s: end,
            centroids: vec![Some(vec![1.0, 0.0])],
            wall_s: 0.0,
        };
        write(run, DIARIZATION, "diarization", diar);
    }

    fn runner(root: &Path, run: &str, gap: &GapFillStage, assign: &AssignWordsStage) -> Runner {
        let graph = StageGraph::new(vec![
            StageDecl::new("asr", ASR, &[]),
            StageDecl::new("diarize", DIARIZATION, &[]),
            stage_decl(gap),
            stage_decl(assign),
        ])
        .unwrap();
        let dir = RunDir::open(&root.join(run), run, Producer::glassrip("0.1.0", None)).unwrap();
        Runner::new(
            dir,
            graph,
            &Selection::default(),
            Cache::new(root.join("cache")),
            RunnerOptions::default(),
            tokio_util::sync::CancellationToken::new(),
        )
        .unwrap()
    }

    fn read_one<T: serde::de::DeserializeOwned>(run: &Path, schema: &str) -> Vec<T> {
        glassrip_core::jsonl::read::<Record<T>>(
            &run.join("artifacts").join(format!("{schema}.jsonl")),
            &SchemaReq::new(schema, 1),
        )
        .unwrap()
        .items
        .into_iter()
        .filter_map(|r| r.outcome.result)
        .collect()
    }

    #[tokio::test]
    async fn gap_fill_loads_nothing_without_long_gaps() {
        let tmp = tempfile::tempdir().unwrap();
        let loads = Arc::new(AtomicUsize::new(0));
        let source = Arc::new(CountingSource {
            loads: Arc::clone(&loads),
            fail: true,
        });
        let gap = GapFillStage::new(Some(GapFillConfig::default()), Some(source));
        let assign = AssignWordsStage::new(AssignWordsParams::default());
        let mut r = runner(tmp.path(), "run", &gap, &assign);
        inputs(&tmp.path().join("run"), &tmp.path().join("audio"), false);
        r.run_stage(&gap).await.unwrap();
        r.run_stage(&assign).await.unwrap();
        assert_eq!(
            loads.load(Ordering::SeqCst),
            0,
            "embedder must stay unloaded"
        );
        let rec: Vec<GapFillRecord> = read_one(&tmp.path().join("run"), GAP_FILL);
        assert!(!rec[0].embedder_loaded);
        let segs: Vec<TranscriptSegment> = read_one(&tmp.path().join("run"), TRANSCRIPT);
        assert!(segs.iter().all(|s| s.speaker_label == "SPEAKER_00"));
    }

    #[tokio::test]
    async fn a_failing_loader_is_an_item_error_not_a_plan_error() {
        let tmp = tempfile::tempdir().unwrap();
        let loads = Arc::new(AtomicUsize::new(0));
        let source = Arc::new(CountingSource {
            loads: Arc::clone(&loads),
            fail: true,
        });
        let gap = GapFillStage::new(Some(GapFillConfig::default()), Some(source));
        let assign = AssignWordsStage::new(AssignWordsParams::default());
        let mut r = runner(tmp.path(), "run", &gap, &assign);
        inputs(&tmp.path().join("run"), &tmp.path().join("audio"), true);
        // plan() only reads the input artifacts; the load happens in process().
        let err = r.run_stage(&gap).await.unwrap_err();
        assert!(
            matches!(err, RunnerError::ErrorRateExceeded { .. }),
            "the loader ran during plan: {err}"
        );
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn gap_fill_labels_gaps_once_and_resumes_from_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let loads = Arc::new(AtomicUsize::new(0));
        let source: Arc<dyn DiarizeEngine> = Arc::new(CountingSource {
            loads: Arc::clone(&loads),
            fail: false,
        });
        let gap = GapFillStage::new(Some(GapFillConfig::default()), Some(Arc::clone(&source)));
        let assign = AssignWordsStage::new(AssignWordsParams::default());
        for run in ["first", "second"] {
            let mut r = runner(tmp.path(), run, &gap, &assign);
            inputs(&tmp.path().join(run), &tmp.path().join("audio"), true);
            let rep = r.run_stage(&gap).await.unwrap();
            r.run_stage(&assign).await.unwrap();
            if run == "second" {
                assert_eq!(rep.status, glassrip_core::manifest::StageStatus::Cached);
            }
        }
        assert_eq!(
            loads.load(Ordering::SeqCst),
            1,
            "the resume must not re-embed"
        );
        let rec: Vec<GapFillRecord> = read_one(&tmp.path().join("second"), GAP_FILL);
        assert!(rec[0].embedder_loaded);
        assert_eq!(rec[0].stats.labeled, 1);
        let segs: Vec<TranscriptSegment> = read_one(&tmp.path().join("second"), TRANSCRIPT);
        let late = segs.iter().find(|s| s.text.contains("four")).unwrap();
        assert_eq!(late.speaker_label, "SPEAKER_00");
        assert!(late
            .words
            .iter()
            .all(|w| w.source == crate::recluster::Source::GapFill));
    }
}
