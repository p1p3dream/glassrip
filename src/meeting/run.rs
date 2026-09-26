//! The meeting-mode run: every stage of [`meeting_mode_stage_decls`] on one core
//! [`Runner`], scheduled in the GPU phases of spec 5.3.
//!
//! | Order | Stages | Resident models |
//! |---|---|---|
//! | media (CPU) | probe, orient, frames, screen_quad, features, keyframes, rectify | orientation model (CPU) |
//! | phase A | ocr_harvest, ocr_vocabulary, classify | PP-OCRv5 sessions, vision model |
//! | phase B | canvas_crop, board_read, board_validate, edge_direction **joined with** audio_extract, asr + diarize, assign_words | vision model, whisper, speakrs (OCR sessions dropped) |
//! | join (CPU) | board_state, name_speakers | |
//! | phase C | notes | text model (the notes stage unloads the vision model first) |
//! | output | render | |
//!
//! Phase B runs the vision and audio chains concurrently only when the vision
//! model is known to be at most `gpu.sequential_model_threshold_gb` (from `/api/ps`
//! when loaded, else the file size or parameter count the server reports, else
//! the tag); larger or unknown models run the chains one after the other, and
//! large models get one client slot. While ASR runs next to board reading, a
//! placement guard pauses vision requests if the model spills (spec 5.3).
//!
//! A model stage whose output restores from cache skips model preflight, so a
//! fully cached rerun works with the model server down (see
//! [`super::backends`]). Stages outside the selection are skipped by the runner
//! and their existing artifacts reused.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use glassrip_audio::extract::ExtractOptions;
use glassrip_audio::gapfill::GapFillConfig;
use glassrip_audio::stages::{
    AsrStage, AssignWordsParams, AssignWordsStage, AudioExtractStage, DiarizeStage, GapFillStage,
};
use glassrip_audio::types::TranscriptSegment;
use glassrip_core::cache::{Cache, DEFAULT_CACHE_DIR};
use glassrip_core::config::Config;
use glassrip_core::envelope::{InputRef, Producer, Record, SchemaReq};
use glassrip_core::graph::{
    meeting_mode_stage_decls, GraphError, Selection, StageDecision, StageGraph,
};
use glassrip_core::manifest::{ManifestError, StageStatus};
use glassrip_core::runner::{Runner, RunnerError, RunnerOptions, Stage, StageReport};
use glassrip_media_stages::blobs::BlobStore;
use glassrip_media_stages::features::FeaturesStage;
use glassrip_media_stages::frames::{FramesParams, FramesStage};
use glassrip_media_stages::keyframes::{KeyframesParams, KeyframesStage};
use glassrip_media_stages::orient::{OrientParams, OrientStage};
use glassrip_media_stages::pipeline::materialize;
use glassrip_media_stages::probe::ProbeStage;
use glassrip_media_stages::quad::{QuadParams, ScreenQuadStage};
use glassrip_media_stages::rectify::{RectifyParams, RectifyStage};
use glassrip_media_stages::schema::{
    FrameRecord, MediaProbe, RectifiedKeyframe, FRAMES, MEDIA_PROBE, RECTIFIED_KEYFRAMES,
};
use glassrip_media_stages::scoring::ScoringParams;
use glassrip_meeting::consolidate::ConsolidationParams;
use glassrip_meeting::pixel_direction::PixelCheckParams;
use glassrip_meeting::stages::{BoardStateStage, EdgeDirectionStage};
use glassrip_meeting::text::{AliasTable, Participant};
use glassrip_meeting::vlm_direction::VlmCheckParams;
use glassrip_notes::named::named_lines;
use glassrip_notes::notes::llm::{
    ChatRequest, ChatResponse, LlmError, LoadedModel, OllamaTextConfig, TextBackend,
};
use glassrip_notes::notes::{NotesParams, NotesStage};
use glassrip_notes::schemas::{SPEAKERS, TRANSCRIPT};
use glassrip_notes::speakers::{
    NameSpeakersParams, NameSpeakersStage, SpeakersDoc, SpeakersRecord,
};
use glassrip_ocr::OcrConfig;
use glassrip_render::markdown::MarkdownMeta;
use glassrip_render::stage::RENDER_SCHEMA;
use glassrip_render::{RenderParams, RenderResult, RenderStage};
use glassrip_vision::VisionClient;
use glassrip_vision_stages::layout::LayoutConfig;
use glassrip_vision_stages::placement::{MonitorConfig, PlacementMonitor};
use glassrip_vision_stages::stages::canvas_crop::CanvasCropParams;
use glassrip_vision_stages::{
    BoardReadParams, BoardReadStage, BoardValidateParams, BoardValidateStage, CanvasCropStage,
    ClassifyParams, ClassifyStage, OcrHarvestStage, VocabularyParams, VocabularyStage,
};
use tokio_util::sync::CancellationToken;

use super::backends::{Backends, VisionBackends};
use super::logging::LogHandle;
use super::preflight::{self, Needs, ToolVersions};

/// On-screen vocabulary terms offered to ASR, at most (the prompt token budget
/// trims further).
pub const MAX_VOCABULARY_TERMS: usize = 200;

/// Everything a meeting run needs besides the backends.
#[derive(Debug, Clone)]
pub struct MeetingOptions {
    /// Source video.
    pub video: PathBuf,
    /// Output (run) directory.
    pub out_dir: PathBuf,
    /// File-name stem of the rendered outputs.
    pub stem: String,
    /// Run id recorded in artifacts.
    pub run_id: String,
    /// Configuration.
    pub config: Config,
    /// Participant display names.
    pub participants: Vec<String>,
    /// `--from-stage`, `--until-stage`, `--force-stage`.
    pub selection: Selection,
    /// Stage cache root.
    pub cache_dir: PathBuf,
    /// Blob store root (frames, keyframes, audio samples).
    pub blobs_dir: PathBuf,
    /// Orientation models directory.
    pub media_models_dir: PathBuf,
    /// Download missing orientation models (pinned URL and hash).
    pub allow_model_download: bool,
    /// `ffmpeg` binary.
    pub ffmpeg: String,
    /// `ffprobe` binary.
    pub ffprobe: String,
    /// Deferred `run.log.jsonl`, activated once preflight passes.
    pub log: Option<LogHandle>,
}

impl MeetingOptions {
    /// Defaults for a video and output directory, with the cache and blobs under
    /// `<workspace>/.glassrip`.
    pub fn new(video: PathBuf, out_dir: PathBuf, workspace: &Path, config: Config) -> Self {
        Self {
            stem: super::args::video_stem(&video),
            run_id: format!(
                "meeting-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            ),
            video,
            out_dir,
            config,
            participants: Vec::new(),
            selection: Selection::default(),
            cache_dir: workspace.join(DEFAULT_CACHE_DIR),
            blobs_dir: workspace.join(".glassrip/blobs"),
            media_models_dir: glassrip_media_stages::models::default_dir(),
            allow_model_download: true,
            ffmpeg: "ffmpeg".into(),
            ffprobe: "ffprobe".into(),
            log: None,
        }
    }
}

/// Phase B mode: the vision and audio chains run concurrently only when the
/// vision model is known to fit next to ASR. `loaded_bytes` is the placement
/// size when the model is loaded; `estimated_gb` comes from the server's file
/// size or parameter count, or the tag. Unknown size means sequential.
pub fn phase_b_concurrent(
    loaded_bytes: Option<u64>,
    estimated_gb: Option<f64>,
    threshold_gb: f64,
) -> bool {
    #[allow(clippy::cast_precision_loss)]
    let gb = loaded_bytes.map(|b| b as f64 / 1e9).or(estimated_gb);
    gb.is_some_and(|g| g <= threshold_gb)
}

/// Meeting run failure.
#[derive(Debug, thiserror::Error)]
pub enum MeetingError {
    /// Preflight found problems; nothing ran.
    #[error("preflight failed; nothing ran:\n  - {}", .0.join("\n  - "))]
    Preflight(Vec<String>),
    /// Stage graph or selection problem.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Run directory problem.
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    /// A stage failed.
    #[error(transparent)]
    Runner(Box<RunnerError>),
    /// A model stage stopped because the model server became unusable.
    #[error("stage `{stage}` aborted: {message}")]
    Aborted {
        /// Stage.
        stage: String,
        /// Reason.
        message: String,
    },
    /// A model stage is not cached and its model server is not used.
    #[error("stage `{stage}` needs its model, which is unavailable: {reason}")]
    Offline {
        /// Stage.
        stage: String,
        /// Why the server is not used.
        reason: String,
    },
    /// Setup problem (bad parameters, unreadable files).
    #[error("{0}")]
    Setup(String),
}

impl From<RunnerError> for MeetingError {
    fn from(e: RunnerError) -> Self {
        Self::Runner(Box::new(e))
    }
}

/// What a run produced.
#[derive(Debug, Clone, Default)]
pub struct MeetingOutcome {
    /// One report per stage run (or restored, or skipped), in run order.
    pub reports: Vec<StageReport>,
    /// Rendered files (notes, SVG, PNG), absolute paths.
    pub outputs: Vec<PathBuf>,
    /// Phase B ran the vision and audio chains concurrently.
    pub phase_b_concurrent: bool,
    /// Wall time per phase, seconds.
    pub phase_wall_s: BTreeMap<String, f64>,
}

/// Collects stage reports from concurrent chains.
#[derive(Default)]
struct Reports(Mutex<Vec<StageReport>>);

impl Reports {
    fn push(&self, r: StageReport) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(r);
    }
    fn take(&self) -> Vec<StageReport> {
        std::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

async fn step<S: Stage>(
    runner: &Runner,
    stage: &S,
    reports: &Reports,
) -> Result<StageStatus, MeetingError> {
    let rep = runner.run_stage_shared(stage).await?;
    tracing::info!(
        stage = %rep.stage,
        status = ?rep.status,
        items = rep.items_total,
        errors = rep.items_error,
        wall_s = rep.wall_s,
        "stage done"
    );
    let status = rep.status;
    reports.push(rep);
    Ok(status)
}

/// Fails a selected stage that would need a model whose server is not used
/// (offline rerun) and that will not restore from cache.
fn require_cached<S: Stage>(
    runner: &Runner,
    stage: &S,
    offline: Option<&str>,
) -> Result<(), MeetingError> {
    let selected = matches!(
        runner.plan().decision(stage.name()),
        Some(StageDecision::Run { .. })
    );
    match offline {
        Some(reason) if selected && !runner.cache_hit(stage) => Err(MeetingError::Offline {
            stage: stage.name().to_string(),
            reason: reason.to_string(),
        }),
        _ => Ok(()),
    }
}

/// A model stage: digest check before it (spec 8.1: the digest must not change
/// mid-run) and the sticky abort after it. A stage that restores from cache
/// skips the check (its key already includes the digest), so a fully cached
/// rerun needs no model server.
async fn model_step<S: Stage>(
    runner: &Runner,
    stage: &S,
    monitor: &PlacementMonitor,
    offline: Option<&str>,
    reports: &Reports,
) -> Result<StageStatus, MeetingError> {
    let name = stage.name();
    require_cached(runner, stage, offline)?;
    let selected = matches!(
        runner.plan().decision(name),
        Some(StageDecision::Run { .. })
    ) && !runner.cache_hit(stage);
    let aborted = |message: String| MeetingError::Aborted {
        stage: name.to_string(),
        message,
    };
    if selected {
        monitor
            .begin_stage()
            .await
            .map_err(|e| aborted(e.message))?;
    }
    let result = step(runner, stage, reports).await;
    if let Some(e) = monitor.abort_error().filter(|_| selected) {
        return Err(aborted(e.message));
    }
    result
}

fn alias_table(participants: &[String]) -> AliasTable {
    AliasTable::new(
        participants
            .iter()
            .map(|name| {
                let words: Vec<&str> = name.split_whitespace().collect();
                let mut aliases = Vec::new();
                if words.len() > 1 {
                    aliases.extend(words.first().map(|w| w.to_string()));
                    aliases.extend(words.last().map(|w| w.to_string()));
                }
                Participant {
                    person_id: glassrip_notes::people::slug(name),
                    display_name: name.clone(),
                    aliases,
                }
            })
            .collect(),
    )
}

/// The text backend when the text model is unavailable: every call fails with
/// the reason. Board-only notes (no speech) never call it.
struct NoTextModel(String);

impl NoTextModel {
    fn err(&self) -> LlmError {
        LlmError::Protocol(format!("text model unavailable: {}", self.0))
    }
}

#[async_trait::async_trait]
impl TextBackend for NoTextModel {
    async fn chat(&self, _model: &str, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        Err(self.err())
    }
    async fn load(&self, _model: &str) -> Result<(), LlmError> {
        Err(self.err())
    }
    async fn unload(&self, _model: &str) -> Result<(), LlmError> {
        Err(self.err())
    }
    async fn loaded(&self) -> Result<Vec<LoadedModel>, LlmError> {
        Err(self.err())
    }
    async fn digest(&self, _model: &str) -> Result<Option<String>, LlmError> {
        Err(self.err())
    }
}

/// Whether the run's transcript has speech, by the test `notes` applies (a
/// named line with words). Unreadable inputs count as speech, so the text
/// model is still required and `notes` reports its own input error.
fn transcript_has_speech(runner: &Runner) -> bool {
    let read = || -> Result<bool, glassrip_core::jsonl::JsonlError> {
        let (transcript, speakers) = {
            let dir = runner.run_dir();
            (dir.artifact_path(TRANSCRIPT), dir.artifact_path(SPEAKERS))
        };
        let segments: Vec<TranscriptSegment> = glassrip_core::jsonl::read::<
            Record<TranscriptSegment>,
        >(
            &transcript, &SchemaReq::new(TRANSCRIPT, 1)
        )?
        .items
        .into_iter()
        .filter_map(|r| r.outcome.result)
        .collect();
        let doc = SpeakersDoc::from_records(
            glassrip_core::jsonl::read::<Record<SpeakersRecord>>(
                &speakers,
                &SchemaReq::new(SPEAKERS, 1),
            )?
            .items
            .into_iter()
            .filter_map(|r| r.outcome.result),
        );
        Ok(named_lines(&segments, &doc)
            .iter()
            .any(|l| !l.text.trim().is_empty()))
    };
    read().unwrap_or(true)
}

/// The recording's length from the media probe, when it has run.
fn media_duration_s(runner: &Runner) -> Option<f64> {
    let path = runner.run_dir().artifact_path(MEDIA_PROBE);
    glassrip_core::jsonl::read::<Record<MediaProbe>>(&path, &SchemaReq::new(MEDIA_PROBE, 1))
        .ok()?
        .items
        .into_iter()
        .find_map(|r| r.outcome.result)
        .map(|p| p.duration_s)
        .filter(|d| d.is_finite() && *d > 0.0)
}

fn setup(e: impl std::fmt::Display) -> MeetingError {
    MeetingError::Setup(e.to_string())
}

/// The vision-model stages of phases A and B, sharing one placement monitor.
struct VisionStages {
    monitor: Arc<PlacementMonitor>,
    classify: ClassifyStage,
    board_read: BoardReadStage,
    client: VisionClient,
}

fn vision_stages(v: VisionBackends, cfg: &Config) -> Result<VisionStages, MeetingError> {
    let slots = v.slots(cfg.gpu.sequential_model_threshold_gb);
    if slots < v.concurrency {
        tracing::info!(
            model = %v.model,
            estimated_gb = v.estimated_gb(),
            slots,
            "large vision model: one request at a time"
        );
    }
    let client = VisionClient::new(v.backend, slots).map_err(setup)?;
    let monitor = Arc::new(PlacementMonitor::new(
        v.probe,
        client.clone(),
        MonitorConfig {
            num_ctx: cfg.ollama.num_ctx,
            ..MonitorConfig::default()
        },
    ));
    let classify = ClassifyStage::new(
        ClassifyParams {
            thumbnail_px: cfg.classify.thumbnail_px,
            seed: cfg.ollama.seed,
            ..ClassifyParams::default()
        },
        Arc::clone(&monitor),
        &v.model,
        v.digest.clone(),
        v.server_version.clone(),
    );
    let br = &cfg.board_read;
    let board_read = BoardReadStage::new(
        BoardReadParams {
            seed: cfg.ollama.seed,
            num_ctx: cfg.ollama.num_ctx,
            target_long_edge_px: br.target_long_edge_px,
            min_long_edge_px: br.min_long_edge_px,
            low_res_upscale: br.low_res_upscale,
            tiling_text_height_px: br.tiling_text_height_px,
            tile_grid: br.tile_grid,
            tile_overlap: br.tile_overlap,
            ..BoardReadParams::default()
        },
        Arc::clone(&monitor),
        &v.model,
        v.digest,
        v.server_version,
    );
    Ok(VisionStages {
        monitor,
        classify,
        board_read,
        client,
    })
}

/// Runs the whole pipeline (see the module docs). Stage reports are returned
/// on success; on failure the error names the stage.
pub async fn run_meeting(
    opts: &MeetingOptions,
    mut backends: Backends,
    cancel: CancellationToken,
) -> Result<MeetingOutcome, MeetingError> {
    let cfg = &opts.config;
    let graph = StageGraph::new(meeting_mode_stage_decls())?;
    let mut selection = opts.selection.clone();
    let plan = graph.plan(&selection)?;
    // Render writes files outside the run directory: rerun it whenever selected.
    if matches!(plan.decision("render"), Some(StageDecision::Run { .. })) {
        selection.force.insert("render".into());
    }
    let needs = Needs::from_plan(&plan);
    let tools: ToolVersions = preflight::check(&needs, &backends, &opts.ffmpeg, &opts.ffprobe)
        .map_err(MeetingError::Preflight)?;
    let video = std::path::absolute(&opts.video).map_err(setup)?;
    let out_dir = std::path::absolute(&opts.out_dir).map_err(setup)?;
    fs_err::create_dir_all(&out_dir).map_err(setup)?;
    if let Some(log) = &opts.log {
        log.activate(&out_dir).map_err(setup)?;
    }

    let run = glassrip_core::manifest::RunDir::open(
        &out_dir,
        &opts.run_id,
        Producer::glassrip(
            env!("CARGO_PKG_VERSION"),
            option_env!("GLASSRIP_GIT_SHA").map(str::to_string),
        ),
    )?;
    let root = run.root().to_path_buf();
    let mut ropts = RunnerOptions::from_config(&cfg.runner);
    ropts.tool_versions = BTreeMap::from([(
        "glassrip".to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
    )]);
    let mut runner = Runner::new(
        run,
        graph,
        &selection,
        Cache::new(&opts.cache_dir),
        ropts,
        cancel.clone(),
    )?;
    let video_hash = {
        let v = video.clone();
        tokio::task::spawn_blocking(move || glassrip_core::blake3_file(&v))
            .await
            .map_err(setup)?
            .map_err(|e| setup(format!("cannot read {}: {e}", video.display())))?
    };
    let tool_map: BTreeMap<String, String> =
        [("ffmpeg", &tools.ffmpeg), ("ffprobe", &tools.ffprobe)]
            .into_iter()
            .filter_map(|(k, v)| v.clone().map(|v| (k.to_string(), v)))
            .collect();
    runner.run_dir_mut().update(|m| {
        m.inputs.retain(|i| i.path != video.display().to_string());
        m.inputs.push(InputRef {
            path: video.display().to_string(),
            blake3: video_hash.clone(),
            schema: None,
            schema_version: None,
        });
        m.tool_versions.extend(tool_map.clone());
        m.model_digests.extend(backends.model_digests.clone());
        // The vision digest keys the model stages' caches; an offline rerun reads it back.
        if let Ok(v) = &backends.vision {
            if let Some(d) = &v.digest {
                m.model_digests.insert(v.model.clone(), d.clone());
            }
        }
    })?;

    let reports = Reports::default();
    let mut outcome = MeetingOutcome::default();
    let lap = |outcome: &mut MeetingOutcome, name: &str, started: Instant| {
        outcome
            .phase_wall_s
            .insert(name.to_string(), started.elapsed().as_secs_f64());
    };
    let blobs = BlobStore::new(&opts.blobs_dir);
    let participants = &opts.participants;
    let layout = LayoutConfig {
        participants: participants.clone(),
        ..LayoutConfig::default()
    };

    // ---- media (CPU)
    let t = Instant::now();
    {
        let ffmpeg_v = tools.ffmpeg.clone().unwrap_or_default();
        let ffprobe_v = tools.ffprobe.clone().unwrap_or_default();
        let scoring = ScoringParams::from_config(&cfg.features);
        let probe = ProbeStage::new(video.clone(), opts.ffprobe.clone(), ffprobe_v.clone());
        let orient = OrientStage::new(
            OrientParams {
                sample_frames: cfg.orient.sample_frames,
                override_rotation_deg: cfg.orient.override_rotation_deg,
                ..OrientParams::default()
            },
            opts.ffmpeg.clone(),
            ffmpeg_v.clone(),
            opts.media_models_dir.clone(),
            opts.allow_model_download,
        );
        let frames = FramesStage::new(
            FramesParams {
                interval_s: cfg.frames.interval_s,
                scale_width: cfg.frames.scale_width,
                hwaccel: cfg.frames.hwaccel,
                ..FramesParams::default()
            },
            opts.ffmpeg.clone(),
            opts.ffprobe.clone(),
            BTreeMap::from([
                ("ffmpeg".to_string(), ffmpeg_v),
                ("ffprobe".to_string(), ffprobe_v),
            ]),
            root.clone(),
            blobs.clone(),
        );
        let quads = ScreenQuadStage::new(QuadParams::default(), root.clone());
        let features = FeaturesStage::new(scoring.clone(), root.clone()).map_err(setup)?;
        let keyframes = KeyframesStage::new(
            KeyframesParams::from_config(&cfg.keyframes, scoring.clone()),
            root.clone(),
        )
        .map_err(setup)?;
        let rectify = RectifyStage::new(
            RectifyParams::with_scoring(scoring),
            root.clone(),
            blobs.clone(),
        )
        .map_err(setup)?;

        step(&runner, &probe, &reports).await?;
        step(&runner, &orient, &reports).await?;
        if step(&runner, &frames, &reports).await? != StageStatus::Skipped {
            materialize::<FrameRecord, _>(&root, &blobs, FRAMES, "frames", |f| {
                (f.path.clone(), f.blake3.clone())
            })
            .map_err(setup)?;
        }
        step(&runner, &quads, &reports).await?;
        step(&runner, &features, &reports).await?;
        step(&runner, &keyframes, &reports).await?;
        if step(&runner, &rectify, &reports).await? != StageStatus::Skipped {
            materialize::<RectifiedKeyframe, _>(
                &root,
                &blobs,
                RECTIFIED_KEYFRAMES,
                "rectify",
                |k| (k.path.clone(), k.blake3.clone()),
            )
            .map_err(setup)?;
        }
    }
    lap(&mut outcome, "media", t);

    // The vision stages exist only with a vision backend; preflight has already
    // failed when a selected stage needs one and it is missing.
    let vision_model = backends.vision.as_ref().ok().map(|v| v.model.clone());
    let vision_offline_reason = backends
        .vision
        .as_ref()
        .ok()
        .and_then(|v| v.offline.clone());
    let vision_offline = vision_offline_reason.as_deref();
    let vision = match backends.vision.as_ref() {
        Ok(v) => Some(vision_stages(v.clone(), cfg)?),
        Err(_) => None,
    };

    // ---- phase A: OCR sessions and the vision model together.
    let t = Instant::now();
    {
        let recognizer =
            std::mem::replace(&mut backends.ocr, Err("released after GPU phase A".into()));
        if let Ok(r) = recognizer {
            let ocr = OcrHarvestStage::new(r, OcrConfig::default(), layout.clone());
            step(&runner, &ocr, &reports).await?;
            // `ocr` and its recognizer drop here: the ONNX sessions are freed
            // before phase B (spec 5.3).
        }
        step(
            &runner,
            &VocabularyStage::new(VocabularyParams {
                participants: opts.participants.clone(),
                ..VocabularyParams::default()
            }),
            &reports,
        )
        .await?;
        if let Some(v) = &vision {
            model_step(&runner, &v.classify, &v.monitor, vision_offline, &reports).await?;
        }
    }
    lap(&mut outcome, "phase_a", t);

    // ---- phase B: board reading next to the audio branch.
    let t = Instant::now();
    let threshold = cfg.gpu.sequential_model_threshold_gb;
    let concurrent = match &backends.vision {
        Ok(v) => {
            let loaded = match v.probe.placement().await {
                Ok(Some(p)) => Some(p.size_bytes),
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!(error = %e, "vision placement unknown before phase B");
                    None
                }
            };
            let concurrent = phase_b_concurrent(loaded, v.estimated_gb(), threshold);
            if !concurrent {
                tracing::info!(
                    loaded_bytes = loaded,
                    estimated_gb = v.estimated_gb(),
                    threshold_gb = threshold,
                    "vision model large or of unknown size: phase B runs board reading and audio one after the other"
                );
            }
            concurrent
        }
        // No vision model: nothing on the GPU competes with ASR.
        Err(_) => true,
    };
    outcome.phase_b_concurrent = concurrent;
    {
        let canvas = CanvasCropStage::new(CanvasCropParams {
            layout: layout.clone(),
            ..CanvasCropParams::default()
        });
        let validate = BoardValidateStage::new({
            let mut p = BoardValidateParams::default();
            p.validation.participant_names = participants.clone();
            p
        });
        let edges = EdgeDirectionStage::new(
            PixelCheckParams::default(),
            VlmCheckParams::default(),
            vision.as_ref().map(|v| v.client.clone()),
        );
        let vision_chain = async {
            step(&runner, &canvas, &reports).await?;
            if let Some(v) = &vision {
                model_step(&runner, &v.board_read, &v.monitor, vision_offline, &reports).await?;
            }
            step(&runner, &validate, &reports).await?;
            require_cached(&runner, &edges, vision_offline)?;
            step(&runner, &edges, &reports).await?;
            Ok::<(), MeetingError>(())
        };

        let asr_engine = backends.asr.as_ref().ok().cloned();
        let diarizer = backends.diarize.as_ref().ok().cloned();
        let extract = AudioExtractStage::new(
            video.clone(),
            opts.blobs_dir.join("audio"),
            ExtractOptions {
                ffmpeg: PathBuf::from(&opts.ffmpeg),
                ffprobe: PathBuf::from(&opts.ffprobe),
            },
            BTreeMap::from([
                (
                    "ffmpeg".to_string(),
                    tools.ffmpeg.clone().unwrap_or_default(),
                ),
                (
                    "ffprobe".to_string(),
                    tools.ffprobe.clone().unwrap_or_default(),
                ),
            ]),
        );
        let asr = asr_engine.map(|e| AsrStage::new(e, MAX_VOCABULARY_TERMS));
        let diarize = diarizer.clone().map(DiarizeStage::new);
        let gap_fill = GapFillStage::new(Some(GapFillConfig::default()), diarizer);
        let assign = AssignWordsStage::new(AssignWordsParams {
            assign: glassrip_audio::assign::AssignConfig {
                max_gap_s: cfg.audio.nearest_turn_s,
                ..glassrip_audio::assign::AssignConfig::default()
            },
            ..AssignWordsParams::default()
        });
        // ASR next to board reading: poll placement and hold vision requests
        // on a spill until ASR finishes.
        let guard = vision
            .as_ref()
            .filter(|_| concurrent)
            .map(|v| Arc::clone(&v.monitor));
        let audio_chain = async {
            step(&runner, &extract, &reports).await?;
            let (a, d) = tokio::join!(
                async {
                    match (&asr, &guard) {
                        (Some(s), Some(m)) => m
                            .guard_asr(step(&runner, s, &reports), m.config().poll_interval)
                            .await
                            .map(|_| ()),
                        (Some(s), None) => step(&runner, s, &reports).await.map(|_| ()),
                        (None, _) => Ok(()),
                    }
                },
                async {
                    match &diarize {
                        Some(s) => step(&runner, s, &reports).await.map(|_| ()),
                        None => Ok(()),
                    }
                }
            );
            a?;
            d?;
            step(&runner, &gap_fill, &reports).await?;
            step(&runner, &assign, &reports).await?;
            Ok::<(), MeetingError>(())
        };
        if concurrent {
            let (v, a) = tokio::join!(vision_chain, audio_chain);
            v?;
            a?;
        } else {
            vision_chain.await?;
            audio_chain.await?;
        }
    }
    lap(&mut outcome, "phase_b", t);

    // ---- join (CPU): board state and speaker names.
    let t = Instant::now();
    {
        let bs = &cfg.board_state;
        let board_state = BoardStateStage::new(ConsolidationParams {
            participants: alias_table(participants),
            fuzzy_threshold: bs.fuzzy_ratio,
            min_support_keyframes: bs.min_support_keyframes as usize,
            min_support_density: bs.min_support_fraction,
            owner_confirm_keyframes: bs.owner_min_keyframes as usize,
            ..ConsolidationParams::default()
        });
        step(&runner, &board_state, &reports).await?;
        let speakers = name_speakers_stage(participants, &video).await;
        step(&runner, &speakers, &reports).await?;
    }
    lap(&mut outcome, "join", t);

    // ---- phase C: the vision stages (and their client) are released; the
    // notes stage unloads the vision model (keep_alive 0) before loading the
    // text model.
    let t = Instant::now();
    drop(vision);
    let text = std::mem::replace(&mut backends.text, Err("released".into()));
    // Why the text model cannot be asked (missing, or its server not used).
    let (text, unavailable): (Arc<dyn TextBackend>, Option<String>) = match text {
        Ok(t) => (t, backends.text_offline.clone()),
        Err(reason) => (Arc::new(NoTextModel(reason.clone())), Some(reason)),
    };
    {
        // The text model's digest keys the notes cache when the notes need the
        // model (speech): the digest resolved when connecting (live, or recorded
        // by an earlier run for an offline rerun, the same pinned-digest policy
        // as the vision stages), else asked of the server and recorded so an
        // offline rerun can key the same entry. Without speech the notes ask no
        // model, so the key names none.
        let speech = transcript_has_speech(&runner);
        let text_model = cfg.models.text.clone();
        let text_digest = if !speech {
            None
        } else if let Some(d) = backends.model_digests.get(&text_model) {
            Some(d.clone())
        } else if unavailable.is_none() {
            let d = text.digest(&text_model).await.ok().flatten();
            if let Some(d) = &d {
                let (model, d) = (text_model.clone(), d.clone());
                runner.run_dir_mut().update(|m| {
                    m.model_digests.insert(model, d);
                })?;
            }
            d
        } else {
            None
        };
        let notes = NotesStage::new(
            NotesParams {
                text_model: cfg.models.text.clone(),
                vision_model: vision_model.clone(),
                ollama: OllamaTextConfig {
                    base_url: cfg.ollama.host.clone(),
                    ..OllamaTextConfig::default()
                },
                allow_spill: cfg.gpu.allow_spill,
                max_drop_rate: cfg.notes.max_drop_fraction,
                alarm_min_transcript_s: cfg.notes.min_transcript_s,
                ..NotesParams::default()
            },
            text,
        )
        .with_text_digest(text_digest);
        let selected = matches!(
            runner.plan().decision("notes"),
            Some(StageDecision::Run { .. })
        );
        match &unavailable {
            // Without speech the notes come from the board alone and ask no
            // model, so the text model is not needed.
            Some(reason) if selected && !speech => tracing::info!(
                %reason,
                "no speech was transcribed: notes come from the board alone, no text model needed"
            ),
            _ => require_cached(&runner, &notes, unavailable.as_deref())?,
        }
        step(&runner, &notes, &reports).await?;
    }
    lap(&mut outcome, "phase_c", t);

    // ---- render.
    let t = Instant::now();
    let render = RenderStage::new(RenderParams {
        out_dir: out_dir.clone(),
        stem: opts.stem.clone(),
        meta: MarkdownMeta {
            date: None,
            source: video.file_name().map(|n| n.to_string_lossy().into_owned()),
            media_duration_s: media_duration_s(&runner),
        },
        ..RenderParams::default()
    });
    if step(&runner, &render, &reports).await? != StageStatus::Skipped {
        let path = runner.run_dir().artifact_path(RENDER_SCHEMA);
        let result = glassrip_core::jsonl::read::<Record<RenderResult>>(
            &path,
            &SchemaReq::new(RENDER_SCHEMA, 1),
        )
        .map_err(setup)?
        .items
        .into_iter()
        .find_map(|r| r.outcome.result);
        if let Some(r) = result {
            outcome.outputs = r.files.iter().map(|f| out_dir.join(&f.name)).collect();
        }
    }
    lap(&mut outcome, "render", t);

    // Graph order, whatever order concurrent chains finished in.
    let position: BTreeMap<&str, usize> = runner
        .plan()
        .order
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let mut all = reports.take();
    all.sort_by_key(|r| {
        position
            .get(r.stage.as_str())
            .copied()
            .unwrap_or(usize::MAX)
    });
    outcome.reports = all;
    let ran: BTreeSet<&str> = outcome.reports.iter().map(|r| r.stage.as_str()).collect();
    tracing::info!(
        stages = ran.len(),
        outputs = outcome.outputs.len(),
        "meeting run finished"
    );
    Ok(outcome)
}

#[cfg(feature = "ocr")]
async fn name_speakers_stage(participants: &[String], video: &Path) -> NameSpeakersStage {
    use glassrip_notes::speakers::frames::FfmpegFrameSource;
    use glassrip_notes::speakers::ocr::{OcrParams, PpOcrReader};
    let base = NameSpeakersParams {
        participants: participants.to_vec(),
        ..NameSpeakersParams::default()
    };
    let models = glassrip_ocr::models::default_dir();
    let frames = FfmpegFrameSource::open(video, 1920, true).await;
    let reader = PpOcrReader::new(&models, OcrParams::default());
    match (frames, reader) {
        (Ok(f), Ok(r)) => NameSpeakersStage::new(NameSpeakersParams {
            video: Some(video.to_path_buf()),
            ..base
        })
        .with_visual(Arc::new(f), Arc::new(r)),
        (f, r) => {
            tracing::warn!(
                frames = ?f.err(),
                ocr = ?r.err(),
                "visual speaker cues unavailable; naming speakers from the transcript"
            );
            NameSpeakersStage::new(base)
        }
    }
}

#[cfg(not(feature = "ocr"))]
async fn name_speakers_stage(participants: &[String], _video: &Path) -> NameSpeakersStage {
    NameSpeakersStage::new(NameSpeakersParams {
        participants: participants.to_vec(),
        ..NameSpeakersParams::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_b_is_sequential_unless_the_model_is_known_small() {
        let gb = |g: f64| Some((g * 1e9) as u64);
        assert!(phase_b_concurrent(gb(9.0), None, 14.0));
        assert!(
            !phase_b_concurrent(gb(20.0), Some(5.0), 14.0),
            "loaded size wins"
        );
        assert!(phase_b_concurrent(None, Some(6.0), 14.0));
        assert!(!phase_b_concurrent(None, Some(21.0), 14.0));
        assert!(
            !phase_b_concurrent(None, None, 14.0),
            "unknown size is sequential"
        );
    }

    #[test]
    fn participants_get_slugs_and_name_aliases() {
        let t = alias_table(&["Ada Quill".into(), "Bo".into()]);
        assert_eq!(
            t.resolve("Ada").map(|p| p.person_id.as_str()),
            Some("ada-quill")
        );
        assert_eq!(
            t.resolve("Quill").map(|p| p.person_id.as_str()),
            Some("ada-quill")
        );
        assert_eq!(t.resolve("Bo").map(|p| p.person_id.as_str()), Some("bo"));
        assert!(t.resolve("Zed").is_none());
    }

    #[tokio::test]
    async fn preflight_failure_runs_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let video = dir.path().join("v.mp4");
        std::fs::write(&video, b"x").unwrap();
        let out = dir.path().join("out");
        let opts = MeetingOptions::new(video, out.clone(), dir.path(), Config::default());
        let err = run_meeting(
            &opts,
            Backends::none("absent in this test"),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
        match err {
            MeetingError::Preflight(p) => {
                assert!(p.iter().any(|m| m.starts_with("asr: absent")), "{p:?}");
                // the text model is checked after transcription, not here
                assert!(!p.iter().any(|m| m.starts_with("notes")), "{p:?}");
            }
            other => panic!("expected a preflight error, got {other}"),
        }
        assert!(!out.exists(), "nothing may run when preflight fails");
    }
}
