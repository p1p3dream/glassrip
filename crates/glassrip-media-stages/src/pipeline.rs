//! Runs the media stages (`probe` through `rectify`) with the core runner.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glassrip_core::cache::Cache;
use glassrip_core::config::Config;
use glassrip_core::envelope::{Producer, Record, SchemaReq};
use glassrip_core::graph::{GraphError, Selection, StageGraph, meeting_mode_stage_decls};
use glassrip_core::jsonl::{self, JsonlError};
use glassrip_core::manifest::{ManifestError, RunDir, StageStatus};
use glassrip_core::runner::{Runner, RunnerError, RunnerOptions, Stage, StageReport};
use tokio_util::sync::CancellationToken;

use crate::blobs::{BlobError, BlobStore};
use crate::features::FeaturesStage;
use crate::frames::{FramesParams, FramesStage, SamplingMode};
use crate::keyframes::{KeyframesParams, KeyframesStage};
use crate::orient::{OrientParams, OrientStage};
use crate::probe::ProbeStage;
use crate::quad::{QuadParams, ScreenQuadStage};
use crate::rectify::{RectifyParams, RectifyStage};
use crate::schema::{FRAMES, FrameRecord, RECTIFIED_KEYFRAMES, RectifiedKeyframe};
use crate::scoring::ScoringParams;

/// The media stages in execution order.
pub const MEDIA_STAGES: [&str; 7] = [
    "probe",
    "orient",
    "frames",
    "screen_quad",
    "features",
    "keyframes",
    "rectify",
];

/// Error running the media stages.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// Stage runner failure (includes cancellation).
    #[error(transparent)]
    Runner(Box<RunnerError>),
    /// Run directory failure.
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    /// Stage graph failure.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Blob store failure.
    #[error(transparent)]
    Blob(#[from] BlobError),
    /// Artifact read failure.
    #[error(transparent)]
    Jsonl(#[from] JsonlError),
    /// Invalid setup (tools missing, bad parameters).
    #[error("{0}")]
    Setup(String),
}

impl From<RunnerError> for PipelineError {
    fn from(e: RunnerError) -> Self {
        Self::Runner(Box::new(e))
    }
}

/// Everything a media run needs.
#[derive(Debug, Clone)]
pub struct MediaRunOptions {
    /// Input video.
    pub video: PathBuf,
    /// Run directory.
    pub out_dir: PathBuf,
    /// Run id recorded in artifacts.
    pub run_id: String,
    /// Stage cache root.
    pub cache_dir: PathBuf,
    /// Blob store root.
    pub blobs_dir: PathBuf,
    /// Models directory.
    pub models_dir: PathBuf,
    /// Download missing models (pinned URL + SHA-256).
    pub allow_model_download: bool,
    /// Configuration (`glassrip.toml`).
    pub config: Config,
    /// Stage selection.
    pub selection: Selection,
    /// Orientation parameters.
    pub orient: OrientParams,
    /// Frame sampling parameters beyond the config (sampling rule, chunking).
    pub sampling: SamplingMode,
    /// Chunk length for frame decoding.
    pub chunk_s: f64,
    /// Parallel ffmpeg processes for frame decoding.
    pub chunk_concurrency: u32,
    /// `ffmpeg` binary.
    pub ffmpeg: String,
    /// `ffprobe` binary.
    pub ffprobe: String,
}

impl MediaRunOptions {
    /// Defaults for a video and run directory: cache and blobs under `<workspace>/.glassrip`.
    pub fn new(video: PathBuf, out_dir: PathBuf, workspace: &Path) -> Self {
        let f = FramesParams::default();
        Self {
            run_id: out_dir
                .file_name()
                .map_or_else(|| "run".to_string(), |n| n.to_string_lossy().into_owned()),
            video,
            out_dir,
            cache_dir: workspace.join(glassrip_core::cache::DEFAULT_CACHE_DIR),
            blobs_dir: workspace.join(".glassrip/blobs"),
            models_dir: crate::models::default_dir(),
            allow_model_download: true,
            config: Config::default(),
            selection: Selection {
                until: Some("rectify".into()),
                ..Selection::default()
            },
            orient: OrientParams::default(),
            sampling: f.sampling,
            chunk_s: f.chunk_s,
            chunk_concurrency: f.chunk_concurrency,
            ffmpeg: "ffmpeg".into(),
            ffprobe: "ffprobe".into(),
        }
    }
}

fn materialize<T, F>(
    run: &Path,
    blobs: &BlobStore,
    schema: &str,
    stage: &str,
    refs: F,
) -> Result<usize, PipelineError>
where
    T: serde::de::DeserializeOwned,
    F: Fn(&T) -> (String, String),
    T: Send,
{
    let path = run.join(RunDir::artifact_rel_path(schema));
    let items = jsonl::read::<Record<T>>(&path, &SchemaReq::new(schema, 1))?.items;
    use rayon::prelude::*;
    let refs: Vec<(String, String)> = items
        .into_iter()
        .filter_map(|r| r.outcome.result)
        .map(|v| refs(&v))
        .collect();
    refs.par_iter()
        .try_for_each(|(rel, hash)| blobs.materialize(run, rel, hash, stage).map(|_| ()))?;
    Ok(refs.len())
}

async fn step<S: Stage>(
    runner: &mut Runner,
    stage: &S,
    reports: &mut Vec<StageReport>,
) -> Result<StageStatus, PipelineError> {
    let rep = runner.run_stage(stage).await?;
    tracing::info!(stage = %rep.stage, status = ?rep.status, items = rep.items_total, errors = rep.items_error, wall_s = rep.wall_s, "stage done");
    let status = rep.status;
    reports.push(rep);
    Ok(status)
}

/// Runs the media stages, appending one report per stage to `reports` (also on error).
pub async fn run_media_stages(
    opts: &MediaRunOptions,
    cancel: CancellationToken,
    reports: &mut Vec<StageReport>,
) -> Result<(), PipelineError> {
    let ffmpeg_v = crate::util::tool_version(&opts.ffmpeg)
        .map_err(|e| PipelineError::Setup(format!("ffmpeg is required: {e}")))?;
    let ffprobe_v = crate::util::tool_version(&opts.ffprobe)
        .map_err(|e| PipelineError::Setup(format!("ffprobe is required: {e}")))?;
    let video = std::path::absolute(&opts.video)
        .map_err(|e| PipelineError::Setup(format!("bad video path: {e}")))?;
    let run = RunDir::open(
        &opts.out_dir,
        &opts.run_id,
        Producer::glassrip(env!("CARGO_PKG_VERSION"), None),
    )?;
    let root = run.root().to_path_buf();
    let graph = StageGraph::new(meeting_mode_stage_decls())?;
    let mut ropts = RunnerOptions::from_config(&opts.config.runner);
    ropts.tool_versions = BTreeMap::from([(
        "glassrip-media-stages".to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
    )]);
    let mut runner = Runner::new(
        run,
        graph,
        &opts.selection,
        Cache::new(&opts.cache_dir),
        ropts,
        cancel,
    )?;
    let blobs = BlobStore::new(&opts.blobs_dir);
    let cfg = &opts.config;
    let scoring = ScoringParams::from_config(&cfg.features);
    let setup = |e: String| PipelineError::Setup(e);

    let probe = ProbeStage::new(video, opts.ffprobe.clone(), ffprobe_v.clone());
    let mut orient_params = opts.orient.clone();
    orient_params.sample_frames = cfg.orient.sample_frames;
    let orient = OrientStage::new(
        orient_params,
        opts.ffmpeg.clone(),
        ffmpeg_v.clone(),
        opts.models_dir.clone(),
        opts.allow_model_download,
    );
    let frames = FramesStage::new(
        FramesParams {
            interval_s: cfg.frames.interval_s,
            scale_width: cfg.frames.scale_width,
            hwaccel: cfg.frames.hwaccel,
            sampling: opts.sampling,
            chunk_s: opts.chunk_s,
            chunk_concurrency: opts.chunk_concurrency,
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

    step(&mut runner, &probe, reports).await?;
    step(&mut runner, &orient, reports).await?;
    if step(&mut runner, &frames, reports).await? != StageStatus::Skipped {
        let n = materialize::<FrameRecord, _>(&root, &blobs, FRAMES, "frames", |f| {
            (f.path.clone(), f.blake3.clone())
        })?;
        tracing::info!(files = n, "frames materialized");
    }
    step(&mut runner, &quads, reports).await?;
    step(&mut runner, &features, reports).await?;
    step(&mut runner, &keyframes, reports).await?;
    if step(&mut runner, &rectify, reports).await? != StageStatus::Skipped {
        let n = materialize::<RectifiedKeyframe, _>(
            &root,
            &blobs,
            RECTIFIED_KEYFRAMES,
            "rectify",
            |k| (k.path.clone(), k.blake3.clone()),
        )?;
        tracing::info!(files = n, "keyframe images materialized");
    }
    Ok(())
}
