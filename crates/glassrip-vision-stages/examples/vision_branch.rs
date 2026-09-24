//! Run the vision branch (ocr_harvest through board_validate) on a directory of
//! keyframes described by an index file, live against Ollama or replayed from a
//! raw response store.
//!
//! ```sh
//! cargo run --release --features cuda --example vision_branch -- \
//!     --index input/keyframes.json --frames-dir input/frames \
//!     --run-dir runs/r1 --raw-store runs/raw --host http://localhost:11434
//! ```
//!
//! OCR needs the `onnx` (CPU) or `cuda` feature and the PP-OCRv5 files in
//! `~/.glassrip/models/ppocrv5/` (or `--ocr-models`).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use glassrip_core::cache::Cache;
use glassrip_core::envelope::Producer;
use glassrip_core::graph::Selection;
use glassrip_core::manifest::RunDir;
use glassrip_core::runner::{Runner, RunnerOptions};
use glassrip_ocr::{OcrConfig, TextRecognizer};
use glassrip_vision::{OllamaBackend, OllamaConfig, VisionBackend, VisionClient};
use glassrip_vision_stages::layout::LayoutConfig;
use glassrip_vision_stages::pipeline::{self, VisionBranch};
use glassrip_vision_stages::placement::{
    MonitorConfig, PlacementMonitor, PlacementProbe, StaticProbe,
};
use glassrip_vision_stages::raw_store::{RawStore, RecordingBackend, ReplayBackend};
use glassrip_vision_stages::stages::canvas_crop::CanvasCropParams;
use glassrip_vision_stages::{
    adapter, BoardReadParams, BoardReadStage, BoardValidateParams, BoardValidateStage,
    CanvasCropStage, ClassifyParams, ClassifyStage, OcrHarvestStage, VocabularyParams,
    VocabularyStage,
};
use tokio_util::sync::CancellationToken;

#[derive(Parser, Debug)]
struct Args {
    /// Keyframe index (JSON with `keyframes: [{file, t_start, t_end, t_rep}]`).
    #[arg(long)]
    index: PathBuf,
    /// Sampled frames `t_NNNNNN.jpg` for the canvas variance tiebreak.
    #[arg(long)]
    frames_dir: Option<PathBuf>,
    #[arg(long)]
    run_dir: PathBuf,
    /// Raw model responses (recorded live, read in replay).
    #[arg(long)]
    raw_store: PathBuf,
    /// Stage cache directory (default: `<run-dir>/../cache`).
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    #[arg(long, default_value = "http://localhost:11434")]
    host: String,
    #[arg(long, default_value = "qwen2.5vl:7b")]
    model: String,
    /// Server slots (client concurrency).
    #[arg(long, default_value_t = 4)]
    slots: usize,
    /// Serve model answers from the raw store only.
    #[arg(long)]
    replay: bool,
    #[arg(long)]
    ocr_models: Option<PathBuf>,
    /// Participant names (optional layout hint).
    #[arg(long, value_delimiter = ',')]
    participants: Vec<String>,
    /// Stop after this stage.
    #[arg(long)]
    until: Option<String>,
    /// Write a JSON run report here.
    #[arg(long)]
    report: Option<PathBuf>,
}

/// Backend for requests, placement probe, model digest, server version.
type ModelSetup = (
    Arc<dyn VisionBackend>,
    Arc<dyn PlacementProbe>,
    Option<String>,
    Option<String>,
);

#[cfg(feature = "onnx")]
fn recognizer(args: &Args, cfg: &OcrConfig) -> Result<Arc<dyn TextRecognizer>, String> {
    let dir = args
        .ocr_models
        .clone()
        .unwrap_or_else(glassrip_ocr::models::default_dir);
    glassrip_ocr::engine::PpOcrEngine::new(&dir, cfg.clone())
        .map(|e| Arc::new(e) as Arc<dyn TextRecognizer>)
        .map_err(|e| e.to_string())
}

#[cfg(not(feature = "onnx"))]
fn recognizer(_args: &Args, _cfg: &OcrConfig) -> Result<Arc<dyn TextRecognizer>, String> {
    Err("OCR needs the `onnx` or `cuda` feature".into())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    let run_id = format!(
        "vision-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let summary = adapter::build_inputs(
        &args.run_dir,
        &args.index,
        args.frames_dir.as_deref(),
        &run_id,
    )?;
    tracing::info!(?summary, "adapter wrote upstream artifacts");

    let cancel = CancellationToken::new();
    {
        let c = cancel.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                c.cancel();
            }
        });
    }
    let run = RunDir::open(
        &args.run_dir,
        &run_id,
        Producer::glassrip(env!("CARGO_PKG_VERSION"), None),
    )?;
    let cache_dir = args
        .cache_dir
        .clone()
        .unwrap_or_else(|| args.run_dir.join("..").join("cache"));
    let mut runner = Runner::new(
        run,
        pipeline::graph()?,
        &Selection::default(),
        Cache::new(cache_dir),
        RunnerOptions::default(),
        cancel.clone(),
    )?;

    let ocr_cfg = OcrConfig::default();
    let recognizer = recognizer(&args, &ocr_cfg)?;
    tracing::info!(provider = %recognizer.execution_provider(), "OCR ready");

    let store = RawStore::new(&args.raw_store);
    let (backend, probe, digest, server_version): ModelSetup = if args.replay {
        (
            Arc::new(ReplayBackend::new(&args.model, store.clone())),
            Arc::new(StaticProbe),
            None,
            None,
        )
    } else {
        let mut cfg = OllamaConfig::new(&args.host, &args.model, 8192);
        cfg.slots = args.slots;
        cfg.request_timeout = Duration::from_secs(180);
        let ollama = Arc::new(OllamaBackend::new(cfg)?);
        let digest = ollama.resolve_digest().await?;
        let version = ollama.server_version().await.ok();
        let self_test = ollama.self_test(cancel.clone()).await?;
        tracing::info!(
            latency_s = self_test.latency.as_secs_f64(),
            "model self-test passed"
        );
        (
            Arc::new(RecordingBackend::new(ollama.clone(), store.clone())),
            ollama,
            Some(digest),
            version,
        )
    };
    let client = VisionClient::new(backend, args.slots)?;
    let monitor = Arc::new(PlacementMonitor::new(
        probe,
        client,
        MonitorConfig::default(),
    ));

    let layout = LayoutConfig {
        participants: args.participants.clone(),
        ..LayoutConfig::default()
    };
    let branch = VisionBranch {
        ocr: OcrHarvestStage::new(recognizer, ocr_cfg, layout.clone()),
        vocabulary: VocabularyStage::new(VocabularyParams::default()),
        classify: ClassifyStage::new(
            ClassifyParams::default(),
            Arc::clone(&monitor),
            &args.model,
            digest.clone(),
            server_version.clone(),
        ),
        canvas: CanvasCropStage::new(CanvasCropParams {
            layout,
            ..CanvasCropParams::default()
        }),
        board_read: BoardReadStage::new(
            BoardReadParams::default(),
            Arc::clone(&monitor),
            &args.model,
            digest,
            server_version,
        ),
        board_validate: BoardValidateStage::new(BoardValidateParams::default()),
    };
    let reports = branch.run(&mut runner, args.until.as_deref()).await?;

    let stage_rows: Vec<serde_json::Value> = reports
        .iter()
        .map(|r| {
            serde_json::json!({
                "stage": r.stage,
                "status": format!("{:?}", r.status),
                "items_total": r.items_total,
                "items_ok": r.items_ok,
                "items_error": r.items_error,
                "items_processed": r.items_processed,
                "wall_s": r.wall_s,
            })
        })
        .collect();
    let report = serde_json::json!({
        "run_id": run_id,
        "adapter": summary,
        "stages": stage_rows,
        "requests_completed": monitor.completed(),
        "placement_checks": monitor.checks(),
    });
    let text = serde_json::to_string_pretty(&report)?;
    match &args.report {
        Some(p) => fs_err::write(p, text)?,
        None => println!("{text}"),
    }
    Ok(())
}
