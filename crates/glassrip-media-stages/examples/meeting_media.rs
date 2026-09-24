//! Dev entry point for the media stages of `glassrip meeting` (probe through rectify).
//!
//! ```text
//! cargo run --release -p glassrip-media-stages --example meeting_media -- VIDEO \
//!     --out RUN_DIR [--workspace DIR] [--until-stage rectify] [--force-stage S]... \
//!     [--features-mode production|prototype_compat] [--sampling auto|grid|sync] \
//!     [--hwaccel none|auto] [--orient-override DEG] [--report FILE]
//! ```
//!
//! Kept thin on purpose: the `glassrip meeting` subcommand will call
//! `glassrip_media_stages::pipeline::run_media_stages` the same way.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use glassrip_core::config::{Config, FeaturesMode, HwAccel};
use glassrip_media_stages::frames::SamplingMode;
use glassrip_media_stages::pipeline::{MediaRunOptions, run_media_stages};
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
struct Args {
    /// Input video.
    video: PathBuf,
    /// Run directory.
    #[arg(long)]
    out: PathBuf,
    /// Workspace holding `.glassrip/cache` and `.glassrip/blobs`.
    #[arg(long, default_value = ".")]
    workspace: PathBuf,
    /// `glassrip.toml`.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Start at this stage (earlier artifacts must exist in the run directory).
    #[arg(long)]
    from_stage: Option<String>,
    /// Stop after this stage.
    #[arg(long, default_value = "rectify")]
    until_stage: String,
    /// Bypass the cache for a stage (repeatable).
    #[arg(long)]
    force_stage: Vec<String>,
    /// Features mode (overrides the config).
    #[arg(long, value_parser = ["production", "prototype_compat"])]
    features_mode: Option<String>,
    /// Frame selection rule.
    #[arg(long, default_value = "auto", value_parser = ["auto", "grid", "sync"])]
    sampling: String,
    /// Decoder (overrides the config).
    #[arg(long, value_parser = ["none", "auto"])]
    hwaccel: Option<String>,
    /// Sampling interval in seconds (overrides the config).
    #[arg(long)]
    interval: Option<f64>,
    /// Skip the orientation model and apply this clockwise rotation.
    #[arg(long)]
    orient_override: Option<u32>,
    /// Chunk length for frame decoding, seconds.
    #[arg(long, default_value_t = 240.0)]
    chunk_s: f64,
    /// Parallel ffmpeg processes.
    #[arg(long, default_value_t = 8)]
    chunk_concurrency: u32,
    /// Do not download missing models.
    #[arg(long)]
    no_download: bool,
    /// Write a JSON summary here.
    #[arg(long)]
    report: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let a = Args::parse();
    let mut config = match &a.config {
        Some(p) => Config::load(p).with_context(|| format!("loading {}", p.display()))?,
        None => Config::default(),
    };
    match a.features_mode.as_deref() {
        Some("production") => config.features.mode = FeaturesMode::Production,
        Some("prototype_compat") => config.features.mode = FeaturesMode::PrototypeCompat,
        _ => {}
    }
    match a.hwaccel.as_deref() {
        Some("auto") => config.frames.hwaccel = HwAccel::Auto,
        Some("none") => config.frames.hwaccel = HwAccel::None,
        _ => {}
    }
    if let Some(i) = a.interval {
        config.frames.interval_s = i;
    }
    let mut opts = MediaRunOptions::new(a.video.clone(), a.out.clone(), &a.workspace);
    opts.config = config;
    opts.selection.from = a.from_stage.clone();
    opts.selection.until = Some(a.until_stage.clone());
    opts.selection.force = a.force_stage.iter().cloned().collect();
    opts.orient.override_rotation_deg = a.orient_override;
    opts.allow_model_download = !a.no_download;
    opts.sampling = match a.sampling.as_str() {
        "grid" => SamplingMode::Grid,
        "sync" => SamplingMode::Sync,
        _ => SamplingMode::Auto,
    };
    opts.chunk_s = a.chunk_s;
    opts.chunk_concurrency = a.chunk_concurrency;

    let cancel = CancellationToken::new();
    let on_signal = cancel.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!("interrupt: stopping after in-flight items; rerun to resume");
            on_signal.cancel();
        }
    });

    let started = Instant::now();
    let mut reports = Vec::new();
    let result = run_media_stages(&opts, cancel, &mut reports).await;
    let total = started.elapsed().as_secs_f64();
    let rows: Vec<serde_json::Value> = reports
        .iter()
        .map(|r| {
            serde_json::json!({
                "stage": r.stage,
                "status": format!("{:?}", r.status).to_lowercase(),
                "items_total": r.items_total,
                "items_ok": r.items_ok,
                "items_error": r.items_error,
                "items_processed": r.items_processed,
                "wall_s": r.wall_s,
            })
        })
        .collect();
    for r in &reports {
        println!(
            "{:<12} {:<8} items {:>5} ok {:>5} err {:>3} processed {:>5} {:>8.2} s",
            r.stage,
            format!("{:?}", r.status).to_lowercase(),
            r.items_total,
            r.items_ok,
            r.items_error,
            r.items_processed,
            r.wall_s
        );
    }
    println!("total {total:.2} s");
    if let Some(p) = &a.report {
        let body = serde_json::json!({
            "stages": rows,
            "total_wall_s": total,
            "error": result.as_ref().err().map(ToString::to_string),
        });
        std::fs::write(p, serde_json::to_vec_pretty(&body)?)?;
    }
    result.map_err(anyhow::Error::from)
}
