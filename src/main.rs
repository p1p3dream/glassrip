#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "glassrip", about = "Extract code from video")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract code from a screencast
    Scrape {
        /// Path to video file
        video: PathBuf,

        /// Output directory
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// Ollama vision model
        #[arg(long, default_value = "qwen2.5vl:7b")]
        model: String,

        /// Ollama API base URL
        #[arg(long, default_value = "http://localhost:11434")]
        ollama_host: String,

        /// SSIM dedup threshold
        #[arg(long, default_value_t = 0.95)]
        ssim_threshold: f64,

        /// Use local Tesseract OCR instead of VLM
        #[arg(long)]
        ocr: bool,

        /// Use GPU-accelerated PaddleOCR via ONNX Runtime (requires --features gpu)
        #[arg(long)]
        gpu_ocr: bool,

        /// Directory containing ONNX models (det.onnx, rec.onnx, dict.txt)
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Temp working directory for frames
        #[arg(long)]
        work_dir: Option<PathBuf>,

        /// Max concurrent extraction workers
        #[arg(long, default_value_t = 4)]
        parallel: usize,

        /// Per-request VLM timeout in seconds
        #[arg(long = "vlm-timeout", default_value_t = glassrip::extract::vlm::DEFAULT_TIMEOUT_SECS)]
        vlm_timeout: u64,

        /// Fraction of frames (0.0 to 1.0) allowed to fail extraction before the run fails
        #[arg(long, default_value_t = glassrip::pipeline::DEFAULT_MAX_FRAME_FAILURE_RATE, value_parser = parse_rate)]
        max_frame_failure_rate: f64,

        /// Refine output via parallel Claude agents (requires ANTHROPIC_API_KEY)
        #[arg(long)]
        refine: bool,

        /// Claude model for refinement
        #[arg(long, default_value = glassrip::stitch::refine::DEFAULT_REFINE_MODEL)]
        refine_model: String,

        /// Number of parallel refine agents
        #[arg(long, default_value_t = 4)]
        refine_agents: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Scrape {
            video,
            output,
            model,
            ollama_host,
            ssim_threshold,
            ocr,
            gpu_ocr,
            model_dir,
            work_dir,
            parallel,
            vlm_timeout,
            max_frame_failure_rate,
            refine,
            refine_model,
            refine_agents,
        } => {
            if !video.exists() {
                bail!("{} not found", video.display());
            }

            std::fs::create_dir_all(&output)
                .with_context(|| format!("failed to create output dir {}", output.display()))?;

            let model_dir = model_dir.unwrap_or_else(glassrip::extract::gpu_ocr::default_model_dir);

            let args = glassrip::pipeline::ScrapeArgs {
                video,
                output,
                model,
                ollama_host,
                ssim_threshold,
                ocr,
                gpu_ocr,
                model_dir,
                work_dir,
                parallel: parallel.max(1),
                refine,
                refine_model,
                refine_agents: refine_agents.max(1),
                vlm_timeout_secs: vlm_timeout.max(1),
                max_frame_failure_rate,
            };
            glassrip::pipeline::run_pipeline(&args).await
        }
    }
}

fn parse_rate(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("{v} is not between 0.0 and 1.0"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scrape(args: &[&str]) -> Result<Commands, clap::Error> {
        let mut argv = vec!["glassrip", "scrape", "video.mp4"];
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv).map(|c| c.command)
    }

    #[test]
    fn robustness_flag_defaults() {
        let Commands::Scrape { vlm_timeout, max_frame_failure_rate, .. } = scrape(&[]).unwrap();
        assert_eq!(vlm_timeout, 120);
        assert_eq!(max_frame_failure_rate, 0.10);
    }

    #[test]
    fn refine_model_default() {
        let Commands::Scrape { refine_model, .. } = scrape(&[]).unwrap();
        assert_eq!(refine_model, "claude-opus-5-5");
    }

    #[test]
    fn robustness_flags_parse() {
        let Commands::Scrape { vlm_timeout, max_frame_failure_rate, .. } =
            scrape(&["--vlm-timeout", "300", "--max-frame-failure-rate", "0.25"]).unwrap();
        assert_eq!(vlm_timeout, 300);
        assert_eq!(max_frame_failure_rate, 0.25);
    }

    #[test]
    fn failure_rate_out_of_range_rejected() {
        assert!(scrape(&["--max-frame-failure-rate", "1.5"]).is_err());
        assert!(scrape(&["--max-frame-failure-rate", "-0.1"]).is_err());
    }
}
