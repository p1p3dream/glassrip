use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "frametap", about = "Extract code from video")]
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

        /// Refine output with a second LLM pass to fix garbled text and remaining duplicates
        #[arg(long)]
        refine: bool,

        /// Text model for refinement (must be available in Ollama)
        #[arg(long, default_value = "qwen2.5:32b")]
        refine_model: String,
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
            refine,
            refine_model,
        } => {
            if !video.exists() {
                bail!("{} not found", video.display());
            }

            std::fs::create_dir_all(&output)
                .with_context(|| format!("failed to create output dir {}", output.display()))?;

            let model_dir = model_dir.unwrap_or_else(frametap::extract::gpu_ocr::default_model_dir);

            let args = frametap::pipeline::ScrapeArgs {
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
            };
            frametap::pipeline::run_pipeline(&args).await
        }
    }
}
