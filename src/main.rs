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
        #[arg(long = "vlm-timeout", default_value_t = glassrip::extract::vlm::DEFAULT_TIMEOUT_SECS, value_parser = clap::value_parser!(u64).range(1..))]
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
    /// Turn a meeting recording into board state, transcript, notes, and an SVG
    Meeting(glassrip::meeting::MeetingArgs),
    /// Evaluate against golden fixtures and gate regressions (spec section 9)
    Eval(glassrip_eval::cli::EvalArgs),
}

fn main() {
    let code = match run_cli() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e:?}");
            1
        }
    };
    exit_now(code);
}

/// Ends the process without running exit-time destructors. ort 2.0.0-rc.13
/// releases its ONNX Runtime environment from `.fini_array`; with the
/// dynamically loaded runtime and CUDA that hook runs during CUDA teardown and
/// corrupts the heap (or panics when no runtime was loaded), so every run
/// ended with SIGABRT. Everything glassrip writes is persisted before this
/// point (artifacts, manifest, and log are written and synced as they are
/// produced), so only stdout and stderr need flushing.
fn exit_now(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(all(unix, feature = "cuda"))]
    {
        // SAFETY: `_exit` takes no pointers and never returns; skipping the
        // exit handlers is the purpose (see above). Only the dynamically
        // loaded CUDA runtime needs this; other builds tear down normally.
        unsafe { libc::_exit(code) }
    }
    #[cfg(not(all(unix, feature = "cuda")))]
    std::process::exit(code)
}

fn run_cli() -> Result<i32> {
    // `try_parse` keeps help, version, and usage errors on the `exit_now`
    // path; `Cli::parse` would exit from inside clap and run the exit hooks.
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return Ok(e.exit_code());
        }
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("cannot start the async runtime")?;
    let result = runtime.block_on(run(cli));
    // Joins the blocking pool, so no worker is mid-write when the process ends.
    drop(runtime);
    result
}

/// Runs a command; returns the process exit code.
async fn run(cli: Cli) -> Result<i32> {
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
                vlm_timeout_secs: vlm_timeout,
                max_frame_failure_rate,
            };
            glassrip::pipeline::run_pipeline(&args).await.map(|()| 0)
        }
        Commands::Meeting(args) => glassrip::meeting::cli::run(args).await.map(|()| 0),
        Commands::Eval(args) => Ok(if glassrip_eval::cli::run(args).await?.passed {
            0
        } else {
            1
        }),
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
        let Commands::Scrape {
            vlm_timeout,
            max_frame_failure_rate,
            ..
        } = scrape(&[]).unwrap()
        else {
            panic!("expected scrape")
        };
        assert_eq!(vlm_timeout, 120);
        assert_eq!(max_frame_failure_rate, 0.10);
    }

    #[test]
    fn refine_model_default() {
        let Commands::Scrape { refine_model, .. } = scrape(&[]).unwrap() else {
            panic!("expected scrape")
        };
        assert_eq!(refine_model, "claude-opus-5-5");
    }

    #[test]
    fn robustness_flags_parse() {
        let Commands::Scrape {
            vlm_timeout,
            max_frame_failure_rate,
            ..
        } = scrape(&["--vlm-timeout", "300", "--max-frame-failure-rate", "0.25"]).unwrap()
        else {
            panic!("expected scrape")
        };
        assert_eq!(vlm_timeout, 300);
        assert_eq!(max_frame_failure_rate, 0.25);
    }

    #[test]
    fn failure_rate_out_of_range_rejected() {
        assert!(scrape(&["--max-frame-failure-rate", "1.5"]).is_err());
        assert!(scrape(&["--max-frame-failure-rate", "-0.1"]).is_err());
        assert!(scrape(&["--max-frame-failure-rate", "NaN"]).is_err());
    }

    #[test]
    fn eval_subcommand_parses() {
        let cli =
            Cli::try_parse_from(["glassrip", "eval", "--suite", "synthetic", "--bench"]).unwrap();
        assert!(matches!(cli.command, Commands::Eval(ref a) if a.bench));
        assert!(Cli::try_parse_from(["glassrip", "eval"]).is_err());
    }

    #[test]
    fn zero_vlm_timeout_rejected() {
        assert!(scrape(&["--vlm-timeout", "0"]).is_err());
        let Commands::Scrape { vlm_timeout, .. } = scrape(&["--vlm-timeout", "1"]).unwrap() else {
            panic!("expected scrape")
        };
        assert_eq!(vlm_timeout, 1);
    }
}
