//! Logging for `glassrip meeting`: human-readable progress on stderr and the
//! structured `run.log.jsonl` (one JSON object per event, with stage and item
//! spans) in the output directory.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{fmt, EnvFilter, Layer};

/// File name of the structured log.
pub const RUN_LOG: &str = "run.log.jsonl";

/// Installs the global subscriber: stderr at `RUST_LOG` (default `info`) and
/// `<out_dir>/run.log.jsonl` at `debug` for glassrip crates. Appends to an
/// existing log (a resumed run keeps its history). Fails if a global subscriber
/// is already installed.
pub fn init(out_dir: &Path) -> anyhow::Result<PathBuf> {
    fs_err::create_dir_all(out_dir)?;
    let path = out_dir.join(RUN_LOG);
    let file = fs_err::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?
        .into_parts()
        .0;
    let stderr_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let file_filter = EnvFilter::new("info,glassrip=debug");
    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_writer(std::io::stderr)
                .with_target(false)
                .with_filter(stderr_filter),
        )
        .with(
            fmt::layer()
                .json()
                .with_current_span(true)
                .with_span_list(true)
                .with_writer(Mutex::new(file))
                .with_filter(file_filter),
        )
        .try_init()
        .map_err(|e| anyhow::anyhow!("cannot install the logger: {e}"))?;
    Ok(path)
}
