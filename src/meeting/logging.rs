//! Logging for `glassrip meeting`: human-readable progress on stderr and the
//! structured `run.log.jsonl` (one JSON object per event, with stage and item
//! spans) in the output directory.
//!
//! The log file is deferred: events are buffered in memory until
//! [`LogHandle::activate`] opens the file, which `run_meeting` does only after
//! preflight passes and the output directory is created. A run that fails
//! preflight therefore leaves nothing on disk (spec 8.1), while a successful
//! run's log still contains the events from before activation.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{fmt, EnvFilter, Layer};

/// File name of the structured log.
pub const RUN_LOG: &str = "run.log.jsonl";

/// Bytes kept in memory before the file is opened; older events beyond this
/// are dropped (the file then starts with a note).
const MAX_BUFFER: usize = 8 << 20;

enum Sink {
    Buffer { bytes: Vec<u8>, dropped: bool },
    File(std::fs::File),
}

/// The deferred log file shared by the subscriber and the run.
pub struct DeferredFile {
    sink: Mutex<Sink>,
}

impl DeferredFile {
    fn new() -> Self {
        Self {
            sink: Mutex::new(Sink::Buffer {
                bytes: Vec::new(),
                dropped: false,
            }),
        }
    }

    fn write_bytes(&self, buf: &[u8]) -> std::io::Result<()> {
        let mut sink = self.sink.lock().unwrap_or_else(PoisonError::into_inner);
        match &mut *sink {
            Sink::File(f) => f.write_all(buf),
            Sink::Buffer { bytes, dropped } => {
                if bytes.len() + buf.len() <= MAX_BUFFER {
                    bytes.extend_from_slice(buf);
                } else {
                    *dropped = true;
                }
                Ok(())
            }
        }
    }

    /// Opens `<out_dir>/run.log.jsonl` for appending and flushes the buffer
    /// into it. Later calls reopen the new path (one run per process in
    /// practice).
    fn activate(&self, out_dir: &Path) -> std::io::Result<PathBuf> {
        let path = out_dir.join(RUN_LOG);
        let mut file = fs_err::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?
            .into_parts()
            .0;
        let mut sink = self.sink.lock().unwrap_or_else(PoisonError::into_inner);
        if let Sink::Buffer { bytes, dropped } = &*sink {
            if *dropped {
                file.write_all(
                    b"{\"level\":\"WARN\",\"fields\":{\"message\":\"early log events dropped (buffer full)\"}}\n",
                )?;
            }
            file.write_all(bytes)?;
        }
        *sink = Sink::File(file);
        Ok(path)
    }
}

/// Writer handed to the JSON layer.
pub struct DeferredWriter(Arc<DeferredFile>);

impl Write for DeferredWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write_bytes(buf)?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for LogHandle {
    type Writer = DeferredWriter;
    fn make_writer(&'a self) -> Self::Writer {
        DeferredWriter(Arc::clone(&self.0))
    }
}

/// Handle to the process-wide deferred log file.
#[derive(Clone)]
pub struct LogHandle(Arc<DeferredFile>);

impl std::fmt::Debug for LogHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LogHandle")
    }
}

impl LogHandle {
    /// Starts writing `run.log.jsonl` in `out_dir` (which must exist).
    pub fn activate(&self, out_dir: &Path) -> std::io::Result<PathBuf> {
        self.0.activate(out_dir)
    }
}

static HANDLE: OnceLock<LogHandle> = OnceLock::new();

/// Installs the global subscriber once per process: stderr at `RUST_LOG`
/// (default `info`) and the deferred JSON file at `debug` for glassrip crates.
/// Returns the file handle; later calls return the same handle. When another
/// subscriber is already installed, the handle still works but receives no
/// events.
pub fn init() -> LogHandle {
    HANDLE
        .get_or_init(|| {
            let handle = LogHandle(Arc::new(DeferredFile::new()));
            let stderr_filter =
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            let file_filter = EnvFilter::new("info,glassrip=debug");
            let installed = tracing_subscriber::registry()
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
                        .with_writer(handle.clone())
                        .with_filter(file_filter),
                )
                .try_init();
            if installed.is_err() {
                eprintln!("glassrip: a logger is already installed; run.log.jsonl stays empty");
            }
            handle
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_before_activation_reach_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let f = DeferredFile::new();
        f.write_bytes(b"{\"early\":1}\n").unwrap();
        assert!(
            !dir.path().join(RUN_LOG).exists(),
            "nothing on disk before activation"
        );
        let path = f.activate(dir.path()).unwrap();
        f.write_bytes(b"{\"late\":2}\n").unwrap();
        let text = std::fs::read_to_string(path).unwrap();
        assert_eq!(text, "{\"early\":1}\n{\"late\":2}\n");
    }
}
