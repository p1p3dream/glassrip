//! Shared helpers: CPU offload, external commands, tool versions.

use std::path::Path;
use std::process::Stdio;
use std::time::Instant;

use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::ItemContext;

/// Bytes of stderr kept in error messages.
pub const STDERR_TAIL: usize = 4000;

/// Runs CPU-bound work on the rayon pool and awaits it. A panic becomes an `internal` error.
pub async fn on_rayon<T, F>(f: F) -> Result<T, ErrorInfo>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, ErrorInfo> + Send + 'static,
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    rayon::spawn(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| {
            Err(ErrorInfo::new(
                ErrorCode::Internal,
                "worker panicked (a bug in glassrip)",
            ))
        });
        // The receiver is gone only when the item was cancelled.
        let _ = tx.send(r);
    });
    rx.await
        .map_err(|_| ErrorInfo::new(ErrorCode::Cancelled, "worker result was dropped"))?
}

/// Last `max` bytes of `bytes` as lossy UTF-8.
pub fn tail(bytes: &[u8], max: usize) -> String {
    let start = bytes.len().saturating_sub(max);
    String::from_utf8_lossy(&bytes[start..]).into_owned()
}

/// Output of [`run_command`].
#[derive(Debug)]
pub struct CommandOutput {
    /// Standard output.
    pub stdout: Vec<u8>,
    /// Standard error.
    pub stderr: Vec<u8>,
    /// Wall time in seconds.
    pub wall_s: f64,
}

/// Runs `argv`, records it in the manifest, and fails with the stderr tail on non-zero
/// exit. The child is killed if the future is dropped (cancellation).
pub async fn run_command(
    ctx: Option<&ItemContext>,
    argv: &[String],
) -> Result<CommandOutput, ErrorInfo> {
    let Some((prog, args)) = argv.split_first() else {
        return Err(ErrorInfo::new(ErrorCode::Internal, "empty command"));
    };
    let started = Instant::now();
    let out = tokio::process::Command::new(prog)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| {
            ErrorInfo::new(
                ErrorCode::ExternalCommand,
                format!("cannot start `{prog}`: {e} (is it installed and on PATH?)"),
            )
        })?;
    let wall_s = started.elapsed().as_secs_f64();
    if let Some(ctx) = ctx {
        ctx.record_command(argv.to_vec(), out.status.code(), Some(wall_s));
    }
    if !out.status.success() {
        return Err(ErrorInfo::new(
            ErrorCode::ExternalCommand,
            format!("`{prog}` exited with {}", out.status),
        )
        .with_raw_text(tail(&out.stderr, STDERR_TAIL)));
    }
    Ok(CommandOutput {
        stdout: out.stdout,
        stderr: out.stderr,
        wall_s,
    })
}

/// First line of `<tool> -version`, used in cache keys.
pub fn tool_version(tool: &str) -> std::io::Result<String> {
    let out = std::process::Command::new(tool)
        .arg("-version")
        .stdin(Stdio::null())
        .output()?;
    if !out.status.success() {
        return Err(std::io::Error::other(format!(
            "`{tool} -version` exited with {}",
            out.status
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_string())
}

/// An `io` error as an item error.
pub fn io_error(what: &str, path: &Path, e: impl std::fmt::Display) -> ErrorInfo {
    ErrorInfo::new(ErrorCode::Io, format!("{what} {}: {e}", path.display()))
}

/// A media error as an item error.
pub fn media_error(e: glassrip_media::MediaError) -> ErrorInfo {
    let code = match e {
        glassrip_media::MediaError::Io { .. } => ErrorCode::Io,
        _ => ErrorCode::InvalidInput,
    };
    ErrorInfo::new(code, e.to_string())
}

/// Parses an ffprobe rational such as `88/3`. `0/0` and malformed values give `None`.
pub fn parse_rational(s: &str) -> Option<(i64, i64)> {
    let (n, d) = s.split_once('/')?;
    let (n, d) = (n.trim().parse::<i64>().ok()?, d.trim().parse::<i64>().ok()?);
    (d != 0 && n != 0).then_some((n, d))
}

/// Rational as f64.
pub fn rational_f64(r: (i64, i64)) -> f64 {
    r.0 as f64 / r.1 as f64
}

/// Default parallelism of this machine.
pub fn cpus() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rationals() {
        assert_eq!(parse_rational("88/3"), Some((88, 3)));
        assert_eq!(parse_rational("0/0"), None);
        assert_eq!(parse_rational("30"), None);
        assert!((rational_f64((30000, 1001)) - 29.97).abs() < 0.01);
    }

    #[test]
    fn tail_keeps_end() {
        assert_eq!(tail(b"abcdef", 3), "def");
        assert_eq!(tail(b"ab", 3), "ab");
    }

    #[tokio::test]
    async fn rayon_panic_is_an_error() {
        let r: Result<(), ErrorInfo> = on_rayon(|| panic!("boom")).await;
        assert_eq!(r.unwrap_err().code, ErrorCode::Internal);
        assert_eq!(on_rayon(|| Ok(3)).await.unwrap(), 3);
    }
}
