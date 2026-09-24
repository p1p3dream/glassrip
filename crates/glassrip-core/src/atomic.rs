//! Atomic whole-file writes.
//!
//! Sequence: create a temp file in the destination's directory, write, `sync_all`,
//! rename over the destination (`persist`), then fsync the parent directory on Unix so
//! the rename itself is durable. If anything fails before `persist`, the destination is
//! untouched and the temp file is removed.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Error from an atomic write.
#[derive(Debug, thiserror::Error)]
pub enum AtomicWriteError {
    /// The destination has no parent directory.
    #[error("destination {0} has no parent directory")]
    NoParent(PathBuf),
    /// The filesystem is full; the destination was not modified.
    #[error("disk full writing {path} ({}): {source}", describe_space(*available_bytes))]
    DiskFull {
        /// Destination path.
        path: PathBuf,
        /// Bytes available to this user on that filesystem, if measurable.
        available_bytes: Option<u64>,
        /// Underlying error.
        #[source]
        source: io::Error,
    },
    /// An I/O error; the destination was not modified if `persisted` is false.
    #[error("atomic write to {path} failed (persisted: {persisted}): {source}")]
    Io {
        /// Destination path.
        path: PathBuf,
        /// Whether the rename had already happened when the error occurred.
        persisted: bool,
        /// Underlying error.
        #[source]
        source: io::Error,
    },
}

/// Atomically replaces `path` with `bytes`.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), AtomicWriteError> {
    write_atomic_with(path, |w| w.write_all(bytes))
}

/// Atomically replaces `path` with whatever `fill` writes.
///
/// `fill` receives a buffered writer over the temp file. If it returns an error, the
/// temp file is discarded and the destination keeps its previous content.
pub fn write_atomic_with<F>(path: &Path, fill: F) -> Result<(), AtomicWriteError>
where
    F: FnOnce(&mut dyn Write) -> io::Result<()>,
{
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        Some(_) => PathBuf::from("."),
        None => return Err(AtomicWriteError::NoParent(path.to_path_buf())),
    };
    let io_err = |persisted: bool| {
        let path = path.to_path_buf();
        move |source: io::Error| classify(path, persisted, source)
    };

    fs_err::create_dir_all(&parent).map_err(io_err(false))?;
    let tmp = tempfile::Builder::new()
        .prefix(".glassrip-tmp-")
        .tempfile_in(&parent)
        .map_err(io_err(false))?;
    {
        let mut writer = io::BufWriter::new(tmp.as_file());
        fill(&mut writer).map_err(io_err(false))?;
        writer.flush().map_err(io_err(false))?;
    }
    tmp.as_file().sync_all().map_err(io_err(false))?;
    tmp.persist(path)
        .map_err(|e| classify(path.to_path_buf(), false, e.error))?;
    sync_dir(&parent).map_err(io_err(true))?;
    Ok(())
}

/// Atomically copies `src` to `dst`.
pub fn copy_atomic(src: &Path, dst: &Path) -> Result<(), AtomicWriteError> {
    let mut input = fs_err::File::open(src).map_err(|source| AtomicWriteError::Io {
        path: dst.to_path_buf(),
        persisted: false,
        source,
    })?;
    write_atomic_with(dst, |w| io::copy(&mut input, w).map(|_| ()))
}

fn classify(path: PathBuf, persisted: bool, source: io::Error) -> AtomicWriteError {
    match available_space_if_full(&path, &source) {
        Some(available_bytes) => AtomicWriteError::DiskFull {
            path,
            available_bytes,
            source,
        },
        None => AtomicWriteError::Io {
            path,
            persisted,
            source,
        },
    }
}

/// For a disk-full error, returns `Some(available bytes)` measured on the filesystem
/// holding `path` (the inner `None` means it could not be measured); `None` for any
/// other error.
pub fn available_space_if_full(path: &Path, err: &io::Error) -> Option<Option<u64>> {
    if err.kind() != io::ErrorKind::StorageFull {
        return None;
    }
    let probe = path
        .ancestors()
        .find(|p| !p.as_os_str().is_empty() && matches!(metadata_opt(p), Ok(Some(_))))
        .unwrap_or_else(|| Path::new("."));
    Some(fs4::available_space(probe).ok())
}

/// Human-readable free space for error messages.
pub fn describe_space(available: Option<u64>) -> String {
    match available {
        Some(bytes) => format!("{bytes} bytes available"),
        None => "available space unknown".to_string(),
    }
}

/// Metadata for `path`, `None` when it does not exist. Other errors (for example
/// permission denied) are returned rather than being mistaken for absence.
pub fn metadata_opt(path: &Path) -> io::Result<Option<std::fs::Metadata>> {
    match fs_err::metadata(path) {
        Ok(m) => Ok(Some(m)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// True when `path` is an existing regular file; errors other than not-found are
/// returned.
pub fn is_file(path: &Path) -> io::Result<bool> {
    Ok(metadata_opt(path)?.is_some_and(|m| m.is_file()))
}

/// True when `path` is an existing directory; errors other than not-found are
/// returned.
pub fn is_dir(path: &Path) -> io::Result<bool> {
    Ok(metadata_opt(path)?.is_some_and(|m| m.is_dir()))
}

/// Fsyncs a directory so renames and creations inside it are durable (Unix only; a
/// no-op elsewhere).
pub fn sync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        fs_err::File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leftover_temps(dir: &Path) -> usize {
        fs_err::read_dir(dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .starts_with(".glassrip-tmp-")
                    })
                    .count()
            })
            .unwrap_or(usize::MAX)
    }

    #[test]
    fn writes_and_replaces() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("sub/out.json");
        write_atomic(&path, b"first")?;
        assert_eq!(fs_err::read(&path)?, b"first");
        write_atomic(&path, b"second")?;
        assert_eq!(fs_err::read(&path)?, b"second");
        assert_eq!(leftover_temps(&dir.path().join("sub")), 0);
        Ok(())
    }

    #[test]
    fn failure_mid_write_keeps_old_content() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("out.json");
        write_atomic(&path, b"original")?;
        let result = write_atomic_with(&path, |w| {
            w.write_all(b"partial new content")?;
            Err(io::Error::new(
                io::ErrorKind::StorageFull,
                "simulated disk full",
            ))
        });
        match &result {
            Err(AtomicWriteError::DiskFull {
                path: p,
                available_bytes,
                source,
            }) => {
                assert_eq!(p, &path);
                assert!(available_bytes.is_some());
                assert_eq!(source.kind(), io::ErrorKind::StorageFull);
            }
            other => panic!("expected simulated disk full, got {other:?}"),
        }
        let message = result.map_err(|e| e.to_string()).unwrap_err();
        assert!(
            message.contains("bytes available") && message.contains("out.json"),
            "{message}"
        );
        assert_eq!(fs_err::read(&path)?, b"original");
        assert_eq!(leftover_temps(dir.path()), 0);
        Ok(())
    }

    #[test]
    fn failure_on_first_write_creates_nothing() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("never.json");
        let result = write_atomic_with(&path, |_| Err(io::Error::other("boom")));
        assert!(result.is_err());
        assert!(!path.exists());
        assert_eq!(leftover_temps(dir.path()), 0);
        Ok(())
    }

    #[test]
    fn metadata_helpers() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        assert!(metadata_opt(&dir.path().join("missing"))?.is_none());
        assert!(is_dir(dir.path())?);
        assert!(!is_file(dir.path())?);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn permission_error_is_not_absence() -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir()?;
        let locked = dir.path().join("locked");
        fs_err::create_dir(&locked)?;
        fs_err::write(locked.join("f"), b"x")?;
        fs_err::set_permissions(&locked, std::fs::Permissions::from_mode(0o000))?;
        let result = metadata_opt(&locked.join("f"));
        fs_err::set_permissions(&locked, std::fs::Permissions::from_mode(0o755))?;
        // Root can read anything; only assert when the permission bit is enforced.
        if let Err(e) = result {
            assert_eq!(e.kind(), io::ErrorKind::PermissionDenied);
        }
        Ok(())
    }

    #[test]
    fn copy_is_atomic() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let src = dir.path().join("a");
        let dst = dir.path().join("b");
        fs_err::write(&src, b"payload")?;
        copy_atomic(&src, &dst)?;
        assert_eq!(fs_err::read(&dst)?, b"payload");
        Ok(())
    }
}
