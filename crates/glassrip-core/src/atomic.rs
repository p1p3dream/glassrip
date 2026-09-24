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
        move |source: io::Error| AtomicWriteError::Io {
            path,
            persisted,
            source,
        }
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
    tmp.persist(path).map_err(|e| AtomicWriteError::Io {
        path: path.to_path_buf(),
        persisted: false,
        source: e.error,
    })?;
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
        match result {
            Err(AtomicWriteError::Io {
                persisted, source, ..
            }) => {
                assert!(!persisted);
                assert_eq!(source.kind(), io::ErrorKind::StorageFull);
            }
            other => panic!("expected simulated failure, got {other:?}"),
        }
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
