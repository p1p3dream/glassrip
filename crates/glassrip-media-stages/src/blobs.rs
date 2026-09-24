//! Content-addressed image store and run-directory materialization.
//!
//! The stage cache stores only JSONL artifacts. Stages that produce image files put them in
//! a [`BlobStore`] (`<root>/<h[0..2]>/<h>.<ext>`, `h` = blake3) and record a path relative
//! to the run directory plus the blake3. [`BlobStore::materialize`] links (or copies) the
//! blob to that path, so a stage restored from cache into a fresh run directory gets its
//! files back without recomputing. If a blob was deleted, materialization fails with a
//! message telling the user to force the producing stage.

use std::io;
use std::path::{Path, PathBuf};

/// Error from the blob store.
#[derive(Debug, thiserror::Error)]
pub enum BlobError {
    /// Filesystem failure.
    #[error("blob store I/O on {path}: {source}")]
    Io {
        /// Path involved.
        path: PathBuf,
        /// Cause.
        #[source]
        source: io::Error,
    },
    /// The blob is gone (for example removed by hand).
    #[error("blob {hash}.{ext} is missing from {root}; rerun with --force-stage {stage}")]
    Missing {
        /// Blob hash.
        hash: String,
        /// Extension.
        ext: String,
        /// Store root.
        root: PathBuf,
        /// Stage that produces it.
        stage: String,
    },
    /// A hash is not 64 lowercase hex characters, or an extension is unusual.
    #[error("invalid blob reference `{0}`")]
    Invalid(String),
}

fn io_err(path: &Path) -> impl FnOnce(io::Error) -> BlobError + '_ {
    move |source| BlobError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn valid_hash(h: &str) -> bool {
    h.len() == 64
        && h.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn valid_ext(e: &str) -> bool {
    !e.is_empty() && e.len() <= 8 && e.bytes().all(|b| b.is_ascii_alphanumeric())
}

/// blake3 of a file, multithreaded and memory-mapped for large files.
pub fn hash_file(path: &Path) -> io::Result<String> {
    let mut h = blake3::Hasher::new();
    h.update_mmap_rayon(path)?;
    Ok(h.finalize().to_hex().to_string())
}

/// A content-addressed file store.
#[derive(Debug, Clone)]
pub struct BlobStore {
    root: PathBuf,
}

impl BlobStore {
    /// Store rooted at `root` (created on first write).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Store root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path of a blob.
    pub fn path(&self, hash: &str, ext: &str) -> Result<PathBuf, BlobError> {
        if !valid_hash(hash) || !valid_ext(ext) {
            return Err(BlobError::Invalid(format!("{hash}.{ext}")));
        }
        Ok(self.root.join(&hash[..2]).join(format!("{hash}.{ext}")))
    }

    /// Moves `src` into the store (or deletes it when the blob already exists) and returns
    /// its blake3.
    pub fn put_move(&self, src: &Path, ext: &str) -> Result<String, BlobError> {
        let hash = hash_file(src).map_err(io_err(src))?;
        let dest = self.path(&hash, ext)?;
        if glassrip_core::atomic::is_file(&dest).map_err(io_err(&dest))? {
            fs_err::remove_file(src).map_err(io_err(src))?;
            return Ok(hash);
        }
        let dir = dest.parent().unwrap_or(&self.root).to_path_buf();
        fs_err::create_dir_all(&dir).map_err(io_err(&dir))?;
        if fs_err::rename(src, &dest).is_err() {
            // Different filesystem: copy atomically, then remove the source.
            glassrip_core::atomic::copy_atomic(src, &dest).map_err(|e| BlobError::Io {
                path: dest.clone(),
                source: io::Error::other(e.to_string()),
            })?;
            fs_err::remove_file(src).map_err(io_err(src))?;
        }
        Ok(hash)
    }

    /// Writes bytes into the store and returns their blake3.
    pub fn put_bytes(&self, bytes: &[u8], ext: &str) -> Result<String, BlobError> {
        let hash = blake3::hash(bytes).to_hex().to_string();
        let dest = self.path(&hash, ext)?;
        if !glassrip_core::atomic::is_file(&dest).map_err(io_err(&dest))? {
            glassrip_core::atomic::write_atomic(&dest, bytes).map_err(|e| BlobError::Io {
                path: dest.clone(),
                source: io::Error::other(e.to_string()),
            })?;
        }
        Ok(hash)
    }

    /// Makes `run_root/rel` a copy of blob `hash.ext` (hard link when possible). An existing
    /// file with the right size is kept when `trust_existing` is set, otherwise it is
    /// re-hashed and replaced on mismatch.
    pub fn materialize(
        &self,
        run_root: &Path,
        rel: &str,
        hash: &str,
        stage: &str,
        trust_existing: bool,
    ) -> Result<PathBuf, BlobError> {
        let ext = Path::new(rel)
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| BlobError::Invalid(rel.to_string()))?;
        let blob = self.path(hash, ext)?;
        let dest = run_root.join(rel);
        let blob_meta = glassrip_core::atomic::metadata_opt(&blob).map_err(io_err(&blob))?;
        if let Some(meta) = glassrip_core::atomic::metadata_opt(&dest).map_err(io_err(&dest))? {
            let same_size = blob_meta.as_ref().is_none_or(|b| b.len() == meta.len());
            if same_size && (trust_existing || hash_file(&dest).map_err(io_err(&dest))? == hash) {
                return Ok(dest);
            }
            fs_err::remove_file(&dest).map_err(io_err(&dest))?;
        }
        if blob_meta.is_none() {
            return Err(BlobError::Missing {
                hash: hash.to_string(),
                ext: ext.to_string(),
                root: self.root.clone(),
                stage: stage.to_string(),
            });
        }
        if let Some(dir) = dest.parent() {
            fs_err::create_dir_all(dir).map_err(io_err(dir))?;
        }
        if fs_err::hard_link(&blob, &dest).is_err() {
            glassrip_core::atomic::copy_atomic(&blob, &dest).map_err(|e| BlobError::Io {
                path: dest.clone(),
                source: io::Error::other(e.to_string()),
            })?;
        }
        Ok(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_and_materialize_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::new(dir.path().join("blobs"));
        let src = dir.path().join("a.jpg");
        fs_err::write(&src, b"synthetic bytes").unwrap();
        let h = store.put_move(&src, "jpg").unwrap();
        assert!(!src.exists());
        assert_eq!(h, blake3::hash(b"synthetic bytes").to_hex().to_string());
        let run = dir.path().join("run");
        let p = store
            .materialize(&run, "frames/x.jpg", &h, "frames", false)
            .unwrap();
        assert_eq!(fs_err::read(&p).unwrap(), b"synthetic bytes");
        // Tampered copy is replaced.
        fs_err::remove_file(&p).unwrap();
        fs_err::write(&p, b"synthetic bytez").unwrap();
        store
            .materialize(&run, "frames/x.jpg", &h, "frames", false)
            .unwrap();
        assert_eq!(fs_err::read(&p).unwrap(), b"synthetic bytes");
        // Missing blob names the stage to force.
        let other = "0".repeat(64);
        let err = store
            .materialize(&run, "frames/y.jpg", &other, "frames", false)
            .unwrap_err();
        assert!(err.to_string().contains("--force-stage frames"), "{err}");
        assert!(store.path("../x", "jpg").is_err());
    }
}
