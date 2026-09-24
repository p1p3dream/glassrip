//! Content-addressed image store and run-directory materialization.
//!
//! The stage cache stores only JSONL artifacts. Stages that produce image files put them in
//! a [`BlobStore`] (`<root>/<h[0..2]>/<h>.<ext>`, `h` = blake3) and record a path relative
//! to the run directory plus the blake3. [`BlobStore::materialize`] links (or copies) the
//! blob to that path, so a stage restored from cache into a fresh run directory gets its
//! files back without recomputing. If a blob was deleted, materialization fails with a
//! message telling the user to force the producing stage.

use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use glassrip_core::cache::{CACHE_LOCK_FILE, Cache, CacheError, DEFAULT_GC_GRACE, GcReport};

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
    /// The blob's content does not match its address (it was removed).
    #[error("blob {path} is corrupt (content hash {actual}); rerun with --force-stage {stage}")]
    Corrupt {
        /// Blob path.
        path: PathBuf,
        /// Actual hash.
        actual: String,
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

    /// Makes `run_root/rel` a copy of blob `hash.ext` (hard link when possible). The blob
    /// is hashed before use: a corrupt blob is removed and reported (force the producing
    /// stage). An existing destination is kept only when its own hash matches.
    pub fn materialize(
        &self,
        run_root: &Path,
        rel: &str,
        hash: &str,
        stage: &str,
    ) -> Result<PathBuf, BlobError> {
        let ext = Path::new(rel)
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| BlobError::Invalid(rel.to_string()))?;
        let blob = self.path(hash, ext)?;
        let dest = run_root.join(rel);
        if glassrip_core::atomic::metadata_opt(&dest)
            .map_err(io_err(&dest))?
            .is_some()
        {
            if hash_file(&dest).map_err(io_err(&dest))? == hash {
                return Ok(dest);
            }
            fs_err::remove_file(&dest).map_err(io_err(&dest))?;
        }
        if glassrip_core::atomic::metadata_opt(&blob)
            .map_err(io_err(&blob))?
            .is_none()
        {
            return Err(BlobError::Missing {
                hash: hash.to_string(),
                ext: ext.to_string(),
                root: self.root.clone(),
                stage: stage.to_string(),
            });
        }
        let actual = hash_file(&blob).map_err(io_err(&blob))?;
        if actual != hash {
            // Never serve it again; the next run of the stage writes a good copy.
            fs_err::remove_file(&blob).map_err(io_err(&blob))?;
            return Err(BlobError::Corrupt {
                path: blob,
                actual,
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

    /// Every blob file in the store: `(hash, path, size, modified)`.
    fn list(&self) -> Result<Vec<(String, PathBuf, u64, SystemTime)>, BlobError> {
        let mut out = Vec::new();
        if !glassrip_core::atomic::is_dir(&self.root).map_err(io_err(&self.root))? {
            return Ok(out);
        }
        for d in fs_err::read_dir(&self.root).map_err(io_err(&self.root))? {
            let d = d.map_err(io_err(&self.root))?.path();
            if !d.is_dir() {
                continue;
            }
            for f in fs_err::read_dir(&d).map_err(io_err(&d))? {
                let path = f.map_err(io_err(&d))?.path();
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                if !valid_hash(stem) {
                    continue;
                }
                let meta = fs_err::metadata(&path).map_err(io_err(&path))?;
                let modified = meta.modified().map_err(io_err(&path))?;
                out.push((stem.to_string(), path, meta.len(), modified));
            }
        }
        Ok(out)
    }
}

/// Result of [`gc_cache_and_blobs`].
#[derive(Debug, Default)]
pub struct BlobGcReport {
    /// The stage cache's own report.
    pub cache: GcReport,
    /// Blobs removed.
    pub blobs_removed: Vec<PathBuf>,
    /// Blobs kept because a cache entry references them.
    pub blobs_referenced: usize,
    /// Bytes freed from the blob store.
    pub blob_bytes_freed: u64,
}

/// Every 64-hex `"blake3"` value in a text (cache entries reference blobs by hash).
pub fn referenced_hashes(text: &str) -> HashSet<String> {
    let key = "\"blake3\":\"";
    let mut out = HashSet::new();
    let mut rest = text;
    while let Some(i) = rest.find(key) {
        rest = &rest[i + key.len()..];
        if let Some(h) = rest.get(..64).filter(|h| valid_hash(h)) {
            out.insert(h.to_string());
        }
    }
    out
}

/// `cache gc` including the blob store: runs [`Cache::gc`], then (holding the cache lock
/// exclusively, so it fails with `Busy` while any run holds a lease) removes blobs that no
/// remaining cache entry references and that are older than `older_than` and the grace
/// window. Run directories keep their hard links, so existing runs are unaffected.
pub fn gc_cache_and_blobs(
    cache: &Cache,
    blobs: &BlobStore,
    older_than: Duration,
    now: SystemTime,
) -> Result<BlobGcReport, BlobGcError> {
    let mut report = BlobGcReport {
        cache: cache.gc(older_than, now)?,
        ..BlobGcReport::default()
    };
    let lock_path = cache.root().join(CACHE_LOCK_FILE);
    let lock = fs_err::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|e| BlobGcError::Blob(io_err(&lock_path)(e)))?;
    match fs4::fs_err3::FileExt::try_lock(&lock) {
        Ok(()) => {}
        Err(fs4::TryLockError::WouldBlock) => {
            return Err(BlobGcError::Cache(CacheError::Busy(
                cache.root().to_path_buf(),
            )));
        }
        Err(fs4::TryLockError::Error(e)) => return Err(BlobGcError::Blob(io_err(&lock_path)(e))),
    }
    let mut referenced = HashSet::new();
    for entry in cache.ls()? {
        if entry.ext != "jsonl" {
            continue;
        }
        let text = fs_err::read_to_string(&entry.path)
            .map_err(|e| BlobGcError::Blob(io_err(&entry.path)(e)))?;
        referenced.extend(referenced_hashes(&text));
    }
    let cutoff = now
        .checked_sub(older_than.max(DEFAULT_GC_GRACE))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    for (hash, path, size, modified) in blobs.list()? {
        if referenced.contains(&hash) {
            report.blobs_referenced += 1;
            continue;
        }
        if modified < cutoff {
            fs_err::remove_file(&path).map_err(|e| BlobGcError::Blob(io_err(&path)(e)))?;
            report.blob_bytes_freed += size;
            report.blobs_removed.push(path);
        }
    }
    drop(lock);
    Ok(report)
}

/// Error from [`gc_cache_and_blobs`].
#[derive(Debug, thiserror::Error)]
pub enum BlobGcError {
    /// Stage cache failure (including `Busy`).
    #[error(transparent)]
    Cache(#[from] CacheError),
    /// Blob store failure.
    #[error(transparent)]
    Blob(#[from] BlobError),
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
            .materialize(&run, "frames/x.jpg", &h, "frames")
            .unwrap();
        assert_eq!(fs_err::read(&p).unwrap(), b"synthetic bytes");
        // Tampered copy is replaced.
        fs_err::remove_file(&p).unwrap();
        fs_err::write(&p, b"synthetic bytez").unwrap();
        store
            .materialize(&run, "frames/x.jpg", &h, "frames")
            .unwrap();
        assert_eq!(fs_err::read(&p).unwrap(), b"synthetic bytes");
        // Missing blob names the stage to force.
        let other = "0".repeat(64);
        let err = store
            .materialize(&run, "frames/y.jpg", &other, "frames")
            .unwrap_err();
        assert!(err.to_string().contains("--force-stage frames"), "{err}");
        assert!(store.path("../x", "jpg").is_err());
    }

    #[test]
    fn corrupt_blob_is_detected_and_removed() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::new(dir.path().join("blobs"));
        let h = store.put_bytes(b"synthetic", "jpg").unwrap();
        let blob = store.path(&h, "jpg").unwrap();
        fs_err::write(&blob, b"bitrot!!!").unwrap();
        let err = store
            .materialize(&dir.path().join("run"), "a.jpg", &h, "rectify")
            .unwrap_err();
        assert!(matches!(err, BlobError::Corrupt { .. }), "{err}");
        assert!(!blob.exists());
    }

    #[test]
    fn gc_removes_only_unreferenced_old_blobs_and_respects_leases() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path().join("cache"));
        let store = BlobStore::new(dir.path().join("blobs"));
        let keep = store.put_bytes(b"kept", "jpg").unwrap();
        let drop_ = store.put_bytes(b"dropped", "jpg").unwrap();
        let key = glassrip_core::cache::CacheKeyParts {
            stage: "frames".into(),
            ..Default::default()
        }
        .key()
        .unwrap();
        let line = format!("{{\"record\":\"item\",\"blake3\":\"{keep}\"}}\n");
        cache.put("frames", &key, "jsonl", line.as_bytes()).unwrap();
        let age = |h: &str, secs: u64| {
            let f = std::fs::File::options()
                .write(true)
                .open(store.path(h, "jpg").unwrap())
                .unwrap();
            f.set_modified(SystemTime::now() - Duration::from_secs(secs))
                .unwrap();
        };
        let two_hours = Duration::from_secs(7200);
        // Fresh blobs are kept (grace window and `older_than`).
        let r = gc_cache_and_blobs(&cache, &store, two_hours, SystemTime::now()).unwrap();
        assert!(r.blobs_removed.is_empty());
        age(&keep, 3 * 3600);
        age(&drop_, 3 * 3600);
        // A run holding a lease blocks gc.
        let lease = cache.lease().unwrap();
        assert!(gc_cache_and_blobs(&cache, &store, two_hours, SystemTime::now()).is_err());
        drop(lease);
        let r = gc_cache_and_blobs(&cache, &store, two_hours, SystemTime::now()).unwrap();
        assert_eq!(r.blobs_removed, vec![store.path(&drop_, "jpg").unwrap()]);
        assert_eq!(r.blobs_referenced, 1);
        assert!(store.path(&keep, "jpg").unwrap().exists());
    }

    #[test]
    fn finds_referenced_hashes() {
        let h = "a".repeat(64);
        let t = format!("x \"blake3\":\"{h}\" y \"blake3\":\"short\"");
        assert_eq!(referenced_hashes(&t), [h].into_iter().collect());
    }
}
