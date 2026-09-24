//! Content-addressed stage cache.
//!
//! Key: blake3 over a domain tag plus the canonical JSON of [`CacheKeyParts`] (stage
//! name, stage version, params, sorted input hashes, model digest, prompt hash,
//! schema hash, decoder, features mode, tool versions). Hashing one canonical JSON
//! document, rather than concatenating strings, makes component boundaries
//! unambiguous.
//!
//! Layout: `<root>/<stage>/<k[0..2]>/<k>.<ext>`, where `<root>` is normally
//! `.glassrip/cache`.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::Serialize;
use serde_json::Value;

use crate::atomic::{self, AtomicWriteError};
use crate::canonical::{self, CanonicalJsonError};

/// Domain separation tag hashed before the canonical JSON.
pub const CACHE_KEY_DOMAIN: &[u8] = b"glassrip.cache_key.v1\n";

/// Default cache directory relative to a workspace.
pub const DEFAULT_CACHE_DIR: &str = ".glassrip/cache";

/// Lock file in the cache root: runs hold it shared, `gc` holds it exclusively.
pub const CACHE_LOCK_FILE: &str = ".glassrip-cache.lock";

/// Directory in the cache root holding entries found to be corrupt.
pub const QUARANTINE_DIR: &str = ".quarantine";

/// Temp files and entries touched more recently than this are never collected.
pub const DEFAULT_GC_GRACE: Duration = Duration::from_secs(10 * 60);

const TEMP_PREFIX: &str = ".glassrip-tmp-";

/// Everything that determines a stage's output.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct CacheKeyParts {
    /// Stage name.
    pub stage: String,
    /// Stage implementation version (`STAGE_VERSION`).
    pub stage_version: u32,
    /// Stage params as JSON (canonicalized when hashing).
    pub params: Value,
    /// blake3 of every input (sorted when hashing, so order does not matter).
    pub input_hashes: Vec<String>,
    /// Model digest for model stages.
    pub model_digest: Option<String>,
    /// Hash of the prompt text.
    pub prompt_hash: Option<String>,
    /// Hash of the output JSON schema.
    pub schema_hash: Option<String>,
    /// Frame decoder (for example `software`, `videotoolbox`).
    pub decoder: Option<String>,
    /// Features mode (`production` or `prototype_compat`).
    pub features_mode: Option<String>,
    /// Tool versions (ffmpeg, Ollama server, glassrip, Cargo.lock hash, ...).
    pub tool_versions: BTreeMap<String, String>,
}

impl CacheKeyParts {
    /// Computes the cache key.
    pub fn key(&self) -> Result<CacheKey, CanonicalJsonError> {
        let mut normalized = self.clone();
        normalized.input_hashes.sort();
        let text = canonical::to_canonical_string(&normalized)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CACHE_KEY_DOMAIN);
        hasher.update(text.as_bytes());
        Ok(CacheKey(hasher.finalize().to_hex().to_string()))
    }
}

/// A 64-hex-character cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct CacheKey(String);

impl CacheKey {
    /// Parses a key from hex, validating its shape.
    pub fn parse(hex: &str) -> Option<Self> {
        (hex.len() == 64
            && hex
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
        .then(|| Self(hex.to_string()))
    }

    /// The hex string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// First two hex characters (the fan-out directory).
    pub fn prefix(&self) -> &str {
        &self.0[..2]
    }
}

impl std::fmt::Display for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Error from the cache.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// Stage names and extensions are restricted to `[a-z0-9_.-]` (no leading dot).
    #[error("invalid cache path component `{0}`")]
    InvalidComponent(String),
    /// Filesystem failure.
    #[error("cache I/O on {path}: {source}")]
    Io {
        /// Path involved.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: io::Error,
    },
    /// Atomic write failed.
    #[error(transparent)]
    Atomic(#[from] AtomicWriteError),
    /// Invalid age specification.
    #[error("invalid age `{0}` (expected a number followed by s, m, h, d, or w)")]
    InvalidAge(String),
    /// A run holds the cache; `gc` refuses to run concurrently with it.
    #[error("cache at {0} is in use by a running glassrip process; retry gc later")]
    Busy(PathBuf),
}

fn is_read_only(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem
    )
}

fn io_err(path: &Path) -> impl FnOnce(io::Error) -> CacheError + '_ {
    move |source| CacheError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn valid_component(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('.')
        && s.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-' || b == b'.'
        })
}

/// One cached file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    /// Stage name.
    pub stage: String,
    /// Key.
    pub key: CacheKey,
    /// Extension (without dot).
    pub ext: String,
    /// Full path.
    pub path: PathBuf,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Last write or read time.
    pub modified: SystemTime,
}

/// Result of [`Cache::gc`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GcReport {
    /// Entries removed.
    pub removed: Vec<CacheEntry>,
    /// Orphaned temp files removed.
    pub temp_files_removed: Vec<PathBuf>,
    /// Total bytes freed.
    pub bytes_freed: u64,
}

/// A shared hold on the cache for the duration of a run. While any lease is alive,
/// [`Cache::gc`] fails with [`CacheError::Busy`]. On a read-only cache the lease
/// holds no lock (nothing can be collected there anyway).
#[derive(Debug)]
pub struct CacheLease {
    file: Option<fs_err::File>,
}

impl CacheLease {
    /// True when a shared lock is actually held.
    pub fn is_locked(&self) -> bool {
        self.file.is_some()
    }
}

/// The on-disk cache.
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    /// A cache rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// A cache at `<workspace>/.glassrip/cache`.
    pub fn in_workspace(workspace: &Path) -> Self {
        Self::new(workspace.join(DEFAULT_CACHE_DIR))
    }

    /// Root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path where an entry lives (whether or not it exists).
    pub fn entry_path(
        &self,
        stage: &str,
        key: &CacheKey,
        ext: &str,
    ) -> Result<PathBuf, CacheError> {
        for c in [stage, ext] {
            if !valid_component(c) {
                return Err(CacheError::InvalidComponent(c.to_string()));
            }
        }
        Ok(self
            .root
            .join(stage)
            .join(key.prefix())
            .join(format!("{}.{ext}", key.as_str())))
    }

    /// True when the entry exists.
    pub fn exists(&self, stage: &str, key: &CacheKey, ext: &str) -> Result<bool, CacheError> {
        let path = self.entry_path(stage, key, ext)?;
        atomic::is_file(&path).map_err(io_err(&path))
    }

    /// Takes a shared lease on the cache (blocks while a `gc` is running).
    pub fn lease(&self) -> Result<CacheLease, CacheError> {
        match fs_err::create_dir_all(&self.root) {
            Ok(()) => {}
            Err(e) if is_read_only(&e) => return Ok(CacheLease { file: None }),
            Err(e) => return Err(io_err(&self.root)(e)),
        }
        let lock_path = self.root.join(CACHE_LOCK_FILE);
        let file = match fs_err::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
        {
            Ok(f) => f,
            Err(e) if is_read_only(&e) => match fs_err::File::open(&lock_path) {
                Ok(f) => f,
                Err(_) => return Ok(CacheLease { file: None }),
            },
            Err(e) => return Err(io_err(&lock_path)(e)),
        };
        fs4::fs_err3::FileExt::lock_shared(&file).map_err(io_err(&lock_path))?;
        Ok(CacheLease { file: Some(file) })
    }

    /// Moves a corrupt entry into `<root>/.quarantine/` so it is no longer served,
    /// returning its new path (or `None` if the entry was already gone).
    pub fn quarantine(
        &self,
        stage: &str,
        key: &CacheKey,
        ext: &str,
    ) -> Result<Option<PathBuf>, CacheError> {
        let path = self.entry_path(stage, key, ext)?;
        if atomic::metadata_opt(&path)
            .map_err(io_err(&path))?
            .is_none()
        {
            return Ok(None);
        }
        let dir = self.root.join(QUARANTINE_DIR);
        fs_err::create_dir_all(&dir).map_err(io_err(&dir))?;
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dest = dir.join(format!("{stage}-{}.{ext}.{stamp}", key.as_str()));
        fs_err::rename(&path, &dest).map_err(io_err(&path))?;
        Ok(Some(dest))
    }

    /// Returns the entry's path if present, refreshing its modification time so
    /// `gc --older-than` measures time since last use.
    pub fn get_path(
        &self,
        stage: &str,
        key: &CacheKey,
        ext: &str,
    ) -> Result<Option<PathBuf>, CacheError> {
        let path = self.entry_path(stage, key, ext)?;
        if !atomic::is_file(&path).map_err(io_err(&path))? {
            return Ok(None);
        }
        // Refresh the mtime for gc; a read-only cache still serves hits.
        let touched = fs_err::OpenOptions::new()
            .write(true)
            .open(&path)
            .and_then(|f| f.file().set_modified(SystemTime::now()));
        match touched {
            Ok(()) => {}
            Err(e) if is_read_only(&e) => {
                tracing::debug!(path = %path.display(), "cache is read-only; not refreshing mtime");
            }
            Err(e) => return Err(io_err(&path)(e)),
        }
        Ok(Some(path))
    }

    /// Reads an entry's bytes if present.
    pub fn get(
        &self,
        stage: &str,
        key: &CacheKey,
        ext: &str,
    ) -> Result<Option<Vec<u8>>, CacheError> {
        match self.get_path(stage, key, ext)? {
            Some(path) => Ok(Some(fs_err::read(&path).map_err(io_err(&path))?)),
            None => Ok(None),
        }
    }

    /// Stores bytes atomically and returns the entry path.
    pub fn put(
        &self,
        stage: &str,
        key: &CacheKey,
        ext: &str,
        bytes: &[u8],
    ) -> Result<PathBuf, CacheError> {
        let path = self.entry_path(stage, key, ext)?;
        atomic::write_atomic(&path, bytes)?;
        Ok(path)
    }

    /// Copies a file into the cache atomically and returns the entry path.
    pub fn put_file(
        &self,
        stage: &str,
        key: &CacheKey,
        ext: &str,
        src: &Path,
    ) -> Result<PathBuf, CacheError> {
        let path = self.entry_path(stage, key, ext)?;
        atomic::copy_atomic(src, &path)?;
        Ok(path)
    }

    /// Lists every entry, sorted by stage then key.
    pub fn ls(&self) -> Result<Vec<CacheEntry>, CacheError> {
        let mut out = Vec::new();
        if !atomic::is_dir(&self.root).map_err(io_err(&self.root))? {
            return Ok(out);
        }
        for stage_dir in read_dir_sorted(&self.root)? {
            let Some(stage) = file_name(&stage_dir) else {
                continue;
            };
            if !valid_component(&stage)
                || !atomic::is_dir(&stage_dir).map_err(io_err(&stage_dir))?
            {
                continue;
            }
            for prefix_dir in read_dir_sorted(&stage_dir)? {
                if !atomic::is_dir(&prefix_dir).map_err(io_err(&prefix_dir))? {
                    continue;
                }
                for file in read_dir_sorted(&prefix_dir)? {
                    let Some(name) = file_name(&file) else {
                        continue;
                    };
                    let Some((key, ext)) = name.split_once('.') else {
                        continue;
                    };
                    let Some(key) = CacheKey::parse(key) else {
                        continue;
                    };
                    let meta = fs_err::metadata(&file).map_err(io_err(&file))?;
                    if !meta.is_file() {
                        continue;
                    }
                    out.push(CacheEntry {
                        stage: stage.clone(),
                        key,
                        ext: ext.to_string(),
                        path: file.clone(),
                        size_bytes: meta.len(),
                        modified: meta.modified().map_err(io_err(&file))?,
                    });
                }
            }
        }
        Ok(out)
    }

    /// Removes entries not written or read within `older_than` of `now`, and orphaned
    /// temp files older than [`DEFAULT_GC_GRACE`].
    ///
    /// Takes the cache lock exclusively and fails with [`CacheError::Busy`] while any
    /// run holds a [`CacheLease`]. Entries touched within the grace window are kept
    /// even when `older_than` is shorter.
    pub fn gc(&self, older_than: Duration, now: SystemTime) -> Result<GcReport, CacheError> {
        let mut report = GcReport::default();
        if !atomic::is_dir(&self.root).map_err(io_err(&self.root))? {
            return Ok(report);
        }
        let lock_path = self.root.join(CACHE_LOCK_FILE);
        let lock = fs_err::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(io_err(&lock_path))?;
        match fs4::fs_err3::FileExt::try_lock(&lock) {
            Ok(()) => {}
            Err(fs4::TryLockError::WouldBlock) => return Err(CacheError::Busy(self.root.clone())),
            Err(fs4::TryLockError::Error(e)) => return Err(io_err(&lock_path)(e)),
        }

        let before = |age: Duration| now.checked_sub(age).unwrap_or(SystemTime::UNIX_EPOCH);
        let entry_cutoff = before(older_than.max(DEFAULT_GC_GRACE));
        for entry in self.ls()? {
            if entry.modified < entry_cutoff {
                fs_err::remove_file(&entry.path).map_err(io_err(&entry.path))?;
                report.bytes_freed += entry.size_bytes;
                report.removed.push(entry);
            }
        }
        let temp_cutoff = before(DEFAULT_GC_GRACE);
        for path in self.temp_files()? {
            let meta = fs_err::metadata(&path).map_err(io_err(&path))?;
            if meta.modified().map_err(io_err(&path))? < temp_cutoff {
                fs_err::remove_file(&path).map_err(io_err(&path))?;
                report.bytes_freed += meta.len();
                report.temp_files_removed.push(path);
            }
        }
        drop(lock);
        Ok(report)
    }

    fn temp_files(&self) -> Result<Vec<PathBuf>, CacheError> {
        let mut out = Vec::new();
        let mut dirs = vec![(self.root.clone(), 0usize)];
        while let Some((dir, depth)) = dirs.pop() {
            for path in read_dir_sorted(&dir)? {
                let name = file_name(&path).unwrap_or_default();
                let meta = fs_err::symlink_metadata(&path).map_err(io_err(&path))?;
                if meta.is_file() && name.starts_with(TEMP_PREFIX) {
                    out.push(path);
                } else if meta.is_dir() && depth < 2 && name != QUARANTINE_DIR {
                    dirs.push((path, depth + 1));
                }
            }
        }
        Ok(out)
    }
}

fn file_name(p: &Path) -> Option<String> {
    p.file_name().map(|n| n.to_string_lossy().into_owned())
}

fn read_dir_sorted(dir: &Path) -> Result<Vec<PathBuf>, CacheError> {
    let mut entries = Vec::new();
    for entry in fs_err::read_dir(dir).map_err(io_err(dir))? {
        entries.push(entry.map_err(io_err(dir))?.path());
    }
    entries.sort();
    Ok(entries)
}

/// Parses an age such as `30d`, `12h`, `90m`, `45s`, or `2w`.
pub fn parse_age(spec: &str) -> Result<Duration, CacheError> {
    let spec = spec.trim();
    let bad = || CacheError::InvalidAge(spec.to_string());
    let unit_at = spec.find(|c: char| !c.is_ascii_digit()).ok_or_else(bad)?;
    let (num, unit) = spec.split_at(unit_at);
    let n: u64 = num.parse().map_err(|_| bad())?;
    let secs_per = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 3_600,
        "d" => 86_400,
        "w" => 604_800,
        _ => return Err(bad()),
    };
    n.checked_mul(secs_per)
        .map(Duration::from_secs)
        .ok_or_else(bad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base() -> CacheKeyParts {
        CacheKeyParts {
            stage: "frames".into(),
            stage_version: 1,
            params: json!({"interval_s": 2.0, "scale": {"w": 1920}}),
            input_hashes: vec!["aa".into(), "bb".into()],
            model_digest: Some("sha256:0001".into()),
            prompt_hash: Some("p1".into()),
            schema_hash: Some("s1".into()),
            decoder: Some("software".into()),
            features_mode: Some("production".into()),
            tool_versions: BTreeMap::from([("ffmpeg".into(), "7.1".into())]),
        }
    }

    #[test]
    fn key_is_stable_and_order_independent() {
        let a = base().key().unwrap();
        assert_eq!(a, base().key().unwrap());
        let mut reordered = base();
        reordered.input_hashes.reverse();
        reordered.params =
            serde_json::from_str(r#"{"scale":{"w":1920},"interval_s":2.0}"#).unwrap();
        assert_eq!(a, reordered.key().unwrap());
        assert_eq!(a.as_str().len(), 64);
    }

    #[test]
    fn every_component_changes_key() {
        let k0 = base().key().unwrap();
        type Mutation = Box<dyn Fn(&mut CacheKeyParts)>;
        let mutations: Vec<(&str, Mutation)> = vec![
            ("stage", Box::new(|p| p.stage = "features".into())),
            ("stage_version", Box::new(|p| p.stage_version = 2)),
            ("params", Box::new(|p| p.params["interval_s"] = json!(1.0))),
            (
                "nested params",
                Box::new(|p| p.params["scale"]["w"] = json!(1280)),
            ),
            (
                "input added",
                Box::new(|p| p.input_hashes.push("cc".into())),
            ),
            (
                "input changed",
                Box::new(|p| p.input_hashes[0] = "ab".into()),
            ),
            (
                "model_digest",
                Box::new(|p| p.model_digest = Some("sha256:0002".into())),
            ),
            ("model_digest none", Box::new(|p| p.model_digest = None)),
            (
                "prompt_hash",
                Box::new(|p| p.prompt_hash = Some("p2".into())),
            ),
            (
                "schema_hash",
                Box::new(|p| p.schema_hash = Some("s2".into())),
            ),
            (
                "decoder",
                Box::new(|p| p.decoder = Some("videotoolbox".into())),
            ),
            (
                "features_mode",
                Box::new(|p| p.features_mode = Some("prototype_compat".into())),
            ),
            (
                "tool version",
                Box::new(|p| {
                    p.tool_versions.insert("ffmpeg".into(), "7.2".into());
                }),
            ),
            (
                "tool added",
                Box::new(|p| {
                    p.tool_versions.insert("ollama".into(), "0.34.3".into());
                }),
            ),
        ];
        let mut seen = std::collections::HashSet::new();
        seen.insert(k0.clone());
        for (name, mutate) in mutations {
            let mut p = base();
            mutate(&mut p);
            let k = p.key().unwrap();
            assert_ne!(k, k0, "changing {name} did not change the key");
            assert!(
                seen.insert(k),
                "changing {name} collided with another mutation"
            );
        }
    }

    #[test]
    fn component_boundaries_are_unambiguous() {
        let mut a = base();
        a.prompt_hash = Some("ab".into());
        a.schema_hash = Some("c".into());
        let mut b = base();
        b.prompt_hash = Some("a".into());
        b.schema_hash = Some("bc".into());
        assert_ne!(a.key().unwrap(), b.key().unwrap());
    }

    #[test]
    fn layout_get_put_exists() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::in_workspace(dir.path());
        let key = base().key().unwrap();
        assert!(!cache.exists("frames", &key, "jsonl").unwrap());
        assert_eq!(cache.get("frames", &key, "jsonl").unwrap(), None);
        let path = cache.put("frames", &key, "jsonl", b"data").unwrap();
        let expected = dir
            .path()
            .join(".glassrip/cache/frames")
            .join(&key.as_str()[..2])
            .join(format!("{key}.jsonl"));
        assert_eq!(path, expected);
        assert!(cache.exists("frames", &key, "jsonl").unwrap());
        assert_eq!(
            cache.get("frames", &key, "jsonl").unwrap().as_deref(),
            Some(&b"data"[..])
        );
        assert!(matches!(
            cache.put("../evil", &key, "jsonl", b"x"),
            Err(CacheError::InvalidComponent(_))
        ));
    }

    #[test]
    fn ls_and_gc() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path().join("c"));
        let mut p = base();
        let k1 = p.key().unwrap();
        p.stage_version = 9;
        let k2 = p.key().unwrap();
        let old = cache.put("frames", &k1, "jsonl", b"old").unwrap();
        cache.put("keyframes", &k2, "json", b"newer").unwrap();
        // Stray temp files and foreign names are ignored.
        fs_err::write(old.with_file_name(".glassrip-tmp-x"), b"").unwrap();

        let entries = cache.ls().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].stage, "frames");
        assert_eq!(entries[1].size_bytes, 5);

        let past = SystemTime::now() - Duration::from_secs(40 * 86_400);
        fs_err::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .file()
            .set_modified(past)
            .unwrap();
        let report = cache
            .gc(parse_age("30d").unwrap(), SystemTime::now())
            .unwrap();
        assert_eq!(report.removed.len(), 1);
        assert_eq!(report.removed[0].key, k1);
        assert_eq!(report.bytes_freed, 3);
        assert_eq!(cache.ls().unwrap().len(), 1);
    }

    #[test]
    fn get_refreshes_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path());
        let key = base().key().unwrap();
        let path = cache.put("frames", &key, "jsonl", b"x").unwrap();
        let past = SystemTime::now() - Duration::from_secs(100 * 86_400);
        fs_err::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .file()
            .set_modified(past)
            .unwrap();
        cache.get_path("frames", &key, "jsonl").unwrap();
        let report = cache
            .gc(Duration::from_secs(86_400), SystemTime::now())
            .unwrap();
        assert!(report.removed.is_empty());
    }

    fn set_mtime(path: &Path, t: SystemTime) {
        fs_err::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .file()
            .set_modified(t)
            .unwrap();
    }

    #[test]
    fn gc_refuses_while_leased() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path().join("c"));
        let key = base().key().unwrap();
        let path = cache.put("frames", &key, "jsonl", b"x").unwrap();
        set_mtime(&path, SystemTime::now() - Duration::from_secs(90 * 86_400));
        let lease = cache.lease().unwrap();
        assert!(lease.is_locked());
        let second = cache.lease().unwrap();
        assert!(matches!(
            cache.gc(Duration::from_secs(86_400), SystemTime::now()),
            Err(CacheError::Busy(_))
        ));
        drop(lease);
        drop(second);
        let report = cache
            .gc(Duration::from_secs(86_400), SystemTime::now())
            .unwrap();
        assert_eq!(report.removed.len(), 1);
    }

    #[test]
    fn gc_grace_window_and_temp_files() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path().join("c"));
        let key = base().key().unwrap();
        let fresh = cache.put("frames", &key, "jsonl", b"fresh").unwrap();
        let old_tmp = fresh.with_file_name(".glassrip-tmp-old");
        let new_tmp = fresh.with_file_name(".glassrip-tmp-new");
        fs_err::write(&old_tmp, b"abc").unwrap();
        fs_err::write(&new_tmp, b"abc").unwrap();
        set_mtime(&old_tmp, SystemTime::now() - Duration::from_secs(3_600));
        // older_than of zero still spares entries inside the grace window.
        let report = cache.gc(Duration::ZERO, SystemTime::now()).unwrap();
        assert!(report.removed.is_empty());
        assert_eq!(report.temp_files_removed, vec![old_tmp.clone()]);
        assert!(!old_tmp.exists() && new_tmp.exists() && fresh.exists());
    }

    #[test]
    fn quarantine_hides_entry() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path().join("c"));
        let key = base().key().unwrap();
        cache.put("frames", &key, "jsonl", b"garbage").unwrap();
        let moved = cache.quarantine("frames", &key, "jsonl").unwrap().unwrap();
        assert!(moved.starts_with(dir.path().join("c").join(QUARANTINE_DIR)));
        assert!(!cache.exists("frames", &key, "jsonl").unwrap());
        assert!(cache.ls().unwrap().is_empty());
        assert_eq!(cache.quarantine("frames", &key, "jsonl").unwrap(), None);
    }

    #[cfg(unix)]
    #[test]
    fn read_only_cache_serves_hits() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path().join("c"));
        let key = base().key().unwrap();
        let path = cache.put("frames", &key, "jsonl", b"data").unwrap();
        let ro = std::fs::Permissions::from_mode(0o444);
        fs_err::set_permissions(&path, ro).unwrap();
        let got = cache.get("frames", &key, "jsonl");
        fs_err::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(got.unwrap().as_deref(), Some(&b"data"[..]));
    }

    #[test]
    fn age_parsing() {
        assert_eq!(parse_age("30d").unwrap(), Duration::from_secs(30 * 86_400));
        assert_eq!(parse_age("2w").unwrap(), Duration::from_secs(14 * 86_400));
        assert_eq!(parse_age("45s").unwrap(), Duration::from_secs(45));
        for bad in ["", "d", "30", "30x", "-1d", "1.5h"] {
            assert!(parse_age(bad).is_err(), "{bad}");
        }
    }
}
