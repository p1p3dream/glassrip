//! Model manifest: download URLs, SHA-256 hashes and licenses.
//!
//! Model binaries are never committed; `models.toml` pins each file to an exact
//! upstream revision and hash. Files live under `$GLASSRIP_MODELS_DIR` or
//! `~/.glassrip/models`, at the manifest's relative `path`.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{AudioError, Result};

const MANIFEST: &str = include_str!("../models.toml");

/// One file in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelFile {
    /// Short name used by callers.
    pub name: String,
    /// Path relative to the models directory.
    pub path: String,
    /// Download URL pinned to a revision.
    pub url: String,
    /// Lowercase hex SHA-256.
    pub sha256: String,
    /// SPDX-style license id.
    pub license: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    model: Vec<ModelFile>,
}

/// Parse the embedded manifest.
pub fn manifest() -> Result<Vec<ModelFile>> {
    let m: Manifest = toml::from_str(MANIFEST).map_err(|e| AudioError::Parse {
        what: "models.toml".into(),
        message: e.to_string(),
    })?;
    Ok(m.model)
}

/// Look up a manifest entry by name.
pub fn find(name: &str) -> Result<ModelFile> {
    manifest()?
        .into_iter()
        .find(|m| m.name == name)
        .ok_or_else(|| AudioError::Model {
            name: name.into(),
            message: "not in models.toml".into(),
        })
}

/// Models directory: `$GLASSRIP_MODELS_DIR`, else `$HOME/.glassrip/models`.
pub fn models_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("GLASSRIP_MODELS_DIR") {
        return Some(PathBuf::from(d));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".glassrip").join("models"))
}

/// Hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).map_err(|e| AudioError::io(path, e))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| AudioError::io(path, e))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Check a file against its manifest hash.
pub fn verify(dir: &Path, entry: &ModelFile) -> Result<PathBuf> {
    let path = dir.join(&entry.path);
    if !path.is_file() {
        return Err(AudioError::Model {
            name: entry.name.clone(),
            message: format!("missing {}; download from {}", path.display(), entry.url),
        });
    }
    let got = sha256_file(&path)?;
    if got != entry.sha256 {
        return Err(AudioError::Model {
            name: entry.name.clone(),
            message: format!("sha256 mismatch for {}: got {got}", path.display()),
        });
    }
    Ok(path)
}

/// Name of the per-directory verification cache.
pub const VERIFY_CACHE: &str = ".glassrip-verified.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CacheEntry {
    size: u64,
    mtime_ns: u128,
    sha256: String,
}

fn file_stamp(path: &Path) -> Result<(u64, u128)> {
    let md = std::fs::metadata(path).map_err(|e| AudioError::io(path, e))?;
    let mtime = md
        .modified()
        .map_err(|e| AudioError::io(path, e))?
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Ok((md.len(), mtime))
}

/// SHA-256 of `path`, reusing a cached value while size and mtime are unchanged.
///
/// The cache is `.glassrip-verified.json` next to the file. Failing to write
/// the cache is not an error; the hash is simply recomputed next time.
pub fn cached_sha256(path: &Path) -> Result<String> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or_else(|| AudioError::InvalidInput(format!("no file name in {}", path.display())))?;
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let cache_path = dir.join(VERIFY_CACHE);
    let mut cache: BTreeMap<String, CacheEntry> = std::fs::read_to_string(&cache_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let (size, mtime_ns) = file_stamp(path)?;
    if let Some(e) = cache.get(&name) {
        if e.size == size && e.mtime_ns == mtime_ns {
            return Ok(e.sha256.clone());
        }
    }
    let sha256 = sha256_file(path)?;
    cache.insert(
        name,
        CacheEntry {
            size,
            mtime_ns,
            sha256: sha256.clone(),
        },
    );
    if let Ok(text) = serde_json::to_string_pretty(&cache) {
        let tmp = dir.join(format!("{VERIFY_CACHE}.{}.tmp", std::process::id()));
        if std::fs::write(&tmp, text).is_ok() && std::fs::rename(&tmp, &cache_path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
    Ok(sha256)
}

fn manifest_entries_named(entries: &[ModelFile], file_name: &str) -> Vec<ModelFile> {
    entries
        .iter()
        .filter(|e| {
            Path::new(&e.path)
                .file_name()
                .is_some_and(|n| n == file_name)
        })
        .cloned()
        .collect()
}

/// Check a model file against the manifest entry with the same file name.
///
/// Errors when the file is not in the manifest or its hash differs.
pub fn verify_model_file(path: &Path) -> Result<()> {
    verify_model_file_in(&manifest()?, path)
}

fn verify_model_file_in(entries: &[ModelFile], path: &Path) -> Result<()> {
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let candidates = manifest_entries_named(entries, &file_name);
    if candidates.is_empty() {
        return Err(AudioError::Model {
            name: path.display().to_string(),
            message: "not listed in models.toml; disable verification to use it".into(),
        });
    }
    let got = cached_sha256(path)?;
    if candidates.iter().any(|e| e.sha256 == got) {
        Ok(())
    } else {
        Err(AudioError::Model {
            name: path.display().to_string(),
            message: format!(
                "sha256 mismatch: got {got}, expected {}",
                candidates[0].sha256
            ),
        })
    }
}

/// Check every manifest file under `speakrs/` that exists in `dir`.
///
/// Returns how many files were checked; errors on the first mismatch.
pub fn verify_speakrs_dir(dir: &Path) -> Result<usize> {
    verify_dir_in(&manifest()?, dir, "speakrs/")
}

fn verify_dir_in(entries: &[ModelFile], dir: &Path, prefix: &str) -> Result<usize> {
    let mut n = 0;
    for e in entries.iter().filter(|e| e.path.starts_with(prefix)) {
        let Some(name) = Path::new(&e.path).file_name() else {
            continue;
        };
        let p = dir.join(name);
        if p.is_file() {
            verify_model_file_in(entries, &p)?;
            n += 1;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn manifest_is_well_formed() {
        let m = manifest().unwrap();
        assert!(m.len() >= 3);
        for e in &m {
            assert_eq!(e.sha256.len(), 64, "{}", e.name);
            assert!(e
                .sha256
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
            assert!(e.url.starts_with("https://"), "{}", e.name);
            assert!(!e.path.starts_with('/'), "{}", e.name);
        }
        let mut names: Vec<&str> = m.iter().map(|e| e.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), m.len(), "duplicate names");
        assert!(find("whisper-large-v3-turbo").is_ok());
        assert!(find("silero-vad-v6.2.0").is_ok());
    }

    fn entry(name: &str, path: &str, sha: &str) -> ModelFile {
        ModelFile {
            name: name.into(),
            path: path.into(),
            url: "https://example.invalid/x".into(),
            sha256: sha.into(),
            license: "MIT".into(),
        }
    }

    const ABC_SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn model_file_verification_uses_and_refreshes_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.bin");
        std::fs::write(&p, b"abc").unwrap();
        let entries = vec![entry("m", "m.bin", ABC_SHA)];
        verify_model_file_in(&entries, &p).unwrap();
        let cache = std::fs::read_to_string(dir.path().join(VERIFY_CACHE)).unwrap();
        assert!(cache.contains(ABC_SHA));
        // content change with a new size is detected despite the cache
        std::fs::write(&p, b"abcd").unwrap();
        assert!(verify_model_file_in(&entries, &p).is_err());
        // unknown files are rejected
        let q = dir.path().join("other.bin");
        std::fs::write(&q, b"abc").unwrap();
        assert!(verify_model_file_in(&entries, &q).is_err());
    }

    #[test]
    fn directory_verification_checks_present_prefixed_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.onnx"), b"abc").unwrap();
        let entries = vec![
            entry("speakrs/a.onnx", "speakrs/a.onnx", ABC_SHA),
            entry("speakrs/missing.onnx", "speakrs/missing.onnx", ABC_SHA),
            entry("other", "other.bin", ABC_SHA),
        ];
        assert_eq!(verify_dir_in(&entries, dir.path(), "speakrs/").unwrap(), 1);
        std::fs::write(dir.path().join("a.onnx"), b"xyz").unwrap();
        assert!(verify_dir_in(&entries, dir.path(), "speakrs/").is_err());
    }

    #[test]
    fn verify_detects_mismatch_and_accepts_match() {
        let dir = tempfile::tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join("x.bin")).unwrap();
        f.write_all(b"abc").unwrap();
        let mut e = ModelFile {
            name: "x".into(),
            path: "x.bin".into(),
            url: "https://example.invalid/x.bin".into(),
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            license: "MIT".into(),
        };
        assert!(verify(dir.path(), &e).is_ok());
        e.sha256 = "0".repeat(64);
        assert!(verify(dir.path(), &e).is_err());
        e.path = "missing.bin".into();
        assert!(verify(dir.path(), &e).is_err());
    }
}
