//! Model manifest: download URLs, SHA-256 hashes and licenses.
//!
//! Model binaries are never committed; `models.toml` pins each file to an exact
//! upstream revision and hash. Files live under `$GLASSRIP_MODELS_DIR` or
//! `~/.glassrip/models`, at the manifest's relative `path`.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
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
            assert!(e.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
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
