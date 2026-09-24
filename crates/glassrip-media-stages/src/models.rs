//! Pinned model files (`models.toml`): URL + SHA-256, never committed.
//!
//! Files live under `$GLASSRIP_MODELS_DIR` or `~/.glassrip/models` at the listed relative
//! paths. [`ensure`] verifies a file and, when allowed, downloads it with `curl` into a
//! temporary file that is renamed into place only after its hash matches.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const MANIFEST: &str = include_str!("../models.toml");

/// Name of the orientation model in the manifest.
pub const ORIENT_MODEL: &str = "pp-lcnet-x1-0-doc-ori";
/// Text-line detector for the 180 degree confirmation.
pub const OCR_DET_MODEL: &str = "pp-ocrv3-det";
/// PP-OCRv5 recognizer for the 180 degree confirmation.
pub const OCR_REC_MODEL: &str = "en-pp-ocrv5-mobile-rec";
/// `ocrs` detection model (fallback).
pub const OCRS_DET_MODEL: &str = "ocrs-text-detection";
/// `ocrs` recognition model (fallback).
pub const OCRS_REC_MODEL: &str = "ocrs-text-recognition";

/// Error resolving a model.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// The embedded manifest is malformed (a build problem).
    #[error("models.toml is invalid: {0}")]
    Manifest(String),
    /// Unknown model name.
    #[error("model `{0}` is not in models.toml")]
    Unknown(String),
    /// File missing and downloads disabled.
    #[error("model file {path} is missing; download it from {url} (SHA-256 {sha256})")]
    Missing {
        /// Expected path.
        path: PathBuf,
        /// Source URL.
        url: String,
        /// Expected hash.
        sha256: String,
    },
    /// Hash mismatch.
    #[error("model file {path} has SHA-256 {actual}, expected {expected}")]
    HashMismatch {
        /// File.
        path: PathBuf,
        /// Expected.
        expected: String,
        /// Actual.
        actual: String,
    },
    /// Download or filesystem failure.
    #[error("model {what}: {message}")]
    Io {
        /// Operation.
        what: String,
        /// Details.
        message: String,
    },
}

/// One manifest entry.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelFile {
    /// Name.
    pub name: String,
    /// Path relative to the models directory.
    pub path: String,
    /// Pinned download URL.
    pub url: String,
    /// Expected SHA-256.
    pub sha256: String,
    /// License of the model file.
    pub license: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    model: Vec<ModelFile>,
}

/// All manifest entries.
pub fn manifest() -> Result<Vec<ModelFile>, ModelError> {
    toml::from_str::<Manifest>(MANIFEST)
        .map(|m| m.model)
        .map_err(|e| ModelError::Manifest(e.to_string()))
}

/// Entry by name.
pub fn entry(name: &str) -> Result<ModelFile, ModelError> {
    manifest()?
        .into_iter()
        .find(|m| m.name == name)
        .ok_or_else(|| ModelError::Unknown(name.to_string()))
}

/// `$GLASSRIP_MODELS_DIR`, else `~/.glassrip/models`, else `.glassrip/models`.
pub fn default_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("GLASSRIP_MODELS_DIR") {
        return PathBuf::from(d);
    }
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from(".glassrip/models"),
        |h| PathBuf::from(h).join(".glassrip/models"),
    )
}

/// SHA-256 of a file as lowercase hex.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = fs_err::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Checks `dir/entry.path` against the pinned hash.
pub fn verify(dir: &Path, entry: &ModelFile) -> Result<PathBuf, ModelError> {
    let path = dir.join(&entry.path);
    if !path.is_file() {
        return Err(ModelError::Missing {
            path,
            url: entry.url.clone(),
            sha256: entry.sha256.clone(),
        });
    }
    let actual = sha256_file(&path).map_err(|e| ModelError::Io {
        what: format!("read {}", path.display()),
        message: e.to_string(),
    })?;
    if actual != entry.sha256 {
        return Err(ModelError::HashMismatch {
            path,
            expected: entry.sha256.clone(),
            actual,
        });
    }
    Ok(path)
}

/// Verifies the model, downloading it first when missing and `download` is set.
pub fn ensure(dir: &Path, entry: &ModelFile, download: bool) -> Result<PathBuf, ModelError> {
    match verify(dir, entry) {
        Err(ModelError::Missing { .. }) if download => {}
        other => return other,
    }
    let dest = dir.join(&entry.path);
    let parent = dest.parent().unwrap_or(dir).to_path_buf();
    fs_err::create_dir_all(&parent).map_err(|e| ModelError::Io {
        what: "create directory".into(),
        message: e.to_string(),
    })?;
    let tmp = tempfile::NamedTempFile::new_in(&parent).map_err(|e| ModelError::Io {
        what: "create temp file".into(),
        message: e.to_string(),
    })?;
    tracing::info!(model = %entry.name, url = %entry.url, "downloading model");
    let status = std::process::Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(tmp.path())
        .arg(&entry.url)
        .status()
        .map_err(|e| ModelError::Io {
            what: "run curl".into(),
            message: e.to_string(),
        })?;
    if !status.success() {
        return Err(ModelError::Io {
            what: format!("download {}", entry.url),
            message: format!("curl exited with {status}"),
        });
    }
    let actual = sha256_file(tmp.path()).map_err(|e| ModelError::Io {
        what: "hash download".into(),
        message: e.to_string(),
    })?;
    if actual != entry.sha256 {
        return Err(ModelError::HashMismatch {
            path: dest,
            expected: entry.sha256.clone(),
            actual,
        });
    }
    tmp.persist(&dest).map_err(|e| ModelError::Io {
        what: format!("persist {}", dest.display()),
        message: e.to_string(),
    })?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_pins_orientation_model() {
        let e = entry(ORIENT_MODEL).unwrap();
        assert_eq!(e.sha256.len(), 64);
        assert!(e.url.starts_with("https://") && e.url.contains("/resolve/"));
        for n in [OCR_DET_MODEL, OCR_REC_MODEL, OCRS_DET_MODEL, OCRS_REC_MODEL] {
            let e = entry(n).unwrap();
            assert_eq!(e.sha256.len(), 64, "{n}");
            assert!(e.url.starts_with("https://"), "{n}");
        }
        assert!(entry("nope").is_err());
    }

    #[test]
    fn verify_reports_missing_and_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let mut e = entry(ORIENT_MODEL).unwrap();
        assert!(matches!(
            verify(dir.path(), &e),
            Err(ModelError::Missing { .. })
        ));
        let p = dir.path().join(&e.path);
        fs_err::create_dir_all(p.parent().unwrap()).unwrap();
        fs_err::write(&p, b"not a model").unwrap();
        assert!(matches!(
            verify(dir.path(), &e),
            Err(ModelError::HashMismatch { .. })
        ));
        e.sha256 = sha256_file(&p).unwrap();
        assert_eq!(verify(dir.path(), &e).unwrap(), p);
    }
}
