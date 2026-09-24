//! Model files and their pinned hashes.
//!
//! Files come from the `monkt/paddleocr-onnx` Hugging Face repository at
//! revision `7b02d0a30a07ba2b92ad1ff5a8941ae2c633de65` (Apache-2.0, converted
//! from PaddleOCR): `detection/v5/det.onnx` (PP-OCRv5 server detector),
//! `languages/english/rec.onnx` (English PP-OCRv5 mobile recognizer) and
//! `languages/english/dict.txt`. They live under
//! `$GLASSRIP_MODELS_DIR/ppocrv5/` or `~/.glassrip/models/ppocrv5/`.

use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::OcrError;

pub const DET_FILE: &str = "det.onnx";
pub const REC_FILE: &str = "rec.onnx";
pub const DICT_FILE: &str = "dict.txt";

/// Pinned SHA-256 of the detector.
pub const DET_SHA256: &str = "61824840edf6e74581898930b8091b1b2318f4b2705a2e8a40ad3de7ac480133";
/// Pinned SHA-256 of the recognizer.
pub const REC_SHA256: &str = "4e16deb22c4da6468bdca539b2cd3c8687825538b67109177c47d359ab994cd7";

const HINT: &str = "Download PP-OCRv5 from huggingface.co/monkt/paddleocr-onnx: \
detection/v5/det.onnx -> det.onnx, languages/english/rec.onnx -> rec.onnx, \
languages/english/dict.txt -> dict.txt";

/// Default model directory.
pub fn default_dir() -> PathBuf {
    let base = std::env::var_os("GLASSRIP_MODELS_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".glassrip/models")))
        .unwrap_or_else(|| PathBuf::from(".glassrip/models"));
    base.join("ppocrv5")
}

/// Fail with the download hint when a model file is missing.
pub fn check_present(dir: &Path) -> Result<(), OcrError> {
    let missing: Vec<&str> = [DET_FILE, REC_FILE, DICT_FILE]
        .into_iter()
        .filter(|f| !dir.join(f).is_file())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(OcrError::ModelsMissing {
            dir: dir.display().to_string(),
            missing: missing.join(", "),
            hint: HINT,
        })
    }
}

/// SHA-256 of a file as lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String, OcrError> {
    let io = |source| OcrError::Io {
        path: path.display().to_string(),
        source,
    };
    let mut file = std::fs::File::open(path).map_err(io)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf).map_err(io)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Check both ONNX files against their pinned hashes; returns a fingerprint.
pub fn verify(dir: &Path) -> Result<String, OcrError> {
    check_present(dir)?;
    let mut parts = Vec::new();
    for (file, expected) in [(DET_FILE, DET_SHA256), (REC_FILE, REC_SHA256)] {
        let actual = sha256_file(&dir.join(file))?;
        if actual != expected {
            return Err(OcrError::ModelHashMismatch {
                file: file.to_string(),
                actual,
                expected: expected.to_string(),
            });
        }
        parts.push(actual);
    }
    parts.push(sha256_file(&dir.join(DICT_FILE))?);
    Ok(parts.join(":"))
}

/// Read the recognizer dictionary (one symbol per line).
pub fn read_dict(dir: &Path) -> Result<Vec<String>, OcrError> {
    let path = dir.join(DICT_FILE);
    let text = std::fs::read_to_string(&path).map_err(|source| OcrError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(text
        .lines()
        .map(|l| l.trim_end_matches('\r').to_string())
        .collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn missing_files_are_listed_with_hint() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join(DICT_FILE), "a\nb\n")?;
        let err = check_present(dir.path()).err().map(|e| e.to_string());
        let msg = err.unwrap_or_default();
        assert!(msg.contains("det.onnx, rec.onnx"), "{msg}");
        assert!(msg.contains("paddleocr-onnx"));
        assert_eq!(read_dict(dir.path())?, vec!["a", "b"]);
        Ok(())
    }

    #[test]
    fn hash_mismatch_is_reported() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        for f in [DET_FILE, REC_FILE, DICT_FILE] {
            std::fs::write(dir.path().join(f), "not a model")?;
        }
        assert!(matches!(
            verify(dir.path()),
            Err(OcrError::ModelHashMismatch { .. })
        ));
        Ok(())
    }
}
