//! Writes externally produced data as core JSONL input artifacts.
//!
//! Bridge until the upstream stages run on the core runner: the audio crate
//! still writes whole-file JSON envelopes, and board state comes from another
//! crate. Each import becomes `<run>/artifacts/<schema>.jsonl` with ok records.

use std::path::{Path, PathBuf};

use glassrip_audio::types::TranscriptArtifact;
use glassrip_core::envelope::{EnvelopeHeader, Outcome, Producer, Record};
use glassrip_core::jsonl::{self, JsonlError};
use glassrip_core::manifest::RunDir;
use semver::Version;
use serde::Serialize;
use serde_json::Value;

use crate::board::BoardStateItem;
use crate::schemas;

/// Writes `items` (id, value) as an artifact of `schema` in the run directory.
pub fn write_artifact<T: Serialize>(
    run: &RunDir,
    schema: &str,
    version: Version,
    params: Value,
    items: Vec<(String, T)>,
) -> Result<PathBuf, JsonlError> {
    let path = run.artifact_path(schema);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|source| JsonlError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
    }
    // envelope params must be an object
    let params = match params {
        Value::Object(_) => params,
        Value::Null => serde_json::json!({ "imported": true }),
        other => serde_json::json!({ "imported": true, "value": other }),
    };
    let header = EnvelopeHeader {
        schema: schema.to_string(),
        schema_version: version,
        run_id: run.manifest().run_id.clone(),
        producer: Producer::glassrip(env!("CARGO_PKG_VERSION"), None),
        inputs: Vec::new(),
        params,
        content_hash: None,
        restored_from: None,
    };
    let records: Vec<Record<T>> = items
        .into_iter()
        .map(|(id, v)| Record {
            id,
            outcome: Outcome::ok(v),
        })
        .collect();
    jsonl::write_atomic(&path, &header, &records)?;
    Ok(path)
}

/// Imports a `glassrip.transcript` envelope from the audio crate.
pub fn import_transcript(run: &RunDir, t: &TranscriptArtifact) -> Result<PathBuf, JsonlError> {
    let params = serde_json::to_value(&t.params).unwrap_or(Value::Null);
    let items = t
        .items
        .iter()
        .map(|s| (s.segment_id.clone(), s.clone()))
        .collect();
    write_artifact(
        run,
        schemas::TRANSCRIPT,
        Version::new(1, 0, 0),
        params,
        items,
    )
}

/// Imports board states (one item per board).
pub fn import_boards(run: &RunDir, boards: &[BoardStateItem]) -> Result<PathBuf, JsonlError> {
    let items = boards
        .iter()
        .map(|b| (b.board_id.clone(), b.clone()))
        .collect();
    write_artifact(
        run,
        schemas::BOARD_STATE,
        Version::new(1, 0, 0),
        Value::Null,
        items,
    )
}

/// Copies an existing core artifact file (for example keyframes from another run).
pub fn copy_artifact(run: &RunDir, schema: &str, from: &Path) -> Result<PathBuf, JsonlError> {
    let path = run.artifact_path(schema);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|source| JsonlError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
    }
    std::fs::copy(from, &path).map_err(|source| JsonlError::Io {
        path: from.to_path_buf(),
        source,
    })?;
    Ok(path)
}

/// Writes an empty artifact (an upstream branch that produced nothing).
pub fn write_empty(run: &RunDir, schema: &str) -> Result<PathBuf, JsonlError> {
    write_artifact::<Value>(run, schema, Version::new(1, 0, 0), Value::Null, Vec::new())
}
