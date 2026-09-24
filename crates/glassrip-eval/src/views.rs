//! What eval reads from a pipeline run.
//!
//! Meeting-mode artifacts are read as the stage crates' own item types through
//! glassrip-core's JSONL reader: the header is checked (schema name and major
//! version), records are deduplicated by id (last write wins), and only `ok`
//! outcomes are scored. A change to a stage's item type is therefore a compile
//! error or a load error here, never a silently empty metric.
//!
//! | Artifact (major 1) | Item type | Producer |
//! |---|---|---|
//! | `glassrip.keyframes` | [`Keyframe`] | `glassrip-media-stages` `keyframes` |
//! | `glassrip.screen_class` | [`ScreenClassItem`] | `glassrip-vision-stages` `classify` |
//! | `glassrip.board_reading` | [`BoardReadingItem`] | `glassrip-vision-stages` `board_read` |
//! | `glassrip.board_state` | [`BoardStateItem`] | `glassrip-meeting` `board_state` |
//! | `glassrip.transcript` | [`TranscriptSegment`] | `glassrip-audio` `assign_words` |
//! | `glassrip.speakers` | [`SpeakersRecord`] | `glassrip-notes` `name_speakers` |
//! | `glassrip.meeting_notes` | [`MeetingNotes`] | `glassrip-notes` `notes` |
//!
//! Scoring rules that depend on these types:
//! - The final board is the item flagged `final` (the last stable board window);
//!   with several flagged items, the latest by time; without any flag, the
//!   latest by time. See [`select_final_state`]. Eval never picks an item by
//!   comparing it to gold.
//! - Nodes, edges, and stickies count toward the final board when their
//!   `in_final_state` / `in_final` flag is set.
//! - Owner assignments and events are timed, so owner and event metrics use the
//!   assignments and events of every board item, not only the final one.
//!
//! `glassrip.documents` has no producing stage yet; [`load_items`] reads it
//! leniently from `.json` envelopes or JSONL (with or without core record
//! wrapping), as document mode fixtures supply it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glassrip_core::envelope::{EnvelopeHeader, Record, SchemaReq};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{read_to_string, EvalError, Result};

pub use glassrip_audio::types::TranscriptSegment;
pub use glassrip_media_stages::schema::Keyframe;
pub use glassrip_meeting::consolidate::events::BoardEvent;
pub use glassrip_meeting::consolidate::owners::{OwnerAssignment, OwnerTarget};
pub use glassrip_meeting::consolidate::{BoardStateItem, EdgeOrientation};
pub use glassrip_notes::notes::{MeetingNotes, NotesStatus};
pub use glassrip_notes::speakers::SpeakersRecord;
pub use glassrip_vision_stages::artifacts::{BoardReadingItem, ScreenClassItem};

/// Artifact schema names read by eval.
pub mod schema {
    /// Keyframes.
    pub const KEYFRAMES: &str = "glassrip.keyframes";
    /// Screen classification.
    pub const SCREEN_CLASS: &str = "glassrip.screen_class";
    /// Per-keyframe board readings.
    pub const BOARD_READING: &str = "glassrip.board_reading";
    /// Consolidated board state.
    pub const BOARD_STATE: &str = "glassrip.board_state";
    /// Transcript.
    pub const TRANSCRIPT: &str = "glassrip.transcript";
    /// Speaker naming.
    pub const SPEAKERS: &str = "glassrip.speakers";
    /// Meeting notes.
    pub const MEETING_NOTES: &str = "glassrip.meeting_notes";
    /// Document-mode pages.
    pub const DOCUMENTS: &str = "glassrip.documents";
}

/// Major version of every artifact eval understands.
pub const ARTIFACT_MAJOR: u64 = 1;

/// Time of a board state: `t_end_s`, else its latest node, owner, or event time.
pub fn state_time_s(b: &BoardStateItem) -> Option<f64> {
    b.t_end_s.or_else(|| {
        b.nodes
            .iter()
            .filter_map(|n| n.last_seen_s)
            .chain(b.owner_assignments.iter().map(|o| o.valid_from_s))
            .chain(b.events.iter().map(|e| e.t_s))
            .reduce(f64::max)
    })
}

/// Index of the pipeline's final board state (see the module docs): the latest
/// item flagged `final`, else the latest item by time; ties go to the later item.
pub fn select_final_state(items: &[BoardStateItem]) -> Option<usize> {
    let flagged: Vec<usize> = (0..items.len()).filter(|&i| items[i].is_final).collect();
    let pool: Vec<usize> = if flagged.is_empty() {
        (0..items.len()).collect()
    } else {
        flagged
    };
    pool.into_iter().reduce(|best, i| {
        let (tb, ti) = (state_time_s(&items[best]), state_time_s(&items[i]));
        let later = match (tb, ti) {
            (Some(b), Some(t)) => t >= b,
            (None, _) => true,
            (Some(_), None) => false,
        };
        if later {
            i
        } else {
            best
        }
    })
}

/// Artifacts of one run, indexed by schema name.
#[derive(Debug, Default)]
pub struct RunArtifacts {
    by_schema: BTreeMap<String, PathBuf>,
}

impl RunArtifacts {
    /// Indexes the artifacts of a run. `dir` is a run directory (its
    /// `artifacts/` subdirectory is read) or an artifacts directory; every
    /// `.jsonl` / `.json` file is indexed by its header's `schema`.
    pub fn scan(dir: &Path) -> Result<Self> {
        let nested = dir.join("artifacts");
        let dir = if nested.is_dir() {
            nested
        } else {
            dir.to_path_buf()
        };
        let mut by_schema = BTreeMap::new();
        let entries = fs_err::read_dir(&dir).map_err(|e| EvalError::io(&dir, e))?;
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("json" | "jsonl")
                )
            })
            .collect();
        paths.sort();
        for p in paths {
            if let Some(schema) = peek_schema(&p) {
                by_schema.entry(schema).or_insert(p);
            }
        }
        Ok(Self { by_schema })
    }

    /// Path of the artifact with `schema`, if present.
    pub fn path(&self, schema: &str) -> Option<&Path> {
        self.by_schema.get(schema).map(PathBuf::as_path)
    }

    /// Every record of a stage artifact (all outcomes); `Ok(None)` when absent.
    pub fn records<T: DeserializeOwned>(&self, schema: &str) -> Result<Option<Vec<Record<T>>>> {
        let Some(p) = self.path(schema) else {
            return Ok(None);
        };
        glassrip_core::jsonl::read::<Record<T>>(p, &SchemaReq::new(schema, ARTIFACT_MAJOR))
            .map(|a| Some(a.items))
            .map_err(|e| EvalError::json(p, e))
    }

    /// The `ok` results of a stage artifact; `Ok(None)` when absent.
    pub fn items<T: DeserializeOwned>(&self, schema: &str) -> Result<Option<Vec<T>>> {
        Ok(self.records::<T>(schema)?.map(|records| {
            records
                .into_iter()
                .filter_map(|r| r.outcome.result)
                .collect()
        }))
    }

    /// Items of a stage artifact that failed, by id; empty when absent.
    pub fn failed_ids(&self, schema: &str) -> Result<Vec<String>> {
        Ok(self
            .records::<Value>(schema)?
            .unwrap_or_default()
            .into_iter()
            .filter(|r| r.outcome.is_error())
            .map(|r| r.id)
            .collect())
    }
}

fn peek_schema(path: &Path) -> Option<String> {
    let text = fs_err::read_to_string(path).ok()?;
    let first: Value = if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
        serde_json::from_str(text.lines().next()?).ok()?
    } else {
        serde_json::from_str(&text).ok()?
    };
    first.get("schema")?.as_str().map(str::to_string)
}

fn header_from(mut v: Value) -> Value {
    if let Value::Object(m) = &mut v {
        m.remove("items");
        m.remove("record");
    }
    v
}

/// A core record (`{"id", "outcome": {"status", "result"}}`) becomes its result
/// (skipped when not ok); a bare item is returned as is.
fn unwrap_record(v: Value) -> Option<Value> {
    match v.get("outcome") {
        Some(outcome) => outcome.get("result").cloned(),
        None => Some(v),
    }
}

/// Lenient loader for artifacts without a stage type (`glassrip.documents`): an
/// envelope (`.json`) or JSONL artifact, checked for schema and major, items
/// returned as `T` (JSONL: last write per id wins; core records unwrapped).
pub fn load_items<T: DeserializeOwned>(path: &Path, schema: &str) -> Result<Vec<T>> {
    let text = read_to_string(path)?;
    let req = SchemaReq::new(schema, ARTIFACT_MAJOR);
    let check = |header: Value| -> Result<()> {
        EnvelopeHeader::from_value_checked(header_from(header), &req)
            .map(|_| ())
            .map_err(|e| EvalError::json(path, e))
    };
    let raw_items: Vec<Value> = if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header: Value = serde_json::from_str(lines.next().unwrap_or("{}"))
            .map_err(|e| EvalError::json(path, e))?;
        check(header)?;
        let mut order: Vec<String> = Vec::new();
        let mut by_id: BTreeMap<String, Value> = BTreeMap::new();
        for (n, line) in lines.enumerate() {
            let Ok(mut v) = serde_json::from_str::<Value>(line) else {
                // A trailing partial line after a crash is ignored (8, crash-safe JSONL).
                continue;
            };
            if let Value::Object(m) = &mut v {
                m.remove("record");
            }
            let id = item_id(&v).unwrap_or_else(|| format!("#{n}"));
            if !by_id.contains_key(&id) {
                order.push(id.clone());
            }
            by_id.insert(id, v);
        }
        order
            .into_iter()
            .filter_map(|id| by_id.remove(&id))
            .collect()
    } else {
        let v: Value = serde_json::from_str(&text).map_err(|e| EvalError::json(path, e))?;
        let items = v
            .get("items")
            .and_then(|i| i.as_array())
            .cloned()
            .ok_or_else(|| EvalError::json(path, "envelope has no items list"))?;
        check(v)?;
        items
    };
    raw_items
        .into_iter()
        .filter_map(unwrap_record)
        .map(|v| serde_json::from_value(v).map_err(|e| EvalError::json(path, e)))
        .collect()
}

fn item_id(v: &Value) -> Option<String> {
    let m = v.as_object()?;
    ["id", "keyframe_id", "segment_id", "page_id", "board_id"]
        .iter()
        .find_map(|k| m.get(*k).and_then(|x| x.as_str()).map(str::to_string))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn header(schema: &str, version: &str) -> Value {
        json!({
            "schema": schema,
            "schema_version": version,
            "run_id": "r1",
            "producer": {"tool": "glassrip", "version": "0.1.0", "git_sha": null},
            "inputs": [],
            "params": {}
        })
    }

    #[test]
    fn lenient_loader_unwraps_records_and_keeps_last_write() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = header(schema::DOCUMENTS, "1.0.0");
        h["record"] = json!("header");
        let lines = [
            h.to_string(),
            json!({"record":"item","id":"p1","outcome":{"status":"ok","result":{"page_id":"p1","v":1}}}).to_string(),
            json!({"record":"item","id":"p1","outcome":{"status":"ok","result":{"page_id":"p1","v":2}}}).to_string(),
            json!({"record":"item","id":"p2","outcome":{"status":"error","error":{"code":"internal","message":"x"}}}).to_string(),
            "{\"record\":\"item\",\"id\":\"p3\",\"outc".to_string(),
        ];
        let p = dir.path().join("docs.jsonl");
        fs_err::write(&p, lines.join("\n")).unwrap();
        let items: Vec<Value> = load_items(&p, schema::DOCUMENTS).unwrap();
        assert_eq!(items, vec![json!({"page_id": "p1", "v": 2})]);
    }

    #[test]
    fn wrong_major_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut env = header(schema::DOCUMENTS, "2.0.0");
        env["items"] = json!([]);
        let p = dir.path().join("k.json");
        fs_err::write(&p, env.to_string()).unwrap();
        assert!(load_items::<Value>(&p, schema::DOCUMENTS).is_err());
    }

    #[test]
    fn scan_accepts_a_run_directory() {
        let dir = tempfile::tempdir().unwrap();
        let art = dir.path().join("artifacts");
        fs_err::create_dir_all(&art).unwrap();
        let mut h = header(schema::KEYFRAMES, "1.0.0");
        h["record"] = json!("header");
        fs_err::write(art.join("anything.jsonl"), format!("{h}\n")).unwrap();
        let run = RunArtifacts::scan(dir.path()).unwrap();
        assert!(run.path(schema::KEYFRAMES).is_some());
        assert_eq!(run.items::<Value>(schema::KEYFRAMES).unwrap(), Some(vec![]));
        assert!(run.items::<Value>(schema::BOARD_STATE).unwrap().is_none());
    }
}
