//! What eval reads from a pipeline run, and how it maps to the spec 7 artifacts.
//!
//! Eval never depends on the stage crates' Rust types. It reads artifacts from a
//! run's `artifacts/` directory as glassrip-core [`Envelope`]s, finds each one by
//! its `schema` name (file names do not matter; `.json` envelopes and `.jsonl`
//! files with a `{"record":"header"}` first line are both accepted), checks the
//! major version, and deserializes the items into the lenient views below
//! (unknown fields ignored, optional fields defaulted). A stage that emits the
//! spec 7 fields is readable without changes here.
//!
//! | Artifact (major 1) | View | Fields eval reads |
//! |---|---|---|
//! | `glassrip.keyframes` | [`KeyframeItem`] | `keyframe_id`, `t_start_s`, `t_end_s`, `t_rep_s` |
//! | `glassrip.screen_class` | [`ScreenClassItem`] | `keyframe_id`, `screen_type` (snake_case enum of 6.7) |
//! | `glassrip.board_reading` | [`BoardReadingItem`] | `keyframe_id`, `status`, `result.{nodes[local_id,text], edges[src,dst,label,style,direction], stickies[text], owner_tags[name_raw,person_id], other_visible_text[text]}` |
//! | `glassrip.board_state` | [`BoardStateItem`] (one item per board id or per stable window) | `final`, `t_end_s`, `nodes[{node_id,text,variants,in_final_state,last_seen_s}]`, `edges[{src,dst,label,style,direction}]` (node ids), `stickies[{text,kind,in_final_state}]`, `owner_assignments[{person_id,name_raw,target{kind:node,node_id}|{kind:edge,src,dst},valid_from_s,valid_to_s}]`, `events[{event_id,kind,t_s}]` |
//! | `glassrip.transcript` | [`TranscriptItem`] | `segment_id`, `start_s`, `end_s`, `speaker_label`, `text`, `words[{w,start_s}]` |
//! | `glassrip.meeting_notes` | [`NotesItem`] (first item) | `status`, `decisions[text]`, `action_items[person_id,task]`, `open_questions[text]`, `caveats` |
//! | `glassrip.documents` | [`crate::metrics::docs::PredDocument`] | `page_id`, `type`, `title`, `fields` (ticket), `blocks`, `provenance[{field,source,model_only}]`, `completeness.coverage` |
//!
//! Assumptions to confirm when the stages land (each is a one-line change here):
//! - `glassrip.board_state` edges and owner targets reference `node_id`s of the
//!   same item; `direction` is `forward` (src is the tail), `uncertain`, or
//!   `bidirectional`.
//! - A node or sticky is in the final board unless `in_final_state` is false.
//! - The final board is the item flagged `final: true` (the last stable board
//!   window); with several flagged items, the latest by `t_end_s`. Without any
//!   flag, the latest item by time (`t_end_s`, else its latest node, owner, or
//!   event time; ties go to the later item in the file). See
//!   [`select_final_state`]. Eval never picks an item by comparing it to gold.
//! - Owner assignments and events are timed, so owner and event metrics use
//!   the assignments and events of every item, not only the final one.
//! - `glassrip.meeting_notes` has one item holding the whole notes object, with
//!   `status: degraded` when the minimum-output alarm fires (6.14).
//! - `glassrip.documents` provenance is flattened to one entry per field path
//!   (`title`, `description`, `comments[1].body`, `blocks[3]`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glassrip_core::envelope::{EnvelopeHeader, SchemaReq};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{read_to_string, EvalError, Result};
use crate::metrics::board::{Direction, LineStyle};

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
    /// Meeting notes.
    pub const MEETING_NOTES: &str = "glassrip.meeting_notes";
    /// Document-mode pages.
    pub const DOCUMENTS: &str = "glassrip.documents";
}

/// Major version of every artifact eval understands.
pub const ARTIFACT_MAJOR: u64 = 1;

/// `glassrip.keyframes` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyframeItem {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Start, seconds.
    pub t_start_s: f64,
    /// End, seconds.
    pub t_end_s: f64,
    /// Representative time, seconds.
    pub t_rep_s: f64,
}

/// `glassrip.screen_class` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreenClassItem {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Screen type (snake_case).
    pub screen_type: String,
}

/// Node inside a board reading.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReadingNode {
    /// Local id.
    pub local_id: String,
    /// Text.
    pub text: String,
}

/// Edge inside a board reading or board state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewEdge {
    /// Tail id.
    pub src: String,
    /// Head id.
    pub dst: String,
    /// Label.
    pub label: String,
    /// Style.
    pub style: LineStyle,
    /// Direction.
    pub direction: Direction,
}

/// Text-bearing element.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextElement {
    /// Text.
    pub text: String,
}

/// Owner tag inside a board reading.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReadingOwnerTag {
    /// Name as read.
    pub name_raw: String,
    /// Resolved person, if any.
    pub person_id: Option<String>,
}

/// Result of one board reading.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReadingResult {
    /// Nodes.
    pub nodes: Vec<ReadingNode>,
    /// Edges (local ids).
    pub edges: Vec<ViewEdge>,
    /// Stickies.
    pub stickies: Vec<TextElement>,
    /// Owner tags.
    pub owner_tags: Vec<ReadingOwnerTag>,
    /// Other canvas text.
    pub other_visible_text: Vec<TextElement>,
}

/// `glassrip.board_reading` item.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BoardReadingItem {
    /// Keyframe id.
    pub keyframe_id: String,
    /// `ok`, `error`, or `skipped`.
    pub status: String,
    /// Reading, when ok.
    pub result: Option<ReadingResult>,
}

/// Node in the board state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateNode {
    /// Node id.
    pub node_id: String,
    /// Canonical text.
    pub text: String,
    /// Other readings.
    #[serde(default)]
    pub variants: Vec<String>,
    /// Alive in the final board.
    #[serde(default = "yes")]
    pub in_final_state: bool,
    /// Last time the node was seen, seconds.
    #[serde(default)]
    pub last_seen_s: Option<f64>,
}

fn yes() -> bool {
    true
}

/// Sticky in the board state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateSticky {
    /// Text.
    pub text: String,
    /// Kind.
    #[serde(default)]
    pub kind: Option<String>,
    /// Alive in the final board.
    #[serde(default = "yes")]
    pub in_final_state: bool,
}

/// Owner anchor in the board state (node ids).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StateTarget {
    /// Node.
    Node {
        /// Node id.
        node_id: String,
    },
    /// Edge.
    Edge {
        /// Tail node id.
        src: String,
        /// Head node id.
        dst: String,
    },
}

/// Timed owner assignment in the board state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateOwner {
    /// Resolved person id.
    #[serde(default)]
    pub person_id: Option<String>,
    /// Name as read on the tag.
    #[serde(default)]
    pub name_raw: String,
    /// Anchor.
    pub target: StateTarget,
    /// Start, seconds.
    pub valid_from_s: f64,
    /// End, seconds (open when absent).
    #[serde(default)]
    pub valid_to_s: Option<f64>,
}

/// Computed change event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateEvent {
    /// Event id.
    #[serde(default)]
    pub event_id: String,
    /// Kind (`NodeAdded` or `node_added`).
    pub kind: String,
    /// Time, seconds.
    pub t_s: f64,
}

/// `glassrip.board_state` item (one per board).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BoardStateItem {
    /// Board id.
    pub board_id: String,
    /// True for the final board state (last stable board window).
    #[serde(rename = "final")]
    pub is_final: Option<bool>,
    /// End of the window this state describes, seconds.
    pub t_end_s: Option<f64>,
    /// Nodes.
    pub nodes: Vec<StateNode>,
    /// Edges between node ids.
    pub edges: Vec<ViewEdge>,
    /// Stickies.
    pub stickies: Vec<StateSticky>,
    /// Owner assignments.
    pub owner_assignments: Vec<StateOwner>,
    /// Events.
    pub events: Vec<StateEvent>,
}

impl BoardStateItem {
    /// Time of the state: `t_end_s`, else its latest node, owner, or event time.
    pub fn time_s(&self) -> Option<f64> {
        self.t_end_s.or_else(|| {
            self.nodes
                .iter()
                .filter_map(|n| n.last_seen_s)
                .chain(self.owner_assignments.iter().map(|o| o.valid_from_s))
                .chain(self.events.iter().map(|e| e.t_s))
                .reduce(f64::max)
        })
    }
}

/// Index of the pipeline's final board state (see the module docs): the latest
/// item flagged `final`, else the latest item by time; ties go to the later item.
pub fn select_final_state(items: &[BoardStateItem]) -> Option<usize> {
    let flagged: Vec<usize> = (0..items.len())
        .filter(|&i| items[i].is_final == Some(true))
        .collect();
    let pool: Vec<usize> = if flagged.is_empty() {
        (0..items.len()).collect()
    } else {
        flagged
    };
    pool.into_iter().reduce(|best, i| {
        let (tb, ti) = (items[best].time_s(), items[i].time_s());
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

/// Word with timing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptWord {
    /// Word.
    pub w: String,
    /// Start, seconds.
    pub start_s: f64,
}

/// `glassrip.transcript` item.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TranscriptItem {
    /// Segment id.
    pub segment_id: String,
    /// Start.
    pub start_s: f64,
    /// End.
    pub end_s: f64,
    /// Speaker label.
    pub speaker_label: String,
    /// Corrected text.
    pub text: String,
    /// Words.
    pub words: Vec<TranscriptWord>,
}

/// Notes text item.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NotesText {
    /// Text.
    pub text: String,
}

/// Notes action item.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NotesAction {
    /// Person id or name.
    pub person_id: Option<String>,
    /// Task.
    pub task: String,
}

/// `glassrip.meeting_notes` item.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NotesItem {
    /// `ok` or `degraded`.
    pub status: String,
    /// Decisions.
    pub decisions: Vec<NotesText>,
    /// Action items.
    pub action_items: Vec<NotesAction>,
    /// Open questions.
    pub open_questions: Vec<NotesText>,
    /// Caveats.
    pub caveats: Vec<String>,
}

/// Artifacts of one run, indexed by schema name.
#[derive(Debug, Default)]
pub struct RunArtifacts {
    by_schema: BTreeMap<String, PathBuf>,
}

impl RunArtifacts {
    /// Indexes every `.json` / `.jsonl` artifact in `dir` by its envelope schema.
    pub fn scan(dir: &Path) -> Result<Self> {
        let mut by_schema = BTreeMap::new();
        let entries = fs_err::read_dir(dir).map_err(|e| EvalError::io(dir, e))?;
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

    /// Loads the items of `schema` as `T`; `Ok(None)` when absent.
    pub fn items<T: DeserializeOwned>(&self, schema: &str) -> Result<Option<Vec<T>>> {
        match self.path(schema) {
            None => Ok(None),
            Some(p) => load_items(p, schema).map(Some),
        }
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

/// Loads an envelope (`.json`) or JSONL artifact, checks schema and major, and
/// returns its items (JSONL: last write per id wins, ids from `*_id` fields).
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
        .map(|v| serde_json::from_value(v).map_err(|e| EvalError::json(path, e)))
        .collect()
}

fn item_id(v: &Value) -> Option<String> {
    let m = v.as_object()?;
    ["keyframe_id", "segment_id", "page_id", "board_id", "id"]
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
    fn json_envelope_and_jsonl_last_write_wins() {
        let dir = tempfile::tempdir().unwrap();
        let mut env = header(schema::KEYFRAMES, "1.2.0");
        env["items"] = json!([
            {"keyframe_id": "k1", "t_start_s": 0.0, "t_end_s": 4.0, "t_rep_s": 2.0, "n_frames": 2}
        ]);
        fs_err::write(dir.path().join("a.json"), env.to_string()).unwrap();
        let mut h = header(schema::SCREEN_CLASS, "1.0.0");
        h["record"] = json!("header");
        let lines = [
            h.to_string(),
            json!({"record":"item","keyframe_id":"k1","screen_type":"cms"}).to_string(),
            json!({"record":"item","keyframe_id":"k1","screen_type":"whiteboard"}).to_string(),
            "{\"record\":\"item\",\"keyfr".to_string(),
        ];
        fs_err::write(dir.path().join("zz.jsonl"), lines.join("\n")).unwrap();
        let run = RunArtifacts::scan(dir.path()).unwrap();
        let kf: Vec<KeyframeItem> = run.items(schema::KEYFRAMES).unwrap().unwrap();
        assert_eq!(kf[0].t_rep_s, 2.0);
        let sc: Vec<ScreenClassItem> = run.items(schema::SCREEN_CLASS).unwrap().unwrap();
        assert_eq!(sc.len(), 1);
        assert_eq!(sc[0].screen_type, "whiteboard");
        assert!(run.items::<Value>(schema::BOARD_STATE).unwrap().is_none());
    }

    #[test]
    fn wrong_major_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut env = header(schema::KEYFRAMES, "2.0.0");
        env["items"] = json!([]);
        let p = dir.path().join("k.json");
        fs_err::write(&p, env.to_string()).unwrap();
        assert!(load_items::<KeyframeItem>(&p, schema::KEYFRAMES).is_err());
    }

    fn state(v: Value) -> BoardStateItem {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn final_state_selection() {
        let a = state(json!({"board_id": "a", "t_end_s": 100.0}));
        let b = state(json!({"board_id": "b", "t_end_s": 50.0, "final": true}));
        let c = state(json!({"board_id": "c", "t_end_s": 300.0, "final": false}));
        // A flagged item wins over later unflagged ones.
        assert_eq!(
            select_final_state(&[a.clone(), b.clone(), c.clone()]),
            Some(1)
        );
        // Without flags, the latest by time.
        let a2 = state(json!({"board_id": "a", "events": [{"kind": "node_added", "t_s": 400.0}]}));
        assert_eq!(select_final_state(&[a2.clone(), c.clone()]), Some(0));
        // Without any time, the last item in the file.
        let n1 = state(json!({"board_id": "x"}));
        let n2 = state(json!({"board_id": "y"}));
        assert_eq!(select_final_state(&[n1, n2]), Some(1));
        assert_eq!(select_final_state(&[]), None);
    }

    #[test]
    fn board_state_view_defaults() {
        let v = json!({
            "board_id": "b1",
            "nodes": [{"node_id": "n1", "text": "Ledger API"}],
            "edges": [{"src": "n1", "dst": "n2", "direction": "uncertain"}],
            "owner_assignments": [{"name_raw": "Avery", "target": {"kind": "edge", "src": "n1", "dst": "n2"}, "valid_from_s": 3.0}],
            "events": [{"kind": "NodeAdded", "t_s": 1.0}],
            "extra_field": 1
        });
        let b: BoardStateItem = serde_json::from_value(v).unwrap();
        assert!(b.nodes[0].in_final_state);
        assert_eq!(b.edges[0].direction, Direction::Uncertain);
        assert_eq!(b.edges[0].style, LineStyle::Solid);
        assert!(matches!(
            b.owner_assignments[0].target,
            StateTarget::Edge { .. }
        ));
    }
}
