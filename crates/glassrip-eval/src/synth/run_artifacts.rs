//! Builders for synthetic run artifacts in the real stage formats, for tests of
//! the meeting scorer. Each builder returns the JSON of one stage item type with
//! every required field; [`write_artifact`] writes items as a core JSONL
//! artifact (`<dir>/artifacts/<schema>.jsonl`, ok records), which eval then
//! reads back through the stage crates' own types. Content is fictional.

use std::path::{Path, PathBuf};

use glassrip_core::envelope::{EnvelopeHeader, Outcome, Producer, Record};
use semver::Version;
use serde_json::{json, Value};

use crate::error::{EvalError, Result};

/// Writes `(id, item)` pairs as an ok-record artifact of `schema` under
/// `<run>/artifacts/`.
pub fn write_artifact(run: &Path, schema: &str, items: Vec<(String, Value)>) -> Result<PathBuf> {
    let dir = run.join("artifacts");
    fs_err::create_dir_all(&dir).map_err(|e| EvalError::io(&dir, e))?;
    let path = dir.join(format!("{schema}.jsonl"));
    let header = EnvelopeHeader {
        schema: schema.to_string(),
        schema_version: Version::new(1, 0, 0),
        run_id: "synthetic".into(),
        producer: Producer::glassrip("0.0.0", None),
        inputs: Vec::new(),
        params: json!({}),
        content_hash: None,
        restored_from: None,
    };
    let records: Vec<Record<Value>> = items
        .into_iter()
        .map(|(id, v)| Record {
            id,
            outcome: Outcome::ok(v),
        })
        .collect();
    glassrip_core::jsonl::write_atomic(&path, &header, &records)
        .map_err(|e| EvalError::Other(format!("{}: {e}", path.display())))?;
    Ok(path)
}

/// Writes an artifact whose `errored` ids failed (error records) and whose
/// `ok` items succeeded.
pub fn write_with_errors(
    run: &Path,
    schema: &str,
    ok: Vec<(String, Value)>,
    errored: &[&str],
) -> Result<PathBuf> {
    let path = write_artifact(run, schema, ok)?;
    let mut text = fs_err::read_to_string(&path).map_err(|e| EvalError::io(&path, e))?;
    for id in errored {
        let r: Record<Value> = Record {
            id: (*id).to_string(),
            outcome: Outcome::error(glassrip_core::envelope::ErrorInfo::new(
                glassrip_core::envelope::ErrorCode::Internal,
                "synthetic failure",
            )),
        };
        let mut v = serde_json::to_value(&r).map_err(|e| EvalError::Other(e.to_string()))?;
        v["record"] = json!("item");
        if !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&v.to_string());
        text.push('\n');
    }
    fs_err::write(&path, text).map_err(|e| EvalError::io(&path, e))?;
    Ok(path)
}

/// `glassrip.keyframes` item.
pub fn keyframe(id: &str, t_start_s: f64, t_end_s: f64, t_rep_s: f64) -> Value {
    json!({
        "keyframe_id": id, "index": 0, "rep_frame_id": format!("f_{id}"),
        "frame_ids": [format!("f_{id}")], "t_start_s": t_start_s, "t_end_s": t_end_s,
        "t_rep_s": t_rep_s, "n_frames": 1, "sharpness_lapvar": 100.0,
        "boundary": {"reason": "start", "anchor_frame_id": null, "ssim": null,
                     "changed_frac": null, "ink_change": null, "align_ok": null},
        "merged": []
    })
}

/// `glassrip.screen_class` item.
pub fn screen_class(keyframe_id: &str, screen_type: &str) -> Value {
    json!({
        "keyframe_id": keyframe_id, "screen_type": screen_type, "app_hint": null,
        "confidence": 0.9, "method": "model", "source": "combined", "canvas_bbox": null,
        "reads_board": screen_type == "whiteboard", "model": null, "model_error": null,
        "rule_hits": [], "smoothed_from": null
    })
}

fn lifetime(t0: f64, t1: f64) -> Value {
    json!([{"first_seen_s": t0, "last_seen_s": t1, "keyframes": 2, "removed_at_s": null}])
}

/// Board-state node (`in_final` sets `in_final_state`).
pub fn node(id: &str, text: &str, in_final: bool) -> Value {
    json!({
        "node_id": id, "text": text, "variants": [text],
        "variant_counts": [{"text": text, "count": 2}], "lifetimes": lifetime(0.0, 60.0),
        "last_seen_s": 60.0, "in_final_state": in_final, "registration": "position",
        "bbox": null, "list_votes": {"node": 2, "sticky": 0, "other": 0, "edge_label": 0}
    })
}

/// Board-state edge between node ids (`direction`: forward, uncertain, bidirectional).
pub fn edge(src: (&str, &str), dst: (&str, &str), label: &str, direction: &str) -> Value {
    let zero = json!({"a_to_b": 0.0, "b_to_a": 0.0, "bidirectional": 0.0, "no_arrowhead": 0.0});
    let votes = serde_json::to_value(glassrip_meeting::direction::DirectionVotes::default())
        .unwrap_or_else(|_| json!({"pixel": zero, "vlm": zero, "reader": zero}));
    json!({
        "id": format!("e_{}_{}", src.0, dst.0), "a": src.0, "b": dst.0,
        "a_text": src.1, "b_text": dst.1, "decision": "a_to_b", "src": src.0, "dst": dst.0,
        "direction": direction, "direction_votes": votes, "direction_basis": "pixel",
        "label": label, "style": "solid", "lifetimes": lifetime(0.0, 60.0), "in_final": true
    })
}

/// Board-state sticky.
pub fn sticky(id: &str, text: &str, in_final: bool) -> Value {
    json!({
        "id": id, "text": text, "kind": "note", "color": "yellow",
        "lifetimes": lifetime(0.0, 60.0), "in_final_state": in_final
    })
}

/// Owner target on a node.
pub fn on_node(node_id: &str, text: &str) -> Value {
    json!({"kind": "node", "node_id": node_id, "text": text})
}

/// Owner target on an edge.
pub fn on_edge(src: (&str, &str), dst: (&str, &str)) -> Value {
    json!({"kind": "edge", "edge_id": format!("e_{}_{}", src.0, dst.0), "src": src.0,
           "dst": dst.0, "a_text": src.1, "b_text": dst.1})
}

/// Timed owner assignment.
pub fn owner(person_id: &str, name_raw: &str, target: Value, from_s: f64, to_s: f64) -> Value {
    json!({
        "person_id": person_id, "display_name": name_raw, "name_raw": name_raw,
        "target": target, "valid_from_s": from_s, "valid_to_s": to_s,
        "opened_at_keyframe": "k1", "opened_by": "consistent_keyframes",
        "corroboration": null, "moved_from": null, "backfill_from_s": null, "sightings": []
    })
}

/// Board event (`kind` as serialized, for example `NodeAdded`).
pub fn event(id: &str, kind: &str, t_s: f64) -> Value {
    json!({
        "event_id": id, "kind": kind, "t_s": t_s, "keyframe_id": "k1", "subject": "n1",
        "detail": "", "ink_change": null, "baseline": false
    })
}

/// `glassrip.board_state` item.
#[allow(clippy::too_many_arguments)]
pub fn board(
    board_id: &str,
    is_final: bool,
    t_end_s: Option<f64>,
    nodes: Vec<Value>,
    edges: Vec<Value>,
    stickies: Vec<Value>,
    owners: Vec<Value>,
    events: Vec<Value>,
) -> Value {
    json!({
        "board_id": board_id, "final": is_final, "t_end_s": t_end_s, "board_keyframes": [],
        "registration": [], "final_window": null, "nodes": nodes, "edges": edges,
        "stickies": stickies, "owner_assignments": owners, "rejected_owner_tags": [],
        "events": events, "suppressed_events": []
    })
}

/// `glassrip.transcript` segment; words are `(word, start_s)` (0.4 s each).
pub fn segment(
    id: &str,
    label: &str,
    start_s: f64,
    end_s: f64,
    text: &str,
    words: &[(&str, f64)],
) -> Value {
    let words: Vec<Value> = words
        .iter()
        .map(|(w, t)| {
            json!({"w": w, "start_s": t, "end_s": t + 0.4, "p": 0.9, "speaker_label": label,
                   "assign_conf": 0.9, "source": "diarizer"})
        })
        .collect();
    json!({
        "segment_id": id, "start_s": start_s, "end_s": end_s, "speaker_label": label,
        "speaker_conf": 0.9, "text": text, "text_raw": text, "gap_fill_words": 0,
        "unassigned_words": 0, "words": words
    })
}

/// `glassrip.speakers` label record.
pub fn speaker_label(label: &str, person_id: Option<&str>) -> Value {
    json!({
        "kind": "label", "label": label,
        "status": if person_id.is_some() { "mapped" } else { "unresolved" },
        "person_id": person_id, "confidence": 0.8, "evidence": [], "talk_time_s": 10.0
    })
}

fn evidence(segment_id: &str) -> Value {
    json!({"segment_ids": [segment_id], "event_ids": [], "keyframe_ids": []})
}

/// `glassrip.meeting_notes` item. `actions` are `(person_id, task)`.
pub fn notes(
    status: &str,
    decisions: &[&str],
    actions: &[(Option<&str>, &str)],
    questions: &[&str],
) -> Value {
    let decisions: Vec<Value> = decisions
        .iter()
        .enumerate()
        .map(|(i, t)| {
            json!({"id": format!("D{i}"), "text": t, "t_start_s": 1.0, "t_end_s": 2.0,
                   "evidence": evidence("s1"), "quote": null})
        })
        .collect();
    let actions: Vec<Value> = actions
        .iter()
        .enumerate()
        .map(|(i, (p, t))| {
            json!({"id": format!("A{i}"), "person_id": p, "owner": p.unwrap_or("Everyone"),
                   "task": t, "t_s": 1.0, "t_end_s": 2.0, "evidence": evidence("s1"), "quote": null})
        })
        .collect();
    let questions: Vec<Value> = questions
        .iter()
        .enumerate()
        .map(|(i, t)| {
            json!({"id": format!("Q{i}"), "text": t, "source": "transcript", "t_s": 1.0,
                   "evidence": evidence("s1"), "quote": null})
        })
        .collect();
    json!({
        "title": null, "duration_s": 60.0, "people": [], "presenter": null, "summary": [],
        "decisions": decisions, "action_items": actions, "open_questions": questions,
        "timeline": [], "caveats": [], "speakers": [], "transcript": [],
        "report": {"status": status, "model": "synthetic", "model_digest": null, "windows": 1,
                   "calls": [], "items_drafted": 0, "items_kept": 0,
                   "items_failed_first_pass": 0, "items_repaired": 0, "dropped": [],
                   "drop_rate": 0.0, "placement": null, "unloaded_vision_model": null,
                   "wall_s": 0.0}
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::views::{self, RunArtifacts};

    /// Every builder deserializes into the real stage type.
    #[test]
    fn builders_match_the_stage_types() {
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path();
        write_artifact(
            run,
            views::schema::KEYFRAMES,
            vec![("k1".into(), keyframe("k1", 0.0, 4.0, 2.0))],
        )
        .unwrap();
        write_artifact(
            run,
            views::schema::SCREEN_CLASS,
            vec![("k1".into(), screen_class("k1", "whiteboard"))],
        )
        .unwrap();
        let b = board(
            "board-1",
            true,
            Some(60.0),
            vec![
                node("n1", "Ledger API", true),
                node("n2", "Orbit Queue", true),
            ],
            vec![edge(
                ("n1", "Ledger API"),
                ("n2", "Orbit Queue"),
                "REST",
                "forward",
            )],
            vec![sticky("s1", "Who owns retries?", true)],
            vec![
                owner("avery", "Avery", on_node("n1", "Ledger API"), 10.0, 30.0),
                owner(
                    "avery",
                    "Avery",
                    on_edge(("n1", "Ledger API"), ("n2", "Orbit Queue")),
                    30.0,
                    60.0,
                ),
            ],
            vec![event("E1", "NodeAdded", 5.0)],
        );
        write_artifact(run, views::schema::BOARD_STATE, vec![("board-1".into(), b)]).unwrap();
        write_artifact(
            run,
            views::schema::TRANSCRIPT,
            vec![(
                "s1".into(),
                segment(
                    "s1",
                    "S0",
                    0.0,
                    2.0,
                    "hi there",
                    &[("hi", 0.0), ("there", 0.4)],
                ),
            )],
        )
        .unwrap();
        write_artifact(
            run,
            views::schema::SPEAKERS,
            vec![("label:S0".into(), speaker_label("S0", Some("avery")))],
        )
        .unwrap();
        write_artifact(
            run,
            views::schema::MEETING_NOTES,
            vec![(
                "meeting_notes".into(),
                notes("ok", &["x"], &[(Some("avery"), "y")], &["z?"]),
            )],
        )
        .unwrap();
        let art = RunArtifacts::scan(run).unwrap();
        assert_eq!(
            art.items::<views::Keyframe>(views::schema::KEYFRAMES)
                .unwrap()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            art.items::<views::ScreenClassItem>(views::schema::SCREEN_CLASS)
                .unwrap()
                .unwrap()
                .len(),
            1
        );
        let states = art
            .items::<views::BoardStateItem>(views::schema::BOARD_STATE)
            .unwrap()
            .unwrap();
        assert_eq!(states[0].nodes.len(), 2);
        assert_eq!(
            art.items::<views::TranscriptSegment>(views::schema::TRANSCRIPT)
                .unwrap()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            art.items::<views::SpeakersRecord>(views::schema::SPEAKERS)
                .unwrap()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            art.items::<views::MeetingNotes>(views::schema::MEETING_NOTES)
                .unwrap()
                .unwrap()
                .len(),
            1
        );
    }
}
