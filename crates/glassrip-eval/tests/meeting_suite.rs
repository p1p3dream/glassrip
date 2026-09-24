//! End-to-end meeting-suite scoring on a fictional golden set and fictional run
//! artifacts written as glassrip-core envelopes, with hand-computed expectations.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use glassrip_eval::golden::MeetingGolden;
use glassrip_eval::suite::run_meeting;
use glassrip_eval::views::RunArtifacts;
use serde_json::{json, Value};

fn envelope(schema: &str, items: Value) -> Value {
    json!({
        "schema": schema,
        "schema_version": "1.0.0",
        "run_id": "run-1",
        "producer": {"tool": "glassrip", "version": "0.1.0", "git_sha": null},
        "inputs": [],
        "params": {},
        "items": items
    })
}

fn write(dir: &Path, name: &str, v: Value) {
    fs_err::write(dir.join(name), serde_json::to_string_pretty(&v).unwrap()).unwrap();
}

fn golden() -> MeetingGolden {
    serde_json::from_value(json!({
        "golden_version": 1,
        "meeting": "fictional weekly sync",
        "participants": [
            {"person_id": "avery", "display_name": "Avery Stone", "aliases": []},
            {"person_id": "jordan", "display_name": "Jordan Vale", "aliases": []}
        ],
        "screen_types": [
            {"t_rep_s": 2.0, "screen_type": "whiteboard"},
            {"t_rep_s": 12.0, "screen_type": "cms"},
            {"t_rep_s": 22.0, "screen_type": "whiteboard"},
            {"t_rep_s": 40.0, "screen_type": "chat", "confirmed": false}
        ],
        "final_board": {
            "nodes": [
                {"id": "api", "text": "Ledger API"},
                {"id": "queue", "text": "Orbit Queue"}
            ],
            "edges": [{"src": "api", "dst": "queue", "label": "REST"}],
            "stickies": [{"text": "Who owns retries?"}]
        },
        "owners": {
            "probes_s": [25.0, 35.0],
            "assignments": [
                {"person_id": "avery", "target": {"kind": "node", "node": "api"}, "valid_from_s": 20.0, "valid_to_s": 30.0},
                {"person_id": "avery", "target": {"kind": "node", "node": "queue"}, "valid_from_s": 30.0}
            ],
            "moves": [{"person_id": "avery", "from": {"kind": "node", "node": "api"}, "to": {"kind": "node", "node": "queue"}, "t_s": 30.0}],
            "negatives": ["Riley Park"]
        },
        "static_windows": [{"t_start_s": 20.0, "t_end_s": 28.0}],
        "chrome_terms": ["Share"],
        "transcript": {
            "decisions": [{"text": "defer the importer"}],
            "action_items": [{"text": "write the parser", "person_id": "jordan"}],
            "open_questions": [{"text": "who owns retries"}],
            "negative_action_items": ["see you later"],
            "hotwords": [{"word": "Quorra", "exhaustive": true, "windows": [{"t_start_s": 0.0, "t_end_s": 10.0, "count": 2}]}],
            "speaker_count": 2
        }
    }))
    .unwrap()
}

#[test]
fn meeting_suite_hand_computed() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    write(
        d,
        "keyframes.json",
        envelope(
            "glassrip.keyframes",
            json!([
                {"keyframe_id": "k1", "t_start_s": 0.0, "t_end_s": 10.0, "t_rep_s": 2.1},
                {"keyframe_id": "k2", "t_start_s": 10.0, "t_end_s": 20.0, "t_rep_s": 12.3},
                {"keyframe_id": "k3", "t_start_s": 20.0, "t_end_s": 45.0, "t_rep_s": 22.2}
            ]),
        ),
    );
    write(
        d,
        "screen_class.json",
        envelope(
            "glassrip.screen_class",
            json!([
                {"keyframe_id": "k1", "screen_type": "whiteboard"},
                {"keyframe_id": "k2", "screen_type": "whiteboard"},
                {"keyframe_id": "k3", "screen_type": "whiteboard"}
            ]),
        ),
    );
    write(
        d,
        "board_state.json",
        envelope(
            "glassrip.board_state",
            json!([{
                "board_id": "b1",
                "nodes": [
                    {"node_id": "n1", "text": "Ledger API"},
                    {"node_id": "n2", "text": "Orbit Queue"},
                    {"node_id": "n3", "text": "Share"}
                ],
                "edges": [{"src": "n2", "dst": "n1", "label": "REST", "direction": "forward"}],
                "stickies": [{"text": "Who owns retries?"}],
                "owner_assignments": [
                    {"name_raw": "Avery", "target": {"kind": "node", "node_id": "n1"}, "valid_from_s": 21.0, "valid_to_s": 40.0},
                    {"name_raw": "Avery", "target": {"kind": "node", "node_id": "n2"}, "valid_from_s": 40.0},
                    {"name_raw": "Riley Park", "target": {"kind": "node", "node_id": "n2"}, "valid_from_s": 21.0}
                ],
                "events": [
                    {"kind": "NodeAdded", "t_s": 24.0},
                    {"kind": "OwnerMoved", "t_s": 40.0}
                ]
            }]),
        ),
    );
    write(
        d,
        "transcript.json",
        envelope(
            "glassrip.transcript",
            json!([
                {"segment_id": "s1", "start_s": 0.0, "end_s": 9.0, "speaker_label": "S0", "text": "hi",
                 "words": [{"w": "Quorra,", "start_s": 1.0}, {"w": "Cora", "start_s": 5.0}]},
                {"segment_id": "s2", "start_s": 9.0, "end_s": 20.0, "speaker_label": "S1", "text": "Quorra again",
                 "words": []},
                {"segment_id": "s3", "start_s": 20.0, "end_s": 30.0, "speaker_label": "S2", "text": "ok", "words": []}
            ]),
        ),
    );
    write(
        d,
        "notes.json",
        envelope(
            "glassrip.meeting_notes",
            json!([{
                "status": "ok",
                "decisions": [{"text": "defer importer now"}, {"text": "lunch at noon"}],
                "action_items": [{"person_id": "Jordan", "task": "write the parser"}, {"person_id": "Avery", "task": "I'll see you guys later"}],
                "open_questions": [{"text": "who owns the retries"}]
            }]),
        ),
    );

    let run = run_meeting(&golden(), &RunArtifacts::scan(d).unwrap(), 2.0).unwrap();
    let m = &run.metrics;
    let get = |k: &str| *m.get(k).unwrap_or_else(|| panic!("missing {k}: {m:#?}"));

    // Screens: 3 confirmed labels; the CMS keyframe was read as whiteboard.
    assert!((get("screen.accuracy") - 2.0 / 3.0).abs() < 1e-12);
    assert_eq!(get("screen.cms_as_whiteboard"), 1.0);
    assert_eq!(get("screen.unconfirmed_labels"), 1.0);

    // Board: 3 predicted nodes, 2 gold; "Share" is chrome. Edge reversed.
    assert!((get("board.node.precision") - 2.0 / 3.0).abs() < 1e-12);
    assert_eq!(get("board.node.recall"), 1.0);
    assert_eq!(get("board.chrome_fp"), 1.0);
    assert_eq!(get("board.edge.f1"), 1.0);
    assert_eq!(get("board.edge.direction_accuracy"), 0.0);
    assert_eq!(get("board.sticky.recall"), 1.0);

    // Owners: probe 25 expects avery on api (found); probe 35 expects avery on
    // queue (not found: predicted move at 40), and the prediction still has avery
    // on api (extra). Riley Park never resolves to a participant.
    assert_eq!(get("owners.attribution"), 0.5);
    assert_eq!(get("owners.extra"), 1.0);
    assert_eq!(get("owners.unresolved"), 1.0);
    assert_eq!(get("owners.negative_hits"), 1.0);
    // Move predicted at 40 vs gold 30: 10 - 2 tolerance.
    assert_eq!(get("owners.move_error_max_s"), 8.0);

    // Events: NodeAdded at 24 inside the static window 20..28.
    assert_eq!(get("events.false_change"), 1.0);

    // Notes: decision P 1/2, R 1; action P 1/2 (farewell), R 1; question matched.
    assert_eq!(get("notes.decision.recall"), 1.0);
    assert_eq!(get("notes.decision.precision"), 0.5);
    assert_eq!(get("notes.action.precision"), 0.5);
    assert_eq!(get("notes.question.recall"), 1.0);
    assert_eq!(get("notes.negative_action_hits"), 1.0);

    // Audio: window 0..10 has 2 truth occurrences; "Quorra," at 1 s hits, "Cora"
    // misses; "Quorra" at 9 s (segment text, start time) is inside the window with
    // tolerance and fills the second slot. 3 speaker labels vs 2 people.
    assert_eq!(get("audio.hotword_wer"), 0.0);
    assert_eq!(get("audio.speaker_labels"), 3.0);
    assert_eq!(get("audio.speaker_label_error"), 1.0);
    assert!(run.not_run.is_empty(), "{:?}", run.not_run);
    assert!(run.gate_failures.is_empty());
}

#[test]
fn degraded_notes_fail_the_gate_and_missing_artifacts_are_reported() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "notes.json",
        envelope("glassrip.meeting_notes", json!([{"status": "degraded"}])),
    );
    let run = run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    assert_eq!(run.gate_failures.len(), 1);
    assert_eq!(run.not_run.len(), 3);
}
