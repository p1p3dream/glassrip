//! End-to-end meeting-suite scoring on a fictional golden set and fictional run
//! artifacts in the real stage formats (core JSONL records of the stage crates'
//! item types), with hand-computed expectations.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use glassrip_eval::golden::MeetingGolden;
use glassrip_eval::suite::run_meeting;
use glassrip_eval::synth::run_artifacts as ra;
use glassrip_eval::views::{schema, RunArtifacts};
use serde_json::{json, Value};

fn write(run: &Path, schema: &str, items: Vec<(&str, Value)>) {
    ra::write_artifact(
        run,
        schema,
        items
            .into_iter()
            .map(|(id, v)| (id.to_string(), v))
            .collect(),
    )
    .unwrap();
}

fn golden() -> MeetingGolden {
    serde_json::from_value(json!({
        "golden_version": 1,
        "meeting": "fictional weekly sync",
        "frame_clock": {"kind": "pts"},
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
        schema::KEYFRAMES,
        vec![
            ("k1", ra::keyframe("k1", 0.0, 10.0, 2.1)),
            ("k2", ra::keyframe("k2", 10.0, 20.0, 12.3)),
            ("k3", ra::keyframe("k3", 20.0, 45.0, 22.2)),
        ],
    );
    write(
        d,
        schema::SCREEN_CLASS,
        vec![
            ("k1", ra::screen_class("k1", "whiteboard")),
            ("k2", ra::screen_class("k2", "whiteboard")),
            ("k3", ra::screen_class("k3", "whiteboard")),
        ],
    );
    let (api, queue, share) = (("n1", "Ledger API"), ("n2", "Orbit Queue"), ("n3", "Share"));
    write(
        d,
        schema::BOARD_STATE,
        vec![(
            "board-1",
            ra::board(
                "b1",
                false,
                None,
                vec![
                    ra::node(api.0, api.1, true),
                    ra::node(queue.0, queue.1, true),
                    ra::node(share.0, share.1, true),
                ],
                // Read in the reverse direction of gold.
                vec![ra::edge(queue, api, "REST", "forward")],
                vec![ra::sticky("s1", "Who owns retries?", true)],
                vec![
                    ra::owner("Avery", "Avery", ra::on_node(api.0, api.1), 21.0, 40.0),
                    ra::owner(
                        "Avery",
                        "Avery",
                        ra::on_node(queue.0, queue.1),
                        40.0,
                        3600.0,
                    ),
                    ra::owner(
                        "riley-park",
                        "Riley Park",
                        ra::on_node(queue.0, queue.1),
                        21.0,
                        3600.0,
                    ),
                ],
                vec![
                    ra::event("E1", "NodeAdded", 24.0),
                    ra::event("E2", "OwnerMoved", 40.0),
                ],
            ),
        )],
    );
    write(
        d,
        schema::TRANSCRIPT,
        vec![
            (
                "s1",
                ra::segment(
                    "s1",
                    "S0",
                    0.0,
                    9.0,
                    "hi",
                    &[("Quorra,", 1.0), ("Cora", 5.0)],
                ),
            ),
            (
                "s2",
                ra::segment("s2", "S1", 9.0, 20.0, "Quorra again", &[]),
            ),
            ("s3", ra::segment("s3", "S2", 20.0, 30.0, "ok", &[])),
        ],
    );
    write(
        d,
        schema::SPEAKERS,
        vec![
            ("label:S0", ra::speaker_label("S0", Some("avery"))),
            ("label:S1", ra::speaker_label("S1", Some("jordan"))),
            ("label:S2", ra::speaker_label("S2", None)),
        ],
    );
    write(
        d,
        schema::MEETING_NOTES,
        vec![(
            "meeting_notes",
            ra::notes(
                "ok",
                &["defer importer now", "lunch at noon"],
                &[
                    (Some("Jordan"), "write the parser"),
                    (Some("Avery"), "I'll see you guys later"),
                ],
                &["who owns the retries"],
            ),
        )],
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
    // After name mapping: avery, jordan, and the unresolved S2 voice.
    assert_eq!(get("audio.speaker_identities"), 3.0);
    assert_eq!(get("audio.speaker_label_error"), 1.0);
    // Speakers: 3 labels, 2 mapped people, both in the golden set.
    assert_eq!(get("speakers.labels"), 3.0);
    assert_eq!(get("speakers.distinct_people"), 2.0);
    assert_eq!(get("speakers.distinct_people_error"), 0.0);
    assert_eq!(get("speakers.people_in_golden"), 2.0);
    assert!(run.not_run.is_empty(), "{:?}", run.not_run);
    assert!(run.gate_failures.is_empty());
}

#[test]
fn degraded_notes_fail_the_gate_and_missing_artifacts_are_reported() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        schema::MEETING_NOTES,
        vec![("meeting_notes", ra::notes("degraded", &[], &[], &[]))],
    );
    let run = run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    assert_eq!(run.gate_failures.len(), 1);
    // Screens, board, speakers, and audio.
    assert_eq!(run.not_run.len(), 4, "{:?}", run.not_run);
}

#[test]
fn errored_notes_fail_the_section_and_errored_items_are_case_errors() {
    let dir = tempfile::tempdir().unwrap();
    ra::write_with_errors(
        dir.path(),
        schema::MEETING_NOTES,
        vec![],
        &["meeting_notes"],
    )
    .unwrap();
    ra::write_with_errors(
        dir.path(),
        schema::BOARD_STATE,
        vec![(
            "board-1".into(),
            ra::board(
                "board-1",
                true,
                Some(60.0),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            ),
        )],
        &["board-2"],
    )
    .unwrap();
    let run = run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    // never skipped: the notes section fails and its metrics read zero
    assert!(
        run.gate_failures
            .iter()
            .any(|g| g.contains("notes section FAIL")),
        "{:?}",
        run.gate_failures
    );
    assert!(
        !run.not_run.iter().any(|n| n.contains("notes")),
        "{:?}",
        run.not_run
    );
    assert_eq!(run.metrics.get("notes.decision.recall"), Some(&0.0));
    assert_eq!(run.metrics.get("notes.action.precision"), Some(&0.0));
    // both errored artifacts are case errors (the no_case_errors gate)
    assert_eq!(run.errors.len(), 2, "{:?}", run.errors);
    assert!(run.errors.iter().any(|e| e.contains("board-2")));
}

#[test]
fn everyone_actions_match_by_owner_name() {
    let dir = tempfile::tempdir().unwrap();
    let mut g = golden();
    g.transcript.action_items[0].person_id = Some("everyone".into());
    write(
        dir.path(),
        schema::MEETING_NOTES,
        vec![(
            "meeting_notes",
            ra::notes("ok", &[], &[(None, "write the parser")], &[]),
        )],
    );
    let run = run_meeting(&g, &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    assert_eq!(run.metrics.get("notes.action.recall"), Some(&1.0));
}

fn board_metrics(states: Vec<Value>) -> glassrip_eval::suite::Metrics {
    let dir = tempfile::tempdir().unwrap();
    let items: Vec<(String, Value)> = states
        .into_iter()
        .enumerate()
        .map(|(i, v)| (format!("board-{i}"), v))
        .collect();
    ra::write_artifact(dir.path(), schema::BOARD_STATE, items).unwrap();
    run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0)
        .unwrap()
        .metrics
}

fn nodes(list: &[(&str, &str)]) -> Vec<Value> {
    list.iter().map(|(id, t)| ra::node(id, t, true)).collect()
}

#[test]
fn noisy_extra_state_cannot_improve_the_score() {
    let (api, queue) = (("n1", "Ledger API"), ("n2", "Orbit Queue"));
    // The pipeline's final state reads only one of the two gold nodes.
    let final_state = ra::board(
        "b1",
        true,
        Some(50.0),
        nodes(&[api, ("n9", "Noise")]),
        vec![],
        vec![],
        vec![],
        vec![],
    );
    // A non-final state that happens to match gold perfectly.
    let perfect = ra::board(
        "b2",
        false,
        Some(60.0),
        nodes(&[api, queue]),
        vec![ra::edge(api, queue, "REST", "forward")],
        vec![ra::sticky("s1", "Who owns retries?", true)],
        vec![],
        vec![],
    );
    let alone = board_metrics(vec![final_state.clone()]);
    let with_noise = board_metrics(vec![perfect, final_state]);
    for k in [
        "board.node.f1",
        "board.node.recall",
        "board.node.precision",
        "board.edge.f1",
        "board.sticky.f1",
    ] {
        assert_eq!(alone[k], with_noise[k], "{k}");
    }
    assert_eq!(with_noise["board.node.recall"], 0.5);

    // Without flags, the latest state by time is final: the earlier perfect state is ignored.
    let early_perfect = ra::board(
        "b2",
        false,
        Some(10.0),
        nodes(&[api, queue]),
        vec![],
        vec![],
        vec![],
        vec![],
    );
    let late_partial = ra::board(
        "b1",
        false,
        Some(50.0),
        nodes(&[api]),
        vec![],
        vec![],
        vec![],
        vec![],
    );
    let m = board_metrics(vec![late_partial, early_perfect]);
    assert_eq!(m["board.node.recall"], 0.5);
}

#[test]
fn owners_and_events_use_every_state_once() {
    // Owner timing lives in a non-final state; the same event appears in both.
    let early = ra::board(
        "w1",
        false,
        Some(30.0),
        nodes(&[("a", "Ledger API")]),
        vec![],
        vec![],
        vec![ra::owner(
            "Avery",
            "Avery",
            ra::on_node("a", "Ledger API"),
            20.0,
            30.0,
        )],
        vec![ra::event("E1", "NodeAdded", 24.0)],
    );
    let last = ra::board(
        "w2",
        true,
        Some(60.0),
        nodes(&[("q", "Orbit Queue")]),
        vec![],
        vec![],
        vec![ra::owner(
            "Avery",
            "Avery",
            ra::on_node("q", "Orbit Queue"),
            30.0,
            3600.0,
        )],
        vec![ra::event("E1", "NodeAdded", 24.0)],
    );
    let m = board_metrics(vec![early, last]);
    assert_eq!(m["owners.attribution"], 1.0);
    assert_eq!(m["owners.move_error_max_s"], 0.0);
    assert_eq!(m["events.false_change"], 1.0);
}

#[test]
fn nominal_grid_labels_join_where_their_frame_was_taken() {
    use glassrip_eval::golden::FrameClock;
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    write(
        d,
        schema::KEYFRAMES,
        vec![
            ("k1", ra::keyframe("k1", 0.0, 10.0, 5.0)),
            ("k2", ra::keyframe("k2", 10.0, 20.0, 15.0)),
        ],
    );
    write(
        d,
        schema::SCREEN_CLASS,
        vec![
            ("k1", ra::screen_class("k1", "whiteboard")),
            ("k2", ra::screen_class("k2", "cms")),
        ],
    );
    // A prototype frame named 9 s shows the screen just before 11 s.
    let mut g = golden();
    g.screen_types =
        serde_json::from_value(json!([{"t_rep_s": 9.0, "screen_type": "cms"}])).unwrap();
    let score = |g: &MeetingGolden| {
        run_meeting(g, &RunArtifacts::scan(d).unwrap(), 2.0)
            .unwrap()
            .metrics["screen.accuracy"]
    };
    g.frame_clock = FrameClock::default();
    assert_eq!(score(&g), 1.0);
    g.frame_clock = FrameClock::Pts;
    assert_eq!(score(&g), 0.0);
    // A golden without the field is on the prototype grid (spec 9.2).
    let mut v = serde_json::to_value(&g).unwrap();
    v.as_object_mut().unwrap().remove("frame_clock");
    let g: MeetingGolden = serde_json::from_value(v).unwrap();
    assert_eq!(g.frame_clock, FrameClock::default());
}
