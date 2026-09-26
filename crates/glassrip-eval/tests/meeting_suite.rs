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
    // Raw diarization: 3 labels for 2 people.
    assert_eq!(get("audio.diarizer_label_error"), 1.0);
    assert!(!m.contains_key("audio.speaker_label_error"), "retired key");
    // After name mapping: avery, jordan, and the unresolved S2 voice.
    assert_eq!(get("audio.speaker_identities"), 3.0);
    assert_eq!(get("audio.speaker_identity_error"), 1.0);
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

/// Codex finding 8 end to end: a negated decision in the notes is not the decision.
#[test]
fn a_negated_decision_is_not_recalled() {
    let score = |decision: &str| {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            schema::MEETING_NOTES,
            vec![("meeting_notes", ra::notes("ok", &[decision], &[], &[]))],
        );
        run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0)
            .unwrap()
            .metrics["notes.decision.recall"]
    };
    assert_eq!(score("We decided to defer the importer."), 1.0);
    assert_eq!(score("We decided not to defer the importer."), 0.0);
    assert_eq!(score("Do not defer the importer"), 0.0);
}

/// Codex round-1 M5: without a speakers artifact there is no mapping, so the
/// post-mapping target has no value to pass on.
#[test]
fn no_speaker_mapping_means_no_identity_metric() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        schema::TRANSCRIPT,
        vec![
            ("s1", ra::segment("s1", "S0", 0.0, 5.0, "hi", &[])),
            ("s2", ra::segment("s2", "S1", 5.0, 9.0, "ok", &[])),
        ],
    );
    let run = run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    assert_eq!(run.metrics.get("audio.diarizer_label_error"), Some(&0.0));
    assert!(!run.metrics.contains_key("audio.speaker_identity_error"));
    assert!(run.not_run.iter().any(|n| n.contains("glassrip.speakers")));
    // Codex round-2 M6: the skipped target fails the run, and so does an empty
    // speakers artifact.
    assert!(
        run.gate_failures
            .iter()
            .any(|g| g.contains("speaker_identity_error not evaluated")),
        "{:?}",
        run.gate_failures
    );
    ra::write_artifact(dir.path(), schema::SPEAKERS, vec![]).unwrap();
    let run = run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    assert!(!run.metrics.contains_key("audio.speaker_identity_error"));
    assert_eq!(run.gate_failures.len(), 1, "{:?}", run.gate_failures);
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
    let grid = FrameClock::PrototypeGrid {
        interval_s: 2.0,
        lead_s: 0.05,
    };
    g.clocks.screen_types = Some(grid);
    assert_eq!(score(&g), 1.0);
    g.clocks.screen_types = Some(FrameClock::Pts);
    assert_eq!(score(&g), 0.0);
    // The legacy field still declares the screen clock.
    g.clocks.screen_types = None;
    g.frame_clock = Some(grid);
    assert_eq!(score(&g), 1.0);
    // A golden without any clock scores as before clocks existed (PTS), and the
    // report says so.
    let mut v = serde_json::to_value(&g).unwrap();
    v.as_object_mut().unwrap().remove("frame_clock");
    let g: MeetingGolden = serde_json::from_value(v).unwrap();
    g.validate().unwrap();
    assert_eq!(score(&g), 0.0);
    let run = run_meeting(&g, &RunArtifacts::scan(d).unwrap(), 2.0).unwrap();
    assert!(
        run.warnings.iter().any(|w| w.contains("screen_types")),
        "{:?}",
        run.warnings
    );
}

/// Kimi finding 4, Codex round-1 M7 and M8: probes, assignment windows, moves,
/// window bounds, and allowed events each join on their own declared clock, and
/// never on the screen clock.
#[test]
fn owner_and_static_window_clocks_are_declared_per_section() {
    use glassrip_eval::golden::FrameClock;
    let owned = |from: f64| {
        ra::board(
            "w1",
            true,
            Some(60.0),
            nodes(&[("a", "Ledger API"), ("q", "Orbit Queue")]),
            vec![],
            vec![],
            vec![
                ra::owner("Avery", "Avery", ra::on_node("a", "Ledger API"), 0.0, from),
                ra::owner(
                    "Avery",
                    "Avery",
                    ra::on_node("q", "Orbit Queue"),
                    from,
                    3600.0,
                ),
            ],
            vec![ra::event("E1", "NodeAdded", 29.0)],
        )
    };
    let metrics = |g: &MeetingGolden| {
        let dir = tempfile::tempdir().unwrap();
        ra::write_artifact(
            dir.path(),
            schema::BOARD_STATE,
            vec![("b".into(), owned(32.0))],
        )
        .unwrap();
        run_meeting(g, &RunArtifacts::scan(dir.path()).unwrap(), 2.0)
            .unwrap()
            .metrics
    };
    let grid = |interval_s: f64| FrameClock::PrototypeGrid {
        interval_s,
        lead_s: 0.0,
    };
    // Gold move at 30 (PTS), predicted at 32: inside the 2 s tolerance.
    let mut g = golden();
    let pts = metrics(&g);
    assert_eq!(pts["owners.move_error_max_s"], 0.0);
    assert_eq!(pts["owners.attribution"], 1.0);
    // The screen clock never moves owner or window times.
    g.frame_clock = Some(grid(5.0));
    assert_eq!(metrics(&g), pts);
    // Moves on a 5 s grid: the move names 30, content time 35; error 3 - 2.
    g.clocks.owner_moves = Some(grid(5.0));
    assert_eq!(metrics(&g)["owners.move_error_max_s"], 1.0);
    // Probes on a 5 s grid (25 -> 30, 35 -> 40) while assignments stay on PTS: the
    // probe at 30 now falls in avery's queue assignment, which the prediction only
    // starts at 32.
    g.clocks.owner_moves = None;
    g.clocks.owner_probes = Some(grid(5.0));
    assert_eq!(metrics(&g)["owners.attribution"], 0.5);
    // Assignments shifted with their probes keep the same answer.
    g.clocks.owner_assignments = Some(grid(5.0));
    assert_eq!(metrics(&g)["owners.attribution"], 1.0);
    // The static window 20..28 holds the event at 29 on PTS (29 <= 28 + 4 s
    // tolerance). On a 10 s grid its bounds are 30..38, after the event.
    assert_eq!(pts["events.false_change"], 1.0);
    g.clocks.static_windows = Some(grid(10.0));
    assert_eq!(metrics(&g)["events.false_change"], 0.0);
    // Allowed events keep their own clock: an allowed NodeAdded at 29 on PTS
    // absorbs the event inside a grid-named window; shifted with the window it
    // would sit at 39, too far to absorb it.
    g.clocks.static_windows = Some(grid(5.0));
    g.static_windows[0].allowed_events =
        serde_json::from_value(json!([{"kind": "node_added", "t_s": 29.0}])).unwrap();
    assert_eq!(metrics(&g)["events.false_change"], 0.0);
    g.clocks.allowed_events = Some(grid(10.0));
    assert_eq!(metrics(&g)["events.false_change"], 1.0);
}

/// Codex final round 3 BLOCKER: `glassrip eval --suite meeting` with a golden set
/// and no run artifacts (or only some of them) scored nothing and still reported
/// PASS. Every section that could not be scored fails the suite.
#[tokio::test]
async fn meeting_eval_without_run_artifacts_fails() {
    #[derive(clap::Parser)]
    struct Wrap {
        #[command(flatten)]
        args: glassrip_eval::cli::EvalArgs,
    }
    let tmp = tempfile::tempdir().unwrap();
    let private = tmp.path().join("private");
    std::fs::create_dir_all(private.join("golden")).unwrap();
    std::fs::write(
        private.join("golden/meeting_golden.json"),
        serde_json::to_vec_pretty(&golden()).unwrap(),
    )
    .unwrap();
    let cfg = tmp.path().join("eval.toml");
    std::fs::write(
        &cfg,
        format!("[eval]\nprivate_fixtures = \"{}\"\n", private.display()),
    )
    .unwrap();
    let eval = |artifacts: Option<&Path>, out: &str| {
        let out = tmp.path().join(out);
        let mut argv = vec![
            "eval".to_string(),
            "--suite".into(),
            "meeting".into(),
            "--config".into(),
            cfg.display().to_string(),
            "--out".into(),
            out.display().to_string(),
        ];
        if let Some(a) = artifacts {
            argv.extend(["--artifacts".to_string(), a.display().to_string()]);
        }
        let args = <Wrap as clap::Parser>::try_parse_from(argv).unwrap().args;
        async move {
            let outcome = glassrip_eval::cli::run(args).await.unwrap();
            let report: Value =
                serde_json::from_slice(&std::fs::read(out.join("eval_report.json")).unwrap())
                    .unwrap();
            (outcome, report)
        }
    };
    let gate = |report: &Value| {
        report["gates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["name"] == "all_sections_scored")
            .cloned()
            .unwrap_or_else(|| panic!("no completeness gate: {report}"))
    };

    // No artifacts directory at all (the default under the private root).
    let (outcome, report) = eval(None, "none").await;
    assert!(!outcome.passed, "{report}");
    assert_eq!(outcome.status, glassrip_eval::report::Status::Fail);
    assert_eq!(gate(&report)["pass"], json!(false), "{report}");
    // An explicit artifacts path that does not exist.
    let (outcome, _) = eval(Some(&tmp.path().join("missing")), "missing").await;
    assert!(!outcome.passed);

    // Only the notes: every other section is unscored.
    let run = tmp.path().join("run");
    write(
        &run,
        schema::MEETING_NOTES,
        vec![(
            "meeting_notes",
            ra::notes("ok", &["defer the importer"], &[], &[]),
        )],
    );
    let (outcome, report) = eval(Some(&run), "partial").await;
    assert!(!outcome.passed, "{report}");
    let detail = gate(&report)["detail"].as_str().unwrap().to_string();
    assert!(detail.contains("glassrip.board_state"), "{detail}");
}

/// Codex final round 3 MAJOR: a speakers artifact whose labels were all left
/// unresolved is not a name mapping, and an unresolved voice never stands in for
/// a person in the post-mapping error.
#[test]
fn unresolved_labels_do_not_satisfy_the_post_mapping_speaker_check() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        schema::TRANSCRIPT,
        vec![
            ("s1", ra::segment("s1", "S0", 0.0, 5.0, "hi", &[])),
            ("s2", ra::segment("s2", "S1", 5.0, 9.0, "ok", &[])),
        ],
    );
    let speakers = |s0: Option<&str>| {
        vec![
            ("S0", ra::speaker_label("S0", s0)),
            ("S1", ra::speaker_label("S1", None)),
        ]
    };
    // Two labels, two people in the golden set, nobody named.
    write(dir.path(), schema::SPEAKERS, speakers(None));
    let run = run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    assert!(
        !run.metrics.contains_key("audio.speaker_identity_error"),
        "{:?}",
        run.metrics
    );
    assert!(
        run.gate_failures.iter().any(|g| g.contains("names nobody")),
        "{:?}",
        run.gate_failures
    );
    // A blank person id names no one either (Codex final round 3, second pass).
    write(dir.path(), schema::SPEAKERS, speakers(Some(" ")));
    let run = run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    assert!(!run.metrics.contains_key("audio.speaker_identity_error"));
    assert!(
        run.gate_failures.iter().any(|g| g.contains("names nobody")),
        "{:?}",
        run.gate_failures
    );
    // One named, one unresolved: the unresolved voice does not fill in for the
    // second person.
    write(dir.path(), schema::SPEAKERS, speakers(Some("avery")));
    let run = run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    assert_eq!(run.metrics["audio.speaker_identities"], 2.0);
    assert_eq!(run.metrics["audio.speaker_unresolved"], 1.0);
    assert_eq!(run.metrics["audio.speaker_mapping_coverage"], 0.5);
    assert_eq!(run.metrics["audio.speaker_identity_error"], 2.0);
    assert!(run.gate_failures.is_empty(), "{:?}", run.gate_failures);
}

/// Codex final round 3, second pass: artifacts that exist but hold nothing do
/// not let the suite pass unscored. A board state with no final board is an
/// unscored board section, and a transcript with no speech cannot pass the
/// speaker check when the golden set has speakers.
#[test]
fn empty_artifacts_leave_sections_unscored_and_fail() {
    let dir = tempfile::tempdir().unwrap();
    for schema in [
        schema::KEYFRAMES,
        schema::SCREEN_CLASS,
        schema::BOARD_STATE,
        schema::TRANSCRIPT,
        schema::SPEAKERS,
    ] {
        write(dir.path(), schema, vec![]);
    }
    write(
        dir.path(),
        schema::MEETING_NOTES,
        vec![(
            "meeting_notes",
            ra::notes("ok", &["defer the importer"], &[], &[]),
        )],
    );
    let run = run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    assert!(
        run.not_run.iter().any(|n| n.contains("no final board")),
        "{:?}",
        run.not_run
    );
    assert!(
        run.gate_failures
            .iter()
            .any(|g| g.contains("no speech but the golden set has 2 speaker(s)")),
        "{:?}",
        run.gate_failures
    );
}

/// Codex final round 3, third pass: segments without speech (blank text, no
/// words) are not speech to attribute, and a word range the speakers artifact
/// gives its own speaker counts as that speaker.
#[test]
fn only_speech_is_attributed_and_word_spans_count() {
    let mapped = || {
        vec![
            ("S0", ra::speaker_label("S0", Some("avery"))),
            ("S1", ra::speaker_label("S1", Some("jordan"))),
        ]
    };
    // Blank segments mapped to both people: no speech, and the golden set has
    // two speakers, so the check cannot pass.
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        schema::TRANSCRIPT,
        vec![
            ("s1", ra::segment("s1", "S0", 0.0, 5.0, "  ", &[])),
            ("s2", ra::segment("s2", "S1", 5.0, 9.0, "", &[])),
        ],
    );
    write(dir.path(), schema::SPEAKERS, mapped());
    let run = run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    assert!(!run.metrics.contains_key("audio.speaker_identity_error"));
    assert!(
        run.gate_failures.iter().any(|g| g.contains("no speech")),
        "{:?}",
        run.gate_failures
    );

    // A third voice inside avery's segment: three people spoke, not two.
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        schema::TRANSCRIPT,
        vec![
            (
                "s1",
                ra::segment(
                    "s1",
                    "S0",
                    0.0,
                    5.0,
                    "ship it now",
                    &[("ship", 0.0), ("it", 0.5), ("now", 1.0)],
                ),
            ),
            (
                "s2",
                ra::segment("s2", "S1", 5.0, 9.0, "ok", &[("ok", 5.0)]),
            ),
        ],
    );
    let mut records = mapped();
    records.push((
        "seg-s1",
        json!({
            "kind": "segment", "segment_id": "s1", "start_s": 0.0, "end_s": 5.0,
            "label": "S0", "person_id": "avery", "confidence": 0.9,
            "source": "label_map", "scores": {}, "observations": [],
            "spans": [{"word_start": 2, "word_end": 3, "person_id": "riley",
                       "confidence": 0.8, "source": "visual_relabel", "reason": "tile"}]
        }),
    ));
    write(dir.path(), schema::SPEAKERS, records);
    let run = run_meeting(&golden(), &RunArtifacts::scan(dir.path()).unwrap(), 2.0).unwrap();
    assert_eq!(run.metrics["audio.speaker_identities"], 3.0);
    assert_eq!(run.metrics["audio.speaker_identity_error"], 1.0);
    assert!(run.gate_failures.is_empty(), "{:?}", run.gate_failures);
}
