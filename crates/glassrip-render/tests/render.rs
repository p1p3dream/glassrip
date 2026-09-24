//! Rendering the synthetic meeting: markdown and SVG snapshots plus validation.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use glassrip_core::cache::Cache;
use glassrip_core::envelope::{Producer, Record, SchemaReq};
use glassrip_core::graph::{meeting_mode_stage_decls, Selection, StageGraph};
use glassrip_core::jsonl;
use glassrip_core::manifest::RunDir;
use glassrip_core::runner::{Runner, RunnerOptions};
use glassrip_notes::board::BoardState;
use glassrip_notes::import;
use glassrip_notes::notes::MeetingNotes;
use glassrip_notes::schemas;
use glassrip_render::markdown::MarkdownMeta;
use glassrip_render::scene::{build_scene, overlaps};
use glassrip_render::stage::RENDER_SCHEMA;
use glassrip_render::{render_all, RenderParams, RenderResult, RenderStage};
use semver::Version;
use tokio_util::sync::CancellationToken;

const BOARD: &str = include_str!("../../glassrip-notes/tests/fixtures/synthetic_board.json");
const NOTES: &str = include_str!("fixtures/synthetic_notes.json");

fn inputs() -> (MeetingNotes, BoardState) {
    (
        serde_json::from_str(NOTES).unwrap(),
        serde_json::from_str(BOARD).unwrap(),
    )
}

fn meta() -> MarkdownMeta {
    MarkdownMeta {
        date: Some("2031-04-02, 10:00 to 10:02".into()),
        source: None,
    }
}

#[test]
fn synthetic_meeting_renders_and_validates() {
    let (notes, board) = inputs();
    let dir = tempfile::tempdir().unwrap();
    // GLASSRIP_RENDER_DUMP_DIR keeps the files for a visual check
    let out = std::env::var("GLASSRIP_RENDER_DUMP_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dir.path().to_path_buf());
    let r = render_all(&notes, &[board], &out, "kiosk", &meta()).unwrap();
    assert!(r.markdown.ok, "{:?}", r.markdown);
    assert_eq!(r.markdown.tables_expected, 4);
    let (_, svg) = &r.svg[0];
    assert!(svg.parsed && svg.nonblank, "{svg:?}");
    assert!(svg.overlaps.is_empty(), "{:?}", svg.overlaps);
    assert!(
        svg.style_violations.is_empty(),
        "{:?}",
        svg.style_violations
    );
    assert_eq!(svg.layout_method, "board_positions");
    if svg.font_faces > 0 {
        assert!(svg.ok, "{svg:?}");
    }
    for f in &r.files {
        assert!(out.join(&f.name).is_file(), "{}", f.name);
    }
    let text = &r.svg_text[0].1;
    assert!(!text.contains("<marker"));
    assert!(
        text.contains("DEFERRED (00:26)"),
        "deferred badge from the decision"
    );
    assert!(
        text.contains("url(#glow-api)"),
        "focus glow on the relay card"
    );
    assert!(text.contains("moved from Ledger Service (01:10)"));
    insta::assert_snapshot!("synthetic_markdown", r.markdown_text);
    insta::assert_snapshot!("synthetic_svg", text);
}

#[test]
fn missing_positions_fall_back_to_sugiyama() {
    let (notes, mut board) = inputs();
    for n in &mut board.nodes {
        n.bbox = None;
    }
    let scene = build_scene(&board, &notes);
    assert_eq!(scene.layout_method, "sugiyama");
    assert!(overlaps(&scene).is_empty(), "{:?}", overlaps(&scene));
}

#[test]
fn coincident_boxes_are_separated() {
    let (notes, mut board) = inputs();
    for n in &mut board.nodes {
        n.bbox = Some(glassrip_notes::board::BBox {
            x: 10.0,
            y: 10.0,
            w: 50.0,
            h: 50.0,
        });
    }
    let scene = build_scene(&board, &notes);
    assert!(overlaps(&scene).is_empty(), "{:?}", overlaps(&scene));
}

#[tokio::test]
async fn render_stage_runs_on_the_core_runner() {
    let (notes, board) = inputs();
    let dir = tempfile::tempdir().unwrap();
    let run = RunDir::open(
        &dir.path().join("run"),
        "synthetic",
        Producer::glassrip("0.1.0", None),
    )
    .unwrap();
    import::write_artifact(
        &run,
        schemas::MEETING_NOTES,
        Version::new(1, 0, 0),
        serde_json::json!({}),
        vec![("meeting_notes".to_string(), notes)],
    )
    .unwrap();
    import::import_boards(&run, &[board]).unwrap();
    let graph = StageGraph::new(meeting_mode_stage_decls()).unwrap();
    let sel = Selection {
        from: Some("render".into()),
        ..Default::default()
    };
    let mut runner = Runner::new(
        run,
        graph,
        &sel,
        Cache::in_workspace(dir.path()),
        RunnerOptions::default(),
        CancellationToken::new(),
    )
    .unwrap();
    let out = dir.path().join("out");
    let stage = RenderStage::new(RenderParams {
        out_dir: out.clone(),
        stem: "kiosk".into(),
        meta: meta(),
        strict: false,
    });
    let rep = runner.run_stage(&stage).await.unwrap();
    assert_eq!(rep.items_error, 0);
    let recs = jsonl::read::<Record<RenderResult>>(
        &runner.run_dir().artifact_path(RENDER_SCHEMA),
        &SchemaReq::new(RENDER_SCHEMA, 1),
    )
    .unwrap()
    .items;
    let result = recs[0].outcome.result.clone().unwrap();
    assert_eq!(result.files.len(), 3);
    assert!(out.join("kiosk-meeting-notes.md").is_file());
}
