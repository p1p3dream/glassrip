//! Rendering the synthetic meeting: markdown and SVG snapshots plus validation.
//!
//! The board state is produced by glassrip-meeting's own consolidation (shared
//! generator), and text is rendered with a bundled OFL font (Inter) with system
//! fonts off, so validation is asserted the same way on every host.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "../../glassrip-notes/tests/common/synthetic_board.rs"]
mod synthetic_board;

use std::path::PathBuf;

use glassrip_core::cache::Cache;
use glassrip_core::envelope::{ErrorCode, Producer, Record, SchemaReq};
use glassrip_core::graph::{meeting_mode_stage_decls, Selection, StageGraph};
use glassrip_core::jsonl;
use glassrip_core::manifest::RunDir;
use glassrip_core::runner::{Runner, RunnerOptions};
use glassrip_notes::board::{BoardExt, BoardStateItem, EdgeStyle, OwnerTarget};
use glassrip_notes::import;
use glassrip_notes::notes::MeetingNotes;
use glassrip_notes::schemas;
use glassrip_render::markdown::MarkdownMeta;
use glassrip_render::scene::{build_scene, overlaps};
use glassrip_render::stage::RENDER_SCHEMA;
use glassrip_render::svg::FontConfig;
use glassrip_render::{render_all, RenderParams, RenderResult, RenderStage};
use semver::Version;
use tokio_util::sync::CancellationToken;

const NOTES: &str = include_str!("fixtures/synthetic_notes.json");

fn inputs() -> (MeetingNotes, BoardStateItem) {
    (
        serde_json::from_str(NOTES).unwrap(),
        synthetic_board::synthetic_board().0,
    )
}

fn bundled_fonts() -> FontConfig {
    FontConfig {
        system: false,
        dirs: vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/assets/fonts")],
    }
}

fn params(out: PathBuf) -> RenderParams {
    RenderParams {
        out_dir: out,
        stem: "kiosk".into(),
        meta: MarkdownMeta {
            date: Some("2031-04-02, 10:00 to 10:02".into()),
            source: None,
            media_duration_s: None,
        },
        fonts: bundled_fonts(),
        ..RenderParams::default()
    }
}

#[test]
fn defaults_are_strict() {
    assert!(RenderParams::default().strict);
}

#[test]
fn synthetic_meeting_renders_and_validates() {
    let (notes, board) = inputs();
    // the producer's board, as consumed here
    assert!(board.is_final);
    assert_eq!(board.final_nodes().len(), 4, "{:#?}", board.nodes);
    assert!(board
        .nodes
        .iter()
        .any(|n| n.text == "Old Sketch" && !n.in_final));
    let dir = tempfile::tempdir().unwrap();
    // GLASSRIP_RENDER_DUMP_DIR keeps the files for a visual check
    let out = std::env::var("GLASSRIP_RENDER_DUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dir.path().to_path_buf());
    let r = render_all(&notes, &[board], &params(out.clone())).unwrap();
    assert!(r.markdown.ok, "{:?}", r.markdown);
    let (_, svg) = &r.svg[0];
    assert!(svg.ok, "{svg:?}");
    assert!(svg.font_faces > 0);
    assert_eq!(svg.text_rendered, svg.text_nodes);
    assert_eq!(svg.layout_method, "board_positions");
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
    assert!(text.contains("moved from Ledger Service"));
    assert!(text.contains("MILESTONE"));
    assert!(r.markdown_text.contains("Removed (seen"));
    insta::assert_snapshot!("synthetic_markdown", r.markdown_text);
    insta::assert_snapshot!("synthetic_svg", text);
}

/// An owner on a short edge hemmed in at its middle (a channel-routed edge runs
/// beside it, its label sits next to it, and cards close both ends) was left
/// out, failing strict validation, although the rest of the edge had room.
#[test]
fn owners_of_a_crowded_edge_slide_along_it() {
    let (notes, mut board) = inputs();
    // Ledger, Relay and Kiosk in one column; Design Kit to the bottom right
    for (text, cx, cy) in [
        ("Ledger Service", 250.0, 200.0),
        ("Relay API", 250.0, 520.0),
        ("Kiosk App", 250.0, 840.0),
        ("Design Kit", 1300.0, 840.0),
    ] {
        let n = board.nodes.iter_mut().find(|n| n.text == text).unwrap();
        let b = n.bbox.as_mut().unwrap();
        (b.x1, b.y1, b.x2, b.y2) = (cx - 100.0, cy - 40.0, cx + 100.0, cy + 40.0);
    }
    // The dashed edge runs from the kit to the ledger: its direct route crosses
    // the kiosk card, so it takes the channel below the cards and climbs back
    // 12 px beside the Kiosk to Relay edge.
    let dashed = board
        .edges
        .iter_mut()
        .find(|e| e.style == EdgeStyle::Dashed)
        .unwrap();
    std::mem::swap(&mut dashed.src, &mut dashed.dst);
    let edge = board
        .final_edges()
        .into_iter()
        .find(|e| {
            let ends = [e.a_text.as_str(), e.b_text.as_str()];
            ends.contains(&"Kiosk App") && ends.contains(&"Relay API")
        })
        .unwrap()
        .clone();
    // every participant tags that edge
    let base = board.current_owners()[0].clone();
    board.owner_assignments = [
        ("avery-quinn", "Avery Quinn"),
        ("rohan-dasgupta", "Rohan Dasgupta"),
        ("mira-okafor", "Mira Okafor"),
    ]
    .into_iter()
    .map(|(id, name)| {
        let mut o = base.clone();
        o.person_id = id.into();
        o.display_name = name.into();
        o.moved_from = None;
        o.target = OwnerTarget::Edge {
            edge_id: edge.id.clone(),
            src: edge.src.clone(),
            dst: edge.dst.clone(),
            a_text: edge.a_text.clone(),
            b_text: edge.b_text.clone(),
        };
        o
    })
    .collect();
    assert_eq!(board.current_owners().len(), 3);

    let scene = build_scene(&board, &notes);
    // the crowding this reproduces: the dashed edge is routed via the channel
    let channel = scene
        .edges
        .iter()
        .find(|e| e.dash.is_some())
        .expect("dashed edge");
    assert_eq!(channel.d.matches(" L ").count(), 3, "{}", channel.d);
    assert!(
        scene.unplaced.is_empty(),
        "{:?}\ncards {:?}\npills {:?}\nedges {:?}",
        scene.unplaced,
        scene.cards.iter().map(|c| (&c.id, c.r)).collect::<Vec<_>>(),
        scene
            .pills
            .iter()
            .map(|p| (&p.text.text, p.r))
            .collect::<Vec<_>>(),
        scene
            .edges
            .iter()
            .map(|e| (&e.d, e.label.as_ref().map(|l| (&l.text.text, l.r))))
            .collect::<Vec<_>>()
    );
    assert!(overlaps(&scene).is_empty(), "{:?}", overlaps(&scene));
    // each owner pill sits beside its edge, between the two cards
    let card = |text: &str| {
        let id = &board.nodes.iter().find(|n| n.text == text).unwrap().id;
        scene.cards.iter().find(|c| &c.id == id).unwrap().r
    };
    let (relay, kiosk) = (card("Relay API"), card("Kiosk App"));
    let owners: Vec<_> = scene.pills.iter().filter(|p| p.fill == "#16a34a").collect();
    assert_eq!(owners.len(), 3);
    for p in owners {
        let cy = p.r.y + p.r.h / 2.0;
        assert!(cy > relay.bottom() && cy < kiosk.y, "{:?}", p.r);
        assert!(
            (p.r.x + p.r.w / 2.0 - (relay.x + relay.w / 2.0)).abs() < relay.w,
            "{:?}",
            p.r
        );
    }
    // and the full render passes strict validation
    let dir = tempfile::tempdir().unwrap();
    let r = render_all(&notes, &[board], &params(dir.path().to_path_buf())).unwrap();
    assert!(r.ok, "{:?}", r.failures());
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
        n.bbox = Some(glassrip_notes::board::BBox::new(10.0, 10.0, 60.0, 60.0));
    }
    let scene = build_scene(&board, &notes);
    assert!(overlaps(&scene).is_empty(), "{:?}", overlaps(&scene));
}

#[test]
fn edge_labels_never_cover_zone_titles() {
    // Move each box over a grid of offsets: wherever an edge runs, its label must
    // not land on a zone title, a card, or anything else (a long relation label
    // wraps when it does not fit on one line).
    let (notes, board) = inputs();
    let mut titled = 0;
    for i in 0..board.nodes.len() {
        for dx in [-300.0, -150.0, 0.0, 150.0, 300.0] {
            for dy in [-200.0, -100.0, 0.0, 100.0, 200.0] {
                let mut b = board.clone();
                if let Some(bb) = &mut b.nodes[i].bbox {
                    *bb = glassrip_notes::board::BBox::new(
                        bb.x1 + dx,
                        bb.y1 + dy,
                        bb.x2 + dx,
                        bb.y2 + dy,
                    );
                }
                let scene = build_scene(&b, &notes);
                titled += scene
                    .blocking
                    .iter()
                    .filter(|(n, _)| n.starts_with("zone title"))
                    .count();
                // Crowded positions may leave a note unplaced (reported, not
                // drawn); what this checks is that no label covers anything.
                let o: Vec<String> = overlaps(&scene)
                    .into_iter()
                    .filter(|m| m.contains("zone title") || m.contains("edge label"))
                    .collect();
                assert!(o.is_empty(), "node {i} moved by ({dx}, {dy}): {o:?}");
            }
        }
    }
    assert!(titled > 0, "the sweep produced zones with titles");
}

#[test]
fn a_long_relation_label_between_touching_cards_is_placed_clear() {
    // Put the design kit right next to the ledger: the dashed relation between
    // them has no room on its path at full width.
    let (notes, mut board) = inputs();
    let ledger = board
        .nodes
        .iter()
        .find(|n| n.text == "Ledger Service")
        .and_then(|n| n.bbox)
        .unwrap();
    let w = ledger.x2 - ledger.x1;
    for n in &mut board.nodes {
        if n.text == "Design Kit" {
            n.bbox = Some(glassrip_notes::board::BBox::new(
                ledger.x2 + 0.05 * w,
                ledger.y1,
                ledger.x2 + 1.05 * w,
                ledger.y2,
            ));
        }
    }
    let scene = build_scene(&board, &notes);
    let o: Vec<String> = overlaps(&scene)
        .into_iter()
        .filter(|m| m.contains("edge label"))
        .collect();
    assert!(o.is_empty(), "{o:?}");
    let e = scene
        .edges
        .iter()
        .find(|e| {
            e.label
                .as_ref()
                .is_some_and(|l| l.text.text.starts_with("Links between"))
        })
        .expect("the relation label is drawn");
    if let Some((x1, y1, x2, y2)) = e.leader {
        assert!(
            (x1 - x2).abs() + (y1 - y2).abs() > 0.0,
            "a leader has length"
        );
    }
}

/// Runs the render stage; returns the stage outcome and the records written.
async fn run_render_stage(
    params: RenderParams,
) -> (Result<u64, String>, Vec<Record<RenderResult>>) {
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
    let params = RenderParams {
        out_dir: dir.path().join("out"),
        ..params
    };
    let rep = runner.run_stage(&RenderStage::new(params)).await;
    let path = runner.run_dir().artifact_path(RENDER_SCHEMA);
    let recs = if path.is_file() {
        jsonl::read::<Record<RenderResult>>(&path, &SchemaReq::new(RENDER_SCHEMA, 1))
            .unwrap()
            .items
    } else {
        Vec::new()
    };
    (
        rep.map(|r| r.items_error).map_err(|e| format!("{e:?}")),
        recs,
    )
}

#[tokio::test]
async fn render_stage_runs_on_the_core_runner() {
    let (errors, recs) = run_render_stage(params(PathBuf::new())).await;
    assert_eq!(errors, Ok(0));
    let result = recs[0].outcome.result.clone().unwrap();
    assert!(result.ok);
    assert_eq!(result.files.len(), 3);
}

#[tokio::test]
async fn failed_validation_fails_the_item_when_strict() {
    // no fonts at all: text cannot render, so validation fails
    let no_fonts = RenderParams {
        fonts: FontConfig {
            system: false,
            dirs: vec![],
        },
        ..params(PathBuf::new())
    };
    let (outcome, recs) = run_render_stage(no_fonts.clone()).await;
    let e = outcome.unwrap_err();
    assert!(
        e.contains("ErrorRateExceeded") && e.contains("errors: 1"),
        "{e}"
    );
    if let Some(r) = recs.first() {
        let err = r.outcome.error.clone().unwrap();
        assert_eq!(err.code, ErrorCode::Validation);
        // the message names the failed check, not only that validation failed
        assert!(err.message.contains("text elements"), "{}", err.message);
    }
    // the same failure with strict off is recorded but does not fail the item
    let (errors, recs) = run_render_stage(RenderParams {
        strict: false,
        ..no_fonts
    })
    .await;
    assert_eq!(errors, Ok(0));
    let result = recs[0].outcome.result.clone().unwrap();
    assert!(!result.ok);
    let failures = result.failures();
    assert!(
        failures
            .iter()
            .any(|f| f.starts_with("svg ") && f.contains("text elements")),
        "{failures:?}"
    );
}

#[test]
fn board_text_with_css_like_words_passes_the_style_check() {
    let (notes, mut board) = inputs();
    board.nodes[0].text = "font: bold (style guide)".into();
    let dir = tempfile::tempdir().unwrap();
    let r = render_all(&notes, &[board], &params(dir.path().to_path_buf())).unwrap();
    let (_, svg) = &r.svg[0];
    assert!(r.svg_text[0].1.contains("font: bold"));
    assert!(
        svg.style_violations.is_empty(),
        "{:?}",
        svg.style_violations
    );
    assert!(svg.ok, "{svg:?}");
}

#[test]
fn em_dashes_in_board_text_are_sanitized() {
    let (notes, mut board) = inputs();
    // real boards carry titles like "Name \u{2014} Subtitle"
    board.nodes[0].text = "Relay API \u{2014} Hackathon".into();
    let dir = tempfile::tempdir().unwrap();
    let r = render_all(&notes, &[board], &params(dir.path().to_path_buf())).unwrap();
    let (_, svg) = &r.svg[0];
    assert!(svg.ok, "{svg:?}");
    assert!(!r.svg_text[0].1.contains('\u{2014}'));
    assert!(!r.markdown_text.contains('\u{2014}'));
}

/// Codex final round 3 MAJOR: the render stage's output is the files it writes
/// outside the run directory, so a rerun never restores it from the cache: files
/// deleted since the last run are written again, not reported from a cached
/// artifact.
#[tokio::test]
async fn a_rerun_rewrites_the_rendered_files() {
    let (notes, board) = inputs();
    let dir = tempfile::tempdir().unwrap();
    let run_path = dir.path().join("run");
    let run = RunDir::open(&run_path, "synthetic", Producer::glassrip("0.1.0", None)).unwrap();
    import::write_artifact(
        &run,
        schemas::MEETING_NOTES,
        Version::new(1, 0, 0),
        serde_json::json!({}),
        vec![("meeting_notes".to_string(), notes)],
    )
    .unwrap();
    import::import_boards(&run, &[board]).unwrap();
    drop(run);
    let out = dir.path().join("out");
    let stage = RenderStage::new(RenderParams {
        out_dir: out.clone(),
        ..params(PathBuf::new())
    });
    let runner = || {
        Runner::new(
            RunDir::open(&run_path, "synthetic", Producer::glassrip("0.1.0", None)).unwrap(),
            StageGraph::new(meeting_mode_stage_decls()).unwrap(),
            // Selected without being forced (`from` would force it).
            &Selection::default(),
            Cache::in_workspace(dir.path()),
            RunnerOptions::default(),
            CancellationToken::new(),
        )
        .unwrap()
    };
    let files = |recs: &[Record<RenderResult>]| -> Vec<PathBuf> {
        recs[0]
            .outcome
            .result
            .as_ref()
            .unwrap()
            .files
            .iter()
            .map(|f| out.join(&f.name))
            .collect()
    };
    let read = |r: &Runner| {
        jsonl::read::<Record<RenderResult>>(
            &r.run_dir().artifact_path(RENDER_SCHEMA),
            &SchemaReq::new(RENDER_SCHEMA, 1),
        )
        .unwrap()
        .items
    };

    let mut r = runner();
    let rep = r.run_stage(&stage).await.unwrap();
    assert_eq!(rep.items_error, 0);
    let written = files(&read(&r));
    assert_eq!(written.len(), 3);
    drop(r);
    assert!(
        Cache::in_workspace(dir.path()).ls().unwrap().is_empty(),
        "nothing is stored for a stage that is never restored"
    );
    for f in &written {
        std::fs::remove_file(f).unwrap();
    }

    let mut r = runner();
    assert!(!r.cache_hit(&stage), "render never restores from the cache");
    let rep = r.run_stage(&stage).await.unwrap();
    assert_ne!(
        rep.status,
        glassrip_core::manifest::StageStatus::Cached,
        "{rep:?}"
    );
    for f in files(&read(&r)) {
        assert!(f.is_file(), "{} was not rewritten", f.display());
    }
}
