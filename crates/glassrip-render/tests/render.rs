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
use glassrip_render::scene::{build_scene, edge_defects, overlaps, Scene, R};
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
    // The dashed edge runs from the kit to the ledger: a straight or L-shaped
    // line would cross the kiosk card, so it is routed around the cards.
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
    // the dashed edge bends around the cards, through none of them
    let dashed = scene
        .edges
        .iter()
        .find(|e| e.dash.is_some())
        .expect("dashed edge");
    assert!(dashed.d.matches(" L ").count() >= 2, "{}", dashed.d);
    assert!(
        edge_defects(&scene).is_empty(),
        "{:?}",
        edge_defects(&scene)
    );
    assert!(
        scene.degraded.is_empty(),
        "{:?}\ncards {:?}\npills {:?}\nedges {:?}",
        scene.degraded,
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

/// The synthetic board with far more annotations than fit: 36 owner tags with
/// long move notes split between one card and one edge, and an edge label that
/// is one unbreakable token of wide glyphs, wider than the canvas.
fn crowded_board() -> (MeetingNotes, BoardStateItem) {
    let (notes, mut board) = inputs();
    let node = board
        .final_nodes()
        .into_iter()
        .find(|n| n.text == "Relay API")
        .unwrap()
        .clone();
    let edge = board
        .final_edges()
        .into_iter()
        .find(|e| {
            let ends = [e.a_text.as_str(), e.b_text.as_str()];
            ends.contains(&"Kiosk App") && ends.contains(&"Relay API")
        })
        .unwrap()
        .clone();
    let on_node = OwnerTarget::Node {
        node_id: node.id.clone(),
        text: node.text.clone(),
    };
    let on_edge = OwnerTarget::Edge {
        edge_id: edge.id.clone(),
        src: edge.src.clone(),
        dst: edge.dst.clone(),
        a_text: edge.a_text.clone(),
        b_text: edge.b_text.clone(),
    };
    let base = board.current_owners()[0].clone();
    board.owner_assignments = (0..36)
        .map(|i| {
            let mut o = base.clone();
            o.person_id = format!("person-{i}");
            o.display_name = format!("Maximiliana{i:02} Example");
            let (target, from) = if i % 2 == 0 {
                (on_node.clone(), on_edge.clone())
            } else {
                (on_edge.clone(), on_node.clone())
            };
            o.target = target;
            o.moved_from = Some(from);
            o
        })
        .collect();
    let long = "W".repeat(320);
    board
        .edges
        .iter_mut()
        .find(|e| e.id == edge.id)
        .unwrap()
        .label = long;
    (notes, board)
}

#[test]
fn a_crowded_board_lists_what_does_not_fit_below_it() {
    let (notes, board) = crowded_board();
    let scene = build_scene(&board, &notes);
    // no overlaps and everything inside the canvas, markers and list included
    assert!(overlaps(&scene).is_empty(), "{:?}", overlaps(&scene));
    let f = scene.footnotes.as_ref().expect("an annotation list");
    assert!(f.items.len() >= 3, "{}", f.items.len());
    // one numbered note per footnote (an edge drawn straight for want of a
    // route is a warning after them, with no number)
    let numbered: Vec<&String> = scene
        .degraded
        .iter()
        .filter(|d| d.contains(" as note "))
        .collect();
    assert_eq!(f.items.len(), numbered.len());
    // the unbreakable label is among them, split over lines inside the list
    assert!(scene
        .degraded
        .iter()
        .any(|d| d.contains("Label on the") && d.contains("WWWW")));
    assert!(scene.edges.iter().all(|e| e
        .label
        .as_ref()
        .is_none_or(|l| !l.text.text.contains("WWWW"))));
    // entries are numbered in order, and every line stays inside the list
    for (i, it) in f.items.iter().enumerate() {
        assert_eq!(it.marker.text.text, (i + 1).to_string());
        assert!(it.marker.r.x >= f.r.x && it.marker.r.bottom() <= f.r.bottom());
        for l in &it.lines {
            let right = l.x + glassrip_render::scene::list_text_w(&l.text);
            assert!(l.x >= f.r.x && right <= f.r.right(), "{l:?} {:?}", f.r);
            assert!(l.y > f.r.y && l.y <= f.r.bottom(), "{l:?} {:?}", f.r);
        }
    }
    // entries do not overlap each other
    let spans: Vec<(f64, f64)> = f
        .items
        .iter()
        .map(|it| {
            let last = it.lines.last().map_or(it.marker.r.bottom(), |l| l.y);
            (it.marker.r.y, last.max(it.marker.r.bottom()))
        })
        .collect();
    for w in spans.windows(2) {
        assert!(w[0].1 <= w[1].0, "{spans:?}");
    }
    // markers on the board point at the entries, one per number at most, and
    // every entry without one says so in its warning
    assert!(!scene.markers.is_empty());
    let numbers: std::collections::BTreeSet<usize> = scene
        .markers
        .iter()
        .map(|m| m.text.text.parse::<usize>().unwrap())
        .collect();
    assert_eq!(numbers.len(), scene.markers.len());
    for n in 1..=f.items.len() {
        let w = &scene.degraded[n - 1];
        assert!(w.contains(&format!("as note {n}")), "{w}");
        assert_eq!(numbers.contains(&n), !w.contains("(no marker"), "{w}");
    }
    // leaders have length and stay inside the canvas
    for l in &scene.leaders {
        assert!((l.0 - l.2).abs() + (l.1 - l.3).abs() > 0.0, "{l:?}");
        for (x, y) in [(l.0, l.1), (l.2, l.3)] {
            assert!(x >= 0.0 && x <= scene.width && y >= 0.0 && y <= scene.height);
        }
    }
    // the list sits below everything else and the canvas grew to hold it
    let (_, plain) = inputs();
    let plain = build_scene(&plain, &notes);
    assert!(scene.height > plain.height);
    for (name, r) in &scene.blocking {
        if name != "annotation list" {
            assert!(r.bottom() <= f.r.y, "{name} {r:?} {:?}", f.r);
        }
    }
    // placement tried harder first: some owners are on the board
    assert!(scene.pills.iter().any(|p| p.fill == "#16a34a"));

    // the full render passes strict validation and reports warnings
    let dir = tempfile::tempdir().unwrap();
    let out = std::env::var("GLASSRIP_RENDER_DUMP_DIR")
        .map(|d| PathBuf::from(d).join("crowded"))
        .unwrap_or_else(|_| dir.path().to_path_buf());
    let r = render_all(&notes, std::slice::from_ref(&board), &params(out)).unwrap();
    assert!(r.ok, "{:?}", r.failures());
    let warnings = r.warnings();
    // the render lays out with the real glyph widths
    let measured = glassrip_render::scene::build_scene_with(
        &board,
        &notes,
        &glassrip_render::svg::TextMeasure::new(&bundled_fonts()),
    );
    assert_eq!(warnings.len(), measured.degraded.len());
    assert!(
        warnings.iter().all(|w| w.starts_with("svg ")),
        "{warnings:?}"
    );
    let text = &r.svg_text[0].1;
    assert!(text.contains("Annotations without room on the board"));
    assert!(text.contains("Listed below the board"));
    // the list's text renders inside the canvas with the real font
    assert!(
        r.svg[0].1.overlaps.iter().all(|o| !o.starts_with("text ")),
        "{:?}",
        r.svg[0].1.overlaps
    );
}

/// Both reviewers of the first version: a relation label on a vertical edge in
/// the rightmost column took its first spot, right of the edge, although that
/// spot ran past the canvas, failing the run. Spots outside the canvas are
/// never taken now; the label goes left, wraps, moves out, or is listed.
#[test]
fn a_relation_label_in_the_rightmost_column_stays_inside_the_canvas() {
    let (notes, mut board) = inputs();
    // Ledger above Design Kit at the far right: the dashed relation between
    // them is a vertical edge in the rightmost column
    for (text, cx, cy) in [
        ("Ledger Service", 1300.0, 200.0),
        ("Relay API", 250.0, 360.0),
        ("Kiosk App", 800.0, 360.0),
        ("Design Kit", 1300.0, 520.0),
    ] {
        let n = board.nodes.iter_mut().find(|n| n.text == text).unwrap();
        let b = n.bbox.as_mut().unwrap();
        (b.x1, b.y1, b.x2, b.y2) = (cx - 100.0, cy - 40.0, cx + 100.0, cy + 40.0);
    }
    let scene = build_scene(&board, &notes);
    let dashed = scene.edges.iter().find(|e| e.dash.is_some()).unwrap();
    assert_eq!(dashed.d.matches(" L ").count(), 1, "vertical: {}", dashed.d);
    // the architecture spans the canvas: the right of the edge has no room
    // for the label (the old first pick ran 147 px past the canvas)
    let right = scene.cards.iter().map(|c| c.r.right()).fold(0.0, f64::max);
    assert!(right >= scene.width - 80.0, "{right} {}", scene.width);
    let l = dashed.label.as_ref().expect("placed, not listed");
    assert!(l.r.x >= 0.0 && l.r.right() <= scene.width, "{:?}", l.r);
    assert!(overlaps(&scene).is_empty(), "{:?}", overlaps(&scene));
    let dir = tempfile::tempdir().unwrap();
    let r = render_all(&notes, &[board], &params(dir.path().to_path_buf())).unwrap();
    assert!(r.ok, "{:?}", r.failures());
}

/// Glyphs past the canvas fail validation, wherever they are (measured on
/// the rendered glyphs, not estimated).
#[test]
fn text_outside_the_canvas_fails() {
    let env = glassrip_render::svg::environment();
    let (notes, board) = inputs();
    let mut scene = build_scene(&board, &notes);
    scene.footer.x = scene.width + 200.0;
    let text = glassrip_render::svg::render_svg(&env, &scene).unwrap();
    let (checks, _) = glassrip_render::svg::validate_svg(&text, &scene, &bundled_fonts());
    assert!(!checks.ok);
    assert!(
        checks
            .overlaps
            .iter()
            .any(|o| o.starts_with("text \"Source:") && o.ends_with("outside the canvas")),
        "{:?}",
        checks.overlaps
    );

    let (notes, board) = crowded_board();
    let mut scene = build_scene(&board, &notes);
    let width = scene.width;
    let f = scene.footnotes.as_mut().unwrap();
    f.items[0].lines[0].x = width + 200.0;
    let text = glassrip_render::svg::render_svg(&env, &scene).unwrap();
    let (checks, _) = glassrip_render::svg::validate_svg(&text, &scene, &bundled_fonts());
    assert!(!checks.ok);
    assert!(
        checks
            .overlaps
            .iter()
            .any(|o| o.starts_with("text \"Label on the") && o.ends_with("outside the canvas")),
        "{:?}",
        checks.overlaps
    );
}

/// The reproducer of the review: a label of wide glyphs on an edge in the
/// rightmost column fit its estimated pill while its real glyphs ran past the
/// canvas. Labels (and all other text) are sized by the real glyph widths
/// now, so the render keeps every glyph inside the canvas and passes.
#[test]
fn wide_glyphs_at_the_right_edge_stay_inside_the_canvas() {
    use glassrip_render::scene::{build_scene_with, Estimate, Measure};
    use glassrip_render::svg::TextMeasure;
    let label = "W".repeat(20);
    let (notes, board) = board_of(
        &[
            ("Top", (1300.0, 200.0)),
            ("Bottom", (1300.0, 700.0)),
            ("Left", (100.0, 450.0)),
        ],
        &[(0, 1, &label, false), (2, 0, "", false)],
    );
    let measure = TextMeasure::new(&bundled_fonts());
    // the estimate is far narrower than the real glyphs
    let real = measure.width(&label, "edge-label");
    assert!(real > Estimate.width(&label, "edge-label") + 40.0, "{real}");
    let scene = build_scene_with(&board, &notes, &measure);
    let e = scene.edges.iter().find(|e| e.label.is_some()).unwrap();
    let l = e.label.as_ref().unwrap();
    assert!(l.r.w >= real, "{:?} {real}", l.r);
    assert!(l.r.x >= 0.0 && l.r.x + l.r.w <= scene.width, "{:?}", l.r);
    let dir = tempfile::tempdir().unwrap();
    let r = render_all(&notes, &[board], &params(dir.path().to_path_buf())).unwrap();
    assert!(r.ok, "{:?}", r.failures());
    assert!(
        r.svg[0]
            .1
            .overlaps
            .iter()
            .all(|o| !o.ends_with("outside the canvas")),
        "{:?}",
        r.svg[0].1.overlaps
    );
}

/// Long text of wide glyphs everywhere (title, card, sticky, decision) is
/// fitted to its box by the real glyph widths: nothing leaves the canvas.
#[test]
fn wide_glyphs_everywhere_are_fitted() {
    let (mut notes, mut board) = inputs();
    let wide = "WMW@".repeat(40);
    notes.title = Some(wide.clone());
    for d in &mut notes.decisions {
        d.text = wide.clone();
    }
    for n in &mut board.nodes {
        n.text = format!("{wide} {}", n.text);
    }
    for s in &mut board.stickies {
        s.text = wide.clone();
    }
    let dir = tempfile::tempdir().unwrap();
    let r = render_all(&notes, &[board], &params(dir.path().to_path_buf())).unwrap();
    assert!(r.ok, "{:?}", r.failures());
}

/// Owner names, history notes and the annotation list with wide glyphs on a
/// crowded board: everything is fitted by the real glyph widths, the render
/// passes, and what does not fit is listed below the board.
#[test]
fn wide_glyphs_in_annotations_are_fitted() {
    let (notes, mut board) = crowded_board();
    for (i, o) in board.owner_assignments.iter_mut().enumerate() {
        o.display_name = format!("WMW{i:02}@WMWMW");
        if let Some(OwnerTarget::Node { text, .. }) = o.moved_from.as_mut() {
            *text = "WMWMWMWM@@ ".repeat(8);
        }
    }
    for n in &mut board.nodes {
        n.text = format!("{} WWMMWWMM@@", n.text);
    }
    let dir = tempfile::tempdir().unwrap();
    let r = render_all(&notes, &[board], &params(dir.path().to_path_buf())).unwrap();
    assert!(r.ok, "{:?}", r.failures());
    assert!(!r.warnings().is_empty());
    assert!(r.svg_text[0]
        .1
        .contains("Annotations without room on the board"));
}

#[test]
fn a_normal_board_has_no_annotation_list() {
    let (notes, board) = inputs();
    let scene = build_scene(&board, &notes);
    assert!(scene.footnotes.is_none());
    assert!(scene.markers.is_empty());
    assert!(scene.degraded.is_empty());
    let dir = tempfile::tempdir().unwrap();
    let r = render_all(&notes, &[board], &params(dir.path().to_path_buf())).unwrap();
    assert!(r.ok, "{:?}", r.failures());
    assert!(r.warnings().is_empty());
    let text = &r.svg_text[0].1;
    assert!(!text.contains("Annotations without room"));
    assert!(!text.contains("Listed below the board"));
}

/// Real defects stay failures: two cards on top of each other are not an
/// annotation and are never moved below the board.
#[test]
fn overlapping_primary_boxes_still_fail() {
    let (notes, board) = inputs();
    let mut scene = build_scene(&board, &notes);
    let first = scene.cards[0].r;
    let i = scene
        .blocking
        .iter()
        .position(|(n, _)| n.starts_with("card ") && !n.ends_with(&scene.cards[0].id))
        .unwrap();
    scene.blocking[i].1 = first;
    let o = overlaps(&scene);
    assert!(o.iter().any(|m| m.contains("overlaps")), "{o:?}");
    let env = glassrip_render::svg::environment();
    let text = glassrip_render::svg::render_svg(&env, &scene).unwrap();
    let (checks, _) = glassrip_render::svg::validate_svg(&text, &scene, &bundled_fonts());
    assert!(!checks.ok);
    assert!(checks.warnings.is_empty());
}

/// Codex r5 integration, MAJOR 4: validation skipped every fallback edge, so
/// a straight fallback over an unrelated card reported ok.
#[test]
fn a_fallback_edge_through_another_card_fails_validation() {
    let (notes, board) = inputs();
    let mut scene = build_scene(&board, &notes);
    assert!(overlaps(&scene).is_empty(), "{:?}", overlaps(&scene));
    let (src, dst) = (scene.routes[0].src.clone(), scene.routes[0].dst.clone());
    let other = scene
        .cards
        .iter()
        .find(|c| c.id != src && c.id != dst)
        .expect("a third card")
        .r;
    // A fallback clear of every card is a warning, not a defect.
    scene.routes[0].fallback = true;
    scene.routes[0].points = vec![(2.0, 2.0), (6.0, 2.0)];
    assert!(
        edge_defects(&scene).is_empty(),
        "{:?}",
        edge_defects(&scene)
    );
    // One straight across an unrelated card fails.
    let y = other.y + other.h / 2.0;
    scene.routes[0].points = vec![(other.x - 30.0, y), (other.x + other.w + 30.0, y)];
    let d = edge_defects(&scene);
    assert!(
        d.iter()
            .any(|m| m.contains("runs through card") && m.contains("fallback")),
        "{d:?}"
    );
    let env = glassrip_render::svg::environment();
    let text = glassrip_render::svg::render_svg(&env, &scene).unwrap();
    let (checks, _) = glassrip_render::svg::validate_svg(&text, &scene, &bundled_fonts());
    assert!(!checks.ok);
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
                // Crowded positions may move a note below the board (a warning,
                // drawn); what this checks is that no label covers anything.
                let o: Vec<String> = overlaps(&scene)
                    .into_iter()
                    .filter(|m| {
                        m.contains("zone title")
                            || m.contains("edge label")
                            || m.contains("runs through card")
                    })
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

/// A board of `cards` (text, center on the board) and `links` (source,
/// target, label, dashed), built from the synthetic board's own records, with
/// no owners.
fn board_of(
    cards: &[(&str, (f64, f64))],
    links: &[(usize, usize, &str, bool)],
) -> (MeetingNotes, BoardStateItem) {
    let (notes, mut board) = inputs();
    let node0 = board.final_nodes()[0].clone();
    let edge0 = board.final_edges()[0].clone();
    board.nodes = cards
        .iter()
        .enumerate()
        .map(|(i, (text, (cx, cy)))| {
            let mut n = node0.clone();
            n.id = format!("node-{i}");
            n.text = (*text).into();
            n.in_final = true;
            n.bbox = Some(glassrip_notes::board::BBox::new(
                cx - 100.0,
                cy - 40.0,
                cx + 100.0,
                cy + 40.0,
            ));
            n
        })
        .collect();
    board.edges = links
        .iter()
        .enumerate()
        .map(|(k, (a, b, label, dashed))| {
            let mut e = edge0.clone();
            e.id = format!("edge-{k}");
            (e.a, e.b) = (format!("node-{a}"), format!("node-{b}"));
            (e.src, e.dst) = (e.a.clone(), e.b.clone());
            (e.a_text, e.b_text) = (cards[*a].0.into(), cards[*b].0.into());
            e.label = (*label).into();
            e.style = if *dashed {
                EdgeStyle::Dashed
            } else {
                EdgeStyle::Solid
            };
            e.in_final = true;
            e
        })
        .collect();
    board.owner_assignments.clear();
    (notes, board)
}

/// Vertices of an edge's drawn path (`M x,y L x,y ...`).
fn path_points(d: &str) -> Vec<(f64, f64)> {
    d.split(['M', 'L'])
        .filter_map(|p| {
            let (x, y) = p.trim().split_once(',')?;
            Some((x.parse().ok()?, y.parse().ok()?))
        })
        .collect()
}

fn card_r(scene: &Scene, i: usize) -> R {
    scene
        .cards
        .iter()
        .find(|c| c.id == format!("node-{i}"))
        .unwrap()
        .r
}

/// Every edge orthogonal, through no card, labels crossed by no edge, no
/// overlaps, nothing drawn straight for want of a route, and the render
/// passes strict validation.
fn assert_clean(notes: &MeetingNotes, board: &BoardStateItem, scene: &Scene) {
    assert!(overlaps(scene).is_empty(), "{:?}", overlaps(scene));
    assert!(
        scene.routes.iter().all(|r| !r.fallback),
        "{:?}",
        scene.degraded
    );
    for r in &scene.routes {
        for w in r.points.windows(2) {
            assert!(
                (w[0].0 - w[1].0).abs() < 0.5 || (w[0].1 - w[1].1).abs() < 0.5,
                "{} is not orthogonal: {:?}",
                r.name,
                r.points
            );
        }
    }
    let dir = tempfile::tempdir().unwrap();
    // GLASSRIP_RENDER_DUMP_DIR keeps the files for a visual check
    let out = std::env::var("GLASSRIP_RENDER_DUMP_DIR")
        .map(|d| PathBuf::from(d).join(file_name(&board.nodes[0].text)))
        .unwrap_or_else(|_| dir.path().to_path_buf());
    let r = render_all(notes, std::slice::from_ref(board), &params(out)).unwrap();
    assert!(r.ok, "{:?}", r.failures());
}

fn file_name(s: &str) -> String {
    glassrip_render::file_safe(&s.to_lowercase())
}

#[test]
fn an_edge_whose_straight_line_crosses_a_card_routes_around_it() {
    let (notes, board) = board_of(
        &[
            ("Gateway", (250.0, 200.0)),
            ("Queue", (250.0, 520.0)),
            ("Store", (250.0, 840.0)),
        ],
        &[(0, 2, "writes", false), (0, 1, "", false)],
    );
    let scene = build_scene(&board, &notes);
    let route = scene
        .routes
        .iter()
        .find(|r| r.dst == "node-2")
        .expect("the long edge");
    // a straight line would run through the queue: the route bends around it
    assert!(route.points.len() >= 4, "{:?}", route.points);
    let queue = card_r(&scene, 1);
    let beside = route
        .points
        .iter()
        .any(|p| p.0 < queue.x || p.0 > queue.x + queue.w);
    assert!(beside, "{:?} {queue:?}", route.points);
    // the short edge keeps its straight line
    let short = scene.routes.iter().find(|r| r.dst == "node-1").unwrap();
    assert_eq!(short.points.len(), 2, "{:?}", short.points);
    assert_clean(&notes, &board, &scene);
}

#[test]
fn parallel_edges_keep_apart() {
    let (notes, board) = board_of(
        &[("Client", (250.0, 300.0)), ("Server", (900.0, 300.0))],
        &[
            (0, 1, "REST", false),
            (0, 1, "gRPC", false),
            (1, 0, "events", false),
            (0, 1, "Shared schema between both", true),
        ],
    );
    let scene = build_scene(&board, &notes);
    assert_eq!(scene.routes.len(), 4);
    // no two edges run along each other closer than a track apart, and no two
    // share an end point on a card
    for (i, a) in scene.routes.iter().enumerate() {
        for b in scene.routes.iter().skip(i + 1) {
            for s in a.points.windows(2) {
                for t in b.points.windows(2) {
                    let hz = |w: &[(f64, f64)]| (w[0].1 - w[1].1).abs() < 0.5;
                    let vt = |w: &[(f64, f64)]| (w[0].0 - w[1].0).abs() < 0.5;
                    let overlap = |a0: f64, a1: f64, b0: f64, b1: f64| {
                        a0.max(a1).min(b0.max(b1)) - a0.min(a1).max(b0.min(b1))
                    };
                    if hz(s) && hz(t) && (s[0].1 - t[0].1).abs() < 12.0 {
                        assert!(
                            overlap(s[0].0, s[1].0, t[0].0, t[1].0) <= 0.0,
                            "{a:?} {b:?}"
                        );
                    }
                    if vt(s) && vt(t) && (s[0].0 - t[0].0).abs() < 12.0 {
                        assert!(
                            overlap(s[0].1, s[1].1, t[0].1, t[1].1) <= 0.0,
                            "{a:?} {b:?}"
                        );
                    }
                }
            }
            for p in [a.points[0], *a.points.last().unwrap()] {
                for q in [b.points[0], *b.points.last().unwrap()] {
                    assert!((p.0 - q.0).abs() + (p.1 - q.1).abs() >= 24.0, "{p:?} {q:?}");
                }
            }
        }
    }
    assert!(scene.edges.iter().all(|e| e.label.is_some()));
    assert_clean(&notes, &board, &scene);
}

#[test]
fn a_label_on_a_vertical_edge_sits_beside_it() {
    let (notes, board) = board_of(
        &[("Frontend", (600.0, 200.0)), ("Backend", (600.0, 700.0))],
        &[(0, 1, "GraphQL", false)],
    );
    let scene = build_scene(&board, &notes);
    let e = &scene.edges[0];
    let pts = path_points(&e.d);
    assert_eq!(pts.len(), 2, "{}", e.d);
    let x = pts[0].0;
    assert!((pts[1].0 - x).abs() < 0.5, "vertical: {}", e.d);
    let l = e.label.as_ref().unwrap().r;
    // beside the line, not across it, close to it, and within its extent
    assert!(l.x + l.w <= x - 4.0 || l.x >= x + 4.0, "{l:?} {x}");
    assert!(
        (l.x + l.w - x).abs() <= 8.0 || (l.x - x).abs() <= 8.0,
        "{l:?} {x}"
    );
    let (y0, y1) = (pts[0].1.min(pts[1].1), pts[0].1.max(pts[1].1));
    assert!(l.y >= y0 && l.y + l.h <= y1, "{l:?} {y0} {y1}");
    assert!(e.leader.is_none());
    assert_clean(&notes, &board, &scene);
}

#[test]
fn a_dense_board_routes_every_edge_around_the_cards() {
    let names = [
        "Web", "Mobile", "Admin", "Partner", "Gateway", "Auth", "Billing", "Search", "Orders",
        "Ledger", "Index", "Archive",
    ];
    let cards: Vec<(&str, (f64, f64))> = names
        .iter()
        .enumerate()
        .map(|(i, n)| {
            (
                *n,
                (
                    150.0 + 400.0 * (i % 4) as f64,
                    150.0 + 300.0 * (i / 4) as f64,
                ),
            )
        })
        .collect();
    let links = [
        (0, 4, "HTTPS", false),
        (1, 4, "HTTPS", false),
        (2, 4, "", false),
        (3, 4, "webhook", false),
        (4, 5, "token", false),
        (4, 6, "", false),
        (4, 7, "query", false),
        (4, 8, "", false),
        (5, 9, "", false),
        (6, 9, "entries", false),
        (8, 9, "", false),
        (7, 10, "", false),
        (10, 11, "", false),
        (0, 11, "exports", true),
        (3, 8, "", false),
        (1, 10, "offline sync of cached data", true),
        (2, 9, "", false),
        (11, 5, "", false),
        (6, 8, "", false),
        (0, 3, "", false),
    ];
    let (notes, board) = board_of(&cards, &links);
    let scene = build_scene(&board, &notes);
    assert_eq!(scene.routes.len(), links.len());
    assert_clean(&notes, &board, &scene);
}

/// Validation names an edge through a card and a label an edge runs through;
/// an edge drawn without a route is a warning while it crosses no other card,
/// and a failure once it does.
#[test]
fn validation_fails_edges_through_cards_and_crossed_labels() {
    let (notes, board) = inputs();
    let clean = build_scene(&board, &notes);
    assert!(
        edge_defects(&clean).is_empty(),
        "{:?}",
        edge_defects(&clean)
    );

    // a route through a card that is not one of its ends
    let mut scene = clean.clone();
    let i = scene
        .routes
        .iter()
        .position(|r| r.src != scene.cards[0].id && r.dst != scene.cards[0].id)
        .unwrap();
    let c = scene.cards[0].r;
    scene.routes[i].points = vec![
        (c.x - 50.0, c.y + c.h / 2.0),
        (c.x + c.w + 50.0, c.y + c.h / 2.0),
    ];
    let d = edge_defects(&scene);
    assert!(d.iter().any(|m| m.contains("runs through card")), "{d:?}");
    let env = glassrip_render::svg::environment();
    let text = glassrip_render::svg::render_svg(&env, &scene).unwrap();
    let (checks, _) = glassrip_render::svg::validate_svg(&text, &scene, &bundled_fonts());
    assert!(!checks.ok);
    // the same line flagged as a fallback still crosses a card that is not
    // one of its ends
    scene.routes[i].fallback = true;
    let d = edge_defects(&scene);
    assert!(
        d.iter()
            .any(|m| m.contains("runs through card") && m.contains("fallback")),
        "{d:?}"
    );

    // a label moved onto its own edge
    let mut scene = clean.clone();
    let k = scene.edges.iter().position(|e| e.label.is_some()).unwrap();
    let (p, q) = (scene.routes[k].points[0], scene.routes[k].points[1]);
    let l = scene.edges[k].label.as_mut().unwrap();
    l.r.x = (p.0 + q.0) / 2.0 - l.r.w / 2.0;
    l.r.y = (p.1 + q.1) / 2.0 - l.r.h / 2.0;
    let d = edge_defects(&scene);
    assert!(
        d.iter().any(|m| m.ends_with("is crossed by its own edge")),
        "{d:?}"
    );
}

/// A history note ("moved from ...") sits right next to its owner pill.
#[test]
fn a_history_note_sits_next_to_its_owner_pill() {
    let (notes, board) = inputs();
    let scene = build_scene(&board, &notes);
    let note = scene
        .blocking
        .iter()
        .find(|(n, _)| n.starts_with("note "))
        .map(|(_, r)| *r)
        .expect("the moved-from note is on the board");
    let gap = |a: &R, b: &R| {
        let dx = (b.x - (a.x + a.w)).max(a.x - (b.x + b.w)).max(0.0);
        let dy = (b.y - (a.y + a.h)).max(a.y - (b.y + b.h)).max(0.0);
        dx.max(dy)
    };
    let nearest = scene
        .pills
        .iter()
        .filter(|p| p.fill == "#16a34a")
        .map(|p| gap(&p.r, &note))
        .fold(f64::INFINITY, f64::min);
    assert!(nearest <= 10.0, "{nearest} {note:?}");
    // no leader needed
    assert!(scene.leaders.iter().all(|l| {
        let (x, y) = (l.0, l.1);
        !(x >= note.x - 1.0
            && x <= note.x + note.w + 1.0
            && y >= note.y - 1.0
            && y <= note.y + note.h + 1.0)
    }));
}
