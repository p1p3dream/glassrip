//! Board-state consolidation on synthetic reading sequences (fictional board and names).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use glassrip_meeting::artifacts::{CanvasDims, EdgeDirectionItem, EdgeEvidence};
use glassrip_meeting::consolidate::events::EventKind;
use glassrip_meeting::consolidate::events::SuppressReason;
use glassrip_meeting::consolidate::owners::{
    AnchorKind, Corroboration, Corroborator, MoveQuery, NameRead, NoCorroboration, OpenReason,
    OwnerTarget,
};
use glassrip_meeting::consolidate::{
    consolidate, consolidate_with_probe, split_boards, BoardFrame, BoardStateItem, CanvasSource,
    ConsolidationParams, EdgeOrientation, FoldReason, Hooks, RegionProbe, StickyKind, TextAnchor,
};
use glassrip_meeting::direction::{DirectionBasis, EdgeDirection, EndVerdict};
use glassrip_meeting::pixel_direction::{EndEvidence, PixelEvidence, PixelStatus};
use glassrip_meeting::register::RegistrationMode;
use glassrip_meeting::similarity::Similarity;
use glassrip_meeting::text::{AliasTable, Participant};
use glassrip_vision::board::{
    BoardEdge, BoardNode, EdgeStyle, OwnerTag, Sticky, StickyColor, TextItem, ValidatedBoard,
};
use glassrip_vision::BBox;

const W: f64 = 1600.0;
const H: f64 = 900.0;

/// Fictional board in reference coordinates: (id, text, center).
const NODES: [(&str, &str, (f64, f64)); 4] = [
    ("n1", "Ingest Gateway", (200.0, 200.0)),
    ("n2", "Queue", (700.0, 220.0)),
    ("n3", "Ledger Store", (1200.0, 200.0)),
    ("n4", "Report Builder", (700.0, 650.0)),
];
const STICKIES: [(&str, (f64, f64)); 3] = [
    ("Is the queue durable?", (1100.0, 600.0)),
    ("Idea: batch the writes", (300.0, 600.0)),
    ("Beta milestone in March", (1400.0, 750.0)),
];

fn bbox_at(t: &Similarity, c: (f64, f64), hw: f64, hh: f64) -> BBox {
    let p = t.apply(c);
    BBox::new(
        p.0 - hw * t.scale,
        p.1 - hh * t.scale,
        p.0 + hw * t.scale,
        p.1 + hh * t.scale,
    )
}

struct Spec {
    t: Similarity,
    nodes: Vec<(&'static str, String, (f64, f64))>,
    edges: Vec<(&'static str, &'static str, &'static str)>,
    stickies: Vec<(&'static str, (f64, f64))>,
    owners: Vec<(&'static str, (f64, f64), &'static str)>,
    ink: Option<f64>,
}

fn base() -> Spec {
    Spec {
        t: Similarity::IDENTITY,
        nodes: NODES
            .iter()
            .map(|(i, s, c)| (*i, s.to_string(), *c))
            .collect(),
        edges: vec![("n1", "n2", "HTTP"), ("n2", "n3", "gRPC"), ("n2", "n4", "")],
        stickies: STICKIES.to_vec(),
        owners: vec![],
        ink: Some(0.2),
    }
}

fn board(s: &Spec) -> ValidatedBoard {
    let t = &s.t;
    let inside = |b: &BBox| b.x1 >= 0.0 && b.y1 >= 0.0 && b.x2 <= W && b.y2 <= H;
    let nodes: Vec<BoardNode> = s
        .nodes
        .iter()
        .map(|(id, text, c)| BoardNode {
            local_id: id.to_string(),
            text: text.clone(),
            bbox: bbox_at(t, *c, 90.0, 40.0),
            conf: 0.9,
        })
        .filter(|n| inside(&n.bbox))
        .collect();
    let ids: Vec<&str> = nodes.iter().map(|n| n.local_id.as_str()).collect();
    let edges = s
        .edges
        .iter()
        .filter(|(a, b, _)| ids.contains(a) && ids.contains(b))
        .map(|(a, b, l)| BoardEdge {
            src: a.to_string(),
            dst: b.to_string(),
            label: l.to_string(),
            label_bbox: None,
            style: EdgeStyle::Solid,
            conf: 0.8,
        })
        .collect();
    let stickies = s
        .stickies
        .iter()
        .map(|(text, c)| Sticky {
            text: text.to_string(),
            color: StickyColor::Yellow,
            bbox: bbox_at(t, *c, 60.0, 50.0),
        })
        .filter(|x| inside(&x.bbox))
        .collect();
    let owner_tags = s
        .owners
        .iter()
        .map(|(name, c, near)| OwnerTag {
            name_raw: name.to_string(),
            near: near.to_string(),
            bbox: bbox_at(t, *c, 30.0, 20.0),
        })
        .collect();
    ValidatedBoard {
        nodes,
        edges,
        stickies,
        owner_tags,
        other_visible_text: vec![TextItem {
            text: "Platform sketch".into(),
            bbox: bbox_at(t, (800.0, 60.0), 120.0, 15.0),
        }],
        confidence: 0.9,
        chrome_rejected: vec![],
        issues: vec![],
        needs_reclassification: false,
    }
}

fn frames(specs: &[Spec]) -> Vec<BoardFrame> {
    specs
        .iter()
        .enumerate()
        .map(|(i, s)| BoardFrame {
            keyframe_id: format!("kf{i:02}"),
            keyframe_index: i,
            t_start_s: i as f64 * 20.0,
            t_end_s: i as f64 * 20.0 + 20.0,
            t_rep_s: i as f64 * 20.0 + 10.0,
            canvas: Some(CanvasDims {
                width: W,
                height: H,
            }),
            board: board(s),
            ink_change: s.ink,
            directions: None,
            board_title: None,
            ocr_anchors: vec![],
            title_hints: vec![],
        })
        .collect()
}

fn hooks(c: &dyn Corroborator) -> Hooks<'_> {
    Hooks {
        corroborator: c,
        second_reader: None,
    }
}

fn run(frames: Vec<BoardFrame>, p: &ConsolidationParams) -> BoardStateItem {
    consolidate(frames, "board-1", p, &hooks(&NoCorroboration))
}

fn params() -> ConsolidationParams {
    ConsolidationParams {
        participants: AliasTable::new(vec![
            Participant {
                person_id: "p-avery".into(),
                display_name: "Avery".into(),
                aliases: vec![],
            },
            Participant {
                person_id: "p-jordan".into(),
                display_name: "Jordan".into(),
                aliases: vec!["Jordy".into()],
            },
        ]),
        final_window_s: 60.0,
        ..ConsolidationParams::default()
    }
}

fn node_texts(s: &BoardStateItem) -> Vec<String> {
    let mut v: Vec<String> = s
        .nodes
        .iter()
        .filter(|n| n.in_final)
        .map(|n| n.text.clone())
        .collect();
    v.sort();
    v
}

fn content_events(s: &BoardStateItem) -> Vec<(EventKind, String)> {
    s.events
        .iter()
        .filter(|e| !e.baseline)
        .map(|e| (e.kind, e.keyframe_id.clone()))
        .collect()
}

#[test]
fn pan_and_zoom_register_and_produce_no_content_events() {
    let moves = [
        Similarity::IDENTITY,
        Similarity {
            scale: 1.0,
            angle: 0.0,
            tx: -120.0,
            ty: 30.0,
        },
        Similarity {
            scale: 1.3,
            angle: 0.0,
            tx: -250.0,
            ty: -80.0,
        },
        Similarity {
            scale: 0.8,
            angle: 0.0,
            tx: 100.0,
            ty: 60.0,
        },
        Similarity {
            scale: 1.0,
            angle: 0.0,
            tx: 0.0,
            ty: 0.0,
        },
    ];
    let specs: Vec<Spec> = moves
        .iter()
        .map(|t| Spec {
            t: *t,
            ink: Some(0.01),
            ..base()
        })
        .collect();
    let s = run(frames(&specs), &params());
    let clusters: Vec<usize> = s
        .registration
        .iter()
        .map(|r| r.registration.cluster)
        .collect();
    assert!(clusters.iter().all(|c| *c == clusters[0]), "{clusters:?}");
    assert!(s
        .registration
        .iter()
        .all(|r| r.registration.mode != RegistrationMode::TextOnly));
    assert_eq!(content_events(&s), vec![], "{:#?}", s.events);
    assert!(s.events.iter().all(|e| e.baseline));
    assert!(s.suppressed_events.is_empty(), "{:#?}", s.suppressed_events);
    assert_eq!(
        node_texts(&s),
        vec!["Ingest Gateway", "Ledger Store", "Queue", "Report Builder"]
    );
    // Zoomed frame 2 cuts off part of the board; nothing is removed because absence
    // is only counted where the element's place is inside the view.
    assert!(s
        .nodes
        .iter()
        .all(|n| n.lifetimes.iter().all(|l| l.removed_at_s.is_none())));
}

#[test]
fn additions_removals_and_label_changes_are_ink_gated() {
    let mut specs: Vec<Spec> = (0..8)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    // A node appears at keyframe 3 with a real ink change.
    for s in specs.iter_mut().skip(3) {
        s.nodes.push(("n5", "Archive".into(), (1200.0, 650.0)));
    }
    specs[3].ink = Some(0.3);
    // The queue box is relabelled at keyframe 5 and stays relabelled.
    for s in specs.iter_mut().skip(5) {
        s.nodes[1].1 = "Event Bus".into();
    }
    specs[5].ink = Some(0.2);
    // The report builder box is erased at keyframe 6 (its place stays in view).
    for s in specs.iter_mut().skip(6) {
        s.nodes.retain(|n| n.0 != "n4");
        s.edges.retain(|e| e.1 != "n4");
    }
    specs[6].ink = Some(0.2);
    let s = run(frames(&specs), &params());
    let ev = content_events(&s);
    assert!(
        ev.contains(&(EventKind::NodeAdded, "kf03".into())),
        "{ev:?}"
    );
    assert!(
        ev.contains(&(EventKind::LabelChanged, "kf05".into())),
        "{ev:?}"
    );
    assert!(
        ev.contains(&(EventKind::NodeRemoved, "kf06".into())),
        "{ev:?}"
    );
    assert!(
        ev.contains(&(EventKind::EdgeRemoved, "kf06".into())),
        "{ev:?}"
    );
    assert_eq!(
        node_texts(&s),
        vec!["Archive", "Event Bus", "Ingest Gateway", "Ledger Store"]
    );
    // The same addition without an ink change is suppressed, not emitted.
    specs[3].ink = Some(0.01);
    let s = run(frames(&specs), &params());
    assert!(!content_events(&s).contains(&(EventKind::NodeAdded, "kf03".into())));
    assert!(s
        .suppressed_events
        .iter()
        .any(|e| e.event.kind == EventKind::NodeAdded && e.event.keyframe_id == "kf03"));
}

#[test]
fn single_sightings_and_unresolved_endpoints_do_not_create_nodes() {
    let mut specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    // One keyframe reads a spurious node plus an edge to it.
    specs[2]
        .nodes
        .push(("n9", "Phantom Box".into(), (1400.0, 450.0)));
    specs[2].edges.push(("n3", "n9", ""));
    let s = run(frames(&specs), &params());
    assert!(!s.nodes.iter().any(|n| n.text == "Phantom Box"));
    assert!(!s
        .edges
        .iter()
        .any(|e| e.a_text == "Phantom Box" || e.b_text == "Phantom Box"));
    assert_eq!(s.edges.iter().filter(|e| e.in_final).count(), 3);
    let kinds: Vec<(String, StickyKind)> = s
        .stickies
        .iter()
        .map(|x| (x.text.clone(), x.kind))
        .collect();
    assert!(kinds.contains(&("Is the queue durable?".into(), StickyKind::Question)));
    assert!(kinds.contains(&("Idea: batch the writes".into(), StickyKind::Idea)));
    assert!(kinds.contains(&("Beta milestone in March".into(), StickyKind::Milestone)));
}

#[test]
fn readings_without_geometry_merge_by_text_only() {
    let specs: Vec<Spec> = (0..4).map(|_| base()).collect();
    let mut fr = frames(&specs);
    // Every element gets the same full-canvas box, as when readings carry no boxes.
    let full = BBox::new(0.0, 0.0, W, H);
    for f in &mut fr {
        for n in &mut f.board.nodes {
            n.bbox = full;
        }
        for x in &mut f.board.stickies {
            x.bbox = full;
        }
        for x in &mut f.board.other_visible_text {
            x.bbox = full;
        }
    }
    // A slightly different reading of one label still merges (ratio >= 0.85).
    fr[1].board.nodes[0].text = "Ingest Gatway".into();
    let s = run(fr, &params());
    assert!(s
        .registration
        .iter()
        .all(|r| r.registration.mode == RegistrationMode::TextOnly));
    assert_eq!(s.nodes.len(), 4);
    let g = s
        .nodes
        .iter()
        .find(|n| n.text == "Ingest Gateway")
        .expect("merged");
    assert_eq!(g.variants.len(), 2);
}

fn end(x: f64, y: f64) -> Option<EndEvidence> {
    Some(EndEvidence {
        x,
        y,
        node_distance_px: 2.0,
        area_ratio: 1.0,
        spread_px: 1.0,
        off_axis_min_side: 0,
        arrow: false,
    })
}

fn evidence(src: &str, dst: &str, pixel: EndVerdict, vlm: Option<EndVerdict>) -> EdgeEvidence {
    EdgeEvidence {
        src: src.into(),
        dst: dst.into(),
        src_text: String::new(),
        dst_text: String::new(),
        label: String::new(),
        style: EdgeStyle::Solid,
        pixel: PixelEvidence {
            status: PixelStatus::Traced,
            src_end: end(0.0, 0.0),
            dst_end: end(10.0, 0.0),
            stroke_px: 2.0,
            verdict: pixel,
        },
        vlm: vlm.map(|v| glassrip_meeting::vlm_direction::VlmEvidence {
            answer: glassrip_meeting::vlm_direction::ArrowheadSide::Unclear,
            src_point: (0.0, 0.0),
            dst_point: (10.0, 0.0),
            verdict: v,
        }),
    }
}

fn dir_item(keyframe_id: &str, edges: Vec<EdgeEvidence>) -> EdgeDirectionItem {
    EdgeDirectionItem {
        keyframe_id: keyframe_id.into(),
        canvas: CanvasDims {
            width: W,
            height: H,
        },
        board_to_image: glassrip_meeting::artifacts::AxisScale { x: 1.0, y: 1.0 },
        coordinates: glassrip_meeting::artifacts::CoordinateCheck::Verified1to1,
        crop_in_frame: None,
        sharpness: 300.0,
        zoom: 80.0,
        edges,
        error: None,
    }
}

#[test]
fn pixel_verdict_is_authoritative_and_vlm_fills_inconclusive_edges() {
    let specs: Vec<Spec> = (0..4).map(|_| base()).collect();
    let mut fr = frames(&specs);
    for f in &mut fr {
        f.directions = Some(dir_item(
            &f.keyframe_id,
            vec![
                // Reader says n1 -> n2 but the head is at n1; the VLM disagrees and is
                // ignored because the pixel vote is decisive.
                evidence("n1", "n2", EndVerdict::Reverse, Some(EndVerdict::Forward)),
                // Pixels found no head: the VLM answer decides.
                evidence(
                    "n2",
                    "n3",
                    EndVerdict::NoArrowhead,
                    Some(EndVerdict::Reverse),
                ),
                // Only the pixel check ran.
                evidence("n2", "n4", EndVerdict::Forward, None),
            ],
        ));
    }
    let s = run(fr, &params());
    let by = |a: &str, b: &str| {
        s.edges
            .iter()
            .find(|e| (e.a_text == a && e.b_text == b) || (e.a_text == b && e.b_text == a))
            .expect("edge")
            .clone()
    };
    let name = |id: &str| {
        s.nodes
            .iter()
            .find(|n| n.id == id)
            .map(|n| n.text.clone())
            .unwrap_or_default()
    };
    let e1 = by("Ingest Gateway", "Queue");
    assert_eq!(
        (name(&e1.src), name(&e1.dst)),
        ("Queue".into(), "Ingest Gateway".into())
    );
    assert_eq!(
        (e1.direction, e1.direction_basis),
        (EdgeOrientation::Forward, DirectionBasis::Pixel)
    );
    let e2 = by("Queue", "Ledger Store");
    assert_eq!(
        (name(&e2.src), name(&e2.dst)),
        ("Ledger Store".into(), "Queue".into())
    );
    assert_eq!(e2.direction_basis, DirectionBasis::Vlm);
    let e3 = by("Queue", "Report Builder");
    assert_eq!(
        (name(&e3.src), name(&e3.dst)),
        ("Queue".into(), "Report Builder".into())
    );
    assert_ne!(e3.decision, EdgeDirection::Uncertain);
}

#[test]
fn edges_without_any_verdict_stay_uncertain() {
    let specs: Vec<Spec> = (0..3).map(|_| base()).collect();
    let mut fr = frames(&specs);
    for f in &mut fr {
        f.directions = Some(dir_item(
            &f.keyframe_id,
            vec![evidence("n1", "n2", EndVerdict::NoArrowhead, None)],
        ));
    }
    let s = run(fr, &params());
    let e = s.edges.iter().find(|e| e.label == "HTTP").expect("edge");
    assert_eq!(e.direction, EdgeOrientation::Uncertain);
    assert_eq!(e.direction_basis, DirectionBasis::None);
}

struct CueAt(f64);
impl Corroborator for CueAt {
    fn corroborate(&self, q: &MoveQuery<'_>) -> Option<Corroboration> {
        (q.from.is_some() && (q.t_start_s - self.0).abs() <= 30.0).then(|| Corroboration {
            source: "transcript".into(),
            t_s: self.0,
            detail: "synthetic cue".into(),
        })
    }
}

#[test]
fn owners_are_timed_edge_anchored_and_filtered() {
    let mut specs: Vec<Spec> = (0..8).map(|_| base()).collect();
    for (i, s) in specs.iter_mut().enumerate() {
        // Jordan's tag sits on the Queue -> Ledger Store connector the whole time.
        s.owners.push(("Jordy", (950.0, 210.0), ""));
        // Avery starts next to Ingest Gateway and moves to Report Builder at kf05.
        let avery = if i < 5 {
            (200.0, 120.0)
        } else {
            (840.0, 650.0)
        };
        s.owners.push(("Avery", avery, ""));
        // A name that is not a participant.
        s.owners.push(("Morgan", (1400.0, 100.0), ""));
    }
    // A one-off misplaced reading of Avery (for example a video tile name).
    specs[2].owners[1].1 = (1200.0, 120.0);
    let s = run(frames(&specs), &params());
    let jordan: Vec<_> = s
        .owner_assignments
        .iter()
        .filter(|a| a.person_id == "p-jordan")
        .collect();
    assert_eq!(jordan.len(), 1);
    assert!(
        matches!(jordan[0].target, OwnerTarget::Edge { .. }),
        "{:?}",
        jordan[0].target
    );
    assert!(jordan[0]
        .sightings
        .iter()
        .all(|x| x.anchor == AnchorKind::GeometryEdge));
    let avery: Vec<_> = s
        .owner_assignments
        .iter()
        .filter(|a| a.person_id == "p-avery")
        .collect();
    assert_eq!(avery.len(), 2, "{avery:#?}");
    assert_eq!(avery[0].target.texts(), vec!["Ingest Gateway"]);
    assert_eq!(avery[1].target.texts(), vec!["Report Builder"]);
    assert_eq!(avery[1].valid_from_s, 100.0);
    assert!(avery[0].valid_at(50.0) && avery[1].valid_at(150.0));
    assert!(s.rejected_owner_tags.iter().all(|r| r.name_raw == "Morgan"));
    assert_eq!(s.rejected_owner_tags.len(), 8);
    let moved: Vec<_> = s
        .events
        .iter()
        .filter(|e| e.kind == EventKind::OwnerMoved)
        .collect();
    assert_eq!(moved.len(), 1);
    assert_eq!(moved[0].keyframe_id, "kf05");

    // With only one sighting after the move, a transcript cue confirms it.
    let mut short: Vec<Spec> = (0..6).map(|_| base()).collect();
    for (i, s) in short.iter_mut().enumerate() {
        let avery = if i < 5 {
            (200.0, 120.0)
        } else {
            (840.0, 650.0)
        };
        s.owners.push(("Avery", avery, ""));
    }
    let no = run(frames(&short), &params());
    assert_eq!(no.owner_assignments.len(), 1);
    let yes = consolidate(frames(&short), "board-1", &params(), &hooks(&CueAt(100.0)));
    assert_eq!(yes.owner_assignments.len(), 2);
    assert_eq!(yes.owner_assignments[1].valid_from_s, 100.0);
}

#[test]
fn misread_participant_node_becomes_an_owner_sighting() {
    let mut specs: Vec<Spec> = (0..4).map(|_| base()).collect();
    for s in &mut specs {
        // "Jordann" is not an exact participant name, so validation keeps it as a node.
        s.nodes.push(("n7", "Jordann".into(), (1000.0, 420.0)));
        s.edges.push(("n7", "n2", ""));
    }
    let s = run(frames(&specs), &params());
    assert!(!s.nodes.iter().any(|n| n.text == "Jordann"));
    assert!(!s
        .edges
        .iter()
        .any(|e| e.a_text == "Jordann" || e.b_text == "Jordann"));
    let jordan: Vec<_> = s
        .owner_assignments
        .iter()
        .filter(|a| a.person_id == "p-jordan")
        .collect();
    assert_eq!(jordan.len(), 1, "{:#?}", s.owner_assignments);
}

fn only_two_nodes() -> Spec {
    Spec {
        nodes: NODES[..2]
            .iter()
            .map(|(i, s, c)| (*i, s.to_string(), *c))
            .collect(),
        edges: vec![("n1", "n2", "")],
        stickies: vec![],
        ..base()
    }
}

#[test]
fn ocr_spans_register_sparse_keyframes() {
    let moves = [
        Similarity::IDENTITY,
        Similarity {
            scale: 1.1,
            angle: 0.0,
            tx: -90.0,
            ty: 40.0,
        },
    ];
    let specs: Vec<Spec> = moves
        .iter()
        .map(|t| Spec {
            t: *t,
            ..only_two_nodes()
        })
        .collect();
    let mut fr = frames(&specs);
    for f in &mut fr {
        f.board.other_visible_text.clear();
    }
    // Two shared labels are not enough to register.
    let s = run(fr.clone(), &params());
    assert!(s
        .registration
        .iter()
        .all(|r| r.registration.mode == glassrip_meeting::register::RegistrationMode::TextOnly));
    // OCR spans inside the canvas add anchors.
    let ocr = [
        ("retention thirty days", (1100.0, 500.0)),
        ("replay window", (300.0, 700.0)),
        ("shard map v2", (900.0, 820.0)),
    ];
    for (f, t) in fr.iter_mut().zip(&moves) {
        f.ocr_anchors = ocr
            .iter()
            .map(|(text, c)| TextAnchor {
                text: text.to_string(),
                bbox: bbox_at(t, *c, 50.0, 10.0),
            })
            .collect();
    }
    let s = run(fr, &params());
    assert!(s
        .registration
        .iter()
        .all(|r| r.registration.mode != glassrip_meeting::register::RegistrationMode::TextOnly));
    assert!(s.registration.iter().all(|r| r.ocr_anchors == 3));
}

fn other_board(i: usize) -> Spec {
    let _ = i;
    Spec {
        nodes: vec![
            ("n1", "Alpha Panel".into(), (200.0, 200.0)),
            ("n2", "Beta Panel".into(), (700.0, 200.0)),
            ("n3", "Gamma Panel".into(), (700.0, 600.0)),
        ],
        edges: vec![("n1", "n2", ""), ("n2", "n3", "")],
        stickies: vec![],
        ..base()
    }
}

#[test]
fn two_boards_split_into_two_items() {
    let mut specs: Vec<Spec> = (0..3).map(|_| base()).collect();
    specs.extend((0..3).map(other_board));
    let mut fr = frames(&specs);
    for f in fr.iter_mut().skip(3) {
        f.board.other_visible_text.clear();
    }
    // A classify switch (keyframe 3 is not a board keyframe) separates the boards.
    let mut gap = fr.clone();
    for f in gap.iter_mut().skip(3) {
        f.keyframe_index += 1;
    }
    let boards = split_boards(gap, &params());
    assert_eq!(boards.len(), 2);
    assert_eq!(boards[0].len(), 3);
    assert!(boards[1]
        .iter()
        .all(|f| f.board.nodes[0].text == "Alpha Panel"));
    let items: Vec<BoardStateItem> = boards
        .into_iter()
        .enumerate()
        .map(|(i, b)| {
            consolidate(
                b,
                &format!("board-{}", i + 1),
                &params(),
                &hooks(&NoCorroboration),
            )
        })
        .collect();
    assert_eq!(items[1].board_id, "board-2");
    assert_eq!(items[1].nodes.len(), 3);
    assert!(!items[0].nodes.iter().any(|n| n.text == "Alpha Panel"));
    // In one contiguous run with no title difference, nothing separates them: one
    // board, merged by text only.
    assert_eq!(split_boards(fr.clone(), &params()).len(), 1);
    // Distinct board titles separate them even in one run.
    for (i, f) in fr.iter_mut().enumerate() {
        f.board_title = Some(
            if i < 3 {
                "Platform sketch board"
            } else {
                "Hiring loop board"
            }
            .into(),
        );
    }
    assert_eq!(split_boards(fr, &params()).len(), 2);
}

#[test]
fn single_sighting_needs_pixel_check_and_confidence() {
    let make = |conf: f64, traced: bool| {
        let mut specs: Vec<Spec> = (0..5).map(|_| base()).collect();
        specs[2]
            .nodes
            .push(("n9", "Late Box".into(), (1400.0, 450.0)));
        specs[2].edges.push(("n3", "n9", ""));
        let mut fr = frames(&specs);
        for n in &mut fr[2].board.nodes {
            if n.local_id == "n9" {
                n.conf = conf;
            }
        }
        let mut ev = evidence("n3", "n9", EndVerdict::Forward, None);
        if !traced {
            ev.pixel.dst_end = None;
            ev.pixel.status = PixelStatus::OneEnd;
        }
        fr[2].directions = Some(dir_item("kf02", vec![ev]));
        run(fr, &params())
    };
    assert!(make(0.9, true).nodes.iter().any(|n| n.text == "Late Box"));
    assert!(!make(0.5, true).nodes.iter().any(|n| n.text == "Late Box"));
    assert!(!make(0.9, false).nodes.iter().any(|n| n.text == "Late Box"));
}

#[test]
fn unknown_ink_suppresses_content_events() {
    let mut specs: Vec<Spec> = (0..6)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    for s in specs.iter_mut().skip(3) {
        s.nodes.push(("n5", "Archive".into(), (1200.0, 650.0)));
    }
    specs[3].ink = None;
    let s = run(frames(&specs), &params());
    assert!(!s
        .events
        .iter()
        .any(|e| e.kind == EventKind::NodeAdded && !e.baseline));
    assert!(s.suppressed_events.iter().any(|e| {
        e.event.kind == EventKind::NodeAdded
            && e.event.keyframe_id == "kf03"
            && e.reason == SuppressReason::InkUnknown
    }));
}

#[test]
fn rotated_and_scaled_views_register_without_content_events() {
    let moves = [
        Similarity::IDENTITY,
        Similarity {
            scale: 1.25,
            angle: 0.12,
            tx: -150.0,
            ty: -140.0,
        },
        Similarity {
            scale: 0.8,
            angle: -0.08,
            tx: 160.0,
            ty: 120.0,
        },
        Similarity {
            scale: 1.0,
            angle: 0.05,
            tx: 20.0,
            ty: -30.0,
        },
    ];
    let specs: Vec<Spec> = moves
        .iter()
        .map(|t| Spec {
            t: *t,
            ink: Some(0.01),
            ..base()
        })
        .collect();
    let s = run(frames(&specs), &params());
    let clusters: Vec<usize> = s
        .registration
        .iter()
        .map(|r| r.registration.cluster)
        .collect();
    assert!(clusters.iter().all(|c| *c == clusters[0]), "{clusters:?}");
    assert!(s
        .registration
        .iter()
        .all(|r| r.registration.mode != glassrip_meeting::register::RegistrationMode::TextOnly));
    assert!(s.events.iter().all(|e| e.baseline), "{:#?}", s.events);
    assert_eq!(s.nodes.len(), 4);
    assert!(s
        .nodes
        .iter()
        .all(|n| n.registration == glassrip_meeting::consolidate::ElementRegistration::Position));
}

#[test]
fn canvas_source_is_recorded() {
    let specs: Vec<Spec> = (0..3).map(|_| base()).collect();
    let mut fr = frames(&specs);
    fr[1].canvas = None;
    let s = run(fr, &params());
    assert_eq!(s.registration[0].canvas_source, CanvasSource::Reading);
    assert_eq!(s.registration[1].canvas_source, CanvasSource::Extent);
}

#[test]
fn final_window_holds_at_least_three_keyframes() {
    let specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    let p = ConsolidationParams {
        final_window_s: 1.0,
        ..params()
    };
    let s = run(frames(&specs), &p);
    let w = s.final_window.expect("window");
    assert_eq!(w.keyframe_ids, vec!["kf03", "kf04", "kf05"]);
}

#[test]
fn output_matches_the_eval_contract_names() {
    let mut specs: Vec<Spec> = (0..4).map(|_| base()).collect();
    for s in &mut specs {
        s.owners.push(("Jordy", (950.0, 210.0), ""));
    }
    let s = run(frames(&specs), &params());
    let v = serde_json::to_value(&s).unwrap();
    assert_eq!(v["final"], serde_json::json!(true));
    assert!(v["t_end_s"].is_number());
    assert!(v["nodes"][0]["node_id"].is_string());
    assert!(v["nodes"][0]["in_final_state"].is_boolean());
    assert!(v["nodes"][0]["variants"][0].is_string());
    let dir = v["edges"][0]["direction"].as_str().unwrap();
    assert!(["forward", "uncertain", "bidirectional"].contains(&dir));
    assert!(v["edges"][0]["src"].is_string() && v["edges"][0]["dst"].is_string());
    let t = &v["owner_assignments"][0]["target"];
    assert_eq!(t["kind"], serde_json::json!("edge"));
    assert!(t["src"].is_string() && t["dst"].is_string());
    assert!(v["owner_assignments"][0]["name_raw"].is_string());
    assert!(v["events"][0]["event_id"].is_string());
}

#[test]
fn corroborated_title_bar_and_app_panel_text_are_not_content() {
    let mut fr = frames(&(0..4).map(|_| base()).collect::<Vec<_>>());
    for (i, f) in fr.iter_mut().enumerate() {
        // The board title in the title bar band in every keyframe (persistent).
        f.board.nodes.push(BoardNode {
            local_id: "t1".into(),
            text: "Widget Platform Plan (Draft)".into(),
            bbox: BBox::new(20.0, 8.0, 330.0, 30.0),
            conf: 0.9,
        });
        // An app panel entry (board list, elided) read as a node in one keyframe.
        f.title_hints = vec!["Quarterly Roadmap Review Bo...".into()];
        if i == 2 {
            f.board.nodes.push(BoardNode {
                local_id: "t3".into(),
                text: "Quarterly Roadmap Review Board".into(),
                bbox: BBox::new(900.0, 800.0, 1200.0, 830.0),
                conf: 0.9,
            });
        }
    }
    let s = run(fr, &params());
    let texts: Vec<&str> = s.nodes.iter().map(|n| n.text.as_str()).collect();
    assert!(
        !texts
            .iter()
            .any(|t| t.contains("Widget Platform") || t.contains("Roadmap")),
        "{texts:?}"
    );
    assert_eq!(
        s.board_title.as_deref(),
        Some("Widget Platform Plan (Draft)")
    );
    assert_eq!(s.nodes.len(), 4);
}

#[test]
fn a_same_text_node_far_from_the_title_bar_survives() {
    let mut fr = frames(&(0..4).map(|_| base()).collect::<Vec<_>>());
    for f in fr.iter_mut() {
        f.board.nodes.push(BoardNode {
            local_id: "t1".into(),
            text: "Widget Platform Plan (Draft)".into(),
            bbox: BBox::new(20.0, 8.0, 330.0, 30.0),
            conf: 0.9,
        });
        // A real box on the board that captions the plan with the same words.
        f.board.nodes.push(BoardNode {
            local_id: "t2".into(),
            text: "Widget Platform Plan (Draft)".into(),
            bbox: BBox::new(900.0, 700.0, 1200.0, 760.0),
            conf: 0.9,
        });
    }
    let s = run(fr, &params());
    let n: Vec<_> = s
        .nodes
        .iter()
        .filter(|n| n.text.contains("Widget Platform"))
        .collect();
    assert_eq!(
        n.len(),
        1,
        "{:?}",
        s.nodes.iter().map(|n| &n.text).collect::<Vec<_>>()
    );
    assert!(n[0].in_final);
}

#[test]
fn a_real_node_near_the_top_edge_survives() {
    // A box whose bottom sits inside the top band in one zoomed keyframe, and lower
    // down in the others: not persistent, not corroborated, so it is content.
    let mut fr = frames(&(0..4).map(|_| base()).collect::<Vec<_>>());
    for (i, f) in fr.iter_mut().enumerate() {
        let y = if i == 1 { 5.0 } else { 300.0 };
        f.board.nodes.push(BoardNode {
            local_id: "t9".into(),
            text: "Edge Cache Tier".into(),
            bbox: BBox::new(1300.0, y, 1500.0, y + 35.0),
            conf: 0.9,
        });
    }
    // Also a single band reading of a board-like phrase never seen elsewhere.
    fr[3].board.other_visible_text.push(TextItem {
        text: "Settlement Batch Window".into(),
        bbox: BBox::new(600.0, 5.0, 900.0, 30.0),
    });
    let s = run(fr, &params());
    assert!(s
        .nodes
        .iter()
        .any(|n| n.text == "Edge Cache Tier" && n.in_final));
    assert!(s.board_title.is_none(), "{:?}", s.board_title);
}

fn grid_spec() -> Spec {
    let cards = [
        "Alpha card",
        "Beta card",
        "Gamma card",
        "Delta card",
        "Epsilon card",
        "Zeta card",
        "Eta card",
        "Theta card",
        "Iota card",
    ];
    let mut s = base();
    s.stickies = cards
        .iter()
        .enumerate()
        .map(|(i, t)| {
            (
                *t,
                (
                    900.0 + (i % 3) as f64 * 130.0,
                    450.0 + (i / 3) as f64 * 110.0,
                ),
            )
        })
        .collect();
    s
}

#[test]
fn grid_stickies_carry_boxes_and_form_a_group_with_its_heading() {
    let specs: Vec<Spec> = (0..3).map(|_| grid_spec()).collect();
    let mut fr = frames(&specs);
    for f in &mut fr {
        f.board.other_visible_text.push(TextItem {
            text: "Things to show first".into(),
            bbox: BBox::new(900.0, 350.0, 1150.0, 380.0),
        });
    }
    let s = run(fr, &params());
    assert!(s
        .stickies
        .iter()
        .all(|x| x.bbox.is_some() && x.last_seen.is_some()));
    let g = s
        .groups
        .iter()
        .find(|g| g.sticky_ids.len() == 9)
        .unwrap_or_else(|| {
            panic!(
                "{:#?} {:?}",
                s.groups,
                s.stickies
                    .iter()
                    .map(|x| (&x.text, x.in_final))
                    .collect::<Vec<_>>()
            )
        });
    assert_eq!((g.rows, g.cols), (3, 3));
    assert_eq!(g.title.as_deref(), Some("Things to show first"));
    let first = s.stickies.iter().find(|x| x.id == g.sticky_ids[0]).unwrap();
    assert_eq!(first.text, "Alpha card");
    assert!(!s.stickies.iter().any(|x| x.text == "Things to show first"));
    assert!(!s.nodes.iter().any(|x| x.text == "Things to show first"));
}

#[test]
fn a_later_view_adds_only_its_new_cards_as_a_group() {
    // Frames 0-2 read the 3x3 grid. Frames 3-5 read the top two rows slightly off
    // their columns (detached from the bottom row) and a new row of four cards under
    // the bottom row. The late seven-card view shares three cards with the grid:
    // a majority-overlap rule keeps it whole (duplicating those three cards); the
    // new cards alone must form the second group.
    let new_row = ["Lambda card", "Mu card", "Nu card", "Xi card"];
    let specs: Vec<Spec> = (0..6)
        .map(|f| {
            let mut s = grid_spec();
            if f >= 3 {
                for c in s.stickies.iter_mut().take(6) {
                    c.1 .0 += 50.0;
                }
                for (i, t) in new_row.iter().enumerate() {
                    s.stickies.push((t, (900.0 + i as f64 * 130.0, 780.0)));
                }
            }
            s
        })
        .collect();
    let s = run(frames(&specs), &params());
    let sizes: Vec<usize> = s.groups.iter().map(|g| g.sticky_ids.len()).collect();
    assert_eq!(sizes, vec![9, 4], "{:#?}", s.groups);
    let text_of = |id: &String| {
        s.stickies
            .iter()
            .find(|x| &x.id == id)
            .map(|x| x.text.clone())
            .unwrap_or_default()
    };
    let second: Vec<String> = s.groups[1].sticky_ids.iter().map(text_of).collect();
    assert_eq!(second, new_row.map(String::from).to_vec());
}

/// Base board with a "gRPC" box at `at` in every keyframe; the Queue to Ledger
/// Store link is labeled "gRPC" in the keyframes `labeled` says.
fn label_box_frames(at: (f64, f64), n: usize, labeled: impl Fn(usize) -> bool) -> Vec<BoardFrame> {
    let specs: Vec<Spec> = (0..n)
        .map(|f| {
            let mut s = base();
            s.nodes.push(("n5", "gRPC".to_string(), at));
            if !labeled(f) {
                s.edges[1].2 = "";
            }
            s
        })
        .collect();
    frames(&specs)
}

fn lifted_as_label(s: &BoardStateItem, text: &str) -> bool {
    s.folded.iter().any(|f| {
        f.text == text && f.reason == FoldReason::EdgeLabel && f.into.starts_with("label of ")
    })
}

#[test]
fn an_edge_label_read_as_a_box_on_its_link_is_not_a_node() {
    let s = run(label_box_frames((950.0, 215.0), 4, |_| true), &params());
    assert!(!s.nodes.iter().any(|n| n.text == "gRPC"), "{:?}", s.nodes);
    assert!(lifted_as_label(&s, "gRPC"), "{:?}", s.folded);
    assert!(!s.events.iter().any(|e| e.detail == "gRPC"));
    assert_eq!(s.edges.len(), 3);
}

#[test]
fn an_edge_label_read_as_a_box_at_its_label_box_is_not_a_node() {
    // Off the straight link (a curved connector), but touching the label box.
    let mut fr = label_box_frames((950.0, 380.0), 4, |_| true);
    for f in &mut fr {
        for e in &mut f.board.edges {
            if e.label == "gRPC" {
                e.label_bbox = Some(BBox::new(900.0, 425.0, 1000.0, 445.0));
            }
        }
    }
    let s = run(fr, &params());
    assert!(!s.nodes.iter().any(|n| n.text == "gRPC"), "{:?}", s.nodes);
    assert!(lifted_as_label(&s, "gRPC"), "{:?}", s.folded);
}

#[test]
fn an_isolated_caption_named_like_a_far_edge_label_stays_a_node() {
    // Nothing touches the caption and nobody owns it; the same-label edge runs
    // across the other side of the board.
    let s = run(label_box_frames((250.0, 800.0), 4, |_| true), &params());
    assert!(
        s.nodes.iter().any(|n| n.text == "gRPC" && n.in_final),
        "{:?} {:?}",
        s.nodes,
        s.folded
    );
    assert!(!lifted_as_label(&s, "gRPC"));
}

#[test]
fn a_box_that_rarely_coincides_with_the_label_stays_a_node() {
    // On the link, but the label was read in only one of six keyframes.
    let s = run(label_box_frames((950.0, 215.0), 6, |f| f == 0), &params());
    assert!(
        s.nodes.iter().any(|n| n.text == "gRPC" && n.in_final),
        "{:?} {:?}",
        s.nodes,
        s.folded
    );
}

#[test]
fn an_edge_label_read_as_a_sticky_on_its_link_is_not_a_sticky() {
    let specs: Vec<Spec> = (0..4)
        .map(|_| {
            let mut s = base();
            s.stickies.push(("gRPC", (950.0, 215.0)));
            s
        })
        .collect();
    let s = run(frames(&specs), &params());
    assert!(
        !s.stickies.iter().any(|x| x.text == "gRPC"),
        "{:?}",
        s.stickies
    );
    assert!(lifted_as_label(&s, "gRPC"), "{:?}", s.folded);
}

#[test]
fn a_box_named_like_a_label_that_ends_an_edge_stays_a_node() {
    let specs: Vec<Spec> = (0..4)
        .map(|_| {
            let mut s = base();
            s.nodes.push(("n5", "HTTP".to_string(), (250.0, 800.0)));
            s.edges.push(("n5", "n4", ""));
            s
        })
        .collect();
    let s = run(frames(&specs), &params());
    assert!(
        s.nodes.iter().any(|n| n.text == "HTTP" && n.in_final),
        "{:?}",
        s.nodes
    );
}

#[test]
fn fragments_fold_into_colocated_longer_stickies_only() {
    let mut specs: Vec<Spec> = (0..5).map(|_| base()).collect();
    for (i, s) in specs.iter_mut().enumerate() {
        s.stickies.push(("Shared updates", (300.0, 780.0)));
        s.stickies.push(("Tasks - due this week", (1300.0, 420.0)));
        s.stickies.push(("Tasks", (450.0, 400.0)));
        if i == 1 || i == 3 {
            // A cut-off reading of the same card.
            s.stickies.push(("updates", (320.0, 785.0)));
        }
    }
    let s = run(frames(&specs), &params());
    let texts: Vec<&str> = s.stickies.iter().map(|x| x.text.as_str()).collect();
    assert!(!texts.contains(&"updates"), "{texts:?}");
    assert!(
        texts.contains(&"Shared updates")
            && texts.contains(&"Tasks")
            && texts.contains(&"Tasks - due this week")
    );
    assert!(s
        .folded
        .iter()
        .any(|f| f.text == "updates" && f.into == "Shared updates"));
}

#[test]
fn a_tag_moved_onto_a_connector_beside_a_box_is_a_move_to_that_box() {
    let mut specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    for (i, s) in specs.iter_mut().enumerate() {
        // Next to Ingest Gateway, then on the Queue -> Ledger Store connector right
        // beside Ledger Store.
        let at = if i < 4 {
            (200.0, 120.0)
        } else {
            (1060.0, 210.0)
        };
        s.owners.push(("Avery", at, ""));
    }
    let s = run(frames(&specs), &params());
    let a: Vec<_> = s
        .owner_assignments
        .iter()
        .filter(|x| x.person_id == "p-avery")
        .collect();
    assert_eq!(a.len(), 2, "{a:#?}");
    assert_eq!(a[1].target.texts(), vec!["Ledger Store"]);
    assert_eq!(a[1].valid_from_s, 80.0);
    assert!(a[1]
        .moved_from
        .as_ref()
        .is_some_and(|m| m.texts() == vec!["Ingest Gateway"]));
    assert!(s
        .events
        .iter()
        .any(|e| e.kind == EventKind::OwnerMoved && e.keyframe_id == "kf04"));
}

#[test]
fn one_tag_bridging_two_boxes_owns_both_nodes_not_their_link() {
    let mut specs: Vec<Spec> = (0..4).map(|_| base()).collect();
    for s in &mut specs {
        s.nodes.push(("n6", "Left Pane".into(), (300.0, 450.0)));
        s.nodes.push(("n7", "Right Pane".into(), (520.0, 450.0)));
        s.edges.push(("n6", "n7", ""));
        // Above the gap between the two boxes, touching neither the connector nor
        // the gap.
        s.owners.push(("Jordy", (410.0, 375.0), ""));
    }
    let s = run(frames(&specs), &params());
    let mut t: Vec<String> = s
        .owner_assignments
        .iter()
        .filter(|x| x.person_id == "p-jordan")
        .flat_map(|x| {
            x.target
                .texts()
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        })
        .collect();
    t.sort();
    assert_eq!(
        t,
        vec!["Left Pane", "Right Pane"],
        "{:#?}",
        s.owner_assignments
    );
    assert!(s
        .owner_assignments
        .iter()
        .all(|x| !matches!(x.target, OwnerTarget::Edge { .. })));
}

#[test]
fn glyph_and_style_labels_are_stripped() {
    let mut specs: Vec<Spec> = (0..3).map(|_| base()).collect();
    for s in &mut specs {
        s.edges = vec![
            ("n1", "n2", "\u{2192}"),
            ("n2", "n3", "solid"),
            ("n2", "n4", "gRPC \u{2192}"),
        ];
    }
    let s = run(frames(&specs), &params());
    let mut labels: Vec<&str> = s.edges.iter().map(|e| e.label.as_str()).collect();
    labels.sort();
    assert_eq!(labels, vec!["", "", "gRPC"]);
}

#[test]
fn a_node_read_with_an_imprecise_box_after_a_pan_stays_one_node() {
    // Keyframes 2 and 3 are panned; in them the reader's Queue box is 120 px off.
    let pan = Similarity {
        scale: 1.0,
        angle: 0.0,
        tx: -100.0,
        ty: 40.0,
    };
    let mut specs: Vec<Spec> = (0..4).map(|_| base()).collect();
    specs[2].t = pan;
    specs[3].t = pan;
    let mut fr = frames(&specs);
    for f in fr.iter_mut().skip(2) {
        for n in &mut f.board.nodes {
            if n.text == "Queue" {
                n.bbox = BBox::new(n.bbox.x1 - 120.0, n.bbox.y1, n.bbox.x2 - 120.0, n.bbox.y2);
            }
        }
    }
    let s = run(fr, &params());
    assert_eq!(
        s.nodes.iter().filter(|n| n.text == "Queue").count(),
        1,
        "{:#?}",
        s.nodes
            .iter()
            .map(|n| (&n.text, n.lifetimes.len()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn same_text_in_two_places_never_seen_together_stays_two_nodes() {
    // "Client" on the left in keyframes 0-2 and a second "Client" on the right in
    // keyframes 3-5, same view throughout: two elements.
    let mut specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    for (i, s) in specs.iter_mut().enumerate() {
        let x = if i < 3 { 250.0 } else { 1300.0 };
        s.nodes.push(("n8", "Client".into(), (x, 420.0)));
    }
    let s = run(frames(&specs), &params());
    assert_eq!(s.nodes.iter().filter(|n| n.text == "Client").count(), 2);
}

#[test]
fn direction_votes_use_every_sighting_of_the_edge() {
    // The edge is seen in keyframes 0-2 (head at Ingest Gateway), erased while both
    // boxes stay in view, then seen again without a decisive head.
    let mut specs: Vec<Spec> = (0..9).map(|_| base()).collect();
    for s in specs.iter_mut().take(7).skip(3) {
        s.edges.retain(|e| e.0 != "n1");
    }
    let mut fr = frames(&specs);
    for (i, f) in fr.iter_mut().enumerate() {
        let v = match i {
            0..=2 => Some(EndVerdict::Reverse),
            7 | 8 => Some(EndVerdict::NoArrowhead),
            _ => None,
        };
        if let Some(v) = v {
            f.directions = Some(dir_item(
                &f.keyframe_id,
                vec![evidence("n1", "n2", v, None)],
            ));
        }
    }
    let s = run(fr, &params());
    let e = s.edges.iter().find(|e| e.label == "HTTP").expect("edge");
    assert_eq!(e.direction, EdgeOrientation::Forward);
    let src = s.nodes.iter().find(|n| n.id == e.src).unwrap();
    assert_eq!(src.text, "Queue");
}

#[test]
fn a_node_well_above_a_card_row_is_not_its_heading() {
    let mut specs: Vec<Spec> = (0..3).map(|_| base()).collect();
    for s in &mut specs {
        s.stickies = vec![
            ("Plan the rollout", (700.0, 845.0)),
            ("Order the badges", (830.0, 845.0)),
            ("Book the venue", (960.0, 845.0)),
        ];
    }
    // Report Builder sits more than a card height above the row: a component, not a
    // heading.
    let s = run(frames(&specs), &params());
    assert!(s.nodes.iter().any(|n| n.text == "Report Builder"));
    let g = s
        .groups
        .iter()
        .find(|g| g.sticky_ids.len() == 3)
        .expect("row");
    assert_eq!((g.rows, g.cols), (1, 3));
    assert!(g.title.is_none());
}

#[test]
fn late_reversal_sets_the_final_direction() {
    // Head at Queue for six keyframes, then redrawn toward Ingest Gateway.
    let mut fr = frames(&(0..9).map(|_| base()).collect::<Vec<_>>());
    for (i, f) in fr.iter_mut().enumerate() {
        let v = if i < 6 {
            EndVerdict::Forward
        } else {
            EndVerdict::Reverse
        };
        f.directions = Some(dir_item(
            &f.keyframe_id,
            vec![evidence("n1", "n2", v, None)],
        ));
    }
    let s = run(fr, &params());
    let e = s.edges.iter().find(|e| e.label == "HTTP").expect("edge");
    let src = s.nodes.iter().find(|n| n.id == e.src).unwrap();
    assert_eq!(src.text, "Queue");
    assert!(s
        .events
        .iter()
        .chain(s.suppressed_events.iter().map(|x| &x.event))
        .any(|ev| ev.kind == EventKind::EdgeReversed && ev.keyframe_id == "kf06"));
}

#[test]
fn fragments_need_positive_geometric_evidence() {
    // "Cache" on the left early; "Cache warmer" drawn elsewhere later. Never seen
    // together, placed far apart: two elements.
    let mut specs: Vec<Spec> = (0..8).map(|_| base()).collect();
    for (i, s) in specs.iter_mut().enumerate() {
        if i < 3 {
            s.nodes.push(("n8", "Cache".into(), (250.0, 420.0)));
        } else {
            s.nodes.push(("n9", "Cache warmer".into(), (1300.0, 420.0)));
        }
    }
    let s = run(frames(&specs), &params());
    assert!(
        s.nodes.iter().any(|n| n.text == "Cache"),
        "{:?} {:?}",
        s.nodes
            .iter()
            .map(|n| (&n.text, n.in_final))
            .collect::<Vec<_>>(),
        s.folded
    );
    assert!(s.folded.is_empty(), "{:?}", s.folded);
    // Without any geometry (text-only readings) there is nothing to compare: no fold.
    let mut fr = frames(&specs);
    let full = BBox::new(0.0, 0.0, W, H);
    for f in &mut fr {
        for n in &mut f.board.nodes {
            n.bbox = full;
        }
        for x in &mut f.board.stickies {
            x.bbox = full;
        }
        for x in &mut f.board.other_visible_text {
            x.bbox = full;
        }
    }
    let s = run(fr, &params());
    assert!(
        s.nodes.iter().any(|n| n.text == "Cache"),
        "{:?} {:?} {:?}",
        s.nodes
            .iter()
            .map(|n| (&n.text, &n.variants))
            .collect::<Vec<_>>(),
        s.folded,
        s.registration
            .iter()
            .map(|r| r.registration.mode)
            .collect::<Vec<_>>()
    );
    assert!(s.folded.iter().all(|x| x.text != "Cache"));
}

#[test]
fn edge_endpoints_are_never_group_headings() {
    // A row of cards right under Report Builder, which is an edge endpoint.
    let mut specs: Vec<Spec> = (0..3).map(|_| base()).collect();
    for s in &mut specs {
        s.stickies = vec![
            ("Plan the rollout", (560.0, 760.0)),
            ("Order the badges", (690.0, 760.0)),
            ("Book the venue", (820.0, 760.0)),
        ];
    }
    let s = run(frames(&specs), &params());
    assert!(s.nodes.iter().any(|n| n.text == "Report Builder"));
    assert!(s
        .edges
        .iter()
        .any(|e| e.a_text == "Report Builder" || e.b_text == "Report Builder"));
    let g = s
        .groups
        .iter()
        .find(|g| g.sticky_ids.len() == 3)
        .expect("row");
    assert!(g.title.is_none());
}

#[test]
fn text_well_above_a_card_row_is_not_its_heading() {
    let mut specs: Vec<Spec> = (0..3).map(|_| base()).collect();
    for s in &mut specs {
        s.stickies = vec![
            ("Plan the rollout", (560.0, 800.0)),
            ("Order the badges", (690.0, 800.0)),
            ("Book the venue", (820.0, 800.0)),
        ];
    }
    let mut fr = frames(&specs);
    for f in &mut fr {
        // 250 px (2.5 card heights) above the row.
        f.board.other_visible_text.push(TextItem {
            text: "Venue notes".into(),
            bbox: BBox::new(560.0, 470.0, 800.0, 500.0),
        });
    }
    let s = run(fr, &params());
    let g = s
        .groups
        .iter()
        .find(|g| g.sticky_ids.len() == 3)
        .expect("row");
    assert!(g.title.is_none());
}

#[test]
fn a_tag_between_two_unconnected_boxes_is_not_a_bridge() {
    let mut specs: Vec<Spec> = (0..4).map(|_| base()).collect();
    for s in &mut specs {
        s.nodes.push(("n6", "Left Pane".into(), (300.0, 450.0)));
        s.nodes.push(("n7", "Right Pane".into(), (520.0, 450.0)));
        s.owners.push(("Jordy", (410.0, 375.0), ""));
    }
    let s = run(frames(&specs), &params());
    assert!(s
        .owner_assignments
        .iter()
        .filter(|x| x.person_id == "p-jordan")
        .all(
            |x| x.target.texts().len() == 1 && x.target.texts()[0] != "Left Pane"
                || x.target.texts()[0] != "Right Pane"
        ));
    let owned: Vec<_> = s
        .owner_assignments
        .iter()
        .filter(|x| x.person_id == "p-jordan")
        .collect();
    assert!(owned.len() < 2, "{owned:#?}");
}

#[test]
fn board_state_1_0_0_items_still_parse() {
    let s = run(
        frames(&(0..3).map(|_| base()).collect::<Vec<_>>()),
        &params(),
    );
    let mut v = serde_json::to_value(&s).unwrap();
    // A 1.0.0 item has none of the fields added in 1.1.0.
    for k in ["board_title", "groups", "folded"] {
        v.as_object_mut().unwrap().remove(k);
    }
    for st in v["stickies"].as_array_mut().unwrap() {
        st.as_object_mut().unwrap().remove("bbox");
        st.as_object_mut().unwrap().remove("last_seen");
    }
    let old: BoardStateItem = serde_json::from_value(v).unwrap();
    assert!(old.groups.is_empty() && old.board_title.is_none());
    assert_eq!(old.stickies.len(), s.stickies.len());
}

/// Final state is "observed and not later removed", not "seen in the last window".
#[test]
fn elements_outside_a_zoomed_final_view_stay_final() {
    // Keyframes 0 to 5 show the whole board; 6 to 8 zoom onto its right part, so
    // "Ingest Gateway" and its HTTP edge are never in the final window's view.
    let zoom = Similarity {
        scale: 1.3,
        angle: 0.0,
        tx: -600.0,
        ty: 0.0,
    };
    let specs: Vec<Spec> = (0..9)
        .map(|i| Spec {
            t: if i >= 6 { zoom } else { Similarity::IDENTITY },
            ink: Some(0.01),
            ..base()
        })
        .collect();
    let s = run(frames(&specs), &params());
    let w = s.final_window.as_ref().unwrap();
    assert!(w.keyframe_ids.iter().all(|k| k.as_str() >= "kf06"), "{w:?}");
    assert!(
        s.registration
            .iter()
            .all(|r| r.registration.mode != RegistrationMode::TextOnly),
        "{:?}",
        s.registration
    );
    assert_eq!(
        node_texts(&s),
        vec!["Ingest Gateway", "Ledger Store", "Queue", "Report Builder"]
    );
    let http = s.edges.iter().find(|e| e.label == "HTTP").unwrap();
    assert!(http.in_final, "{http:?}");
    assert!(http.lifetimes.iter().all(|l| l.removed_at_s.is_none()));
}

#[test]
fn a_view_that_read_nothing_known_near_the_place_is_not_a_removal() {
    // Keyframes 6 to 8 keep the whole board in view but read only its left part:
    // "Ledger Store" and everything established near it are missing. Its place is
    // in the view, yet nothing confirms the reader looked there.
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    for s in specs.iter_mut().skip(6) {
        s.nodes.retain(|n| n.0 != "n3");
        s.edges.retain(|e| e.1 != "n3");
        s.stickies
            .retain(|x| x.0 != "Is the queue durable?" && x.0 != "Beta milestone in March");
    }
    let mut fr = frames(&specs);
    for f in fr.iter_mut().skip(6) {
        f.board.other_visible_text.clear();
    }
    let s = run(fr, &params());
    let ledger = s.nodes.iter().find(|n| n.text == "Ledger Store").unwrap();
    assert!(ledger.in_final, "{ledger:?}");
    assert!(ledger.lifetimes.iter().all(|l| l.removed_at_s.is_none()));
    let grpc = s.edges.iter().find(|e| e.label == "gRPC").unwrap();
    assert!(grpc.in_final, "{grpc:?}");
}

#[test]
fn ocr_text_at_the_place_means_the_reader_missed_it() {
    // The reader drops "Report Builder" from keyframes 6 to 8, but OCR still reads
    // its words at its place: that is a reading omission, not an erasure.
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    for s in specs.iter_mut().skip(6) {
        s.nodes.retain(|n| n.0 != "n4");
        s.edges.retain(|e| e.1 != "n4");
    }
    let without_ocr = run(frames(&specs), &params());
    let rb = without_ocr
        .nodes
        .iter()
        .find(|n| n.text == "Report Builder")
        .unwrap();
    assert!(
        !rb.in_final,
        "absent while its neighbors are read: removed ({rb:?})"
    );

    let mut fr = frames(&specs);
    for f in fr.iter_mut().skip(6) {
        f.ocr_anchors = vec![
            TextAnchor {
                text: "Report".into(),
                bbox: BBox::new(640.0, 635.0, 700.0, 655.0),
            },
            TextAnchor {
                text: "Builder".into(),
                bbox: BBox::new(705.0, 635.0, 765.0, 655.0),
            },
        ];
    }
    let s = run(fr, &params());
    let rb = s.nodes.iter().find(|n| n.text == "Report Builder").unwrap();
    assert!(rb.in_final, "{rb:?}");
    assert!(rb.lifetimes.iter().all(|l| l.removed_at_s.is_none()));
}

#[test]
fn an_edge_left_out_of_readings_stays_until_the_ink_changes() {
    // Keyframes 5 to 8 read both ends of the gRPC edge but not the edge itself.
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    for s in specs.iter_mut().skip(5) {
        s.edges.retain(|e| e.2 != "gRPC");
    }
    let grpc = |s: &BoardStateItem| s.edges.iter().find(|e| e.label == "gRPC").cloned().unwrap();
    // No ink change: the reader left it out.
    let s = run(frames(&specs), &params());
    let e = grpc(&s);
    assert!(e.in_final, "{e:?}");
    assert!(e.lifetimes.iter().all(|l| l.removed_at_s.is_none()));
    // The connector is erased at keyframe 5: the ink changes there.
    specs[5].ink = Some(0.2);
    let s = run(frames(&specs), &params());
    let e = grpc(&s);
    assert!(!e.in_final, "{e:?}");
    assert_eq!(e.lifetimes.last().unwrap().removed_at_s, Some(100.0));
}

#[test]
fn one_odd_reading_of_a_known_box_is_not_a_second_final_box() {
    // The last keyframe reads "Queue" far from its place (a confident, traced
    // single sighting) and not at its place: one Queue on the final board.
    let mut specs: Vec<Spec> = (0..5).map(|_| base()).collect();
    specs[4].nodes.retain(|n| n.0 != "n2");
    specs[4].edges.retain(|e| e.0 != "n2" && e.1 != "n2");
    specs[4].nodes.push(("n9", "Queue".into(), (1400.0, 450.0)));
    specs[4].edges.push(("n3", "n9", ""));
    let mut fr = frames(&specs);
    fr[4].directions = Some(dir_item(
        "kf04",
        vec![evidence("n3", "n9", EndVerdict::Forward, None)],
    ));
    let s = run(fr, &params());
    let queues: Vec<_> = s
        .nodes
        .iter()
        .filter(|n| n.text == "Queue" && n.in_final)
        .collect();
    assert_eq!(queues.len(), 1, "{:#?}", s.nodes);
    assert!(queues[0].lifetimes[0].keyframes >= 2);
    // The odd single sighting itself is not final.
    assert!(!s
        .nodes
        .iter()
        .any(|n| n.text == "Queue" && n.lifetimes[0].keyframes == 1 && n.in_final));
}

/// Scripted canvas pixels: per keyframe index, the line cover of every corridor and
/// the ink share of every box.
struct Pixels {
    line: Vec<f64>,
    ink: Vec<f64>,
}

impl Pixels {
    fn at(v: &[f64], keyframe_id: &str) -> Option<f64> {
        let i: usize = keyframe_id.trim_start_matches("kf").parse().ok()?;
        v.get(i).copied()
    }
}

impl RegionProbe for Pixels {
    fn ink_share(&self, keyframe_id: &str, _region: &BBox) -> Option<f64> {
        Self::at(&self.ink, keyframe_id)
    }
    fn line_cover(
        &self,
        keyframe_id: &str,
        _a: (f64, f64),
        _b: (f64, f64),
        _half_width: f64,
    ) -> Option<f64> {
        Self::at(&self.line, keyframe_id)
    }
}

fn run_probed(frames: Vec<BoardFrame>, p: &ConsolidationParams, px: &Pixels) -> BoardStateItem {
    consolidate_with_probe(
        frames,
        "board-1",
        p,
        &hooks(&NoCorroboration),
        Some(px as &dyn RegionProbe),
    )
}

#[test]
fn unrelated_ink_does_not_remove_a_connector_still_drawn() {
    // A sticky is added in a corner at keyframe 5 (ink 0.06, over the event
    // threshold), and keyframes 5 to 8 read both ends of the gRPC edge but not the
    // edge. Its corridor still holds the line: the reader left it out.
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    specs[5].ink = Some(0.06);
    for s in specs.iter_mut().skip(5) {
        s.edges.retain(|e| e.2 != "gRPC");
        s.stickies.push(("Corner note", (1500.0, 850.0)));
    }
    let grpc = |s: &BoardStateItem| s.edges.iter().find(|e| e.label == "gRPC").cloned().unwrap();
    let px = Pixels {
        line: vec![0.95; 9],
        ink: vec![0.5; 9],
    };
    let e = grpc(&run_probed(frames(&specs), &params(), &px));
    assert!(e.in_final, "{e:?}");
    assert!(e.lifetimes.iter().all(|l| l.removed_at_s.is_none()));
    // Without pixels, the whole-board ink is all there is to go on.
    let e = grpc(&run(frames(&specs), &params()));
    assert!(!e.in_final, "{e:?}");
}

#[test]
fn an_erased_connector_is_removed_without_a_board_ink_event() {
    // A thin connector on a large canvas: its erasure at keyframe 5 stays under the
    // board's ink threshold (or the ink is unknown across a classify switch), but
    // its corridor emptied.
    for ink5 in [Some(0.01), None] {
        let mut specs: Vec<Spec> = (0..9)
            .map(|_| Spec {
                ink: Some(0.01),
                ..base()
            })
            .collect();
        specs[5].ink = ink5;
        for s in specs.iter_mut().skip(5) {
            s.edges.retain(|e| e.2 != "gRPC");
        }
        let mut line = vec![0.9; 9];
        for v in line.iter_mut().skip(5) {
            *v = 0.05;
        }
        let px = Pixels {
            line,
            ink: vec![0.5; 9],
        };
        let s = run_probed(frames(&specs), &params(), &px);
        let e = s.edges.iter().find(|e| e.label == "gRPC").unwrap();
        assert!(!e.in_final, "{e:?}");
        assert_eq!(e.lifetimes.last().unwrap().removed_at_s, Some(100.0));
    }
}

#[test]
fn a_routed_connector_the_corridor_misses_falls_back_to_board_ink() {
    // The corridor never held a straight line (a routed connector): the pixels
    // cannot tell, and without a board ink change the edge stays.
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    for s in specs.iter_mut().skip(5) {
        s.edges.retain(|e| e.2 != "gRPC");
    }
    let px = Pixels {
        line: vec![0.1; 9],
        ink: vec![0.5; 9],
    };
    let s = run_probed(frames(&specs), &params(), &px);
    assert!(s.edges.iter().find(|e| e.label == "gRPC").unwrap().in_final);
}

fn sparse() -> Spec {
    // One sticky far from every other element (no established neighbor within the
    // coverage radius of its place).
    Spec {
        stickies: vec![("Beta milestone in March", (1400.0, 780.0))],
        ..base()
    }
}

#[test]
fn an_erased_sticky_on_a_sparse_board_is_removed_by_its_pixels() {
    let mut specs: Vec<Spec> = (0..8).map(|_| sparse()).collect();
    for s in specs.iter_mut().skip(4) {
        s.stickies.clear();
    }
    let beta = |s: &BoardStateItem| {
        s.stickies
            .iter()
            .find(|x| x.text == "Beta milestone in March")
            .cloned()
            .unwrap()
    };
    // No pixels: nothing near the place confirms the view, so it is never removed.
    assert!(beta(&run(frames(&specs), &params())).in_final);
    // The sticky's box emptied from keyframe 4 on.
    let mut ink = vec![0.6; 8];
    for v in ink.iter_mut().skip(4) {
        *v = 0.01;
    }
    let px = Pixels {
        line: vec![0.9; 8],
        ink,
    };
    let b = beta(&run_probed(frames(&specs), &params(), &px));
    assert!(!b.in_final, "{b:?}");
    assert_eq!(b.lifetimes.last().unwrap().removed_at_s, Some(80.0));
    // The box still holds the card: the reader left it out, it stays.
    let px = Pixels {
        line: vec![0.9; 8],
        ink: vec![0.6; 8],
    };
    assert!(beta(&run_probed(frames(&specs), &params(), &px)).in_final);
}

#[test]
fn a_second_box_that_ocr_reads_at_its_own_place_is_not_an_echo() {
    // As in the echo case, but OCR reads "Queue" at the new place: a second Queue
    // box is really there.
    let mut specs: Vec<Spec> = (0..5).map(|_| base()).collect();
    specs[4].nodes.retain(|n| n.0 != "n2");
    specs[4].edges.retain(|e| e.0 != "n2" && e.1 != "n2");
    specs[4].nodes.push(("n9", "Queue".into(), (1400.0, 450.0)));
    specs[4].edges.push(("n3", "n9", ""));
    let mut fr = frames(&specs);
    fr[4].directions = Some(dir_item(
        "kf04",
        vec![evidence("n3", "n9", EndVerdict::Forward, None)],
    ));
    fr[4].ocr_anchors.push(TextAnchor {
        text: "Queue".into(),
        bbox: BBox::new(1370.0, 440.0, 1430.0, 460.0),
    });
    let s = run(fr, &params());
    let queues = s
        .nodes
        .iter()
        .filter(|n| n.text == "Queue" && n.in_final)
        .count();
    assert_eq!(queues, 2, "{:#?}", s.nodes);
}

#[test]
fn the_final_window_stays_inside_its_contiguous_run() {
    // Five keyframes, a classify switch, then two more: the window holds the two
    // after the switch, not a keyframe from before it.
    let specs: Vec<Spec> = (0..7).map(|_| base()).collect();
    let mut fr = frames(&specs);
    for f in fr.iter_mut().skip(5) {
        f.keyframe_index += 3;
    }
    let p = ConsolidationParams {
        final_window_s: 1.0,
        ..params()
    };
    let w = run(fr, &p).final_window.expect("window");
    assert_eq!(w.keyframe_ids, vec!["kf05", "kf06"]);
    assert_eq!(w.start_s, 100.0);
}

#[test]
fn a_routed_connector_is_not_removed_by_ink_added_elsewhere() {
    // The corridor never held a straight line (a routed connector) and a sticky is
    // added elsewhere at keyframe 5 (board ink 0.2). The ink around the edge's
    // ends did not fall: the reader left the edge out.
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    specs[5].ink = Some(0.2);
    for s in specs.iter_mut().skip(5) {
        s.edges.retain(|e| e.2 != "gRPC");
        s.stickies.push(("Corner note", (1500.0, 850.0)));
    }
    let grpc = |s: &BoardStateItem| s.edges.iter().find(|e| e.label == "gRPC").cloned().unwrap();
    let px = Pixels {
        line: vec![0.1; 9],
        ink: vec![0.3; 9],
    };
    let e = grpc(&run_probed(frames(&specs), &params(), &px));
    assert!(e.in_final, "{e:?}");
    // The ink around the ends falling is not enough by itself: without a traced
    // stroke (these scripted pixels trace none) the pixels cannot tell, and the
    // edge stays. The traced cases are in
    // `a_routed_connector_is_traced_around_the_elements_between_its_ends`.
    let mut ink = vec![0.3; 9];
    for v in ink.iter_mut().skip(5) {
        *v = 0.25;
    }
    let px = Pixels {
        line: vec![0.1; 9],
        ink,
    };
    let e = grpc(&run_probed(frames(&specs), &params(), &px));
    assert!(e.in_final, "{e:?}");
    // Real pixels: the routed connector drawn until keyframe 5 and erased there.
    let canvases = paint_all(&specs, |i| if i < 5 { vec![routed_grpc()] } else { vec![] });
    let e = grpc(&run_canvases(frames(&specs), &canvases));
    assert!(!e.in_final, "{e:?}");
    assert_eq!(e.lifetimes.last().unwrap().removed_at_s, Some(100.0));
}

#[test]
fn erasing_the_only_element_is_a_removal() {
    // A board holding one sticky; from keyframe 4 the canvas is empty, so the
    // later views read nothing and cannot be registered.
    let specs: Vec<Spec> = (0..8)
        .map(|i| Spec {
            nodes: vec![],
            edges: vec![],
            stickies: if i < 4 {
                vec![("Beta milestone in March", (800.0, 450.0))]
            } else {
                vec![]
            },
            ..base()
        })
        .collect();
    let mut fr = frames(&specs);
    for f in &mut fr {
        f.board.other_visible_text.clear();
    }
    let only = |s: &BoardStateItem| s.stickies.first().cloned().unwrap();
    assert!(only(&run(fr.clone(), &params())).in_final, "control");
    let mut ink = vec![0.6; 8];
    for v in ink.iter_mut().skip(4) {
        *v = 0.0;
    }
    let px = Pixels {
        line: vec![0.0; 8],
        ink,
    };
    let s = only(&run_probed(fr.clone(), &params(), &px));
    assert!(!s.in_final, "{s:?}");
    assert_eq!(s.lifetimes.last().unwrap().removed_at_s, Some(80.0));
    // OCR still reads the card's text on the "blank" canvas: not a removal.
    for f in fr.iter_mut().skip(4) {
        f.ocr_anchors.push(TextAnchor {
            text: "Beta milestone".into(),
            bbox: BBox::new(760.0, 440.0, 840.0, 460.0),
        });
    }
    assert!(only(&run_probed(fr, &params(), &px)).in_final);
}

#[test]
fn a_blank_view_that_aligned_ink_does_not_link_is_not_an_erasure() {
    // The same empty canvas from keyframe 4, but reached by a pan or cut the
    // aligner could not follow (ink unknown), or with no ink change at all:
    // nothing proves it is the same view, so the card stays.
    for ink in [None, Some(0.0)] {
        let specs: Vec<Spec> = (0..8)
            .map(|i| Spec {
                nodes: vec![],
                edges: vec![],
                stickies: if i < 4 {
                    vec![("Beta milestone in March", (800.0, 450.0))]
                } else {
                    vec![]
                },
                ink: if i < 4 { Some(0.2) } else { ink },
                ..base()
            })
            .collect();
        let mut fr = frames(&specs);
        for f in &mut fr {
            f.board.other_visible_text.clear();
        }
        let mut shares = vec![0.6; 8];
        for v in shares.iter_mut().skip(4) {
            *v = 0.0;
        }
        let px = Pixels {
            line: vec![0.0; 8],
            ink: shares,
        };
        let s = run_probed(fr, &params(), &px);
        assert!(s.stickies[0].in_final, "{ink:?}: {:?}", s.stickies[0]);
    }
}

#[test]
fn erasing_another_card_between_a_routed_connector_s_ends_does_not_remove_it() {
    // A card sits between Queue and Ledger Store and is erased at keyframe 5 (the
    // ink around the ends falls, board ink 0.2) while the reader leaves the
    // routed connector out: the drop belongs to the card.
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    for s in specs.iter_mut().take(5) {
        s.stickies.push(("Temp note", (950.0, 210.0)));
    }
    specs[5].ink = Some(0.2);
    for s in specs.iter_mut().skip(5) {
        s.edges.retain(|e| e.2 != "gRPC");
    }
    let mut ink = vec![0.3; 9];
    for v in ink.iter_mut().skip(5) {
        *v = 0.25;
    }
    let px = Pixels {
        line: vec![0.1; 9],
        ink,
    };
    let s = run_probed(frames(&specs), &params(), &px);
    let e = s.edges.iter().find(|e| e.label == "gRPC").unwrap();
    assert!(e.in_final, "{e:?}");
}

/// Pixels of one straight connector between two fixed points: a corridor query
/// covers it only when its ends are within the corridor's half width of the
/// connector's (a real corridor covers a line anywhere inside it).
struct OneLine {
    a: (f64, f64),
    b: (f64, f64),
}

impl RegionProbe for OneLine {
    fn ink_share(&self, _keyframe_id: &str, _region: &BBox) -> Option<f64> {
        Some(0.3)
    }
    fn line_cover(
        &self,
        _keyframe_id: &str,
        a: (f64, f64),
        b: (f64, f64),
        half_width: f64,
    ) -> Option<f64> {
        let near = |p: (f64, f64), q: (f64, f64)| (p.0 - q.0).hypot(p.1 - q.1) <= half_width;
        Some(if near(a, self.a) && near(b, self.b) {
            0.95
        } else {
            0.0
        })
    }
}

#[test]
fn reader_box_jitter_does_not_erase_a_drawn_connector() {
    // From keyframe 5 the reader places Queue and Ledger Store 30 px lower and
    // leaves their connector out, while the board ink changes elsewhere. The
    // corridor is measured where the edge was read, mapped into each keyframe, so
    // it stays on the drawn line.
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    specs[5].ink = Some(0.2);
    for s in specs.iter_mut().skip(5) {
        s.edges.retain(|e| e.2 != "gRPC");
        for n in &mut s.nodes {
            if n.0 == "n2" || n.0 == "n3" {
                n.2 .1 += 30.0;
            }
        }
    }
    let t = Similarity::IDENTITY;
    let (q, l) = (
        bbox_at(&t, (700.0, 220.0), 90.0, 40.0),
        bbox_at(&t, (1200.0, 200.0), 90.0, 40.0),
    );
    let c = |b: &BBox| ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0);
    let line = OneLine {
        a: glassrip_meeting::pixel_direction::exit_point(&q, c(&l)),
        b: glassrip_meeting::pixel_direction::exit_point(&l, c(&q)),
    };
    let s = consolidate_with_probe(
        frames(&specs),
        "board-1",
        &params(),
        &hooks(&NoCorroboration),
        Some(&line as &dyn RegionProbe),
    );
    let e = s.edges.iter().find(|e| e.label == "gRPC").unwrap();
    assert!(e.in_final, "{e:?}");
    // The jittered boxes alone put a corridor 30 px off the line (more than its
    // 20 px half width): measuring there would have read the line as gone.
    let moved = bbox_at(&t, (700.0, 250.0), 90.0, 40.0);
    let off = line.line_cover("kf05", c(&moved), c(&l), 20.0).unwrap();
    assert_eq!(off, 0.0);
}

// ---------------------------------------------------------------- real pixels

use glassrip_meeting::consolidate::StrokeTrace;
use glassrip_meeting::stages::ProbeCanvas;
use image::{Rgb, RgbImage};

/// Something drawn on a synthetic canvas, in reading coordinates.
#[derive(Clone)]
enum Mark {
    /// A box outline (2 px).
    Outline(BBox),
    /// A filled box.
    Fill(BBox),
    /// A polyline (2 px).
    Path(Vec<(f64, f64)>),
}

const BACKGROUND: Rgb<u8> = Rgb([250, 250, 247]);
const INK: Rgb<u8> = Rgb([40, 44, 52]);

fn dot(img: &mut RgbImage, x: f64, y: f64) {
    for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
        let (px, py) = (x + dx, y + dy);
        if px >= 0.0 && py >= 0.0 && px < f64::from(img.width()) && py < f64::from(img.height()) {
            img.put_pixel(px as u32, py as u32, INK);
        }
    }
}

fn segment(img: &mut RgbImage, p: (f64, f64), q: (f64, f64)) {
    let n = ((q.0 - p.0).hypot(q.1 - p.1) * 2.0).ceil().max(1.0) as usize;
    for i in 0..=n {
        let t = i as f64 / n as f64;
        dot(img, p.0 + (q.0 - p.0) * t, p.1 + (q.1 - p.1) * t);
    }
}

/// The spec's boxes (outlines) and every read edge but gRPC (straight lines between
/// the boxes' exit points), plus `extra`.
fn paint(spec: &Spec, extra: &[Mark]) -> ProbeCanvas {
    let mut img = RgbImage::from_pixel(W as u32, H as u32, BACKGROUND);
    let b = board(spec);
    let mut marks: Vec<Mark> = b.nodes.iter().map(|n| Mark::Outline(n.bbox)).collect();
    marks.extend(b.stickies.iter().map(|x| Mark::Outline(x.bbox)));
    let bx = |id: &str| b.nodes.iter().find(|n| n.local_id == id).map(|n| n.bbox);
    let c = |r: &BBox| ((r.x1 + r.x2) / 2.0, (r.y1 + r.y2) / 2.0);
    for e in b.edges.iter().filter(|e| e.label != "gRPC") {
        if let (Some(p), Some(q)) = (bx(&e.src), bx(&e.dst)) {
            marks.push(Mark::Path(vec![
                glassrip_meeting::pixel_direction::exit_point(&p, c(&q)),
                glassrip_meeting::pixel_direction::exit_point(&q, c(&p)),
            ]));
        }
    }
    marks.extend(extra.iter().cloned());
    for m in &marks {
        match m {
            Mark::Outline(r) => {
                let corners = [(r.x1, r.y1), (r.x2, r.y1), (r.x2, r.y2), (r.x1, r.y2)];
                for i in 0..4 {
                    segment(&mut img, corners[i], corners[(i + 1) % 4]);
                }
            }
            Mark::Fill(r) => {
                for y in (r.y1 as u32)..(r.y2 as u32) {
                    for x in (r.x1 as u32)..(r.x2 as u32) {
                        img.put_pixel(x, y, INK);
                    }
                }
            }
            Mark::Path(pts) => {
                for w in pts.windows(2) {
                    segment(&mut img, w[0], w[1]);
                }
            }
        }
    }
    ProbeCanvas::new(
        img,
        Some(CanvasDims {
            width: W,
            height: H,
        }),
    )
}

/// The gRPC connector from Queue to Ledger Store routed below the place between
/// them (x 890 to 1010), leaving the straight corridor and the ends' union box.
fn routed_grpc() -> Mark {
    Mark::Path(vec![
        (790.0, 220.0),
        (850.0, 220.0),
        (850.0, 330.0),
        (1050.0, 330.0),
        (1050.0, 200.0),
        (1110.0, 200.0),
    ])
}

/// One synthetic canvas per keyframe (`kfNN`).
struct Canvases(Vec<ProbeCanvas>);

impl Canvases {
    fn at(&self, keyframe_id: &str) -> Option<&ProbeCanvas> {
        let i: usize = keyframe_id.trim_start_matches("kf").parse().ok()?;
        self.0.get(i)
    }
}

impl RegionProbe for Canvases {
    fn ink_share(&self, keyframe_id: &str, region: &BBox) -> Option<f64> {
        self.at(keyframe_id)?.ink_share(region)
    }
    fn line_cover(
        &self,
        keyframe_id: &str,
        a: (f64, f64),
        b: (f64, f64),
        half_width: f64,
    ) -> Option<f64> {
        self.at(keyframe_id)?.line_cover(a, b, half_width)
    }
    fn stroke_between(
        &self,
        keyframe_id: &str,
        a: &BBox,
        b: &BBox,
        region: &BBox,
        masks: &[BBox],
        ring: f64,
    ) -> Option<StrokeTrace> {
        self.at(keyframe_id)?
            .stroke_between(a, b, region, masks, ring)
    }
    fn traces(&self) -> bool {
        true
    }
    fn uniform(&self, keyframe_id: &str) -> Option<bool> {
        Some(self.at(keyframe_id)?.uniform())
    }
}

fn paint_all(specs: &[Spec], extra: impl Fn(usize) -> Vec<Mark>) -> Canvases {
    Canvases(
        specs
            .iter()
            .enumerate()
            .map(|(i, s)| paint(s, &extra(i)))
            .collect(),
    )
}

fn run_canvases(frames: Vec<BoardFrame>, c: &Canvases) -> BoardStateItem {
    consolidate_with_probe(
        frames,
        "board-1",
        &params(),
        &hooks(&NoCorroboration),
        Some(c as &dyn RegionProbe),
    )
}

#[test]
fn a_routed_connector_is_traced_around_the_elements_between_its_ends() {
    // The reader reads the routed gRPC connector in keyframes 0 to 4 and leaves it
    // out from keyframe 5, where the board ink changes (0.2). Something sits
    // between Queue and Ledger Store: a card the reader reads ("Temp note"), or an
    // unread filled mark. Only the connector's own stroke decides.
    let grpc = |s: &BoardStateItem| s.edges.iter().find(|e| e.label == "gRPC").cloned().unwrap();
    let blob = Mark::Fill(BBox::new(930.0, 190.0, 970.0, 230.0));
    // (card read until, card drawn until, blob drawn until, connector drawn until,
    // removed at)
    let cases: [(usize, usize, usize, usize, Option<f64>); 4] = [
        // The card stays, the connector is erased: removed (the card is masked,
        // not a veto).
        (9, 9, 0, 5, Some(100.0)),
        // The card is erased, the connector stays: kept.
        (5, 5, 0, 9, None),
        // An unread mark between the ends is erased, the connector stays: kept.
        (0, 0, 5, 9, None),
        // Both the card and the connector are erased: the region changed around
        // the connector, so the pixels cannot tell and it is kept.
        (5, 5, 0, 5, None),
    ];
    for (card_read, card_drawn, blob_drawn, line_drawn, removed) in cases {
        let mut specs: Vec<Spec> = (0..9)
            .map(|_| Spec {
                ink: Some(0.01),
                ..base()
            })
            .collect();
        specs[5].ink = Some(0.2);
        for (i, s) in specs.iter_mut().enumerate() {
            if i < card_read {
                s.stickies.push(("Temp note", (950.0, 210.0)));
            }
            if i >= 5 {
                s.edges.retain(|e| e.2 != "gRPC");
            }
        }
        let canvases = paint_all(&specs, |i| {
            let mut m = Vec::new();
            if i < line_drawn {
                m.push(routed_grpc());
            }
            if i < blob_drawn {
                m.push(blob.clone());
            }
            if i < card_drawn && i >= card_read {
                m.push(Mark::Outline(BBox::new(890.0, 160.0, 1010.0, 260.0)));
            }
            m
        });
        let e = grpc(&run_canvases(frames(&specs), &canvases));
        let case = (card_read, blob_drawn, line_drawn);
        assert_eq!(e.in_final, removed.is_none(), "{case:?}: {e:?}");
        assert_eq!(
            e.lifetimes.last().unwrap().removed_at_s,
            removed,
            "{case:?}: {e:?}"
        );
    }
}

#[test]
fn a_moved_end_card_does_not_read_as_an_erased_connector() {
    // From keyframe 5 Ledger Store (and its connector) sits 60 px lower (still the
    // same node); the reader leaves the connector out. The connector is gone from
    // where it was traced, but an end read away from where it was cannot speak
    // for it.
    let grpc = |s: &BoardStateItem| s.edges.iter().find(|e| e.label == "gRPC").cloned().unwrap();
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    specs[5].ink = Some(0.2);
    for s in specs.iter_mut().skip(5) {
        s.edges.retain(|e| e.2 != "gRPC");
        for n in &mut s.nodes {
            if n.0 == "n3" {
                n.2 .1 += 60.0;
            }
        }
    }
    let canvases = paint_all(&specs, |i| {
        if i < 5 {
            vec![routed_grpc()]
        } else {
            vec![Mark::Path(vec![
                (790.0, 220.0),
                (850.0, 220.0),
                (850.0, 390.0),
                (1050.0, 390.0),
                (1050.0, 260.0),
                (1110.0, 260.0),
            ])]
        }
    });
    let s = run_canvases(frames(&specs), &canvases);
    let ledger: Vec<_> = s
        .nodes
        .iter()
        .filter(|n| n.text == "Ledger Store")
        .collect();
    assert_eq!(ledger.len(), 1, "one node, moved: {ledger:?}");
    let e = grpc(&s);
    assert!(
        e.lifetimes.iter().all(|l| l.removed_at_s.is_none()),
        "{e:?}"
    );
    // Without pixels the board ink alone removes it.
    let e = grpc(&run(frames(&specs), &params()));
    assert!(!e.in_final, "{e:?}");
}

/// A board holding one sticky, read at `place(i)` in keyframes 0 to 3; from
/// keyframe 4 the canvas is one flat color. `erase_ink` is the ink change measured
/// into keyframe 4; the flat views after it cannot be aligned with each other, so
/// their ink change is unknown. `uniform` lets the probe report flat canvases.
fn flat_canvas_case(
    place: impl Fn(usize) -> (f64, f64),
    erase_ink: Option<f64>,
    uniform: bool,
) -> BoardStateItem {
    let specs: Vec<Spec> = (0..8)
        .map(|i| Spec {
            nodes: vec![],
            edges: vec![],
            stickies: if i < 4 {
                vec![("Beta milestone in March", place(i))]
            } else {
                vec![]
            },
            ink: match i {
                0..=3 => Some(0.01),
                4 => erase_ink,
                _ => None,
            },
            ..base()
        })
        .collect();
    let mut fr = frames(&specs);
    for f in &mut fr {
        f.board.other_visible_text.clear();
    }
    // The sticky's handwriting: two strokes across its middle.
    let canvases = paint_all(&specs, |i| {
        if i >= 4 {
            return vec![];
        }
        let (x, y) = place(i);
        vec![
            Mark::Fill(BBox::new(x - 45.0, y - 18.0, x + 45.0, y - 10.0)),
            Mark::Fill(BBox::new(x - 45.0, y + 10.0, x + 30.0, y + 18.0)),
        ]
    });
    assert!(canvases.0[4].uniform() && !canvases.0[3].uniform());
    struct NoUniform<'a>(&'a Canvases);
    impl RegionProbe for NoUniform<'_> {
        fn ink_share(&self, k: &str, r: &BBox) -> Option<f64> {
            self.0.ink_share(k, r)
        }
        fn line_cover(&self, k: &str, a: (f64, f64), b: (f64, f64), h: f64) -> Option<f64> {
            self.0.line_cover(k, a, b, h)
        }
    }
    let hidden = NoUniform(&canvases);
    let probe: &dyn RegionProbe = if uniform { &canvases } else { &hidden };
    consolidate_with_probe(
        fr,
        "board-1",
        &params(),
        &hooks(&NoCorroboration),
        Some(probe),
    )
}

#[test]
fn erasing_the_only_element_on_a_flat_canvas_is_a_removal() {
    // The erasure was measured into keyframe 4 (aligned ink 0.2); the flat views
    // after it cannot be aligned. Two flat adjacent views are no change, so the
    // chain stays linked through them and the sticky is removed.
    let s = flat_canvas_case(|_| (800.0, 450.0), Some(0.2), true);
    let x = &s.stickies[0];
    assert!(!x.in_final, "{x:?}");
    assert_eq!(x.lifetimes.last().unwrap().removed_at_s, Some(80.0));
    // Without the flat-canvas measure the unknown links break the chain.
    let s = flat_canvas_case(|_| (800.0, 450.0), Some(0.2), false);
    assert!(s.stickies[0].in_final, "{:?}", s.stickies[0]);
}

#[test]
fn a_cut_or_pan_to_flat_canvas_is_not_an_erasure() {
    // The aligner could not measure the step into the flat view: an erasure, a
    // cut or a pan to empty canvas look the same there. Whether the view was
    // still before (the sticky at the same place) or moving (120 px further left
    // each keyframe), the sticky stays.
    let still = flat_canvas_case(|_| (800.0, 450.0), None, true);
    assert!(still.stickies[0].in_final, "{:?}", still.stickies[0]);
    let moving = flat_canvas_case(|i| (1100.0 - 120.0 * i as f64, 450.0), None, true);
    assert!(moving.stickies[0].in_final, "{:?}", moving.stickies[0]);
}

#[test]
fn an_erased_straight_mark_across_the_corridor_is_not_the_routed_connector() {
    // An unread straight line runs through the straight corridor between Queue
    // and Ledger Store (touching the routed connector's far leg) and is erased at
    // keyframe 5, while the routed connector stays. The corridor empties, but the
    // traced connector still joins the ends.
    let grpc = |s: &BoardStateItem| s.edges.iter().find(|e| e.label == "gRPC").cloned().unwrap();
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    specs[5].ink = Some(0.2);
    for s in specs.iter_mut().skip(5) {
        s.edges.retain(|e| e.2 != "gRPC");
    }
    let canvases = paint_all(&specs, |i| {
        let mut m = vec![routed_grpc()];
        if i < 5 {
            m.push(Mark::Path(vec![(830.0, 212.0), (1070.0, 208.0)]));
        }
        m
    });
    let e = grpc(&run_canvases(frames(&specs), &canvases));
    assert!(e.in_final, "{e:?}");
    assert!(
        e.lifetimes.iter().all(|l| l.removed_at_s.is_none()),
        "{e:?}"
    );
}

#[test]
fn a_card_moved_onto_a_drawn_connector_does_not_remove_it() {
    // A card moves from below the routed connector onto its lower leg at keyframe
    // 5 (covering part of it); the connector is still drawn and the reader leaves
    // it out. The card changed place over the traced region, so the pixels cannot
    // tell and the connector stays.
    let grpc = |s: &BoardStateItem| s.edges.iter().find(|e| e.label == "gRPC").cloned().unwrap();
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    specs[5].ink = Some(0.2);
    for (i, s) in specs.iter_mut().enumerate() {
        s.stickies.push((
            "Temp note",
            if i < 5 {
                (950.0, 460.0)
            } else {
                (950.0, 330.0)
            },
        ));
        if i >= 5 {
            s.edges.retain(|e| e.2 != "gRPC");
        }
    }
    let canvases = paint_all(&specs, |_| vec![routed_grpc()]);
    let e = grpc(&run_canvases(frames(&specs), &canvases));
    assert!(
        e.lifetimes.iter().all(|l| l.removed_at_s.is_none()),
        "{e:?}"
    );
}

#[test]
fn a_connector_erased_while_ink_is_added_near_it_is_not_vetoed_by_the_trace() {
    // A straight connector between Queue and Ledger Store is erased at keyframe 5
    // while an unread scribble is added under it; the corridor empties. The trace
    // breaks without an ink drop, which cannot tell: the corridor decides.
    let grpc = |s: &BoardStateItem| s.edges.iter().find(|e| e.label == "gRPC").cloned().unwrap();
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    specs[5].ink = Some(0.2);
    for s in specs.iter_mut().skip(5) {
        s.edges.retain(|e| e.2 != "gRPC");
    }
    let straight = Mark::Path(vec![(790.0, 216.0), (1110.0, 204.0)]);
    let canvases = paint_all(&specs, |i| {
        if i < 5 {
            vec![straight.clone()]
        } else {
            vec![Mark::Fill(BBox::new(820.0, 290.0, 1080.0, 330.0))]
        }
    });
    let e = grpc(&run_canvases(frames(&specs), &canvases));
    assert!(!e.in_final, "{e:?}");
    assert_eq!(e.lifetimes.last().unwrap().removed_at_s, Some(100.0));
}

#[test]
fn a_card_moved_off_a_rerouted_connector_does_not_remove_it() {
    // A card between Queue and Ledger Store moves 120 px down at keyframe 5 and
    // the connector is rerouted straight through the place it left; the reader
    // leaves the connector out. Each keyframe masks the card where it is read
    // there, so the new route is traced.
    let grpc = |s: &BoardStateItem| s.edges.iter().find(|e| e.label == "gRPC").cloned().unwrap();
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    specs[5].ink = Some(0.2);
    for (i, s) in specs.iter_mut().enumerate() {
        s.stickies.push((
            "Temp note",
            if i < 5 {
                (950.0, 210.0)
            } else {
                (950.0, 330.0)
            },
        ));
        if i >= 5 {
            s.edges.retain(|e| e.2 != "gRPC");
        }
    }
    let canvases = paint_all(&specs, |i| {
        if i < 5 {
            vec![Mark::Path(vec![
                (790.0, 220.0),
                (850.0, 220.0),
                (850.0, 300.0),
                (1050.0, 300.0),
                (1050.0, 200.0),
                (1110.0, 200.0),
            ])]
        } else {
            vec![Mark::Path(vec![
                (790.0, 220.0),
                (850.0, 220.0),
                (850.0, 205.0),
                (1110.0, 205.0),
            ])]
        }
    });
    let e = grpc(&run_canvases(frames(&specs), &canvases));
    assert!(
        e.lifetimes.iter().all(|l| l.removed_at_s.is_none()),
        "{e:?}"
    );
}

/// Corridor pixels as in [`Pixels`], with a stroke trace that joins the two ends
/// only while `joined[i]` holds in keyframe `i` (ink `trace_ink[i]`); `None`
/// traces nothing there.
struct TracedPixels {
    px: Pixels,
    joined: Vec<Option<bool>>,
    trace_ink: Vec<f64>,
}

impl RegionProbe for TracedPixels {
    fn ink_share(&self, keyframe_id: &str, region: &BBox) -> Option<f64> {
        self.px.ink_share(keyframe_id, region)
    }
    fn line_cover(
        &self,
        keyframe_id: &str,
        a: (f64, f64),
        b: (f64, f64),
        half_width: f64,
    ) -> Option<f64> {
        self.px.line_cover(keyframe_id, a, b, half_width)
    }
    fn stroke_between(
        &self,
        keyframe_id: &str,
        _a: &BBox,
        _b: &BBox,
        _region: &BBox,
        _masks: &[BBox],
        _ring: f64,
    ) -> Option<StrokeTrace> {
        let i: usize = keyframe_id.trim_start_matches("kf").parse().ok()?;
        Some(StrokeTrace {
            joined: (*self.joined.get(i)?)?,
            ink: *self.trace_ink.get(i)?,
        })
    }
    fn traces(&self) -> bool {
        true
    }
}

/// Codex final round 3 MAJOR (Kimi N4): the straight corridor empties (an
/// unrelated straight mark across it is erased) while the connector itself is a
/// stroke the trace never saw join its ends (dashed, or occluded by a card read
/// on its route). A trace that cannot tell does not confirm the corridor: the
/// connector stays. A connector whose own trace joined and is gone is still
/// removed.
#[test]
fn an_emptied_corridor_without_a_joined_trace_keeps_the_connector() {
    let mut specs: Vec<Spec> = (0..9)
        .map(|_| Spec {
            ink: Some(0.01),
            ..base()
        })
        .collect();
    specs[5].ink = Some(0.2);
    for s in specs.iter_mut().skip(5) {
        s.edges.retain(|e| e.2 != "gRPC");
    }
    let mut line = vec![0.9; 9];
    for v in line.iter_mut().skip(5) {
        *v = 0.05;
    }
    let probe = |joined: Vec<Option<bool>>, trace_ink: Vec<f64>| TracedPixels {
        px: Pixels {
            line: line.clone(),
            ink: vec![0.5; 9],
        },
        joined,
        trace_ink,
    };
    let run_with = |p: &TracedPixels| {
        consolidate_with_probe(
            frames(&specs),
            "board-1",
            &params(),
            &hooks(&NoCorroboration),
            Some(p as &dyn RegionProbe),
        )
    };
    let grpc = |s: &BoardStateItem| s.edges.iter().find(|e| e.label == "gRPC").cloned().unwrap();

    // The trace never joins the ends, or a tracing probe traces nothing where
    // the edge was read: the corridor alone cannot remove it.
    for joined in [Some(false), None] {
        let e = grpc(&run_with(&probe(vec![joined; 9], vec![0.2; 9])));
        assert!(e.in_final, "{joined:?}: {e:?}");
        assert!(
            e.lifetimes.iter().all(|l| l.removed_at_s.is_none()),
            "{joined:?}: {e:?}"
        );
    }

    // Positive control: the trace joined through keyframe 4 and its ink fell.
    let joined: Vec<Option<bool>> = (0..9).map(|i| Some(i < 5)).collect();
    let ink: Vec<f64> = (0..9).map(|i| if i < 5 { 0.2 } else { 0.02 }).collect();
    let e = grpc(&run_with(&probe(joined, ink)));
    assert!(!e.in_final, "{e:?}");
    assert_eq!(e.lifetimes.last().unwrap().removed_at_s, Some(100.0));
}

/// OCR spans of the fictional nodes (and optional name tags) at their true places.
fn true_ocr(names: &[(&str, (f64, f64))]) -> Vec<TextAnchor> {
    NODES
        .iter()
        .map(|(_, text, c)| (*text, *c))
        .chain(names.iter().copied())
        .map(|(text, c)| TextAnchor {
            text: text.to_string(),
            bbox: BBox::new(c.0 - 60.0, c.1 - 12.0, c.0 + 60.0, c.1 + 12.0),
        })
        .collect()
}

/// A reading whose node boxes collapsed into one strip along the top, laid out left to
/// right in list order, with the owner tag read in that strip too.
fn collapse_nodes(f: &mut BoardFrame, tag_at: (f64, f64)) {
    for (i, n) in f.board.nodes.iter_mut().enumerate() {
        let x = 10.0 + 190.0 * i as f64;
        n.bbox = BBox::new(x, 5.0, x + 180.0, 45.0);
    }
    for o in &mut f.board.owner_tags {
        o.bbox = BBox::new(
            tag_at.0 - 30.0,
            tag_at.1 - 20.0,
            tag_at.0 + 30.0,
            tag_at.1 + 20.0,
        );
    }
}

fn avery(s: &BoardStateItem) -> Vec<&glassrip_meeting::consolidate::owners::OwnerAssignment> {
    s.owner_assignments
        .iter()
        .filter(|a| a.person_id == "p-avery")
        .collect()
}

/// Avery beside Ingest Gateway for four keyframes, then a keyframe with no tag at all
/// (Ingest Gateway read and on OCR), then a long last keyframe whose reading collapsed
/// every node box into one strip; OCR places the tag beside Ledger Store.
fn end_move_frames(ocr_in_last: bool) -> Vec<BoardFrame> {
    let mut specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    for s in specs.iter_mut().take(4) {
        s.owners.push(("Avery", (200.0, 120.0), ""));
    }
    // The last view no longer shows Ingest Gateway.
    specs[5].nodes.retain(|n| n.0 != "n1");
    specs[5].owners.push(("Avery", (1200.0, 120.0), ""));
    let mut fr = frames(&specs);
    fr[4].ocr_anchors = true_ocr(&[]);
    // Read in the strip right under the collapsed Queue box.
    collapse_nodes(&mut fr[5], (100.0, 75.0));
    if ocr_in_last {
        fr[5].ocr_anchors = true_ocr(&[("Avery", (1200.0, 120.0))])
            .into_iter()
            .filter(|a| a.text != "Ingest Gateway")
            .collect();
    }
    fr[5].t_end_s = fr[5].t_start_s + 50.0;
    fr
}

#[test]
fn a_collapsed_reading_places_the_tag_on_ocr_and_a_long_last_view_moves_it() {
    let s = run(end_move_frames(true), &params());
    let a = avery(&s);
    assert_eq!(a.len(), 2, "{a:#?}");
    assert_eq!(a[0].target.texts(), vec!["Ingest Gateway"]);
    // Closed where Ingest Gateway was read with Avery nowhere.
    assert_eq!(a[0].valid_to_s, 80.0);
    assert_eq!(a[1].target.texts(), vec!["Ledger Store"]);
    assert_eq!(a[1].opened_by, OpenReason::FinalHold);
    assert_eq!(a[1].valid_from_s, 100.0);
    assert!(a[1]
        .moved_from
        .as_ref()
        .is_some_and(|m| m.texts() == vec!["Ingest Gateway"]));
    assert!(a[1].sightings.iter().all(|x| x.ocr_located));
    assert!(s
        .events
        .iter()
        .any(|e| e.kind == EventKind::OwnerMoved && e.keyframe_id == "kf05"));
    // Never tied to the strip box it was read next to.
    assert!(s
        .owner_assignments
        .iter()
        .all(|x| x.target.texts() != vec!["Queue"]));
}

#[test]
fn without_ocr_a_collapsed_last_reading_moves_nothing() {
    let s = run(end_move_frames(false), &params());
    let a = avery(&s);
    assert_eq!(a.len(), 1, "{a:#?}");
    assert_eq!(a[0].target.texts(), vec!["Ingest Gateway"]);
    assert!(s
        .owner_assignments
        .iter()
        .all(|x| x.target.texts() != vec!["Queue"]));
}

#[test]
fn one_strong_sighting_mid_meeting_opens_nothing() {
    // The same OCR-placed sighting, followed by another keyframe: a later view can
    // confirm, so one sighting is not enough.
    let mut fr = end_move_frames(true);
    let mut next = frames(&[base()]).remove(0);
    next.keyframe_id = "kf06".into();
    next.keyframe_index = 6;
    next.t_start_s = fr[5].t_end_s;
    next.t_end_s = next.t_start_s + 20.0;
    next.t_rep_s = next.t_start_s + 10.0;
    fr.push(next);
    let s = run(fr, &params());
    let a = avery(&s);
    assert_eq!(a.len(), 1, "{a:#?}");
    assert_eq!(a[0].target.texts(), vec!["Ingest Gateway"]);
}

#[test]
fn a_pan_reveals_elements_without_add_events_but_drawn_ones_are_added() {
    // Frames 0-2 zoomed in on the left part (Ledger Store and one sticky out of view),
    // frames 3-5 zoomed out to the whole board. "Cache" is drawn during the zoom-out at
    // a place the zoomed view showed empty; "Audit Log" is drawn later in a still view.
    let zoomed = Similarity {
        scale: 1.3,
        angle: 0.0,
        tx: 0.0,
        ty: 0.0,
    };
    let specs: Vec<Spec> = (0..6)
        .map(|i| {
            let mut s = Spec {
                t: if i < 3 { zoomed } else { Similarity::IDENTITY },
                ink: Some(0.2),
                ..base()
            };
            if i >= 3 {
                s.nodes.push(("n7", "Cache".into(), (450.0, 420.0)));
            }
            if i >= 4 {
                s.nodes.push(("n8", "Audit Log".into(), (1000.0, 420.0)));
            }
            s
        })
        .collect();
    let s = run(frames(&specs), &params());
    let added: Vec<(EventKind, String)> = s
        .events
        .iter()
        .filter(|e| !e.baseline)
        .filter(|e| {
            matches!(
                e.kind,
                EventKind::NodeAdded | EventKind::StickyAdded | EventKind::EdgeAdded
            )
        })
        .map(|e| (e.kind, e.detail.clone()))
        .collect();
    assert_eq!(
        added,
        vec![
            (EventKind::NodeAdded, "Cache".to_string()),
            (EventKind::NodeAdded, "Audit Log".to_string()),
        ],
        "{:#?}",
        s.events
    );
    let revealed: Vec<(EventKind, &str)> = s
        .suppressed_events
        .iter()
        .filter(|x| x.reason == SuppressReason::RevealedByView)
        .map(|x| (x.event.kind, x.event.keyframe_id.as_str()))
        .collect();
    assert!(
        revealed.contains(&(EventKind::NodeAdded, "kf03")),
        "{revealed:?}"
    );
    assert!(
        revealed.contains(&(EventKind::StickyAdded, "kf03")),
        "{revealed:?}"
    );
    assert!(
        revealed.contains(&(EventKind::EdgeAdded, "kf03")),
        "{revealed:?}"
    );
    // Revealed elements are still on the board.
    assert!(node_texts(&s).contains(&"Ledger Store".to_string()));
    assert!(s
        .stickies
        .iter()
        .any(|x| x.text == "Beta milestone in March" && x.in_final));
}

/// Ink marks on the fictional board, in reference coordinates, each drawn from a
/// keyframe on; `views` maps reference to each keyframe's canvas.
struct InkMarks {
    views: Vec<Similarity>,
    marks: Vec<(usize, BBox)>,
}

impl RegionProbe for InkMarks {
    fn ink_share(&self, keyframe_id: &str, region: &BBox) -> Option<f64> {
        let i: usize = keyframe_id.trim_start_matches("kf").parse().ok()?;
        let t = self.views.get(i)?;
        let c = ((region.x1 + region.x2) / 2.0, (region.y1 + region.y2) / 2.0);
        let r = ((c.0 - t.tx) / t.scale, (c.1 - t.ty) / t.scale);
        let inked = self.marks.iter().any(|(from, b)| {
            *from <= i && r.0 >= b.x1 && r.0 <= b.x2 && r.1 >= b.y1 && r.1 <= b.y2
        });
        Some(if inked { 0.4 } else { 0.0 })
    }
    fn line_cover(&self, _: &str, _: (f64, f64), _: (f64, f64), _: f64) -> Option<f64> {
        None
    }
}

#[test]
fn pixels_tell_a_missed_element_revealed_by_a_pan_from_one_drawn_during_it() {
    // Zoomed in (frames 0-2), then the whole board (frames 3-5). "Cache" was on the
    // board all along (ink at its place from frame 0) but only read after the zoom-out;
    // "Spare" was drawn during the zoom-out (no ink at its place before frame 3).
    let zoomed = Similarity {
        scale: 1.3,
        angle: 0.0,
        tx: 0.0,
        ty: 0.0,
    };
    let views: Vec<Similarity> = (0..6)
        .map(|i| if i < 3 { zoomed } else { Similarity::IDENTITY })
        .collect();
    let specs: Vec<Spec> = views
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let mut s = Spec {
                t: *t,
                ink: Some(0.2),
                ..base()
            };
            if i >= 3 {
                s.nodes.push(("n7", "Cache".into(), (450.0, 420.0)));
                s.nodes.push(("n8", "Spare".into(), (450.0, 540.0)));
            }
            s
        })
        .collect();
    let px = InkMarks {
        views,
        marks: vec![
            (0, BBox::new(360.0, 380.0, 540.0, 460.0)),
            (3, BBox::new(360.0, 500.0, 540.0, 580.0)),
        ],
    };
    let s = consolidate_with_probe(
        frames(&specs),
        "board-1",
        &params(),
        &hooks(&NoCorroboration),
        Some(&px as &dyn RegionProbe),
    );
    let added: Vec<String> = s
        .events
        .iter()
        .filter(|e| !e.baseline && e.kind == EventKind::NodeAdded)
        .map(|e| e.detail.clone())
        .collect();
    assert_eq!(added, vec!["Spare".to_string()], "{:#?}", s.events);
    assert!(s
        .suppressed_events
        .iter()
        .any(|x| { x.reason == SuppressReason::RevealedByView && x.event.detail == "Cache" }));
    // Without pixels, coverage alone cannot tell them apart.
    let s = run(frames(&specs), &params());
    let added: Vec<String> = s
        .events
        .iter()
        .filter(|e| !e.baseline && e.kind == EventKind::NodeAdded)
        .map(|e| e.detail.clone())
        .collect();
    assert!(added.contains(&"Cache".to_string()) && added.contains(&"Spare".to_string()));
}

/// Pixels whose traced strokes join every pair of boxes (`joined`), and whose
/// straight corridors are all inked; nothing about boxes.
struct Strokes {
    joined: bool,
}

impl RegionProbe for Strokes {
    fn ink_share(&self, _: &str, _: &BBox) -> Option<f64> {
        None
    }
    fn line_cover(&self, _: &str, _: (f64, f64), _: (f64, f64), _: f64) -> Option<f64> {
        Some(0.9)
    }
    fn stroke_between(
        &self,
        _: &str,
        _: &BBox,
        _: &BBox,
        _: &BBox,
        _: &[BBox],
        _: f64,
    ) -> Option<StrokeTrace> {
        Some(StrokeTrace {
            joined: self.joined,
            ink: 0.1,
        })
    }
    fn traces(&self) -> bool {
        true
    }
}

#[test]
fn a_connector_missed_before_a_zoom_out_is_not_added_when_a_traced_stroke_shows_it() {
    // Both ends of the HTTP edge are in the zoomed view from the start, but the
    // reader only lists the edge after the zoom-out.
    let zoomed = Similarity {
        scale: 1.3,
        angle: 0.0,
        tx: 0.0,
        ty: 0.0,
    };
    let specs: Vec<Spec> = (0..6)
        .map(|i| {
            let mut s = Spec {
                t: if i < 3 { zoomed } else { Similarity::IDENTITY },
                ink: Some(0.2),
                ..base()
            };
            if i < 3 {
                s.edges.retain(|e| e.2 != "HTTP");
            }
            s
        })
        .collect();
    let http_added = |s: &BoardStateItem| {
        s.events.iter().any(|e| {
            e.kind == EventKind::EdgeAdded && e.detail.contains("Ingest Gateway") && !e.baseline
        })
    };
    let probed = |joined: bool| {
        consolidate_with_probe(
            frames(&specs),
            "board-1",
            &params(),
            &hooks(&NoCorroboration),
            Some(&Strokes { joined } as &dyn RegionProbe),
        )
    };
    let s = probed(true);
    assert!(!http_added(&s), "{:#?}", s.events);
    assert!(s.suppressed_events.iter().any(|x| {
        x.reason == SuppressReason::RevealedByView && x.event.kind == EventKind::EdgeAdded
    }));
    // No stroke joined the ends before (corridor ink alone proves nothing), or no
    // pixels: both ends were in view, so a new connector.
    assert!(http_added(&probed(false)));
    assert!(http_added(&run(frames(&specs), &params())));
}

/// OCR spans of a spec's nodes and of the given names (reference places), mapped into
/// the reading's coordinates; spans leaving the canvas are dropped.
fn ocr_for(s: &Spec, names: &[(&str, (f64, f64))]) -> Vec<TextAnchor> {
    s.nodes
        .iter()
        .map(|(_, text, c)| (text.as_str(), *c))
        .chain(names.iter().copied())
        .map(|(text, c)| TextAnchor {
            text: text.to_string(),
            bbox: bbox_at(&s.t, c, 60.0, 12.0),
        })
        .filter(|a| a.bbox.x1 >= 0.0 && a.bbox.y1 >= 0.0 && a.bbox.x2 <= W && a.bbox.y2 <= H)
        .collect()
}

/// Frames of `specs` with OCR from `names_of(keyframe index)`.
fn frames_with_ocr(
    specs: &[Spec],
    names_of: &dyn Fn(usize) -> Vec<(&'static str, (f64, f64))>,
) -> Vec<BoardFrame> {
    let mut fr = frames(specs);
    for (i, f) in fr.iter_mut().enumerate() {
        f.ocr_anchors = ocr_for(&specs[i], &names_of(i));
    }
    fr
}

/// Above Ingest Gateway, above Ledger Store, on the Ingest Gateway to Queue connector,
/// and under Report Builder.
const ABOVE_GATEWAY: (f64, f64) = (200.0, 120.0);
const ABOVE_LEDGER: (f64, f64) = (1200.0, 120.0);
const ON_HTTP_EDGE: (f64, f64) = (450.0, 205.0);
const UNDER_REPORT: (f64, f64) = (700.0, 745.0);

#[test]
fn ocr_reads_a_tag_the_reader_missed() {
    let specs: Vec<Spec> = (0..4).map(|_| base()).collect();
    let s = run(
        frames_with_ocr(&specs, &|_| vec![("Avery", ABOVE_GATEWAY)]),
        &params(),
    );
    let a = avery(&s);
    assert_eq!(a.len(), 1, "{a:#?}");
    assert_eq!(a[0].target.texts(), vec!["Ingest Gateway"]);
    assert_eq!((a[0].valid_from_s, a[0].valid_to_s), (0.0, 80.0));
    assert!(a[0].sightings.iter().all(|x| x.name_read == NameRead::Ocr));
}

#[test]
fn reader_tags_alone_open_nothing_where_ocr_reads_the_board() {
    // The reader tags Avery in every keyframe; OCR reads the board but no name.
    let specs: Vec<Spec> = (0..4)
        .map(|_| {
            let mut s = base();
            s.owners.push(("Avery", ABOVE_GATEWAY, ""));
            s
        })
        .collect();
    let s = run(frames_with_ocr(&specs, &|_| vec![]), &params());
    assert!(avery(&s).is_empty(), "{:#?}", s.owner_assignments);
    // Without any OCR there is nothing to check against: the reader's tags count.
    let s = run(frames(&specs), &params());
    assert_eq!(avery(&s).len(), 1);
}

#[test]
fn a_misplaced_reader_tag_follows_its_ocr_text() {
    // The reader puts Avery's tag above Ledger Store; OCR reads it above Ingest Gateway.
    let specs: Vec<Spec> = (0..4)
        .map(|_| {
            let mut s = base();
            s.owners.push(("Avery", ABOVE_LEDGER, ""));
            s
        })
        .collect();
    let s = run(
        frames_with_ocr(&specs, &|_| vec![("Avery", ABOVE_GATEWAY)]),
        &params(),
    );
    let a = avery(&s);
    assert_eq!(a.len(), 1, "{a:#?}");
    assert_eq!(a[0].target.texts(), vec!["Ingest Gateway"]);
}

#[test]
fn a_misspelled_ocr_name_resolves_to_its_participant() {
    let specs: Vec<Spec> = (0..3).map(|_| base()).collect();
    let s = run(
        frames_with_ocr(&specs, &|_| vec![("Averry", ABOVE_GATEWAY)]),
        &params(),
    );
    let a = avery(&s);
    assert_eq!(a.len(), 1, "{a:#?}");
    assert_eq!(a[0].name_raw, "Averry");
    assert_eq!(a[0].target.texts(), vec!["Ingest Gateway"]);
    // A name no participant is close to is not a tag.
    let s = run(
        frames_with_ocr(&specs, &|_| vec![("Morgan", ABOVE_GATEWAY)]),
        &params(),
    );
    assert!(s.owner_assignments.is_empty());
}

#[test]
fn an_edge_tag_does_not_jump_to_the_adjacent_node() {
    // Avery's tag sits on the Ingest Gateway to Queue connector. In keyframes 2 and 3
    // the reader draws Queue's box wider, reaching the tag: those views alone anchor
    // it beside Queue. It is the same tag at the same place: the edge is kept.
    let specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    let mut fr = frames_with_ocr(&specs, &|_| vec![("Avery", ON_HTTP_EDGE)]);
    for f in &mut fr[2..4] {
        for n in &mut f.board.nodes {
            if n.local_id == "n2" {
                n.bbox.x1 = 500.0;
            }
        }
    }
    let s = run(fr, &params());
    let a = avery(&s);
    assert_eq!(a.len(), 1, "{a:#?}");
    assert_eq!(a[0].target.texts(), vec!["Ingest Gateway", "Queue"]);
    assert!(a[0].moved_from.is_none());
    assert_eq!((a[0].valid_from_s, a[0].valid_to_s), (0.0, 120.0));
    assert!(a[0]
        .sightings
        .iter()
        .any(|x| x.keyframe_id == "kf02" && x.anchor == AnchorKind::Registered));
    assert!(s.events.iter().all(|e| e.kind != EventKind::OwnerMoved));
}

#[test]
fn an_ocr_tag_moved_to_another_node_is_a_move() {
    let specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    let s = run(
        frames_with_ocr(&specs, &|i| {
            vec![("Avery", if i < 3 { ABOVE_GATEWAY } else { ABOVE_LEDGER })]
        }),
        &params(),
    );
    let a = avery(&s);
    assert_eq!(a.len(), 2, "{a:#?}");
    assert_eq!(a[0].target.texts(), vec!["Ingest Gateway"]);
    assert_eq!((a[0].valid_from_s, a[0].valid_to_s), (0.0, 60.0));
    assert_eq!(a[1].target.texts(), vec!["Ledger Store"]);
    assert_eq!(a[1].valid_from_s, 60.0);
    assert!(a[1]
        .moved_from
        .as_ref()
        .is_some_and(|m| m.texts() == vec!["Ingest Gateway"]));
}

#[test]
fn a_tag_panned_out_of_view_is_not_absent() {
    // Avery owns Ingest Gateway and Report Builder. Keyframes 2 and 3 pan the view so
    // Report Builder sits at the bottom edge and its tag below it is cut off.
    let panned = Similarity {
        scale: 1.0,
        angle: 0.0,
        tx: 0.0,
        ty: 200.0,
    };
    let specs: Vec<Spec> = (0..6)
        .map(|i| Spec {
            t: if (2..4).contains(&i) {
                panned
            } else {
                Similarity::IDENTITY
            },
            ..base()
        })
        .collect();
    let s = run(
        frames_with_ocr(&specs, &|_| {
            vec![("Avery", ABOVE_GATEWAY), ("Avery", UNDER_REPORT)]
        }),
        &params(),
    );
    assert!(s
        .registration
        .iter()
        .all(|r| r.registration.mode != RegistrationMode::TextOnly));
    let a = avery(&s);
    let report: Vec<_> = a
        .iter()
        .filter(|x| x.target.texts() == vec!["Report Builder"])
        .collect();
    assert_eq!(report.len(), 1, "{a:#?}");
    assert_eq!((report[0].valid_from_s, report[0].valid_to_s), (0.0, 120.0));
    assert_eq!(a.len(), 2, "{a:#?}");
}

#[test]
fn a_name_mentioned_in_a_sticky_is_no_tag() {
    // A sticky under Ledger Store says "Ask Avery about retention"; OCR reads the
    // name on its own line inside it.
    let specs: Vec<Spec> = (0..4)
        .map(|_| {
            let mut s = base();
            s.stickies = vec![("Ask Avery about retention", (1200.0, 300.0))];
            s
        })
        .collect();
    let s = run(
        frames_with_ocr(&specs, &|_| {
            vec![
                ("Ask", (1200.0, 272.0)),
                ("Avery", (1200.0, 300.0)),
                ("about retention", (1200.0, 328.0)),
            ]
        }),
        &params(),
    );
    assert!(avery(&s).is_empty(), "{:#?}", s.owner_assignments);
    // The same sticky's reader box thrown over a real tag elsewhere: OCR reads no
    // word of the sticky around the name, so the tag stands.
    let specs: Vec<Spec> = (0..4)
        .map(|_| {
            let mut s = base();
            s.stickies = vec![("Ask Avery about retention", ABOVE_GATEWAY)];
            s
        })
        .collect();
    let s = run(
        frames_with_ocr(&specs, &|_| vec![("Avery", ABOVE_GATEWAY)]),
        &params(),
    );
    let a = avery(&s);
    assert_eq!(a.len(), 1, "{a:#?}");
    assert_eq!(a[0].target.texts(), vec!["Ingest Gateway"]);
}

#[test]
fn a_reader_tag_ocr_does_not_see_where_it_reads_the_name_is_dropped() {
    // The reader sees two Avery tags throughout (above Ingest Gateway and above Ledger
    // Store); OCR reads the one above Ingest Gateway, and the one above Ledger Store
    // only from keyframe 4. The reader's extra tag is not evidence: Ledger Store opens
    // when OCR reads it, not before.
    let specs: Vec<Spec> = (0..6)
        .map(|_| {
            let mut s = base();
            s.owners.push(("Avery", ABOVE_GATEWAY, ""));
            s.owners.push(("Avery", ABOVE_LEDGER, ""));
            s
        })
        .collect();
    let s = run(
        frames_with_ocr(&specs, &|i| {
            let mut v = vec![("Avery", ABOVE_GATEWAY)];
            if i >= 4 {
                v.push(("Avery", ABOVE_LEDGER));
            }
            v
        }),
        &params(),
    );
    let a = avery(&s);
    let ledger: Vec<_> = a
        .iter()
        .filter(|x| x.target.texts() == vec!["Ledger Store"])
        .collect();
    assert_eq!(ledger.len(), 1, "{a:#?}");
    assert_eq!(ledger[0].valid_from_s, 80.0);
    assert!(a
        .iter()
        .any(|x| x.target.texts() == vec!["Ingest Gateway"] && x.valid_from_s == 0.0));
}

#[test]
fn a_tag_moved_next_to_the_adjacent_node_is_a_move() {
    // Avery's tag sits mid-connector for three keyframes, then is moved along it to
    // touch Queue: a new place, so a new physical tag, anchored beside Queue.
    let specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    let s = run(
        frames_with_ocr(&specs, &|i| {
            vec![("Avery", if i < 3 { ON_HTTP_EDGE } else { (560.0, 205.0) })]
        }),
        &params(),
    );
    let a = avery(&s);
    assert_eq!(a.len(), 2, "{a:#?}");
    assert_eq!(a[0].target.texts(), vec!["Ingest Gateway", "Queue"]);
    assert_eq!(a[0].valid_to_s, 60.0);
    assert_eq!(a[1].target.texts(), vec!["Queue"]);
    assert!(a[1].moved_from.is_some());
}

#[test]
fn two_tags_on_one_node_are_two_physical_tags() {
    // Two Avery tags above Ingest Gateway (left and right of it) in keyframe 0; only
    // the right one is read afterwards. The right one is seen twice in a row and opens.
    let left = (140.0, 120.0);
    let right = (290.0, 120.0);
    let specs: Vec<Spec> = (0..2).map(|_| base()).collect();
    let s = run(
        frames_with_ocr(&specs, &|i| {
            if i == 0 {
                vec![("Avery", left), ("Avery", right)]
            } else {
                vec![("Avery", right)]
            }
        }),
        &params(),
    );
    let a = avery(&s);
    assert_eq!(a.len(), 1, "{a:#?}");
    assert_eq!(a[0].target.texts(), vec!["Ingest Gateway"]);
    assert_eq!(a[0].valid_from_s, 0.0);
}

/// Where the misplaced or second Queue box sits (far from the real Queue).
const FAR_QUEUE: (f64, f64) = (1400.0, 450.0);

/// Six keyframes of the base board with OCR of the real board in every keyframe; in
/// keyframes 4 and 5 the reader puts Queue's box at `FAR_QUEUE` (OCR still reads
/// Queue at its real place). `avery_of(i)` places Avery's OCR name tag.
fn misplaced_queue_frames(
    avery_of: &dyn Fn(usize) -> Vec<(&'static str, (f64, f64))>,
) -> Vec<BoardFrame> {
    let specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    let mut fr = frames_with_ocr(&specs, avery_of);
    for f in &mut fr[4..] {
        for n in &mut f.board.nodes {
            if n.local_id == "n2" {
                n.bbox = bbox_at(&Similarity::IDENTITY, FAR_QUEUE, 90.0, 40.0);
            }
        }
    }
    fr
}

fn queue_nodes(s: &BoardStateItem) -> Vec<&glassrip_meeting::consolidate::NodeState> {
    s.nodes
        .iter()
        .filter(|n| n.text == "Queue" && n.in_final)
        .collect()
}

#[test]
fn a_misplaced_same_text_track_merges_into_the_established_node() {
    // Two keyframes read Queue far from its place while OCR reads it where it has
    // been: one Queue, with its edges, and the second track reported as a duplicate.
    let s = run(misplaced_queue_frames(&|_| vec![]), &params());
    let q = queue_nodes(&s);
    assert_eq!(q.len(), 1, "{:#?}", s.nodes);
    assert_eq!(q[0].lifetimes.last().unwrap().keyframes, 6);
    assert_eq!(
        node_texts(&s),
        vec!["Ingest Gateway", "Ledger Store", "Queue", "Report Builder"]
    );
    assert!(s
        .folded
        .iter()
        .any(|f| f.text == "Queue" && f.into == "Queue" && f.reason == FoldReason::Duplicate));
    // The edges read to the misplaced box in keyframes 4 and 5 are Queue's edges:
    // every Queue edge is seen in all six keyframes.
    let queue_edges: Vec<_> = s
        .edges
        .iter()
        .filter(|e| e.a_text == "Queue" || e.b_text == "Queue")
        .collect();
    assert_eq!(queue_edges.len(), 3, "{:#?}", s.edges);
    for e in queue_edges {
        assert!(e.in_final);
        assert_eq!(e.lifetimes.last().unwrap().keyframes, 6, "{e:#?}");
    }
}

#[test]
fn a_same_text_box_on_a_collapsed_reading_merges() {
    // No OCR at all, but the last two readings collapse every node box into one
    // place: the far Queue box there is no evidence of a second Queue.
    let specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    let mut fr = frames(&specs);
    for f in &mut fr[4..] {
        for n in &mut f.board.nodes {
            n.bbox = bbox_at(&Similarity::IDENTITY, FAR_QUEUE, 90.0, 40.0);
        }
    }
    let s = run(fr, &params());
    let q = queue_nodes(&s);
    assert_eq!(q.len(), 1, "{:#?}", s.nodes);
    assert_eq!(q[0].lifetimes.last().unwrap().keyframes, 6);
    assert!(s
        .folded
        .iter()
        .any(|f| f.text == "Queue" && f.reason == FoldReason::Duplicate));
}

#[test]
fn a_distant_second_box_that_ocr_confirms_stays_a_second_node() {
    // A real second Queue box from keyframe 2 on, read by the reader and by OCR at
    // its own place on a reliable registration: two Queues.
    let mut specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    for s in specs.iter_mut().skip(2) {
        s.nodes.push(("n9", "Queue".into(), FAR_QUEUE));
    }
    let s = run(frames_with_ocr(&specs, &|_| vec![]), &params());
    assert_eq!(queue_nodes(&s).len(), 2, "{:#?}", s.nodes);
    assert!(s.folded.iter().all(|f| f.reason != FoldReason::Duplicate));
    // Without OCR nothing says the second place is wrong: still two.
    let s = run(frames(&specs), &params());
    assert_eq!(queue_nodes(&s).len(), 2, "{:#?}", s.nodes);
}

#[test]
fn an_adjacent_second_box_that_ocr_confirms_stays_a_second_node() {
    // A second Queue box 150 px right of the first (the two boxes nearly touch):
    // OCR reads Queue inside each box. The margin around the first box does not claim
    // the second box's text: two Queues.
    let mut specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    for s in specs.iter_mut().skip(2) {
        s.nodes.push(("n9", "Queue".into(), (850.0, 220.0)));
    }
    let s = run(frames_with_ocr(&specs, &|_| vec![]), &params());
    assert_eq!(queue_nodes(&s).len(), 2, "{:#?}", s.nodes);
    assert!(s.folded.iter().all(|f| f.reason != FoldReason::Duplicate));
}

#[test]
fn an_owner_tag_on_a_merged_duplicate_targets_the_established_node() {
    // Avery sits above Ledger Store, then moves above Queue in the two keyframes that
    // misplace Queue's box: the move targets the one Queue node.
    let s = run(
        misplaced_queue_frames(&|i| {
            vec![("Avery", if i < 4 { ABOVE_LEDGER } else { (700.0, 140.0) })]
        }),
        &params(),
    );
    let q = queue_nodes(&s);
    assert_eq!(q.len(), 1, "{:#?}", s.nodes);
    let a = avery(&s);
    let last = a.last().expect("an assignment");
    assert_eq!(last.target.texts(), vec!["Queue"], "{a:#?}");
    match &last.target {
        OwnerTarget::Node { node_id, .. } => assert_eq!(node_id, &q[0].id),
        other => panic!("{other:?}"),
    }
    assert!(last
        .moved_from
        .as_ref()
        .is_some_and(|m| m.texts() == vec!["Ledger Store"]));
}

#[test]
fn a_node_read_as_a_sticky_as_often_or_marked_as_one_is_a_sticky() {
    // "Cache warmup plan" is read as a node in keyframes 0-1 and as a sticky in 2-3 at
    // one place; "Pilot milestone in May" is always read as a node; neither has a
    // connector. "Is the ledger durable?" is read as a node joined to Ledger Store.
    let mut specs: Vec<Spec> = (0..4).map(|_| base()).collect();
    for (i, s) in specs.iter_mut().enumerate() {
        if i < 2 {
            s.nodes
                .push(("n7", "Cache warmup plan".into(), (1400.0, 450.0)));
        } else {
            s.stickies.push(("Cache warmup plan", (1400.0, 450.0)));
        }
        s.nodes
            .push(("n8", "Pilot milestone in May".into(), (200.0, 450.0)));
        s.nodes
            .push(("n9", "Is the ledger durable?".into(), (1200.0, 450.0)));
        s.edges.push(("n3", "n9", ""));
    }
    let s = run(frames(&specs), &params());
    let texts = node_texts(&s);
    assert!(
        !texts.contains(&"Cache warmup plan".to_string()),
        "{texts:?}"
    );
    assert!(!texts.contains(&"Pilot milestone in May".to_string()));
    assert!(texts.contains(&"Is the ledger durable?".to_string()));
    let sticky = |t: &str| s.stickies.iter().find(|x| x.text == t && x.in_final);
    assert_eq!(
        sticky("Pilot milestone in May").map(|x| x.kind),
        Some(StickyKind::Milestone)
    );
    assert!(sticky("Cache warmup plan").is_some(), "{:#?}", s.stickies);
}

#[test]
fn a_same_text_box_where_the_registration_is_off_merges() {
    // No OCR. In keyframes 4 and 5 the reader throws Queue far from every other
    // element, and Ledger Store to another far place (a different one in each
    // keyframe): nothing read near the Queue box confirms the registration there, and
    // the keyframe misplaces another known element. The Queue box is no second Queue.
    let specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    let mut fr = frames(&specs);
    let id = Similarity::IDENTITY;
    for (k, ledger_at) in [(4, (-1200.0, 300.0)), (5, (-1200.0, 1400.0))] {
        for n in &mut fr[k].board.nodes {
            match n.local_id.as_str() {
                "n2" => n.bbox = bbox_at(&id, (-700.0, -600.0), 90.0, 40.0),
                "n3" => n.bbox = bbox_at(&id, ledger_at, 90.0, 40.0),
                _ => {}
            }
        }
    }
    let s = run(fr, &params());
    let q = queue_nodes(&s);
    assert_eq!(q.len(), 1, "{:#?}", s.nodes);
    assert_eq!(q[0].lifetimes.last().unwrap().keyframes, 6);
    assert!(s
        .folded
        .iter()
        .any(|f| f.text == "Queue" && f.reason == FoldReason::Duplicate));
    // With the Ledger Store boxes where they belong, nothing in those keyframes says
    // the registration is off: the far Queue box stays unexplained.
    let mut fr = frames(&specs);
    for f in &mut fr[4..] {
        for n in &mut f.board.nodes {
            if n.local_id == "n2" {
                n.bbox = bbox_at(&id, (-700.0, -600.0), 90.0, 40.0);
            }
        }
    }
    let s = run(fr, &params());
    assert!(s.folded.iter().all(|f| f.reason != FoldReason::Duplicate));
}

/// Codex r5 integration, MAJOR 2: connector ends were taken before fragments
/// were redirected, so a connector read at a fragment did not keep the element
/// it folds into a node; the element became a sticky and the edge was dropped.
#[test]
fn a_supported_connector_at_a_fragment_keeps_its_element_a_node_with_the_edge() {
    // "Pilot milestone in May" is read whole in keyframes 0, 1, 4 and 5; keyframes 2
    // and 3 read only its fragment "Pilot milestone" over the element's place
    // (offset, so a separate reading rather than a label change), with a connector
    // from Queue in both (an edge needs support). The fragment folds into the
    // element, its connector with it: the element is a node (only nodes carry
    // edges) and the connector survives.
    let mut specs: Vec<Spec> = (0..6).map(|_| base()).collect();
    for (i, s) in specs.iter_mut().enumerate() {
        if !(2..4).contains(&i) {
            s.nodes
                .push(("n8", "Pilot milestone in May".into(), (200.0, 450.0)));
        } else {
            s.nodes
                .push(("n8", "Pilot milestone".into(), (280.0, 450.0)));
            s.edges.push(("n2", "n8", ""));
        }
    }
    let s = run(frames(&specs), &params());
    assert!(s
        .folded
        .iter()
        .any(|f| f.text == "Pilot milestone" && f.into == "Pilot milestone in May"));
    let texts = node_texts(&s);
    assert!(
        texts.contains(&"Pilot milestone in May".to_string()),
        "{texts:?}"
    );
    assert!(!s
        .stickies
        .iter()
        .any(|x| x.text.starts_with("Pilot milestone")));
    let milestone_edge = |s: &BoardStateItem| {
        s.edges.iter().any(|e| {
            let ends = [e.a_text.as_str(), e.b_text.as_str()];
            ends.contains(&"Queue") && ends.contains(&"Pilot milestone in May")
        })
    };
    assert!(milestone_edge(&s), "{:#?}", s.edges);
}

#[test]
fn a_connector_read_once_at_a_fragment_does_not_keep_a_marked_element_a_node() {
    // As above, but only keyframe 4 reads the fragment with its connector: the
    // edge has no support and is dropped whatever the element's kind, so it does
    // not override the milestone marker; the element is a sticky.
    let mut specs: Vec<Spec> = (0..5).map(|_| base()).collect();
    for (i, s) in specs.iter_mut().enumerate() {
        if i < 4 {
            s.nodes
                .push(("n8", "Pilot milestone in May".into(), (200.0, 450.0)));
        } else {
            s.nodes
                .push(("n8", "Pilot milestone".into(), (280.0, 450.0)));
            s.edges.push(("n2", "n8", ""));
        }
    }
    let s = run(frames(&specs), &params());
    let texts = node_texts(&s);
    assert!(
        !texts.iter().any(|t| t.starts_with("Pilot milestone")),
        "{texts:?}"
    );
    assert!(s
        .stickies
        .iter()
        .any(|x| x.text == "Pilot milestone in May" && x.kind == StickyKind::Milestone));
}

/// Codex r5 fix round 3: two fragment-connector sightings met the keyframe
/// count, but lying far apart they fail the edge's density rule, so no edge was
/// drawn while the milestone was still kept a node.
#[test]
fn far_apart_fragment_connectors_that_draw_no_edge_leave_a_marked_element_a_sticky() {
    // 25 keyframes: "Pilot milestone in May" whole everywhere but keyframes 2 and
    // 24, which read its fragment with a connector from Queue; two sightings over
    // 23 keyframes are below the support density, so the edge is not drawn.
    let mut specs: Vec<Spec> = (0..25).map(|_| base()).collect();
    for (i, s) in specs.iter_mut().enumerate() {
        if i != 2 && i != 24 {
            s.nodes
                .push(("n8", "Pilot milestone in May".into(), (200.0, 450.0)));
        } else {
            s.nodes
                .push(("n8", "Pilot milestone".into(), (280.0, 450.0)));
            s.edges.push(("n2", "n8", ""));
        }
    }
    let s = run(frames(&specs), &params());
    assert!(
        !s.edges
            .iter()
            .any(|e| e.a_text.starts_with("Pilot milestone")
                || e.b_text.starts_with("Pilot milestone")),
        "{:#?}",
        s.edges
    );
    assert!(
        !s.nodes
            .iter()
            .any(|n| n.text.starts_with("Pilot milestone")),
        "{:?}",
        node_texts(&s)
    );
    assert!(s
        .stickies
        .iter()
        .any(|x| x.text == "Pilot milestone in May" && x.kind == StickyKind::Milestone));
}
