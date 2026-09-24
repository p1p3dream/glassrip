//! Board-state consolidation on synthetic reading sequences (fictional board and names).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use glassrip_meeting::artifacts::{CanvasDims, EdgeDirectionItem, EdgeEvidence};
use glassrip_meeting::consolidate::events::EventKind;
use glassrip_meeting::consolidate::events::SuppressReason;
use glassrip_meeting::consolidate::owners::{
    AnchorKind, Corroboration, Corroborator, MoveQuery, NoCorroboration, OwnerTarget,
};
use glassrip_meeting::consolidate::{
    consolidate, split_boards, BoardFrame, BoardStateItem, CanvasSource, ConsolidationParams,
    EdgeOrientation, Hooks, StickyKind, TextAnchor,
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
