//! A synthetic `glassrip.board_state` produced by glassrip-meeting's own
//! consolidation from fictional per-keyframe readings (no hand-written board
//! state). Shared by the notes and render tests through `#[path]`.

#![allow(dead_code, clippy::type_complexity)]

use std::collections::BTreeMap;

use glassrip_meeting::artifacts::{
    AxisScale, CanvasDims, CoordinateCheck, EdgeDirectionItem, EdgeEvidence,
};
use glassrip_meeting::consolidate::events::EventKind;
use glassrip_meeting::consolidate::owners::NoCorroboration;
use glassrip_meeting::consolidate::{
    consolidate, BoardFrame, BoardStateItem, ConsolidationParams, Hooks,
};
use glassrip_meeting::direction::EndVerdict;
use glassrip_meeting::pixel_direction::{EndEvidence, PixelEvidence, PixelStatus};
use glassrip_meeting::text::{AliasTable, Participant};
use glassrip_vision::board::{
    BoardEdge, BoardNode, EdgeStyle, OwnerTag, Sticky, StickyColor, TextItem, ValidatedBoard,
};
use glassrip_vision::BBox;

const W: f64 = 1600.0;
const H: f64 = 900.0;
/// Keyframes, 10 s each.
pub const FRAMES: usize = 8;

fn bbox(c: (f64, f64), hw: f64, hh: f64) -> BBox {
    BBox::new(c.0 - hw, c.1 - hh, c.0 + hw, c.1 + hh)
}

/// Nodes: (local id, text, center, first keyframe, last keyframe inclusive).
const NODES: [(&str, &str, (f64, f64), usize, usize); 5] = [
    ("n_ledger", "Ledger Service", (250.0, 200.0), 0, 7),
    ("n_relay", "Relay API", (250.0, 520.0), 0, 7),
    ("n_sketch", "Old Sketch", (800.0, 170.0), 0, 2),
    ("n_kiosk", "Kiosk App", (800.0, 520.0), 1, 7),
    ("n_kit", "Design Kit", (1300.0, 520.0), 1, 7),
];

/// Edges: (src, dst, label, style, first keyframe).
const EDGES: [(&str, &str, &str, EdgeStyle, usize); 4] = [
    ("n_relay", "n_ledger", "REST", EdgeStyle::Solid, 0),
    ("n_kiosk", "n_relay", "GraphQL", EdgeStyle::Solid, 1),
    ("n_kiosk", "n_kit", "", EdgeStyle::Solid, 1),
    (
        "n_ledger",
        "n_kit",
        "Links between ledger entries + kit widgets",
        EdgeStyle::Dashed,
        4,
    ),
];

/// Stickies: (text, center, first keyframe).
const STICKIES: [(&str, (f64, f64), usize); 5] = [
    (
        "Which widgets do we need for the kiosk?",
        (1400.0, 180.0),
        1,
    ),
    ("Do we change the badge flow?", (550.0, 780.0), 2),
    ("Idea: visitors pick their badge color", (850.0, 780.0), 4),
    ("Clone the lobby page", (1100.0, 780.0), 4),
    ("Pilot milestone in May", (1350.0, 780.0), 4),
];

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

fn frame(i: usize) -> BoardFrame {
    let alive = |a: usize, b: usize| i >= a && i <= b;
    let nodes: Vec<BoardNode> = NODES
        .iter()
        .filter(|n| alive(n.3, n.4))
        .map(|(id, text, c, _, _)| BoardNode {
            local_id: (*id).into(),
            text: (*text).into(),
            bbox: bbox(*c, 100.0, 40.0),
            conf: 0.9,
        })
        .collect();
    let ids: Vec<&str> = nodes.iter().map(|n| n.local_id.as_str()).collect();
    let edges_here: Vec<&(&str, &str, &str, EdgeStyle, usize)> = EDGES
        .iter()
        .filter(|e| i >= e.4 && ids.contains(&e.0) && ids.contains(&e.1))
        .collect();
    let edges = edges_here
        .iter()
        .map(|(a, b, l, st, _)| BoardEdge {
            src: (*a).into(),
            dst: (*b).into(),
            label: (*l).into(),
            label_bbox: None,
            style: *st,
            conf: 0.8,
        })
        .collect();
    let stickies = STICKIES
        .iter()
        .filter(|s| i >= s.2)
        .map(|(text, c, _)| Sticky {
            text: (*text).into(),
            color: StickyColor::Yellow,
            bbox: bbox(*c, 70.0, 50.0),
        })
        .collect();
    // owner tags from keyframe 4; Mira's tag moves from the ledger to the kit at 6
    let mut owner_tags = Vec::new();
    if i >= 4 {
        owner_tags.push(("Avery", (800.0, 440.0)));
        owner_tags.push(("Rohan", (525.0, 520.0)));
        owner_tags.push((
            "Mira",
            if i < 6 {
                (250.0, 120.0)
            } else {
                (1300.0, 440.0)
            },
        ));
    }
    let owner_tags = owner_tags
        .into_iter()
        .map(|(name, c)| OwnerTag {
            name_raw: name.into(),
            near: String::new(),
            bbox: bbox(c, 30.0, 18.0),
        })
        .collect();
    // pixel evidence: every arrowhead at the reader's dst
    let evidence = edges_here
        .iter()
        .map(|(a, b, l, st, _)| {
            let at = |id: &str| {
                NODES
                    .iter()
                    .find(|n| n.0 == id)
                    .map(|n| n.2)
                    .unwrap_or((0.0, 0.0))
            };
            let (pa, pb) = (at(a), at(b));
            EdgeEvidence {
                src: (*a).into(),
                dst: (*b).into(),
                src_text: String::new(),
                dst_text: String::new(),
                label: (*l).into(),
                style: *st,
                pixel: PixelEvidence {
                    status: PixelStatus::Traced,
                    // line ends at the node centers (the connector the tags sit on)
                    src_end: end(pa.0, pa.1),
                    dst_end: end(pb.0, pb.1),
                    stroke_px: 2.0,
                    verdict: EndVerdict::Forward,
                },
                vlm: None,
            }
        })
        .collect();
    let id = format!("kf_{:06}", i * 10);
    BoardFrame {
        keyframe_id: id.clone(),
        keyframe_index: i,
        t_start_s: i as f64 * 10.0,
        t_end_s: i as f64 * 10.0 + 10.0,
        t_rep_s: i as f64 * 10.0 + 5.0,
        canvas: Some(CanvasDims {
            width: W,
            height: H,
        }),
        board_title: None,
        board: ValidatedBoard {
            nodes,
            edges,
            stickies,
            owner_tags,
            other_visible_text: vec![TextItem {
                text: "Kiosk relay pilot".into(),
                bbox: bbox((800.0, 40.0), 150.0, 15.0),
            }],
            confidence: 0.9,
            chrome_rejected: vec![],
            issues: vec![],
            needs_reclassification: false,
        },
        ink_change: if i == 0 { None } else { Some(0.2) },
        directions: Some(EdgeDirectionItem {
            keyframe_id: id,
            canvas: CanvasDims {
                width: W,
                height: H,
            },
            board_to_image: AxisScale { x: 1.0, y: 1.0 },
            coordinates: CoordinateCheck::Verified1to1,
            crop_in_frame: None,
            sharpness: 300.0,
            zoom: 80.0,
            edges: evidence,
            error: None,
        }),
        ocr_anchors: vec![],
    }
}

/// Participants of the fictional meeting (same ids the notes derive).
pub fn participants() -> AliasTable {
    let p = |id: &str, name: &str, alias: &str| Participant {
        person_id: id.into(),
        display_name: name.into(),
        aliases: vec![alias.into()],
    };
    AliasTable::new(vec![
        p("avery-quinn", "Avery Quinn", "Avery"),
        p("rohan-dasgupta", "Rohan Dasgupta", "Rohan"),
        p("mira-okafor", "Mira Okafor", "Mira"),
    ])
}

/// The consolidated board and the matching `glassrip.keyframes` items.
pub fn synthetic_board() -> (BoardStateItem, Vec<serde_json::Value>) {
    let frames: Vec<BoardFrame> = (0..FRAMES).map(frame).collect();
    let keyframes = frames
        .iter()
        .map(|f| {
            serde_json::json!({
                "keyframe_id": f.keyframe_id,
                "t_start_s": f.t_start_s,
                "t_end_s": f.t_end_s,
                "t_rep_s": f.t_rep_s,
            })
        })
        .collect();
    let params = ConsolidationParams {
        participants: participants(),
        final_window_s: 30.0,
        ..ConsolidationParams::default()
    };
    let hooks = Hooks {
        corroborator: &NoCorroboration,
        second_reader: None,
    };
    (consolidate(frames, "board-1", &params, &hooks), keyframes)
}

/// Symbolic ids used in the replayed model responses, resolved against the
/// consolidated board (event ids are assigned by the producer).
pub fn symbolic_ids(b: &BoardStateItem) -> BTreeMap<&'static str, String> {
    let find = |kind: EventKind, detail: &str| {
        b.events
            .iter()
            .find(|e| e.kind == kind && e.detail.contains(detail))
            .map(|e| (e.event_id.clone(), e.keyframe_id.clone()))
            .unwrap_or_else(|| panic!("no {kind:?} event with {detail:?}: {:#?}", b.events))
    };
    let mut m = BTreeMap::new();
    m.insert("EV_KIOSK", find(EventKind::NodeAdded, "Kiosk App").0);
    m.insert("EV_BADGE_Q", find(EventKind::StickyAdded, "badge flow").0);
    let (ev, kf) = find(EventKind::StickyAdded, "widgets");
    m.insert("EV_WIDGETS_Q", ev);
    m.insert("KF_WIDGETS", kf);
    m.insert("EV_OWNERS", find(EventKind::OwnerAssigned, "").0);
    m.insert("EV_MOVE", find(EventKind::OwnerMoved, "").0);
    m
}

/// Replaces symbolic ids in `text`.
pub fn resolve(text: &str, ids: &BTreeMap<&'static str, String>) -> String {
    let mut out = text.to_string();
    // longest keys first so EV_WIDGETS_Q is not cut by a shorter key
    let mut keys: Vec<&&str> = ids.keys().collect();
    keys.sort_by_key(|k| std::cmp::Reverse(k.len()));
    for k in keys {
        out = out.replace(*k, &ids[*k]);
    }
    out
}
