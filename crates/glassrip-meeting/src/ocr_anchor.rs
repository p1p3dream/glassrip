//! Re-anchoring reader boxes to OCR text before the pixel check.
//!
//! The vision reader's boxes can sit well off the drawn shapes (tens of pixels on a
//! zoomed canvas), far beyond what snapping each side to its outline can recover.
//! The pixel check then treats a node's real outline, outside the reader's box, as
//! part of the connector, and attributes an arrowhead that touches one node to the
//! other end. OCR boxes are pixel-accurate, so they locate the text of each node and
//! label:
//!
//! - **Nodes.** Each OCR span belongs to at most one node: among the nodes whose
//!   text it matches and whose search window holds it, the one whose reader box is
//!   nearest. A node's own spans are grouped around the one closest to its box,
//!   each further span adding words the group does not have yet (two boxes both
//!   named "Service" never pool their spans). When the group covers enough of the
//!   node's text, does not sit on another node's box, and is displaced from the box
//!   center by more than a share of the box size, the box is translated onto it
//!   (keeping its size, grown to contain it). Smaller offsets are left to the
//!   outline snap.
//! - **Edge labels.** The label box becomes the OCR box of the label text found near
//!   the reader's label box, or, when there is none, in the corridor between the two
//!   nodes (a repeated short label elsewhere around them is another connector's).
//!   Spans inside a node's box are never a label.
//! - **Masking.** Every OCR text box is returned for masking, so label glyphs never
//!   join a connector wherever the reader placed its boxes.
//!
//! OCR boxes come from the DB detector, which expands each text region by
//! `area * unclip_ratio / perimeter` on every side (about 0.75 of the text height at
//! the default ratio 1.5). Used as is, such a box covers an arrowhead drawn next to
//! a label, so every OCR box is first shrunk back to its text core
//! ([`glyph_box`]).

use glassrip_vision::board::ValidatedBoard;
use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::consolidate::TextAnchor;
use crate::text::{clean_label, normalize};

/// Re-anchoring settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OcrAnchorParams {
    /// Re-anchor at all (OCR spans must be available).
    pub enabled: bool,
    /// Search window around a reader box, as a share of its width and height on
    /// each side.
    pub search_share: f64,
    /// Share of the node's text (in characters) the grouped spans must cover.
    pub min_text_cover: f64,
    /// Displacements under this share of the box's smaller side are left to the
    /// outline snap.
    pub min_shift_share: f64,
    /// Two words of four or more letters match when their similarity ratio
    /// ([`crate::difflib::ratio`]) reaches this (OCR misreads a letter or two).
    pub word_similarity: f64,
    /// The OCR detector's box expansion ratio (DB `unclip_ratio`), undone by
    /// [`glyph_box`]; 0 keeps OCR boxes as they are.
    #[serde(default = "default_unclip_ratio")]
    pub ocr_unclip_ratio: f64,
}

fn default_unclip_ratio() -> f64 {
    1.5
}

/// The text core of a DB-detected box: the detector grew a region of `w x h` by
/// `d = w h r / (2 (w + h))` on every side, so for the final `W x H` box `d` is
/// the smaller root of `(8 + 4r) d^2 - 2 (1 + r)(W + H) d + r W H = 0`.
pub fn glyph_box(b: &BBox, unclip_ratio: f64) -> BBox {
    let (w, h, r) = (b.width(), b.height(), unclip_ratio);
    if r <= 0.0 || w <= 0.0 || h <= 0.0 {
        return *b;
    }
    let a = 8.0 + 4.0 * r;
    let bb = 2.0 * (1.0 + r) * (w + h);
    let disc = bb * bb - 4.0 * a * r * w * h;
    if disc < 0.0 {
        return *b;
    }
    let d = ((bb - disc.sqrt()) / (2.0 * a)).clamp(0.0, w.min(h) / 2.0 - 0.5);
    if d <= 0.0 {
        return *b;
    }
    BBox::new(b.x1 + d, b.y1 + d, b.x2 - d, b.y2 - d)
}

impl Default for OcrAnchorParams {
    fn default() -> Self {
        Self {
            enabled: true,
            search_share: 1.0,
            min_text_cover: 0.5,
            min_shift_share: 0.15,
            word_similarity: 0.75,
            ocr_unclip_ratio: default_unclip_ratio(),
        }
    }
}

/// What re-anchoring changed in one keyframe.
#[derive(Debug, Clone, PartialEq)]
pub struct Reanchored {
    /// The reading with node and label boxes moved onto their OCR text.
    pub board: ValidatedBoard,
    /// Nodes whose box was translated.
    pub nodes_moved: usize,
    /// Edge labels whose box now comes from OCR.
    pub labels_found: usize,
    /// Every usable OCR text box (to mask).
    pub text_boxes: Vec<BBox>,
}

fn words(text: &str) -> Vec<String> {
    normalize(text)
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn center(b: &BBox) -> (f64, f64) {
    ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
}

fn union(boxes: &[BBox]) -> Option<BBox> {
    boxes.iter().copied().reduce(|a, b| {
        BBox::new(
            a.x1.min(b.x1),
            a.y1.min(b.y1),
            a.x2.max(b.x2),
            a.y2.max(b.y2),
        )
    })
}

fn dist2(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}

fn inside(p: (f64, f64), b: &BBox) -> bool {
    p.0 >= b.x1 && p.0 <= b.x2 && p.1 >= b.y1 && p.1 <= b.y2
}

fn grow(b: &BBox, share: f64) -> BBox {
    let (dx, dy) = (b.width() * share, b.height() * share);
    BBox::new(b.x1 - dx, b.y1 - dy, b.x2 + dx, b.y2 + dy)
}

/// A usable OCR span: at least two alphanumeric characters.
struct Span {
    words: Vec<String>,
    bbox: BBox,
}

/// Indexes of `target` words matched by `span` words, or `None` when a span word
/// matches none of them (the span is other text).
fn matched_words(span: &[String], target: &[String], sim: f64) -> Option<Vec<usize>> {
    let mut hit = Vec::new();
    for w in span {
        let best = target
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                *t == w || (w.chars().count() >= 4 && crate::difflib::ratio(t, w) >= sim)
            })
            .map(|(i, _)| i)
            .next()?;
        hit.push(best);
    }
    Some(hit)
}

/// Spans of `text` near `window`, grouped around the one closest to `anchor`, and
/// the share of `text`'s characters they cover. Spans join the group in order of
/// distance from the seed, within `reach` of it on each axis, and only when they add
/// words the group does not cover yet: a second copy of the same text is another
/// element's.
fn locate(
    spans: &[&Span],
    text: &str,
    window: &BBox,
    anchor: (f64, f64),
    reach: (f64, f64),
    sim: f64,
) -> Option<(BBox, f64)> {
    let target = words(text);
    if target.is_empty() {
        return None;
    }
    let cands: Vec<(&Span, Vec<usize>)> = spans
        .iter()
        .copied()
        .filter(|s| inside(center(&s.bbox), window))
        .filter_map(|s| matched_words(&s.words, &target, sim).map(|m| (s, m)))
        .collect();
    let dist = |b: &BBox, p: (f64, f64)| {
        let c = center(b);
        (c.0 - p.0).powi(2) + (c.1 - p.1).powi(2)
    };
    let seed = cands
        .iter()
        .min_by(|a, b| dist(&a.0.bbox, anchor).total_cmp(&dist(&b.0.bbox, anchor)))?;
    let sc = center(&seed.0.bbox);
    let mut order: Vec<&(&Span, Vec<usize>)> = cands
        .iter()
        .filter(|(s, _)| {
            let c = center(&s.bbox);
            (c.0 - sc.0).abs() <= reach.0 && (c.1 - sc.1).abs() <= reach.1
        })
        .collect();
    order.sort_by(|a, b| dist(&a.0.bbox, sc).total_cmp(&dist(&b.0.bbox, sc)));
    let mut covered: Vec<usize> = Vec::new();
    let mut boxes: Vec<BBox> = Vec::new();
    for (s, m) in order {
        if m.iter().all(|i| covered.contains(i)) {
            continue;
        }
        covered.extend(m.iter().copied());
        boxes.push(s.bbox);
    }
    covered.sort_unstable();
    covered.dedup();
    let total: usize = target.iter().map(|w| w.chars().count()).sum();
    let got: usize = covered.iter().map(|&i| target[i].chars().count()).sum();
    union(&boxes).map(|u| (u, got as f64 / total.max(1) as f64))
}

/// Distance from `p` to the segment `a`-`b`.
fn segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p.0 - (a.0 + t * dx)).hypot(p.1 - (a.1 + t * dy))
}

/// Distance from `p` to box `b` (0 inside).
fn box_distance(p: (f64, f64), b: &BBox) -> f64 {
    let dx = (b.x1 - p.0).max(p.0 - b.x2).max(0.0);
    let dy = (b.y1 - p.1).max(p.1 - b.y2).max(0.0);
    dx.hypot(dy)
}

/// Move node and label boxes of `board` onto their OCR text. `ocr` must share the
/// board's coordinate space.
pub fn reanchor(board: &ValidatedBoard, ocr: &[TextAnchor], p: &OcrAnchorParams) -> Reanchored {
    let spans: Vec<Span> = ocr
        .iter()
        .filter(|a| a.bbox.is_well_formed())
        .filter(|a| a.text.chars().filter(|c| c.is_alphanumeric()).count() >= 2)
        .map(|a| Span {
            words: words(&a.text),
            bbox: glyph_box(&a.bbox, p.ocr_unclip_ratio),
        })
        .filter(|s| !s.words.is_empty())
        .collect();
    let mut out = Reanchored {
        board: board.clone(),
        nodes_moved: 0,
        labels_found: 0,
        text_boxes: spans.iter().map(|s| s.bbox).collect(),
    };
    if !p.enabled || spans.is_empty() {
        return out;
    }
    // One owner per span: the matching node whose reader box is nearest.
    let owner: Vec<Option<usize>> = spans
        .iter()
        .map(|s| {
            let c = center(&s.bbox);
            board
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.bbox.width() > 0.0 && n.bbox.height() > 0.0)
                .filter(|(_, n)| inside(c, &grow(&n.bbox, p.search_share)))
                .filter(|(_, n)| {
                    matched_words(&s.words, &words(&n.text), p.word_similarity).is_some()
                })
                .min_by(|a, b| {
                    (box_distance(c, &a.1.bbox), dist2(c, center(&a.1.bbox)))
                        .partial_cmp(&(box_distance(c, &b.1.bbox), dist2(c, center(&b.1.bbox))))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
        })
        .collect();
    for (ni, n) in out.board.nodes.iter_mut().enumerate() {
        let b = n.bbox;
        let (w, h) = (b.width(), b.height());
        if w <= 0.0 || h <= 0.0 {
            continue;
        }
        let own: Vec<&Span> = spans
            .iter()
            .zip(&owner)
            .filter(|(_, o)| **o == Some(ni))
            .map(|(s, _)| s)
            .collect();
        let window = grow(&b, p.search_share);
        let Some((u, cover)) = locate(
            &own,
            &n.text,
            &window,
            center(&b),
            (w, h),
            p.word_similarity,
        ) else {
            continue;
        };
        if cover < p.min_text_cover {
            continue;
        }
        // Text that sits on another node's box is that node's.
        let on_other = board
            .nodes
            .iter()
            .enumerate()
            .any(|(j, m)| j != ni && inside(center(&u), &m.bbox) && !inside(center(&u), &b));
        if on_other {
            continue;
        }
        let (bc, uc) = (center(&b), center(&u));
        let (dx, dy) = (uc.0 - bc.0, uc.1 - bc.1);
        let min_shift = p.min_shift_share * w.min(h);
        if dx.abs() <= min_shift && dy.abs() <= min_shift {
            continue;
        }
        let moved = BBox::new(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy);
        n.bbox = BBox::new(
            moved.x1.min(u.x1 - 2.0),
            moved.y1.min(u.y1 - 2.0),
            moved.x2.max(u.x2 + 2.0),
            moved.y2.max(u.y2 + 2.0),
        );
        out.nodes_moved += 1;
    }
    let node_box = |id: &str, nodes: &[glassrip_vision::board::BoardNode]| {
        nodes.iter().find(|n| n.local_id == id).map(|n| n.bbox)
    };
    let nodes = out.board.nodes.clone();
    // A span inside a node's (re-anchored) box is that node's text, never a label;
    // a span that merely shares a word with a node nearby stays a label candidate.
    let free: Vec<&Span> = spans
        .iter()
        .filter(|x| !nodes.iter().any(|n| inside(center(&x.bbox), &n.bbox)))
        .collect();
    for e in &mut out.board.edges {
        let label = clean_label(&e.label);
        if label.is_empty() {
            continue;
        }
        let (Some(s), Some(d)) = (node_box(&e.src, &nodes), node_box(&e.dst, &nodes)) else {
            continue;
        };
        let (sc, dc) = (center(&s), center(&d));
        let mid = ((sc.0 + dc.0) / 2.0, (sc.1 + dc.1) / 2.0);
        // Near the reader's label box when there is one, else in the corridor
        // between the two nodes (closest to their midpoint).
        let (window, anchor, cands) = match e.label_bbox {
            Some(l) if l.is_well_formed() => {
                let pad = l.width().max(l.height());
                (
                    BBox::new(l.x1 - pad, l.y1 - pad, l.x2 + pad, l.y2 + pad),
                    center(&l),
                    free.clone(),
                )
            }
            _ => {
                let across = (0.5 * s.height().max(d.height())).max(20.0);
                let corridor: Vec<&Span> = free
                    .iter()
                    .copied()
                    .filter(|x| segment_distance(center(&x.bbox), sc, dc) <= across)
                    .collect();
                (
                    BBox::new(
                        s.x1.min(d.x1) - 20.0,
                        s.y1.min(d.y1) - 20.0,
                        s.x2.max(d.x2) + 20.0,
                        s.y2.max(d.y2) + 20.0,
                    ),
                    mid,
                    corridor,
                )
            }
        };
        // Label words on one line or two stacked lines.
        let reach = (window.width(), window.height() / 2.0);
        if let Some((u, cover)) = locate(&cands, &label, &window, anchor, reach, p.word_similarity)
        {
            if cover >= p.min_text_cover {
                e.label_bbox = Some(u);
                out.labels_found += 1;
            }
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use glassrip_vision::board::{BoardEdge, BoardNode, EdgeStyle};

    fn node(id: &str, text: &str, b: BBox) -> BoardNode {
        BoardNode {
            local_id: id.into(),
            text: text.into(),
            bbox: b,
            conf: 0.9,
        }
    }

    fn anchor(text: &str, x1: f64, y1: f64, x2: f64, y2: f64) -> TextAnchor {
        TextAnchor {
            text: text.into(),
            bbox: BBox::new(x1, y1, x2, y2),
        }
    }

    /// The fixtures below give text-core boxes: no unclip to undo.
    fn tight() -> OcrAnchorParams {
        OcrAnchorParams {
            ocr_unclip_ratio: 0.0,
            ..OcrAnchorParams::default()
        }
    }

    #[test]
    fn detector_boxes_shrink_back_to_their_text_core() {
        // A 100 x 10 text core grown by d = 100 * 10 * 1.5 / 220 on each side.
        let d = 1500.0 / 220.0;
        let det = BBox::new(50.0 - d, 20.0 - d, 150.0 + d, 30.0 + d);
        let g = glyph_box(&det, 1.5);
        for (got, want) in [(g.x1, 50.0), (g.y1, 20.0), (g.x2, 150.0), (g.y2, 30.0)] {
            assert!((got - want).abs() < 1e-6, "{g:?}");
        }
        assert_eq!(glyph_box(&det, 0.0), det);
        // The shrunk label box leaves an arrowhead drawn just above the text
        // unmasked; the detector's box would have covered it.
        let label = [anchor("GRPC", det.x1, det.y1, det.x2, det.y2)];
        let r = reanchor(&board(vec![], vec![]), &label, &OcrAnchorParams::default());
        let head_tip = (100.0, 15.0);
        let m = r.text_boxes[0];
        assert!(head_tip.1 < m.y1, "{m:?}");
        assert!(det.y1 < head_tip.1);
    }

    fn board(nodes: Vec<BoardNode>, edges: Vec<BoardEdge>) -> ValidatedBoard {
        ValidatedBoard {
            nodes,
            edges,
            stickies: vec![],
            owner_tags: vec![],
            other_visible_text: vec![],
            confidence: 0.9,
            chrome_rejected: vec![],
            issues: vec![],
            needs_reclassification: false,
        }
    }

    #[test]
    fn a_box_read_far_off_its_text_moves_onto_it() {
        // Three text lines centered at y 239; the reader's box is centered 34 px
        // higher.
        let b = board(
            vec![node(
                "n1",
                "Ledger Sync Service",
                BBox::new(100.0, 155.0, 300.0, 255.0),
            )],
            vec![],
        );
        let ocr = vec![
            anchor("Ledger", 170.0, 215.0, 230.0, 228.0),
            anchor("Sync", 180.0, 232.0, 220.0, 245.0),
            anchor("Service", 168.0, 250.0, 232.0, 263.0),
            anchor("Other box", 600.0, 230.0, 680.0, 245.0),
        ];
        let r = reanchor(&b, &ocr, &tight());
        assert_eq!(r.nodes_moved, 1);
        let m = r.board.nodes[0].bbox;
        let c = center(&m);
        assert!((c.1 - 239.0).abs() < 1.0, "{m:?}");
        assert!((m.height() - 100.0).abs() < 1.0, "size kept: {m:?}");
        assert_eq!(r.text_boxes.len(), 4);
    }

    #[test]
    fn small_offsets_unrelated_text_and_other_boxes_leave_the_box() {
        let b = board(
            vec![
                node(
                    "n1",
                    "Ledger Sync Service",
                    BBox::new(100.0, 196.0, 300.0, 296.0),
                ),
                node("n2", "Queue", BBox::new(400.0, 200.0, 500.0, 260.0)),
            ],
            vec![],
        );
        let ocr = vec![
            // n1's text, 4 px off: below the shift threshold.
            anchor("Ledger", 170.0, 215.0, 230.0, 228.0),
            anchor("Sync", 180.0, 232.0, 220.0, 245.0),
            anchor("Service", 168.0, 250.0, 232.0, 263.0),
            // Text near n2 that is not n2's.
            anchor("Retry policy", 380.0, 300.0, 470.0, 315.0),
            // n2's word, but outside its search window.
            anchor("Queue", 900.0, 600.0, 950.0, 615.0),
        ];
        let r = reanchor(&b, &ocr, &tight());
        assert_eq!(r.nodes_moved, 0);
        assert_eq!(r.board.nodes, b.nodes);
    }

    #[test]
    fn a_single_short_word_of_a_long_title_is_not_enough() {
        let b = board(
            vec![node(
                "n1",
                "Customer Onboarding Workflow Service",
                BBox::new(100.0, 100.0, 300.0, 200.0),
            )],
            vec![],
        );
        // Only "Service" found, far below the reader's center: 7 of 34 characters.
        let ocr = vec![anchor("Service", 170.0, 250.0, 230.0, 262.0)];
        let r = reanchor(&b, &ocr, &tight());
        assert_eq!(r.nodes_moved, 0);
    }

    #[test]
    fn edge_labels_take_their_ocr_box() {
        let edge = |label_bbox| BoardEdge {
            src: "n1".into(),
            dst: "n2".into(),
            label: "gRPC".into(),
            label_bbox,
            style: EdgeStyle::Solid,
            conf: 0.9,
        };
        let nodes = vec![
            node("n1", "Alpha", BBox::new(100.0, 100.0, 200.0, 150.0)),
            node("n2", "Beta", BBox::new(100.0, 300.0, 200.0, 350.0)),
        ];
        let ocr = vec![
            anchor("Alpha", 130.0, 118.0, 170.0, 132.0),
            anchor("Beta", 132.0, 318.0, 168.0, 332.0),
            anchor("GRPC", 155.0, 212.0, 190.0, 224.0),
        ];
        // Reader label box 30 px off, and no label box at all.
        for lb in [Some(BBox::new(150.0, 180.0, 185.0, 192.0)), None] {
            let r = reanchor(&board(nodes.clone(), vec![edge(lb)]), &ocr, &tight());
            assert_eq!(r.labels_found, 1, "{lb:?}");
            assert_eq!(
                r.board.edges[0].label_bbox,
                Some(BBox::new(155.0, 212.0, 190.0, 224.0))
            );
        }
    }

    #[test]
    fn disabled_or_without_ocr_nothing_changes() {
        let b = board(
            vec![node("n1", "Ledger", BBox::new(100.0, 155.0, 300.0, 255.0))],
            vec![],
        );
        let ocr = vec![anchor("Ledger", 170.0, 235.0, 230.0, 248.0)];
        let off = OcrAnchorParams {
            enabled: false,
            ..OcrAnchorParams::default()
        };
        assert_eq!(reanchor(&b, &ocr, &off).board, b);
        assert_eq!(reanchor(&b, &[], &OcrAnchorParams::default()).board, b);
    }

    #[test]
    fn two_boxes_with_the_same_word_never_pool_their_text() {
        // Two neighbouring boxes both read "Service", each text centered in its
        // box. Pooling the spans would pull the first box halfway to the second.
        let a = BBox::new(100.0, 100.0, 260.0, 160.0);
        let b = BBox::new(280.0, 100.0, 400.0, 160.0);
        let board_ = board(
            vec![node("n1", "Service", a), node("n2", "Service", b)],
            vec![],
        );
        let ocr = vec![
            anchor("Service", 155.0, 123.0, 205.0, 137.0),
            anchor("Service", 315.0, 123.0, 365.0, 137.0),
        ];
        let r = reanchor(&board_, &ocr, &tight());
        assert_eq!(r.nodes_moved, 0, "{:?}", r.board.nodes);
        assert_eq!(r.board.nodes[0].bbox, a);
        assert_eq!(r.board.nodes[1].bbox, b);
    }

    #[test]
    fn text_on_another_box_is_not_this_box_s_anchor() {
        // "Queue" is read only inside the "Orders" box (a note written on it): the
        // "Queue" box does not jump onto another node.
        let q = BBox::new(100.0, 100.0, 200.0, 150.0);
        let o = BBox::new(230.0, 100.0, 330.0, 150.0);
        let board_ = board(
            vec![node("n1", "Queue", q), node("n2", "Orders", o)],
            vec![],
        );
        let ocr = vec![anchor("Queue", 260.0, 118.0, 300.0, 132.0)];
        let r = reanchor(&board_, &ocr, &tight());
        assert_eq!(r.nodes_moved, 0);
        assert_eq!(r.board.nodes[0].bbox, q);
    }

    #[test]
    fn a_label_without_a_reader_box_is_found_on_its_own_connector() {
        // Diagonal connector n1 -> n2 labelled "yes" a quarter of the way along; a
        // second "yes" (another connector's) sits nearer the midpoint but off this
        // connector's corridor.
        let n1 = BBox::new(100.0, 100.0, 200.0, 150.0);
        let n2 = BBox::new(500.0, 300.0, 600.0, 350.0);
        let edge = BoardEdge {
            src: "n1".into(),
            dst: "n2".into(),
            label: "yes".into(),
            label_bbox: None,
            style: EdgeStyle::Solid,
            conf: 0.9,
        };
        let board_ = board(
            vec![node("n1", "Start", n1), node("n2", "Finish", n2)],
            vec![edge],
        );
        let ocr = vec![
            anchor("yes", 235.0, 168.0, 265.0, 182.0),
            anchor("yes", 315.0, 133.0, 345.0, 147.0),
        ];
        let r = reanchor(&board_, &ocr, &tight());
        assert_eq!(r.labels_found, 1);
        assert_eq!(
            r.board.edges[0].label_bbox,
            Some(BBox::new(235.0, 168.0, 265.0, 182.0))
        );
    }

    #[test]
    fn a_label_sharing_a_word_with_a_nearby_box_is_still_found() {
        // Node "Say yes" sits right beside the connector's "yes" label: the label
        // span lies in the node's search window but outside its box.
        let n1 = BBox::new(100.0, 100.0, 200.0, 150.0);
        let n2 = BBox::new(500.0, 100.0, 600.0, 150.0);
        let n3 = BBox::new(260.0, 20.0, 360.0, 70.0);
        let edge = BoardEdge {
            src: "n1".into(),
            dst: "n2".into(),
            label: "yes".into(),
            label_bbox: None,
            style: EdgeStyle::Solid,
            conf: 0.9,
        };
        let board_ = board(
            vec![
                node("n1", "Start", n1),
                node("n2", "Finish", n2),
                node("n3", "Say yes", n3),
            ],
            vec![edge],
        );
        let ocr = vec![
            anchor("Say", 275.0, 38.0, 305.0, 52.0),
            anchor("yes", 310.0, 38.0, 340.0, 52.0),
            anchor("yes", 335.0, 108.0, 365.0, 122.0),
        ];
        let r = reanchor(&board_, &ocr, &tight());
        assert_eq!(r.labels_found, 1);
        assert_eq!(
            r.board.edges[0].label_bbox,
            Some(BBox::new(335.0, 108.0, 365.0, 122.0))
        );
    }
}
