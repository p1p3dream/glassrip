//! Owner-tag geometry checked against OCR.
//!
//! The reader's boxes can sit far off the drawn shapes: laid out on a made-up grid,
//! collapsed into one strip, or one box thrown elsewhere. Owner anchoring compares a
//! tag's box with the node boxes around it, so a misplaced box ties a tag to the wrong
//! node. OCR boxes are pixel-accurate, so every node and name tag is looked up in the
//! keyframe's OCR text:
//!
//! - **Located.** A node's text, or a span naming the tag's participant, found within
//!   `search_share` box sizes of the reader's box on each side (and at least
//!   `min_window_lines` OCR line heights and `min_displacement_share` of the canvas
//!   diagonal) confirms the element where the reader put it: jitter of a consistent
//!   reading is not a displacement. Otherwise text that occurs exactly once on the canvas,
//!   does not sit on another element's box, and belongs to no other element of the
//!   reading with the same text, locates it there: the reader displaced it. Tag spans
//!   go to the nearest tag of the same person first; a person's only tag left takes
//!   the person's only span left.
//! - **Reliable reading.** The reader's boxes are one consistent picture, so they are
//!   kept; only displaced elements move onto their OCR text (keeping their size, grown
//!   to contain the text).
//! - **Unreliable reading.** When at least `displaced_share` of the located elements
//!   (nodes, stickies, tags; at least `min_located` of them) are displaced, as when the
//!   reader collapsed its boxes into one strip or laid them out on a made-up grid, the
//!   reader's boxes are not geometry: every located element moves onto its OCR text
//!   and the others are left out of owner geometry.
//!
//! Without OCR the reader's boxes are used as they are.
//!
//! Every OCR span that names a participant is also reported ([`NameSpan`]), whether or
//! not the reader emitted a tag there: owner sightings start from what OCR read, and a
//! span inside another element that mentions the name (a sticky saying "ask Avery")
//! is flagged as a mention, not a tag.

use glassrip_vision::board::ValidatedBoard;
use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::TextAnchor;
use crate::ocr_anchor::{glyph_box, locate_text_groups, OcrAnchorParams};
use crate::text::normalize;

/// Owner geometry settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OwnerGeometryParams {
    /// Check owner geometry against OCR at all.
    pub enabled: bool,
    /// Text matching and search window (shared with the pixel check's re-anchoring).
    pub ocr: OcrAnchorParams,
    /// Located elements needed before the reading can be judged unreliable.
    pub min_located: usize,
    /// Share of displaced located elements that makes the reading unreliable.
    pub displaced_share: f64,
    /// Grouping reach for text lines of one element, in OCR line heights (across,
    /// along): an element's text may wrap over several lines.
    pub line_reach: (f64, f64),
    /// Smallest margin of the confirming window around a reader box, in OCR line
    /// heights: a box one text line tall is not displaced by a line or two.
    pub min_window_lines: f64,
    /// Smallest margin of the confirming window, as a share of the canvas diagonal.
    pub min_displacement_share: f64,
    /// Padding around an OCR name span, in OCR line heights, that gives the tag's
    /// box for owner geometry (a name tag is its text plus a margin).
    #[serde(default = "default_tag_pad_lines")]
    pub tag_pad_lines: f64,
}

fn default_tag_pad_lines() -> f64 {
    1.0
}

impl Default for OwnerGeometryParams {
    fn default() -> Self {
        Self {
            enabled: true,
            ocr: OcrAnchorParams::default(),
            min_located: 4,
            displaced_share: 0.5,
            line_reach: (4.0, 2.0),
            min_window_lines: 2.0,
            min_displacement_share: 0.05,
            tag_pad_lines: default_tag_pad_lines(),
        }
    }
}

/// Where one element sits for owner geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placed {
    /// Box in the reading's coordinates.
    pub bbox: BBox,
    /// OCR located the element (confirmed where the reader put it, or moved).
    pub ocr: bool,
}

/// Owner geometry of one keyframe.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameGeometry {
    /// Per reading node (same order as `board.nodes`); `None` when left out.
    pub nodes: Vec<Option<Placed>>,
    /// Per tag (same order as the tags passed in); `None` when left out.
    pub tags: Vec<Option<Placed>>,
    /// The reading's own boxes were judged unreliable.
    pub unreliable: bool,
    /// Per tag: the OCR name span (index into `name_spans`) that located it.
    pub tag_spans: Vec<Option<usize>>,
    /// OCR spans naming a participant, in OCR order.
    pub name_spans: Vec<NameSpan>,
    /// Median OCR line height (0 without OCR).
    pub line: f64,
}

/// An OCR span that names a participant.
#[derive(Debug, Clone, PartialEq)]
pub struct NameSpan {
    /// Participant id.
    pub person: String,
    /// Text as read.
    pub text: String,
    /// Glyph box.
    pub glyph: BBox,
    /// The span lies in another element whose text mentions the name: not a tag.
    pub mention: bool,
}

/// The box of a name tag read by OCR: the glyph box padded by `pad_lines` line
/// heights on every side.
pub fn tag_box(glyph: &BBox, line: f64, pad_lines: f64) -> BBox {
    let m = (line * pad_lines).max(0.0);
    BBox::new(glyph.x1 - m, glyph.y1 - m, glyph.x2 + m, glyph.y2 + m)
}

/// True when some word or pair of adjacent words of `text` names `pid`.
fn mentions(text: &str, pid: &str, names: &dyn Fn(&str) -> Option<String>) -> bool {
    let norm = normalize(text);
    let words: Vec<&str> = norm.split_whitespace().collect();
    words.iter().any(|w| names(w).as_deref() == Some(pid))
        || words
            .windows(2)
            .any(|w| names(&format!("{} {}", w[0], w[1])).as_deref() == Some(pid))
}

/// One name tag to place: its reader box and the participant it names.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TagIn<'a> {
    /// Reader box.
    pub bbox: BBox,
    /// Participant id, when the name resolved.
    pub person: Option<&'a str>,
}

/// An element OCR located: its text box, and whether it lies in the reader window.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Found {
    text: BBox,
    local: bool,
}

fn center(b: &BBox) -> (f64, f64) {
    ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
}

fn in_window(p: (f64, f64), b: &BBox, share: f64) -> bool {
    in_margin(p, b, b.width() * share, b.height() * share)
}

fn in_margin(p: (f64, f64), b: &BBox, dx: f64, dy: f64) -> bool {
    p.0 >= b.x1 - dx && p.0 <= b.x2 + dx && p.1 >= b.y1 - dy && p.1 <= b.y2 + dy
}

/// The reader box moved so its center is `c`, grown to contain `text`.
fn moved_onto(b: &BBox, text: &BBox) -> BBox {
    let (bc, c) = (center(b), center(text));
    let (dx, dy) = (c.0 - bc.0, c.1 - bc.1);
    BBox::new(
        (b.x1 + dx).min(text.x1 - 2.0),
        (b.y1 + dy).min(text.y1 - 2.0),
        (b.x2 + dx).max(text.x2 + 2.0),
        (b.y2 + dy).max(text.y2 + 2.0),
    )
}

/// Place the nodes and tags of one reading for owner geometry. `ocr` shares the
/// reading's coordinates; `names(text)` is the participant an OCR text names;
/// `diagonal` is the canvas diagonal (0 when unknown).
pub fn place(
    board: &ValidatedBoard,
    tags: &[TagIn<'_>],
    ocr: &[TextAnchor],
    names: &dyn Fn(&str) -> Option<String>,
    diagonal: f64,
    p: &OwnerGeometryParams,
) -> FrameGeometry {
    let as_read = |b: &BBox| {
        Some(Placed {
            bbox: *b,
            ocr: false,
        })
    };
    if !p.enabled || ocr.is_empty() {
        return FrameGeometry {
            nodes: board.nodes.iter().map(|n| as_read(&n.bbox)).collect(),
            tags: tags.iter().map(|t| as_read(&t.bbox)).collect(),
            unreliable: false,
            tag_spans: vec![None; tags.len()],
            name_spans: Vec::new(),
            line: 0.0,
        };
    }
    // Index-aligned with `ocr`.
    let glyphs: Vec<BBox> = ocr
        .iter()
        .map(|a| glyph_box(&a.bbox, p.ocr.ocr_unclip_ratio))
        .collect();
    let line = {
        let mut h: Vec<f64> = ocr
            .iter()
            .zip(&glyphs)
            .filter(|(a, _)| a.bbox.is_well_formed())
            .map(|(_, g)| g.height())
            .collect();
        h.sort_by(f64::total_cmp);
        h.get(h.len() / 2).copied().unwrap_or(0.0)
    };
    // The window that confirms an element where the reader put it.
    let min_margin = (p.min_window_lines * line).max(p.min_displacement_share * diagonal.max(0.0));
    let confirms = |c: (f64, f64), b: &BBox| {
        let s = p.ocr.search_share;
        in_margin(
            c,
            b,
            (b.width() * s).max(min_margin),
            (b.height() * s).max(min_margin),
        )
    };
    // Every reading element with text: (text, box), nodes first.
    let elements: Vec<(&str, BBox)> = board
        .nodes
        .iter()
        .map(|n| (n.text.as_str(), n.bbox))
        .chain(board.stickies.iter().map(|s| (s.text.as_str(), s.bbox)))
        .chain(
            board
                .other_visible_text
                .iter()
                .map(|t| (t.text.as_str(), t.bbox)),
        )
        .collect();
    let locate = |i: usize| -> Option<Found> {
        let (text, b) = elements[i];
        if !b.is_well_formed() || text.chars().filter(|c| c.is_alphanumeric()).count() < 3 {
            return None;
        }
        let reach = (
            b.width().max(p.line_reach.0 * line),
            b.height().max(p.line_reach.1 * line),
        );
        let groups = locate_text_groups(ocr, text, center(&b), reach, &p.ocr);
        if let Some((u, _)) = groups.iter().find(|(u, _)| confirms(center(u), &b)) {
            return Some(Found {
                text: *u,
                local: true,
            });
        }
        // Displaced: only text that occurs once, belongs to no other element with the
        // same text, and does not sit on another element's box says where.
        let norm = normalize(text);
        let same_text = elements
            .iter()
            .enumerate()
            .any(|(j, e)| j != i && normalize(e.0) == norm);
        match groups.as_slice() {
            [(u, _)] if !same_text => {
                let c = center(u);
                let on_other = elements
                    .iter()
                    .enumerate()
                    .any(|(j, e)| j != i && in_window(c, &e.1, 0.0));
                (!on_other).then_some(Found {
                    text: *u,
                    local: false,
                })
            }
            _ => None,
        }
    };
    let found: Vec<Option<Found>> = (0..elements.len()).map(locate).collect();

    // Tags: OCR spans naming each tag's person, nearest first, each span once.
    let name_spans: Vec<(String, BBox, &str)> = ocr
        .iter()
        .zip(&glyphs)
        .filter(|(a, _)| a.bbox.is_well_formed())
        .filter_map(|(a, g)| names(&a.text).map(|pid| (pid, *g, a.text.as_str())))
        .collect();
    let mut tag_found: Vec<Option<Found>> = vec![None; tags.len()];
    let mut tag_spans: Vec<Option<usize>> = vec![None; tags.len()];
    let mut span_used = vec![false; name_spans.len()];
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (ti, t) in tags.iter().enumerate() {
        let Some(pid) = t.person else { continue };
        for (si, (spid, sb, _)) in name_spans.iter().enumerate() {
            let c = center(sb);
            if spid == pid && confirms(c, &t.bbox) {
                let tc = center(&t.bbox);
                pairs.push(((c.0 - tc.0).hypot(c.1 - tc.1), ti, si));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, ti, si) in pairs {
        if tag_found[ti].is_none() && !span_used[si] {
            tag_found[ti] = Some(Found {
                text: name_spans[si].1,
                local: true,
            });
            tag_spans[ti] = Some(si);
            span_used[si] = true;
        }
    }
    for (ti, t) in tags.iter().enumerate() {
        let Some(pid) = t.person else { continue };
        if tag_found[ti].is_some() {
            continue;
        }
        let tags_left = tags
            .iter()
            .enumerate()
            .filter(|(j, u)| u.person == Some(pid) && tag_found[*j].is_none())
            .count();
        let spans_left: Vec<usize> = name_spans
            .iter()
            .enumerate()
            .filter(|(si, (spid, _, _))| spid == pid && !span_used[*si])
            .map(|(si, _)| si)
            .collect();
        // A name inside another element's box is a mention in its text, unless that
        // element is itself read as the name.
        let on_other = |si: usize| {
            let c = center(&name_spans[si].1);
            elements
                .iter()
                .any(|e| in_window(c, &e.1, 0.0) && names(e.0).as_deref() != Some(pid))
        };
        if tags_left == 1 && spans_left.len() == 1 && !on_other(spans_left[0]) {
            tag_found[ti] = Some(Found {
                text: name_spans[spans_left[0]].1,
                local: false,
            });
            tag_spans[ti] = Some(spans_left[0]);
            span_used[spans_left[0]] = true;
        }
    }

    // Reading check over every located element.
    let located: Vec<bool> = found
        .iter()
        .chain(&tag_found)
        .filter_map(|f| f.map(|f| f.local))
        .collect();
    let displaced = located.iter().filter(|local| !**local).count();
    let unreliable = located.len() >= p.min_located.max(1)
        && displaced as f64 >= p.displaced_share * located.len() as f64;

    // OCR reads another word (3+ characters, not a name of `pid`) of `text` within two
    // lines of `c`: the name is part of that text as written on the canvas.
    let words_near = |c: (f64, f64), text: &str, pid: &str| {
        let norm = normalize(text);
        let words: Vec<&str> = norm
            .split_whitespace()
            .filter(|w| w.chars().count() >= 3 && names(w).as_deref() != Some(pid))
            .collect();
        ocr.iter().zip(&glyphs).any(|(a, g)| {
            let dx = (g.x1 - c.0).max(c.0 - g.x2).max(0.0);
            let dy = (g.y1 - c.1).max(c.1 - g.y2).max(0.0);
            dx <= 2.0 * line
                && dy <= 2.0 * line
                && normalize(&a.text)
                    .split_whitespace()
                    .any(|w| words.contains(&w))
        })
    };
    // Every name span, flagged when it lies in another element whose text mentions the
    // name without being the name: inside the element's located text when OCR found
    // it, else inside its reader box with more of its words read around the name.
    let spans_out: Vec<NameSpan> = name_spans
        .iter()
        .map(|(pid, g, text)| {
            let c = center(g);
            // Where OCR located the element, its text says where it is; the reader's
            // box only for elements OCR did not locate.
            let mention = elements.iter().zip(&found).any(|(e, f)| {
                let inside = match f {
                    Some(f) => in_margin(c, &f.text, line, line),
                    None => in_window(c, &e.1, 0.0) && words_near(c, e.0, pid),
                };
                inside && names(e.0).as_deref() != Some(pid) && mentions(e.0, pid, names)
            });
            NameSpan {
                person: pid.clone(),
                text: (*text).to_string(),
                glyph: *g,
                mention,
            }
        })
        .collect();

    let placed = |b: &BBox, f: &Option<Found>| match f {
        Some(f) if unreliable || !f.local => Some(Placed {
            bbox: moved_onto(b, &f.text),
            ocr: true,
        }),
        Some(_) => Some(Placed {
            bbox: *b,
            ocr: true,
        }),
        None if unreliable => None,
        None => as_read(b),
    };
    FrameGeometry {
        nodes: board
            .nodes
            .iter()
            .zip(&found)
            .map(|(n, f)| placed(&n.bbox, f))
            .collect(),
        tags: tags
            .iter()
            .zip(&tag_found)
            .map(|(t, f)| placed(&t.bbox, f))
            .collect(),
        unreliable,
        tag_spans,
        name_spans: spans_out,
        line,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use glassrip_vision::board::{BoardNode, Sticky, StickyColor};

    fn b(x1: f64, y1: f64, x2: f64, y2: f64) -> BBox {
        BBox::new(x1, y1, x2, y2)
    }

    fn node(id: &str, text: &str, bb: BBox) -> BoardNode {
        BoardNode {
            local_id: id.into(),
            text: text.into(),
            bbox: bb,
            conf: 0.9,
        }
    }

    fn board(nodes: Vec<BoardNode>) -> ValidatedBoard {
        ValidatedBoard {
            nodes,
            edges: vec![],
            stickies: vec![],
            owner_tags: vec![],
            other_visible_text: vec![],
            confidence: 0.9,
            chrome_rejected: vec![],
            issues: vec![],
            needs_reclassification: false,
        }
    }

    fn anchor(text: &str, bb: BBox) -> TextAnchor {
        TextAnchor {
            text: text.into(),
            bbox: bb,
        }
    }

    fn names(t: &str) -> Option<String> {
        (normalize(t) == "avery").then(|| "p-avery".to_string())
    }

    fn tight() -> OwnerGeometryParams {
        OwnerGeometryParams {
            ocr: OcrAnchorParams {
                ocr_unclip_ratio: 0.0,
                ..OcrAnchorParams::default()
            },
            ..OwnerGeometryParams::default()
        }
    }

    #[test]
    fn without_ocr_the_reading_is_used_as_is() {
        let r = board(vec![node("n1", "Ingest Gateway", b(0.0, 0.0, 100.0, 50.0))]);
        let tags = [TagIn {
            bbox: b(10.0, 60.0, 40.0, 80.0),
            person: Some("p-avery"),
        }];
        let g = place(&r, &tags, &[], &names, 0.0, &tight());
        assert!(!g.unreliable);
        assert_eq!(g.nodes[0].unwrap().bbox, r.nodes[0].bbox);
        assert!(!g.tags[0].unwrap().ocr);
    }

    #[test]
    fn collapsed_boxes_are_placed_on_their_text_and_the_rest_left_out() {
        // Four nodes on a 2 x 2 layout plus one OCR did not read, all read as one
        // strip along the top.
        let r = board(vec![
            node("n1", "Ingest Gateway", b(0.0, 0.0, 100.0, 40.0)),
            node("n2", "Ledger Store", b(110.0, 0.0, 210.0, 40.0)),
            node("n3", "Report Builder", b(220.0, 0.0, 320.0, 40.0)),
            node("n4", "Queue", b(330.0, 0.0, 430.0, 40.0)),
            node("n5", "Unread", b(440.0, 0.0, 540.0, 40.0)),
        ]);
        let ocr = [
            anchor("Ingest Gateway", b(100.0, 100.0, 200.0, 120.0)),
            anchor("Ledger Store", b(700.0, 100.0, 800.0, 120.0)),
            anchor("Report Builder", b(100.0, 600.0, 200.0, 620.0)),
            anchor("Queue", b(700.0, 600.0, 760.0, 620.0)),
            anchor("Avery", b(720.0, 660.0, 770.0, 680.0)),
        ];
        let tags = [
            TagIn {
                bbox: b(500.0, 0.0, 540.0, 30.0),
                person: Some("p-avery"),
            },
            TagIn {
                bbox: b(0.0, 50.0, 40.0, 80.0),
                person: None,
            },
        ];
        let g = place(&r, &tags, &ocr, &names, 0.0, &tight());
        assert!(g.unreliable);
        let queue = g.nodes[3].unwrap();
        assert!(queue.ocr && queue.bbox.x1 < 720.0 && queue.bbox.x2 > 740.0);
        assert!(queue.bbox.y1 < 610.0 && queue.bbox.y2 > 610.0);
        // No OCR for "Unread": its strip box is not geometry.
        assert!(g.nodes[4].is_none());
        let tag = g.tags[0].unwrap();
        assert!(tag.ocr && tag.bbox.y1 > 600.0);
        assert!(g.tags[1].is_none());
    }

    #[test]
    fn a_reliable_reading_keeps_its_boxes_and_moves_only_a_displaced_one() {
        let r = board(vec![
            node("n1", "Ingest Gateway", b(0.0, 0.0, 100.0, 40.0)),
            node("n2", "Ledger Store", b(300.0, 0.0, 400.0, 40.0)),
            node("n3", "Report Builder", b(600.0, 0.0, 700.0, 40.0)),
            node("n4", "Queue", b(900.0, 0.0, 1000.0, 40.0)),
        ]);
        let ocr = [
            // Jittered by 20 px: confirmed, kept.
            anchor("Ingest Gateway", b(30.0, 10.0, 110.0, 30.0)),
            anchor("Ledger Store", b(300.0, 10.0, 380.0, 30.0)),
            anchor("Report Builder", b(610.0, 10.0, 690.0, 30.0)),
            // Thrown far off: moved.
            anchor("Queue", b(900.0, 400.0, 960.0, 420.0)),
        ];
        let g = place(&r, &[], &ocr, &names, 0.0, &tight());
        assert!(!g.unreliable);
        let n1 = g.nodes[0].unwrap();
        assert!(n1.ocr && n1.bbox == r.nodes[0].bbox);
        let q = g.nodes[3].unwrap();
        assert!(q.ocr && q.bbox.y1 < 410.0 && q.bbox.y2 > 410.0);
    }

    #[test]
    fn a_shift_under_the_canvas_share_is_jitter() {
        let r = board(vec![node("n1", "Queue", b(0.0, 0.0, 100.0, 20.0))]);
        // 60 px below a 20 px box: displaced on its own scale, jitter on a canvas
        // with a 1600 px diagonal (80 px).
        let ocr = [anchor("Queue", b(20.0, 70.0, 80.0, 90.0))];
        let n1 = place(&r, &[], &ocr, &names, 1600.0, &tight()).nodes[0].unwrap();
        assert!(n1.ocr && n1.bbox == r.nodes[0].bbox);
        let n1 = place(&r, &[], &ocr, &names, 0.0, &tight()).nodes[0].unwrap();
        assert!(n1.ocr && n1.bbox != r.nodes[0].bbox);
    }

    #[test]
    fn a_tag_name_near_its_box_confirms_it_far_off_moves_it() {
        let r = board(vec![]);
        let tags = [TagIn {
            bbox: b(100.0, 100.0, 200.0, 200.0),
            person: Some("p-avery"),
        }];
        let ocr = [anchor("Avery", b(210.0, 150.0, 270.0, 170.0))];
        let t = place(&r, &tags, &ocr, &names, 0.0, &tight()).tags[0].unwrap();
        assert!(t.ocr && t.bbox == tags[0].bbox);
        let ocr = [anchor("Avery", b(600.0, 500.0, 660.0, 520.0))];
        let t = place(&r, &tags, &ocr, &names, 0.0, &tight()).tags[0].unwrap();
        assert!(t.ocr && (t.bbox.x1 - 580.0).abs() < 1.0);
    }

    #[test]
    fn a_name_mentioned_in_another_element_is_not_the_tag() {
        let mut r = board(vec![]);
        r.stickies.push(Sticky {
            text: "ask Avery about retention".into(),
            color: StickyColor::Yellow,
            bbox: b(580.0, 480.0, 760.0, 540.0),
        });
        let tags = [TagIn {
            bbox: b(100.0, 100.0, 200.0, 200.0),
            person: Some("p-avery"),
        }];
        let ocr = [anchor("Avery", b(600.0, 500.0, 660.0, 520.0))];
        let t = place(&r, &tags, &ocr, &names, 0.0, &tight()).tags[0].unwrap();
        assert!(!t.ocr && t.bbox == tags[0].bbox);
    }

    #[test]
    fn displaced_text_that_is_ambiguous_or_on_another_element_does_not_move_a_box() {
        // Read twice elsewhere.
        let r = board(vec![node("n1", "Queue", b(0.0, 0.0, 100.0, 40.0))]);
        let ocr = [
            anchor("Queue", b(600.0, 400.0, 660.0, 420.0)),
            anchor("Queue", b(900.0, 700.0, 960.0, 720.0)),
        ];
        let g = place(&r, &[], &ocr, &names, 0.0, &tight());
        let n1 = g.nodes[0].unwrap();
        assert!(!n1.ocr && n1.bbox == r.nodes[0].bbox);
        // Once, but inside a sticky whose text mentions it.
        let mut r = board(vec![node("n1", "Ledger Store", b(0.0, 0.0, 100.0, 40.0))]);
        r.stickies.push(Sticky {
            text: "is the ledger store durable?".into(),
            color: StickyColor::Yellow,
            bbox: b(580.0, 380.0, 700.0, 440.0),
        });
        let ocr = [anchor("ledger store", b(600.0, 400.0, 680.0, 420.0))];
        let n1 = place(&r, &[], &ocr, &names, 0.0, &tight()).nodes[0].unwrap();
        assert!(!n1.ocr && n1.bbox == r.nodes[0].bbox);
    }

    #[test]
    fn a_node_text_wrapped_over_two_lines_is_one_occurrence() {
        let r = board(vec![node(
            "n1",
            "Report Builder Service",
            b(0.0, 0.0, 120.0, 16.0),
        )]);
        let ocr = [
            anchor("Report Builder", b(500.0, 300.0, 600.0, 316.0)),
            anchor("Service", b(510.0, 322.0, 580.0, 338.0)),
        ];
        let n1 = place(&r, &[], &ocr, &names, 0.0, &tight()).nodes[0].unwrap();
        assert!(n1.ocr && n1.bbox.y1 < 305.0 && n1.bbox.y2 > 335.0);
    }
}
