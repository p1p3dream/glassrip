//! Layout: turns a board state and validated notes into positioned shapes.
//!
//! All geometry is computed here; the SVG template only draws. Node positions
//! come from the board's own canvas bboxes (scaled, snapped into rows and
//! columns, and pushed apart until cards keep their gaps); when positions are
//! missing, a Sugiyama layered layout is used instead.

use std::collections::{BTreeMap, BTreeSet};

use glassrip_notes::board::{
    center, first_seen, target_text, BoardExt, BoardStateItem, EdgeOrientation, EdgeStyle,
    OwnerAssignment, OwnerTarget, StickyKind,
};
use glassrip_notes::notes::MeetingNotes;
use glassrip_notes::text::{mmss, sanitize_dashes};
use serde::Serialize;

use crate::facts::{deferred_nodes, derive_grids, focus_node, short_name};
use crate::route::{Pt, Router, CLEAR};
use crate::style::{role_of, text_width, wrap, wrap_lines, Role};

/// Canvas width unless the board needs more.
pub const MIN_WIDTH: f64 = 1640.0;
const MARGIN: f64 = 40.0;
const CARD_W: f64 = 220.0;
const CARD_H: f64 = 132.0;
const GAP_X: f64 = 110.0;
const GAP_Y: f64 = 130.0;
const ARCH_TOP: f64 = 248.0;
const ZONE_PAD_X: f64 = 32.0;
const ZONE_PAD_TOP: f64 = 100.0;
const ZONE_PAD_BOTTOM: f64 = 40.0;
const PILL_H: f64 = 24.0;
/// Bottom of the title and legend band; nothing else is placed above it.
const LEGEND_BOTTOM: f64 = 124.0;
/// Fractions along a segment that labels and owner pills slide to when the
/// middle is taken.
const SLIDE: [f64; 5] = [0.5, 0.3, 0.7, 0.15, 0.85];
/// Gap between an edge and an owner pill beside it.
const PILL_GAP: f64 = 10.0;

/// Axis-aligned rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct R {
    /// Left.
    pub x: f64,
    /// Top.
    pub y: f64,
    /// Width.
    pub w: f64,
    /// Height.
    pub h: f64,
}

impl R {
    pub(crate) fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self {
            x: x.round(),
            y: y.round(),
            w: w.round(),
            h: h.round(),
        }
    }
    /// Right edge.
    pub fn right(&self) -> f64 {
        self.x + self.w
    }
    /// Bottom edge.
    pub fn bottom(&self) -> f64 {
        self.y + self.h
    }
    pub(crate) fn cx(&self) -> f64 {
        self.x + self.w / 2.0
    }
    pub(crate) fn cy(&self) -> f64 {
        self.y + self.h / 2.0
    }
    /// True when the interiors overlap (touching edges do not count).
    pub fn intersects(&self, o: &R) -> bool {
        self.x < o.right() && o.x < self.right() && self.y < o.bottom() && o.y < self.bottom()
    }
    pub(crate) fn inflate(&self, d: f64) -> R {
        R {
            x: self.x - d,
            y: self.y - d,
            w: self.w + 2.0 * d,
            h: self.h + 2.0 * d,
        }
    }
}

/// A line of text.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TextLine {
    /// Anchor x.
    pub x: f64,
    /// Baseline y.
    pub y: f64,
    /// Text.
    pub text: String,
    /// CSS class.
    pub class: String,
    /// `start` or `middle`.
    pub anchor: &'static str,
    /// Extra fill color.
    pub fill: Option<String>,
    /// Element id (lines of the annotation list, whose glyphs validation
    /// requires inside the canvas).
    pub id: Option<String>,
}

fn tl(x: f64, y: f64, text: impl Into<String>, class: &str) -> TextLine {
    TextLine {
        x: x.round(),
        y: y.round(),
        text: text.into(),
        class: class.into(),
        anchor: "start",
        fill: None,
        id: None,
    }
}

fn tc(x: f64, y: f64, text: impl Into<String>, class: &str) -> TextLine {
    TextLine {
        anchor: "middle",
        ..tl(x, y, text, class)
    }
}

/// A component card.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Card {
    /// Node id.
    pub id: String,
    /// Box.
    pub r: R,
    /// Header color.
    pub color: &'static str,
    /// Role key.
    pub role: &'static str,
    /// Header text.
    pub title: TextLine,
    /// Body lines.
    pub lines: Vec<TextLine>,
    /// Dashed border (deferred).
    pub dashed: bool,
    /// Glow (focus).
    pub glow: bool,
}

/// A role zone.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Zone {
    /// Box.
    pub r: R,
    /// Border color.
    pub color: &'static str,
    /// Role key (gradient id).
    pub role: &'static str,
    /// Title.
    pub label: TextLine,
    /// Optional badge (focus).
    pub badge: Option<Pill>,
    /// Width of the title's glyphs.
    #[serde(skip)]
    pub label_w: f64,
}

/// A rounded label (owner pill, badge, edge label).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Pill {
    /// Box.
    pub r: R,
    /// Corner radius.
    pub rx: f64,
    /// Fill.
    pub fill: String,
    /// Stroke.
    pub stroke: Option<String>,
    /// Text.
    pub text: TextLine,
}

/// A drawn edge.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EdgeArt {
    /// Path data.
    pub d: String,
    /// Arrowhead polygons (none when the direction is not established, two when
    /// the arrow points both ways).
    pub arrows: Vec<String>,
    /// Stroke color.
    pub color: &'static str,
    /// Stroke width.
    pub width: f64,
    /// Dash pattern.
    pub dash: Option<&'static str>,
    /// Label.
    pub label: Option<Pill>,
    /// Further lines of a label wrapped to fit (drawn inside its pill, under the
    /// first line).
    pub label_lines: Vec<TextLine>,
    /// Leader line `(x1, y1, x2, y2)` from a label placed away from its connector
    /// back to the connector.
    pub leader: Option<(f64, f64, f64, f64)>,
}

/// A sticky card.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StickyArt {
    /// Box.
    pub r: R,
    /// Header color.
    pub header: &'static str,
    /// Kind header text.
    pub kind: TextLine,
    /// Text lines.
    pub lines: Vec<TextLine>,
}

/// The decision banner.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Banner {
    /// Box.
    pub r: R,
    /// Label pill.
    pub label: Pill,
    /// Lines.
    pub lines: Vec<TextLine>,
}

/// A shape inside a panel.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Shape {
    /// Box.
    pub r: R,
    /// Corner radius.
    pub rx: f64,
    /// Fill.
    pub fill: &'static str,
    /// Stroke.
    pub stroke: &'static str,
    /// Stroke width.
    pub stroke_width: f64,
    /// Dash pattern.
    pub dash: Option<&'static str>,
    /// Small shadow.
    pub shadow: bool,
}

/// A group panel.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Panel {
    /// Box.
    pub r: R,
    /// Header color.
    pub color: &'static str,
    /// Header text.
    pub title: TextLine,
    /// Shapes.
    pub shapes: Vec<Shape>,
    /// Texts.
    pub texts: Vec<TextLine>,
    /// Pills (badges).
    pub pills: Vec<Pill>,
    /// Arrows (path, arrowhead).
    pub arrows: Vec<(String, String)>,
}

/// A legend entry.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LegendItem {
    /// `solid`, `dashed`, `owner`, `deferred`, `sticky`.
    pub kind: &'static str,
    /// Left offset inside the legend.
    pub x: f64,
    /// Label.
    pub label: String,
}

/// The legend.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Legend {
    /// Box.
    pub r: R,
    /// Items.
    pub items: Vec<LegendItem>,
}

/// An annotation that had no free place on the board, listed below it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Footnote {
    /// Number marker at the start of the entry (same look as on the board).
    pub marker: Pill,
    /// Wrapped text lines.
    pub lines: Vec<TextLine>,
}

/// The block below the board listing annotations that did not fit on it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Footnotes {
    /// Box.
    pub r: R,
    /// Heading.
    pub title: TextLine,
    /// Entries, top to bottom.
    pub items: Vec<Footnote>,
}

/// Geometry of a drawn edge, for validation.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteInfo {
    /// `edge <source text> to <target text>`.
    pub name: String,
    /// Source card id.
    pub src: String,
    /// Target card id.
    pub dst: String,
    /// The full polyline, from border to border (arrow tips included).
    pub points: Vec<(f64, f64)>,
    /// The router found no route around the cards: drawn by a cheap detour or
    /// straight (a warning; through another card, a failure).
    pub fallback: bool,
}

/// Everything the SVG template draws.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Scene {
    /// Canvas width.
    pub width: f64,
    /// Canvas height.
    pub height: f64,
    /// Title.
    pub title: TextLine,
    /// Subtitle.
    pub subtitle: TextLine,
    /// Legend.
    pub legend: Legend,
    /// Roles used (for gradients and filters).
    pub roles: Vec<(&'static str, &'static str)>,
    /// Zones.
    pub zones: Vec<Zone>,
    /// Cards.
    pub cards: Vec<Card>,
    /// Edges.
    pub edges: Vec<EdgeArt>,
    /// Owner pills and badges.
    pub pills: Vec<Pill>,
    /// Small annotations.
    pub notes: Vec<TextLine>,
    /// Stickies.
    pub stickies: Vec<StickyArt>,
    /// Decision banner.
    pub banner: Option<Banner>,
    /// Group panels.
    pub panels: Vec<Panel>,
    /// Numbered markers on the board for annotations listed in `footnotes`.
    pub markers: Vec<Pill>,
    /// Leader lines `(x1, y1, x2, y2)` from owner pills, notes and badges placed
    /// away from their element back to it.
    pub leaders: Vec<(f64, f64, f64, f64)>,
    /// Annotations without room on the board, listed below it.
    pub footnotes: Option<Footnotes>,
    /// Footer.
    pub footer: TextLine,
    /// Layout method (`board_positions` or `sugiyama` or `grid`).
    pub layout_method: &'static str,
    /// Boxes that must not overlap (name, box).
    #[serde(skip)]
    pub blocking: Vec<(String, R)>,
    /// Edge routes, checked against the cards and the edge labels.
    #[serde(skip)]
    pub routes: Vec<RouteInfo>,
    /// Annotations (owner pills, notes, badges, edge labels) that found no free
    /// place on the board and were moved to `footnotes`: warnings, not failures.
    #[serde(skip)]
    pub degraded: Vec<String>,
}

fn arrowhead(tip: (f64, f64), from: (f64, f64)) -> ((f64, f64), String) {
    let (dx, dy) = (tip.0 - from.0, tip.1 - from.1);
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    let (ux, uy) = (dx / len, dy / len);
    let base = (tip.0 - ux * 11.0, tip.1 - uy * 11.0);
    let (px, py) = (-uy * 6.0, ux * 6.0);
    let pts = format!(
        "{:.0},{:.0} {:.0},{:.0} {:.0},{:.0}",
        tip.0,
        tip.1,
        base.0 + px,
        base.1 + py,
        base.0 - px,
        base.1 - py
    );
    (base, pts)
}

fn path_d(points: &[(f64, f64)]) -> String {
    points
        .iter()
        .enumerate()
        .map(|(i, p)| format!("{} {:.0},{:.0}", if i == 0 { "M" } else { "L" }, p.0, p.1))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Card centers in board units plus the method used.
fn raw_positions(board: &BoardStateItem, ids: &[String]) -> (Vec<(f64, f64)>, &'static str) {
    let nodes: Vec<_> = ids.iter().filter_map(|id| board.node(id)).collect();
    if !nodes.is_empty() && nodes.iter().all(|n| n.bbox.is_some()) {
        return (
            nodes
                .iter()
                .filter_map(|n| n.bbox.map(|b| center(&b)))
                .collect(),
            "board_positions",
        );
    }
    let index: BTreeMap<&str, u32> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i as u32))
        .collect();
    let vertices: Vec<(u32, (f64, f64))> = (0..ids.len() as u32)
        .map(|i| (i, (CARD_W, CARD_H)))
        .collect();
    let edges: Vec<(u32, u32)> = board
        .final_edges()
        .into_iter()
        .filter_map(|e| Some((*index.get(e.src.as_str())?, *index.get(e.dst.as_str())?)))
        .filter(|(a, b)| a != b)
        .collect();
    let cfg = rust_sugiyama::configure::Config {
        vertex_spacing: GAP_X,
        ..Default::default()
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rust_sugiyama::from_vertices_and_edges(&vertices, &edges, &cfg)
    }));
    if let Ok(layouts) = result {
        let mut pos = vec![None; ids.len()];
        let mut offset = 0.0;
        for (items, w, _h) in layouts {
            for (v, (x, y)) in items {
                if let Some(slot) = pos.get_mut(v) {
                    *slot = Some((x + offset, y * 1.6));
                }
            }
            offset += w + CARD_W + GAP_X;
        }
        if pos.iter().all(Option::is_some) {
            return (pos.into_iter().flatten().collect(), "sugiyama");
        }
    }
    let grid = (0..ids.len())
        .map(|i| {
            (
                (i % 4) as f64 * (CARD_W + GAP_X),
                (i / 4) as f64 * (CARD_H + GAP_Y),
            )
        })
        .collect();
    (grid, "grid")
}

fn snap(values: &mut [f64], tol: f64) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut group: Vec<usize> = Vec::new();
    let flush = |group: &mut Vec<usize>, values: &mut [f64]| {
        if group.len() > 1 {
            let mean = group.iter().map(|i| values[*i]).sum::<f64>() / group.len() as f64;
            for i in group.iter() {
                values[*i] = mean;
            }
        }
        group.clear();
    };
    for i in order {
        if let Some(&last) = group.last() {
            if values[i] - values[last] >= tol {
                flush(&mut group, values);
            }
        }
        group.push(i);
    }
    flush(&mut group, values);
}

/// Pushes cards apart until every pair keeps its gaps.
fn separate(pos: &mut [(f64, f64)]) {
    for _ in 0..500 {
        let mut moved = false;
        for i in 0..pos.len() {
            for j in (i + 1)..pos.len() {
                let (dx, dy) = (pos[j].0 - pos[i].0, pos[j].1 - pos[i].1);
                let ox = CARD_W + GAP_X - dx.abs();
                let oy = CARD_H + GAP_Y - dy.abs();
                if ox <= 0.0 || oy <= 0.0 {
                    continue;
                }
                moved = true;
                if ox / (CARD_W + GAP_X) <= oy / (CARD_H + GAP_Y) {
                    let s = if dx >= 0.0 { 1.0 } else { -1.0 };
                    pos[i].0 -= s * ox / 2.0;
                    pos[j].0 += s * ox / 2.0;
                } else {
                    let s = if dy >= 0.0 { 1.0 } else { -1.0 };
                    pos[i].1 -= s * oy / 2.0;
                    pos[j].1 += s * oy / 2.0;
                }
            }
        }
        if !moved {
            break;
        }
    }
}

/// Glyph widths for layout, so text is sized to what is actually drawn.
pub trait Measure {
    /// Width in px of `text` set in the template class `class`.
    fn width(&self, text: &str, class: &str) -> f64;
}

/// Font size and boldness of a template class, for [`Estimate`].
fn class_font(class: &str) -> (f64, bool) {
    match class {
        "title" => (24.0, true),
        "subtitle" => (13.0, false),
        "section-label" => (18.0, true),
        "heading" => (14.0, true),
        "body-strong" | "grid-text" | "grid-text-green" => (12.0, true),
        "small" => (10.0, false),
        "mono" => (11.0, false),
        "label" => (10.0, true),
        "edge-label" | "pill-text" => (11.0, true),
        "card-title" => (13.0, true),
        _ => (12.0, false),
    }
}

/// Widths estimated from character counts ([`text_width`]), for layout
/// without fonts; [`crate::svg::TextMeasure`] measures the real glyphs.
pub struct Estimate;

impl Measure for Estimate {
    fn width(&self, text: &str, class: &str) -> f64 {
        let (size, bold) = class_font(class);
        text_width(text, size, bold)
    }
}

/// `text` wrapped to at most `max_lines` lines (the last cut with `...`) of at
/// most `chars` characters, fewer when a line's glyphs would be wider than
/// `max_px` in `class`.
fn fit_wrap(
    m: &dyn Measure,
    text: &str,
    class: &str,
    max_px: f64,
    chars: usize,
    max_lines: usize,
) -> Vec<String> {
    let mut n = chars.max(1);
    loop {
        let lines = wrap_lines(text, n, max_lines);
        if n == 1 || lines.iter().all(|l| m.width(l, class) <= max_px) {
            return lines;
        }
        n = (n * 9 / 10).min(n - 1).max(1);
    }
}

/// `text` on one line whose glyphs fit `max_px` in `class`, cut with `...`
/// when needed.
fn fit_line(m: &dyn Measure, text: &str, class: &str, max_px: f64) -> String {
    if m.width(text, class) <= max_px {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let cut = |n: usize| {
        let head: String = chars[..n].iter().collect();
        format!("{}...", head.trim_end())
    };
    // the longest prefix that fits (widths grow with the prefix)
    let (mut lo, mut hi) = (0usize, chars.len());
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if m.width(&cut(mid), class) <= max_px {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if lo == 0 {
        String::new()
    } else {
        cut(lo)
    }
}

/// A pill of text width `tw` (measured in `class`).
fn pill(x: f64, y: f64, text: &str, fill: &str, class: &str, tw: f64) -> Pill {
    let w = (tw + 24.0).max(56.0);
    let r = R::new(x, y, w, PILL_H);
    Pill {
        r,
        rx: 12.0,
        fill: fill.into(),
        stroke: None,
        text: tc(r.cx(), r.y + 16.0, text, class),
    }
}

/// A badge of measured text width `tw` (class `label`).
fn badge(x: f64, y: f64, text: &str, fill: &str, tw: f64) -> Pill {
    // bold 10 px capitals with letter spacing
    let w = (text.chars().count() as f64 * 7.2 + 20.0).max(tw + 20.0);
    let r = R::new(x, y, w, PILL_H);
    Pill {
        r,
        rx: 6.0,
        fill: fill.into(),
        stroke: None,
        text: tc(r.cx(), r.y + 16.0, text, "label"),
    }
}

/// Owner pill spots beside an edge path: both sides of every segment (longest
/// first) at the [`SLIDE`] fractions, then every 5% along each segment (a
/// short edge between two cards has room for a pill only in a narrow band).
/// Paths are axis-aligned.
fn beside_path(pts: &[(f64, f64)], w: f64, h: f64) -> Vec<R> {
    let seg_len = |i: usize| (pts[i + 1].0 - pts[i].0).abs() + (pts[i + 1].1 - pts[i].1).abs();
    let mut order: Vec<usize> = (0..pts.len().saturating_sub(1)).collect();
    order.sort_by(|a, b| seg_len(*b).total_cmp(&seg_len(*a)));
    let fine = (1..20).map(|k| f64::from(k) / 20.0);
    let fractions: Vec<f64> = SLIDE
        .into_iter()
        .chain(fine.filter(|t| !SLIDE.iter().any(|s| (s - t).abs() < 1e-9)))
        .collect();
    let mut out = Vec::new();
    for i in order {
        let (a, b) = (pts[i], pts[i + 1]);
        let horizontal = (a.1 - b.1).abs() < 0.5;
        for &t in &fractions {
            let m = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            if horizontal {
                out.push(R::new(m.0 - w / 2.0, m.1 + PILL_GAP, w, h));
                out.push(R::new(m.0 - w / 2.0, m.1 - PILL_GAP - h, w, h));
            } else {
                out.push(R::new(m.0 + PILL_GAP, m.1 - h / 2.0, w, h));
                out.push(R::new(m.0 - w - PILL_GAP, m.1 - h / 2.0, w, h));
            }
        }
    }
    out
}

/// The middle of the longest segment of a path, and whether that segment is
/// horizontal.
fn longest_mid(pts: &[Pt]) -> (f64, f64, bool) {
    let (li, _) = pts
        .windows(2)
        .enumerate()
        .map(|(i, w)| (i, (w[1].0 - w[0].0).abs() + (w[1].1 - w[0].1).abs()))
        .fold((0, -1.0), |acc, x| if x.1 > acc.1 { x } else { acc });
    match (pts.get(li), pts.get(li + 1)) {
        (Some(p0), Some(p1)) => (
            (p0.0 + p1.0) / 2.0,
            (p0.1 + p1.1) / 2.0,
            (p0.1 - p1.1).abs() < 0.5,
        ),
        (Some(p), None) => (p.0, p.1, true),
        _ => (0.0, 0.0, true),
    }
}

/// The four sides of a box as lines.
fn box_lines(r: &R) -> [Segment; 4] {
    let (a, b, c, d) = (
        (r.x, r.y),
        (r.right(), r.y),
        (r.right(), r.bottom()),
        (r.x, r.bottom()),
    );
    [(a, b), (b, c), (c, d), (d, a)]
}

/// Gap between an edge label and its segment.
const LABEL_GAP: f64 = 6.0;
/// Room a strictly placed label leaves at a segment's ends (arrowheads).
const LABEL_END: f64 = 14.0;

/// Spots of size `w` x `h` for an edge label beside the straight segments of
/// its path (longest first): above and below a horizontal segment, right and
/// left of a vertical one, [`LABEL_GAP`] from the line, never across it.
/// `strict` keeps the label within the segment's extent (clear of its ends);
/// otherwise only the label's middle is on the segment. Diagonal segments (a
/// fallback line) get none.
fn label_spots(pts: &[Pt], w: f64, h: f64, strict: bool) -> Vec<R> {
    let seg_len = |i: usize| (pts[i + 1].0 - pts[i].0).abs() + (pts[i + 1].1 - pts[i].1).abs();
    let mut order: Vec<usize> = (0..pts.len().saturating_sub(1)).collect();
    order.sort_by(|a, b| seg_len(*b).total_cmp(&seg_len(*a)));
    let fine = (1..20).map(|k| f64::from(k) / 20.0);
    let fractions: Vec<f64> = SLIDE
        .into_iter()
        .chain(fine.filter(|t| !SLIDE.iter().any(|s| (s - t).abs() < 1e-9)))
        .collect();
    let mut out: Vec<R> = Vec::new();
    for i in order {
        let (a, b) = (pts[i], pts[i + 1]);
        let horizontal = (a.1 - b.1).abs() < 0.5;
        let vertical = (a.0 - b.0).abs() < 0.5;
        if horizontal == vertical {
            continue;
        }
        let (lo, hi) = if horizontal {
            (a.0.min(b.0), a.0.max(b.0))
        } else {
            (a.1.min(b.1), a.1.max(b.1))
        };
        let ext = if horizontal { w } else { h };
        let (min_c, max_c) = if strict {
            (lo + ext / 2.0 + LABEL_END, hi - ext / 2.0 - LABEL_END)
        } else {
            (lo, hi)
        };
        if max_c < min_c {
            continue;
        }
        for &t in &fractions {
            let c = (lo + (hi - lo) * t).clamp(min_c, max_c);
            let spots = if horizontal {
                [
                    R::new(c - w / 2.0, a.1 - LABEL_GAP - h, w, h),
                    R::new(c - w / 2.0, a.1 + LABEL_GAP, w, h),
                ]
            } else {
                [
                    R::new(a.0 + LABEL_GAP, c - h / 2.0, w, h),
                    R::new(a.0 - LABEL_GAP - w, c - h / 2.0, w, h),
                ]
            };
            for s in spots {
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// Spots of size `w` x `h` just outside a card: along the top (right to
/// left), down both sides, then along the bottom.
fn around_card(r: &R, w: f64, h: f64) -> Vec<R> {
    around_box(r, w, h, 6.0, 12.0)
}

/// Spots of size `w` x `h` `gap` outside `r`, every `step` px: along the top
/// (right to left), down both sides, then along the bottom.
fn around_box(r: &R, w: f64, h: f64, gap: f64, step: f64) -> Vec<R> {
    let mut out = Vec::new();
    let mut x = r.right() - w;
    while x >= r.x - 0.5 {
        out.push(R::new(x, r.y - h - gap, w, h));
        x -= step;
    }
    let mut y = r.y;
    while y <= r.bottom() - h + 0.5 {
        out.push(R::new(r.right() + gap, y, w, h));
        out.push(R::new(r.x - w - gap, y, w, h));
        y += step;
    }
    let mut x = r.right() - w;
    while x >= r.x - 0.5 {
        out.push(R::new(x, r.bottom() + gap, w, h));
        x -= step;
    }
    out
}

/// Spots for a history note of size `w` x `h` next to the owner pill `p`,
/// nearest first: right of it, above and below it (left, then right aligned),
/// left of it, then sliding along its top and bottom.
fn note_spots(p: &R, w: f64, h: f64) -> Vec<R> {
    const G: f64 = 4.0;
    let mut v = vec![
        R::new(p.right() + 8.0, p.cy() - h / 2.0, w, h),
        R::new(p.x, p.y - h - G, w, h),
        R::new(p.right() - w, p.y - h - G, w, h),
        R::new(p.x, p.bottom() + G, w, h),
        R::new(p.right() - w, p.bottom() + G, w, h),
        R::new(p.x - w - 8.0, p.cy() - h / 2.0, w, h),
    ];
    // every 12 px while the note still overlaps the pill's span by 24 px
    for k in 1..40 {
        let d = 12.0 * f64::from(k);
        for x in [p.x - d, p.right() - w + d] {
            if x + w < p.x + 24.0 || x > p.right() - 24.0 {
                continue;
            }
            v.push(R::new(x, p.y - h - G, w, h));
            v.push(R::new(x, p.bottom() + G, w, h));
        }
    }
    v
}

/// Spots of size `w` x `h` centered on rings around `c` (16 per ring).
fn ring_spots(c: (f64, f64), w: f64, h: f64, radii: &[f64]) -> Vec<R> {
    let mut out = Vec::new();
    for rad in radii {
        for step in 0..16 {
            let a = f64::from(step) * std::f64::consts::PI / 8.0;
            let (x, y) = (c.0 + rad * a.cos(), c.1 + rad * a.sin());
            out.push(R::new(x - w / 2.0, y - h / 2.0, w, h));
        }
    }
    out
}

/// Id prefix of the annotation list's text lines.
pub const LIST_ID: &str = "annotation-list";

/// Upper bound of a character's advance in the list's 12 px regular text, by
/// width class (wide capitals and symbols, other capitals and digits, the
/// rest); generous so a line never runs past the list.
pub fn list_char_w(c: char) -> f64 {
    let em = match c {
        'W' | 'M' | 'm' | 'w' | '@' | '%' | '&' => 1.0,
        // full-width scripts and emoji
        c if !c.is_ascii() => 1.3,
        c if c.is_uppercase() || c.is_ascii_digit() => 0.8,
        _ => 0.6,
    };
    12.0 * em
}

/// Width of `s` in the list by [`list_char_w`].
pub fn list_text_w(s: &str) -> f64 {
    s.chars().map(list_char_w).sum()
}

/// Word wrap to at most `max_px` per line by the width function `w`,
/// splitting a word longer than a line.
fn wrap_px(text: &str, max_px: f64, w: &dyn Fn(&str) -> f64) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        let joined = if cur.is_empty() {
            word.to_string()
        } else {
            format!("{cur} {word}")
        };
        if w(&joined) <= max_px {
            cur = joined;
            continue;
        }
        if !cur.is_empty() {
            lines.push(std::mem::take(&mut cur));
        }
        for c in word.chars() {
            if !cur.is_empty() && w(&format!("{cur}{c}")) > max_px {
                lines.push(std::mem::take(&mut cur));
            }
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// Size of the numbered marker for footnote `n`.
fn marker_size(n: usize) -> (f64, f64) {
    (if n < 10 { 18.0 } else { 26.0 }, 18.0)
}

/// Numbered marker for footnote `n` with its top left at `(x, y)`.
fn marker(n: usize, x: f64, y: f64) -> Pill {
    let (w, h) = marker_size(n);
    let r = R::new(x, y, w, h);
    Pill {
        r,
        rx: 9.0,
        fill: "#334155".into(),
        stroke: None,
        text: tc(r.cx(), r.y + 13.0, n.to_string(), "label"),
    }
}

/// The point of a polyline nearest to `p`.
fn nearest_on_path(pts: &[(f64, f64)], p: (f64, f64)) -> Option<(f64, f64)> {
    pts.windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let len2 = dx * dx + dy * dy;
            let t = if len2 > 0.0 {
                (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (a.0 + dx * t, a.1 + dy * t)
        })
        .min_by(|a, b| {
            let d = |q: &(f64, f64)| (q.0 - p.0).powi(2) + (q.1 - p.1).powi(2);
            d(a).total_cmp(&d(b))
        })
}

/// A leader line.
type Leader = (f64, f64, f64, f64);

/// Leader from the nearest point of `r` to `anchor`.
fn leader_to(r: &R, anchor: (f64, f64)) -> Leader {
    let near = (
        anchor.0.clamp(r.x, r.right()),
        anchor.1.clamp(r.y, r.bottom()),
    );
    (near.0, near.1, anchor.0, anchor.1)
}

/// True when the line runs through the interior of `r` (touching its border
/// does not count); exact (Liang-Barsky clipping).
pub(crate) fn line_hits(l: &Leader, r: &R) -> bool {
    let (dx, dy) = (l.2 - l.0, l.3 - l.1);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, q) in [
        (-dx, l.0 - r.x),
        (dx, r.right() - l.0),
        (-dy, l.1 - r.y),
        (dy, r.bottom() - l.1),
    ] {
        if p.abs() < 1e-12 {
            if q <= 0.0 {
                return false;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }
    t1 - t0 > 1e-9
}

/// True when the leader and the segment `s` intersect (touching counts).
fn segments_cross(l: &Leader, s: (Pt, Pt)) -> bool {
    let (p1, p2) = ((l.0, l.1), (l.2, l.3));
    let (q1, q2) = s;
    let orient = |a: Pt, b: Pt, c: Pt| (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
    let on = |a: Pt, b: Pt, c: Pt| {
        c.0 >= a.0.min(b.0) - 1e-9
            && c.0 <= a.0.max(b.0) + 1e-9
            && c.1 >= a.1.min(b.1) - 1e-9
            && c.1 <= a.1.max(b.1) + 1e-9
    };
    let (d1, d2) = (orient(q1, q2, p1), orient(q1, q2, p2));
    let (d3, d4) = (orient(p1, p2, q1), orient(p1, p2, q2));
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }
    (d1 == 0.0 && on(q1, q2, p1))
        || (d2 == 0.0 && on(q1, q2, p2))
        || (d3 == 0.0 && on(p1, p2, q1))
        || (d4 == 0.0 && on(p1, p2, q2))
}

/// True when a leader crosses no taken box except `target`, the box of the
/// element it points at.
fn leader_clear(l: &Leader, taken: &[R], target: Option<R>) -> bool {
    taken
        .iter()
        .filter(|t| Some(**t) != target)
        .all(|t| !line_hits(l, t))
}

/// Free space for annotations: inside the canvas below the title band and
/// above the stickies and banner, clear of taken boxes, edges and leaders.
struct Space<'a> {
    width: f64,
    limit: f64,
    segments: &'a [Segment],
}

/// A straight piece of a drawn edge.
type Segment = ((f64, f64), (f64, f64));

impl Space<'_> {
    fn free(&self, r: &R, taken: &[R], leaders: &[Leader]) -> bool {
        r.x >= MARGIN / 2.0
            && r.right() <= self.width - MARGIN / 2.0
            && r.y >= LEGEND_BOTTOM
            && r.bottom() <= self.limit
            && !taken.iter().any(|t| t.intersects(r))
            && !self
                .segments
                .iter()
                .any(|(a, b)| line_hits(&(a.0, a.1, b.0, b.1), &r.inflate(3.0)))
            && !leaders.iter().any(|l| line_hits(l, &r.inflate(2.0)))
    }

    /// The first free spot in `direct`; else the first free spot in `ringed`
    /// whose leader to `anchor(spot)` crosses no taken box but `target`.
    fn place(
        &self,
        direct: &[R],
        ringed: &[R],
        anchor: impl Fn(&R) -> (f64, f64),
        target: Option<R>,
        taken: &[R],
        leaders: &[Leader],
    ) -> Option<(R, Option<Leader>)> {
        if let Some(r) = direct.iter().find(|r| self.free(r, taken, leaders)) {
            return Some((*r, None));
        }
        ringed.iter().find_map(|r| {
            if !self.free(r, taken, leaders) {
                return None;
            }
            let l = leader_to(r, anchor(r));
            leader_clear(&l, taken, target).then_some((*r, Some(l)))
        })
    }
}

/// Where a leader line from a spot ends on its element.
type Anchor = Box<dyn Fn(&R) -> (f64, f64)>;

/// Spots for an owner pill: next to its element, further out (with a leader
/// back to `anchor`), and for its marker when it has no room at all.
struct OwnerSpots {
    direct: Vec<R>,
    ringed: Vec<R>,
    anchor: Anchor,
    /// The taken box of the element a leader may end in.
    target: Option<R>,
    marker: Vec<R>,
}

/// Where an owner pill goes: its box, a leader back to its element, and the
/// spot of its history note right next to it (box, wrap form) when found.
type PillChoice = (R, Option<Leader>, Option<(R, usize)>);

/// An annotation moved to the footnotes, waiting for its marker.
struct Pending {
    text: String,
    /// Marker spots, best first (sized for this footnote's number).
    spots: Vec<R>,
    /// False when its element is not drawn on the board at all.
    drawn: bool,
}

/// Builds the scene for one board, with estimated text widths.
pub fn build_scene(board: &BoardStateItem, notes: &MeetingNotes) -> Scene {
    build_scene_with(board, notes, &Estimate)
}

/// Builds the scene for one board, sizing and fitting text by `m` (the
/// render measures the real glyphs, so no text runs past its box or the
/// canvas).
pub fn build_scene_with(board: &BoardStateItem, notes: &MeetingNotes, m: &dyn Measure) -> Scene {
    let grids = derive_grids(board);
    let in_grid: BTreeSet<&str> = grids
        .iter()
        .flat_map(|g| g.members.iter().map(|n| n.id.as_str()))
        .collect();
    let nodes: Vec<_> = board
        .final_nodes()
        .into_iter()
        .filter(|n| !in_grid.contains(n.id.as_str()))
        .collect();
    let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
    let (raw, layout_method) = raw_positions(board, &ids);
    let deferred = deferred_nodes(board, &notes.decisions);
    let focus = focus_node(board, &notes.decisions);

    // scale board units to pixels, snap rows and columns, then separate
    let (minx, maxx) = raw
        .iter()
        .fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.0), a.1.max(p.0)));
    let (miny, maxy) = raw
        .iter()
        .fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.1), a.1.max(p.1)));
    let avail_w = MIN_WIDTH - 2.0 * (MARGIN + ZONE_PAD_X) - CARD_W;
    let sx = if maxx - minx > 1e-9 {
        avail_w / (maxx - minx)
    } else {
        f64::INFINITY
    };
    let sy = if maxy - miny > 1e-9 {
        560.0 / (maxy - miny)
    } else {
        f64::INFINITY
    };
    let s = if sx.min(sy).is_finite() {
        sx.min(sy)
    } else {
        1.0
    };
    let mut xs: Vec<f64> = raw.iter().map(|p| (p.0 - minx) * s).collect();
    let mut ys: Vec<f64> = raw.iter().map(|p| (p.1 - miny) * s).collect();
    snap(&mut xs, CARD_W * 0.5);
    snap(&mut ys, CARD_H * 0.5);
    let mut pos: Vec<(f64, f64)> = xs.into_iter().zip(ys).collect();
    separate(&mut pos);
    let left = pos.iter().map(|p| p.0).fold(f64::MAX, f64::min);
    let top = pos.iter().map(|p| p.1).fold(f64::MAX, f64::min);
    let x0 = MARGIN + ZONE_PAD_X;
    let mut cards_r: Vec<R> = pos
        .iter()
        .map(|p| R::new(p.0 - left + x0, p.1 - top + ARCH_TOP, CARD_W, CARD_H))
        .collect();
    let content_right = cards_r.iter().map(R::right).fold(0.0, f64::max);
    let width = MIN_WIDTH.max(content_right + MARGIN + ZONE_PAD_X);
    // center the architecture horizontally
    let shift = ((width - 2.0 * (MARGIN + ZONE_PAD_X)) - (content_right - x0)) / 2.0;
    if shift > 0.0 {
        for r in &mut cards_r {
            r.x = (r.x + shift).round();
        }
    }
    let card_of: BTreeMap<&str, R> = ids
        .iter()
        .map(String::as_str)
        .zip(cards_r.iter().copied())
        .collect();

    // cards
    let mut cards = Vec::new();
    let mut roles_used: BTreeSet<Role> = BTreeSet::new();
    for (n, r) in nodes.iter().zip(&cards_r) {
        let role = role_of(&n.text);
        roles_used.insert(role);
        // a snake_case or path token becomes a mono body line
        // board text is read from the video: sanitize like model text
        let text = sanitize_dashes(&n.text);
        let words: Vec<&str> = text.split_whitespace().collect();
        let (head_words, mono): (Vec<&str>, Vec<&str>) = words
            .iter()
            .partition(|w| !(w.contains('_') && words.len() > 1));
        let mut title = head_words.join(" ");
        let mut lines = Vec::new();
        let mut y = r.y + 58.0;
        // everything stays inside the card: long titles wrap into at most two
        // body lines, a long mono token is truncated, overflow lines are dropped
        let inner = CARD_W - 24.0;
        let max_title = (inner / (14.0 * 0.6)) as usize;
        // the title may use 4 px of the right padding
        let title_max = CARD_W - 20.0;
        if m.width(&title, "heading") > title_max {
            let parts = fit_wrap(m, &title, "heading", title_max, max_title, usize::MAX);
            title = parts.first().cloned().unwrap_or_default();
            let rest = parts.get(1..).unwrap_or_default().join(" ");
            let chars = (inner / (12.0 * 0.6)) as usize;
            for l in fit_wrap(m, &rest, "body-strong", inner, chars, 2) {
                lines.push(tl(r.x + 12.0, y, l, "body-strong"));
                y += 20.0;
            }
        }
        if !mono.is_empty() {
            let mono = fit_wrap(
                m,
                &mono.join(" "),
                "mono",
                inner,
                (inner / (11.0 * 0.62)) as usize,
                1,
            );
            lines.push(tl(r.x + 12.0, y, mono.join(" "), "mono"));
            y += 20.0;
        }
        lines.push(tl(r.x + 12.0, y, role.describe(), "body"));
        y += 20.0;
        lines.push(tl(
            r.x + 12.0,
            y,
            format!("On the board from {}", mmss(first_seen(&n.lifetimes))),
            "small",
        ));
        if let Some(t) = deferred.get(&n.id) {
            y += 18.0;
            lines.push(TextLine {
                fill: Some("#b91c1c".into()),
                ..tl(r.x + 12.0, y, format!("Deferred at {}", mmss(*t)), "small")
            });
        }
        lines.retain(|l| l.y <= r.bottom() - 8.0);
        cards.push(Card {
            id: n.id.clone(),
            r: *r,
            color: role.color(),
            role: role.key(),
            title: tl(r.x + 12.0, r.y + 21.0, title, "heading"),
            lines,
            dashed: deferred.contains_key(&n.id),
            glow: focus.as_deref() == Some(n.id.as_str()),
        });
    }

    // zones where a role's cards form a cluster that covers no other card
    let mut zones: Vec<Zone> = Vec::new();
    let mut role_order: Vec<(Role, f64)> = roles_used
        .iter()
        .map(|r| {
            (
                *r,
                cards
                    .iter()
                    .filter(|c| c.role == r.key())
                    .map(|c| c.r.x)
                    .fold(f64::MAX, f64::min),
            )
        })
        .collect();
    role_order.sort_by(|a, b| a.1.total_cmp(&b.1));
    for (role, _) in role_order {
        let mine: Vec<&Card> = cards.iter().filter(|c| c.role == role.key()).collect();
        let (x, y) = (
            mine.iter().map(|c| c.r.x).fold(f64::MAX, f64::min),
            mine.iter().map(|c| c.r.y).fold(f64::MAX, f64::min),
        );
        let (xr, yb) = (
            mine.iter().map(|c| c.r.right()).fold(0.0, f64::max),
            mine.iter().map(|c| c.r.bottom()).fold(0.0, f64::max),
        );
        let zr = R::new(
            x - ZONE_PAD_X,
            y - ZONE_PAD_TOP,
            xr - x + 2.0 * ZONE_PAD_X,
            yb - y + ZONE_PAD_TOP + ZONE_PAD_BOTTOM,
        );
        let clash = cards
            .iter()
            .any(|c| c.role != role.key() && c.r.inflate(8.0).intersects(&zr))
            || zones.iter().any(|z| z.r.intersects(&zr));
        if clash {
            continue;
        }
        let has_focus = mine.iter().any(|c| c.glow);
        // the title stays inside its zone
        let zone_title = fit_line(m, role.zone_title(), "section-label", zr.w - 40.0);
        zones.push(Zone {
            r: zr,
            color: role.color(),
            role: role.key(),
            label: tl(
                zr.x + 20.0,
                zr.y + 28.0,
                zone_title.clone(),
                "section-label",
            ),
            badge: has_focus.then(|| {
                badge(
                    zr.x + 20.0,
                    zr.y + 40.0,
                    "FOCUS",
                    role.color(),
                    m.width("FOCUS", "label"),
                )
            }),
            label_w: m.width(&zone_title, "section-label"),
        });
    }
    let arch_bottom = cards_r.iter().map(R::bottom).fold(ARCH_TOP, f64::max);

    // edges: every edge is routed around the cards first (the shortest first,
    // so direct links keep their straight lines), then labeled once every
    // line is known, so no label sits where a later edge runs
    let mut taken: Vec<R> = cards_r.iter().map(|r| r.inflate(4.0)).collect();
    // zone titles and badges: labels never cover them, routes avoid them
    // when they can
    let soft: Vec<R> = zones
        .iter()
        .filter_map(|z| z.badge.as_ref().map(|b| b.r.inflate(2.0)))
        .chain(zones.iter().map(|z| zone_title_r(z).inflate(2.0)))
        .collect();
    taken.extend(soft.iter().copied());
    let mut edges = Vec::new();
    let mut routes: Vec<RouteInfo> = Vec::new();
    let mut route_warnings: Vec<String> = Vec::new();
    let mut edge_anchor: BTreeMap<String, (f64, f64, bool)> = BTreeMap::new();
    let mut edge_path: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    let mut segments: Vec<Segment> = Vec::new();
    // the segments of each drawn edge (by its index in `edges`)
    let mut seg_range: Vec<std::ops::Range<usize>> = Vec::new();
    // annotations moved to the footnotes, in footnote order
    let mut pending: Vec<Pending> = Vec::new();
    // leader lines drawn so far (later annotations keep off them)
    let mut leaders: Vec<Leader> = Vec::new();
    let final_edges = board.final_edges();
    let card_index: BTreeMap<&str, usize> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    let ends: Vec<Option<(usize, usize)>> = final_edges
        .iter()
        .map(|e| {
            Some((
                *card_index.get(e.src.as_str())?,
                *card_index.get(e.dst.as_str())?,
            ))
        })
        .collect();
    let span = |i: usize| {
        ends[i].map_or(0.0, |(a, b)| {
            (cards_r[a].cx() - cards_r[b].cx()).abs() + (cards_r[a].cy() - cards_r[b].cy()).abs()
        })
    };
    let mut order: Vec<usize> = (0..final_edges.len())
        .filter(|i| ends[*i].is_some())
        .collect();
    order.sort_by(|x, y| span(*x).total_cmp(&span(*y)));
    let route_top = LEGEND_BOTTOM + 16.0;
    // zone borders: an edge running along one reads as part of it
    let borders: Vec<Segment> = zones.iter().flat_map(|z| box_lines(&z.r)).collect();
    let mut router = Router::new(
        cards_r.clone(),
        soft,
        borders.clone(),
        R::new(
            MARGIN / 2.0,
            route_top,
            width - MARGIN,
            arch_bottom + 160.0 - route_top,
        ),
    );
    let mut routed: Vec<Option<crate::route::Routed>> = vec![None; final_edges.len()];
    for i in order {
        if let Some((a, b)) = ends[i] {
            // room for the label (and its gaps) between parallel neighbors
            let e = &final_edges[i];
            let label = e.label.trim();
            let gap = if label.is_empty() {
                0.0
            } else if e.style == EdgeStyle::Dashed || label.chars().count() > 22 {
                24.0 + 2.0 * (LABEL_GAP + 3.0)
            } else {
                18.0 + 2.0 * (LABEL_GAP + 3.0)
            };
            routed[i] = Some(router.route(a, b, gap));
        }
    }
    // labels placed once every edge is drawn: (edge, text, link, path)
    let mut to_label: Vec<(usize, String, String, Vec<Pt>)> = Vec::new();
    for (e, route) in final_edges.iter().zip(routed) {
        let Some(route) = route else {
            continue;
        };
        if route.points.len() < 2 {
            continue;
        }
        let link = format!(
            "{} to {}",
            sanitize_dashes(&e.a_text),
            sanitize_dashes(&e.b_text)
        );
        if route.fallback {
            let how = if route.points.len() == 2 {
                "drawn straight"
            } else {
                "drawn with a simple detour"
            };
            route_warnings.push(format!(
                "no route around the cards for the {link} link: {how}"
            ));
        }
        let full = route.points;
        let mut pts = full.clone();
        // arrowheads per the producer's direction: at dst (forward), at both
        // ends (bidirectional) or none (not established)
        let n = pts.len();
        let mut arrows = Vec::new();
        let (tip, start) = (pts[n - 1], pts[0]);
        if matches!(
            e.direction,
            EdgeOrientation::Forward | EdgeOrientation::Bidirectional
        ) {
            let (base, arrow) = arrowhead(tip, pts[n - 2]);
            pts[n - 1] = base;
            arrows.push(arrow);
        }
        if e.direction == EdgeOrientation::Bidirectional {
            let (base, arrow) = arrowhead(start, pts[1]);
            pts[0] = base;
            arrows.push(arrow);
        }
        let first = segments.len();
        segments.extend(full.windows(2).map(|w| (w[0], w[1])));
        seg_range.push(first..segments.len());
        edge_anchor.insert(e.id.clone(), longest_mid(&full));
        edge_path.insert(e.id.clone(), full.clone());
        routes.push(RouteInfo {
            name: format!("edge {link}"),
            src: e.src.clone(),
            dst: e.dst.clone(),
            points: full.clone(),
            fallback: route.fallback,
        });
        let dashed = e.style == EdgeStyle::Dashed;
        let label_text = sanitize_dashes(e.label.trim());
        if !label_text.is_empty() {
            to_label.push((edges.len(), label_text, link, full));
        }
        edges.push(EdgeArt {
            d: path_d(&pts),
            arrows,
            color: if dashed { "#7c3aed" } else { "#0f172a" },
            width: if dashed { 1.5 } else { 2.0 },
            dash: dashed.then_some("6,4"),
            label: None,
            label_lines: Vec::new(),
            leader: None,
        });
    }
    let routes_bottom = segments
        .iter()
        .map(|(a, b)| a.1.max(b.1))
        .fold(0.0, f64::max);

    // Edge labels sit beside a straight segment of their own edge (the
    // longest segment first), clear of every edge line, card, zone title and
    // label; a long relation label wraps; then the nearest free spot around
    // the edge's middle with a leader to it; else the list below the board.
    let label_space = Space {
        width,
        limit: f64::INFINITY,
        segments: &segments,
    };
    for (k, text, link, path) in to_label {
        let relation = edges[k].dash.is_some() || text.chars().count() > 22;
        let size = |lines: &[String]| -> (f64, f64) {
            if relation {
                (
                    lines.iter().map(|l| m.width(l, "body")).fold(0.0, f64::max) + 24.0,
                    24.0 + 16.0 * lines.len().saturating_sub(1) as f64,
                )
            } else {
                (m.width(&text, "edge-label") + 16.0, 18.0)
            }
        };
        let mut variants = vec![vec![text.clone()]];
        if relation {
            let wrapped = wrap(&text, 26);
            if wrapped.len() > 1 {
                variants.push(wrapped);
            }
        }
        let mut found: Option<(R, Vec<String>, Option<Leader>)> = None;
        'spots: for strict in [true, false] {
            for lines in &variants {
                let (w, h) = size(lines);
                if let Some(r) = label_spots(&path, w, h, strict)
                    .into_iter()
                    .find(|r| label_space.free(r, &taken, &leaders))
                {
                    found = Some((r, lines.clone(), None));
                    break 'spots;
                }
            }
        }
        if found.is_none() {
            let lines = variants.last().cloned().unwrap_or_default();
            let (lw, lh) = size(&lines);
            let (mx, my, _) = longest_mid(&path);
            // rings every 40 px, each spot also slid sideways by a quarter
            // and a half of the label's width (a leader then drops straight
            // down a gap the centered spot would block)
            let spots = (1..=8).flat_map(|ring| {
                let rad = 40.0 * f64::from(ring);
                (0..12).flat_map(move |step| {
                    let a = f64::from(step) * std::f64::consts::PI / 6.0;
                    [0.0, 0.25, -0.25, 0.5, -0.5]
                        .map(|k| (mx + rad * a.cos() + k * lw, my + rad * a.sin()))
                })
            });
            for c in spots {
                let r = R::new(c.0 - lw / 2.0, c.1 - lh / 2.0, lw, lh);
                if !label_space.free(&r, &taken, &leaders) {
                    continue;
                }
                // the leader ends on the nearest point of the edge and
                // may cross nothing on its way back
                let anchor = nearest_on_path(&path, (r.cx(), r.cy())).unwrap_or((mx, my));
                let l = leader_to(&r, anchor);
                // nor any other edge (it would point at the wrong one)
                let own = seg_range.get(k).cloned().unwrap_or(0..0);
                let crosses_other = segments
                    .iter()
                    .enumerate()
                    .any(|(si, sg)| !own.contains(&si) && segments_cross(&l, *sg));
                if crosses_other || !leader_clear(&l, &taken, None) {
                    continue;
                }
                found = Some((r, lines.clone(), Some(l)));
                break;
            }
        }
        let Some((r, lines, leader)) = found else {
            let (mw, mh) = marker_size(pending.len() + 1);
            pending.push(Pending {
                text: format!("Label on the {link} link: {text}"),
                spots: beside_path(&path, mw, mh),
                drawn: true,
            });
            continue;
        };
        let mut label_lines = Vec::new();
        for (j, l) in lines.iter().enumerate().skip(1) {
            let mut t = tc(r.cx(), r.y + 16.0 + 16.0 * j as f64, l.clone(), "body");
            t.fill = Some("#5b21b6".into());
            label_lines.push(t);
        }
        let (class, stroke) = if relation {
            ("body", "#7c3aed")
        } else {
            ("edge-label", "#cbd5e1")
        };
        let mut t = tc(
            r.cx(),
            r.y + if relation { 16.0 } else { 13.0 },
            lines[0].clone(),
            class,
        );
        if relation {
            t.fill = Some("#5b21b6".into());
        }
        taken.push(r);
        leaders.extend(leader);
        let art = &mut edges[k];
        art.label = Some(Pill {
            r,
            rx: if relation { 6.0 } else { 4.0 },
            fill: "#ffffff".into(),
            stroke: Some(stroke.into()),
            text: t,
        });
        art.label_lines = label_lines;
        art.leader = leader;
    }
    let channels_bottom = routes_bottom + CLEAR;

    // owner pills, moved-from notes, deferred badges; none may cover a card,
    // another label or an edge. Each tries spots next to its element, then
    // spots further out joined by a leader line; one with no room at all goes
    // to the footnotes below the board, marked by its number on the board.
    let zones_bottom = zones.iter().map(|z| z.r.bottom()).fold(0.0, f64::max);
    // stickies, the banner and panels start at or below this line
    // (the first sticky row starts 24 px lower; the banner or a panel starts
    // right there when there are no stickies; an edge label placed away from
    // its path may reach below the cards, and pushes them down)
    let sticky_gap = if board.final_stickies().is_empty() {
        0.0
    } else {
        24.0
    };
    let labels_bottom = edges
        .iter()
        .filter_map(|e| e.label.as_ref().map(|l| l.r.bottom()))
        .fold(0.0, f64::max);
    let annot_limit = channels_bottom
        .max(zones_bottom)
        .max(arch_bottom + ZONE_PAD_BOTTOM)
        .max(labels_bottom + 8.0 - sticky_gap);
    let space = Space {
        width,
        limit: annot_limit + sticky_gap,
        segments: &segments,
    };
    let node_text: BTreeMap<&str, String> = nodes
        .iter()
        .map(|n| (n.id.as_str(), sanitize_dashes(&n.text)))
        .collect();
    let mut pills: Vec<Pill> = Vec::new();
    let mut notes_txt: Vec<TextLine> = Vec::new();
    let mut note_boxes: Vec<R> = Vec::new();
    let mut per_target: BTreeMap<String, Vec<&OwnerAssignment>> = BTreeMap::new();
    for o in board.current_owners() {
        let key = match &o.target {
            OwnerTarget::Node { node_id, .. } => node_id.clone(),
            OwnerTarget::Edge { edge_id, .. } => edge_id.clone(),
        };
        per_target.entry(key).or_default().push(o);
    }
    for (target, owners) in per_target {
        for o in owners {
            let name = sanitize_dashes(&short_name(notes, &o.person_id, &o.display_name));
            let name_w = m.width(&name, "pill-text");
            let proto = pill(0.0, 0.0, &name, "#16a34a", "pill-text", name_w);
            let (w, h) = (proto.r.w, proto.r.h);
            let moved = o.moved_from.as_ref().map(|from| {
                format!(
                    "moved from {} ({})",
                    sanitize_dashes(&target_text(from)),
                    mmss(o.valid_from_s)
                )
            });
            let what = sanitize_dashes(&target_text(&o.target));
            let spots = match &o.target {
                OwnerTarget::Node { .. } => {
                    let Some(r) = card_of.get(target.as_str()).copied() else {
                        // not drawn as a card (a grid member): listed only
                        pending.push(Pending {
                            text: format!("Owner {name} of {what}"),
                            spots: Vec::new(),
                            drawn: false,
                        });
                        continue;
                    };
                    // above the card, sliding right (bounded), then below it,
                    // then all around it
                    let mut v: Vec<R> = (0..16)
                        .map(|k| R::new(r.x + 12.0 * k as f64, r.y - 34.0, w, h))
                        .collect();
                    v.extend((0..4).map(|k| R::new(r.x + 12.0 * k as f64, r.bottom() + 8.0, w, h)));
                    v.extend(around_card(&r, w, h));
                    let rings =
                        ring_spots((r.cx(), r.cy()), w, h, &[110.0, 140.0, 170.0, 200.0, 240.0]);
                    let (mw, mh) = marker_size(pending.len() + 1);
                    let anchor: Anchor = Box::new(move |p: &R| {
                        (p.cx().clamp(r.x, r.right()), p.cy().clamp(r.y, r.bottom()))
                    });
                    OwnerSpots {
                        direct: v,
                        ringed: rings,
                        anchor,
                        target: Some(r.inflate(4.0)),
                        marker: around_card(&r, mw, mh),
                    }
                }
                OwnerTarget::Edge { .. } => {
                    let Some((mx, my, horizontal)) = edge_anchor.get(target.as_str()).copied()
                    else {
                        // the edge is not drawn (an end is not a card): listed only
                        pending.push(Pending {
                            text: format!("Owner {name} of {what}"),
                            spots: Vec::new(),
                            drawn: false,
                        });
                        continue;
                    };
                    let mut v = Vec::new();
                    for step in 0..6 {
                        let d = 28.0 * step as f64;
                        if horizontal {
                            v.push(R::new(mx - w / 2.0, my + 10.0 + d, w, h));
                            v.push(R::new(mx - w / 2.0, my - 34.0 - d, w, h));
                        } else {
                            v.push(R::new(mx - w - 10.0, my + 4.0 + d, w, h));
                            v.push(R::new(mx - w - 10.0, my - 28.0 - d, w, h));
                            v.push(R::new(mx + 10.0, my + 28.0 + d, w, h));
                        }
                    }
                    // Then beside the rest of the edge: a parallel edge or a
                    // label next to the middle, with cards at both ends, must
                    // not leave the owner out while the edge has room elsewhere.
                    let path = edge_path.get(target.as_str()).cloned().unwrap_or_default();
                    v.extend(beside_path(&path, w, h));
                    let rings = ring_spots((mx, my), w, h, &[60.0, 90.0, 120.0, 160.0]);
                    let (mw, mh) = marker_size(pending.len() + 1);
                    // the nearest point of the edge (not its middle, where a
                    // label may sit on the line)
                    let line = path.clone();
                    let anchor: Anchor = Box::new(move |p: &R| {
                        nearest_on_path(&line, (p.cx(), p.cy())).unwrap_or((mx, my))
                    });
                    OwnerSpots {
                        direct: v,
                        ringed: rings,
                        anchor,
                        target: None,
                        marker: beside_path(&path, mw, mh),
                    }
                }
            };
            let OwnerSpots {
                direct,
                ringed,
                anchor,
                target: lead_target,
                marker: marker_spots,
            } = spots;
            // a history note wraps to one, two or three lines
            let note_forms: Vec<(Vec<String>, f64, f64)> = moved
                .as_ref()
                .map(|txt| {
                    let n = txt.chars().count();
                    let mut forms: Vec<Vec<String>> = vec![vec![txt.clone()]];
                    for k in [2usize, 3] {
                        let w = wrap(txt, n.div_ceil(k) + 4);
                        if forms.last().is_some_and(|f| w.len() > f.len()) {
                            forms.push(w);
                        }
                    }
                    forms
                        .into_iter()
                        .map(|lines| {
                            let tw = lines
                                .iter()
                                .map(|l| m.width(l, "small"))
                                .fold(0.0, f64::max);
                            let th = 12.0 * lines.len() as f64 + 4.0;
                            (lines, tw, th)
                        })
                        .collect()
                })
                .unwrap_or_default();
            // clear of zone borders when `clean` (text across a dashed border
            // is hard to read)
            let off_borders = |r: &R| {
                !borders
                    .iter()
                    .any(|(a, b)| line_hits(&(a.0, a.1, b.0, b.1), &r.inflate(2.0)))
            };
            let note_next_to = |pr: &R, taken: &[R], leaders: &[Leader], clean: bool| {
                note_forms.iter().enumerate().find_map(|(k, (_, tw, th))| {
                    note_spots(pr, *tw, *th)
                        .into_iter()
                        .find(|r| space.free(r, taken, leaders) && (!clean || off_borders(r)))
                        .map(|r| (r, k))
                })
            };
            // The first free spot next to the element where the history note
            // also fits right next to the pill; else the first free spot at
            // all (the note then goes further out).
            let mut chosen: Option<PillChoice> = None;
            if !note_forms.is_empty() {
                // Every free spot next to the element, best class first: pill
                // and note clear of zone borders, then the note, then either.
                let mut best: [Option<PillChoice>; 3] = [None, None, None];
                for pr in &direct {
                    if !space.free(pr, &taken, &leaders) {
                        continue;
                    }
                    taken.push(*pr);
                    let clean = note_next_to(pr, &taken, &leaders, true);
                    let any = if clean.is_some() {
                        None
                    } else {
                        note_next_to(pr, &taken, &leaders, false)
                    };
                    taken.pop();
                    let class = match (&clean, off_borders(pr)) {
                        (Some(_), true) => 0,
                        (Some(_), false) => 1,
                        (None, _) if any.is_some() => 2,
                        _ => continue,
                    };
                    if best[class].is_none() {
                        best[class] = Some((*pr, None, clean.or(any)));
                    }
                    if class == 0 {
                        break;
                    }
                }
                chosen = best.into_iter().flatten().next();
                // then a pill further out (with its leader) whose note fits
                // right next to it, before a note goes further out or below
                if chosen.is_none() {
                    for pr in &ringed {
                        if !space.free(pr, &taken, &leaders) {
                            continue;
                        }
                        let l = leader_to(pr, anchor(pr));
                        if !leader_clear(&l, &taken, lead_target) {
                            continue;
                        }
                        taken.push(*pr);
                        leaders.push(l);
                        let n = note_next_to(pr, &taken, &leaders, true)
                            .or_else(|| note_next_to(pr, &taken, &leaders, false));
                        taken.pop();
                        leaders.pop();
                        if n.is_some() {
                            chosen = Some((*pr, Some(l), n));
                            break;
                        }
                    }
                }
            }
            if chosen.is_none() {
                chosen = space
                    .place(&direct, &ringed, anchor, lead_target, &taken, &leaders)
                    .map(|(pr, lead)| (pr, lead, None));
            }
            let Some((pr, lead, near)) = chosen else {
                let mut text = format!("Owner {name} of {what}");
                if let Some(m) = &moved {
                    text.push_str(&format!(", {m}"));
                }
                pending.push(Pending {
                    text,
                    spots: marker_spots,
                    drawn: true,
                });
                continue;
            };
            let p = pill(pr.x, pr.y, &name, "#16a34a", "pill-text", name_w);
            taken.push(p.r);
            leaders.extend(lead);
            if let Some(txt) = moved {
                let placed = match near
                    .or_else(|| note_next_to(&p.r, &taken, &leaders, true))
                    .or_else(|| note_next_to(&p.r, &taken, &leaders, false))
                {
                    Some((nb, k)) => Some((nb, None, k)),
                    // further out, nearest first, with a leader back to the pill
                    None => note_forms.iter().enumerate().find_map(|(k, (_, tw, th))| {
                        let pr = p.r;
                        let base = (tw + pr.w) / 2.0;
                        let ringed = ring_spots(
                            (pr.cx(), pr.cy()),
                            *tw,
                            *th,
                            &[base + 16.0, base + 40.0, base + 70.0],
                        );
                        let anchor = move |nb: &R| {
                            (
                                nb.cx().clamp(pr.x, pr.right()),
                                nb.cy().clamp(pr.y, pr.bottom()),
                            )
                        };
                        space
                            .place(&[], &ringed, anchor, Some(pr), &taken, &leaders)
                            .map(|(nb, l)| (nb, l, k))
                    }),
                };
                match placed {
                    Some((nb, lead, k)) => {
                        let lines = note_forms.get(k).map(|f| f.0.clone()).unwrap_or_default();
                        for (j, l) in lines.into_iter().enumerate() {
                            notes_txt.push(tl(nb.x, nb.y + 12.0 + 12.0 * j as f64, l, "small"));
                        }
                        taken.push(nb);
                        note_boxes.push(nb);
                        leaders.extend(lead);
                    }
                    None => {
                        let (mw, mh) = marker_size(pending.len() + 1);
                        let mut spots = vec![
                            R::new(p.r.right() + 4.0, p.r.y + 3.0, mw, mh),
                            R::new(p.r.x - mw - 4.0, p.r.y + 3.0, mw, mh),
                            R::new(p.r.right() - mw, p.r.y - mh - 4.0, mw, mh),
                            R::new(p.r.x, p.r.bottom() + 4.0, mw, mh),
                        ];
                        spots.extend(ring_spots(
                            (p.r.cx(), p.r.cy()),
                            mw,
                            mh,
                            &[p.r.w / 2.0 + 16.0],
                        ));
                        pending.push(Pending {
                            text: format!("Owner {name} of {what}: {txt}"),
                            spots,
                            drawn: true,
                        });
                    }
                }
            }
            pills.push(p);
        }
    }
    for (id, t) in &deferred {
        let Some(r) = card_of.get(id.as_str()).copied() else {
            continue;
        };
        let text = format!("DEFERRED ({})", mmss(*t));
        let mut b = badge(0.0, 0.0, &text, "#ef4444", m.width(&text, "label"));
        // right-aligned above the card; slide left, then up a row; then below
        let mut cands = Vec::new();
        for row in 0..4 {
            let y = r.y - 34.0 - 30.0 * row as f64;
            let mut x = r.right() - b.r.w;
            while x >= r.x - 200.0 {
                cands.push(R::new(x, y, b.r.w, b.r.h));
                x -= 12.0;
            }
        }
        cands.extend((0..3).map(|k| {
            R::new(
                r.right() - b.r.w,
                r.bottom() + 6.0 + 30.0 * k as f64,
                b.r.w,
                b.r.h,
            )
        }));
        cands.extend(around_card(&r, b.r.w, b.r.h));
        let ringed = ring_spots(
            (r.cx(), r.cy()),
            b.r.w,
            b.r.h,
            &[120.0, 160.0, 200.0, 240.0],
        );
        let anchor = move |p: &R| (p.cx().clamp(r.x, r.right()), p.cy().clamp(r.y, r.bottom()));
        let Some((br, lead)) = space.place(
            &cands,
            &ringed,
            anchor,
            Some(r.inflate(4.0)),
            &taken,
            &leaders,
        ) else {
            let (mw, mh) = marker_size(pending.len() + 1);
            pending.push(Pending {
                text: format!(
                    "{} {text}",
                    node_text.get(id.as_str()).cloned().unwrap_or_default()
                ),
                spots: around_card(&r, mw, mh),
                drawn: true,
            });
            continue;
        };
        b.r = br;
        b.text.x = b.r.cx();
        b.text.y = b.r.y + 16.0;
        taken.push(b.r);
        leaders.extend(lead);
        pills.push(b);
    }

    // markers for the footnotes: the first free spot next to each element
    let mut markers: Vec<Pill> = Vec::new();
    let mut degraded: Vec<String> = Vec::new();
    for (i, p) in pending.iter().enumerate() {
        let n = i + 1;
        let spot = p
            .spots
            .iter()
            .find(|r| space.free(r, &taken, &leaders))
            .copied();
        match spot {
            Some(r) => {
                let m = marker(n, r.x, r.y);
                taken.push(m.r);
                markers.push(m);
                degraded.push(format!(
                    "no room on the board, listed below it as note {n}: {}",
                    p.text
                ));
            }
            None if !p.drawn => degraded.push(format!(
                "its element is not drawn on the board, listed below it as note {n} (no marker): {}",
                p.text
            )),
            None => degraded.push(format!(
                "no room on the board, listed below it as note {n} (no marker, no room for one): {}",
                p.text
            )),
        }
    }

    // edges drawn straight for want of a route (after the numbered notes)
    degraded.extend(route_warnings);

    // stickies
    let mut stickies = Vec::new();
    let mut sorted = board.final_stickies();
    sorted.sort_by(|a, b| first_seen(&a.lifetimes).total_cmp(&first_seen(&b.lifetimes)));
    let mut y = annot_limit + 24.0;
    let per_row = 6usize;
    for row in sorted.chunks(per_row) {
        let n = row.len() as f64;
        let step = if n > 1.0 {
            ((width - 2.0 * MARGIN - CARD_W) / (n - 1.0)).min(CARD_W + 100.0)
        } else {
            0.0
        };
        let span = step * (n - 1.0) + CARD_W;
        let x_start = (width - span) / 2.0;
        let mut row_h: f64 = 0.0;
        for (i, s) in row.iter().enumerate() {
            let (kind, color) = match s.kind {
                StickyKind::Question => ("OPEN QUESTION", "#f59e0b"),
                StickyKind::Milestone => ("MILESTONE", "#0891b2"),
                // Appendix B specifies the idea header in indigo; no analytics
                // cards share the canvas, so it is not ambiguous here
                StickyKind::Idea => ("IDEA", "#6366f1"),
                StickyKind::Note => ("NOTE", "#2563eb"),
            };
            let text = s.text.trim();
            let text = if s.kind == StickyKind::Idea {
                text.trim_start_matches(|c: char| c.is_alphabetic())
                    .trim_start_matches(':')
                    .trim()
            } else {
                text
            };
            let lines = fit_wrap(
                m,
                &sanitize_dashes(text),
                "sticky-text",
                CARD_W - 24.0,
                26,
                4,
            );
            let h = (24.0 + 22.0 + 18.0 * lines.len() as f64).max(96.0);
            let r = R::new(x_start + step * i as f64, y, CARD_W, h);
            row_h = row_h.max(h);
            stickies.push(StickyArt {
                r,
                header: color,
                kind: tl(r.x + 12.0, r.y + 16.0, kind, "label"),
                lines: lines
                    .iter()
                    .enumerate()
                    .map(|(k, l)| {
                        tl(
                            r.x + 12.0,
                            r.y + 46.0 + 18.0 * k as f64,
                            l.clone(),
                            "sticky-text",
                        )
                    })
                    .collect(),
            });
        }
        y += row_h + 16.0;
    }
    let mut bottom = if stickies.is_empty() {
        y - 24.0
    } else {
        y + 8.0
    };

    // decision banner: a decision that defers a component, else the earliest
    let banner = notes
        .decisions
        .iter()
        .max_by(|a, b| {
            let score = |d: &glassrip_notes::notes::Decision| {
                i32::from(deferred.values().any(|t| (*t - d.t_start_s).abs() < 1e-6))
            };
            score(a)
                .cmp(&score(b))
                .then(b.t_start_s.total_cmp(&a.t_start_s))
        })
        .map(|d| {
            let r = R::new(MARGIN, bottom, width - 2.0 * MARGIN, 56.0);
            let when = if d.t_end_s - d.t_start_s >= 5.0 {
                format!(" ({} to {})", mmss(d.t_start_s), mmss(d.t_end_s))
            } else {
                format!(" ({})", mmss(d.t_start_s))
            };
            let chars = ((r.w - 130.0) / 6.6) as usize;
            let lines = fit_wrap(
                m,
                &format!("{}{}", sanitize_dashes(&d.text), when),
                "body-strong",
                r.w - 128.0,
                chars,
                2,
            );
            let mut ls = Vec::new();
            for (k, l) in lines.iter().enumerate() {
                let mut t = tl(
                    r.x + 112.0,
                    r.y + 25.0 + 18.0 * k as f64,
                    l.clone(),
                    if k == 0 { "body-strong" } else { "body" },
                );
                if k > 0 {
                    t.fill = Some("#92400e".into());
                }
                ls.push(t);
            }
            let more = notes.decisions.len().saturating_sub(1);
            if more > 0 && ls.len() < 2 {
                let mut t = tl(
                    r.x + 112.0,
                    r.y + 43.0,
                    format!(
                        "{more} more decision{} in the notes",
                        if more == 1 { "" } else { "s" }
                    ),
                    "small",
                );
                t.fill = Some("#92400e".into());
                ls.push(t);
            }
            let label = badge(
                r.x + 16.0,
                r.y + 16.0,
                "DECISION",
                "#f59e0b",
                m.width("DECISION", "label"),
            );
            Banner {
                r,
                label,
                lines: ls,
            }
        });
    if let Some(b) = &banner {
        bottom = b.r.bottom() + 24.0;
    }

    // grids of unconnected boxes, derived from their positions on the board
    let mut panels = Vec::new();
    for g in &grids {
        let pw = width - 2.0 * MARGIN;
        let px = MARGIN;
        let color = "#2563eb";
        let mut shapes = Vec::new();
        let mut texts = Vec::new();
        let mut inner = bottom + 60.0;
        texts.push(tl(
            px + 16.0,
            inner,
            format!(
                "Boxes without arrows, laid out in {} rows and {} columns; on the board from {}",
                g.rows,
                g.columns,
                mmss(g.first_seen_s)
            ),
            "small",
        ));
        inner += 16.0;
        let cols = g.columns.clamp(1, 6);
        let cw = (pw - 32.0 - (cols as f64 - 1.0) * 16.0) / cols as f64;
        for (k, member) in g.members.iter().enumerate() {
            let r = R::new(
                px + 16.0 + (k % cols) as f64 * (cw + 16.0),
                inner + (k / cols) as f64 * 68.0,
                cw,
                52.0,
            );
            shapes.push(Shape {
                r,
                rx: 8.0,
                fill: "#dbeafe",
                stroke: "#2563eb",
                stroke_width: 1.0,
                dash: None,
                shadow: true,
            });
            let lines = fit_wrap(
                m,
                &sanitize_dashes(&member.text),
                "grid-text",
                cw,
                (cw / 7.2) as usize,
                2,
            );
            let y0 = r.cy() + 4.0 - 8.0 * (lines.len() as f64 - 1.0);
            for (li, l) in lines.iter().enumerate() {
                texts.push(tc(r.cx(), y0 + 16.0 * li as f64, l.clone(), "grid-text"));
            }
        }
        inner += g.members.len().div_ceil(cols) as f64 * 68.0;
        let r = R::new(px, bottom, pw, inner - bottom + 16.0);
        panels.push(Panel {
            r,
            color,
            title: tl(px + 16.0, bottom + 23.0, "Grouped boxes", "heading"),
            shapes,
            texts,
            pills: Vec::new(),
            arrows: Vec::new(),
        });
        bottom += r.h + 24.0;
    }

    // annotations without room on the board, one numbered entry per line
    // group, below everything else (the canvas grows to hold them)
    let footnotes = (!pending.is_empty()).then(|| {
        let (fx, fw) = (MARGIN, width - 2.0 * MARGIN);
        let max_px = fw - 72.0;
        let mut y = bottom + 44.0;
        let mut items = Vec::new();
        for (i, p) in pending.iter().enumerate() {
            let mk = marker(i + 1, fx + 16.0, y);
            // the generous estimate and the real glyphs, whichever is wider
            let fit = |t: &str| list_text_w(t).max(m.width(t, "body"));
            let lines = wrap_px(&p.text, max_px, &fit)
                .into_iter()
                .enumerate()
                .map(|(k, l)| TextLine {
                    id: Some(format!("{LIST_ID}-{}-{k}", i + 1)),
                    ..tl(fx + 52.0, y + 13.0 + 18.0 * k as f64, l, "body")
                })
                .collect::<Vec<_>>();
            y += 18.0 * lines.len().max(1) as f64 + 8.0;
            items.push(Footnote { marker: mk, lines });
        }
        let r = R::new(fx, bottom, fw, y - bottom + 8.0);
        Footnotes {
            r,
            title: tl(
                fx + 16.0,
                bottom + 26.0,
                "Annotations without room on the board (numbered where they belong when a number fits)",
                "body-strong",
            ),
            items,
        }
    });
    if let Some(f) = &footnotes {
        bottom += f.r.h + 24.0;
    }

    let height = bottom + 40.0;
    let title_text = notes
        .title
        .clone()
        .unwrap_or_else(|| "Meeting board".into());
    let subtitle = format!(
        "Architecture as drawn on the board during the meeting (final state at {})",
        mmss(board.end_s())
    );
    let footer = format!(
        "Source: board read from the meeting recording (final state at {}); owners from name tags on the board; deferrals and the banner from decisions validated against the transcript.",
        mmss(board.end_s())
    );

    // legend
    let mut items = Vec::new();
    let mut lx = 16.0;
    let mut add = |kind: &'static str, label: &str, lead: f64| {
        items.push(LegendItem {
            kind,
            x: lx,
            label: label.to_string(),
        });
        lx += lead + m.width(label, "body") + 28.0;
    };
    add("solid", "Request (caller to callee)", 48.0);
    if final_edges
        .iter()
        .any(|e| e.direction == EdgeOrientation::Uncertain)
    {
        add("undirected", "Direction not established", 40.0);
    }
    if final_edges.iter().any(|e| e.style == EdgeStyle::Dashed) {
        add("dashed", "Proposed relationship", 48.0);
    }
    if !pills.iter().all(|p| p.fill != "#16a34a") {
        add("owner", "Owner tag on the board", 72.0);
    }
    if !deferred.is_empty() {
        add("deferred", "Deferred by a decision", 84.0);
    }
    if !stickies.is_empty() {
        add("sticky", "Board sticky", 28.0);
    }
    if !markers.is_empty() {
        add("marker", "Listed below the board", 28.0);
    }
    let lw = lx;
    let legend = Legend {
        r: R::new((width - lw) / 2.0, 76.0, lw, 44.0),
        items,
    };

    let mut blocking: Vec<(String, R)> = Vec::new();
    blocking.push(("legend".into(), legend.r));
    for c in &cards {
        blocking.push((format!("card {}", c.id), c.r));
    }
    for z in &zones {
        blocking.push((format!("zone title {}", z.label.text), zone_title_r(z)));
        if let Some(b) = &z.badge {
            blocking.push((format!("zone badge {}", z.label.text), b.r));
        }
    }
    for (i, p) in pills.iter().enumerate() {
        blocking.push((format!("pill {i} {}", p.text.text), p.r));
    }
    for (i, nb) in note_boxes.iter().enumerate() {
        blocking.push((format!("note {i}"), *nb));
    }
    for e in &edges {
        if let Some(l) = &e.label {
            blocking.push((format!("edge label {}", l.text.text), l.r));
        }
    }
    for (i, s) in stickies.iter().enumerate() {
        blocking.push((format!("sticky {i}"), s.r));
    }
    if let Some(b) = &banner {
        blocking.push(("banner".into(), b.r));
    }
    for (i, p) in panels.iter().enumerate() {
        blocking.push((format!("panel {i}"), p.r));
    }
    for m in &markers {
        blocking.push((format!("marker {}", m.text.text), m.r));
    }
    if let Some(f) = &footnotes {
        blocking.push(("annotation list".into(), f.r));
    }

    // edge label leaders are drawn with their edge
    let leaders: Vec<Leader> = leaders
        .into_iter()
        .filter(|l| !edges.iter().any(|e| e.leader == Some(*l)))
        .collect();
    Scene {
        width: width.round(),
        height: height.round(),
        title: tc(
            width / 2.0,
            40.0,
            fit_line(
                m,
                &sanitize_dashes(&title_text),
                "title",
                width - 2.0 * MARGIN,
            ),
            "title",
        ),
        subtitle: tc(
            width / 2.0,
            62.0,
            fit_line(m, &subtitle, "subtitle", width - 2.0 * MARGIN),
            "subtitle",
        ),
        legend,
        roles: roles_used.iter().map(|r| (r.key(), r.color())).collect(),
        zones,
        cards,
        edges,
        pills,
        notes: notes_txt,
        stickies,
        banner,
        panels,
        markers,
        leaders,
        footnotes,
        footer: tc(
            width / 2.0,
            height - 24.0,
            fit_line(m, &footer, "small", width - 2.0 * MARGIN),
            "small",
        ),
        layout_method,
        blocking,
        routes,
        degraded,
    }
}

/// Box of a zone's title text (`section-label`: 18 px bold, baseline 28 px below
/// the zone's top, starting 20 px in).
fn zone_title_r(z: &Zone) -> R {
    R::new(z.r.x + 20.0, z.r.y + 12.0, z.label_w, 20.0)
}

/// Blocking boxes outside the canvas and pairs that overlap, edges that run
/// through a card, and edge labels an edge runs through. Annotations with no
/// room are not listed: they are moved below the board (`Scene::degraded`);
/// neither is a fallback edge clear of every other card (a warning there).
pub fn overlaps(scene: &Scene) -> Vec<String> {
    let mut out = overlapping_boxes(scene);
    out.extend(edge_defects(scene));
    out
}

/// Edges through a card's interior (a routed edge's own cards included: it
/// only touches those at its ports; a fallback edge through any other card:
/// its degraded route is a warning only while it obscures nothing), and edge
/// labels crossed by any edge.
pub fn edge_defects(scene: &Scene) -> Vec<String> {
    let mut out = Vec::new();
    for route in &scene.routes {
        for w in route.points.windows(2) {
            let l = (w[0].0, w[0].1, w[1].0, w[1].1);
            for c in &scene.cards {
                let own = c.id == route.src || c.id == route.dst;
                // a fallback's straight line leaves its own cards at computed
                // border points: only the other cards are checked
                if (route.fallback && own) || !line_hits(&l, &c.r) {
                    continue;
                }
                let note = match (own, route.fallback) {
                    (true, _) => " (its own)",
                    (false, true) => " (a fallback with no route around the cards)",
                    (false, false) => "",
                };
                out.push(format!("{} runs through card {}{note}", route.name, c.id));
            }
        }
    }
    for (i, e) in scene.edges.iter().enumerate() {
        let Some(label) = &e.label else {
            continue;
        };
        for (j, route) in scene.routes.iter().enumerate() {
            if route
                .points
                .windows(2)
                .any(|w| line_hits(&(w[0].0, w[0].1, w[1].0, w[1].1), &label.r))
            {
                let whose = if i == j {
                    "its own edge".to_string()
                } else {
                    route.name.clone()
                };
                out.push(format!(
                    "edge label {} is crossed by {whose}",
                    label.text.text
                ));
            }
        }
    }
    out
}

fn overlapping_boxes(scene: &Scene) -> Vec<String> {
    let b = &scene.blocking;
    let mut out = Vec::new();
    let canvas = R {
        x: 0.0,
        y: 0.0,
        w: scene.width,
        h: scene.height,
    };
    for (name, r) in b {
        if r.x < canvas.x
            || r.y < canvas.y
            || r.right() > canvas.right()
            || r.bottom() > canvas.bottom()
        {
            out.push(format!("{name} is outside the canvas"));
        }
    }
    for i in 0..b.len() {
        for j in (i + 1)..b.len() {
            if b[i].1.intersects(&b[j].1) {
                out.push(format!("{} overlaps {}", b[i].0, b[j].0));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separation_keeps_gaps() {
        let mut p = vec![(0.0, 0.0), (10.0, 5.0), (20.0, 200.0)];
        separate(&mut p);
        for i in 0..p.len() {
            for j in (i + 1)..p.len() {
                let (dx, dy) = ((p[i].0 - p[j].0).abs(), (p[i].1 - p[j].1).abs());
                assert!(dx >= CARD_W + GAP_X - 1e-6 || dy >= CARD_H + GAP_Y - 1e-6);
            }
        }
    }

    #[test]
    fn snapping_aligns_near_values() {
        let mut v = vec![0.0, 30.0, 300.0, 310.0];
        snap(&mut v, 100.0);
        assert_eq!(v, vec![15.0, 15.0, 305.0, 305.0]);
    }

    #[test]
    fn owner_spots_cover_both_sides_of_every_segment() {
        // an L-shaped path: a long vertical leg, then a short horizontal one
        let pts = [(100.0, 100.0), (100.0, 400.0), (200.0, 400.0)];
        let spots = beside_path(&pts, 60.0, 24.0);
        // SLIDE is a subset of the 5% steps: 19 fractions per segment
        let per_segment = 2 * 19;
        assert_eq!(spots.len(), 2 * per_segment);
        // the longest segment comes first, starting at its middle, right side
        assert_eq!(spots[0], R::new(110.0, 238.0, 60.0, 24.0));
        assert_eq!(spots[1], R::new(30.0, 238.0, 60.0, 24.0));
        // every spot clears its own segment by the gap (a spot near a corner
        // may touch the other segment; placement rejects those)
        let (long, short) = spots.split_at(per_segment);
        for s in long {
            assert!(
                !line_hits(
                    &(pts[0].0, pts[0].1, pts[1].0, pts[1].1),
                    &s.inflate(PILL_GAP - 1.0)
                ),
                "{s:?}"
            );
        }
        for s in short {
            assert!(
                !line_hits(
                    &(pts[1].0, pts[1].1, pts[2].0, pts[2].1),
                    &s.inflate(PILL_GAP - 1.0)
                ),
                "{s:?}"
            );
        }
        // the horizontal leg gets spots above and below it
        assert!(short.iter().any(|s| s.bottom() <= 400.0 - PILL_GAP));
        assert!(short.iter().any(|s| s.y >= 400.0 + PILL_GAP));
    }

    #[test]
    fn leaders_hit_box_interiors_only() {
        let r = R::new(10.0, 10.0, 20.0, 20.0);
        // through the middle, and a diagonal clipping a corner
        assert!(line_hits(&(0.0, 20.0, 40.0, 20.0), &r));
        assert!(line_hits(&(5.0, 20.0, 20.0, 5.0), &r));
        // ending inside counts; ending on the border does not
        assert!(line_hits(&(0.0, 20.0, 15.0, 20.0), &r));
        assert!(!line_hits(&(0.0, 20.0, 10.0, 20.0), &r));
        // along a border, beside it, stopping short, a point on the border
        assert!(!line_hits(&(10.0, 0.0, 10.0, 40.0), &r));
        assert!(!line_hits(&(0.0, 5.0, 40.0, 5.0), &r));
        assert!(!line_hits(&(0.0, 20.0, 5.0, 20.0), &r));
        assert!(!line_hits(&(10.0, 15.0, 10.0, 15.0), &r));
        // a point inside is a hit
        assert!(line_hits(&(15.0, 15.0, 15.0, 15.0), &r));
        // a corner graze outside the box
        assert!(!line_hits(&(0.0, 20.0, 20.0, 0.0), &r));
    }

    #[test]
    fn the_list_wraps_wide_scripts_and_long_words_within_its_width() {
        let cjk: String = "\u{4e2d}\u{6587}\u{5185}\u{5bb9}".repeat(60);
        for text in [
            format!("Owner \u{674e}\u{56db} of {cjk}"),
            "W".repeat(400),
            "https://example.com/".repeat(30),
        ] {
            let lines = wrap_px(&text, 600.0, &list_text_w);
            assert!(lines.len() > 1);
            for l in &lines {
                assert!(list_text_w(l) <= 600.0, "{l}");
            }
            assert_eq!(lines.concat().replace(' ', ""), text.replace(' ', ""));
        }
        // a full-width glyph is bounded above its real advance (about 1 em)
        assert!(list_char_w('\u{4e2d}') >= 12.0);
    }

    #[test]
    fn arrowhead_points_at_the_tip() {
        let (base, pts) = arrowhead((100.0, 50.0), (0.0, 50.0));
        assert_eq!(base, (89.0, 50.0));
        assert_eq!(pts, "100,50 89,56 89,44");
    }
}
