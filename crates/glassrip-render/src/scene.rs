//! Layout: turns a board state and validated notes into positioned shapes.
//!
//! All geometry is computed here; the SVG template only draws. Node positions
//! come from the board's own canvas bboxes (scaled, snapped into rows and
//! columns, and pushed apart until cards keep their gaps); when positions are
//! missing, a Sugiyama layered layout is used instead.

use std::collections::{BTreeMap, BTreeSet};

use glassrip_notes::board::{BoardState, EdgeStyle, GroupKind, StickyKind, TargetKind};
use glassrip_notes::notes::MeetingNotes;
use glassrip_notes::text::{mmss, sanitize_dashes};
use serde::Serialize;

use crate::facts::{deferred_nodes, final_nodes, focus_node, short_name};
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
    fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
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
    fn cx(&self) -> f64 {
        self.x + self.w / 2.0
    }
    fn cy(&self) -> f64 {
        self.y + self.h / 2.0
    }
    /// True when the interiors overlap (touching edges do not count).
    pub fn intersects(&self, o: &R) -> bool {
        self.x < o.right() && o.x < self.right() && self.y < o.bottom() && o.y < self.bottom()
    }
    fn inflate(&self, d: f64) -> R {
        R {
            x: self.x - d,
            y: self.y - d,
            w: self.w + 2.0 * d,
            h: self.h + 2.0 * d,
        }
    }
    fn hits_segment(&self, a: (f64, f64), b: (f64, f64)) -> bool {
        // axis-aligned segments only
        let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
        let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
        x0 < self.right()
            && self.x < x1.max(x0 + 0.5)
            && y0 < self.bottom()
            && self.y < y1.max(y0 + 0.5)
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
}

fn tl(x: f64, y: f64, text: impl Into<String>, class: &str) -> TextLine {
    TextLine {
        x: x.round(),
        y: y.round(),
        text: text.into(),
        class: class.into(),
        anchor: "start",
        fill: None,
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
    /// Arrowhead polygon points.
    pub arrow: String,
    /// Stroke color.
    pub color: &'static str,
    /// Stroke width.
    pub width: f64,
    /// Dash pattern.
    pub dash: Option<&'static str>,
    /// Label.
    pub label: Option<Pill>,
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
    /// Footer.
    pub footer: TextLine,
    /// Layout method (`board_positions` or `sugiyama` or `grid`).
    pub layout_method: &'static str,
    /// Boxes that must not overlap (name, box).
    #[serde(skip)]
    pub blocking: Vec<(String, R)>,
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
fn raw_positions(board: &BoardState, ids: &[String]) -> (Vec<(f64, f64)>, &'static str) {
    let nodes: Vec<_> = ids.iter().filter_map(|id| board.node(id)).collect();
    if !nodes.is_empty() && nodes.iter().all(|n| n.bbox.is_some()) {
        return (
            nodes
                .iter()
                .filter_map(|n| n.bbox.map(|b| b.center()))
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
        .edges
        .iter()
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

fn pill(x: f64, y: f64, text: &str, fill: &str, class: &str, size: f64, bold: bool) -> Pill {
    let w = (text_width(text, size, bold) + 24.0).max(56.0);
    let r = R::new(x, y, w, PILL_H);
    Pill {
        r,
        rx: 12.0,
        fill: fill.into(),
        stroke: None,
        text: tc(r.cx(), r.y + 16.0, text, class),
    }
}

fn badge(x: f64, y: f64, text: &str, fill: &str) -> Pill {
    // bold 10 px capitals with letter spacing
    let w = text.chars().count() as f64 * 7.2 + 20.0;
    let r = R::new(x, y, w, PILL_H);
    Pill {
        r,
        rx: 6.0,
        fill: fill.into(),
        stroke: None,
        text: tc(r.cx(), r.y + 16.0, text, "label"),
    }
}

/// Builds the scene for one board.
pub fn build_scene(board: &BoardState, notes: &MeetingNotes) -> Scene {
    let nodes = final_nodes(board);
    let ids: Vec<String> = nodes.iter().map(|n| n.node_id.clone()).collect();
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
        let words: Vec<&str> = n.text.split_whitespace().collect();
        let (head_words, mono): (Vec<&str>, Vec<&str>) = words
            .iter()
            .partition(|w| !(w.contains('_') && words.len() > 1));
        let mut title = head_words.join(" ");
        let mut lines = Vec::new();
        let mut y = r.y + 58.0;
        let max_title = ((CARD_W - 24.0) / (14.0 * 0.6)) as usize;
        if title.chars().count() > max_title {
            let parts = wrap(&title, max_title);
            title = parts.first().cloned().unwrap_or_default();
            let rest = parts[1..].join(" ");
            lines.push(tl(r.x + 12.0, y, rest, "body-strong"));
            y += 20.0;
        }
        if !mono.is_empty() {
            lines.push(tl(r.x + 12.0, y, mono.join(" "), "mono"));
            y += 20.0;
        }
        lines.push(tl(r.x + 12.0, y, role.describe(), "body"));
        y += 20.0;
        lines.push(tl(
            r.x + 12.0,
            y,
            format!("On the board from {}", mmss(n.first_seen_s)),
            "small",
        ));
        if let Some(t) = deferred.get(&n.node_id) {
            y += 18.0;
            lines.push(TextLine {
                fill: Some("#b91c1c".into()),
                ..tl(r.x + 12.0, y, format!("Deferred at {}", mmss(*t)), "small")
            });
        }
        cards.push(Card {
            id: n.node_id.clone(),
            r: *r,
            color: role.color(),
            role: role.key(),
            title: tl(r.x + 12.0, r.y + 21.0, title, "heading"),
            lines,
            dashed: deferred.contains_key(&n.node_id),
            glow: focus.as_deref() == Some(n.node_id.as_str()),
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
        zones.push(Zone {
            r: zr,
            color: role.color(),
            role: role.key(),
            label: tl(zr.x + 20.0, zr.y + 28.0, role.zone_title(), "section-label"),
            badge: has_focus.then(|| badge(zr.x + 20.0, zr.y + 40.0, "FOCUS", role.color())),
        });
    }
    let arch_bottom = cards_r.iter().map(R::bottom).fold(ARCH_TOP, f64::max);

    // edges
    let mut taken: Vec<R> = cards_r.iter().map(|r| r.inflate(4.0)).collect();
    taken.extend(
        zones
            .iter()
            .filter_map(|z| z.badge.as_ref().map(|b| b.r.inflate(2.0))),
    );
    let mut edges = Vec::new();
    let mut edge_anchor: BTreeMap<String, (f64, f64, bool)> = BTreeMap::new();
    let mut channel = 0usize;
    let mut segments: Vec<((f64, f64), (f64, f64))> = Vec::new();
    for e in &board.edges {
        let (Some(a), Some(b)) = (card_of.get(e.src.as_str()), card_of.get(e.dst.as_str())) else {
            continue;
        };
        let others: Vec<R> = cards_r
            .iter()
            .filter(|r| *r != a && *r != b)
            .map(|r| r.inflate(6.0))
            .collect();
        let vo = (a.y.max(b.y), a.bottom().min(b.bottom()));
        let ho = (a.x.max(b.x), a.right().min(b.right()));
        let mut pts: Vec<(f64, f64)> = if vo.1 - vo.0 > 20.0 {
            let y = (vo.0 + vo.1) / 2.0;
            if b.cx() > a.cx() {
                vec![(a.right(), y), (b.x, y)]
            } else {
                vec![(a.x, y), (b.right(), y)]
            }
        } else if ho.1 - ho.0 > 20.0 {
            let x = (ho.0 + ho.1) / 2.0;
            if b.cy() > a.cy() {
                vec![(x, a.bottom()), (x, b.y)]
            } else {
                vec![(x, a.y), (x, b.bottom())]
            }
        } else {
            let sx = if b.cx() > a.cx() { a.right() } else { a.x };
            let ey = if b.cy() > a.cy() { b.y } else { b.bottom() };
            vec![(sx, a.cy()), (b.cx(), a.cy()), (b.cx(), ey)]
        };
        let crosses = pts
            .windows(2)
            .any(|w| others.iter().any(|o| o.hits_segment(w[0], w[1])));
        let mut via_channel = false;
        if crosses {
            let yc = arch_bottom + 36.0 + 26.0 * channel as f64;
            let off = (channel as f64 - 0.5) * 24.0;
            pts = vec![
                (a.cx() + off, a.bottom()),
                (a.cx() + off, yc),
                (b.cx() + off, yc),
                (b.cx() + off, b.bottom()),
            ];
            channel += 1;
            via_channel = true;
        }
        let n = pts.len();
        let tip = pts[n - 1];
        let (base, arrow) = arrowhead(tip, pts[n - 2]);
        pts[n - 1] = base;
        for w in pts.windows(2) {
            segments.push((w[0], w[1]));
        }
        segments.push((base, tip));
        let dashed = e.style == EdgeStyle::Dashed;
        // label at the middle of the longest segment
        let (li, _) = pts
            .windows(2)
            .enumerate()
            .map(|(i, w)| (i, (w[1].0 - w[0].0).abs() + (w[1].1 - w[0].1).abs()))
            .fold((0, -1.0), |acc, x| if x.1 > acc.1 { x } else { acc });
        let (p0, p1) = (pts[li], pts[li + 1]);
        let horizontal = (p0.1 - p1.1).abs() < 0.5;
        let mid = ((p0.0 + p1.0) / 2.0, (p0.1 + p1.1) / 2.0);
        edge_anchor.insert(e.edge_id.clone(), (mid.0, mid.1, horizontal));
        let label = e
            .label
            .as_ref()
            .map(|t| sanitize_dashes(t))
            .filter(|t| !t.is_empty())
            .map(|text| {
                let relation = dashed || text.chars().count() > 22;
                let (h, tw) = if relation {
                    (24.0, text_width(&text, 12.0, false) + 24.0)
                } else {
                    (18.0, text_width(&text, 11.0, true) + 16.0)
                };
                let candidates: Vec<(f64, f64)> = if via_channel || (relation && horizontal) {
                    vec![
                        (mid.0 - tw / 2.0, mid.1 - h / 2.0),
                        (mid.0 - tw / 2.0, mid.1 + 8.0),
                    ]
                } else if horizontal {
                    vec![
                        (mid.0 - tw / 2.0, mid.1 - h - 6.0),
                        (mid.0 - tw / 2.0, mid.1 + 6.0),
                    ]
                } else {
                    vec![
                        (mid.0 + 8.0, mid.1 - h / 2.0),
                        (mid.0 - tw - 8.0, mid.1 - h / 2.0),
                    ]
                };
                let r = candidates
                    .iter()
                    .map(|(x, y)| R::new(*x, *y, tw, h))
                    .find(|r| !taken.iter().any(|t| t.intersects(r)))
                    .unwrap_or_else(|| R::new(candidates[0].0, candidates[0].1, tw, h));
                let (class, fill, stroke) = if relation {
                    ("body", "#ffffff", "#7c3aed")
                } else {
                    ("edge-label", "#ffffff", "#cbd5e1")
                };
                let mut t = tc(
                    r.cx(),
                    r.y + if relation { 16.0 } else { 13.0 },
                    text,
                    class,
                );
                if relation {
                    t.fill = Some("#5b21b6".into());
                }
                Pill {
                    r,
                    rx: if relation { 6.0 } else { 4.0 },
                    fill: fill.into(),
                    stroke: Some(stroke.into()),
                    text: t,
                }
            });
        if let Some(l) = &label {
            taken.push(l.r);
        }
        edges.push(EdgeArt {
            d: path_d(&pts),
            arrow,
            color: if dashed { "#7c3aed" } else { "#0f172a" },
            width: if dashed { 1.5 } else { 2.0 },
            dash: dashed.then_some("6,4"),
            label,
        });
    }
    let channels_bottom = if channel > 0 {
        arch_bottom + 36.0 + 26.0 * channel as f64
    } else {
        arch_bottom
    };

    // owner pills, moved-from notes, deferred badges; none may cover a card,
    // another label or an edge
    let mut pills: Vec<Pill> = Vec::new();
    let mut notes_txt: Vec<TextLine> = Vec::new();
    let mut note_boxes: Vec<R> = Vec::new();
    let free = |r: &R, taken: &[R]| {
        !taken.iter().any(|t| t.intersects(r))
            && !segments
                .iter()
                .any(|(a, b)| r.inflate(3.0).hits_segment(*a, *b))
    };
    let label_of = |id: &str| {
        board
            .node(id)
            .map(|n| n.text.clone())
            .unwrap_or_else(|| id.to_string())
    };
    let mut per_node: BTreeMap<&str, Vec<&glassrip_notes::board::OwnerAssignment>> =
        BTreeMap::new();
    for o in board
        .owner_assignments
        .iter()
        .filter(|o| o.valid_to_s.is_none())
    {
        per_node.entry(o.target_id.as_str()).or_default().push(o);
    }
    for (target, owners) in per_node {
        let Some(first) = owners.first() else {
            continue;
        };
        match first.target_kind {
            TargetKind::Node => {
                let Some(r) = card_of.get(target) else {
                    continue;
                };
                let mut x = r.x;
                for o in owners {
                    let name = short_name(notes, o.person_id.as_deref(), &o.name_raw);
                    let mut p = pill(x, r.y - 34.0, &name, "#16a34a", "pill-text", 11.0, true);
                    for _ in 0..40 {
                        if free(&p.r, &taken) {
                            break;
                        }
                        p.r.x += 12.0;
                        p.text.x += 12.0;
                    }
                    x = p.r.right() + 8.0;
                    let moved = board.owner_assignments.iter().find(|q| {
                        q.owner_id != o.owner_id
                            && q.person_id == o.person_id
                            && q.name_raw == o.name_raw
                            && q.valid_to_s
                                .is_some_and(|t| (t - o.valid_from_s).abs() <= 5.0)
                    });
                    taken.push(p.r);
                    if let Some(q) = moved {
                        let from = match q.target_kind {
                            TargetKind::Node => label_of(&q.target_id),
                            TargetKind::Edge => "a link".into(),
                        };
                        let txt = format!("moved from {} ({})", from, mmss(o.valid_from_s));
                        let tw = text_width(&txt, 10.0, false);
                        let beside = R::new(p.r.right() + 8.0, p.r.y + 4.0, tw, 16.0);
                        let above = R::new(p.r.x, p.r.y - 20.0, tw, 16.0);
                        let nb = if free(&beside, &taken) || !free(&above, &taken) {
                            beside
                        } else {
                            above
                        };
                        notes_txt.push(tl(nb.x, nb.y + 12.0, txt, "small"));
                        taken.push(nb);
                        note_boxes.push(nb);
                        x = nb.right() + 8.0;
                    }
                    pills.push(p);
                }
            }
            TargetKind::Edge => {
                let Some((mx, my, horizontal)) = edge_anchor.get(target).copied() else {
                    continue;
                };
                for o in owners.iter() {
                    let name = short_name(notes, o.person_id.as_deref(), &o.name_raw);
                    let w = (text_width(&name, 11.0, true) + 24.0).max(56.0);
                    // candidates around the label anchor, nearest first
                    let mut cands: Vec<(f64, f64)> = Vec::new();
                    for step in 0..6 {
                        let d = 28.0 * step as f64;
                        if horizontal {
                            cands.push((mx - w / 2.0, my + 10.0 + d));
                            cands.push((mx - w / 2.0, my - 34.0 - d));
                        } else {
                            cands.push((mx - w - 10.0, my + 4.0 + d));
                            cands.push((mx - w - 10.0, my - 28.0 - d));
                            cands.push((mx + 10.0, my + 28.0 + d));
                        }
                    }
                    let mut p = pill(
                        cands[0].0,
                        cands[0].1,
                        &name,
                        "#16a34a",
                        "pill-text",
                        11.0,
                        true,
                    );
                    for (x, y) in &cands {
                        let cand = pill(*x, *y, &name, "#16a34a", "pill-text", 11.0, true);
                        if free(&cand.r, &taken) {
                            p = cand;
                            break;
                        }
                    }
                    taken.push(p.r);
                    pills.push(p);
                }
            }
        }
    }
    for (id, t) in &deferred {
        let Some(r) = card_of.get(id.as_str()) else {
            continue;
        };
        let text = format!("DEFERRED ({})", mmss(*t));
        let mut b = badge(0.0, r.y - 34.0, &text, "#ef4444");
        // right-aligned above the card; slide left, then up a row, to a free slot
        let mut placed = false;
        'rows: for row in 0..6 {
            let y = r.y - 34.0 - 30.0 * row as f64;
            let mut x = r.right() - b.r.w;
            while x >= r.x - 200.0 {
                let cand = R::new(x, y, b.r.w, b.r.h);
                if free(&cand, &taken) {
                    b.r = cand;
                    placed = true;
                    break 'rows;
                }
                x -= 12.0;
            }
        }
        if !placed {
            // last resort: below the card, where no pill or note is placed
            b.r = R::new(r.right() - b.r.w, r.bottom() + 6.0, b.r.w, b.r.h);
        }
        b.text.x = b.r.cx();
        b.text.y = b.r.y + 16.0;
        taken.push(b.r);
        pills.push(b);
    }

    // stickies
    let mut stickies = Vec::new();
    let mut sorted: Vec<_> = board.stickies.iter().collect();
    sorted.sort_by(|a, b| a.first_seen_s.total_cmp(&b.first_seen_s));
    let zones_bottom = zones.iter().map(|z| z.r.bottom()).fold(0.0, f64::max);
    let mut y = channels_bottom
        .max(zones_bottom)
        .max(arch_bottom + ZONE_PAD_BOTTOM)
        + 24.0;
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
            let (kind, color) = match s.effective_kind() {
                StickyKind::Question => ("OPEN QUESTION", "#f59e0b"),
                StickyKind::Idea => ("IDEA", "#6366f1"),
                StickyKind::Note => ("NOTE", "#2563eb"),
            };
            let text = s.text.trim();
            let text = if s.effective_kind() == StickyKind::Idea {
                text.trim_start_matches(|c: char| c.is_alphabetic())
                    .trim_start_matches(':')
                    .trim()
            } else {
                text
            };
            let lines = wrap_lines(&sanitize_dashes(text), 26, 4);
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
            let lines = wrap_lines(&format!("{}{}", sanitize_dashes(&d.text), when), chars, 2);
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
            let label = badge(r.x + 16.0, r.y + 16.0, "DECISION", "#f59e0b");
            Banner {
                r,
                label,
                lines: ls,
            }
        });
    if let Some(b) = &banner {
        bottom = b.r.bottom() + 24.0;
    }

    // group panels
    let mut panels = Vec::new();
    let groups: Vec<_> = board.groups.iter().collect();
    for row in groups.chunks(2) {
        let pw = if row.len() == 1 {
            width - 2.0 * MARGIN
        } else {
            (width - 2.0 * MARGIN - 40.0) / 2.0
        };
        let mut row_h: f64 = 0.0;
        for (i, g) in row.iter().enumerate() {
            let px = MARGIN + i as f64 * (pw + 40.0);
            let color = if g.kind == GroupKind::Grid {
                "#2563eb"
            } else {
                "#7c3aed"
            };
            let mut shapes = Vec::new();
            let mut texts = Vec::new();
            let mut ppills = Vec::new();
            let mut arrows = Vec::new();
            let mut inner = bottom + 60.0;
            texts.push(tl(
                px + 16.0,
                inner,
                format!(
                    "On the board from {} ({} cards)",
                    mmss(g.first_seen_s),
                    g.members.len()
                ),
                "small",
            ));
            inner += 16.0;
            match g.kind {
                GroupKind::Grid => {
                    let cols = g.columns.unwrap_or(3).max(1) as usize;
                    let cw = (pw - 32.0 - (cols as f64 - 1.0) * 16.0) / cols as f64;
                    for (k, m) in g.members.iter().enumerate() {
                        let r = R::new(
                            px + 16.0 + (k % cols) as f64 * (cw + 16.0),
                            inner + (k / cols) as f64 * 68.0,
                            cw,
                            52.0,
                        );
                        let (fill, stroke, sw, class) = if m.highlight {
                            ("#dcfce7", "#10b981", 2.0, "grid-text-green")
                        } else {
                            ("#dbeafe", "#2563eb", 1.0, "grid-text")
                        };
                        shapes.push(Shape {
                            r,
                            rx: 8.0,
                            fill,
                            stroke,
                            stroke_width: sw,
                            dash: None,
                            shadow: true,
                        });
                        let lines = wrap_lines(&m.text, (cw / 7.2) as usize, 2);
                        let y0 = r.cy() + 4.0 - 8.0 * (lines.len() as f64 - 1.0);
                        for (li, l) in lines.iter().enumerate() {
                            texts.push(tc(r.cx(), y0 + 16.0 * li as f64, l.clone(), class));
                        }
                    }
                    let rows = g.members.len().div_ceil(cols) as f64;
                    inner += rows * 68.0;
                    if g.members.iter().any(|m| m.highlight) {
                        let sw = R::new(px + 16.0, inner + 4.0, 16.0, 16.0);
                        shapes.push(Shape {
                            r: sw,
                            rx: 4.0,
                            fill: "#dcfce7",
                            stroke: "#10b981",
                            stroke_width: 2.0,
                            dash: None,
                            shadow: false,
                        });
                        texts.push(tl(
                            px + 40.0,
                            inner + 16.0,
                            "Green: drawn in a different color on the board",
                            "small",
                        ));
                        inner += 28.0;
                    }
                }
                GroupKind::Container => {
                    let mut cx = px + 16.0;
                    if let Some(f) = &g.fed_by {
                        let fr = R::new(cx, inner + 28.0, 164.0, 72.0);
                        shapes.push(Shape {
                            r: fr,
                            rx: 8.0,
                            fill: "#e0e7ff",
                            stroke: "#6366f1",
                            stroke_width: 2.0,
                            dash: None,
                            shadow: true,
                        });
                        let lines = wrap_lines(f, 18, 2);
                        let y0 = fr.cy() + 5.0 - 9.0 * (lines.len() as f64 - 1.0);
                        for (li, l) in lines.iter().enumerate() {
                            let mut t = tc(fr.cx(), y0 + 18.0 * li as f64, l.clone(), "card-title");
                            t.fill = Some("#3730a3".into());
                            texts.push(t);
                        }
                        let (base, arrow) =
                            arrowhead((fr.right() + 52.0, fr.cy()), (fr.right(), fr.cy()));
                        arrows.push((path_d(&[(fr.right(), fr.cy()), base]), arrow));
                        cx = fr.right() + 60.0;
                    }
                    let cr = R::new(cx, inner, px + pw - 16.0 - cx, 128.0);
                    shapes.push(Shape {
                        r: cr,
                        rx: 10.0,
                        fill: "#f5f3ff",
                        stroke: "#7c3aed",
                        stroke_width: 1.5,
                        dash: Some("6,3"),
                        shadow: false,
                    });
                    let mut t = tl(cr.x + 16.0, cr.y + 18.0, g.label.to_uppercase(), "small");
                    t.fill = Some("#5b21b6".into());
                    texts.push(t);
                    let m = g.members.len().max(1) as f64;
                    let mw = ((cr.w - 32.0 - (m - 1.0) * 8.0) / m).min(160.0);
                    for (k, mem) in g.members.iter().enumerate() {
                        let r = R::new(cr.x + 16.0 + k as f64 * (mw + 8.0), cr.y + 30.0, mw, 80.0);
                        shapes.push(Shape {
                            r,
                            rx: 8.0,
                            fill: "#ede9fe",
                            stroke: "#7c3aed",
                            stroke_width: 1.0,
                            dash: None,
                            shadow: true,
                        });
                        let title = wrap_lines(&mem.text, (mw / 7.8) as usize, 2);
                        for (li, l) in title.iter().enumerate() {
                            texts.push(tl(
                                r.x + 12.0,
                                r.y + 26.0 + 16.0 * li as f64,
                                l.clone(),
                                "card-title",
                            ));
                        }
                        if let Some(d) = &mem.detail {
                            if d.trim().eq_ignore_ascii_case("new") {
                                ppills.push(Pill {
                                    r: R::new(r.x + 12.0, r.y + 52.0, 44.0, 18.0),
                                    rx: 6.0,
                                    fill: "#22c55e".into(),
                                    stroke: None,
                                    text: tc(r.x + 34.0, r.y + 65.0, "NEW", "label"),
                                });
                            } else {
                                texts.push(tl(
                                    r.x + 12.0,
                                    r.y + 26.0 + 16.0 * title.len() as f64 + 4.0,
                                    d.clone(),
                                    "small",
                                ));
                            }
                        }
                    }
                    inner += 128.0 + 12.0;
                }
            }
            let r = R::new(px, bottom, pw, inner - bottom + 16.0);
            row_h = row_h.max(r.h);
            panels.push(Panel {
                r,
                color,
                title: tl(px + 16.0, bottom + 23.0, g.label.clone(), "heading"),
                shapes,
                texts,
                pills: ppills,
                arrows,
            });
        }
        for p in panels.iter_mut().rev().take(row.len()) {
            p.r.h = row_h;
        }
        bottom += row_h + 24.0;
    }

    let height = bottom + 40.0;
    let title_text = notes
        .title
        .clone()
        .or_else(|| board.title.clone())
        .unwrap_or_else(|| "Meeting board".into());
    let subtitle = format!(
        "Architecture as drawn on the board during the meeting (final state at {})",
        mmss(board.final_t_s)
    );
    let footer = format!(
        "Source: board read from the meeting recording (final state at {}); owners from name tags on the board; deferrals and the banner from decisions validated against the transcript.",
        mmss(board.final_t_s)
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
        lx += lead + text_width(label, 12.0, false) + 28.0;
    };
    add("solid", "Request (caller to callee)", 48.0);
    if board.edges.iter().any(|e| e.style == EdgeStyle::Dashed) {
        add("dashed", "Proposed relationship", 48.0);
    }
    if !board.owner_assignments.is_empty() {
        add("owner", "Owner tag on the board", 72.0);
    }
    if !deferred.is_empty() {
        add("deferred", "Deferred by a decision", 84.0);
    }
    if !board.stickies.is_empty() {
        add("sticky", "Board sticky", 28.0);
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

    Scene {
        width: width.round(),
        height: height.round(),
        title: tc(width / 2.0, 40.0, sanitize_dashes(&title_text), "title"),
        subtitle: tc(width / 2.0, 62.0, subtitle, "subtitle"),
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
        footer: tc(width / 2.0, height - 24.0, footer, "small"),
        layout_method,
        blocking,
    }
}

/// Pairs of blocking boxes that overlap.
pub fn overlaps(scene: &Scene) -> Vec<String> {
    let b = &scene.blocking;
    let mut out = Vec::new();
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
    fn arrowhead_points_at_the_tip() {
        let (base, pts) = arrowhead((100.0, 50.0), (0.0, 50.0));
        assert_eq!(base, (89.0, 50.0));
        assert_eq!(pts, "100,50 89,56 89,44");
    }
}
