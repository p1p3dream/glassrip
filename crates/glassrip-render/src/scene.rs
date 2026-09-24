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
    /// Owner pills, notes and badges that found no free place (left out).
    #[serde(skip)]
    pub unplaced: Vec<String>,
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
pub fn build_scene(board: &BoardStateItem, notes: &MeetingNotes) -> Scene {
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
        let max_title = ((CARD_W - 24.0) / (14.0 * 0.6)) as usize;
        if title.chars().count() > max_title {
            let parts = wrap(&title, max_title);
            title = parts.first().cloned().unwrap_or_default();
            let rest = parts[1..].join(" ");
            for l in wrap_lines(&rest, ((CARD_W - 24.0) / (12.0 * 0.6)) as usize, 2) {
                lines.push(tl(r.x + 12.0, y, l, "body-strong"));
                y += 20.0;
            }
        }
        if !mono.is_empty() {
            let mono = wrap_lines(
                &mono.join(" "),
                ((CARD_W - 24.0) / (11.0 * 0.62)) as usize,
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
    // Zone titles too: an edge label placed on one hides it.
    taken.extend(zones.iter().map(|z| zone_title_r(z).inflate(2.0)));
    let mut edges = Vec::new();
    let mut edge_anchor: BTreeMap<String, (f64, f64, bool)> = BTreeMap::new();
    let mut channel = 0usize;
    let mut segments: Vec<((f64, f64), (f64, f64))> = Vec::new();
    let final_edges = board.final_edges();
    for e in &final_edges {
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
        for w in pts.windows(2) {
            segments.push((w[0], w[1]));
        }
        segments.push((pts[n - 1], tip));
        segments.push((start, pts[0]));
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
        edge_anchor.insert(e.id.clone(), (mid.0, mid.1, horizontal));
        let label = Some(sanitize_dashes(e.label.trim()))
            .filter(|t| !t.is_empty())
            .map(|text| {
                let relation = dashed || text.chars().count() > 22;
                let (h, tw) = if relation {
                    (24.0, text_width(&text, 12.0, false) + 24.0)
                } else {
                    (18.0, text_width(&text, 11.0, true) + 16.0)
                };
                let around = |mid: (f64, f64), horizontal: bool| -> Vec<(f64, f64)> {
                    if via_channel || (relation && horizontal) {
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
                    }
                };
                // The middle of the longest segment first, then points sliding
                // along every segment (longest first) when that spot is taken
                // (a card, another label, a zone title or badge).
                let mut candidates = around(mid, horizontal);
                let mut order: Vec<usize> = (0..pts.len() - 1).collect();
                let seg_len =
                    |i: usize| (pts[i + 1].0 - pts[i].0).abs() + (pts[i + 1].1 - pts[i].1).abs();
                order.sort_by(|a, b| seg_len(*b).total_cmp(&seg_len(*a)));
                for i in order {
                    let (a, b) = (pts[i], pts[i + 1]);
                    let hz = (a.1 - b.1).abs() < 0.5;
                    for t in [0.5, 0.3, 0.7, 0.15, 0.85] {
                        let m = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
                        candidates.extend(around(m, hz));
                    }
                }
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
            arrows,
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
    // inside the canvas (below the legend) and clear of cards, labels and edges
    let free = |r: &R, taken: &[R]| {
        r.x >= MARGIN / 2.0
            && r.right() <= width - MARGIN / 2.0
            && r.y >= LEGEND_BOTTOM
            && !taken.iter().any(|t| t.intersects(r))
            && !segments
                .iter()
                .any(|(a, b)| r.inflate(3.0).hits_segment(*a, *b))
    };
    // first free candidate, else None (the item is left out and reported)
    let first_free = |cands: &[R], taken: &[R]| cands.iter().copied().find(|c| free(c, taken));
    let mut per_target: BTreeMap<String, Vec<&OwnerAssignment>> = BTreeMap::new();
    for o in board.current_owners() {
        let key = match &o.target {
            OwnerTarget::Node { node_id, .. } => node_id.clone(),
            OwnerTarget::Edge { edge_id, .. } => edge_id.clone(),
        };
        per_target.entry(key).or_default().push(o);
    }
    let mut unplaced: Vec<String> = Vec::new();
    for (target, owners) in per_target {
        for o in owners {
            let name = sanitize_dashes(&short_name(notes, &o.person_id, &o.display_name));
            let proto = pill(0.0, 0.0, &name, "#16a34a", "pill-text", 11.0, true);
            let (w, h) = (proto.r.w, proto.r.h);
            let cands: Vec<R> = match &o.target {
                OwnerTarget::Node { .. } => {
                    let Some(r) = card_of.get(target.as_str()) else {
                        continue;
                    };
                    // above the card, sliding right (bounded), then below it
                    let mut v: Vec<R> = (0..16)
                        .map(|k| R::new(r.x + 12.0 * k as f64, r.y - 34.0, w, h))
                        .collect();
                    v.extend((0..4).map(|k| R::new(r.x + 12.0 * k as f64, r.bottom() + 8.0, w, h)));
                    v
                }
                OwnerTarget::Edge { .. } => {
                    let Some((mx, my, horizontal)) = edge_anchor.get(target.as_str()).copied()
                    else {
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
                    v
                }
            };
            let Some(pr) = first_free(&cands, &taken) else {
                unplaced.push(format!("owner {name}"));
                continue;
            };
            let p = pill(pr.x, pr.y, &name, "#16a34a", "pill-text", 11.0, true);
            taken.push(p.r);
            if let Some(from) = &o.moved_from {
                let txt = format!(
                    "moved from {} ({})",
                    sanitize_dashes(&target_text(from)),
                    mmss(o.valid_from_s)
                );
                let tw = text_width(&txt, 10.0, false);
                let spots = [
                    R::new(p.r.right() + 8.0, p.r.y + 4.0, tw, 16.0),
                    R::new(p.r.x, p.r.y - 20.0, tw, 16.0),
                    R::new(p.r.right() - tw, p.r.y - 20.0, tw, 16.0),
                    R::new(p.r.x, p.r.bottom() + 4.0, tw, 16.0),
                ];
                match first_free(&spots, &taken) {
                    Some(nb) => {
                        notes_txt.push(tl(nb.x, nb.y + 12.0, txt, "small"));
                        taken.push(nb);
                        note_boxes.push(nb);
                    }
                    None => unplaced.push(format!("note {txt}")),
                }
            }
            pills.push(p);
        }
    }
    for (id, t) in &deferred {
        let Some(r) = card_of.get(id.as_str()) else {
            continue;
        };
        let text = format!("DEFERRED ({})", mmss(*t));
        let mut b = badge(0.0, 0.0, &text, "#ef4444");
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
        let Some(br) = first_free(&cands, &taken) else {
            unplaced.push(text);
            continue;
        };
        b.r = br;
        b.text.x = b.r.cx();
        b.text.y = b.r.y + 16.0;
        taken.push(b.r);
        pills.push(b);
    }

    // stickies
    let mut stickies = Vec::new();
    let mut sorted = board.final_stickies();
    sorted.sort_by(|a, b| first_seen(&a.lifetimes).total_cmp(&first_seen(&b.lifetimes)));
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
        for (k, m) in g.members.iter().enumerate() {
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
            let lines = wrap_lines(&sanitize_dashes(&m.text), (cw / 7.2) as usize, 2);
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
        lx += lead + text_width(label, 12.0, false) + 28.0;
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
        unplaced,
    }
}

/// Box of a zone's title text (`section-label`: 18 px bold, baseline 28 px below
/// the zone's top, starting 20 px in).
fn zone_title_r(z: &Zone) -> R {
    R::new(
        z.r.x + 20.0,
        z.r.y + 12.0,
        text_width(&z.label.text, 18.0, true),
        20.0,
    )
}

/// Pairs of blocking boxes that overlap.
pub fn overlaps(scene: &Scene) -> Vec<String> {
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
    for u in &scene.unplaced {
        out.push(format!("{u} could not be placed without overlapping"));
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
    fn arrowhead_points_at_the_tip() {
        let (base, pts) = arrowhead((100.0, 50.0), (0.0, 50.0));
        assert_eq!(base, (89.0, 50.0));
        assert_eq!(pts, "100,50 89,56 89,44");
    }
}
