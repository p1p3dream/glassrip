//! Degenerate readings: a complete board reply that lists one text at many places.
//!
//! With a large output budget the model can fill a list towards its `maxItems`
//! with one element under fresh ids and boxes (one box text 28 times, 60
//! stickies holding five texts). Such a reply is valid JSON, so the repetition
//! guard (which stops a runaway *generation*) lets it through. This module judges
//! a *finished* reading in two steps: [`detect`] flags a reading worth one retry,
//! and [`collapse`] removes copies only where the reply shows it made them up.
//!
//! Per list (nodes, stickies, owner tags, other text; edges keyed by endpoints
//! and label), for every normalized text:
//!
//! - `count`: items with that text;
//! - `supported`: items with an OCR span of that text of their own inside their
//!   box (each span backs the nearest matching item, and backs nothing when that
//!   item is already backed or a span of its text at the same place backed an
//!   item: stepped copies around one real box share a single span);
//! - `repeats = count - max(supported, 1) + 1`: the copies that neither OCR nor a
//!   single original explains, plus that original.
//!
//! A text is **repeated** when `repeats >= min_repeats` and the list's excess
//! copies (`repeats - 1`, summed over the texts that reach `min_repeats` and are
//! stacked or reach `strong_repeats`; pairs and triples never count) make up at
//! least `min_duplicate_share` of the list, or when the text is **stacked** and
//! `repeats >= strong_repeats`; copies side by side need both the share and
//! `strong_repeats`. A text is stacked when at least
//! `min_repeats` of its boxes each cover another box of the same text by
//! [`STACK_OVERLAP`] of the smaller one (the stepped or piled copies a runaway
//! reply writes); edges, which have no box, count as stacked. Two "API" boxes or
//! three "TODO" stickies never reach `min_repeats`; four "TODO" stickies among a
//! dozen notes stay under the share, however many other labels come in pairs.
//!
//! A repeated text only earns the retry. [`collapse`] removes copies of it only
//! where the reply shows it fabricated them ([`Fabrication`]), and only those:
//!
//! - copies piled on each other: each pile keeps its best-supported copy;
//! - at least `min_repeats` copies lying wholly outside the image the request
//!   sent: those copies go, OCR-backed or not (nothing outside the image was
//!   read from it; a tile's OCR anchors cover the whole canvas).
//!
//! Every other copy stays as read, however many there are: a row of identical
//! cards, or eight "Avery" tags on eight nodes, that OCR misses is still a row,
//! and real copies of a text keep their place beside fabricated ones. A list
//! that ran to its `maxItems` is no evidence either: thirty separate "TODO"
//! notes fill a compact list as surely as a runaway does. Separate copies a
//! runaway spreads over the canvas are left to the consensus vote of
//! `board_read`, where the other reads confirm at most one copy per real
//! element.
//!
//! Best-supported items: every item backed by its own OCR span, or the first
//! item when OCR backs none. Edges and owner tags pointing at a removed node
//! move to the nearest node of its text that was kept.

use std::collections::{HashMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::board::{normalize, BoardReading, ElementList};
use crate::geometry::BBox;

/// Share of the smaller box that two boxes of one text must share for either
/// to count as a copy stacked on the other.
pub const STACK_OVERLAP: f64 = 0.5;

/// Thresholds of the degenerate-reading rule.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DegenerateParams {
    /// Unexplained copies of one text (the original included) in one list that
    /// can make the list degenerate; `0` disables the rule.
    pub min_repeats: usize,
    /// ...when the list's excess copies make up at least this share of it...
    pub min_duplicate_share: f64,
    /// ...or whatever the share, at this many copies (`0`: no such bound).
    pub strong_repeats: usize,
    /// An OCR span backs an item when its center lies inside the item's box grown
    /// by this fraction of the box's width and height on each side.
    pub anchor_margin: f64,
}

impl Default for DegenerateParams {
    fn default() -> Self {
        Self {
            min_repeats: 4,
            min_duplicate_share: 0.5,
            strong_repeats: 8,
            anchor_margin: 0.25,
        }
    }
}

impl DegenerateParams {
    pub fn enabled(&self) -> bool {
        self.min_repeats >= 2
    }
}

/// What a reply was asked to read: the evidence [`collapse`] weighs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReplyFrame {
    /// The image the request sent, in the reading's (canvas) pixels.
    pub extent: BBox,
}

/// Why the copies of a repeated text were made up by the reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Fabrication {
    /// At least `min_repeats` copies each cover another copy (every repeated
    /// edge: the same endpoints and label); each pile keeps one copy.
    Piled,
    /// At least `min_repeats` copies lie wholly outside the image; they go,
    /// OCR-backed or not (a tile's OCR anchors cover the whole canvas).
    OffCanvas,
    /// No longer produced (a full list is no evidence of fabrication); kept so
    /// that readings recorded before `board_read` version 10 still parse.
    ListCap,
}

/// One repeated text in one list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RepeatedText {
    pub list: ElementList,
    /// Normalized text (edges: `src -> dst: label`).
    pub text: String,
    /// Items with this text.
    pub count: usize,
    /// Items backed by an OCR span of their own.
    pub supported: usize,
    /// Items in the list.
    pub list_len: usize,
}

/// Why a reading is degenerate: every repeated text, in list order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DegenerateFinding {
    pub repeated: Vec<RepeatedText>,
}

impl std::fmt::Display for DegenerateFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, r) in self.repeated.iter().enumerate() {
            if i > 0 {
                f.write_str("; ")?;
            }
            let text: String = r.text.chars().take(60).collect();
            write!(
                f,
                "{:?} {text:?} x{} of {} ({} OCR-backed)",
                r.list, r.count, r.list_len, r.supported
            )?;
        }
        Ok(())
    }
}

/// A repeated text reduced to its best-supported items.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CollapsedText {
    pub list: ElementList,
    pub text: String,
    pub before: usize,
    pub after: usize,
    /// What showed the copies were made up.
    #[serde(default)]
    pub evidence: Vec<Fabrication>,
}

/// OCR text that belongs to an element: the same text (at any length), a
/// similar one, or one line of it.
pub fn span_matches(element: &str, span: &str) -> bool {
    let (e, s) = (normalize(element), normalize(span));
    if s.is_empty() {
        return false;
    }
    e == s
        || (s.chars().count() >= 3
            && (strsim::normalized_levenshtein(&e, &s) >= 0.8
                || (s.chars().count() >= 4 && e.contains(&s))))
}

fn center(b: &BBox) -> (f64, f64) {
    ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
}

fn distance2(a: &BBox, b: &BBox) -> f64 {
    let ((ax, ay), (bx, by)) = (center(a), center(b));
    (ax - bx).powi(2) + (ay - by).powi(2)
}

/// One list item as the rule sees it.
struct Item<'a> {
    key: String,
    /// Text and box an OCR span can back (edges have none).
    anchorable: Option<(&'a str, &'a BBox)>,
}

fn center_inside(span: &BBox, b: &BBox, margin: f64) -> bool {
    let (cx, cy) = ((span.x1 + span.x2) / 2.0, (span.y1 + span.y2) / 2.0);
    let (mx, my) = (b.width() * margin, b.height() * margin);
    cx >= b.x1 - mx && cx <= b.x2 + mx && cy >= b.y1 - my && cy <= b.y2 + my
}

/// Two OCR spans of one text at the same place (by IoU) are one text read twice.
pub const REREAD_IOU: f64 = 0.7;

/// Per item: backed by an OCR span of its own. Each span resolves to one item:
/// the nearest (by box center) matching item whose grown box holds the span.
/// The span backs that item unless the item is backed already, or the span
/// lies where a span of the same text that backed an item lies
/// ([`REREAD_IOU`]): then it is a second reading of that text and backs nothing.
/// A span never falls through to another item.
fn supported(items: &[Item<'_>], anchors: &[(String, BBox)], margin: f64) -> Vec<bool> {
    let mut backed = vec![false; items.len()];
    // Spans that backed an item: normalized text and box.
    let mut used: Vec<(String, &BBox)> = Vec::new();
    for (s, sb) in anchors {
        let best = items
            .iter()
            .enumerate()
            .filter_map(|(i, it)| {
                let (text, bbox) = it.anchorable?;
                (span_matches(text, s) && center_inside(sb, bbox, margin))
                    .then(|| (i, distance2(sb, bbox)))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let Some((i, _)) = best else {
            continue;
        };
        let ns = normalize(s);
        let reread = backed[i]
            || used
                .iter()
                .any(|(t, ub)| *t == ns && ub.iou(sb) >= REREAD_IOU);
        if !reread {
            backed[i] = true;
            used.push((ns, sb));
        }
    }
    backed
}

/// Two well-formed boxes (positive width and height) cover each other by
/// [`STACK_OVERLAP`] of the smaller one. A box without area piles on nothing.
fn stacked_pair(a: &BBox, b: &BBox) -> bool {
    let formed = |b: &BBox| b.x2 - b.x1 > 0.0 && b.y2 - b.y1 > 0.0;
    if !(formed(a) && formed(b)) {
        return false;
    }
    let w = a.x2.min(b.x2) - a.x1.max(b.x1);
    let h = a.y2.min(b.y2) - a.y1.max(b.y1);
    let area = |b: &BBox| (b.x2 - b.x1) * (b.y2 - b.y1);
    w > 0.0 && h > 0.0 && w * h >= STACK_OVERLAP * area(a).min(area(b))
}

/// Items of one text whose box is stacked on another box of that text; every
/// item when the list has no boxes (edges).
fn stacked(items: &[Item<'_>], idx: &[usize]) -> usize {
    let boxes: Vec<Option<&BBox>> = idx
        .iter()
        .map(|&i| items[i].anchorable.map(|(_, b)| b))
        .collect();
    (0..boxes.len())
        .filter(|&a| match boxes[a] {
            None => true,
            Some(ba) => {
                (0..boxes.len()).any(|b| b != a && boxes[b].is_some_and(|bb| stacked_pair(ba, bb)))
            }
        })
        .count()
}

/// A well-formed box (positive width and height) with no point inside the image
/// (touching its edge from outside counts as outside). A box without area is
/// nowhere, so it is not outside either.
fn outside(b: &BBox, extent: &BBox) -> bool {
    b.x2 > b.x1
        && b.y2 > b.y1
        && (b.x1 >= extent.x2 || b.x2 <= extent.x1 || b.y1 >= extent.y2 || b.y2 <= extent.y1)
}

/// Piles among `idx`: groups (two or more, in list order) of items whose boxes
/// cover each other, directly or through a chain of stepped copies; items with
/// no box (edges) all form one pile.
fn piles(items: &[Item<'_>], idx: &[usize]) -> Vec<Vec<usize>> {
    let mut pile_of: Vec<Option<usize>> = vec![None; idx.len()];
    let mut out: Vec<Vec<usize>> = Vec::new();
    for a in 0..idx.len() {
        if pile_of[a].is_some() {
            continue;
        }
        let mut members = vec![a];
        pile_of[a] = Some(out.len());
        let mut k = 0;
        while k < members.len() {
            let m = members[k];
            for b in 0..idx.len() {
                if pile_of[b].is_some() {
                    continue;
                }
                let on = match (items[idx[m]].anchorable, items[idx[b]].anchorable) {
                    (Some((_, x)), Some((_, y))) => stacked_pair(x, y),
                    (None, None) => true,
                    _ => false,
                };
                if on {
                    pile_of[b] = Some(out.len());
                    members.push(b);
                }
            }
            k += 1;
        }
        members.sort_unstable();
        out.push(members.into_iter().map(|j| idx[j]).collect());
    }
    out.retain(|p| p.len() >= 2);
    out
}

/// Groups of item indices by key, in order of first appearance.
fn groups(items: &[Item<'_>]) -> Vec<(String, Vec<usize>)> {
    let mut at: HashMap<&str, usize> = HashMap::new();
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, it) in items.iter().enumerate() {
        match at.get(it.key.as_str()) {
            Some(&g) => out[g].1.push(i),
            None => {
                at.insert(it.key.as_str(), out.len());
                out.push((it.key.clone(), vec![i]));
            }
        }
    }
    out
}

/// One repeated text of a list.
struct Repeat {
    text: RepeatedText,
    /// Its items.
    idx: Vec<usize>,
    /// Its items backed by an OCR span of their own.
    backed: Vec<usize>,
    piled: bool,
}

/// Repeated texts of one list.
fn list_repeats(
    list: ElementList,
    items: &[Item<'_>],
    anchors: &[(String, BBox)],
    p: &DegenerateParams,
) -> Vec<Repeat> {
    let n = items.len();
    if !p.enabled() || n < p.min_repeats {
        return Vec::new();
    }
    let backed = supported(items, anchors, p.anchor_margin);
    let gs = groups(items);
    let repeats = |idx: &[usize]| {
        let s = idx.iter().filter(|&&i| backed[i]).count();
        (idx.len() - s.max(1) + 1, s)
    };
    let strong = |r: usize| p.strong_repeats > 0 && r >= p.strong_repeats;
    // Per group: (repeats, supported, stacked, strong).
    let judged: Vec<(usize, usize, bool, bool)> = gs
        .iter()
        .map(|(_, idx)| {
            let (r, s) = repeats(idx);
            let piled = r >= p.min_repeats && stacked(items, idx) >= p.min_repeats;
            (r, s, piled, strong(r))
        })
        .collect();
    let excess: usize = judged
        .iter()
        .filter(|(r, _, piled, strong)| *r >= p.min_repeats && (*piled || *strong))
        .map(|(r, ..)| r - 1)
        .sum();
    let share = excess as f64 / n as f64;
    gs.into_iter()
        .zip(judged)
        .filter_map(|((key, idx), (r, s, piled, strong))| {
            // Piled copies: the strong bound or the share. Copies side by side
            // (a runaway grid, or a board's real identical notes that OCR
            // missed): the strong bound and the share.
            let shared = share >= p.min_duplicate_share;
            let flagged = if piled {
                strong || shared
            } else {
                strong && shared
            };
            if r < p.min_repeats || !flagged {
                return None;
            }
            let own: Vec<usize> = idx.iter().copied().filter(|&i| backed[i]).collect();
            Some(Repeat {
                text: RepeatedText {
                    list,
                    text: key,
                    count: idx.len(),
                    supported: s,
                    list_len: n,
                },
                idx,
                backed: own,
                piled,
            })
        })
        .collect()
}

/// The best-supported of `members`: those backed by OCR, or the first.
fn best(r: &Repeat, members: &[usize]) -> Vec<usize> {
    let own: Vec<usize> = members
        .iter()
        .copied()
        .filter(|i| r.backed.contains(i))
        .collect();
    if own.is_empty() {
        members.iter().copied().take(1).collect()
    } else {
        own
    }
}

/// What shows the reply made up copies of one repeated text, and the items
/// that stay; `None` when nothing does (every copy stays as read).
fn plan(
    r: &Repeat,
    items: &[Item<'_>],
    frame: &ReplyFrame,
    p: &DegenerateParams,
) -> Option<(Vec<Fabrication>, Vec<usize>)> {
    let off: Vec<usize> = r
        .idx
        .iter()
        .copied()
        .filter(|&i| {
            items[i]
                .anchorable
                .is_some_and(|(_, b)| outside(b, &frame.extent))
        })
        .collect();
    let off_canvas = off.len() >= p.min_repeats;
    let evidence: Vec<Fabrication> = [
        (r.piled, Fabrication::Piled),
        (off_canvas, Fabrication::OffCanvas),
    ]
    .into_iter()
    .filter_map(|(on, f)| on.then_some(f))
    .collect();
    if evidence.is_empty() {
        return None;
    }
    // Keepers come from inside the image when any member is: a tile's OCR
    // anchors cover the whole canvas, so OCR can back a copy the tile never saw.
    let prefer_inside = |members: &[usize]| -> Vec<usize> {
        let inside: Vec<usize> = members
            .iter()
            .copied()
            .filter(|i| !off.contains(i))
            .collect();
        if inside.is_empty() {
            best(r, members)
        } else {
            best(r, &inside)
        }
    };
    // Copies outside the image go, OCR-backed or not.
    let mut gone: HashSet<usize> = if off_canvas {
        off.iter().copied().collect()
    } else {
        HashSet::new()
    };
    if r.piled {
        for pile in piles(items, &r.idx) {
            let left: Vec<usize> = pile.into_iter().filter(|i| !gone.contains(i)).collect();
            if left.is_empty() {
                continue;
            }
            let keep = prefer_inside(&left);
            gone.extend(left.into_iter().filter(|i| !keep.contains(i)));
        }
    }
    let keep: Vec<usize> = r
        .idx
        .iter()
        .copied()
        .filter(|i| !gone.contains(i))
        .collect();
    let keep = if keep.is_empty() {
        best(r, &r.idx)
    } else {
        keep
    };
    Some((evidence, keep))
}

fn edge_key(src: &str, dst: &str, label: &str) -> String {
    format!("{src} -> {dst}: {}", normalize(label))
}

fn node_items(r: &BoardReading) -> Vec<Item<'_>> {
    r.nodes
        .iter()
        .map(|n| Item {
            key: normalize(&n.text),
            anchorable: Some((n.text.as_str(), &n.bbox)),
        })
        .collect()
}

fn edge_items(r: &BoardReading) -> Vec<Item<'_>> {
    r.edges
        .iter()
        .map(|e| Item {
            key: edge_key(&e.src, &e.dst, &e.label),
            anchorable: None,
        })
        .collect()
}

fn sticky_items(r: &BoardReading) -> Vec<Item<'_>> {
    r.stickies
        .iter()
        .map(|s| Item {
            key: normalize(&s.text),
            anchorable: Some((s.text.as_str(), &s.bbox)),
        })
        .collect()
}

fn owner_items(r: &BoardReading) -> Vec<Item<'_>> {
    r.owner_tags
        .iter()
        .map(|o| Item {
            key: normalize(&o.name_raw),
            anchorable: Some((o.name_raw.as_str(), &o.bbox)),
        })
        .collect()
}

fn other_items(r: &BoardReading) -> Vec<Item<'_>> {
    r.other_visible_text
        .iter()
        .map(|t| Item {
            key: normalize(&t.text),
            anchorable: Some((t.text.as_str(), &t.bbox)),
        })
        .collect()
}

/// The repeated texts of a complete reading (worth one retry), or `None` when
/// it is sound. `anchors` are OCR spans in the reading's (canvas) pixels.
pub fn detect(
    reading: &BoardReading,
    anchors: &[(String, BBox)],
    p: &DegenerateParams,
) -> Option<DegenerateFinding> {
    let lists = [
        (ElementList::Nodes, node_items(reading)),
        (ElementList::Edges, edge_items(reading)),
        (ElementList::Stickies, sticky_items(reading)),
        (ElementList::OwnerTags, owner_items(reading)),
        (ElementList::OtherVisibleText, other_items(reading)),
    ];
    let repeated: Vec<RepeatedText> = lists
        .iter()
        .flat_map(|(list, items)| list_repeats(*list, items, anchors, p))
        .map(|r| r.text)
        .collect();
    (!repeated.is_empty()).then_some(DegenerateFinding { repeated })
}

/// One fabricated text: the repeat, its evidence, and the items that stay.
type Fabricated = (Repeat, Vec<Fabrication>, Vec<usize>);

/// The repeated texts of one list that the reply fabricated.
fn fabricated(
    list: ElementList,
    items: &[Item<'_>],
    anchors: &[(String, BBox)],
    frame: &ReplyFrame,
    p: &DegenerateParams,
) -> Vec<Fabricated> {
    list_repeats(list, items, anchors, p)
        .into_iter()
        .filter_map(|r| {
            let (ev, keep) = plan(&r, items, frame, p)?;
            Some((r, ev, keep))
        })
        .collect()
}

/// Indices to drop from one list: every fabricated text keeps only its chosen
/// items.
fn drops(repeats: &[Fabricated]) -> HashSet<usize> {
    repeats
        .iter()
        .flat_map(|(r, _, keep)| r.idx.iter().copied().filter(|i| !keep.contains(i)))
        .collect()
}

fn retain_indexed<T>(v: &mut Vec<T>, gone: &HashSet<usize>) {
    let mut i = 0;
    v.retain(|_| {
        let keep = !gone.contains(&i);
        i += 1;
        keep
    });
}

fn record(out: &mut Vec<CollapsedText>, repeats: &[Fabricated]) {
    out.extend(repeats.iter().map(|(r, ev, keep)| CollapsedText {
        list: r.text.list,
        text: r.text.text.clone(),
        before: r.text.count,
        after: keep.len(),
        evidence: ev.clone(),
    }));
}

/// Remove the copies the reply fabricated (see [`Fabrication`]); repeated texts
/// without such evidence, and copies the evidence does not reach, stay as read.
/// Returns the reading unchanged, with no record, when nothing is fabricated.
pub fn collapse(
    mut reading: BoardReading,
    anchors: &[(String, BBox)],
    frame: &ReplyFrame,
    p: &DegenerateParams,
) -> (BoardReading, Vec<CollapsedText>) {
    let mut done = Vec::new();

    // Nodes first: edges and owner tags follow a removed copy to the nearest
    // kept node of its text.
    let items = node_items(&reading);
    let repeats = fabricated(ElementList::Nodes, &items, anchors, frame, p);
    let gone = drops(&repeats);
    let mut moved: HashMap<String, String> = HashMap::new();
    if !gone.is_empty() {
        let kept_ids: HashSet<&str> = reading
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| !gone.contains(i))
            .map(|(_, n)| n.local_id.as_str())
            .collect();
        for (i, n) in reading.nodes.iter().enumerate() {
            if !gone.contains(&i) || kept_ids.contains(n.local_id.as_str()) {
                continue;
            }
            let key = normalize(&n.text);
            let nearest = reading
                .nodes
                .iter()
                .enumerate()
                .filter(|(j, k)| !gone.contains(j) && normalize(&k.text) == key)
                .map(|(j, k)| (j, distance2(&n.bbox, &k.bbox), &k.local_id))
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            if let Some((_, _, to)) = nearest {
                moved.insert(n.local_id.clone(), to.clone());
            }
        }
        record(&mut done, &repeats);
        retain_indexed(&mut reading.nodes, &gone);
    }
    if !moved.is_empty() {
        let mut edges: Vec<(bool, crate::board::BoardEdge)> = std::mem::take(&mut reading.edges)
            .into_iter()
            .map(|mut e| {
                let remapped = moved.contains_key(&e.src) || moved.contains_key(&e.dst);
                if let Some(to) = moved.get(&e.src) {
                    e.src = to.clone();
                }
                if let Some(to) = moved.get(&e.dst) {
                    e.dst = to.clone();
                }
                (remapped, e)
            })
            .collect();
        // An edge between two copies of one box is no edge.
        edges.retain(|(remapped, e)| !(*remapped && e.src == e.dst));
        // Edges that now coincide with a remapped copy are one edge: keep the
        // first. Coinciding edges no remap touched stay as the model read them.
        let touched: HashSet<String> = edges
            .iter()
            .filter(|(r, _)| *r)
            .map(|(_, e)| edge_key(&e.src, &e.dst, &e.label))
            .collect();
        let mut seen: HashSet<String> = HashSet::new();
        edges.retain(|(_, e)| {
            let key = edge_key(&e.src, &e.dst, &e.label);
            !touched.contains(&key) || seen.insert(key)
        });
        let kept: Vec<crate::board::BoardEdge> = edges.into_iter().map(|(_, e)| e).collect();
        reading.edges = kept;
        for o in &mut reading.owner_tags {
            if let Some(to) = moved.get(&o.near) {
                o.near = to.clone();
            }
        }
    }

    let items = edge_items(&reading);
    let repeats = fabricated(ElementList::Edges, &items, anchors, frame, p);
    let gone = drops(&repeats);
    record(&mut done, &repeats);
    retain_indexed(&mut reading.edges, &gone);

    let items = sticky_items(&reading);
    let repeats = fabricated(ElementList::Stickies, &items, anchors, frame, p);
    let gone = drops(&repeats);
    record(&mut done, &repeats);
    retain_indexed(&mut reading.stickies, &gone);

    let items = owner_items(&reading);
    let repeats = fabricated(ElementList::OwnerTags, &items, anchors, frame, p);
    let gone = drops(&repeats);
    record(&mut done, &repeats);
    retain_indexed(&mut reading.owner_tags, &gone);

    let items = other_items(&reading);
    let repeats = fabricated(ElementList::OtherVisibleText, &items, anchors, frame, p);
    let gone = drops(&repeats);
    record(&mut done, &repeats);
    retain_indexed(&mut reading.other_visible_text, &gone);

    (reading, done)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::board::{BoardEdge, BoardNode, EdgeStyle, OwnerTag, Sticky, StickyColor, TextItem};

    // Every board below is fictional and generated here.

    fn bb(x: f64, y: f64) -> BBox {
        BBox::new(x, y, x + 100.0, y + 50.0)
    }

    fn node(id: &str, text: &str, x: f64, y: f64) -> BoardNode {
        BoardNode {
            local_id: id.into(),
            text: text.into(),
            bbox: bb(x, y),
            conf: 0.9,
        }
    }

    fn sticky(text: &str, x: f64, y: f64) -> Sticky {
        Sticky {
            text: text.into(),
            color: StickyColor::Yellow,
            bbox: bb(x, y),
        }
    }

    fn edge(src: &str, dst: &str, label: &str) -> BoardEdge {
        BoardEdge {
            src: src.into(),
            dst: dst.into(),
            label: label.into(),
            label_bbox: None,
            style: EdgeStyle::Solid,
            conf: 0.8,
        }
    }

    fn reading() -> BoardReading {
        BoardReading {
            nodes: vec![],
            edges: vec![],
            stickies: vec![],
            owner_tags: vec![],
            other_visible_text: vec![],
            confidence: 0.8,
        }
    }

    /// An OCR span at the center of `bb(x, y)`.
    fn span(text: &str, x: f64, y: f64) -> (String, BBox) {
        (
            text.into(),
            BBox::new(x + 20.0, y + 15.0, x + 80.0, y + 35.0),
        )
    }

    fn distinct_nodes(n: usize) -> Vec<BoardNode> {
        (0..n)
            .map(|i| {
                node(
                    &format!("n{}", i + 1),
                    &format!("Service {i}"),
                    (i % 8) as f64 * 150.0,
                    (i / 8) as f64 * 100.0,
                )
            })
            .collect()
    }

    fn p() -> DegenerateParams {
        DegenerateParams::default()
    }

    /// A large image: only piled copies are evidence.
    fn frame() -> ReplyFrame {
        ReplyFrame {
            extent: BBox::new(0.0, 0.0, 10_000.0, 10_000.0),
        }
    }

    // Legitimate boards: never flagged, never changed.

    #[test]
    fn a_few_repeated_labels_pass_untouched() {
        let mut r = reading();
        r.nodes = distinct_nodes(5);
        r.nodes.push(node("n6", "API", 0.0, 400.0));
        r.nodes.push(node("n7", "API", 300.0, 400.0));
        r.edges = vec![edge("n1", "n6", "REST"), edge("n2", "n7", "REST")];
        r.stickies = vec![
            sticky("TODO", 0.0, 600.0),
            sticky("TODO", 150.0, 600.0),
            sticky("TODO", 300.0, 600.0),
            sticky("Ship on Friday?", 450.0, 600.0),
        ];
        assert_eq!(detect(&r, &[], &p()), None);
        let (after, done) = collapse(r.clone(), &[], &frame(), &p());
        assert_eq!(after, r);
        assert!(done.is_empty());
    }

    #[test]
    fn four_todo_notes_among_a_dozen_pass() {
        let mut r = reading();
        for i in 0..8 {
            r.stickies
                .push(sticky(&format!("Idea {i}"), f64::from(i) * 120.0, 0.0));
        }
        for i in 0..4 {
            r.stickies.push(sticky("TODO", f64::from(i) * 120.0, 200.0));
        }
        assert_eq!(detect(&r, &[], &p()), None);
    }

    #[test]
    fn four_todo_notes_among_paired_labels_pass() {
        // Pairs never count towards the share: 3 excess copies of 12, not 7.
        let mut r = reading();
        for i in 0..4 {
            r.stickies.push(sticky("TODO", f64::from(i) * 120.0, 0.0));
        }
        for (i, t) in ["Cache", "Queue", "Retry", "Owner?"].iter().enumerate() {
            for j in 0..2 {
                r.stickies
                    .push(sticky(t, (i * 2 + j) as f64 * 120.0, 200.0));
            }
        }
        assert_eq!(r.stickies.len(), 12);
        assert_eq!(detect(&r, &[], &p()), None);
    }

    #[test]
    fn a_card_row_that_ocr_reads_at_every_card_passes() {
        let mut r = reading();
        let mut anchors = Vec::new();
        for i in 0..6 {
            let x = f64::from(i) * 150.0;
            r.nodes.push(node(&format!("n{}", i + 1), "Card", x, 100.0));
            anchors.push(span("Card", x, 100.0));
        }
        assert_eq!(detect(&r, &anchors, &p()), None);
        // Without OCR the row is still six cards side by side, not copies.
        assert_eq!(detect(&r, &[], &p()), None);
        let (after, done) = collapse(r.clone(), &[], &frame(), &p());
        assert_eq!(after, r);
        assert!(done.is_empty());
    }

    #[test]
    fn identical_cards_that_ocr_misses_keep_their_edges_and_owners() {
        // Four separate "Card" boxes and two others, OCR silent: 3 excess of 6
        // reaches the share, but the cards are side by side, not piled.
        let mut r = reading();
        r.nodes = distinct_nodes(2);
        for i in 0..4 {
            r.nodes
                .push(node(&format!("c{i}"), "Card", f64::from(i) * 150.0, 300.0));
        }
        r.edges = (0..4).map(|i| edge("n1", &format!("c{i}"), "")).collect();
        r.owner_tags = vec![OwnerTag {
            name_raw: "Rowan".into(),
            near: "c3".into(),
            bbox: bb(450.0, 380.0),
        }];
        assert_eq!(detect(&r, &[], &p()), None);
        let (after, done) = collapse(r.clone(), &[], &frame(), &p());
        assert_eq!(after, r);
        assert!(done.is_empty());
    }

    #[test]
    fn piled_copies_without_ocr_are_flagged_below_the_strong_bound() {
        // Five copies stepped 10 px over one box among three real ones.
        let mut r = reading();
        r.nodes = distinct_nodes(3);
        for k in 0..5 {
            r.nodes.push(node(
                &format!("c{k}"),
                "Billing",
                600.0 + f64::from(k) * 10.0,
                400.0,
            ));
        }
        let f = detect(&r, &[], &p()).expect("piled copies");
        assert_eq!((f.repeated[0].count, f.repeated[0].list_len), (5, 8));
        let (after, _) = collapse(r, &[], &frame(), &p());
        assert_eq!(after.nodes.len(), 4);
        assert!(!stacked_pair(&bb(0.0, 0.0), &bb(60.0, 0.0)));
        assert!(stacked_pair(&bb(0.0, 0.0), &bb(40.0, 0.0)));
        // Equal boxes without area, or inverted, pile on nothing.
        let dot = BBox::new(5.0, 5.0, 5.0, 5.0);
        assert!(!stacked_pair(&dot, &dot));
        let inverted = BBox {
            x1: 100.0,
            y1: 50.0,
            x2: 0.0,
            y2: 0.0,
        };
        assert!(!stacked_pair(&inverted, &inverted));
    }

    #[test]
    fn repeated_edge_labels_between_distinct_nodes_pass() {
        let mut r = reading();
        r.nodes = distinct_nodes(9);
        r.edges = (2..=9)
            .map(|i| edge("n1", &format!("n{i}"), "gRPC"))
            .collect();
        assert_eq!(detect(&r, &[], &p()), None);
    }

    #[test]
    fn disabled_rule_flags_nothing() {
        let mut r = reading();
        for i in 0..20 {
            r.stickies.push(sticky("Same", f64::from(i) * 10.0, 0.0));
        }
        let off = DegenerateParams {
            min_repeats: 0,
            ..p()
        };
        assert_eq!(detect(&r, &[], &off), None);
        assert!(collapse(r, &[], &frame(), &off).1.is_empty());
    }

    // Degenerate readings.

    /// Seven real boxes, then one box restated 28 times under fresh ids, with
    /// an edge from a real box to every copy and an owner tag on a copy.
    fn runaway_node_list() -> (BoardReading, Vec<(String, BBox)>) {
        let mut r = reading();
        r.nodes = distinct_nodes(7);
        for k in 0..28 {
            r.nodes.push(node(
                &format!("n{}", 8 + k),
                "Mobile App",
                600.0 + f64::from(k) * 4.0,
                500.0,
            ));
        }
        r.edges = (0..28)
            .map(|k| edge("n1", &format!("n{}", 8 + k), "HTTPS"))
            .collect();
        r.edges.push(edge("n2", "n3", ""));
        r.owner_tags = vec![OwnerTag {
            name_raw: "Rowan".into(),
            near: "n20".into(),
            bbox: bb(800.0, 600.0),
        }];
        // OCR reads the real box once, at the 5th copy's place.
        let anchors = vec![
            span("Mobile App", 616.0, 500.0),
            span("Service 0", 0.0, 0.0),
        ];
        (r, anchors)
    }

    #[test]
    fn a_runaway_node_list_is_flagged_and_collapsed_to_the_ocr_backed_copy() {
        let (r, anchors) = runaway_node_list();
        let f = detect(&r, &anchors, &p()).expect("degenerate");
        assert_eq!(f.repeated.len(), 1, "{f}");
        let rep = &f.repeated[0];
        assert_eq!(
            (rep.list, rep.text.as_str(), rep.count, rep.supported),
            (ElementList::Nodes, "mobile app", 28, 1)
        );
        let (after, done) = collapse(r, &anchors, &frame(), &p());
        assert_eq!(after.nodes.len(), 8);
        let app: Vec<&BoardNode> = after
            .nodes
            .iter()
            .filter(|n| n.text == "Mobile App")
            .collect();
        assert_eq!(app.len(), 1);
        assert_eq!(app[0].local_id, "n12", "the OCR-backed copy is kept");
        // 28 edges to the copies become one edge to the kept box.
        assert_eq!(after.edges.len(), 2);
        assert_eq!(
            (after.edges[0].src.as_str(), after.edges[0].dst.as_str()),
            ("n1", "n12")
        );
        assert_eq!(after.owner_tags[0].near, "n12");
        assert_eq!(
            done,
            vec![CollapsedText {
                list: ElementList::Nodes,
                text: "mobile app".into(),
                before: 28,
                after: 1,
                evidence: vec![Fabrication::Piled],
            }]
        );
        assert_eq!(detect(&after, &anchors, &p()), None);
    }

    #[test]
    fn without_ocr_the_first_copy_is_kept() {
        let (r, _) = runaway_node_list();
        let (after, _) = collapse(r, &[], &frame(), &p());
        let app: Vec<&BoardNode> = after
            .nodes
            .iter()
            .filter(|n| n.text == "Mobile App")
            .collect();
        assert_eq!(app.len(), 1);
        assert_eq!(app[0].local_id, "n8");
    }

    #[test]
    fn sixty_stickies_with_five_texts_are_flagged() {
        let mut r = reading();
        let texts = [
            "Latency budget",
            "Retry policy",
            "Owner?",
            "Cache TTL",
            "SLO",
        ];
        for i in 0..60u32 {
            r.stickies.push(sticky(
                texts[(i % 5) as usize],
                f64::from(i % 10) * 110.0,
                f64::from(i / 10) * 60.0,
            ));
        }
        let anchors: Vec<(String, BBox)> = (0..5u32)
            .map(|i| span(texts[i as usize], f64::from(i) * 110.0, 0.0))
            .collect();
        let f = detect(&r, &anchors, &p()).expect("degenerate");
        assert_eq!(f.repeated.len(), 5, "{f}");
        // Side by side on the canvas, the notes are retried but kept as read,
        // though they fill the sticky list to its limit of 60: a full list is
        // no evidence (the consensus vote judges a grid only one read has).
        let (after, done) = collapse(r.clone(), &anchors, &frame(), &p());
        assert_eq!(after, r);
        assert!(done.is_empty());
    }

    #[test]
    fn many_copies_are_flagged_even_in_a_long_list() {
        // Eight copies among 30 otherwise distinct notes: a low share, but past
        // the strong bound.
        let mut r = reading();
        for i in 0..22 {
            r.stickies
                .push(sticky(&format!("Note {i}"), f64::from(i) * 50.0, 0.0));
        }
        for i in 0..8 {
            r.stickies
                .push(sticky("Follow up", f64::from(i) * 50.0, 300.0));
        }
        let f = detect(&r, &[], &p()).expect("strong bound");
        assert_eq!(f.repeated[0].count, 8);
        // The notes step half their width: piled, so made up.
        let (after, done) = collapse(r.clone(), &[], &frame(), &p());
        assert_eq!(after.stickies.len(), 23);
        assert_eq!(done[0].evidence, [Fabrication::Piled]);
        let weak = DegenerateParams {
            strong_repeats: 0,
            ..p()
        };
        assert_eq!(detect(&r, &[], &weak), None);
    }

    #[test]
    fn repeated_owner_tags_and_other_text_are_flagged() {
        let mut r = reading();
        r.nodes = distinct_nodes(3);
        for i in 0..5 {
            r.owner_tags.push(OwnerTag {
                name_raw: "Rowan".into(),
                near: "n1".into(),
                bbox: bb(f64::from(i) * 30.0, 400.0),
            });
            r.other_visible_text.push(TextItem {
                text: "v2".into(),
                bbox: bb(f64::from(i) * 30.0, 700.0),
            });
        }
        let f = detect(&r, &[], &p()).expect("degenerate");
        let lists: Vec<ElementList> = f.repeated.iter().map(|x| x.list).collect();
        assert_eq!(
            lists,
            [ElementList::OwnerTags, ElementList::OtherVisibleText]
        );
        let (after, _) = collapse(r, &[], &frame(), &p());
        assert_eq!(after.owner_tags.len(), 1);
        assert_eq!(after.other_visible_text.len(), 1);
        assert_eq!(after.nodes.len(), 3);
    }

    #[test]
    fn short_labels_read_by_ocr_at_every_box_pass() {
        let mut r = reading();
        let mut anchors = Vec::new();
        for i in 0..4 {
            let x = f64::from(i) * 150.0;
            r.nodes.push(node(&format!("n{}", i + 1), "OK", x, 100.0));
            anchors.push(span("OK", x, 100.0));
        }
        assert_eq!(detect(&r, &anchors, &p()), None);
        assert!(span_matches("OK", "ok") && !span_matches("OK", "on"));
    }

    #[test]
    fn copies_follow_the_nearest_kept_box() {
        // Two real "API" boxes (both read by OCR) far apart; the model restates
        // the second one four times next to it, with edges and a tag on copies.
        let mut r = reading();
        r.nodes = vec![
            node("n1", "API", 0.0, 0.0),
            node("n2", "API", 1000.0, 600.0),
            node("n3", "API", 1004.0, 600.0),
            node("n4", "API", 1008.0, 600.0),
            node("n5", "API", 1012.0, 600.0),
            node("n6", "Queue", 500.0, 300.0),
        ];
        r.edges = vec![
            edge("n6", "n3", "AMQP"),
            edge("n6", "n4", "AMQP"),
            edge("n3", "n4", ""),
            edge("n6", "n1", "AMQP"),
        ];
        r.owner_tags = vec![OwnerTag {
            name_raw: "Rowan".into(),
            near: "n5".into(),
            bbox: bb(1120.0, 600.0),
        }];
        let anchors = vec![span("API", 0.0, 0.0), span("API", 1000.0, 600.0)];
        let f = detect(&r, &anchors, &p()).expect("degenerate");
        assert_eq!((f.repeated[0].count, f.repeated[0].supported), (5, 2));
        let (after, _) = collapse(r, &anchors, &frame(), &p());
        let ids: Vec<&str> = after.nodes.iter().map(|n| n.local_id.as_str()).collect();
        assert_eq!(ids, ["n1", "n2", "n6"]);
        let edges: Vec<(&str, &str)> = after
            .edges
            .iter()
            .map(|e| (e.src.as_str(), e.dst.as_str()))
            .collect();
        // The copies' edges land on n2 (once), the copy-to-copy edge is gone,
        // and the edge to n1 is untouched.
        assert_eq!(edges, [("n6", "n2"), ("n6", "n1")]);
        assert_eq!(after.owner_tags[0].near, "n2");
    }

    #[test]
    fn one_span_backs_one_copy() {
        // Four stacked copies over a single real box all cover its span.
        let mut r = reading();
        for k in 0..4 {
            r.nodes.push(node(
                &format!("n{k}"),
                "Gateway",
                100.0 + f64::from(k),
                100.0,
            ));
        }
        let anchors = vec![span("Gateway", 100.0, 100.0)];
        let f = detect(&r, &anchors, &p()).expect("degenerate");
        assert_eq!(f.repeated[0].supported, 1);
    }

    #[test]
    fn a_text_ocr_read_twice_backs_one_copy() {
        // Four piled copies of one box; OCR reports its one text twice.
        let mut r = reading();
        for k in 0..4 {
            r.nodes.push(node(
                &format!("n{k}"),
                "Gateway",
                100.0 + f64::from(k),
                100.0,
            ));
        }
        let twice = vec![span("Gateway", 100.0, 100.0), span("gateway", 101.0, 100.0)];
        let f = detect(&r, &twice, &p()).expect("degenerate");
        assert_eq!(f.repeated[0].supported, 1);
        // Two separate spans of one text still back two boxes.
        let apart = vec![span("Gateway", 100.0, 100.0), span("Gateway", 400.0, 100.0)];
        let items = node_items(&r);
        assert_eq!(
            supported(&items, &apart, 0.25)
                .iter()
                .filter(|b| **b)
                .count(),
            1
        );
        r.nodes[3].bbox = bb(400.0, 100.0);
        let items = node_items(&r);
        assert_eq!(
            supported(&items, &apart, 0.25)
                .iter()
                .filter(|b| **b)
                .count(),
            2
        );
    }

    #[test]
    fn wide_spans_of_two_cards_back_both() {
        // Two separate "Deploy" cards; OCR's wide spans overlap by more than
        // half, but each span's center lies in its own card.
        let mut r = reading();
        r.nodes = vec![
            node("a", "Deploy", 0.0, 0.0),
            node("b", "Deploy", 110.0, 0.0),
        ];
        let spans = vec![
            ("Deploy".to_string(), BBox::new(-100.0, 15.0, 200.0, 35.0)),
            ("Deploy".to_string(), BBox::new(10.0, 15.0, 310.0, 35.0)),
        ];
        assert!(stacked_pair(&spans[0].1, &spans[1].1));
        let items = node_items(&r);
        assert_eq!(supported(&items, &spans, 0.25), vec![true, true]);
    }

    #[test]
    fn eight_separate_todo_notes_that_ocr_misses_stay() {
        // Eight "TODO" notes side by side among twenty, OCR silent: past the
        // strong bound, but 7 excess of 20 is under the share.
        let mut r = reading();
        for i in 0..12 {
            r.stickies
                .push(sticky(&format!("Idea {i}"), f64::from(i) * 120.0, 0.0));
        }
        for i in 0..8 {
            r.stickies.push(sticky("TODO", f64::from(i) * 120.0, 200.0));
        }
        assert_eq!(detect(&r, &[], &p()), None);
        let (after, done) = collapse(r.clone(), &[], &frame(), &p());
        assert_eq!(after, r);
        assert!(done.is_empty());
    }

    #[test]
    fn finding_display_names_the_list_and_counts() {
        let (r, anchors) = runaway_node_list();
        let f = detect(&r, &anchors, &p()).expect("degenerate");
        assert_eq!(
            f.to_string(),
            "Nodes \"mobile app\" x28 of 35 (1 OCR-backed)"
        );
    }

    // Regressions: copies side by side are retried, never collapsed by count.

    /// Eight nodes, each with an "Avery" owner tag under it that OCR missed.
    fn owner_on_every_node(n: usize) -> BoardReading {
        let mut r = reading();
        r.nodes = distinct_nodes(n);
        for node in r.nodes.clone() {
            let b = node.bbox;
            r.owner_tags.push(OwnerTag {
                name_raw: "Avery".into(),
                near: node.local_id.clone(),
                bbox: BBox::new(b.x1, b.y2 + 2.0, b.x1 + 40.0, b.y2 + 14.0),
            });
        }
        r
    }

    #[test]
    fn one_owner_on_eight_nodes_is_flagged_but_kept() {
        let r = owner_on_every_node(8);
        // Eight copies, seven excess of eight: worth the retry...
        let f = detect(&r, &[], &p()).expect("flagged for the retry");
        assert_eq!(
            (f.repeated[0].list, f.repeated[0].count),
            (ElementList::OwnerTags, 8)
        );
        // ...but when the retry repeats them, every assignment stays.
        let (after, done) = collapse(r.clone(), &[], &frame(), &p());
        assert_eq!(after, r);
        assert!(done.is_empty());
        // Twenty tags on twenty nodes fill the owner list (20) and stay.
        let full = owner_on_every_node(20);
        assert!(detect(&full, &[], &p()).is_some());
        let (after, done) = collapse(full.clone(), &[], &frame(), &p());
        assert_eq!(after, full);
        assert!(done.is_empty());
    }

    #[test]
    fn owner_tags_off_the_canvas_go_and_the_rest_stay() {
        // Three real tags, five copies wholly outside the 800 x 400 image: the
        // copies go, the three real assignments stay.
        let mut r = owner_on_every_node(8);
        let small = ReplyFrame {
            extent: BBox::new(0.0, 0.0, 800.0, 400.0),
        };
        for (i, t) in r.owner_tags.iter_mut().enumerate().skip(3) {
            let x = 900.0 + i as f64 * 50.0;
            t.bbox = BBox::new(x, 500.0, x + 40.0, 512.0);
        }
        let (after, done) = collapse(r.clone(), &[], &small, &p());
        assert_eq!(after.owner_tags, r.owner_tags[..3]);
        assert_eq!((done[0].before, done[0].after), (8, 3));
        assert_eq!(done[0].evidence, [Fabrication::OffCanvas]);
        // With every copy outside, the first stays.
        for (i, t) in r.owner_tags.iter_mut().enumerate() {
            let x = 900.0 + i as f64 * 50.0;
            t.bbox = BBox::new(x, 500.0, x + 40.0, 512.0);
        }
        let (after, _) = collapse(r.clone(), &[], &small, &p());
        assert_eq!(after.owner_tags, r.owner_tags[..1]);
    }

    #[test]
    fn a_tile_drops_its_off_tile_copies_even_where_canvas_ocr_backs_them() {
        // A tile covering x 0..500 reads one real "Queue" box and restates it
        // eight times to the right of the tile, where OCR of the whole canvas
        // happens to read "Queue" under one of the copies.
        let mut r = reading();
        r.nodes = vec![node("q0", "Queue", 100.0, 100.0)];
        for k in 0..8 {
            r.nodes.push(node(
                &format!("q{}", k + 1),
                "Queue",
                600.0 + f64::from(k) * 150.0,
                100.0,
            ));
        }
        let anchors = vec![span("Queue", 100.0, 100.0), span("Queue", 750.0, 100.0)];
        let tile = ReplyFrame {
            extent: BBox::new(0.0, 0.0, 500.0, 400.0),
        };
        assert!(detect(&r, &anchors, &p()).is_some());
        let (after, done) = collapse(r.clone(), &anchors, &tile, &p());
        let ids: Vec<&str> = after.nodes.iter().map(|n| n.local_id.as_str()).collect();
        assert_eq!(ids, ["q0"]);
        assert_eq!(done[0].evidence, [Fabrication::OffCanvas]);
    }

    #[test]
    fn a_pile_collapses_but_separate_copies_of_its_text_stay() {
        // Five "Card" copies stepped 10 px over one box, and three real "Card"
        // boxes elsewhere: the pile keeps one copy, the three stay.
        let mut r = reading();
        r.nodes = distinct_nodes(2);
        for k in 0..5 {
            r.nodes.push(node(
                &format!("p{k}"),
                "Card",
                600.0 + f64::from(k) * 10.0,
                400.0,
            ));
        }
        for k in 0..3 {
            r.nodes
                .push(node(&format!("c{k}"), "Card", f64::from(k) * 150.0, 800.0));
        }
        let (after, done) = collapse(r, &[], &frame(), &p());
        let ids: Vec<&str> = after.nodes.iter().map(|n| n.local_id.as_str()).collect();
        assert_eq!(ids, ["n1", "n2", "p0", "c0", "c1", "c2"]);
        assert_eq!(done[0].evidence, [Fabrication::Piled]);
    }

    /// Seven pairs of real "Deploy" cards overlapping by 40% of a card, each
    /// read by OCR with a wide span (the two spans of a pair pile on each other
    /// and the second span's center lies in the first card too), and an edge
    /// from each card to its pair.
    fn deploy_pairs() -> (BoardReading, Vec<(String, BBox)>) {
        let mut r = reading();
        let mut spans = Vec::new();
        for k in 0..7 {
            let y = f64::from(k) * 100.0;
            for (id, x) in [("a", 0.0), ("b", 60.0)] {
                r.nodes.push(BoardNode {
                    local_id: format!("{id}{k}"),
                    text: "Deploy".into(),
                    bbox: BBox::new(x, y, x + 100.0, y + 50.0),
                    conf: 0.9,
                });
            }
            r.edges.push(edge(&format!("a{k}"), &format!("b{k}"), ""));
            spans.push((
                "Deploy".to_string(),
                BBox::new(0.0, y + 15.0, 120.0, y + 35.0),
            ));
            spans.push((
                "Deploy".to_string(),
                BBox::new(40.0, y + 15.0, 160.0, y + 35.0),
            ));
        }
        (r, spans)
    }

    #[test]
    fn overlapping_cards_each_keep_their_own_span() {
        let (r, spans) = deploy_pairs();
        assert!(stacked_pair(&spans[0].1, &spans[1].1));
        assert!(!stacked_pair(&r.nodes[0].bbox, &r.nodes[1].bbox));
        let items = node_items(&r);
        assert_eq!(supported(&items, &spans, 0.25), vec![true; 14]);
        assert_eq!(detect(&r, &spans, &p()), None);
        let (after, done) = collapse(r.clone(), &spans, &frame(), &p());
        assert_eq!(after, r);
        assert!(done.is_empty());
    }

    #[test]
    fn cards_overlapping_by_more_than_half_each_keep_their_own_span() {
        // Seven pairs of real "Deploy" cards overlapping by 60% (piled by the
        // box rule), each read by OCR once at its own text.
        let mut r = reading();
        let mut spans = Vec::new();
        for k in 0..7 {
            let y = f64::from(k) * 100.0;
            for (id, x) in [("a", 0.0), ("b", 40.0)] {
                r.nodes.push(BoardNode {
                    local_id: format!("{id}{k}"),
                    text: "Deploy".into(),
                    bbox: BBox::new(x, y, x + 100.0, y + 50.0),
                    conf: 0.9,
                });
                spans.push((
                    "Deploy".to_string(),
                    BBox::new(x + 10.0, y + 15.0, x + 90.0, y + 35.0),
                ));
            }
        }
        assert!(stacked_pair(&r.nodes[0].bbox, &r.nodes[1].bbox));
        let items = node_items(&r);
        assert_eq!(supported(&items, &spans, 0.25), vec![true; 14]);
        assert_eq!(detect(&r, &spans, &p()), None);
    }

    #[test]
    fn a_span_that_resolves_to_a_backed_card_backs_nothing() {
        // One card, OCR reports its text twice: the second span resolves to the
        // same card and does not fall through to a neighbor.
        let mut r = reading();
        r.nodes = vec![
            node("a", "Deploy", 0.0, 0.0),
            node("b", "Deploy", 160.0, 0.0),
        ];
        let twice = vec![span("Deploy", 0.0, 0.0), span("Deploy", 2.0, 0.0)];
        let items = node_items(&r);
        assert_eq!(supported(&items, &twice, 0.25), vec![true, false]);
    }

    /// A runaway row: two real boxes, then one text restated 28 times in a row
    /// with a fixed step that runs past the right edge of the 1920 px image.
    fn row_past_the_edge() -> BoardReading {
        let mut r = reading();
        r.nodes = vec![
            node("n1", "Camera", 220.0, 255.0),
            node("n2", "Upload", 320.0, 255.0),
        ];
        for k in 0..28 {
            let x = 565.0 + f64::from(k) * 135.0;
            r.nodes.push(BoardNode {
                local_id: format!("n{}", k + 3),
                text: "Mobile App".into(),
                bbox: BBox::new(x, 255.0, x + 80.0, 295.0),
                conf: 0.9,
            });
        }
        r.edges = (0..28)
            .map(|k| edge("n2", &format!("n{}", k + 3), ""))
            .collect();
        r
    }

    #[test]
    fn a_runaway_row_past_the_canvas_edge_is_collapsed() {
        let r = row_past_the_edge();
        assert!(detect(&r, &[], &p()).is_some());
        let image = ReplyFrame {
            extent: BBox::new(0.0, 0.0, 1920.0, 954.0),
        };
        // The 17 copies wholly past the edge go; their edges join the nearest
        // kept copy's. The eleven inside stay (the retry repeated them).
        let (after, done) = collapse(r.clone(), &[], &image, &p());
        assert_eq!(after.nodes.len(), 13);
        assert_eq!(after.edges.len(), 11);
        assert!(after.nodes.iter().all(|n| n.bbox.x1 < 1920.0));
        assert_eq!((done[0].before, done[0].after), (28, 11));
        assert_eq!(done[0].evidence, [Fabrication::OffCanvas]);
        // Inside a wide enough image the row is kept.
        let (after, done) = collapse(r.clone(), &[], &frame(), &p());
        assert_eq!(after, r);
        assert!(done.is_empty());
    }

    #[test]
    fn a_box_touching_the_edge_from_outside_is_outside() {
        let e = BBox::new(0.0, 0.0, 100.0, 100.0);
        assert!(outside(&BBox::new(100.0, 0.0, 120.0, 10.0), &e));
        assert!(!outside(&BBox::new(90.0, 0.0, 120.0, 10.0), &e));
        assert!(outside(&BBox::new(-30.0, 0.0, 0.0, 10.0), &e));
        // A box without area is nowhere.
        assert!(!outside(&BBox::new(0.0, 0.0, 0.0, 0.0), &e));
        assert!(!outside(&BBox::new(200.0, 0.0, 200.0, 10.0), &e));
    }
}
