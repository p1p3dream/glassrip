//! Degenerate readings: a complete board reply that lists one text at many places.
//!
//! With a large output budget the model can fill a list towards its `maxItems`
//! with one element under fresh ids and boxes (one box text 28 times, 60
//! stickies holding five texts). Such a reply is valid JSON, so the repetition
//! guard (which stops a runaway *generation*) lets it through. This module judges
//! a *finished* reading.
//!
//! Per list (nodes, stickies, owner tags, other text; edges keyed by endpoints
//! and label), for every normalized text:
//!
//! - `count`: items with that text;
//! - `supported`: items with an OCR span of that text of their own inside their
//!   box (one span backs one item, so stepped copies around one real box share a
//!   single span);
//! - `repeats = count - max(supported, 1) + 1`: the copies that neither OCR nor a
//!   single original explains, plus that original.
//!
//! A text is **repeated** when `repeats >= min_repeats` and either the list's
//! excess copies (`repeats - 1`, summed over its texts) make up at least
//! `min_duplicate_share` of the list, or `repeats >= strong_repeats`. Two "API"
//! boxes or three "TODO" stickies never reach `min_repeats`; four "TODO"
//! stickies among a dozen notes stay under the share; a row of identical cards
//! that OCR reads at every card has no unexplained copies.
//!
//! [`collapse`] reduces each repeated text to its best-supported items: every item
//! backed by its own OCR span, or the first item when OCR backs none. Edges and
//! owner tags pointing at a removed node move to the node that was kept.

use std::collections::{HashMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::board::{normalize, BoardReading, ElementList};
use crate::geometry::BBox;

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

/// Per item: backed by an OCR span of its own. Each span backs at most one item:
/// the nearest (by box center) matching item whose grown box holds the span.
fn supported(items: &[Item<'_>], anchors: &[(String, BBox)], margin: f64) -> Vec<bool> {
    let mut backed = vec![false; items.len()];
    for (s, sb) in anchors {
        let best = items
            .iter()
            .enumerate()
            .filter(|(i, _)| !backed[*i])
            .filter_map(|(i, it)| {
                let (text, bbox) = it.anchorable?;
                (span_matches(text, s) && center_inside(sb, bbox, margin))
                    .then(|| (i, distance2(sb, bbox)))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        if let Some((i, _)) = best {
            backed[i] = true;
        }
    }
    backed
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

/// Repeated texts of one list, each with the indices to keep when collapsed.
fn list_repeats(
    list: ElementList,
    items: &[Item<'_>],
    anchors: &[(String, BBox)],
    p: &DegenerateParams,
) -> Vec<(RepeatedText, Vec<usize>)> {
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
    let excess: usize = gs.iter().map(|(_, idx)| repeats(idx).0 - 1).sum();
    let share = excess as f64 / n as f64;
    gs.into_iter()
        .filter_map(|(key, idx)| {
            let (r, s) = repeats(&idx);
            let strong = p.strong_repeats > 0 && r >= p.strong_repeats;
            if r < p.min_repeats || !(share >= p.min_duplicate_share || strong) {
                return None;
            }
            let mut keep: Vec<usize> = idx.iter().copied().filter(|&i| backed[i]).collect();
            if keep.is_empty() {
                keep.push(idx[0]);
            }
            Some((
                RepeatedText {
                    list,
                    text: key,
                    count: idx.len(),
                    supported: s,
                    list_len: n,
                },
                keep,
            ))
        })
        .collect()
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

/// The repeated texts of a complete reading, or `None` when it is sound.
/// `anchors` are OCR spans in the reading's (canvas) pixels.
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
        .map(|(r, _)| r)
        .collect();
    (!repeated.is_empty()).then_some(DegenerateFinding { repeated })
}

/// Indices to drop from one list: every repeated text keeps only its chosen items.
fn drops(repeats: &[(RepeatedText, Vec<usize>)], items: &[Item<'_>]) -> HashSet<usize> {
    let flagged: HashMap<&str, &Vec<usize>> = repeats
        .iter()
        .map(|(r, keep)| (r.text.as_str(), keep))
        .collect();
    items
        .iter()
        .enumerate()
        .filter(|(i, it)| {
            flagged
                .get(it.key.as_str())
                .is_some_and(|keep| !keep.contains(i))
        })
        .map(|(i, _)| i)
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

fn record(out: &mut Vec<CollapsedText>, repeats: &[(RepeatedText, Vec<usize>)]) {
    out.extend(repeats.iter().map(|(r, keep)| CollapsedText {
        list: r.list,
        text: r.text.clone(),
        before: r.count,
        after: keep.len(),
    }));
}

/// Reduce every repeated text to its best-supported items. Returns the reading
/// unchanged, with no record, when [`detect`] finds nothing.
pub fn collapse(
    mut reading: BoardReading,
    anchors: &[(String, BBox)],
    p: &DegenerateParams,
) -> (BoardReading, Vec<CollapsedText>) {
    let mut done = Vec::new();

    // Nodes first: edges and owner tags follow a removed copy to the nearest
    // kept node of its text.
    let items = node_items(&reading);
    let repeats = list_repeats(ElementList::Nodes, &items, anchors, p);
    let gone = drops(&repeats, &items);
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
    let repeats = list_repeats(ElementList::Edges, &items, anchors, p);
    let gone = drops(&repeats, &items);
    record(&mut done, &repeats);
    retain_indexed(&mut reading.edges, &gone);

    let items = sticky_items(&reading);
    let repeats = list_repeats(ElementList::Stickies, &items, anchors, p);
    let gone = drops(&repeats, &items);
    record(&mut done, &repeats);
    retain_indexed(&mut reading.stickies, &gone);

    let items = owner_items(&reading);
    let repeats = list_repeats(ElementList::OwnerTags, &items, anchors, p);
    let gone = drops(&repeats, &items);
    record(&mut done, &repeats);
    retain_indexed(&mut reading.owner_tags, &gone);

    let items = other_items(&reading);
    let repeats = list_repeats(ElementList::OtherVisibleText, &items, anchors, p);
    let gone = drops(&repeats, &items);
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
        let (after, done) = collapse(r.clone(), &[], &p());
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
    fn a_card_row_that_ocr_reads_at_every_card_passes() {
        let mut r = reading();
        let mut anchors = Vec::new();
        for i in 0..6 {
            let x = f64::from(i) * 150.0;
            r.nodes.push(node(&format!("n{}", i + 1), "Card", x, 100.0));
            anchors.push(span("Card", x, 100.0));
        }
        assert_eq!(detect(&r, &anchors, &p()), None);
        // Without OCR the same row is six unexplained copies.
        assert!(detect(&r, &[], &p()).is_some());
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
        assert!(collapse(r, &[], &off).1.is_empty());
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
        let (after, done) = collapse(r, &anchors, &p());
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
                after: 1
            }]
        );
        assert_eq!(detect(&after, &anchors, &p()), None);
    }

    #[test]
    fn without_ocr_the_first_copy_is_kept() {
        let (r, _) = runaway_node_list();
        let (after, _) = collapse(r, &[], &p());
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
        let (after, done) = collapse(r, &anchors, &p());
        assert_eq!(after.stickies.len(), 5);
        assert!(done.iter().all(|c| c.before == 12 && c.after == 1));
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
        let (after, _) = collapse(r, &[], &p());
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
        let (after, _) = collapse(r, &anchors, &p());
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
    fn finding_display_names_the_list_and_counts() {
        let (r, anchors) = runaway_node_list();
        let f = detect(&r, &anchors, &p()).expect("degenerate");
        assert_eq!(
            f.to_string(),
            "Nodes \"mobile app\" x28 of 35 (1 OCR-backed)"
        );
    }
}
