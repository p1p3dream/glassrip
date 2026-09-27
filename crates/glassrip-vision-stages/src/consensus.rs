//! Consensus over several readings of one keyframe.
//!
//! A vision model served with concurrent batching does not answer one request
//! the same way twice, even at temperature 0 with a fixed seed, so a single
//! reading is one sample. `board_read` can read a keyframe several times and
//! keep what enough reads agree on:
//!
//! - Elements of one list (nodes, stickies, owner tags, other text) are matched
//!   across reads one to one: two elements are one when their boxes overlap at
//!   least `merge_iou`, or their normalized texts are at least `text_ratio`
//!   similar and their centers lie within the larger box's long side. Each read
//!   contributes at most one element to a match, so a text one read repeats
//!   many times finds at most one partner per other read.
//! - A match is kept when at least `min_agree` reads have it. Its box is the
//!   per-coordinate median of the reads' boxes, its text the most common
//!   normalized form (a tie goes to the form the OCR spans at the box back,
//!   then to the earliest read).
//! - Kept matches of one list whose boxes overlap at least `merge_iou` are one
//!   element (the rule the tile merge uses) unless `min_agree` reads list both
//!   separately: a pile of copies in one read cannot pair with two reads'
//!   element twice.
//! - Edges are matched by their endpoints mapped through the node matching
//!   (then by label when a read has several edges between the same two nodes)
//!   and kept with `min_agree` reads whose endpoints both survived. Direction
//!   and style go by majority; a tie keeps the earliest read's and is flagged.
//! - An owner tag's node is the majority of the reads' nodes mapped through the
//!   node matching; a tie leaves it unset (uncertain).
//!
//! Everything here is a pure function of the readings and the OCR spans, and
//! iterates in read order, so a replay of the same replies votes the same way.

use std::collections::{BTreeMap, HashMap};

use glassrip_vision::board::{
    normalize, BoardEdge, BoardNode, BoardReading, EdgeStyle, OwnerTag, Sticky, TextItem,
};
use glassrip_vision::BBox;

use crate::artifacts::{DroppedCounts, EdgeVote, ElementVote};

/// Vote settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoteParams {
    /// Reads an element needs to be kept (at least 1).
    pub min_agree: usize,
    /// Two elements are one when their boxes overlap at least this much...
    pub merge_iou: f64,
    /// ...or their normalized texts are at least this similar and their
    /// centers are close.
    pub text_ratio: f64,
}

/// The voted reading and its per-element vote records (parallel to its lists).
#[derive(Debug, Clone, PartialEq)]
pub struct Vote {
    pub result: BoardReading,
    pub nodes: Vec<ElementVote>,
    pub edges: Vec<EdgeVote>,
    pub stickies: Vec<ElementVote>,
    pub owner_tags: Vec<ElementVote>,
    pub other_visible_text: Vec<ElementVote>,
    pub dropped: DroppedCounts,
}

/// One element as the matcher sees it.
#[derive(Debug, Clone, Copy)]
struct El<'a> {
    text: &'a str,
    bbox: &'a BBox,
}

/// Element `idx` of read `read`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Member {
    read: usize,
    idx: usize,
}

fn center(b: &BBox) -> (f64, f64) {
    ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
}

fn text_sim(a: &str, b: &str) -> f64 {
    strsim::normalized_levenshtein(&normalize(a), &normalize(b))
}

/// How well two elements match, or `None` when they are not one element.
fn match_score(a: El<'_>, b: El<'_>, p: &VoteParams) -> Option<f64> {
    let iou = a.bbox.iou(b.bbox);
    let sim = text_sim(a.text, b.text);
    let (ca, cb) = (center(a.bbox), center(b.bbox));
    let d = ((ca.0 - cb.0).powi(2) + (ca.1 - cb.1).powi(2)).sqrt();
    let reach = a
        .bbox
        .width()
        .max(a.bbox.height())
        .max(b.bbox.width().max(b.bbox.height()));
    let same = iou >= p.merge_iou || (sim >= p.text_ratio && d <= reach);
    let close = if reach > 0.0 {
        1.0 - (d / reach).min(1.0)
    } else {
        0.0
    };
    same.then_some(iou + sim + close)
}

/// Greedy one-to-one assignment of `n` items to `m` open slots by descending
/// score (ties by item, then slot index): `out[i]` is item `i`'s slot.
fn assign(n: usize, m: usize, mut cand: Vec<(f64, usize, usize)>) -> Vec<Option<usize>> {
    cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut out = vec![None; n];
    let mut taken = vec![false; m];
    for (_, i, c) in cand {
        if out[i].is_none() && !taken[c] {
            out[i] = Some(c);
            taken[c] = true;
        }
    }
    out
}

/// Match the elements of every read: each cluster holds at most one element
/// per read, in read order.
fn cluster(lists: &[Vec<El<'_>>], p: &VoteParams) -> Vec<Vec<Member>> {
    let mut clusters: Vec<Vec<Member>> = Vec::new();
    for (r, list) in lists.iter().enumerate() {
        let open = clusters.len();
        let mut cand = Vec::new();
        for (i, e) in list.iter().enumerate() {
            for (c, members) in clusters[..open].iter().enumerate() {
                let best = members
                    .iter()
                    .filter_map(|m| match_score(*e, lists[m.read][m.idx], p))
                    .max_by(f64::total_cmp);
                if let Some(s) = best {
                    cand.push((s, i, c));
                }
            }
        }
        for (i, slot) in assign(list.len(), open, cand).into_iter().enumerate() {
            let m = Member { read: r, idx: i };
            match slot {
                Some(c) => clusters[c].push(m),
                None => clusters.push(vec![m]),
            }
        }
    }
    clusters
}

/// Median of a non-empty list (the mean of the middle two for an even count).
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn median_box<'a>(boxes: impl Iterator<Item = &'a BBox> + Clone) -> BBox {
    BBox::new(
        median(boxes.clone().map(|b| b.x1).collect()),
        median(boxes.clone().map(|b| b.y1).collect()),
        median(boxes.clone().map(|b| b.x2).collect()),
        median(boxes.map(|b| b.y2).collect()),
    )
}

/// How well the OCR spans at `bbox` back `text` (0 to 1): the best similarity
/// to one span or to all of them in reading order (a text on several lines).
fn ocr_support(text: &str, bbox: &BBox, anchors: &[(String, BBox)]) -> f64 {
    let n = normalize(text);
    if n.is_empty() {
        return 0.0;
    }
    let pad = 0.25 * bbox.width().max(bbox.height());
    let mut near: Vec<&(String, BBox)> = anchors
        .iter()
        .filter(|(_, b)| {
            let (x, y) = center(b);
            x >= bbox.x1 - pad && x <= bbox.x2 + pad && y >= bbox.y1 - pad && y <= bbox.y2 + pad
        })
        .collect();
    near.sort_by(|a, b| a.1.y1.total_cmp(&b.1.y1).then(a.1.x1.total_cmp(&b.1.x1)));
    let joined = near
        .iter()
        .map(|(t, _)| t.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let single = near
        .iter()
        .map(|(t, _)| strsim::normalized_levenshtein(&n, &normalize(t)))
        .fold(0.0, f64::max);
    single.max(strsim::normalized_levenshtein(&n, &normalize(&joined)))
}

/// The most common normalized form of `texts` (read order), as first written;
/// a tie goes to the form OCR backs best at `bbox`, then to the earliest.
/// Returns the text and how many reads wrote it.
fn vote_text(texts: &[&str], bbox: &BBox, anchors: &[(String, BBox)]) -> (String, u32) {
    // (normalized form, first original, count), in first-appearance order.
    let mut forms: Vec<(String, &str, u32)> = Vec::new();
    for &t in texts {
        let n = normalize(t);
        match forms.iter_mut().find(|f| f.0 == n) {
            Some(f) => f.2 += 1,
            None => forms.push((n, t, 1)),
        }
    }
    let top = forms.iter().map(|f| f.2).max().unwrap_or(0);
    let tied: Vec<&(String, &str, u32)> = forms.iter().filter(|f| f.2 == top).collect();
    let pick = if tied.len() > 1 {
        let mut best = tied[0];
        let mut best_support = ocr_support(best.1, bbox, anchors);
        for &f in &tied[1..] {
            let s = ocr_support(f.1, bbox, anchors);
            if s > best_support {
                best = f;
                best_support = s;
            }
        }
        best
    } else {
        tied[0]
    };
    (pick.1.to_string(), pick.2)
}

/// The value most members hold, and its count; `None` as the value when the
/// top count is shared by different values (a tie).
fn majority<T: PartialEq + Copy>(values: &[T]) -> (Option<T>, u32) {
    let mut counts: Vec<(T, u32)> = Vec::new();
    for v in values {
        match counts.iter_mut().find(|c| c.0 == *v) {
            Some(c) => c.1 += 1,
            None => counts.push((*v, 1)),
        }
    }
    let top = counts.iter().map(|c| c.1).max().unwrap_or(0);
    let mut winners = counts.iter().filter(|c| c.1 == top);
    match (winners.next(), winners.next()) {
        (Some(w), None) => (Some(w.0), top),
        _ => (None, top),
    }
}

/// Kept clusters of one list after the overlap merge, with the map from every
/// cluster to its kept index (`None`: dropped).
struct Kept {
    clusters: Vec<Vec<Member>>,
    of_cluster: Vec<Option<usize>>,
    dropped: u32,
}

/// Keep clusters with `min_agree` reads, then fold a kept cluster into an
/// earlier one when their median boxes overlap at least `merge_iou` and fewer
/// than `min_agree` reads list both (one read's pile of copies paired with
/// another read's element); the fold keeps one member per read, the larger
/// cluster's first (ties to the earlier). Overlapping elements that
/// `min_agree` reads list separately stay separate.
fn keep(clusters: Vec<Vec<Member>>, lists: &[Vec<El<'_>>], p: &VoteParams) -> Kept {
    let boxes = |ms: &[Member]| median_box(ms.iter().map(|m| lists[m.read][m.idx].bbox));
    let mut kept: Vec<Vec<Member>> = Vec::new();
    let mut of_cluster = Vec::with_capacity(clusters.len());
    let mut dropped = 0;
    for c in clusters {
        if c.len() < p.min_agree {
            of_cluster.push(None);
            dropped += 1;
            continue;
        }
        let b = boxes(&c);
        // Reads listing both: that many reads saw two elements there.
        let both = |k: &[Member]| {
            c.iter()
                .filter(|m| k.iter().any(|x| x.read == m.read))
                .count()
        };
        match kept
            .iter()
            .position(|k| boxes(k).iou(&b) >= p.merge_iou && both(k) < p.min_agree)
        {
            Some(k) => {
                let (first, second) = if c.len() > kept[k].len() {
                    (c, std::mem::take(&mut kept[k]))
                } else {
                    (std::mem::take(&mut kept[k]), c)
                };
                let mut merged = first;
                for m in second {
                    if !merged.iter().any(|x| x.read == m.read) {
                        merged.push(m);
                    }
                }
                merged.sort_by_key(|m| (m.read, m.idx));
                kept[k] = merged;
                of_cluster.push(Some(k));
            }
            None => {
                kept.push(c);
                of_cluster.push(Some(kept.len() - 1));
            }
        }
    }
    Kept {
        clusters: kept,
        of_cluster,
        dropped,
    }
}

/// Vote over `reads` (in read order, at least one). `anchors` are the
/// keyframe's OCR spans in canvas pixels (text tie-breaks).
pub fn vote(reads: &[BoardReading], anchors: &[(String, BBox)], p: &VoteParams) -> Vote {
    let p = VoteParams {
        min_agree: p.min_agree.clamp(1, reads.len().max(1)),
        ..*p
    };
    let els = |f: &dyn Fn(&BoardReading) -> Vec<El<'_>>| reads.iter().map(f).collect::<Vec<_>>();
    let mut dropped = DroppedCounts::default();

    // Nodes.
    let node_lists = els(&|r| {
        r.nodes
            .iter()
            .map(|n| El {
                text: &n.text,
                bbox: &n.bbox,
            })
            .collect()
    });
    let node_clusters = cluster(&node_lists, &p);
    let node_member_cluster: Vec<(Member, usize)> = node_clusters
        .iter()
        .enumerate()
        .flat_map(|(c, ms)| ms.iter().map(move |m| (*m, c)))
        .collect();
    let kept_nodes = keep(node_clusters, &node_lists, &p);
    dropped.nodes = kept_nodes.dropped;
    let mut nodes = Vec::new();
    let mut node_votes = Vec::new();
    for (k, ms) in kept_nodes.clusters.iter().enumerate() {
        let bbox = median_box(ms.iter().map(|m| &reads[m.read].nodes[m.idx].bbox));
        let texts: Vec<&str> = ms
            .iter()
            .map(|m| reads[m.read].nodes[m.idx].text.as_str())
            .collect();
        let (text, text_votes) = vote_text(&texts, &bbox, anchors);
        nodes.push(BoardNode {
            local_id: format!("n{}", k + 1),
            text,
            bbox,
            conf: median(ms.iter().map(|m| reads[m.read].nodes[m.idx].conf).collect()),
        });
        node_votes.push(ElementVote {
            votes: ms.len() as u32,
            text_votes,
            uncertain: false,
        });
    }
    // (read, local id) -> kept node index; the first node of a read with an id wins.
    let mut node_of: Vec<HashMap<&str, usize>> = vec![HashMap::new(); reads.len()];
    let mut by_member: Vec<Vec<Option<usize>>> =
        reads.iter().map(|r| vec![None; r.nodes.len()]).collect();
    for (m, c) in &node_member_cluster {
        by_member[m.read][m.idx] = kept_nodes.of_cluster[*c];
    }
    for (r, read) in reads.iter().enumerate() {
        for (i, n) in read.nodes.iter().enumerate() {
            if let Some(k) = by_member[r][i] {
                node_of[r].entry(n.local_id.as_str()).or_insert(k);
            }
        }
    }
    let node_id = |r: usize, id: &str| node_of[r].get(id).copied();

    // Edges: by kept endpoints (unordered), then by label within a read.
    struct EdgeMember {
        read: usize,
        idx: usize,
        forward: bool,
    }
    let mut edge_clusters: Vec<((usize, usize), Vec<EdgeMember>)> = Vec::new();
    let mut unmapped = 0u32;
    for (r, read) in reads.iter().enumerate() {
        let mut groups: BTreeMap<(usize, usize), Vec<(usize, bool)>> = BTreeMap::new();
        for (i, e) in read.edges.iter().enumerate() {
            match (node_id(r, &e.src), node_id(r, &e.dst)) {
                (Some(s), Some(d)) => groups
                    .entry((s.min(d), s.max(d)))
                    .or_default()
                    .push((i, s <= d)),
                _ => unmapped += 1,
            }
        }
        for (key, items) in groups {
            let open: Vec<usize> = edge_clusters
                .iter()
                .enumerate()
                .filter(|(_, (k, ms))| *k == key && !ms.iter().any(|m| m.read == r))
                .map(|(c, _)| c)
                .collect();
            let mut cand = Vec::new();
            for (j, (i, _)) in items.iter().enumerate() {
                for (slot, c) in open.iter().enumerate() {
                    let best = edge_clusters[*c]
                        .1
                        .iter()
                        .map(|m| text_sim(&read.edges[*i].label, &reads[m.read].edges[m.idx].label))
                        .fold(0.0, f64::max);
                    cand.push((best, j, slot));
                }
            }
            for (j, slot) in assign(items.len(), open.len(), cand)
                .into_iter()
                .enumerate()
            {
                let (idx, forward) = items[j];
                let m = EdgeMember {
                    read: r,
                    idx,
                    forward,
                };
                match slot {
                    Some(s) => edge_clusters[open[s]].1.push(m),
                    None => edge_clusters.push((key, vec![m])),
                }
            }
        }
    }
    let mut edges = Vec::new();
    let mut edge_votes = Vec::new();
    dropped.edges = unmapped;
    for ((a, b), ms) in &edge_clusters {
        if ms.len() < p.min_agree {
            dropped.edges += 1;
            continue;
        }
        let edge = |m: &EdgeMember| &reads[m.read].edges[m.idx];
        let first = &ms[0];
        let (dir, direction_votes) = majority(&ms.iter().map(|m| m.forward).collect::<Vec<_>>());
        let forward = dir.unwrap_or(first.forward);
        let (sty, style_votes) = majority(&ms.iter().map(|m| edge(m).style).collect::<Vec<_>>());
        let style: EdgeStyle = sty.unwrap_or(edge(first).style);
        let label_boxes: Vec<&BBox> = ms
            .iter()
            .filter_map(|m| edge(m).label_bbox.as_ref())
            .collect();
        let label_bbox =
            (2 * label_boxes.len() >= ms.len()).then(|| median_box(label_boxes.iter().copied()));
        let around = label_bbox.unwrap_or_else(|| {
            let (na, nb) = (&nodes[*a].bbox, &nodes[*b].bbox);
            BBox::new(
                na.x1.min(nb.x1),
                na.y1.min(nb.y1),
                na.x2.max(nb.x2),
                na.y2.max(nb.y2),
            )
        });
        let labels: Vec<&str> = ms.iter().map(|m| edge(m).label.as_str()).collect();
        let (label, label_votes) = vote_text(&labels, &around, anchors);
        let (src, dst) = if forward { (*a, *b) } else { (*b, *a) };
        edges.push(BoardEdge {
            src: format!("n{}", src + 1),
            dst: format!("n{}", dst + 1),
            label,
            label_bbox,
            style,
            conf: median(ms.iter().map(|m| edge(m).conf).collect()),
        });
        edge_votes.push(EdgeVote {
            votes: ms.len() as u32,
            direction_votes: if dir.is_some() {
                direction_votes
            } else {
                ms.iter().filter(|m| m.forward == forward).count() as u32
            },
            style_votes: if sty.is_some() {
                style_votes
            } else {
                ms.iter().filter(|m| edge(m).style == style).count() as u32
            },
            label_votes,
            direction_uncertain: dir.is_none(),
            style_uncertain: sty.is_none(),
        });
    }

    // Stickies.
    let sticky_lists = els(&|r| {
        r.stickies
            .iter()
            .map(|s| El {
                text: &s.text,
                bbox: &s.bbox,
            })
            .collect()
    });
    let kept = keep(cluster(&sticky_lists, &p), &sticky_lists, &p);
    dropped.stickies = kept.dropped;
    let mut stickies = Vec::new();
    let mut sticky_votes = Vec::new();
    for ms in &kept.clusters {
        let s = |m: &Member| &reads[m.read].stickies[m.idx];
        let bbox = median_box(ms.iter().map(|m| &s(m).bbox));
        let texts: Vec<&str> = ms.iter().map(|m| s(m).text.as_str()).collect();
        let (text, text_votes) = vote_text(&texts, &bbox, anchors);
        let (color, _) = majority(&ms.iter().map(|m| s(m).color).collect::<Vec<_>>());
        stickies.push(Sticky {
            text,
            color: color.unwrap_or(s(&ms[0]).color),
            bbox,
        });
        sticky_votes.push(ElementVote {
            votes: ms.len() as u32,
            text_votes,
            uncertain: color.is_none(),
        });
    }

    // Owner tags: the node each read put the tag at, mapped through the nodes.
    let owner_lists = els(&|r| {
        r.owner_tags
            .iter()
            .map(|o| El {
                text: &o.name_raw,
                bbox: &o.bbox,
            })
            .collect()
    });
    let kept = keep(cluster(&owner_lists, &p), &owner_lists, &p);
    dropped.owner_tags = kept.dropped;
    let mut owner_tags = Vec::new();
    let mut owner_votes = Vec::new();
    for ms in &kept.clusters {
        let o = |m: &Member| &reads[m.read].owner_tags[m.idx];
        let bbox = median_box(ms.iter().map(|m| &o(m).bbox));
        let texts: Vec<&str> = ms.iter().map(|m| o(m).name_raw.as_str()).collect();
        let (name_raw, text_votes) = vote_text(&texts, &bbox, anchors);
        let nears: Vec<Option<usize>> = ms
            .iter()
            .map(|m| {
                let near = &o(m).near;
                (!near.is_empty()).then(|| node_id(m.read, near)).flatten()
            })
            .collect();
        let (near, _) = majority(&nears);
        owner_tags.push(OwnerTag {
            name_raw,
            near: near
                .flatten()
                .map(|k| format!("n{}", k + 1))
                .unwrap_or_default(),
            bbox,
        });
        owner_votes.push(ElementVote {
            votes: ms.len() as u32,
            text_votes,
            uncertain: near.is_none(),
        });
    }

    // Other text.
    let other_lists = els(&|r| {
        r.other_visible_text
            .iter()
            .map(|t| El {
                text: &t.text,
                bbox: &t.bbox,
            })
            .collect()
    });
    let kept = keep(cluster(&other_lists, &p), &other_lists, &p);
    dropped.other_visible_text = kept.dropped;
    let mut other_visible_text = Vec::new();
    let mut other_votes = Vec::new();
    for ms in &kept.clusters {
        let t = |m: &Member| &reads[m.read].other_visible_text[m.idx];
        let bbox = median_box(ms.iter().map(|m| &t(m).bbox));
        let texts: Vec<&str> = ms.iter().map(|m| t(m).text.as_str()).collect();
        let (text, text_votes) = vote_text(&texts, &bbox, anchors);
        other_visible_text.push(TextItem { text, bbox });
        other_votes.push(ElementVote {
            votes: ms.len() as u32,
            text_votes,
            uncertain: false,
        });
    }

    Vote {
        result: BoardReading {
            nodes,
            edges,
            stickies,
            owner_tags,
            other_visible_text,
            confidence: median(reads.iter().map(|r| r.confidence).collect()),
        },
        nodes: node_votes,
        edges: edge_votes,
        stickies: sticky_votes,
        owner_tags: owner_votes,
        other_visible_text: other_votes,
        dropped,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use glassrip_vision::board::StickyColor;

    fn p() -> VoteParams {
        VoteParams {
            min_agree: 2,
            merge_iou: 0.5,
            text_ratio: 0.85,
        }
    }

    fn node(id: &str, text: &str, x: f64, y: f64) -> BoardNode {
        BoardNode {
            local_id: id.into(),
            text: text.into(),
            bbox: BBox::new(x, y, x + 100.0, y + 40.0),
            conf: 0.9,
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

    fn reading(nodes: Vec<BoardNode>, edges: Vec<BoardEdge>) -> BoardReading {
        BoardReading {
            nodes,
            edges,
            stickies: vec![],
            owner_tags: vec![],
            other_visible_text: vec![],
            confidence: 0.8,
        }
    }

    fn texts(r: &BoardReading) -> Vec<&str> {
        r.nodes.iter().map(|n| n.text.as_str()).collect()
    }

    #[test]
    fn keeps_what_two_of_three_reads_have_and_drops_singletons() {
        let a = reading(
            vec![
                node("n1", "Order Service", 100.0, 100.0),
                node("n2", "Ledger", 400.0, 100.0),
                node("n3", "Ghost", 700.0, 500.0),
            ],
            vec![],
        );
        let b = reading(
            vec![
                node("n1", "Ledger", 402.0, 101.0),
                node("n2", "Order Service", 99.0, 102.0),
            ],
            vec![],
        );
        let c = reading(
            vec![
                node("n1", "Order Service", 101.0, 99.0),
                node("n2", "Phantom", 50.0, 600.0),
            ],
            vec![],
        );
        let v = vote(&[a, b, c], &[], &p());
        assert_eq!(texts(&v.result), ["Order Service", "Ledger"]);
        assert_eq!(v.nodes[0].votes, 3);
        assert_eq!(v.nodes[1].votes, 2);
        assert_eq!(v.dropped.nodes, 2, "Ghost and Phantom are singletons");
        let ids: Vec<&str> = v.result.nodes.iter().map(|n| n.local_id.as_str()).collect();
        assert_eq!(ids, ["n1", "n2"]);
    }

    #[test]
    fn boxes_are_per_coordinate_medians() {
        let mk = |x1: f64, y1: f64, x2: f64, y2: f64| {
            reading(
                vec![BoardNode {
                    local_id: "n1".into(),
                    text: "Box".into(),
                    bbox: BBox::new(x1, y1, x2, y2),
                    conf: 0.5,
                }],
                vec![],
            )
        };
        let v = vote(
            &[
                mk(10.0, 20.0, 110.0, 60.0),
                mk(14.0, 18.0, 104.0, 66.0),
                mk(12.0, 30.0, 108.0, 62.0),
            ],
            &[],
            &p(),
        );
        assert_eq!(v.result.nodes[0].bbox, BBox::new(12.0, 20.0, 108.0, 62.0));
        // Two reads: the mean of the two.
        let v = vote(
            &[mk(10.0, 20.0, 110.0, 60.0), mk(14.0, 18.0, 104.0, 66.0)],
            &[],
            &p(),
        );
        assert_eq!(v.result.nodes[0].bbox, BBox::new(12.0, 19.0, 107.0, 63.0));
    }

    #[test]
    fn edges_map_endpoints_through_the_node_matching() {
        // Three reads name the same two nodes with different local ids and list
        // them in different orders; two agree on the direction.
        let a = reading(
            vec![
                node("a", "Order Service", 100.0, 100.0),
                node("b", "Ledger", 400.0, 100.0),
            ],
            vec![edge("a", "b", "REST")],
        );
        let b = reading(
            vec![
                node("x7", "Ledger", 401.0, 99.0),
                node("x3", "Order Service", 100.0, 101.0),
            ],
            vec![edge("x3", "x7", "REST")],
        );
        let c = reading(
            vec![
                node("n1", "Order Service", 98.0, 100.0),
                node("n2", "Ledger", 399.0, 100.0),
            ],
            vec![edge("n2", "n1", "rest")],
        );
        let v = vote(&[a, b.clone(), c], &[], &p());
        assert_eq!(v.result.edges.len(), 1, "{:?}", v.result.edges);
        let e = &v.result.edges[0];
        let name = |id: &str| {
            v.result
                .nodes
                .iter()
                .find(|n| n.local_id == id)
                .unwrap()
                .text
                .clone()
        };
        assert_eq!(
            (name(&e.src), name(&e.dst)),
            ("Order Service".into(), "Ledger".into())
        );
        assert_eq!(e.label, "REST");
        let ev = &v.edges[0];
        assert_eq!((ev.votes, ev.direction_votes, ev.label_votes), (3, 2, 3));
        assert!(!ev.direction_uncertain && !ev.style_uncertain);

        // Two reads disagreeing on the direction: kept, first read's, flagged.
        let d = reading(
            vec![
                node("q", "Ledger", 400.0, 100.0),
                node("r", "Order Service", 100.0, 100.0),
            ],
            vec![edge("q", "r", "REST")],
        );
        let v = vote(&[b, d], &[], &p());
        assert_eq!(v.result.edges.len(), 1);
        assert!(v.edges[0].direction_uncertain);
        let e = &v.result.edges[0];
        let src = v.result.nodes.iter().find(|n| n.local_id == e.src).unwrap();
        assert_eq!(src.text, "Order Service", "the first read's direction");
    }

    #[test]
    fn an_edge_to_a_dropped_node_goes() {
        let a = reading(
            vec![
                node("a", "Order Service", 100.0, 100.0),
                node("g", "Ghost", 700.0, 500.0),
            ],
            vec![edge("a", "g", "")],
        );
        let b = reading(
            vec![
                node("a", "Order Service", 100.0, 100.0),
                node("h", "Phantom", 50.0, 600.0),
            ],
            vec![edge("a", "h", "")],
        );
        let v = vote(&[a, b], &[], &p());
        assert_eq!(texts(&v.result), ["Order Service"]);
        assert!(v.result.edges.is_empty());
        assert_eq!(v.dropped.edges, 2);
    }

    #[test]
    fn a_runaway_in_one_read_is_dropped() {
        // Read 2 lists "Order Service" ten times, piled with small steps, each
        // with an edge from Ledger; the other reads list it once.
        let clean = || {
            reading(
                vec![
                    node("n1", "Ledger", 400.0, 100.0),
                    node("n2", "Order Service", 100.0, 300.0),
                ],
                vec![edge("n1", "n2", "")],
            )
        };
        let mut nodes = vec![node("n1", "Ledger", 400.0, 100.0)];
        let mut edges = Vec::new();
        for k in 0..10 {
            let id = format!("n{}", k + 2);
            nodes.push(node(
                &id,
                "Order Service",
                100.0 + f64::from(k) * 3.0,
                300.0,
            ));
            edges.push(edge("n1", &id, ""));
        }
        let runaway = reading(nodes, edges);
        for order in [
            vec![clean(), runaway.clone(), clean()],
            vec![runaway.clone(), clean(), clean()],
            vec![clean(), clean(), runaway.clone()],
        ] {
            let v = vote(&order, &[], &p());
            let mut t = texts(&v.result);
            t.sort_unstable();
            assert_eq!(t, ["Ledger", "Order Service"], "one copy survives");
            assert_eq!(v.result.edges.len(), 1, "{:?}", v.result.edges);
            assert!(v.nodes.iter().all(|n| n.votes == 3), "{:?}", v.nodes);
            assert_eq!(v.edges[0].votes, 3);
            assert_eq!(v.dropped.nodes, 9, "nine fabricated copies");
            assert_eq!(v.dropped.edges, 9, "nine edges to them");
        }
    }

    #[test]
    fn a_pile_pairing_with_two_reads_is_one_element() {
        // Read 1 lists the box twice, 6 px apart; read 2's box is nearer the
        // second copy, so the matching pairs read 0 with one copy and read 2
        // with the other. Only read 1 lists both: one element.
        let a = reading(vec![node("n1", "Order Service", 100.0, 100.0)], vec![]);
        let b = reading(
            vec![
                node("n1", "Order Service", 100.0, 100.0),
                node("n2", "Order Service", 106.0, 100.0),
            ],
            vec![],
        );
        let c = reading(vec![node("n1", "Order Service", 107.0, 100.0)], vec![]);
        let v = vote(&[a, b, c], &[], &p());
        assert_eq!(texts(&v.result), ["Order Service"]);
        assert_eq!(v.nodes[0].votes, 3);
    }

    #[test]
    fn overlapping_elements_every_read_lists_stay_two() {
        let r = || {
            reading(
                vec![
                    BoardNode {
                        local_id: "g".into(),
                        text: "Payments".into(),
                        bbox: BBox::new(100.0, 100.0, 300.0, 200.0),
                        conf: 0.9,
                    },
                    BoardNode {
                        local_id: "i".into(),
                        text: "Refunds".into(),
                        bbox: BBox::new(110.0, 110.0, 290.0, 190.0),
                        conf: 0.9,
                    },
                ],
                vec![],
            )
        };
        let v = vote(&[r(), r(), r()], &[], &p());
        assert_eq!(texts(&v.result), ["Payments", "Refunds"]);
    }

    #[test]
    fn a_text_tie_goes_to_the_form_ocr_backs() {
        let mk = |t: &str| reading(vec![node("n1", t, 100.0, 100.0)], vec![]);
        let anchors = vec![(
            "Order Service".to_string(),
            BBox::new(110.0, 110.0, 190.0, 130.0),
        )];
        let v = vote(&[mk("Order Servce"), mk("Order Service")], &anchors, &p());
        assert_eq!(v.result.nodes[0].text, "Order Service");
        assert_eq!(v.nodes[0].text_votes, 1);
        // Without OCR, the earliest read's form.
        let v = vote(&[mk("Order Servce"), mk("Order Service")], &[], &p());
        assert_eq!(v.result.nodes[0].text, "Order Servce");
        // A majority beats OCR.
        let v = vote(
            &[mk("Order Servce"), mk("Order Service"), mk("order  servce")],
            &anchors,
            &p(),
        );
        assert_eq!(v.result.nodes[0].text, "Order Servce");
        assert_eq!(v.nodes[0].text_votes, 2);
    }

    #[test]
    fn owner_tags_vote_and_point_at_the_voted_nodes() {
        let tag = |near: &str, x: f64| OwnerTag {
            name_raw: "Avery".into(),
            near: near.into(),
            bbox: BBox::new(x, 150.0, x + 40.0, 170.0),
        };
        let mut a = reading(
            vec![
                node("p", "Ledger", 400.0, 100.0),
                node("q", "Order Service", 100.0, 100.0),
            ],
            vec![],
        );
        a.owner_tags = vec![tag("q", 100.0), tag("p", 700.0)];
        let mut b = reading(
            vec![
                node("n1", "Order Service", 100.0, 100.0),
                node("n2", "Ledger", 400.0, 100.0),
            ],
            vec![],
        );
        b.owner_tags = vec![tag("n1", 101.0)];
        let mut c = reading(
            vec![
                node("z", "Order Service", 100.0, 100.0),
                node("y", "Ledger", 400.0, 100.0),
            ],
            vec![],
        );
        c.owner_tags = vec![tag("y", 100.0)];
        let v = vote(&[a, b, c], &[], &p());
        assert_eq!(
            v.result.owner_tags.len(),
            1,
            "the tag at x=700 is a singleton"
        );
        let o = &v.result.owner_tags[0];
        let near = v
            .result
            .nodes
            .iter()
            .find(|n| n.local_id == o.near)
            .unwrap();
        assert_eq!(near.text, "Order Service", "two of three reads");
        assert_eq!(v.owner_tags[0].votes, 3);
        assert!(!v.owner_tags[0].uncertain);
    }

    #[test]
    fn a_tied_owner_node_is_left_unset() {
        let tag = |near: &str| OwnerTag {
            name_raw: "Avery".into(),
            near: near.into(),
            bbox: BBox::new(250.0, 150.0, 290.0, 170.0),
        };
        let mut a = reading(
            vec![
                node("a", "Order Service", 100.0, 100.0),
                node("b", "Ledger", 400.0, 100.0),
            ],
            vec![],
        );
        a.owner_tags = vec![tag("a")];
        let mut b = a.clone();
        b.owner_tags = vec![tag("b")];
        let v = vote(&[a, b], &[], &p());
        assert_eq!(v.result.owner_tags[0].near, "");
        assert!(v.owner_tags[0].uncertain);
    }

    #[test]
    fn stickies_and_other_text_vote_too() {
        let sticky = |t: &str, color, x: f64| Sticky {
            text: t.into(),
            color,
            bbox: BBox::new(x, 400.0, x + 80.0, 460.0),
        };
        let mut a = reading(vec![], vec![]);
        a.stickies = vec![sticky("Ship it?", StickyColor::Yellow, 500.0)];
        a.other_visible_text = vec![TextItem {
            text: "Q3 plan".into(),
            bbox: BBox::new(10.0, 10.0, 80.0, 25.0),
        }];
        let mut b = reading(vec![], vec![]);
        b.stickies = vec![
            sticky("Ship it?", StickyColor::Orange, 502.0),
            sticky("Only here", StickyColor::Pink, 100.0),
        ];
        b.other_visible_text = a.other_visible_text.clone();
        let v = vote(&[a, b], &[], &p());
        assert_eq!(v.result.stickies.len(), 1);
        assert_eq!(v.result.stickies[0].color, StickyColor::Yellow);
        assert!(v.stickies[0].uncertain, "the color was tied");
        assert_eq!(v.result.other_visible_text.len(), 1);
        assert_eq!(v.dropped.stickies, 1);
    }

    #[test]
    fn side_by_side_copies_are_separate_elements() {
        // A row of identical cards that every read lists stays a row.
        let row = || {
            reading(
                (0..4)
                    .map(|k| {
                        node(
                            &format!("n{k}"),
                            "Card",
                            100.0 + f64::from(k) * 150.0,
                            100.0,
                        )
                    })
                    .collect(),
                vec![],
            )
        };
        let v = vote(&[row(), row(), row()], &[], &p());
        assert_eq!(v.result.nodes.len(), 4);
        assert!(v.nodes.iter().all(|n| n.votes == 3));
    }

    #[test]
    fn the_vote_is_deterministic() {
        let a = reading(
            vec![
                node("a", "Order Service", 100.0, 100.0),
                node("b", "Ledger", 400.0, 100.0),
            ],
            vec![edge("a", "b", "REST")],
        );
        let mut b = a.clone();
        b.nodes[0].bbox = BBox::new(103.0, 98.0, 205.0, 139.0);
        let mut c = a.clone();
        c.edges[0].style = EdgeStyle::Dashed;
        let reads = [a, b, c];
        assert_eq!(vote(&reads, &[], &p()), vote(&reads, &[], &p()));
    }
}
