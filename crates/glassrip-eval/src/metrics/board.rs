//! Board metrics: node, edge, and sticky P/R/F1; edge direction, label, and style
//! accuracy; sticky CER; owner tags; UI-chrome false positives.
//!
//! Matching rules (spec 9.3):
//! - A predicted element matches a gold element when the Jaro-Winkler similarity
//!   of the normalized labels (gold text or any alias) is at least 0.9 **and**,
//!   when the gold element has a bbox, the predicted bbox exists and has
//!   IoU >= 0.3 with it. Matching is one-to-one and greedy by score.
//! - Edges are resolved to gold node ids through their endpoint labels and match
//!   when the unordered endpoint pair is the same. Direction is then correct
//!   when the prediction is `forward` and its tail is the gold tail.
//! - A chrome false positive is any predicted element (node, sticky, owner tag,
//!   edge label, other visible text) whose label matches a chrome string, or
//!   contains one as a contiguous token sequence (`Riley Park (Presenting)`
//!   contains `Riley Park`), unless the element matches a gold label of the
//!   board (so a real node such as `Share Service` is not chrome).

use glassrip_vision::BBox;
use serde::{Deserialize, Serialize};

use super::{greedy_match, Counts, Prf, Tally};
use crate::text::{best_label_similarity, cer_count, labels_match, ErrorCount, LABEL_MATCH_JW};

/// Minimum IoU for a bbox-checked match.
pub const BBOX_MATCH_IOU: f64 = 0.3;

/// Line style of an edge.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LineStyle {
    /// Continuous line.
    #[default]
    Solid,
    /// Dashed or dotted line.
    Dashed,
}

impl From<glassrip_vision::board::EdgeStyle> for LineStyle {
    fn from(s: glassrip_vision::board::EdgeStyle) -> Self {
        match s {
            glassrip_vision::board::EdgeStyle::Solid => Self::Solid,
            glassrip_vision::board::EdgeStyle::Dashed => Self::Dashed,
        }
    }
}

/// Predicted direction of an edge relative to its `src`/`dst` fields.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Arrowhead at `dst`.
    #[default]
    Forward,
    /// Direction not established.
    Uncertain,
    /// Arrowheads at both ends.
    Bidirectional,
}

fn default_true() -> bool {
    true
}

/// Gold node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldNode {
    /// Stable id used by edges and owners.
    pub id: String,
    /// Label as written on the board.
    pub text: String,
    /// Accepted alternative spellings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// Box in frame pixels, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<BBox>,
    /// Part of the core diagram (recall target 1.0).
    #[serde(default = "default_true")]
    pub core: bool,
}

/// Gold edge between two gold node ids.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldEdge {
    /// Tail node id.
    pub src: String,
    /// Head node id (arrowhead end).
    pub dst: String,
    /// Label on the line; empty when unlabelled.
    #[serde(default)]
    pub label: String,
    /// Accepted alternative label texts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub label_aliases: Vec<String>,
    /// Line style.
    #[serde(default)]
    pub style: LineStyle,
    /// Whether the edge has a single arrowhead (scored for direction).
    #[serde(default = "default_true")]
    pub directed: bool,
}

/// Gold sticky note.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldSticky {
    /// Full text.
    pub text: String,
    /// Accepted alternative texts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// Box in frame pixels, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<BBox>,
    /// Kind (`question`, `idea`, `milestone`, `note`), informational.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

/// Gold owner tag on one frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldOwnerTag {
    /// Name written on the tag.
    pub name: String,
    /// Gold node id the tag sits on.
    pub near: String,
}

/// A titled group of cards (for example a grid of candidate modules).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldGroup {
    /// Group title.
    pub title: String,
    /// Card texts.
    pub items: Vec<String>,
}

/// Gold board content.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldBoard {
    /// Nodes.
    #[serde(default)]
    pub nodes: Vec<GoldNode>,
    /// Edges.
    #[serde(default)]
    pub edges: Vec<GoldEdge>,
    /// Stickies.
    #[serde(default)]
    pub stickies: Vec<GoldSticky>,
    /// Owner tags visible in this frame (single-frame fixtures only).
    #[serde(default)]
    pub owners: Vec<GoldOwnerTag>,
    /// Card groups; the meeting suite scores titles and items as stickies.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GoldGroup>,
}

/// Predicted node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredNode {
    /// Label.
    pub text: String,
    /// Box in frame pixels.
    #[serde(default)]
    pub bbox: Option<BBox>,
}

/// Predicted edge with endpoints given as node labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredEdge {
    /// Tail node label.
    pub src: String,
    /// Head node label.
    pub dst: String,
    /// Edge label.
    #[serde(default)]
    pub label: String,
    /// Line style.
    #[serde(default)]
    pub style: LineStyle,
    /// Direction.
    #[serde(default)]
    pub direction: Direction,
}

/// Predicted sticky.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredSticky {
    /// Text.
    pub text: String,
    /// Box in frame pixels.
    #[serde(default)]
    pub bbox: Option<BBox>,
}

/// Predicted owner tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredOwnerTag {
    /// Name as read.
    pub name: String,
    /// Label of the node it sits on (empty when unanchored).
    #[serde(default)]
    pub near: String,
}

/// Predicted board content.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PredBoard {
    /// Nodes.
    #[serde(default)]
    pub nodes: Vec<PredNode>,
    /// Edges.
    #[serde(default)]
    pub edges: Vec<PredEdge>,
    /// Stickies.
    #[serde(default)]
    pub stickies: Vec<PredSticky>,
    /// Owner tags.
    #[serde(default)]
    pub owner_tags: Vec<PredOwnerTag>,
    /// Other visible canvas text.
    #[serde(default)]
    pub other_text: Vec<String>,
}

/// Scores for one board comparison (all counts poolable).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BoardScore {
    /// Node matching counts.
    pub nodes: Counts,
    /// Recall over core gold nodes.
    pub core_nodes: Tally,
    /// Edge matching counts.
    pub edges: Counts,
    /// Direction correct over matched directed gold edges.
    pub edge_direction: Tally,
    /// Matched edges whose direction came back `uncertain`.
    pub edge_uncertain: usize,
    /// Label correct over matched edges.
    pub edge_label: Tally,
    /// Style correct over matched edges.
    pub edge_style: Tally,
    /// Sticky matching counts.
    pub stickies: Counts,
    /// Character errors over matched stickies.
    pub sticky_cer: ErrorCount,
    /// Gold owner tags recovered (name and anchor node).
    pub owners: Tally,
    /// Predicted owner tags matching no gold tag.
    pub owner_fp: usize,
    /// Predicted elements that are UI chrome.
    pub chrome_fp: usize,
    /// Texts counted as chrome false positives.
    pub chrome_hits: Vec<String>,
}

impl BoardScore {
    /// Pools another score into this one.
    pub fn add(&mut self, o: &BoardScore) {
        self.nodes.add(o.nodes);
        self.core_nodes.add(o.core_nodes);
        self.edges.add(o.edges);
        self.edge_direction.add(o.edge_direction);
        self.edge_uncertain += o.edge_uncertain;
        self.edge_label.add(o.edge_label);
        self.edge_style.add(o.edge_style);
        self.stickies.add(o.stickies);
        self.sticky_cer.add(o.sticky_cer);
        self.owners.add(o.owners);
        self.owner_fp += o.owner_fp;
        self.chrome_fp += o.chrome_fp;
        self.chrome_hits.extend(o.chrome_hits.iter().cloned());
    }

    /// Node P/R/F1.
    pub fn node_prf(&self) -> Prf {
        self.nodes.prf()
    }
}

fn bbox_ok(gold: Option<&BBox>, pred: Option<&BBox>) -> Option<f64> {
    match (gold, pred) {
        (None, _) => Some(0.0),
        (Some(_), None) => None,
        (Some(g), Some(p)) => {
            let iou = g.iou(p);
            (iou >= BBOX_MATCH_IOU).then_some(iou)
        }
    }
}

fn texts<'a>(text: &'a str, aliases: &'a [String]) -> impl Iterator<Item = &'a str> {
    std::iter::once(text).chain(aliases.iter().map(String::as_str))
}

/// Score of a label pair with optional bbox gate; `None` when not admissible.
fn element_score(
    gold_text: &str,
    gold_aliases: &[String],
    gold_bbox: Option<&BBox>,
    pred_text: &str,
    pred_bbox: Option<&BBox>,
) -> Option<f64> {
    let sim = best_label_similarity(pred_text, texts(gold_text, gold_aliases));
    if sim < LABEL_MATCH_JW {
        return None;
    }
    let iou = bbox_ok(gold_bbox, pred_bbox)?;
    Some(sim + iou * 1e-3)
}

/// Matches predicted nodes to gold nodes; returns `(gold, pred)` pairs.
pub fn match_nodes(gold: &[GoldNode], pred: &[PredNode]) -> Vec<(usize, usize)> {
    greedy_match(gold.len(), pred.len(), |g, p| {
        element_score(
            &gold[g].text,
            &gold[g].aliases,
            gold[g].bbox.as_ref(),
            &pred[p].text,
            pred[p].bbox.as_ref(),
        )
    })
}

/// Resolves predicted node labels to gold node ids.
#[derive(Debug, Clone)]
pub struct NodeResolver<'a> {
    gold: &'a [GoldNode],
    /// (normalized pred label, gold id) from the node matching.
    matched: Vec<(String, &'a str)>,
}

impl<'a> NodeResolver<'a> {
    /// Builds a resolver from the gold nodes and the node matching.
    pub fn new(gold: &'a [GoldNode], pred: &[PredNode], matches: &[(usize, usize)]) -> Self {
        let matched = matches
            .iter()
            .map(|&(g, p)| {
                (
                    crate::text::normalize_label(&pred[p].text),
                    gold[g].id.as_str(),
                )
            })
            .collect();
        Self { gold, matched }
    }

    /// Resolver that matches by label only (no predicted node list).
    pub fn text_only(gold: &'a [GoldNode]) -> Self {
        Self {
            gold,
            matched: Vec::new(),
        }
    }

    /// Gold node id for a predicted label: a matched predicted node with the same
    /// label first, else the best gold label at the Jaro-Winkler threshold.
    pub fn resolve(&self, label: &str) -> Option<&'a str> {
        let n = crate::text::normalize_label(label);
        if n.is_empty() {
            return None;
        }
        if let Some((_, id)) = self.matched.iter().find(|(t, _)| *t == n) {
            return Some(id);
        }
        let mut best: Option<(f64, &'a str)> = None;
        for g in self.gold {
            let s = best_label_similarity(label, texts(&g.text, &g.aliases));
            if s >= LABEL_MATCH_JW && best.is_none_or(|(b, _)| s > b) {
                best = Some((s, g.id.as_str()));
            }
        }
        best.map(|(_, id)| id)
    }
}

fn same_pair(a: (&str, &str), b: (&str, &str)) -> bool {
    (a.0 == b.0 && a.1 == b.1) || (a.0 == b.1 && a.1 == b.0)
}

/// True when the normalized tokens of `needle` occur contiguously in `hay`.
pub fn contains_tokens(hay: &str, needle: &str) -> bool {
    let h = crate::text::normalize_label(hay);
    let n = crate::text::normalize_label(needle);
    if n.is_empty() {
        return false;
    }
    let h: Vec<&str> = h.split(' ').collect();
    let n: Vec<&str> = n.split(' ').collect();
    h.windows(n.len()).any(|w| w == n.as_slice())
}

/// True when `text` matches a chrome string or contains one as tokens.
pub fn is_chrome(text: &str, chrome: &[String]) -> bool {
    !text.trim().is_empty()
        && chrome
            .iter()
            .any(|c| labels_match(text, c) || contains_tokens(text, c))
}

fn gold_labels(gold: &GoldBoard) -> Vec<&str> {
    gold.nodes
        .iter()
        .flat_map(|n| texts(&n.text, &n.aliases))
        .chain(
            gold.stickies
                .iter()
                .flat_map(|s| texts(&s.text, &s.aliases)),
        )
        .chain(
            gold.edges
                .iter()
                .flat_map(|e| texts(&e.label, &e.label_aliases)),
        )
        .chain(gold.owners.iter().map(|o| o.name.as_str()))
        .filter(|t| !t.trim().is_empty())
        .collect()
}

/// Scores a predicted board against gold content and chrome strings.
pub fn score_board(gold: &GoldBoard, pred: &PredBoard, chrome: &[String]) -> BoardScore {
    let mut s = BoardScore::default();

    // Nodes.
    let node_matches = match_nodes(&gold.nodes, &pred.nodes);
    s.nodes = Counts::from_matches(node_matches.len(), gold.nodes.len(), pred.nodes.len());
    for (g, node) in gold.nodes.iter().enumerate() {
        if node.core {
            s.core_nodes
                .record(node_matches.iter().any(|&(mg, _)| mg == g));
        }
    }
    let resolver = NodeResolver::new(&gold.nodes, &pred.nodes, &node_matches);

    // Edges.
    let resolved: Vec<(Option<&str>, Option<&str>)> = pred
        .edges
        .iter()
        .map(|e| (resolver.resolve(&e.src), resolver.resolve(&e.dst)))
        .collect();
    let edge_matches = greedy_match(gold.edges.len(), pred.edges.len(), |g, p| {
        let (Some(ps), Some(pd)) = resolved[p] else {
            return None;
        };
        let ge = &gold.edges[g];
        if !same_pair((ps, pd), (&ge.src, &ge.dst)) {
            return None;
        }
        let label_sim = if ge.label.trim().is_empty() && pred.edges[p].label.trim().is_empty() {
            1.0
        } else {
            best_label_similarity(&pred.edges[p].label, texts(&ge.label, &ge.label_aliases))
        };
        Some(1.0 + label_sim)
    });
    s.edges = Counts::from_matches(edge_matches.len(), gold.edges.len(), pred.edges.len());
    for &(g, p) in &edge_matches {
        let ge = &gold.edges[g];
        let pe = &pred.edges[p];
        if ge.directed {
            let tail_ok = resolved[p].0 == Some(ge.src.as_str());
            s.edge_direction
                .record(pe.direction == Direction::Forward && tail_ok);
        }
        if pe.direction == Direction::Uncertain {
            s.edge_uncertain += 1;
        }
        let label_ok = if ge.label.trim().is_empty() {
            pe.label.trim().is_empty()
        } else {
            best_label_similarity(&pe.label, texts(&ge.label, &ge.label_aliases)) >= LABEL_MATCH_JW
        };
        s.edge_label.record(label_ok);
        s.edge_style.record(ge.style == pe.style);
    }

    // Stickies.
    let sticky_matches = greedy_match(gold.stickies.len(), pred.stickies.len(), |g, p| {
        element_score(
            &gold.stickies[g].text,
            &gold.stickies[g].aliases,
            gold.stickies[g].bbox.as_ref(),
            &pred.stickies[p].text,
            pred.stickies[p].bbox.as_ref(),
        )
    });
    s.stickies = Counts::from_matches(
        sticky_matches.len(),
        gold.stickies.len(),
        pred.stickies.len(),
    );
    for &(g, p) in &sticky_matches {
        s.sticky_cer
            .add(cer_count(&pred.stickies[p].text, &gold.stickies[g].text));
    }

    // Owner tags.
    let owner_matches = greedy_match(gold.owners.len(), pred.owner_tags.len(), |g, p| {
        let go = &gold.owners[g];
        let po = &pred.owner_tags[p];
        let sim = crate::text::label_similarity(&po.name, &go.name);
        (sim >= LABEL_MATCH_JW && resolver.resolve(&po.near) == Some(go.near.as_str()))
            .then_some(sim)
    });
    for g in 0..gold.owners.len() {
        s.owners
            .record(owner_matches.iter().any(|&(mg, _)| mg == g));
    }
    s.owner_fp = pred.owner_tags.len() - owner_matches.len();

    // Chrome false positives.
    let candidates = pred
        .nodes
        .iter()
        .map(|n| n.text.as_str())
        .chain(pred.stickies.iter().map(|x| x.text.as_str()))
        .chain(pred.owner_tags.iter().map(|o| o.name.as_str()))
        .chain(pred.edges.iter().map(|e| e.label.as_str()))
        .chain(pred.other_text.iter().map(String::as_str));
    let gold_texts = gold_labels(gold);
    for t in candidates {
        let is_gold = gold_texts.iter().any(|g| labels_match(t, g));
        if !is_gold && is_chrome(t, chrome) {
            s.chrome_fp += 1;
            s.chrome_hits.push(t.to_string());
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gnode(id: &str, text: &str, bbox: Option<BBox>) -> GoldNode {
        GoldNode {
            id: id.into(),
            text: text.into(),
            aliases: vec![],
            bbox,
            core: true,
        }
    }

    fn pnode(text: &str, bbox: Option<BBox>) -> PredNode {
        PredNode {
            text: text.into(),
            bbox,
        }
    }

    fn gedge(src: &str, dst: &str, label: &str) -> GoldEdge {
        GoldEdge {
            src: src.into(),
            dst: dst.into(),
            label: label.into(),
            label_aliases: vec![],
            style: LineStyle::Solid,
            directed: true,
        }
    }

    fn pedge(src: &str, dst: &str, label: &str, direction: Direction) -> PredEdge {
        PredEdge {
            src: src.into(),
            dst: dst.into(),
            label: label.into(),
            style: LineStyle::Solid,
            direction,
        }
    }

    fn b(x1: f64, y1: f64, x2: f64, y2: f64) -> Option<BBox> {
        Some(BBox::new(x1, y1, x2, y2))
    }

    #[test]
    fn nodes_need_label_and_iou() {
        let gold = GoldBoard {
            nodes: vec![
                gnode("a", "Ledger API", b(0.0, 0.0, 100.0, 50.0)),
                gnode("b", "Orbit Queue", b(200.0, 0.0, 300.0, 50.0)),
                gnode("c", "Parcel Store", None),
            ],
            ..Default::default()
        };
        let pred = PredBoard {
            nodes: vec![
                // label ok, IoU 1.0 -> match
                pnode("ledger api", b(0.0, 0.0, 100.0, 50.0)),
                // label ok but IoU = 0 -> no match
                pnode("Orbit Queue", b(500.0, 500.0, 600.0, 550.0)),
                // gold has no bbox -> label only
                pnode("Parcel Stor", b(9.0, 9.0, 10.0, 10.0)),
                // unrelated
                pnode("Share", None),
            ],
            ..Default::default()
        };
        let s = score_board(&gold, &pred, &["Share".to_string()]);
        // tp 2, fp 2, fn 1: P = 0.5, R = 2/3
        assert_eq!(
            s.nodes,
            Counts {
                tp: 2,
                fp: 2,
                fn_: 1
            }
        );
        assert_eq!(
            s.core_nodes,
            Tally {
                correct: 2,
                total: 3
            }
        );
        assert_eq!(s.chrome_fp, 1);
        assert_eq!(s.chrome_hits, vec!["Share".to_string()]);
    }

    #[test]
    fn iou_threshold_boundary() {
        // Gold 0..100, pred 50..150 on x: IoU = 50*10 / (1000+1000-500) = 1/3 >= 0.3
        let gold = GoldBoard {
            nodes: vec![gnode("a", "Atlas", b(0.0, 0.0, 100.0, 10.0))],
            ..Default::default()
        };
        let ok = PredBoard {
            nodes: vec![pnode("Atlas", b(50.0, 0.0, 150.0, 10.0))],
            ..Default::default()
        };
        assert_eq!(score_board(&gold, &ok, &[]).nodes.tp, 1);
        // pred 60..160: IoU = 40 / 160 = 0.25 < 0.3
        let bad = PredBoard {
            nodes: vec![pnode("Atlas", b(60.0, 0.0, 160.0, 10.0))],
            ..Default::default()
        };
        assert_eq!(score_board(&gold, &bad, &[]).nodes.tp, 0);
    }

    #[test]
    fn edges_direction_label_style() {
        let gold = GoldBoard {
            nodes: vec![
                gnode("a", "Ledger API", None),
                gnode("b", "Orbit Queue", None),
                gnode("c", "Parcel Store", None),
            ],
            edges: vec![
                gedge("a", "b", "REST"),
                gedge("b", "c", "gRPC"),
                gedge("a", "c", ""),
            ],
            ..Default::default()
        };
        let pred = PredBoard {
            nodes: vec![
                pnode("Ledger API", None),
                pnode("Orbit Queue", None),
                pnode("Parcel Store", None),
            ],
            edges: vec![
                // correct direction and label
                pedge("Ledger API", "Orbit Queue", "REST", Direction::Forward),
                // reversed, label ok
                pedge("Parcel Store", "Orbit Queue", "gRPC", Direction::Forward),
                // uncertain, wrong label (gold unlabelled)
                pedge("Ledger API", "Parcel Store", "events", Direction::Uncertain),
                // endpoint unknown -> fp
                pedge("Ledger API", "Nowhere", "", Direction::Forward),
            ],
            ..Default::default()
        };
        let s = score_board(&gold, &pred, &[]);
        assert_eq!(
            s.edges,
            Counts {
                tp: 3,
                fp: 1,
                fn_: 0
            }
        );
        assert_eq!(
            s.edge_direction,
            Tally {
                correct: 1,
                total: 3
            }
        );
        assert_eq!(s.edge_uncertain, 1);
        assert_eq!(
            s.edge_label,
            Tally {
                correct: 2,
                total: 3
            }
        );
        assert_eq!(
            s.edge_style,
            Tally {
                correct: 3,
                total: 3
            }
        );
    }

    #[test]
    fn stickies_cer_and_owners() {
        let gold = GoldBoard {
            nodes: vec![
                gnode("a", "Ledger API", None),
                gnode("b", "Orbit Queue", None),
            ],
            stickies: vec![GoldSticky {
                text: "Who owns retries?".into(),
                aliases: vec![],
                bbox: None,
                kind: Some("question".into()),
            }],
            owners: vec![
                GoldOwnerTag {
                    name: "Avery".into(),
                    near: "a".into(),
                },
                GoldOwnerTag {
                    name: "Jordan".into(),
                    near: "b".into(),
                },
            ],
            ..Default::default()
        };
        let pred = PredBoard {
            nodes: vec![pnode("Ledger API", None), pnode("Orbit Queue", None)],
            stickies: vec![PredSticky {
                // one substitution in 17 chars
                text: "Who owns retrics?".into(),
                bbox: None,
            }],
            owner_tags: vec![
                PredOwnerTag {
                    name: "Avery".into(),
                    near: "Ledger API".into(),
                },
                // right name, wrong node
                PredOwnerTag {
                    name: "Jordan".into(),
                    near: "Ledger API".into(),
                },
                // tile name read as owner
                PredOwnerTag {
                    name: "Riley Park".into(),
                    near: "".into(),
                },
            ],
            ..Default::default()
        };
        let s = score_board(&gold, &pred, &["Riley Park".into()]);
        assert_eq!(
            s.stickies,
            Counts {
                tp: 1,
                fp: 0,
                fn_: 0
            }
        );
        assert_eq!(
            s.sticky_cer,
            ErrorCount {
                errors: 1,
                reference_len: 17
            }
        );
        assert_eq!(
            s.owners,
            Tally {
                correct: 1,
                total: 2
            }
        );
        assert_eq!(s.owner_fp, 2);
        assert_eq!(s.chrome_fp, 1);
    }

    #[test]
    fn chrome_containment_but_not_gold_labels() {
        let chrome = vec!["Riley Park".to_string(), "Share".to_string()];
        assert!(is_chrome("Riley Park (Presenting)", &chrome));
        assert!(is_chrome("100% Share Undo", &chrome));
        assert!(!is_chrome("Shared cache", &chrome));
        let gold = GoldBoard {
            nodes: vec![gnode("s", "Share Service", None)],
            ..Default::default()
        };
        let pred = PredBoard {
            nodes: vec![pnode("Share Service", None)],
            other_text: vec!["Riley Park (Presenting)".into(), "Share".into()],
            ..Default::default()
        };
        let s = score_board(&gold, &pred, &chrome);
        // the gold node containing "Share" is not chrome; the banner and "Share" are
        assert_eq!(s.chrome_fp, 2);
    }

    #[test]
    fn resolver_uses_aliases() {
        let gold = vec![GoldNode {
            id: "lg".into(),
            text: "Ledger gateway ledger_gw_py".into(),
            aliases: vec!["Ledger gateway ledger_gw.py".into()],
            bbox: None,
            core: true,
        }];
        let r = NodeResolver::text_only(&gold);
        assert_eq!(r.resolve("Ledger gateway ledger gw.py"), Some("lg"));
        assert_eq!(r.resolve(""), None);
        assert_eq!(r.resolve("Mobile App"), None);
    }

    #[test]
    fn pooling_adds_counts() {
        let mut a = BoardScore::default();
        let b2 = BoardScore {
            nodes: Counts {
                tp: 1,
                fp: 2,
                fn_: 3,
            },
            chrome_fp: 2,
            ..Default::default()
        };
        a.add(&b2);
        a.add(&b2);
        assert_eq!(
            a.nodes,
            Counts {
                tp: 2,
                fp: 4,
                fn_: 6
            }
        );
        assert_eq!(a.chrome_fp, 4);
    }
}
