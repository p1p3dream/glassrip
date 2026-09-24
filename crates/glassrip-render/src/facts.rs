//! Facts derived from the board and the validated notes, shared by the
//! markdown and the SVG.

use std::collections::{BTreeMap, BTreeSet};

use glassrip_notes::board::{
    center, first_seen, target_text, BoardExt, BoardStateItem, NodeState, OwnerTarget,
};
use glassrip_notes::notes::{Decision, MeetingNotes};
use glassrip_notes::people::first_name;
use glassrip_notes::text::{mmss, tokens};

const GENERIC: &[&str] = &[
    "api", "service", "services", "app", "server", "system", "manager", "the", "a", "an", "of",
    "and", "new", "page",
];
const DEFER_WORDS: &[&str] = &[
    "skip",
    "skipping",
    "defer",
    "deferring",
    "deferred",
    "defers",
    "postpone",
    "postponed",
    "park",
    "parked",
    "drop",
    "dropped",
    "shelve",
    "pause",
];
/// Words that negate a deferral when they come shortly before it ("not
/// deferring", "no longer deferred", "we won't skip", "don't drop").
const NEGATIONS: &[&str] = &[
    "not", "no", "never", "don", "dont", "won", "wont", "isn", "aren", "shouldn", "stop",
    "stopped", "cancel", "undo", "without",
];
/// Tokens between a deferral verb and the component it applies to, at most.
const DEFER_WINDOW: usize = 4;
/// Tokens before a deferral verb searched for a negation.
const NEGATION_WINDOW: usize = 3;

/// Distinctive tokens of a label (generic words removed unless nothing is left).
pub fn key_tokens(label: &str) -> Vec<String> {
    let all = tokens(label);
    let key: Vec<String> = all
        .iter()
        .filter(|t| !GENERIC.contains(&t.as_str()))
        .cloned()
        .collect();
    if key.is_empty() {
        all
    } else {
        key
    }
}

fn mentions(text_tokens: &[String], key: &[String]) -> Vec<usize> {
    // any distinctive token counts: people often say only one word of a name
    text_tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| key.contains(t))
        .map(|(i, _)| i)
        .collect()
}

/// Positions of deferral verbs that are not negated.
fn deferrals(toks: &[String]) -> Vec<usize> {
    toks.iter()
        .enumerate()
        .filter(|(_, t)| DEFER_WORDS.contains(&t.as_str()))
        .map(|(i, _)| i)
        .filter(|&i| {
            !toks[i.saturating_sub(NEGATION_WINDOW)..i]
                .iter()
                .any(|t| NEGATIONS.contains(&t.as_str()))
        })
        .collect()
}

/// Nodes a decision defers, with the decision time.
pub fn deferred_nodes(board: &BoardStateItem, decisions: &[Decision]) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for d in decisions {
        let toks = tokens(&d.text);
        let defer_at = deferrals(&toks);
        for n in &board.nodes {
            let hit = mentions(&toks, &key_tokens(&n.text))
                .into_iter()
                .any(|m| defer_at.iter().any(|&v| m > v && m - v <= DEFER_WINDOW));
            if hit {
                out.entry(n.id.clone()).or_insert(d.t_start_s);
            }
        }
    }
    out
}

/// The node a decision says to focus on.
pub fn focus_node(board: &BoardStateItem, decisions: &[Decision]) -> Option<String> {
    for d in decisions {
        let toks = tokens(&d.text);
        let Some(f) = toks.iter().position(|t| t == "focus") else {
            continue;
        };
        let best = board
            .final_nodes()
            .into_iter()
            .filter_map(|n| {
                mentions(&toks, &key_tokens(&n.text))
                    .into_iter()
                    .filter(|&m| m > f)
                    .min()
                    .map(|m| (m, n.id.clone()))
            })
            .min();
        if let Some((_, id)) = best {
            return Some(id);
        }
    }
    None
}

/// First name of a participant, or the fallback text.
pub fn short_name(notes: &MeetingNotes, person_id: &str, fallback: &str) -> String {
    notes
        .people
        .iter()
        .find(|p| p.person_id == person_id)
        .map(|p| first_name(p).to_string())
        .unwrap_or_else(|| {
            fallback
                .split_whitespace()
                .next()
                .unwrap_or(fallback)
                .trim()
                .to_string()
        })
}

/// One owner of a node over time.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeOwner {
    /// Short name.
    pub name: String,
    /// From, seconds.
    pub from_s: f64,
    /// Until, seconds (None: still the owner at the end).
    pub to_s: Option<f64>,
    /// Via an edge (`to X` / `from X`), when the tag sits on a link.
    pub via: Option<String>,
    /// Target the assignment replaced, when it was a move.
    pub moved_from: Option<String>,
}

/// Owner history per node id (edge tags are listed on both ends).
pub fn node_owners(
    board: &BoardStateItem,
    notes: &MeetingNotes,
) -> BTreeMap<String, Vec<NodeOwner>> {
    let label = |id: &str| {
        board
            .node(id)
            .map(|n| n.text.clone())
            .unwrap_or_else(|| id.to_string())
    };
    let end = board.end_s();
    let mut out: BTreeMap<String, Vec<NodeOwner>> = BTreeMap::new();
    for o in &board.owner_assignments {
        let base = NodeOwner {
            name: short_name(notes, &o.person_id, &o.display_name),
            from_s: o.valid_from_s,
            to_s: (o.valid_to_s < end - 0.5).then_some(o.valid_to_s),
            via: None,
            moved_from: o.moved_from.as_ref().map(target_text),
        };
        match &o.target {
            OwnerTarget::Node { node_id, .. } => out.entry(node_id.clone()).or_default().push(base),
            OwnerTarget::Edge { src, dst, .. } => {
                out.entry(src.clone()).or_default().push(NodeOwner {
                    via: Some(format!("on the link to {}", label(dst))),
                    ..base.clone()
                });
                out.entry(dst.clone()).or_default().push(NodeOwner {
                    via: Some(format!("on the link from {}", label(src))),
                    ..base
                });
            }
        }
    }
    for v in out.values_mut() {
        v.sort_by(|a, b| a.from_s.total_cmp(&b.from_s));
    }
    out
}

/// "Name (from 17:22)" style owner summary for a node.
pub fn owner_summary(owners: &[NodeOwner]) -> String {
    if owners.is_empty() {
        return "none shown".into();
    }
    owners
        .iter()
        .map(|o| {
            let mut s = o.name.clone();
            let mut notes = Vec::new();
            if let Some(v) = &o.via {
                notes.push(v.clone());
            }
            match o.to_s {
                Some(t) => notes.push(format!("{} to {}", mmss(o.from_s), mmss(t))),
                None => notes.push(format!("from {}", mmss(o.from_s))),
            }
            if let Some(m) = &o.moved_from {
                notes.push(format!("moved from {m}"));
            }
            s.push_str(&format!(" ({})", notes.join(", ")));
            s
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Boxes with no arrows laid out in rows and columns on the board (for example
/// a set of cards listing things to build). Found from positions only.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivedGrid<'a> {
    /// Members in reading order (row by row).
    pub members: Vec<&'a NodeState>,
    /// Columns.
    pub columns: usize,
    /// Rows.
    pub rows: usize,
    /// Earliest first sighting of a member, seconds.
    pub first_seen_s: f64,
}

fn clusters(values: &[f64], tol: f64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut out = vec![0; values.len()];
    let (mut id, mut start) = (0usize, None::<f64>);
    for i in order {
        match start {
            Some(s0) if values[i] - s0 <= tol => {}
            Some(_) => {
                id += 1;
                start = Some(values[i]);
            }
            None => start = Some(values[i]),
        }
        out[i] = id;
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// Grids of at least 4 unconnected boxes in 2 or more rows and columns that fill
/// at least 75% of their cells.
pub fn derive_grids(board: &BoardStateItem) -> Vec<DerivedGrid<'_>> {
    let linked: BTreeSet<&str> = board
        .final_edges()
        .iter()
        .flat_map(|e| [e.src.as_str(), e.dst.as_str()])
        .collect();
    let cand: Vec<(&NodeState, (f64, f64), f64, f64)> = board
        .final_nodes()
        .into_iter()
        .filter(|n| !linked.contains(n.id.as_str()))
        .filter_map(|n| {
            n.bbox
                .map(|b| (n, center(&b), (b.x2 - b.x1).abs(), (b.y2 - b.y1).abs()))
        })
        .collect();
    if cand.len() < 4 {
        return Vec::new();
    }
    let mw = median(cand.iter().map(|c| c.2).collect()).max(1.0);
    let mh = median(cand.iter().map(|c| c.3).collect()).max(1.0);
    let rows = clusters(&cand.iter().map(|c| c.1 .1).collect::<Vec<_>>(), 0.6 * mh);
    let cols = clusters(&cand.iter().map(|c| c.1 .0).collect::<Vec<_>>(), 0.6 * mw);
    let n_rows = rows.iter().copied().max().map_or(0, |m| m + 1);
    let n_cols = cols.iter().copied().max().map_or(0, |m| m + 1);
    let filled = cand.len() as f64 / (n_rows * n_cols).max(1) as f64;
    if n_rows < 2 || n_cols < 2 || filled < 0.75 {
        return Vec::new();
    }
    let mut members: Vec<(usize, usize, &NodeState)> = cand
        .iter()
        .enumerate()
        .map(|(i, c)| (rows[i], cols[i], c.0))
        .collect();
    members.sort_by_key(|m| (m.0, m.1));
    vec![DerivedGrid {
        first_seen_s: members
            .iter()
            .map(|m| first_seen(&m.2.lifetimes))
            .fold(f64::INFINITY, f64::min),
        members: members.into_iter().map(|m| m.2).collect(),
        columns: n_cols,
        rows: n_rows,
    }]
}

/// Final nodes (present at the end), falling back to every node.
pub fn final_nodes(board: &BoardStateItem) -> Vec<&NodeState> {
    board.final_nodes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glassrip_notes::board::build;

    fn decision(text: &str, t: f64) -> Decision {
        Decision {
            id: "d".into(),
            text: text.into(),
            t_start_s: t,
            t_end_s: t,
            evidence: Default::default(),
            quote: None,
        }
    }

    fn board() -> BoardStateItem {
        let mut b = build::board("b", 60.0);
        b.nodes = vec![
            build::node("a", "Acme CMS", 0.0, 60.0, None),
            build::node("b", "Relay API", 0.0, 60.0, None),
        ];
        b
    }

    #[test]
    fn deferral_needs_the_verb_close_to_the_component() {
        let b = board();
        let d = vec![decision(
            "Skip the Acme step for now and focus on the relay side",
            30.0,
        )];
        let def = deferred_nodes(&b, &d);
        assert_eq!(def.keys().collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(focus_node(&b, &d).as_deref(), Some("b"));
        let d = vec![decision("Keep Acme; skip nothing else", 5.0)];
        assert!(deferred_nodes(&b, &d).is_empty());
    }

    #[test]
    fn unconnected_boxes_in_rows_and_columns_form_a_grid() {
        use glassrip_notes::board::{BBox, EdgeStyle};
        let mut b = build::board("b", 60.0);
        let at = |x: f64, y: f64| Some(BBox::new(x, y, x + 100.0, y + 50.0));
        b.nodes = vec![
            build::node("a", "Relay API", 0.0, 60.0, at(0.0, 0.0)),
            build::node("b", "Ledger", 0.0, 60.0, at(300.0, 0.0)),
        ];
        for (i, t) in [
            "Visitor log",
            "Hosts",
            "Wayfinding",
            "Badge preview",
            "Safety",
        ]
        .iter()
        .enumerate()
        {
            let (r, c) = (i / 3, i % 3);
            b.nodes.push(build::node(
                &format!("g{i}"),
                t,
                10.0 + i as f64,
                60.0,
                at(1000.0 + 150.0 * c as f64, 400.0 + 90.0 * r as f64),
            ));
        }
        b.edges = vec![build::edge("e", &b, "a", "b", "", EdgeStyle::Solid)];
        let g = derive_grids(&b);
        assert_eq!(g.len(), 1);
        assert_eq!((g[0].rows, g[0].columns, g[0].members.len()), (2, 3, 5));
        assert_eq!(g[0].members[3].text, "Badge preview");
        assert_eq!(g[0].first_seen_s, 10.0);
        // too few boxes: no grid
        b.nodes.truncate(5);
        assert!(derive_grids(&b).is_empty());
    }

    #[test]
    fn negated_deferrals_do_not_defer() {
        let b = board();
        for text in [
            "We are not deferring the Acme work",
            "We are no longer deferring Acme",
            "We won't skip the Acme step",
            "Don't drop Acme from the pilot",
        ] {
            assert!(
                deferred_nodes(&b, &[decision(text, 1.0)]).is_empty(),
                "{text}"
            );
        }
        assert_eq!(
            deferred_nodes(&b, &[decision("Defer Acme to the next phase", 1.0)]).len(),
            1
        );
    }
}
