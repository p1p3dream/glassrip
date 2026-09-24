//! Facts derived from the board and the validated notes, shared by the
//! markdown and the SVG.

use std::collections::BTreeMap;

use glassrip_notes::board::{BoardNode, BoardState, TargetKind};
use glassrip_notes::notes::{Decision, MeetingNotes};
use glassrip_notes::text::{mmss, tokens};

const GENERIC: &[&str] = &[
    "api", "service", "services", "app", "server", "system", "manager", "the", "a", "an", "of",
    "and", "new", "page",
];
const DEFER_WORDS: &[&str] = &[
    "skip",
    "skipping",
    "defer",
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
/// Tokens between a deferral verb and the component it applies to, at most.
const DEFER_WINDOW: usize = 4;

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

/// Nodes a decision defers, with the decision time.
pub fn deferred_nodes(board: &BoardState, decisions: &[Decision]) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for d in decisions {
        let toks = tokens(&d.text);
        let defer_at: Vec<usize> = toks
            .iter()
            .enumerate()
            .filter(|(_, t)| DEFER_WORDS.contains(&t.as_str()))
            .map(|(i, _)| i)
            .collect();
        for n in &board.nodes {
            let hit = mentions(&toks, &key_tokens(&n.text))
                .into_iter()
                .any(|m| defer_at.iter().any(|&v| m > v && m - v <= DEFER_WINDOW));
            if hit {
                out.entry(n.node_id.clone()).or_insert(d.t_start_s);
            }
        }
    }
    out
}

/// The node a decision says to focus on.
pub fn focus_node(board: &BoardState, decisions: &[Decision]) -> Option<String> {
    for d in decisions {
        let toks = tokens(&d.text);
        let Some(f) = toks.iter().position(|t| t == "focus") else {
            continue;
        };
        let best = board
            .final_nodes()
            .filter_map(|n| {
                mentions(&toks, &key_tokens(&n.text))
                    .into_iter()
                    .filter(|&m| m > f)
                    .min()
                    .map(|m| (m, n.node_id.clone()))
            })
            .min();
        if let Some((_, id)) = best {
            return Some(id);
        }
    }
    None
}

/// First name of a participant, or the raw tag text.
pub fn short_name(notes: &MeetingNotes, person_id: Option<&str>, raw: &str) -> String {
    person_id
        .and_then(|pid| notes.people.iter().find(|p| p.person_id == pid))
        .map(|p| p.first_name().to_string())
        .unwrap_or_else(|| raw.trim().to_string())
}

/// One owner of a node over time.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeOwner {
    /// Short name.
    pub name: String,
    /// From, seconds.
    pub from_s: f64,
    /// Until, seconds.
    pub to_s: Option<f64>,
    /// Via an edge (`to X` / `from X`), when the tag sits on a link.
    pub via: Option<String>,
    /// Target the same person moved from when this assignment started.
    pub moved_from: Option<String>,
}

/// Owner history per node id (edge tags are listed on both ends).
pub fn node_owners(board: &BoardState, notes: &MeetingNotes) -> BTreeMap<String, Vec<NodeOwner>> {
    let label = |id: &str| {
        board
            .node(id)
            .map(|n| n.text.clone())
            .unwrap_or_else(|| id.to_string())
    };
    let target_label = |kind: TargetKind, id: &str| match kind {
        TargetKind::Node => label(id),
        TargetKind::Edge => board
            .edge(id)
            .map(|e| format!("the {} to {} link", label(&e.src), label(&e.dst)))
            .unwrap_or_else(|| id.to_string()),
    };
    let mut out: BTreeMap<String, Vec<NodeOwner>> = BTreeMap::new();
    for o in &board.owner_assignments {
        let name = short_name(notes, o.person_id.as_deref(), &o.name_raw);
        let moved_from = board
            .owner_assignments
            .iter()
            .filter(|p| {
                p.owner_id != o.owner_id && p.name_raw == o.name_raw && p.person_id == o.person_id
            })
            .find(|p| {
                p.valid_to_s
                    .is_some_and(|t| (t - o.valid_from_s).abs() <= 5.0)
            })
            .map(|p| target_label(p.target_kind, &p.target_id));
        let base = NodeOwner {
            name,
            from_s: o.valid_from_s,
            to_s: o.valid_to_s,
            via: None,
            moved_from,
        };
        match o.target_kind {
            TargetKind::Node => out.entry(o.target_id.clone()).or_default().push(base),
            TargetKind::Edge => {
                if let Some(e) = board.edge(&o.target_id) {
                    out.entry(e.src.clone()).or_default().push(NodeOwner {
                        via: Some(format!("on the link to {}", label(&e.dst))),
                        ..base.clone()
                    });
                    out.entry(e.dst.clone()).or_default().push(NodeOwner {
                        via: Some(format!("on the link from {}", label(&e.src))),
                        ..base
                    });
                }
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

/// Final nodes (present at the end), falling back to every node.
pub fn final_nodes(board: &BoardState) -> Vec<&BoardNode> {
    let f: Vec<&BoardNode> = board.final_nodes().collect();
    if f.is_empty() {
        board.nodes.iter().collect()
    } else {
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn board() -> BoardState {
        serde_json::from_value(serde_json::json!({
            "board_id": "b", "final_t_s": 1.0, "edges": [],
            "nodes": [
                {"node_id": "a", "text": "Acme CMS", "first_seen_s": 0.0},
                {"node_id": "b", "text": "Relay API", "first_seen_s": 0.0}
            ]
        }))
        .unwrap()
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
}
