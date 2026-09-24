//! Owner tags as timed assignments.
//!
//! Each owner-tag sighting is resolved to a participant through the alias table
//! (unknown names are rejected) and anchored to a target:
//!
//! 1. **Geometry**, when the keyframe's boxes are distinct: a tag whose center lies on
//!    a connector (within half the tag size of the segment) and that does not overlap
//!    a node anchors to that edge; otherwise the unique nearest node within
//!    `node_anchor_share` tag sizes.
//! 2. The reader's `near` node.
//! 3. Otherwise the sighting is untargeted (presence only).
//!
//! Without geometry, a tag on a connector shows up as `near` flipping between the
//! edge's two endpoints. When a person's node targets alternate between the two ends
//! of a supported edge at least `alternation_min_switches` times, with at least two
//! sightings on each end, those sightings are re-anchored to the edge.
//!
//! Per person, sightings are replayed in time order. A new target opens (or moves) an
//! assignment after `confirm_keyframes` consecutive sightings agree, or after one
//! sighting that a [`Corroborator`] confirms (for example a transcript cue). Isolated
//! disagreeing sightings are ignored. `valid_from_s` is the start of the first
//! confirming keyframe; the previous assignment ends there. Untargeted sightings
//! before the first assignment (with no conflicting target in between) extend it
//! backward, flagged as backfilled.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// What an owner tag is attached to.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OwnerTarget {
    /// A node track.
    Node {
        /// Node id.
        node_id: String,
        /// Node text.
        text: String,
    },
    /// An edge track.
    Edge {
        /// Edge id.
        edge_id: String,
        /// One end's text.
        a_text: String,
        /// Other end's text.
        b_text: String,
    },
}

impl OwnerTarget {
    /// Texts of the target (one node, or both edge ends).
    pub fn texts(&self) -> Vec<&str> {
        match self {
            Self::Node { text, .. } => vec![text.as_str()],
            Self::Edge { a_text, b_text, .. } => vec![a_text.as_str(), b_text.as_str()],
        }
    }
}

/// How a sighting was anchored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnchorKind {
    /// Tag geometry on a connector.
    GeometryEdge,
    /// Tag geometry next to a node.
    GeometryNode,
    /// The reader's `near` field.
    Near,
    /// `near` alternating between the two ends of one edge.
    NearAlternation,
    /// Presence without a target.
    Untargeted,
}

/// How an assignment was opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OpenReason {
    /// Consecutive consistent keyframes.
    ConsistentKeyframes,
    /// One keyframe plus corroboration.
    Corroborated,
}

/// One owner-tag sighting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OwnerSighting {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Keyframe start.
    pub t_start_s: f64,
    /// Keyframe end.
    pub t_end_s: f64,
    /// Name as written.
    pub name_raw: String,
    /// Target, when anchored.
    pub target: Option<OwnerTarget>,
    /// Anchor method.
    pub anchor: AnchorKind,
}

/// A move or opening that a corroborator may confirm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveQuery<'a> {
    /// Person id.
    pub person_id: &'a str,
    /// Display name.
    pub display_name: &'a str,
    /// Current target, if any.
    pub from: Option<&'a OwnerTarget>,
    /// Proposed target.
    pub to: &'a OwnerTarget,
    /// Keyframe start of the sighting.
    pub t_start_s: f64,
    /// Keyframe end of the sighting.
    pub t_end_s: f64,
}

/// Evidence that confirms a move.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Corroboration {
    /// Source kind, for example `transcript`.
    pub source: String,
    /// Time of the cue.
    pub t_s: f64,
    /// Short justification (for example a segment id); no quoted content required.
    pub detail: String,
}

/// Confirms owner moves from another modality. The transcript arrives after the board
/// branch (spec 5.2), so the default confirms nothing.
pub trait Corroborator: Send + Sync {
    /// Confirmation for the query, if any.
    fn corroborate(&self, query: &MoveQuery<'_>) -> Option<Corroboration>;
}

/// The default: no corroboration.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCorroboration;

impl Corroborator for NoCorroboration {
    fn corroborate(&self, _query: &MoveQuery<'_>) -> Option<Corroboration> {
        None
    }
}

/// A timed owner assignment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OwnerAssignment {
    /// Person id.
    pub person_id: String,
    /// Display name.
    pub display_name: String,
    /// Target.
    pub target: OwnerTarget,
    /// Start of validity.
    pub valid_from_s: f64,
    /// End of validity (exclusive).
    pub valid_to_s: f64,
    /// Keyframe that opened it.
    pub opened_at_keyframe: String,
    /// Opening rule.
    pub opened_by: OpenReason,
    /// Corroboration used, if any.
    pub corroboration: Option<Corroboration>,
    /// Start was extended backward over untargeted sightings.
    pub backfilled: bool,
    /// Sightings supporting the assignment.
    pub sightings: Vec<OwnerSighting>,
}

impl OwnerAssignment {
    /// True when valid at `t_s`.
    pub fn valid_at(&self, t_s: f64) -> bool {
        self.valid_from_s <= t_s && t_s < self.valid_to_s
    }
}

/// State-machine settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OwnerParams {
    /// Consecutive consistent sightings needed to open or move.
    pub confirm_keyframes: usize,
    /// Extend the first assignment backward over untargeted sightings.
    pub backfill_untargeted: bool,
}

/// Replay one person's sightings (time order) into assignments. `timeline_end_s`
/// closes the last one.
pub fn assign(
    person_id: &str,
    display_name: &str,
    sightings: &[OwnerSighting],
    timeline_end_s: f64,
    params: &OwnerParams,
    corroborator: &dyn Corroborator,
) -> Vec<OwnerAssignment> {
    let mut out: Vec<OwnerAssignment> = Vec::new();
    let mut pending: Vec<&OwnerSighting> = Vec::new();
    let mut untargeted_since_last: Vec<&OwnerSighting> = Vec::new();
    for s in sightings {
        let Some(target) = &s.target else {
            if out.is_empty() && pending.is_empty() {
                untargeted_since_last.push(s);
            }
            continue;
        };
        if let Some(cur) = out.last_mut() {
            if &cur.target == target {
                cur.sightings.push(s.clone());
                pending.clear();
                continue;
            }
        }
        if pending.first().and_then(|p| p.target.as_ref()) != Some(target) {
            if !pending.is_empty() {
                // A conflicting target was seen: earlier presence no longer backs
                // whatever opens next.
                untargeted_since_last.clear();
            }
            pending.clear();
        }
        pending.push(s);
        let current = out.last().map(|a| &a.target);
        let corroboration = if pending.len() < params.confirm_keyframes.max(1) {
            corroborator.corroborate(&MoveQuery {
                person_id,
                display_name,
                from: current,
                to: target,
                t_start_s: s.t_start_s,
                t_end_s: s.t_end_s,
            })
        } else {
            None
        };
        if pending.len() >= params.confirm_keyframes.max(1) || corroboration.is_some() {
            let first = pending[0];
            let start = first.t_start_s;
            if let Some(prev) = out.last_mut() {
                prev.valid_to_s = start;
            }
            let (valid_from_s, backfilled) = match untargeted_since_last.first() {
                Some(u) if out.is_empty() && params.backfill_untargeted && u.t_start_s < start => {
                    (u.t_start_s, true)
                }
                _ => (start, false),
            };
            let mut sight: Vec<OwnerSighting> = if backfilled {
                untargeted_since_last.iter().map(|u| (*u).clone()).collect()
            } else {
                Vec::new()
            };
            sight.extend(pending.iter().map(|p| (*p).clone()));
            out.push(OwnerAssignment {
                person_id: person_id.to_string(),
                display_name: display_name.to_string(),
                target: target.clone(),
                valid_from_s,
                valid_to_s: timeline_end_s,
                opened_at_keyframe: first.keyframe_id.clone(),
                opened_by: if corroboration.is_some() {
                    OpenReason::Corroborated
                } else {
                    OpenReason::ConsistentKeyframes
                },
                corroboration,
                backfilled,
                sightings: sight,
            });
            pending.clear();
            untargeted_since_last.clear();
        }
    }
    out
}

/// Re-anchor sightings that alternate between the two ends of one edge. `edges` lists
/// `(edge target, a node target, b node target)`.
pub fn apply_alternation(
    sightings: &mut [OwnerSighting],
    edges: &[(OwnerTarget, OwnerTarget, OwnerTarget)],
    min_switches: usize,
) {
    for (edge, a, b) in edges {
        let seq: Vec<bool> = sightings
            .iter()
            .filter_map(|s| s.target.as_ref())
            .filter(|t| *t == a || *t == b)
            .map(|t| t == a)
            .collect();
        let (na, nb) = (
            seq.iter().filter(|v| **v).count(),
            seq.iter().filter(|v| !**v).count(),
        );
        let switches = seq.windows(2).filter(|w| w[0] != w[1]).count();
        if na >= 2 && nb >= 2 && switches >= min_switches {
            for s in sightings.iter_mut() {
                if s.target.as_ref() == Some(a) || s.target.as_ref() == Some(b) {
                    s.target = Some(edge.clone());
                    s.anchor = AnchorKind::NearAlternation;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str) -> OwnerTarget {
        OwnerTarget::Node {
            node_id: id.into(),
            text: id.to_uppercase(),
        }
    }

    fn s(t: f64, target: Option<OwnerTarget>) -> OwnerSighting {
        OwnerSighting {
            keyframe_id: format!("kf{t}"),
            t_start_s: t,
            t_end_s: t + 10.0,
            name_raw: "Avery".into(),
            anchor: if target.is_some() {
                AnchorKind::Near
            } else {
                AnchorKind::Untargeted
            },
            target,
        }
    }

    fn params() -> OwnerParams {
        OwnerParams {
            confirm_keyframes: 2,
            backfill_untargeted: true,
        }
    }

    #[test]
    fn two_consistent_keyframes_open_and_move() {
        let seq = [
            s(100.0, Some(node("n1"))),
            s(110.0, Some(node("n1"))),
            s(120.0, Some(node("n2"))), // isolated: ignored
            s(130.0, Some(node("n1"))),
            s(140.0, Some(node("n3"))),
            s(150.0, Some(node("n3"))),
        ];
        let a = assign("p1", "Avery", &seq, 500.0, &params(), &NoCorroboration);
        assert_eq!(a.len(), 2);
        assert_eq!((a[0].valid_from_s, a[0].valid_to_s), (100.0, 140.0));
        assert_eq!(a[1].target, node("n3"));
        assert_eq!((a[1].valid_from_s, a[1].valid_to_s), (140.0, 500.0));
        assert!(!a.iter().any(|x| x.target == node("n2")));
    }

    struct Always;
    impl Corroborator for Always {
        fn corroborate(&self, q: &MoveQuery<'_>) -> Option<Corroboration> {
            (q.from.is_some()).then(|| Corroboration {
                source: "transcript".into(),
                t_s: q.t_start_s,
                detail: "synthetic".into(),
            })
        }
    }

    #[test]
    fn corroboration_moves_on_one_sighting_and_backfill_extends_start() {
        let seq = [
            s(50.0, None),
            s(60.0, None),
            s(100.0, Some(node("n1"))),
            s(110.0, Some(node("n1"))),
            s(200.0, Some(node("n2"))),
        ];
        let a = assign("p1", "Avery", &seq, 300.0, &params(), &NoCorroboration);
        assert_eq!(a.len(), 1);
        assert!(a[0].backfilled && a[0].valid_from_s == 50.0);
        let b = assign("p1", "Avery", &seq, 300.0, &params(), &Always);
        assert_eq!(b.len(), 2);
        assert_eq!(b[1].opened_by, OpenReason::Corroborated);
        assert_eq!((b[0].valid_to_s, b[1].valid_from_s), (200.0, 200.0));
        assert!(b[1].valid_at(250.0) && !b[0].valid_at(250.0));
    }

    #[test]
    fn alternation_re_anchors_to_edge() {
        let edge = OwnerTarget::Edge {
            edge_id: "e1".into(),
            a_text: "N1".into(),
            b_text: "N2".into(),
        };
        let mut seq = vec![
            s(1.0, Some(node("n1"))),
            s(2.0, Some(node("n2"))),
            s(3.0, Some(node("n1"))),
            s(4.0, Some(node("n2"))),
            s(5.0, Some(node("n9"))),
        ];
        apply_alternation(&mut seq, &[(edge.clone(), node("n1"), node("n2"))], 2);
        assert_eq!(seq[0].target, Some(edge.clone()));
        assert_eq!(seq[3].anchor, AnchorKind::NearAlternation);
        assert_eq!(seq[4].target, Some(node("n9")));
        // A single switch (a move) is not alternation.
        let mut mv = vec![
            s(1.0, Some(node("n1"))),
            s(2.0, Some(node("n1"))),
            s(3.0, Some(node("n2"))),
            s(4.0, Some(node("n2"))),
        ];
        apply_alternation(&mut mv, &[(edge, node("n1"), node("n2"))], 2);
        assert_eq!(mv[0].target, Some(node("n1")));
    }
}
