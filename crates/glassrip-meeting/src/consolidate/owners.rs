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
//! A person may be tagged on several targets in one keyframe (multi-target owners);
//! all are kept. A person tagged on both ends of one edge in the same keyframe is
//! anchored to that edge ([`collapse_edge_pairs`]). Without geometry, a tag on a
//! connector shows up as `near` alternating between the edge's two ends; only strictly
//! interleaving stretches (A, B, A, B: at least 3 alternations, no repeated target)
//! are re-anchored to the edge ([`apply_alternation`]), so genuine moves such as
//! A, A, B, B, A, A stay node assignments.
//!
//! Per person, keyframes with at least one target are replayed in time order
//! ([`assign`]): a target opens after `confirm_keyframes` consecutive such keyframes
//! show it, or after one that a [`Corroborator`] confirms (for example a transcript
//! cue); an open target closes after `confirm_keyframes` consecutive such keyframes
//! lack it. An opening at the moment another target of the same person closes is a
//! move. Keyframes where the person appears without a target are neutral. Backfill
//! over earlier untargeted sightings is off by default; when enabled it is recorded
//! separately (`backfill_from_s`) and never counts toward opening.

use std::collections::BTreeMap;

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
        /// Tail node id (or end `a` when the direction is uncertain).
        src: String,
        /// Head node id (or end `b`).
        dst: String,
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
    /// Tagged on both ends of one edge in the same keyframe (two separate tags).
    BothEnds,
    /// One tag bridging two nodes, adjacent to both.
    GeometryBridge,
    /// `near` strictly alternating between the two ends of one edge.
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
    /// Index of the physical tag within its keyframe: one tag bridging two nodes
    /// yields two sightings with the same index.
    #[serde(default)]
    pub tag: u32,
}

/// A move or opening that a corroborator may confirm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveQuery<'a> {
    /// Person id.
    pub person_id: &'a str,
    /// Display name.
    pub display_name: &'a str,
    /// Open target the person is leaving (open but not tagged in this keyframe).
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
    /// Name as written on the tag (first supporting sighting).
    pub name_raw: String,
    /// Target.
    pub target: OwnerTarget,
    /// Start of validity (confirmed opening).
    pub valid_from_s: f64,
    /// End of validity (exclusive).
    pub valid_to_s: f64,
    /// Keyframe whose sighting started the confirmed opening.
    pub opened_at_keyframe: String,
    /// Opening rule.
    pub opened_by: OpenReason,
    /// Corroboration used, if any.
    pub corroboration: Option<Corroboration>,
    /// The target this assignment replaced (a move), if any.
    pub moved_from: Option<OwnerTarget>,
    /// Start of a backfilled interval before `valid_from_s` (untargeted presence only;
    /// flagged, not part of the confirmed assignment).
    pub backfill_from_s: Option<f64>,
    /// Sightings supporting the assignment.
    pub sightings: Vec<OwnerSighting>,
}

impl OwnerAssignment {
    /// True when the confirmed assignment is valid at `t_s`.
    pub fn valid_at(&self, t_s: f64) -> bool {
        self.valid_from_s <= t_s && t_s < self.valid_to_s
    }

    /// Like [`OwnerAssignment::valid_at`], also counting the backfilled interval.
    pub fn valid_at_with_backfill(&self, t_s: f64) -> bool {
        self.backfill_from_s.unwrap_or(self.valid_from_s) <= t_s && t_s < self.valid_to_s
    }
}

/// State-machine settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OwnerParams {
    /// Consecutive consistent keyframes needed to open, move, or close.
    pub confirm_keyframes: usize,
    /// Record a backfilled interval over earlier untargeted sightings.
    pub backfill_untargeted: bool,
}

/// Sightings of one person in one keyframe.
struct KeySight<'a> {
    keyframe_id: &'a str,
    t_start_s: f64,
    t_end_s: f64,
    targets: Vec<&'a OwnerSighting>,
    untargeted: Vec<&'a OwnerSighting>,
}

fn by_keyframe(sightings: &[OwnerSighting]) -> Vec<KeySight<'_>> {
    let mut out: Vec<KeySight<'_>> = Vec::new();
    for s in sightings {
        let same = out.last().is_some_and(|k| k.keyframe_id == s.keyframe_id);
        if !same {
            out.push(KeySight {
                keyframe_id: &s.keyframe_id,
                t_start_s: s.t_start_s,
                t_end_s: s.t_end_s,
                targets: Vec::new(),
                untargeted: Vec::new(),
            });
        }
        if let Some(k) = out.last_mut() {
            match &s.target {
                Some(t) if !k.targets.iter().any(|x| x.target.as_ref() == Some(t)) => {
                    k.targets.push(s)
                }
                Some(_) => {}
                None => k.untargeted.push(s),
            }
        }
    }
    out
}

struct Open<'a> {
    target: OwnerTarget,
    from_s: f64,
    opened_at: String,
    opened_by: OpenReason,
    corroboration: Option<Corroboration>,
    moved_from: Option<OwnerTarget>,
    backfill_from_s: Option<f64>,
    sightings: Vec<OwnerSighting>,
    missing: Vec<&'a KeySight<'a>>,
}

fn close(o: Open<'_>, person_id: &str, display_name: &str, to_s: f64) -> OwnerAssignment {
    OwnerAssignment {
        person_id: person_id.to_string(),
        display_name: display_name.to_string(),
        name_raw: o
            .sightings
            .first()
            .map(|s| s.name_raw.clone())
            .unwrap_or_default(),
        target: o.target,
        valid_from_s: o.from_s,
        valid_to_s: to_s,
        opened_at_keyframe: o.opened_at,
        opened_by: o.opened_by,
        corroboration: o.corroboration,
        moved_from: o.moved_from,
        backfill_from_s: o.backfill_from_s,
        sightings: o.sightings,
    }
}

/// Replay one person's sightings (time order; several per keyframe allowed) into
/// assignments. `timeline_end_s` closes those still open. `visible(keyframe_id,
/// target)` says whether a target was read in a keyframe: an open target only counts
/// as absent where it was visible (a node the reader missed is not a move).
pub fn assign(
    person_id: &str,
    display_name: &str,
    sightings: &[OwnerSighting],
    timeline_end_s: f64,
    params: &OwnerParams,
    corroborator: &dyn Corroborator,
    visible: &dyn Fn(&str, &OwnerTarget) -> bool,
) -> Vec<OwnerAssignment> {
    let confirm = params.confirm_keyframes.max(1);
    let keys = by_keyframe(sightings);
    let mut done: Vec<OwnerAssignment> = Vec::new();
    let mut open: Vec<Open<'_>> = Vec::new();
    // Pending targets with their consecutive targeted keyframes.
    let mut pending: BTreeMap<OwnerTarget, Vec<&KeySight<'_>>> = BTreeMap::new();
    // Untargeted presence since the last targeted keyframe, before any assignment.
    let mut presence: Vec<&OwnerSighting> = Vec::new();
    let mut ever_opened = false;
    for k in &keys {
        if k.targets.is_empty() {
            if !ever_opened && pending.is_empty() {
                presence.extend(k.untargeted.iter().copied());
            }
            continue;
        }
        let here: Vec<&OwnerTarget> = k.targets.iter().filter_map(|s| s.target.as_ref()).collect();
        for o in open.iter_mut() {
            match k
                .targets
                .iter()
                .find(|s| s.target.as_ref() == Some(&o.target))
            {
                Some(s) => {
                    o.sightings.push((*s).clone());
                    o.missing.clear();
                }
                None if visible(k.keyframe_id, &o.target) => o.missing.push(k),
                None => {}
            }
        }
        let before = pending.len();
        pending.retain(|t, _| here.contains(&t));
        if pending.len() < before && !ever_opened {
            // A conflicting target was abandoned: earlier presence no longer backs
            // whatever opens next.
            presence.clear();
        }
        for s in &k.targets {
            let Some(t) = &s.target else { continue };
            if open.iter().any(|o| &o.target == t) {
                continue;
            }
            pending.entry(t.clone()).or_default().push(k);
        }
        // Closings by sustained absence.
        let mut closed_now: Vec<(OwnerTarget, f64)> = Vec::new();
        let mut i = 0;
        while i < open.len() {
            if open[i].missing.len() >= confirm {
                let at = open[i].missing[0].t_start_s;
                let o = open.remove(i);
                closed_now.push((o.target.clone(), at));
                done.push(close(o, person_id, display_name, at));
            } else {
                i += 1;
            }
        }
        // Openings.
        let ready: Vec<OwnerTarget> = pending.keys().cloned().collect();
        for t in ready {
            let Some(run) = pending.get(&t) else { continue };
            let leaving = open
                .iter()
                .find(|o| !o.missing.is_empty())
                .map(|o| o.target.clone());
            let corroboration = if run.len() < confirm {
                corroborator.corroborate(&MoveQuery {
                    person_id,
                    display_name,
                    from: leaving.as_ref(),
                    to: &t,
                    t_start_s: k.t_start_s,
                    t_end_s: k.t_end_s,
                })
            } else {
                None
            };
            if run.len() < confirm && corroboration.is_none() {
                continue;
            }
            let first = run[0];
            let at = first.t_start_s;
            let sight: Vec<OwnerSighting> = run
                .iter()
                .filter_map(|ks| ks.targets.iter().find(|s| s.target.as_ref() == Some(&t)))
                .map(|s| (*s).clone())
                .collect();
            // A corroborated move closes the target being left now.
            let mut moved_from = closed_now
                .iter()
                .find(|(_, c)| (*c - at).abs() < 1e-9)
                .map(|(x, _)| x.clone());
            if corroboration.is_some() {
                if let Some(pos) = open
                    .iter()
                    .position(|o| Some(&o.target) == leaving.as_ref())
                {
                    let o = open.remove(pos);
                    moved_from = Some(o.target.clone());
                    done.push(close(o, person_id, display_name, at));
                }
            }
            // A target that was visible but untagged from the start of this run was
            // left for the new one: a move, even if it is not read again afterwards.
            if moved_from.is_none() {
                if let Some(pos) = open
                    .iter()
                    .position(|o| o.missing.first().is_some_and(|m| m.t_start_s <= at + 1e-9))
                {
                    let o = open.remove(pos);
                    moved_from = Some(o.target.clone());
                    done.push(close(o, person_id, display_name, at));
                }
            }
            let backfill_from_s = if !ever_opened && params.backfill_untargeted {
                presence.first().map(|p| p.t_start_s).filter(|s| *s < at)
            } else {
                None
            };
            pending.remove(&t);
            ever_opened = true;
            open.push(Open {
                target: t,
                from_s: at,
                opened_at: first.keyframe_id.to_string(),
                opened_by: if corroboration.is_some() {
                    OpenReason::Corroborated
                } else {
                    OpenReason::ConsistentKeyframes
                },
                corroboration,
                moved_from,
                backfill_from_s,
                sightings: sight,
                missing: Vec::new(),
            });
        }
    }
    for o in open {
        done.push(close(o, person_id, display_name, timeline_end_s));
    }
    done.sort_by(|a, b| {
        a.valid_from_s
            .total_cmp(&b.valid_from_s)
            .then(a.target.cmp(&b.target))
    });
    done
}

/// Replace, per keyframe, a person's two node targets that are the ends of one edge
/// with that edge. `edges` lists `(edge target, a node target, b node target)`.
pub fn collapse_edge_pairs(
    sightings: &mut Vec<OwnerSighting>,
    edges: &[(OwnerTarget, OwnerTarget, OwnerTarget)],
) {
    let mut out: Vec<OwnerSighting> = Vec::with_capacity(sightings.len());
    let mut i = 0;
    while i < sightings.len() {
        let kf = sightings[i].keyframe_id.clone();
        let mut j = i;
        while j < sightings.len() && sightings[j].keyframe_id == kf {
            j += 1;
        }
        let mut group: Vec<OwnerSighting> = sightings[i..j].to_vec();
        for (edge, a, b) in edges {
            let pa = group.iter().position(|s| s.target.as_ref() == Some(a));
            let pb = group.iter().position(|s| s.target.as_ref() == Some(b));
            // Only two separate tags; one tag bridging both ends stays two node targets.
            if let (Some(pa), Some(pb)) = (pa, pb) {
                if group[pa].tag == group[pb].tag {
                    continue;
                }
                let mut merged = group[pa.min(pb)].clone();
                merged.target = Some(edge.clone());
                merged.anchor = AnchorKind::BothEnds;
                let (lo, hi) = (pa.min(pb), pa.max(pb));
                group.remove(hi);
                group[lo] = merged;
            }
        }
        out.extend(group);
        i = j;
    }
    *sightings = out;
}

/// Re-anchor strictly interleaving stretches between the two ends of an edge. Only
/// keyframes where the person is tagged on exactly one of the two ends take part; a
/// stretch qualifies when consecutive entries always switch end and it has at least
/// `min_alternations + 1` entries. `edges` lists `(edge target, a, b)`.
pub fn apply_alternation(
    sightings: &mut [OwnerSighting],
    edges: &[(OwnerTarget, OwnerTarget, OwnerTarget)],
    min_alternations: usize,
) {
    for (edge, a, b) in edges {
        // (sighting index, is a), one per keyframe with exactly one of a or b.
        let mut seq: Vec<(usize, bool)> = Vec::new();
        let mut i = 0;
        while i < sightings.len() {
            let kf = sightings[i].keyframe_id.clone();
            let mut hits: Vec<(usize, bool)> = Vec::new();
            while i < sightings.len() && sightings[i].keyframe_id == kf {
                match sightings[i].target.as_ref() {
                    Some(t) if t == a => hits.push((i, true)),
                    Some(t) if t == b => hits.push((i, false)),
                    _ => {}
                }
                i += 1;
            }
            if hits.len() == 1 {
                seq.push(hits[0]);
            }
        }
        let mut start = 0;
        while start < seq.len() {
            let mut end = start + 1;
            while end < seq.len() && seq[end].1 != seq[end - 1].1 {
                end += 1;
            }
            if end - start > min_alternations {
                for &(idx, _) in &seq[start..end] {
                    sightings[idx].target = Some(edge.clone());
                    sightings[idx].anchor = AnchorKind::NearAlternation;
                }
            }
            start = end;
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

    fn edge() -> OwnerTarget {
        OwnerTarget::Edge {
            edge_id: "e1".into(),
            src: "n1".into(),
            dst: "n2".into(),
            a_text: "N1".into(),
            b_text: "N2".into(),
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
            tag: 0,
        }
    }

    fn params() -> OwnerParams {
        OwnerParams {
            confirm_keyframes: 2,
            backfill_untargeted: false,
        }
    }

    fn seq(ids: &[&str]) -> Vec<OwnerSighting> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| s(10.0 * i as f64, Some(node(id))))
            .collect()
    }

    fn pairs() -> Vec<(OwnerTarget, OwnerTarget, OwnerTarget)> {
        vec![(edge(), node("n1"), node("n2"))]
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
        let a = assign(
            "p1",
            "Avery",
            &seq,
            500.0,
            &params(),
            &NoCorroboration,
            &|_, _| true,
        );
        assert_eq!(a.len(), 2, "{a:#?}");
        assert_eq!((a[0].valid_from_s, a[0].valid_to_s), (100.0, 140.0));
        assert_eq!(a[1].target, node("n3"));
        assert_eq!((a[1].valid_from_s, a[1].valid_to_s), (140.0, 500.0));
        assert_eq!(a[1].moved_from, Some(node("n1")));
        assert!(a[0].moved_from.is_none());
    }

    struct Always;
    impl Corroborator for Always {
        fn corroborate(&self, q: &MoveQuery<'_>) -> Option<Corroboration> {
            q.from.is_some().then(|| Corroboration {
                source: "transcript".into(),
                t_s: q.t_start_s,
                detail: "synthetic".into(),
            })
        }
    }

    #[test]
    fn corroboration_moves_on_one_sighting() {
        let seq = [
            s(100.0, Some(node("n1"))),
            s(110.0, Some(node("n1"))),
            s(200.0, Some(node("n2"))),
        ];
        let a = assign(
            "p1",
            "Avery",
            &seq,
            300.0,
            &params(),
            &NoCorroboration,
            &|_, _| true,
        );
        assert_eq!(a.len(), 1);
        let b = assign("p1", "Avery", &seq, 300.0, &params(), &Always, &|_, _| true);
        assert_eq!(b.len(), 2);
        assert_eq!(b[1].opened_by, OpenReason::Corroborated);
        assert_eq!(b[1].moved_from, Some(node("n1")));
        assert_eq!((b[0].valid_to_s, b[1].valid_from_s), (200.0, 200.0));
        assert!(b[1].valid_at(250.0) && !b[0].valid_at(250.0));
    }

    #[test]
    fn backfill_is_off_by_default_and_flagged_when_on() {
        let seq = [
            s(50.0, None),
            s(60.0, None),
            s(100.0, Some(node("n1"))),
            s(110.0, Some(node("n1"))),
        ];
        let a = assign(
            "p1",
            "Avery",
            &seq,
            300.0,
            &params(),
            &NoCorroboration,
            &|_, _| true,
        );
        assert_eq!(a[0].valid_from_s, 100.0);
        assert!(a[0].backfill_from_s.is_none());
        let on = OwnerParams {
            backfill_untargeted: true,
            ..params()
        };
        let b = assign(
            "p1",
            "Avery",
            &seq,
            300.0,
            &on,
            &NoCorroboration,
            &|_, _| true,
        );
        // The confirmed opening is unchanged; the backfill is separate.
        assert_eq!(b[0].valid_from_s, 100.0);
        assert_eq!(b[0].backfill_from_s, Some(50.0));
        assert!(!b[0].valid_at(55.0) && b[0].valid_at_with_backfill(55.0));
        // Presence alone never opens anything.
        let only = [s(50.0, None), s(60.0, None), s(100.0, Some(node("n1")))];
        assert!(assign(
            "p1",
            "Avery",
            &only,
            300.0,
            &on,
            &NoCorroboration,
            &|_, _| true
        )
        .is_empty());
    }

    #[test]
    fn multi_target_owner_keeps_both_targets() {
        let mut v = Vec::new();
        for t in [0.0, 10.0, 20.0] {
            v.push(s(t, Some(node("n1"))));
            v.push(s(t, Some(node("n3"))));
        }
        let a = assign(
            "p1",
            "Avery",
            &v,
            100.0,
            &params(),
            &NoCorroboration,
            &|_, _| true,
        );
        assert_eq!(a.len(), 2);
        assert!(a.iter().all(|x| x.valid_at(50.0) && x.moved_from.is_none()));
    }

    #[test]
    fn both_ends_of_an_edge_in_one_keyframe_anchor_to_the_edge() {
        let mut v = Vec::new();
        for t in [0.0, 10.0] {
            v.push(s(t, Some(node("n1"))));
            v.push(OwnerSighting {
                tag: 1,
                ..s(t, Some(node("n2")))
            });
        }
        // One tag bridging both ends stays two node targets.
        let mut bridge: Vec<OwnerSighting> = v
            .iter()
            .cloned()
            .map(|x| OwnerSighting { tag: 0, ..x })
            .collect();
        collapse_edge_pairs(&mut bridge, &pairs());
        assert_eq!(bridge.len(), 4);
        collapse_edge_pairs(&mut v, &pairs());
        assert_eq!(v.len(), 2);
        assert!(v
            .iter()
            .all(|x| x.target == Some(edge()) && x.anchor == AnchorKind::BothEnds));
        let a = assign(
            "p1",
            "Avery",
            &v,
            100.0,
            &params(),
            &NoCorroboration,
            &|_, _| true,
        );
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].target, edge());
    }

    #[test]
    fn strict_alternation_re_anchors_to_edge() {
        let mut v = seq(&["n1", "n2", "n1", "n2", "n9"]);
        apply_alternation(&mut v, &pairs(), 3);
        assert!(v[..4].iter().all(|x| x.target == Some(edge())));
        assert_eq!(v[3].anchor, AnchorKind::NearAlternation);
        assert_eq!(v[4].target, Some(node("n9")));
    }

    #[test]
    fn two_moves_stay_node_assignments() {
        // A, A, B, B, A, A: move to B, then back.
        let mut v = seq(&["n1", "n1", "n2", "n2", "n1", "n1"]);
        apply_alternation(&mut v, &pairs(), 3);
        assert!(v.iter().all(|x| x.anchor == AnchorKind::Near));
        let a = assign(
            "p1",
            "Avery",
            &v,
            100.0,
            &params(),
            &NoCorroboration,
            &|_, _| true,
        );
        let targets: Vec<&OwnerTarget> = a.iter().map(|x| &x.target).collect();
        assert_eq!(targets, vec![&node("n1"), &node("n2"), &node("n1")]);
        assert_eq!(a[1].moved_from, Some(node("n1")));
        assert_eq!(a[2].moved_from, Some(node("n2")));
    }

    #[test]
    fn a_b_b_a_is_not_alternation() {
        let mut v = seq(&["n1", "n2", "n2", "n1"]);
        apply_alternation(&mut v, &pairs(), 3);
        assert!(v.iter().all(|x| x.anchor == AnchorKind::Near));
        // Three alternations is the minimum.
        let mut w = seq(&["n1", "n2", "n1"]);
        apply_alternation(&mut w, &pairs(), 3);
        assert!(w.iter().all(|x| x.anchor == AnchorKind::Near));
    }

    #[test]
    fn close_and_reopen_without_a_move_is_an_assignment() {
        // n3 stays open throughout; n1 disappears for two keyframes, then returns.
        let mut v = Vec::new();
        for (t, with_n1) in [
            (0.0, true),
            (10.0, true),
            (20.0, false),
            (30.0, false),
            (40.0, true),
            (50.0, true),
        ] {
            v.push(s(t, Some(node("n3"))));
            if with_n1 {
                v.push(s(t, Some(node("n1"))));
            }
        }
        let a = assign(
            "p1",
            "Avery",
            &v,
            100.0,
            &params(),
            &NoCorroboration,
            &|_, _| true,
        );
        let n1: Vec<&OwnerAssignment> = a.iter().filter(|x| x.target == node("n1")).collect();
        assert_eq!(n1.len(), 2, "{a:#?}");
        assert_eq!((n1[0].valid_from_s, n1[0].valid_to_s), (0.0, 20.0));
        assert_eq!(n1[1].valid_from_s, 40.0);
        assert!(n1.iter().all(|x| x.moved_from.is_none()));
    }

    #[test]
    fn unread_targets_do_not_close() {
        // n1 and n3 both owned; n3 is not read in two keyframes, which is not absence.
        let mut v = Vec::new();
        for t in [0.0, 10.0, 20.0, 30.0, 40.0] {
            v.push(s(t, Some(node("n1"))));
            if !(15.0..35.0).contains(&t) {
                v.push(OwnerSighting {
                    tag: 1,
                    ..s(t, Some(node("n3")))
                });
            }
        }
        let vis = |kf: &str, t: &OwnerTarget| !(t == &node("n3") && (kf == "kf20" || kf == "kf30"));
        let a = assign("p1", "Avery", &v, 100.0, &params(), &NoCorroboration, &vis);
        assert_eq!(a.len(), 2, "{a:#?}");
        assert!(a.iter().all(|x| x.valid_at(25.0)));
        // Read and untagged: that is absence.
        let b = assign(
            "p1",
            "Avery",
            &v,
            100.0,
            &params(),
            &NoCorroboration,
            &|_, _| true,
        );
        let n3: Vec<_> = b.iter().filter(|x| x.target == node("n3")).collect();
        assert_eq!(n3[0].valid_to_s, 20.0);
    }

    #[test]
    fn a_move_closes_the_old_target_even_if_it_is_not_read_again() {
        // n1 held; n2 tagged twice while n1 is visible at the first of those keyframes
        // only (n1 is not read at the second).
        let v = seq(&["n1", "n1", "n2", "n2"]);
        let vis = |kf: &str, t: &OwnerTarget| !(t == &node("n1") && kf == "kf30");
        let a = assign("p1", "Avery", &v, 100.0, &params(), &NoCorroboration, &vis);
        assert_eq!(a.len(), 2, "{a:#?}");
        assert_eq!(a[0].valid_to_s, 20.0);
        assert_eq!(a[1].moved_from, Some(node("n1")));
    }
}
