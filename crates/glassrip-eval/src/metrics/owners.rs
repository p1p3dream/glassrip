//! Owner metrics over time: attribution at probe times and owner-move time error
//! (spec 9.2, 9.3).
//!
//! Owners are timed assignments `(person, target, [valid_from_s, valid_to_s))`,
//! where the target is a node or an edge (unordered endpoint pair). Predicted
//! assignments are resolved to gold person ids and gold node ids before scoring
//! (see [`crate::suite`]); unresolved ones never match.

use serde::{Deserialize, Serialize};

use super::Tally;

/// What an owner tag is anchored to (gold node ids).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Target {
    /// A node.
    Node {
        /// Node id.
        node: String,
    },
    /// An edge; endpoint order is ignored.
    Edge {
        /// One endpoint node id.
        src: String,
        /// Other endpoint node id.
        dst: String,
    },
}

impl Target {
    /// Equality with edges compared as unordered pairs.
    pub fn same_as(&self, other: &Target) -> bool {
        match (self, other) {
            (Target::Node { node: a }, Target::Node { node: b }) => a == b,
            (Target::Edge { src: a1, dst: a2 }, Target::Edge { src: b1, dst: b2 }) => {
                (a1 == b1 && a2 == b2) || (a1 == b2 && a2 == b1)
            }
            _ => false,
        }
    }
}

/// A timed owner assignment (gold, or predicted after resolution).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assignment {
    /// Gold person id.
    pub person_id: String,
    /// Anchor.
    pub target: Target,
    /// Start of validity, seconds.
    pub valid_from_s: f64,
    /// End of validity (exclusive); `None` means until the end.
    #[serde(default)]
    pub valid_to_s: Option<f64>,
}

impl Assignment {
    /// True when valid at `t`.
    pub fn active_at(&self, t: f64) -> bool {
        self.valid_from_s <= t && self.valid_to_s.is_none_or(|end| t < end)
    }
}

/// A gold owner move.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Move {
    /// Person id.
    pub person_id: String,
    /// Previous anchor.
    pub from: Target,
    /// New anchor.
    pub to: Target,
    /// Time of the move, seconds.
    pub t_s: f64,
}

/// Result at one probe time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeResult {
    /// Probe time.
    pub t_s: f64,
    /// Whether the probe joined to a predicted keyframe.
    pub joined: bool,
    /// Gold assignments active at the probe.
    pub expected: usize,
    /// Of those, found in the prediction.
    pub correct: usize,
    /// Predicted active assignments not in the gold set.
    pub extra: usize,
}

/// Owner attribution score.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AttributionScore {
    /// Correct over expected (person, target) pairs across probes.
    pub tally: Tally,
    /// Predicted pairs not expected, across probes.
    pub extra: usize,
    /// Per-probe detail.
    pub probes: Vec<ProbeResult>,
}

/// Owner attribution at probe times. `joined(t)` says whether the probe time
/// joins to a predicted keyframe (9.2); unjoined probes score all expected pairs wrong.
pub fn owner_attribution(
    gold: &[Assignment],
    pred: &[Assignment],
    probes: &[f64],
    joined: impl Fn(f64) -> bool,
) -> AttributionScore {
    let mut out = AttributionScore::default();
    for &t in probes {
        let expected: Vec<&Assignment> = gold.iter().filter(|a| a.active_at(t)).collect();
        let is_joined = joined(t);
        let active: Vec<&Assignment> = if is_joined {
            pred.iter().filter(|a| a.active_at(t)).collect()
        } else {
            Vec::new()
        };
        let found = |a: &Assignment, set: &[&Assignment]| {
            set.iter()
                .any(|b| b.person_id == a.person_id && b.target.same_as(&a.target))
        };
        let correct = expected.iter().filter(|a| found(a, &active)).count();
        let extra = active.iter().filter(|a| !found(a, &expected)).count();
        for i in 0..expected.len() {
            out.tally.record(i < correct);
        }
        out.extra += extra;
        out.probes.push(ProbeResult {
            t_s: t,
            joined: is_joined,
            expected: expected.len(),
            correct,
            extra,
        });
    }
    out
}

/// Result for one gold move.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoveResult {
    /// Person id.
    pub person_id: String,
    /// Gold move time.
    pub gold_t_s: f64,
    /// Predicted start of the new assignment closest to the gold time.
    pub pred_t_s: Option<f64>,
    /// `max(0, |pred - gold| - tolerance)`; `None` when the move was not found.
    pub error_s: Option<f64>,
}

/// Owner-move time error: for each gold move, the predicted assignment of the
/// same person to the new target whose `valid_from_s` is closest to the gold
/// time. The 9.2 join tolerance (`tolerance_s`) is subtracted from the error.
pub fn owner_move_errors(moves: &[Move], pred: &[Assignment], tolerance_s: f64) -> Vec<MoveResult> {
    moves
        .iter()
        .map(|m| {
            let pred_t = pred
                .iter()
                .filter(|a| a.person_id == m.person_id && a.target.same_as(&m.to))
                .map(|a| a.valid_from_s)
                .min_by(|a, b| (a - m.t_s).abs().total_cmp(&(b - m.t_s).abs()));
            MoveResult {
                person_id: m.person_id.clone(),
                gold_t_s: m.t_s,
                pred_t_s: pred_t,
                error_s: pred_t.map(|p| ((p - m.t_s).abs() - tolerance_s).max(0.0)),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(n: &str) -> Target {
        Target::Node { node: n.into() }
    }

    fn a(p: &str, t: Target, from: f64, to: Option<f64>) -> Assignment {
        Assignment {
            person_id: p.into(),
            target: t,
            valid_from_s: from,
            valid_to_s: to,
        }
    }

    #[test]
    fn edge_targets_unordered() {
        let e1 = Target::Edge {
            src: "x".into(),
            dst: "y".into(),
        };
        let e2 = Target::Edge {
            src: "y".into(),
            dst: "x".into(),
        };
        assert!(e1.same_as(&e2));
        assert!(!e1.same_as(&node("x")));
    }

    #[test]
    fn attribution_hand_computed() {
        let gold = vec![
            a("p1", node("app"), 100.0, None),
            a("p2", node("cms"), 100.0, Some(200.0)),
            a("p2", node("app"), 200.0, None),
        ];
        let pred = vec![
            a("p1", node("app"), 90.0, None),
            // p2 stays on cms too long (until 250) and moves late
            a("p2", node("cms"), 100.0, Some(250.0)),
            a("p2", node("app"), 250.0, None),
        ];
        // probe 150: expected {p1 app, p2 cms}: both found, 0 extra
        // probe 220: expected {p1 app, p2 app}: p1 found, p2 not; extra p2 cms
        // probe 300: unjoined: expected 2, both wrong
        let s = owner_attribution(&gold, &pred, &[150.0, 220.0, 300.0], |t| t < 299.0);
        assert_eq!(
            s.tally,
            Tally {
                correct: 3,
                total: 6
            }
        );
        assert_eq!(s.extra, 1);
        assert_eq!(s.probes[1].correct, 1);
        assert!(!s.probes[2].joined);
    }

    #[test]
    fn move_error_subtracts_tolerance() {
        let moves = vec![Move {
            person_id: "p2".into(),
            from: node("cms"),
            to: node("app"),
            t_s: 200.0,
        }];
        let pred = vec![
            a("p2", node("app"), 250.0, None),
            a("p2", node("app"), 190.0, Some(195.0)),
        ];
        let r = owner_move_errors(&moves, &pred, 2.0);
        // closest start is 190: |190-200| - 2 = 8
        assert_eq!(r[0].pred_t_s, Some(190.0));
        assert_eq!(r[0].error_s, Some(8.0));
        let none = owner_move_errors(&moves, &[], 2.0);
        assert_eq!(none[0].error_s, None);
        let within = owner_move_errors(&moves, &[a("p2", node("app"), 201.5, None)], 2.0);
        assert_eq!(within[0].error_s, Some(0.0));
    }
}
