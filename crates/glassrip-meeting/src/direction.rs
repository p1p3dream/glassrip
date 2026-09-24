//! Edge direction evidence and the cross-frame vote (spec 6.11, edge direction).
//!
//! Every keyframe contributes up to two independent verdicts per edge: the pixel check
//! (skeleton walk plus arrowhead blob test) and the binary VLM check on endpoint
//! crops. Verdicts are weighted by crop sharpness and zoom and summed per channel. A
//! direction is kept only when both channels elect the same winner; otherwise the edge
//! is `uncertain`. `bidirectional` needs both channels to see arrowheads at both ends,
//! so opposite single-ended readings are never merged into it.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Verdict for one reading of an edge, relative to the edge's `(src, dst)` as read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EndVerdict {
    /// Arrowhead at `dst` only.
    Forward,
    /// Arrowhead at `src` only.
    Reverse,
    /// Arrowheads at both ends.
    Bidirectional,
    /// Both ends found, neither has an arrowhead.
    NoArrowhead,
    /// The check could not decide.
    Unknown,
}

impl EndVerdict {
    /// Verdict from arrowhead presence at each end (`None` = end not observed).
    pub fn from_ends(src_arrow: Option<bool>, dst_arrow: Option<bool>) -> Self {
        match (src_arrow, dst_arrow) {
            (Some(true), Some(true)) => Self::Bidirectional,
            (Some(false), Some(true)) | (None, Some(true)) => Self::Forward,
            (Some(true), Some(false)) | (Some(true), None) => Self::Reverse,
            (Some(false), Some(false)) => Self::NoArrowhead,
            _ => Self::Unknown,
        }
    }

    /// The same verdict seen from the opposite orientation.
    pub fn flipped(self) -> Self {
        match self {
            Self::Forward => Self::Reverse,
            Self::Reverse => Self::Forward,
            other => other,
        }
    }
}

/// Answer of the binary VLM check for one endpoint crop. `A` is the box at this end,
/// `B` the box at the other end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum EndpointAnswer {
    /// An arrowhead at this end points into box A.
    #[serde(rename = "points_to_A")]
    PointsToA,
    /// The arrowhead at this end points away from A, toward B.
    #[serde(rename = "points_to_B")]
    PointsToB,
    /// The line ends here without an arrowhead.
    #[serde(rename = "no_arrowhead")]
    NoArrowhead,
    /// Not decidable from the crop.
    #[serde(rename = "unclear")]
    Unclear,
}

/// Combine the answers for the `src` end (A = src) and the `dst` end (A = dst).
pub fn vlm_verdict(src_end: EndpointAnswer, dst_end: EndpointAnswer) -> EndVerdict {
    // An arrowhead pointing into the box at this end marks this end as the head.
    let into = |a: EndpointAnswer| match a {
        EndpointAnswer::PointsToA => Some(true),
        EndpointAnswer::NoArrowhead => Some(false),
        EndpointAnswer::PointsToB | EndpointAnswer::Unclear => None,
    };
    let (s, d) = (into(src_end), into(dst_end));
    // "points_to_B" at one end is direct evidence that the other end is the head.
    let s = s.or((dst_end == EndpointAnswer::PointsToB).then_some(true));
    let d = d.or((src_end == EndpointAnswer::PointsToB).then_some(true));
    EndVerdict::from_ends(s, d)
}

/// Final direction of a consolidated edge between nodes `a` and `b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EdgeDirection {
    /// From `a` to `b`.
    AToB,
    /// From `b` to `a`.
    BToA,
    /// Arrowheads at both ends.
    Bidirectional,
    /// Channels disagree or there is no decisive evidence.
    Uncertain,
}

/// How the final direction was decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DirectionPolicy {
    /// Keep a direction only when the pixel and VLM votes agree (spec default).
    RequireAgreement,
    /// Pixel vote alone; for runs without a vision backend. Recorded in the output.
    PixelOnly,
}

/// Weighted votes of one channel, in the `(a, b)` orientation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChannelVotes {
    /// Weight for `a -> b`.
    pub a_to_b: f64,
    /// Weight for `b -> a`.
    pub b_to_a: f64,
    /// Weight for arrowheads at both ends.
    pub both: f64,
    /// Weight for no arrowhead at either end.
    pub none: f64,
    /// Frames that voted (any verdict other than unknown).
    pub frames: u32,
}

impl ChannelVotes {
    /// Add a verdict already expressed in the `(a, b)` orientation.
    pub fn add(&mut self, v: EndVerdict, weight: f64) {
        let w = if weight.is_finite() {
            weight.max(0.0)
        } else {
            0.0
        };
        match v {
            EndVerdict::Forward => self.a_to_b += w,
            EndVerdict::Reverse => self.b_to_a += w,
            EndVerdict::Bidirectional => self.both += w,
            EndVerdict::NoArrowhead => self.none += w,
            EndVerdict::Unknown => return,
        }
        self.frames += 1;
    }

    /// The winning direction when it holds at least `min_share` of the decisive
    /// weight (`a_to_b + b_to_a + both`).
    pub fn winner(&self, min_share: f64) -> Option<EdgeDirection> {
        let decisive = self.a_to_b + self.b_to_a + self.both;
        if decisive <= 0.0 {
            return None;
        }
        let options = [
            (EdgeDirection::AToB, self.a_to_b),
            (EdgeDirection::BToA, self.b_to_a),
            (EdgeDirection::Bidirectional, self.both),
        ];
        let mut best = options[0];
        let mut tie = false;
        for o in &options[1..] {
            if o.1 > best.1 {
                best = *o;
                tie = false;
            } else if o.1 == best.1 {
                tie = true;
            }
        }
        (!tie && best.1 / decisive >= min_share).then_some(best.0)
    }
}

/// Both channels' votes for one consolidated edge (`direction_votes` in the artifact).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DirectionVotes {
    /// Pixel check votes.
    pub pixel: ChannelVotes,
    /// VLM endpoint check votes.
    pub vlm: ChannelVotes,
    /// The board reader's own `src -> dst` (informational; never decides).
    pub reader: ChannelVotes,
}

impl DirectionVotes {
    /// Apply the policy: see the module docs.
    pub fn decide(&self, policy: DirectionPolicy, min_share: f64) -> EdgeDirection {
        let p = self.pixel.winner(min_share);
        match policy {
            DirectionPolicy::PixelOnly => p.unwrap_or(EdgeDirection::Uncertain),
            DirectionPolicy::RequireAgreement => match (p, self.vlm.winner(min_share)) {
                (Some(a), Some(b)) if a == b => a,
                _ => EdgeDirection::Uncertain,
            },
        }
    }
}

/// Frame weight from crop sharpness and zoom, each relative to the run median and
/// clamped to `[0.25, 2]`, so no single frame dominates the vote.
pub fn frame_weight(sharpness: f64, median_sharpness: f64, zoom: f64, median_zoom: f64) -> f64 {
    let rel = |v: f64, m: f64| {
        if v.is_finite() && m.is_finite() && m > 0.0 {
            (v / m).clamp(0.25, 2.0)
        } else {
            1.0
        }
    };
    rel(sharpness, median_sharpness) * rel(zoom, median_zoom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn ends_to_verdict() {
        assert_eq!(
            EndVerdict::from_ends(Some(false), Some(true)),
            EndVerdict::Forward
        );
        assert_eq!(EndVerdict::from_ends(Some(true), None), EndVerdict::Reverse);
        assert_eq!(
            EndVerdict::from_ends(Some(true), Some(true)),
            EndVerdict::Bidirectional
        );
        assert_eq!(
            EndVerdict::from_ends(None, Some(false)),
            EndVerdict::Unknown
        );
        assert_eq!(EndVerdict::Forward.flipped(), EndVerdict::Reverse);
    }

    #[test]
    fn vlm_answers_combine() {
        use EndpointAnswer::*;
        assert_eq!(vlm_verdict(NoArrowhead, PointsToA), EndVerdict::Forward);
        assert_eq!(vlm_verdict(PointsToA, NoArrowhead), EndVerdict::Reverse);
        assert_eq!(vlm_verdict(Unclear, PointsToB), EndVerdict::Reverse);
        assert_eq!(vlm_verdict(PointsToA, PointsToA), EndVerdict::Bidirectional);
        assert_eq!(vlm_verdict(Unclear, Unclear), EndVerdict::Unknown);
    }

    #[test]
    fn agreement_rule() {
        let mut v = DirectionVotes::default();
        v.pixel.add(EndVerdict::Forward, 1.0);
        v.vlm.add(EndVerdict::Forward, 0.5);
        assert_eq!(
            v.decide(DirectionPolicy::RequireAgreement, 0.6),
            EdgeDirection::AToB
        );
        v.vlm.add(EndVerdict::Reverse, 2.0);
        assert_eq!(
            v.decide(DirectionPolicy::RequireAgreement, 0.6),
            EdgeDirection::Uncertain
        );
        assert_eq!(
            v.decide(DirectionPolicy::PixelOnly, 0.6),
            EdgeDirection::AToB
        );
    }

    #[test]
    fn opposite_readings_never_become_bidirectional() {
        let mut v = DirectionVotes::default();
        v.pixel.add(EndVerdict::Forward, 1.0);
        v.pixel.add(EndVerdict::Reverse, 1.0);
        v.vlm.add(EndVerdict::Forward, 1.0);
        v.vlm.add(EndVerdict::Reverse, 1.0);
        assert_eq!(
            v.decide(DirectionPolicy::RequireAgreement, 0.6),
            EdgeDirection::Uncertain
        );
        let mut b = DirectionVotes::default();
        b.pixel.add(EndVerdict::Bidirectional, 1.0);
        b.vlm.add(EndVerdict::Bidirectional, 1.0);
        assert_eq!(
            b.decide(DirectionPolicy::RequireAgreement, 0.6),
            EdgeDirection::Bidirectional
        );
    }

    proptest! {
        #[test]
        fn voting_is_order_independent(votes in proptest::collection::vec((0u8..5, 0.0f64..3.0), 0..20)) {
            let to_v = |k: u8| match k {
                0 => EndVerdict::Forward,
                1 => EndVerdict::Reverse,
                2 => EndVerdict::Bidirectional,
                3 => EndVerdict::NoArrowhead,
                _ => EndVerdict::Unknown,
            };
            let mut fwd = ChannelVotes::default();
            for (k, w) in &votes { fwd.add(to_v(*k), *w); }
            let mut rev = ChannelVotes::default();
            for (k, w) in votes.iter().rev() { rev.add(to_v(*k), *w); }
            prop_assert_eq!(fwd.winner(0.6), rev.winner(0.6));
            prop_assert_eq!(fwd.frames, rev.frames);
        }
    }
}
