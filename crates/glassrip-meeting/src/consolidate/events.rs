//! Computed change events, gated on aligned ink change.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::owners::OwnerTarget;

/// Event kinds (spec 6.11).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
pub enum EventKind {
    /// A node appeared.
    NodeAdded,
    /// A node was removed (visible place, absent).
    NodeRemoved,
    /// A node at the same place got different text.
    LabelChanged,
    /// An edge appeared.
    EdgeAdded,
    /// An edge was removed.
    EdgeRemoved,
    /// An edge's direction flipped.
    EdgeReversed,
    /// A sticky appeared.
    StickyAdded,
    /// An owner was first assigned.
    OwnerAssigned,
    /// An owner moved to another target.
    OwnerMoved,
}

/// One event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardEvent {
    /// Stable id within the board state (assigned in time order).
    pub event_id: String,
    /// Kind.
    pub kind: EventKind,
    /// Time (start of the keyframe where the change is first seen).
    pub t_s: f64,
    /// Keyframe id.
    pub keyframe_id: String,
    /// Subject id (node, edge, sticky, or person).
    pub subject: String,
    /// Human-readable detail (element texts).
    pub detail: String,
    /// Aligned ink change at the keyframe, when known.
    pub ink_change: Option<f64>,
    /// In the first board keyframe (initial content, not a change during the meeting).
    pub baseline: bool,
    /// Owner events: the target the owner tag is on after the change, by id, so
    /// consumers match an event to its assignment exactly (texts overlap: a tag
    /// on "Ledger Store" is not a tag on "Ledger"). Absent on other kinds and in
    /// states written before it existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_target: Option<OwnerTarget>,
    /// Owner moves: the target the tag left.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_from: Option<OwnerTarget>,
}

/// Why an event was not emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SuppressReason {
    /// Aligned ink change below the threshold (pan or zoom only).
    InkBelowThreshold,
    /// The ink change of this keyframe pair is unknown (non-adjacent keyframes or no
    /// alignment), so the change cannot be told from pan or zoom.
    InkUnknown,
}

/// A computed state change that did not pass the gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SuppressedEvent {
    /// The event that would have been emitted.
    pub event: BoardEvent,
    /// Reason.
    pub reason: SuppressReason,
}

/// Sorts candidates into emitted and suppressed events.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EventGate {
    /// Threshold on aligned ink change.
    pub threshold: f64,
    /// Emitted.
    pub events: Vec<BoardEvent>,
    /// Suppressed.
    pub suppressed: Vec<SuppressedEvent>,
}

impl EventGate {
    /// Gate with a threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Self::default()
        }
    }

    /// Emit when the event is baseline or the keyframe pair's ink change reaches the
    /// threshold; otherwise record it as suppressed (below threshold, or unknown).
    pub fn offer(&mut self, e: BoardEvent) {
        let reason = match (e.baseline, e.ink_change) {
            (true, _) => None,
            (false, Some(v)) if v >= self.threshold => None,
            (false, Some(_)) => Some(SuppressReason::InkBelowThreshold),
            (false, None) => Some(SuppressReason::InkUnknown),
        };
        match reason {
            None => self.events.push(e),
            Some(reason) => self.suppressed.push(SuppressedEvent { event: e, reason }),
        }
    }

    /// Sort both lists by time, then kind, then subject, and number the events.
    pub fn finish(mut self) -> (Vec<BoardEvent>, Vec<SuppressedEvent>) {
        let key = |e: &BoardEvent| (e.t_s, e.kind, e.subject.clone());
        self.events.sort_by(|a, b| {
            key(a)
                .partial_cmp(&key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.suppressed.sort_by(|a, b| {
            key(&a.event)
                .partial_cmp(&key(&b.event))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, e) in self.events.iter_mut().enumerate() {
            e.event_id = format!("ev-{}", i + 1);
        }
        for (i, e) in self.suppressed.iter_mut().enumerate() {
            e.event.event_id = format!("sup-{}", i + 1);
        }
        (self.events, self.suppressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(ink: Option<f64>, baseline: bool) -> BoardEvent {
        BoardEvent {
            event_id: String::new(),
            kind: EventKind::NodeAdded,
            t_s: 1.0,
            keyframe_id: "kf".into(),
            subject: "node-1".into(),
            detail: String::new(),
            ink_change: ink,
            baseline,
            owner_target: None,
            owner_from: None,
        }
    }

    #[test]
    fn gate_rules() {
        let mut g = EventGate::new(0.05);
        g.offer(ev(Some(0.01), false));
        g.offer(ev(Some(0.2), false));
        g.offer(ev(Some(0.0), true));
        g.offer(ev(None, false));
        let (e, s) = g.finish();
        assert_eq!(e.len(), 2);
        assert_eq!(s.len(), 2);
        assert!(s
            .iter()
            .any(|x| x.reason == SuppressReason::InkBelowThreshold));
        assert!(s.iter().any(|x| x.reason == SuppressReason::InkUnknown));
    }
}
