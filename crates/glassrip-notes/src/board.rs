//! The `glassrip.board_state` contract, as produced by `glassrip-meeting`.
//!
//! This module re-exports the producer's types and adds read-only helpers the
//! notes and renderer need (lookups, first-seen times, the board end, owners
//! valid at the end, readable event text). Keyframe times are not part of the
//! board state; they come from `glassrip.keyframes` through [`KeyframeTimes`].

use std::collections::BTreeMap;

pub use glassrip_meeting::artifacts::KeyframeView;
pub use glassrip_meeting::consolidate::events::{BoardEvent, EventKind};
pub use glassrip_meeting::consolidate::owners::{OwnerAssignment, OwnerTarget};
pub use glassrip_meeting::consolidate::{
    BoardStateItem, EdgeOrientation, EdgeState, Lifetime, NodeState, StickyKind, StickyState,
};
pub use glassrip_vision::board::{EdgeStyle, StickyColor};
pub use glassrip_vision::BBox;

/// Major version of `glassrip.board_state` this crate reads.
pub const BOARD_STATE_MAJOR: u64 = 1;
/// Major version of `glassrip.keyframes` this crate reads.
pub const KEYFRAMES_MAJOR: u64 = 1;

/// Earliest start over lifetimes (0 when there are none).
pub fn first_seen(lifetimes: &[Lifetime]) -> f64 {
    lifetimes
        .iter()
        .map(|l| l.first_seen_s)
        .reduce(f64::min)
        .unwrap_or(0.0)
}

/// Latest end over lifetimes.
pub fn last_seen(lifetimes: &[Lifetime]) -> Option<f64> {
    lifetimes.iter().map(|l| l.last_seen_s).reduce(f64::max)
}

/// Removal time: the last lifetime's removal, when the element was removed.
pub fn removed_at(lifetimes: &[Lifetime]) -> Option<f64> {
    lifetimes.last().and_then(|l| l.removed_at_s)
}

/// Center of a box.
pub fn center(b: &BBox) -> (f64, f64) {
    ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
}

/// Readable name of an event kind.
pub fn event_kind_text(k: EventKind) -> &'static str {
    match k {
        EventKind::NodeAdded => "box added",
        EventKind::NodeRemoved => "box removed",
        EventKind::LabelChanged => "label changed",
        EventKind::EdgeAdded => "arrow added",
        EventKind::EdgeRemoved => "arrow removed",
        EventKind::EdgeReversed => "arrow reversed",
        EventKind::StickyAdded => "sticky added",
        EventKind::OwnerAssigned => "owner assigned",
        EventKind::OwnerMoved => "owner moved",
    }
}

/// Readable text of one event ("box added: Ledger Store").
pub fn event_text(e: &BoardEvent) -> String {
    if e.detail.trim().is_empty() {
        event_kind_text(e.kind).to_string()
    } else {
        format!("{}: {}", event_kind_text(e.kind), e.detail.trim())
    }
}

/// Id of an owner target (node or edge id).
pub fn target_id(t: &OwnerTarget) -> &str {
    match t {
        OwnerTarget::Node { node_id, .. } => node_id,
        OwnerTarget::Edge { edge_id, .. } => edge_id,
    }
}

/// Readable text of an owner target.
pub fn target_text(t: &OwnerTarget) -> String {
    match t {
        OwnerTarget::Node { text, .. } => text.clone(),
        OwnerTarget::Edge { a_text, b_text, .. } => format!("the {a_text} to {b_text} link"),
    }
}

/// Read-only helpers on a board state.
pub trait BoardExt {
    /// Node by id.
    fn node(&self, id: &str) -> Option<&NodeState>;
    /// Edge by id.
    fn edge(&self, id: &str) -> Option<&EdgeState>;
    /// Sticky by id.
    fn sticky(&self, id: &str) -> Option<&StickyState>;
    /// Nodes present in the final state (every node when none is flagged final).
    fn final_nodes(&self) -> Vec<&NodeState>;
    /// Edges present in the final state whose ends are final nodes.
    fn final_edges(&self) -> Vec<&EdgeState>;
    /// Stickies present in the final state (every sticky when none is flagged).
    fn final_stickies(&self) -> Vec<&StickyState>;
    /// End of the board (final window end, else the latest sighting or event).
    fn end_s(&self) -> f64;
    /// Owner assignments still valid at the end of the board.
    fn current_owners(&self) -> Vec<&OwnerAssignment>;
    /// Events whose subject is `id`.
    fn events_of(&self, id: &str) -> Vec<&BoardEvent>;
}

impl BoardExt for BoardStateItem {
    fn node(&self, id: &str) -> Option<&NodeState> {
        self.nodes.iter().find(|n| n.id == id)
    }

    fn edge(&self, id: &str) -> Option<&EdgeState> {
        self.edges.iter().find(|e| e.id == id)
    }

    fn sticky(&self, id: &str) -> Option<&StickyState> {
        self.stickies.iter().find(|s| s.id == id)
    }

    fn final_nodes(&self) -> Vec<&NodeState> {
        let f: Vec<&NodeState> = self.nodes.iter().filter(|n| n.in_final).collect();
        if f.is_empty() {
            self.nodes.iter().collect()
        } else {
            f
        }
    }

    fn final_edges(&self) -> Vec<&EdgeState> {
        let nodes: Vec<&str> = self.final_nodes().iter().map(|n| n.id.as_str()).collect();
        let ends_ok =
            |e: &&EdgeState| nodes.contains(&e.src.as_str()) && nodes.contains(&e.dst.as_str());
        let f: Vec<&EdgeState> = self
            .edges
            .iter()
            .filter(|e| e.in_final)
            .filter(ends_ok)
            .collect();
        if f.is_empty() && !self.edges.iter().any(|e| e.in_final) {
            self.edges.iter().filter(ends_ok).collect()
        } else {
            f
        }
    }

    fn final_stickies(&self) -> Vec<&StickyState> {
        let f: Vec<&StickyState> = self.stickies.iter().filter(|s| s.in_final).collect();
        if f.is_empty() {
            self.stickies.iter().collect()
        } else {
            f
        }
    }

    fn end_s(&self) -> f64 {
        if let Some(t) = self.t_end_s.or(self.final_window.as_ref().map(|w| w.end_s)) {
            return t;
        }
        let lifetimes = self
            .nodes
            .iter()
            .flat_map(|n| n.lifetimes.iter())
            .chain(self.edges.iter().flat_map(|e| e.lifetimes.iter()))
            .chain(self.stickies.iter().flat_map(|s| s.lifetimes.iter()))
            .map(|l| l.last_seen_s);
        lifetimes
            .chain(self.events.iter().map(|e| e.t_s))
            .fold(0.0, f64::max)
    }

    fn current_owners(&self) -> Vec<&OwnerAssignment> {
        let end = self.end_s();
        let mut v: Vec<&OwnerAssignment> = self
            .owner_assignments
            .iter()
            .filter(|o| o.valid_to_s >= end - 0.5)
            .collect();
        v.sort_by(|a, b| a.valid_from_s.total_cmp(&b.valid_from_s));
        v
    }

    fn events_of(&self, id: &str) -> Vec<&BoardEvent> {
        self.events.iter().filter(|e| e.subject == id).collect()
    }
}

/// Keyframe id to representative time, from `glassrip.keyframes`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeyframeTimes(pub BTreeMap<String, f64>);

impl KeyframeTimes {
    /// Builds the map from keyframe items.
    pub fn from_views<'a, I: IntoIterator<Item = &'a KeyframeView>>(views: I) -> Self {
        Self(
            views
                .into_iter()
                .map(|k| (k.keyframe_id.clone(), k.t_rep_s))
                .collect(),
        )
    }

    /// Time of a keyframe.
    pub fn get(&self, id: &str) -> Option<f64> {
        self.0.get(id).copied()
    }
}

/// Builders for board states in tests and fixtures (the producer's types have
/// many fields that do not matter to the notes or the renderer).
pub mod build {
    use glassrip_meeting::consolidate::owners::OpenReason;
    use glassrip_meeting::consolidate::tracks::ListVotes;
    use glassrip_meeting::consolidate::ElementRegistration;
    use glassrip_meeting::direction::{DirectionBasis, DirectionVotes, EdgeDirection};

    use super::*;

    /// One lifetime from `a` to `b`.
    pub fn life(a: f64, b: f64) -> Lifetime {
        Lifetime {
            first_seen_s: a,
            last_seen_s: b,
            keyframes: 1,
            removed_at_s: None,
        }
    }

    /// An empty final board ending at `end_s`.
    pub fn board(id: &str, end_s: f64) -> BoardStateItem {
        BoardStateItem {
            board_id: id.into(),
            board_title: None,
            groups: vec![],
            folded: vec![],
            is_final: true,
            t_end_s: Some(end_s),
            board_keyframes: vec![],
            registration: vec![],
            final_window: None,
            nodes: vec![],
            edges: vec![],
            stickies: vec![],
            owner_assignments: vec![],
            rejected_owner_tags: vec![],
            events: vec![],
            suppressed_events: vec![],
        }
    }

    /// A final node seen from `first_s`, with an optional box.
    pub fn node(id: &str, text: &str, first_s: f64, end_s: f64, bbox: Option<BBox>) -> NodeState {
        NodeState {
            id: id.into(),
            text: text.into(),
            variants: vec![text.into()],
            variant_counts: vec![],
            lifetimes: vec![life(first_s, end_s)],
            last_seen_s: Some(end_s),
            in_final: true,
            registration: ElementRegistration::Position,
            bbox,
            list_votes: ListVotes::default(),
        }
    }

    /// A final edge `src -> dst`.
    pub fn edge(
        id: &str,
        board: &BoardStateItem,
        src: &str,
        dst: &str,
        label: &str,
        style: EdgeStyle,
    ) -> EdgeState {
        let text = |n: &str| board.node(n).map(|x| x.text.clone()).unwrap_or_default();
        EdgeState {
            id: id.into(),
            a: src.into(),
            b: dst.into(),
            a_text: text(src),
            b_text: text(dst),
            decision: EdgeDirection::AToB,
            src: src.into(),
            dst: dst.into(),
            direction: EdgeOrientation::Forward,
            direction_votes: DirectionVotes::default(),
            direction_basis: DirectionBasis::Pixel,
            label: label.into(),
            style,
            lifetimes: vec![life(0.0, board.end_s())],
            in_final: true,
        }
    }

    /// A final sticky classified like the producer does.
    pub fn sticky(id: &str, text: &str, first_s: f64, end_s: f64) -> StickyState {
        StickyState {
            id: id.into(),
            text: text.into(),
            kind: glassrip_meeting::consolidate::sticky_kind(text),
            color: Some(StickyColor::Yellow),
            bbox: None,
            last_seen: None,
            lifetimes: vec![life(first_s, end_s)],
            in_final: true,
        }
    }

    /// A node target.
    pub fn node_target(board: &BoardStateItem, node_id: &str) -> OwnerTarget {
        OwnerTarget::Node {
            node_id: node_id.into(),
            text: board
                .node(node_id)
                .map(|n| n.text.clone())
                .unwrap_or_default(),
        }
    }

    /// An edge target.
    pub fn edge_target(board: &BoardStateItem, edge_id: &str) -> OwnerTarget {
        let e = board.edge(edge_id);
        OwnerTarget::Edge {
            edge_id: edge_id.into(),
            src: e.map(|e| e.src.clone()).unwrap_or_default(),
            dst: e.map(|e| e.dst.clone()).unwrap_or_default(),
            a_text: e.map(|e| e.a_text.clone()).unwrap_or_default(),
            b_text: e.map(|e| e.b_text.clone()).unwrap_or_default(),
        }
    }

    /// An owner assignment.
    pub fn owner(
        person_id: &str,
        display_name: &str,
        target: OwnerTarget,
        from_s: f64,
        to_s: f64,
        moved_from: Option<OwnerTarget>,
    ) -> OwnerAssignment {
        OwnerAssignment {
            person_id: person_id.into(),
            display_name: display_name.into(),
            name_raw: display_name
                .split_whitespace()
                .next()
                .unwrap_or(display_name)
                .into(),
            target,
            valid_from_s: from_s,
            valid_to_s: to_s,
            opened_at_keyframe: String::new(),
            opened_by: OpenReason::ConsistentKeyframes,
            corroboration: None,
            moved_from,
            backfill_from_s: None,
            sightings: vec![],
        }
    }

    /// An event.
    pub fn event(
        id: &str,
        kind: EventKind,
        t_s: f64,
        keyframe_id: &str,
        subject: &str,
        detail: &str,
    ) -> BoardEvent {
        BoardEvent {
            event_id: id.into(),
            kind,
            t_s,
            keyframe_id: keyframe_id.into(),
            subject: subject.into(),
            detail: detail.into(),
            ink_change: Some(0.2),
            baseline: false,
            owner_target: None,
            owner_from: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn life(a: f64, b: f64, removed: Option<f64>) -> Lifetime {
        Lifetime {
            first_seen_s: a,
            last_seen_s: b,
            keyframes: 1,
            removed_at_s: removed,
        }
    }

    #[test]
    fn lifetime_helpers() {
        let l = [life(20.0, 40.0, Some(45.0)), life(60.0, 90.0, None)];
        assert_eq!(first_seen(&l), 20.0);
        assert_eq!(last_seen(&l), Some(90.0));
        assert_eq!(removed_at(&l), None);
        assert_eq!(removed_at(&l[..1]), Some(45.0));
        assert_eq!(first_seen(&[]), 0.0);
    }
}
