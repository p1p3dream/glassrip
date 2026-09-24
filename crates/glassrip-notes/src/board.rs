//! Local copy of the `glassrip.board_state` item shape (design section 7).
//!
//! UNIFY: the board-state consolidation crate owns this contract. Until it merges,
//! `notes` and `render` read board state through these types; when it lands,
//! replace this module with a re-export (or a `From` conversion) of its types and
//! keep the field names below, which follow the data contract: nodes with
//! lifetimes and variants, edges with direction votes, owner assignments with
//! validity intervals, and events. Stickies, groups (grids and containers) and
//! the keyframes that support each element are included because the notes and
//! the SVG cite and draw them.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Current major version of the board state contract read here.
pub const BOARD_STATE_MAJOR: u64 = 1;

/// Axis-aligned box in board canvas coordinates (any unit; only relative
/// positions matter).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BBox {
    /// Left.
    pub x: f64,
    /// Top.
    pub y: f64,
    /// Width.
    pub w: f64,
    /// Height.
    pub h: f64,
}

impl BBox {
    /// Center point.
    pub fn center(&self) -> (f64, f64) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
}

/// A box on the board.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BoardNode {
    /// Stable node id.
    pub node_id: String,
    /// Consolidated label.
    pub text: String,
    /// Other readings of the label seen over time.
    #[serde(default)]
    pub variants: Vec<String>,
    /// Canvas position, when known.
    #[serde(default)]
    pub bbox: Option<BBox>,
    /// First time the node was seen, seconds.
    pub first_seen_s: f64,
    /// Last time the node was seen, seconds (None: still present at the end).
    #[serde(default)]
    pub last_seen_s: Option<f64>,
    /// Keyframes supporting the node.
    #[serde(default)]
    pub keyframe_ids: Vec<String>,
}

/// Line style of an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EdgeStyle {
    /// Solid line.
    #[default]
    Solid,
    /// Dashed or dotted line.
    Dashed,
}

/// A directed connection (`src` is the tail, `dst` the arrowhead end).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BoardEdge {
    /// Stable edge id.
    pub edge_id: String,
    /// Tail node id.
    pub src: String,
    /// Head node id.
    pub dst: String,
    /// Text written on the line.
    #[serde(default)]
    pub label: Option<String>,
    /// Line style.
    #[serde(default)]
    pub style: EdgeStyle,
    /// Direction votes by outcome (for example `forward`, `reverse`, `none`).
    #[serde(default)]
    pub direction_votes: BTreeMap<String, u32>,
    /// Keyframes supporting the edge.
    #[serde(default)]
    pub keyframe_ids: Vec<String>,
}

/// What a sticky note says, by its form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StickyKind {
    /// A question (text ends in `?`).
    Question,
    /// An idea.
    Idea,
    /// Anything else.
    #[default]
    Note,
}

/// A sticky note on the canvas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Sticky {
    /// Stable sticky id.
    pub sticky_id: String,
    /// Full text.
    pub text: String,
    /// Kind (derived from the text when absent).
    #[serde(default)]
    pub kind: Option<StickyKind>,
    /// Canvas position.
    #[serde(default)]
    pub bbox: Option<BBox>,
    /// First time seen, seconds.
    pub first_seen_s: f64,
    /// Keyframes supporting the sticky.
    #[serde(default)]
    pub keyframe_ids: Vec<String>,
}

impl Sticky {
    /// Kind, derived from the text when the state does not say.
    pub fn effective_kind(&self) -> StickyKind {
        if let Some(k) = self.kind {
            return k;
        }
        let t = self.text.trim();
        if t.ends_with('?') {
            StickyKind::Question
        } else if t.to_lowercase().starts_with("idea") {
            StickyKind::Idea
        } else {
            StickyKind::Note
        }
    }
}

/// What an owner tag points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TargetKind {
    /// A node.
    Node,
    /// An edge.
    Edge,
}

/// An owner tag valid over an interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OwnerAssignment {
    /// Stable id.
    pub owner_id: String,
    /// Resolved participant id, when the tag matched one.
    #[serde(default)]
    pub person_id: Option<String>,
    /// Name as written on the tag.
    pub name_raw: String,
    /// Target kind.
    pub target_kind: TargetKind,
    /// Target node or edge id.
    pub target_id: String,
    /// Start of validity, seconds.
    pub valid_from_s: f64,
    /// End of validity, seconds (None: still valid at the end).
    #[serde(default)]
    pub valid_to_s: Option<f64>,
    /// Keyframes supporting the assignment.
    #[serde(default)]
    pub keyframe_ids: Vec<String>,
}

/// A computed board change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BoardEvent {
    /// Stable event id.
    pub event_id: String,
    /// Time, seconds.
    pub t_s: f64,
    /// Event kind (for example `node_added`, `owner_moved`).
    pub kind: String,
    /// One-line description computed from the diff.
    pub summary: String,
    /// Ids of the elements involved.
    #[serde(default)]
    pub refs: Vec<String>,
    /// Keyframes before and after the change.
    #[serde(default)]
    pub keyframe_ids: Vec<String>,
}

/// Kind of a group of board elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GroupKind {
    /// Cards arranged in a grid under a title.
    Grid,
    /// A labelled container holding cards (for example a module list).
    Container,
}

/// One card inside a group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GroupMember {
    /// Card text.
    pub text: String,
    /// Secondary text (for example after a separator on the card).
    #[serde(default)]
    pub detail: Option<String>,
    /// Card drawn in a distinct color on the board.
    #[serde(default)]
    pub highlight: bool,
}

/// A grid or container of cards.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BoardGroup {
    /// Stable group id.
    pub group_id: String,
    /// Title.
    pub label: String,
    /// Kind.
    pub kind: GroupKind,
    /// Cards in reading order.
    pub members: Vec<GroupMember>,
    /// Grid columns (grids only).
    #[serde(default)]
    pub columns: Option<u32>,
    /// Node that feeds this container, when an arrow points into it.
    #[serde(default)]
    pub fed_by: Option<String>,
    /// Canvas position.
    #[serde(default)]
    pub bbox: Option<BBox>,
    /// First time seen, seconds.
    pub first_seen_s: f64,
    /// Keyframes supporting the group.
    #[serde(default)]
    pub keyframe_ids: Vec<String>,
}

/// A keyframe the board state refers to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct KeyframeRef {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Representative time, seconds.
    pub t_rep_s: f64,
}

/// Consolidated state of one board.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BoardState {
    /// Board id.
    pub board_id: String,
    /// Board title read from the app chrome, when available.
    #[serde(default)]
    pub title: Option<String>,
    /// Time of the final state, seconds.
    pub final_t_s: f64,
    /// Nodes.
    pub nodes: Vec<BoardNode>,
    /// Edges.
    pub edges: Vec<BoardEdge>,
    /// Stickies.
    #[serde(default)]
    pub stickies: Vec<Sticky>,
    /// Owner assignments.
    #[serde(default)]
    pub owner_assignments: Vec<OwnerAssignment>,
    /// Events.
    #[serde(default)]
    pub events: Vec<BoardEvent>,
    /// Grids and containers.
    #[serde(default)]
    pub groups: Vec<BoardGroup>,
    /// Keyframes referenced by this state.
    #[serde(default)]
    pub keyframes: Vec<KeyframeRef>,
}

impl BoardState {
    /// Node by id.
    pub fn node(&self, id: &str) -> Option<&BoardNode> {
        self.nodes.iter().find(|n| n.node_id == id)
    }

    /// Edge by id.
    pub fn edge(&self, id: &str) -> Option<&BoardEdge> {
        self.edges.iter().find(|e| e.edge_id == id)
    }

    /// Nodes present at the final state.
    pub fn final_nodes(&self) -> impl Iterator<Item = &BoardNode> {
        self.nodes.iter().filter(|n| n.last_seen_s.is_none())
    }

    /// True when `id` names an event of this board.
    pub fn has_event(&self, id: &str) -> bool {
        self.events.iter().any(|e| e.event_id == id)
    }

    /// True when `id` names a keyframe referenced by this board.
    pub fn has_keyframe(&self, id: &str) -> bool {
        self.keyframes.iter().any(|k| k.keyframe_id == id)
    }

    /// Time of an event or keyframe id.
    pub fn time_of(&self, id: &str) -> Option<f64> {
        self.events
            .iter()
            .find(|e| e.event_id == id)
            .map(|e| e.t_s)
            .or_else(|| {
                self.keyframes
                    .iter()
                    .find(|k| k.keyframe_id == id)
                    .map(|k| k.t_rep_s)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sticky_kind_is_derived_from_text() {
        let s = |t: &str| Sticky {
            sticky_id: "s".into(),
            text: t.into(),
            kind: None,
            bbox: None,
            first_seen_s: 0.0,
            keyframe_ids: vec![],
        };
        assert_eq!(
            s("Which cache do we need?").effective_kind(),
            StickyKind::Question
        );
        assert_eq!(s("Idea: badge wall").effective_kind(), StickyKind::Idea);
        assert_eq!(s("Clone the lobby page").effective_kind(), StickyKind::Note);
    }

    #[test]
    fn minimal_state_parses_with_defaults() {
        let v = serde_json::json!({
            "board_id": "b1", "final_t_s": 10.0,
            "nodes": [{"node_id": "n1", "text": "Relay", "first_seen_s": 1.0}],
            "edges": []
        });
        let b: BoardState = serde_json::from_value(v).unwrap();
        assert_eq!(b.final_nodes().count(), 1);
        assert!(b.stickies.is_empty());
    }
}
