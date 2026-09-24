//! Board-state consolidation (spec 6.11, board state).
//!
//! Pipeline over the board keyframes in time order:
//!
//! 1. Register keyframes into canvas clusters through text anchors ([`crate::register`]).
//! 2. Track text elements across keyframes: fuzzy text ([`crate::difflib`] at 0.85)
//!    plus, within a registered cluster, position (same text elsewhere is another
//!    element, except a non-sticky reading within `off_position_share` of the
//!    diagonal after the view moved, which is taken as an imprecise box; different
//!    text at the same node place is a label variant). Nodes,
//!    stickies, other text and edge labels share one pool, so an element read in
//!    different lists in different keyframes is one track whose kind is the majority
//!    list. Illegible or elided texts are not tracked.
//! 3. Lifetimes and support ([`tracks::intervals`]). The final state is what was
//!    observed and not later removed: an element is final when its last supported
//!    interval was never ended by a removal. Removal needs evidence: consecutive
//!    later keyframes that cover the element's region and lack it. A keyframe covers
//!    the region when the region lies inside its registered view, OCR read none of
//!    the element's text there, and the keyframe read another established element
//!    near that place, which confirms the registration there. Absence outside the
//!    view, or in a view with nothing known read near the place, is not removal.
//!    An element never placed on the board (text-only registration) seen in a
//!    single keyframe cannot be checked against later views and is not final.
//! 4. Edges are keyed by their two node tracks; an endpoint that is not a supported
//!    node drops the edge (nodes are never created from endpoints). Directions come
//!    from the weighted vote over `glassrip.edge_direction` evidence. An edge is
//!    absent from a keyframe that read (or covers) both of its ends without it, but
//!    only once the board's aligned ink changed since the edge was last read:
//!    readers often leave a connector out of a reading, erasing one changes ink.
//! 5. Owner tags become timed assignments ([`owners`]).
//! 6. Events are computed from the state changes and gated on ink ([`events`]).

pub mod anchor;
mod cleanup;
pub mod events;
pub mod owners;
pub mod tracks;

use std::collections::{BTreeMap, HashMap, HashSet};

use glassrip_vision::board::{EdgeStyle, ElementList, RejectReason, StickyColor, ValidatedBoard};
use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::artifacts::{CanvasDims, EdgeDirectionItem, EdgeEvidence};
use crate::direction::{frame_weight, DirectionBasis, DirectionVotes, EdgeDirection, EndVerdict};
use crate::pixel_direction::exit_point;
use crate::register::{
    map_bbox, register, Anchors, FrameRegistration, RegistrationMode, RegistrationParams,
};
use crate::text::{clean_label, is_unreliable, normalize, AliasTable};

use anchor::{anchor_tag, box_distance, AnchorParams, Anchored, EdgeGeom};
use cleanup::{derive_groups, fold_fragments, Titles};
use events::{BoardEvent, EventGate, EventKind, SuppressedEvent};
use owners::{
    apply_alternation, assign, collapse_edge_pairs, AnchorKind, Corroborator, OwnerAssignment,
    OwnerParams, OwnerSighting, OwnerTarget,
};
use tracks::{
    assign_frame, intervals, Interval, MatchParams, Obs, ObsList, SupportParams, Track, Visibility,
};

/// One board keyframe as consolidation input.
#[derive(Debug, Clone, PartialEq)]
pub struct BoardFrame {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Index among all keyframes (board or not), for contiguity.
    pub keyframe_index: usize,
    /// Run start.
    pub t_start_s: f64,
    /// Run end.
    pub t_end_s: f64,
    /// Representative time.
    pub t_rep_s: f64,
    /// Canvas size of the reading's coordinates, when recorded.
    pub canvas: Option<CanvasDims>,
    /// Board title, when a producer extracted one (board splitting).
    pub board_title: Option<String>,
    /// Validated reading.
    pub board: ValidatedBoard,
    /// Aligned ink change between this board keyframe and the previous one, known
    /// only when the two are adjacent keyframes (`None` otherwise).
    pub ink_change: Option<f64>,
    /// Edge direction evidence, when computed.
    pub directions: Option<EdgeDirectionItem>,
    /// OCR spans inside the canvas, in the reading's coordinates (anchors).
    pub ocr_anchors: Vec<TextAnchor>,
    /// Texts from the app's title bar and board list (they name the board, and are
    /// never board content).
    pub title_hints: Vec<String>,
}

/// Keys used for tag geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnchorKey {
    Node(usize),
    Edge((usize, usize)),
}

/// A text with its box, used as a registration anchor.
#[derive(Debug, Clone, PartialEq)]
pub struct TextAnchor {
    /// Text.
    pub text: String,
    /// Box in the reading's canvas coordinates.
    pub bbox: BBox,
}

/// A second board-reader pass that may confirm a single sighting (optional).
pub trait SecondReader: Send + Sync {
    /// True when the second pass also reads `text` in `keyframe_id`.
    fn confirms(&self, keyframe_id: &str, text: &str) -> bool;
}

/// Hooks from other stages.
#[derive(Clone, Copy)]
pub struct Hooks<'a> {
    /// Transcript corroboration of owner moves.
    pub corroborator: &'a dyn Corroborator,
    /// Second reader pass for single sightings.
    pub second_reader: Option<&'a dyn SecondReader>,
}

/// Consolidation parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConsolidationParams {
    /// Participants and aliases for owner tags.
    pub participants: AliasTable,
    /// Fuzzy text threshold (difflib ratio).
    pub fuzzy_threshold: f64,
    /// Registration.
    pub registration: RegistrationParams,
    /// Same-place tolerance as a share of the reference canvas diagonal.
    pub position_tolerance_share: f64,
    /// IoU for a same-place node with new text to count as a label change.
    pub label_change_min_iou: f64,
    /// Largest offset, as a share of the reference diagonal, at which a same-text
    /// node reading joins its track after the view moved.
    #[serde(default = "default_off_position_share")]
    pub off_position_share: f64,
    /// Minimum share of an element's sightings that were also read as an edge's
    /// label before the element may be lifted out as that label.
    #[serde(default = "default_label_box_min_share")]
    pub label_box_min_share: f64,
    /// Minimum keyframes supporting an interval.
    pub min_support_keyframes: usize,
    /// Minimum share of board keyframes inside an interval that saw the element.
    pub min_support_density: f64,
    /// Consecutive visible-but-absent keyframes that remove an element.
    pub removal_absent_keyframes: usize,
    /// Length of the final stable board window, in seconds, measured back from the end
    /// of the last contiguous run of board keyframes. Default 120 s: long enough to
    /// span a few keyframes at the 2 s sampling grid, short enough to exclude earlier
    /// board states. Not a spec number; tune per corpus.
    pub final_window_s: f64,
    /// The final window holds at least this many board keyframes when that many exist
    /// (it is extended backward otherwise). The window only dates the final state
    /// (`t_end_s`, "final board state at"); membership is "observed and not removed".
    pub min_final_keyframes: usize,
    /// A keyframe that lacks an element evidences its removal only when it read
    /// another established element (seen in at least two keyframes) within this
    /// share of the reference diagonal of the element's center: nothing known read
    /// near the place means the view did not really cover it, or was registered
    /// wrongly there. A quarter of the diagonal reaches the neighbors of a box on a
    /// sparse board.
    #[serde(default = "default_coverage_radius_share")]
    pub coverage_radius_share: f64,
    /// Reading confidence (element and board) a single sighting needs, together with
    /// the pixel check, to be kept.
    pub single_sighting_min_conf: f64,
    /// Top band of the canvas, as a share of its height, treated as the title bar:
    /// corroborated title text inside it is the board title, not content.
    #[serde(default = "default_title_band_share")]
    pub title_band_share: f64,
    /// Minimum cards for a derived sticky group (grid or row).
    #[serde(default = "default_min_group_cards")]
    pub min_group_cards: usize,
    /// Aligned ink change needed for a content event.
    pub ink_event_threshold: f64,
    /// Share of the decisive weight a direction needs within a channel.
    pub vote_min_share: f64,
    /// Consecutive consistent keyframes to open or move an owner assignment.
    pub owner_confirm_keyframes: usize,
    /// Owner tag geometry thresholds.
    #[serde(default)]
    pub owner_anchor: AnchorParams,
    /// Alternations (A, B, A, B has 3) a strictly interleaving stretch between an
    /// edge's ends needs before its sightings are re-anchored to the edge.
    pub alternation_min_alternations: usize,
    /// Record a flagged backfill interval over untargeted sightings before a first
    /// assignment (never part of the confirmed assignment).
    pub backfill_untargeted_owners: bool,
}

impl Default for ConsolidationParams {
    fn default() -> Self {
        Self {
            participants: AliasTable::default(),
            fuzzy_threshold: crate::text::FUZZY_THRESHOLD,
            registration: RegistrationParams::default(),
            position_tolerance_share: 0.04,
            label_change_min_iou: 0.5,
            off_position_share: default_off_position_share(),
            label_box_min_share: default_label_box_min_share(),
            min_support_keyframes: 2,
            min_support_density: 0.10,
            removal_absent_keyframes: 2,
            final_window_s: 120.0,
            min_final_keyframes: 3,
            coverage_radius_share: default_coverage_radius_share(),
            single_sighting_min_conf: 0.8,
            title_band_share: default_title_band_share(),
            min_group_cards: default_min_group_cards(),
            ink_event_threshold: 0.05,
            vote_min_share: 0.6,
            owner_confirm_keyframes: 2,
            owner_anchor: AnchorParams::default(),
            alternation_min_alternations: 3,
            backfill_untargeted_owners: false,
        }
    }
}

fn default_coverage_radius_share() -> f64 {
    0.25
}

fn default_title_band_share() -> f64 {
    0.05
}

fn default_min_group_cards() -> usize {
    3
}

fn default_off_position_share() -> f64 {
    0.1
}

fn default_label_box_min_share() -> f64 {
    0.3
}

/// A lifetime interval in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Lifetime {
    /// Start of the first supporting keyframe.
    pub first_seen_s: f64,
    /// End of the last supporting keyframe.
    pub last_seen_s: f64,
    /// Supporting keyframes.
    pub keyframes: u32,
    /// Start of the absence that removed the element, when removed.
    pub removed_at_s: Option<f64>,
}

/// A text variant and how often it was read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextVariant {
    /// Text as read.
    pub text: String,
    /// Keyframes.
    pub count: u32,
}

/// Registration basis of an element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ElementRegistration {
    /// Matched by position and text in a registered cluster.
    Position,
    /// Matched by text only.
    TextOnly,
}

/// A consolidated node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeState {
    /// Stable id.
    #[serde(rename = "node_id")]
    pub id: String,
    /// Current label.
    pub text: String,
    /// Texts read for this node, most frequent first.
    pub variants: Vec<String>,
    /// Variants with their keyframe counts.
    pub variant_counts: Vec<TextVariant>,
    /// Supported lifetimes.
    pub lifetimes: Vec<Lifetime>,
    /// End of the last supported sighting.
    pub last_seen_s: Option<f64>,
    /// Alive in the final window.
    #[serde(rename = "in_final_state")]
    pub in_final: bool,
    /// Matching basis.
    pub registration: ElementRegistration,
    /// Box in the largest registered cluster's reference frame.
    pub bbox: Option<BBox>,
    /// List votes across keyframes.
    pub list_votes: tracks::ListVotes,
}

/// Sticky kinds (spec 6.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StickyKind {
    /// Ends with `?`.
    Question,
    /// Contains "milestone".
    Milestone,
    /// Starts with "Idea".
    Idea,
    /// Anything else.
    Note,
}

/// Classify a sticky by its text.
pub fn sticky_kind(text: &str) -> StickyKind {
    let t = text.trim();
    let lower = t.to_lowercase();
    if t.ends_with('?') {
        StickyKind::Question
    } else if lower.contains("milestone") {
        StickyKind::Milestone
    } else if lower.starts_with("idea") {
        StickyKind::Idea
    } else {
        StickyKind::Note
    }
}

/// A consolidated sticky.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StickyState {
    /// Stable id.
    pub id: String,
    /// Text.
    pub text: String,
    /// Kind.
    pub kind: StickyKind,
    /// Most frequent color.
    pub color: Option<StickyColor>,
    /// Box in the largest registered cluster's reference frame (since 1.1.0).
    #[serde(default)]
    pub bbox: Option<BBox>,
    /// Box at the last sighting, in that keyframe's canvas coordinates (since 1.1.0).
    #[serde(default)]
    pub last_seen: Option<SeenBox>,
    /// Supported lifetimes.
    pub lifetimes: Vec<Lifetime>,
    /// Alive in the final window.
    #[serde(rename = "in_final_state")]
    pub in_final: bool,
}

/// Edge direction in the tail-to-head form downstream consumers read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EdgeOrientation {
    /// Arrowhead at `dst`.
    Forward,
    /// Not established (`src`, `dst` are `a`, `b`).
    Uncertain,
    /// Arrowheads at both ends.
    Bidirectional,
}

/// A box in one keyframe's canvas coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeenBox {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Box in that keyframe's canvas pixels.
    pub bbox: BBox,
}

/// Stickies laid out together (a grid or a row of same-size cards).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StickyGroup {
    /// Stable id.
    pub id: String,
    /// Heading written just above the group, when there is one.
    pub title: Option<String>,
    /// Member sticky ids, row by row.
    pub sticky_ids: Vec<String>,
    /// Rows.
    pub rows: u32,
    /// Columns.
    pub cols: u32,
    /// Keyframe the layout was taken from.
    pub keyframe_id: String,
    /// Group extent in that keyframe's canvas pixels.
    pub bbox: BBox,
}

/// Why an element left the node and sticky lists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FoldReason {
    /// A piece of a longer element at the same place.
    #[default]
    Fragment,
    /// The heading of a derived sticky group.
    GroupHeading,
    /// An edge's label also read as a box on that edge.
    EdgeLabel,
}

/// An element taken out of the node and sticky lists, with where it went.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FoldedElement {
    /// Text of the element.
    pub text: String,
    /// What it was folded into: the longer element's text, `heading of <group>`,
    /// or `label of <edge>`.
    pub into: String,
    /// Why it left the lists.
    #[serde(default)]
    pub reason: FoldReason,
}

/// A consolidated edge between nodes `a` and `b`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeState {
    /// Stable id.
    pub id: String,
    /// Node id `a`.
    pub a: String,
    /// Node id `b`.
    pub b: String,
    /// Text of `a`.
    pub a_text: String,
    /// Text of `b`.
    pub b_text: String,
    /// Decision in the `(a, b)` orientation.
    pub decision: EdgeDirection,
    /// Tail node id (`a` unless the decision is `b_to_a`).
    pub src: String,
    /// Head node id.
    pub dst: String,
    /// Direction of `src -> dst`.
    pub direction: EdgeOrientation,
    /// Votes behind the decision.
    pub direction_votes: DirectionVotes,
    /// Which channel decided (pixel is authoritative; VLM only when pixels were
    /// inconclusive).
    pub direction_basis: DirectionBasis,
    /// Most frequent non-empty label.
    pub label: String,
    /// Majority style.
    pub style: EdgeStyle,
    /// Supported lifetimes.
    pub lifetimes: Vec<Lifetime>,
    /// Alive in the final window.
    pub in_final: bool,
}

/// An owner tag that did not resolve to a participant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RejectedOwnerTag {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Name as written.
    pub name_raw: String,
}

/// Where a keyframe's canvas size (and so the registration tolerance) came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CanvasSource {
    /// Recorded with the reading.
    Reading,
    /// From the crop the pixel check ran on.
    Crop,
    /// Estimated from the extent of the elements (a lower bound; flagged).
    Extent,
    /// Unknown: no registration for this keyframe.
    Unknown,
}

/// Registration record of one board keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KeyframeRegistration {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Registration details.
    pub registration: FrameRegistration,
    /// Source of the canvas size behind the residual tolerance.
    pub canvas_source: CanvasSource,
    /// OCR spans used as anchors.
    pub ocr_anchors: usize,
}

/// The final stable board window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FinalWindow {
    /// Start.
    pub start_s: f64,
    /// End.
    pub end_s: f64,
    /// Keyframes in the window.
    pub keyframe_ids: Vec<String>,
}

/// `glassrip.board_state` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardStateItem {
    /// Board id.
    pub board_id: String,
    /// Board title (corroborated title bar text), when seen (since 1.1.0).
    #[serde(default)]
    pub board_title: Option<String>,
    /// Sticky groups derived from the layout (since 1.1.0).
    #[serde(default)]
    pub groups: Vec<StickyGroup>,
    /// Fragments folded into longer elements, and group headings and edge-label
    /// boxes taken out of the node and sticky lists (since 1.1.0).
    #[serde(default)]
    pub folded: Vec<FoldedElement>,
    /// This is the board's final state (the last stable board window).
    #[serde(rename = "final")]
    pub is_final: bool,
    /// End of the final window.
    pub t_end_s: Option<f64>,
    /// Board keyframes in time order.
    pub board_keyframes: Vec<String>,
    /// Registration per keyframe.
    pub registration: Vec<KeyframeRegistration>,
    /// Final window.
    pub final_window: Option<FinalWindow>,
    /// Nodes.
    pub nodes: Vec<NodeState>,
    /// Edges.
    pub edges: Vec<EdgeState>,
    /// Stickies.
    pub stickies: Vec<StickyState>,
    /// Timed owner assignments.
    pub owner_assignments: Vec<OwnerAssignment>,
    /// Owner tags whose name is not a participant.
    pub rejected_owner_tags: Vec<RejectedOwnerTag>,
    /// Emitted events.
    pub events: Vec<BoardEvent>,
    /// Computed changes suppressed by the ink gate.
    pub suppressed_events: Vec<SuppressedEvent>,
}

/// A line segment in keyframe pixels.
type Segment = ((f64, f64), (f64, f64));

#[derive(Debug, Clone)]
struct EdgeObs {
    frame: usize,
    /// Reader's tail track.
    src: usize,
    label: String,
    style: EdgeStyle,
    evidence: Option<EdgeEvidence>,
    /// Segment in keyframe pixels (for owner geometry).
    segment: Segment,
}

fn canvas_of(f: &BoardFrame) -> (Option<CanvasDims>, CanvasSource) {
    if let Some(c) = f.canvas {
        return (Some(c), CanvasSource::Reading);
    }
    if let Some(d) = f
        .directions
        .as_ref()
        .filter(|d| d.canvas.width > 0.0 && d.canvas.height > 0.0)
    {
        return (Some(d.canvas), CanvasSource::Crop);
    }
    let extent = {
        let b = &f.board;
        let xs = b
            .nodes
            .iter()
            .map(|n| n.bbox.x2)
            .chain(b.stickies.iter().map(|s| s.bbox.x2))
            .chain(b.other_visible_text.iter().map(|t| t.bbox.x2));
        let ys = b
            .nodes
            .iter()
            .map(|n| n.bbox.y2)
            .chain(b.stickies.iter().map(|s| s.bbox.y2))
            .chain(b.other_visible_text.iter().map(|t| t.bbox.y2));
        let w = xs.fold(0.0, f64::max);
        let h = ys.fold(0.0, f64::max);
        (w > 0.0 && h > 0.0).then_some(CanvasDims {
            width: w,
            height: h,
        })
    };
    match extent {
        Some(c) => (Some(c), CanvasSource::Extent),
        None => (None, CanvasSource::Unknown),
    }
}

/// Boxes are usable for geometry when nodes do not all share one box.
fn geometry_ok(board: &ValidatedBoard) -> bool {
    let n = &board.nodes;
    for i in 0..n.len() {
        for j in i + 1..n.len() {
            if n[i].bbox.iou(&n[j].bbox) > 0.9 {
                return false;
            }
        }
    }
    true
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Consolidate the keyframes of one board into its board state.
pub fn consolidate(
    mut frames: Vec<BoardFrame>,
    board_id: &str,
    params: &ConsolidationParams,
    hooks: &Hooks<'_>,
) -> BoardStateItem {
    let corroborator = hooks.corroborator;
    frames.sort_by(|a, b| a.t_rep_s.total_cmp(&b.t_rep_s));
    let n = frames.len();
    let fz = params.fuzzy_threshold;

    // 0. Board title: title bar text and app panel text are never content.
    let titles = Titles::collect(&frames, params);
    let is_title = |fi: usize, text: &str, b: &BBox| titles.is_title(&frames[fi], text, b, params);

    // 1. Registration.
    let anchors: Vec<Anchors> = frames
        .iter()
        .enumerate()
        .map(|(fi, f)| {
            let b = &f.board;
            Anchors::from_items(
                b.nodes
                    .iter()
                    .map(|x| (x.text.as_str(), &x.bbox))
                    .chain(b.stickies.iter().map(|x| (x.text.as_str(), &x.bbox)))
                    .chain(
                        b.other_visible_text
                            .iter()
                            .map(|x| (x.text.as_str(), &x.bbox)),
                    )
                    .chain(f.ocr_anchors.iter().map(|x| (x.text.as_str(), &x.bbox)))
                    .filter(|(t, bb)| !is_title(fi, t, bb)),
            )
        })
        .collect();
    let canvas_info: Vec<(Option<CanvasDims>, CanvasSource)> =
        frames.iter().map(canvas_of).collect();
    let canvases: Vec<Option<CanvasDims>> = canvas_info.iter().map(|c| c.0).collect();
    let diagonals: Vec<f64> = canvases
        .iter()
        .map(|c| c.map(|c| c.diagonal()).unwrap_or(0.0))
        .collect();
    let regs = register(&anchors, &diagonals, &params.registration);
    let positioned = |i: usize| regs[i].mode != RegistrationMode::TextOnly;
    let ref_diag = median(diagonals.clone());
    let match_params = MatchParams {
        fuzzy: fz,
        position_tolerance_px: params.position_tolerance_share
            * if ref_diag.is_finite() { ref_diag } else { 0.0 },
        label_change_min_iou: params.label_change_min_iou,
        off_position_max_px: params.off_position_share
            * if ref_diag.is_finite() { ref_diag } else { 0.0 },
    };
    // The registered view changed between two keyframes (other cluster, or scale or
    // translation of the transform to the reference moved noticeably).
    let view_moved = |a: usize, b: usize| {
        let (ra, rb) = (&regs[a], &regs[b]);
        if ra.cluster != rb.cluster {
            return true;
        }
        let (ta, tb) = (ra.to_reference, rb.to_reference);
        let d = if ref_diag.is_finite() { ref_diag } else { 0.0 };
        (ta.scale / tb.scale - 1.0).abs() > 0.05
            || ((ta.tx - tb.tx).powi(2) + (ta.ty - tb.ty).powi(2)).sqrt() > 0.02 * d
    };

    // 2. Element tracks.
    let mut tracks: Vec<Track> = Vec::new();
    let mut node_track: HashMap<(usize, String), usize> = HashMap::new();
    // Readings whose text matched an edge's label in the same keyframe: (track,
    // keyframe, edge index in that keyframe, the reading's own box).
    let mut label_merges: Vec<(usize, usize, usize, BBox)> = Vec::new();
    for (fi, f) in frames.iter().enumerate() {
        let to_ref = |b: &BBox| positioned(fi).then(|| map_bbox(&regs[fi].to_reference, b));
        let cluster = regs[fi].cluster;
        let mut obs: Vec<Obs> = Vec::new();
        for nd in &f.board.nodes {
            // A node whose text resolves to a participant (fuzzy, unlike the
            // validator's exact check) is an owner tag read in the wrong list.
            if is_unreliable(&nd.text)
                || params.participants.resolve(&nd.text).is_some()
                || is_title(fi, &nd.text, &nd.bbox)
            {
                continue;
            }
            // Single sightings: the pixel check traced a connector to this node and the
            // reading is confident (plus the second reader pass when configured).
            let pixel_ok = f.directions.as_ref().is_some_and(|d| {
                d.edges.iter().any(|e| {
                    (e.src == nd.local_id && e.pixel.src_end.is_some())
                        || (e.dst == nd.local_id && e.pixel.dst_end.is_some())
                })
            });
            let conf_ok = nd.conf >= params.single_sighting_min_conf
                && f.board.confidence >= params.single_sighting_min_conf;
            let reader_ok = hooks
                .second_reader
                .is_none_or(|r| r.confirms(&f.keyframe_id, &nd.text));
            obs.push(Obs {
                frame: fi,
                lists: vec![ObsList::Node],
                text: nd.text.clone(),
                bbox: to_ref(&nd.bbox),
                raw_bbox: Some(nd.bbox),
                cluster,
                local_id: Some(nd.local_id.clone()),
                color: None,
                single_ok: pixel_ok && conf_ok && reader_ok,
            });
        }
        for s in &f.board.stickies {
            if is_unreliable(&s.text) || is_title(fi, &s.text, &s.bbox) {
                continue;
            }
            obs.push(Obs {
                frame: fi,
                lists: vec![ObsList::Sticky],
                text: s.text.clone(),
                bbox: to_ref(&s.bbox),
                raw_bbox: Some(s.bbox),
                cluster,
                local_id: None,
                color: Some(s.color),
                single_ok: false,
            });
        }
        for t in &f.board.other_visible_text {
            if is_unreliable(&t.text) || is_title(fi, &t.text, &t.bbox) {
                continue;
            }
            obs.push(Obs {
                frame: fi,
                lists: vec![ObsList::Other],
                text: t.text.clone(),
                bbox: to_ref(&t.bbox),
                raw_bbox: Some(t.bbox),
                cluster,
                local_id: None,
                color: None,
                single_ok: false,
            });
        }
        let mut merged: Vec<(usize, usize, BBox)> = Vec::new();
        for (ei, e) in f.board.edges.iter().enumerate() {
            let cleaned = clean_label(&e.label);
            let l = cleaned.as_str();
            if l.is_empty() || is_unreliable(l) {
                continue;
            }
            let nl = normalize(l);
            match obs.iter().position(|o| normalize(&o.text) == nl) {
                Some(oi) => {
                    let o = &mut obs[oi];
                    if !o.lists.contains(&ObsList::EdgeLabel) {
                        o.lists.push(ObsList::EdgeLabel);
                    }
                    if let Some(b) = o.raw_bbox {
                        merged.push((oi, ei, b));
                    }
                }
                None => obs.push(Obs {
                    frame: fi,
                    lists: vec![ObsList::EdgeLabel],
                    text: l.to_string(),
                    bbox: None,
                    raw_bbox: None,
                    cluster,
                    local_id: None,
                    color: None,
                    single_ok: false,
                }),
            }
        }
        let locals: Vec<Option<String>> = obs.iter().map(|o| o.local_id.clone()).collect();
        let assigned = assign_frame(&mut tracks, obs, &match_params, &view_moved);
        for (oi, ei, b) in merged {
            label_merges.push((assigned[oi], fi, ei, b));
        }
        for (ti, local) in assigned.into_iter().zip(locals) {
            if let Some(l) = local {
                node_track.insert((fi, l), ti);
            }
        }
    }

    // Fragments: a shorter reading of a longer element at the same place is folded
    // into it. The fragment's sightings do not count as support; references to it
    // (edge endpoints, owner targets) are redirected.
    let fold_to = fold_fragments(&tracks);
    let resolve = |mut t: usize| {
        let mut guard = 0;
        while let Some(&u) = fold_to.get(&t) {
            t = u;
            guard += 1;
            if guard > tracks.len() {
                break;
            }
        }
        t
    };
    for ti in node_track.values_mut() {
        *ti = resolve(*ti);
    }
    let folded: Vec<FoldedElement> = fold_to
        .iter()
        .map(|(&t, &u)| FoldedElement {
            text: tracks[t].text(),
            into: tracks[resolve(u)].text(),
            reason: FoldReason::Fragment,
        })
        .collect();

    // Views and visibility.
    let views: Vec<Option<BBox>> = (0..n)
        .map(|i| {
            let c = canvases[i]?;
            positioned(i).then(|| {
                map_bbox(
                    &regs[i].to_reference,
                    &BBox::new(0.0, 0.0, c.width, c.height),
                )
            })
        })
        .collect();
    let track_visible = |t: &Track, f: usize| -> Visibility {
        let Some(view) = views[f] else {
            return Visibility::Unknown;
        };
        let Some(b) = t.bbox_in(regs[f].cluster) else {
            return Visibility::Unknown;
        };
        let c = ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0);
        let (mx, my) = (view.width() * 0.03, view.height() * 0.03);
        if c.0 > view.x1 + mx && c.0 < view.x2 - mx && c.1 > view.y1 + my && c.1 < view.y2 - my {
            Visibility::Visible
        } else {
            Visibility::Unknown
        }
    };
    // Coverage evidence per positioned keyframe, in its cluster's reference frame:
    // the centers of the established elements it read (tracks seen in at least two
    // keyframes, matched there by text and position, so the keyframe's registration
    // agrees with theirs around that place), and the OCR anchors with their
    // normalized text. Anchors never corroborate coverage on their own: a wrongly
    // registered view carries its anchors along with it.
    let mut content: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n];
    for t in &tracks {
        if t.frames().len() < 2 {
            continue;
        }
        for o in &t.obs {
            if let Some(b) = o.bbox {
                content[o.frame].push(((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0));
            }
        }
    }
    let ocr_ref: Vec<Vec<(String, BBox)>> = (0..n)
        .map(|f| {
            if !positioned(f) {
                return Vec::new();
            }
            frames[f]
                .ocr_anchors
                .iter()
                .map(|a| (normalize(&a.text), map_bbox(&regs[f].to_reference, &a.bbox)))
                .filter(|(t, _)| t.chars().filter(|c| c.is_alphanumeric()).count() >= 3)
                .collect()
        })
        .collect();
    let coverage_radius = params.coverage_radius_share * ref_diag;
    // `Visible` only when keyframe `f` really covers the track's place: inside the
    // view (`track_visible`), no OCR text of the track there, and other content near.
    let track_covered = |t: &Track, f: usize| -> Visibility {
        if track_visible(t, f) != Visibility::Visible {
            return Visibility::Unknown;
        }
        let Some(b) = t.bbox_in(regs[f].cluster) else {
            return Visibility::Unknown;
        };
        let text = normalize(&t.text());
        let (bw, bh) = (b.width() * 0.5, b.height() * 0.5);
        let near_box = BBox::new(b.x1 - bw, b.y1 - bh, b.x2 + bw, b.y2 + bh);
        let ocr_saw_it = ocr_ref[f].iter().any(|(a, ab)| {
            let c = ((ab.x1 + ab.x2) / 2.0, (ab.y1 + ab.y2) / 2.0);
            c.0 >= near_box.x1
                && c.0 <= near_box.x2
                && c.1 >= near_box.y1
                && c.1 <= near_box.y2
                && (text.contains(a.as_str()) || a.contains(text.as_str()))
        });
        if ocr_saw_it {
            return Visibility::Unknown;
        }
        let c = ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0);
        let corroborated = coverage_radius.is_finite()
            && content[f]
                .iter()
                .any(|p| ((p.0 - c.0).powi(2) + (p.1 - c.1).powi(2)).sqrt() <= coverage_radius);
        if corroborated {
            Visibility::Visible
        } else {
            Visibility::Unknown
        }
    };
    let support = SupportParams {
        min_keyframes: params.min_support_keyframes,
        min_density: params.min_support_density,
        removal_absent: params.removal_absent_keyframes,
    };
    let track_intervals: Vec<Vec<Interval>> = tracks
        .iter()
        .map(|t| {
            let seen = t.frames();
            intervals(
                &seen,
                |f| {
                    if seen.binary_search(&f).is_ok() {
                        Visibility::Unknown
                    } else {
                        track_covered(t, f)
                    }
                },
                n,
                &support,
                |f| t.obs.iter().any(|o| o.frame == f && o.single_ok),
            )
        })
        .collect();
    // Folded fragments have no lifetime of their own.
    let track_intervals: Vec<Vec<Interval>> = track_intervals
        .into_iter()
        .enumerate()
        .map(|(ti, iv)| {
            if fold_to.contains_key(&ti) {
                Vec::new()
            } else {
                iv
            }
        })
        .collect();

    // Final window: the last contiguous run of board keyframes, trimmed to
    // `final_window_s` before its end. It dates the final state; which elements
    // belong to it is decided by removal evidence, not by this window.
    let final_frames: Vec<usize> = if n == 0 {
        Vec::new()
    } else {
        let mut start = n - 1;
        while start > 0 && frames[start - 1].keyframe_index + 1 == frames[start].keyframe_index {
            start -= 1;
        }
        let end_s = frames[n - 1].t_end_s;
        let mut v: Vec<usize> = (start..n)
            .filter(|&i| frames[i].t_rep_s >= end_s - params.final_window_s)
            .collect();
        let want = params.min_final_keyframes.max(1).min(n);
        if v.len() < want {
            v = (n - want..n).collect();
        }
        v
    };
    let window = final_frames
        .first()
        .zip(final_frames.last())
        .map(|(&a, &b)| FinalWindow {
            start_s: frames[a].t_start_s,
            end_s: frames[b].t_end_s,
            keyframe_ids: final_frames
                .iter()
                .map(|&i| frames[i].keyframe_id.clone())
                .collect(),
        });
    let lifetime = |iv: &Interval| Lifetime {
        first_seen_s: frames[iv.first].t_start_s,
        last_seen_s: frames[iv.last].t_end_s,
        keyframes: iv.count as u32,
        removed_at_s: iv.removed_at.map(|f| frames[f].t_start_s),
    };
    // Final: observed and not later removed (see the module docs). A single
    // sighting that was never placed on the board cannot be checked against later
    // views, so it is not final.
    let alive_in_final = |ivs: &[Interval], placed: bool| -> bool {
        ivs.last().is_some_and(|iv| {
            iv.removed_at.is_none() && (placed || iv.count >= params.min_support_keyframes)
        })
    };
    let largest_cluster = {
        let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
        for (i, r) in regs.iter().enumerate() {
            if positioned(i) {
                *counts.entry(r.cluster).or_default() += 1;
            }
        }
        counts
            .into_iter()
            .max_by_key(|e| (e.1, std::cmp::Reverse(e.0)))
            .map(|e| e.0)
    };

    // Node and sticky outputs (ids in order of first appearance).
    let mut order: Vec<usize> = (0..tracks.len()).collect();
    order.sort_by_key(|&t| tracks[t].frames().first().copied().unwrap_or(usize::MAX));
    let mut node_id: HashMap<usize, String> = HashMap::new();
    let mut sticky_id: HashMap<usize, String> = HashMap::new();
    let mut nodes: Vec<NodeState> = Vec::new();
    let mut stickies: Vec<StickyState> = Vec::new();
    // A node read in a single keyframe with the text of an established node (seen
    // in at least `min_support_keyframes` keyframes, and not read in that keyframe)
    // is that node read at an imprecise place: it never becomes a second final
    // element.
    let mut established: HashMap<String, Vec<(usize, Vec<usize>)>> = HashMap::new();
    for (ti, t) in tracks.iter().enumerate() {
        if !track_intervals[ti].is_empty() && t.votes().kind() == ObsList::Node {
            established
                .entry(normalize(&t.text()))
                .or_default()
                .push((ti, t.frames()));
        }
    }
    let echo_of_established = |ti: usize, t: &Track| -> bool {
        let seen = t.frames();
        seen.len() == 1
            && established.get(&normalize(&t.text())).is_some_and(|v| {
                v.iter().any(|(u, frames)| {
                    *u != ti
                        && frames.len() >= params.min_support_keyframes
                        && !frames.contains(&seen[0])
                })
            })
    };
    for &ti in &order {
        let t = &tracks[ti];
        let ivs = &track_intervals[ti];
        if ivs.is_empty() {
            continue;
        }
        let votes = t.votes();
        let lifetimes: Vec<Lifetime> = ivs.iter().map(lifetime).collect();
        match votes.kind() {
            ObsList::Node => {
                let id = format!("node-{}", nodes.len() + 1);
                node_id.insert(ti, id.clone());
                let mut variant_counts: Vec<TextVariant> = t
                    .variants()
                    .into_values()
                    .map(|(text, count)| TextVariant { text, count })
                    .collect();
                variant_counts.sort_by(|a, b| b.count.cmp(&a.count).then(a.text.cmp(&b.text)));
                nodes.push(NodeState {
                    id,
                    text: t.current_text(fz),
                    variants: variant_counts.iter().map(|v| v.text.clone()).collect(),
                    variant_counts,
                    last_seen_s: lifetimes.iter().map(|l| l.last_seen_s).reduce(f64::max),
                    lifetimes,
                    in_final: alive_in_final(ivs, t.obs.iter().any(|o| o.bbox.is_some()))
                        && !echo_of_established(ti, t),
                    registration: if t.obs.iter().any(|o| o.bbox.is_some()) {
                        ElementRegistration::Position
                    } else {
                        ElementRegistration::TextOnly
                    },
                    bbox: largest_cluster.and_then(|c| t.bbox_in(c)),
                    list_votes: votes,
                });
            }
            ObsList::Sticky => {
                let text = t.current_text(fz);
                let id = format!("sticky-{}", stickies.len() + 1);
                sticky_id.insert(ti, id.clone());
                let last_seen = t.obs.iter().rev().find_map(|o| {
                    o.raw_bbox.map(|b| SeenBox {
                        keyframe_id: frames[o.frame].keyframe_id.clone(),
                        bbox: b,
                    })
                });
                stickies.push(StickyState {
                    id,
                    kind: sticky_kind(&text),
                    text,
                    color: t.color(),
                    bbox: largest_cluster.and_then(|c| t.bbox_in(c)),
                    last_seen,
                    lifetimes,
                    in_final: alive_in_final(ivs, t.obs.iter().any(|o| o.bbox.is_some())),
                });
            }
            ObsList::Other | ObsList::EdgeLabel => {}
        }
    }

    // 4. Edges.
    let mut edge_obs: BTreeMap<(usize, usize), Vec<EdgeObs>> = BTreeMap::new();
    for (fi, f) in frames.iter().enumerate() {
        let boxes: HashMap<&str, &BBox> = f
            .board
            .nodes
            .iter()
            .map(|x| (x.local_id.as_str(), &x.bbox))
            .collect();
        for e in &f.board.edges {
            let (Some(&s), Some(&d)) = (
                node_track.get(&(fi, e.src.clone())),
                node_track.get(&(fi, e.dst.clone())),
            ) else {
                continue;
            };
            if s == d || !node_id.contains_key(&s) || !node_id.contains_key(&d) {
                continue;
            }
            let key = (s.min(d), s.max(d));
            let list = edge_obs.entry(key).or_default();
            if list.iter().any(|o| o.frame == fi) {
                continue;
            }
            let evidence = f
                .directions
                .as_ref()
                .and_then(|dir| dir.edges.iter().find(|x| x.src == e.src && x.dst == e.dst))
                .cloned();
            let segment = match (
                &evidence,
                boxes.get(e.src.as_str()),
                boxes.get(e.dst.as_str()),
            ) {
                (Some(ev), _, _) if ev.pixel.src_end.is_some() && ev.pixel.dst_end.is_some() => {
                    let (a, b) = (ev.pixel.src_end, ev.pixel.dst_end);
                    match (a, b) {
                        (Some(a), Some(b)) => ((a.x, a.y), (b.x, b.y)),
                        _ => ((0.0, 0.0), (0.0, 0.0)),
                    }
                }
                (_, Some(sb), Some(db)) => {
                    let sc = ((sb.x1 + sb.x2) / 2.0, (sb.y1 + sb.y2) / 2.0);
                    let dc = ((db.x1 + db.x2) / 2.0, (db.y1 + db.y2) / 2.0);
                    (exit_point(sb, dc), exit_point(db, sc))
                }
                _ => ((0.0, 0.0), (0.0, 0.0)),
            };
            list.push(EdgeObs {
                frame: fi,
                src: s,
                label: clean_label(&e.label),
                style: e.style,
                evidence,
                segment,
            });
        }
    }
    let dir_items: Vec<&EdgeDirectionItem> = frames
        .iter()
        .filter_map(|f| f.directions.as_ref())
        .collect();
    let med_sharp = median(dir_items.iter().map(|d| d.sharpness).collect());
    let med_zoom = median(dir_items.iter().map(|d| d.zoom).collect());
    let weight_of = |f: usize| {
        frames[f]
            .directions
            .as_ref()
            .map(|d| frame_weight(d.sharpness, med_sharp, d.zoom, med_zoom))
            .unwrap_or(1.0)
    };
    let final_node = |ti: usize| {
        nodes
            .iter()
            .any(|x| Some(&x.id) == node_id.get(&ti) && x.in_final)
    };
    let text_of = |ti: usize| {
        nodes
            .iter()
            .find(|x| Some(&x.id) == node_id.get(&ti))
            .map(|x| x.text.clone())
            .unwrap_or_default()
    };

    let mut edges: Vec<EdgeState> = Vec::new();
    let mut edge_id_of: HashMap<(usize, usize), String> = HashMap::new();
    let mut edge_intervals: HashMap<(usize, usize), Vec<Interval>> = HashMap::new();
    for (&(a, b), list) in &edge_obs {
        let seen: Vec<usize> = list.iter().map(|o| o.frame).collect();
        let (ta, tb) = (&tracks[a], &tracks[b]);
        let ivs = intervals(
            &seen,
            // An edge is absent from a keyframe that read both of its ends (or
            // covers both places) without reading the edge, once the board's ink
            // changed since the edge was last read: readers often leave a
            // connector out of one reading, while erasing one changes the ink.
            |f| {
                let present = |t: &Track| {
                    t.obs.iter().any(|o| o.frame == f && o.bbox.is_some())
                        || track_covered(t, f) == Visibility::Visible
                };
                let from = seen.iter().rev().find(|&&s| s < f).map_or(0, |&s| s + 1);
                let inked = (from..=f).any(|k| {
                    frames[k]
                        .ink_change
                        .is_some_and(|x| x >= params.ink_event_threshold)
                });
                if seen.contains(&f) {
                    Visibility::Unknown
                } else if inked && present(ta) && present(tb) {
                    Visibility::Visible
                } else {
                    Visibility::Unknown
                }
            },
            n,
            &support,
            |_| false,
        );
        let Some(last_iv) = ivs.last().copied() else {
            continue;
        };
        let in_iv: Vec<&EdgeObs> = list
            .iter()
            .filter(|o| o.frame >= last_iv.first && o.frame <= last_iv.last)
            .collect();
        // The direction vote uses every sighting since the edge's last detected
        // reversal (all of them when none was detected): more keyframes outvote a bad
        // one, and the final direction agrees with the last detected reversal. A flip
        // seen in a single keyframe is not a reversal, so it neither emits an
        // EdgeReversed event nor restarts the vote window.
        let since = reversals(list, a, params.vote_min_share).last().copied();
        let mut votes = DirectionVotes::default();
        for o in list.iter().filter(|o| since.is_none_or(|f| o.frame >= f)) {
            let w = weight_of(o.frame);
            let orient = |v: EndVerdict| if o.src == a { v } else { v.flipped() };
            votes.reader.add(orient(EndVerdict::Forward), 1.0);
            if let Some(ev) = &o.evidence {
                let (p, v) = ev.verdicts();
                votes.pixel.add(orient(p), w);
                if let Some(v) = v {
                    votes.vlm.add(orient(v), w);
                }
            }
        }
        let (direction, direction_basis) = votes.decide(params.vote_min_share);
        let (ida, idb) = match (node_id.get(&a), node_id.get(&b)) {
            (Some(x), Some(y)) => (x.clone(), y.clone()),
            _ => continue,
        };
        let (src, dst) = match direction {
            EdgeDirection::BToA => (idb.clone(), ida.clone()),
            _ => (ida.clone(), idb.clone()),
        };
        let orientation = match direction {
            EdgeDirection::AToB | EdgeDirection::BToA => EdgeOrientation::Forward,
            EdgeDirection::Bidirectional => EdgeOrientation::Bidirectional,
            EdgeDirection::Uncertain => EdgeOrientation::Uncertain,
        };
        let mut label_counts: BTreeMap<String, (String, u32)> = BTreeMap::new();
        for o in in_iv.iter().filter(|o| !o.label.is_empty()) {
            let e = label_counts
                .entry(normalize(&o.label))
                .or_insert((o.label.clone(), 0));
            e.1 += 1;
            e.0.clone_from(&o.label);
        }
        let label = label_counts
            .into_values()
            .max_by_key(|e| e.1)
            .map(|e| e.0)
            .unwrap_or_default();
        let dashed = in_iv
            .iter()
            .filter(|o| o.style == EdgeStyle::Dashed)
            .count();
        let id = format!("edge-{}", edges.len() + 1);
        edge_id_of.insert((a, b), id.clone());
        edge_intervals.insert((a, b), ivs.clone());
        edges.push(EdgeState {
            id,
            a: ida,
            b: idb,
            a_text: text_of(a),
            b_text: text_of(b),
            decision: direction,
            src,
            dst,
            direction: orientation,
            direction_votes: votes,
            direction_basis,
            label,
            style: if dashed * 2 > in_iv.len() {
                EdgeStyle::Dashed
            } else {
                EdgeStyle::Solid
            },
            lifetimes: ivs.iter().map(lifetime).collect(),
            in_final: alive_in_final(&ivs, true) && final_node(a) && final_node(b),
        });
    }

    // 5. Owners.
    let node_target = |ti: usize| -> Option<OwnerTarget> {
        node_id.get(&ti).map(|id| OwnerTarget::Node {
            node_id: id.clone(),
            text: text_of(ti),
        })
    };
    let edge_target = |key: (usize, usize)| -> Option<OwnerTarget> {
        let id = edge_id_of.get(&key)?;
        let e = edges.iter().find(|e| &e.id == id)?;
        Some(OwnerTarget::Edge {
            edge_id: id.clone(),
            src: e.src.clone(),
            dst: e.dst.clone(),
            a_text: text_of(key.0),
            b_text: text_of(key.1),
        })
    };
    let mut rejected_owner_tags: Vec<RejectedOwnerTag> = Vec::new();
    let mut by_person: BTreeMap<String, (String, Vec<OwnerSighting>)> = BTreeMap::new();
    for (fi, f) in frames.iter().enumerate() {
        let geo = geometry_ok(&f.board);
        let mut tags: Vec<(String, String, BBox)> = f
            .board
            .owner_tags
            .iter()
            .map(|o| (o.name_raw.clone(), o.near.clone(), o.bbox))
            .collect();
        // A participant name read as a node is an owner tag in the wrong list.
        for r in &f.board.chrome_rejected {
            if r.list == ElementList::Nodes && r.reason == RejectReason::ParticipantName {
                if let Some(b) = r.bbox {
                    tags.push((r.text.clone(), String::new(), b));
                }
            }
        }
        for nd in &f.board.nodes {
            if params.participants.resolve(&nd.text).is_some() {
                tags.push((nd.text.clone(), String::new(), nd.bbox));
            }
        }
        // Boxes of the tracked nodes in this keyframe, and of the supported edges seen
        // here (for tag geometry).
        let node_boxes: Vec<(AnchorKey, BBox)> = f
            .board
            .nodes
            .iter()
            .filter(|x| params.participants.resolve(&x.text).is_none())
            .filter_map(|x| {
                node_track
                    .get(&(fi, x.local_id.clone()))
                    .filter(|ti| node_id.contains_key(ti))
                    .map(|&ti| (AnchorKey::Node(ti), x.bbox))
            })
            .collect();
        let box_of = |ti: usize| {
            node_boxes
                .iter()
                .find(|(k, _)| *k == AnchorKey::Node(ti))
                .map(|(_, b)| *b)
        };
        let frame_edges: Vec<EdgeGeom<AnchorKey>> = edge_obs
            .iter()
            .filter(|(k, _)| edge_id_of.contains_key(k))
            .filter_map(|(k, l)| {
                let o = l.iter().find(|o| o.frame == fi)?;
                Some(EdgeGeom {
                    key: AnchorKey::Edge(*k),
                    a: (AnchorKey::Node(k.0), box_of(k.0)?),
                    b: (AnchorKey::Node(k.1), box_of(k.1)?),
                    segment: o.segment,
                })
            })
            .collect();
        for (tag_index, (name, near, tb)) in tags.into_iter().enumerate() {
            let Some(person) = params.participants.resolve(&name) else {
                rejected_owner_tags.push(RejectedOwnerTag {
                    keyframe_id: f.keyframe_id.clone(),
                    name_raw: name,
                });
                continue;
            };
            let mut targets: Vec<(OwnerTarget, AnchorKind)> = Vec::new();
            if geo {
                let node_of = |k: &AnchorKey| match k {
                    AnchorKey::Node(t) => node_target(*t),
                    AnchorKey::Edge(_) => None,
                };
                match anchor_tag(&tb, &node_boxes, &frame_edges, &params.owner_anchor) {
                    Some(Anchored::Node(k)) => {
                        targets.extend(node_of(&k).map(|t| (t, AnchorKind::GeometryNode)));
                    }
                    Some(Anchored::Bridge(x, y)) => {
                        targets.extend(node_of(&x).map(|t| (t, AnchorKind::GeometryBridge)));
                        targets.extend(node_of(&y).map(|t| (t, AnchorKind::GeometryBridge)));
                    }
                    Some(Anchored::Edge(AnchorKey::Edge(k))) => {
                        targets.extend(edge_target(k).map(|t| (t, AnchorKind::GeometryEdge)));
                    }
                    Some(Anchored::Edge(AnchorKey::Node(_))) | None => {}
                }
            }
            if targets.is_empty() && !near.is_empty() {
                targets.extend(
                    node_track
                        .get(&(fi, near.clone()))
                        .and_then(|&ti| node_target(ti))
                        .map(|t| (t, AnchorKind::Near)),
                );
            }
            let entry = by_person
                .entry(person.person_id.clone())
                .or_insert_with(|| (person.display_name.clone(), Vec::new()));
            let targets: Vec<(Option<OwnerTarget>, AnchorKind)> = if targets.is_empty() {
                vec![(None, AnchorKind::Untargeted)]
            } else {
                targets.into_iter().map(|(t, a)| (Some(t), a)).collect()
            };
            for (t, anchor) in targets {
                // Several tags of one person in a keyframe are kept when their targets
                // differ (multi-target owners).
                if entry
                    .1
                    .iter()
                    .any(|s| s.keyframe_id == f.keyframe_id && s.target == t)
                {
                    continue;
                }
                entry.1.push(OwnerSighting {
                    keyframe_id: f.keyframe_id.clone(),
                    t_start_s: f.t_start_s,
                    t_end_s: f.t_end_s,
                    name_raw: name.clone(),
                    target: t,
                    anchor,
                    tag: tag_index as u32,
                });
            }
        }
    }
    let alternation_edges: Vec<(OwnerTarget, OwnerTarget, OwnerTarget)> = edge_id_of
        .keys()
        .filter_map(|&(a, b)| Some((edge_target((a, b))?, node_target(a)?, node_target(b)?)))
        .collect();
    let timeline_end_s = frames.last().map(|f| f.t_end_s).unwrap_or(0.0);
    let owner_params = OwnerParams {
        confirm_keyframes: params.owner_confirm_keyframes,
        backfill_untargeted: params.backfill_untargeted_owners,
    };
    // A target is visible in a keyframe where its node (or both edge ends) was read.
    let track_of_id: HashMap<&str, usize> =
        node_id.iter().map(|(t, id)| (id.as_str(), *t)).collect();
    let edge_of_id: HashMap<&str, (usize, usize)> =
        edge_id_of.iter().map(|(k, id)| (id.as_str(), *k)).collect();
    let frame_of_kf: HashMap<&str, usize> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| (f.keyframe_id.as_str(), i))
        .collect();
    let seen = |ti: usize, kf: &str| {
        frame_of_kf
            .get(kf)
            .is_some_and(|fi| tracks[ti].obs.iter().any(|o| o.frame == *fi))
    };
    let target_visible = |kf: &str, t: &OwnerTarget| match t {
        OwnerTarget::Node { node_id, .. } => track_of_id
            .get(node_id.as_str())
            .is_some_and(|&ti| seen(ti, kf)),
        OwnerTarget::Edge { edge_id, .. } => edge_of_id
            .get(edge_id.as_str())
            .is_some_and(|&(a, b)| seen(a, kf) && seen(b, kf)),
    };
    let mut owner_assignments: Vec<OwnerAssignment> = Vec::new();
    for (pid, (name, mut sightings)) in by_person {
        collapse_edge_pairs(&mut sightings, &alternation_edges);
        apply_alternation(
            &mut sightings,
            &alternation_edges,
            params.alternation_min_alternations,
        );
        owner_assignments.extend(assign(
            &pid,
            &name,
            &sightings,
            timeline_end_s,
            &owner_params,
            corroborator,
            &target_visible,
        ));
    }

    // Sticky groups (grids, rows of cards). Their headings leave the node and sticky
    // lists, recorded in `folded`; edge endpoints and owner targets are never
    // headings, so no connectivity or ownership is lost. Label boxes leave the same
    // way under the same exclusions.
    let final_sticky: HashMap<usize, String> = sticky_id
        .iter()
        .filter(|(_, id)| stickies.iter().any(|s| &s.id == *id && s.in_final))
        .map(|(t, id)| (*t, id.clone()))
        .collect();
    let mut bound: HashSet<String> = edges
        .iter()
        .flat_map(|e| [e.a.clone(), e.b.clone()])
        .collect();
    for a in &owner_assignments {
        if let OwnerTarget::Node { node_id, .. } = &a.target {
            bound.insert(node_id.clone());
        }
    }
    let excluded = |ti: usize| node_id.get(&ti).is_some_and(|id| bound.contains(id));
    let (groups, heading_tracks) = derive_groups(
        &tracks,
        &frames,
        &final_sticky,
        params.min_group_cards,
        &excluded,
    );
    let mut folded = folded;
    // Tracks lifted out of the node and sticky lists (headings and label boxes).
    let mut lifted: HashSet<usize> = HashSet::new();
    for (t, gid) in &heading_tracks {
        if node_id.contains_key(t) || sticky_id.contains_key(t) {
            lifted.insert(*t);
            folded.push(FoldedElement {
                text: tracks[*t].text(),
                into: format!("heading of {gid}"),
                reason: FoldReason::GroupHeading,
            });
        }
    }
    // A box or sticky whose text is also an edge's label is that label read twice
    // only on positive evidence: a meaningful share of its sightings coincided with
    // the label, and in one of those keyframes the edge whose label merged into it
    // is a consolidated edge and the reading sits on that edge (within about one
    // text height of its label box, or beside the link between its two ends).
    // Endpoints and owner targets are never lifted. Anything else stays an element.
    let mut label_lifts: BTreeMap<usize, String> = BTreeMap::new();
    for &(t, fi, ei, b) in &label_merges {
        let ti = resolve(t);
        let Some(id) = node_id.get(&ti).or_else(|| sticky_id.get(&ti)) else {
            continue;
        };
        if bound.contains(id) || lifted.contains(&ti) || label_lifts.contains_key(&ti) {
            continue;
        }
        let tr = &tracks[ti];
        let hits = tr
            .obs
            .iter()
            .filter(|o| o.lists.contains(&ObsList::EdgeLabel))
            .count();
        if (hits as f64) < params.label_box_min_share * tr.obs.len() as f64 {
            continue;
        }
        let f = &frames[fi];
        let Some(e) = f.board.edges.get(ei) else {
            continue;
        };
        let (Some(s), Some(d)) = (
            node_track.get(&(fi, e.src.clone())),
            node_track.get(&(fi, e.dst.clone())),
        ) else {
            continue;
        };
        let (Some(sa), Some(sb)) = (node_id.get(s), node_id.get(d)) else {
            continue;
        };
        let Some(edge) = edges
            .iter()
            .find(|x| (x.a == *sa && x.b == *sb) || (x.a == *sb && x.b == *sa))
        else {
            continue;
        };
        let text_h = b.height();
        let near_label = e
            .label_bbox
            .is_some_and(|lb| box_distance(&b, &lb) <= text_h);
        let end_box = |l: &str| {
            f.board
                .nodes
                .iter()
                .find(|n| n.local_id == l)
                .map(|n| n.bbox)
        };
        let on_path = match (end_box(&e.src), end_box(&e.dst)) {
            (Some(p), Some(q)) => beside_link(&b, &p, &q),
            _ => false,
        };
        if near_label || on_path {
            label_lifts.insert(ti, edge.id.clone());
        }
    }
    for (ti, eid) in label_lifts {
        if lifted.insert(ti) {
            folded.push(FoldedElement {
                text: tracks[ti].text(),
                into: format!("label of {eid}"),
                reason: FoldReason::EdgeLabel,
            });
        }
    }

    // 6. Events.
    let mut gate = EventGate::new(params.ink_event_threshold);
    let event = |kind: EventKind, f: usize, subject: &str, detail: String| BoardEvent {
        event_id: String::new(),
        kind,
        t_s: frames[f].t_start_s,
        keyframe_id: frames[f].keyframe_id.clone(),
        subject: subject.to_string(),
        detail,
        ink_change: frames[f].ink_change,
        baseline: f == 0,
    };
    for &ti in &order {
        if lifted.contains(&ti) {
            continue;
        }
        let ivs = &track_intervals[ti];
        if let Some(id) = node_id.get(&ti) {
            let text = text_of(ti);
            for iv in ivs {
                gate.offer(event(EventKind::NodeAdded, iv.first, id, text.clone()));
                if let Some(r) = iv.removed_at {
                    gate.offer(event(EventKind::NodeRemoved, r, id, text.clone()));
                }
            }
            // Label changes: a new variant that holds for two consecutive sightings.
            let obs = &tracks[ti].obs;
            let hist = tracks[ti].label_changes(fz);
            for w in hist.windows(2) {
                let f = obs[w[1].0].frame;
                gate.offer(event(
                    EventKind::LabelChanged,
                    f,
                    id,
                    format!("{} -> {}", w[0].1, obs[w[1].0].text),
                ));
            }
        } else if let Some(id) = sticky_id.get(&ti) {
            let text = tracks[ti].current_text(fz);
            for iv in ivs {
                gate.offer(event(EventKind::StickyAdded, iv.first, id, text.clone()));
            }
        }
    }
    for e in &edges {
        let Some(key) = edge_id_of
            .iter()
            .find(|(_, v)| **v == e.id)
            .map(|(k, _)| *k)
        else {
            continue;
        };
        let detail = format!("{} - {}", e.a_text, e.b_text);
        for iv in edge_intervals.get(&key).map(Vec::as_slice).unwrap_or(&[]) {
            gate.offer(event(EventKind::EdgeAdded, iv.first, &e.id, detail.clone()));
            if let Some(r) = iv.removed_at {
                gate.offer(event(EventKind::EdgeRemoved, r, &e.id, detail.clone()));
            }
        }
        // Reversal: per-keyframe decisions that flip for two consecutive evidence frames.
        let list = edge_obs.get(&key).map(Vec::as_slice).unwrap_or(&[]);
        for f in reversals(list, key.0, params.vote_min_share) {
            gate.offer(event(EventKind::EdgeReversed, f, &e.id, detail.clone()));
        }
    }
    let frame_of = |id: &str| frames.iter().position(|f| f.keyframe_id == id);
    for a in &owner_assignments {
        let Some(f) = frame_of(&a.opened_at_keyframe) else {
            continue;
        };
        // A move replaces an open target of the same person; anything else (a first
        // assignment, an added target, a reopening) is an assignment.
        let kind = if a.moved_from.is_some() {
            EventKind::OwnerMoved
        } else {
            EventKind::OwnerAssigned
        };
        gate.offer(event(
            kind,
            f,
            &a.person_id,
            format!("{} -> {}", a.display_name, a.target.texts().join(" - ")),
        ));
    }
    let (events, suppressed_events) = gate.finish();
    // Lifted tracks leave the node and sticky lists.
    let heading_ids: HashSet<String> = lifted
        .iter()
        .filter_map(|t| node_id.get(t).or_else(|| sticky_id.get(t)).cloned())
        .collect();
    nodes.retain(|n| !heading_ids.contains(&n.id));
    stickies.retain(|s| !heading_ids.contains(&s.id));

    BoardStateItem {
        board_id: board_id.to_string(),
        board_title: titles.board_title(),
        groups,
        folded,
        is_final: true,
        t_end_s: window.as_ref().map(|w| w.end_s),
        board_keyframes: frames.iter().map(|f| f.keyframe_id.clone()).collect(),
        registration: frames
            .iter()
            .zip(regs)
            .zip(&canvas_info)
            .map(|((f, r), c)| KeyframeRegistration {
                keyframe_id: f.keyframe_id.clone(),
                registration: r,
                canvas_source: c.1,
                ocr_anchors: f.ocr_anchors.len(),
            })
            .collect(),
        final_window: window,
        nodes,
        edges,
        stickies,
        owner_assignments,
        rejected_owner_tags,
        events,
        suppressed_events,
    }
}

/// Keyframes where an edge's direction flips: per-keyframe decisions (pixel first,
/// VLM when pixels are inconclusive), a flip being a new direction that holds for two
/// consecutive decided keyframes. `a` is the node track the `(a, b)` orientation
/// starts from.
/// Whether box `b` sits beside the straight link between boxes `p` and `q`: clear
/// of both ends, projecting inside the link, and within about one text height
/// (its own height) of the line between their centers.
fn beside_link(b: &BBox, p: &BBox, q: &BBox) -> bool {
    if box_distance(b, p) <= 0.0 || box_distance(b, q) <= 0.0 {
        return false;
    }
    let c = ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0);
    let a = ((p.x1 + p.x2) / 2.0, (p.y1 + p.y2) / 2.0);
    let z = ((q.x1 + q.x2) / 2.0, (q.y1 + q.y2) / 2.0);
    let (dx, dy) = (z.0 - a.0, z.1 - a.1);
    let len2 = dx * dx + dy * dy;
    if len2 <= 0.0 {
        return false;
    }
    let t = ((c.0 - a.0) * dx + (c.1 - a.1) * dy) / len2;
    if !(0.0..=1.0).contains(&t) {
        return false;
    }
    let dist = ((c.0 - a.0) * dy - (c.1 - a.1) * dx).abs() / len2.sqrt();
    dist <= 1.5 * b.height()
}

fn reversals(list: &[EdgeObs], a: usize, min_share: f64) -> Vec<usize> {
    let mut per_frame: Vec<(usize, EdgeDirection)> = Vec::new();
    for o in list {
        let Some(ev) = &o.evidence else { continue };
        let orient = |v: EndVerdict| if o.src == a { v } else { v.flipped() };
        let mut v = DirectionVotes::default();
        let (p, m) = ev.verdicts();
        v.pixel.add(orient(p), 1.0);
        if let Some(m) = m {
            v.vlm.add(orient(m), 1.0);
        }
        let (d, _) = v.decide(min_share);
        if matches!(d, EdgeDirection::AToB | EdgeDirection::BToA) {
            per_frame.push((o.frame, d));
        }
    }
    let mut out = Vec::new();
    let mut established = per_frame.first().map(|p| p.1);
    for w in per_frame.windows(2) {
        if Some(w[0].1) != established && w[0].1 == w[1].1 {
            out.push(w[0].0);
            established = Some(w[0].1);
        }
    }
    out
}

/// Split board keyframes into distinct boards (spec 8.1).
///
/// Keyframes are grouped by shared anchor texts (normalized, reliable texts of nodes,
/// stickies, other text, and OCR spans): two keyframes sharing any text are on the
/// same board. Groups that share no text are still one board (merged by text only)
/// unless something separates them: their board titles differ, or no contiguous run of
/// board keyframes contains both (a classify switch lies between every pair).
/// Keyframes without any anchor text join the board of the nearest keyframe in time.
/// Boards are returned in order of their first keyframe.
pub fn split_boards(
    mut frames: Vec<BoardFrame>,
    params: &ConsolidationParams,
) -> Vec<Vec<BoardFrame>> {
    frames.sort_by(|a, b| a.t_rep_s.total_cmp(&b.t_rep_s));
    let n = frames.len();
    if n == 0 {
        return Vec::new();
    }
    let texts: Vec<std::collections::BTreeSet<String>> = frames
        .iter()
        .map(|f| {
            let b = &f.board;
            b.nodes
                .iter()
                .map(|x| x.text.as_str())
                .chain(b.stickies.iter().map(|x| x.text.as_str()))
                .chain(b.other_visible_text.iter().map(|x| x.text.as_str()))
                .chain(f.ocr_anchors.iter().map(|x| x.text.as_str()))
                .filter(|t| !is_unreliable(t))
                .map(normalize)
                .collect()
        })
        .collect();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for i in 0..n {
        for j in i + 1..n {
            if !texts[i].is_empty() && texts[i].intersection(&texts[j]).next().is_some() {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a.max(b)] = a.min(b);
            }
        }
    }
    // Contiguous runs of board keyframes.
    let mut run = vec![0usize; n];
    for i in 1..n {
        run[i] =
            run[i - 1] + usize::from(frames[i].keyframe_index != frames[i - 1].keyframe_index + 1);
    }
    let anchored: Vec<usize> = (0..n).filter(|&i| !texts[i].is_empty()).collect();
    let roots: Vec<usize> = {
        let mut r: Vec<usize> = anchored.iter().map(|&i| find(&mut parent, i)).collect();
        r.sort_unstable();
        r.dedup();
        r
    };
    let title = |root: usize, parent: &mut Vec<usize>| -> Option<String> {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (i, f) in frames.iter().enumerate() {
            if find(parent, i) == root {
                if let Some(t) = f
                    .board_title
                    .as_deref()
                    .map(normalize)
                    .filter(|t| !t.is_empty())
                {
                    *counts.entry(t).or_default() += 1;
                }
            }
        }
        counts.into_iter().max_by_key(|e| e.1).map(|e| e.0)
    };
    let runs_of = |root: usize, parent: &mut Vec<usize>| -> std::collections::BTreeSet<usize> {
        (0..n)
            .filter(|&i| find(parent, i) == root)
            .map(|i| run[i])
            .collect()
    };
    let infos: Vec<(usize, Option<String>, std::collections::BTreeSet<usize>)> = roots
        .iter()
        .map(|&r| (r, title(r, &mut parent), runs_of(r, &mut parent)))
        .collect();
    // Merge groups that nothing separates.
    let mut board_of: Vec<usize> = (0..infos.len()).collect();
    for x in 0..infos.len() {
        for y in x + 1..infos.len() {
            let titles_differ = match (&infos[x].1, &infos[y].1) {
                (Some(a), Some(b)) => {
                    a != b && crate::difflib::ratio(a, b) < params.fuzzy_threshold
                }
                _ => false,
            };
            let share_run = infos[x].2.intersection(&infos[y].2).next().is_some();
            if !titles_differ && share_run {
                let (a, b) = (board_of[x], board_of[y]);
                let (lo, hi) = (a.min(b), a.max(b));
                for v in board_of.iter_mut() {
                    if *v == hi {
                        *v = lo;
                    }
                }
            }
        }
    }
    let mut board_idx = vec![usize::MAX; n];
    for (k, info) in infos.iter().enumerate() {
        for i in 0..n {
            if find(&mut parent, i) == info.0 && !texts[i].is_empty() {
                board_idx[i] = board_of[k];
            }
        }
    }
    // Keyframes without anchors join the nearest anchored keyframe's board.
    for i in 0..n {
        if board_idx[i] == usize::MAX {
            let nearest = anchored
                .iter()
                .min_by(|&&a, &&b| {
                    (frames[a].t_rep_s - frames[i].t_rep_s)
                        .abs()
                        .total_cmp(&(frames[b].t_rep_s - frames[i].t_rep_s).abs())
                })
                .map(|&a| board_idx[a]);
            board_idx[i] = nearest.unwrap_or(0);
        }
    }
    let mut boards: BTreeMap<usize, Vec<BoardFrame>> = BTreeMap::new();
    let mut first_seen: Vec<usize> = Vec::new();
    for (i, f) in frames.into_iter().enumerate() {
        if !first_seen.contains(&board_idx[i]) {
            first_seen.push(board_idx[i]);
        }
        boards.entry(board_idx[i]).or_default().push(f);
    }
    first_seen
        .into_iter()
        .filter_map(|b| boards.remove(&b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sticky_kinds() {
        assert_eq!(sticky_kind("Do we need a cache?"), StickyKind::Question);
        assert_eq!(sticky_kind("Beta milestone: March"), StickyKind::Milestone);
        assert_eq!(sticky_kind("Idea: batch the writes"), StickyKind::Idea);
        assert_eq!(sticky_kind("Clone the landing page"), StickyKind::Note);
    }
}
