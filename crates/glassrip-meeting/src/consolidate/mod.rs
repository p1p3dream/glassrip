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
//!    list. Illegible or elided texts are not tracked. A newer node track with the
//!    text of an established node is merged into it when its boxes are no evidence
//!    of a second place ([`duplicates`]). A node-majority track that no connector
//!    ends at is a sticky when it was read as a sticky at least as often, or when its
//!    text carries a sticky marker (question, milestone).
//! 3. Lifetimes and support ([`tracks::intervals`]). With a consensus reading
//!    ([`FrameVotes`]) every sighting weighs the share of the answering reads that
//!    listed the element: an interval needs `min_support_weight` in all, and
//!    `min_presence_share` per keyframe that could have read it (its sightings and
//!    the keyframes with its place in view), so an element the reads carried in a
//!    few of the views that showed it, or in one keyframe on two of three reads,
//!    does not decide the board. The final state is what was
//!    observed and not later removed: an element is final when its last supported
//!    interval was never ended by a removal. Removal needs evidence: consecutive
//!    later keyframes that cover the element's region and lack it. A keyframe covers
//!    the region when the region lies inside its registered view, OCR read none of
//!    the element's text there, and either the keyframe read another established
//!    element near that place, which confirms the registration there, or (with a
//!    [`RegionProbe`]) a well-supported registration maps the element's box onto
//!    canvas pixels that lost the ink the box held where the element was last read
//!    (a sparse board has no neighbor to confirm). Absence outside the view, or in a
//!    view with nothing known read near the place and no pixel evidence, is not
//!    removal. A keyframe whose whole canvas went blank (and whose OCR read none of
//!    the element's text) covers every element read before it, registered or not:
//!    erasing the last element leaves a view with nothing to register.
//!    An element never placed on the board (text-only registration) seen in a
//!    single keyframe cannot be checked against later views and is not final.
//! 4. Edges are keyed by their two node tracks; an endpoint that is not a supported
//!    node drops the edge (nodes are never created from endpoints). Edge support is
//!    weighted like element support, over the keyframes that read both ends.
//!    Directions come from the weighted vote over `glassrip.edge_direction` evidence
//!    (the reader's channel weighted by how many reads agreed on the direction), and
//!    the label is the one with the most label votes across keyframes. An edge is
//!    absent from a keyframe that read (or covers) both of its ends without it only
//!    with evidence that the connector is gone: readers often leave a connector out
//!    of a reading. With a [`RegionProbe`], the straight corridor between the two
//!    boxes decides when it held a line where the edge was last read: a line still
//!    there keeps the edge whatever else changed on the board, a line gone removes
//!    it. Otherwise (no pixels, or a routed connector the corridor does not follow)
//!    the board's aligned ink must have changed since the edge was last read.
//! 5. Owner tags become timed assignments ([`owners`]), anchored on the OCR positions of
//!    tags and nodes where OCR read them ([`owner_geometry`]). A reader tag OCR did
//!    not read corroborates only when at least `owner_reader_min_share` of the reads
//!    listed it.
//! 6. Events are computed from the state changes and gated on ink ([`events`]).

pub mod anchor;
mod cleanup;
pub mod duplicates;
pub mod events;
pub mod owner_geometry;
pub mod owners;
pub mod tracks;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Mutex;

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
use crate::similarity::Similarity;
use crate::text::{clean_label, is_unreliable, normalize, AliasTable};

use anchor::{anchor_tag, box_distance, edge_end_ambiguities, AnchorParams, Anchored, EdgeGeom};
use cleanup::{derive_groups, fold_fragments, Titles};
use duplicates::{apply_merges, duplicate_merges, Sighting};
use events::{BoardEvent, EventGate, EventKind, SuppressReason, SuppressedEvent};
use owner_geometry::{FrameGeometry, OwnerGeometryParams, TagIn};
use owners::{
    apply_alternation, assign_with, collapse_edge_pairs, consolidate_tags, AnchorKind,
    Corroborator, NameRead, OwnerAssignment, OwnerParams, OwnerSighting, OwnerTarget, TagPlace,
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

/// Consensus vote shares of one keyframe's validated reading (the `votes` of its
/// `glassrip.board_validate` item): per element, the share of the answering reads
/// that listed it, parallel to the reading's lists.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct FrameVotes {
    /// Node shares.
    pub nodes: Vec<f64>,
    /// Edge shares.
    pub edges: Vec<EdgeVoteShares>,
    /// Sticky shares.
    pub stickies: Vec<f64>,
    /// Owner tag shares.
    pub owner_tags: Vec<f64>,
    /// Other text shares.
    pub other_visible_text: Vec<f64>,
}

/// Consensus vote shares of one edge.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EdgeVoteShares {
    /// Share of the answering reads that listed the edge.
    pub share: f64,
    /// Share of the reads listing it whose direction is the kept one.
    pub direction: f64,
    /// Share of the answering reads that carry its label.
    pub label: f64,
}

impl FrameVotes {
    /// The votes run parallel to `board`'s lists (otherwise they are ignored).
    pub fn fits(&self, board: &ValidatedBoard) -> bool {
        self.nodes.len() == board.nodes.len()
            && self.edges.len() == board.edges.len()
            && self.stickies.len() == board.stickies.len()
            && self.owner_tags.len() == board.owner_tags.len()
            && self.other_visible_text.len() == board.other_visible_text.len()
    }
}

/// A share as a sighting weight: within `[0, 1]`, and 1 when unknown.
fn share_of(v: Option<&f64>) -> f64 {
    v.copied()
        .filter(|x| x.is_finite())
        .map_or(1.0, |x| x.clamp(0.0, 1.0))
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

/// Pixel evidence about regions of a keyframe's canvas (optional). Coordinates are
/// the keyframe reading's canvas coordinates. `None` when the keyframe's pixels are
/// unavailable or the region is degenerate.
pub trait RegionProbe: Send + Sync {
    /// Share of the pixels inside `region` that differ from the canvas background
    /// (0 on blank canvas, near 1 over a filled card).
    fn ink_share(&self, keyframe_id: &str, region: &BBox) -> Option<f64>;
    /// Share of positions along the straight corridor from `a` to `b` (`half_width`
    /// on each side; the ends, where the boxes sit, are skipped) at which ink crosses
    /// the corridor: near 1 over a drawn straight connector, 0 on blank canvas.
    fn line_cover(
        &self,
        keyframe_id: &str,
        a: (f64, f64),
        b: (f64, f64),
        half_width: f64,
    ) -> Option<f64>;
    /// Traces ink from box `a` to box `b` inside `region`, whatever route it
    /// takes: ink inside the two boxes and inside every box of `masks` (the other
    /// elements read there) is ignored, and a connector must reach the band of
    /// width `ring` around each box. `None` without pixels, for a degenerate
    /// region, or when the two bands touch (nothing to trace between them).
    fn stroke_between(
        &self,
        _keyframe_id: &str,
        _a: &BBox,
        _b: &BBox,
        _region: &BBox,
        _masks: &[BBox],
        _ring: f64,
    ) -> Option<StrokeTrace> {
        None
    }
    /// True when [`RegionProbe::stroke_between`] traces strokes. A probe that
    /// traces and still returns `None` for the keyframe that read a connector
    /// cannot tell whether the connector went; one that does not trace leaves
    /// the straight corridor to decide alone.
    fn traces(&self) -> bool {
        false
    }
    /// `Some(true)` when the keyframe's whole canvas is one flat color (no drawn
    /// content, not even faint marks: near-zero luminance spread); `None` without
    /// pixels.
    fn uniform(&self, _keyframe_id: &str) -> Option<bool> {
        None
    }
}

/// What the pixels say about a connector's own traced stroke between the
/// keyframe that read it and one that did not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Trace {
    /// It joined the two boxes and no longer does, and its ink fell.
    Gone,
    /// It joined the two boxes and no longer does, without an ink drop.
    Broken,
    /// It still joins the two boxes.
    Joined,
    /// The probe does not trace strokes.
    Untraced,
    /// The pixels cannot tell (see `routed_trace`).
    Inconclusive,
}

/// Result of [`RegionProbe::stroke_between`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeTrace {
    /// Connected ink (small gaps bridged) runs from the band around one box to
    /// the band around the other.
    pub joined: bool,
    /// Share of the region holding ink outside the ignored boxes.
    pub ink: f64,
}

/// Thresholds for [`RegionProbe`] evidence. Every comparison is between the same
/// measure in two keyframes: where the element or edge was last read, and where it
/// is missing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RegionProbeParams {
    /// A corridor keeps at least this share of its line cover: the connector is
    /// still drawn and the reading left it out.
    pub still_drawn_ratio: f64,
    /// A corridor or box keeps at most this share of its measure: the content is gone.
    pub gone_ratio: f64,
    /// The corridor speaks only when the line covered at least this share of it
    /// where the edge was read (a routed connector leaves the straight corridor).
    pub min_line_cover: f64,
    /// The box speaks only when it held at least this ink share where the element
    /// was read.
    pub min_ink_share: f64,
    /// Corridor half width as a share of the smaller end box height (at least 3 px).
    pub corridor_half_width_share: f64,
    /// Registration inliers a keyframe needs before its mapped box is trusted
    /// without a neighboring established element (references always are).
    pub min_registration_inliers: usize,
    /// A routed connector (no corridor verdict) is removed on board ink only when
    /// its traced stroke no longer joins its ends and the unmasked ink share of
    /// the traced region fell by at least this much...
    #[serde(default = "default_min_ink_drop")]
    pub min_ink_drop: f64,
    /// ...and by at least this share of its previous value.
    #[serde(default = "default_min_ink_drop_share")]
    pub min_ink_drop_share: f64,
    /// A whole canvas under this ink share is blank: everything read on it before
    /// is gone, even when the blank view cannot be registered.
    #[serde(default = "default_blank_canvas_share")]
    pub blank_canvas_share: f64,
    /// A routed connector is traced inside the ends' union box grown by this share
    /// of the taller end box on every side (a connector routed around a card
    /// leaves the union box).
    #[serde(default = "default_stroke_margin_share")]
    pub stroke_margin_share: f64,
}

fn default_stroke_margin_share() -> f64 {
    1.5
}

fn default_min_ink_drop() -> f64 {
    0.002
}

fn default_min_ink_drop_share() -> f64 {
    0.05
}

fn default_blank_canvas_share() -> f64 {
    0.002
}

impl Default for RegionProbeParams {
    fn default() -> Self {
        Self {
            still_drawn_ratio: 0.6,
            gone_ratio: 0.25,
            min_line_cover: 0.5,
            min_ink_share: 0.04,
            corridor_half_width_share: 0.25,
            min_registration_inliers: 3,
            min_ink_drop: default_min_ink_drop(),
            min_ink_drop_share: default_min_ink_drop_share(),
            blank_canvas_share: default_blank_canvas_share(),
            stroke_margin_share: default_stroke_margin_share(),
        }
    }
}

/// Hooks from other stages.
#[derive(Clone, Copy)]
pub struct Hooks<'a> {
    /// Transcript corroboration of owner moves.
    pub corroborator: &'a dyn Corroborator,
    /// Second reader pass for single sightings.
    pub second_reader: Option<&'a dyn SecondReader>,
}

// The fallback pass must see the same reader answers without calling the reader
// twice for the same keyframe and text.
struct MemoSecondReader<'a> {
    inner: &'a dyn SecondReader,
    answers: Mutex<BTreeMap<(String, String), bool>>,
}

impl SecondReader for MemoSecondReader<'_> {
    fn confirms(&self, keyframe_id: &str, text: &str) -> bool {
        let key = (keyframe_id.to_string(), text.to_string());
        let cached = self
            .answers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .copied();
        if let Some(answer) = cached {
            return answer;
        }
        let answer = self.inner.confirms(keyframe_id, text);
        self.answers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, answer);
        answer
    }
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
    /// Minimum summed consensus vote share of an interval's sightings (each
    /// keyframe weighs the share of the answering reads that listed the element;
    /// 1 for a single read). 1.5 needs two unanimous keyframes, or three at two of
    /// three reads.
    #[serde(default = "default_min_support_weight")]
    pub min_support_weight: f64,
    /// Minimum summed vote share per keyframe of an interval that could have read
    /// the element: its sightings and the keyframes with its place in view.
    #[serde(default = "default_min_presence_share")]
    pub min_presence_share: f64,
    /// The same for an edge, over the keyframes around its interval that read
    /// both of its ends or had their places in view.
    #[serde(default = "default_edge_min_presence_share")]
    pub edge_min_presence_share: f64,
    /// Presence is measured over at least this many board keyframes around an
    /// interval (a short one is widened on both sides).
    #[serde(default = "default_presence_window_keyframes")]
    pub presence_window_keyframes: usize,
    /// Vote share a single confirmed sighting needs to support an element alone.
    #[serde(default = "default_single_min_share")]
    pub single_min_share: f64,
    /// A same-text node or sticky whose summed vote share is at most this share
    /// of an established element's, and that was only read where the established
    /// element's place was in view but unread, is that element at a displaced
    /// place: never a second final element.
    #[serde(default = "default_echo_max_weight_share")]
    pub echo_max_weight_share: f64,
    /// Vote share a reader tag OCR did not read needs to count as an owner
    /// sighting.
    #[serde(default = "default_owner_reader_min_share")]
    pub owner_reader_min_share: f64,
    /// Consecutive visible-but-absent keyframes that remove an element.
    pub removal_absent_keyframes: usize,
    /// Length of the final stable board window, in seconds, measured back from the end
    /// of the last contiguous run of board keyframes. Default 120 s: long enough to
    /// span a few keyframes at the 2 s sampling grid, short enough to exclude earlier
    /// board states. Not a spec number; tune per corpus.
    pub final_window_s: f64,
    /// The final window holds at least this many board keyframes when its contiguous
    /// run has that many (it is extended backward, never past the run's start). The
    /// window only dates the final state
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
    /// Pixel evidence thresholds ([`RegionProbe`]).
    #[serde(default)]
    pub region_probe: RegionProbeParams,
    /// Share of the decisive weight a direction needs within a channel.
    pub vote_min_share: f64,
    /// Consecutive consistent keyframes to open or move an owner assignment.
    pub owner_confirm_keyframes: usize,
    /// Owner tag geometry thresholds.
    #[serde(default)]
    pub owner_anchor: AnchorParams,
    /// Owner tags and nodes placed on their OCR text for owner geometry.
    #[serde(default)]
    pub owner_geometry: OwnerGeometryParams,
    /// Shortest last board keyframe, in seconds, whose single OCR-placed geometric
    /// sighting of an owner tag opens an assignment (no later keyframe can confirm it).
    #[serde(default = "default_owner_final_hold_s")]
    pub owner_final_hold_s: f64,
    /// Reach, in tag sizes, within which a person's OCR-read tags at one registered
    /// place are the same physical tag (one target set for all its keyframes).
    #[serde(default = "default_owner_tag_reach_share")]
    pub owner_tag_reach_share: f64,
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
            min_support_weight: default_min_support_weight(),
            min_presence_share: default_min_presence_share(),
            edge_min_presence_share: default_edge_min_presence_share(),
            presence_window_keyframes: default_presence_window_keyframes(),
            single_min_share: default_single_min_share(),
            echo_max_weight_share: default_echo_max_weight_share(),
            owner_reader_min_share: default_owner_reader_min_share(),
            removal_absent_keyframes: 2,
            final_window_s: 120.0,
            min_final_keyframes: 3,
            coverage_radius_share: default_coverage_radius_share(),
            single_sighting_min_conf: 0.8,
            title_band_share: default_title_band_share(),
            min_group_cards: default_min_group_cards(),
            ink_event_threshold: 0.05,
            region_probe: RegionProbeParams::default(),
            vote_min_share: 0.6,
            owner_confirm_keyframes: 2,
            owner_anchor: AnchorParams::default(),
            owner_geometry: OwnerGeometryParams::default(),
            owner_final_hold_s: default_owner_final_hold_s(),
            owner_tag_reach_share: default_owner_tag_reach_share(),
            alternation_min_alternations: 3,
            backfill_untargeted_owners: false,
        }
    }
}

fn default_min_support_weight() -> f64 {
    1.5
}

fn default_min_presence_share() -> f64 {
    0.3
}

fn default_edge_min_presence_share() -> f64 {
    0.3
}

fn default_single_min_share() -> f64 {
    1.0
}

fn default_presence_window_keyframes() -> usize {
    6
}

fn default_echo_max_weight_share() -> f64 {
    0.25
}

fn default_owner_reader_min_share() -> f64 {
    1.0
}

fn default_owner_final_hold_s() -> f64 {
    30.0
}

fn default_owner_tag_reach_share() -> f64 {
    0.5
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
    /// A second reading of an established node at a misplaced or duplicated box
    /// (since board_state 1.4.0).
    Duplicate,
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
    /// Consensus vote shares of the edge in this keyframe.
    shares: EdgeVoteShares,
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

/// The owner tags of one reading: (name as read, the reader's `near`, box). A
/// participant name read as a node, or rejected from the node list as one, is a tag
/// read in the wrong list.
fn reading_tags(f: &BoardFrame, participants: &AliasTable) -> Vec<(String, String, BBox)> {
    let mut tags: Vec<(String, String, BBox)> = f
        .board
        .owner_tags
        .iter()
        .map(|o| (o.name_raw.clone(), o.near.clone(), o.bbox))
        .collect();
    for r in &f.board.chrome_rejected {
        if r.list == ElementList::Nodes && r.reason == RejectReason::ParticipantName {
            if let Some(b) = r.bbox {
                tags.push((r.text.clone(), String::new(), b));
            }
        }
    }
    for nd in &f.board.nodes {
        if participants.resolve(&nd.text).is_some() {
            tags.push((nd.text.clone(), String::new(), nd.bbox));
        }
    }
    tags
}

/// Consensus vote shares of [`reading_tags`], in its order (a participant name
/// rejected from the node list has no vote: 1).
fn reading_tag_shares(
    f: &BoardFrame,
    votes: Option<&FrameVotes>,
    participants: &AliasTable,
) -> Vec<f64> {
    let mut shares: Vec<f64> = (0..f.board.owner_tags.len())
        .map(|i| share_of(votes.and_then(|v| v.owner_tags.get(i))))
        .collect();
    for r in &f.board.chrome_rejected {
        if r.list == ElementList::Nodes
            && r.reason == RejectReason::ParticipantName
            && r.bbox.is_some()
        {
            shares.push(1.0);
        }
    }
    for (i, nd) in f.board.nodes.iter().enumerate() {
        if participants.resolve(&nd.text).is_some() {
            shares.push(share_of(votes.and_then(|v| v.nodes.get(i))));
        }
    }
    shares
}

/// Owner geometry of one keyframe for its reading's `tags`
/// ([`owner_geometry::place`]).
fn frame_geometry(
    f: &BoardFrame,
    tags: &[(String, String, BBox)],
    params: &ConsolidationParams,
    diagonal: f64,
) -> FrameGeometry {
    let person_of = |text: &str| {
        params
            .participants
            .resolve(text)
            .map(|p| p.person_id.clone())
    };
    let people: Vec<Option<String>> = tags.iter().map(|t| person_of(&t.0)).collect();
    let tag_in: Vec<TagIn<'_>> = tags
        .iter()
        .zip(&people)
        .map(|(t, p)| TagIn {
            bbox: t.2,
            person: p.as_deref(),
        })
        .collect();
    owner_geometry::place(
        &f.board,
        &tag_in,
        &f.ocr_anchors,
        &person_of,
        diagonal,
        &params.owner_geometry,
    )
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
    frames: Vec<BoardFrame>,
    board_id: &str,
    params: &ConsolidationParams,
    hooks: &Hooks<'_>,
) -> BoardStateItem {
    consolidate_with_probe(frames, board_id, params, hooks, None)
}

/// [`consolidate`] with pixel evidence about the keyframes' canvases.
///
/// A connector read only at fragments of an element keeps that element a node
/// only when the edge builder draws one of its edges. Whether it does depends on
/// evidence gathered after the kind decision (coverage, corridor probes, removal
/// intervals), so the builder's own result decides: when an element kept a node
/// only by fragment connectors ends with no edge, the board is consolidated
/// again with that protection withdrawn. Withdrawing it only removes edges that
/// were not drawn, so a second pass is the last.
pub fn consolidate_with_probe(
    frames: Vec<BoardFrame>,
    board_id: &str,
    params: &ConsolidationParams,
    hooks: &Hooks<'_>,
    probe: Option<&dyn RegionProbe>,
) -> BoardStateItem {
    consolidate_voted(frames, &HashMap::new(), board_id, params, hooks, probe)
}

/// [`consolidate_with_probe`] with the consensus vote shares of the keyframes'
/// readings, by keyframe id ([`FrameVotes`]; a keyframe without votes weighs
/// every sighting 1).
pub fn consolidate_voted(
    frames: Vec<BoardFrame>,
    votes: &HashMap<String, FrameVotes>,
    board_id: &str,
    params: &ConsolidationParams,
    hooks: &Hooks<'_>,
    probe: Option<&dyn RegionProbe>,
) -> BoardStateItem {
    let second_reader = hooks.second_reader.map(|inner| MemoSecondReader {
        inner,
        answers: Mutex::new(BTreeMap::new()),
    });
    let memo_hooks = Hooks {
        corroborator: hooks.corroborator,
        second_reader: second_reader.as_ref().map(|r| r as &dyn SecondReader),
    };
    let orphaned = match consolidate_pass(
        frames.clone(),
        votes,
        board_id,
        params,
        &memo_hooks,
        probe,
        &BTreeSet::new(),
    ) {
        Ok(state) => return state,
        Err(orphaned) => orphaned,
    };
    match consolidate_pass(
        frames,
        votes,
        board_id,
        params,
        &memo_hooks,
        probe,
        &orphaned,
    ) {
        Ok(state) => state,
        Err(_) => unreachable!("a pass with fragment protection withdrawn cannot retry"),
    }
}

/// One consolidation. Tracks in `withdrawn` are not kept nodes by connectors read
/// at their fragments. Before owner callbacks, returns the tracks that need the
/// fallback pass instead of an incomplete state.
fn consolidate_pass(
    mut frames: Vec<BoardFrame>,
    votes: &HashMap<String, FrameVotes>,
    board_id: &str,
    params: &ConsolidationParams,
    hooks: &Hooks<'_>,
    probe: Option<&dyn RegionProbe>,
    withdrawn: &BTreeSet<usize>,
) -> Result<BoardStateItem, BTreeSet<usize>> {
    let corroborator = hooks.corroborator;
    frames.sort_by(|a, b| a.t_rep_s.total_cmp(&b.t_rep_s));
    let n = frames.len();
    let fz = params.fuzzy_threshold;
    // Consensus vote shares per keyframe (in time order), when they fit its reading.
    let frame_votes: Vec<Option<&FrameVotes>> = frames
        .iter()
        .map(|f| votes.get(&f.keyframe_id).filter(|v| v.fits(&f.board)))
        .collect();
    let edge_shares = |fi: usize, ei: usize| -> EdgeVoteShares {
        frame_votes[fi].and_then(|v| v.edges.get(ei)).map_or(
            EdgeVoteShares {
                share: 1.0,
                direction: 1.0,
                label: 1.0,
            },
            |e| EdgeVoteShares {
                share: share_of(Some(&e.share)),
                direction: share_of(Some(&e.direction)),
                label: share_of(Some(&e.label)),
            },
        )
    };

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

    // OCR anchors of every positioned keyframe in its cluster's reference frame, with
    // their normalized text.
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
    // OCR of keyframe `f` read `text` at (or around) reference box `b`.
    let ocr_reads = |f: usize, text: &str, b: &BBox| -> bool {
        let (bw, bh) = (b.width() * 0.5, b.height() * 0.5);
        let near_box = BBox::new(b.x1 - bw, b.y1 - bh, b.x2 + bw, b.y2 + bh);
        ocr_ref[f].iter().any(|(a, ab)| {
            let c = ((ab.x1 + ab.x2) / 2.0, (ab.y1 + ab.y2) / 2.0);
            c.0 >= near_box.x1
                && c.0 <= near_box.x2
                && c.1 >= near_box.y1
                && c.1 <= near_box.y2
                && (text.contains(a.as_str()) || a.contains(text))
        })
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
        let fv = frame_votes[fi];
        for (ni, nd) in f.board.nodes.iter().enumerate() {
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
                weight: share_of(fv.and_then(|v| v.nodes.get(ni))),
            });
        }
        for (si, s) in f.board.stickies.iter().enumerate() {
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
                weight: share_of(fv.and_then(|v| v.stickies.get(si))),
            });
        }
        for (xi, t) in f.board.other_visible_text.iter().enumerate() {
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
                weight: share_of(fv.and_then(|v| v.other_visible_text.get(xi))),
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
            let label_share = edge_shares(fi, ei).label;
            match obs.iter().position(|o| normalize(&o.text) == nl) {
                Some(oi) => {
                    let o = &mut obs[oi];
                    if !o.lists.contains(&ObsList::EdgeLabel) {
                        o.lists.push(ObsList::EdgeLabel);
                    }
                    o.weight = o.weight.max(label_share);
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
                    weight: label_share,
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

    // Duplicate node tracks ([`duplicates`]). A registration is trusted as for pixel
    // evidence; a reading is unreliable when its boxes coincide or the OCR geometry
    // check found most of them displaced.
    let trusted_registration = |f: usize| {
        let r = &regs[f];
        r.mode == RegistrationMode::Reference
            || (r.mode == RegistrationMode::Registered
                && r.inliers >= params.region_probe.min_registration_inliers)
    };
    let geometries: Vec<FrameGeometry> = frames
        .iter()
        .enumerate()
        .map(|(fi, f)| {
            frame_geometry(
                f,
                &reading_tags(f, &params.participants),
                params,
                diagonals[fi],
            )
        })
        .collect();
    let reading_bad: Vec<bool> = frames
        .iter()
        .zip(&geometries)
        .map(|(f, g)| !geometry_ok(&f.board) || g.unreliable)
        .collect();
    // OCR spans of keyframe `f` reading `text` with their center inside `b` grown by
    // `grow` of its size on each side.
    let ocr_hits = |f: usize, text: &str, b: &BBox, grow: f64| -> Vec<usize> {
        let (gx, gy) = (b.width() * grow, b.height() * grow);
        ocr_ref[f]
            .iter()
            .enumerate()
            .filter(|(_, (a, ab))| {
                let c = ((ab.x1 + ab.x2) / 2.0, (ab.y1 + ab.y2) / 2.0);
                c.0 >= b.x1 - gx
                    && c.0 <= b.x2 + gx
                    && c.1 >= b.y1 - gy
                    && c.1 <= b.y2 + gy
                    && (text.contains(a.as_str()) || a.contains(text))
            })
            .map(|(i, _)| i)
            .collect()
    };
    let same_place = |a: &BBox, b: &BBox| {
        let (ca, cb) = (
            ((a.x1 + a.x2) / 2.0, (a.y1 + a.y2) / 2.0),
            ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0),
        );
        (ca.0 - cb.0).hypot(ca.1 - cb.1) <= match_params.position_tolerance_px || a.iou(b) >= 0.3
    };
    // Tracks placed in at least two keyframes, and where each track's element is: its
    // own place when it is placed so, else the place of any such track with its text.
    let placed_twice: Vec<bool> = tracks
        .iter()
        .map(|u| u.obs.iter().filter(|o| o.bbox.is_some()).count() >= 2)
        .collect();
    let texts: Vec<String> = tracks.iter().map(Track::text).collect();
    let homes: Vec<Vec<usize>> = (0..tracks.len())
        .map(|u| {
            if placed_twice[u] {
                return vec![u];
            }
            if tracks[u].obs.iter().all(|o| o.bbox.is_none()) {
                return Vec::new();
            }
            (0..tracks.len())
                .filter(|&v| {
                    v != u && placed_twice[v] && duplicates::same_text(&texts[u], &texts[v], fz)
                })
                .collect()
        })
        .collect();
    // Each track's box per registered cluster, and per keyframe the readings of tracks
    // whose element has a known place: (track, box, whether the box agrees with where
    // the element is).
    let track_box: Vec<BTreeMap<usize, BBox>> = tracks
        .iter()
        .map(|t| {
            let clusters: BTreeSet<usize> = t
                .obs
                .iter()
                .filter(|o| o.bbox.is_some())
                .map(|o| o.cluster)
                .collect();
            clusters
                .into_iter()
                .filter_map(|c| t.bbox_in(c).map(|b| (c, b)))
                .collect()
        })
        .collect();
    let box_in = |u: usize, c: usize| track_box[u].get(&c).copied();
    let mut frame_placed: Vec<Vec<(usize, BBox, bool)>> = vec![Vec::new(); n];
    for (u, t) in tracks.iter().enumerate() {
        for o in &t.obs {
            let Some(b) = o.bbox else { continue };
            let at: Vec<BBox> = homes[u]
                .iter()
                .filter_map(|&v| box_in(v, o.cluster))
                .collect();
            if !at.is_empty() {
                frame_placed[o.frame].push((u, b, at.iter().any(|h| same_place(&b, h))));
            }
        }
    }
    let radius = params.coverage_radius_share * ref_diag;
    // The registration of keyframe `f` is unconfirmed around `b` (no reading of a
    // track placed in two keyframes lies within the coverage radius where its track
    // is) and wrong elsewhere in the keyframe (some reading lies away from where its
    // element is): the sighting's registered place says nothing. The readings of
    // tracks `skip` and of every track with the candidate's text are left out of
    // both: same-label readings never vouch for each other.
    let registration_off = |f: usize, b: &BBox, skip: [usize; 2]| -> bool {
        let c = ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0);
        let others = || {
            frame_placed[f].iter().filter(|x| {
                !skip.contains(&x.0) && !duplicates::same_text(&texts[x.0], &texts[skip[0]], fz)
            })
        };
        let confirmed = radius.is_finite()
            && others().any(|&(u, ub, agrees)| {
                let uc = ((ub.x1 + ub.x2) / 2.0, (ub.y1 + ub.y2) / 2.0);
                placed_twice[u] && agrees && (uc.0 - c.0).hypot(uc.1 - c.1) <= radius
            });
        !confirmed && others().any(|x| !x.2)
    };
    // OCR reading the text at the sighting's own place (and not at the established
    // track's) is never explained away: a second element there on a trusted
    // registration and a reliable reading, else unknown.
    let judge = |ti: usize, i: usize, ei: usize| -> Sighting {
        let o = &tracks[ti].obs[i];
        let bad = reading_bad[o.frame];
        let Some((b, eb)) = o.bbox.and_then(|b| box_in(ei, o.cluster).map(|eb| (b, eb))) else {
            return if bad {
                Sighting::Explained
            } else {
                Sighting::Unknown
            };
        };
        if same_place(&b, &eb) {
            return Sighting::Near;
        }
        // A span inside the sighting's own box is its own unless it also lies inside
        // the established box itself (a neighbor's margin does not claim it); the
        // established place, grown by half its size, explains a sighting only when
        // OCR reads nothing at the sighting's own place.
        let text = normalize(&o.text);
        let inside_established = ocr_hits(o.frame, &text, &eb, 0.0);
        let own = ocr_hits(o.frame, &text, &b, 0.0)
            .into_iter()
            .any(|x| !inside_established.contains(&x));
        let at_established = ocr_hits(o.frame, &text, &eb, 0.5);
        let off = registration_off(o.frame, &b, [ti, ei]);
        if own {
            if trusted_registration(o.frame) && !bad && !off {
                Sighting::Apart
            } else {
                Sighting::Unknown
            }
        } else if bad || off || !at_established.is_empty() {
            Sighting::Explained
        } else {
            Sighting::Unknown
        }
    };
    let merges = duplicate_merges(&tracks, params.min_support_keyframes, fz, &judge);
    let near: HashSet<(usize, usize)> = merges
        .iter()
        .flat_map(|(&t, &e)| (0..tracks[t].obs.len()).map(move |i| (t, i, e)))
        .filter(|&(t, i, e)| judge(t, i, e) == Sighting::Near)
        .map(|(t, i, _)| (t, i))
        .collect();
    let duplicate_folds: Vec<FoldedElement> = merges
        .iter()
        .map(|(&t, &e)| FoldedElement {
            text: tracks[t].text(),
            into: tracks[e].text(),
            reason: FoldReason::Duplicate,
        })
        .collect();
    apply_merges(&mut tracks, &merges, &|t, i| near.contains(&(t, i)));
    for ti in node_track.values_mut() {
        if let Some(&e) = merges.get(ti) {
            *ti = e;
        }
    }
    for m in &mut label_merges {
        if let Some(&e) = merges.get(&m.0) {
            m.0 = e;
        }
    }

    // Fragments: a shorter reading of a longer element at the same place is folded
    // into it. The fragment's sightings do not count as support; references to it
    // (edge endpoints, owner targets) are redirected.
    let fold_to = fold_fragments(&tracks);
    let n_tracks = tracks.len();
    let resolve = |mut t: usize| {
        let mut guard = 0;
        while let Some(&u) = fold_to.get(&t) {
            t = u;
            guard += 1;
            if guard > n_tracks {
                break;
            }
        }
        t
    };
    // Connector ends. A connector read at an element's own readings keeps it a
    // node. One read at a fragment reaches the element through the redirect the
    // edges take (section 4): it keeps the element a node when that edge, between
    // the resolved ends, was read in at least `min_support_keyframes` keyframes
    // (necessary for an edge to be kept) and the edge builder draws an edge at the
    // element (checked at the end, see `consolidate_with_probe`). A connector the
    // state draws is then never lost to its end's kind, and one it drops cannot
    // turn a marked sticky into a node.
    let mut edge_ends: HashSet<usize> = HashSet::new();
    let mut via_fragment: BTreeMap<(usize, usize), (BTreeSet<usize>, BTreeSet<usize>)> =
        BTreeMap::new();
    for (fi, f) in frames.iter().enumerate() {
        for e in &f.board.edges {
            let raw = [&e.src, &e.dst].map(|end| node_track.get(&(fi, end.clone())).copied());
            let ends = raw.map(|t| t.map(resolve));
            for (r, t) in raw.iter().zip(&ends) {
                if let (Some(r), Some(t)) = (r, t) {
                    if r == t {
                        edge_ends.insert(*t);
                    }
                }
            }
            if let [Some(s), Some(d)] = ends {
                if s != d {
                    let entry = via_fragment.entry((s.min(d), s.max(d))).or_default();
                    entry.0.insert(fi);
                    for (r, t) in raw.iter().zip(&ends) {
                        if let (Some(r), Some(t)) = (r, t) {
                            if r != t {
                                entry.1.insert(*t);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut fragment_ends: BTreeSet<usize> = BTreeSet::new();
    for (read_in, ends) in via_fragment.values() {
        if read_in.len() >= params.min_support_keyframes {
            fragment_ends.extend(
                ends.iter()
                    .copied()
                    .filter(|t| !edge_ends.contains(t) && !withdrawn.contains(t)),
            );
        }
    }
    edge_ends.extend(fragment_ends.iter().copied());
    for ti in node_track.values_mut() {
        *ti = resolve(*ti);
    }

    // Kind beyond the list votes, decided after fragments are folded on the vote
    // kinds. A connector end (above) is a node: only nodes carry edges;
    // otherwise a node-majority track is a sticky when it was read as a sticky (a
    // colored card) at least as often as a node, or when its text carries a
    // sticky marker (a question or a milestone, spec 6.11): node labels name
    // components.
    for (ti, t) in tracks.iter_mut().enumerate() {
        let v = t.votes();
        if v.kind() != ObsList::Node || edge_ends.contains(&ti) {
            continue;
        }
        let marked = matches!(
            sticky_kind(&t.text()),
            StickyKind::Question | StickyKind::Milestone
        );
        if marked || (v.sticky > 0 && v.sticky >= v.node) {
            t.kind_override = Some(ObsList::Sticky);
        }
    }

    let folded: Vec<FoldedElement> = fold_to
        .iter()
        .map(|(&t, &u)| FoldedElement {
            text: tracks[t].text(),
            into: tracks[resolve(u)].text(),
            reason: FoldReason::Fragment,
        })
        .chain(duplicate_folds)
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
    let coverage_radius = params.coverage_radius_share * ref_diag;
    // Region-local disappearance: a well-supported registration maps the track's
    // reference box `b` into keyframe `f`, and those canvas pixels lost the ink the
    // box held in the keyframe that last read the track.
    let region_gone = |t: &Track, f: usize, b: &BBox| -> bool {
        let Some(probe) = probe else {
            return false;
        };
        let rp = &params.region_probe;
        let r = &regs[f];
        let trusted = r.mode == RegistrationMode::Reference
            || (r.mode == RegistrationMode::Registered && r.inliers >= rp.min_registration_inliers);
        if !trusted {
            return false;
        }
        let Some(inv) = r.to_reference.inverse() else {
            return false;
        };
        let Some((s, sb)) = t
            .obs
            .iter()
            .filter(|o| o.frame < f)
            .filter_map(|o| o.raw_bbox.map(|rb| (o.frame, rb)))
            .max_by_key(|x| x.0)
        else {
            return false;
        };
        let before = probe.ink_share(&frames[s].keyframe_id, &sb);
        let now = probe.ink_share(&frames[f].keyframe_id, &map_bbox(&inv, b));
        match (before, now) {
            (Some(before), Some(now)) => {
                before >= rp.min_ink_share && now <= rp.gone_ratio * before
            }
            _ => false,
        }
    };
    // The whole canvas of keyframe `f` is blank (and OCR read none of the track's
    // text anywhere on it), while the track's box held ink where it was last read,
    // and aligned ink links the two keyframes. Needs no registration: a view with
    // nothing on it cannot be registered, and an erased last element leaves
    // exactly that.
    let canvas_blanked = |t: &Track, f: usize| -> bool {
        let (Some(probe), Some(c)) = (probe, canvases[f]) else {
            return false;
        };
        let rp = &params.region_probe;
        let text = normalize(&t.text());
        if frames[f].ocr_anchors.iter().any(|a| {
            let a = normalize(&a.text);
            !a.is_empty() && (text.contains(a.as_str()) || a.contains(text.as_str()))
        }) {
            return false;
        }
        let Some((s, sb)) = t
            .obs
            .iter()
            .filter(|o| o.frame < f)
            .filter_map(|o| o.raw_bbox.map(|rb| (o.frame, rb)))
            .max_by_key(|x| x.0)
        else {
            return false;
        };
        // The blank view must be the same view: every board keyframe since the
        // last sighting was compared with its predecessor on aligned ink (a pan
        // or a cut the aligner cannot follow leaves the ink unknown), and the ink
        // changed on the way. A flat canvas has nothing to align on, so a link
        // between two adjacent keyframes that are both confidently blank is
        // measured from the pixels instead: no change. A link from content into
        // a blank view stays unknown when the aligner could not measure it: an
        // erasure, a pan and a cut to empty canvas look the same there, and the
        // element stays.
        let kid = |k: usize| frames[k].keyframe_id.as_str();
        let link = |k: usize| -> Option<f64> {
            if let Some(x) = frames[k].ink_change {
                return Some(x);
            }
            let adjacent = k > 0 && frames[k - 1].keyframe_index + 1 == frames[k].keyframe_index;
            (adjacent
                && probe.uniform(kid(k)) == Some(true)
                && probe.uniform(kid(k - 1)) == Some(true))
            .then_some(0.0)
        };
        let chain: Vec<Option<f64>> = (s + 1..=f).map(link).collect();
        if !chain.iter().all(Option::is_some)
            || !chain
                .iter()
                .flatten()
                .any(|&x| x >= params.ink_event_threshold)
        {
            return false;
        }
        let whole = BBox::new(0.0, 0.0, c.width, c.height);
        match (
            probe.ink_share(&frames[s].keyframe_id, &sb),
            probe.ink_share(&frames[f].keyframe_id, &whole),
        ) {
            (Some(before), Some(now)) => before >= rp.min_ink_share && now <= rp.blank_canvas_share,
            _ => false,
        }
    };
    // `Visible` only when keyframe `f` really covers the track's place: inside the
    // view (`track_visible`), no OCR text of the track there, and either other
    // content near or the place's pixels emptied.
    let track_covered = |t: &Track, f: usize| -> Visibility {
        if canvas_blanked(t, f) {
            return Visibility::Visible;
        }
        if track_visible(t, f) != Visibility::Visible {
            return Visibility::Unknown;
        }
        let Some(b) = t.bbox_in(regs[f].cluster) else {
            return Visibility::Unknown;
        };
        if ocr_reads(f, &normalize(&t.text()), &b) {
            return Visibility::Unknown;
        }
        let c = ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0);
        let corroborated = coverage_radius.is_finite()
            && content[f]
                .iter()
                .any(|p| ((p.0 - c.0).powi(2) + (p.1 - c.1).powi(2)).sqrt() <= coverage_radius);
        if corroborated || region_gone(t, f, &b) {
            Visibility::Visible
        } else {
            Visibility::Unknown
        }
    };
    let support = SupportParams {
        min_keyframes: params.min_support_keyframes,
        min_density: params.min_support_density,
        removal_absent: params.removal_absent_keyframes,
        min_weight: params.min_support_weight,
        min_presence: params.min_presence_share,
        single_min_share: params.single_min_share,
        presence_window: params.presence_window_keyframes.max(1),
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
                |f| t.weight_at(f),
                |f| track_visible(t, f) == Visibility::Visible,
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
            // Extend backward within the contiguous run only: an earlier run is
            // another stretch of the meeting (a classify switch between them).
            v = ((n - want).max(start)..n).collect();
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
        if !track_intervals[ti].is_empty() && t.kind() == ObsList::Node {
            established
                .entry(normalize(&t.text()))
                .or_default()
                .push((ti, t.frames()));
        }
    }
    // OCR reading the text at the single sighting's own place says the element is
    // really there (OCR boxes are pixel-accurate): a second element with that text,
    // not an echo.
    let ocr_at_own_place = |t: &Track| -> bool {
        let text = normalize(&t.text());
        t.obs.iter().any(|o| {
            o.bbox.is_some_and(|b| {
                ocr_ref[o.frame].iter().any(|(a, ab)| {
                    let c = ((ab.x1 + ab.x2) / 2.0, (ab.y1 + ab.y2) / 2.0);
                    c.0 >= b.x1
                        && c.0 <= b.x2
                        && c.1 >= b.y1
                        && c.1 <= b.y2
                        && (text.contains(a.as_str()) || a.contains(text.as_str()))
                })
            })
        })
    };
    let echo_of_established = |ti: usize, t: &Track| -> bool {
        let seen = t.frames();
        seen.len() == 1
            && !ocr_at_own_place(t)
            && established.get(&normalize(&t.text())).is_some_and(|v| {
                v.iter().any(|(u, frames)| {
                    *u != ti
                        && frames.len() >= params.min_support_keyframes
                        && !frames.contains(&seen[0])
                })
            })
    };
    // A same-text node or sticky the reads carried at most `echo_max_weight_share`
    // as strongly as an established element, read only in keyframes that had the
    // established element's place in view and did not read it there, is that
    // element read at a displaced place (a registration or box error): the
    // established element it stands for. A sighting that is evidence of a second
    // place (`apart_from`) makes it a second element.
    let track_weight = |t: &Track| t.frames().iter().map(|&f| t.weight_at(f)).sum::<f64>();
    // Sighting `o` of track `ti` is evidence of a place apart from established
    // track `u` (the `Apart` verdict of the duplicate judge): OCR reads its text at
    // its own box and not only inside `u`'s, on a trusted registration, a reliable
    // reading, and a registration confirmed around the box.
    let apart_from = |ti: usize, o: &Obs, u: usize| -> bool {
        let Some((b, eb)) = o.bbox.and_then(|b| box_in(u, o.cluster).map(|eb| (b, eb))) else {
            return false;
        };
        if same_place(&b, &eb) {
            return false;
        }
        let text = normalize(&o.text);
        let inside_established = ocr_hits(o.frame, &text, &eb, 0.0);
        let own = ocr_hits(o.frame, &text, &b, 0.0)
            .into_iter()
            .any(|x| !inside_established.contains(&x));
        own && trusted_registration(o.frame)
            && !reading_bad[o.frame]
            && !registration_off(o.frame, &b, [ti, u])
    };
    let displaced_echo = |ti: usize, t: &Track| -> Option<usize> {
        let boxy = |x: &Track| matches!(x.kind(), ObsList::Node | ObsList::Sticky);
        let seen = t.frames();
        if seen.is_empty() || !boxy(t) {
            return None;
        }
        let (w, text) = (track_weight(t), t.text());
        tracks
            .iter()
            .enumerate()
            .filter(|&(u, e)| {
                u != ti
                    && !track_intervals[u].is_empty()
                    && boxy(e)
                    && duplicates::same_text(&e.text(), &text, fz)
                    && w <= params.echo_max_weight_share * track_weight(e) + 1e-9
                    && seen.iter().all(|&f| {
                        e.weight_at(f) == 0.0 && track_visible(e, f) == Visibility::Visible
                    })
                    && !t.obs.iter().any(|o| apart_from(ti, o, u))
            })
            .max_by(|a, b| {
                track_weight(a.1)
                    .total_cmp(&track_weight(b.1))
                    .then(b.0.cmp(&a.0))
            })
            .map(|(u, _)| u)
    };
    for &ti in &order {
        let t = &tracks[ti];
        let ivs = &track_intervals[ti];
        if ivs.is_empty() {
            continue;
        }
        let votes = t.votes();
        let lifetimes: Vec<Lifetime> = ivs.iter().map(lifetime).collect();
        match t.kind() {
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
                        && !echo_of_established(ti, t)
                        && displaced_echo(ti, t).is_none(),
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
                    in_final: alive_in_final(ivs, t.obs.iter().any(|o| o.bbox.is_some()))
                        && displaced_echo(ti, t).is_none(),
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
        for (ei, e) in f.board.edges.iter().enumerate() {
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
                shares: edge_shares(fi, ei),
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

    // Keyframe `s` coordinates to keyframe `f` coordinates through the cluster
    // reference (both registered in one cluster), with the scale factor.
    let s_to_f = |s: usize, f: usize| -> Option<(Similarity, f64)> {
        if !(positioned(s) && positioned(f)) || regs[s].cluster != regs[f].cluster {
            return None;
        }
        let t = regs[f]
            .to_reference
            .inverse()?
            .compose(&regs[s].to_reference);
        (t.scale.is_finite() && t.scale > 0.0).then_some((t, t.scale))
    };
    // The straight corridor between the boxes of tracks `a` and `b` as read in
    // keyframe `s` (both read there): its two ends and half width, in `s`
    // coordinates.
    type Corridor = ((f64, f64), (f64, f64), f64);
    let corridor_geom = |a: &Track, b: &Track, s: usize| -> Option<Corridor> {
        let raw = |t: &Track| t.obs.iter().find(|o| o.frame == s).and_then(|o| o.raw_bbox);
        let (ba, bb) = (raw(a)?, raw(b)?);
        let (ca, cb) = (
            ((ba.x1 + ba.x2) / 2.0, (ba.y1 + ba.y2) / 2.0),
            ((bb.x1 + bb.x2) / 2.0, (bb.y1 + bb.y2) / 2.0),
        );
        let half =
            (params.region_probe.corridor_half_width_share * ba.height().min(bb.height())).max(3.0);
        Some((exit_point(&ba, cb), exit_point(&bb, ca), half))
    };
    // The straight corridor's line cover of the edge between `a` and `b`, read in
    // keyframe `s` and missing in `f`, measured on the same board pixels in both
    // keyframes (the geometry is taken where the edge was read and mapped into
    // `f`, so reader box jitter in `f` does not move the corridor).
    // `(cover_s, cover_f)`; `None` without pixels or a common registration.
    let edge_cover = |a: &Track, b: &Track, s: usize, f: usize| -> Option<(f64, f64)> {
        let probe = probe?;
        let (p, q, half) = corridor_geom(a, b, s)?;
        let (t, k) = s_to_f(s, f)?;
        let (ks, kf) = (&frames[s].keyframe_id, &frames[f].keyframe_id);
        Some((
            probe.line_cover(ks, p, q, half)?,
            probe.line_cover(kf, t.apply(p), t.apply(q), half * k)?,
        ))
    };
    // A connector between tracks `a` and `b` (read in keyframe `s`) traced on the
    // pixels, whatever route it takes: `Trace::Gone` when the stroke that joined
    // the two boxes in `s` no longer joins them in `f` and the ink it was traced
    // through fell; `Trace::Broken` when it no longer joins them but the ink did
    // not fall (ink added nearby); `Trace::Joined` when it still joins them;
    // `Trace::Untraced` when the probe does not trace strokes at all
    // ([`RegionProbe::traces`]); `Trace::Inconclusive` when the pixels cannot
    // tell: no trace in `s` from a probe that traces (a degenerate region, boxes
    // too close to trace between), the stroke did not join the boxes in `s` (dashed, or occluded by a
    // card read on its route), no common registration, an end read in `f` away
    // from where `s` puts it (a moved card takes its connector along), or any
    // other element over the traced region added, removed, moved or left unread
    // between the two keyframes (a
    // card placed on a connector hides part of it, a card moved off it may let it
    // be rerouted through the place it left). The other elements are masked out at
    // their boxes in each keyframe, so an unchanged card between the ends neither
    // vetoes nor fakes the erasure, and an unread mark erased near the ends does
    // not break a stroke that is still drawn. Traces in `s` are cached per
    // (a, b, s).
    let raw_at = |t: &Track, k: usize| t.obs.iter().find(|o| o.frame == k).and_then(|o| o.raw_bbox);
    let others = |a: usize, b: usize| {
        tracks
            .iter()
            .enumerate()
            .filter(move |(i, _)| *i != a && *i != b)
            .map(|(_, x)| x)
    };
    // (trace, traced region, band width) in the keyframe that read the edge.
    type Traced = Option<(StrokeTrace, BBox, f64)>;
    let traced: std::cell::RefCell<HashMap<(usize, usize, usize), Traced>> =
        std::cell::RefCell::new(HashMap::new());
    let routed_trace = |a: usize, b: usize, s: usize, f: usize| -> Trace {
        let Some(probe) = probe else {
            return Trace::Untraced;
        };
        let (ta, tb) = (&tracks[a], &tracks[b]);
        let (Some(ba), Some(bb)) = (raw_at(ta, s), raw_at(tb, s)) else {
            return Trace::Inconclusive;
        };
        let Some((t, k)) = s_to_f(s, f) else {
            return Trace::Inconclusive;
        };
        let center = |r: &BBox| ((r.x1 + r.x2) / 2.0, (r.y1 + r.y2) / 2.0);
        let inside =
            |c: (f64, f64), m: &BBox| c.0 >= m.x1 && c.0 <= m.x2 && c.1 >= m.y1 && c.1 <= m.y2;
        for (tr, bs) in [(ta, &ba), (tb, &bb)] {
            if let Some(r) = raw_at(tr, f) {
                if !inside(center(&r), &map_bbox(&t, bs)) {
                    return Trace::Inconclusive;
                }
            }
        }
        let rp = &params.region_probe;
        let before = *traced.borrow_mut().entry((a, b, s)).or_insert_with(|| {
            let ring = (rp.corridor_half_width_share * ba.height().min(bb.height())).max(3.0);
            let margin = rp.stroke_margin_share * ba.height().max(bb.height());
            let region = BBox::new(
                ba.x1.min(bb.x1) - margin,
                ba.y1.min(bb.y1) - margin,
                ba.x2.max(bb.x2) + margin,
                ba.y2.max(bb.y2) + margin,
            );
            let masks: Vec<BBox> = others(a, b).filter_map(|x| raw_at(x, s)).collect();
            probe
                .stroke_between(&frames[s].keyframe_id, &ba, &bb, &region, &masks, ring)
                .map(|x| (x, region, ring))
        });
        let Some((before, region, ring)) = before else {
            return if probe.traces() {
                Trace::Inconclusive
            } else {
                Trace::Untraced
            };
        };
        if !before.joined {
            return Trace::Inconclusive;
        }
        // Every other element over the traced region is read in both keyframes at
        // the same place (within the band width).
        let region_f = map_bbox(&t, &region);
        let meets = |r: &BBox| {
            r.x1 < region_f.x2 && region_f.x1 < r.x2 && r.y1 < region_f.y2 && region_f.y1 < r.y2
        };
        let mut masks = Vec::new();
        for x in others(a, b) {
            let (at_s, at_f) = (raw_at(x, s).map(|r| map_bbox(&t, &r)), raw_at(x, f));
            if !(at_s.as_ref().is_some_and(meets) || at_f.as_ref().is_some_and(meets)) {
                continue;
            }
            let (Some(ms), Some(rf)) = (at_s, at_f) else {
                return Trace::Inconclusive;
            };
            let (cs, cf) = (center(&ms), center(&rf));
            if (cs.0 - cf.0).hypot(cs.1 - cf.1) > ring * k {
                return Trace::Inconclusive;
            }
            masks.push(rf);
        }
        let Some(now) = probe.stroke_between(
            &frames[f].keyframe_id,
            &map_bbox(&t, &ba),
            &map_bbox(&t, &bb),
            &region_f,
            &masks,
            ring * k,
        ) else {
            return Trace::Inconclusive;
        };
        if now.joined {
            return Trace::Joined;
        }
        let fell = now.ink <= before.ink - rp.min_ink_drop.max(rp.min_ink_drop_share * before.ink);
        if fell {
            Trace::Gone
        } else {
            Trace::Broken
        }
    };

    let edge_support = SupportParams {
        min_presence: params.edge_min_presence_share,
        ..support
    };
    let mut edges: Vec<EdgeState> = Vec::new();
    let mut edge_id_of: HashMap<(usize, usize), String> = HashMap::new();
    let mut edge_intervals: HashMap<(usize, usize), Vec<Interval>> = HashMap::new();
    for (&(a, b), list) in &edge_obs {
        let seen: Vec<usize> = list.iter().map(|o| o.frame).collect();
        let (ta, tb) = (&tracks[a], &tracks[b]);
        // Pixel verdict for keyframe `f` against the last keyframe `s` that read the
        // edge: `Some(true)` the straight line is still drawn, `Some(false)` it is
        // gone, `None` when the corridor cannot tell (a routed connector leaves it).
        let rp = &params.region_probe;
        let last_read = |f: usize| seen.iter().rev().find(|&&s| s < f).copied();
        let verdict = |f: usize| -> Option<bool> {
            let (before, now) = edge_cover(ta, tb, last_read(f)?, f)?;
            if before < rp.min_line_cover {
                return None;
            }
            let ratio = now / before;
            if ratio >= rp.still_drawn_ratio {
                Some(true)
            } else if ratio <= rp.gone_ratio {
                Some(false)
            } else {
                None
            }
        };
        let ivs = intervals(
            &seen,
            // An edge is absent from a keyframe that read both of its ends (or
            // covers both places) without reading the edge only with evidence that
            // the connector is gone: readers often leave a connector out of one
            // reading. The corridor's pixels decide when they can; otherwise the
            // board's aligned ink must have changed since the edge was last read,
            // and (with pixels) the ink around the edge's ends must have fallen.
            |f| {
                if seen.contains(&f) {
                    return Visibility::Unknown;
                }
                let present = |t: &Track| {
                    t.obs.iter().any(|o| o.frame == f && o.bbox.is_some())
                        || track_covered(t, f) == Visibility::Visible
                };
                if !(present(ta) && present(tb)) {
                    return Visibility::Unknown;
                }
                // A corridor that emptied needs evidence about the connector
                // itself: its own traced stroke gone, or broken (a straight
                // connector erased while ink is added near it). A stroke still
                // joined, or a trace that cannot tell (the stroke never joined
                // the boxes in `s`, being dashed or occluded by a card read on
                // its route, or the region changed around it), keeps the
                // connector: an unread straight mark erased across the corridor
                // of a connector routed around it is not the connector. Only a
                // probe that traces nothing leaves the corridor to decide alone.
                let trace =
                    || last_read(f).map_or(Trace::Inconclusive, |s| routed_trace(a, b, s, f));
                let gone = match verdict(f) {
                    Some(true) => false,
                    Some(false) => matches!(trace(), Trace::Gone | Trace::Broken | Trace::Untraced),
                    None => {
                        let from = last_read(f).map_or(0, |s| s + 1);
                        let inked = (from..=f).any(|k| {
                            frames[k]
                                .ink_change
                                .is_some_and(|x| x >= params.ink_event_threshold)
                        });
                        // With pixels, the traced stroke must say the connector went
                        // (pixels that cannot tell keep it); without pixels, the board
                        // ink is all there is to go on.
                        inked && (probe.is_none() || trace() == Trace::Gone)
                    }
                };
                if gone {
                    Visibility::Visible
                } else {
                    Visibility::Unknown
                }
            },
            n,
            &edge_support,
            |_| false,
            |f| {
                list.iter()
                    .filter(|o| o.frame == f)
                    .map(|o| o.shares.share)
                    .fold(0.0, f64::max)
            },
            // Both ends read, or with their places in view: the keyframe could
            // have read the connector.
            |f| {
                let there = |t: &Track| {
                    t.obs.iter().any(|o| o.frame == f) || track_visible(t, f) == Visibility::Visible
                };
                there(ta) && there(tb)
            },
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
        let since = reversals(list, a, params.vote_min_share, params.min_support_weight)
            .last()
            .copied();
        let mut votes = DirectionVotes::default();
        for o in list.iter().filter(|o| since.is_none_or(|f| o.frame >= f)) {
            // A keyframe's evidence weighs its sharpness and zoom and the share of
            // its reads that listed the edge.
            let w = weight_of(o.frame) * o.shares.share;
            let orient = |v: EndVerdict| if o.src == a { v } else { v.flipped() };
            // The reader's direction weighs the share of its reads that agreed on it.
            votes
                .reader
                .add(orient(EndVerdict::Forward), o.shares.direction);
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
        // The label with the most label votes across the interval's keyframes (a
        // keyframe weighs the share of its reads that carried the label; ties go
        // to the label read last).
        let mut label_counts: BTreeMap<String, (String, f64, usize)> = BTreeMap::new();
        for o in in_iv.iter().filter(|o| !o.label.is_empty()) {
            let e =
                label_counts
                    .entry(normalize(&o.label))
                    .or_insert((o.label.clone(), 0.0, o.frame));
            e.1 += o.shares.label;
            e.2 = o.frame;
            e.0.clone_from(&o.label);
        }
        // A label needs support of its own: label votes of at least
        // `min_support_weight`, and at least `min_presence_share` of the edge's own
        // votes over the interval (a label read now and then on a connector read
        // throughout is not its label).
        let edge_weight: f64 = in_iv.iter().map(|o| o.shares.share).sum();
        let label = label_counts
            .into_values()
            .filter(|e| {
                e.1 >= params.min_support_weight - 1e-9
                    && e.1 >= params.min_presence_share * edge_weight - 1e-9
            })
            .max_by(|x, y| x.1.total_cmp(&y.1).then(x.2.cmp(&y.2)))
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

    // Stop before owner corroboration when a fallback is needed. Owner callbacks
    // may have side effects, and node IDs can change when an orphan becomes a
    // sticky, so replaying their first-pass answers would not be safe.
    if withdrawn.is_empty() {
        let orphaned: BTreeSet<usize> = fragment_ends
            .iter()
            .copied()
            .filter(|t| {
                node_id
                    .get(t)
                    .is_some_and(|id| !edges.iter().any(|e| &e.a == id || &e.b == id))
            })
            .collect();
        if !orphaned.is_empty() {
            return Err(orphaned);
        }
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
    // A node read once with the text of an established node, at an imprecise place,
    // is that node (see `echo_of_established`), as is a displaced echo of one (see
    // `displaced_echo`): owner targets follow it there.
    let owner_track = |ti: usize| -> usize {
        if tracks[ti].kind() != ObsList::Node {
            return ti;
        }
        if !echo_of_established(ti, &tracks[ti]) {
            return displaced_echo(ti, &tracks[ti])
                .filter(|u| node_id.contains_key(u) && tracks[*u].kind() == ObsList::Node)
                .unwrap_or(ti);
        }
        let seen = tracks[ti].frames();
        established
            .get(&normalize(&tracks[ti].text()))
            .and_then(|v| {
                v.iter()
                    .filter(|(u, fr)| {
                        *u != ti
                            && node_id.contains_key(u)
                            && fr.len() >= params.min_support_keyframes
                            && !fr.contains(&seen[0])
                    })
                    .max_by_key(|(u, fr)| (fr.len(), std::cmp::Reverse(*u)))
                    .map(|(u, _)| *u)
            })
            .unwrap_or(ti)
    };
    let owner_node = |ti: usize| node_target(owner_track(ti));
    let person_of = |text: &str| {
        params
            .participants
            .resolve(text)
            .map(|p| p.person_id.clone())
    };
    let mut rejected_owner_tags: Vec<RejectedOwnerTag> = Vec::new();
    let mut by_person: BTreeMap<String, (String, Vec<OwnerSighting>)> = BTreeMap::new();
    // Per keyframe: people named anywhere in it (reading or OCR), and node tracks
    // whose text OCR placed.
    let mut named: Vec<HashSet<String>> = vec![HashSet::new(); n];
    let mut ocr_placed: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (fi, f) in frames.iter().enumerate() {
        let geo = geometry_ok(&f.board);
        let tags = reading_tags(f, &params.participants);
        let tag_shares = reading_tag_shares(f, frame_votes[fi], &params.participants);
        let b = &f.board;
        for text in b
            .nodes
            .iter()
            .map(|x| x.text.as_str())
            .chain(b.stickies.iter().map(|x| x.text.as_str()))
            .chain(b.other_visible_text.iter().map(|x| x.text.as_str()))
            .chain(b.owner_tags.iter().map(|x| x.name_raw.as_str()))
            .chain(b.chrome_rejected.iter().map(|x| x.text.as_str()))
            .chain(f.ocr_anchors.iter().map(|x| x.text.as_str()))
        {
            if let Some(pid) = person_of(text) {
                named[fi].insert(pid);
            }
        }
        // Tags and nodes checked against OCR (displaced ones moved onto their text);
        // an unreliable reading keeps only what OCR placed.
        let people: Vec<Option<String>> = tags.iter().map(|t| person_of(&t.0)).collect();
        let placed = &geometries[fi];
        // Boxes of the tracked nodes in this keyframe, and of the supported edges seen
        // here (for tag geometry).
        let mut node_boxes: Vec<(AnchorKey, BBox)> = Vec::new();
        for (x, pl) in f.board.nodes.iter().zip(&placed.nodes) {
            if params.participants.resolve(&x.text).is_some() {
                continue;
            }
            let (Some(pl), Some(&raw)) = (pl, node_track.get(&(fi, x.local_id.clone()))) else {
                continue;
            };
            // A single misplaced reading of an established node stands for it.
            let ti = if node_id.contains_key(&raw) {
                raw
            } else {
                owner_track(raw)
            };
            if !node_id.contains_key(&ti) {
                continue;
            }
            if pl.ocr {
                ocr_placed[fi].insert(ti);
                ocr_placed[fi].insert(owner_track(ti));
            }
            node_boxes.push((AnchorKey::Node(ti), pl.bbox));
        }
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
                let (a, b) = (box_of(k.0)?, box_of(k.1)?);
                // Traced termini where the reading is geometry; otherwise the facing
                // border points of the placed boxes.
                let traced = o
                    .evidence
                    .as_ref()
                    .is_some_and(|ev| ev.pixel.src_end.is_some() && ev.pixel.dst_end.is_some());
                let segment = if traced && !placed.unreliable {
                    o.segment
                } else {
                    let (ac, bc) = (
                        ((a.x1 + a.x2) / 2.0, (a.y1 + a.y2) / 2.0),
                        ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0),
                    );
                    (exit_point(&a, bc), exit_point(&b, ac))
                };
                Some(EdgeGeom {
                    key: AnchorKey::Edge(*k),
                    a: (AnchorKey::Node(k.0), a),
                    b: (AnchorKey::Node(k.1), b),
                    segment,
                })
            })
            .collect();
        // Reader boxes that all coincide are no geometry, unless OCR placed what is
        // used (an unreliable reading keeps nothing else).
        let use_geometry = geo || placed.unreliable;
        // OCR can vouch for names only where the OCR check runs.
        let has_ocr = params.owner_geometry.enabled && !f.ocr_anchors.is_empty();
        let pad =
            |g: &BBox| owner_geometry::tag_box(g, placed.line, params.owner_geometry.tag_pad_lines);
        // OCR-first tags. A reader tag OCR read sits on its OCR text; in a keyframe
        // where OCR read a person's name, that person's other reader tags are
        // misplaced or duplicated and dropped; every other OCR name span is a tag the
        // reader missed. A span inside an element that mentions the name is no tag.
        let ocr_people: HashSet<&str> = placed
            .name_spans
            .iter()
            .filter(|s| !s.mention)
            .map(|s| s.person.as_str())
            .collect();
        let span_of = |si: &usize| Some(*si).filter(|si| !placed.name_spans[*si].mention);
        // (name, reader's near, geometry box, box on OCR text, who read the name)
        let mut all: Vec<(String, String, Option<BBox>, bool, NameRead)> = Vec::new();
        for (i, ((name, near, _), pl)) in tags.into_iter().zip(&placed.tags).enumerate() {
            match placed.tag_spans[i].as_ref().and_then(span_of) {
                Some(si) => all.push((
                    name,
                    near,
                    Some(pad(&placed.name_spans[si].glyph)),
                    true,
                    NameRead::Ocr,
                )),
                None if people[i].as_deref().is_some_and(|p| ocr_people.contains(p)) => {}
                // A reader tag no OCR span confirmed counts only when enough of the
                // reads listed it.
                None if tag_shares.get(i).copied().unwrap_or(1.0)
                    < params.owner_reader_min_share - 1e-9 => {}
                None => all.push((
                    name,
                    near,
                    pl.map(|p| p.bbox),
                    pl.is_some_and(|p| p.ocr),
                    if has_ocr {
                        NameRead::Reader
                    } else {
                        NameRead::Unchecked
                    },
                )),
            }
        }
        let claimed: HashSet<usize> = placed.tag_spans.iter().flatten().copied().collect();
        for (si, ns) in placed.name_spans.iter().enumerate() {
            if !ns.mention && !claimed.contains(&si) {
                all.push((
                    ns.text.clone(),
                    String::new(),
                    Some(pad(&ns.glyph)),
                    true,
                    NameRead::Ocr,
                ));
            }
        }
        let place_of = |b: &BBox| {
            positioned(fi).then(|| {
                let c = regs[fi]
                    .to_reference
                    .apply(((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0));
                let k = regs[fi].to_reference.scale.abs();
                TagPlace {
                    cluster: regs[fi].cluster,
                    x: c.0,
                    y: c.1,
                    w: b.width() * k,
                    h: b.height() * k,
                }
            })
        };
        for (tag_index, (name, near, bx, box_ocr, name_read)) in all.into_iter().enumerate() {
            let Some(person) = params.participants.resolve(&name) else {
                rejected_owner_tags.push(RejectedOwnerTag {
                    keyframe_id: f.keyframe_id.clone(),
                    name_raw: name,
                });
                continue;
            };
            // (target, anchor, the tag and every target node placed on OCR text,
            // other targets the geometry fits)
            type Found = (OwnerTarget, AnchorKind, bool, Vec<OwnerTarget>);
            let mut targets: Vec<Found> = Vec::new();
            if let (true, Some(bx)) = (use_geometry, bx) {
                let on_ocr =
                    |ks: &[usize]| box_ocr && ks.iter().all(|t| ocr_placed[fi].contains(t));
                let amb = edge_end_ambiguities(&bx, &frame_edges, &params.owner_anchor);
                match anchor_tag(&bx, &node_boxes, &frame_edges, &params.owner_anchor) {
                    Some(Anchored::Node(AnchorKey::Node(x))) => {
                        let alts: Vec<OwnerTarget> = amb
                            .iter()
                            .filter(|(_, end)| *end == AnchorKey::Node(x))
                            .filter_map(|(e, _)| match e {
                                AnchorKey::Edge(k) => edge_target(*k),
                                AnchorKey::Node(_) => None,
                            })
                            .collect();
                        targets.extend(
                            owner_node(x)
                                .map(|t| (t, AnchorKind::GeometryNode, on_ocr(&[x]), alts)),
                        );
                    }
                    Some(Anchored::Bridge(AnchorKey::Node(x), AnchorKey::Node(y))) => {
                        for z in [x, y] {
                            targets.extend(owner_node(z).map(|t| {
                                (t, AnchorKind::GeometryBridge, on_ocr(&[z]), Vec::new())
                            }));
                        }
                    }
                    Some(Anchored::Edge(AnchorKey::Edge(k))) => {
                        let alts: Vec<OwnerTarget> = amb
                            .iter()
                            .filter(|(e, _)| *e == AnchorKey::Edge(k))
                            .filter_map(|(_, end)| match end {
                                AnchorKey::Node(x) => owner_node(*x),
                                AnchorKey::Edge(_) => None,
                            })
                            .collect();
                        targets.extend(
                            edge_target(k)
                                .map(|t| (t, AnchorKind::GeometryEdge, on_ocr(&[k.0, k.1]), alts)),
                        );
                    }
                    _ => {}
                }
            }
            if targets.is_empty() && !near.is_empty() {
                targets.extend(
                    node_track
                        .get(&(fi, near.clone()))
                        .and_then(|&ti| owner_node(ti))
                        .map(|t| (t, AnchorKind::Near, false, Vec::new())),
                );
            }
            let entry = by_person
                .entry(person.person_id.clone())
                .or_insert_with(|| (person.display_name.clone(), Vec::new()));
            let targets: Vec<(Option<OwnerTarget>, AnchorKind, bool, Vec<OwnerTarget>)> =
                if targets.is_empty() {
                    vec![(None, AnchorKind::Untargeted, false, Vec::new())]
                } else {
                    targets
                        .into_iter()
                        .map(|(t, a, l, alts)| (Some(t), a, l, alts))
                        .collect()
                };
            for (t, anchor, located, alternates) in targets {
                let sighting = OwnerSighting {
                    keyframe_id: f.keyframe_id.clone(),
                    t_start_s: f.t_start_s,
                    t_end_s: f.t_end_s,
                    name_raw: name.clone(),
                    target: t,
                    anchor,
                    tag: tag_index as u32,
                    ocr_located: located,
                    name_read,
                    alternates,
                    place: bx.as_ref().and_then(&place_of),
                    physical: None,
                };
                // Several tags of one person in a keyframe are kept when their targets
                // differ (multi-target owners); for one target the stronger evidence,
                // unless both are placed: two placed tags are two physical tags.
                let strength = |s: &OwnerSighting| (s.ocr_located, s.name_read != NameRead::Reader);
                match entry.1.iter().position(|s| {
                    s.keyframe_id == f.keyframe_id
                        && s.target == sighting.target
                        && !(s.place.is_some() && sighting.place.is_some() && s.tag != sighting.tag)
                }) {
                    None => entry.1.push(sighting),
                    Some(j) if strength(&sighting) > strength(&entry.1[j]) => entry.1[j] = sighting,
                    Some(_) => {}
                }
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
        final_hold_min_s: params.owner_final_hold_s,
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
    // Tracks that stand for another one (single imprecise readings of it).
    let mut echoes: HashMap<usize, Vec<usize>> = HashMap::new();
    for ti in 0..tracks.len() {
        let u = owner_track(ti);
        if u != ti {
            echoes.entry(u).or_default().push(ti);
        }
    }
    let seen = |ti: usize, kf: &str| {
        frame_of_kf.get(kf).is_some_and(|fi| {
            std::iter::once(&ti)
                .chain(echoes.get(&ti).into_iter().flatten())
                .any(|&t| tracks[t].obs.iter().any(|o| o.frame == *fi))
        })
    };
    // The tag's registered position lies inside the keyframe's canvas. Without a
    // known tag position the target alone decides; with one, a keyframe that cannot
    // map it (not registered, another registration cluster, no canvas) is not
    // evidence either way.
    let in_view = |fi: usize, place: Option<&TagPlace>| -> bool {
        let Some(p) = place else { return true };
        if !positioned(fi) || regs[fi].cluster != p.cluster {
            return false;
        }
        let (Some(c), Some(inv)) = (canvases[fi], regs[fi].to_reference.inverse()) else {
            return false;
        };
        let q = inv.apply((p.x, p.y));
        q.0 >= 0.0 && q.1 >= 0.0 && q.0 <= c.width && q.1 <= c.height
    };
    let target_visible = |kf: &str, t: &OwnerTarget, place: Option<&TagPlace>| {
        let read = match t {
            OwnerTarget::Node { node_id, .. } => track_of_id
                .get(node_id.as_str())
                .is_some_and(|&ti| seen(ti, kf)),
            OwnerTarget::Edge { edge_id, .. } => edge_of_id
                .get(edge_id.as_str())
                .is_some_and(|&(a, b)| seen(a, kf) && seen(b, kf)),
        };
        read && frame_of_kf.get(kf).is_some_and(|&fi| in_view(fi, place))
    };
    let target_tracks = |t: &OwnerTarget| -> Vec<usize> {
        match t {
            OwnerTarget::Node { node_id, .. } => track_of_id
                .get(node_id.as_str())
                .copied()
                .into_iter()
                .collect(),
            OwnerTarget::Edge { edge_id, .. } => edge_of_id
                .get(edge_id.as_str())
                .map(|&(a, b)| vec![a, b])
                .unwrap_or_default(),
        }
    };
    let mut owner_assignments: Vec<OwnerAssignment> = Vec::new();
    for (pid, (name, mut sightings)) in by_person {
        collapse_edge_pairs(&mut sightings, &alternation_edges);
        apply_alternation(
            &mut sightings,
            &alternation_edges,
            params.alternation_min_alternations,
        );
        consolidate_tags(&mut sightings, params.owner_tag_reach_share);
        // The first keyframe in [after, before) that read the target, OCR placed it,
        // has the tag's last position in view, and names the person nowhere.
        let absent =
            |t: &OwnerTarget, place: Option<&TagPlace>, after: f64, before: f64| -> Option<f64> {
                let ts = target_tracks(t);
                if ts.is_empty() {
                    return None;
                }
                frames
                    .iter()
                    .enumerate()
                    .find(|(fi, f)| {
                        f.t_start_s >= after - 1e-9
                            && f.t_start_s < before - 1e-9
                            && !named[*fi].contains(&pid)
                            && in_view(*fi, place)
                            && ts.iter().all(|&ti| {
                                seen(ti, &f.keyframe_id) && ocr_placed[*fi].contains(&ti)
                            })
                    })
                    .map(|(_, f)| f.t_start_s)
            };
        owner_assignments.extend(assign_with(
            &pid,
            &name,
            &sightings,
            timeline_end_s,
            &owner_params,
            corroborator,
            &target_visible,
            &absent,
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
    // Came into view rather than drawn: first seen right after a registered pan or
    // zoom (two positioned keyframes of one cluster whose transforms differ), at a
    // place the previous view did not show empty.
    let view_changed_into = |f: usize| {
        f > 0
            && positioned(f - 1)
            && positioned(f)
            && regs[f - 1].cluster == regs[f].cluster
            && view_moved(f - 1, f)
    };
    // The previous keyframe showed the place empty. With pixels: the place, mapped into
    // the previous keyframe through a trusted registration, held almost no ink while
    // the element's own reading holds ink. Without them: coverage evidence.
    let shown_empty_before = |t: &Track, f: usize| -> bool {
        let p = f - 1;
        let pixels = || -> Option<bool> {
            let probe = probe?;
            let rp = &params.region_probe;
            let r = &regs[p];
            let trusted = r.mode == RegistrationMode::Reference
                || (r.mode == RegistrationMode::Registered
                    && r.inliers >= rp.min_registration_inliers);
            if !trusted || track_visible(t, p) != Visibility::Visible {
                return Some(false);
            }
            let b = t.bbox_in(r.cluster)?;
            let inv = r.to_reference.inverse()?;
            let own = t
                .obs
                .iter()
                .find(|o| o.frame == f)
                .and_then(|o| o.raw_bbox)?;
            let now = probe.ink_share(&frames[f].keyframe_id, &own)?;
            let before = probe.ink_share(&frames[p].keyframe_id, &map_bbox(&inv, &b))?;
            Some(now >= rp.min_ink_share && before <= rp.gone_ratio * now)
        };
        pixels().unwrap_or_else(|| track_covered(t, p) == Visibility::Visible)
    };
    let revealed =
        |ti: usize, f: usize| view_changed_into(f) && !shown_empty_before(&tracks[ti], f);
    // An edge first read after a view change came into view when an end was not in
    // the previous view, or when, both ends read there, a traced stroke already
    // joined them there (straight-corridor ink proves no connection, and a probe
    // that does not trace leaves the edge added).
    let revealed_edge = |key: (usize, usize), f: usize| {
        if !view_changed_into(f) {
            return false;
        }
        let before = |ti: usize| {
            tracks[ti]
                .obs
                .iter()
                .find(|o| o.frame == f - 1)
                .and_then(|o| o.raw_bbox)
        };
        if ![key.0, key.1]
            .iter()
            .all(|&ti| before(ti).is_some() || shown_empty_before(&tracks[ti], f))
        {
            return true;
        }
        let rp = &params.region_probe;
        match (probe, before(key.0), before(key.1)) {
            (Some(probe), Some(ba), Some(bb)) if probe.traces() => {
                let p = f - 1;
                let ring = (rp.corridor_half_width_share * ba.height().min(bb.height())).max(3.0);
                let margin = rp.stroke_margin_share * ba.height().max(bb.height());
                let region = BBox::new(
                    ba.x1.min(bb.x1) - margin,
                    ba.y1.min(bb.y1) - margin,
                    ba.x2.max(bb.x2) + margin,
                    ba.y2.max(bb.y2) + margin,
                );
                let masks: Vec<BBox> = others(key.0, key.1).filter_map(|x| raw_at(x, p)).collect();
                probe
                    .stroke_between(&frames[p].keyframe_id, &ba, &bb, &region, &masks, ring)
                    .is_some_and(|t| t.joined)
            }
            _ => false,
        }
    };
    let offer_added = |gate: &mut EventGate, e: BoardEvent, is_revealed: bool| {
        if !e.baseline && is_revealed {
            gate.suppress(e, SuppressReason::RevealedByView);
        } else {
            gate.offer(e);
        }
    };
    let event = |kind: EventKind, f: usize, subject: &str, detail: String| BoardEvent {
        event_id: String::new(),
        kind,
        t_s: frames[f].t_start_s,
        keyframe_id: frames[f].keyframe_id.clone(),
        subject: subject.to_string(),
        detail,
        ink_change: frames[f].ink_change,
        baseline: f == 0,
        owner_target: None,
        owner_from: None,
    };
    for &ti in &order {
        if lifted.contains(&ti) {
            continue;
        }
        let ivs = &track_intervals[ti];
        if let Some(id) = node_id.get(&ti) {
            let text = text_of(ti);
            for iv in ivs {
                offer_added(
                    &mut gate,
                    event(EventKind::NodeAdded, iv.first, id, text.clone()),
                    revealed(ti, iv.first),
                );
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
                offer_added(
                    &mut gate,
                    event(EventKind::StickyAdded, iv.first, id, text.clone()),
                    revealed(ti, iv.first),
                );
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
            offer_added(
                &mut gate,
                event(EventKind::EdgeAdded, iv.first, &e.id, detail.clone()),
                revealed_edge(key, iv.first),
            );
            if let Some(r) = iv.removed_at {
                gate.offer(event(EventKind::EdgeRemoved, r, &e.id, detail.clone()));
            }
        }
        // Reversal: per-keyframe decisions that flip for two consecutive evidence frames.
        let list = edge_obs.get(&key).map(Vec::as_slice).unwrap_or(&[]);
        for f in reversals(
            list,
            key.0,
            params.vote_min_share,
            params.min_support_weight,
        ) {
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
        gate.offer(BoardEvent {
            owner_target: Some(a.target.clone()),
            owner_from: a.moved_from.clone(),
            ..event(
                kind,
                f,
                &a.person_id,
                format!("{} -> {}", a.display_name, a.target.texts().join(" - ")),
            )
        });
    }
    let (events, suppressed_events) = gate.finish();
    // Lifted tracks leave the node and sticky lists.
    let heading_ids: HashSet<String> = lifted
        .iter()
        .filter_map(|t| node_id.get(t).or_else(|| sticky_id.get(t)).cloned())
        .collect();
    nodes.retain(|n| !heading_ids.contains(&n.id));
    stickies.retain(|s| !heading_ids.contains(&s.id));
    let state = BoardStateItem {
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
    };
    Ok(state)
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

/// Keyframes where an edge's direction was reversed: a decisive direction other
/// than the established one that the next deciding keyframe repeats, the two
/// together carried by at least `min_weight` of vote shares (two borderline
/// readings do not reverse an edge).
fn reversals(list: &[EdgeObs], a: usize, min_share: f64, min_weight: f64) -> Vec<usize> {
    let mut per_frame: Vec<(usize, EdgeDirection, f64)> = Vec::new();
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
            per_frame.push((o.frame, d, o.shares.share));
        }
    }
    let mut out = Vec::new();
    let mut established = per_frame.first().map(|p| p.1);
    for w in per_frame.windows(2) {
        if Some(w[0].1) != established && w[0].1 == w[1].1 && w[0].2 + w[1].2 >= min_weight - 1e-9 {
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
    fn stability_defaults_match_the_run_config() {
        let p = ConsolidationParams::default();
        let c = glassrip_core::config::Config::default().board_state;
        assert_eq!(p.min_support_weight, c.min_support_weight);
        assert_eq!(p.min_presence_share, c.min_presence_share);
        assert_eq!(p.edge_min_presence_share, c.edge_min_presence_share);
        assert_eq!(p.single_min_share, c.single_min_share);
        assert_eq!(p.owner_reader_min_share, c.owner_reader_min_share);
        assert_eq!(
            p.presence_window_keyframes,
            c.presence_window_keyframes as usize
        );
        assert_eq!(p.echo_max_weight_share, c.echo_max_weight_share);
        assert_eq!(p.min_support_keyframes, c.min_support_keyframes as usize);
    }

    #[test]
    fn sticky_kinds() {
        assert_eq!(sticky_kind("Do we need a cache?"), StickyKind::Question);
        assert_eq!(sticky_kind("Beta milestone: March"), StickyKind::Milestone);
        assert_eq!(sticky_kind("Idea: batch the writes"), StickyKind::Idea);
        assert_eq!(sticky_kind("Clone the landing page"), StickyKind::Note);
    }
}
