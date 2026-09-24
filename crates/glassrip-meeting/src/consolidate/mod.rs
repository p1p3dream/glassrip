//! Board-state consolidation (spec 6.11, board state).
//!
//! Pipeline over the board keyframes in time order:
//!
//! 1. Register keyframes into canvas clusters through text anchors ([`crate::register`]).
//! 2. Track text elements across keyframes: fuzzy text ([`crate::difflib`] at 0.85)
//!    plus, within a registered cluster, position (same text elsewhere is another
//!    element; different text at the same node place is a label variant). Nodes,
//!    stickies, other text and edge labels share one pool, so an element read in
//!    different lists in different keyframes is one track whose kind is the majority
//!    list. Illegible or elided texts are not tracked.
//! 3. Lifetimes and support ([`tracks::intervals`]); the final state is what is alive
//!    in the last stable board window.
//! 4. Edges are keyed by their two node tracks; an endpoint that is not a supported
//!    node drops the edge (nodes are never created from endpoints). Directions come
//!    from the weighted vote over `glassrip.edge_direction` evidence.
//! 5. Owner tags become timed assignments ([`owners`]).
//! 6. Events are computed from the state changes and gated on ink ([`events`]).

pub mod events;
pub mod owners;
pub mod tracks;

use std::collections::{BTreeMap, HashMap};

use glassrip_vision::board::{EdgeStyle, ElementList, RejectReason, StickyColor, ValidatedBoard};
use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::artifacts::{CanvasDims, EdgeDirectionItem, EdgeEvidence};
use crate::direction::{frame_weight, DirectionBasis, DirectionVotes, EdgeDirection, EndVerdict};
use crate::pixel_direction::{dist_to_bbox, exit_point};
use crate::register::{
    map_bbox, register, Anchors, FrameRegistration, RegistrationMode, RegistrationParams,
};
use crate::text::{is_unreliable, normalize, AliasTable};

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
    /// (it is extended backward otherwise).
    pub min_final_keyframes: usize,
    /// Reading confidence (element and board) a single sighting needs, together with
    /// the pixel check, to be kept.
    pub single_sighting_min_conf: f64,
    /// Aligned ink change needed for a content event.
    pub ink_event_threshold: f64,
    /// Share of the decisive weight a direction needs within a channel.
    pub vote_min_share: f64,
    /// Consecutive consistent keyframes to open or move an owner assignment.
    pub owner_confirm_keyframes: usize,
    /// A tag anchors to the nearest node within this many tag sizes.
    pub owner_node_anchor_share: f64,
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
            min_support_keyframes: 2,
            min_support_density: 0.10,
            removal_absent_keyframes: 2,
            final_window_s: 120.0,
            min_final_keyframes: 3,
            single_sighting_min_conf: 0.8,
            ink_event_threshold: 0.05,
            vote_min_share: 0.6,
            owner_confirm_keyframes: 2,
            owner_node_anchor_share: 1.5,
            alternation_min_alternations: 3,
            backfill_untargeted_owners: false,
        }
    }
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

fn point_segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (qx, qy) = (a.0 + t * dx, a.1 + t * dy);
    ((p.0 - qx).powi(2) + (p.1 - qy).powi(2)).sqrt()
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

    // 1. Registration.
    let anchors: Vec<Anchors> = frames
        .iter()
        .map(|f| {
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
                    .chain(f.ocr_anchors.iter().map(|x| (x.text.as_str(), &x.bbox))),
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
    };

    // 2. Element tracks.
    let mut tracks: Vec<Track> = Vec::new();
    let mut node_track: HashMap<(usize, String), usize> = HashMap::new();
    for (fi, f) in frames.iter().enumerate() {
        let to_ref = |b: &BBox| positioned(fi).then(|| map_bbox(&regs[fi].to_reference, b));
        let cluster = regs[fi].cluster;
        let mut obs: Vec<Obs> = Vec::new();
        for nd in &f.board.nodes {
            // A node whose text resolves to a participant (fuzzy, unlike the
            // validator's exact check) is an owner tag read in the wrong list.
            if is_unreliable(&nd.text) || params.participants.resolve(&nd.text).is_some() {
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
                cluster,
                local_id: Some(nd.local_id.clone()),
                color: None,
                single_ok: pixel_ok && conf_ok && reader_ok,
            });
        }
        for s in &f.board.stickies {
            if is_unreliable(&s.text) {
                continue;
            }
            obs.push(Obs {
                frame: fi,
                lists: vec![ObsList::Sticky],
                text: s.text.clone(),
                bbox: to_ref(&s.bbox),
                cluster,
                local_id: None,
                color: Some(s.color),
                single_ok: false,
            });
        }
        for t in &f.board.other_visible_text {
            if is_unreliable(&t.text) {
                continue;
            }
            obs.push(Obs {
                frame: fi,
                lists: vec![ObsList::Other],
                text: t.text.clone(),
                bbox: to_ref(&t.bbox),
                cluster,
                local_id: None,
                color: None,
                single_ok: false,
            });
        }
        for e in &f.board.edges {
            let l = e.label.trim();
            if l.is_empty() || is_unreliable(l) {
                continue;
            }
            let nl = normalize(l);
            match obs.iter_mut().find(|o| normalize(&o.text) == nl) {
                Some(o) => {
                    if !o.lists.contains(&ObsList::EdgeLabel) {
                        o.lists.push(ObsList::EdgeLabel);
                    }
                }
                None => obs.push(Obs {
                    frame: fi,
                    lists: vec![ObsList::EdgeLabel],
                    text: l.to_string(),
                    bbox: None,
                    cluster,
                    local_id: None,
                    color: None,
                    single_ok: false,
                }),
            }
        }
        let locals: Vec<Option<String>> = obs.iter().map(|o| o.local_id.clone()).collect();
        let assigned = assign_frame(&mut tracks, obs, &match_params);
        for (ti, local) in assigned.into_iter().zip(locals) {
            if let Some(l) = local {
                node_track.insert((fi, l), ti);
            }
        }
    }

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
                        track_visible(t, f)
                    }
                },
                n,
                &support,
                |f| t.obs.iter().any(|o| o.frame == f && o.single_ok),
            )
        })
        .collect();

    // Final window: the last contiguous run of board keyframes, trimmed to
    // `final_window_s` before its end.
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
    let alive_in_final = |ivs: &[Interval]| -> bool {
        let Some(w) = &window else {
            return false;
        };
        ivs.iter().any(|iv| {
            iv.removed_at.is_none()
                && frames[iv.first].t_start_s <= w.end_s
                && frames[iv.last].t_end_s >= w.start_s
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
                    in_final: alive_in_final(ivs),
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
                stickies.push(StickyState {
                    id,
                    kind: sticky_kind(&text),
                    text,
                    color: t.color(),
                    lifetimes,
                    in_final: alive_in_final(ivs),
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
                label: e.label.trim().to_string(),
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
            |f| {
                if seen.contains(&f) {
                    Visibility::Unknown
                } else if track_visible(ta, f) == Visibility::Visible
                    && track_visible(tb, f) == Visibility::Visible
                {
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
        let mut votes = DirectionVotes::default();
        for o in &in_iv {
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
            in_final: alive_in_final(&ivs) && final_node(a) && final_node(b),
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
        let geo_nodes: Vec<&glassrip_vision::board::BoardNode> = f
            .board
            .nodes
            .iter()
            .filter(|x| params.participants.resolve(&x.text).is_none())
            .collect();
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
        let frame_edges: Vec<((usize, usize), Segment)> = edge_obs
            .iter()
            .filter(|(k, _)| edge_id_of.contains_key(k))
            .filter_map(|(k, l)| l.iter().find(|o| o.frame == fi).map(|o| (*k, o.segment)))
            .collect();
        for (name, near, tb) in tags {
            let Some(person) = params.participants.resolve(&name) else {
                rejected_owner_tags.push(RejectedOwnerTag {
                    keyframe_id: f.keyframe_id.clone(),
                    name_raw: name,
                });
                continue;
            };
            let c = ((tb.x1 + tb.x2) / 2.0, (tb.y1 + tb.y2) / 2.0);
            let size = tb.width().max(tb.height()).max(1.0);
            let mut target: Option<(OwnerTarget, AnchorKind)> = None;
            if geo {
                let overlaps_node = geo_nodes.iter().any(|x| x.bbox.iou(&tb) > 0.0);
                let on_edge = frame_edges
                    .iter()
                    .map(|(k, (p, q))| (*k, point_segment_distance(c, *p, *q)))
                    .filter(|(_, d)| *d <= size / 2.0)
                    .min_by(|x, y| x.1.total_cmp(&y.1));
                if let (false, Some((k, _))) = (overlaps_node, on_edge) {
                    target = edge_target(k).map(|t| (t, AnchorKind::GeometryEdge));
                }
                if target.is_none() {
                    let mut d: Vec<(f64, &str)> = geo_nodes
                        .iter()
                        .map(|x| (dist_to_bbox(c, &x.bbox), x.local_id.as_str()))
                        .collect();
                    d.sort_by(|x, y| x.0.total_cmp(&y.0));
                    let unique = d.len() == 1 || (d.len() > 1 && d[1].0 - d[0].0 > 1.0);
                    if let Some(&(dist, local)) = d.first() {
                        if unique && dist <= params.owner_node_anchor_share * size {
                            target = node_track
                                .get(&(fi, local.to_string()))
                                .and_then(|&ti| node_target(ti))
                                .map(|t| (t, AnchorKind::GeometryNode));
                        }
                    }
                }
            }
            if target.is_none() && !near.is_empty() {
                target = node_track
                    .get(&(fi, near.clone()))
                    .and_then(|&ti| node_target(ti))
                    .map(|t| (t, AnchorKind::Near));
            }
            let entry = by_person
                .entry(person.person_id.clone())
                .or_insert_with(|| (person.display_name.clone(), Vec::new()));
            let (t, anchor) = match target {
                Some((t, a)) => (Some(t), a),
                None => (None, AnchorKind::Untargeted),
            };
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
                name_raw: name,
                target: t,
                anchor,
            });
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
        ));
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
        let mut per_frame: Vec<(usize, EdgeDirection)> = Vec::new();
        for o in edge_obs.get(&key).map(Vec::as_slice).unwrap_or(&[]) {
            let Some(ev) = &o.evidence else { continue };
            let orient = |v: EndVerdict| if o.src == key.0 { v } else { v.flipped() };
            let mut v = DirectionVotes::default();
            let (p, m) = ev.verdicts();
            v.pixel.add(orient(p), 1.0);
            if let Some(m) = m {
                v.vlm.add(orient(m), 1.0);
            }
            let (d, _) = v.decide(params.vote_min_share);
            if matches!(d, EdgeDirection::AToB | EdgeDirection::BToA) {
                per_frame.push((o.frame, d));
            }
        }
        let mut established = per_frame.first().map(|p| p.1);
        for w in per_frame.windows(2) {
            if Some(w[0].1) != established && w[0].1 == w[1].1 {
                gate.offer(event(
                    EventKind::EdgeReversed,
                    w[0].0,
                    &e.id,
                    detail.clone(),
                ));
                established = Some(w[0].1);
            }
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

    BoardStateItem {
        board_id: board_id.to_string(),
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
