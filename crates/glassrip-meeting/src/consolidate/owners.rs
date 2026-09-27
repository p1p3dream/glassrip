//! Owner tags as timed assignments.
//!
//! Sightings start from OCR: every OCR span on the canvas that names a participant
//! (exactly or by a fuzzy spelling) is a tag, whether or not the reader emitted one,
//! and a reader tag that OCR read takes the OCR text's position. A reader tag OCR did
//! not read is corroboration only ([`NameRead::Reader`]): it sustains an open target
//! and extends a pending run, but a target never opens on such sightings alone. In a
//! keyframe where OCR read the person's name, reader tags of that person that no span
//! located are dropped (the reader misplaced or duplicated them).
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
//! A person's OCR-read tags at one registered place are one physical tag, and every
//! keyframe of it takes the target set most of them anchored to ([`consolidate_tags`]);
//! consecutive keyframes confirm an opening only when they show the same physical tag.
//! A tag without a known place whose geometry fits both an edge and one of its end
//! nodes (on the connector next to the node) carries the other as an alternate: when
//! an alternate is already open (or else pending), the sighting counts for it. The
//! incumbent wins ambiguous geometry, so an edge owner does not jump to the adjacent
//! node on one tilted view.
//!
//! Per person, keyframes with at least one target are replayed in time order
//! ([`assign`]): a target opens after `confirm_keyframes` consecutive such keyframes
//! show it (`confirm_keyframes` keyframes between two of them that showed the
//! target, and the tag's place, with the person named nowhere break the run, as
//! they would close an open target: [`Absent`]), or after one that a [`Corroborator`] confirms (for example a transcript
//! cue), or after one strong sighting that no later keyframe can confirm: in the last
//! board keyframe, held for at least `final_hold_min_s`, anchored by geometry with
//! the tag and its target both placed on OCR text ([`OpenReason::FinalHold`]). A
//! single sighting anywhere else never opens. Every opening needs at
//! least one sighting in its run whose name was not read by the reader alone. An open
//! target closes after `confirm_keyframes` consecutive such keyframes lack it where
//! it is in view: the target was read, and the tag's last registered position lies
//! inside the keyframe's canvas (a pan that cuts the tag off is not absence). An opening at the moment
//! another target of the same person closes is a move; so is an opening while another
//! open target was read, since its last sighting, in a keyframe where the person's
//! name appears nowhere ([`assign_with`]): that target closes there. Keyframes where
//! the person appears without a target are neutral. Backfill over earlier untargeted
//! sightings is off by default; when enabled it is recorded separately
//! (`backfill_from_s`) and never counts toward opening.

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
    /// The same physical tag (same registered place) anchored there in most of its
    /// keyframes ([`consolidate_tags`]).
    Registered,
}

/// Who read the name on a tag.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NameRead {
    /// OCR read the name at the tag (the reader may or may not have tagged it).
    Ocr,
    /// Only the reader: the keyframe has OCR, but no OCR span names the person there.
    Reader,
    /// The keyframe has no OCR to check against (older artifacts too).
    #[default]
    Unchecked,
}

/// A tag's center in its registration cluster's reference frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TagPlace {
    /// Registration cluster.
    pub cluster: usize,
    /// Reference x.
    pub x: f64,
    /// Reference y.
    pub y: f64,
    /// Tag box width, in reference units.
    pub w: f64,
    /// Tag box height, in reference units.
    pub h: f64,
}

impl TagPlace {
    /// The tag box's larger side.
    pub fn size(&self) -> f64 {
        self.w.max(self.h)
    }
}

/// How an assignment was opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OpenReason {
    /// Consecutive consistent keyframes.
    ConsistentKeyframes,
    /// One keyframe plus corroboration.
    Corroborated,
    /// One OCR-placed geometric sighting in the last board keyframe, held long.
    FinalHold,
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
    /// The tag and its target were both placed on OCR text in this keyframe (since
    /// board_state 1.2.0).
    #[serde(default)]
    pub ocr_located: bool,
    /// Who read the name (since board_state 1.3.0).
    #[serde(default)]
    pub name_read: NameRead,
    /// Other targets the tag's geometry fits as well (an edge and its adjacent end
    /// node); an open or pending one takes the sighting (since board_state 1.3.0).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternates: Vec<OwnerTarget>,
    /// Registered tag position (in-memory only).
    #[serde(skip)]
    #[schemars(skip)]
    pub place: Option<TagPlace>,
    /// The physical tag this sighting belongs to, per person ([`consolidate_tags`];
    /// in-memory only).
    #[serde(skip)]
    #[schemars(skip)]
    pub physical: Option<usize>,
}

impl OwnerSighting {
    /// Evidence rank when two sightings of a keyframe land on one target.
    fn strength(&self) -> (bool, bool) {
        (self.ocr_located, self.name_read != NameRead::Reader)
    }
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
    /// Shortest last board keyframe, in seconds, whose single strong sighting opens
    /// ([`OpenReason::FinalHold`]); infinite disables it.
    pub final_hold_min_s: f64,
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
            if s.target.is_some() {
                k.targets.push(s);
            } else {
                k.untargeted.push(s);
            }
        }
    }
    out
}

/// A keyframe's targeted sightings with their effective targets: an alternate that is
/// open (else pending) takes the sighting; one sighting per target, the strongest,
/// with the physical tags of all of them.
fn resolve_targets(
    raw: &[&OwnerSighting],
    open: &[&OwnerTarget],
    pending: &[&OwnerTarget],
) -> Vec<(OwnerSighting, Vec<Option<usize>>)> {
    let mut out: Vec<(OwnerSighting, Vec<Option<usize>>)> = Vec::new();
    for s in raw {
        let Some(primary) = &s.target else { continue };
        // A placed tag is settled by its physical tag ([`consolidate_tags`]); at a new
        // place it is a new tag, which may be a move.
        let alternates: &[OwnerTarget] = if s.physical.is_none() {
            &s.alternates
        } else {
            &[]
        };
        let cands = || std::iter::once(primary).chain(alternates);
        let chosen = cands()
            .find(|t| open.contains(t))
            .or_else(|| cands().find(|t| pending.contains(t)))
            .unwrap_or(primary)
            .clone();
        let mut e = (*s).clone();
        e.target = Some(chosen);
        match out.iter().position(|x| x.0.target == e.target) {
            None => {
                let p = vec![e.physical];
                out.push((e, p));
            }
            Some(i) => {
                out[i].1.push(e.physical);
                if e.strength() > out[i].0.strength() {
                    out[i].0 = e;
                }
            }
        }
    }
    out
}

/// The longest stretch of consecutive run keyframes that show one physical tag, and
/// the index where it starts. A keyframe shows tag `p` when one of its sightings of the
/// target belongs to `p`, or has no known place (neutral: it matches any tag). Other
/// physical tags are the same target read at another place (a misregistered view, or
/// another tag): they break the stretch.
fn run_support(run: &[RunStep]) -> (usize, usize) {
    let mut ids: Vec<Option<usize>> = vec![None];
    for r in run {
        for p in r.physicals.iter().flatten() {
            if !ids.contains(&Some(*p)) {
                ids.push(Some(*p));
            }
        }
    }
    let mut best = (0, 0);
    for id in ids {
        let shows = |r: &RunStep| r.physicals.contains(&None) || r.physicals.contains(&id);
        let mut start = 0;
        for (i, r) in run.iter().enumerate() {
            if !shows(r) {
                start = i + 1;
                continue;
            }
            let len = i + 1 - start;
            if len > best.0 || (len == best.0 && start < best.1) {
                best = (len, start);
            }
        }
    }
    best
}

/// One keyframe of a pending run.
struct RunStep {
    keyframe_id: String,
    t_start_s: f64,
    sighting: OwnerSighting,
    /// Physical tags of every sighting of the target in this keyframe.
    physicals: Vec<Option<usize>>,
}

struct Open {
    target: OwnerTarget,
    from_s: f64,
    opened_at: String,
    opened_by: OpenReason,
    corroboration: Option<Corroboration>,
    moved_from: Option<OwnerTarget>,
    backfill_from_s: Option<f64>,
    sightings: Vec<OwnerSighting>,
    /// Starts of the consecutive targeted keyframes, in view, that lacked it.
    missing: Vec<f64>,
}

impl Open {
    fn last_end(&self) -> f64 {
        self.sightings.last().map_or(self.from_s, |s| s.t_end_s)
    }

    /// The last registered position of the tag.
    fn place(&self) -> Option<TagPlace> {
        self.sightings.iter().rev().find_map(|s| s.place)
    }
}

fn close(o: Open, person_id: &str, display_name: &str, to_s: f64) -> OwnerAssignment {
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

/// Where an open target is in view: `visible(keyframe_id, target, tag)` says the
/// target was read in the keyframe and the tag's last registered position (when
/// known) lies inside its canvas.
pub type Visible<'a> = dyn Fn(&str, &OwnerTarget, Option<&TagPlace>) -> bool + 'a;

/// Absence evidence: `absent(target, tag, after_s, before_s)` is the start of the
/// first keyframe starting in `[after_s, before_s)` that has the target (and the tag's
/// last position) in view while the person's name appears nowhere in it.
pub type Absent<'a> = dyn Fn(&OwnerTarget, Option<&TagPlace>, f64, f64) -> Option<f64> + 'a;

/// Replay one person's sightings (time order; several per keyframe allowed) into
/// assignments. `timeline_end_s` closes those still open. An open target only counts
/// as absent where it is in view (`visible`): a node the reader missed, or a tag a pan
/// cut off, is not a move.
pub fn assign(
    person_id: &str,
    display_name: &str,
    sightings: &[OwnerSighting],
    timeline_end_s: f64,
    params: &OwnerParams,
    corroborator: &dyn Corroborator,
    visible: &Visible<'_>,
) -> Vec<OwnerAssignment> {
    assign_with(
        person_id,
        display_name,
        sightings,
        timeline_end_s,
        params,
        corroborator,
        visible,
        &|_, _, _, _| None,
    )
}

/// [`assign`] with absence evidence ([`Absent`]).
#[allow(clippy::too_many_arguments)]
pub fn assign_with(
    person_id: &str,
    display_name: &str,
    sightings: &[OwnerSighting],
    timeline_end_s: f64,
    params: &OwnerParams,
    corroborator: &dyn Corroborator,
    visible: &Visible<'_>,
    absent: &Absent<'_>,
) -> Vec<OwnerAssignment> {
    let confirm = params.confirm_keyframes.max(1);
    let keys = by_keyframe(sightings);
    let mut done: Vec<OwnerAssignment> = Vec::new();
    let mut open: Vec<Open> = Vec::new();
    // Pending targets with their consecutive targeted keyframes.
    let mut pending: BTreeMap<OwnerTarget, Vec<RunStep>> = BTreeMap::new();
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
        let (targets, physicals): (Vec<OwnerSighting>, Vec<Vec<Option<usize>>>) = {
            let open_t: Vec<&OwnerTarget> = open.iter().map(|o| &o.target).collect();
            let pend_t: Vec<&OwnerTarget> = pending.keys().collect();
            resolve_targets(&k.targets, &open_t, &pend_t)
                .into_iter()
                .unzip()
        };
        let here: Vec<&OwnerTarget> = targets.iter().filter_map(|s| s.target.as_ref()).collect();
        for o in open.iter_mut() {
            match targets
                .iter()
                .find(|s| s.target.as_ref() == Some(&o.target))
            {
                Some(s) => {
                    o.sightings.push(s.clone());
                    o.missing.clear();
                }
                None if visible(k.keyframe_id, &o.target, o.place().as_ref()) => {
                    o.missing.push(k.t_start_s)
                }
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
        for (s, phys) in targets.iter().zip(&physicals) {
            let Some(t) = &s.target else { continue };
            if open.iter().any(|o| &o.target == t) {
                continue;
            }
            let run = pending.entry(t.clone()).or_default();
            // As many keyframes since the run's last step as it takes to close an
            // open target (`confirm_keyframes`) that showed the target (and the
            // tag's place) with the person named nowhere break the run: its steps
            // were not consecutive views of the tag.
            if let Some(last) = run.last() {
                let place = last.sighting.place.or(s.place);
                let mut after = last.sighting.t_end_s;
                let mut absences = 0;
                while absences < confirm {
                    match absent(t, place.as_ref(), after, k.t_start_s) {
                        Some(x) if x >= after - 1e-9 => {
                            absences += 1;
                            after = x + 1e-6;
                        }
                        _ => break,
                    }
                }
                if absences >= confirm {
                    run.clear();
                }
            }
            run.push(RunStep {
                keyframe_id: k.keyframe_id.to_string(),
                t_start_s: k.t_start_s,
                sighting: s.clone(),
                physicals: phys.clone(),
            });
        }
        // Closings by sustained absence (`closed_now` keeps the first missing
        // keyframe, which a move opening there matches); an earlier keyframe that read
        // the target with the person named nowhere dates the close.
        let mut closed_now: Vec<(OwnerTarget, f64)> = Vec::new();
        let mut i = 0;
        while i < open.len() {
            if open[i].missing.len() >= confirm {
                let at = open[i].missing[0];
                let o = open.remove(i);
                let to = absent(&o.target, o.place().as_ref(), o.last_end(), at)
                    .filter(|x| *x >= o.from_s)
                    .unwrap_or(at);
                closed_now.push((o.target.clone(), at));
                done.push(close(o, person_id, display_name, to));
            } else {
                i += 1;
            }
        }
        // Openings.
        let ready: Vec<OwnerTarget> = pending.keys().cloned().collect();
        for t in ready {
            let Some(run) = pending.get(&t) else { continue };
            // Reader-only sightings corroborate; they never open a target alone.
            if run.iter().all(|r| r.sighting.name_read == NameRead::Reader) {
                continue;
            }
            let leaving = open
                .iter()
                .find(|o| !o.missing.is_empty())
                .map(|o| o.target.clone());
            let (support, first) = run_support(run);
            let corroboration = if support < confirm {
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
            // One strong sighting that no later keyframe can confirm or contradict.
            let final_hold = support < confirm
                && corroboration.is_none()
                && k.t_end_s >= timeline_end_s - 1e-9
                && k.t_end_s - k.t_start_s >= params.final_hold_min_s
                && targets.iter().any(|s| {
                    s.target.as_ref() == Some(&t)
                        && s.ocr_located
                        && s.name_read != NameRead::Reader
                        && matches!(
                            s.anchor,
                            AnchorKind::GeometryNode
                                | AnchorKind::GeometryEdge
                                | AnchorKind::GeometryBridge
                        )
                });
            if support < confirm && corroboration.is_none() && !final_hold {
                continue;
            }
            // Confirmed by a stretch: from its start. Opened on this keyframe alone
            // (corroboration, final hold): from this keyframe, the run's last step.
            let from = if support < confirm {
                run.len() - 1
            } else {
                first
            };
            let at = run[from].t_start_s;
            let opened_at = run[from].keyframe_id.clone();
            let sight: Vec<OwnerSighting> =
                run[from..].iter().map(|r| r.sighting.clone()).collect();
            // Where an open target being left was first read with the person named
            // nowhere since its last sighting, else `at`.
            let left_at = |o: &Open| {
                absent(&o.target, o.place().as_ref(), o.last_end(), at)
                    .filter(|x| *x >= o.from_s && *x <= at)
                    .unwrap_or(at)
            };
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
                    let to = left_at(&o);
                    moved_from = Some(o.target.clone());
                    done.push(close(o, person_id, display_name, to));
                }
            }
            // A target that was in view but untagged from the start of this run was
            // left for the new one: a move, even if it is not read again afterwards.
            if moved_from.is_none() {
                if let Some(pos) = open
                    .iter()
                    .position(|o| o.missing.first().is_some_and(|m| *m <= at + 1e-9))
                {
                    let o = open.remove(pos);
                    let to = left_at(&o);
                    moved_from = Some(o.target.clone());
                    done.push(close(o, person_id, display_name, to));
                }
            }
            // An open target read, since its last sighting, where the person's name
            // appears nowhere was left: a move, closed where it was first seen empty.
            if moved_from.is_none() {
                let left = open
                    .iter()
                    .enumerate()
                    .filter_map(|(i, o)| {
                        absent(&o.target, o.place().as_ref(), o.last_end(), at)
                            .filter(|x| *x >= o.from_s && *x <= at)
                            .map(|x| (i, x))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((pos, x)) = left {
                    let o = open.remove(pos);
                    moved_from = Some(o.target.clone());
                    done.push(close(o, person_id, display_name, x));
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
                opened_at,
                opened_by: if corroboration.is_some() {
                    OpenReason::Corroborated
                } else if final_hold {
                    OpenReason::FinalHold
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

/// Group one person's OCR-read sightings into physical tags and give each tag one
/// target set. Two sightings are the same physical tag when they share a registration
/// cluster and their centers lie within `reach_share` tag sizes of the tag's mean
/// center; one physical tag holds at most one tag of a keyframe (the two sightings of
/// a bridge are one tag). A tag that has not moved has not changed target: the target set its
/// keyframes anchored to most often (a keyframe's set is all targets the tag took
/// there, so a bridge counts as one pair) is given to every keyframe of the tag,
/// including keyframes where geometry found nothing. Ties go to the set with more
/// OCR-placed sightings, then to the earliest seen. Sightings without a registered
/// place, or not read by OCR, are left as they are. Each sighting of a physical tag
/// records its index (`physical`): consecutive keyframes confirm an opening only when
/// they show the same physical tag ([`assign`]).
pub fn consolidate_tags(sightings: &mut Vec<OwnerSighting>, reach_share: f64) {
    // (cluster, sum x, sum y, count, max size, member indexes)
    struct Tag {
        cluster: usize,
        sx: f64,
        sy: f64,
        n: f64,
        size: f64,
        members: Vec<usize>,
    }
    let mut tags: Vec<Tag> = Vec::new();
    for (i, s) in sightings.iter().enumerate() {
        let (Some(p), NameRead::Ocr) = (s.place, s.name_read) else {
            continue;
        };
        // A bridge is two sightings of one physical tag in one keyframe.
        if let Some(t) = tags.iter_mut().find(|t| {
            t.members
                .iter()
                .any(|&j| sightings[j].keyframe_id == s.keyframe_id && sightings[j].tag == s.tag)
        }) {
            t.members.push(i);
            continue;
        }
        // Another tag of the same keyframe is another physical tag, however near.
        let near = tags
            .iter_mut()
            .filter(|t| t.cluster == p.cluster)
            .filter(|t| {
                !t.members
                    .iter()
                    .any(|&j| sightings[j].keyframe_id == s.keyframe_id)
            })
            .map(|t| {
                let d = (p.x - t.sx / t.n).hypot(p.y - t.sy / t.n);
                (d, reach_share * t.size.max(p.size()), t)
            })
            .filter(|(d, r, _)| d <= r)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        match near {
            Some((_, _, t)) => {
                t.sx += p.x;
                t.sy += p.y;
                t.n += 1.0;
                t.size = t.size.max(p.size());
                t.members.push(i);
            }
            None => tags.push(Tag {
                cluster: p.cluster,
                sx: p.x,
                sy: p.y,
                n: 1.0,
                size: p.size(),
                members: vec![i],
            }),
        }
    }
    for (ti, t) in tags.iter().enumerate() {
        for &i in &t.members {
            sightings[i].physical = Some(ti);
        }
    }
    let mut replace: BTreeMap<usize, Vec<OwnerSighting>> = BTreeMap::new();
    for t in &tags {
        // Target set per keyframe, in first-seen order.
        let mut frames: Vec<(&str, Vec<usize>)> = Vec::new();
        for &i in &t.members {
            let kf = sightings[i].keyframe_id.as_str();
            match frames.iter_mut().find(|(k, _)| *k == kf) {
                Some((_, v)) => v.push(i),
                None => frames.push((kf, vec![i])),
            }
        }
        // (target set, keyframes, OCR-placed sightings), in first-seen order.
        let mut votes: Vec<(Vec<OwnerTarget>, usize, usize)> = Vec::new();
        for (_, idx) in &frames {
            let mut set: Vec<OwnerTarget> = idx
                .iter()
                .filter_map(|&i| sightings[i].target.clone())
                .collect();
            set.sort();
            set.dedup();
            if set.is_empty() {
                continue;
            }
            let located = idx.iter().filter(|&&i| sightings[i].ocr_located).count();
            match votes.iter_mut().find(|v| v.0 == set) {
                Some(v) => {
                    v.1 += 1;
                    v.2 += located;
                }
                None => votes.push((set, 1, located)),
            }
        }
        let Some(best) = votes
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| (a.1, a.2).cmp(&(b.1, b.2)).then(ib.cmp(ia)))
            .map(|(_, v)| v.0.clone())
        else {
            continue;
        };
        for (_, idx) in &frames {
            let mut own: Vec<OwnerTarget> = idx
                .iter()
                .filter_map(|&i| sightings[i].target.clone())
                .collect();
            own.sort();
            own.dedup();
            if own == best {
                continue;
            }
            let Some(&first) = idx.first() else { continue };
            let base = &sightings[first];
            let new: Vec<OwnerSighting> = best
                .iter()
                .map(|target| OwnerSighting {
                    target: Some(target.clone()),
                    anchor: AnchorKind::Registered,
                    alternates: Vec::new(),
                    ..base.clone()
                })
                .collect();
            replace.insert(first, new);
            for &i in &idx[1..] {
                replace.insert(i, Vec::new());
            }
        }
    }
    if replace.is_empty() {
        return;
    }
    let old = std::mem::take(sightings);
    for (i, s) in old.into_iter().enumerate() {
        match replace.remove(&i) {
            Some(v) => sightings.extend(v),
            None => sightings.push(s),
        }
    }
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
            // Only two separate tags; one tag bridging both ends stays two node targets,
            // and two placed tags are two physical tags ([`consolidate_tags`]).
            if let (Some(pa), Some(pb)) = (pa, pb) {
                if group[pa].tag == group[pb].tag
                    || (group[pa].place.is_some() && group[pb].place.is_some())
                {
                    continue;
                }
                let mut merged = group[pa.min(pb)].clone();
                merged.target = Some(edge.clone());
                merged.anchor = AnchorKind::BothEnds;
                merged.alternates.clear();
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
                    sightings[idx].alternates.clear();
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
            ocr_located: false,
            name_read: NameRead::Unchecked,
            alternates: Vec::new(),
            place: None,
            physical: None,
        }
    }

    fn params() -> OwnerParams {
        OwnerParams {
            confirm_keyframes: 2,
            backfill_untargeted: false,
            final_hold_min_s: 30.0,
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
            &|_, _, _| true,
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
            &|_, _, _| true,
        );
        assert_eq!(a.len(), 1);
        let b = assign(
            "p1",
            "Avery",
            &seq,
            300.0,
            &params(),
            &Always,
            &|_, _, _| true,
        );
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
            &|_, _, _| true,
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
            &|_, _, _| true,
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
            &|_, _, _| true
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
            &|_, _, _| true,
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
            &|_, _, _| true,
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
            &|_, _, _| true,
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
            &|_, _, _| true,
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
        let vis = |kf: &str, t: &OwnerTarget, _: Option<&TagPlace>| {
            !(t == &node("n3") && (kf == "kf20" || kf == "kf30"))
        };
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
            &|_, _, _| true,
        );
        let n3: Vec<_> = b.iter().filter(|x| x.target == node("n3")).collect();
        assert_eq!(n3[0].valid_to_s, 20.0);
    }

    #[test]
    fn a_move_closes_the_old_target_even_if_it_is_not_read_again() {
        // n1 held; n2 tagged twice while n1 is visible at the first of those keyframes
        // only (n1 is not read at the second).
        let v = seq(&["n1", "n1", "n2", "n2"]);
        let vis =
            |kf: &str, t: &OwnerTarget, _: Option<&TagPlace>| !(t == &node("n1") && kf == "kf30");
        let a = assign("p1", "Avery", &v, 100.0, &params(), &NoCorroboration, &vis);
        assert_eq!(a.len(), 2, "{a:#?}");
        assert_eq!(a[0].valid_to_s, 20.0);
        assert_eq!(a[1].moved_from, Some(node("n1")));
    }

    fn strong(t: f64, end: f64, target: OwnerTarget) -> OwnerSighting {
        OwnerSighting {
            t_end_s: end,
            anchor: AnchorKind::GeometryNode,
            ocr_located: true,
            ..s(t, Some(target))
        }
    }

    #[test]
    fn a_strong_sighting_in_a_long_last_keyframe_moves_the_owner() {
        // n1 held; the last keyframe (100 to 160 s) shows the tag on n2, placed on OCR.
        // n1 was read at 80 s with the person's name nowhere on the canvas.
        let v = [
            s(0.0, Some(node("n1"))),
            s(10.0, Some(node("n1"))),
            strong(100.0, 160.0, node("n2")),
        ];
        let absent = |t: &OwnerTarget, _: Option<&TagPlace>, after: f64, before: f64| {
            (t == &node("n1") && after <= 80.0 && 80.0 < before).then_some(80.0)
        };
        // n1 is out of view in the last keyframe.
        let vis =
            |kf: &str, t: &OwnerTarget, _: Option<&TagPlace>| !(kf == "kf100" && t == &node("n1"));
        let a = assign_with(
            "p1",
            "Avery",
            &v,
            160.0,
            &params(),
            &NoCorroboration,
            &vis,
            &absent,
        );
        assert_eq!(a.len(), 2, "{a:#?}");
        assert_eq!((a[0].valid_from_s, a[0].valid_to_s), (0.0, 80.0));
        assert_eq!(a[1].target, node("n2"));
        assert_eq!(a[1].opened_by, OpenReason::FinalHold);
        assert_eq!(a[1].moved_from, Some(node("n1")));
        assert_eq!(a[1].valid_from_s, 100.0);
        // Without the absence the old target is kept: n2 is an added target.
        let b = assign("p1", "Avery", &v, 160.0, &params(), &NoCorroboration, &vis);
        assert_eq!(b.len(), 2);
        assert!(b
            .iter()
            .all(|x| x.moved_from.is_none() && x.valid_at(150.0)));
    }

    #[test]
    fn one_strong_sighting_opens_nothing_unless_it_is_last_long_and_located() {
        let base = [s(0.0, Some(node("n1"))), s(10.0, Some(node("n1")))];
        let run = |last: OwnerSighting, end: f64| {
            let mut v = base.to_vec();
            v.push(last);
            assign(
                "p1",
                "Avery",
                &v,
                end,
                &params(),
                &NoCorroboration,
                &|_, _, _| false,
            )
        };
        // Mid-meeting: a later keyframe exists (timeline ends after it).
        assert_eq!(run(strong(100.0, 160.0, node("n2")), 400.0).len(), 1);
        // Last but short.
        assert_eq!(run(strong(100.0, 110.0, node("n2")), 110.0).len(), 1);
        // Last and long but not placed on OCR.
        let unplaced = OwnerSighting {
            ocr_located: false,
            ..strong(100.0, 160.0, node("n2"))
        };
        assert_eq!(run(unplaced, 160.0).len(), 1);
        // Last and long but only the reader's `near`.
        let near = OwnerSighting {
            anchor: AnchorKind::Near,
            ..strong(100.0, 160.0, node("n2"))
        };
        assert_eq!(run(near, 160.0).len(), 1);
        let a = run(strong(100.0, 160.0, node("n2")), 160.0);
        assert_eq!(a.len(), 2);
        assert_eq!(a[1].opened_by, OpenReason::FinalHold);
    }

    #[test]
    fn a_move_is_dated_by_the_earliest_absence_whichever_rule_closes_it() {
        // n1 held; read at 40 s with the person named nowhere; n2 tagged at 60 and 80 s
        // while n1 is still read (the visible-but-untagged rule closes it).
        let v = seq(&["n1", "n1"])
            .into_iter()
            .chain([s(60.0, Some(node("n2"))), s(80.0, Some(node("n2")))])
            .collect::<Vec<_>>();
        let absent = |t: &OwnerTarget, _: Option<&TagPlace>, after: f64, before: f64| {
            (t == &node("n1") && after <= 40.0 && 40.0 < before).then_some(40.0)
        };
        let a = assign_with(
            "p1",
            "Avery",
            &v,
            200.0,
            &params(),
            &NoCorroboration,
            &|_, _, _| true,
            &absent,
        );
        assert_eq!(a.len(), 2, "{a:#?}");
        assert_eq!(a[0].valid_to_s, 40.0);
        assert_eq!(a[1].valid_from_s, 60.0);
        assert_eq!(a[1].moved_from, Some(node("n1")));
    }

    /// Two sightings of a target far apart are consecutive keyframes of the
    /// person only while fewer keyframes between them than it takes to close an
    /// open target showed the target with the person named nowhere: that many
    /// restart the run, and the target opens at the next two consecutive
    /// sightings. One such keyframe does not.
    #[test]
    fn a_keyframe_showing_the_target_untagged_breaks_a_pending_run() {
        let v = vec![
            s(10.0, Some(node("n1"))),
            s(500.0, Some(node("n1"))),
            s(520.0, Some(node("n1"))),
        ];
        let run = |absent: &Absent<'_>| {
            assign_with(
                "p1",
                "Avery",
                &v,
                600.0,
                &params(),
                &NoCorroboration,
                &|_, _, _| true,
                absent,
            )
        };
        let a = run(&|_, _, _, _| None);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].valid_from_s, 10.0);
        let untagged_at = |at: &'static [f64]| {
            move |_: &OwnerTarget, _: Option<&TagPlace>, after: f64, before: f64| {
                at.iter().copied().find(|&x| after <= x && x < before)
            }
        };
        let a = run(&untagged_at(&[200.0]));
        assert_eq!(a[0].valid_from_s, 10.0, "one absence is not a break");
        let a = run(&untagged_at(&[200.0, 300.0]));
        assert_eq!(a.len(), 1, "{a:#?}");
        assert_eq!(a[0].valid_from_s, 500.0);
    }

    fn read_by(mut x: OwnerSighting, by: NameRead) -> OwnerSighting {
        x.name_read = by;
        x
    }

    fn at(mut x: OwnerSighting, px: f64, py: f64) -> OwnerSighting {
        x.name_read = NameRead::Ocr;
        x.place = Some(TagPlace {
            cluster: 0,
            x: px,
            y: py,
            w: 60.0,
            h: 30.0,
        });
        x
    }

    #[test]
    fn reader_only_sightings_corroborate_but_never_open() {
        let reader = |t: f64| read_by(s(t, Some(node("n1"))), NameRead::Reader);
        let ocr = |t: f64| read_by(s(t, Some(node("n1"))), NameRead::Ocr);
        let run = |v: &[OwnerSighting]| {
            assign(
                "p1",
                "Avery",
                v,
                100.0,
                &params(),
                &NoCorroboration,
                &|_, _, _| true,
            )
        };
        assert!(run(&[reader(0.0), reader(10.0), reader(20.0)]).is_empty());
        // A reader tag the OCR confirms in the next keyframe opens from the first.
        let a = run(&[reader(0.0), ocr(10.0)]);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].valid_from_s, 0.0);
        // Once open, reader-only sightings keep it (no absence).
        let a = run(&[ocr(0.0), ocr(10.0), reader(20.0), reader(30.0)]);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].sightings.len(), 4);
    }

    #[test]
    fn ambiguous_geometry_keeps_the_incumbent_edge() {
        // Held on the edge; two keyframes anchor to its end n2, with the edge as an
        // alternate: the edge is kept, no move.
        let near_end = |t: f64| OwnerSighting {
            alternates: vec![edge()],
            ..s(t, Some(node("n2")))
        };
        let v = [
            s(0.0, Some(edge())),
            s(10.0, Some(edge())),
            near_end(20.0),
            near_end(30.0),
            s(40.0, Some(edge())),
        ];
        let a = assign(
            "p1",
            "Avery",
            &v,
            100.0,
            &params(),
            &NoCorroboration,
            &|_, _, _| true,
        );
        assert_eq!(a.len(), 1, "{a:#?}");
        assert_eq!(a[0].target, edge());
        assert_eq!(a[0].sightings.len(), 5);
        // Without the alternate the node takes over: a move.
        let w: Vec<OwnerSighting> = v
            .iter()
            .cloned()
            .map(|x| OwnerSighting {
                alternates: Vec::new(),
                ..x
            })
            .collect();
        let b = assign(
            "p1",
            "Avery",
            &w,
            100.0,
            &params(),
            &NoCorroboration,
            &|_, _, _| true,
        );
        assert!(b
            .iter()
            .any(|x| x.target == node("n2") && x.moved_from.is_some()));
    }

    #[test]
    fn one_physical_tag_gets_its_majority_target_everywhere() {
        // One tag at (500, 100): anchored n1, n1, n2, untargeted, n1; another tag of the
        // same person at (900, 400) anchored n3.
        let mut v = vec![
            at(s(0.0, Some(node("n1"))), 500.0, 100.0),
            at(s(10.0, Some(node("n1"))), 503.0, 98.0),
            at(s(20.0, Some(node("n2"))), 497.0, 104.0),
            at(s(30.0, None), 501.0, 101.0),
            at(s(40.0, Some(node("n1"))), 499.0, 99.0),
            at(s(40.0, Some(node("n3"))), 900.0, 400.0),
        ];
        v[5].tag = 1;
        consolidate_tags(&mut v, 1.0);
        let targets: Vec<Option<OwnerTarget>> = v.iter().map(|x| x.target.clone()).collect();
        assert_eq!(
            targets,
            vec![
                Some(node("n1")),
                Some(node("n1")),
                Some(node("n1")),
                Some(node("n1")),
                Some(node("n1")),
                Some(node("n3")),
            ]
        );
        assert_eq!(v[2].anchor, AnchorKind::Registered);
        assert_eq!(v[3].anchor, AnchorKind::Registered);
        assert_eq!(v[0].anchor, AnchorKind::Near);
        assert_eq!(v[0].physical, v[4].physical);
        assert_ne!(v[0].physical, v[5].physical);
        // Reader-only sightings are not grouped.
        let mut r = vec![read_by(s(0.0, Some(node("n2"))), NameRead::Reader)];
        consolidate_tags(&mut r, 1.0);
        assert_eq!(r[0].target, Some(node("n2")));
        assert!(r[0].physical.is_none());
    }

    #[test]
    fn two_tags_of_one_keyframe_are_two_physical_tags() {
        // Keyframe 0 reads two nearby Avery tags, on n1 and on n2; keyframe 10 reads
        // only the n1 tag. The n2 tag must not be copied into keyframe 10 and
        // confirmed there.
        let mut v = vec![
            at(s(0.0, Some(node("n1"))), 500.0, 100.0),
            OwnerSighting {
                tag: 1,
                ..at(s(0.0, Some(node("n2"))), 520.0, 100.0)
            },
            at(s(10.0, Some(node("n1"))), 505.0, 100.0),
        ];
        consolidate_tags(&mut v, 1.0);
        assert_ne!(v[0].physical, v[1].physical);
        assert_eq!(v[0].physical, v[2].physical);
        let at10: Vec<&OwnerTarget> = v
            .iter()
            .filter(|x| x.t_start_s == 10.0)
            .filter_map(|x| x.target.as_ref())
            .collect();
        assert_eq!(at10, vec![&node("n1")]);
        let a = assign(
            "p1",
            "Avery",
            &v,
            100.0,
            &params(),
            &NoCorroboration,
            &|_, _, _| true,
        );
        assert!(a.iter().all(|x| x.target != node("n2")), "{a:#?}");
        assert!(a.iter().any(|x| x.target == node("n1")), "{a:#?}");
    }

    #[test]
    fn a_bridge_counts_as_one_target_pair() {
        let bridge = |t: f64| {
            [node("n1"), node("n2")].map(|n| OwnerSighting {
                anchor: AnchorKind::GeometryBridge,
                ..at(s(t, Some(n)), 300.0, 300.0)
            })
        };
        let mut v: Vec<OwnerSighting> = bridge(0.0).into_iter().chain(bridge(10.0)).collect();
        v.push(at(s(20.0, Some(node("n1"))), 302.0, 301.0));
        consolidate_tags(&mut v, 1.0);
        let at20: Vec<&OwnerTarget> = v
            .iter()
            .filter(|x| x.t_start_s == 20.0)
            .filter_map(|x| x.target.as_ref())
            .collect();
        assert_eq!(at20, vec![&node("n1"), &node("n2")]);
    }

    #[test]
    fn sightings_of_one_target_at_two_places_do_not_confirm_each_other() {
        // Two keyframes read the name next to n1, but at places far apart (one view
        // misregistered): no opening.
        let mut v = vec![
            at(s(0.0, Some(node("n1"))), 100.0, 100.0),
            at(s(10.0, Some(node("n1"))), 400.0, 300.0),
        ];
        consolidate_tags(&mut v, 1.0);
        let run = |v: &[OwnerSighting]| {
            assign(
                "p1",
                "Avery",
                v,
                100.0,
                &params(),
                &NoCorroboration,
                &|_, _, _| true,
            )
        };
        assert!(run(&v).is_empty(), "{:#?}", run(&v));
    }

    #[test]
    fn a_tag_out_of_view_is_not_missing() {
        // n1 and n3 held; in two keyframes n3 is read but its tag is out of view.
        let tagged = |t: f64, with_n3: bool| {
            let mut v = vec![at(s(t, Some(node("n1"))), 100.0, 100.0)];
            if with_n3 {
                v.push(OwnerSighting {
                    tag: 1,
                    ..at(s(t, Some(node("n3"))), 800.0, 900.0)
                });
            }
            v
        };
        let v: Vec<OwnerSighting> = [
            (0.0, true),
            (10.0, true),
            (20.0, false),
            (30.0, false),
            (40.0, true),
        ]
        .into_iter()
        .flat_map(|(t, n3)| tagged(t, n3))
        .collect();
        // The view of keyframes 20 and 30 ends above y = 700.
        let vis = |kf: &str, _: &OwnerTarget, p: Option<&TagPlace>| {
            !((kf == "kf20" || kf == "kf30") && p.is_some_and(|p| p.y > 700.0))
        };
        let a = assign("p1", "Avery", &v, 100.0, &params(), &NoCorroboration, &vis);
        let n3: Vec<_> = a.iter().filter(|x| x.target == node("n3")).collect();
        assert_eq!(n3.len(), 1, "{a:#?}");
        assert_eq!((n3[0].valid_from_s, n3[0].valid_to_s), (0.0, 100.0));
    }

    #[test]
    fn support_needs_consecutive_keyframes_of_one_physical_tag() {
        // Tags A, B, A (all anchored to n1): no two consecutive keyframes show one tag.
        let mut v = vec![
            at(s(0.0, Some(node("n1"))), 100.0, 100.0),
            at(s(10.0, Some(node("n1"))), 400.0, 300.0),
            at(s(20.0, Some(node("n1"))), 101.0, 100.0),
        ];
        consolidate_tags(&mut v, 0.5);
        let run = |v: &[OwnerSighting]| {
            assign(
                "p1",
                "Avery",
                v,
                100.0,
                &params(),
                &NoCorroboration,
                &|_, _, _| true,
            )
        };
        assert!(run(&v).is_empty(), "{:#?}", run(&v));
        // A fourth keyframe of tag A: A, A in a row opens at the start of that stretch.
        v.push(at(s(30.0, Some(node("n1"))), 99.0, 101.0));
        consolidate_tags(&mut v, 0.5);
        let a = run(&v);
        assert_eq!(a.len(), 1, "{a:#?}");
        assert_eq!(a[0].valid_from_s, 20.0);
        // A keyframe with an unplaced sighting of the target continues any stretch.
        let mut w = vec![
            at(s(0.0, Some(node("n1"))), 100.0, 100.0),
            s(10.0, Some(node("n1"))),
        ];
        consolidate_tags(&mut w, 0.5);
        assert_eq!(run(&w)[0].valid_from_s, 0.0);
    }

    #[test]
    fn a_final_hold_after_another_tag_opens_at_the_last_keyframe() {
        // n2 read once at one place, then (last, long, OCR placed) at another place:
        // the final hold opens at the last keyframe, not at the earlier sighting.
        let mut v = vec![
            at(s(0.0, Some(node("n1"))), 50.0, 50.0),
            at(s(10.0, Some(node("n1"))), 51.0, 50.0),
            at(s(50.0, Some(node("n2"))), 300.0, 300.0),
            OwnerSighting {
                t_end_s: 160.0,
                anchor: AnchorKind::GeometryNode,
                ocr_located: true,
                ..at(s(100.0, Some(node("n2"))), 700.0, 700.0)
            },
        ];
        consolidate_tags(&mut v, 0.5);
        let a = assign(
            "p1",
            "Avery",
            &v,
            160.0,
            &params(),
            &NoCorroboration,
            &|_, _, _| false,
        );
        let n2: Vec<_> = a.iter().filter(|x| x.target == node("n2")).collect();
        assert_eq!(n2.len(), 1, "{a:#?}");
        assert_eq!(n2[0].opened_by, OpenReason::FinalHold);
        assert_eq!(n2[0].valid_from_s, 100.0);
        assert_eq!(n2[0].sightings.len(), 1);
    }
}
