//! Private golden set for the meeting suite (spec 9.2).
//!
//! The golden file lives only in the private fixtures root
//! (`<eval.private_fixtures>/golden/meeting_golden.json`, or `.toml`), never in
//! the repository. Its shape is [`MeetingGolden`]:
//!
//! | Field | Truth |
//! |---|---|
//! | `participants` | person ids, display names, aliases (used to resolve predicted names) |
//! | `screen_types` | one label per golden keyframe `t_rep_s` (unconfirmed labels are reported, not scored) |
//! | `final_board` | nodes (with `core`), directed edges, stickies, card groups |
//! | `owners` | timed assignments, probe times, moves, negative names |
//! | `traps` | trap frames and windows (coverage is reported) |
//! | `static_windows` | windows where only pans and zooms happen, with allowed real events |
//! | `chrome_terms` | UI strings that must never become board content |
//! | `transcript` | decisions, action items, open questions, negative action items, hotwords, speaker count |
//! | `clocks` | per-section [`FrameClock`] ([`ClockSection`]); PTS unless declared |
//!
//! Authoring can use `screen_type_ranges` (inclusive `t_rep` ranges) instead of
//! per-keyframe labels; the `golden_convert` example expands them against the
//! reference keyframe list and writes the JSON the loader reads. The loader
//! rejects unexpanded ranges so a half-converted file is never scored.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{read_json, read_toml, EvalError, Result};
use crate::metrics::audio::Hotword;
use crate::metrics::board::GoldBoard;
use crate::metrics::events::StaticWindow;
use crate::metrics::notes::GoldItem;
use crate::metrics::owners::{Assignment, Move, Target};
use crate::metrics::screen::ScreenType;
use crate::text::{claim_words, contradicts, key_terms, Vocabulary};

/// Current golden format version.
pub const GOLDEN_VERSION: u32 = 1;

fn default_true() -> bool {
    true
}

/// A meeting participant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Participant {
    /// Stable id.
    pub person_id: String,
    /// Display name.
    pub display_name: String,
    /// Other spellings (first names, misrecognitions seen on tags).
    #[serde(default)]
    pub aliases: Vec<String>,
}

/// Screen type of one golden keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreenLabel {
    /// Golden representative time, seconds.
    pub t_rep_s: f64,
    /// Screen type.
    pub screen_type: ScreenType,
    /// False when the label still needs confirmation (reported, not scored).
    #[serde(default = "default_true")]
    pub confirmed: bool,
}

/// An inclusive range of golden keyframes sharing a screen type (authoring form).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreenRange {
    /// First `t_rep` in the range, seconds.
    pub from_t_rep_s: f64,
    /// Last `t_rep` in the range, seconds.
    pub to_t_rep_s: f64,
    /// Screen type.
    pub screen_type: ScreenType,
    /// Confirmation flag applied to every expanded label.
    #[serde(default = "default_true")]
    pub confirmed: bool,
}

/// Owner truth.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnerTruth {
    /// Probe times, seconds.
    pub probes_s: Vec<f64>,
    /// Timed assignments.
    pub assignments: Vec<Assignment>,
    /// Moves.
    #[serde(default)]
    pub moves: Vec<Move>,
    /// Names that must never become owners.
    #[serde(default)]
    pub negatives: Vec<String>,
}

/// A trap frame or window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trap {
    /// Single golden keyframe time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_rep_s: Option<f64>,
    /// Window start (with `t_to_s`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_from_s: Option<f64>,
    /// Window end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_to_s: Option<f64>,
    /// Trap kind (for example `sidebar_chrome`, `cms_as_board`).
    pub kind: String,
    /// What the trap tests.
    pub note: String,
}

/// Clock of a golden section's times.
///
/// Spec 9.2: golden `t_rep` values may sit on the prototype's nominal frame grid.
/// The prototype sampled with ffmpeg `fps=1/INTERVAL` and named each frame by its
/// grid time, but the frame it keeps for grid time `t` is the last video frame before
/// `t + INTERVAL`. A label at nominal `t` then describes what was on screen just
/// before `t + INTERVAL`, while the pipeline's keyframes carry true PTS times, so the
/// join maps the label there. Whether a golden's times are grid names is a property
/// of how that golden was authored, never of the eval: a section that declares no
/// clock ([`GoldenClocks`]) is on [`FrameClock::Pts`], the eval's scoring before
/// clocks existed, so an old golden scores as it did until it opts in. Screen
/// labels and traps without a declared clock are reported as warnings
/// ([`MeetingGolden::clock_warnings`]), since those are the times a golden may have
/// named on the prototype grid.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrameClock {
    /// Times are presentation times (the default).
    #[default]
    Pts,
    /// Times are nominal grid names of prototype frames.
    PrototypeGrid {
        /// Grid interval, seconds.
        interval_s: f64,
        /// How far before the next grid point the kept frame lies, seconds (default
        /// 0.05: under one frame at 20 fps or more).
        #[serde(default = "default_grid_lead_s")]
        lead_s: f64,
    },
}

fn default_grid_lead_s() -> f64 {
    0.05
}

impl FrameClock {
    /// Content time (PTS) of a golden time on this clock. On the prototype grid that
    /// is `lead_s` before the next grid point.
    pub fn content_time(&self, t: f64) -> f64 {
        match self {
            Self::PrototypeGrid { interval_s, lead_s } => t + interval_s - lead_s,
            Self::Pts => t,
        }
    }

    fn validate(&self, section: &str) -> Result<()> {
        if let Self::PrototypeGrid { interval_s, lead_s } = *self {
            if !(interval_s.is_finite() && interval_s > 0.0) {
                return Err(golden_err(format!(
                    "{section} clock: interval_s {interval_s} is not positive"
                )));
            }
            if !(lead_s.is_finite() && (0.0..interval_s).contains(&lead_s)) {
                return Err(golden_err(format!(
                    "{section} clock: lead_s {lead_s} is not in [0, interval_s)"
                )));
            }
        }
        Ok(())
    }
}

/// A kind of golden time that can be on a declared clock. Each is declared on its
/// own, because one golden can take them from different sources (a probe named by
/// a grid frame, an assignment timed from the transcript).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockSection {
    /// `screen_types[].t_rep_s`.
    ScreenTypes,
    /// `traps[]` times.
    Traps,
    /// `owners.probes_s`.
    OwnerProbes,
    /// `owners.assignments[]` validity windows.
    OwnerAssignments,
    /// `owners.moves[].t_s`.
    OwnerMoves,
    /// `static_windows[]` bounds.
    StaticWindows,
    /// `static_windows[].allowed_events[].t_s`.
    AllowedEvents,
}

impl ClockSection {
    /// Every section, in report order.
    pub const ALL: [ClockSection; 7] = [
        Self::ScreenTypes,
        Self::Traps,
        Self::OwnerProbes,
        Self::OwnerAssignments,
        Self::OwnerMoves,
        Self::StaticWindows,
        Self::AllowedEvents,
    ];

    /// Field name under `clocks`.
    pub fn name(self) -> &'static str {
        match self {
            Self::ScreenTypes => "screen_types",
            Self::Traps => "traps",
            Self::OwnerProbes => "owner_probes",
            Self::OwnerAssignments => "owner_assignments",
            Self::OwnerMoves => "owner_moves",
            Self::StaticWindows => "static_windows",
            Self::AllowedEvents => "allowed_events",
        }
    }
}

/// Per-section clocks ([`ClockSection`]). A section left out is on
/// [`FrameClock::Pts`]. Transcript times (hotword windows, notes) are always PTS.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenClocks {
    /// Clock of `screen_types`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screen_types: Option<FrameClock>,
    /// Clock of `traps`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub traps: Option<FrameClock>,
    /// Clock of `owners.probes_s`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_probes: Option<FrameClock>,
    /// Clock of `owners.assignments` windows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_assignments: Option<FrameClock>,
    /// Clock of `owners.moves`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_moves: Option<FrameClock>,
    /// Clock of `static_windows` bounds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub static_windows: Option<FrameClock>,
    /// Clock of `static_windows[].allowed_events`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_events: Option<FrameClock>,
}

impl GoldenClocks {
    /// True when no section declares a clock.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// The clock a section declares, if any.
    pub fn get(&self, s: ClockSection) -> Option<FrameClock> {
        match s {
            ClockSection::ScreenTypes => self.screen_types,
            ClockSection::Traps => self.traps,
            ClockSection::OwnerProbes => self.owner_probes,
            ClockSection::OwnerAssignments => self.owner_assignments,
            ClockSection::OwnerMoves => self.owner_moves,
            ClockSection::StaticWindows => self.static_windows,
            ClockSection::AllowedEvents => self.allowed_events,
        }
    }
}

/// Transcript and notes truth.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranscriptTruth {
    /// Decisions.
    pub decisions: Vec<GoldItem>,
    /// Action items (with `person_id`).
    pub action_items: Vec<GoldItem>,
    /// Open questions.
    pub open_questions: Vec<GoldItem>,
    /// Texts that must not appear as action items (farewells, greetings).
    #[serde(default)]
    pub negative_action_items: Vec<String>,
    /// Hotwords with occurrence windows.
    #[serde(default)]
    pub hotwords: Vec<Hotword>,
    /// Number of distinct speakers.
    pub speaker_count: usize,
}

/// The private golden set for one meeting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingGolden {
    /// Format version ([`GOLDEN_VERSION`]).
    pub golden_version: u32,
    /// Meeting label.
    pub meeting: String,
    /// Where the truth came from, caveats.
    #[serde(default)]
    pub sources: Vec<String>,
    /// Media duration, seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_s: Option<f64>,
    /// Legacy single clock for `screen_types` and `traps` (the only sections it ever
    /// covered). Declares both unless [`GoldenClocks`] names either; prefer `clocks`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_clock: Option<FrameClock>,
    /// Per-section clocks ([`GoldenClocks`]).
    #[serde(default, skip_serializing_if = "GoldenClocks::is_empty")]
    pub clocks: GoldenClocks,
    /// Participants.
    pub participants: Vec<Participant>,
    /// Per-keyframe screen types.
    #[serde(default)]
    pub screen_types: Vec<ScreenLabel>,
    /// Authoring ranges (must be expanded by `golden_convert` before scoring).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub screen_type_ranges: Vec<ScreenRange>,
    /// Final board.
    pub final_board: GoldBoard,
    /// Owners over time.
    pub owners: OwnerTruth,
    /// Trap frames.
    #[serde(default)]
    pub traps: Vec<Trap>,
    /// Pan/zoom-only windows.
    #[serde(default)]
    pub static_windows: Vec<StaticWindow>,
    /// UI chrome strings.
    #[serde(default)]
    pub chrome_terms: Vec<String>,
    /// Transcript truth.
    pub transcript: TranscriptTruth,
}

fn golden_err(message: impl Into<String>) -> EvalError {
    EvalError::Fixture {
        case: "meeting golden".into(),
        message: message.into(),
    }
}

impl MeetingGolden {
    /// The clock a section declares: its `clocks` entry, else (screen types and
    /// traps only) the legacy `frame_clock`; `None` when undeclared.
    pub fn declared_clock(&self, section: ClockSection) -> Option<FrameClock> {
        let legacy = matches!(section, ClockSection::ScreenTypes | ClockSection::Traps)
            .then_some(self.frame_clock)
            .flatten();
        self.clocks.get(section).or(legacy)
    }

    /// Warnings for frame-named sections (screen labels, traps) that have times but
    /// declare no clock: they are scored on PTS, which is right for a golden timed
    /// on the recording and wrong for one named on the prototype grid.
    pub fn clock_warnings(&self) -> Vec<String> {
        [
            (ClockSection::ScreenTypes, !self.screen_types.is_empty()),
            (ClockSection::Traps, !self.traps.is_empty()),
        ]
        .into_iter()
        .filter(|(s, present)| *present && self.declared_clock(*s).is_none())
        .map(|(s, _)| {
            format!(
                "golden {0} declare no clock and are scored on PTS; set clocks.{0} to pts or prototype_grid to say which",
                s.name()
            )
        })
        .collect()
    }

    /// The clock of one section: [`Self::declared_clock`], else [`FrameClock::Pts`].
    pub fn clock(&self, section: ClockSection) -> FrameClock {
        self.declared_clock(section).unwrap_or_default()
    }

    /// Participant and entity names for the notes matcher: every participant's
    /// display name and aliases, and every final-board node label, node alias, and
    /// hotword.
    pub fn vocabulary(&self) -> Vocabulary {
        let mut v = Vocabulary::default();
        for p in &self.participants {
            v.add_person(
                &p.person_id,
                std::iter::once(p.display_name.as_str())
                    .chain(p.aliases.iter().map(String::as_str)),
            );
        }
        for n in &self.final_board.nodes {
            v.add_entity(&n.text);
            for a in &n.aliases {
                v.add_entity(a);
            }
        }
        for h in &self.transcript.hotwords {
            v.add_entity(&h.word);
        }
        v
    }

    /// Warnings for name words two participants share: the notes matcher reads such
    /// a word as naming neither participant alone ([`Vocabulary::ambiguous_names`]).
    pub fn name_warnings(&self) -> Vec<String> {
        self.vocabulary()
            .ambiguous_names()
            .into_iter()
            .map(|w| format!("golden participants: {w}"))
            .collect()
    }

    /// Notes items whose phrasings carry no key term. They are matched strictly
    /// (every claim word, [`crate::text::allowed_missing`]); listed in the report so
    /// an author can see which items rely on wording alone.
    pub fn unkeyed_notes_items(&self) -> Vec<String> {
        let v = self.vocabulary();
        self.notes_items()
            .filter(|(_, item)| {
                std::iter::once(&item.text)
                    .chain(&item.aliases)
                    .all(|t| key_terms(t, &v).is_empty())
            })
            .map(|(kind, item)| format!("{kind}: {}", item.text))
            .collect()
    }

    fn notes_items(&self) -> impl Iterator<Item = (&'static str, &GoldItem)> {
        let t = &self.transcript;
        t.decisions
            .iter()
            .map(|i| ("decision", i))
            .chain(t.action_items.iter().map(|i| ("action item", i)))
            .chain(t.open_questions.iter().map(|i| ("open question", i)))
    }

    /// Every alias of a notes item states the same claim about the same subject as
    /// its text. An alias must name every participant the text names and, for each
    /// other key term of the text, that term or a word of the same entity name that
    /// no other entity shares ("Ledger step" for "Ledger API step", never "Orbit
    /// Queue" for "Ledger Queue"); otherwise it would match a claim about someone or
    /// something else. And no word the two share may flip polarity ("Ship
    /// Ledger" is not an alias of "Do not ship Ledger").
    fn validate_notes_aliases(&self) -> Result<()> {
        let v = self.vocabulary();
        for (kind, item) in self.notes_items() {
            let keys = key_terms(&item.text, &v);
            for alias in &item.aliases {
                let terms: BTreeSet<String> =
                    claim_words(alias, &v).into_iter().map(|w| w.term).collect();
                let unnamed = keys.iter().find(|k| {
                    if k.starts_with('@') {
                        !terms.contains(*k)
                    } else {
                        !v.names(k, &terms)
                    }
                });
                if let Some(missing) = unnamed {
                    return Err(golden_err(format!(
                        "{kind} `{}`: alias `{alias}` does not name `{missing}`, so it would match a claim about someone or something else",
                        item.text
                    )));
                }
                if contradicts(&item.text, alias, &v) || contradicts(alias, &item.text, &v) {
                    return Err(golden_err(format!(
                        "{kind} `{}`: alias `{alias}` states a shared word with the opposite polarity",
                        item.text
                    )));
                }
            }
        }
        Ok(())
    }

    /// Resolves a predicted person name or id to a participant id (exact id,
    /// else Jaro-Winkler >= 0.9 against display name, first name, or alias).
    pub fn resolve_person(&self, name: &str) -> Option<&str> {
        let t = name.trim();
        if t.is_empty() {
            return None;
        }
        if let Some(p) = self.participants.iter().find(|p| p.person_id == t) {
            return Some(&p.person_id);
        }
        let mut best: Option<(f64, &str)> = None;
        for p in &self.participants {
            let first = p.display_name.split_whitespace().next().unwrap_or("");
            let candidates = std::iter::once(p.display_name.as_str())
                .chain(std::iter::once(first))
                .chain(p.aliases.iter().map(String::as_str));
            let s = crate::text::best_label_similarity(t, candidates);
            if s >= crate::text::LABEL_MATCH_JW && best.is_none_or(|(b, _)| s > b) {
                best = Some((s, p.person_id.as_str()));
            }
        }
        best.map(|(_, id)| id)
    }

    /// Checks internal consistency: version, ids, references, expanded ranges.
    pub fn validate(&self) -> Result<()> {
        if self.golden_version != GOLDEN_VERSION {
            return Err(golden_err(format!(
                "golden_version {} is not {GOLDEN_VERSION}",
                self.golden_version
            )));
        }
        if !self.screen_type_ranges.is_empty() {
            return Err(golden_err(
                "screen_type_ranges are not expanded; run the golden_convert example first",
            ));
        }
        if self.frame_clock.is_some()
            && (self.clocks.screen_types.is_some() || self.clocks.traps.is_some())
        {
            return Err(golden_err(
                "frame_clock and clocks.screen_types/clocks.traps both declare a clock; keep only `clocks`",
            ));
        }
        if let Some(c) = self.frame_clock {
            c.validate("frame_clock")?;
        }
        for section in ClockSection::ALL {
            self.clock(section).validate(section.name())?;
        }
        crate::fixture::validate_board("meeting golden", &self.final_board)?;
        self.validate_notes_aliases()?;
        let people: BTreeSet<&str> = self
            .participants
            .iter()
            .map(|p| p.person_id.as_str())
            .collect();
        if people.len() != self.participants.len() {
            return Err(golden_err("duplicate person_id"));
        }
        let nodes: BTreeSet<&str> = self
            .final_board
            .nodes
            .iter()
            .map(|n| n.id.as_str())
            .collect();
        let check_target = |t: &Target| -> Result<()> {
            let ids: Vec<&String> = match t {
                Target::Node { node } => vec![node],
                Target::Edge { src, dst } => vec![src, dst],
            };
            for id in ids {
                if !nodes.contains(id.as_str()) {
                    return Err(golden_err(format!(
                        "owner target `{id}` is not a final-board node id"
                    )));
                }
            }
            Ok(())
        };
        for a in &self.owners.assignments {
            if !people.contains(a.person_id.as_str()) {
                return Err(golden_err(format!(
                    "unknown person `{}` in assignments",
                    a.person_id
                )));
            }
            check_target(&a.target)?;
        }
        for m in &self.owners.moves {
            if !people.contains(m.person_id.as_str()) {
                return Err(golden_err(format!(
                    "unknown person `{}` in moves",
                    m.person_id
                )));
            }
            check_target(&m.from)?;
            check_target(&m.to)?;
        }
        for a in &self.transcript.action_items {
            if let Some(p) = &a.person_id {
                if !people.contains(p.as_str()) && p != "everyone" {
                    return Err(golden_err(format!("unknown person `{p}` in action items")));
                }
            }
        }
        let mut seen = BTreeSet::new();
        for l in &self.screen_types {
            if !seen.insert(l.t_rep_s.to_bits()) {
                return Err(golden_err(format!(
                    "duplicate screen label at t_rep {}",
                    l.t_rep_s
                )));
            }
        }
        Ok(())
    }

    /// Expands `screen_type_ranges` against a list of golden keyframe times.
    /// Explicit `screen_types` entries win over ranges at the same time; times
    /// covered by no range and no explicit label are left unlabelled.
    pub fn expand_ranges(&mut self, t_reps: &[f64]) -> Result<()> {
        let mut labels = std::mem::take(&mut self.screen_types);
        let explicit: BTreeSet<u64> = labels.iter().map(|l| l.t_rep_s.to_bits()).collect();
        for r in &self.screen_type_ranges {
            if r.to_t_rep_s < r.from_t_rep_s {
                return Err(golden_err(format!(
                    "range {} to {} is inverted",
                    r.from_t_rep_s, r.to_t_rep_s
                )));
            }
            let mut matched = 0;
            for &t in t_reps {
                if t >= r.from_t_rep_s && t <= r.to_t_rep_s {
                    matched += 1;
                    if !explicit.contains(&t.to_bits()) {
                        labels.push(ScreenLabel {
                            t_rep_s: t,
                            screen_type: r.screen_type,
                            confirmed: r.confirmed,
                        });
                    }
                }
            }
            if matched == 0 {
                return Err(golden_err(format!(
                    "range {} to {} covers no keyframe",
                    r.from_t_rep_s, r.to_t_rep_s
                )));
            }
        }
        labels.sort_by(|a, b| a.t_rep_s.total_cmp(&b.t_rep_s));
        let mut seen = BTreeSet::new();
        for l in &labels {
            if !seen.insert(l.t_rep_s.to_bits()) {
                return Err(golden_err(format!(
                    "t_rep {} is covered by two ranges",
                    l.t_rep_s
                )));
            }
        }
        self.screen_types = labels;
        self.screen_type_ranges.clear();
        Ok(())
    }
}

/// Parses a golden file (`.json` or `.toml`) without validating it.
pub fn parse_golden(path: &Path) -> Result<MeetingGolden> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("toml") => read_toml(path),
        _ => read_json(path),
    }
}

/// Loads and validates a golden file.
pub fn load_golden(path: &Path) -> Result<MeetingGolden> {
    let g = parse_golden(path)?;
    g.validate()?;
    Ok(g)
}

/// Finds the golden file in a private root: `golden/meeting_golden.json`, then `.toml`.
pub fn find_golden(private_root: &Path) -> Option<std::path::PathBuf> {
    ["meeting_golden.json", "meeting_golden.toml"]
        .iter()
        .map(|f| private_root.join("golden").join(f))
        .find(|p| p.is_file())
}

/// Reads golden keyframe times from a JSON file: either a bare list of numbers or
/// an object with `keyframes: [{t_rep, ...}]` (the prototype's `keyframes.json`).
pub fn read_keyframe_times(path: &Path) -> Result<Vec<f64>> {
    let v: serde_json::Value = read_json(path)?;
    let list = match &v {
        serde_json::Value::Array(a) => a.clone(),
        serde_json::Value::Object(o) => o
            .get("keyframes")
            .and_then(|k| k.as_array())
            .cloned()
            .ok_or_else(|| EvalError::json(path, "no `keyframes` array"))?,
        _ => return Err(EvalError::json(path, "expected a list or an object")),
    };
    list.iter()
        .map(|item| {
            item.as_f64()
                .or_else(|| item.get("t_rep_s").and_then(|t| t.as_f64()))
                .or_else(|| item.get("t_rep").and_then(|t| t.as_f64()))
                .ok_or_else(|| EvalError::json(path, "keyframe entry without t_rep"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::board::GoldNode;

    #[test]
    fn frame_clock_maps_nominal_grid_names_to_content_time() {
        let g: FrameClock = serde_json::from_value(serde_json::json!({"kind": "pts"})).unwrap();
        assert_eq!(g.content_time(24.0), 24.0);
        assert_eq!(
            FrameClock::default(),
            FrameClock::Pts,
            "no clock is assumed"
        );
        let two: FrameClock = serde_json::from_value(
            serde_json::json!({"kind": "prototype_grid", "interval_s": 2.0}),
        )
        .unwrap();
        assert!((two.content_time(24.0) - 25.95).abs() < 1e-9);
        let five: FrameClock = serde_json::from_value(
            serde_json::json!({"kind": "prototype_grid", "interval_s": 5.0, "lead_s": 0.5}),
        )
        .unwrap();
        assert!((five.content_time(10.0) - 14.5).abs() < 1e-9);
    }

    /// GLM M3, Codex 11, Kimi 8: a golden that declares no clock is scored on PTS,
    /// as before clocks existed; the legacy field covers only the sections it
    /// covered; each section can declare its own.
    #[test]
    fn clocks_are_declared_per_section_and_default_to_pts() {
        let mut g = minimal();
        g.screen_type_ranges.clear();
        let mut v = serde_json::to_value(&g).unwrap();
        let o = v.as_object_mut().unwrap();
        o.remove("frame_clock");
        o.remove("clocks");
        let g: MeetingGolden = serde_json::from_value(v.clone()).unwrap();
        for s in ClockSection::ALL {
            assert_eq!(g.clock(s), FrameClock::Pts, "{}", s.name());
        }
        let grid = FrameClock::PrototypeGrid {
            interval_s: 2.0,
            lead_s: 0.05,
        };
        v["frame_clock"] = serde_json::json!({"kind": "prototype_grid", "interval_s": 2.0});
        let legacy: MeetingGolden = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(legacy.clock(ClockSection::ScreenTypes), grid);
        assert_eq!(legacy.clock(ClockSection::Traps), grid);
        for s in &ClockSection::ALL[2..] {
            assert_eq!(legacy.clock(*s), FrameClock::Pts, "{}", s.name());
        }
        assert_eq!(legacy.declared_clock(ClockSection::OwnerMoves), None);
        legacy.validate().unwrap();
        // both forms at once are ambiguous
        v["clocks"] = serde_json::json!({"screen_types": {"kind": "pts"}});
        let both: MeetingGolden = serde_json::from_value(v.clone()).unwrap();
        assert!(both.validate().is_err());
        // per-section
        v.as_object_mut().unwrap().remove("frame_clock");
        v["clocks"] =
            serde_json::json!({"owner_probes": {"kind": "prototype_grid", "interval_s": 2.0}});
        let per: MeetingGolden = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(per.clock(ClockSection::OwnerProbes), grid);
        assert_eq!(per.clock(ClockSection::OwnerAssignments), FrameClock::Pts);
        assert_eq!(per.clock(ClockSection::OwnerMoves), FrameClock::Pts);
        assert_eq!(per.clock(ClockSection::ScreenTypes), FrameClock::Pts);
        per.validate().unwrap();
        // nonsense grids are rejected
        v["clocks"] = serde_json::json!({"traps": {"kind": "prototype_grid", "interval_s": 0.0}});
        let bad: MeetingGolden = serde_json::from_value(v).unwrap();
        assert!(bad.validate().is_err());
    }

    /// Kimi round-1 B1, Codex round-1 m11: screen labels and traps with no declared
    /// clock score on PTS (the scoring before clocks existed) and are reported.
    #[test]
    fn undeclared_frame_clocks_are_reported() {
        let mut g = minimal();
        g.screen_type_ranges.clear();
        g.screen_types.push(ScreenLabel {
            t_rep_s: 6.0,
            screen_type: ScreenType::Whiteboard,
            confirmed: true,
        });
        assert!(g.clock_warnings().is_empty(), "screen clock declared");
        g.clocks.screen_types = None;
        g.validate().unwrap();
        assert_eq!(g.clock(ClockSection::ScreenTypes), FrameClock::Pts);
        let w = g.clock_warnings();
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("screen_types declare no clock"), "{w:?}");
        g.frame_clock = Some(FrameClock::PrototypeGrid {
            interval_s: 2.0,
            lead_s: 0.05,
        });
        assert!(g.clock_warnings().is_empty());
        g.frame_clock = None;
        g.clocks.screen_types = Some(FrameClock::Pts);
        g.traps.push(Trap {
            t_rep_s: Some(8.0),
            t_from_s: None,
            t_to_s: None,
            kind: "sidebar_chrome".into(),
            note: "synthetic".into(),
        });
        assert_eq!(g.clock_warnings().len(), 1, "traps too");
        g.clocks.traps = Some(FrameClock::Pts);
        assert!(g.clock_warnings().is_empty());
    }

    /// Kimi finding 2: an alias that drops the name of its item is rejected.
    #[test]
    fn aliases_must_name_what_the_item_names() {
        let item = |text: &str, aliases: &[&str]| GoldItem {
            text: text.into(),
            aliases: aliases.iter().map(|a| a.to_string()).collect(),
            person_id: None,
            t_s: None,
        };
        let mut g = minimal();
        g.screen_type_ranges.clear();
        g.transcript.decisions = vec![item(
            "Avery moves from the importer to the dashboard work",
            &["Aveery works on the dashboard instead"],
        )];
        g.validate().unwrap();
        g.transcript.decisions = vec![item(
            "Avery moves from the importer to the dashboard work",
            &["dashboard work instead of the importer"],
        )];
        let e = g.validate().unwrap_err().to_string();
        assert!(e.contains("does not name"), "{e}");
        // the Ledger API entity is named even without capitals
        g.transcript.decisions = vec![item("skip the Ledger step", &["the step is deferred"])];
        assert!(g.validate().is_err());
        g.transcript.decisions = vec![
            item("Ship weekly builds", &[]),
            item("skip the ledger step", &["ledger is deferred"]),
        ];
        g.validate().unwrap();
        // Codex round-1 B1: an alias of the opposite claim is rejected.
        g.transcript.decisions = vec![item("Do not ship Ledger", &["Ship Ledger"])];
        let e = g.validate().unwrap_err().to_string();
        assert!(e.contains("opposite polarity"), "{e}");
        // Final review BLOCKER: an alias that states a name-headed or noun-headed
        // claim and then retracts it is rejected; the retraction is anchored on the
        // predicate, not the first word.
        for (text, alias) in [
            (
                "Avery ships weekly builds",
                "Avery ships weekly builds; Avery does not ship weekly builds",
            ),
            (
                "weekly builds ship on Friday",
                "weekly builds ship on Friday; weekly builds do not ship on Friday",
            ),
            (
                "ship weekly builds",
                "ship weekly builds; we never ship weekly builds",
            ),
        ] {
            // the repaired direction on its own: the alias retracts the text
            assert!(contradicts(text, alias, &g.vocabulary()), "{alias}");
            g.transcript.decisions = vec![item(text, &[alias])];
            let e = g.validate().unwrap_err().to_string();
            assert!(e.contains("opposite polarity"), "{alias}: {e}");
        }
        g.transcript.decisions = vec![item(
            "Avery ships weekly builds",
            &["Avery will ship weekly builds"],
        )];
        g.validate().unwrap();
        // Codex round-1 M10: an alias may shorten an entity name to one of its words,
        // but not drop a second entity.
        g.transcript.decisions = vec![item("Skip Ledger API step", &["Skip Ledger step"])];
        g.validate().unwrap();
        g.final_board.nodes.push(GoldNode {
            id: "queue".into(),
            text: "Orbit Queue".into(),
            aliases: vec![],
            bbox: None,
            core: false,
        });
        g.transcript.decisions = vec![item("Move Ledger to Orbit", &["Move Ledger"])];
        assert!(g.validate().is_err());
        // Codex round-2 M4: a word two entities share does not name either one.
        g.final_board.nodes.push(GoldNode {
            id: "lq".into(),
            text: "Ledger Queue".into(),
            aliases: vec![],
            bbox: None,
            core: false,
        });
        g.transcript.decisions = vec![item("Move the ledger queue", &["Move the Orbit Queue"])];
        assert!(g.validate().is_err());
        g.transcript.decisions = vec![
            item("Ship weekly builds", &[]),
            item("skip the ledger step", &["ledger is deferred"]),
        ];
        g.validate().unwrap();
        assert_eq!(
            g.unkeyed_notes_items(),
            vec!["decision: Ship weekly builds".to_string()]
        );
    }

    fn minimal() -> MeetingGolden {
        MeetingGolden {
            golden_version: GOLDEN_VERSION,
            meeting: "synthetic".into(),
            sources: vec![],
            duration_s: None,
            frame_clock: None,
            clocks: GoldenClocks {
                screen_types: Some(FrameClock::Pts),
                ..GoldenClocks::default()
            },
            participants: vec![
                Participant {
                    person_id: "avery".into(),
                    display_name: "Avery Stone".into(),
                    aliases: vec!["Aveery".into()],
                },
                Participant {
                    person_id: "jordan".into(),
                    display_name: "Jordan Vale".into(),
                    aliases: vec![],
                },
            ],
            screen_types: vec![],
            screen_type_ranges: vec![ScreenRange {
                from_t_rep_s: 2.0,
                to_t_rep_s: 10.0,
                screen_type: ScreenType::Whiteboard,
                confirmed: true,
            }],
            final_board: GoldBoard {
                nodes: vec![GoldNode {
                    id: "api".into(),
                    text: "Ledger API".into(),
                    aliases: vec![],
                    bbox: None,
                    core: true,
                }],
                ..Default::default()
            },
            owners: OwnerTruth {
                probes_s: vec![5.0],
                assignments: vec![Assignment {
                    person_id: "avery".into(),
                    target: Target::Node { node: "api".into() },
                    valid_from_s: 0.0,
                    valid_to_s: None,
                }],
                moves: vec![],
                negatives: vec![],
            },
            traps: vec![],
            static_windows: vec![],
            chrome_terms: vec![],
            transcript: TranscriptTruth::default(),
        }
    }

    #[test]
    fn ranges_must_be_expanded() {
        let mut g = minimal();
        assert!(g.validate().is_err());
        g.screen_types.push(ScreenLabel {
            t_rep_s: 6.0,
            screen_type: ScreenType::Cms,
            confirmed: false,
        });
        g.expand_ranges(&[2.0, 6.0, 10.0, 12.0]).unwrap();
        g.validate().unwrap();
        let got: Vec<(f64, ScreenType, bool)> = g
            .screen_types
            .iter()
            .map(|l| (l.t_rep_s, l.screen_type, l.confirmed))
            .collect();
        // explicit label at 6 wins; 12 is outside every range
        assert_eq!(
            got,
            vec![
                (2.0, ScreenType::Whiteboard, true),
                (6.0, ScreenType::Cms, false),
                (10.0, ScreenType::Whiteboard, true)
            ]
        );
    }

    #[test]
    fn empty_or_overlapping_ranges_rejected() {
        let mut g = minimal();
        assert!(g.clone().expand_ranges(&[50.0]).is_err());
        g.screen_type_ranges.push(ScreenRange {
            from_t_rep_s: 10.0,
            to_t_rep_s: 20.0,
            screen_type: ScreenType::Chat,
            confirmed: true,
        });
        assert!(g.expand_ranges(&[2.0, 10.0]).is_err());
    }

    #[test]
    fn bad_references_rejected() {
        let mut g = minimal();
        g.screen_type_ranges.clear();
        g.owners.assignments[0].target = Target::Node {
            node: "nope".into(),
        };
        assert!(g.validate().is_err());
        let mut g = minimal();
        g.screen_type_ranges.clear();
        g.owners.assignments[0].person_id = "ghost".into();
        assert!(g.validate().is_err());
    }

    #[test]
    fn person_resolution() {
        let g = minimal();
        assert_eq!(g.resolve_person("avery"), Some("avery"));
        assert_eq!(g.resolve_person("Avery"), Some("avery"));
        assert_eq!(g.resolve_person("Aveery"), Some("avery"));
        assert_eq!(g.resolve_person("Jordan Vale"), Some("jordan"));
        assert_eq!(g.resolve_person("Morgan"), None);
        assert_eq!(g.resolve_person(""), None);
    }

    #[test]
    fn json_round_trip() {
        let mut g = minimal();
        g.expand_ranges(&[2.0]).unwrap();
        let text = serde_json::to_string(&g).unwrap();
        let back: MeetingGolden = serde_json::from_str(&text).unwrap();
        assert_eq!(back, g);
    }
}
