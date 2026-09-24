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
        crate::fixture::validate_board("meeting golden", &self.final_board)?;
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

    fn minimal() -> MeetingGolden {
        MeetingGolden {
            golden_version: GOLDEN_VERSION,
            meeting: "synthetic".into(),
            sources: vec![],
            duration_s: None,
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
