//! Deterministic candidate passes and guards around the notes model.
//!
//! Every rule here is general (no meeting-specific words) and has synthetic
//! regression tests:
//!
//! - [`board_facts`]: owner assignments, owner moves and removed owner tags as
//!   candidate decisions and action items, with the event ids to cite.
//! - [`cue_lines`]: transcript sentences with decision or question cue phrases,
//!   which the model must accept (and cite) or leave out.
//! - [`has_commitment`]: whether a line carries a speaker commitment (the
//!   decision precision guard).
//! - [`strip_item_prefix`]: removes narrative prefixes ("The team decided to")
//!   so items read as the decision itself.
//! - [`windows_by_time`]: overlapping time windows, so short exchanges are not
//!   diluted in long windows.

use std::fmt::Write as _;

use super::prompt::{format_line, Window};
use crate::board::{target_text, BoardEvent, BoardExt, BoardStateItem, EventKind};
use crate::named::NamedLine;
use crate::text::{estimate_tokens, mmss, normalize};

/// Owner events of a person near a time (the owner tag appearing or moving).
fn owner_events<'a>(b: &'a BoardStateItem, person_id: &str, near_s: f64) -> Vec<&'a BoardEvent> {
    b.events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::OwnerAssigned | EventKind::OwnerMoved))
        .filter(|e| e.subject == person_id || e.detail.contains(person_id))
        .filter(|e| (e.t_s - near_s).abs() <= 30.0)
        .collect()
}

/// Board facts that are likely decisions or action items: who owns what (from
/// owner tags), who moved to what, and owner tags taken off. Each line names
/// the event ids to cite.
pub fn board_facts(boards: &[BoardStateItem]) -> String {
    let mut s = String::new();
    for b in boards {
        let end = b.end_s();
        for o in &b.owner_assignments {
            let mut evs: Vec<String> = owner_events(b, &o.person_id, o.valid_from_s)
                .iter()
                .map(|e| e.event_id.clone())
                .collect();
            evs.dedup();
            // events can be suppressed by the ink gate; the keyframe that opened
            // the assignment is always recorded
            if evs.is_empty() && !o.opened_at_keyframe.is_empty() {
                evs.push(o.opened_at_keyframe.clone());
            }
            let cite = if evs.is_empty() {
                String::new()
            } else {
                format!(" [cite {}]", evs.join(", "))
            };
            let target = target_text(&o.target);
            match &o.moved_from {
                Some(from) => {
                    let _ = writeln!(
                        s,
                        "- {} moved from {} to {} at {}{cite}",
                        o.display_name,
                        target_text(from),
                        target,
                        mmss(o.valid_from_s)
                    );
                }
                None => {
                    let _ = writeln!(
                        s,
                        "- {} owns {} (owner tag from {}){cite}",
                        o.display_name,
                        target,
                        mmss(o.valid_from_s)
                    );
                }
            }
            if o.valid_to_s < end - 0.5 {
                let _ = writeln!(
                    s,
                    "- {}'s owner tag was taken off {} at {}",
                    o.display_name,
                    target,
                    mmss(o.valid_to_s)
                );
            }
        }
    }
    s
}

/// Phrases that mark a decision or a commitment.
const DECISION_CUES: &[&str] = &[
    "let's",
    "let us",
    "we'll",
    "we will",
    "we're going to",
    "i think we should",
    "we should",
    "we can just",
    "maybe we just",
    "skip",
    "for now",
    "go with",
    "decided",
    "we agree",
    "i'll",
    "i will",
    "you'll",
    "you will",
    "take a stab",
    "i'm going to",
    "i'm gonna",
    "we're gonna",
];
/// Owner tags still valid at the end of the board are task assignments: each
/// becomes an "Own <target>" action item for that person, unless one of the
/// person's action items already names the target. Cites the keyframe that
/// opened the assignment (or its owner event).
pub fn owner_actions(
    boards: &[BoardStateItem],
    existing: &[super::ActionItem],
    people: &[crate::people::Person],
) -> Vec<super::ActionItem> {
    let mut out: Vec<super::ActionItem> = Vec::new();
    for b in boards {
        for o in b.current_owners() {
            let target = target_text(&o.target);
            let key = crate::text::content_tokens(&target);
            let named = |task: &str| {
                let t = crate::text::content_tokens(task);
                key.iter().any(|k| t.contains(k))
            };
            let mine =
                |a: &&super::ActionItem| a.person_id.as_deref() == Some(o.person_id.as_str());
            if existing.iter().filter(mine).any(|a| named(&a.task))
                || out.iter().filter(mine).any(|a| named(&a.task))
            {
                continue;
            }
            let mut event_ids: Vec<String> = owner_events(b, &o.person_id, o.valid_from_s)
                .iter()
                .map(|e| e.event_id.clone())
                .collect();
            event_ids.dedup();
            let keyframe_ids = if o.opened_at_keyframe.is_empty() {
                Vec::new()
            } else {
                vec![o.opened_at_keyframe.clone()]
            };
            let owner = people
                .iter()
                .find(|p| p.person_id == o.person_id)
                .map_or_else(|| o.display_name.clone(), |p| p.display_name.clone());
            out.push(super::ActionItem {
                id: String::new(),
                person_id: Some(o.person_id.clone()),
                owner,
                task: crate::text::sanitize_dashes(&format!("Own {target}")),
                t_s: o.valid_from_s,
                t_end_s: o.valid_from_s,
                evidence: super::Evidence {
                    segment_ids: vec![],
                    event_ids,
                    keyframe_ids,
                },
                quote: None,
            });
        }
    }
    out
}

/// Phrases that mark a question.
const QUESTION_CUES: &[&str] = &[
    "how do we",
    "how would",
    "what about",
    "should we",
    "do we",
    "is it",
    "what does",
    "what do",
    "which",
];

/// Lowercase words (letters, digits and apostrophes) joined by single spaces,
/// with curly apostrophes made straight, padded with a space on each side.
fn cue_text(text: &str) -> String {
    let mapped: String = text
        .chars()
        .map(|c| match c {
            '\u{2019}' | '\u{2018}' => '\'',
            c if c.is_alphanumeric() || c == '\'' => c.to_ascii_lowercase(),
            _ => ' ',
        })
        .collect();
    format!(
        " {} ",
        mapped.split_whitespace().collect::<Vec<_>>().join(" ")
    )
}

/// True when a transcript line carries a speaker commitment or decision cue.
pub fn has_commitment(text: &str) -> bool {
    let n = cue_text(text);
    DECISION_CUES.iter().any(|c| n.contains(&format!(" {c} ")))
}

/// Sentences of a window with decision or question cues, as
/// `segment_id [mm:ss] speaker: sentence (cue)`.
pub fn cue_lines(win: &Window, lines: &[NamedLine], max: usize) -> Vec<String> {
    let mut out = Vec::new();
    for l in win.lines.iter().filter_map(|i| lines.get(*i)) {
        for sentence in split_sentences(&l.text) {
            let n = cue_text(&sentence);
            let question = sentence.trim_end().ends_with('?')
                || QUESTION_CUES
                    .iter()
                    .any(|c| n.starts_with(&format!(" {c} ")));
            let decision = DECISION_CUES.iter().any(|c| n.contains(&format!(" {c} ")));
            let kind = match (decision, question) {
                (true, _) => "decision or action cue",
                (false, true) => "question cue",
                _ => continue,
            };
            // very short fragments ("Okay?") carry no content
            if normalize(&sentence).split(' ').count() < 4 {
                continue;
            }
            out.push(format!(
                "{} [{}] {}: {} ({kind})",
                l.segment_id,
                mmss(l.start_s),
                l.speaker,
                sentence.trim()
            ));
            if out.len() >= max {
                return out;
            }
        }
    }
    out
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        if matches!(c, '.' | '?' | '!') {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Narrative prefixes removed from decisions and action items.
const PREFIXES: &[&str] = &[
    "the team decided to ",
    "the team agreed to ",
    "the team will ",
    "the team has decided to ",
    "the group decided to ",
    "the group agreed to ",
    "it was decided to ",
    "it was agreed to ",
    "we decided to ",
    "we agreed to ",
    "they decided to ",
    "decided to ",
    "agreed to ",
];

/// Removes a narrative prefix and restores the leading capital.
pub fn strip_item_prefix(text: &str) -> String {
    let t = text.trim();
    let lower = t.to_lowercase();
    for p in PREFIXES {
        if lower.starts_with(p) && t.len() > p.len() {
            let rest = &t[p.len()..];
            let mut c = rest.chars();
            return match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => t.to_string(),
            };
        }
    }
    t.to_string()
}

/// Time windows of `window_s` seconds starting every `window_s - overlap_s`
/// seconds, each cut to `budget_tokens`.
pub fn windows_by_time(
    lines: &[NamedLine],
    window_s: f64,
    overlap_s: f64,
    budget_tokens: usize,
) -> Vec<Window> {
    let Some(first) = lines.first() else {
        return Vec::new();
    };
    let end = lines.iter().map(|l| l.end_s).fold(0.0, f64::max);
    let step = (window_s - overlap_s).max(window_s / 4.0).max(1.0);
    let mut out: Vec<Window> = Vec::new();
    let mut start = first.start_s;
    while start < end {
        let mut used = 0;
        let idx: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.start_s >= start && l.start_s < start + window_s)
            .map(|(i, _)| i)
            .take_while(|i| {
                used += estimate_tokens(&format_line(&lines[*i])) + 1;
                used <= budget_tokens
            })
            .collect();
        if !idx.is_empty() && out.last().is_none_or(|w| w.lines != idx) {
            out.push(Window { lines: idx });
        }
        start += step;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{build, EdgeStyle, EventKind};

    fn line(i: usize, t: f64, text: &str) -> NamedLine {
        NamedLine {
            segment_id: format!("seg_{i:05}"),
            start_s: t,
            end_s: t + 4.0,
            person_id: None,
            speaker: "Avery Quinn".into(),
            text: text.into(),
            text_raw: text.into(),
            speaker_confidence: 1.0,
            relabeled: false,
            gap_fill_share: 0.0,
        }
    }

    #[test]
    fn board_facts_list_owners_moves_and_removed_tags() {
        let mut b = build::board("b", 100.0);
        b.nodes = vec![
            build::node("n1", "Ledger Store", 0.0, 100.0, None),
            build::node("n2", "Kiosk App", 0.0, 100.0, None),
        ];
        b.edges = vec![build::edge("e1", &b, "n2", "n1", "", EdgeStyle::Solid)];
        let t1 = build::node_target(&b, "n1");
        let t2 = build::node_target(&b, "n2");
        b.owner_assignments = vec![
            build::owner("mira", "Mira Okafor", t1.clone(), 20.0, 60.0, None),
            build::owner("mira", "Mira Okafor", t2, 60.0, 100.0, Some(t1)),
        ];
        b.events = vec![
            build::event(
                "ev-1",
                EventKind::OwnerAssigned,
                20.0,
                "kf1",
                "mira",
                "Mira on Ledger Store",
            ),
            build::event(
                "ev-2",
                EventKind::OwnerMoved,
                60.0,
                "kf2",
                "mira",
                "Mira to Kiosk App",
            ),
        ];
        let f = board_facts(&[b]);
        assert!(
            f.contains("- Mira Okafor owns Ledger Store (owner tag from 00:20) [cite ev-1]"),
            "{f}"
        );
        assert!(
            f.contains("- Mira Okafor moved from Ledger Store to Kiosk App at 01:00 [cite ev-2]"),
            "{f}"
        );
        assert!(
            f.contains("owner tag was taken off Ledger Store at 01:00"),
            "{f}"
        );
        assert!(board_facts(&[build::board("empty", 10.0)]).is_empty());
        // no owner events (suppressed): cite the keyframe that opened the tag
        let mut b2 = build::board("b2", 100.0);
        b2.nodes = vec![build::node("n1", "Ledger Store", 0.0, 100.0, None)];
        let mut o = build::owner(
            "rohan",
            "Rohan Dasgupta",
            build::node_target(&b2, "n1"),
            10.0,
            100.0,
            None,
        );
        o.opened_at_keyframe = "kf_000010".into();
        b2.owner_assignments = vec![o];
        assert!(board_facts(&[b2]).contains("[cite kf_000010]"));
    }

    #[test]
    fn current_owner_tags_become_actions_unless_already_covered() {
        let mut b = build::board("b", 100.0);
        b.nodes = vec![
            build::node("n1", "Ledger Store", 0.0, 100.0, None),
            build::node("n2", "Kiosk App", 0.0, 100.0, None),
        ];
        let mut o1 = build::owner(
            "mira",
            "Mira Okafor",
            build::node_target(&b, "n1"),
            10.0,
            100.0,
            None,
        );
        o1.opened_at_keyframe = "kf_000010".into();
        let o2 = build::owner(
            "rohan",
            "Rohan Dasgupta",
            build::node_target(&b, "n2"),
            10.0,
            100.0,
            None,
        );
        // an owner tag taken off before the end is not a current assignment
        let o3 = build::owner(
            "avery",
            "Avery Quinn",
            build::node_target(&b, "n2"),
            5.0,
            40.0,
            None,
        );
        b.owner_assignments = vec![o1, o2, o3];
        let existing = vec![super::super::ActionItem {
            id: "a1".into(),
            person_id: Some("rohan".into()),
            owner: "Rohan Dasgupta".into(),
            task: "Build the kiosk flow".into(),
            t_s: 1.0,
            t_end_s: 1.0,
            evidence: Default::default(),
            quote: None,
        }];
        let add = owner_actions(&[b], &existing, &[]);
        assert_eq!(add.len(), 1, "{add:?}");
        assert_eq!(add[0].person_id.as_deref(), Some("mira"));
        assert_eq!(add[0].task, "Own Ledger Store");
        assert_eq!(add[0].evidence.keyframe_ids, vec!["kf_000010"]);
    }

    #[test]
    fn cues_find_decisions_and_questions_only() {
        let lines = vec![
            line(0, 0.0, "Let's skip the ledger import for now."),
            line(1, 5.0, "How do we pick the layout for small screens?"),
            line(2, 10.0, "The weather was nice. Okay?"),
            line(3, 15.0, "I'll write up where I left things tonight."),
        ];
        let w = Window {
            lines: vec![0, 1, 2, 3],
        };
        let c = cue_lines(&w, &lines, 10);
        assert_eq!(c.len(), 3, "{c:?}");
        assert!(c[0].starts_with("seg_00000") && c[0].ends_with("(decision or action cue)"));
        assert!(c[1].ends_with("(question cue)"));
        assert!(c[2].starts_with("seg_00003"));
        assert_eq!(cue_lines(&w, &lines, 1).len(), 1);
    }

    #[test]
    fn commitment_cues() {
        assert!(has_commitment(
            "Maybe we just use random ordering for the demo"
        ));
        assert!(has_commitment("Let's skip that step for now."));
        assert!(has_commitment("I'll take the backend."));
        assert!(!has_commitment("The importer reads the ledger nightly."));
        assert!(!has_commitment("Skipper is the name of the bot."));
    }

    #[test]
    fn narrative_prefixes_are_removed() {
        assert_eq!(
            strip_item_prefix("The team decided to skip the importer"),
            "Skip the importer"
        );
        assert_eq!(
            strip_item_prefix("It was agreed to ship weekly"),
            "Ship weekly"
        );
        assert_eq!(strip_item_prefix("Skip the importer"), "Skip the importer");
    }

    #[test]
    fn time_windows_overlap_and_cover_everything() {
        let lines: Vec<NamedLine> = (0..60)
            .map(|i| line(i, i as f64 * 10.0, "a short line"))
            .collect();
        let w = windows_by_time(&lines, 120.0, 30.0, 100_000);
        assert!(w.len() >= 6);
        let covered: std::collections::BTreeSet<usize> =
            w.iter().flat_map(|x| x.lines.clone()).collect();
        assert_eq!(covered.len(), 60);
        // consecutive windows share 30 s of lines
        let shared = w[0].lines.iter().filter(|i| w[1].lines.contains(i)).count();
        assert_eq!(shared, 3);
        assert!(windows_by_time(&[], 60.0, 10.0, 100).is_empty());
    }
}
