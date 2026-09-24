//! Deterministic candidate passes and guards around the notes model.
//!
//! Every rule here is general (no meeting-specific words) and has synthetic
//! regression tests:
//!
//! - [`board_facts`]: owner assignments, owner moves and removed owner tags as
//!   candidate decisions and action items, with the ids to cite.
//! - [`cue_lines`]: transcript sentences with decision, action or question cue
//!   phrases (recall cues, hedges included), which the model must accept
//!   (citing them) or leave out. Interrogative sentences are question cues;
//!   first-person futures count only when they name a work task.
//! - [`has_commitment`]: the decision precision guard: a sentence with an
//!   assertive commitment cue (a separate, stricter list), not a question and
//!   not hedged.
//! - [`owner_actions`]: owner tags still valid at the end become "Own <target>"
//!   action items when the transcript corroborates them; they pass the same
//!   validation as model items.
//! - [`strip_item_prefix`]: removes narrative prefixes ("The team decided to").
//! - [`windows_by_time`]: overlapping time windows.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use super::prompt::{format_line, Window};
use super::validate::{
    check_with, is_personal_activity, is_task_verb, CheckOptions, Corpus, DraftItem, Section,
};
use super::{ActionItem, Evidence};
use crate::board::{
    target_id, target_text, BoardEvent, BoardExt, BoardStateItem, EventKind, OwnerAssignment,
};
use crate::named::NamedLine;
use crate::text::{content_tokens, estimate_tokens, mmss, sanitize_dashes};

/// Owner events of an assignment: the owner tag of this person appearing on,
/// or moving to, this target near the assignment's start.
fn owner_events<'a>(b: &'a BoardStateItem, o: &OwnerAssignment) -> Vec<&'a BoardEvent> {
    let tid = target_id(&o.target);
    let ttext = target_text(&o.target).to_lowercase();
    b.events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::OwnerAssigned | EventKind::OwnerMoved))
        .filter(|e| e.subject == o.person_id || e.detail.contains(&o.person_id))
        .filter(|e| e.detail.contains(tid) || e.detail.to_lowercase().contains(&ttext))
        .filter(|e| (e.t_s - o.valid_from_s).abs() <= 30.0)
        .collect()
}

/// Ids that show an assignment: its owner events, else the keyframe that opened it.
fn assignment_ids(b: &BoardStateItem, o: &OwnerAssignment) -> (Vec<String>, Vec<String>) {
    let mut events: Vec<String> = owner_events(b, o)
        .iter()
        .map(|e| e.event_id.clone())
        .collect();
    events.dedup();
    let keyframes = if events.is_empty() && !o.opened_at_keyframe.is_empty() {
        vec![o.opened_at_keyframe.clone()]
    } else {
        Vec::new()
    };
    (events, keyframes)
}

/// Board facts that are likely decisions or action items: who owns what (from
/// owner tags), who moved to what, and owner tags taken off. Each line names
/// the ids to cite.
pub fn board_facts(boards: &[BoardStateItem]) -> String {
    let mut s = String::new();
    for b in boards {
        let end = b.end_s();
        for o in &b.owner_assignments {
            let (events, keyframes) = assignment_ids(b, o);
            let ids: Vec<String> = events.into_iter().chain(keyframes).collect();
            let cite = if ids.is_empty() {
                String::new()
            } else {
                format!(" [cite {}]", ids.join(", "))
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

/// Recall cues for the candidate pass: decisions, plans and assignments,
/// including hedged forms ("maybe we just", "we should") the model may reject.
const RECALL_CUES: &[&str] = &[
    "let's",
    "let us",
    "we'll",
    "we will",
    "we're going to",
    "we are going to",
    "we're gonna",
    "i think we should",
    "we should",
    "we can just",
    "maybe we just",
    "skip",
    "for now",
    "go with",
    "decided",
    "we agree",
    "you'll",
    "you will",
    "take a stab",
];
/// First-person futures: cues only when the rest names a work task.
const FIRST_PERSON_FUTURES: &[&str] = &[
    "i'll",
    "i will",
    "i'm going to",
    "i'm gonna",
    "i am going to",
];
/// Assertive commitment cues for the precision guard (no hedges, no temporal
/// scoping such as "for now").
const COMMIT_CUES: &[&str] = &[
    "let's",
    "let us",
    "we'll",
    "we will",
    "i'll",
    "i will",
    "go with",
    "decided",
    "we are going to",
    "we're going to",
];
/// Decision verbs that commit when the group is the subject ("we keep the API
/// on REST", "we're switching to weekly builds"). Status verbs ("we use", "we
/// have") are not here: they describe, they do not decide.
const DECISION_VERBS: &[&str] = &[
    "keep",
    "stick",
    "stay",
    "switch",
    "move",
    "drop",
    "defer",
    "ship",
    "skip",
    "pick",
    "choose",
    "adopt",
    "keeping",
    "sticking",
    "staying",
    "switching",
    "moving",
    "dropping",
    "deferring",
    "shipping",
    "skipping",
    "picking",
    "choosing",
    "adopting",
];

fn group_decision(n: &str) -> bool {
    let w: Vec<&str> = n.split_whitespace().collect();
    w.windows(2)
        .any(|p| matches!(p[0], "we" | "we're") && DECISION_VERBS.contains(&p[1]))
}

/// Hedges that make a sentence tentative rather than a commitment.
const HEDGES: &[&str] = &[
    "maybe", "perhaps", "i think", "should", "might", "probably", "could", "not sure", "if we",
];
/// Phrases that open a question.
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

fn has_any(n: &str, cues: &[&str]) -> bool {
    cues.iter().any(|c| n.contains(&format!(" {c} ")))
}

fn is_question(sentence: &str, n: &str) -> bool {
    sentence.trim_end().ends_with('?')
        || QUESTION_CUES
            .iter()
            .any(|c| n.starts_with(&format!(" {c} ")))
}

/// The words after the first first-person future cue name a work task: a task
/// verb within the next three words and no personal activity.
fn first_person_work(n: &str) -> bool {
    FIRST_PERSON_FUTURES.iter().any(|c| {
        let needle = format!(" {c} ");
        n.find(&needle).is_some_and(|i| {
            let rest = &n[i + needle.len()..];
            let verb = rest
                .split_whitespace()
                .take(3)
                .any(|w| is_task_verb(w.trim_matches('\'')));
            verb && !is_personal_activity(rest)
        })
    })
}

/// True when a line holds a sentence with an assertive commitment: a
/// commitment cue, not a question, and not hedged.
pub fn has_commitment(text: &str) -> bool {
    split_sentences(text).iter().any(|s| {
        let n = cue_text(s);
        (has_any(&n, COMMIT_CUES) || group_decision(&n))
            && !is_question(s, &n)
            && !has_any(&n, HEDGES)
    })
}

/// Sentences of a window with cues, as
/// `segment_id [mm:ss] speaker: sentence (kind)`.
pub fn cue_lines(win: &Window, lines: &[NamedLine], max: usize) -> Vec<String> {
    let mut out = Vec::new();
    for l in win.lines.iter().filter_map(|i| lines.get(*i)) {
        for sentence in split_sentences(&l.text) {
            let n = cue_text(&sentence);
            // very short fragments ("Okay?") carry no content
            if n.split_whitespace().count() < 4 {
                continue;
            }
            let kind = if is_question(&sentence, &n) {
                "question cue"
            } else if has_any(&n, RECALL_CUES) || first_person_work(&n) {
                "decision or action cue"
            } else {
                continue;
            };
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

/// How owner-tag actions are corroborated and validated.
pub struct OwnerActionContext<'a> {
    /// Named transcript lines.
    pub lines: &'a [NamedLine],
    /// Citation corpus.
    pub corpus: &'a Corpus,
    /// Validation switches (board support is forced on).
    pub opts: CheckOptions,
    /// Transcript lines this close to the tag's appearance can corroborate it, seconds.
    pub near_s: f64,
}

/// Owner tags still valid at the end of the board become "Own <target>" action
/// items when corroborated: the owner's name or the target is mentioned in the
/// transcript near the tag's appearance, or the tag was seen in two or more
/// keyframes and the owner spoke in the meeting. Each item passes the same
/// validation as model items and cites the board ids plus the corroborating
/// lines. Owners who already have an action naming the target are skipped.
pub fn owner_actions(
    boards: &[BoardStateItem],
    existing: &[ActionItem],
    ctx: &OwnerActionContext<'_>,
) -> Vec<ActionItem> {
    let opts = CheckOptions {
        board_support: true,
        ..ctx.opts
    };
    let mut out: Vec<ActionItem> = Vec::new();
    for b in boards {
        for o in b.current_owners() {
            let target = sanitize_dashes(&target_text(&o.target));
            let key = content_tokens(&target);
            let named = |a: &ActionItem| {
                a.person_id.as_deref() == Some(o.person_id.as_str()) && {
                    let t = content_tokens(&a.task);
                    key.iter().any(|k| t.contains(k))
                }
            };
            if existing.iter().chain(out.iter()).any(named) {
                continue;
            }
            // corroboration
            let first = o
                .display_name
                .split_whitespace()
                .next()
                .unwrap_or(&o.display_name)
                .to_lowercase();
            let mentions: Vec<String> = ctx
                .lines
                .iter()
                .filter(|l| (l.start_s - o.valid_from_s).abs() <= ctx.near_s)
                .filter(|l| {
                    let toks: BTreeSet<String> = content_tokens(&l.text).into_iter().collect();
                    toks.contains(&first) || key.iter().any(|k| toks.contains(k))
                })
                .map(|l| l.segment_id.clone())
                .collect();
            let keyframes: BTreeSet<&str> =
                o.sightings.iter().map(|s| s.keyframe_id.as_str()).collect();
            let spoke = ctx
                .lines
                .iter()
                .any(|l| l.person_id.as_deref() == Some(o.person_id.as_str()));
            if mentions.is_empty() && !(keyframes.len() >= 2 && spoke) {
                continue;
            }
            let (event_ids, keyframe_ids) = assignment_ids(b, o);
            let mut segment_ids = mentions;
            segment_ids.dedup();
            segment_ids.truncate(3);
            let item = DraftItem {
                owner: o.display_name.clone(),
                task: format!("Own {target}"),
                segment_ids,
                event_ids,
                keyframe_ids,
                ..DraftItem::default()
            };
            let Ok(c) = check_with(Section::ActionItems, &item, ctx.corpus, &opts) else {
                continue;
            };
            if c.evidence.is_empty() {
                continue;
            }
            for owner in c.owners {
                out.push(ActionItem {
                    id: String::new(),
                    person_id: owner.person_id,
                    owner: owner.name,
                    task: c.text.clone(),
                    t_s: o.valid_from_s,
                    t_end_s: c.t_end_s.max(o.valid_from_s),
                    evidence: Evidence {
                        segment_ids: c.evidence.segment_ids.clone(),
                        event_ids: c.evidence.event_ids.clone(),
                        keyframe_ids: c.evidence.keyframe_ids.clone(),
                    },
                    quote: None,
                });
            }
        }
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

/// Removes a narrative prefix and restores the leading capital. Compares
/// character by character, so slicing always lands on a char boundary.
pub fn strip_item_prefix(text: &str) -> String {
    let t = text.trim();
    for p in PREFIXES {
        let mut ti = t.char_indices();
        let mut matched = true;
        for pc in p.chars() {
            match ti.next() {
                Some((_, tc)) if tc.to_lowercase().eq(pc.to_lowercase()) => {}
                _ => {
                    matched = false;
                    break;
                }
            }
        }
        if !matched {
            continue;
        }
        let rest = ti.next().map_or("", |(i, _)| &t[i..]);
        let mut c = rest.chars();
        return match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => t.to_string(),
        };
    }
    t.to_string()
}

/// Time windows of `window_s` seconds starting every `window_s - overlap_s`
/// seconds, each cut to `budget_tokens` (each window's first fresh line is
/// always included, as in the token windows, so no line is lost). Returns no
/// windows when `window_s` is not positive.
pub fn windows_by_time(
    lines: &[NamedLine],
    window_s: f64,
    overlap_s: f64,
    budget_tokens: usize,
) -> Vec<Window> {
    let Some(first) = lines.first() else {
        return Vec::new();
    };
    if !window_s.is_finite() || window_s <= 0.0 {
        return Vec::new();
    }
    let end = lines.iter().map(|l| l.end_s).fold(0.0, f64::max);
    let step = (window_s - overlap_s.max(0.0)).max(window_s / 4.0);
    let mut covered = vec![false; lines.len()];
    let mut out: Vec<Window> = Vec::new();
    let mut start = first.start_s;
    while start < end {
        let in_range: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.start_s >= start && l.start_s < start + window_s)
            .map(|(i, _)| i)
            .collect();
        let first_fresh = in_range.iter().copied().find(|i| !covered[*i]);
        let mut used = 0;
        let mut idx = Vec::new();
        for i in in_range {
            let cost = estimate_tokens(&format_line(&lines[i])) + 1;
            if used + cost > budget_tokens && Some(i) != first_fresh {
                if first_fresh.is_some_and(|f| i < f) {
                    // earlier overlap lines may be dropped to make room
                    continue;
                }
                break;
            }
            used += cost;
            idx.push(i);
        }
        for i in &idx {
            covered[*i] = true;
        }
        if !idx.is_empty() && out.last().is_none_or(|w| w.lines != idx) {
            out.push(Window { lines: idx });
        }
        start += step;
    }
    // lines a budget cut still left out start windows of their own
    let mut i = 0;
    while i < lines.len() {
        if covered[i] {
            i += 1;
            continue;
        }
        let mut used = 0;
        let mut idx = Vec::new();
        while i < lines.len() && !covered[i] {
            let cost = estimate_tokens(&format_line(&lines[i])) + 1;
            if !idx.is_empty() && used + cost > budget_tokens {
                break;
            }
            used += cost;
            idx.push(i);
            covered[i] = true;
            i += 1;
        }
        out.push(Window { lines: idx });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::KeyframeTimes;
    use crate::board::{build, EdgeStyle, EventKind};
    use crate::people::AliasTable;

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
            // same person, other target, within 30 s of the move: not cited for it
            build::event(
                "ev-3",
                EventKind::OwnerAssigned,
                58.0,
                "kf2",
                "mira",
                "Mira on Ledger Store",
            ),
        ];
        let f = board_facts(&[b]);
        assert!(
            f.contains("- Mira Okafor owns Ledger Store (owner tag from 00:20) [cite ev-1]"),
            "{f}"
        );
        assert!(
            f.contains("- Mira Okafor moved from Ledger Store to Kiosk App at 01:00 [cite ev-2]\n"),
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

    fn sighting(kf: &str) -> glassrip_meeting::consolidate::owners::OwnerSighting {
        glassrip_meeting::consolidate::owners::OwnerSighting {
            keyframe_id: kf.into(),
            t_start_s: 0.0,
            t_end_s: 1.0,
            name_raw: String::new(),
            target: None,
            anchor: glassrip_meeting::consolidate::owners::AnchorKind::GeometryNode,
            tag: 0,
        }
    }

    fn owner_board() -> BoardStateItem {
        let mut b = build::board("b", 300.0);
        b.nodes = vec![
            build::node("n1", "Ledger Store", 0.0, 300.0, None),
            build::node("n2", "Kiosk App", 0.0, 300.0, None),
            build::node("n3", "Badge Printer", 0.0, 300.0, None),
        ];
        let mut o1 = build::owner(
            "mira-okafor",
            "Mira Okafor",
            build::node_target(&b, "n1"),
            100.0,
            300.0,
            None,
        );
        o1.opened_at_keyframe = "kf_000100".into();
        o1.sightings = vec![sighting("kf_000100")];
        let mut o2 = build::owner(
            "rohan-dasgupta",
            "Rohan Dasgupta",
            build::node_target(&b, "n2"),
            100.0,
            300.0,
            None,
        );
        o2.opened_at_keyframe = "kf_000100".into();
        o2.sightings = vec![sighting("kf_000100")];
        // a participant's name used as a label on a diagram box: one sighting,
        // never mentioned, and this participant never spoke
        let mut o3 = build::owner(
            "avery-quinn",
            "Avery Quinn",
            build::node_target(&b, "n3"),
            100.0,
            300.0,
            None,
        );
        o3.opened_at_keyframe = "kf_000100".into();
        o3.sightings = vec![sighting("kf_000100")];
        // a tag taken off before the end is not a current assignment
        let o4 = build::owner(
            "mira-okafor",
            "Mira Okafor",
            build::node_target(&b, "n2"),
            5.0,
            40.0,
            None,
        );
        b.owner_assignments = vec![o1, o2, o3, o4];
        b
    }

    #[test]
    fn owner_tags_become_actions_only_when_corroborated_and_valid() {
        let b = owner_board();
        let mut lines = vec![
            line(0, 90.0, "Mira, can you look after the ledger side?"),
            line(1, 400.0, "The kiosk looks fine to me."),
        ];
        lines[1].person_id = Some("rohan-dasgupta".into());
        let corpus = Corpus::new(
            &lines,
            std::slice::from_ref(&b),
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Mira Okafor", "Rohan Dasgupta", "Avery Quinn"]),
        );
        let ctx = OwnerActionContext {
            lines: &lines,
            corpus: &corpus,
            opts: CheckOptions::default(),
            near_s: 120.0,
        };
        let add = owner_actions(std::slice::from_ref(&b), &[], &ctx);
        // Mira: named near the tag. Rohan: one sighting and his mention is far
        // from the tag. Avery: a label with no transcript support.
        let got: Vec<(&str, &str)> = add
            .iter()
            .map(|a| (a.owner.as_str(), a.task.as_str()))
            .collect();
        assert_eq!(got, vec![("Mira Okafor", "Own Ledger Store")], "{add:?}");
        assert_eq!(add[0].evidence.segment_ids, vec!["seg_00000"]);
        assert_eq!(add[0].evidence.keyframe_ids, vec!["kf_000100"]);
        // persistence plus the owner speaking corroborates too
        let mut b2 = b.clone();
        b2.owner_assignments[1].sightings = vec![sighting("kf_000100"), sighting("kf_000120")];
        let add = owner_actions(std::slice::from_ref(&b2), &[], &ctx);
        assert!(add.iter().any(|a| a.owner == "Rohan Dasgupta"), "{add:?}");
        assert!(!add.iter().any(|a| a.owner == "Avery Quinn"));
        // an owner who already has an action naming the target gets no duplicate
        let existing = vec![ActionItem {
            id: "a1".into(),
            person_id: Some("mira-okafor".into()),
            owner: "Mira Okafor".into(),
            task: "Migrate the ledger records".into(),
            t_s: 1.0,
            t_end_s: 1.0,
            evidence: Evidence::default(),
            quote: None,
        }];
        assert!(owner_actions(std::slice::from_ref(&b), &existing, &ctx).is_empty());
        // an owner who is not a participant fails validation
        let corpus2 = Corpus::new(
            &lines,
            std::slice::from_ref(&b),
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Rohan Dasgupta"]),
        );
        let ctx2 = OwnerActionContext {
            corpus: &corpus2,
            ..ctx
        };
        assert!(owner_actions(std::slice::from_ref(&b), &[], &ctx2).is_empty());
    }

    #[test]
    fn cues_find_decisions_actions_and_questions_only() {
        let lines = vec![
            line(0, 0.0, "Let's skip the ledger import for now."),
            line(1, 5.0, "How do we pick the layout for small screens?"),
            line(2, 10.0, "The weather was nice. Okay?"),
            line(3, 15.0, "I'll write up where I left things tonight."),
            line(4, 20.0, "Should we skip the review this week?"),
            line(5, 25.0, "I'm going to grab a coffee before we start."),
            line(6, 30.0, "I'll be back in a minute or so."),
        ];
        let w = Window {
            lines: (0..7).collect(),
        };
        let c = cue_lines(&w, &lines, 10);
        assert_eq!(c.len(), 4, "{c:#?}");
        assert!(c[0].starts_with("seg_00000") && c[0].ends_with("(decision or action cue)"));
        assert!(c[1].ends_with("(question cue)"));
        assert!(c[2].starts_with("seg_00003") && c[2].ends_with("(decision or action cue)"));
        // interrogative wins over the "skip" cue
        assert!(c[3].starts_with("seg_00004") && c[3].ends_with("(question cue)"));
        assert_eq!(cue_lines(&w, &lines, 1).len(), 1);
    }

    #[test]
    fn the_guard_accepts_only_assertive_commitments() {
        assert!(has_commitment("Let's skip that step."));
        assert!(has_commitment("I'll take the backend."));
        assert!(has_commitment("Okay. We'll go with the second layout."));
        assert!(has_commitment(
            "We keep the ledger API on REST for the pilot."
        ));
        assert!(has_commitment("We're switching to weekly builds."));
        assert!(!has_commitment("We use Postgres today."));
        assert!(!has_commitment("Maybe we keep the old API?"));
        // the reviewer's constructions: status with "for now", a hedge, a question
        assert!(!has_commitment("For now the demo runs on the old cluster."));
        assert!(!has_commitment("Maybe we just park that decision."));
        assert!(!has_commitment("Should we skip the review?"));
        assert!(!has_commitment("I think we should move the importer."));
        assert!(!has_commitment("Will we ship on Friday?"));
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
        // non-ASCII text never splits a character
        assert_eq!(
            strip_item_prefix("\u{212a}eep the importer"),
            "\u{212a}eep the importer"
        );
        assert_eq!(
            strip_item_prefix("We decided to \u{e9}tudier it"),
            "\u{c9}tudier it"
        );
    }

    #[test]
    fn time_windows_overlap_and_cover_everything() {
        let lines: Vec<NamedLine> = (0..60)
            .map(|i| line(i, i as f64 * 10.0, "a short line"))
            .collect();
        let w = windows_by_time(&lines, 120.0, 30.0, 100_000);
        assert!(w.len() >= 6);
        let covered: BTreeSet<usize> = w.iter().flat_map(|x| x.lines.clone()).collect();
        assert_eq!(covered.len(), 60);
        // consecutive windows share 30 s of lines
        let shared = w[0].lines.iter().filter(|i| w[1].lines.contains(i)).count();
        assert_eq!(shared, 3);
        assert!(windows_by_time(&[], 60.0, 10.0, 100).is_empty());
        // invalid lengths make no windows
        assert!(windows_by_time(&lines, 0.0, 10.0, 100).is_empty());
        assert!(windows_by_time(&lines, -5.0, 10.0, 100).is_empty());
        // a line longer than the budget still reaches the model
        let mut long = lines.clone();
        long[7].text = "word ".repeat(400);
        let w = windows_by_time(&long, 120.0, 30.0, 60);
        let covered: BTreeSet<usize> = w.iter().flat_map(|x| x.lines.clone()).collect();
        assert_eq!(covered.len(), 60, "{w:?}");
    }
}
