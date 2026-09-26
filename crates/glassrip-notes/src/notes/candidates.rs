//! Deterministic candidate passes and guards around the notes model.
//!
//! Every rule here is general (no meeting-specific words) and has synthetic
//! regression tests:
//!
//! - [`board_facts`]: owner tags still on the board at the end as candidate
//!   action items for their owner, owner moves as candidate decisions, and
//!   owner tags taken off as context, with the ids to cite.
//! - [`cue_lines`]: transcript sentences with decision, action or question cue
//!   phrases (recall cues, hedges included), which the model must accept
//!   (citing them) or leave out. Interrogative sentences are question cues;
//!   first-person futures count only when they name a work task.
//! - [`has_commitment`]: the decision precision guard, judged per clause: a
//!   subject (a group cue, "we" with a decision verb, or a first-person future
//!   with a task verb) and a verb with a real object (no flow continuations
//!   or navigation objects); not a question, and the committing clause is not
//!   hedged, conditional or a personal activity.
//! - [`owner_actions`]: owner tags still valid at the end become "Own <target>"
//!   action items when the owner speaks about the target nearby, or someone
//!   else names the owner about the target nearby; they pass the same
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
    OwnerTarget,
};
use crate::named::NamedLine;
use crate::text::{
    content_tokens, estimate_tokens, is_stopword, mmss, names_target, sanitize_dashes, tokens,
};

/// Owner events of an assignment: the owner tag of this person appearing on,
/// or moving to, this target near the assignment's start. Matched exactly: the
/// event's kind agrees with the assignment (a move or not), and its structured
/// target (and, for a move, the target it left) is the assignment's by id. An
/// event without a structured target (a state written before events carried
/// one) matches no assignment: its detail text cannot tell two targets with the
/// same name apart, and the assignment is still cited by the keyframe that
/// opened it.
pub(crate) fn owner_events<'a>(b: &'a BoardStateItem, o: &OwnerAssignment) -> Vec<&'a BoardEvent> {
    let want = if o.moved_from.is_some() {
        EventKind::OwnerMoved
    } else {
        EventKind::OwnerAssigned
    };
    b.events
        .iter()
        .filter(|e| e.kind == want && e.subject == o.person_id)
        .filter(|e| {
            e.owner_target
                .as_ref()
                .is_some_and(|t| same_target(t, &o.target))
                && match (&e.owner_from, &o.moved_from) {
                    (None, None) => true,
                    (Some(a), Some(b)) => same_target(a, b),
                    _ => false,
                }
        })
        .filter(|e| (e.t_s - o.valid_from_s).abs() <= 30.0)
        .collect()
}

/// Two owner targets are the same board element: the same kind and id.
fn same_target(a: &OwnerTarget, b: &OwnerTarget) -> bool {
    matches!(
        (a, b),
        (OwnerTarget::Node { .. }, OwnerTarget::Node { .. })
            | (OwnerTarget::Edge { .. }, OwnerTarget::Edge { .. })
    ) && target_id(a) == target_id(b)
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

/// What a board fact is offered to the notes model as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BoardFactKind {
    /// An owner tag on a box or link: the owner is responsible for it, so it
    /// is a candidate action item for that owner (not a decision).
    ActionItem,
    /// An owner tag moved from one target to another: a change the group made,
    /// so it is a candidate decision.
    Decision,
    /// An owner tag taken off: context for the model, not an item by itself.
    Context,
}

/// One board fact with the ids to cite.
#[derive(Debug, Clone, PartialEq)]
pub struct BoardFact {
    /// Candidate kind.
    pub kind: BoardFactKind,
    /// The fact, as one line of text (no ids).
    pub text: String,
    /// Event or keyframe ids that show it (may be empty).
    pub ids: Vec<String>,
}

/// Board facts from owner tags, classified: who owns what at the end
/// (candidate action items for the owner), who moved from what to what
/// (candidate decisions), and owner tags taken off before the end (context).
pub fn board_fact_list(boards: &[BoardStateItem]) -> Vec<BoardFact> {
    let mut out = Vec::new();
    for b in boards {
        let end = b.end_s();
        for o in &b.owner_assignments {
            let (events, keyframes) = assignment_ids(b, o);
            let ids: Vec<String> = events.into_iter().chain(keyframes).collect();
            let target = target_text(&o.target);
            let ended = o.valid_to_s < end - 0.5;
            if let Some(from) = &o.moved_from {
                out.push(BoardFact {
                    kind: BoardFactKind::Decision,
                    text: format!(
                        "{} moved from {} to {} at {}",
                        o.display_name,
                        target_text(from),
                        target,
                        mmss(o.valid_from_s)
                    ),
                    ids: ids.clone(),
                });
            }
            if ended {
                // an owner tag no longer on the board is not a current task
                out.push(BoardFact {
                    kind: BoardFactKind::Context,
                    text: format!(
                        "{}'s owner tag on {} (from {}) was taken off at {}",
                        o.display_name,
                        target,
                        mmss(o.valid_from_s),
                        mmss(o.valid_to_s)
                    ),
                    ids,
                });
            } else {
                out.push(BoardFact {
                    kind: BoardFactKind::ActionItem,
                    text: format!(
                        "{} owns {} (owner tag from {})",
                        o.display_name,
                        target,
                        mmss(o.valid_from_s)
                    ),
                    ids,
                });
            }
        }
    }
    out
}

/// Heading of each group in [`board_facts`].
fn fact_heading(kind: BoardFactKind) -> &'static str {
    match kind {
        BoardFactKind::ActionItem => {
            "Action item candidates (an owner tag says who is responsible: an action item for that owner, not a decision):"
        }
        BoardFactKind::Decision => {
            "Decision candidates (an owner tag moved from one target to another: a change the group made):"
        }
        BoardFactKind::Context => "Context only (not an item by itself):",
    }
}

/// [`board_fact_list`] as prompt text, grouped by candidate kind, each line
/// naming the ids to cite. Empty when the boards have no owner tags.
pub fn board_facts(boards: &[BoardStateItem]) -> String {
    let facts = board_fact_list(boards);
    let mut s = String::new();
    for kind in [
        BoardFactKind::ActionItem,
        BoardFactKind::Decision,
        BoardFactKind::Context,
    ] {
        let group: Vec<&BoardFact> = facts.iter().filter(|f| f.kind == kind).collect();
        if group.is_empty() {
            continue;
        }
        let _ = writeln!(s, "{}", fact_heading(kind));
        for f in group {
            let cite = if f.ids.is_empty() {
                String::new()
            } else {
                format!(" [cite {}]", f.ids.join(", "))
            };
            let _ = writeln!(s, "- {}{cite}", f.text);
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
/// Group commitment cues for the precision guard, as word sequences; the verb
/// follows (after fillers). No hedges and no temporal scoping ("for now").
const GROUP_CUES: &[&[&str]] = &[
    &["let's"],
    &["let", "us"],
    &["we'll"],
    &["we", "will"],
    &["we", "are", "going", "to"],
    &["we're", "going", "to"],
    &["we're", "gonna"],
    &["decided"],
];
/// First-person future cues for the precision guard; a task verb follows
/// within three words.
const FIRST_PERSON_CUES: &[&[&str]] = &[
    &["i'll"],
    &["i", "will"],
    &["i'm", "going", "to"],
    &["i'm", "gonna"],
    &["i", "am", "going", "to"],
];
/// Words between a cue and its verb ("let's just ship", "we'll not change",
/// "decided to move").
const FILLERS: &[&str] = &[
    "just",
    "also",
    "definitely",
    "then",
    "now",
    "actually",
    "still",
    "officially",
    "not",
    "to",
    "all",
];
/// Verbs after a group cue that steer the conversation rather than decide
/// ("let's see", "let's say", "we'll talk about it later").
const NONCOMMITTAL: &[&str] = &[
    "see", "say", "think", "hear", "talk", "chat", "discuss", "wait", "be", "have", "need", "hope",
    "assume", "imagine", "suppose", "look", "recap",
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
    "push",
    "postpone",
    "cancel",
    "go",
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
    "pushing",
    "postponing",
    "cancelling",
    "canceling",
];

/// Words that make a verb phrase a meeting-flow phrase ("we keep going", "we
/// skip ahead", "we drop off", "let's get started") when they come before any
/// object.
const CONTINUATIONS: &[&str] = &[
    "going", "ahead", "along", "forward", "off", "rolling", "started",
];
/// Movement verbs whose "on" means "proceed" ("we move on", "let's go on").
const PROCEED_VERBS: &[&str] = &["move", "moving", "keep", "keeping", "go", "carry"];
/// Particles and prepositions between a verb and its object.
const PARTICLES: &[&str] = &[
    "on", "to", "with", "over", "up", "down", "out", "in", "into", "onto", "at", "for", "from",
    "by", "back", "away", "around", "aside",
];
/// Particles that, after a pronoun, make a phrasal verb ("pick this up",
/// "figure it out", "circle back").
const PHRASAL: &[&str] = &[
    "up", "down", "off", "back", "out", "over", "along", "away", "around", "aside",
];
/// Pronouns that stand in for a decided object ("we ship it").
const PRONOUNS: &[&str] = &["it", "this", "that", "them", "these", "those"];
/// Words that fill an object slot without naming anything decided: time,
/// agenda and navigation words ("we stay on track", "we move to the next
/// slide", "we skip to the end").
const FLOW_WORDS: &[&str] = &[
    "track",
    "now",
    "today",
    "tomorrow",
    "later",
    "next",
    "topic",
    "topics",
    "agenda",
    "item",
    "items",
    "point",
    "thing",
    "things",
    "one",
    "bit",
    "minute",
    "second",
    "moment",
    "time",
    "then",
    "anyway",
    "instead",
    "too",
    "all",
    "everyone",
    "guys",
    "folks",
    "people",
    "okay",
    "right",
    "again",
    "same",
    "quickly",
    "real",
    "quick",
    "slide",
    "slides",
    "end",
    "start",
    "beginning",
    "top",
    "section",
    "question",
    "questions",
    "recap",
    "here",
    "there",
];
/// Words that open a subordinate clause; the guard judges clauses separately.
const SUBORDINATORS: &[&str] = &[
    "because", "though", "although", "before", "after", "while", "since", "unless", "if", "whereas",
];
/// Hedges and hypotheticals that make a clause tentative rather than a
/// commitment.
const HEDGES: &[&str] = &[
    "maybe",
    "perhaps",
    "i think",
    "should",
    "might",
    "probably",
    "could",
    "not sure",
    "if we",
    "let's say",
    "say we",
    "suppose",
    "supposing",
    "imagine",
    "assuming",
    "hypothetically",
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
/// verb within the next three words and no personal activity. Used by the
/// recall cues; the guard applies the stricter [`clause_commits`].
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

fn starts_with_seq(w: &[&str], i: usize, seq: &[&str]) -> bool {
    w.len() >= i + seq.len() && w[i..i + seq.len()] == *seq
}

/// The verb and what follows it name something decided: a content word that
/// is not a flow word, reached before any continuation ("we keep going",
/// "let's move on", "we keep right on going" do not commit), or a pronoun
/// object of a decision or task verb that is not a phrasal particle's ("we
/// ship it" commits, "we pick this up" does not).
fn verb_commits(verb: &str, rest: &[&str]) -> bool {
    if rest.first() == Some(&"on") && PROCEED_VERBS.contains(&verb) {
        return false;
    }
    let pronoun_ok = DECISION_VERBS.contains(&verb) || is_task_verb(verb);
    for (k, t) in rest.iter().enumerate() {
        if CONTINUATIONS.contains(t) {
            return false;
        }
        if PRONOUNS.contains(t) {
            if pronoun_ok && !rest.get(k + 1).is_some_and(|p| PHRASAL.contains(p)) {
                return true;
            }
            continue;
        }
        if !is_stopword(t) && !PARTICLES.contains(t) && !FLOW_WORDS.contains(t) {
            return true;
        }
    }
    false
}

/// One clause holds a commitment. Every leg finds a subject and a verb and
/// then applies the same object test ([`verb_commits`]):
///
/// - a group cue ("let's", "we'll", "we're going to", "decided") and the verb
///   after it, unless the verb steers the conversation ("let's see");
/// - "we" or "we're" and a decision verb ("we keep the API", "we ship it");
/// - a first-person future and a task verb within three words ("I'll write
///   the spec"); "go with" needs one of these subjects too.
fn clause_commits(w: &[&str]) -> bool {
    for i in 0..w.len() {
        for cue in GROUP_CUES {
            if !starts_with_seq(w, i, cue) {
                continue;
            }
            let mut j = i + cue.len();
            if starts_with_seq(w, j, &["go", "ahead", "and"]) {
                j += 3;
            }
            while w.get(j).is_some_and(|t| FILLERS.contains(t)) {
                j += 1;
            }
            if let Some(verb) = w.get(j) {
                if !NONCOMMITTAL.contains(verb) && verb_commits(verb, &w[j + 1..]) {
                    return true;
                }
            }
        }
        if matches!(w[i], "we" | "we're")
            && w.get(i + 1).is_some_and(|v| DECISION_VERBS.contains(v))
            && verb_commits(w[i + 1], &w[i + 2..])
        {
            return true;
        }
        for cue in FIRST_PERSON_CUES {
            if !starts_with_seq(w, i, cue) {
                continue;
            }
            let rest = &w[i + cue.len()..];
            for k in 0..rest.len().min(3) {
                let v = rest[k];
                let go_with = v == "go" && rest.get(k + 1) == Some(&"with");
                if is_task_verb(v) || go_with {
                    if verb_commits(v, &rest[k + 1..]) {
                        return true;
                    }
                    break;
                }
            }
        }
    }
    false
}

/// Clauses of a sentence, as padded cue text: split on , ; : and before
/// subordinators ("because", "though", "before", "if").
fn clauses(sentence: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in sentence.split([',', ';', ':']) {
        let n = cue_text(part);
        let mut cur: Vec<&str> = Vec::new();
        for w in n.split_whitespace() {
            if SUBORDINATORS.contains(&w) && !cur.is_empty() {
                out.push(format!(" {} ", cur.join(" ")));
                cur.clear();
            }
            cur.push(w);
        }
        if !cur.is_empty() {
            out.push(format!(" {} ", cur.join(" ")));
        }
    }
    out
}

fn is_conditional(clause: &str) -> bool {
    clause.starts_with(" if ") || clause.starts_with(" unless ")
}

/// A short clause that only hedges what follows ("maybe, ...", "I think, ...").
fn is_hedge_only(clause: &str) -> bool {
    has_any(clause, HEDGES) && clause.split_whitespace().count() <= 3
}

/// True when a line holds a clause with an assertive commitment
/// ([`clause_commits`]) in a sentence that is not a question. The hedge and
/// personal-activity vetoes apply to the committing clause only, plus a
/// conditional clause next to it or a hedge-only clause before it: "we'll go
/// with the second layout, though we should revisit it" commits, "I'll be
/// right back" and "let's ship it if the tests pass" do not.
pub fn has_commitment(text: &str) -> bool {
    split_sentences(text).iter().any(|s| {
        if is_question(s, &cue_text(s)) {
            return false;
        }
        let cs = clauses(s);
        (0..cs.len()).any(|i| {
            let c = &cs[i];
            let w: Vec<&str> = c.split_whitespace().collect();
            clause_commits(&w)
                && !has_any(c, HEDGES)
                && !is_personal_activity(c)
                && !(i > 0 && (is_conditional(&cs[i - 1]) || is_hedge_only(&cs[i - 1])))
                && !cs.get(i + 1).is_some_and(|n| is_conditional(n))
        })
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

/// First names that are also common English words ("let's mark that", "your
/// bill"): they count as a mention only with the surname, or when addressing
/// someone about the target ("Mark, can you take the kiosk?").
const WORD_NAMES: &[&str] = &[
    "mark", "bill", "drew", "grant", "may", "chase", "pat", "skip", "will", "rose", "art", "bob",
    "frank", "joy", "faith", "hope", "grace", "dawn", "summer", "june", "april", "august", "rich",
    "sandy", "ray", "gene", "jack", "sue", "guy", "rob", "don", "sky", "dean", "pierce", "hunter",
    "cliff", "glen", "wade", "miles", "lane", "reed", "clay", "penny", "ruby", "amber", "iris",
    "lily", "ivy", "holly", "crystal", "robin", "jay", "max", "rusty", "sonny", "sunny", "bud",
    "buck", "cash", "earl", "harmony", "mercy", "page", "paige", "carol", "nick", "ted", "chip",
    "sterling", "win", "early", "major", "royal", "star",
];

/// True when `words` introduce `first` ("I'm Avery", "my name is Avery",
/// "this is Avery").
fn self_introduction(words: &[String], first: &str) -> bool {
    let intros: [&[&str]; 5] = [
        &["im"],
        &["i", "am"],
        &["name", "is"],
        &["this", "is"],
        &["its"],
    ];
    words.iter().enumerate().any(|(i, w)| {
        w == first
            && intros.iter().any(|p| {
                i >= p.len()
                    && words[i - p.len()..i]
                        .iter()
                        .map(String::as_str)
                        .eq(p.iter().copied())
            })
    })
}

/// True when `first` is used to address someone: a comma-, period- or
/// question-delimited piece of the raw text that is only the name ("Mark,
/// can you ...", "..., Mark?").
fn vocative(text: &str, first: &str) -> bool {
    text.split([',', '.', '?', '!', ';', ':'])
        .any(|piece| tokens(piece) == [first])
}

/// Transcript lines within `near_s` of the tag's appearance that corroborate
/// an owner tag:
///
/// - the owner speaking about the target (a distinctive word of it); or
/// - someone else mentioning the owner by name about the target: the target
///   is named in the same line or an adjacent line within the window. The
///   owner's own speech and self-introductions ("I'm Avery") are not
///   mentions, and a first name that is a common word ("mark", "bill")
///   counts only with the surname or when addressing the owner in a line
///   that names the target.
fn corroborating_lines(
    ctx: &OwnerActionContext<'_>,
    o: &OwnerAssignment,
    key: &[String],
) -> Vec<String> {
    let name = tokens(&o.display_name);
    let Some(first) = name.first() else {
        return Vec::new();
    };
    let surname = name.get(1);
    let owner_id = Some(o.person_id.as_str());
    let near = |l: &NamedLine| (l.start_s - o.valid_from_s).abs() <= ctx.near_s;
    let names = |l: &NamedLine| {
        let toks: BTreeSet<String> = content_tokens(&l.text).into_iter().collect();
        names_target(&toks, key)
    };
    let lines = ctx.lines;
    let topical = |i: usize| {
        names(&lines[i])
            || [i.checked_sub(1), Some(i + 1)]
                .into_iter()
                .flatten()
                .filter_map(|j| lines.get(j))
                .any(|l| near(l) && names(l))
    };
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if !near(l) {
            continue;
        }
        let ok = if l.person_id.as_deref() == owner_id {
            names(l)
        } else {
            let words = tokens(&l.text);
            let full =
                surname.is_some_and(|s| words.windows(2).any(|p| &p[0] == first && &p[1] == s));
            if self_introduction(&words, first) {
                false
            } else if WORD_NAMES.contains(&first.as_str()) {
                (full && topical(i)) || (vocative(&l.text, first) && names(l))
            } else {
                words.contains(first) && topical(i)
            }
        };
        if ok {
            out.push(l.segment_id.clone());
        }
    }
    out
}

/// Owner tags still valid at the end of the board become "Own <target>" action
/// items when a transcript line near the tag's appearance corroborates them
/// (see [`corroborating_lines`]): the owner speaking about the target, or
/// someone naming the owner about the target (one generic word such as "app"
/// does not name it, see [`names_target`]). How often the tag was seen and
/// whether the owner spoke elsewhere are not evidence: a name used as a label
/// persists on the board too. Each item passes the same validation as model
/// items and cites the board ids plus the corroborating lines. Owners who
/// already have an action naming the target are skipped.
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
            let owner_id = Some(o.person_id.as_str());
            // an existing action of this owner that names the target (a
            // distinctive word of it; "app" alone names nothing)
            let named = |a: &ActionItem| {
                a.person_id.as_deref() == owner_id && {
                    let t: BTreeSet<String> = content_tokens(&a.task).into_iter().collect();
                    names_target(&t, &key)
                }
            };
            if existing.iter().chain(out.iter()).any(named) {
                continue;
            }
            let mentions = corroborating_lines(ctx, o, &key);
            if mentions.is_empty() {
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
        let (n1, n2) = (build::node_target(&b, "n1"), build::node_target(&b, "n2"));
        for (e, to, from) in [(0, &n1, None), (1, &n2, Some(&n1)), (2, &n1, None)] {
            b.events[e].owner_target = Some(to.clone());
            b.events[e].owner_from = from.cloned();
        }
        let f = board_facts(&[b]);
        assert!(
            f.contains(
                "- Mira Okafor's owner tag on Ledger Store (from 00:20) was taken off at 01:00 [cite ev-1]"
            ),
            "{f}"
        );
        assert!(
            f.contains("- Mira Okafor moved from Ledger Store to Kiosk App at 01:00 [cite ev-2]\n"),
            "{f}"
        );
        assert!(
            f.contains("- Mira Okafor owns Kiosk App (owner tag from 01:00) [cite ev-2]"),
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
    fn owner_tags_are_action_candidates_and_moves_are_decision_candidates() {
        let mut b = build::board("b", 100.0);
        b.nodes = vec![
            build::node("n1", "Ledger Store", 0.0, 100.0, None),
            build::node("n2", "Kiosk App", 0.0, 100.0, None),
            build::node("n3", "Badge Printer", 0.0, 100.0, None),
        ];
        let t1 = build::node_target(&b, "n1");
        let t2 = build::node_target(&b, "n2");
        let t3 = build::node_target(&b, "n3");
        b.owner_assignments = vec![
            build::owner("mira", "Mira Okafor", t1.clone(), 20.0, 60.0, None),
            build::owner("mira", "Mira Okafor", t2, 60.0, 100.0, Some(t1)),
            build::owner("rohan", "Rohan Dasgupta", t3, 30.0, 100.0, None),
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
            build::event(
                "ev-3",
                EventKind::OwnerAssigned,
                30.0,
                "kf1",
                "rohan",
                "Rohan on Badge Printer",
            ),
        ];
        for (e, to, from) in [(0, "n1", None), (1, "n2", Some("n1")), (2, "n3", None)] {
            b.events[e].owner_target = Some(build::node_target(&b, to));
            b.events[e].owner_from = from.map(|f| build::node_target(&b, f));
        }
        let facts = board_fact_list(std::slice::from_ref(&b));
        let kinds: Vec<(BoardFactKind, &str, Vec<&str>)> = facts
            .iter()
            .map(|f| {
                (
                    f.kind,
                    f.text.as_str(),
                    f.ids.iter().map(String::as_str).collect(),
                )
            })
            .collect();
        assert_eq!(
            kinds,
            vec![
                // taken off before the end: context, not a current task
                (
                    BoardFactKind::Context,
                    "Mira Okafor's owner tag on Ledger Store (from 00:20) was taken off at 01:00",
                    vec!["ev-1"]
                ),
                // a move is a decision, and the new owner tag a task
                (
                    BoardFactKind::Decision,
                    "Mira Okafor moved from Ledger Store to Kiosk App at 01:00",
                    vec!["ev-2"]
                ),
                (
                    BoardFactKind::ActionItem,
                    "Mira Okafor owns Kiosk App (owner tag from 01:00)",
                    vec!["ev-2"]
                ),
                (
                    BoardFactKind::ActionItem,
                    "Rohan Dasgupta owns Badge Printer (owner tag from 00:30)",
                    vec!["ev-3"]
                ),
            ]
        );
        // the prompt text groups them: every plain owner tag under the action
        // item candidates, the move under the decision candidates
        let f = board_facts(&[b]);
        let at = |needle: &str| {
            f.find(needle)
                .unwrap_or_else(|| panic!("{needle:?} not in {f}"))
        };
        let actions = at("Action item candidates");
        let decisions = at("Decision candidates");
        let context = at("Context only");
        assert!(actions < at("- Mira Okafor owns Kiosk App"));
        assert!(at("- Rohan Dasgupta owns Badge Printer") < decisions);
        assert!(decisions < at("- Mira Okafor moved from Ledger Store to Kiosk App"));
        assert!(at("- Mira Okafor moved from") < context);
        assert!(context < at("owner tag on Ledger Store (from 00:20) was taken off"));
        assert!(
            !f.contains("owns Ledger Store"),
            "an ended tag is not a task: {f}"
        );
        assert!(f[..decisions].contains("not a decision"), "{f}");
        assert!(!f.contains("likely a decision or an action item"), "{f}");
        // a board with owner tags but no move has no decision group
        let mut plain = build::board("p", 100.0);
        plain.nodes = vec![build::node("n1", "Ledger Store", 0.0, 100.0, None)];
        plain.owner_assignments = vec![build::owner(
            "mira",
            "Mira Okafor",
            build::node_target(&plain, "n1"),
            10.0,
            100.0,
            None,
        )];
        let f = board_facts(&[plain]);
        assert!(f.starts_with("Action item candidates"), "{f}");
        assert!(!f.contains("Decision candidates"), "{f}");
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
            ocr_located: false,
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
        // persistence plus the owner speaking elsewhere is not corroboration: a
        // name used as a label persists too
        let mut b2 = b.clone();
        b2.owner_assignments[1].sightings = vec![sighting("kf_000100"), sighting("kf_000120")];
        let add = owner_actions(std::slice::from_ref(&b2), &[], &ctx);
        assert!(!add.iter().any(|a| a.owner == "Rohan Dasgupta"), "{add:?}");
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

    /// "Own <target>" actions for label tags (seen in two keyframes, so
    /// persistence alone would have counted) given owners as
    /// `(person_id, name, target)` and a transcript.
    fn tag_actions(owners: &[(&str, &str, &str)], lines: &[NamedLine]) -> Vec<(String, String)> {
        let mut b = build::board("b", 600.0);
        b.nodes = owners
            .iter()
            .enumerate()
            .map(|(i, (_, _, t))| build::node(&format!("n{i}"), t, 0.0, 600.0, None))
            .collect();
        b.owner_assignments = owners
            .iter()
            .enumerate()
            .map(|(i, (id, name, _))| {
                let mut o = build::owner(
                    id,
                    name,
                    build::node_target(&b, &format!("n{i}")),
                    100.0,
                    600.0,
                    None,
                );
                o.opened_at_keyframe = "kf_000100".into();
                o.sightings = vec![sighting("kf_000100"), sighting("kf_000160")];
                o
            })
            .collect();
        let names: Vec<&str> = owners.iter().map(|o| o.1).chain(["Avery Quinn"]).collect();
        let corpus = Corpus::new(
            lines,
            std::slice::from_ref(&b),
            &KeyframeTimes::default(),
            AliasTable::from_names(&names),
        );
        let ctx = OwnerActionContext {
            lines,
            corpus: &corpus,
            opts: CheckOptions::default(),
            near_s: 120.0,
        };
        owner_actions(std::slice::from_ref(&b), &[], &ctx)
            .into_iter()
            .map(|a| (a.owner, a.task))
            .collect()
    }

    const ROHAN_MIRA: &[(&str, &str, &str)] = &[
        ("rohan-dasgupta", "Rohan Dasgupta", "Kiosk App"),
        ("mira-okafor", "Mira Okafor", "Design Kit"),
    ];

    fn label_tag_actions(lines: &[NamedLine]) -> Vec<(String, String)> {
        tag_actions(ROHAN_MIRA, lines)
    }

    fn spoken(i: usize, t: f64, who: &str, text: &str) -> NamedLine {
        let mut l = line(i, t, text);
        l.person_id = Some(who.into());
        l
    }

    #[test]
    fn owner_tag_corroboration_needs_a_name_or_the_owner_on_topic() {
        // talkative owners near the tag, off topic: not corroborated
        let lines = vec![
            spoken(0, 110.0, "rohan-dasgupta", "The weather is nice today."),
            spoken(1, 115.0, "mira-okafor", "I can share my screen next."),
        ];
        assert!(label_tag_actions(&lines).is_empty());
        // one generic target word nearby, from someone else: not corroborated
        let lines = vec![
            spoken(0, 110.0, "avery-quinn", "The app is slow again."),
            spoken(1, 115.0, "avery-quinn", "The design looks off."),
        ];
        assert!(label_tag_actions(&lines).is_empty());
        // one generic target word from the owners themselves: still not
        let lines = vec![
            spoken(0, 110.0, "rohan-dasgupta", "The app is slow again."),
            spoken(1, 115.0, "mira-okafor", "The design looks off."),
        ];
        assert!(label_tag_actions(&lines).is_empty());
        // a distinctive target word, but from someone other than the owner
        let lines = vec![spoken(
            0,
            110.0,
            "avery-quinn",
            "The kiosk crashed twice today.",
        )];
        assert!(label_tag_actions(&lines).is_empty());
        // the owner on topic, but far from the tag's appearance
        let lines = vec![spoken(
            0,
            400.0,
            "rohan-dasgupta",
            "I'll look at the kiosk crash.",
        )];
        assert!(label_tag_actions(&lines).is_empty());
        // the owner on topic near the tag, and a target made of generic words
        // named in full: both corroborated
        let lines = vec![
            spoken(0, 110.0, "rohan-dasgupta", "I'll look at the kiosk crash."),
            spoken(1, 115.0, "mira-okafor", "The design kit needs new icons."),
        ];
        let got = label_tag_actions(&lines);
        assert_eq!(
            got,
            vec![
                ("Rohan Dasgupta".to_string(), "Own Kiosk App".to_string()),
                ("Mira Okafor".to_string(), "Own Design Kit".to_string()),
            ]
        );
    }

    #[test]
    fn name_mentions_corroborate_only_about_the_target() {
        // the review's constructions: a name mention about something else
        for text in [
            "Rohan, your mic is muted.",
            "Rohan, can you take that box?",
            "Thanks Rohan, that helps.",
        ] {
            let lines = vec![spoken(0, 110.0, "avery-quinn", text)];
            assert!(label_tag_actions(&lines).is_empty(), "{text}");
        }
        // a self-introduction, even when misattributed and on topic
        let lines = vec![spoken(
            0,
            110.0,
            "avery-quinn",
            "Hi, I'm Rohan Dasgupta, I lead the kiosk team.",
        )];
        assert!(label_tag_actions(&lines).is_empty());
        let lines = vec![spoken(
            0,
            110.0,
            "avery-quinn",
            "My name is Rohan and I work on the kiosk.",
        )];
        assert!(label_tag_actions(&lines).is_empty());
        // the owner saying their own name is not a mention
        let lines = vec![spoken(
            0,
            110.0,
            "rohan-dasgupta",
            "Rohan here, the build is green.",
        )];
        assert!(label_tag_actions(&lines).is_empty());
        // a mention about the target, in the line or the adjacent line
        let lines = vec![spoken(
            0,
            110.0,
            "avery-quinn",
            "Rohan, can you take the kiosk?",
        )];
        assert_eq!(label_tag_actions(&lines).len(), 1);
        let lines = vec![
            spoken(0, 110.0, "avery-quinn", "Rohan, one more thing."),
            spoken(1, 114.0, "avery-quinn", "The kiosk needs an owner."),
        ];
        let got = label_tag_actions(&lines);
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].0, "Rohan Dasgupta");
    }

    #[test]
    fn common_word_first_names_need_the_surname_or_an_address() {
        let owners = &[("mark-ellery", "Mark Ellery", "Kiosk App")];
        // the review's constructions: the word, not the person
        for text in [
            "Let's mark that as done.",
            "Put a question mark next to the kiosk box.",
            "Let's mark the kiosk as blocked.",
        ] {
            let lines = vec![spoken(0, 110.0, "avery-quinn", text)];
            assert!(tag_actions(owners, &lines).is_empty(), "{text}");
        }
        // addressing Mark about the target, or naming him in full, counts
        for text in [
            "Mark, can you take the kiosk?",
            "Mark Ellery will look after the kiosk.",
        ] {
            let lines = vec![spoken(0, 110.0, "avery-quinn", text)];
            assert_eq!(tag_actions(owners, &lines).len(), 1, "{text}");
        }
        // addressing Mark about something else does not
        let lines = vec![spoken(0, 110.0, "avery-quinn", "Mark, your mic is muted.")];
        assert!(tag_actions(owners, &lines).is_empty());
    }

    #[test]
    fn an_existing_action_suppresses_only_when_it_names_the_target() {
        let b = {
            let mut b = build::board("b", 600.0);
            b.nodes = vec![build::node("n0", "Kiosk App", 0.0, 600.0, None)];
            let mut o = build::owner(
                "rohan-dasgupta",
                "Rohan Dasgupta",
                build::node_target(&b, "n0"),
                100.0,
                600.0,
                None,
            );
            o.opened_at_keyframe = "kf_000100".into();
            b.owner_assignments = vec![o];
            b
        };
        let lines = vec![spoken(
            0,
            110.0,
            "rohan-dasgupta",
            "I'll look at the kiosk crash.",
        )];
        let corpus = Corpus::new(
            &lines,
            std::slice::from_ref(&b),
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Rohan Dasgupta"]),
        );
        let ctx = OwnerActionContext {
            lines: &lines,
            corpus: &corpus,
            opts: CheckOptions::default(),
            near_s: 120.0,
        };
        let existing = |task: &str| ActionItem {
            id: "a1".into(),
            person_id: Some("rohan-dasgupta".into()),
            owner: "Rohan Dasgupta".into(),
            task: task.into(),
            t_s: 1.0,
            t_end_s: 1.0,
            evidence: Evidence::default(),
            quote: None,
        };
        // "app" alone is another artifact: the owner action is still added
        let add = owner_actions(
            std::slice::from_ref(&b),
            &[existing("Ship the app redesign")],
            &ctx,
        );
        assert_eq!(add.len(), 1, "{add:?}");
        // naming the kiosk suppresses the duplicate
        let add = owner_actions(
            std::slice::from_ref(&b),
            &[existing("Fix the kiosk crash")],
            &ctx,
        );
        assert!(add.is_empty(), "{add:?}");
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
    fn the_guard_ignores_personal_first_person_futures() {
        // the review's constructions: personal futures are not commitments
        assert!(!has_commitment("I'll be right back."));
        assert!(!has_commitment("I'll grab a coffee before we start."));
        assert!(!has_commitment("I'll be back in a minute or so."));
        assert!(!has_commitment("I will be back in five."));
        assert!(!has_commitment("I'll talk to my mom after this."));
        assert!(!has_commitment("Let's take a quick break."));
        assert!(!has_commitment("We'll grab lunch after this."));
        // a multi-sentence segment: the personal first sentence does not lend
        // a commitment to the descriptive second one
        assert!(!has_commitment(
            "I'll be right back. The importer reads the ledger nightly."
        ));
        // first-person futures that name a work task still commit
        assert!(has_commitment("I'll take the backend."));
        assert!(has_commitment("I will write the migration spec tonight."));
        assert!(has_commitment("I'm going to draft the importer docs."));
        // "break" as a work verb is not a pause
        assert!(has_commitment("Let's break the importer into two jobs."));
    }

    #[test]
    fn group_decisions_need_an_object() {
        // the review's meeting-flow constructions
        for s in [
            "We move on.",
            "We keep going.",
            "We're moving on.",
            "We drop off now.",
            "We stay on track.",
            "We skip ahead.",
            "We pick this up tomorrow.",
            "We move on to the next item.",
            "We keep going with the demo.",
            "We move on, the importer is done.",
        ] {
            assert!(!has_commitment(s), "{s}");
        }
        // decisions with an object
        for s in [
            "We keep the ledger API on REST for the pilot.",
            "We're switching to weekly builds.",
            "We stay on REST for the pilot.",
            "We're moving back to the old importer.",
            "We skip the review step.",
            "Okay, we drop the badge printer.",
            // pronoun objects and the added decision verbs
            "We ship it.",
            "We keep it.",
            "We push the release to next week.",
            "We postpone the kiosk pilot.",
            "We cancel the badge rollout.",
            "We go with the second layout.",
        ] {
            assert!(has_commitment(s), "{s}");
        }
    }

    #[test]
    fn every_commitment_leg_needs_an_object() {
        // round 3: the same meeting-flow phrases through "let's", "we'll",
        // "we will", "decided" and first-person futures
        for s in [
            "Let's move on.",
            "We'll keep going.",
            "Let's wrap up.",
            "We'll move on to the next topic.",
            "Let's circle back to this.",
            "Okay, let's move on people.",
            "We will keep going.",
            "We're going to move on.",
            "We decided to move on.",
            "Let's go ahead.",
            "Let's get started.",
            "Let's see how it goes.",
            "Let's say we ship on Friday.",
            "I'll move on to the next slide.",
            // navigation objects and continuations anywhere before an object
            "We move to the next slide.",
            "We skip to the end.",
            "We keep right on going.",
            "Let's go to the agenda.",
            "Let's jump to the next topic.",
            // "go with" needs a subject that commits
            "The banner doesn't go with the logo.",
            "That color would go with anything.",
        ] {
            assert!(!has_commitment(s), "{s}");
        }
        for s in [
            "Let's skip that step.",
            "We'll go with the second layout.",
            "Let's go ahead and ship the importer.",
            "We decided to move the demo to Thursday.",
            "Let's not change the ledger API.",
            "I'll go with the blue theme.",
            "We will ship the importer going forward.",
        ] {
            assert!(has_commitment(s), "{s}");
        }
    }

    #[test]
    fn vetoes_apply_to_the_committing_clause_only() {
        // round 3: incidental personal words or hedges in another clause
        for s in [
            "We decided to move the demo to Thursday because of the kids' schedules.",
            "Let's push the release because of the daycare closure.",
            "We'll ship on Friday; my wife is on call all week.",
            "Before I get the kids, I'll write the migration spec.",
            "We'll go with the second layout, though we should revisit it next quarter.",
            "Let's keep the REST API, and someone should write it up.",
        ] {
            assert!(has_commitment(s), "{s}");
        }
        // the committing clause itself, a conditional next to it, or a
        // hedge-only clause before it still veto
        for s in [
            "Let's grab lunch, the importer can wait.",
            "Let's ship it if the tests pass.",
            "If the tests pass, let's ship it.",
            "Maybe, we'll ship the importer.",
            "I think, we keep the old API.",
            "We should probably ship the importer, because it is ready.",
            "I'll take a walk.",
            "I'll get some air.",
            "I'll take a quick call.",
            "I'll get the door.",
        ] {
            assert!(!has_commitment(s), "{s}");
        }
    }

    #[test]
    fn personal_topics_outside_the_old_list_are_not_action_cues() {
        let lines = vec![
            line(0, 0.0, "I'll talk to my mom after this."),
            line(1, 5.0, "I'll get the kids from school at three."),
            line(2, 10.0, "I'll talk to the vendor about the kiosk."),
        ];
        let w = Window {
            lines: (0..3).collect(),
        };
        let c = cue_lines(&w, &lines, 10);
        assert_eq!(c.len(), 1, "{c:#?}");
        assert!(c[0].starts_with("seg_00002"));
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

    #[test]
    fn owner_events_match_their_target_by_id_not_by_a_shared_name() {
        // Mira's plain tag on "Ledger" and, 10 s later, her tag moved to
        // "Ledger Store" from the Kiosk App: each event belongs to its own
        // assignment by target id. Events without a structured target (a
        // state written before events carried one) match none.
        let mut b = build::board("b", 100.0);
        b.nodes = vec![
            build::node("n1", "Ledger", 0.0, 100.0, None),
            build::node("n2", "Ledger Store", 0.0, 100.0, None),
            build::node("n3", "Kiosk App", 0.0, 100.0, None),
        ];
        let (n1, n2, n3) = (
            build::node_target(&b, "n1"),
            build::node_target(&b, "n2"),
            build::node_target(&b, "n3"),
        );
        let plain = build::owner("mira", "Mira Okafor", n1.clone(), 40.0, 100.0, None);
        let moved = build::owner(
            "mira",
            "Mira Okafor",
            n2.clone(),
            50.0,
            100.0,
            Some(n3.clone()),
        );
        let structured = vec![
            BoardEvent {
                owner_target: Some(n1.clone()),
                ..build::event("ev-a", EventKind::OwnerAssigned, 40.0, "kf1", "mira", "x")
            },
            BoardEvent {
                owner_target: Some(n2.clone()),
                owner_from: Some(n3.clone()),
                ..build::event("ev-m", EventKind::OwnerMoved, 50.0, "kf2", "mira", "x")
            },
        ];
        let legacy = vec![
            build::event(
                "ev-a",
                EventKind::OwnerAssigned,
                40.0,
                "kf1",
                "mira",
                "Mira Okafor -> Ledger",
            ),
            build::event(
                "ev-m",
                EventKind::OwnerMoved,
                50.0,
                "kf2",
                "mira",
                "Mira Okafor -> Ledger Store",
            ),
        ];
        for (events, matched) in [(structured, true), (legacy, false)] {
            b.events = events;
            let ids = |o: &OwnerAssignment| -> Vec<String> {
                owner_events(&b, o)
                    .iter()
                    .map(|e| e.event_id.clone())
                    .collect()
            };
            let want = |id: &str| {
                if matched {
                    vec![id.to_string()]
                } else {
                    vec![]
                }
            };
            assert_eq!(ids(&plain), want("ev-a"));
            assert_eq!(ids(&moved), want("ev-m"));
        }
        // Two targets with the same text: the event names one of them by id.
        b.nodes.push(build::node("n4", "Ledger", 0.0, 100.0, None));
        let twin = build::owner(
            "mira",
            "Mira Okafor",
            build::node_target(&b, "n4"),
            40.0,
            100.0,
            None,
        );
        b.events = vec![BoardEvent {
            owner_target: Some(n1.clone()),
            ..build::event("ev-a", EventKind::OwnerAssigned, 40.0, "kf1", "mira", "x")
        }];
        assert_eq!(owner_events(&b, &plain).len(), 1);
        assert!(owner_events(&b, &twin).is_empty());
        // A move with the right destination but another origin is not this move.
        b.events = vec![BoardEvent {
            owner_target: Some(n2),
            owner_from: Some(n1),
            ..build::event("ev-x", EventKind::OwnerMoved, 50.0, "kf2", "mira", "x")
        }];
        assert!(owner_events(&b, &moved).is_empty());
    }
}
