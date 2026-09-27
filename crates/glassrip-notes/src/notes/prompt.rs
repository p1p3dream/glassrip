//! Prompts: board digest, transcript windows, map, reduce and repair messages.

use std::fmt::Write as _;

use super::llm::{ChatRequest, Message};
use super::validate::{draft_schema, Draft, DraftItem, Section};
use crate::board::{
    event_text, first_seen, last_seen, target_text, BoardEvent, BoardExt, BoardStateItem,
    EdgeOrientation, EdgeStyle, KeyframeTimes, NodeState, StickyKind,
};
use crate::named::NamedLine;
use crate::people::Person;
use crate::text::{content_tokens, estimate_tokens, jaccard, mmss};

/// System prompt for every call.
pub const SYSTEM: &str = "You write meeting notes from a speaker-attributed transcript and a summary of the whiteboard shown in the meeting. \
Use only what was said or shown; never add facts. \
Every item must cite the ids of the transcript lines that support it (segment_ids, for example seg_00012), and may also cite board event ids and keyframe ids from the board summary; an action item or decision backed only by an owner tag on the board may cite just that board id. Cite only ids that appear in the input. \
The quote field must be copied exactly from one cited transcript line (3 to 20 consecutive words, keep the original wording and spelling) or be an empty string. \
Decisions: choices the group settled on (what to do, what not to do, a change in who does what), not ideas that were only floated; a proposal is settled once another participant agrees to it (\"that makes sense\", \"sounds good\") and nobody objects or takes it back; giving someone a task or ownership is an action item for that person, not a decision. \
Action items: one person (or everyone) who will do one concrete task; owner is a participant name or everyone; the task starts with a verb and names what is done. List the small ones too: someone offering to do something (\"I'll send the draft\"), a request to a person (\"can you\", \"it would be great if you could\"; the owner is the person asked) and an instruction to the whole group (\"make sure to\", \"please\", \"I'd encourage everyone to\"; the owner is everyone). Never list greetings, farewells, thanks or small talk. \
Open questions: questions raised and left unanswered; leave out questions answered later in the meeting, rhetorical questions and quick checks on the participants (whether someone knows, has or heard something). \
Timeline: the main phases of the meeting in order, one short entry per phase. \
Summary: the three to six most important points. \
Write short plain sentences. Do not use em dashes.";

/// Extra instructions and candidate blocks (see [`super::candidates`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PromptExtras {
    /// Owner-tag facts from the board, with ids to cite.
    pub board_facts: String,
    /// Cue sentences per window (map calls only).
    pub cues: Vec<Vec<String>>,
    /// Question stickies already in the notes (reduce call only).
    pub board_questions: Vec<String>,
    /// Ask for short items in the speakers' own words.
    pub concise: bool,
}

/// Rule appended to the system prompt when [`PromptExtras::concise`] is set.
pub const CONCISE: &str = " Write each decision and task as the thing itself in a few words, keeping the speakers' key words: 'Skip the importer for now', not 'The team decided that the importer step would be skipped for the time being'. One item per decision or task; do not split one decision into several items.";

fn system(extras: &PromptExtras) -> String {
    if extras.concise {
        format!("{SYSTEM}{CONCISE}")
    } else {
        SYSTEM.to_string()
    }
}

fn board_block(extras: &PromptExtras, reduce: bool) -> String {
    if extras.board_facts.trim().is_empty() {
        return String::new();
    }
    let intro = if reduce {
        BOARD_FACTS_REDUCE
    } else {
        BOARD_FACTS_INTRO
    };
    format!("\n{intro}\n{}", extras.board_facts)
}

/// The board facts introduction of the reduce call, which classifies drafted
/// items and may cite only ids that are in the drafts.
pub const BOARD_FACTS_REDUCE: &str = "Board facts from owner tags, grouped by what each one likely is. Use them to put each drafted item in the right section: a plain owner tag (\"X owns Y\") only says who is responsible for Y, so it is an action item for X, never a decision; an owner tag that moved from one target to another is a decision. Cite only ids that appear in the drafts.";

/// Introduction of the board facts block: a plain owner tag is an action item
/// for its owner, a moved owner tag is a decision.
pub const BOARD_FACTS_INTRO: &str = "Board facts from owner tags, grouped by what each one likely is. Include the ones the transcript does not contradict, citing the id in brackets as an event or keyframe id plus any transcript lines that discuss them. A plain owner tag (\"X owns Y\") only says who is responsible for Y: list it as an action item for X, never as a decision. An owner tag that moved from one target to another is a change the group made: list it as a decision. An item backed only by an owner tag may cite just that id, without a transcript line.";

/// One transcript line as the model sees it.
pub fn format_line(l: &NamedLine) -> String {
    format!(
        "{} [{}] {}: {}",
        l.segment_id,
        mmss(l.start_s),
        l.speaker,
        l.text.trim()
    )
}

/// Board summary listing every id the model may cite (event ids and keyframe
/// ids; board elements themselves have no citable id).
pub fn board_digest(boards: &[BoardStateItem], keyframes: &KeyframeTimes) -> String {
    let mut s = String::new();
    if boards.is_empty() {
        return "No whiteboard was read.".into();
    }
    for b in boards {
        let _ = writeln!(
            s,
            "Board {} (final state at {})",
            b.board_id,
            mmss(b.end_s())
        );
        s.push_str("Boxes in the final state:\n");
        for n in b.final_nodes() {
            let _ = writeln!(
                s,
                "- \"{}\" (from {})",
                n.text,
                mmss(first_seen(&n.lifetimes))
            );
        }
        let removed: Vec<&NodeState> = b.nodes.iter().filter(|n| !n.in_final).collect();
        if !removed.is_empty() {
            s.push_str("Boxes removed during the meeting:\n");
            for n in removed {
                let until = last_seen(&n.lifetimes).map(mmss).unwrap_or_default();
                let _ = writeln!(s, "- \"{}\" (seen until {until})", n.text);
            }
        }
        s.push_str("Arrows (from caller to callee):\n");
        let label = |id: &str| {
            b.node(id)
                .map(|n| n.text.clone())
                .unwrap_or_else(|| id.to_string())
        };
        for e in b.final_edges() {
            let arrow = match e.direction {
                EdgeOrientation::Forward => "->",
                EdgeOrientation::Bidirectional => "<->",
                EdgeOrientation::Uncertain => "--",
            };
            let _ = writeln!(
                s,
                "- {} {arrow} {}{}{}",
                label(&e.src),
                label(&e.dst),
                if e.label.trim().is_empty() {
                    String::new()
                } else {
                    format!(" labeled \"{}\"", e.label.trim())
                },
                if e.style == EdgeStyle::Dashed {
                    " (dashed)"
                } else {
                    ""
                }
            );
        }
        let stickies = b.final_stickies();
        if !stickies.is_empty() {
            s.push_str("Sticky notes:\n");
            for st in stickies {
                let kind = match st.kind {
                    StickyKind::Question => "question",
                    StickyKind::Milestone => "milestone",
                    StickyKind::Idea => "idea",
                    StickyKind::Note => "note",
                };
                let _ = writeln!(
                    s,
                    "- {kind}: \"{}\" (from {})",
                    st.text,
                    mmss(first_seen(&st.lifetimes))
                );
            }
        }
        if !b.owner_assignments.is_empty() {
            s.push_str("Owner tags:\n");
            for o in &b.owner_assignments {
                let until = if o.valid_to_s >= b.end_s() - 0.5 {
                    String::new()
                } else {
                    format!(" until {}", mmss(o.valid_to_s))
                };
                let _ = writeln!(
                    s,
                    "- {} on {} from {}{until}",
                    o.display_name,
                    target_text(&o.target),
                    mmss(o.valid_from_s)
                );
            }
        }
        let events: Vec<&BoardEvent> = b.events.iter().filter(|e| !e.baseline).collect();
        if !events.is_empty() {
            s.push_str("Board events (id, time, keyframe, what changed):\n");
            for e in events {
                let _ = writeln!(
                    s,
                    "- {} [{}] {} {}",
                    e.event_id,
                    mmss(e.t_s),
                    e.keyframe_id,
                    event_text(e)
                );
            }
        }
        let ks: Vec<String> = b
            .board_keyframes
            .iter()
            .map(|k| match keyframes.get(k) {
                Some(t) => format!("{k} ({})", mmss(t)),
                None => k.clone(),
            })
            .collect();
        if !ks.is_empty() {
            let _ = writeln!(s, "Board keyframes: {}", ks.join(", "));
        }
    }
    s
}

/// A transcript window.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    /// Line indices (into the named lines).
    pub lines: Vec<usize>,
}

/// Splits lines into windows of at most `budget_tokens`, repeating the last
/// `overlap` lines of a window at the start of the next.
pub fn windows(lines: &[NamedLine], budget_tokens: usize, overlap: usize) -> Vec<Window> {
    let cost: Vec<usize> = lines
        .iter()
        .map(|l| estimate_tokens(&format_line(l)) + 1)
        .collect();
    let mut out: Vec<Window> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let mut w = Vec::new();
        let mut used = 0;
        if let Some(prev) = out.last() {
            let tail: Vec<usize> = prev
                .lines
                .iter()
                .rev()
                .take(overlap)
                .rev()
                .copied()
                .collect();
            for j in tail {
                used += cost[j];
                w.push(j);
            }
        }
        let fresh_start = w.len();
        while i < lines.len() && (w.len() == fresh_start || used + cost[i] <= budget_tokens) {
            used += cost[i];
            w.push(i);
            i += 1;
        }
        out.push(Window { lines: w });
    }
    out
}

fn participants_line(people: &[Person]) -> String {
    if people.is_empty() {
        return NO_PARTICIPANTS.into();
    }
    people
        .iter()
        .map(|p| p.display_name.clone())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Participants line when no names are known.
pub const NO_PARTICIPANTS: &str = "none known (speakers are unnamed diarization labels; an action item's owner is everyone or a person named by an owner tag on the board)";

/// Map request for one window.
/// `part` is (index, count) of the window.
pub fn map_request(
    part: (usize, usize),
    win: &Window,
    lines: &[NamedLine],
    digest: &str,
    people: &[Person],
    num_predict: u32,
    extras: &PromptExtras,
) -> ChatRequest {
    let (idx, total) = part;
    let body: Vec<String> = win
        .lines
        .iter()
        .filter_map(|i| lines.get(*i))
        .map(format_line)
        .collect();
    let span = match (
        win.lines.first().and_then(|i| lines.get(*i)),
        win.lines.last().and_then(|i| lines.get(*i)),
    ) {
        (Some(a), Some(b)) => format!("{} to {}", mmss(a.start_s), mmss(b.end_s)),
        _ => String::new(),
    };
    let cues = extras.cues.get(idx).filter(|c| !c.is_empty()).map(|c| {
        format!(
            "\n\nLines with decision or question cues (for each: include it as a decision, action item or open question when it is one, citing its segment id; leave it out when it is not):\n{}",
            c.join("\n")
        )
    });
    let user = format!(
        "Participants: {}\n\nWhiteboard:\n{digest}{}\nTranscript part {} of {total} ({span}). Each line is: segment_id [mm:ss] speaker: text\n{}{}\n\nExtract the decisions, action items, open questions, timeline entries and summary points supported by this part. Return JSON only.",
        participants_line(people),
        board_block(extras, false),
        idx + 1,
        body.join("\n"),
        cues.unwrap_or_default(),
    );
    ChatRequest {
        purpose: format!("map {}/{total}", idx + 1),
        messages: vec![
            Message {
                role: "system".into(),
                content: system(extras),
            },
            Message {
                role: "user".into(),
                content: user,
            },
        ],
        format: draft_schema(),
        num_predict,
    }
}

/// Reduce request merging the drafts of several windows.
pub fn reduce_request(
    drafts: &[Draft],
    digest: &str,
    people: &[Person],
    num_predict: u32,
    extras: &PromptExtras,
) -> ChatRequest {
    let mut merged = Draft::default();
    for d in drafts {
        for s in Section::ALL {
            merged.section_mut(s).extend(d.section(s).iter().cloned());
        }
    }
    // compact: the drafts are the largest part of the prompt
    let candidates = serde_json::to_string(&merged).unwrap_or_default();
    let user = format!(
        "Participants: {}\n\nWhiteboard:\n{digest}\nThe notes below were drafted separately for consecutive parts of one meeting, so the same point can appear more than once. Merge them into one set of notes: combine duplicates into one item and keep the union of their citations, keep the clearest wording, drop items that do not meet the rules, order the timeline by time and merge it into at most 12 phases, and give three to six summary points. Copy segment_ids, event_ids, keyframe_ids and quotes only from the drafts.{}{} Return JSON only.\n\nDrafts:\n{candidates}",
        participants_line(people),
        board_block(extras, true),
        if extras.board_questions.is_empty() {
            String::new()
        } else {
            format!(
                "\nThese questions are already in the notes from the board's question stickies; do not list them (or rewordings of them) as open questions:\n{}\n",
                extras
                    .board_questions
                    .iter()
                    .map(|q| format!("- {q}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        },
    );
    ChatRequest {
        purpose: "reduce".into(),
        messages: vec![
            Message {
                role: "system".into(),
                content: system(extras),
            },
            Message {
                role: "user".into(),
                content: user,
            },
        ],
        format: draft_schema(),
        num_predict,
    }
}

/// A failed item sent for repair.
#[derive(Debug, Clone)]
pub struct RepairCase {
    /// Section.
    pub section: Section,
    /// The item.
    pub item: DraftItem,
    /// Validation failures.
    pub reasons: Vec<String>,
}

/// Transcript lines relevant to the failing items: cited lines with neighbours,
/// plus the best-matching lines by content words.
pub fn repair_context(cases: &[RepairCase], lines: &[NamedLine], max_lines: usize) -> Vec<usize> {
    let mut pick = std::collections::BTreeSet::new();
    for c in cases {
        for (i, l) in lines.iter().enumerate() {
            if c.item.segment_ids.iter().any(|id| id == &l.segment_id) {
                for j in i.saturating_sub(2)..=(i + 2).min(lines.len().saturating_sub(1)) {
                    pick.insert(j);
                }
            }
        }
        let toks = content_tokens(&format!(
            "{} {} {}",
            c.item.main_text(),
            c.item.quote,
            c.item.owner
        ));
        let mut scored: Vec<(f64, usize)> = lines
            .iter()
            .enumerate()
            .map(|(i, l)| (jaccard(&content_tokens(&l.text), &toks), i))
            .filter(|(s, _)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for (_, i) in scored.into_iter().take(5) {
            for j in i.saturating_sub(1)..=(i + 1).min(lines.len().saturating_sub(1)) {
                pick.insert(j);
            }
        }
    }
    pick.into_iter().take(max_lines).collect()
}

/// Repair request listing each failure.
pub fn repair_request(
    cases: &[RepairCase],
    lines: &[NamedLine],
    context: &[usize],
    digest: &str,
    people: &[Person],
    num_predict: u32,
) -> ChatRequest {
    let mut failures = String::new();
    for (i, c) in cases.iter().enumerate() {
        let item = serde_json::to_string(&c.item).unwrap_or_default();
        let _ = writeln!(
            failures,
            "{}. section {}: {item}\n   problems: {}",
            i + 1,
            c.section.key(),
            c.reasons.join("; ")
        );
    }
    let ctx: Vec<String> = context
        .iter()
        .filter_map(|i| lines.get(*i))
        .map(format_line)
        .collect();
    let user = format!(
        "Participants: {}\n\nWhiteboard:\n{digest}\nThese drafted items failed validation:\n{failures}\nRelevant transcript lines (segment_id [mm:ss] speaker: text):\n{}\n\nReturn corrected versions of these items in their sections: cite segment ids that exist and support the item, copy quotes exactly from a cited line (or leave the quote empty), use a participant name or everyone as owner, and start tasks with a verb. Leave out any item that neither the transcript nor a cited owner tag on the board supports. Return JSON only.",
        participants_line(people),
        ctx.join("\n"),
    );
    ChatRequest {
        purpose: "repair".into(),
        messages: vec![
            Message {
                role: "system".into(),
                content: SYSTEM.into(),
            },
            Message {
                role: "user".into(),
                content: user,
            },
        ],
        format: draft_schema(),
        num_predict,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(i: usize) -> NamedLine {
        NamedLine {
            segment_id: format!("seg_{i:05}"),
            start_s: i as f64 * 5.0,
            end_s: i as f64 * 5.0 + 4.0,
            person_id: None,
            speaker: "Avery Quinn".into(),
            text: "We should move the relay config into the ledger service this week.".into(),
            text_raw: String::new(),
            speaker_confidence: 1.0,
            relabeled: false,
            gap_fill_share: 0.0,
        }
    }

    #[test]
    fn windows_respect_budget_and_overlap() {
        let lines: Vec<NamedLine> = (0..40).map(line).collect();
        let per = estimate_tokens(&format_line(&lines[0])) + 1;
        let ws = windows(&lines, per * 12, 2);
        assert!(ws.len() >= 4);
        for w in &ws {
            assert!(w.lines.len() <= 12);
        }
        assert_eq!(ws[1].lines[..2], ws[0].lines[ws[0].lines.len() - 2..]);
        let covered: std::collections::BTreeSet<usize> =
            ws.iter().flat_map(|w| w.lines.clone()).collect();
        assert_eq!(covered.len(), 40);
        assert_eq!(windows(&lines[..3], 1_000_000, 2).len(), 1);
    }

    #[test]
    fn board_block_offers_owner_tags_as_actions_and_moves_as_decisions() {
        let lines: Vec<NamedLine> = (0..2).map(line).collect();
        let win = Window { lines: vec![0, 1] };
        let facts = "Action item candidates (x):\n- Mira Okafor owns Ledger Store (owner tag from 00:20) [cite ev-1]\nDecision candidates (y):\n- Mira Okafor moved from Ledger Store to Kiosk App at 01:00 [cite ev-2]\n";
        let extras = PromptExtras {
            board_facts: facts.into(),
            ..PromptExtras::default()
        };
        let people = vec![Person {
            person_id: "mira-okafor".into(),
            display_name: "Mira Okafor".into(),
            aliases: vec![],
        }];
        let map = map_request((0, 1), &win, &lines, "digest", &people, 100, &extras);
        let user = &map.messages[1].content;
        assert!(user.contains(BOARD_FACTS_INTRO), "{user}");
        assert!(user.contains(facts), "facts and their ids are kept: {user}");
        assert!(
            !user.contains("likely a decision or an action item"),
            "{user}"
        );
        assert!(BOARD_FACTS_INTRO.contains("list it as an action item for X, never as a decision"));
        assert!(BOARD_FACTS_INTRO.contains("moved from one target to another"));
        assert!(BOARD_FACTS_INTRO.contains("list it as a decision"));
        assert!(user.starts_with("Participants: Mira Okafor\n"), "{user}");
        let reduce = reduce_request(
            &[Draft::default(), Draft::default()],
            "digest",
            &people,
            100,
            &extras,
        );
        let merge = &reduce.messages[1].content;
        assert!(merge.contains(BOARD_FACTS_REDUCE) && merge.contains(facts));
        assert!(
            !merge.contains(BOARD_FACTS_INTRO),
            "the reduce call cites only draft ids"
        );
        assert!(
            !user.contains(BOARD_FACTS_REDUCE),
            "map calls cite the board ids"
        );
        assert!(SYSTEM.contains("a change in who does what"));
        assert!(SYSTEM.contains("ownership is an action item for that person, not a decision"));
        // agreed proposals, requests and group instructions
        assert!(SYSTEM.contains("a proposal is settled once another participant agrees to it"));
        assert!(SYSTEM.contains("the owner is the person asked"));
        assert!(SYSTEM.contains("the owner is everyone"));
        assert!(SYSTEM.contains("leave out questions answered later in the meeting"));
        // no facts: no block
        let bare = map_request(
            (0, 1),
            &win,
            &lines,
            "digest",
            &people,
            100,
            &PromptExtras::default(),
        );
        assert!(!bare.messages[1].content.contains("Board facts"));
    }

    #[test]
    fn no_participant_names_are_said_plainly() {
        let lines: Vec<NamedLine> = (0..2).map(line).collect();
        let win = Window { lines: vec![0, 1] };
        let map = map_request(
            (0, 1),
            &win,
            &lines,
            "digest",
            &[],
            100,
            &PromptExtras::default(),
        );
        assert!(map.messages[1]
            .content
            .starts_with(&format!("Participants: {NO_PARTICIPANTS}\n")));
    }

    #[test]
    fn line_format() {
        assert!(format_line(&line(3)).starts_with("seg_00003 [00:15] Avery Quinn: We should"));
    }
}
