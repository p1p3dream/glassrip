//! Prompts: board digest, transcript windows, map, reduce and repair messages.

use std::fmt::Write as _;

use super::llm::{ChatRequest, Message};
use super::validate::{draft_schema, Draft, DraftItem, Section};
use crate::board::BoardState;
use crate::named::NamedLine;
use crate::people::Person;
use crate::text::{content_tokens, estimate_tokens, jaccard, mmss};

/// System prompt for every call.
pub const SYSTEM: &str = "You write meeting notes from a speaker-attributed transcript and a summary of the whiteboard shown in the meeting. \
Use only what was said or shown; never add facts. \
Every item must cite the ids of the transcript lines that support it (segment_ids, for example seg_00012), and may also cite board event ids and keyframe ids from the board summary. Cite only ids that appear in the input. \
The quote field must be copied exactly from one cited transcript line (3 to 20 consecutive words, keep the original wording and spelling) or be an empty string. \
Decisions: choices the group settled on (what to do, what not to do, who does what), not ideas that were only floated. \
Action items: one person (or everyone) who will do one concrete task; owner is a participant name or everyone; the task starts with a verb and names what is done. Never list greetings, farewells, thanks or small talk. \
Open questions: questions raised and left unanswered. \
Timeline: the main phases of the meeting in order, one short entry per phase. \
Summary: the three to six most important points. \
Write short plain sentences. Do not use em dashes.";

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

/// Board summary listing every id the model may cite.
pub fn board_digest(boards: &[BoardState], people: &[Person]) -> String {
    let mut s = String::new();
    if boards.is_empty() {
        return "No whiteboard was read.".into();
    }
    let name = |pid: &Option<String>, raw: &str| -> String {
        pid.as_ref()
            .and_then(|p| people.iter().find(|x| &x.person_id == p))
            .map(|p| p.display_name.clone())
            .unwrap_or_else(|| raw.to_string())
    };
    for b in boards {
        let _ = writeln!(
            s,
            "Board {}{} (final state at {})",
            b.board_id,
            b.title
                .as_ref()
                .map(|t| format!(" \"{t}\""))
                .unwrap_or_default(),
            mmss(b.final_t_s)
        );
        let label = |id: &str| {
            b.node(id)
                .map(|n| n.text.clone())
                .unwrap_or_else(|| id.to_string())
        };
        s.push_str("Boxes:\n");
        for n in &b.nodes {
            let until = n
                .last_seen_s
                .map(|t| format!(" until {}", mmss(t)))
                .unwrap_or_default();
            let _ = writeln!(s, "- \"{}\" (from {}{until})", n.text, mmss(n.first_seen_s));
        }
        s.push_str("Arrows (from caller to callee):\n");
        for e in &b.edges {
            let _ = writeln!(
                s,
                "- {} -> {}{}{}",
                label(&e.src),
                label(&e.dst),
                e.label
                    .as_ref()
                    .map(|l| format!(" labeled \"{l}\""))
                    .unwrap_or_default(),
                if e.style == crate::board::EdgeStyle::Dashed {
                    " (dashed)"
                } else {
                    ""
                }
            );
        }
        if !b.stickies.is_empty() {
            s.push_str("Sticky notes:\n");
            for st in &b.stickies {
                let _ = writeln!(
                    s,
                    "- \"{}\" (first seen {}; keyframes {})",
                    st.text,
                    mmss(st.first_seen_s),
                    st.keyframe_ids.join(", ")
                );
            }
        }
        for g in &b.groups {
            let members: Vec<String> = g
                .members
                .iter()
                .map(|m| match &m.detail {
                    Some(d) => format!("{} ({d})", m.text),
                    None => m.text.clone(),
                })
                .collect();
            let _ = writeln!(
                s,
                "Group \"{}\" (from {}): {}",
                g.label,
                mmss(g.first_seen_s),
                members.join("; ")
            );
        }
        if !b.owner_assignments.is_empty() {
            s.push_str("Owner tags:\n");
            for o in &b.owner_assignments {
                let target = match o.target_kind {
                    crate::board::TargetKind::Node => label(&o.target_id),
                    crate::board::TargetKind::Edge => b
                        .edge(&o.target_id)
                        .map(|e| format!("the arrow {} -> {}", label(&e.src), label(&e.dst)))
                        .unwrap_or_else(|| o.target_id.clone()),
                };
                let until = o
                    .valid_to_s
                    .map(|t| format!(" until {}", mmss(t)))
                    .unwrap_or_default();
                let _ = writeln!(
                    s,
                    "- {} on {target} from {}{until}",
                    name(&o.person_id, &o.name_raw),
                    mmss(o.valid_from_s)
                );
            }
        }
        if !b.events.is_empty() {
            s.push_str("Board events (id, time, what changed):\n");
            for e in &b.events {
                let _ = writeln!(s, "- {} [{}] {}", e.event_id, mmss(e.t_s), e.summary);
            }
        }
        if !b.keyframes.is_empty() {
            let ks: Vec<String> = b
                .keyframes
                .iter()
                .map(|k| format!("{} ({})", k.keyframe_id, mmss(k.t_rep_s)))
                .collect();
            let _ = writeln!(s, "Keyframes: {}", ks.join(", "));
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
    people
        .iter()
        .map(|p| p.display_name.clone())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Map request for one window.
pub fn map_request(
    idx: usize,
    total: usize,
    win: &Window,
    lines: &[NamedLine],
    digest: &str,
    people: &[Person],
    num_predict: u32,
) -> ChatRequest {
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
    let user = format!(
        "Participants: {}\n\nWhiteboard:\n{digest}\nTranscript part {} of {total} ({span}). Each line is: segment_id [mm:ss] speaker: text\n{}\n\nExtract the decisions, action items, open questions, timeline entries and summary points supported by this part. Return JSON only.",
        participants_line(people),
        idx + 1,
        body.join("\n"),
    );
    ChatRequest {
        purpose: format!("map {}/{total}", idx + 1),
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

/// Reduce request merging the drafts of several windows.
pub fn reduce_request(
    drafts: &[Draft],
    digest: &str,
    people: &[Person],
    num_predict: u32,
) -> ChatRequest {
    let mut merged = Draft::default();
    for d in drafts {
        for s in Section::ALL {
            merged.section_mut(s).extend(d.section(s).iter().cloned());
        }
    }
    let candidates = serde_json::to_string_pretty(&merged).unwrap_or_default();
    let user = format!(
        "Participants: {}\n\nWhiteboard:\n{digest}\nThe notes below were drafted separately for consecutive parts of one meeting, so the same point can appear more than once. Merge them into one set of notes: combine duplicates into one item and keep the union of their citations, keep the clearest wording, drop items that do not meet the rules, order the timeline by time and merge it into at most 12 phases, and give three to six summary points. Copy segment_ids, event_ids, keyframe_ids and quotes only from the drafts. Return JSON only.\n\nDrafts:\n{candidates}",
        participants_line(people),
    );
    ChatRequest {
        purpose: "reduce".into(),
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
        "Participants: {}\n\nWhiteboard:\n{digest}\nThese drafted items failed validation:\n{failures}\nRelevant transcript lines (segment_id [mm:ss] speaker: text):\n{}\n\nReturn corrected versions of these items in their sections: cite segment ids that exist and support the item, copy quotes exactly from a cited line (or leave the quote empty), use a participant name or everyone as owner, and start tasks with a verb. Leave out any item the transcript does not support. Return JSON only.",
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
    fn line_format() {
        assert!(format_line(&line(3)).starts_with("seg_00003 [00:15] Avery Quinn: We should"));
    }
}
