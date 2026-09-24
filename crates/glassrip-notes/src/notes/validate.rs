//! The Rust side of the notes: what the model drafts is checked here.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{
    ActionItem, Decision, DroppedItem, Evidence, OpenQuestion, QuestionSource, Quote, QuoteMatch,
    SummaryPoint, TimelineEntry,
};
use crate::board::{BoardState, StickyKind};
use crate::named::NamedLine;
use crate::people::AliasTable;
use crate::text::{content_tokens, jaccard, normalize, sanitize_dashes, tokens};

/// One drafted item (fields depend on the section).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DraftItem {
    /// Statement (decisions, questions, timeline, summary).
    #[serde(default)]
    pub text: String,
    /// Owner name (action items).
    #[serde(default)]
    pub owner: String,
    /// Task (action items).
    #[serde(default)]
    pub task: String,
    /// Transcript segment ids.
    #[serde(default)]
    pub segment_ids: Vec<String>,
    /// Board event ids.
    #[serde(default)]
    pub event_ids: Vec<String>,
    /// Keyframe ids.
    #[serde(default)]
    pub keyframe_ids: Vec<String>,
    /// Verbatim quote from a cited segment (may be empty).
    #[serde(default)]
    pub quote: String,
}

impl DraftItem {
    /// The item's main text (task for action items).
    pub fn main_text(&self) -> String {
        if self.task.is_empty() {
            self.text.clone()
        } else if self.owner.is_empty() {
            self.task.clone()
        } else {
            format!("{}: {}", self.owner, self.task)
        }
    }
}

/// A drafted set of notes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Draft {
    /// Decisions.
    #[serde(default)]
    pub decisions: Vec<DraftItem>,
    /// Action items.
    #[serde(default)]
    pub action_items: Vec<DraftItem>,
    /// Open questions.
    #[serde(default)]
    pub open_questions: Vec<DraftItem>,
    /// Timeline entries.
    #[serde(default)]
    pub timeline: Vec<DraftItem>,
    /// Summary points.
    #[serde(default)]
    pub summary: Vec<DraftItem>,
}

/// A notes section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Section {
    /// Decisions.
    Decisions,
    /// Action items.
    ActionItems,
    /// Open questions.
    OpenQuestions,
    /// Timeline.
    Timeline,
    /// Summary.
    Summary,
}

impl Section {
    /// All sections.
    pub const ALL: [Section; 5] = [
        Section::Decisions,
        Section::ActionItems,
        Section::OpenQuestions,
        Section::Timeline,
        Section::Summary,
    ];

    /// JSON key.
    pub fn key(self) -> &'static str {
        match self {
            Section::Decisions => "decisions",
            Section::ActionItems => "action_items",
            Section::OpenQuestions => "open_questions",
            Section::Timeline => "timeline",
            Section::Summary => "summary",
        }
    }
}

impl Draft {
    /// Items of a section.
    pub fn section(&self, s: Section) -> &Vec<DraftItem> {
        match s {
            Section::Decisions => &self.decisions,
            Section::ActionItems => &self.action_items,
            Section::OpenQuestions => &self.open_questions,
            Section::Timeline => &self.timeline,
            Section::Summary => &self.summary,
        }
    }

    /// Mutable items of a section.
    pub fn section_mut(&mut self, s: Section) -> &mut Vec<DraftItem> {
        match s {
            Section::Decisions => &mut self.decisions,
            Section::ActionItems => &mut self.action_items,
            Section::OpenQuestions => &mut self.open_questions,
            Section::Timeline => &mut self.timeline,
            Section::Summary => &mut self.summary,
        }
    }

    /// Items in all sections.
    pub fn len(&self) -> usize {
        Section::ALL.iter().map(|s| self.section(*s).len()).sum()
    }

    /// True when no section has items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn ids_schema() -> Value {
    json!({"type": "array", "items": {"type": "string"}, "maxItems": 8})
}

/// JSON schema of a [`Draft`] (sent as the Ollama `format`).
pub fn draft_schema() -> Value {
    let item = |fields: &[&str], max: u32| {
        let mut props = serde_json::Map::new();
        for f in fields {
            props.insert((*f).to_string(), json!({"type": "string"}));
        }
        for f in ["segment_ids", "event_ids", "keyframe_ids"] {
            props.insert(f.to_string(), ids_schema());
        }
        let mut required: Vec<&str> = fields.to_vec();
        required.extend(["segment_ids", "event_ids", "keyframe_ids"]);
        json!({
            "type": "array", "maxItems": max,
            "items": {"type": "object", "properties": props, "required": required}
        })
    };
    json!({
        "type": "object",
        "properties": {
            "decisions": item(&["text", "quote"], 12),
            "action_items": item(&["owner", "task", "quote"], 16),
            "open_questions": item(&["text", "quote"], 12),
            "timeline": item(&["text"], 16),
            "summary": item(&["text"], 8),
        },
        "required": ["decisions", "action_items", "open_questions", "timeline", "summary"]
    })
}

#[derive(Debug, Clone)]
struct SegText {
    text: String,
    text_raw: String,
    norm: String,
    norm_raw: String,
    start_s: f64,
    end_s: f64,
}

/// What citations and quotes are checked against.
#[derive(Debug, Clone)]
pub struct Corpus {
    segs: BTreeMap<String, SegText>,
    times: BTreeMap<String, f64>,
    table: AliasTable,
}

impl Corpus {
    /// Builds the corpus from named transcript lines and board states.
    pub fn new(lines: &[NamedLine], boards: &[BoardState], table: AliasTable) -> Self {
        let mut segs: BTreeMap<String, SegText> = BTreeMap::new();
        for l in lines {
            let e = segs.entry(l.segment_id.clone()).or_insert_with(|| SegText {
                text: String::new(),
                text_raw: String::new(),
                norm: String::new(),
                norm_raw: String::new(),
                start_s: l.start_s,
                end_s: l.end_s,
            });
            for (dst, src) in [(&mut e.text, &l.text), (&mut e.text_raw, &l.text_raw)] {
                if !dst.is_empty() {
                    dst.push(' ');
                }
                dst.push_str(src);
            }
            e.start_s = e.start_s.min(l.start_s);
            e.end_s = e.end_s.max(l.end_s);
        }
        for s in segs.values_mut() {
            s.norm = normalize(&s.text);
            s.norm_raw = normalize(&s.text_raw);
        }
        let mut times = BTreeMap::new();
        for b in boards {
            for e in &b.events {
                times.insert(e.event_id.clone(), e.t_s);
            }
            for k in &b.keyframes {
                times.insert(k.keyframe_id.clone(), k.t_rep_s);
            }
        }
        Self { segs, times, table }
    }

    /// The alias table.
    pub fn table(&self) -> &AliasTable {
        &self.table
    }

    /// Start and end of a segment.
    pub fn segment_span(&self, id: &str) -> Option<(f64, f64)> {
        self.segs.get(id).map(|s| (s.start_s, s.end_s))
    }

    /// Segment ids in time order.
    pub fn segment_ids(&self) -> Vec<&str> {
        let mut v: Vec<(&str, f64)> = self
            .segs
            .iter()
            .map(|(k, s)| (k.as_str(), s.start_s))
            .collect();
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        v.into_iter().map(|x| x.0).collect()
    }

    /// Finds a quote (with optional `...` elisions) in the cited segments.
    fn find_quote(&self, quote: &str, cited: &[&str]) -> Option<(QuoteMatch, String)> {
        let parts: Vec<String> = quote
            .split("...")
            .flat_map(|p| p.split('\u{2026}'))
            .map(normalize)
            .filter(|p| !p.is_empty())
            .collect();
        if parts.is_empty() {
            return None;
        }
        let mut segs: Vec<(&str, &SegText)> = cited
            .iter()
            .filter_map(|id| self.segs.get(*id).map(|s| (*id, s)))
            .collect();
        segs.sort_by(|a, b| a.1.start_s.total_cmp(&b.1.start_s));
        let contains_in_order = |hay: &str| {
            let hay = format!(" {hay} ");
            let mut pos = 0;
            for p in &parts {
                let needle = format!(" {p} ");
                match hay.get(pos..).and_then(|h| h.find(&needle)) {
                    Some(i) => pos += i + needle.len() - 1,
                    None => return false,
                }
            }
            true
        };
        let first_part = parts.first().cloned().unwrap_or_default();
        let owner = |raw: bool| -> String {
            segs.iter()
                .find(|(_, s)| {
                    format!(" {} ", if raw { &s.norm_raw } else { &s.norm })
                        .contains(&format!(" {first_part} "))
                })
                .or(segs.first())
                .map(|(id, _)| (*id).to_string())
                .unwrap_or_default()
        };
        for (raw, kind) in [(true, QuoteMatch::Raw), (false, QuoteMatch::Corrected)] {
            let pick = |s: &SegText| {
                if raw {
                    s.norm_raw.clone()
                } else {
                    s.norm.clone()
                }
            };
            if segs.iter().any(|(_, s)| contains_in_order(&pick(s))) {
                return Some((kind, owner(raw)));
            }
            let joined = segs
                .iter()
                .map(|(_, s)| pick(s))
                .collect::<Vec<_>>()
                .join(" ");
            if contains_in_order(&joined) {
                return Some((kind, owner(raw)));
            }
        }
        None
    }
}

/// An action-item owner.
#[derive(Debug, Clone, PartialEq)]
pub struct Owner {
    /// Participant (None: everyone).
    pub person_id: Option<String>,
    /// Display name.
    pub name: String,
}

/// A drafted item that passed validation.
#[derive(Debug, Clone, PartialEq)]
pub struct Checked {
    /// Section.
    pub section: Section,
    /// Text (task for action items), sanitized.
    pub text: String,
    /// Owners (action items).
    pub owners: Vec<Owner>,
    /// Citations.
    pub evidence: Evidence,
    /// Verified quote.
    pub quote: Option<Quote>,
    /// Start of the evidence, seconds.
    pub t_start_s: f64,
    /// End of the evidence, seconds.
    pub t_end_s: f64,
}

/// Why a drafted item failed.
#[derive(Debug, Clone, PartialEq)]
pub struct Failure {
    /// Reasons.
    pub reasons: Vec<String>,
    /// A repair cannot help (greetings, farewells).
    pub fatal: bool,
}

const TASK_VERBS: &[&str] = &[
    "add",
    "adjust",
    "align",
    "analyze",
    "ask",
    "assess",
    "assign",
    "audit",
    "build",
    "change",
    "check",
    "clarify",
    "clean",
    "clone",
    "collect",
    "compare",
    "configure",
    "confirm",
    "connect",
    "contact",
    "convert",
    "coordinate",
    "create",
    "decide",
    "define",
    "deliver",
    "demo",
    "deploy",
    "design",
    "determine",
    "develop",
    "document",
    "draft",
    "draw",
    "enable",
    "estimate",
    "evaluate",
    "explore",
    "export",
    "extend",
    "figure",
    "file",
    "finalize",
    "find",
    "finish",
    "fix",
    "follow",
    "gather",
    "get",
    "hook",
    "identify",
    "implement",
    "import",
    "improve",
    "inspect",
    "install",
    "integrate",
    "investigate",
    "learn",
    "list",
    "look",
    "make",
    "map",
    "measure",
    "meet",
    "merge",
    "migrate",
    "model",
    "move",
    "offer",
    "organize",
    "own",
    "pair",
    "plan",
    "post",
    "prepare",
    "present",
    "prioritize",
    "produce",
    "propose",
    "prototype",
    "provide",
    "publish",
    "pull",
    "push",
    "read",
    "rebuild",
    "record",
    "refactor",
    "reach",
    "remove",
    "render",
    "reorder",
    "replace",
    "report",
    "request",
    "research",
    "resolve",
    "review",
    "run",
    "schedule",
    "scope",
    "send",
    "set",
    "settle",
    "share",
    "ship",
    "sketch",
    "sort",
    "spec",
    "specify",
    "split",
    "start",
    "store",
    "study",
    "submit",
    "support",
    "sync",
    "take",
    "talk",
    "test",
    "track",
    "train",
    "translate",
    "try",
    "understand",
    "update",
    "upload",
    "validate",
    "verify",
    "wire",
    "work",
    "write",
];

const TASK_PREFIXES: &[&str] = &["to", "will", "should", "must", "needs", "need", "please"];

const FAREWELLS: &[&str] = &[
    "see you",
    "see ya",
    "bye",
    "goodbye",
    "talk to you later",
    "talk later",
    "catch you later",
    "have a good",
    "have a great",
    "nice to meet",
    "thanks everyone",
    "thank you everyone",
    "thanks for joining",
    "hello",
    "good morning",
    "good afternoon",
];

/// True when the text is a greeting, thanks or farewell rather than content.
pub fn is_greeting_or_farewell(text: &str) -> bool {
    let n = normalize(text);
    let toks = tokens(text);
    if toks.len() <= 3
        && toks.iter().any(|t| {
            matches!(
                t.as_str(),
                "hi" | "hey" | "hello" | "bye" | "thanks" | "thank"
            )
        })
    {
        return true;
    }
    FAREWELLS
        .iter()
        .any(|f| n.starts_with(f) || n.contains(&format!(" {f} ")) || n.ends_with(&format!(" {f}")))
        && content_tokens(text).len() <= 6
}

/// Checks the task has a leading verb and an object. Returns the reason if not.
pub fn check_task(task: &str) -> Option<String> {
    let toks = tokens(task);
    let mut i = 0;
    while toks
        .get(i)
        .is_some_and(|t| TASK_PREFIXES.contains(&t.as_str()))
    {
        i += 1;
    }
    let Some(verb) = toks.get(i) else {
        return Some("empty task".into());
    };
    if !TASK_VERBS.contains(&verb.as_str()) {
        return Some(format!(
            "task must start with a task verb (found \"{verb}\")"
        ));
    }
    let object = toks[i + 1..]
        .iter()
        .filter(|t| !crate::text::is_stopword(t))
        .count();
    if object == 0 || toks.len() < i + 3 {
        return Some("task needs an object after the verb".into());
    }
    None
}

fn resolve_owners(raw: &str, table: &AliasTable) -> Result<Vec<Owner>, String> {
    let mut out = Vec::new();
    let replaced = raw.replace(" and ", ",").replace(['&', '/'], ",");
    for part in replaced.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let n = normalize(part);
        if matches!(
            n.as_str(),
            "everyone"
                | "everybody"
                | "all"
                | "team"
                | "the team"
                | "whole team"
                | "group"
                | "the group"
                | "all of us"
        ) {
            out.push(Owner {
                person_id: None,
                name: "Everyone".into(),
            });
            continue;
        }
        let m = table.match_screen_text(part).or_else(|| {
            part.split_whitespace()
                .next()
                .and_then(|w| table.match_word(w))
        });
        match m.and_then(|m| table.person(m.person)) {
            Some(p) => out.push(Owner {
                person_id: Some(p.person_id.clone()),
                name: p.display_name.clone(),
            }),
            None => return Err(format!("owner \"{part}\" is not a participant")),
        }
    }
    if out.is_empty() {
        return Err("action item has no owner".into());
    }
    out.dedup();
    Ok(out)
}

/// Validates one drafted item.
pub fn check(section: Section, item: &DraftItem, corpus: &Corpus) -> Result<Checked, Failure> {
    let mut reasons = Vec::new();
    let text = sanitize_dashes(if section == Section::ActionItems {
        &item.task
    } else {
        &item.text
    });
    if tokens(&text).len() < 3 {
        reasons.push("text too short".to_string());
    }
    if matches!(section, Section::Decisions | Section::ActionItems)
        && is_greeting_or_farewell(&text)
    {
        return Err(Failure {
            reasons: vec!["greeting or farewell".into()],
            fatal: true,
        });
    }
    let mut owners = Vec::new();
    if section == Section::ActionItems {
        if let Some(r) = check_task(&text) {
            reasons.push(r);
        }
        match resolve_owners(&item.owner, corpus.table()) {
            Ok(o) => owners = o,
            Err(e) => reasons.push(e),
        }
    }
    let evidence = Evidence {
        segment_ids: dedup(&item.segment_ids),
        event_ids: dedup(&item.event_ids),
        keyframe_ids: dedup(&item.keyframe_ids),
    };
    if evidence.is_empty() {
        reasons.push("no citations".into());
    }
    let needs_segment = matches!(
        section,
        Section::Decisions | Section::ActionItems | Section::OpenQuestions
    );
    if needs_segment && evidence.segment_ids.is_empty() {
        reasons.push("must cite at least one transcript segment".into());
    }
    for id in &evidence.segment_ids {
        if !corpus.segs.contains_key(id) {
            reasons.push(format!("unknown segment id {id}"));
        }
    }
    for id in evidence.event_ids.iter().chain(&evidence.keyframe_ids) {
        if !corpus.times.contains_key(id) {
            reasons.push(format!("unknown board id {id}"));
        }
    }
    let mut quote = None;
    let q = sanitize_dashes(item.quote.trim().trim_matches('"'));
    if !q.is_empty() && reasons.is_empty() {
        let cited: Vec<&str> = evidence.segment_ids.iter().map(String::as_str).collect();
        match corpus.find_quote(&q, &cited) {
            Some((matched, segment_id)) => {
                quote = Some(Quote {
                    text: q,
                    segment_id,
                    matched,
                })
            }
            None => reasons.push(format!("quote \"{q}\" is not in the cited segments")),
        }
    }
    if !reasons.is_empty() {
        return Err(Failure {
            reasons,
            fatal: false,
        });
    }
    let mut times: Vec<(f64, f64)> = evidence
        .segment_ids
        .iter()
        .filter_map(|id| corpus.segment_span(id))
        .collect();
    times.extend(
        evidence
            .event_ids
            .iter()
            .chain(&evidence.keyframe_ids)
            .filter_map(|id| corpus.times.get(id).map(|t| (*t, *t))),
    );
    let t_start_s = times.iter().map(|t| t.0).fold(f64::INFINITY, f64::min);
    let t_end_s = times.iter().map(|t| t.1).fold(f64::NEG_INFINITY, f64::max);
    Ok(Checked {
        section,
        text,
        owners,
        evidence,
        quote,
        t_start_s: if t_start_s.is_finite() {
            t_start_s
        } else {
            0.0
        },
        t_end_s: if t_end_s.is_finite() { t_end_s } else { 0.0 },
    })
}

fn dedup(ids: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    ids.iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && seen.insert(s.clone()))
        .collect()
}

/// Validated notes sections.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sections {
    /// Decisions.
    pub decisions: Vec<Decision>,
    /// Action items.
    pub action_items: Vec<ActionItem>,
    /// Open questions.
    pub open_questions: Vec<OpenQuestion>,
    /// Timeline.
    pub timeline: Vec<TimelineEntry>,
    /// Summary.
    pub summary: Vec<SummaryPoint>,
}

/// Near-duplicate threshold (content-token Jaccard) within a section.
const DUP_JACCARD: f64 = 0.7;

/// Assembles checked items into sections: duplicates merged, times ordered, ids assigned.
pub fn assemble(checked: Vec<Checked>) -> Sections {
    let mut by_section: BTreeMap<Section, Vec<Checked>> = BTreeMap::new();
    for c in checked {
        let list = by_section.entry(c.section).or_default();
        let toks = content_tokens(&c.text);
        let same_owner = |a: &Checked| a.owners == c.owners;
        match list
            .iter_mut()
            .find(|x| same_owner(x) && jaccard(&content_tokens(&x.text), &toks) >= DUP_JACCARD)
        {
            Some(x) => {
                x.evidence.merge(&c.evidence);
                x.t_start_s = x.t_start_s.min(c.t_start_s);
                x.t_end_s = x.t_end_s.max(c.t_end_s);
                if x.quote.is_none() {
                    x.quote = c.quote;
                }
            }
            None => list.push(c),
        }
    }
    let mut out = Sections::default();
    for (section, mut list) in by_section {
        if section == Section::Timeline {
            list.sort_by(|a, b| a.t_start_s.total_cmp(&b.t_start_s));
        }
        for (i, c) in list.into_iter().enumerate() {
            let n = i + 1;
            match section {
                Section::Decisions => out.decisions.push(Decision {
                    id: format!("d{n}"),
                    text: c.text,
                    t_start_s: c.t_start_s,
                    t_end_s: c.t_end_s,
                    evidence: c.evidence,
                    quote: c.quote,
                }),
                Section::ActionItems => {
                    for o in &c.owners {
                        let k = out.action_items.len() + 1;
                        out.action_items.push(ActionItem {
                            id: format!("a{k}"),
                            person_id: o.person_id.clone(),
                            owner: o.name.clone(),
                            task: c.text.clone(),
                            t_s: c.t_start_s,
                            t_end_s: c.t_end_s,
                            evidence: c.evidence.clone(),
                            quote: c.quote.clone(),
                        });
                    }
                }
                Section::OpenQuestions => out.open_questions.push(OpenQuestion {
                    id: format!("q{n}"),
                    text: c.text,
                    source: QuestionSource::Transcript,
                    t_s: Some(c.t_start_s),
                    evidence: c.evidence,
                    quote: c.quote,
                }),
                Section::Timeline => out.timeline.push(TimelineEntry {
                    id: format!("t{n}"),
                    t_start_s: c.t_start_s,
                    t_end_s: c.t_end_s,
                    text: c.text,
                    evidence: c.evidence,
                }),
                Section::Summary => out.summary.push(SummaryPoint {
                    id: format!("s{n}"),
                    text: c.text,
                    t_start_s: c.t_start_s,
                    t_end_s: c.t_end_s,
                    evidence: c.evidence,
                }),
            }
        }
    }
    out
}

/// Similarity at which a board question and a transcript question are merged.
const BOARD_MERGE_JACCARD: f64 = 0.3;

/// Merges question stickies into the open questions. Returns how many were added.
pub fn merge_board_questions(qs: &mut Vec<OpenQuestion>, boards: &[BoardState]) -> usize {
    let mut added = 0;
    for b in boards {
        for s in b
            .stickies
            .iter()
            .filter(|s| s.effective_kind() == StickyKind::Question)
        {
            let events: Vec<String> = b
                .events
                .iter()
                .filter(|e| e.refs.iter().any(|r| r == &s.sticky_id))
                .map(|e| e.event_id.clone())
                .collect();
            let ev = Evidence {
                segment_ids: vec![],
                event_ids: events,
                keyframe_ids: s.keyframe_ids.clone(),
            };
            let toks = content_tokens(&s.text);
            let cites_sticky = |q: &OpenQuestion| {
                q.evidence
                    .event_ids
                    .iter()
                    .any(|e| ev.event_ids.contains(e))
            };
            let best = qs
                .iter_mut()
                .map(|q| {
                    let score = jaccard(&content_tokens(&q.text), &toks);
                    (score, q)
                })
                .filter(|(score, q)| *score >= BOARD_MERGE_JACCARD || cites_sticky(q))
                .max_by(|a, b| a.0.total_cmp(&b.0));
            match best {
                Some((_, q)) => {
                    q.source = QuestionSource::BoardAndTranscript;
                    q.evidence.merge(&ev);
                }
                None => {
                    added += 1;
                    qs.push(OpenQuestion {
                        id: String::new(),
                        text: sanitize_dashes(s.text.trim()),
                        source: QuestionSource::Board,
                        t_s: Some(s.first_seen_s),
                        evidence: ev,
                        quote: None,
                    });
                }
            }
        }
    }
    qs.sort_by(|a, b| {
        a.t_s
            .unwrap_or(f64::MAX)
            .total_cmp(&b.t_s.unwrap_or(f64::MAX))
    });
    for (i, q) in qs.iter_mut().enumerate() {
        q.id = format!("q{}", i + 1);
    }
    added
}

/// Records a dropped item.
pub fn dropped(section: Section, item: &DraftItem, reasons: Vec<String>) -> DroppedItem {
    DroppedItem {
        section: section.key().into(),
        text: item.main_text(),
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::named::NamedLine;

    fn line(id: &str, t: f64, text: &str, raw: &str) -> NamedLine {
        NamedLine {
            segment_id: id.into(),
            start_s: t,
            end_s: t + 3.0,
            person_id: None,
            speaker: "X".into(),
            text: text.into(),
            text_raw: raw.into(),
            speaker_confidence: 0.9,
            relabeled: false,
            gap_fill_share: 0.0,
        }
    }

    fn corpus() -> Corpus {
        let lines = vec![
            line(
                "s1",
                10.0,
                "Let's skip the Ledgerly step for now.",
                "Let's skip the Ledger Lee step for now.",
            ),
            line(
                "s2",
                14.0,
                "We'll focus on the relay side.",
                "We'll focus on the relay side.",
            ),
            line(
                "s3",
                60.0,
                "I'll see you guys later.",
                "I'll see you guys later.",
            ),
        ];
        Corpus::new(
            &lines,
            &[],
            AliasTable::from_names(&["Avery Quinn", "Rohan Dasgupta"]),
        )
    }

    fn item(text: &str, ids: &[&str], quote: &str) -> DraftItem {
        DraftItem {
            text: text.into(),
            segment_ids: ids.iter().map(|s| (*s).to_string()).collect(),
            quote: quote.into(),
            ..Default::default()
        }
    }

    #[test]
    fn quotes_match_raw_then_corrected_and_across_segments() {
        let c = corpus();
        let ok = check(
            Section::Decisions,
            &item(
                "Skip the ledger step for now",
                &["s1"],
                "skip the Ledger Lee step",
            ),
            &c,
        )
        .unwrap();
        assert_eq!(ok.quote.unwrap().matched, QuoteMatch::Raw);
        let ok = check(
            Section::Decisions,
            &item(
                "Skip the ledger step for now",
                &["s1"],
                "skip the Ledgerly step",
            ),
            &c,
        )
        .unwrap();
        assert_eq!(ok.quote.unwrap().matched, QuoteMatch::Corrected);
        let ok = check(
            Section::Decisions,
            &item(
                "Skip it and focus on the relay",
                &["s2", "s1"],
                "skip the Ledgerly step ... focus on the relay side",
            ),
            &c,
        )
        .unwrap();
        assert_eq!((ok.t_start_s, ok.t_end_s), (10.0, 17.0));
        let err = check(
            Section::Decisions,
            &item("Skip the ledger step", &["s2"], "skip the Ledgerly step"),
            &c,
        )
        .unwrap_err();
        assert!(err.reasons[0].contains("not in the cited segments"));
    }

    #[test]
    fn unknown_ids_and_missing_citations_fail() {
        let c = corpus();
        let e = check(
            Section::Decisions,
            &item("Skip the ledger step", &["s9"], ""),
            &c,
        )
        .unwrap_err();
        assert!(e
            .reasons
            .iter()
            .any(|r| r.contains("unknown segment id s9")));
        let e = check(Section::Summary, &item("A summary point here", &[], ""), &c).unwrap_err();
        assert!(e.reasons.iter().any(|r| r == "no citations"));
    }

    #[test]
    fn action_items_need_owner_verb_object_and_no_farewells() {
        let c = corpus();
        let mut a = item("", &["s2"], "");
        a.owner = "Rowan".into();
        a.task = "Design the relay storage schema".into();
        let ok = check(Section::ActionItems, &a, &c).unwrap();
        assert_eq!(ok.owners[0].person_id.as_deref(), Some("rohan-dasgupta"));
        a.task = "Backend: the relay".into();
        assert!(check(Section::ActionItems, &a, &c).is_err());
        a.task = "See you guys later".into();
        a.segment_ids = vec!["s3".into()];
        assert!(check(Section::ActionItems, &a, &c).unwrap_err().fatal);
        a.owner = "Everyone and Avery".into();
        a.task = "Push work to a branch early".into();
        let ok = check(Section::ActionItems, &a, &c).unwrap();
        assert_eq!(ok.owners.len(), 2);
        a.owner = "Kristen".into();
        assert!(check(Section::ActionItems, &a, &c).is_err());
    }

    #[test]
    fn em_dashes_are_sanitized() {
        let c = corpus();
        let ok = check(
            Section::Decisions,
            &item("Skip the ledger step \u{2014} for now", &["s1"], ""),
            &c,
        )
        .unwrap();
        assert_eq!(ok.text, "Skip the ledger step, for now");
    }

    #[test]
    fn duplicates_merge_and_board_questions_join() {
        let c = corpus();
        let a = check(
            Section::OpenQuestions,
            &item("Which widgets does the kiosk need?", &["s1"], ""),
            &c,
        )
        .unwrap();
        let b = check(
            Section::OpenQuestions,
            &item("Which widgets does the kiosk need", &["s2"], ""),
            &c,
        )
        .unwrap();
        let mut s = assemble(vec![a, b]);
        assert_eq!(s.open_questions.len(), 1);
        assert_eq!(s.open_questions[0].evidence.segment_ids, vec!["s1", "s2"]);
        let board: BoardState = serde_json::from_value(serde_json::json!({
            "board_id": "b", "final_t_s": 90.0, "nodes": [], "edges": [],
            "stickies": [
                {"sticky_id": "st1", "text": "What widgets do we need for the kiosk?", "first_seen_s": 5.0, "keyframe_ids": ["kf_1"]},
                {"sticky_id": "st2", "text": "Do we change the badge flow?", "first_seen_s": 40.0, "keyframe_ids": ["kf_2"]},
                {"sticky_id": "st3", "text": "Clone the lobby page", "first_seen_s": 41.0}
            ],
            "keyframes": [{"keyframe_id": "kf_1", "t_rep_s": 5.0}, {"keyframe_id": "kf_2", "t_rep_s": 40.0}]
        }))
        .unwrap();
        let added = merge_board_questions(&mut s.open_questions, &[board]);
        assert_eq!(added, 1);
        assert_eq!(
            s.open_questions[0].source,
            QuestionSource::BoardAndTranscript
        );
        assert_eq!(s.open_questions[1].source, QuestionSource::Board);
        assert_eq!(s.open_questions[1].id, "q2");
    }

    #[test]
    fn schema_requires_every_section() {
        let s = draft_schema();
        assert_eq!(s["required"].as_array().unwrap().len(), 5);
        assert_eq!(
            s["properties"]["action_items"]["items"]["required"][0],
            "owner"
        );
    }
}
