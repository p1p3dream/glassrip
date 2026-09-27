//! The Rust side of the notes: what the model drafts is checked here.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{
    ActionItem, Decision, DroppedItem, Evidence, OpenQuestion, QuestionSource, Quote, QuoteMatch,
    SummaryPoint, TimelineEntry,
};
use crate::board::{first_seen, BoardExt, BoardStateItem, KeyframeTimes, StickyKind};
use crate::named::NamedLine;
use crate::people::AliasTable;
use crate::text::{
    content_tokens, distinctive_tokens, is_stopword, jaccard, names_target, normalize,
    sanitize_dashes, tokens,
};

/// Most `...` elisions in one quote.
pub const MAX_ELISIONS: usize = 2;
/// Fewest words in a quote.
pub const MIN_QUOTE_TOKENS: usize = 5;
/// Fewest words in each part of an elided quote.
pub const MIN_PART_TOKENS: usize = 3;
/// Share of the matched transcript span the quote's words must cover.
pub const MIN_QUOTE_COVERAGE: f64 = 0.6;

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
    /// Ids that show an owner tag appearing or moving (owner events, and the
    /// keyframes that opened owner assignments), with the owner and target of
    /// each assignment they concern.
    owner_keys: BTreeMap<String, Vec<OwnerKey>>,
    /// Every owner assignment on the boards.
    assignments: Vec<OwnerKey>,
    /// Transcript lines in time order (who replied to whom).
    lines: Vec<NamedLine>,
    /// Indices into `lines` of each segment's lines.
    line_index: BTreeMap<String, Vec<usize>>,
}

/// The owner and target of an owner-tag change, as tokens.
#[derive(Debug, Clone)]
struct OwnerKey {
    /// Tokens of the owner's display name.
    name: Vec<String>,
    /// The owner's display name.
    owner: String,
    /// Content tokens of the target.
    target: Vec<String>,
    /// The target as text.
    target_text: String,
    /// The owner tag moved here from another target (a change the group made);
    /// a plain owner tag only says who is responsible.
    moved: bool,
    /// The owner tag is still on the board at the end (a current task).
    current: bool,
    /// Content tokens of the target the tag moved from (empty: not a move).
    from: Vec<String>,
    /// The target the tag moved from, as text.
    from_text: String,
    /// When the tag appeared here, seconds.
    since_s: f64,
}

/// Words that state ownership or assignment ("Avery owns the kiosk", "take
/// the ledger", "assigned to").
const OWNERSHIP_WORDS: &[&str] = &[
    "own",
    "owns",
    "owned",
    "owning",
    "owner",
    "owners",
    "ownership",
    "take",
    "takes",
    "taking",
    "took",
    "assign",
    "assigns",
    "assigned",
    "assigning",
    "assignment",
    "responsible",
    "lead",
    "leads",
    "leading",
];

impl OwnerKey {
    /// The item's words refer to this owner-tag change: they name the target
    /// and either the owner or the ownership ("Avery owns the kiosk rollout",
    /// "Own the kiosk work"), or they name the target in full (every content
    /// word, or most of at least two distinctive words). One shared word
    /// ("Buy a new office printer" against "Badge Printer") is not enough.
    fn backs(&self, words: &BTreeSet<String>) -> bool {
        if self.target.is_empty() {
            return false;
        }
        let owner_ref = self.name.iter().any(|t| words.contains(t))
            || OWNERSHIP_WORDS.iter().any(|t| words.contains(*t));
        if owner_ref && names_target(words, &self.target) {
            return true;
        }
        if self.target.iter().all(|t| words.contains(t)) {
            return true;
        }
        let distinctive = distinctive_tokens(&self.target);
        let hit = distinctive.iter().filter(|t| words.contains(*t)).count();
        hit >= 2 && hit * 2 > distinctive.len()
    }
}

/// Verbs of an ownership task ("Own the kiosk", "Work on the importer",
/// "Take the ledger").
const OWNERSHIP_VERBS: &[&str] = &["own", "work", "lead", "handle", "manage", "take", "drive"];
/// Words that only say "ownership" in an ownership task.
const OWNERSHIP_FILLERS: &[&str] = &["ownership", "over", "owner", "responsibility"];

/// Explicit negation changes the action even when the remaining content
/// words are identical ("cache the photos" versus "do not cache the photos").
fn explicitly_negated(text: &str) -> bool {
    tokens(text).iter().any(|t| {
        matches!(
            t.as_str(),
            "not"
                | "no"
                | "never"
                | "dont"
                | "doesnt"
                | "wont"
                | "cant"
                | "cannot"
                | "shouldnt"
                | "wouldnt"
                | "couldnt"
        )
    })
}

/// The content words an ownership task says the owner owns: the task opens
/// (after "to", "will") with an ownership verb, and what follows is kept
/// without stopwords and ownership fillers. `None` for other tasks.
pub fn ownership_words(task: &str) -> Option<BTreeSet<String>> {
    let toks = tokens(task);
    let mut i = 0;
    while toks
        .get(i)
        .is_some_and(|t| TASK_PREFIXES.contains(&t.as_str()))
    {
        i += 1;
    }
    if !OWNERSHIP_VERBS.contains(&toks.get(i)?.as_str()) {
        return None;
    }
    let words: BTreeSet<String> = toks[i + 1..]
        .iter()
        .filter(|t| !is_stopword(t) && !OWNERSHIP_FILLERS.contains(&t.as_str()))
        .cloned()
        .collect();
    (!words.is_empty()).then_some(words)
}

/// The ownership words name this target and add at most one word of their
/// own ("Own the kiosk rollout" owns the Kiosk App; "Work on the kiosk crash
/// report for the audit" is a task about it, not its ownership).
fn owns_target(words: &BTreeSet<String>, target: &[String]) -> bool {
    names_target(words, target) && words.iter().filter(|w| !target.contains(w)).count() <= 1
}

/// Decisions backed only by a plain owner tag (none moved) that is still on
/// the board are refiled as "Own <target>" action items for the tag's owner,
/// keeping their citations: an owner tag says who is responsible, which is a
/// task, not a decision. Skipped when the draft already has an action of that
/// owner naming the target. Returns how many were refiled.
pub fn refile_owner_decisions(draft: &mut Draft, corpus: &Corpus) -> usize {
    let mut kept = Vec::new();
    let mut refiled = 0;
    for d in std::mem::take(&mut draft.decisions) {
        let words: BTreeSet<String> = tokens(&d.text).into_iter().collect();
        let backing: Vec<&OwnerKey> = d
            .event_ids
            .iter()
            .chain(&d.keyframe_ids)
            .filter_map(|id| corpus.owner_keys.get(id.trim()))
            .flatten()
            .filter(|k| k.backs(&words))
            .collect();
        let plain = backing.iter().all(|k| !k.moved);
        let Some(k) = backing.iter().find(|k| k.current).filter(|_| plain) else {
            kept.push(d);
            continue;
        };
        let has_action = draft.action_items.iter().any(|a| {
            let owner = tokens(&a.owner);
            let task: BTreeSet<String> = content_tokens(&a.task).into_iter().collect();
            (owner == k.name || owner.first() == k.name.first()) && names_target(&task, &k.target)
        });
        if !has_action {
            draft.action_items.push(DraftItem {
                owner: k.owner.clone(),
                task: format!("Own {}", k.target_text),
                text: String::new(),
                ..d
            });
        }
        refiled += 1;
    }
    draft.decisions = kept;
    refiled
}

/// Validation switches (see [`super::candidates`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CheckOptions {
    /// Action items may be supported by an owner tag on the board, and
    /// decisions by an owner tag moved on the board, instead of a transcript
    /// segment.
    pub board_support: bool,
    /// Decisions need a speaker commitment in a cited segment, or an owner tag
    /// moved on the board.
    pub precision_guard: bool,
    /// Remove narrative prefixes ("The team decided to") from decisions and tasks.
    pub strip_prefix: bool,
}

impl Corpus {
    /// Builds the corpus from named transcript lines and board states.
    pub fn new(
        lines: &[NamedLine],
        boards: &[BoardStateItem],
        keyframes: &KeyframeTimes,
        table: AliasTable,
    ) -> Self {
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
        // citable board ids: events, and keyframes (times from glassrip.keyframes;
        // an event's keyframe falls back to the event time)
        let mut times = BTreeMap::new();
        for b in boards {
            for e in &b.events {
                times.insert(e.event_id.clone(), e.t_s);
                times
                    .entry(e.keyframe_id.clone())
                    .or_insert(keyframes.get(&e.keyframe_id).unwrap_or(e.t_s));
            }
            for k in &b.board_keyframes {
                if let Some(t) = keyframes.get(k) {
                    times.insert(k.clone(), t);
                }
            }
        }
        let mut owner_keys: BTreeMap<String, Vec<OwnerKey>> = BTreeMap::new();
        let mut assignments = Vec::new();
        for b in boards {
            for o in &b.owner_assignments {
                let target_text = sanitize_dashes(&crate::board::target_text(&o.target));
                let from_text = o
                    .moved_from
                    .as_ref()
                    .map(|t| sanitize_dashes(&crate::board::target_text(t)))
                    .unwrap_or_default();
                let words = OwnerKey {
                    name: tokens(&o.display_name),
                    owner: o.display_name.trim().to_string(),
                    target: content_tokens(&target_text),
                    target_text,
                    moved: o.moved_from.is_some(),
                    current: o.valid_to_s >= b.end_s() - 0.5,
                    from: content_tokens(&from_text),
                    from_text,
                    since_s: o.valid_from_s,
                };
                assignments.push(words.clone());
                // the owner events of this assignment (same person, same target
                // by id, same kind, near its start): an event backs a move only
                // through the move assignment it was matched to
                for e in super::candidates::owner_events(b, o) {
                    owner_keys
                        .entry(e.event_id.clone())
                        .or_default()
                        .push(words.clone());
                }
                if !o.opened_at_keyframe.is_empty() {
                    owner_keys
                        .entry(o.opened_at_keyframe.clone())
                        .or_default()
                        .push(words.clone());
                    times.entry(o.opened_at_keyframe.clone()).or_insert(
                        keyframes
                            .get(&o.opened_at_keyframe)
                            .unwrap_or(o.valid_from_s),
                    );
                }
            }
        }
        let mut ordered: Vec<NamedLine> = lines.to_vec();
        ordered.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
        let mut line_index: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, l) in ordered.iter().enumerate() {
            line_index.entry(l.segment_id.clone()).or_default().push(i);
        }
        Self {
            segs,
            times,
            table,
            owner_keys,
            assignments,
            lines: ordered,
            line_index,
        }
    }

    /// A cited segment holds a standing commitment that no reply retracts or
    /// objects to, shares a topic word with the item (`words`, see
    /// [`super::candidates::topic_words`]), and does not commit to the
    /// opposite: when the committing clauses on the item's topic all have the
    /// other polarity ("Let's not ship the importer" for "Ship the importer"),
    /// the line does not support it. Polarity is judged per clause, so a
    /// negation elsewhere in a long line does not flip it.
    fn commits(&self, id: &str, claim: &str, words: &BTreeSet<String>) -> bool {
        let Some(s) = self.segs.get(id) else {
            return false;
        };
        let supports = |source: &str| {
            if !super::candidates::has_standing_commitment(source)
                || super::candidates::topic_words(source).is_disjoint(words)
            {
                return false;
            }
            let on_topic: Vec<String> = super::candidates::committing_clauses(source)
                .into_iter()
                .filter(|c| !super::candidates::topic_words(c).is_disjoint(words))
                .collect();
            let opposite = !on_topic.is_empty()
                && on_topic
                    .iter()
                    .all(|c| explicitly_negated(c) != explicitly_negated(claim));
            !opposite
        };
        let committed = supports(&s.text) || supports(&s.text_raw);
        committed
            && !self
                .line_index
                .get(id)
                .and_then(|v| v.last())
                .is_some_and(|i| super::candidates::retracted_after(&self.lines, *i))
    }

    /// The segment proposes something another speaker then agreed to
    /// ([`super::candidates::accepted_proposal`]); the latest proposed clause
    /// of the line (the one an agreement answers) shares at least two of the
    /// item's topic words (`words`; all of them when it has fewer) and has the
    /// same polarity. One shared word, or a negated proposal, does not back
    /// the item. Returns the agreeing segment.
    fn agreement(&self, id: &str, words: &BTreeSet<String>, claim: &str) -> Option<String> {
        self.line_index.get(id)?.iter().find_map(|i| {
            let l = &self.lines[*i];
            // Acceptance is evaluated on the corrected line, so a word that
            // appears only in a different raw transcript cannot support it.
            let proposal = super::candidates::latest_proposal_clause(&l.text)?;
            let shared = super::candidates::topic_words(&proposal)
                .intersection(words)
                .count();
            if words.is_empty()
                || shared < words.len().min(2)
                || explicitly_negated(claim) != explicitly_negated(&proposal)
            {
                return None;
            }
            super::candidates::accepted_proposal(&self.lines, *i)
                .map(|j| self.lines[j].segment_id.clone())
        })
    }

    /// For an ownership task ([`ownership_words`]) of `owner`: the target it
    /// names when every owner tag of this person on that target was taken
    /// off before the end (moved elsewhere or removed) and none is still on
    /// it. "Work on the kiosk" after the owner's tag moved from the kiosk to
    /// the ledger is no longer their task.
    fn superseded_ownership(&self, owner: &str, task: &str) -> Option<String> {
        let words = ownership_words(task)?;
        let name = tokens(owner);
        let mine: Vec<&OwnerKey> = self.assignments.iter().filter(|k| k.name == name).collect();
        let named: Vec<&OwnerKey> = mine
            .into_iter()
            .filter(|k| owns_target(&words, &k.target))
            .collect();
        if named.iter().any(|k| k.current) {
            return None;
        }
        named
            .into_iter()
            .find(|k| !k.current)
            .map(|k| k.target_text.clone())
    }

    /// For a task of `owner` said only before (`said_until`, the end of its
    /// last cited line) this person's owner tag moved away from a target the
    /// task names: the move, as (from, to). A change in who does what
    /// supersedes the earlier plan for the old target. A task with any word of
    /// the new target, a task said after the move, and a person whose tag is
    /// back on the old target at the end are kept.
    fn moved_off(
        &self,
        owner: &str,
        task: &str,
        said_until: Option<f64>,
    ) -> Option<(String, String)> {
        let end = said_until?;
        let words: BTreeSet<String> = content_tokens(task).into_iter().collect();
        let name = tokens(owner);
        let mine: Vec<&OwnerKey> = self.assignments.iter().filter(|k| k.name == name).collect();
        mine.iter()
            .filter(|k| k.moved && !k.from.is_empty() && end < k.since_s)
            .find(|k| {
                names_target(&words, &k.from)
                    // any word of the new target, generic ones included,
                    // ties the task to it ("map ledger entries to kit widgets"
                    // after a move from the Ledger Service to the Design Kit)
                    && !k.target.iter().any(|t| words.contains(t))
                    && !mine.iter().any(|a| a.current && a.target == k.from)
            })
            .map(|k| (k.from_text.clone(), k.target_text.clone()))
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
    /// Checks a quote (with optional `...` elisions) against the cited segments.
    ///
    /// A quote passes when it has at most [`MAX_ELISIONS`] elisions and at least
    /// [`MIN_QUOTE_TOKENS`] tokens, every part has at least [`MIN_PART_TOKENS`]
    /// tokens with one that is not a stopword, the parts appear in order in the
    /// cited segments' verbatim text (else their corrected text), and the quote's
    /// tokens cover at least [`MIN_QUOTE_COVERAGE`] of the transcript span they
    /// match (elisions cannot skip most of what was said).
    fn find_quote(&self, quote: &str, cited: &[&str]) -> Result<(QuoteMatch, String), String> {
        let parts: Vec<Vec<String>> = quote
            .split("...")
            .flat_map(|p| p.split('\u{2026}'))
            .map(tokens)
            .filter(|p| !p.is_empty())
            .collect();
        if parts.len() > MAX_ELISIONS + 1 {
            return Err(format!("quote has more than {MAX_ELISIONS} elisions"));
        }
        let total: usize = parts.iter().map(Vec::len).sum();
        if total < MIN_QUOTE_TOKENS {
            return Err(format!("quote is shorter than {MIN_QUOTE_TOKENS} words"));
        }
        if parts
            .iter()
            .any(|p| p.len() < MIN_PART_TOKENS || p.iter().all(|t| is_stopword(t)))
        {
            return Err(format!(
                "each quoted part needs {MIN_PART_TOKENS} or more words, not only common words"
            ));
        }
        let mut segs: Vec<(&str, &SegText)> = cited
            .iter()
            .filter_map(|id| self.segs.get(*id).map(|s| (*id, s)))
            .collect();
        segs.sort_by(|a, b| a.1.start_s.total_cmp(&b.1.start_s));
        let mut best_coverage = 0.0f64;
        for (raw, kind) in [(true, QuoteMatch::Raw), (false, QuoteMatch::Corrected)] {
            // tokens of the cited segments in time order, with their segment
            let hay: Vec<(&str, &str)> = segs
                .iter()
                .flat_map(|(id, s)| {
                    let t = if raw { &s.norm_raw } else { &s.norm };
                    t.split(' ')
                        .filter(|w| !w.is_empty())
                        .map(move |w| (*id, w))
                })
                .collect();
            let at = |i: usize, part: &[String]| {
                part.iter()
                    .enumerate()
                    .all(|(k, t)| hay.get(i + k).is_some_and(|h| h.1 == t))
            };
            let mut best: Option<(usize, usize)> = None;
            for start in 0..hay.len() {
                if !at(start, &parts[0]) {
                    continue;
                }
                let mut pos = start + parts[0].len();
                let mut ok = true;
                for part in &parts[1..] {
                    match (pos..hay.len()).find(|i| at(*i, part)) {
                        Some(i) => pos = i + part.len(),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && best.is_none_or(|(a, b)| pos - start < b - a) {
                    best = Some((start, pos));
                }
            }
            if let Some((a, b)) = best {
                let coverage = total as f64 / (b - a) as f64;
                if coverage >= MIN_QUOTE_COVERAGE {
                    return Ok((kind, hay[a].0.to_string()));
                }
                best_coverage = best_coverage.max(coverage);
            }
        }
        if best_coverage > 0.0 {
            Err(format!(
                "quote elides too much: it covers {:.0}% of the words it spans",
                best_coverage * 100.0
            ))
        } else {
            Err(format!("quote \"{quote}\" is not in the cited segments"))
        }
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
    "drive",
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
    "handle",
    "hook",
    "identify",
    "implement",
    "import",
    "improve",
    "inspect",
    "install",
    "integrate",
    "investigate",
    "lead",
    "learn",
    "list",
    "look",
    "make",
    "manage",
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

/// True for a verb that starts a task.
pub fn is_task_verb(word: &str) -> bool {
    TASK_VERBS.contains(&word)
}

/// Words of personal activities that are not work tasks ("grab a coffee",
/// "call my mom").
const PERSONAL: &[&str] = &[
    "coffee",
    "lunch",
    "breakfast",
    "dinner",
    "snack",
    "bathroom",
    "restroom",
    "water",
    "drink",
    "nap",
    "doctor",
    "appointment",
    "errand",
    "errands",
    "gym",
    "dentist",
    "haircut",
    "groceries",
    "grocery",
    "pharmacy",
    "babysitter",
    "daycare",
    "mom",
    "dad",
    "mother",
    "father",
    "kids",
    "wife",
    "husband",
    "son",
    "daughter",
];
/// Phrases of leaving or pausing ("head out", "step away").
const PERSONAL_PHRASES: &[&str] = &[
    "head out",
    "step away",
    "step out",
    "grab a",
    "be right back",
    "a break",
    "quick break",
    "take a walk",
    "go for a walk",
    "some air",
    "fresh air",
    "take a quick call",
    "get the door",
    "answer the door",
    "drive home",
    "head home",
    "go home",
];

/// Words that open a time adjunct ("after lunch", "when I'm back from the
/// dentist"): it says when, not what.
const TIME_ADJUNCTS: &[&str] = &[
    "after", "before", "during", "until", "till", "once", "while", "when",
];

/// The text without its time adjuncts: a trailing one ("review the logs
/// after lunch") is cut at its first word, and a fronted one set off by a comma
/// ("After lunch, review the logs") is left out. A fronted adjunct with nothing after
/// it is kept, since it may be all there is.
fn without_time_adjuncts(text: &str) -> String {
    let pieces: Vec<Vec<String>> = text
        .split([',', ';', ':'])
        .map(tokens)
        .filter(|p| !p.is_empty())
        .collect();
    let mut kept: Vec<String> = Vec::new();
    for (i, p) in pieces.iter().enumerate() {
        let fronted = TIME_ADJUNCTS.contains(&p[0].as_str());
        if fronted {
            if i + 1 < pieces.len() {
                continue;
            }
            kept.push(p.join(" "));
            continue;
        }
        let cut = p
            .iter()
            .position(|t| TIME_ADJUNCTS.contains(&t.as_str()))
            .unwrap_or(p.len());
        kept.push(p[..cut].join(" "));
    }
    kept.join(" , ")
}

/// True when a task is a personal activity rather than work. A time adjunct
/// does not make work personal ("Review the kiosk logs after lunch" is
/// work; "Grab lunch after the demo" is not).
pub fn is_personal_activity(text: &str) -> bool {
    let toks = tokens(&without_time_adjuncts(text));
    // "walk through the design" is a work walkthrough, not a walk
    let n = format!(" {} ", toks.join(" ")).replace(" walk through ", " walkthrough ");
    toks.iter().any(|t| PERSONAL.contains(&t.as_str()))
        || PERSONAL_PHRASES
            .iter()
            .any(|p| n.contains(&format!(" {p} ")))
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
    check_with(section, item, corpus, &CheckOptions::default())
}

/// Validates one drafted item with options.
pub fn check_with(
    section: Section,
    item: &DraftItem,
    corpus: &Corpus,
    opts: &CheckOptions,
) -> Result<Checked, Failure> {
    let mut reasons = Vec::new();
    let raw_text = if section == Section::ActionItems {
        &item.task
    } else {
        &item.text
    };
    let text = if opts.strip_prefix && matches!(section, Section::Decisions | Section::ActionItems)
    {
        sanitize_dashes(&super::candidates::strip_item_prefix(raw_text))
    } else {
        sanitize_dashes(raw_text)
    };
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
    if section == Section::ActionItems && is_personal_activity(&text) {
        return Err(Failure {
            reasons: vec!["personal activity, not a work task".into()],
            fatal: true,
        });
    }
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
    // board support counts only when the item's own text refers to the
    // owner-tag change it cites (see OwnerKey::backs): a name-drop, one shared
    // target word or a generic word does not; the owner field is not text
    // a decision needs an owner-tag move: a plain owner tag only says who is
    // responsible, which is an action item, not a decision
    let item_words: BTreeSet<String> = tokens(&text).into_iter().collect();
    let board_backed = evidence
        .event_ids
        .iter()
        .chain(&evidence.keyframe_ids)
        .filter_map(|id| corpus.owner_keys.get(id))
        .flatten()
        .filter(|k| match section {
            Section::Decisions => k.moved,
            // a tag taken off before the end is not a current task
            Section::ActionItems => k.current,
            _ => true,
        })
        .any(|k| k.backs(&item_words));
    let board_ok = opts.board_support
        && board_backed
        && matches!(section, Section::Decisions | Section::ActionItems);
    if needs_segment && evidence.segment_ids.is_empty() && !board_ok {
        reasons.push("must cite at least one transcript segment".into());
    }
    // the line where another speaker agreed to a cited proposal, added to the
    // evidence once the item passes
    let mut agreed_segment = None;
    if opts.precision_guard && section == Section::Decisions && !board_backed {
        let content_words = super::candidates::topic_words(&text);
        let committed = evidence
            .segment_ids
            .iter()
            .any(|id| corpus.commits(id, &text, &content_words));
        if !committed {
            agreed_segment = evidence
                .segment_ids
                .iter()
                .find_map(|id| corpus.agreement(id, &content_words, &text));
            if agreed_segment.is_none() {
                reasons.push(
                    "a decision must cite a line where someone commits to it and does not take it back, or a proposal another participant agreed to (or an owner tag moved on the board)"
                        .into(),
                );
            }
        }
    }
    if section == Section::ActionItems {
        let said_until = evidence
            .segment_ids
            .iter()
            .filter_map(|id| corpus.segment_span(id))
            .map(|s| s.1)
            .reduce(f64::max);
        for o in &owners {
            if let Some(target) = corpus.superseded_ownership(&o.name, &text) {
                return Err(Failure {
                    reasons: vec![format!(
                        "{}'s owner tag on {target} was taken off the board: the task moved on",
                        o.name
                    )],
                    fatal: true,
                });
            }
            if let Some((from, to)) = corpus.moved_off(&o.name, &text, said_until) {
                return Err(Failure {
                    reasons: vec![format!(
                        "{}'s owner tag moved from {from} to {to} after this was said: the task moved on",
                        o.name
                    )],
                    fatal: true,
                });
            }
        }
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
            Ok((matched, segment_id)) => {
                quote = Some(Quote {
                    text: q,
                    segment_id,
                    matched,
                })
            }
            Err(why) => reasons.push(why),
        }
    }
    if !reasons.is_empty() {
        return Err(Failure {
            reasons,
            fatal: false,
        });
    }
    let mut evidence = evidence;
    if let Some(id) = agreed_segment {
        if !evidence.segment_ids.contains(&id) {
            evidence.segment_ids.push(id);
        }
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

/// Two items restate one claim: they cite a common id, and the content
/// words of one (at least two) are all in the other ("Archive the draft on
/// Friday" and "Archive the draft on Friday after review", both citing
/// the line that says it), or both are ownership tasks whose owned words are
/// nested or share two words and differ by at most one on each side ("Own the
/// Kiosk App", "Work on the Kiosk App"; "Own the kiosk app link", "Work on
/// the kiosk app integration"). Items citing different evidence are kept
/// apart however alike.
fn restates(a: &Checked, b: &Checked) -> bool {
    if explicitly_negated(&a.text) != explicitly_negated(&b.text) {
        return false;
    }
    let ids = |e: &Evidence| -> BTreeSet<String> {
        e.segment_ids
            .iter()
            .chain(&e.event_ids)
            .chain(&e.keyframe_ids)
            .cloned()
            .collect()
    };
    if ids(&a.evidence).is_disjoint(&ids(&b.evidence)) {
        return false;
    }
    let nested = |x: &BTreeSet<String>, y: &BTreeSet<String>| {
        let (small, large) = if x.len() <= y.len() { (x, y) } else { (y, x) };
        small.is_subset(large)
    };
    let ta: BTreeSet<String> = content_tokens(&a.text).into_iter().collect();
    let tb: BTreeSet<String> = content_tokens(&b.text).into_iter().collect();
    if ta.len().min(tb.len()) >= 2 && nested(&ta, &tb) {
        return true;
    }
    a.section == Section::ActionItems
        && match (ownership_words(&a.text), ownership_words(&b.text)) {
            // "Own the kiosk app link" and "Work on the kiosk app
            // integration": the same target with one word of framing each
            (Some(x), Some(y)) => {
                nested(&x, &y)
                    || (x.intersection(&y).count() >= 2
                        && x.difference(&y).count() <= 1
                        && y.difference(&x).count() <= 1)
            }
            _ => false,
        }
}

/// Assembles checked items into sections: duplicates merged, times ordered, ids assigned.
pub fn assemble(checked: Vec<Checked>) -> Sections {
    let mut by_section: BTreeMap<Section, Vec<Checked>> = BTreeMap::new();
    for c in checked {
        let list = by_section.entry(c.section).or_default();
        let toks = content_tokens(&c.text);
        let same_owner = |a: &Checked| a.owners == c.owners;
        match list.iter_mut().find(|x| {
            same_owner(x)
                && explicitly_negated(&x.text) == explicitly_negated(&c.text)
                && (jaccard(&content_tokens(&x.text), &toks) >= DUP_JACCARD || restates(x, &c))
        }) {
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
pub fn merge_board_questions(qs: &mut Vec<OpenQuestion>, boards: &[BoardStateItem]) -> usize {
    let mut added = 0;
    for b in boards {
        for s in b.stickies.iter().filter(|s| s.kind == StickyKind::Question) {
            let events = b.events_of(&s.id);
            let mut keyframe_ids: Vec<String> =
                events.iter().map(|e| e.keyframe_id.clone()).collect();
            keyframe_ids.dedup();
            let ev = Evidence {
                segment_ids: vec![],
                event_ids: events.iter().map(|e| e.event_id.clone()).collect(),
                keyframe_ids,
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
                        t_s: Some(first_seen(&s.lifetimes)),
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
    use crate::board::EventKind;
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
                "I'll see you folks tomorrow.",
                "I'll see you folks tomorrow.",
            ),
        ];
        Corpus::new(
            &lines,
            &[],
            &KeyframeTimes::default(),
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
                "skip the Ledger Lee step for",
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
                "skip the Ledgerly step for now",
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
            &item(
                "Skip the ledger step",
                &["s2"],
                "skip the Ledgerly step for now",
            ),
            &c,
        )
        .unwrap_err();
        assert!(err.reasons[0].contains("not in the cited segments"));
    }

    #[test]
    fn elided_quotes_cannot_launder_common_words() {
        let mut lines = vec![line(
            "s9",
            100.0,
            "We should move the relay config into the ledger service so that the kiosk team can ship the badge printer changes next month without waiting on us to review it.",
            "We should move the relay config into the ledger service so that the kiosk team can ship the badge printer changes next month without waiting on us to review it.",
        )];
        lines.push(line(
            "s1",
            10.0,
            "Let's skip the Ledgerly step for now.",
            "Let's skip the Ledger Lee step for now.",
        ));
        let c = Corpus::new(
            &lines,
            &[],
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Avery Quinn"]),
        );
        let fails = |quote: &str, why: &str| {
            let e = check(
                Section::Decisions,
                &item("Move the relay config into the ledger", &["s9"], quote),
                &c,
            )
            .unwrap_err();
            assert!(
                e.reasons.iter().any(|r| r.contains(why)),
                "{quote}: {:?}",
                e.reasons
            );
        };
        // stopword-only parts
        fails("we ... the ... to", "words");
        fails("we should ... the ... to us", "words");
        // too many elisions
        fails(
            "we should move ... relay config into ... ledger service so ... kiosk team can",
            "elisions",
        );
        // too short overall
        fails("relay config into", "shorter than");
        // elision skips most of the segment
        fails("we should move ... on us to review it", "elides too much");
        // a real quote with one short elision passes
        let ok = check(
            Section::Decisions,
            &item(
                "Move the relay config into the ledger",
                &["s9"],
                "move the relay config ... the ledger service",
            ),
            &c,
        )
        .unwrap();
        assert_eq!(ok.quote.unwrap().segment_id, "s9");
    }

    #[test]
    fn board_support_precision_guard_and_prefixes() {
        use crate::board::build;
        let mut b = build::board("b", 100.0);
        b.nodes = vec![build::node("n1", "Kiosk App", 0.0, 100.0, None)];
        let mut o = build::owner(
            "avery-quinn",
            "Avery Quinn",
            build::node_target(&b, "n1"),
            40.0,
            100.0,
            None,
        );
        o.opened_at_keyframe = "kf_000040".into();
        b.owner_assignments = vec![o];
        let lines = vec![
            line(
                "s1",
                10.0,
                "The importer reads the ledger nightly.",
                "The importer reads the ledger nightly.",
            ),
            line(
                "s2",
                20.0,
                "Let's skip the importer for now.",
                "Let's skip the importer for now.",
            ),
        ];
        let c = Corpus::new(
            &lines,
            &[b],
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Avery Quinn"]),
        );
        let all = CheckOptions {
            board_support: true,
            precision_guard: true,
            strip_prefix: true,
        };
        // an owner-tag action with only the board citation
        let mut a = DraftItem {
            owner: "Avery".into(),
            task: "Own the Kiosk App work".into(),
            keyframe_ids: vec!["kf_000040".into()],
            ..Default::default()
        };
        assert!(check_with(Section::ActionItems, &a, &c, &all).is_ok());
        assert!(
            check(Section::ActionItems, &a, &c).is_err(),
            "off by default"
        );
        // a keyframe that did not open an owner tag is not board support
        a.keyframe_ids = vec!["kf_999999".into()];
        assert!(check_with(Section::ActionItems, &a, &c, &all).is_err());
        // the guard: a decision citing only a descriptive line fails
        let d = item("The team decided to skip the importer", &["s1"], "");
        let e = check_with(Section::Decisions, &d, &c, &all).unwrap_err();
        assert!(e.reasons.iter().any(|r| r.contains("commits")), "{e:?}");
        // citing an unrelated owner-tag change does not bypass the guard
        let mut d = item("The team decided to skip the importer", &["s1"], "");
        d.keyframe_ids = vec!["kf_000040".into()];
        let e = check_with(Section::Decisions, &d, &c, &all).unwrap_err();
        assert!(e.reasons.iter().any(|r| r.contains("commits")), "{e:?}");
        // a plain owner tag backs an action, not a decision: "Avery owns the
        // Kiosk App work" is who is responsible, not a choice the group made
        let mut d = item("Avery owns the Kiosk App work", &["s1"], "");
        d.keyframe_ids = vec!["kf_000040".into()];
        let e = check_with(Section::Decisions, &d, &c, &all).unwrap_err();
        assert!(e.reasons.iter().any(|r| r.contains("commits")), "{e:?}");
        // personal activities are not tasks
        let coffee = DraftItem {
            owner: "Avery".into(),
            task: "Get coffee before the next session".into(),
            segment_ids: vec!["s2".into()],
            ..Default::default()
        };
        let e = check_with(Section::ActionItems, &coffee, &c, &all).unwrap_err();
        assert!(
            e.fatal && e.reasons[0].contains("personal activity"),
            "{e:?}"
        );
        // citing the committing line passes, with the prefix removed
        let d = item("The team decided to skip the importer", &["s2"], "");
        let ok = check_with(Section::Decisions, &d, &c, &all).unwrap();
        assert_eq!(ok.text, "Skip the importer");
    }

    /// Avery owns the Kiosk App (keyframe kf_000040); s1 is descriptive, s2
    /// opens with a personal future before a descriptive sentence.
    fn guard_corpus() -> Corpus {
        use crate::board::build;
        let mut b = build::board("b", 100.0);
        b.nodes = vec![build::node("n1", "Kiosk App", 0.0, 100.0, None)];
        let mut o = build::owner(
            "avery-quinn",
            "Avery Quinn",
            build::node_target(&b, "n1"),
            40.0,
            100.0,
            None,
        );
        o.opened_at_keyframe = "kf_000040".into();
        b.owner_assignments = vec![o];
        let text = "I'll be right back. The importer reads the ledger nightly.";
        let lines = vec![
            line(
                "s1",
                10.0,
                "The importer reads the ledger nightly.",
                "The importer reads the ledger nightly.",
            ),
            line("s2", 20.0, text, text),
        ];
        Corpus::new(
            &lines,
            &[b],
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Avery Quinn"]),
        )
    }

    const ALL: CheckOptions = CheckOptions {
        board_support: true,
        precision_guard: true,
        strip_prefix: true,
    };

    #[test]
    fn a_personal_future_does_not_lend_a_commitment_to_its_segment() {
        let c = guard_corpus();
        let d = item("Keep the importer on REST", &["s2"], "");
        let e = check_with(Section::Decisions, &d, &c, &ALL).unwrap_err();
        assert!(e.reasons.iter().any(|r| r.contains("commits")), "{e:?}");
    }

    #[test]
    fn board_support_needs_the_target_not_the_owner_name() {
        let c = guard_corpus();
        // the review's construction: the decision name-drops the owner of the
        // cited keyframe and cites no transcript line
        let mut d = item("Avery will draft the importer docs", &[], "");
        d.keyframe_ids = vec!["kf_000040".into()];
        let e = check_with(Section::Decisions, &d, &c, &ALL).unwrap_err();
        assert!(
            e.reasons.iter().any(|r| r.contains("transcript segment")),
            "{e:?}"
        );
        // with a descriptive line cited, the guard is not bypassed either
        d.segment_ids = vec!["s1".into()];
        let e = check_with(Section::Decisions, &d, &c, &ALL).unwrap_err();
        assert!(e.reasons.iter().any(|r| r.contains("commits")), "{e:?}");
        // the owner named as an action's owner does not back the task
        let a = DraftItem {
            owner: "Avery".into(),
            task: "Draft the importer docs".into(),
            keyframe_ids: vec!["kf_000040".into()],
            ..Default::default()
        };
        let e = check_with(Section::ActionItems, &a, &c, &ALL).unwrap_err();
        assert!(
            e.reasons.iter().any(|r| r.contains("transcript segment")),
            "{e:?}"
        );
        // one generic target word is not the target
        let mut d = item("Rewrite the app login", &["s1"], "");
        d.keyframe_ids = vec!["kf_000040".into()];
        let e = check_with(Section::Decisions, &d, &c, &ALL).unwrap_err();
        assert!(e.reasons.iter().any(|r| r.contains("commits")), "{e:?}");
        // naming the target and the ownership backs an action, but a plain
        // owner tag is not a decision
        let a = DraftItem {
            owner: "Avery".into(),
            task: "Own the kiosk rollout".into(),
            keyframe_ids: vec!["kf_000040".into()],
            ..Default::default()
        };
        assert!(check_with(Section::ActionItems, &a, &c, &ALL).is_ok());
        let mut d = item("Avery owns the kiosk rollout", &[], "");
        d.keyframe_ids = vec!["kf_000040".into()];
        let e = check_with(Section::Decisions, &d, &c, &ALL).unwrap_err();
        assert!(
            e.reasons.iter().any(|r| r.contains("transcript segment")),
            "{e:?}"
        );
    }

    #[test]
    fn only_a_moved_owner_tag_backs_a_decision() {
        use crate::board::build;
        let mut b = build::board("b", 100.0);
        b.nodes = vec![
            build::node("n1", "Ledger Store", 0.0, 100.0, None),
            build::node("n2", "Kiosk App", 0.0, 100.0, None),
        ];
        let t1 = build::node_target(&b, "n1");
        let t2 = build::node_target(&b, "n2");
        let mut first = build::owner("mira-okafor", "Mira Okafor", t1.clone(), 10.0, 50.0, None);
        first.opened_at_keyframe = "kf_000010".into();
        let mut moved = build::owner("mira-okafor", "Mira Okafor", t2, 50.0, 100.0, Some(t1));
        moved.opened_at_keyframe = "kf_000050".into();
        b.owner_assignments = vec![first, moved];
        b.events = vec![
            build::event(
                "ev-1",
                EventKind::OwnerAssigned,
                10.0,
                "kf_000010",
                "mira-okafor",
                "Mira on Ledger Store",
            ),
            build::event(
                "ev-2",
                EventKind::OwnerMoved,
                50.0,
                "kf_000050",
                "mira-okafor",
                "Mira to Kiosk App",
            ),
        ];
        let (n1, n2) = (build::node_target(&b, "n1"), build::node_target(&b, "n2"));
        b.events[0].owner_target = Some(n1.clone());
        b.events[1].owner_target = Some(n2);
        b.events[1].owner_from = Some(n1);
        let text = "The importer reads the ledger nightly.";
        let c = Corpus::new(
            &[line("s1", 5.0, text, text)],
            &[b],
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Mira Okafor"]),
        );
        // the move backs a decision by its event or by the keyframe that opened it
        for (ev, kf) in [(vec!["ev-2"], vec![]), (vec![], vec!["kf_000050"])] {
            let mut d = item("Mira moves to the Kiosk App", &[], "");
            d.event_ids = ev.into_iter().map(String::from).collect();
            d.keyframe_ids = kf.into_iter().map(String::from).collect();
            assert!(
                check_with(Section::Decisions, &d, &c, &ALL).is_ok(),
                "{d:?}"
            );
        }
        // the plain assignment is not a decision, and, taken off at 00:50, not
        // a current task either
        let mut d = item("Mira owns the Ledger Store", &[], "");
        d.event_ids = vec!["ev-1".into()];
        assert!(check_with(Section::Decisions, &d, &c, &ALL).is_err());
        let action = |task: &str, ev: &str| DraftItem {
            owner: "Mira".into(),
            task: task.into(),
            event_ids: vec![ev.into()],
            ..Default::default()
        };
        let a = action("Own the Ledger Store", "ev-1");
        assert!(check_with(Section::ActionItems, &a, &c, &ALL).is_err());
        // the tag still on the board backs an action for its owner
        let a = action("Own the Kiosk App", "ev-2");
        assert!(check_with(Section::ActionItems, &a, &c, &ALL).is_ok());
    }

    #[test]
    fn a_plain_owner_event_does_not_inherit_a_nearby_move() {
        use crate::board::build;
        // one person tagged on the Ledger Store, then (within 30 s) moved onto
        // the Kiosk App from somewhere else: each event belongs to its target
        let mut b = build::board("b", 100.0);
        b.nodes = vec![
            build::node("n1", "Ledger Store", 0.0, 100.0, None),
            build::node("n2", "Kiosk App", 0.0, 100.0, None),
            build::node("n3", "Badge Printer", 0.0, 100.0, None),
        ];
        let plain = build::owner(
            "mira-okafor",
            "Mira Okafor",
            build::node_target(&b, "n1"),
            40.0,
            100.0,
            None,
        );
        let moved = build::owner(
            "mira-okafor",
            "Mira Okafor",
            build::node_target(&b, "n2"),
            50.0,
            100.0,
            Some(build::node_target(&b, "n3")),
        );
        b.owner_assignments = vec![plain, moved];
        b.events = vec![
            build::event(
                "ev-a",
                EventKind::OwnerAssigned,
                40.0,
                "kf_000040",
                "mira-okafor",
                "Mira on Ledger Store",
            ),
            build::event(
                "ev-m",
                EventKind::OwnerMoved,
                50.0,
                "kf_000050",
                "mira-okafor",
                "Mira to Kiosk App",
            ),
        ];
        b.events[0].owner_target = Some(build::node_target(&b, "n1"));
        b.events[1].owner_target = Some(build::node_target(&b, "n2"));
        b.events[1].owner_from = Some(build::node_target(&b, "n3"));
        let text = "The importer reads the ledger nightly.";
        let c = Corpus::new(
            &[line("s1", 5.0, text, text)],
            &[b],
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Mira Okafor"]),
        );
        let decision = |ev: &str| {
            let mut d = item("Mira moves to the Kiosk App", &[], "");
            d.event_ids = vec![ev.into()];
            check_with(Section::Decisions, &d, &c, &ALL)
        };
        assert!(decision("ev-m").is_ok());
        assert!(
            decision("ev-a").is_err(),
            "the plain event is about the Ledger Store"
        );
    }

    #[test]
    fn a_move_to_ledger_store_does_not_back_the_plain_tag_on_ledger() {
        use crate::board::{build, BoardEvent};
        // Mira's plain tag on "Ledger" stays to the end; 10 s after it she is
        // moved onto "Ledger Store" from the Kiosk App, and that tag is taken
        // off at 01:10. The move event is about Ledger Store only: it cannot
        // back a current task on Ledger.
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
        b.owner_assignments = vec![
            build::owner("mira-okafor", "Mira Okafor", n1.clone(), 40.0, 100.0, None),
            build::owner(
                "mira-okafor",
                "Mira Okafor",
                n2.clone(),
                50.0,
                70.0,
                Some(n3.clone()),
            ),
        ];
        b.events = vec![
            BoardEvent {
                owner_target: Some(n1),
                ..build::event(
                    "ev-a",
                    EventKind::OwnerAssigned,
                    40.0,
                    "kf_000040",
                    "mira-okafor",
                    "Mira Okafor -> Ledger",
                )
            },
            BoardEvent {
                owner_target: Some(n2),
                owner_from: Some(n3),
                ..build::event(
                    "ev-m",
                    EventKind::OwnerMoved,
                    50.0,
                    "kf_000050",
                    "mira-okafor",
                    "Mira Okafor -> Ledger Store",
                )
            },
        ];
        let text = "The importer reads the ledger nightly.";
        let c = Corpus::new(
            &[line("s1", 5.0, text, text)],
            &[b],
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Mira Okafor"]),
        );
        let action = |ev: &str| DraftItem {
            owner: "Mira".into(),
            task: "Own the Ledger".into(),
            event_ids: vec![ev.into()],
            ..Default::default()
        };
        assert!(check_with(Section::ActionItems, &action("ev-a"), &c, &ALL).is_ok());
        assert!(
            check_with(Section::ActionItems, &action("ev-m"), &c, &ALL).is_err(),
            "the move event is about Ledger Store, whose tag was taken off"
        );
        // the move still backs the decision it is about
        let mut d = item("Mira moves to the Ledger Store", &[], "");
        d.event_ids = vec!["ev-m".into()];
        assert!(check_with(Section::Decisions, &d, &c, &ALL).is_ok());
    }

    #[test]
    fn one_shared_target_word_is_not_board_support() {
        use crate::board::build;
        let mut b = build::board("b", 100.0);
        b.nodes = vec![
            build::node("n1", "Badge Printer", 0.0, 100.0, None),
            build::node("n2", "Kiosk App", 0.0, 100.0, None),
        ];
        let mut o1 = build::owner(
            "mira-okafor",
            "Mira Okafor",
            build::node_target(&b, "n1"),
            40.0,
            100.0,
            // moved here, so it can back a decision (a plain tag cannot); the
            // cases below test which words refer to it
            Some(build::node_target(&b, "n2")),
        );
        o1.opened_at_keyframe = "kf_000040".into();
        let mut o2 = build::owner(
            "avery-quinn",
            "Avery Quinn",
            build::node_target(&b, "n2"),
            50.0,
            100.0,
            None,
        );
        o2.opened_at_keyframe = "kf_000050".into();
        b.owner_assignments = vec![o1, o2];
        let text = "The importer reads the ledger nightly.";
        let c = Corpus::new(
            &[line("s1", 10.0, text, text)],
            &[b],
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Mira Okafor", "Avery Quinn"]),
        );
        let reject = |section: Section, d: &DraftItem| {
            let e = check_with(section, d, &c, &ALL).unwrap_err();
            assert!(
                e.reasons
                    .iter()
                    .any(|r| r.contains("transcript segment") || r.contains("commits")),
                "{d:?}: {e:?}"
            );
        };
        // the review's constructions: one distinctive target word, no
        // reference to the ownership
        let mut d = item("Buy a new office printer", &[], "");
        d.keyframe_ids = vec!["kf_000040".into()];
        reject(Section::Decisions, &d);
        d.segment_ids = vec!["s1".into()];
        reject(Section::Decisions, &d);
        let a = DraftItem {
            owner: "Avery".into(),
            task: "Rewrite the kiosk login flow".into(),
            keyframe_ids: vec!["kf_000050".into()],
            ..Default::default()
        };
        reject(Section::ActionItems, &a);
        // the owner named in the text, an ownership verb, or the full target
        let ok = |section: Section, d: &DraftItem| {
            assert!(check_with(section, d, &c, &ALL).is_ok(), "{d:?}");
        };
        let mut d = item("Mira keeps the printer queue", &[], "");
        d.keyframe_ids = vec!["kf_000040".into()];
        ok(Section::Decisions, &d);
        let mut d = item("Take over the printer drivers", &[], "");
        d.keyframe_ids = vec!["kf_000040".into()];
        ok(Section::Decisions, &d);
        let mut d = item("Retire the badge printer", &[], "");
        d.keyframe_ids = vec!["kf_000040".into()];
        ok(Section::Decisions, &d);
        let a = DraftItem {
            owner: "Avery".into(),
            task: "Own the Kiosk App".into(),
            keyframe_ids: vec!["kf_000050".into()],
            ..Default::default()
        };
        ok(Section::ActionItems, &a);
    }

    #[test]
    fn personal_errands_outside_the_old_list_are_not_tasks() {
        let c = guard_corpus();
        for task in ["Talk to mom after this", "Get the kids from school"] {
            let a = DraftItem {
                owner: "Avery".into(),
                task: task.into(),
                segment_ids: vec!["s1".into()],
                ..Default::default()
            };
            let e = check_with(Section::ActionItems, &a, &c, &ALL).unwrap_err();
            assert!(e.fatal, "{task}: {e:?}");
        }
        // "break" as a work verb is not a pause
        assert!(!is_personal_activity("Break the importer into two jobs"));
        assert!(is_personal_activity("Take a quick break"));
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
        a.task = "See you folks tomorrow".into();
        a.segment_ids = vec!["s3".into()];
        assert!(check(Section::ActionItems, &a, &c).unwrap_err().fatal);
        a.owner = "Everyone and Avery".into();
        a.task = "Publish each kiosk release in the tracker".into();
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
            &item("Which widgets suit the kiosk?", &["s1"], ""),
            &c,
        )
        .unwrap();
        let b = check(
            Section::OpenQuestions,
            &item("Which widgets suit the kiosk", &["s2"], ""),
            &c,
        )
        .unwrap();
        let mut s = assemble(vec![a, b]);
        assert_eq!(s.open_questions.len(), 1);
        assert_eq!(s.open_questions[0].evidence.segment_ids, vec!["s1", "s2"]);
        use crate::board::build;
        let mut board = build::board("b", 90.0);
        board.stickies = vec![
            build::sticky("st1", "What widgets suit the kiosk?", 5.0, 90.0),
            build::sticky("st2", "Does the badge flow need edits?", 40.0, 90.0),
            build::sticky("st3", "Print the lobby maps", 41.0, 90.0),
        ];
        board.events = vec![build::event(
            "ev_1",
            crate::board::EventKind::StickyAdded,
            40.0,
            "kf_2",
            "st2",
            "Does the badge flow need edits?",
        )];
        let added = merge_board_questions(&mut s.open_questions, &[board]);
        assert_eq!(added, 1);
        assert_eq!(s.open_questions[1].evidence.event_ids, vec!["ev_1"]);
        assert_eq!(s.open_questions[1].evidence.keyframe_ids, vec!["kf_2"]);
        assert_eq!(
            s.open_questions[0].source,
            QuestionSource::BoardAndTranscript
        );
        assert_eq!(s.open_questions[1].source, QuestionSource::Board);
        assert_eq!(s.open_questions[1].id, "q2");
    }

    fn said(id: &str, t: f64, who: &str, text: &str) -> NamedLine {
        let mut l = line(id, t, text, text);
        l.person_id = Some(who.into());
        l.speaker = who.into();
        l
    }

    fn talk(lines: &[NamedLine]) -> Corpus {
        Corpus::new(
            lines,
            &[],
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Avery Quinn", "Rohan Dasgupta"]),
        )
    }

    #[test]
    fn an_agreed_proposal_is_a_decision_and_cites_the_agreement() {
        let proposal = "Maybe we just cache the badge photos on the kiosk.";
        let d = item("Cache the badge photos on the kiosk", &["s1"], "");
        // agreed by another speaker: kept, and the agreement is cited
        let c = talk(&[
            said("s1", 0.0, "avery-quinn", proposal),
            said("s2", 5.0, "rohan-dasgupta", "Yeah, that makes sense to me."),
        ]);
        let ok = check_with(Section::Decisions, &d, &c, &ALL).unwrap();
        assert_eq!(ok.evidence.segment_ids, vec!["s1", "s2"]);
        assert_eq!((ok.t_start_s, ok.t_end_s), (0.0, 8.0));
        // only floated, agreed by the proposer alone, objected to, or agreed
        // and then taken back: rejected
        for replies in [
            vec![],
            vec![said("s2", 5.0, "avery-quinn", "Sounds good to me.")],
            vec![said(
                "s2",
                5.0,
                "rohan-dasgupta",
                "I don't think so, the kiosk has no disk space.",
            )],
            vec![
                said("s2", 5.0, "rohan-dasgupta", "Sounds good."),
                said(
                    "s3",
                    9.0,
                    "avery-quinn",
                    "Actually, scratch that, we keep loading them live.",
                ),
            ],
        ] {
            let mut lines = vec![said("s1", 0.0, "avery-quinn", proposal)];
            lines.extend(replies.clone());
            let e = check_with(Section::Decisions, &d, &talk(&lines), &ALL).unwrap_err();
            assert!(
                e.reasons.iter().any(|r| r.contains("commits")),
                "{replies:?}"
            );
        }
        // an agreed proposal does not vouch for a decision about something else
        let c = talk(&[
            said("s1", 0.0, "avery-quinn", proposal),
            said("s2", 5.0, "rohan-dasgupta", "Yeah, that makes sense to me."),
        ]);
        let other = item("Migrate the ledger service to the new cluster", &["s1"], "");
        assert!(check_with(Section::Decisions, &other, &c, &ALL).is_err());
        // a paraphrase sharing two topic words with the proposal is backed
        let para = item("Keep cached badge photos on each kiosk", &["s1"], "");
        assert!(check_with(Section::Decisions, &para, &c, &ALL).is_ok());
        // one shared topic word does not tie a decision to the proposal
        let one_word = item("Remove the ledger photos", &["s1"], "");
        assert!(check_with(Section::Decisions, &one_word, &c, &ALL).is_err());
        let mut mixed = said("s1", 0.0, "avery-quinn", proposal);
        mixed.text_raw = "Maybe we should remove the ledger.".into();
        let c = talk(&[mixed, said("s2", 5.0, "rohan-dasgupta", "Sounds good.")]);
        let invented = item("Cache the ledger", &["s1"], "");
        assert!(check_with(Section::Decisions, &invented, &c, &ALL).is_err());
        let c = talk(&[
            said(
                "s1",
                0.0,
                "avery-quinn",
                "Maybe we cache the badge photos. Maybe we move the ledger.",
            ),
            said("s2", 5.0, "rohan-dasgupta", "Sounds good."),
        ]);
        assert!(check_with(Section::Decisions, &invented, &c, &ALL).is_err());
        let c = talk(&[
            said(
                "s1",
                0.0,
                "avery-quinn",
                "Maybe we should not cache the badge photos on the kiosk.",
            ),
            said("s2", 5.0, "rohan-dasgupta", "Sounds good."),
        ]);
        assert!(check_with(Section::Decisions, &d, &c, &ALL).is_err());
        // Assent to a later offer in the same turn is not assent to this one.
        let c = talk(&[
            said("s1", 0.0, "avery-quinn", proposal),
            said(
                "s2",
                5.0,
                "avery-quinn",
                "Maybe we should move the ledger to the kiosk.",
            ),
            said("s3", 9.0, "rohan-dasgupta", "Sounds good."),
        ]);
        assert!(check_with(Section::Decisions, &d, &c, &ALL).is_err());
        let c = talk(&[
            said("s1", 0.0, "avery-quinn", proposal),
            said(
                "s2",
                5.0,
                "rohan-dasgupta",
                "Maybe we should move the ledger. Sounds good.",
            ),
        ]);
        assert!(check_with(Section::Decisions, &d, &c, &ALL).is_err());
        let c = talk(&[
            said("s1", 0.0, "avery-quinn", proposal),
            said("s2", 5.0, "rohan-dasgupta", "Not a good idea. Sounds good."),
        ]);
        assert!(check_with(Section::Decisions, &d, &c, &ALL).is_err());
        // an agreement elsewhere does not back a decision citing a line with no proposal
        let c = talk(&[
            said("s1", 0.0, "avery-quinn", "The kiosk demo runs on Friday."),
            said("s2", 5.0, "rohan-dasgupta", "Sounds good."),
        ]);
        assert!(check_with(Section::Decisions, &d, &c, &ALL).is_err());
    }

    #[test]
    fn a_retracted_commitment_is_not_a_decision() {
        let d = item("Ship the importer on Friday", &["s1"], "");
        let commit = "Let's ship the importer on Friday.";
        // kept when it stands
        let c = talk(&[
            said("s1", 0.0, "avery-quinn", commit),
            said("s2", 5.0, "rohan-dasgupta", "Great, I'll tell the vendor."),
        ]);
        assert!(check_with(Section::Decisions, &d, &c, &ALL).is_ok());
        // taken back in the same line, or by anyone right after
        for lines in [
            vec![said(
                "s1",
                0.0,
                "avery-quinn",
                "Let's ship the importer on Friday. Actually, never mind.",
            )],
            vec![said(
                "s1",
                0.0,
                "avery-quinn",
                "Let's ship the importer on Friday, actually scratch that.",
            )],
            vec![said(
                "s1",
                0.0,
                "avery-quinn",
                "Let's ship the importer on Friday, but let's not ship it.",
            )],
            vec![
                said("s1", 0.0, "avery-quinn", commit),
                said(
                    "s2",
                    5.0,
                    "rohan-dasgupta",
                    "Wait, no, legal has not signed off.",
                ),
            ],
            vec![
                said("s1", 0.0, "avery-quinn", commit),
                said("s2", 5.0, "rohan-dasgupta", "No, let's not ship it."),
            ],
        ] {
            assert!(check_with(Section::Decisions, &d, &talk(&lines), &ALL).is_err());
        }
        let c = talk(&[said(
            "s1",
            0.0,
            "avery-quinn",
            "Let's not ship the importer on Friday.",
        )]);
        assert!(check_with(Section::Decisions, &d, &c, &ALL).is_err());
        // a negation in another clause of the line does not flip the claim
        let c2 = talk(&[said(
            "s1",
            0.0,
            "avery-quinn",
            "Maybe we ship the kiosk build today, and we'll not touch the ledger at all.",
        )]);
        let kiosk = item("Ship the kiosk build today", &["s1"], "");
        assert!(check_with(Section::Decisions, &kiosk, &c2, &ALL).is_ok());
        let no_ship = item("Do not ship the importer on Friday", &["s1"], "");
        assert!(check_with(Section::Decisions, &no_ship, &c, &ALL).is_ok());
        let other = item("Move the badge printer to the kiosk", &["s1"], "");
        assert!(check_with(Section::Decisions, &other, &c, &ALL).is_err());
    }

    #[test]
    fn a_time_adjunct_does_not_make_work_personal() {
        let c = guard_corpus();
        let task = |t: &str| DraftItem {
            owner: "Avery".into(),
            task: t.into(),
            segment_ids: vec!["s1".into()],
            ..Default::default()
        };
        for t in [
            "Review the kiosk logs after lunch and the dentist",
            "Sync with the vendor when back from the dentist",
        ] {
            assert!(
                check_with(Section::ActionItems, &task(t), &c, &ALL).is_ok(),
                "{t}"
            );
        }
        assert!(!is_personal_activity("After lunch, review the kiosk logs"));
        for t in [
            "Grab lunch after the demo",
            "Take a break after the standup",
            "Pick up the kids before the review",
            "After lunch, grab a coffee",
            "After the dentist",
        ] {
            assert!(is_personal_activity(t), "{t}");
        }
        let e = check_with(
            Section::ActionItems,
            &task("Grab lunch after the demo"),
            &c,
            &ALL,
        );
        assert!(e.unwrap_err().fatal);
    }

    /// Mira's owner tag on the Ledger Store (plain, taken off at 60 s) moved to
    /// the Kiosk App (current); Rohan's plain tag on the Badge Printer (current).
    fn moved_board_corpus(lines: &[NamedLine]) -> Corpus {
        use crate::board::build;
        let mut b = build::board("b", 100.0);
        b.nodes = vec![
            build::node("n1", "Ledger Store", 0.0, 100.0, None),
            build::node("n2", "Kiosk App", 0.0, 100.0, None),
            build::node("n3", "Badge Printer", 0.0, 100.0, None),
        ];
        let (t1, t2, t3) = (
            build::node_target(&b, "n1"),
            build::node_target(&b, "n2"),
            build::node_target(&b, "n3"),
        );
        let mut plain = build::owner("mira-okafor", "Mira Okafor", t1.clone(), 20.0, 60.0, None);
        plain.opened_at_keyframe = "kf_000020".into();
        let mut moved = build::owner("mira-okafor", "Mira Okafor", t2, 60.0, 100.0, Some(t1));
        moved.opened_at_keyframe = "kf_000060".into();
        let mut rohan = build::owner("rohan-dasgupta", "Rohan Dasgupta", t3, 30.0, 100.0, None);
        rohan.opened_at_keyframe = "kf_000030".into();
        b.owner_assignments = vec![plain, moved, rohan];
        Corpus::new(
            lines,
            &[b],
            &KeyframeTimes::default(),
            AliasTable::from_names(&["Mira Okafor", "Rohan Dasgupta"]),
        )
    }

    #[test]
    fn plain_owner_tag_decisions_are_refiled_as_tasks() {
        let lines = vec![said(
            "s1",
            25.0,
            "mira-okafor",
            "Let's put Rohan on this one.",
        )];
        let c = moved_board_corpus(&lines);
        let dec = |text: &str, kf: &str| DraftItem {
            text: text.into(),
            segment_ids: vec!["s1".into()],
            keyframe_ids: vec![kf.into()],
            ..Default::default()
        };
        let mut d = Draft {
            decisions: vec![
                // a plain, current tag: a task for its owner
                dec("Rohan Dasgupta owns the Badge Printer", "kf_000030"),
                // a move: stays a decision
                dec(
                    "Mira Okafor moves from Ledger Store to Kiosk App",
                    "kf_000060",
                ),
                // a plain tag taken off: not a current task, left for the guard
                dec("Mira Okafor owns the Ledger Store", "kf_000020"),
                // no owner tag behind it
                dec("Keep the ledger API on REST", "kf_000030"),
            ],
            ..Draft::default()
        };
        assert_eq!(refile_owner_decisions(&mut d, &c), 1);
        let texts: Vec<&str> = d.decisions.iter().map(|x| x.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "Mira Okafor moves from Ledger Store to Kiosk App",
                "Mira Okafor owns the Ledger Store",
                "Keep the ledger API on REST",
            ]
        );
        assert_eq!(d.action_items.len(), 1);
        let a = &d.action_items[0];
        assert_eq!(
            (a.owner.as_str(), a.task.as_str()),
            ("Rohan Dasgupta", "Own Badge Printer")
        );
        assert_eq!(a.keyframe_ids, vec!["kf_000030"]);
        let ok = check_with(Section::ActionItems, a, &c, &ALL).unwrap();
        assert_eq!(ok.owners[0].person_id.as_deref(), Some("rohan-dasgupta"));
        // the owner already has an action naming the target: no duplicate
        let mut d = Draft {
            decisions: vec![dec("Rohan Dasgupta owns the Badge Printer", "kf_000030")],
            action_items: vec![DraftItem {
                owner: "Rohan".into(),
                task: "Fix the badge printer driver".into(),
                segment_ids: vec!["s1".into()],
                ..Default::default()
            }],
            ..Draft::default()
        };
        assert_eq!(refile_owner_decisions(&mut d, &c), 1);
        assert!(d.decisions.is_empty());
        assert_eq!(d.action_items.len(), 1);
    }

    #[test]
    fn ownership_of_a_target_the_owner_moved_off_is_superseded() {
        let lines = vec![said(
            "s1",
            25.0,
            "mira-okafor",
            "I'll work on the ledger store.",
        )];
        let c = moved_board_corpus(&lines);
        let act = |owner: &str, task: &str| DraftItem {
            owner: owner.into(),
            task: task.into(),
            segment_ids: vec!["s1".into()],
            ..Default::default()
        };
        for task in ["Work on the Ledger Store", "Own Ledger Store integration"] {
            let e = check_with(Section::ActionItems, &act("Mira", task), &c, &ALL).unwrap_err();
            assert!(
                e.fatal && e.reasons[0].contains("taken off"),
                "{task}: {e:?}"
            );
        }
        // her current target, a task about the old one that is not its
        // ownership, and another person's ownership are kept
        for (owner, task) in [
            ("Mira", "Own the Kiosk App"),
            ("Rohan", "Work on the Ledger Store"),
            ("Rohan", "Own the Badge Printer"),
        ] {
            assert!(
                check_with(Section::ActionItems, &act(owner, task), &c, &ALL).is_ok(),
                "{owner}: {task}"
            );
        }
        assert_eq!(
            ownership_words("Take ownership of the kiosk app"),
            Some(
                ["kiosk".to_string(), "app".to_string()]
                    .into_iter()
                    .collect()
            )
        );
        assert_eq!(ownership_words("Fix the kiosk app"), None);
    }

    #[test]
    fn tasks_on_a_target_the_owner_later_moved_off_are_superseded() {
        // Mira's tag moves from the Ledger Store to the Kiosk App at 60 s
        let lines = vec![
            said(
                "s1",
                25.0,
                "avery-quinn",
                "Mira, could you migrate the ledger store records?",
            ),
            said(
                "s2",
                80.0,
                "avery-quinn",
                "Mira, please send the ledger store notes to Rohan.",
            ),
        ];
        let c = moved_board_corpus(&lines);
        let act = |owner: &str, task: &str, seg: &str| DraftItem {
            owner: owner.into(),
            task: task.into(),
            segment_ids: vec![seg.into()],
            ..Default::default()
        };
        // said before the move, about the old target: superseded
        let e = check_with(
            Section::ActionItems,
            &act("Mira", "Migrate the ledger store records", "s1"),
            &c,
            &ALL,
        )
        .unwrap_err();
        assert!(
            e.fatal && e.reasons[0].contains("moved from Ledger Store to Kiosk App"),
            "{e:?}"
        );
        // said after the move, naming the new target too, or another person's
        for (owner, task, seg) in [
            ("Mira", "Send the ledger store notes to Rohan", "s2"),
            (
                "Mira",
                "Migrate the ledger store records into the kiosk app",
                "s1",
            ),
            ("Mira", "Map ledger store entries to app widgets", "s1"),
            ("Rohan", "Migrate the ledger store records", "s1"),
        ] {
            assert!(
                check_with(Section::ActionItems, &act(owner, task, seg), &c, &ALL).is_ok(),
                "{owner}: {task}"
            );
        }
    }

    #[test]
    fn restated_items_merge_only_on_shared_evidence() {
        let c = talk(&[
            said(
                "s1",
                0.0,
                "avery-quinn",
                "Let's skip the ledger importer for now.",
            ),
            said(
                "s2",
                5.0,
                "avery-quinn",
                "Let's skip the ledger importer tests for the kiosk.",
            ),
            said("s3", 9.0, "avery-quinn", "I'll own the kiosk app."),
        ]);
        let get = |s: Section, text: &str, ids: &[&str]| {
            let mut d = item(text, ids, "");
            if s == Section::ActionItems {
                d.owner = "Avery".into();
                d.task = text.into();
            }
            check_with(s, &d, &c, &CheckOptions::default()).unwrap()
        };
        // one line, stated twice, once with an added reason
        let s = assemble(vec![
            get(
                Section::Decisions,
                "Skip the ledger importer for now",
                &["s1"],
            ),
            get(
                Section::Decisions,
                "Skip the ledger importer for now to focus on the kiosk",
                &["s1"],
            ),
        ]);
        assert_eq!(s.decisions.len(), 1);
        // alike, but from different lines: two decisions
        let s = assemble(vec![
            get(Section::Decisions, "Skip the ledger importer", &["s1"]),
            get(
                Section::Decisions,
                "Skip the ledger importer tests for the kiosk",
                &["s2"],
            ),
        ]);
        assert_eq!(s.decisions.len(), 2);
        // Negation is a stopword for similarity, but reverses the claim.
        let s = assemble(vec![
            get(Section::Decisions, "Ship the badge printer", &["s1"]),
            get(Section::Decisions, "Do not ship the badge printer", &["s1"]),
        ]);
        assert_eq!(s.decisions.len(), 2);
        // ownership in other words, same line
        let s = assemble(vec![
            get(Section::ActionItems, "Own the Kiosk App", &["s3"]),
            get(Section::ActionItems, "Work on the Kiosk App", &["s3"]),
        ]);
        assert_eq!(s.action_items.len(), 1);
        // one word of framing on each side, same line: merged; other lines
        // or two different targets: kept apart
        let s = assemble(vec![
            get(Section::ActionItems, "Own the kiosk app link", &["s3"]),
            get(
                Section::ActionItems,
                "Work on the kiosk app integration",
                &["s3"],
            ),
        ]);
        assert_eq!(s.action_items.len(), 1);
        let s = assemble(vec![
            get(Section::ActionItems, "Own the kiosk app link", &["s3"]),
            get(
                Section::ActionItems,
                "Work on the kiosk app integration",
                &["s1"],
            ),
        ]);
        assert_eq!(s.action_items.len(), 2);
        let s = assemble(vec![
            get(Section::ActionItems, "Own the kiosk app", &["s3"]),
            get(Section::ActionItems, "Own the ledger store", &["s3"]),
        ]);
        assert_eq!(s.action_items.len(), 2);
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
