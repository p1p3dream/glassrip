//! `notes`: citation-validated meeting notes from a local text model.
//!
//! The text model (default `qwen3.6:27b` on Ollama) runs in GPU phase C: the
//! vision model is unloaded first (`keep_alive: 0`), the text model is loaded and
//! must sit fully in VRAM. The transcript is cut into windows that fit the
//! context; each window yields candidate items (map), and one more call merges
//! them (reduce). The model only drafts: [`validate`] checks that every cited id
//! exists and that quotes appear in the cited segments, applies the action-item
//! and greeting filters, merges board questions, and computes all times. Items
//! that fail get one repair call; items still failing are dropped and counted.

pub mod candidates;
pub mod llm;
pub mod prompt;
pub mod stage;
pub mod validate;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::named::Paragraph;
use crate::people::Person;

pub use stage::{NotesParams, NotesStage};

/// Ids an item cites.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Evidence {
    /// Transcript segment ids.
    pub segment_ids: Vec<String>,
    /// Board event ids.
    pub event_ids: Vec<String>,
    /// Keyframe ids.
    pub keyframe_ids: Vec<String>,
}

impl Evidence {
    /// True when nothing is cited.
    pub fn is_empty(&self) -> bool {
        self.segment_ids.is_empty() && self.event_ids.is_empty() && self.keyframe_ids.is_empty()
    }

    /// Adds another item's ids (deduplicated, order kept).
    pub fn merge(&mut self, other: &Evidence) {
        for (dst, src) in [
            (&mut self.segment_ids, &other.segment_ids),
            (&mut self.event_ids, &other.event_ids),
            (&mut self.keyframe_ids, &other.keyframe_ids),
        ] {
            for id in src {
                if !dst.contains(id) {
                    dst.push(id.clone());
                }
            }
        }
    }
}

/// Which transcript text a quote matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QuoteMatch {
    /// Verbatim ASR text (authoritative).
    Raw,
    /// Only the vocabulary-corrected text.
    Corrected,
}

/// A verified quote.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Quote {
    /// Quote text as it appears in the transcript.
    pub text: String,
    /// Segment containing (the start of) the quote.
    pub segment_id: String,
    /// Which text matched.
    pub matched: QuoteMatch,
}

/// A decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Decision {
    /// Item id.
    pub id: String,
    /// Statement.
    pub text: String,
    /// Start of the supporting evidence, seconds.
    pub t_start_s: f64,
    /// End of the supporting evidence, seconds.
    pub t_end_s: f64,
    /// Citations.
    pub evidence: Evidence,
    /// Supporting quote.
    pub quote: Option<Quote>,
}

/// An action item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ActionItem {
    /// Item id.
    pub id: String,
    /// Owner (None: everyone).
    pub person_id: Option<String>,
    /// Owner display name (`Everyone` for the whole group).
    pub owner: String,
    /// Task, starting with a verb.
    pub task: String,
    /// Time of the first supporting evidence, seconds.
    pub t_s: f64,
    /// End of the supporting evidence, seconds.
    pub t_end_s: f64,
    /// Citations.
    pub evidence: Evidence,
    /// Supporting quote.
    pub quote: Option<Quote>,
}

/// Where an open question came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QuestionSource {
    /// Asked in the meeting.
    Transcript,
    /// A question sticky on the board.
    Board,
    /// Both.
    BoardAndTranscript,
}

/// An open question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OpenQuestion {
    /// Item id.
    pub id: String,
    /// Question.
    pub text: String,
    /// Source.
    pub source: QuestionSource,
    /// Time first raised, seconds.
    pub t_s: Option<f64>,
    /// Citations.
    pub evidence: Evidence,
    /// Supporting quote.
    pub quote: Option<Quote>,
}

/// A timeline entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TimelineEntry {
    /// Item id.
    pub id: String,
    /// Start, seconds.
    pub t_start_s: f64,
    /// End, seconds.
    pub t_end_s: f64,
    /// What happened.
    pub text: String,
    /// Citations.
    pub evidence: Evidence,
}

/// A summary point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SummaryPoint {
    /// Item id.
    pub id: String,
    /// Statement.
    pub text: String,
    /// Start of the supporting evidence, seconds.
    pub t_start_s: f64,
    /// End of the supporting evidence, seconds.
    pub t_end_s: f64,
    /// Citations.
    pub evidence: Evidence,
}

/// A caveat shown with the notes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Caveat {
    /// Kind (`degraded`, `speakers`, `gap_fill`, `misheard`, `dropped`, `no_board`, `no_audio`).
    pub kind: String,
    /// Text.
    pub text: String,
}

/// An item removed by validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DroppedItem {
    /// Section.
    pub section: String,
    /// Item text.
    pub text: String,
    /// Why.
    pub reasons: Vec<String>,
}

/// Notes status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NotesStatus {
    /// Passed the minimum-output checks.
    Ok,
    /// Minimum-output alarm (see the caveats).
    Degraded,
}

/// One model call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CallRecord {
    /// Purpose (`map 1/2`, `reduce`, `repair`).
    pub purpose: String,
    /// Client wall time, seconds.
    pub wall_s: f64,
    /// Prompt tokens.
    pub prompt_tokens: Option<u64>,
    /// Generated tokens.
    pub eval_tokens: Option<u64>,
    /// Stop reason.
    pub done_reason: Option<String>,
    /// Parse failure of the reply, if any.
    pub parse_error: Option<String>,
}

/// Validation and model statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NotesReport {
    /// Status.
    pub status: NotesStatus,
    /// Text model tag.
    pub model: String,
    /// Model digest.
    pub model_digest: Option<String>,
    /// Transcript windows.
    pub windows: usize,
    /// Model calls.
    pub calls: Vec<CallRecord>,
    /// Items drafted by the model (after reduce) plus board questions.
    pub items_drafted: usize,
    /// Items kept.
    pub items_kept: usize,
    /// Items that failed validation the first time.
    pub items_failed_first_pass: usize,
    /// Items fixed by the repair call.
    pub items_repaired: usize,
    /// Items dropped.
    pub dropped: Vec<DroppedItem>,
    /// Dropped share of drafted items.
    pub drop_rate: f64,
    /// Placement of the text model after loading (`/api/ps`).
    pub placement: Option<llm::LoadedModel>,
    /// Vision model unloaded before loading the text model.
    pub unloaded_vision_model: Option<String>,
    /// Stage wall time, seconds.
    pub wall_s: f64,
}

/// How one diarization label was resolved (for caveats and rendering).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SpeakerLine {
    /// Diarization label.
    pub label: String,
    /// Participant name or `unresolved` / `noise`.
    pub name: String,
    /// Status (`mapped`, `noise`, `unresolved`).
    pub status: String,
    /// Confidence in [0, 1].
    pub confidence: f32,
    /// Talk time, seconds.
    pub talk_time_s: f64,
}

/// The `glassrip.meeting_notes` item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MeetingNotes {
    /// Title (board title or configured).
    pub title: Option<String>,
    /// Meeting length, seconds.
    pub duration_s: f64,
    /// Participants.
    pub people: Vec<Person>,
    /// Presenter, when known.
    pub presenter: Option<String>,
    /// Summary points.
    pub summary: Vec<SummaryPoint>,
    /// Decisions.
    pub decisions: Vec<Decision>,
    /// Action items.
    pub action_items: Vec<ActionItem>,
    /// Open questions.
    pub open_questions: Vec<OpenQuestion>,
    /// Timeline.
    pub timeline: Vec<TimelineEntry>,
    /// Caveats.
    pub caveats: Vec<Caveat>,
    /// Speaker resolution per label.
    pub speakers: Vec<SpeakerLine>,
    /// Full transcript with mapped names.
    pub transcript: Vec<Paragraph>,
    /// Validation report.
    pub report: NotesReport,
}
