//! `name_speakers`: map diarization labels to participants by an evidence vote.
//!
//! Evidence sources (design 6.13):
//!
//! 1. **Visual active-speaker cues.** Frames are decoded on demand from the source
//!    video at each transcript segment's midpoint and 0.5 s after its start (and at
//!    the midpoint of every long gap-fill run). Conferencing tiles are found by
//!    reading participant names on screen ([`cues::TileReader`]); a tile whose
//!    border carries the bright speaking ring is highlighted ([`cues::ring_score`]).
//!    When every visible tile is dark and exactly one participant has no visible
//!    tile (typically the local user, whose self view is hidden while a screen is
//!    shared), that participant is the likely speaker (a weaker cue).
//! 2. **Direct address** in the transcript through the [`crate::people::AliasTable`]:
//!    the person addressed is not the speaker, and the next turn by another label
//!    is likely theirs.
//! 3. **Role cues:** the presenter named in the "(Presenting" banner is voted for
//!    the label with the most talk time.
//!
//! Label votes pick one participant per label. Each segment is then re-checked
//! against its own cues and relabeled when a cue contradicts the diarizer strongly
//! enough, and runs of gap-filled words are re-checked the same way.

pub mod address;
pub mod cues;
pub mod frames;
#[cfg(feature = "ocr")]
pub mod ocr;
pub mod stage;
pub mod vote;

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use crate::people::Person;
pub use stage::{NameSpeakersParams, NameSpeakersStage};

/// Kind of evidence behind a vote.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum CueKind {
    /// The participant's tile carried the speaking ring.
    ActiveSpeakerHighlight,
    /// No visible tile was lit and this participant had no visible tile.
    AbsentTile,
    /// The segment answers a direct address to this participant.
    AddressResponse,
    /// The segment addresses this participant by name, so it is not theirs (negative).
    AddressedNotSpeaker,
    /// Presenter banner plus the largest talk time.
    RoleCue,
}

/// One on-screen participant tile in a frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TileSeen {
    /// Matched participant.
    pub person_id: String,
    /// Text as read.
    pub text: String,
    /// Name box `[x0, y0, x1, y1]` in frame pixels.
    pub name_bbox: [u32; 4],
    /// Fraction of columns under the name with a speaking ring.
    pub ring_score: f32,
    /// Ring score passed the threshold.
    pub highlighted: bool,
}

/// What one decoded frame showed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FrameObservation {
    /// Frame time, seconds.
    pub t_s: f64,
    /// Participant tiles found.
    pub tiles: Vec<TileSeen>,
    /// Participant named in a presenting banner.
    #[serde(default)]
    pub presenter: Option<String>,
    /// Decode or OCR failure for this frame.
    #[serde(default)]
    pub error: Option<String>,
}

impl FrameObservation {
    /// Person ids whose tile is highlighted.
    pub fn highlighted(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self
            .tiles
            .iter()
            .filter(|t| t.highlighted)
            .map(|t| t.person_id.as_str())
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Person ids with a visible tile.
    pub fn visible(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tiles.iter().map(|t| t.person_id.as_str()).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

/// One piece of evidence recorded on a label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LabelEvidence {
    /// Time, seconds.
    pub t_s: f64,
    /// Kind.
    pub kind: CueKind,
    /// Participant the evidence is about.
    pub person_id: String,
    /// Signed vote weight.
    pub weight: f64,
    /// Segment the evidence came from.
    #[serde(default)]
    pub segment_id: Option<String>,
    /// Supporting text (transcript words or tile text).
    pub text: String,
}

/// Resolution status of a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LabelStatus {
    /// Mapped to a participant.
    Mapped,
    /// Not a participant (noise, system audio, a stray cluster of a few words).
    Noise,
    /// Not resolved.
    Unresolved,
}

/// One diarization label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LabelItem {
    /// Diarization label.
    pub label: String,
    /// Resolution status.
    pub status: LabelStatus,
    /// Mapped participant.
    pub person_id: Option<String>,
    /// Confidence of the mapping in [0, 1].
    pub confidence: f32,
    /// Summed vote per participant.
    pub votes: BTreeMap<String, f64>,
    /// Strongest evidence items (at most 40, by absolute weight).
    pub evidence: Vec<LabelEvidence>,
    /// Evidence items in total.
    pub evidence_total: usize,
    /// Speaking time of the label, seconds.
    pub talk_time_s: f64,
    /// Part of the talk time from gap-filled words, seconds.
    pub talk_time_gap_fill_s: f64,
}

/// Where a segment's speaker came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerSource {
    /// The label's mapping.
    LabelMap,
    /// A visual cue contradicted the diarizer and won.
    VisualRelabel,
    /// Direct address and absent-tile cues together outweighed the diarizer.
    EvidenceRelabel,
    /// No participant could be assigned.
    Unresolved,
}

/// A word range inside a segment with its own speaker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WordSpan {
    /// First word index (inclusive).
    pub word_start: usize,
    /// Last word index (exclusive).
    pub word_end: usize,
    /// Speaker of the range.
    pub person_id: Option<String>,
    /// Confidence in [0, 1].
    pub confidence: f32,
    /// Where the speaker came from.
    pub source: SpeakerSource,
    /// Why the range differs from the segment.
    pub reason: String,
}

/// Resolved speaker of one transcript segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SegmentSpeaker {
    /// Transcript segment id.
    pub segment_id: String,
    /// Start, seconds.
    pub start_s: f64,
    /// End, seconds.
    pub end_s: f64,
    /// Diarization label.
    pub label: String,
    /// Speaker of the segment.
    pub person_id: Option<String>,
    /// Confidence in [0, 1].
    pub confidence: f32,
    /// Where the speaker came from.
    pub source: SpeakerSource,
    /// Explanation when relabeled.
    #[serde(default)]
    pub reason: Option<String>,
    /// Per-candidate scores used for the decision.
    pub scores: BTreeMap<String, f64>,
    /// Frames observed for this segment.
    pub observations: Vec<FrameObservation>,
    /// Word ranges whose speaker differs from the segment's (gap-fill re-check).
    #[serde(default)]
    pub spans: Vec<WordSpan>,
}

/// Outcome of re-checking gap-filled words against visual cues.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GapFillCheck {
    /// Gap-filled words in the transcript.
    pub words_total: usize,
    /// Runs of consecutive gap-filled words.
    pub runs_total: usize,
    /// Runs with a usable visual cue.
    pub runs_with_cue: usize,
    /// Runs whose cue agrees with the final speaker.
    pub runs_agree: usize,
    /// Runs whose cue names someone else.
    pub runs_contradict: usize,
    /// Runs relabeled because of the contradiction.
    pub runs_relabeled: usize,
    /// Words in agreeing runs.
    pub words_agree: usize,
    /// Words in contradicting runs.
    pub words_contradict: usize,
    /// Words relabeled.
    pub words_relabeled: usize,
    /// Words in runs without a usable cue.
    pub words_no_cue: usize,
}

/// Run-level summary of speaker naming.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SpeakersSummary {
    /// How the mapping was produced.
    pub method: String,
    /// Frames decoded (distinct times).
    pub frames_decoded: usize,
    /// Frames that failed to decode or read.
    pub frames_failed: usize,
    /// Frames with at least one participant tile.
    pub frames_with_tiles: usize,
    /// Frames with a highlighted tile.
    pub frames_with_highlight: usize,
    /// Segments whose speaker differs from their label's mapping.
    pub relabeled_segments: usize,
    /// Presenter from the banner (majority over frames).
    pub presenter: Option<String>,
    /// Gap-fill re-check.
    pub gap_fill: GapFillCheck,
    /// Free-form notes, kept apart from the items.
    pub notes: Option<String>,
}

/// One item of the `glassrip.speakers` artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpeakersRecord {
    /// A participant (`person:<id>`).
    Person(Person),
    /// A diarization label (`label:<label>`).
    Label(LabelItem),
    /// A transcript segment (`segment:<segment_id>`).
    Segment(SegmentSpeaker),
    /// Run summary (`summary`).
    Summary(SpeakersSummary),
}

/// All `glassrip.speakers` records, gathered for consumers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeakersDoc {
    /// Participants.
    pub people: Vec<Person>,
    /// Labels.
    pub labels: Vec<LabelItem>,
    /// Segments by id.
    pub segments: BTreeMap<String, SegmentSpeaker>,
    /// Summary.
    pub summary: Option<SpeakersSummary>,
}

impl SpeakersDoc {
    /// Gathers records (any order).
    pub fn from_records<I: IntoIterator<Item = SpeakersRecord>>(records: I) -> Self {
        let mut d = Self::default();
        for r in records {
            match r {
                SpeakersRecord::Person(p) => d.people.push(p),
                SpeakersRecord::Label(l) => d.labels.push(l),
                SpeakersRecord::Segment(s) => {
                    d.segments.insert(s.segment_id.clone(), s);
                }
                SpeakersRecord::Summary(s) => d.summary = Some(s),
            }
        }
        d.labels.sort_by(|a, b| a.label.cmp(&b.label));
        d
    }

    /// Display name of a person id.
    pub fn display_name(&self, person_id: &str) -> Option<&str> {
        self.people
            .iter()
            .find(|p| p.person_id == person_id)
            .map(|p| p.display_name.as_str())
    }
}
