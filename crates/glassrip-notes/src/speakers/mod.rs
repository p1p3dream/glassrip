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

pub use glassrip_audio::types::{
    Evidence, EvidenceKind, Person, SpeakerItem, SpeakerStatus, SpeakersArtifact, SpeakersParams,
};
pub use stage::{NameSpeakersParams, NameSpeakersStage};

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
    Label(SpeakerItem),
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
    pub labels: Vec<SpeakerItem>,
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

    /// Reads the audio crate's whole-file `glassrip.speakers` envelope (labels and
    /// people; it carries no per-segment decisions).
    pub fn from_artifact(a: &SpeakersArtifact) -> Self {
        let mut d = Self {
            people: a.people.clone(),
            labels: a.envelope.items.clone(),
            ..Self::default()
        };
        d.labels.sort_by(|x, y| x.label.cmp(&y.label));
        if let Some(n) = &a.notes {
            d.summary = Some(SpeakersSummary {
                method: a.envelope.params.method.clone(),
                frames_decoded: 0,
                frames_failed: 0,
                frames_with_tiles: 0,
                frames_with_highlight: 0,
                relabeled_segments: 0,
                presenter: None,
                gap_fill: GapFillCheck::default(),
                notes: Some(n.clone()),
            });
        }
        d
    }

    /// Writes the labels and people as the audio crate's whole-file envelope.
    pub fn to_artifact(&self, run_id: &str) -> SpeakersArtifact {
        use glassrip_audio::types::{Envelope, Producer, SCHEMA_VERSION, SPEAKERS_SCHEMA};
        let summary = self.summary.as_ref();
        SpeakersArtifact {
            envelope: Envelope {
                schema: SPEAKERS_SCHEMA.into(),
                schema_version: SCHEMA_VERSION.into(),
                run_id: run_id.into(),
                producer: Producer::current(),
                inputs: vec![],
                params: SpeakersParams {
                    method: summary.map(|s| s.method.clone()).unwrap_or_default(),
                    num_speakers_requested: None,
                    num_clusters_raw: self.labels.len(),
                    num_speakers_found: self
                        .labels
                        .iter()
                        .filter(|l| l.status == SpeakerStatus::Mapped)
                        .count(),
                },
                items: self.labels.clone(),
            },
            people: self.people.clone(),
            notes: summary.and_then(|s| s.notes.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_the_audio_envelope() {
        let doc = SpeakersDoc {
            people: vec![Person {
                person_id: "avery-quinn".into(),
                display_name: "Avery Quinn".into(),
                aliases: vec!["Avery".into()],
            }],
            labels: vec![SpeakerItem {
                label: "SPEAKER_00".into(),
                status: SpeakerStatus::Mapped,
                person_id: Some("avery-quinn".into()),
                confidence: 0.8,
                evidence: vec![Evidence {
                    t_s: 3.0,
                    kind: EvidenceKind::AbsentTile,
                    text: "no tile lit".into(),
                    person_id: Some("avery-quinn".into()),
                    weight: Some(0.4),
                    segment_id: Some("seg_00001".into()),
                }],
                talk_time_s: 12.0,
                talk_time_gap_fill_s: 1.0,
                words_diarizer: 30,
                words_gap_fill: 3,
                words_unassigned: 0,
                votes: [("avery-quinn".to_string(), 0.4)].into_iter().collect(),
                evidence_total: 1,
            }],
            ..SpeakersDoc::default()
        };
        let art = doc.to_artifact("r1");
        let v = serde_json::to_value(&art).unwrap();
        assert_eq!(v["schema"], "glassrip.speakers");
        assert_eq!(v["people"][0]["display_name"], "Avery Quinn");
        assert_eq!(v["items"][0]["evidence"][0]["kind"], "absent_tile");
        let back: SpeakersArtifact = serde_json::from_value(v).unwrap();
        let again = SpeakersDoc::from_artifact(&back);
        assert_eq!(again.labels, doc.labels);
        assert_eq!(again.people, doc.people);
        // a label record in the JSONL artifact is the audio SpeakerItem plus a kind tag
        let rec = serde_json::to_value(SpeakersRecord::Label(doc.labels[0].clone())).unwrap();
        assert_eq!(rec["kind"], "label");
        let item: SpeakerItem = serde_json::from_value(rec).unwrap();
        assert_eq!(item, doc.labels[0]);
    }
}
