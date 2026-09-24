//! Artifact types for `glassrip.transcript` and `glassrip.speakers`.
//!
//! These follow the envelope contract of the meeting-mode design: `schema`,
//! `schema_version`, `run_id`, `producer`, `inputs`, `params`, `items`. They
//! live here until `glassrip-core` owns the shared envelope.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use crate::recluster::Source;

/// Schema name of the transcript artifact.
pub const TRANSCRIPT_SCHEMA: &str = "glassrip.transcript";
/// Schema name of the speakers artifact.
pub const SPEAKERS_SCHEMA: &str = "glassrip.speakers";
/// Current schema version of both artifacts.
pub const SCHEMA_VERSION: &str = "1.0.0";

/// Tool that produced an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Producer {
    /// Tool name.
    pub tool: String,
    /// Tool version.
    pub version: String,
    /// Git commit of the tool, when known at build time.
    pub git_sha: Option<String>,
}

impl Producer {
    /// Producer record for this crate.
    pub fn current() -> Self {
        Self {
            tool: env!("CARGO_PKG_NAME").to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            // set by build.rs; empty when git was unavailable at build time
            git_sha: Some(env!("GLASSRIP_GIT_SHA"))
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        }
    }
}

/// An input consumed by a stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputRef {
    /// Path as given to the stage.
    pub path: String,
    /// BLAKE3 hash of the file contents (hex).
    pub blake3: String,
    /// Schema of the input when it is a glassrip artifact.
    pub schema: Option<String>,
    /// Schema version of the input when it is a glassrip artifact.
    pub schema_version: Option<String>,
}

/// Versioned artifact envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Envelope<P, I> {
    /// Artifact schema name.
    pub schema: String,
    /// Semver of the schema.
    pub schema_version: String,
    /// Run identifier.
    pub run_id: String,
    /// Producing tool.
    pub producer: Producer,
    /// Inputs with content hashes.
    pub inputs: Vec<InputRef>,
    /// Structured parameters.
    pub params: P,
    /// Items; always a list.
    pub items: Vec<I>,
}

/// Parameters recorded in a transcript artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptParams {
    /// ASR model file name.
    pub asr_model: String,
    /// Vocabulary terms used for the prompt and the correction pass.
    pub vocabulary: Vec<String>,
    /// Decoding language.
    pub language: String,
    /// Beam width.
    pub beam_size: u32,
    /// VAD model file name, if VAD chunking was used.
    pub vad_model: Option<String>,
    /// Diarization backend and mode, if diarization ran.
    pub diarization: Option<String>,
    /// Requested speaker count.
    pub num_speakers: Option<usize>,
    /// Seconds added to audio times to place them on the video timeline.
    pub timeline_offset_s: f64,
    /// Words at or above this probability are never corrected.
    pub correction_max_p: f32,
    /// Limit used for capitalized (proper noun) substitutions.
    pub correction_max_p_proper_noun: f32,
    /// whisper backend device actually used (for example `CUDA0`, `MTL0`, `cpu`).
    pub asr_backend: String,
    /// Gap filling gates, when gap filling ran.
    pub gap_fill: Option<GapFillParams>,
}

/// Gap filling gates recorded in the transcript params.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GapFillParams {
    /// Minimum uncovered span length before widening, seconds.
    pub min_span_s: f64,
    /// Minimum cosine similarity to the chosen centroid.
    pub min_similarity: f32,
    /// Minimum similarity margin over the second-best speaker.
    pub min_margin: f32,
    /// Upper bound on the confidence of a gap-filled word.
    pub conf_cap: f32,
}

/// One word of a transcript segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptWord {
    /// Word text after the vocabulary correction pass.
    pub w: String,
    /// Verbatim ASR word when the correction pass changed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub w_raw: Option<String>,
    /// Start time, seconds.
    pub start_s: f64,
    /// End time, seconds.
    pub end_s: f64,
    /// Mean token probability.
    pub p: f32,
    /// Assigned speaker label.
    pub speaker_label: String,
    /// Confidence of the speaker assignment, in [0, 1].
    pub assign_conf: f32,
    /// Where the speaker label came from.
    #[serde(default)]
    pub source: Source,
    /// Gap fill: cosine similarity of the span to the chosen speaker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gap_sim: Option<f32>,
    /// Gap fill: similarity margin over the second-best speaker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gap_margin: Option<f32>,
}

/// One transcript segment (a run of words by one speaker).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    /// Stable segment id.
    pub segment_id: String,
    /// Start time, seconds.
    pub start_s: f64,
    /// End time, seconds.
    pub end_s: f64,
    /// Speaker label.
    pub speaker_label: String,
    /// Duration-weighted mean of word assignment confidences.
    pub speaker_conf: f32,
    /// Corrected text.
    pub text: String,
    /// Verbatim ASR text.
    pub text_raw: String,
    /// Words labeled by gap filling.
    #[serde(default)]
    pub gap_fill_words: usize,
    /// Words with a placeholder label only.
    #[serde(default)]
    pub unassigned_words: usize,
    /// Words.
    pub words: Vec<TranscriptWord>,
}

/// The `glassrip.transcript` artifact.
pub type TranscriptArtifact = Envelope<TranscriptParams, TranscriptSegment>;

/// Parameters recorded in a speakers artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakersParams {
    /// How labels were produced.
    pub method: String,
    /// Requested speaker count, if any.
    pub num_speakers_requested: Option<usize>,
    /// Clusters found by the diarizer before re-clustering.
    pub num_clusters_raw: usize,
    /// Labels in the output.
    pub num_speakers_found: usize,
}

/// A known participant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Person {
    /// Stable person id.
    pub person_id: String,
    /// Display name.
    pub display_name: String,
    /// Other spellings (ASR variants, nicknames).
    pub aliases: Vec<String>,
}

/// Resolution status of a speaker label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerStatus {
    /// Mapped to a person.
    Mapped,
    /// Not a person (noise, music, system audio).
    Noise,
    /// Not yet resolved.
    Unresolved,
}

/// Kind of evidence for a speaker mapping.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    /// Conferencing UI highlighted the active speaker.
    ActiveSpeakerHighlight,
    /// Someone addressed the speaker by name.
    DirectAddress,
    /// Talk time or role cue.
    RoleCue,
    /// No visible tile is lit and exactly one participant has no visible tile.
    AbsentTile,
    /// The segment answers a direct address to the person.
    AddressResponse,
    /// The segment addresses the person by name, so its speaker is someone else
    /// (negative evidence).
    AddressedNotSpeaker,
}

/// One piece of evidence for a speaker mapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Evidence {
    /// Time of the evidence, seconds.
    pub t_s: f64,
    /// Evidence kind.
    pub kind: EvidenceKind,
    /// Supporting text.
    pub text: String,
    /// Person the evidence points at (or away from), when not the item's person.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub person_id: Option<String>,
    /// Vote weight (negative for evidence against).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    /// Transcript segment the evidence comes from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment_id: Option<String>,
}

/// One speaker label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SpeakerItem {
    /// Diarization label.
    pub label: String,
    /// Resolution status.
    pub status: SpeakerStatus,
    /// Mapped person, if any.
    pub person_id: Option<String>,
    /// Confidence of the mapping, in [0, 1].
    pub confidence: f32,
    /// Evidence for the mapping.
    pub evidence: Vec<Evidence>,
    /// Total speaking time of the label, seconds.
    pub talk_time_s: f64,
    /// Part of `talk_time_s` that comes from gap-fill turns, seconds.
    #[serde(default)]
    pub talk_time_gap_fill_s: f64,
    /// Words labeled from diarizer turns.
    #[serde(default)]
    pub words_diarizer: usize,
    /// Words labeled by gap filling.
    #[serde(default)]
    pub words_gap_fill: usize,
    /// Words carrying this label only as a placeholder.
    #[serde(default)]
    pub words_unassigned: usize,
    /// Vote totals per person (speaker naming).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub votes: BTreeMap<String, f64>,
    /// Evidence found before `evidence` was trimmed to the strongest items.
    #[serde(default)]
    pub evidence_total: usize,
}

/// The `glassrip.speakers` artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakersArtifact {
    /// Envelope with speaker items.
    #[serde(flatten)]
    pub envelope: Envelope<SpeakersParams, SpeakerItem>,
    /// Known participants.
    pub people: Vec<Person>,
    /// Free-form notes, kept separate from items.
    pub notes: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speakers_artifact_flattens_envelope() {
        let art = SpeakersArtifact {
            envelope: Envelope {
                schema: SPEAKERS_SCHEMA.into(),
                schema_version: SCHEMA_VERSION.into(),
                run_id: "r1".into(),
                producer: Producer::current(),
                inputs: vec![],
                params: SpeakersParams {
                    method: "test".into(),
                    num_speakers_requested: Some(2),
                    num_clusters_raw: 3,
                    num_speakers_found: 2,
                },
                items: vec![SpeakerItem {
                    label: "SPEAKER_00".into(),
                    status: SpeakerStatus::Unresolved,
                    person_id: None,
                    confidence: 0.0,
                    evidence: vec![],
                    talk_time_s: 1.5,
                    talk_time_gap_fill_s: 0.0,
                    words_diarizer: 3,
                    words_gap_fill: 0,
                    words_unassigned: 0,
                    votes: BTreeMap::new(),
                    evidence_total: 0,
                }],
            },
            people: vec![],
            notes: None,
        };
        let v = serde_json::to_value(&art).unwrap();
        assert_eq!(v["schema"], "glassrip.speakers");
        assert_eq!(v["items"][0]["status"], "unresolved");
        let back: SpeakersArtifact = serde_json::from_value(v).unwrap();
        assert_eq!(back, art);
    }

    #[test]
    fn word_raw_is_omitted_when_unchanged() {
        let w = TranscriptWord {
            w: "hello".into(),
            w_raw: None,
            start_s: 0.0,
            end_s: 0.5,
            p: 0.9,
            speaker_label: "SPEAKER_00".into(),
            assign_conf: 1.0,
            source: Source::Diarizer,
            gap_sim: None,
            gap_margin: None,
        };
        let v = serde_json::to_value(&w).unwrap();
        assert!(v.get("w_raw").is_none());
        assert!(v.get("gap_sim").is_none());
        assert_eq!(v["source"], "diarizer");
    }
}
