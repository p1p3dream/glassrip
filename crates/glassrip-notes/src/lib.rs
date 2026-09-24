//! glassrip-notes: speaker naming and citation-validated meeting notes.
//!
//! Stages (both implement [`glassrip_core::runner::Stage`]):
//!
//! - [`speakers::NameSpeakersStage`] (`name_speakers`, `glassrip.speakers`): maps
//!   diarization labels to participants by an evidence vote over visual
//!   active-speaker cues (frames decoded on demand from the source video), direct
//!   address in the transcript, and talk-time and presenter cues, and relabels
//!   turns whose visual cue contradicts the diarizer.
//! - [`notes::NotesStage`] (`notes`, `glassrip.meeting_notes`): a local text model
//!   drafts decisions, action items, open questions, a timeline and a summary over
//!   transcript windows; every item must cite evidence ids, and a Rust validator
//!   decides what is kept.
//!
//! The board state is consumed through the local [`board`] shape until the
//! board-state crate lands (see the `UNIFY` notes there).
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod board;
pub mod import;
pub mod named;
pub mod notes;
pub mod people;
pub mod speakers;
pub mod text;

/// Schema names used by this crate.
pub mod schemas {
    /// Transcript artifact (produced by the audio branch).
    pub const TRANSCRIPT: &str = "glassrip.transcript";
    /// Keyframes artifact (produced by the media branch).
    pub const KEYFRAMES: &str = "glassrip.keyframes";
    /// OCR artifact (produced by the vision branch).
    pub const OCR: &str = "glassrip.ocr";
    /// Speakers artifact (this crate).
    pub const SPEAKERS: &str = "glassrip.speakers";
    /// Board state artifact (board-state consolidation).
    pub const BOARD_STATE: &str = "glassrip.board_state";
    /// Meeting notes artifact (this crate).
    pub const MEETING_NOTES: &str = "glassrip.meeting_notes";
}
