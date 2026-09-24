//! glassrip-audio: the audio branch of glassrip meeting mode.
//!
//! Stages:
//! 1. [`extract`]: decode the first audio stream to 16 kHz mono `f32` with ffmpeg,
//!    recording stream start times so every output time sits on the video timeline.
//! 2. [`asr`]: whisper.cpp (via `whisper-rs`) with Silero VAD chunking, beam search,
//!    DTW token timestamps and a vocabulary prompt; words are built from tokens.
//! 3. [`vocab`]: post-pass replacing low-probability words that sound like a
//!    vocabulary term (raw text is kept alongside the corrected text).
//! 4. [`diarize`]: speakrs (pyannote community-1 port) plus optional re-clustering
//!    to a known speaker count ([`recluster`]).
//! 5. [`gapfill`]: ASR words outside every diarization turn are labeled by
//!    embedding the uncovered speech and matching it to the speaker centroids.
//! 6. [`assign`]: each word goes to the exclusive speaker turn it overlaps most.
//! 7. [`transcript`]: `glassrip.transcript` and `glassrip.speakers` artifacts ([`types`]).
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub mod asr;
pub mod assign;
pub mod diarize;
pub mod error;
pub mod extract;
pub mod gapfill;
pub mod metrics;
pub mod models;
pub mod pipeline;
pub mod recluster;
pub mod transcript;
pub mod types;
pub mod vocab;
pub mod words;

pub use error::{AudioError, Result};
