//! `glassrip meeting <VIDEO>` (spec section 4): turns a meeting recording into a
//! verified board state, a speaker-attributed transcript, meeting notes, and a
//! house-style SVG.
//!
//! - [`args`]: command-line flags and their resolution over `glassrip.toml`.
//! - [`backends`]: OCR, vision, text, ASR, and diarization backends for this build.
//! - [`preflight`]: tools, features, and models the selected stages need.
//! - [`run`]: stage construction and the GPU-phase schedule on the core runner.
//! - [`logging`]: stderr progress and `run.log.jsonl` (deferred until preflight passes).
//! - [`cli`]: the command entry point.
//!
//! Outputs in `--out`: `run.lock.json`, `run.log.jsonl`, `artifacts/`,
//! `frames/keyframes/`, `<stem>-meeting-notes.md`, `<stem>-architecture.svg`
//! (plus `.png`), and `raw_responses/` (vision replies for offline replay).

pub mod args;
pub mod backends;
pub mod cli;
pub mod logging;
pub mod preflight;
pub mod run;

pub use args::MeetingArgs;
pub use backends::{Backends, VisionBackends};
pub use run::{run_meeting, MeetingError, MeetingOptions, MeetingOutcome};

/// Directory (inside the output directory) holding recorded vision replies.
pub const RAW_RESPONSES_DIR: &str = "raw_responses";
