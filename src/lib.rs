#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod extract;
pub mod frames;
pub mod meeting;
pub mod output;
pub mod pipeline;
pub mod retry;
pub mod stitch;

#[cfg(test)]
mod test_support;

pub mod types {
    use serde::{Deserialize, Serialize};
    use std::path::PathBuf;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Word {
        pub word: String,
        pub start: f64,
        pub end: f64,
        pub confidence: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SampledFrame {
        pub timestamp: f64,
        pub path: PathBuf,
        pub is_keyframe: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum RegionType {
        Editor,
        Terminal,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CodeRegion {
        pub x: u32,
        pub y: u32,
        pub w: u32,
        pub h: u32,
        pub region_type: RegionType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CodeRevision {
        pub timestamp: f64,
        pub content: String,
        #[serde(default)]
        pub narration: Option<String>,
        #[serde(default)]
        pub frame_index: usize,
        #[serde(default)]
        pub diff: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ExtractedFile {
        pub filename: String,
        pub language: String,
        pub final_content: String,
        #[serde(default)]
        pub revisions: Vec<CodeRevision>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TerminalCommand {
        pub timestamp: f64,
        pub command: String,
        #[serde(default)]
        pub output: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Annotation {
        pub timestamp: f64,
        pub text: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PipelineOutput {
        #[serde(default)]
        pub files: Vec<ExtractedFile>,
        #[serde(default)]
        pub terminal_commands: Vec<TerminalCommand>,
        #[serde(default)]
        pub annotations: Vec<Annotation>,
        #[serde(default)]
        pub duration: f64,
        #[serde(default)]
        pub source_video: String,
    }
}
