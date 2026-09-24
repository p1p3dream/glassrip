//! Vision model access for glassrip meeting mode.
//!
//! The model extracts; Rust decides. This crate sends one schema-constrained
//! image request at a time to a local model server, validates every reply
//! against the schema and the Rust type, and applies deterministic rules on
//! top (screen classification combiner, board validation).

pub mod backend;
pub mod board;
pub mod classify;
pub mod error;
pub mod geometry;
pub mod image_prep;
pub mod ollama;
pub mod pool;
pub mod schema;

pub use backend::{
    BackendId, Durations, GenerationOptions, Placement, RawResponse, VisionBackend, VisionRequest,
};
pub use error::{FieldError, Result, VisionError};
pub use geometry::BBox;
pub use image_prep::{EncodedImage, PreparedImage, SizePlan};
pub use ollama::{ModelSize, OllamaBackend, OllamaConfig, SelfTestReport};
pub use pool::VisionClient;
pub use schema::OutputSchema;
