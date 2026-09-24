//! Media-branch stages of glassrip meeting mode, as [`glassrip_core::runner::Stage`]
//! implementations:
//!
//! | Stage | Output artifact |
//! |---|---|
//! | [`probe::ProbeStage`] | `glassrip.media_probe` |
//! | [`orient::OrientStage`] | `glassrip.orientation` |
//! | [`frames::FramesStage`] | `glassrip.frames` |
//! | [`quad::ScreenQuadStage`] | `glassrip.screen_quads` |
//! | [`features::FeaturesStage`] | `glassrip.features` |
//! | [`keyframes::KeyframesStage`] | `glassrip.keyframes` |
//! | [`rectify::RectifyStage`] | `glassrip.rectified_keyframes` |
//!
//! Image files (sampled frames, rectified representatives) live in a content-addressed
//! [`blobs::BlobStore`] next to the stage cache and are linked into the run directory, so a
//! stage restored from cache in a new run directory still finds its files.
//! [`pipeline::run_media_stages`] wires everything to a [`glassrip_core::runner::Runner`].

#![deny(clippy::unwrap_used, clippy::expect_used)]
#![warn(missing_docs)]

pub mod blobs;
pub mod features;
pub mod frames;
pub mod keyframes;
pub mod models;
pub mod ocr;
pub mod orient;
pub mod pipeline;
pub mod probe;
pub mod quad;
pub mod rectify;
pub mod schema;
pub mod scoring;
pub mod util;
