//! Board-state consolidation for glassrip meeting mode.
//!
//! The vision stages read each whiteboard keyframe independently. This crate turns
//! those readings into one board: it checks edge directions from pixels and a binary
//! model check, registers keyframes into a shared canvas frame through text anchors,
//! merges elements across keyframes with lifetimes and support rules, tracks owner
//! tags as timed assignments, and computes change events.

pub mod difflib;
pub mod register;
pub mod similarity;
pub mod text;
