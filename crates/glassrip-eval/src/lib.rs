//! Evaluation harness for glassrip meeting and document modes (spec section 9).

#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod cli;
pub mod error;
pub mod fixture;
pub mod gate;
pub mod golden;
pub mod metrics;
pub mod privacy;
pub mod replay;
pub mod report;
pub mod suite;
pub mod synth;
pub mod text;
pub mod timejoin;
pub mod views;

pub use error::{EvalError, Result};
