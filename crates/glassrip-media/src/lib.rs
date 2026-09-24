//! Media primitives for glassrip meeting mode: JPEG decoding, frame features, ECC alignment,
//! the ink metric and keyframe segmentation.
//!
//! Everything here reproduces the Python prototype's OpenCV calls in `prototype_compat` mode;
//! see the module docs for the exact OpenCV code paths each function follows.

pub mod decode;
pub mod error;
pub mod jpeg;
pub mod plane;
pub mod util;

pub use error::{MediaError, Result};
pub use plane::{Bgr, Plane};
