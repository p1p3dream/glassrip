//! Media primitives for glassrip meeting mode: JPEG decoding, frame features, ECC alignment,
//! the ink metric and keyframe segmentation.
//!
//! Everything here reproduces the Python prototype's OpenCV calls in `prototype_compat` mode;
//! see the module docs for the exact OpenCV code paths each function follows.

pub mod decode;
pub mod ecc;
pub mod error;
pub mod gaussian;
pub mod jpeg;
pub mod plane;
pub mod resize;
pub mod segment;
pub mod sharpness;
pub mod ssim;
pub mod util;
pub mod warp;

pub use error::{MediaError, Result};
pub use plane::{Bgr, Plane};
