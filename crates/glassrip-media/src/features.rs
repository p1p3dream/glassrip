//! Per-frame features and pair scores in `prototype_compat` mode.

use std::path::Path;

use rayon::prelude::*;

use crate::decode::decode_gray_and_bgr_bytes;
use crate::ecc::{find_transform_ecc_affine, EccOutcome, EccParams};
use crate::error::{MediaError, Result};
use crate::gaussian::{blur_u8, KernelSize};
use crate::ink::{ink_change, ink_frame, InkFrame};
use crate::plane::Plane;
use crate::resize::area_gray;
use crate::sharpness::laplacian_variance;
use crate::ssim::{changed_fraction, inset_mean_f32, ssim_map};
use crate::warp::{affine_linear_f32, IDENTITY};

/// How features and alignment are computed. The mode is part of any cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Mode {
    /// Reproduces the Python prototype exactly (identity ECC start, partial warp kept on
    /// ECC failure, fixed 15 px inset). Used for the parity gate.
    PrototypeCompat,
}

/// Width and height of the pair-score image.
pub const SMALL_W: usize = 320;
/// See [`SMALL_W`].
pub const SMALL_H: usize = 180;
/// Inset excluded from SSIM and `changed_frac`.
pub const INSET: usize = 15;
/// Absolute difference above which a pixel counts as changed.
pub const CHANGE_THRESH: f32 = 25.0;

/// Everything computed once per frame.
#[derive(Debug, Clone)]
pub struct FrameFeatures {
    /// Laplacian variance of the full-resolution gray frame.
    pub sharpness: f64,
    /// 320x180 pair-score image: 6x6 area mean, fixed-point Gaussian 5x5 sigma 1.2, as f32.
    pub small: Plane<f32>,
    /// Ink metric inputs.
    pub ink: InkFrame,
}

/// Builds the pair-score image from full-resolution gray (`load()` in the prototype).
pub fn small_gray(gray: &Plane<u8>) -> Plane<f32> {
    let factor = (gray.width / SMALL_W).max(1);
    blur_u8(&area_gray(gray, factor), KernelSize::K5, 1.2).map(f32::from)
}

/// Computes all per-frame features from JPEG bytes. `path` only labels errors.
pub fn frame_features_from_bytes(path: &Path, bytes: &[u8]) -> Result<FrameFeatures> {
    let (gray, bgr) = decode_gray_and_bgr_bytes(path, bytes)?;
    if (gray.width, gray.height) != crate::ink::INK_INPUT {
        return Err(MediaError::Size {
            width: gray.width,
            height: gray.height,
            reason: "prototype_compat needs 1920x1080 frames",
        });
    }
    let (sharpness, small) = rayon::join(|| laplacian_variance(&gray), || small_gray(&gray));
    let ink = ink_frame(&bgr)?;
    Ok(FrameFeatures {
        sharpness,
        small,
        ink,
    })
}

/// Computes all per-frame features for a JPEG file.
pub fn frame_features(path: &Path) -> Result<FrameFeatures> {
    let bytes = std::fs::read(path).map_err(|source| MediaError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    frame_features_from_bytes(path, &bytes)
}

/// Computes features for many frames in parallel, preserving order.
pub fn frames_features(paths: &[std::path::PathBuf]) -> Result<Vec<FrameFeatures>> {
    paths.par_iter().map(|p| frame_features(p)).collect()
}

/// Output of [`pair_score`] (`score()` in the prototype).
#[derive(Debug, Clone, Copy)]
pub struct PairScore {
    /// Mean SSIM over the inset region.
    pub ssim: f64,
    /// Share of inset pixels whose aligned difference exceeds 25.
    pub changed_frac: f64,
    /// Length of the ECC translation.
    pub shift: f64,
    /// ECC outcome (warp maps `a` coordinates into `b`).
    pub ecc: EccOutcome,
}

/// Aligns `b` to `a` with ECC, warps it, and scores the pair.
pub fn pair_score(a: &Plane<f32>, b: &Plane<f32>) -> PairScore {
    let ecc = find_transform_ecc_affine(a, b, IDENTITY, EccParams::default());
    let wm = ecc.warp;
    let bw = affine_linear_f32(b, &wm, a.width, a.height);
    let ssim = f64::from(inset_mean_f32(&ssim_map(a, &bw), INSET));
    let changed_frac = changed_fraction(a, &bw, INSET, CHANGE_THRESH);
    let shift = f64::from(wm[2].hypot(wm[5]));
    PairScore {
        ssim,
        changed_frac,
        shift,
        ecc,
    }
}

/// Pair score plus ink change between two frames.
#[derive(Debug, Clone, Copy)]
pub struct PairFeatures {
    /// Pair score on the small gray images.
    pub score: PairScore,
    /// Ink change value.
    pub ink_change: f64,
    /// Whether the ink path's ECC converged.
    pub ink_align_ok: bool,
}

/// Computes the pair score and the ink change of `b` relative to `a`.
pub fn pair_features(a: &FrameFeatures, b: &FrameFeatures) -> PairFeatures {
    let (score, ink) = rayon::join(
        || pair_score(&a.small, &b.small),
        || ink_change(&a.ink, &b.ink),
    );
    PairFeatures {
        score,
        ink_change: ink.value,
        ink_align_ok: ink.ecc.ok(),
    }
}

/// [`crate::segment::PairOracle`] backed by precomputed frame features.
pub struct FeatureOracle<'a> {
    /// Features indexed by frame.
    pub frames: &'a [FrameFeatures],
}

impl crate::segment::PairOracle for FeatureOracle<'_> {
    fn score(&self, a: usize, b: usize) -> (f64, f64, bool) {
        let s = pair_score(&self.frames[a].small, &self.frames[b].small);
        (s.ssim, s.changed_frac, s.ecc.ok())
    }

    fn ink(&self, a: usize, b: usize) -> f64 {
        ink_change(&self.frames[a].ink, &self.frames[b].ink).value
    }
}
