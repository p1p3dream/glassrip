//! Per-frame features and pair scores in either features mode, shared by `features` and
//! `keyframes` so anchor comparisons use exactly the consecutive-pair scoring.

use std::path::Path;

use glassrip_core::config::{FeaturesConfig, FeaturesMode};
use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_media::ecc::EccParams;
use glassrip_media::features::{FrameFeatures as MediaFeatures, frame_features_from_bytes};
use glassrip_media::production::{self, AlignParams, ScoreParams};
use schemars::JsonSchema;
use serde::Serialize;

use crate::schema::AlignMethod;

/// Scoring parameters (part of the `features` and `keyframes` keys).
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ScoringParams {
    /// `production` or `prototype_compat`.
    pub mode: FeaturesMode,
    /// ECC iteration limit (`production`; compat uses the prototype's 60).
    pub ecc_iterations: u32,
    /// ECC epsilon (`production`).
    pub ecc_eps: f64,
    /// ECC prefilter size (`production`).
    pub ecc_gauss_filt_size: u32,
    /// Changed-pixel delta (`production`).
    pub changed_pixel_delta: u32,
    /// Minimum phase-correlation response trusted as a translation.
    pub min_phase_response: f64,
    /// Pairs with a smaller valid-pixel share count as `align_failed`.
    pub min_valid_frac: f64,
    /// Largest accepted translation (share of width or height).
    pub max_shift_frac: f64,
    /// Largest accepted deviation of the affine linear part from the identity.
    pub max_linear_dev: f64,
}

impl ScoringParams {
    /// From the `[features]` config section.
    pub fn from_config(c: &FeaturesConfig) -> Self {
        let d = AlignParams::default();
        Self {
            mode: c.mode,
            ecc_iterations: c.ecc_iterations,
            ecc_eps: c.ecc_eps,
            ecc_gauss_filt_size: c.ecc_gauss_filt_size,
            changed_pixel_delta: c.changed_pixel_delta,
            min_phase_response: d.min_phase_response,
            min_valid_frac: ScoreParams::default().min_valid_frac,
            max_shift_frac: d.max_shift_frac,
            max_linear_dev: d.max_linear_dev,
        }
    }

    /// Mode name used in cache keys.
    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            FeaturesMode::Production => "production",
            FeaturesMode::PrototypeCompat => "prototype_compat",
        }
    }
}

/// Everything needed to score pairs.
#[derive(Debug, Clone)]
pub struct Scorer {
    mode: FeaturesMode,
    score: ScoreParams,
}

/// Scores of one ordered pair `(template, input)`.
#[derive(Debug, Clone, Copy)]
pub struct PairOut {
    /// SSIM.
    pub ssim: f64,
    /// Changed fraction.
    pub changed_frac: f64,
    /// Ink change.
    pub ink_change: f64,
    /// Pair-score alignment usable.
    pub align_ok: bool,
    /// Method.
    pub align_method: AlignMethod,
    /// Ink alignment usable.
    pub ink_align_ok: bool,
    /// Translation length.
    pub shift: f64,
    /// Valid share (`production`).
    pub valid_frac: Option<f64>,
}

impl Scorer {
    /// Builds a scorer, validating ECC parameters.
    pub fn new(p: &ScoringParams) -> Result<Self, String> {
        let ecc = EccParams::new(
            p.ecc_iterations as usize,
            p.ecc_eps,
            p.ecc_gauss_filt_size as usize,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            mode: p.mode,
            score: ScoreParams {
                align: AlignParams {
                    ecc,
                    min_phase_response: p.min_phase_response,
                    max_shift_frac: p.max_shift_frac,
                    max_linear_dev: p.max_linear_dev,
                },
                changed_delta: p.changed_pixel_delta as f32,
                min_valid_frac: p.min_valid_frac,
            },
        })
    }

    /// Per-frame features from a JPEG, verifying its blake3 when given.
    pub fn features(&self, path: &Path, blake3: Option<&str>) -> Result<MediaFeatures, ErrorInfo> {
        let bytes =
            fs_err::read(path).map_err(|e| crate::util::io_error("cannot read", path, e))?;
        if let Some(want) = blake3 {
            let got = blake3::hash(&bytes).to_hex().to_string();
            if got != want {
                return Err(ErrorInfo::new(
                    ErrorCode::InvalidInput,
                    format!(
                        "{} changed since it was recorded (blake3 {got}, expected {want}); rerun with --force-stage frames",
                        path.display()
                    ),
                ));
            }
        }
        frame_features_from_bytes(path, &bytes).map_err(crate::util::media_error)
    }

    /// Scores `b` against template `a`, including ink.
    pub fn pair(&self, a: &MediaFeatures, b: &MediaFeatures) -> PairOut {
        let (s, ink) = rayon::join(|| self.score(a, b), || self.ink(a, b));
        PairOut {
            ssim: s.ssim,
            changed_frac: s.changed_frac,
            ink_change: ink.0,
            align_ok: s.align_ok,
            align_method: s.method,
            ink_align_ok: ink.1,
            shift: s.shift,
            valid_frac: s.valid_frac,
        }
    }

    /// Pair score without ink.
    pub fn score(&self, a: &MediaFeatures, b: &MediaFeatures) -> ScoreOut {
        match self.mode {
            FeaturesMode::PrototypeCompat => {
                let s = glassrip_media::features::pair_score(&a.small, &b.small);
                ScoreOut {
                    ssim: s.ssim,
                    changed_frac: s.changed_frac,
                    align_ok: s.ecc.ok(),
                    method: AlignMethod::CompatEcc,
                    shift: s.shift,
                    valid_frac: None,
                }
            }
            FeaturesMode::Production => {
                let s = production::pair_score(&a.small, &b.small, &self.score);
                ScoreOut {
                    ssim: s.ssim,
                    changed_frac: s.changed_frac,
                    align_ok: s.align_ok(),
                    method: s.alignment.method.into(),
                    shift: s.shift,
                    valid_frac: Some(s.valid_frac),
                }
            }
        }
    }

    /// Ink change of `b` against `a`, and whether its alignment was usable.
    pub fn ink(&self, a: &MediaFeatures, b: &MediaFeatures) -> (f64, bool) {
        match self.mode {
            FeaturesMode::PrototypeCompat => {
                let c = glassrip_media::ink::ink_change(&a.ink, &b.ink);
                (c.value, c.ecc.ok())
            }
            FeaturesMode::Production => {
                let c = production::ink_change(&a.ink, &b.ink, &self.score.align);
                (c.value, c.alignment.ok())
            }
        }
    }

    /// True in `production` mode.
    pub fn is_production(&self) -> bool {
        self.mode == FeaturesMode::Production
    }
}

/// Output of [`Scorer::score`].
#[derive(Debug, Clone, Copy)]
pub struct ScoreOut {
    /// SSIM.
    pub ssim: f64,
    /// Changed fraction.
    pub changed_frac: f64,
    /// Alignment usable.
    pub align_ok: bool,
    /// Method.
    pub method: AlignMethod,
    /// Translation length.
    pub shift: f64,
    /// Valid share.
    pub valid_frac: Option<f64>,
}
