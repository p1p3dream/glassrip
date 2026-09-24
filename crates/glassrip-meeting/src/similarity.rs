//! 2D similarity transforms: Umeyama least squares and a deterministic RANSAC.

use nalgebra::{Matrix2, Vector2};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A point in pixels.
pub type Point = (f64, f64);

/// `p -> scale * R(angle) * p + (tx, ty)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Similarity {
    /// Uniform scale.
    pub scale: f64,
    /// Rotation in radians.
    pub angle: f64,
    /// Translation x.
    pub tx: f64,
    /// Translation y.
    pub ty: f64,
}

impl Similarity {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        angle: 0.0,
        tx: 0.0,
        ty: 0.0,
    };

    /// Apply to a point.
    pub fn apply(&self, p: Point) -> Point {
        let (s, c) = self.angle.sin_cos();
        (
            self.scale * (c * p.0 - s * p.1) + self.tx,
            self.scale * (s * p.0 + c * p.1) + self.ty,
        )
    }

    /// `self` after `first`: `x -> self(first(x))`.
    pub fn compose(&self, first: &Self) -> Self {
        let t = self.apply((first.tx, first.ty));
        Self {
            scale: self.scale * first.scale,
            angle: self.angle + first.angle,
            tx: t.0,
            ty: t.1,
        }
    }

    /// The inverse transform (`None` for a zero or non-finite scale).
    pub fn inverse(&self) -> Option<Self> {
        if !(self.scale.is_finite() && self.scale.abs() > 1e-12) {
            return None;
        }
        let inv = Self {
            scale: 1.0 / self.scale,
            angle: -self.angle,
            tx: 0.0,
            ty: 0.0,
        };
        let t = inv.apply((self.tx, self.ty));
        Some(Self {
            tx: -t.0,
            ty: -t.1,
            ..inv
        })
    }
}

/// Umeyama (1991) least-squares similarity mapping `src[i]` onto `dst[i]`, without
/// reflection. Needs at least 2 points with non-zero spread.
pub fn umeyama(src: &[Point], dst: &[Point]) -> Option<Similarity> {
    let n = src.len();
    if n < 2 || dst.len() != n {
        return None;
    }
    let nf = n as f64;
    let mean = |pts: &[Point]| {
        let (sx, sy) = pts
            .iter()
            .fold((0.0, 0.0), |(ax, ay), p| (ax + p.0, ay + p.1));
        Vector2::new(sx / nf, sy / nf)
    };
    let (mx, my) = (mean(src), mean(dst));
    let mut sigma = Matrix2::zeros();
    let mut var_x = 0.0;
    for (s, d) in src.iter().zip(dst) {
        let xs = Vector2::new(s.0, s.1) - mx;
        let yd = Vector2::new(d.0, d.1) - my;
        sigma += yd * xs.transpose();
        var_x += xs.norm_squared();
    }
    sigma /= nf;
    var_x /= nf;
    if var_x <= 1e-12 {
        return None;
    }
    let svd = sigma.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    let d = svd.singular_values;
    let mut s = Matrix2::identity();
    if u.determinant() * vt.determinant() < 0.0 {
        s[(1, 1)] = -1.0;
    }
    let r = u * s * vt;
    let scale = (d[0] * s[(0, 0)] + d[1] * s[(1, 1)]) / var_x;
    let t = my - scale * r * mx;
    let angle = r[(1, 0)].atan2(r[(0, 0)]);
    let out = Similarity {
        scale,
        angle,
        tx: t[0],
        ty: t[1],
    };
    [out.scale, out.angle, out.tx, out.ty]
        .iter()
        .all(|v| v.is_finite())
        .then_some(out)
}

/// RANSAC settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RansacParams {
    /// Inlier residual threshold in destination pixels.
    pub threshold_px: f64,
    /// Minimum inliers for a valid fit.
    pub min_inliers: usize,
    /// Hypotheses to try when exhaustive enumeration would exceed this count.
    pub max_hypotheses: usize,
    /// Accepted scale range (rejects degenerate fits).
    pub min_scale: f64,
    /// Accepted scale range (rejects degenerate fits).
    pub max_scale: f64,
}

impl Default for RansacParams {
    fn default() -> Self {
        Self {
            threshold_px: 20.0,
            min_inliers: 3,
            max_hypotheses: 600,
            min_scale: 0.1,
            max_scale: 10.0,
        }
    }
}

/// A RANSAC result.
#[derive(Debug, Clone, PartialEq)]
pub struct RansacFit {
    /// Refit on the inliers.
    pub transform: Similarity,
    /// Indices of inlier correspondences.
    pub inliers: Vec<usize>,
    /// Root mean square residual over the inliers, in destination pixels.
    pub rms_px: f64,
}

fn residual(t: &Similarity, s: Point, d: Point) -> f64 {
    let p = t.apply(s);
    ((p.0 - d.0).powi(2) + (p.1 - d.1).powi(2)).sqrt()
}

fn inliers_of(t: &Similarity, src: &[Point], dst: &[Point], thr: f64) -> (Vec<usize>, f64) {
    let mut idx = Vec::new();
    let mut sum = 0.0;
    for (i, (s, d)) in src.iter().zip(dst).enumerate() {
        let r = residual(t, *s, *d);
        if r < thr {
            idx.push(i);
            sum += r * r;
        }
    }
    let rms = if idx.is_empty() {
        f64::INFINITY
    } else {
        (sum / idx.len() as f64).sqrt()
    };
    (idx, rms)
}

/// Deterministic RANSAC over 2-point hypotheses: every pair when their number is at
/// most `max_hypotheses`, otherwise a fixed pseudo-random sample. The best hypothesis
/// (most inliers, then lowest RMS) is refit on its inliers with [`umeyama`] and
/// re-scored. Returns `None` below `min_inliers`.
pub fn ransac(src: &[Point], dst: &[Point], params: &RansacParams) -> Option<RansacFit> {
    let n = src.len();
    if n < 2 || dst.len() != n || n < params.min_inliers {
        return None;
    }
    let total = n * (n - 1) / 2;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    if total <= params.max_hypotheses {
        for i in 0..n {
            for j in i + 1..n {
                pairs.push((i, j));
            }
        }
    } else {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = |m: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as usize) % m
        };
        while pairs.len() < params.max_hypotheses {
            let (i, j) = (next(n), next(n));
            if i != j {
                pairs.push((i.min(j), i.max(j)));
            }
        }
    }
    let ok_scale = |t: &Similarity| t.scale >= params.min_scale && t.scale <= params.max_scale;
    let mut best: Option<(Vec<usize>, f64)> = None;
    for (i, j) in pairs {
        let Some(t) = umeyama(&[src[i], src[j]], &[dst[i], dst[j]]) else {
            continue;
        };
        if !ok_scale(&t) {
            continue;
        }
        let (idx, rms) = inliers_of(&t, src, dst, params.threshold_px);
        let better = match &best {
            None => true,
            Some((b, brms)) => idx.len() > b.len() || (idx.len() == b.len() && rms < *brms),
        };
        if better {
            best = Some((idx, rms));
        }
    }
    let (idx, _) = best?;
    if idx.len() < params.min_inliers {
        return None;
    }
    let s: Vec<Point> = idx.iter().map(|&i| src[i]).collect();
    let d: Vec<Point> = idx.iter().map(|&i| dst[i]).collect();
    let t = umeyama(&s, &d)?;
    if !ok_scale(&t) {
        return None;
    }
    let (inliers, rms_px) = inliers_of(&t, src, dst, params.threshold_px);
    (inliers.len() >= params.min_inliers).then_some(RansacFit {
        transform: t,
        inliers,
        rms_px,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    fn close(a: Point, b: Point) -> bool {
        (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6
    }

    #[test]
    fn umeyama_recovers_exact_similarity() {
        let t = Similarity {
            scale: 1.7,
            angle: 0.3,
            tx: 40.0,
            ty: -12.0,
        };
        let src = [(0.0, 0.0), (100.0, 10.0), (30.0, 80.0), (60.0, 60.0)];
        let dst: Vec<Point> = src.iter().map(|p| t.apply(*p)).collect();
        let f = umeyama(&src, &dst).expect("fit");
        assert!((f.scale - 1.7).abs() < 1e-9 && (f.angle - 0.3).abs() < 1e-9);
        for (s, d) in src.iter().zip(&dst) {
            assert!(close(f.apply(*s), *d));
        }
    }

    #[test]
    fn inverse_and_compose() {
        let a = Similarity {
            scale: 2.0,
            angle: 0.5,
            tx: 3.0,
            ty: 4.0,
        };
        let b = Similarity {
            scale: 0.5,
            angle: -0.2,
            tx: -7.0,
            ty: 1.0,
        };
        let p = (12.0, -5.0);
        assert!(close(a.compose(&b).apply(p), a.apply(b.apply(p))));
        let inv = a.inverse().expect("invertible");
        assert!(close(inv.apply(a.apply(p)), p));
    }

    #[test]
    fn ransac_rejects_outliers_and_needs_three_inliers() {
        let t = Similarity {
            scale: 0.8,
            angle: 0.0,
            tx: 100.0,
            ty: 50.0,
        };
        let src = vec![
            (0.0, 0.0),
            (200.0, 0.0),
            (0.0, 150.0),
            (220.0, 170.0),
            (90.0, 60.0),
        ];
        let mut dst: Vec<Point> = src.iter().map(|p| t.apply(*p)).collect();
        dst[4] = (900.0, 900.0);
        let fit = ransac(&src, &dst, &RansacParams::default()).expect("fit");
        assert_eq!(fit.inliers, vec![0, 1, 2, 3]);
        assert!(fit.rms_px < 1e-6);
        // Two correspondences are never enough.
        assert!(ransac(&src[..2], &dst[..2], &RansacParams::default()).is_none());
    }

    proptest! {
        #[test]
        fn umeyama_roundtrip(scale in 0.3f64..3.0, angle in -3.0f64..3.0,
                             tx in -500.0f64..500.0, ty in -500.0f64..500.0) {
            let t = Similarity { scale, angle, tx, ty };
            let src = [(10.0, 20.0), (300.0, 40.0), (120.0, 260.0)];
            let dst: Vec<Point> = src.iter().map(|p| t.apply(*p)).collect();
            let f = umeyama(&src, &dst).expect("fit");
            for (s, d) in src.iter().zip(&dst) {
                let p = f.apply(*s);
                prop_assert!((p.0 - d.0).abs() < 1e-6 && (p.1 - d.1).abs() < 1e-6);
            }
        }
    }
}
