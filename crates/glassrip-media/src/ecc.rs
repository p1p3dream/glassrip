//! Port of OpenCV's `findTransformECC` (modules/video/src/ecc.cpp) for `MOTION_AFFINE`
//! without masks, as called by the prototype:
//!
//! ```text
//! cv2.findTransformECC(template, input, wm, cv2.MOTION_AFFINE,
//!                      (COUNT | EPS, 60, 1e-4), None, 3)
//! ```
//!
//! The arithmetic follows OpenCV's float/double split: images, gradients, Jacobian, Hessian
//! and parameter updates are f32; means, norms and dot products accumulate in f64 the way
//! OpenCV's `meanStdDev`, `norm`, `Mat::dot` and small `gemm` do.

use rayon::prelude::*;

use crate::error::{MediaError, Result};
use crate::gaussian::{blur_f32, KernelSize};
use crate::plane::Plane;
use crate::util::dot_prod_32f;
use crate::warp::{affine_linear_f32, affine_nearest_u8, Affine};

/// Termination criteria (`TERM_CRITERIA_COUNT | TERM_CRITERIA_EPS`) and prefilter size.
/// Construct with [`EccParams::new`]; the default is the prototype's `(60, 1e-4, 3)`.
#[derive(Debug, Clone, Copy)]
pub struct EccParams {
    max_iter: usize,
    eps: f64,
    gauss_size: KernelSize,
}

impl EccParams {
    /// Validates the parameters: `max_iter >= 1`, `eps` finite and non-negative,
    /// `gauss_size` odd and at least 3.
    pub fn new(max_iter: usize, eps: f64, gauss_size: usize) -> Result<Self> {
        if max_iter == 0 {
            return Err(MediaError::Invalid(
                "ECC max_iter must be at least 1".into(),
            ));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(MediaError::Invalid(format!(
                "ECC eps must be finite and >= 0, got {eps}"
            )));
        }
        if gauss_size < 3 {
            return Err(MediaError::Invalid(format!(
                "ECC gauss_size must be odd and at least 3, got {gauss_size}"
            )));
        }
        Ok(Self {
            max_iter,
            eps,
            gauss_size: KernelSize::new(gauss_size)?,
        })
    }

    /// Maximum iterations.
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Correlation change below which iteration stops.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Prefilter size.
    pub fn gauss_size(&self) -> usize {
        self.gauss_size.get()
    }
}

impl Default for EccParams {
    fn default() -> Self {
        Self {
            max_iter: 60,
            eps: 1e-4,
            gauss_size: KernelSize::K3,
        }
    }
}

/// Why ECC stopped early. OpenCV raises `StsNoConv` in both cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EccFailure {
    /// The correlation became NaN.
    NaN,
    /// `lambda_d <= 0`: the images are uncorrelated or do not overlap.
    NotConverging,
}

/// Result of an ECC run.
#[derive(Debug, Clone, Copy)]
pub struct EccOutcome {
    /// Final warp. On failure this is the matrix as it stood when OpenCV raised: the
    /// Python bindings update the caller's array in place, so the prototype kept these
    /// partial updates rather than the identity it passed in.
    pub warp: Affine,
    /// Final correlation coefficient (meaningless on failure).
    pub rho: f64,
    /// Iterations started.
    pub iterations: usize,
    /// `None` on success.
    pub failure: Option<EccFailure>,
}

impl EccOutcome {
    /// True when OpenCV would have returned normally.
    pub fn ok(&self) -> bool {
        self.failure.is_none()
    }
}

/// Central-difference gradient `filter2D(img, -1, [-0.5, 0, 0.5])` (REFLECT_101).
fn gradients(img: &Plane<f32>) -> (Plane<f32>, Plane<f32>) {
    let (w, h) = (img.width, img.height);
    let mut gx = Plane::new(w, h);
    let mut gy = Plane::new(w, h);
    let refl = |i: isize, n: usize| crate::util::reflect101(i, n);
    for y in 0..h {
        let up = refl(y as isize - 1, h);
        let dn = refl(y as isize + 1, h);
        for x in 0..w {
            let l = img.get(refl(x as isize - 1, w), y);
            let r = img.get(refl(x as isize + 1, w), y);
            // FilterVec_32f skips the zero tap: fma(0.5, r, -0.5 * l).
            gx.data[y * w + x] = r.mul_add(0.5, -0.5 * l);
            let u = img.get(x, up);
            let d = img.get(x, dn);
            gy.data[y * w + x] = d.mul_add(0.5, -0.5 * u);
        }
    }
    (gx, gy)
}

/// `LUImpl<float>` solving `A X = I`, returning `A^-1` or all zeros when singular.
///
/// The index loops follow OpenCV's loop structure one to one so the operation order (and
/// therefore the f32 rounding) matches.
#[allow(clippy::needless_range_loop)]
fn invert6(a: &[[f32; 6]; 6]) -> [[f32; 6]; 6] {
    let mut a = *a;
    let mut b = [[0.0f32; 6]; 6];
    for (i, row) in b.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let eps = f32::EPSILON * 10.0;
    let m = 6;
    for i in 0..m {
        let mut k = i;
        for j in i + 1..m {
            if a[j][i].abs() > a[k][i].abs() {
                k = j;
            }
        }
        if a[k][i].abs() < eps {
            return [[0.0; 6]; 6];
        }
        if k != i {
            a.swap(i, k);
            b.swap(i, k);
        }
        let d = -1.0 / a[i][i];
        for j in i + 1..m {
            let alpha = a[j][i] * d;
            for kk in i + 1..m {
                a[j][kk] = alpha.mul_add(a[i][kk], a[j][kk]);
            }
            for kk in 0..m {
                b[j][kk] = alpha.mul_add(b[i][kk], b[j][kk]);
            }
        }
    }
    for i in (0..m).rev() {
        for j in 0..m {
            let mut s = b[i][j];
            for k in i + 1..m {
                s = (-a[i][k]).mul_add(b[k][j], s);
            }
            b[i][j] = s / a[i][i];
        }
    }
    b
}

/// `(float)(A * v)` with f64 accumulation, as OpenCV's small float `gemm` does.
fn matvec(a: &[[f32; 6]; 6], v: &[f32; 6]) -> [f32; 6] {
    let mut out = [0.0f32; 6];
    for (o, row) in out.iter_mut().zip(a.iter()) {
        let mut s = 0.0f64;
        for (x, y) in row.iter().zip(v.iter()) {
            s += f64::from(*x) * f64::from(*y);
        }
        *o = s as f32;
    }
    out
}

/// `Mat::dot` on two 6x1 float vectors (four SIMD lanes, then an f64 tail).
fn dot6(a: &[f32; 6], b: &[f32; 6]) -> f64 {
    dot_prod_32f(a, b)
}

/// The six Jacobian blocks for MOTION_AFFINE, each `w x h`, stored contiguously per block:
/// `[gx*X, gy*X, gx*Y, gy*Y, gx, gy]`.
struct Jacobian {
    blocks: [Vec<f32>; 6],
    w: usize,
    h: usize,
}

impl Jacobian {
    fn new(gx: &Plane<f32>, gy: &Plane<f32>) -> Self {
        let (w, h) = (gx.width, gx.height);
        let mut blocks: [Vec<f32>; 6] = Default::default();
        for b in blocks.iter_mut() {
            *b = vec![0.0; w * h];
        }
        for y in 0..h {
            let yf = y as f32;
            for x in 0..w {
                let i = y * w + x;
                let xf = x as f32;
                let (a, c) = (gx.data[i], gy.data[i]);
                blocks[0][i] = a * xf;
                blocks[1][i] = c * xf;
                blocks[2][i] = a * yf;
                blocks[3][i] = c * yf;
                blocks[4][i] = a;
                blocks[5][i] = c;
            }
        }
        Self { blocks, w, h }
    }

    /// `Mat::dot` between a Jacobian block and a contiguous image: the block is a column
    /// range of a wider matrix, so OpenCV iterates row by row.
    fn dot_rows(&self, k: usize, img: &[f32]) -> f64 {
        let w = self.w;
        (0..self.h)
            .map(|y| {
                dot_prod_32f(
                    &self.blocks[k][y * w..(y + 1) * w],
                    &img[y * w..(y + 1) * w],
                )
            })
            .sum::<f64>()
    }

    /// `project_onto_jacobian_ECC(jacobian, jacobian, hessian)`.
    fn hessian(&self) -> [[f32; 6]; 6] {
        let w = self.w;
        let pairs: Vec<(usize, usize)> = (0..6).flat_map(|i| (i..6).map(move |j| (i, j))).collect();
        let vals: Vec<(usize, usize, f32)> = pairs
            .par_iter()
            .map(|&(i, j)| {
                if i == j {
                    // (float)pow(norm(block), 2); norm accumulates squares in f64 per row.
                    let mut s = 0.0f64;
                    for v in &self.blocks[i] {
                        let d = f64::from(*v);
                        s += d * d;
                    }
                    (i, j, s.sqrt().powi(2) as f32)
                } else {
                    let mut s = 0.0f64;
                    for y in 0..self.h {
                        let r = y * w..(y + 1) * w;
                        s += dot_prod_32f(&self.blocks[i][r.clone()], &self.blocks[j][r]);
                    }
                    (i, j, s as f32)
                }
            })
            .collect();
        let mut hmat = [[0.0f32; 6]; 6];
        for (i, j, v) in vals {
            hmat[i][j] = v;
            hmat[j][i] = v;
        }
        hmat
    }

    /// `project_onto_jacobian_ECC(jacobian, img, out)`.
    fn project(&self, img: &[f32]) -> [f32; 6] {
        let mut out = [0.0f32; 6];
        for (k, o) in out.iter_mut().enumerate() {
            *o = self.dot_rows(k, img) as f32;
        }
        out
    }
}

/// Masked mean and standard deviation in f64 (`meanStdDev` with a mask).
fn mean_std(img: &[f32], mask: &[u8]) -> (f64, f64, usize) {
    let mut s = 0.0f64;
    let mut sq = 0.0f64;
    let mut n = 0usize;
    for (&v, &m) in img.iter().zip(mask) {
        if m != 0 {
            let d = f64::from(v);
            s += d;
            sq += d * d;
            n += 1;
        }
    }
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    let mean = s / n as f64;
    let var = (sq / n as f64 - mean * mean).max(0.0);
    (mean, var.sqrt(), n)
}

/// Runs ECC with `MOTION_AFFINE`, aligning `input` to `template` starting from `init`.
///
/// Both images are expected to be the same size. The returned warp maps template
/// coordinates into the input image (use it with an inverse-map warp).
pub fn find_transform_ecc_affine(
    template: &Plane<f32>,
    input: &Plane<f32>,
    init: Affine,
    params: EccParams,
) -> EccOutcome {
    let (ws, hs) = (template.width, template.height);
    let (wd, hd) = (input.width, input.height);
    let template_f = blur_f32(template, params.gauss_size, 0.0);
    let image_f = blur_f32(input, params.gauss_size, 0.0);
    let (grad_x, grad_y) = gradients(&image_f);
    let pre_mask = Plane::filled(wd, hd, 1u8);

    let mut map = init;
    let mut rho = -1.0f64;
    let mut last_rho = -params.eps;
    let mut iterations = 0usize;
    let mut i = 1usize;
    while i <= params.max_iter && (rho - last_rho).abs() >= params.eps {
        iterations = i;
        let ((mut warped, mask), (gxw, gyw)) = rayon::join(
            || {
                (
                    affine_linear_f32(&image_f, &map, ws, hs),
                    affine_nearest_u8(&pre_mask, &map, ws, hs),
                )
            },
            || {
                rayon::join(
                    || affine_linear_f32(&grad_x, &map, ws, hs),
                    || affine_linear_f32(&grad_y, &map, ws, hs),
                )
            },
        );

        let (img_mean, img_std, _) = mean_std(&warped.data, &mask.data);
        let (tmp_mean, tmp_std, _) = mean_std(&template_f.data, &mask.data);
        let img_mean_f = img_mean as f32;
        let tmp_mean_f = tmp_mean as f32;
        let mut template_zm = vec![0.0f32; ws * hs];
        for (k, &m) in mask.data.iter().enumerate() {
            if m != 0 {
                warped.data[k] -= img_mean_f;
                template_zm[k] = template_f.data[k] - tmp_mean_f;
            }
        }
        let valid = mask.data.iter().filter(|&&m| m != 0).count() as f64;
        let tmp_norm = (valid * tmp_std * tmp_std).sqrt();
        let img_norm = (valid * img_std * img_std).sqrt();

        let jac = Jacobian::new(&gxw, &gyw);
        let hessian = jac.hessian();
        let hessian_inv = invert6(&hessian);
        let correlation = dot_prod_32f(&template_zm, &warped.data);
        last_rho = rho;
        rho = correlation / (img_norm * tmp_norm);
        if rho.is_nan() {
            return EccOutcome {
                warp: map,
                rho,
                iterations,
                failure: Some(EccFailure::NaN),
            };
        }
        let image_proj = jac.project(&warped.data);
        let template_proj = jac.project(&template_zm);
        let image_proj_h = matvec(&hessian_inv, &image_proj);
        let lambda_n = img_norm * img_norm - dot6(&image_proj, &image_proj_h);
        let lambda_d = correlation - dot6(&template_proj, &image_proj_h);
        if lambda_d <= 0.0 {
            return EccOutcome {
                warp: map,
                rho: -1.0,
                iterations,
                failure: Some(EccFailure::NotConverging),
            };
        }
        let lambda = (lambda_n / lambda_d) as f32;
        // error = lambda * templateZM - imageWarped  (addWeighted: fma(t, l, fma(w, -1, 0)))
        let error: Vec<f32> = template_zm
            .iter()
            .zip(&warped.data)
            .map(|(&t, &w)| t.mul_add(lambda, -w))
            .collect();
        let error_proj = jac.project(&error);
        let delta = matvec(&hessian_inv, &error_proj);
        map[0] += delta[0];
        map[3] += delta[1];
        map[1] += delta[2];
        map[4] += delta[3];
        map[2] += delta[4];
        map[5] += delta[5];
        i += 1;
    }
    EccOutcome {
        warp: map,
        rho,
        iterations,
        failure: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::warp::IDENTITY;

    /// Continuous test field: several orientations and scales so every affine parameter is
    /// observable, smooth enough for a wide convergence basin.
    fn field(x: f32, y: f32) -> f32 {
        128.0
            + 50.0 * (x * 0.13).sin() * (y * 0.11).cos()
            + 35.0 * ((x + 2.0 * y) * 0.07).sin()
            + 25.0 * ((3.0 * x - y) * 0.045).cos()
            + 15.0 * (x * 0.31 + 1.0).sin() * (y * 0.27 + 2.0).sin()
    }

    fn sample(w: usize, h: usize, m: &Affine) -> Plane<f32> {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (xf, yf) = (x as f32, y as f32);
                let sx = m[0] * xf + m[1] * yf + m[2];
                let sy = m[3] * xf + m[4] * yf + m[5];
                *p.get_mut(x, y) = field(sx, sy);
            }
        }
        p
    }

    #[test]
    fn identity_pair_converges_to_identity() {
        let a = sample(96, 64, &IDENTITY);
        let out = find_transform_ecc_affine(&a, &a, IDENTITY, EccParams::default());
        assert!(out.ok());
        for (g, w) in out.warp.iter().zip(IDENTITY.iter()) {
            assert!((g - w).abs() < 1e-4, "{:?}", out.warp);
        }
        assert!(out.rho > 0.9999);
    }

    #[test]
    fn recovers_known_affine() {
        let (w, h) = (160, 120);
        let truth: Affine = [1.01, 0.02, 2.5, -0.015, 0.99, -1.75];
        // a(p) = field(p) and b(q) = field(truth^-1 q), so b(truth p) = a(p): ECC, which
        // finds W with a(p) ~ b(W p), should return `truth`.
        let det = truth[0] * truth[4] - truth[1] * truth[3];
        let inv: Affine = [
            truth[4] / det,
            -truth[1] / det,
            (truth[1] * truth[5] - truth[4] * truth[2]) / det,
            -truth[3] / det,
            truth[0] / det,
            (truth[3] * truth[2] - truth[0] * truth[5]) / det,
        ];
        let a = sample(w, h, &IDENTITY);
        let b = sample(w, h, &inv);
        let out = find_transform_ecc_affine(&a, &b, IDENTITY, EccParams::default());
        assert!(out.ok());
        for (k, (g, t)) in out.warp.iter().zip(truth.iter()).enumerate() {
            let tol = if k == 2 || k == 5 { 0.05 } else { 0.001 };
            assert!(
                (g - t).abs() < tol,
                "param {k}: got {g}, want {t} ({:?})",
                out.warp
            );
        }
    }

    #[test]
    fn unrelated_images_report_failure_and_keep_partial_warp() {
        // Two unrelated noise images: OpenCV raises StsNoConv; the outcome must say so.
        let mut state: u32 = 7;
        let mut noise = |_: usize| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as f32
        };
        let a = Plane::from_vec(64, 48, (0..64 * 48).map(&mut noise).collect()).unwrap();
        let b = Plane::from_vec(64, 48, (0..64 * 48).map(&mut noise).collect()).unwrap();
        let out = find_transform_ecc_affine(&a, &b, IDENTITY, EccParams::default());
        assert!(!out.ok());
        assert!(out.warp.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn params_are_validated() {
        assert!(EccParams::new(60, 1e-4, 3).is_ok());
        assert!(EccParams::new(60, 1e-4, 1).is_err());
        assert!(EccParams::new(60, 1e-4, 4).is_err());
        assert!(EccParams::new(0, 1e-4, 3).is_err());
        assert!(EccParams::new(60, f64::NAN, 3).is_err());
        assert_eq!(EccParams::default().gauss_size(), 3);
    }

    #[test]
    fn invert6_inverts() {
        let mut m = [[0.0f32; 6]; 6];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = if i == j {
                    4.0
                } else {
                    1.0 / (1 + i + j) as f32
                };
            }
        }
        let inv = invert6(&m);
        for (i, row) in m.iter().enumerate() {
            for j in 0..6 {
                let s: f32 = row.iter().zip(inv.iter()).map(|(a, r)| a * r[j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((s - want).abs() < 1e-5);
            }
        }
        assert_eq!(invert6(&[[0.0; 6]; 6]), [[0.0; 6]; 6]);
    }
}
