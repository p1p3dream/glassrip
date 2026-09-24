//! SSIM and changed-pixel fraction as computed by the prototype's `score()` in numpy f32.

use crate::gaussian::blur_f32;
use crate::plane::Plane;

/// SSIM stabilizers used by the prototype (for 0..255 data).
pub const C1: f32 = 6.5;
/// See [`C1`].
pub const C2: f32 = 58.5;

/// Per-pixel SSIM map with a 7x7, sigma 1.5 Gaussian window (f32, REFLECT_101).
pub fn ssim_map(a: &Plane<f32>, b: &Plane<f32>) -> Plane<f32> {
    let g = |p: &Plane<f32>| blur_f32(p, 7, 1.5);
    let mul = |x: &Plane<f32>, y: &Plane<f32>| Plane {
        width: x.width,
        height: x.height,
        data: x.data.iter().zip(&y.data).map(|(p, q)| p * q).collect(),
    };
    let (mu1, mu2) = rayon::join(|| g(a), || g(b));
    let ((e11, e22), e12) = rayon::join(
        || rayon::join(|| g(&mul(a, a)), || g(&mul(b, b))),
        || g(&mul(a, b)),
    );
    let mut out = Plane::new(a.width, a.height);
    for i in 0..out.data.len() {
        let (m1, m2) = (mu1.data[i], mu2.data[i]);
        let s1 = e11.data[i] - m1 * m1;
        let s2 = e22.data[i] - m2 * m2;
        let s12 = e12.data[i] - m1 * m2;
        let num = (2.0 * m1 * m2 + C1) * (2.0 * s12 + C2);
        let den = (m1 * m1 + m2 * m2 + C1) * (s1 + s2 + C2);
        out.data[i] = num / den;
    }
    out
}

/// numpy's `pairwise_sum` for float32 (blocks of 8, recursion above 128 elements).
fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0f32;
        for &v in a {
            r += v;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

/// numpy's buffer size for reductions that need buffering.
const NUMPY_BUFSIZE: usize = 8192;

/// `arr[m:-m, m:-m].mean()` for a float32 array, following numpy's reduction order. The
/// inset view is not contiguous, so numpy copies it through its 8192-element buffer in
/// chunks of whole rows, sums each chunk pairwise and accumulates the chunk sums in f32,
/// then divides by the count in f32.
pub fn inset_mean_f32(p: &Plane<f32>, inset: usize) -> f32 {
    if p.width <= 2 * inset || p.height <= 2 * inset {
        return f32::NAN;
    }
    let row_len = p.width - 2 * inset;
    let mut flat = Vec::with_capacity(row_len * (p.height - 2 * inset));
    for y in inset..p.height - inset {
        flat.extend_from_slice(&p.row(y)[inset..p.width - inset]);
    }
    let chunk = if row_len <= NUMPY_BUFSIZE {
        (NUMPY_BUFSIZE / row_len) * row_len
    } else {
        NUMPY_BUFSIZE
    };
    let mut total = 0.0f32;
    for c in flat.chunks(chunk) {
        total += pairwise_sum_f32(c);
    }
    total / flat.len() as f32
}

/// Share of inset pixels with `|a - b| > thresh` (numpy bool mean, exact in f64).
pub fn changed_fraction(a: &Plane<f32>, b: &Plane<f32>, inset: usize, thresh: f32) -> f64 {
    if a.width <= 2 * inset || a.height <= 2 * inset {
        return f64::NAN;
    }
    let mut n = 0usize;
    let mut hit = 0usize;
    for y in inset..a.height - inset {
        for x in inset..a.width - inset {
            let d = (a.get(x, y) - b.get(x, y)).abs();
            if d > thresh {
                hit += 1;
            }
            n += 1;
        }
    }
    hit as f64 / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured(w: usize, h: usize) -> Plane<f32> {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                *p.get_mut(x, y) = ((x * 31 + y * 17) % 97) as f32 * 2.3;
            }
        }
        p
    }

    #[test]
    fn identical_images_have_ssim_one() {
        let a = textured(64, 48);
        let m = ssim_map(&a, &a);
        for v in &m.data {
            assert!((v - 1.0).abs() < 1e-5, "{v}");
        }
        assert!((inset_mean_f32(&m, 15) - 1.0).abs() < 1e-5);
        assert_eq!(changed_fraction(&a, &a, 15, 25.0), 0.0);
    }

    #[test]
    fn different_images_score_lower() {
        let a = textured(64, 48);
        let b = Plane::filled(64, 48, 100.0f32);
        let s = inset_mean_f32(&ssim_map(&a, &b), 15);
        assert!(s < 0.5, "{s}");
    }

    #[test]
    fn pairwise_sum_matches_numpy_structure() {
        let v: Vec<f32> = (0..290).map(|i| i as f32 * 0.5).collect();
        let exact: f32 = (0..290).map(|i| i as f32 * 0.5).sum();
        assert_eq!(pairwise_sum_f32(&v), exact);
        assert_eq!(pairwise_sum_f32(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn changed_fraction_counts_threshold_strictly() {
        let a = Plane::filled(4, 4, 0.0f32);
        let mut b = Plane::filled(4, 4, 25.0f32);
        *b.get_mut(1, 1) = 25.5;
        assert_eq!(changed_fraction(&a, &b, 0, 25.0), 1.0 / 16.0);
    }
}
