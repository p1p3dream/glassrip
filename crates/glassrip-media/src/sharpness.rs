//! Frame sharpness: population variance of the 3x3 Laplacian.

use rayon::prelude::*;

use crate::plane::Plane;
use crate::util::reflect101;

/// numpy's `pairwise_sum` for float64 over a contiguous array. The recursion splits at the
/// same points as numpy, so the result is identical; large halves run in parallel.
fn pairwise_sum_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0;
        for &v in a {
            r += v;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
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
        let (lo, hi) = a.split_at(n2);
        if n > 1 << 16 {
            let (x, y) = rayon::join(|| pairwise_sum_f64(lo), || pairwise_sum_f64(hi));
            x + y
        } else {
            pairwise_sum_f64(lo) + pairwise_sum_f64(hi)
        }
    }
}

/// 3x3 Laplacian `[0,1,0; 1,-4,1; 0,1,0]` with `BORDER_REFLECT_101`, as f64.
pub fn laplacian(g: &Plane<u8>) -> Vec<f64> {
    let (w, h) = (g.width, g.height);
    let mut out = vec![0.0f64; w * h];
    out.par_chunks_mut(w.max(1))
        .enumerate()
        .for_each(|(y, dst)| {
            let up = g.row(reflect101(y as isize - 1, h));
            let mid = g.row(y);
            let dn = g.row(reflect101(y as isize + 1, h));
            for (x, o) in dst.iter_mut().enumerate() {
                let l = i32::from(mid[reflect101(x as isize - 1, w)])
                    + i32::from(mid[reflect101(x as isize + 1, w)])
                    + i32::from(up[x])
                    + i32::from(dn[x])
                    - 4 * i32::from(mid[x]);
                *o = f64::from(l);
            }
        });
    out
}

/// `cv2.Laplacian(gray, cv2.CV_64F).var()`: population variance computed the way numpy
/// does it (pairwise mean, squared deviations, pairwise sum, divide by n).
pub fn laplacian_variance(g: &Plane<u8>) -> f64 {
    let lap = laplacian(g);
    if lap.is_empty() {
        return 0.0;
    }
    let n = lap.len() as f64;
    let mean = pairwise_sum_f64(&lap) / n;
    let sq: Vec<f64> = lap
        .par_iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .collect();
    pairwise_sum_f64(&sq) / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_image_has_zero_variance() {
        assert_eq!(laplacian_variance(&Plane::filled(8, 6, 90u8)), 0.0);
    }

    #[test]
    fn single_impulse_variance() {
        // One pixel of 10 in the interior: Laplacian -40 there and +10 at 4 neighbours.
        let mut p = Plane::filled(7, 7, 0u8);
        *p.get_mut(3, 3) = 10;
        let n = 49.0;
        let mean = 0.0 / n;
        let want = (1600.0 + 4.0 * 100.0) / n - mean * mean;
        assert!((laplacian_variance(&p) - want).abs() < 1e-12);
    }

    #[test]
    fn pairwise_sum_is_exact_on_integers() {
        let v: Vec<f64> = (0..100_000).map(|i| f64::from(i % 17) - 8.0).collect();
        let exact: i64 = (0..100_000).map(|i| i64::from(i % 17) - 8).sum();
        assert_eq!(pairwise_sum_f64(&v), exact as f64);
    }
}
