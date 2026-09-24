//! Gaussian filters matching `cv2.GaussianBlur` with the default `BORDER_REFLECT_101`.
//!
//! * 8-bit input uses OpenCV's bit-exact fixed-point path: kernels quantized to 8 fractional
//!   bits with error diffusion, a horizontal pass kept at 8 fractional bits and a vertical pass
//!   at 16 fractional bits rounded back to 8-bit.
//! * f32 input uses OpenCV's separable float filter, including its operation order and fused
//!   multiply-adds, so results agree to the last bit or within a few ulps.

use crate::error::{MediaError, Result};
use crate::plane::Plane;
use crate::util::reflect101;

/// A Gaussian kernel size: odd and at least 1. Invalid sizes are unrepresentable, so the
/// filters themselves cannot fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSize(usize);

impl KernelSize {
    /// 3 taps.
    pub const K3: Self = Self(3);
    /// 5 taps.
    pub const K5: Self = Self(5);
    /// 7 taps.
    pub const K7: Self = Self(7);

    /// Validates an odd, positive kernel size.
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 || n.is_multiple_of(2) {
            return Err(MediaError::Invalid(format!(
                "Gaussian kernel size must be odd and positive, got {n}"
            )));
        }
        Ok(Self(n))
    }

    /// Number of taps.
    pub fn get(self) -> usize {
        self.0
    }
}

/// `getGaussianKernelBitExact`: normalized Gaussian taps in f64.
pub fn kernel_f64(size: KernelSize, sigma: f64) -> Vec<f64> {
    let n = size.get();
    if sigma <= 0.0 {
        match n {
            1 => return vec![1.0],
            3 => return vec![0.25, 0.5, 0.25],
            5 => return vec![0.0625, 0.25, 0.375, 0.25, 0.0625],
            7 => {
                return vec![
                    0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
                ]
            }
            9 => {
                return [4.0, 13.0, 30.0, 51.0, 60.0, 51.0, 30.0, 13.0, 4.0]
                    .iter()
                    .map(|v| v / 256.0)
                    .collect()
            }
            _ => {}
        }
    }
    let sigma_x = if sigma > 0.0 {
        sigma
    } else {
        (n as f64).mul_add(0.15, 0.35)
    };
    let scale2x = -0.125 / (sigma_x * sigma_x);
    let half = (n - 1) / 2;
    let mut values = Vec::with_capacity(half);
    let mut sum = 0.0f64;
    let mut x = 1 - n as i64;
    for _ in 0..half {
        let t = ((x * x) as f64 * scale2x).exp();
        values.push(t);
        sum += t;
        x += 2;
    }
    sum *= 2.0;
    sum += 1.0;
    if n.is_multiple_of(2) {
        sum += 1.0;
    }
    let mul1 = 1.0 / sum;
    let mut out = vec![0.0; n];
    for (i, v) in values.iter().enumerate() {
        out[i] = v * mul1;
        out[n - 1 - i] = v * mul1;
    }
    out[half] = mul1;
    if n.is_multiple_of(2) {
        out[half + 1] = mul1;
    }
    out
}

/// `getGaussianKernelFixedPoint_ED` with 8 fractional bits (`ufixedpoint16`): taps sum to 256.
pub fn kernel_fixed8(size: KernelSize, sigma: f64) -> Vec<u32> {
    let n = size.get();
    let k = kernel_f64(size, sigma);
    let mut out = vec![0u32; n];
    let half = n / 2;
    let mut err = 0.0f64;
    let mut sum: i64 = 0;
    for i in 0..half {
        let adj = k[i] * 256.0 + err;
        let v = adj.round_ties_even() as i64;
        err = adj - v as f64;
        out[i] = v as u32;
        out[n - 1 - i] = v as u32;
        sum += v;
    }
    out[half] = (256 - 2 * sum) as u32;
    out
}

/// `getGaussianKernel(n, sigma, CV_32F)`.
pub fn kernel_f32(size: KernelSize, sigma: f64) -> Vec<f32> {
    kernel_f64(size, sigma).iter().map(|&v| v as f32).collect()
}

/// 8-bit Gaussian blur through OpenCV's fixed-point path (`GaussianBlurFixedPoint`).
pub fn blur_u8(src: &Plane<u8>, size: KernelSize, sigma: f64) -> Plane<u8> {
    if src.width == 0 || src.height == 0 {
        return src.clone();
    }
    let n = size.get();
    let k = kernel_fixed8(size, sigma);
    let (w, h) = (src.width, src.height);
    let r = (n / 2) as isize;
    // Horizontal pass: 8 fractional bits, at most 255 * 256.
    let mut tmp = vec![0u32; w * h];
    for y in 0..h {
        let row = src.row(y);
        for x in 0..w {
            let mut s = 0u32;
            for (i, &kv) in k.iter().enumerate() {
                let sx = reflect101(x as isize + i as isize - r, w);
                s += kv * u32::from(row[sx]);
            }
            tmp[y * w + x] = s;
        }
    }
    // Vertical pass: 16 fractional bits, rounded to 8-bit.
    let mut out = Plane::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut s = 0u32;
            for (i, &kv) in k.iter().enumerate() {
                let sy = reflect101(y as isize + i as isize - r, h);
                s += kv * tmp[sy * w + x];
            }
            out.data[y * w + x] = ((s + (1 << 15)) >> 16).min(255) as u8;
        }
    }
    out
}

/// Horizontal pass of OpenCV's separable float filter for a symmetric kernel.
fn row_pass(src: &Plane<f32>, k: &[f32]) -> Vec<f32> {
    let (w, h) = (src.width, src.height);
    let n = k.len();
    let r = (n / 2) as isize;
    let c = n / 2;
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let row = src.row(y);
        let at = |x: usize, d: isize| row[reflect101(x as isize + d, w)];
        let dst = &mut out[y * w..(y + 1) * w];
        for (x, o) in dst.iter_mut().enumerate() {
            *o = match n {
                // SymmRowSmallVec_32f, ksize 3: muladd(s0, k0, (s[-1] + s[1]) * k1)
                3 => at(x, 0).mul_add(k[c], (at(x, -1) + at(x, 1)) * k[c + 1]),
                // ksize 5: muladd(s[2] + s[-2], k2, muladd(s0, k0, (s[-1] + s[1]) * k1))
                5 => (at(x, 2) + at(x, -2)).mul_add(
                    k[c + 2],
                    at(x, 0).mul_add(k[c], (at(x, -1) + at(x, 1)) * k[c + 1]),
                ),
                // RowVec_32f: s = s[0]*k[0]; s = muladd(s[i], k[i], s) left to right.
                _ => {
                    let mut s = at(x, -r) * k[0];
                    for (i, &kv) in k.iter().enumerate().skip(1) {
                        s = at(x, i as isize - r).mul_add(kv, s);
                    }
                    s
                }
            };
        }
    }
    out
}

/// Vertical pass of OpenCV's separable float filter for a symmetric kernel.
fn col_pass(tmp: &[f32], w: usize, h: usize, k: &[f32]) -> Plane<f32> {
    let n = k.len();
    let c = n / 2;
    let mut out = Plane::new(w, h);
    for y in 0..h {
        let rowi = |d: isize| reflect101(y as isize + d, h) * w;
        let r0 = rowi(0);
        for x in 0..w {
            let v = if n == 3 {
                // SymmColumnSmallVec_32f: muladd(S0 + S2, k1, muladd(S1, k0, delta))
                let (a, b) = (tmp[rowi(-1) + x], tmp[rowi(1) + x]);
                (a + b).mul_add(k[c + 1], tmp[r0 + x].mul_add(k[c], 0.0))
            } else {
                // SymmColumnVec_32f: s = muladd(S0, k0, delta); s = muladd(S[k] + S[-k], kk, s)
                let mut s = tmp[r0 + x].mul_add(k[c], 0.0);
                for d in 1..=c {
                    let sum = tmp[rowi(d as isize) + x] + tmp[rowi(-(d as isize)) + x];
                    s = sum.mul_add(k[c + d], s);
                }
                s
            };
            out.data[y * w + x] = v;
        }
    }
    out
}

/// f32 Gaussian blur, `cv2.GaussianBlur(src, (n, n), sigma)` on a `CV_32F` image.
pub fn blur_f32(src: &Plane<f32>, size: KernelSize, sigma: f64) -> Plane<f32> {
    if src.width == 0 || src.height == 0 {
        return src.clone();
    }
    let k = kernel_f32(size, sigma);
    let tmp = row_pass(src, &k);
    col_pass(&tmp, src.width, src.height, &k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_kernels_sum_to_256() {
        for (n, s) in [(3, 0.0), (5, 1.2), (7, 1.5), (5, 0.0)] {
            let k = kernel_fixed8(KernelSize::new(n).unwrap(), s);
            assert_eq!(k.iter().sum::<u32>(), 256, "n={n} sigma={s}");
        }
        assert_eq!(kernel_fixed8(KernelSize::K3, 0.0), vec![64, 128, 64]);
    }

    #[test]
    fn sigma_1_2_fixed_kernel_is_symmetric_and_peaked() {
        let k = kernel_fixed8(KernelSize::K5, 1.2);
        assert_eq!(k[0], k[4]);
        assert_eq!(k[1], k[3]);
        assert!(k[2] > k[1] && k[1] > k[0]);
    }

    #[test]
    fn constant_image_is_unchanged() {
        let p = Plane::filled(9, 7, 77u8);
        assert_eq!(blur_u8(&p, KernelSize::K5, 1.2).data, p.data);
        let f = Plane::filled(9, 7, 12.5f32);
        for v in blur_f32(&f, KernelSize::K7, 1.5).data {
            assert!((v - 12.5).abs() < 1e-4);
        }
    }

    #[test]
    fn binomial_3x3_matches_integer_formula() {
        // For [1,2,1] x [1,2,1] / 16 the fixed-point result equals (sum + 8) >> 4.
        let data: Vec<u8> = (0..30u32).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let p = Plane::from_vec(6, 5, data).unwrap();
        let got = blur_u8(&p, KernelSize::K3, 0.0);
        let k = [1u32, 2, 1];
        for y in 0..5 {
            for x in 0..6 {
                let mut s = 0u32;
                for (j, &ky) in k.iter().enumerate() {
                    for (i, &kx) in k.iter().enumerate() {
                        let sx = reflect101(x as isize + i as isize - 1, 6);
                        let sy = reflect101(y as isize + j as isize - 1, 5);
                        s += kx * ky * u32::from(p.get(sx, sy));
                    }
                }
                assert_eq!(u32::from(got.get(x, y)), (s + 8) >> 4);
            }
        }
    }

    #[test]
    fn impulse_response_matches_kernel_outer_product() {
        let mut f = Plane::new(11, 11);
        *f.get_mut(5, 5) = 1.0f32;
        let out = blur_f32(&f, KernelSize::K5, 1.2);
        let k = kernel_f32(KernelSize::K5, 1.2);
        for dy in 0..5 {
            for dx in 0..5 {
                let want = k[dx] * k[dy];
                assert!((out.get(3 + dx, 3 + dy) - want).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn kernel_size_rejects_even_and_zero() {
        assert!(KernelSize::new(0).is_err());
        assert!(KernelSize::new(4).is_err());
        assert_eq!(KernelSize::new(9).unwrap().get(), 9);
    }

    #[test]
    fn empty_planes_pass_through() {
        let p: Plane<u8> = Plane::new(0, 3);
        assert_eq!(blur_u8(&p, KernelSize::K5, 1.2), p);
        let f: Plane<f32> = Plane::new(4, 0);
        assert_eq!(blur_f32(&f, KernelSize::K7, 1.5), f);
    }
}
