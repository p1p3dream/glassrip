//! `cv2.warpAffine(..., flags=INTER_* | WARP_INVERSE_MAP)` with a zero constant border.
//!
//! Mirrors OpenCV 5's vectorized warp kernels: the matrix is rounded to f32, source
//! coordinates are `fma(M0, x, y*M1 + M2)` in f32, bilinear weights come from the fractional
//! part without quantization, and interpolation uses fused multiply-adds.

use rayon::prelude::*;

use crate::plane::Plane;

/// A 2x3 affine matrix `[m0 m1 m2; m3 m4 m5]` stored as f32, like the prototype's `wm`.
pub type Affine = [f32; 6];

/// Identity warp.
pub const IDENTITY: Affine = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

#[inline]
fn row_offsets(m: &Affine, y: usize) -> (f32, f32) {
    let yf = y as f32;
    (yf.mul_add(m[1], m[2]), yf.mul_add(m[4], m[5]))
}

/// Bilinear inverse-map warp of an f32 image into a `dw x dh` output.
pub fn affine_linear_f32(src: &Plane<f32>, m: &Affine, dw: usize, dh: usize) -> Plane<f32> {
    let (sw, sh) = (src.width as i64, src.height as i64);
    let mut out = Plane::new(dw, dh);
    out.data
        .par_chunks_mut(dw.max(1))
        .enumerate()
        .for_each(|(y, dst)| {
            let (mx, my) = row_offsets(m, y);
            for (x, o) in dst.iter_mut().enumerate() {
                let xf = x as f32;
                let sx = m[0].mul_add(xf, mx);
                let sy = m[3].mul_add(xf, my);
                let fx = sx.floor();
                let fy = sy.floor();
                let (ix, iy) = (fx as i64, fy as i64);
                let ax = sx - fx;
                let ay = sy - fy;
                let fetch = |dx: i64, dy: i64| -> f32 {
                    let (px, py) = (ix + dx, iy + dy);
                    if px >= 0 && px < sw && py >= 0 && py < sh {
                        src.data[py as usize * src.width + px as usize]
                    } else {
                        0.0
                    }
                };
                let (p00, p01, p10, p11) = (fetch(0, 0), fetch(1, 0), fetch(0, 1), fetch(1, 1));
                let v0 = ax.mul_add(p01 - p00, p00);
                let v1 = ax.mul_add(p11 - p10, p10);
                *o = ay.mul_add(v1 - v0, v0);
            }
        });
    out
}

/// Nearest-neighbour inverse-map warp of an 8-bit image (coordinates rounded half to even).
pub fn affine_nearest_u8(src: &Plane<u8>, m: &Affine, dw: usize, dh: usize) -> Plane<u8> {
    let (sw, sh) = (src.width as i64, src.height as i64);
    let mut out = Plane::new(dw, dh);
    out.data
        .par_chunks_mut(dw.max(1))
        .enumerate()
        .for_each(|(y, dst)| {
            let (mx, my) = row_offsets(m, y);
            for (x, o) in dst.iter_mut().enumerate() {
                let xf = x as f32;
                let ix = m[0].mul_add(xf, mx).round_ties_even() as i64;
                let iy = m[3].mul_add(xf, my).round_ties_even() as i64;
                *o = if ix >= 0 && ix < sw && iy >= 0 && iy < sh {
                    src.data[iy as usize * src.width + ix as usize]
                } else {
                    0
                };
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_a_copy() {
        let p = Plane::from_vec(4, 3, (0..12).map(|v| v as f32).collect()).unwrap();
        assert_eq!(affine_linear_f32(&p, &IDENTITY, 4, 3), p);
        let q = Plane::from_vec(4, 3, (0..12u8).collect()).unwrap();
        assert_eq!(affine_nearest_u8(&q, &IDENTITY, 4, 3), q);
    }

    #[test]
    fn half_pixel_shift_averages_and_zero_border() {
        let p = Plane::from_vec(3, 1, vec![0.0f32, 10.0, 20.0]).unwrap();
        let m = [1.0, 0.0, 0.5, 0.0, 1.0, 0.0];
        let out = affine_linear_f32(&p, &m, 3, 1);
        assert_eq!(out.data, vec![5.0, 15.0, 10.0]);
    }

    #[test]
    fn nearest_rounds_half_to_even() {
        let q = Plane::from_vec(4, 1, vec![1u8, 2, 3, 4]).unwrap();
        let m = [1.0, 0.0, 0.5, 0.0, 1.0, 0.0];
        // x + 0.5 rounds to even: 0.5->0, 1.5->2, 2.5->2, 3.5->4 (outside).
        assert_eq!(affine_nearest_u8(&q, &m, 4, 1).data, vec![1, 3, 3, 0]);
    }
}
