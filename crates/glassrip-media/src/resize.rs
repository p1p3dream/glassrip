//! Integer-factor downscaling as OpenCV performs it for the prototype's calls.

use crate::plane::{Bgr, Plane};
use crate::util::sat_u8_f32;

/// `cv2.resize(..., interpolation=INTER_AREA)` for an exact integer factor on 8-bit data:
/// OpenCV's `resizeAreaFast` sums each `factor x factor` block as an integer, multiplies by
/// the f32 constant `1/area` and rounds half to even.
fn area_fast(src: &[u8], width: usize, height: usize, cn: usize, factor: usize) -> Vec<u8> {
    let dw = width / factor;
    let dh = height / factor;
    let scale = 1.0f32 / (factor * factor) as f32;
    let mut out = vec![0u8; dw * dh * cn];
    for dy in 0..dh {
        for dx in 0..dw {
            for c in 0..cn {
                let mut sum: i32 = 0;
                for sy in dy * factor..(dy + 1) * factor {
                    let row = &src[sy * width * cn..(sy + 1) * width * cn];
                    for sx in dx * factor..(dx + 1) * factor {
                        sum += i32::from(row[sx * cn + c]);
                    }
                }
                out[(dy * dw + dx) * cn + c] = sat_u8_f32(sum as f32 * scale);
            }
        }
    }
    out
}

/// Single-channel `INTER_AREA` downscale by an exact integer factor.
pub fn area_gray(src: &Plane<u8>, factor: usize) -> Plane<u8> {
    Plane {
        width: src.width / factor,
        height: src.height / factor,
        data: area_fast(&src.data, src.width, src.height, 1, factor),
    }
}

/// Three-channel `INTER_AREA` downscale by an exact integer factor.
pub fn area_bgr(src: &Bgr, factor: usize) -> Bgr {
    Bgr {
        width: src.width / factor,
        height: src.height / factor,
        data: area_fast(&src.data, src.width, src.height, 3, factor),
    }
}

/// `cv2.resize(src, (w/2, h/2))` with the default `INTER_LINEAR` on 8-bit gray. At exactly
/// half size OpenCV (and the KleidiCV HAL it dispatches to on arm64) computes the 2x2 mean
/// `(a + b + c + d + 2) >> 2`.
pub fn half_linear_gray(src: &Plane<u8>) -> Plane<u8> {
    let (dw, dh) = (src.width / 2, src.height / 2);
    let mut out = Plane::new(dw, dh);
    for y in 0..dh {
        let r0 = src.row(2 * y);
        let r1 = src.row(2 * y + 1);
        for x in 0..dw {
            let s = u32::from(r0[2 * x])
                + u32::from(r0[2 * x + 1])
                + u32::from(r1[2 * x])
                + u32::from(r1[2 * x + 1]);
            out.data[y * dw + x] = ((s + 2) >> 2) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_rounds_half_to_even_via_f32_scale() {
        // 6x6 block sums: 18 -> 0.5 -> 0; 54 -> 1.5 -> 2; 90 -> 2.5 -> 2.
        for (ones, want) in [(18, 0u8), (54, 2), (90, 2), (35, 1)] {
            let mut data = vec![0u8; 36];
            let mut left = ones;
            for v in data.iter_mut() {
                let take = left.min(255);
                *v = take as u8;
                left -= take;
                if left == 0 {
                    break;
                }
            }
            let p = Plane::from_vec(6, 6, data).unwrap();
            assert_eq!(area_gray(&p, 6).data, vec![want], "sum {ones}");
        }
    }

    #[test]
    fn area_bgr_keeps_channels_separate() {
        let mut data = Vec::new();
        for _ in 0..9 {
            data.extend_from_slice(&[10u8, 20, 30]);
        }
        let img = Bgr {
            width: 3,
            height: 3,
            data,
        };
        assert_eq!(area_bgr(&img, 3).data, vec![10, 20, 30]);
    }

    #[test]
    fn half_linear_rounds_up_at_half() {
        let p = Plane::from_vec(2, 2, vec![1u8, 2, 2, 1]).unwrap();
        // (6 + 2) >> 2 = 2 (1.5 rounds up).
        assert_eq!(half_linear_gray(&p).data, vec![2]);
        let q = Plane::from_vec(2, 2, vec![1u8, 1, 1, 2]).unwrap();
        assert_eq!(half_linear_gray(&q).data, vec![1]);
    }
}
