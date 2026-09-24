//! Border handling and numeric helpers shared by the filters.

/// OpenCV `BORDER_REFLECT_101` (`gfedcb|abcdefgh|gfedcba`) index mapping.
#[inline]
pub fn reflect101(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * n - 2 - i;
        } else {
            return i as usize;
        }
    }
}

/// OpenCV `BORDER_REPLICATE` (`aaaaaa|abcdefgh|hhhhhhh`) index mapping.
#[inline]
pub fn replicate(i: isize, n: usize) -> usize {
    i.clamp(0, n as isize - 1) as usize
}

/// `saturate_cast<uchar>(float)`: round half to even, then clamp to 0..=255.
#[inline]
pub fn sat_u8_f32(x: f32) -> u8 {
    x.round_ties_even().clamp(0.0, 255.0) as u8
}

/// NEON `vaddvq_f32` style horizontal sum of four lanes.
#[inline]
pub fn reduce4(v: [f32; 4]) -> f32 {
    (v[0] + v[1]) + (v[2] + v[3])
}

/// OpenCV `dotProd_32f` on one contiguous run: blocks of 8192 elements accumulated in four
/// f32 lanes (four independent accumulators over 16-element strides, FMA), reduced into f64,
/// with the tail that does not fill a lane group accumulated in f64.
pub fn dot_prod_32f(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    let len0 = len & !3;
    let mut r = 0.0f64;
    let mut i = 0;
    while i < len0 {
        let block = (len0 - i).min(1 << 13);
        let (pa, pb) = (&a[i..i + block], &b[i..i + block]);
        let mut s = [[0.0f32; 4]; 4];
        let mut j = 0;
        while j + 16 <= block {
            for (u, acc) in s.iter_mut().enumerate() {
                for (l, lane) in acc.iter_mut().enumerate() {
                    let k = j + 4 * u + l;
                    *lane = pa[k].mul_add(pb[k], *lane);
                }
            }
            j += 16;
        }
        let mut v = [0.0f32; 4];
        for l in 0..4 {
            v[l] = s[0][l] + ((s[1][l] + s[2][l]) + s[3][l]);
        }
        while j + 4 <= block {
            for (l, lane) in v.iter_mut().enumerate() {
                *lane = pa[j + l].mul_add(pb[j + l], *lane);
            }
            j += 4;
        }
        r += f64::from(reduce4(v));
        i += block;
    }
    let mut t = 0.0f64;
    while i < len {
        t += f64::from(a[i]) * f64::from(b[i]);
        i += 1;
    }
    r + t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect101_matches_opencv() {
        let got: Vec<usize> = (-3..8).map(|i| reflect101(i, 5)).collect();
        assert_eq!(got, vec![3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]);
    }

    #[test]
    fn replicate_clamps() {
        assert_eq!(replicate(-4, 5), 0);
        assert_eq!(replicate(9, 5), 4);
        assert_eq!(replicate(2, 5), 2);
    }

    #[test]
    fn sat_u8_rounds_half_to_even() {
        assert_eq!(sat_u8_f32(2.5), 2);
        assert_eq!(sat_u8_f32(3.5), 4);
        assert_eq!(sat_u8_f32(-1.0), 0);
        assert_eq!(sat_u8_f32(300.0), 255);
    }

    #[test]
    fn dot_prod_matches_exact_sum_for_integers() {
        let a: Vec<f32> = (0..20_001).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..20_001).map(|i| (i % 5) as f32).collect();
        let exact: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x * y)).sum();
        assert_eq!(dot_prod_32f(&a, &b), exact);
    }
}
