//! Binary masks, connected components, Zhang-Suen thinning, and skeleton walks.

use std::collections::VecDeque;

/// A binary image stored row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Row-major values.
    pub data: Vec<bool>,
}

const N8: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

impl Mask {
    /// All-false mask.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![false; width * height],
        }
    }

    /// Value at `(x, y)`; false outside the image.
    #[inline]
    pub fn get(&self, x: isize, y: isize) -> bool {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return false;
        }
        self.data[y as usize * self.width + x as usize]
    }

    /// Set `(x, y)` when inside the image.
    #[inline]
    pub fn set(&mut self, x: isize, y: isize, v: bool) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.data[y as usize * self.width + x as usize] = v;
        }
    }

    /// Number of set pixels.
    pub fn count(&self) -> usize {
        self.data.iter().filter(|v| **v).count()
    }

    /// Clear an axis-aligned rectangle (inclusive-exclusive, clamped).
    pub fn clear_rect(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) {
        let (w, h) = (self.width as f64, self.height as f64);
        let xa = x1.floor().clamp(0.0, w) as usize;
        let xb = x2.ceil().clamp(0.0, w) as usize;
        let ya = y1.floor().clamp(0.0, h) as usize;
        let yb = y2.ceil().clamp(0.0, h) as usize;
        for y in ya..yb {
            for x in xa..xb {
                self.data[y * self.width + x] = false;
            }
        }
    }

    /// Set pixels of the union.
    pub fn or(&self, other: &Self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| *a || *b)
                .collect(),
        }
    }

    /// Set pixels of the intersection.
    pub fn and(&self, other: &Self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| *a && *b)
                .collect(),
        }
    }

    /// Square dilation with side `2r + 1`.
    pub fn dilate(&self, r: usize) -> Self {
        self.morph(r, true)
    }

    /// Square erosion with side `2r + 1` (pixels outside the image count as set).
    pub fn erode(&self, r: usize) -> Self {
        self.morph(r, false)
    }

    fn morph(&self, r: usize, dilate: bool) -> Self {
        if r == 0 {
            return self.clone();
        }
        let (w, h) = (self.width, self.height);
        let pass = |src: &[bool], horizontal: bool| -> Vec<bool> {
            let mut out = vec![false; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut acc = !dilate;
                    for d in -(r as isize)..=(r as isize) {
                        let (xx, yy) = if horizontal {
                            (x as isize + d, y as isize)
                        } else {
                            (x as isize, y as isize + d)
                        };
                        let v = if xx < 0 || yy < 0 || xx as usize >= w || yy as usize >= h {
                            !dilate
                        } else {
                            src[yy as usize * w + xx as usize]
                        };
                        if dilate {
                            acc |= v;
                        } else {
                            acc &= v;
                        }
                    }
                    out[y * w + x] = acc;
                }
            }
            out
        };
        let tmp = pass(&self.data, true);
        Self {
            width: w,
            height: h,
            data: pass(&tmp, false),
        }
    }

    /// Morphological closing (dilate then erode) with radius `r`; bridges gaps of up
    /// to about `2r` pixels, which joins dashed strokes.
    pub fn close(&self, r: usize) -> Self {
        self.dilate(r).erode(r)
    }

    /// The 8-connected components that contain any of `seeds`, merged into one mask.
    pub fn components_from(&self, seeds: &[(usize, usize)]) -> Self {
        let mut out = Self::new(self.width, self.height);
        let mut queue = VecDeque::new();
        for &(x, y) in seeds {
            let (xi, yi) = (x as isize, y as isize);
            if self.get(xi, yi) && !out.get(xi, yi) {
                out.set(xi, yi, true);
                queue.push_back((xi, yi));
            }
        }
        while let Some((x, y)) = queue.pop_front() {
            for (dx, dy) in N8 {
                let (nx, ny) = (x + dx, y + dy);
                if self.get(nx, ny) && !out.get(nx, ny) {
                    out.set(nx, ny, true);
                    queue.push_back((nx, ny));
                }
            }
        }
        out
    }

    /// Set-pixel coordinates.
    pub fn points(&self) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if self.data[y * self.width + x] {
                    v.push((x, y));
                }
            }
        }
        v
    }

    /// Number of set 8-neighbors of `(x, y)`.
    pub fn neighbors(&self, x: usize, y: usize) -> usize {
        N8.iter()
            .filter(|(dx, dy)| self.get(x as isize + dx, y as isize + dy))
            .count()
    }
}

/// Zhang-Suen thinning (Zhang and Suen 1984), iterated until stable.
pub fn zhang_suen(mask: &Mask) -> Mask {
    let mut m = mask.clone();
    let (w, h) = (m.width as isize, m.height as isize);
    loop {
        let mut changed = false;
        for step in 0..2 {
            let mut remove = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    if !m.get(x, y) {
                        continue;
                    }
                    // P2..P9 clockwise from north.
                    let p = [
                        m.get(x, y - 1),
                        m.get(x + 1, y - 1),
                        m.get(x + 1, y),
                        m.get(x + 1, y + 1),
                        m.get(x, y + 1),
                        m.get(x - 1, y + 1),
                        m.get(x - 1, y),
                        m.get(x - 1, y - 1),
                    ];
                    let b = p.iter().filter(|v| **v).count();
                    if !(2..=6).contains(&b) {
                        continue;
                    }
                    let a = (0..8).filter(|&i| !p[i] && p[(i + 1) % 8]).count();
                    if a != 1 {
                        continue;
                    }
                    let (p2, p4, p6, p8) = (p[0], p[2], p[4], p[6]);
                    let ok = if step == 0 {
                        !(p4 && p6 && (p2 || p8))
                    } else {
                        !(p2 && p8 && (p4 || p6))
                    };
                    if ok {
                        remove.push((x, y));
                    }
                }
            }
            if !remove.is_empty() {
                changed = true;
            }
            for (x, y) in remove {
                m.set(x, y, false);
            }
        }
        if !changed {
            return m;
        }
    }
}

/// Skeleton pixels with exactly one 8-neighbor.
pub fn endpoints(skel: &Mask) -> Vec<(usize, usize)> {
    skel.points()
        .into_iter()
        .filter(|&(x, y)| skel.neighbors(x, y) == 1)
        .collect()
}

/// Geodesic (8-connected step count) distance over the skeleton from `sources`;
/// `None` for unreachable pixels.
pub fn geodesic(skel: &Mask, sources: &[(usize, usize)]) -> Vec<Option<u32>> {
    let mut dist = vec![None; skel.width * skel.height];
    let mut queue = VecDeque::new();
    for &(x, y) in sources {
        if skel.get(x as isize, y as isize) && dist[y * skel.width + x].is_none() {
            dist[y * skel.width + x] = Some(0);
            queue.push_back((x as isize, y as isize));
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        let d = dist[y as usize * skel.width + x as usize].unwrap_or(0);
        for (dx, dy) in N8 {
            let (nx, ny) = (x + dx, y + dy);
            if skel.get(nx, ny) {
                let i = ny as usize * skel.width + nx as usize;
                if dist[i].is_none() {
                    dist[i] = Some(d + 1);
                    queue.push_back((nx, ny));
                }
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(m: &mut Mask, x1: isize, y1: isize, x2: isize, y2: isize) {
        for y in y1..y2 {
            for x in x1..x2 {
                m.set(x, y, true);
            }
        }
    }

    #[test]
    fn thinning_a_thick_bar_leaves_a_one_pixel_line_with_two_ends() {
        let mut m = Mask::new(60, 20);
        rect(&mut m, 5, 7, 55, 12);
        let s = zhang_suen(&m);
        assert!(s.count() >= 40 && s.count() <= 52, "{}", s.count());
        let ends = endpoints(&s);
        assert_eq!(ends.len(), 2, "{ends:?}");
        for (x, y) in s.points() {
            assert!(s.neighbors(x, y) <= 2, "branch at {x},{y}");
        }
    }

    #[test]
    fn closing_bridges_dash_gaps() {
        let mut m = Mask::new(80, 10);
        for k in 0..6 {
            rect(&mut m, 5 + k * 12, 4, 13 + k * 12, 6);
        }
        let open = m.components_from(&[(6, 5)]);
        assert!(open.count() < 20);
        let closed = m.close(3).components_from(&[(6, 5)]);
        assert!(closed.count() > 100);
    }

    #[test]
    fn geodesic_distance_follows_an_elbow() {
        let mut m = Mask::new(30, 30);
        rect(&mut m, 2, 2, 3, 20);
        rect(&mut m, 2, 19, 25, 20);
        let d = geodesic(&m, &[(2, 2)]);
        assert_eq!(d[19 * 30 + 24], Some(38));
        assert_eq!(d[0], None);
    }
}
