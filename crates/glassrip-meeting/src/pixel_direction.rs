//! Pixel check for edge direction (spec 6.11, edge direction step 1).
//!
//! Per keyframe, a stroke mask is built once: dark pixels from an adaptive mean
//! threshold (the ink metric's rule), with every node box and every text box masked
//! out so that box outlines and glyphs cannot join connectors. Per edge:
//!
//! 1. **Seed.** Around the edge label (a ring outside the label box, since the label is
//!    masked and splits the line in two) or, without a label, around the midpoint of
//!    the segment joining the two boxes' facing borders. The darkest candidate sets the
//!    stroke color; seeds are candidates close to that color.
//! 2. **Component.** Stroke pixels near the stroke color, closed morphologically (a
//!    larger radius for dashed lines so dashes join), then the 8-connected components
//!    that contain a seed.
//! 3. **Skeleton.** Zhang-Suen thinning of the component, so curved and elbow
//!    connectors are followed; skeleton endpoints are the candidate termini.
//! 4. **Termini.** For each of the two nodes, the endpoint closest to its box (within a
//!    tolerance; ties go to the endpoint geodesically farthest from the seed). A
//!    terminus inside the tolerance identifies the node.
//! 5. **Arrowhead.** On the unthinned component within `arrow_radius_px` of the
//!    terminus: the area relative to a plain line of the measured stroke width, and
//!    the spread perpendicular to the local skeleton axis. A filled triangle has both
//!    well above a line's.

use glassrip_media::ink::{adaptive_threshold_mean_inv, bgr_to_gray};
use glassrip_media::{Bgr, Plane};
use glassrip_vision::board::EdgeStyle;
use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::direction::EndVerdict;
use crate::skeleton::{endpoints, geodesic, zhang_suen, Mask};

/// Pixel check parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PixelCheckParams {
    /// Adaptive threshold block size (odd).
    pub threshold_block: usize,
    /// Adaptive threshold offset: a pixel is dark when it is this much below the mean.
    pub threshold_c: i32,
    /// Margin added around node boxes before masking, in pixels.
    pub node_margin_px: f64,
    /// Margin added around text boxes before masking, in pixels.
    pub text_margin_px: f64,
    /// Seed search radius around the label ring or the midpoint, in pixels.
    pub seed_radius_px: f64,
    /// Maximum gray-level distance from the stroke color for seed and component pixels.
    pub color_tolerance: u8,
    /// Closing radius for solid connectors.
    pub solid_close_px: usize,
    /// Closing radius for dashed connectors.
    pub dashed_close_px: usize,
    /// A terminus within this distance of a node box identifies the node.
    pub terminus_tolerance_px: f64,
    /// Radius of the arrowhead test around a terminus.
    pub arrow_radius_px: f64,
    /// Minimum area relative to a plain line of the same skeleton length.
    pub min_arrow_area_ratio: f64,
    /// Minimum perpendicular spread relative to half the stroke width.
    pub min_arrow_spread_ratio: f64,
    /// Minimum perpendicular spread in pixels.
    pub min_arrow_spread_px: f64,
    /// Components larger than this share of the canvas are rejected (not a connector).
    pub max_component_share: f64,
}

impl Default for PixelCheckParams {
    fn default() -> Self {
        Self {
            threshold_block: 31,
            threshold_c: 12,
            node_margin_px: 4.0,
            text_margin_px: 2.0,
            seed_radius_px: 24.0,
            color_tolerance: 70,
            solid_close_px: 1,
            dashed_close_px: 5,
            terminus_tolerance_px: 14.0,
            arrow_radius_px: 15.0,
            min_arrow_area_ratio: 1.6,
            min_arrow_spread_ratio: 2.2,
            min_arrow_spread_px: 2.5,
            max_component_share: 0.05,
        }
    }
}

/// Outcome of tracing one connector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PixelStatus {
    /// Both termini found at their nodes.
    Traced,
    /// Only one terminus reached its node.
    OneEnd,
    /// No stroke pixel near the seed.
    NoSeed,
    /// The component reached neither node.
    NoTermini,
    /// The seeded component was implausibly large (fill, texture, or noise).
    ComponentTooLarge,
}

/// Arrowhead test at one terminus.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EndEvidence {
    /// Terminus x in canvas pixels.
    pub x: f64,
    /// Terminus y in canvas pixels.
    pub y: f64,
    /// Distance from the terminus to the node box (0 inside).
    pub node_distance_px: f64,
    /// Component area near the terminus over the area of a plain line.
    pub area_ratio: f64,
    /// Largest perpendicular distance from the local axis, in pixels.
    pub spread_px: f64,
    /// Filled triangular blob found.
    pub arrow: bool,
}

/// Pixel check result for one edge in one keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PixelEvidence {
    /// Tracing outcome.
    pub status: PixelStatus,
    /// Terminus at the `src` node.
    pub src_end: Option<EndEvidence>,
    /// Terminus at the `dst` node.
    pub dst_end: Option<EndEvidence>,
    /// Measured stroke width in pixels.
    pub stroke_px: f64,
    /// Verdict relative to `src -> dst` as read.
    pub verdict: EndVerdict,
}

impl PixelEvidence {
    fn failed(status: PixelStatus) -> Self {
        Self {
            status,
            src_end: None,
            dst_end: None,
            stroke_px: 0.0,
            verdict: EndVerdict::Unknown,
        }
    }
}

/// One edge to check, in canvas pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeQuery {
    /// Box of the tail node as read.
    pub src: BBox,
    /// Box of the head node as read.
    pub dst: BBox,
    /// Box of the edge label text, when known.
    pub label: Option<BBox>,
    /// Line style.
    pub style: EdgeStyle,
}

/// Distance from a point to a box (0 inside).
pub fn dist_to_bbox(p: (f64, f64), b: &BBox) -> f64 {
    let dx = (b.x1 - p.0).max(0.0).max(p.0 - b.x2);
    let dy = (b.y1 - p.1).max(0.0).max(p.1 - b.y2);
    (dx * dx + dy * dy).sqrt()
}

fn center(b: &BBox) -> (f64, f64) {
    ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
}

/// Where the ray from the box center toward `toward` leaves the box.
pub fn exit_point(b: &BBox, toward: (f64, f64)) -> (f64, f64) {
    let c = center(b);
    let (dx, dy) = (toward.0 - c.0, toward.1 - c.1);
    let hw = b.width() / 2.0;
    let hh = b.height() / 2.0;
    let tx = if dx.abs() > 1e-9 {
        hw / dx.abs()
    } else {
        f64::INFINITY
    };
    let ty = if dy.abs() > 1e-9 {
        hh / dy.abs()
    } else {
        f64::INFINITY
    };
    let t = tx.min(ty).min(1.0);
    (c.0 + dx * t, c.1 + dy * t)
}

/// The facing-border midpoint used as the seed when there is no label.
pub fn facing_midpoint(a: &BBox, b: &BBox) -> (f64, f64) {
    let pa = exit_point(a, center(b));
    let pb = exit_point(b, center(a));
    ((pa.0 + pb.0) / 2.0, (pa.1 + pb.1) / 2.0)
}

/// Per-keyframe state shared by all edges of the keyframe.
#[derive(Debug, Clone)]
pub struct PreparedCanvas {
    gray: Plane<u8>,
    /// Dark pixels with node and text boxes removed.
    stroke: Mask,
    params: PixelCheckParams,
}

impl PreparedCanvas {
    /// Build the masked stroke mask. `nodes` and `texts` are every node box and every
    /// text box (labels, stickies, owner tags, other text) of the keyframe.
    pub fn new(image: &Bgr, nodes: &[BBox], texts: &[BBox], params: &PixelCheckParams) -> Self {
        let gray = bgr_to_gray(image);
        let block = params.threshold_block.max(3) | 1;
        let dark = adaptive_threshold_mean_inv(&gray, block, params.threshold_c);
        let mut stroke = Mask {
            width: dark.width,
            height: dark.height,
            data: dark.data.iter().map(|v| *v > 0).collect(),
        };
        let (m, t) = (params.node_margin_px, params.text_margin_px);
        for b in nodes {
            stroke.clear_rect(b.x1 - m, b.y1 - m, b.x2 + m, b.y2 + m);
        }
        for b in texts {
            stroke.clear_rect(b.x1 - t, b.y1 - t, b.x2 + t, b.y2 + t);
        }
        Self {
            gray,
            stroke,
            params: params.clone(),
        }
    }

    /// Canvas width.
    pub fn width(&self) -> usize {
        self.gray.width
    }

    /// Canvas height.
    pub fn height(&self) -> usize {
        self.gray.height
    }

    fn seed_candidates(&self, q: &EdgeQuery) -> Vec<(usize, usize)> {
        let r = self.params.seed_radius_px;
        let (w, h) = (self.gray.width as f64, self.gray.height as f64);
        let region = match q.label {
            Some(l) => BBox::new(l.x1 - r, l.y1 - r, l.x2 + r, l.y2 + r),
            None => {
                let m = facing_midpoint(&q.src, &q.dst);
                BBox::new(m.0 - r, m.1 - r, m.0 + r, m.1 + r)
            }
        }
        .clamped(w, h);
        let mut out = Vec::new();
        for y in region.y1 as usize..region.y2 as usize {
            for x in region.x1 as usize..region.x2 as usize {
                if !self.stroke.get(x as isize, y as isize) {
                    continue;
                }
                if q.label.is_none() {
                    let m = facing_midpoint(&q.src, &q.dst);
                    let d = ((x as f64 - m.0).powi(2) + (y as f64 - m.1).powi(2)).sqrt();
                    if d > r {
                        continue;
                    }
                }
                out.push((x, y));
            }
        }
        out
    }

    fn colored_mask(&self, stroke_level: u8) -> Mask {
        let tol = self.params.color_tolerance;
        let mut colored = self.stroke.clone();
        for (i, v) in colored.data.iter_mut().enumerate() {
            if *v && self.gray.data[i].abs_diff(stroke_level) > tol {
                *v = false;
            }
        }
        colored
    }

    fn close_radius(&self, style: EdgeStyle) -> usize {
        match style {
            EdgeStyle::Solid => self.params.solid_close_px,
            EdgeStyle::Dashed => self.params.dashed_close_px,
        }
    }

    fn ring(&self, b: &BBox) -> Vec<(usize, usize)> {
        let p = &self.params;
        let (lo, hi) = (p.node_margin_px, p.node_margin_px + p.terminus_tolerance_px);
        let area = BBox::new(b.x1 - hi, b.y1 - hi, b.x2 + hi, b.y2 + hi)
            .clamped(self.gray.width as f64, self.gray.height as f64);
        let mut out = Vec::new();
        for y in area.y1 as usize..area.y2 as usize {
            for x in area.x1 as usize..area.x2 as usize {
                let d = dist_to_bbox((x as f64, y as f64), b);
                if d > lo && d <= hi && self.stroke.get(x as isize, y as isize) {
                    out.push((x, y));
                }
            }
        }
        out
    }

    /// Run the pixel check for one edge.
    pub fn check(&self, q: &EdgeQuery) -> PixelEvidence {
        let cands = self.seed_candidates(q);
        let primary = match cands.iter().map(|&(x, y)| self.gray.get(x, y)).min() {
            Some(level) => {
                let tol = self.params.color_tolerance;
                let seeds: Vec<(usize, usize)> = cands
                    .into_iter()
                    .filter(|&(x, y)| self.gray.get(x, y).abs_diff(level) <= tol)
                    .collect();
                let colored = self.colored_mask(level);
                let comp = colored
                    .close(self.close_radius(q.style))
                    .components_from(&seeds);
                self.trace(q, &comp, &colored, &seeds)
            }
            None => PixelEvidence::failed(PixelStatus::NoSeed),
        };
        if primary.status == PixelStatus::Traced || q.label.is_some() {
            return primary;
        }
        // Unlabelled connectors (elbows in particular) may miss the midpoint: flood
        // single components from the ring around the src box and keep the first one
        // that also reaches the ring around the dst box.
        let src_ring = self.ring(&q.src);
        let dst_ring = self.ring(&q.dst);
        let Some(level) = src_ring
            .iter()
            .chain(&dst_ring)
            .map(|&(x, y)| self.gray.get(x, y))
            .min()
        else {
            return primary;
        };
        let colored = self.colored_mask(level);
        let closed = colored.close(self.close_radius(q.style));
        let mut visited = Mask::new(closed.width, closed.height);
        for &seed in &src_ring {
            if visited.get(seed.0 as isize, seed.1 as isize)
                || !closed.get(seed.0 as isize, seed.1 as isize)
            {
                continue;
            }
            let comp = closed.components_from(&[seed]);
            visited = visited.or(&comp);
            if dst_ring
                .iter()
                .any(|&(x, y)| comp.get(x as isize, y as isize))
            {
                let fallback = self.trace(q, &comp, &colored, &[seed]);
                if fallback.status == PixelStatus::Traced {
                    return fallback;
                }
            }
        }
        primary
    }

    fn trace(
        &self,
        q: &EdgeQuery,
        comp: &Mask,
        colored: &Mask,
        seeds: &[(usize, usize)],
    ) -> PixelEvidence {
        let p = &self.params;
        let area = comp.count();
        let canvas_area = (comp.width * comp.height).max(1) as f64;
        if area == 0 {
            return PixelEvidence::failed(PixelStatus::NoSeed);
        }
        if area as f64 / canvas_area > p.max_component_share {
            return PixelEvidence::failed(PixelStatus::ComponentTooLarge);
        }
        let raw = comp.and(colored);
        let (sub, ox, oy) = crop_to_content(comp, 2);
        let sub_raw = crop_region(&raw, ox, oy, sub.width, sub.height);
        let skel = zhang_suen(&sub);
        let ends = endpoints(&skel);
        let seeds_local: Vec<(usize, usize)> = seeds
            .iter()
            .filter(|&&(x, y)| x >= ox && y >= oy)
            .map(|&(x, y)| (x - ox, y - oy))
            .filter(|&(x, y)| x < sub.width && y < sub.height)
            .flat_map(|(x, y)| nearest_on(&skel, x, y, 3))
            .collect();
        let dist = geodesic(&skel, &seeds_local);
        let stroke_px = (sub_raw.count() as f64 / skel.count().max(1) as f64).max(1.0);

        let pick = |b: &BBox| -> Option<(usize, usize, f64)> {
            let lb = BBox::new(
                b.x1 - ox as f64,
                b.y1 - oy as f64,
                b.x2 - ox as f64,
                b.y2 - oy as f64,
            );
            ends.iter()
                .map(|&(x, y)| (x, y, dist_to_bbox((x as f64, y as f64), &lb)))
                .filter(|e| e.2 <= p.terminus_tolerance_px + p.node_margin_px)
                .min_by(|a, b| {
                    a.2.total_cmp(&b.2).then_with(|| {
                        let ga = dist[a.1 * skel.width + a.0].unwrap_or(0);
                        let gb = dist[b.1 * skel.width + b.0].unwrap_or(0);
                        gb.cmp(&ga)
                    })
                })
        };
        let mut src_t = pick(&q.src);
        let dst_t = pick(&q.dst);
        if let (Some(s), Some(d)) = (src_t, dst_t) {
            if (s.0, s.1) == (d.0, d.1) {
                src_t = None;
            }
        }
        let evidence = |t: (usize, usize, f64)| -> EndEvidence {
            let (area_ratio, spread_px) =
                arrow_metrics(&sub_raw, &skel, t.0, t.1, p.arrow_radius_px, stroke_px);
            let arrow = area_ratio >= p.min_arrow_area_ratio
                && spread_px
                    >= p.min_arrow_spread_px
                        .max(p.min_arrow_spread_ratio * stroke_px / 2.0);
            EndEvidence {
                x: (t.0 + ox) as f64,
                y: (t.1 + oy) as f64,
                node_distance_px: t.2,
                area_ratio,
                spread_px,
                arrow,
            }
        };
        let src_end = src_t.map(evidence);
        let dst_end = dst_t.map(evidence);
        let status = match (&src_end, &dst_end) {
            (Some(_), Some(_)) => PixelStatus::Traced,
            (None, None) => PixelStatus::NoTermini,
            _ => PixelStatus::OneEnd,
        };
        let verdict = EndVerdict::from_ends(src_end.map(|e| e.arrow), dst_end.map(|e| e.arrow));
        // A single observed end can show where the head is, but never "no arrowhead".
        let verdict = if status == PixelStatus::OneEnd && verdict == EndVerdict::NoArrowhead {
            EndVerdict::Unknown
        } else {
            verdict
        };
        PixelEvidence {
            status,
            src_end,
            dst_end,
            stroke_px,
            verdict,
        }
    }
}

/// Skeleton pixels within `r` of `(x, y)` (the seed may sit beside the thinned line).
fn nearest_on(skel: &Mask, x: usize, y: usize, r: isize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if skel.get(nx, ny) {
                out.push((nx as usize, ny as usize));
            }
        }
    }
    out
}

fn crop_to_content(m: &Mask, pad: usize) -> (Mask, usize, usize) {
    let (mut x1, mut y1, mut x2, mut y2) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..m.height {
        for x in 0..m.width {
            if m.data[y * m.width + x] {
                x1 = x1.min(x);
                y1 = y1.min(y);
                x2 = x2.max(x);
                y2 = y2.max(y);
            }
        }
    }
    if x1 == usize::MAX {
        return (Mask::new(0, 0), 0, 0);
    }
    let ox = x1.saturating_sub(pad);
    let oy = y1.saturating_sub(pad);
    let w = (x2 + pad + 1).min(m.width) - ox;
    let h = (y2 + pad + 1).min(m.height) - oy;
    (crop_region(m, ox, oy, w, h), ox, oy)
}

fn crop_region(m: &Mask, ox: usize, oy: usize, w: usize, h: usize) -> Mask {
    let mut out = Mask::new(w, h);
    for y in 0..h {
        for x in 0..w {
            out.data[y * w + x] = m.get((x + ox) as isize, (y + oy) as isize);
        }
    }
    out
}

/// `(area_ratio, spread_px)` of the component around a terminus.
fn arrow_metrics(raw: &Mask, skel: &Mask, tx: usize, ty: usize, r: f64, stroke: f64) -> (f64, f64) {
    let ri = r.ceil() as isize;
    let mut comp_pts = Vec::new();
    let mut skel_pts = Vec::new();
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            if ((dx * dx + dy * dy) as f64).sqrt() > r {
                continue;
            }
            let (x, y) = (tx as isize + dx, ty as isize + dy);
            if raw.get(x, y) {
                comp_pts.push((x as f64, y as f64));
            }
            if skel.get(x, y) {
                skel_pts.push((x as f64, y as f64));
            }
        }
    }
    if skel_pts.len() < 2 {
        return (0.0, 0.0);
    }
    let area_ratio = comp_pts.len() as f64 / (stroke * skel_pts.len() as f64).max(1.0);
    // Local axis: principal direction of the skeleton pixels near the terminus.
    let n = skel_pts.len() as f64;
    let (mx, my) = skel_pts
        .iter()
        .fold((0.0, 0.0), |a, p| (a.0 + p.0 / n, a.1 + p.1 / n));
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for p in &skel_pts {
        let (dx, dy) = (p.0 - mx, p.1 - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (s, c) = theta.sin_cos();
    let spread = comp_pts
        .iter()
        .map(|p| ((p.0 - mx) * -s + (p.1 - my) * c).abs())
        .fold(0.0, f64::max);
    (area_ratio, spread)
}

/// Variance of the Laplacian of the gray canvas (crop sharpness for vote weights).
pub fn sharpness(image: &Bgr) -> f64 {
    glassrip_media::sharpness::laplacian_variance(&bgr_to_gray(image))
}

/// Convert an RGB image to the media crate's BGR layout.
pub fn bgr_from_rgb(img: &image::RgbImage) -> Bgr {
    let (w, h) = img.dimensions();
    let mut data = Vec::with_capacity(w as usize * h as usize * 3);
    for p in img.pixels() {
        data.extend_from_slice(&[p[2], p[1], p[0]]);
    }
    Bgr {
        width: w as usize,
        height: h as usize,
        data,
    }
}
