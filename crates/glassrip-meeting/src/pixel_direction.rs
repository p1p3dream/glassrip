//! Pixel check for edge direction (spec 6.11, edge direction step 1).
//!
//! Per keyframe, a stroke mask is built once: dark pixels from an adaptive mean
//! threshold (the ink metric's rule), with every node box and every text box masked
//! out so that box outlines and glyphs cannot join connectors. Per edge:
//!
//! 1. **Seed.** Around the edge label (a ring outside the label box, since the label is
//!    masked and splits the line in two) or, without a label, around the midpoint of
//!    the segment joining the two boxes' facing borders. The stroke color is the
//!    candidates' median gray; seeds are candidates not much lighter than that.
//! 2. **Component.** Stroke pixels not much lighter than the stroke color (filled heads
//!    are darker than anti-aliased lines and always pass), closed morphologically (a
//!    larger radius for dashed lines so dashes join), then the 8-connected components
//!    that contain a seed.
//! 3. **Skeleton.** Zhang-Suen thinning of the component, so curved and elbow
//!    connectors are followed; skeleton endpoints are the candidate termini.
//! 4. **Termini.** For each of the two nodes, the endpoint closest to its box (within a
//!    tolerance; ties go to the endpoint geodesically farthest from the seed). A
//!    terminus inside the tolerance identifies the node.
//! 5. **Arrowhead.** On the unthinned component within `arrow_radius_px` of the
//!    terminus: the spread perpendicular to the local skeleton axis, and off-axis
//!    pixels on both sides of it (a head is symmetric; noise touching a line is not).
//!    A head is compact: a line crossing or branching near the terminus reaches the
//!    edge of the test disk sideways and is rejected.
//!
//! Node boxes are first snapped to their drawn outlines (the strongest straight
//! stroke within a few pixels of each side), and only the outline plus a thin band is
//! masked, so the small heads that touch an outline are kept even when the reader's
//! boxes are a few pixels off.

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
    /// Margin added around node boxes before masking, for sides without a detected
    /// outline, in pixels.
    pub node_margin_px: f64,
    /// Minimum search range for snapping a node box side to its drawn outline, in
    /// pixels.
    pub outline_search_px: f64,
    /// Search range as a share of the box's smaller side (reader boxes are off by
    /// more on large boxes), capped by `outline_search_max_px`.
    pub outline_search_share: f64,
    /// Largest search range, in pixels.
    pub outline_search_max_px: f64,
    /// Share of a side's middle span the outline must cover to be accepted.
    pub outline_min_cover: f64,
    /// Band masked beyond a detected outline, in pixels.
    pub outline_band_px: f64,
    /// Margin added around text boxes before masking, in pixels.
    pub text_margin_px: f64,
    /// Seed search radius around the label ring or the midpoint, in pixels.
    pub seed_radius_px: f64,
    /// How much lighter than the stroke color a seed or component pixel may be. Darker
    /// pixels always pass: filled heads render darker than anti-aliased thin lines.
    pub color_tolerance: u8,
    /// Closing radius for solid connectors.
    pub solid_close_px: usize,
    /// Closing radius for dashed connectors.
    pub dashed_close_px: usize,
    /// A terminus within this distance of a node box identifies the node.
    pub terminus_tolerance_px: f64,
    /// Radius of the arrowhead test around a terminus.
    pub arrow_radius_px: f64,
    /// Minimum off-axis pixels on each side of the local axis (a head is symmetric;
    /// a noise speck touching the line is one-sided).
    pub min_arrow_side_px: u32,
    /// Minimum overhang of the head beyond the line's edge (spread minus half the
    /// local stroke width), in pixels.
    pub min_arrow_overhang_px: f64,
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
            outline_search_px: 8.0,
            outline_search_share: 0.35,
            outline_search_max_px: 48.0,
            outline_min_cover: 0.6,
            outline_band_px: 2.0,
            text_margin_px: 2.0,
            seed_radius_px: 24.0,
            color_tolerance: 70,
            solid_close_px: 1,
            dashed_close_px: 5,
            terminus_tolerance_px: 14.0,
            arrow_radius_px: 15.0,
            min_arrow_side_px: 2,
            min_arrow_overhang_px: 1.5,
            min_arrow_spread_px: 2.0,
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
    /// Off-axis pixels on the smaller side of the local axis.
    pub off_axis_min_side: u32,
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
    /// `(box as given, box snapped to its drawn outline)` for every node.
    snapped: Vec<(BBox, BBox)>,
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
        // Fit every node box to its drawn outline first (on the unmasked strokes), so
        // the mask hugs the outline and small heads touching it survive. Outlines in
        // photographed screens are slightly slanted, so each side is a fitted line.
        let outlines: Vec<Outline> = nodes
            .iter()
            .map(|b| Outline::fit(&stroke, b, params))
            .collect();
        for o in &outlines {
            o.clear(&mut stroke, params.outline_band_px, m);
        }
        for b in texts {
            stroke.clear_rect(b.x1 - t, b.y1 - t, b.x2 + t, b.y2 + t);
        }
        Self {
            gray,
            snapped: nodes
                .iter()
                .copied()
                .zip(outlines.iter().map(Outline::bbox))
                .collect(),
            stroke,
            params: params.clone(),
        }
    }

    /// The outline-snapped version of a node box given to [`PreparedCanvas::new`].
    pub fn snapped(&self, b: &BBox) -> BBox {
        self.snapped
            .iter()
            .find(|(o, _)| o == b)
            .map(|(_, s)| *s)
            .unwrap_or(*b)
    }

    /// The masked stroke mask (diagnostics).
    pub fn stroke_mask(&self) -> &Mask {
        &self.stroke
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
            if *v && self.gray.data[i] > stroke_level.saturating_add(tol) {
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
        let q = &EdgeQuery {
            src: self.snapped(&q.src),
            dst: self.snapped(&q.dst),
            ..*q
        };
        let cands = self.seed_candidates(q);
        let primary = match median_level(cands.iter().map(|&(x, y)| self.gray.get(x, y))) {
            Some(level) => {
                let tol = self.params.color_tolerance;
                let seeds: Vec<(usize, usize)> = cands
                    .into_iter()
                    .filter(|&(x, y)| self.gray.get(x, y) <= level.saturating_add(tol))
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
        let Some(level) = median_level(
            src_ring
                .iter()
                .chain(&dst_ring)
                .map(|&(x, y)| self.gray.get(x, y)),
        ) else {
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
        // Stroke width from area over skeleton length; unreliable on tiny skeletons.
        let skel_n = skel.count();
        let stroke_px = if skel_n >= 10 {
            (sub_raw.count() as f64 / skel_n as f64).clamp(1.0, 4.0)
        } else {
            2.0
        };

        let pick = |b: &BBox| -> Option<(usize, usize, f64, BBox)> {
            let lb = BBox::new(
                b.x1 - ox as f64,
                b.y1 - oy as f64,
                b.x2 - ox as f64,
                b.y2 - oy as f64,
            );
            ends.iter()
                .map(|&(x, y)| (x, y, dist_to_bbox((x as f64, y as f64), &lb), lb))
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
        let evidence = |t: (usize, usize, f64, BBox)| -> EndEvidence {
            let m = arrow_metrics(
                &sub_raw,
                &skel,
                (t.0, t.1),
                &t.3,
                p.arrow_radius_px,
                stroke_px,
            );
            let arrow = !m.crossing
                && m.min_side >= p.min_arrow_side_px
                && m.spread
                    >= p.min_arrow_spread_px
                        .max(m.stroke / 2.0 + p.min_arrow_overhang_px);
            EndEvidence {
                x: (t.0 + ox) as f64,
                y: (t.1 + oy) as f64,
                node_distance_px: t.2,
                area_ratio: m.area_ratio,
                spread_px: m.spread,
                off_axis_min_side: m.min_side,
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

/// Median gray level (the connector's stroke color; robust to a dark filled head).
fn median_level(values: impl Iterator<Item = u8>) -> Option<u8> {
    let mut v: Vec<u8> = values.collect();
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    Some(v[v.len() / 2])
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

/// One fitted side of a node outline: its position (y for top and bottom, x for left
/// and right) at the two ends of the side's span, extrapolated linearly.
#[derive(Debug, Clone, Copy)]
struct Side {
    at_start: f64,
    at_end: f64,
    span: (f64, f64),
    found: bool,
}

impl Side {
    fn fixed(pos: f64, span: (f64, f64)) -> Self {
        Self {
            at_start: pos,
            at_end: pos,
            span,
            found: false,
        }
    }

    /// Position at coordinate `u` along the side.
    fn at(&self, u: f64) -> f64 {
        let len = self.span.1 - self.span.0;
        if len.abs() < 1e-9 {
            return self.at_start;
        }
        self.at_start + (self.at_end - self.at_start) * (u - self.span.0) / len
    }

    fn mean(&self) -> f64 {
        (self.at_start + self.at_end) / 2.0
    }
}

/// A node box fitted to its drawn outline.
#[derive(Debug, Clone, Copy)]
struct Outline {
    top: Side,
    bottom: Side,
    left: Side,
    right: Side,
}

impl Outline {
    /// Fit each side to a straight stroke near it: the search range grows with the box
    /// (`outline_search_share` of its smaller side, between `outline_search_px` and
    /// `outline_search_max_px`), lines may slant by up to a tenth of the side's middle
    /// 70%, and a line counts when it (with a 1 px band) covers `outline_min_cover` of
    /// that span. The qualifying line closest to the given edge wins, so a
    /// neighbouring box's outline is not taken.
    fn fit(stroke: &Mask, b: &BBox, p: &PixelCheckParams) -> Self {
        let (w, h) = (b.width(), b.height());
        let xs = (b.x1 + 0.15 * w, b.x2 - 0.15 * w);
        let ys = (b.y1 + 0.15 * h, b.y2 - 0.15 * h);
        let r = (p.outline_search_share * w.min(h))
            .clamp(
                p.outline_search_px,
                p.outline_search_max_px.max(p.outline_search_px),
            )
            .round()
            .max(0.0) as isize;
        let fit_side = |horizontal: bool, base: f64| -> Side {
            let span = if horizontal { xs } else { ys };
            let (u0, u1) = (span.0.round() as isize, span.1.round() as isize);
            if u1 <= u0 {
                return Side::fixed(base, span);
            }
            let hit = |u: isize, v: isize| {
                (-1..=1).any(|d| {
                    if horizontal {
                        stroke.get(u, v + d)
                    } else {
                        stroke.get(v + d, u)
                    }
                })
            };
            let b0 = base.round() as isize;
            let slope = (((u1 - u0) as f64 * 0.1).ceil() as isize).clamp(1, (r / 2).max(1));
            let mut best: Option<(f64, isize, isize)> = None;
            for a in b0 - r..=b0 + r {
                for e in a - slope..=a + slope {
                    let n = (u0..=u1)
                        .filter(|&u| {
                            let v = a as f64 + (e - a) as f64 * (u - u0) as f64 / (u1 - u0) as f64;
                            hit(u, v.round() as isize)
                        })
                        .count();
                    let cover = n as f64 / (u1 - u0 + 1) as f64;
                    let dist = (a - b0).abs() + (e - b0).abs();
                    if cover < p.outline_min_cover {
                        continue;
                    }
                    let better = match best {
                        None => true,
                        Some((bc, ba, be)) => {
                            let bd = (ba - b0).abs() + (be - b0).abs();
                            dist < bd || (dist == bd && cover > bc)
                        }
                    };
                    if better {
                        best = Some((cover, a, e));
                    }
                }
            }
            match best {
                Some((c, a, e)) if c >= p.outline_min_cover => Side {
                    at_start: a as f64,
                    at_end: e as f64,
                    span: (u0 as f64, u1 as f64),
                    found: true,
                },
                _ => Side::fixed(base, span),
            }
        };
        let out = Self {
            top: fit_side(true, b.y1),
            bottom: fit_side(true, b.y2),
            left: fit_side(false, b.x1),
            right: fit_side(false, b.x2),
        };
        if out.bbox().is_well_formed() {
            out
        } else {
            Self {
                top: Side::fixed(b.y1, xs),
                bottom: Side::fixed(b.y2, xs),
                left: Side::fixed(b.x1, ys),
                right: Side::fixed(b.x2, ys),
            }
        }
    }

    /// Axis-aligned box through the mean position of each side.
    fn bbox(&self) -> BBox {
        BBox::new(
            self.left.mean(),
            self.top.mean(),
            self.right.mean(),
            self.bottom.mean(),
        )
    }

    /// Clear the outline and its interior, plus `band` beyond fitted sides and
    /// `margin` beyond sides that were not found.
    fn clear(&self, m: &mut Mask, band: f64, margin: f64) {
        let e = |s: &Side| if s.found { band } else { margin };
        let bb = self.bbox();
        let pad = margin.max(band) + 16.0;
        let (w, h) = (m.width as f64, m.height as f64);
        let x0 = (bb.x1 - pad).floor().clamp(0.0, w) as usize;
        let x1 = (bb.x2 + pad).ceil().clamp(0.0, w) as usize;
        let y0 = (bb.y1 - pad).floor().clamp(0.0, h) as usize;
        let y1 = (bb.y2 + pad).ceil().clamp(0.0, h) as usize;
        for y in y0..y1 {
            let yf = y as f64;
            let (l, r) = (
                self.left.at(yf) - e(&self.left),
                self.right.at(yf) + e(&self.right),
            );
            for x in x0..x1 {
                let xf = x as f64;
                if xf < l || xf > r {
                    continue;
                }
                let (t, b) = (
                    self.top.at(xf) - e(&self.top),
                    self.bottom.at(xf) + e(&self.bottom),
                );
                if yf >= t && yf <= b {
                    m.data[y * m.width + x] = false;
                }
            }
        }
    }
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

/// Shape measurements of the component around a terminus.
struct ArrowMetrics {
    /// Area over the area of a plain line of the same skeleton length (diagnostic).
    area_ratio: f64,
    /// Largest perpendicular distance from the local axis.
    spread: f64,
    /// Off-axis pixels (beyond the line's edge) on the smaller side of the axis.
    min_side: u32,
    /// Local stroke width of the line near the terminus.
    stroke: f64,
    /// Some pixel leaves the test disk sideways (a crossing or branching line, not
    /// a compact head).
    crossing: bool,
}

/// Unit direction from the nearest point of `b` to `p` (the normal of the side a
/// connector enters through); toward the box center when `p` is inside.
fn approach_direction(p: (f64, f64), b: &BBox) -> (f64, f64) {
    let q = (p.0.clamp(b.x1, b.x2), p.1.clamp(b.y1, b.y2));
    let (mut dx, mut dy) = (p.0 - q.0, p.1 - q.1);
    if dx.abs() + dy.abs() < 1e-9 {
        dx = (b.x1 + b.x2) / 2.0 - p.0;
        dy = (b.y1 + b.y2) / 2.0 - p.1;
    }
    let n = (dx * dx + dy * dy).sqrt().max(1e-9);
    (dx / n, dy / n)
}

fn arrow_metrics(
    raw: &Mask,
    skel: &Mask,
    t: (usize, usize),
    node: &BBox,
    r: f64,
    stroke: f64,
) -> ArrowMetrics {
    let (tx, ty) = t;
    let none = ArrowMetrics {
        area_ratio: 0.0,
        spread: 0.0,
        min_side: 0,
        stroke,
        crossing: false,
    };
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
        return none;
    }
    let area_ratio = comp_pts.len() as f64 / (stroke * skel_pts.len() as f64).max(1.0);
    // Local axis: principal direction of the skeleton away from the terminus (the line
    // itself; thinning spurs inside a head would tilt it).
    let far: Vec<(f64, f64)> = skel_pts
        .iter()
        .copied()
        .filter(|p| ((p.0 - tx as f64).powi(2) + (p.1 - ty as f64).powi(2)).sqrt() >= 0.4 * r)
        .collect();
    // With a well-sampled line near the terminus, its principal direction is the
    // axis; otherwise (a short stub, for example between a head and a label) the
    // connector's approach direction to the box is.
    let use_far = far.len() >= 5;
    let axis_pts = if use_far { far } else { skel_pts.clone() };
    let n = axis_pts.len() as f64;
    let (mx, my) = axis_pts
        .iter()
        .fold((0.0, 0.0), |a, p| (a.0 + p.0 / n, a.1 + p.1 / n));
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for p in &axis_pts {
        let (dx, dy) = (p.0 - mx, p.1 - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let theta = if use_far {
        0.5 * (2.0 * sxy).atan2(sxx - syy)
    } else {
        let d = approach_direction((tx as f64, ty as f64), node);
        d.1.atan2(d.0)
    };
    let (s, c) = theta.sin_cos();
    // Center the axis on the line's own pixels in the same annulus: a thin line's
    // skeleton sits on one of its rows, which would bias offsets by half a pixel.
    let dist_t = |p: &(f64, f64)| ((p.0 - tx as f64).powi(2) + (p.1 - ty as f64).powi(2)).sqrt();
    let ring: Vec<(f64, f64)> = comp_pts
        .iter()
        .copied()
        .filter(|p| dist_t(p) >= 0.4 * r)
        .collect();
    // Local stroke width: line pixels over line length in the outer annulus, clear
    // of the head.
    let outer_px = comp_pts.iter().filter(|p| dist_t(p) >= 0.6 * r).count();
    let outer_len = skel_pts.iter().filter(|p| dist_t(p) >= 0.6 * r).count();
    let stroke = if outer_len >= 3 {
        (outer_px as f64 / outer_len as f64).clamp(1.0, 4.0)
    } else {
        stroke
    };
    let (mx, my) = if ring.len() >= 3 {
        let k = ring.len() as f64;
        ring.iter()
            .fold((0.0, 0.0), |a, p| (a.0 + p.0 / k, a.1 + p.1 / k))
    } else {
        (mx, my)
    };
    // Off-axis: clearly beyond the line's edge.
    let off = (stroke / 2.0 + 0.75).max(1.5);
    let (mut spread, mut left, mut right) = (0.0f64, 0u32, 0u32);
    for p in &comp_pts {
        let d = (p.0 - mx) * -s + (p.1 - my) * c;
        spread = spread.max(d.abs());
        if d >= off {
            left += 1;
        } else if d <= -off {
            right += 1;
        }
    }
    // Arms leaving the disk: skeleton pixels near the rim, grouped by adjacency. A
    // line end or a head has one (the connector); a crossing or branch has more.
    let rim: Vec<(f64, f64)> = skel_pts
        .iter()
        .copied()
        .filter(|p| dist_t(p) >= 0.8 * r)
        .collect();
    let mut arm = vec![usize::MAX; rim.len()];
    let mut arms = 0usize;
    for i in 0..rim.len() {
        if arm[i] != usize::MAX {
            continue;
        }
        arm[i] = arms;
        let mut stack = vec![i];
        while let Some(k) = stack.pop() {
            for j in 0..rim.len() {
                if arm[j] == usize::MAX
                    && (rim[j].0 - rim[k].0).abs() <= 1.5
                    && (rim[j].1 - rim[k].1).abs() <= 1.5
                {
                    arm[j] = arms;
                    stack.push(j);
                }
            }
        }
        arms += 1;
    }
    ArrowMetrics {
        area_ratio,
        spread,
        min_side: left.min(right),
        stroke,
        crossing: arms >= 2 || spread >= 0.8 * r,
    }
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
