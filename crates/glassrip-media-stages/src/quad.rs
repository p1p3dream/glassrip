//! `screen_quad` (spec 6.4): per-frame monitor quadrilateral.
//!
//! Gray frame downscaled to `work_width`, Gaussian blur, Canny, a small morphological close
//! to bridge edge gaps, contours, Douglas-Peucker approximation, and the largest convex
//! 4-gon whose area lies in `[min_area_frac, max_area_frac]` of the frame. Confidence is the
//! share of points sampled along the quad's four sides that have a Canny edge within
//! `edge_tolerance_px`. A quad below `min_confidence` is reported as none. This stage does
//! not change pixels.

use std::path::PathBuf;

use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, Stage, StageError, StageInputs, WorkItem,
};
use image::GrayImage;
use imageproc::contours::{BorderType, find_contours};
use imageproc::distance_transform::Norm;
use imageproc::point::Point;
use schemars::JsonSchema;
use serde::Serialize;

use crate::schema::{FRAMES, FrameRecord, Quad, SCREEN_QUADS, ScreenQuad, v1};

/// Detection parameters.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct QuadParams {
    /// Width the frame is downscaled to for detection.
    pub work_width: u32,
    /// Gaussian blur sigma before Canny.
    pub blur_sigma: f32,
    /// Canny low threshold.
    pub canny_low: f32,
    /// Canny high threshold.
    pub canny_high: f32,
    /// Morphological close radius bridging edge gaps (work pixels).
    pub close_radius: u8,
    /// Douglas-Peucker epsilon as a share of the contour perimeter.
    pub poly_epsilon_frac: f64,
    /// Smallest accepted quad area as a share of the frame.
    pub min_area_frac: f64,
    /// Largest accepted quad area (excludes the frame border itself).
    pub max_area_frac: f64,
    /// Edge search tolerance for confidence (work pixels).
    pub edge_tolerance_px: u32,
    /// Quads below this confidence are reported as none.
    pub min_confidence: f64,
}

impl Default for QuadParams {
    fn default() -> Self {
        Self {
            work_width: 640,
            blur_sigma: 1.2,
            canny_low: 20.0,
            canny_high: 60.0,
            close_radius: 2,
            poly_epsilon_frac: 0.02,
            min_area_frac: 0.15,
            max_area_frac: 0.97,
            edge_tolerance_px: 2,
            min_confidence: 0.5,
        }
    }
}

/// Result of [`detect_quad`].
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Quad in input pixels (TL, TR, BR, BL), when accepted.
    pub quad: Option<Quad>,
    /// Confidence of the accepted quad, or of the best rejected candidate.
    pub confidence: f64,
    /// Area share of the accepted quad.
    pub area_frac: Option<f64>,
}

fn shoelace(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    let mut s = 0.0;
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        s += a[0] * b[1] - b[0] * a[1];
    }
    s.abs() / 2.0
}

fn convex(p: &[[f64; 2]]) -> bool {
    let n = p.len();
    let mut sign = 0.0f64;
    for i in 0..n {
        let (a, b, c) = (p[i], p[(i + 1) % n], p[(i + 2) % n]);
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross.abs() < 1e-9 {
            return false;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    true
}

/// Orders four points as top-left, top-right, bottom-right, bottom-left.
pub fn order_quad(p: [[f64; 2]; 4]) -> Quad {
    let by = |f: &dyn Fn(&[f64; 2]) -> f64, max: bool| -> [f64; 2] {
        let mut best = p[0];
        for q in &p[1..] {
            let (v, b) = (f(q), f(&best));
            if (max && v > b) || (!max && v < b) {
                best = *q;
            }
        }
        best
    };
    let sum = |q: &[f64; 2]| q[0] + q[1];
    let diff = |q: &[f64; 2]| q[1] - q[0];
    [
        by(&sum, false),
        by(&diff, false),
        by(&sum, true),
        by(&diff, true),
    ]
}

fn edge_support(edges: &GrayImage, q: &[[f64; 2]; 4], tol: u32) -> f64 {
    let (w, h) = (edges.width() as i64, edges.height() as i64);
    let t = i64::from(tol);
    let (mut hit, mut total) = (0usize, 0usize);
    for i in 0..4 {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        let len = ((b[0] - a[0]).hypot(b[1] - a[1])).max(1.0);
        let steps = (len / 2.0).ceil() as usize;
        for s in 0..=steps {
            let f = s as f64 / steps.max(1) as f64;
            let (x, y) = (
                (a[0] + f * (b[0] - a[0])).round() as i64,
                (a[1] + f * (b[1] - a[1])).round() as i64,
            );
            total += 1;
            'search: for dy in -t..=t {
                for dx in -t..=t {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx >= 0
                        && yy >= 0
                        && xx < w
                        && yy < h
                        && edges.get_pixel(xx as u32, yy as u32)[0] > 0
                    {
                        hit += 1;
                        break 'search;
                    }
                }
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hit as f64 / total as f64
    }
}

/// Detects the monitor quad in a full-resolution gray frame.
pub fn detect_quad(gray: &GrayImage, p: &QuadParams) -> Detection {
    let (fw, fh) = gray.dimensions();
    let none = Detection {
        quad: None,
        confidence: 0.0,
        area_frac: None,
    };
    if fw < 8 || fh < 8 {
        return none;
    }
    let ww = p.work_width.clamp(8, fw);
    let wh = ((u64::from(fh) * u64::from(ww) / u64::from(fw)) as u32).max(8);
    let small = image::imageops::resize(gray, ww, wh, image::imageops::FilterType::Triangle);
    let blurred = if p.blur_sigma > 0.0 {
        imageproc::filter::gaussian_blur_f32(&small, p.blur_sigma)
    } else {
        small
    };
    let edges = imageproc::edges::canny(&blurred, p.canny_low, p.canny_high);
    let closed = if p.close_radius > 0 {
        imageproc::morphology::close(&edges, Norm::LInf, p.close_radius)
    } else {
        edges.clone()
    };
    let frame_area = f64::from(ww) * f64::from(wh);
    let mut best: Option<(f64, [[f64; 2]; 4], f64)> = None; // (area, quad, conf)
    let mut best_rejected = 0.0f64;
    for c in find_contours::<i32>(&closed) {
        if c.border_type != BorderType::Outer && c.border_type != BorderType::Hole {
            continue;
        }
        if c.points.len() < 16 {
            continue;
        }
        let pts = &c.points;
        let perim: f64 = pts
            .iter()
            .zip(pts.iter().cycle().skip(1))
            .map(|(a, b)| f64::from(a.x - b.x).hypot(f64::from(a.y - b.y)))
            .sum();
        let eps = (p.poly_epsilon_frac * perim).max(1.0);
        let approx: Vec<Point<i32>> = imageproc::geometry::approximate_polygon_dp(pts, eps, true);
        if approx.len() != 4 {
            continue;
        }
        let poly: Vec<[f64; 2]> = approx
            .iter()
            .map(|q| [f64::from(q.x), f64::from(q.y)])
            .collect();
        if !convex(&poly) {
            continue;
        }
        let area = shoelace(&poly);
        let frac = area / frame_area;
        if frac < p.min_area_frac || frac > p.max_area_frac {
            continue;
        }
        let quad = order_quad([poly[0], poly[1], poly[2], poly[3]]);
        let conf = edge_support(&edges, &quad, p.edge_tolerance_px);
        if conf < p.min_confidence {
            best_rejected = best_rejected.max(conf);
            continue;
        }
        if best.as_ref().is_none_or(|(a, _, _)| area > *a) {
            best = Some((area, quad, conf));
        }
    }
    match best {
        Some((area, q, conf)) => {
            let (sx, sy) = (f64::from(fw) / f64::from(ww), f64::from(fh) / f64::from(wh));
            let scaled = q.map(|[x, y]| [x * sx, y * sy]);
            Detection {
                quad: Some(scaled),
                confidence: conf,
                area_frac: Some(area / frame_area),
            }
        }
        None => Detection {
            confidence: best_rejected,
            ..none
        },
    }
}

/// The `screen_quad` stage.
#[derive(Debug, Clone)]
pub struct ScreenQuadStage {
    params: QuadParams,
    run_root: PathBuf,
}

impl ScreenQuadStage {
    /// Stage reading frames under `run_root`.
    pub fn new(params: QuadParams, run_root: PathBuf) -> Self {
        Self { params, run_root }
    }
}

impl Stage for ScreenQuadStage {
    type Params = QuadParams;
    type Work = FrameRecord;
    type Output = ScreenQuad;

    fn name(&self) -> &'static str {
        "screen_quad"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: SCREEN_QUADS,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![InputDecl {
            schema: FRAMES,
            major: 1,
        }]
    }
    fn params(&self) -> &QuadParams {
        &self.params
    }
    fn concurrency(&self) -> usize {
        crate::util::cpus() * 2
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<FrameRecord>>, StageError> {
        Ok(inputs
            .read_ok::<FrameRecord>(FRAMES)?
            .into_iter()
            .map(|(id, f)| WorkItem { id, work: f })
            .collect())
    }
    async fn process(&self, _ctx: &ItemContext, f: FrameRecord) -> Result<ScreenQuad, ErrorInfo> {
        let path = self.run_root.join(&f.path);
        let params = self.params.clone();
        crate::util::on_rayon(move || {
            let img = image::open(&path)
                .map_err(|e| {
                    ErrorInfo::new(
                        ErrorCode::InvalidInput,
                        format!("cannot read frame {}: {e}", path.display()),
                    )
                })?
                .to_luma8();
            let d = detect_quad(&img, &params);
            Ok(ScreenQuad {
                frame_id: f.frame_id,
                quad: d.quad,
                confidence: d.confidence,
                area_frac: d.area_frac,
            })
        })
        .await
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Dark frame with a bright, slightly rotated quad and some inner texture.
    pub(crate) fn synthetic_screen(w: u32, h: u32, q: [[f64; 2]; 4]) -> GrayImage {
        let inside = |x: f64, y: f64| {
            (0..4).all(|i| {
                let (a, b) = (q[i], q[(i + 1) % 4]);
                (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) >= 0.0
            })
        };
        GrayImage::from_fn(w, h, |x, y| {
            let (xf, yf) = (f64::from(x), f64::from(y));
            if inside(xf, yf) {
                let stripe = ((x / 37 + y / 23) % 2) as u8;
                image::Luma([200 + 20 * stripe])
            } else {
                image::Luma([30])
            }
        })
    }

    #[test]
    fn finds_tilted_monitor() {
        let q = [
            [300.0, 180.0],
            [1600.0, 240.0],
            [1540.0, 900.0],
            [260.0, 860.0],
        ];
        let img = synthetic_screen(1920, 1080, q);
        let d = detect_quad(&img, &QuadParams::default());
        let got = d.quad.unwrap();
        for (a, b) in got.iter().zip(q.iter()) {
            assert!(
                (a[0] - b[0]).abs() < 12.0 && (a[1] - b[1]).abs() < 12.0,
                "{got:?}"
            );
        }
        assert!(d.confidence > 0.8, "{d:?}");
    }

    #[test]
    fn flat_frame_has_no_quad() {
        let img = GrayImage::from_pixel(1920, 1080, image::Luma([128]));
        let d = detect_quad(&img, &QuadParams::default());
        assert_eq!(d.quad, None);
    }

    #[test]
    fn order_is_tl_tr_br_bl() {
        let q = order_quad([[10.0, 90.0], [100.0, 5.0], [0.0, 0.0], [95.0, 100.0]]);
        assert_eq!(q, [[0.0, 0.0], [100.0, 5.0], [95.0, 100.0], [10.0, 90.0]]);
        assert!(convex(&q));
    }
}
