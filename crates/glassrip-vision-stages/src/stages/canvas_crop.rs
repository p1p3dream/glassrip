//! `canvas_crop`: the board canvas of every whiteboard keyframe (spec 6.9).
//!
//! 1. Layout first ([`crate::layout`]): tiles from name labels and their grid,
//!    the conferencing title bar, and the whiteboard sidebar and top bar.
//! 2. Tiebreak: a lone name label with no grid is accepted as a tile only when
//!    its box is clearly more variable over the segment's frames (video) than
//!    the rest of the screen. Frames are warped by their screen quad first.
//! 3. Fallbacks: the classifier's canvas box, then the whole image.
//! 4. Stabilization: consecutive whiteboard keyframes whose boxes agree form a
//!    segment; each member within `stabilize_iou` of the segment median takes
//!    the median box.
//! 5. Text outside the canvas is chrome; chrome and tile text inside it is
//!    listed as masks painted over before board reading.

use std::collections::HashMap;
use std::path::PathBuf;

use glassrip_core::envelope::ErrorInfo;
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, Stage, StageError, StageInputs, WorkItem,
};
use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::Serialize;

use crate::artifacts::{
    self, CanvasCropItem, CanvasMethod, ChromeMask, ChromeReason, FrameView, KeyframeView,
    OcrKeyframe, Point, RectifiedKeyframeView, ScreenClassItem, ScreenQuadView, TextRegion,
};
use crate::layout::{self, Layout, LayoutConfig, Span};
use crate::pixels;
use crate::stages::{input, resolve, run_root};

/// Parameters.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CanvasCropParams {
    pub layout: LayoutConfig,
    /// Minimum IoU with the segment median to join a segment and be stabilized.
    pub stabilize_iou: f64,
    /// Width of the gray frames used for the variance tiebreak.
    pub variance_width: u32,
    /// Most frames sampled per segment for the tiebreak.
    pub variance_max_frames: usize,
    /// Tile accepted when inside variance >= ratio x outside variance.
    pub variance_ratio: f64,
}

impl Default for CanvasCropParams {
    fn default() -> Self {
        Self {
            layout: LayoutConfig::default(),
            stabilize_iou: 0.8,
            variance_width: 192,
            variance_max_frames: 12,
            variance_ratio: 2.0,
        }
    }
}

/// Precomputed per-keyframe work.
#[derive(Debug, Clone)]
pub struct CropWork {
    rect: RectifiedKeyframeView,
    source_frame_id: String,
    image_path: PathBuf,
    ocr: OcrKeyframe,
    layout: Layout,
    spans: Vec<Span>,
    raw: BBox,
    canvas: BBox,
    method: CanvasMethod,
    segment: u32,
    stabilized: bool,
}

/// The stage.
pub struct CanvasCropStage {
    params: CanvasCropParams,
}

impl CanvasCropStage {
    pub fn new(params: CanvasCropParams) -> Self {
        Self { params }
    }
}

fn spans_of(ocr: &OcrKeyframe) -> Vec<Span> {
    ocr.spans
        .iter()
        .map(|s| Span {
            text: s.text.clone(),
            bbox: s.bbox,
            confidence: s.confidence,
            bg_luma: Some(s.bg_luma),
        })
        .collect()
}

fn round_box(b: &BBox, w: f64, h: f64) -> BBox {
    BBox::new(
        b.x1.floor().clamp(0.0, w),
        b.y1.floor().clamp(0.0, h),
        b.x2.ceil().clamp(0.0, w),
        b.y2.ceil().clamp(0.0, h),
    )
}

fn median_box(boxes: &[BBox]) -> Option<BBox> {
    let med = |f: &dyn Fn(&BBox) -> f64| -> Option<f64> {
        let mut v: Vec<f64> = boxes.iter().map(f).collect();
        v.sort_by(f64::total_cmp);
        let n = v.len();
        (n > 0).then(|| {
            if n % 2 == 1 {
                v[n / 2]
            } else {
                (v[n / 2 - 1] + v[n / 2]) / 2.0
            }
        })
    };
    Some(BBox::new(
        med(&|b| b.x1)?,
        med(&|b| b.y1)?,
        med(&|b| b.x2)?,
        med(&|b| b.y2)?,
    ))
}

/// Assign stabilization segments and boxes in time order. `raws[i]` is `None`
/// for keyframes that are not whiteboards (they break segments).
pub fn stabilize(raws: &[Option<BBox>], min_iou: f64) -> Vec<Option<(u32, BBox, bool)>> {
    let mut segments: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for (i, r) in raws.iter().enumerate() {
        match r {
            None => {
                if !current.is_empty() {
                    segments.push(std::mem::take(&mut current));
                }
            }
            Some(b) => {
                let med = median_box(&current.iter().filter_map(|&j| raws[j]).collect::<Vec<_>>());
                if med.is_some_and(|m| m.iou(b) < min_iou) {
                    segments.push(std::mem::take(&mut current));
                }
                current.push(i);
            }
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    let mut out = vec![None; raws.len()];
    for (s, members) in segments.iter().enumerate() {
        let boxes: Vec<BBox> = members.iter().filter_map(|&j| raws[j]).collect();
        let Some(med) = median_box(&boxes) else {
            continue;
        };
        for &j in members {
            if let Some(raw) = raws[j] {
                let stable = members.len() > 1 && med.iou(&raw) >= min_iou;
                out[j] = Some((s as u32, if stable { med } else { raw }, stable));
            }
        }
    }
    out
}

/// Per-corner median of screen quads.
pub fn median_quad(quads: &[[Point; 4]]) -> Option<[Point; 4]> {
    if quads.is_empty() {
        return None;
    }
    let mut out = [[0.0; 2]; 4];
    for (c, corner) in out.iter_mut().enumerate() {
        for (a, v) in corner.iter_mut().enumerate() {
            let mut xs: Vec<f64> = quads.iter().map(|q| q[c][a]).collect();
            xs.sort_by(f64::total_cmp);
            let n = xs.len();
            *v = if n % 2 == 1 {
                xs[n / 2]
            } else {
                (xs[n / 2 - 1] + xs[n / 2]) / 2.0
            };
        }
    }
    Some(out)
}

/// Frames for the variance tiebreak and the one warp they share: the run's
/// median quad (as `rectify` computes it) over the frames that have a quad;
/// frames without a quad are left out. With no quads (screen recordings), all
/// frames are used unwarped.
pub fn tiebreak_frames(
    frames: &[(PathBuf, Option<[Point; 4]>)],
) -> (Vec<PathBuf>, Option<[Point; 4]>) {
    let quads: Vec<[Point; 4]> = frames.iter().filter_map(|(_, q)| *q).collect();
    match median_quad(&quads) {
        Some(m) => (
            frames
                .iter()
                .filter(|(_, q)| q.is_some())
                .map(|(p, _)| p.clone())
                .collect(),
            Some(m),
        ),
        None => (frames.iter().map(|(p, _)| p.clone()).collect(), None),
    }
}

/// Temporal-variance tiebreak for an ambiguous tile.
fn variance_accepts(
    tile: &BBox,
    frames: &[(PathBuf, Option<[Point; 4]>)],
    image_w: u32,
    image_h: u32,
    p: &CanvasCropParams,
) -> Option<bool> {
    let (paths, quad) = tiebreak_frames(frames);
    let step = (paths.len() / p.variance_max_frames.max(1)).max(1);
    let out_w = p.variance_width.max(16);
    let out_h =
        ((f64::from(out_w) * f64::from(image_h) / f64::from(image_w.max(1))).round() as u32).max(8);
    let mut grays = Vec::new();
    for path in paths.iter().step_by(step).take(p.variance_max_frames) {
        let Ok(img) = image::open(path) else {
            continue;
        };
        let gray = img.to_luma8();
        let warped = match &quad {
            Some(q) => pixels::warp_quad(&gray, q, out_w, out_h),
            None => Some(image::imageops::resize(
                &gray,
                out_w,
                out_h,
                image::imageops::FilterType::Triangle,
            )),
        };
        if let Some(w) = warped {
            grays.push(w);
        }
    }
    let (map, w, h) = pixels::temporal_std(&grays)?;
    let s = f64::from(out_w) / f64::from(image_w.max(1));
    let scaled = tile.scaled(s, s);
    let (inside, outside) = pixels::inside_outside_mean(&map, w, h, &scaled)?;
    Some(inside >= p.variance_ratio * outside.max(0.5))
}

impl Stage for CanvasCropStage {
    type Params = CanvasCropParams;
    type Work = CropWork;
    type Output = CanvasCropItem;

    fn name(&self) -> &'static str {
        "canvas_crop"
    }
    fn version(&self) -> u32 {
        2
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: artifacts::CANVAS_CROP,
            version: artifacts::output_version(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![
            input(artifacts::SCREEN_CLASS),
            input(artifacts::FRAMES),
            input(artifacts::SCREEN_QUADS),
            input(artifacts::RECTIFIED_KEYFRAMES),
            input(artifacts::KEYFRAMES),
            input(artifacts::OCR),
        ]
    }
    fn params(&self) -> &CanvasCropParams {
        &self.params
    }
    fn concurrency(&self) -> usize {
        4
    }

    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<CropWork>>, StageError> {
        let root = run_root(inputs, artifacts::RECTIFIED_KEYFRAMES)?;
        let classes: HashMap<String, ScreenClassItem> = inputs
            .read_ok::<ScreenClassItem>(artifacts::SCREEN_CLASS)?
            .into_iter()
            .collect();
        let ocr: HashMap<String, OcrKeyframe> = inputs
            .read_ok::<OcrKeyframe>(artifacts::OCR)?
            .into_iter()
            .collect();
        let rect: HashMap<String, RectifiedKeyframeView> = inputs
            .read_ok::<RectifiedKeyframeView>(artifacts::RECTIFIED_KEYFRAMES)?
            .into_iter()
            .map(|(_, r)| (r.keyframe_id.clone(), r))
            .collect();
        let quads: HashMap<String, Option<[Point; 4]>> = inputs
            .read_ok::<ScreenQuadView>(artifacts::SCREEN_QUADS)?
            .into_iter()
            .map(|(_, q)| (q.frame_id, q.quad))
            .collect();
        let mut frames: Vec<FrameView> = inputs
            .read_ok::<FrameView>(artifacts::FRAMES)?
            .into_iter()
            .map(|(_, f)| f)
            .collect();
        frames.sort_by(|a, b| a.pts_s.total_cmp(&b.pts_s));
        let mut keyframes: Vec<KeyframeView> = inputs
            .read_ok::<KeyframeView>(artifacts::KEYFRAMES)?
            .into_iter()
            .map(|(_, k)| k)
            .collect();
        keyframes.sort_by(|a, b| a.t_rep_s.total_cmp(&b.t_rep_s));

        let p = &self.params;
        let mut pending: Vec<Option<CropWork>> = Vec::with_capacity(keyframes.len());
        for k in &keyframes {
            let board = classes.get(&k.keyframe_id).is_some_and(|c| c.reads_board);
            let (Some(o), Some(r)) = (ocr.get(&k.keyframe_id), rect.get(&k.keyframe_id)) else {
                pending.push(None);
                continue;
            };
            if !board {
                pending.push(None);
                continue;
            }
            let (w, h) = (f64::from(o.image_width), f64::from(o.image_height));
            let spans = spans_of(o);
            let mut l = layout::analyze(&spans, w, h, &p.layout);
            let mut method = CanvasMethod::Layout;
            if let Some(t) = l.ambiguous_tile.clone() {
                let seg: Vec<(PathBuf, Option<[Point; 4]>)> = frames
                    .iter()
                    .filter(|f| f.pts_s >= k.t_start_s && f.pts_s < k.t_end_s)
                    .map(|f| {
                        (
                            resolve(&root, &f.path),
                            quads.get(&f.frame_id).copied().flatten(),
                        )
                    })
                    .collect();
                if variance_accepts(&t.bbox, &seg, o.image_width, o.image_height, p) == Some(true) {
                    layout::accept_ambiguous_tile(&mut l, &spans, &p.layout);
                    method = CanvasMethod::LayoutVariance;
                }
            }
            let model_box = classes
                .get(&k.keyframe_id)
                .and_then(|c| c.canvas_bbox)
                .filter(BBox::is_well_formed);
            let raw = match (l.canvas, model_box) {
                (Some(c), _) => c,
                (None, Some(m)) => {
                    method = CanvasMethod::ModelBox;
                    m
                }
                (None, None) => {
                    method = CanvasMethod::FullFrame;
                    BBox::new(0.0, 0.0, w, h)
                }
            };
            let raw = round_box(&raw, w, h);
            pending.push(Some(CropWork {
                image_path: resolve(&root, &r.image_path),
                rect: r.clone(),
                source_frame_id: k.rep_frame_id.clone(),
                ocr: o.clone(),
                layout: l,
                spans,
                raw,
                canvas: raw,
                method,
                segment: 0,
                stabilized: false,
            }));
        }
        let raws: Vec<Option<BBox>> = pending.iter().map(|w| w.as_ref().map(|w| w.raw)).collect();
        let stab = stabilize(&raws, p.stabilize_iou);
        Ok(pending
            .into_iter()
            .zip(stab)
            .filter_map(|(w, s)| {
                let mut w = w?;
                if let Some((segment, b, stable)) = s {
                    w.segment = segment;
                    w.canvas = round_box(
                        &b,
                        f64::from(w.ocr.image_width),
                        f64::from(w.ocr.image_height),
                    );
                    w.stabilized = stable;
                }
                Some(WorkItem {
                    id: w.rect.keyframe_id.clone(),
                    work: w,
                })
            })
            .collect())
    }

    async fn process(&self, _ctx: &ItemContext, w: CropWork) -> Result<CanvasCropItem, ErrorInfo> {
        let canvas = w.canvas;
        let mut span_regions = Vec::with_capacity(w.spans.len());
        let mut masks = Vec::new();
        let mut heights = Vec::new();
        for (i, s) in w.spans.iter().enumerate() {
            let (region, reason) = layout::region_of(&w.layout, i, s, Some(&canvas));
            span_regions.push(region);
            let overlaps = s.bbox.iou(&canvas) > 0.0;
            match region {
                TextRegion::Canvas => heights.push(s.bbox.height()),
                TextRegion::Chrome | TextRegion::Tile if overlaps => {
                    if reason != Some(ChromeReason::OutsideCanvas) {
                        masks.push(ChromeMask {
                            bbox: s.bbox,
                            reason: reason.unwrap_or(ChromeReason::Denylist),
                            text: s.text.clone(),
                        });
                    }
                }
                _ => {}
            }
        }
        for t in &w.layout.tiles {
            if t.bbox.iou(&canvas) > 0.0 {
                masks.push(ChromeMask {
                    bbox: t.bbox,
                    reason: ChromeReason::TileName,
                    text: t.name.clone(),
                });
            }
        }
        heights.sort_by(f64::total_cmp);
        let canvas_text_height_px = (!heights.is_empty()).then(|| heights[heights.len() / 2]);
        Ok(CanvasCropItem {
            keyframe_id: w.rect.keyframe_id.clone(),
            source_frame_id: w.source_frame_id,
            source_image_path: w.image_path.display().to_string(),
            source_image_blake3: w.rect.image_blake3.clone(),
            image_width: w.ocr.image_width,
            image_height: w.ocr.image_height,
            canvas_bbox: canvas,
            raw_canvas_bbox: w.raw,
            method: w.method,
            segment: w.segment,
            stabilized: w.stabilized,
            share_area: w.layout.share_area,
            tiles: w.layout.tiles.clone(),
            masks,
            span_regions,
            canvas_text_height_px,
            participants: w.layout.names.clone(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn tiebreak_uses_the_median_quad_and_drops_frames_without_one() {
        let q = |dx: f64| [[dx, 0.0], [100.0 + dx, 0.0], [100.0 + dx, 50.0], [dx, 50.0]];
        let frames = vec![
            (PathBuf::from("a"), Some(q(0.0))),
            (PathBuf::from("b"), None),
            (PathBuf::from("c"), Some(q(4.0))),
            (PathBuf::from("d"), Some(q(2.0))),
        ];
        let (paths, quad) = tiebreak_frames(&frames);
        assert_eq!(
            paths,
            vec![PathBuf::from("a"), PathBuf::from("c"), PathBuf::from("d")]
        );
        assert_eq!(quad, Some(q(2.0)));
        let plain = vec![(PathBuf::from("a"), None), (PathBuf::from("b"), None)];
        let (paths, quad) = tiebreak_frames(&plain);
        assert_eq!(paths.len(), 2);
        assert!(quad.is_none());
    }

    #[test]
    fn stabilization_segments_and_medians() {
        let a = BBox::new(100.0, 100.0, 900.0, 600.0);
        let a2 = BBox::new(104.0, 98.0, 902.0, 604.0);
        let a3 = BBox::new(98.0, 102.0, 898.0, 598.0);
        let b = BBox::new(0.0, 0.0, 400.0, 300.0);
        let raws = vec![Some(a), Some(a2), Some(a3), None, Some(b), Some(a)];
        let s = stabilize(&raws, 0.8);
        let (seg0, box0, st0) = s[0].unwrap();
        assert_eq!(seg0, 0);
        assert!(st0);
        assert_eq!(box0, BBox::new(100.0, 100.0, 900.0, 600.0));
        assert_eq!(s[1].unwrap().1, box0);
        assert!(s[3].is_none());
        // A different layout after a gap starts new segments; singletons keep their box.
        let (seg4, box4, st4) = s[4].unwrap();
        assert_eq!(seg4, 1);
        assert!(!st4);
        assert_eq!(box4, b);
        assert_eq!(s[5].unwrap().0, 2);
    }
}
