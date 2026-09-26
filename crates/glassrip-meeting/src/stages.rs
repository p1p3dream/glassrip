//! `edge_direction` and `board_state` as [`glassrip_core::runner::Stage`]s.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, KeyExtras, Stage, StageError, StageInputs, WorkItem,
};
use glassrip_vision::board::ValidatedBoard;
use glassrip_vision::{BBox, VisionClient};
use image::imageops::FilterType;
use image::RgbImage;
use schemars::JsonSchema;
use semver::Version;
use serde::{Deserialize, Serialize};

use crate::artifacts::{
    AxisScale, BoardItem, CanvasCropView, CanvasDims, CoordinateCheck, EdgeDirectionBatch,
    EdgeDirectionItem, EdgeEvidence, KeyframeView, OcrSpanView, OcrView, ValidateItemView,
    VlmFallback, BOARD_STATE, BOARD_VALIDATE, CANVAS_CROP, EDGE_DIRECTION, KEYFRAMES, OCR,
};
use crate::consolidate::owners::{Corroborator, NoCorroboration};
use crate::consolidate::{
    consolidate_with_probe, split_boards, BoardFrame, BoardStateItem, ConsolidationParams, Hooks,
    RegionProbe, SecondReader, StrokeTrace, TextAnchor,
};
use crate::direction::{frame_weight, DirectionVotes};
use crate::pixel_direction::{
    bgr_from_rgb, sharpness, EdgeQuery, EndEvidence, PixelCheckParams, PreparedCanvas,
};
use crate::text::{fuzzy_eq, normalize, FUZZY_THRESHOLD};
use crate::vlm_direction::{check_connector, EdgeEnds, VlmCheckParams, CONNECTOR_PROMPT};

/// `edge_direction` parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeDirectionParams {
    /// Pixel check.
    pub pixel: PixelCheckParams,
    /// VLM fallback for edges whose pixel vote is inconclusive.
    pub vlm: VlmCheckParams,
    /// Whether the VLM fallback runs (set from the presence of a client).
    pub vlm_enabled: bool,
    /// Share of the decisive pixel weight a direction needs to count as a verdict.
    pub vote_min_share: f64,
}

/// One board keyframe to check.
#[derive(Debug, Clone)]
pub struct FrameWork {
    /// Keyframe id.
    pub keyframe_id: String,
    /// Validated reading.
    pub board: ValidatedBoard,
    /// Canvas size recorded with the reading.
    pub canvas: Option<CanvasDims>,
    /// Crop image.
    pub image: Option<PathBuf>,
    /// Crop box in frame pixels.
    pub crop: Option<BBox>,
    /// OCR spans of the keyframe, in frame pixels (empty when unavailable).
    pub ocr: Vec<OcrSpanView>,
}

/// All board keyframes (one work item).
#[derive(Debug, Clone)]
pub struct EdgeDirectionWork {
    frames: Vec<FrameWork>,
}

/// Pixel check for every edge of every board keyframe, then the VLM fallback for edges
/// whose cross-keyframe pixel vote is inconclusive.
pub struct EdgeDirectionStage {
    params: EdgeDirectionParams,
    client: Option<VisionClient>,
}

impl EdgeDirectionStage {
    /// Stage with a vision client for the fallback, or `None` for pixels only.
    pub fn new(pixel: PixelCheckParams, vlm: VlmCheckParams, client: Option<VisionClient>) -> Self {
        Self {
            params: EdgeDirectionParams {
                pixel,
                vlm,
                vlm_enabled: client.is_some(),
                vote_min_share: 0.6,
            },
            client,
        }
    }
}

/// Canvas-coordinate box of the other-text item that matches an edge label.
fn label_box(board: &ValidatedBoard, label: &str) -> Option<BBox> {
    if label.trim().is_empty() {
        return None;
    }
    board
        .other_visible_text
        .iter()
        .find(|t| fuzzy_eq(&t.text, label, FUZZY_THRESHOLD))
        .map(|t| t.bbox)
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Pixel evidence for every edge of one keyframe (CPU only). Boxes and the image must
/// share one coordinate space.
pub fn pixel_evidence(
    image: &RgbImage,
    board: &ValidatedBoard,
    params: &PixelCheckParams,
) -> (Vec<EdgeEvidence>, f64, f64) {
    let (edges, sharp, zoom, _) = pixel_evidence_with_ocr(image, board, &[], params);
    (edges, sharp, zoom)
}

/// [`pixel_evidence`] with the keyframe's OCR spans (in the image's coordinates):
/// node and label boxes are first moved onto their OCR text, and every OCR text box
/// is masked ([`crate::ocr_anchor`]). Also returns `(nodes moved, labels found)`.
pub fn pixel_evidence_with_ocr(
    image: &RgbImage,
    board: &ValidatedBoard,
    ocr: &[TextAnchor],
    params: &PixelCheckParams,
) -> (Vec<EdgeEvidence>, f64, f64, (usize, usize)) {
    let re = crate::ocr_anchor::reanchor(board, ocr, &params.ocr_anchor);
    let moved = (re.nodes_moved, re.labels_found);
    let board = &re.board;
    let bgr = bgr_from_rgb(image);
    let nodes: Vec<BBox> = board.nodes.iter().map(|n| n.bbox).collect();
    let texts: Vec<BBox> = board
        .stickies
        .iter()
        .map(|s| s.bbox)
        .chain(board.owner_tags.iter().map(|o| o.bbox))
        .chain(board.other_visible_text.iter().map(|t| t.bbox))
        // Edge labels are text too: unmasked, their glyphs join the connector.
        .chain(board.edges.iter().filter_map(|e| e.label_bbox))
        .chain(re.text_boxes.iter().copied())
        .collect();
    let prepared = PreparedCanvas::new(&bgr, &nodes, &texts, params);
    let by_id: HashMap<&str, &glassrip_vision::board::BoardNode> = board
        .nodes
        .iter()
        .map(|n| (n.local_id.as_str(), n))
        .collect();
    let mut out = Vec::new();
    for e in &board.edges {
        let (Some(s), Some(d)) = (by_id.get(e.src.as_str()), by_id.get(e.dst.as_str())) else {
            continue;
        };
        let pixel = prepared.check(&EdgeQuery {
            src: s.bbox,
            dst: d.bbox,
            label: e
                .label_bbox
                .filter(|_| !crate::text::clean_label(&e.label).is_empty())
                .or_else(|| label_box(board, &crate::text::clean_label(&e.label))),
            style: e.style,
        });
        out.push(EdgeEvidence {
            src: e.src.clone(),
            dst: e.dst.clone(),
            src_text: s.text.clone(),
            dst_text: d.text.clone(),
            label: e.label.clone(),
            style: e.style,
            pixel,
            vlm: None,
        });
    }
    let zoom = median(board.nodes.iter().map(|n| n.bbox.height()).collect());
    (out, sharpness(&bgr), zoom, moved)
}

/// OCR spans of a keyframe in the loaded canvas image's pixels: canvas spans, and
/// unassigned spans inside the crop (chrome and tile names never count), shifted by
/// the crop origin.
pub fn ocr_in_canvas(spans: &[OcrSpanView], crop: Option<BBox>) -> Vec<TextAnchor> {
    let (ox, oy) = crop.filter(|c| c.is_well_formed()).map_or((0.0, 0.0), |c| {
        (c.x1.max(0.0).floor(), c.y1.max(0.0).floor())
    });
    spans
        .iter()
        .filter(|s| {
            let inside = crop.is_none_or(|c| {
                s.bbox.x1 >= c.x1 && s.bbox.y1 >= c.y1 && s.bbox.x2 <= c.x2 && s.bbox.y2 <= c.y2
            });
            match s.region.as_deref() {
                Some("canvas") => true,
                Some("unassigned") | None => inside,
                Some(_) => false,
            }
        })
        .filter(|s| s.bbox.is_well_formed())
        .map(|s| TextAnchor {
            text: s.text.clone(),
            bbox: BBox::new(
                s.bbox.x1 - ox,
                s.bbox.y1 - oy,
                s.bbox.x2 - ox,
                s.bbox.y2 - oy,
            ),
        })
        .collect()
}

fn scale_box(b: &BBox, sx: f64, sy: f64) -> BBox {
    BBox::new(b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy)
}

/// A copy of the reading with every box scaled by `(sx, sy)`.
pub fn scale_board(board: &ValidatedBoard, sx: f64, sy: f64) -> ValidatedBoard {
    let mut b = board.clone();
    for n in &mut b.nodes {
        n.bbox = scale_box(&n.bbox, sx, sy);
    }
    for s in &mut b.stickies {
        s.bbox = scale_box(&s.bbox, sx, sy);
    }
    for o in &mut b.owner_tags {
        o.bbox = scale_box(&o.bbox, sx, sy);
    }
    for t in &mut b.other_visible_text {
        t.bbox = scale_box(&t.bbox, sx, sy);
    }
    for e in &mut b.edges {
        e.label_bbox = e.label_bbox.map(|l| scale_box(&l, sx, sy));
    }
    b
}

/// How reading coordinates map onto a crop of `(iw, ih)` pixels.
pub fn coordinate_check(
    board: &ValidatedBoard,
    canvas: Option<CanvasDims>,
    iw: f64,
    ih: f64,
) -> (CoordinateCheck, AxisScale) {
    match canvas {
        Some(c) if c.width > 0.0 && c.height > 0.0 => {
            let (sx, sy) = (iw / c.width, ih / c.height);
            let one = (sx - 1.0).abs() < 1e-3 && (sy - 1.0).abs() < 1e-3;
            (
                if one {
                    CoordinateCheck::Verified1to1
                } else {
                    CoordinateCheck::Scaled
                },
                AxisScale { x: sx, y: sy },
            )
        }
        _ => {
            let inside = board.nodes.iter().all(|n| n.bbox.is_inside(iw, ih, 2.0));
            (
                if inside {
                    CoordinateCheck::Assumed1to1
                } else {
                    CoordinateCheck::Inconsistent
                },
                AxisScale { x: 1.0, y: 1.0 },
            )
        }
    }
}

fn unscale_end(e: &mut Option<EndEvidence>, s: AxisScale) {
    if let Some(t) = e.as_mut() {
        t.x /= s.x;
        t.y /= s.y;
    }
}

/// Load the crop and compute pixel evidence for one keyframe, in reading coordinates.
/// Load a keyframe's canvas. The file is either the canvas crop itself or the
/// whole frame; in the second case (the image extends past the crop box, as
/// `glassrip.canvas_crop` items that name their source frame do) the crop box
/// cuts the canvas out.
pub fn load_canvas(path: &Path, crop: Option<BBox>) -> Result<RgbImage, String> {
    let image = image::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .to_rgb8();
    let (w, h) = (f64::from(image.width()), f64::from(image.height()));
    let Some(c) = crop.filter(|c| c.is_well_formed()) else {
        return Ok(image);
    };
    let whole_frame =
        w >= c.x2 - 1.0 && h >= c.y2 - 1.0 && (w > c.width() + 1.0 || h > c.height() + 1.0);
    if !whole_frame {
        return Ok(image);
    }
    let x = c.x1.max(0.0).floor() as u32;
    let y = c.y1.max(0.0).floor() as u32;
    let cw = (c.x2.min(w) - f64::from(x)).round().max(1.0) as u32;
    let ch = (c.y2.min(h) - f64::from(y)).round().max(1.0) as u32;
    Ok(image::imageops::crop_imm(&image, x, y, cw, ch).to_image())
}

pub fn pixel_frame(fw: &FrameWork, params: &PixelCheckParams) -> EdgeDirectionItem {
    let empty = |error: String| EdgeDirectionItem {
        keyframe_id: fw.keyframe_id.clone(),
        canvas: fw.canvas.unwrap_or(CanvasDims {
            width: 0.0,
            height: 0.0,
        }),
        board_to_image: AxisScale { x: 1.0, y: 1.0 },
        coordinates: CoordinateCheck::Inconsistent,
        crop_in_frame: fw.crop,
        sharpness: 0.0,
        zoom: 0.0,
        edges: Vec::new(),
        error: Some(error),
    };
    let Some(path) = &fw.image else {
        return empty("no canvas crop for this keyframe".into());
    };
    let image = match load_canvas(path, fw.crop) {
        Ok(i) => i,
        Err(e) => return empty(e),
    };
    let (iw, ih) = (f64::from(image.width()), f64::from(image.height()));
    let (check, scale) = coordinate_check(&fw.board, fw.canvas, iw, ih);
    if check == CoordinateCheck::Inconsistent {
        return EdgeDirectionItem {
            coordinates: check,
            ..empty("reading boxes fall outside the crop and no canvas size is recorded".into())
        };
    }
    let board = scale_board(&fw.board, scale.x, scale.y);
    let ocr = ocr_in_canvas(&fw.ocr, fw.crop);
    let (mut edges, sharp, zoom, _) = pixel_evidence_with_ocr(&image, &board, &ocr, params);
    for e in &mut edges {
        unscale_end(&mut e.pixel.src_end, scale);
        unscale_end(&mut e.pixel.dst_end, scale);
    }
    EdgeDirectionItem {
        keyframe_id: fw.keyframe_id.clone(),
        canvas: CanvasDims {
            width: iw / scale.x,
            height: ih / scale.y,
        },
        board_to_image: scale,
        coordinates: check,
        crop_in_frame: fw.crop,
        sharpness: sharp,
        zoom: zoom / scale.y,
        edges,
        error: None,
    }
}

/// One edge grouped across keyframes by the texts of its ends.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeGroup {
    /// Text of end `a`.
    pub a: String,
    /// Text of end `b`.
    pub b: String,
    /// `(keyframe item index, edge index, src is a)`.
    pub members: Vec<(usize, usize, bool)>,
}

/// Group edge readings across keyframes: two readings are the same edge when both end
/// texts match (fuzzy, either orientation).
pub fn group_edges(items: &[EdgeDirectionItem]) -> Vec<EdgeGroup> {
    let mut groups: Vec<EdgeGroup> = Vec::new();
    for (ii, item) in items.iter().enumerate() {
        for (ei, e) in item.edges.iter().enumerate() {
            let m = |x: &str, y: &str| fuzzy_eq(x, y, FUZZY_THRESHOLD);
            let found = groups.iter_mut().find_map(|g| {
                if m(&g.a, &e.src_text) && m(&g.b, &e.dst_text) {
                    Some((g, true))
                } else if m(&g.a, &e.dst_text) && m(&g.b, &e.src_text) {
                    Some((g, false))
                } else {
                    None
                }
            });
            match found {
                Some((g, same)) => g.members.push((ii, ei, same)),
                None => groups.push(EdgeGroup {
                    a: e.src_text.clone(),
                    b: e.dst_text.clone(),
                    members: vec![(ii, ei, true)],
                }),
            }
        }
    }
    groups
}

/// Weighted pixel votes of a group, and each member's vote weight.
pub fn group_votes(items: &[EdgeDirectionItem], g: &EdgeGroup) -> (DirectionVotes, Vec<f64>) {
    let ok: Vec<&EdgeDirectionItem> = items.iter().filter(|i| i.error.is_none()).collect();
    let med_sharp = median(ok.iter().map(|i| i.sharpness).collect());
    let med_zoom = median(ok.iter().map(|i| i.zoom).collect());
    let mut v = DirectionVotes::default();
    let mut weights = Vec::new();
    for &(ii, ei, same) in &g.members {
        let item = &items[ii];
        let w = frame_weight(item.sharpness, med_sharp, item.zoom, med_zoom);
        let verdict = item.edges[ei].pixel.verdict;
        v.pixel
            .add(if same { verdict } else { verdict.flipped() }, w);
        weights.push(w);
    }
    (v, weights)
}

fn resolve(root: &Path, p: &str) -> PathBuf {
    let path = PathBuf::from(p);
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn run_root(inputs: &StageInputs, schema: &str) -> Result<PathBuf, StageError> {
    let p = inputs.path(schema)?;
    p.parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            StageError::Invalid(format!("artifact path {} has no run root", p.display()))
        })
}

impl EdgeDirectionStage {
    async fn fallback(
        &self,
        client: &VisionClient,
        frames: &[FrameWork],
        items: &mut [EdgeDirectionItem],
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<Vec<VlmFallback>, ErrorInfo> {
        let mut out = Vec::new();
        let groups = group_edges(items);
        for g in groups {
            let (votes, weights) = group_votes(items, &g);
            if !votes.pixel_inconclusive(self.params.vote_min_share) {
                continue;
            }
            let mut order: Vec<usize> = (0..g.members.len()).collect();
            order.sort_by(|a, b| weights[*b].total_cmp(&weights[*a]).then(a.cmp(b)));
            let mut record = VlmFallback {
                ends: (g.a.clone(), g.b.clone()),
                keyframes: Vec::new(),
                errors: Vec::new(),
            };
            for &k in order.iter().take(self.params.vlm.frames_per_edge.max(1)) {
                if cancel.is_cancelled() {
                    return Err(ErrorInfo::new(ErrorCode::Cancelled, "cancelled"));
                }
                let (ii, ei, _) = g.members[k];
                let item = &items[ii];
                let Some(fw) = frames.iter().find(|f| f.keyframe_id == item.keyframe_id) else {
                    continue;
                };
                let Some(path) = &fw.image else { continue };
                let image = match load_canvas(path, fw.crop) {
                    Ok(i) => i,
                    Err(e) => {
                        record.errors.push(format!("{}: {e}", item.keyframe_id));
                        continue;
                    }
                };
                // Work in reading coordinates.
                let image = if item.coordinates == CoordinateCheck::Scaled {
                    image::imageops::resize(
                        &image,
                        item.canvas.width.round().max(1.0) as u32,
                        item.canvas.height.round().max(1.0) as u32,
                        FilterType::Triangle,
                    )
                } else {
                    image
                };
                let ev = &item.edges[ei];
                let nodes: HashMap<&str, &glassrip_vision::board::BoardNode> = fw
                    .board
                    .nodes
                    .iter()
                    .map(|n| (n.local_id.as_str(), n))
                    .collect();
                let (Some(s), Some(d)) = (nodes.get(ev.src.as_str()), nodes.get(ev.dst.as_str()))
                else {
                    continue;
                };
                let ends = EdgeEnds {
                    src: &s.bbox,
                    dst: &d.bbox,
                    src_text: &s.text,
                    dst_text: &d.text,
                    src_end: ev.pixel.src_end.as_ref(),
                    dst_end: ev.pixel.dst_end.as_ref(),
                };
                record.keyframes.push(item.keyframe_id.clone());
                match check_connector(client, &image, ends, &self.params.vlm, cancel.clone()).await
                {
                    Ok(v) => items[ii].edges[ei].vlm = v,
                    Err(e) => record.errors.push(format!("{}: {e}", item.keyframe_id)),
                }
            }
            out.push(record);
        }
        Ok(out)
    }
}

impl Stage for EdgeDirectionStage {
    type Params = EdgeDirectionParams;
    type Work = EdgeDirectionWork;
    type Output = EdgeDirectionBatch;

    fn name(&self) -> &'static str {
        "edge_direction"
    }
    fn version(&self) -> u32 {
        // 2: node and label boxes re-anchored to OCR text; OCR is an input.
        2
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: EDGE_DIRECTION,
            version: Version::new(1, 0, 0),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![
            InputDecl {
                schema: BOARD_VALIDATE,
                major: 1,
            },
            InputDecl {
                schema: CANVAS_CROP,
                major: 1,
            },
            InputDecl {
                schema: OCR,
                major: 1,
            },
        ]
    }
    fn params(&self) -> &EdgeDirectionParams {
        &self.params
    }
    fn key_extras(&self) -> KeyExtras {
        match &self.client {
            Some(c) => KeyExtras {
                model_digest: c.backend().id().digest,
                prompt_hash: Some(glassrip_core::blake3_hex(CONNECTOR_PROMPT.as_bytes())),
                ..KeyExtras::default()
            },
            None => KeyExtras::default(),
        }
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<EdgeDirectionWork>>, StageError> {
        let root = run_root(inputs, CANVAS_CROP)?;
        let crops: HashMap<String, (PathBuf, Option<BBox>)> = inputs
            .read_ok::<CanvasCropView>(CANVAS_CROP)?
            .into_iter()
            .map(|(id, c)| {
                (
                    c.keyframe_id.unwrap_or(id),
                    (resolve(&root, &c.path), c.crop),
                )
            })
            .collect();
        let mut ocr: HashMap<String, Vec<OcrSpanView>> = inputs
            .read_ok::<OcrView>(OCR)?
            .into_iter()
            .map(|(id, o)| (o.keyframe_id.clone().unwrap_or(id), o.spans))
            .collect();
        let frames = inputs
            .read_ok::<ValidateItemView>(BOARD_VALIDATE)?
            .into_iter()
            .map(|(id, v)| {
                let item = v.into_item(&id);
                let crop = crops.get(&item.keyframe_id);
                FrameWork {
                    ocr: ocr.remove(&item.keyframe_id).unwrap_or_default(),
                    image: crop.map(|c| c.0.clone()),
                    crop: crop.and_then(|c| c.1),
                    keyframe_id: item.keyframe_id,
                    board: item.board,
                    canvas: item.canvas,
                }
            })
            .collect();
        Ok(vec![WorkItem {
            id: "edge-direction".into(),
            work: EdgeDirectionWork { frames },
        }])
    }
    async fn process(
        &self,
        ctx: &ItemContext,
        work: EdgeDirectionWork,
    ) -> Result<EdgeDirectionBatch, ErrorInfo> {
        let frames = Arc::new(work.frames);
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .clamp(1, 8);
        let mut items: Vec<Option<EdgeDirectionItem>> = vec![None; frames.len()];
        let mut set = tokio::task::JoinSet::new();
        let mut next = 0usize;
        while next < frames.len() || !set.is_empty() {
            while next < frames.len() && set.len() < workers {
                if ctx.cancel_token().is_cancelled() {
                    return Err(ErrorInfo::new(ErrorCode::Cancelled, "cancelled"));
                }
                let (f, p, i) = (Arc::clone(&frames), self.params.pixel.clone(), next);
                set.spawn_blocking(move || (i, pixel_frame(&f[i], &p)));
                next += 1;
            }
            if let Some(r) = set.join_next().await {
                let (i, item) = r.map_err(|e| {
                    ErrorInfo::new(ErrorCode::Internal, format!("pixel check task: {e}"))
                })?;
                items[i] = Some(item);
            }
        }
        let mut items: Vec<EdgeDirectionItem> = items.into_iter().flatten().collect();
        let vlm_fallback = match &self.client {
            Some(client) => {
                self.fallback(client, &frames, &mut items, ctx.cancel_token())
                    .await?
            }
            None => Vec::new(),
        };
        Ok(EdgeDirectionBatch {
            keyframes: items,
            vlm_fallback,
            vlm_available: self.client.is_some(),
        })
    }
}

/// Work for one board.
#[derive(Debug, Clone)]
pub struct BoardStateWork {
    frames: Vec<BoardFrame>,
    probe: Option<Arc<CropProbe>>,
}

/// Canvas pixels of one keyframe for [`CropProbe`].
pub struct ProbeCanvas {
    image: RgbImage,
    background: [u8; 3],
    /// Reading coordinates to image pixels.
    sx: f64,
    sy: f64,
    tolerance: u8,
    /// No pixel is ink and the luminance spread is under
    /// [`ProbeCanvas::UNIFORM_MAX_STD`].
    uniform: bool,
}

impl ProbeCanvas {
    /// A pixel is ink when one of its channels differs from the canvas background
    /// (the per-channel median of a sample grid) by more than this.
    pub const DEFAULT_TOLERANCE: u8 = 48;
    /// Largest luminance standard deviation (0 to 255) of a canvas that counts
    /// as one flat color: compression noise on a flat fill stays well under it,
    /// and any drawn mark or text raises it.
    pub const UNIFORM_MAX_STD: f64 = 2.0;
    /// Cell size (image pixels) of the stroke trace grid: gaps up to about a
    /// cell (dashes, anti-aliasing breaks) are bridged.
    const STROKE_CELL: usize = 3;
    /// Image pixels added around every ignored box (outlines drawn on the edge).
    const BOX_PAD: f64 = 2.0;

    /// `image` is the canvas crop; `canvas` the size of the reading's coordinate
    /// space (identity scale when unknown).
    pub fn new(image: RgbImage, canvas: Option<CanvasDims>) -> Self {
        let (w, h) = (image.width(), image.height());
        let step = ((u64::from(w) * u64::from(h) / 10_000).max(1) as f64).sqrt() as u32;
        let mut ch: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for y in (0..h).step_by(step.max(1) as usize) {
            for x in (0..w).step_by(step.max(1) as usize) {
                let p = image.get_pixel(x, y).0;
                for c in 0..3 {
                    ch[c].push(p[c]);
                }
            }
        }
        let background = ch.map(|mut v| {
            v.sort_unstable();
            v.get(v.len() / 2).copied().unwrap_or(255)
        });
        let (sx, sy) = match canvas {
            Some(c) if c.width > 0.0 && c.height > 0.0 => {
                (f64::from(w) / c.width, f64::from(h) / c.height)
            }
            _ => (1.0, 1.0),
        };
        let mut canvas = Self {
            image,
            background,
            sx,
            sy,
            tolerance: Self::DEFAULT_TOLERANCE,
            uniform: false,
        };
        canvas.uniform = canvas.measure_uniform();
        canvas
    }

    fn measure_uniform(&self) -> bool {
        let (w, h) = (self.image.width(), self.image.height());
        if w == 0 || h == 0 {
            return false;
        }
        let (mut sum, mut sq, mut n) = (0.0f64, 0.0f64, 0.0f64);
        for y in 0..h {
            for x in 0..w {
                let p = self.image.get_pixel(x, y).0;
                if (0..3).any(|c| p[c].abs_diff(self.background[c]) > self.tolerance) {
                    return false;
                }
                let l = 0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]);
                sum += l;
                sq += l * l;
                n += 1.0;
            }
        }
        let mean = sum / n;
        (sq / n - mean * mean).max(0.0).sqrt() <= Self::UNIFORM_MAX_STD
    }

    /// [`RegionProbe::uniform`] on this canvas.
    pub fn uniform(&self) -> bool {
        self.uniform
    }

    /// [`RegionProbe::stroke_between`] on this canvas: ink pixels outside the
    /// ignored boxes are pooled into cells of [`ProbeCanvas::STROKE_CELL`] pixels,
    /// and the cells holding ink in the band around `a` are flooded (8-connected)
    /// through ink cells; the stroke joins when the flood reaches a cell holding
    /// ink in the band around `b`.
    pub fn stroke_between(
        &self,
        a: &BBox,
        b: &BBox,
        region: &BBox,
        masks: &[BBox],
        ring: f64,
    ) -> Option<StrokeTrace> {
        type Rect = (f64, f64, f64, f64);
        let finite = |r: &BBox| [r.x1, r.y1, r.x2, r.y2].iter().all(|v| v.is_finite());
        if !(finite(a) && finite(b) && finite(region) && ring.is_finite()) {
            return None;
        }
        let px = |r: &BBox| -> Rect {
            (
                r.x1 * self.sx,
                r.y1 * self.sy,
                r.x2 * self.sx,
                r.y2 * self.sy,
            )
        };
        let grow = |r: Rect, d: f64| -> Rect { (r.0 - d, r.1 - d, r.2 + d, r.3 + d) };
        let inside = |r: &Rect, x: f64, y: f64| x >= r.0 && x < r.2 && y >= r.1 && y < r.3;
        let ring = (ring * (self.sx + self.sy) / 2.0).max(2.0);
        let (pa, pb) = (grow(px(a), Self::BOX_PAD), grow(px(b), Self::BOX_PAD));
        let (ra, rb) = (grow(pa, ring), grow(pb, ring));
        if ra.0 < rb.2 && rb.0 < ra.2 && ra.1 < rb.3 && rb.1 < ra.3 {
            return None;
        }
        let r = px(region);
        let (w, h) = (
            f64::from(self.image.width()),
            f64::from(self.image.height()),
        );
        let (x1, y1) = (r.0.max(0.0).floor(), r.1.max(0.0).floor());
        let (x2, y2) = (r.2.min(w).ceil(), r.3.min(h).ceil());
        let c = Self::STROKE_CELL;
        if x2 - x1 < (4 * c) as f64 || y2 - y1 < (4 * c) as f64 {
            return None;
        }
        let (x0, y0) = (x1 as usize, y1 as usize);
        let (ww, wh) = (x2 as usize - x0, y2 as usize - y0);
        // The ignored pixels of the window, rasterized once (pixel centers inside
        // a grown box).
        let mut ignored = vec![false; ww * wh];
        for m in masks
            .iter()
            .filter(|m| finite(m))
            .map(|m| grow(px(m), Self::BOX_PAD))
            .chain([pa, pb])
        {
            let lo = |v: f64, o: usize| ((v - 0.5).ceil().max(o as f64) as usize).saturating_sub(o);
            let (mx1, my1) = (lo(m.0, x0), lo(m.1, y0));
            let (mx2, my2) = (lo(m.2, x0).min(ww), lo(m.3, y0).min(wh));
            for y in my1..my2 {
                ignored[y * ww + mx1.min(mx2)..y * ww + mx2].fill(true);
            }
        }
        let cols = ww.div_ceil(c);
        let rows = wh.div_ceil(c);
        // Per cell: holds ink, ink in a's band, ink in b's band.
        let mut ink = vec![false; cols * rows];
        let mut near_a = vec![false; cols * rows];
        let mut near_b = vec![false; cols * rows];
        for y in y0..(y2 as usize) {
            for x in x0..(x2 as usize) {
                let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
                if ignored[(y - y0) * ww + (x - x0)] || !self.ink_at(x as f64, y as f64) {
                    continue;
                }
                let i = ((y - y0) / c) * cols + (x - x0) / c;
                ink[i] = true;
                near_a[i] |= inside(&ra, fx, fy);
                near_b[i] |= inside(&rb, fx, fy);
            }
        }
        let mut seen = near_a.clone();
        let mut stack: Vec<usize> = (0..seen.len()).filter(|&i| seen[i]).collect();
        let mut joined = false;
        while let Some(i) = stack.pop() {
            if near_b[i] {
                joined = true;
                break;
            }
            let (cx, cy) = ((i % cols) as isize, (i / cols) as isize);
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let (nx, ny) = (cx + dx, cy + dy);
                    if nx < 0 || ny < 0 || nx >= cols as isize || ny >= rows as isize {
                        continue;
                    }
                    let j = ny as usize * cols + nx as usize;
                    if ink[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        let share = ink.iter().filter(|&&v| v).count() as f64 / ink.len().max(1) as f64;
        Some(StrokeTrace { joined, ink: share })
    }

    fn ink_at(&self, x: f64, y: f64) -> bool {
        if x < 0.0 || y < 0.0 {
            return false;
        }
        let (x, y) = (x as u32, y as u32);
        if x >= self.image.width() || y >= self.image.height() {
            return false;
        }
        let p = self.image.get_pixel(x, y).0;
        (0..3).any(|c| p[c].abs_diff(self.background[c]) > self.tolerance)
    }

    /// [`RegionProbe::ink_share`] on this canvas.
    pub fn ink_share(&self, region: &BBox) -> Option<f64> {
        // A non-finite region measures nothing (it must never read as emptied);
        // checked before clamping, which would turn NaN into a bound.
        if ![region.x1, region.y1, region.x2, region.y2]
            .iter()
            .all(|v| v.is_finite())
        {
            return None;
        }
        let x1 = (region.x1 * self.sx).max(0.0).floor();
        let y1 = (region.y1 * self.sy).max(0.0).floor();
        let x2 = (region.x2 * self.sx)
            .min(f64::from(self.image.width()))
            .ceil();
        let y2 = (region.y2 * self.sy)
            .min(f64::from(self.image.height()))
            .ceil();
        if x2 - x1 < 2.0 || y2 - y1 < 2.0 {
            return None;
        }
        let (mut ink, mut all) = (0usize, 0usize);
        let mut y = y1;
        while y < y2 {
            let mut x = x1;
            while x < x2 {
                all += 1;
                if self.ink_at(x, y) {
                    ink += 1;
                }
                x += 1.0;
            }
            y += 1.0;
        }
        Some(ink as f64 / all.max(1) as f64)
    }

    /// [`RegionProbe::line_cover`] on this canvas: the middle 70 % of the segment
    /// is sampled every 1.5 px, and a sample is covered when ink lies within
    /// `half_width` of the line across it.
    pub fn line_cover(&self, a: (f64, f64), b: (f64, f64), half_width: f64) -> Option<f64> {
        let pa = (a.0 * self.sx, a.1 * self.sy);
        let pb = (b.0 * self.sx, b.1 * self.sy);
        let (dx, dy) = (pb.0 - pa.0, pb.1 - pa.1);
        let len = dx.hypot(dy);
        if !len.is_finite() || len < 8.0 {
            return None;
        }
        let (ux, uy) = (dx / len, dy / len);
        let (nx, ny) = (-uy, ux);
        let half = (half_width * (self.sx + self.sy) / 2.0).max(1.0).round() as i64;
        let samples = ((0.7 * len) / 1.5).ceil().max(4.0) as usize;
        let mut hits = 0usize;
        for i in 0..samples {
            let t = 0.15 + 0.7 * (i as f64 + 0.5) / samples as f64;
            let (cx, cy) = (pa.0 + dx * t, pa.1 + dy * t);
            if (-half..=half).any(|k| {
                let k = k as f64;
                self.ink_at(cx + nx * k, cy + ny * k)
            }) {
                hits += 1;
            }
        }
        Some(hits as f64 / samples as f64)
    }
}

/// Where [`CropProbe`] finds one keyframe's canvas.
#[derive(Debug, Clone)]
pub struct CropSource {
    /// Canvas crop image.
    pub path: PathBuf,
    /// Crop box in frame pixels (the image may be the whole frame).
    pub crop: Option<BBox>,
    /// Size of the reading's coordinate space.
    pub canvas: Option<CanvasDims>,
}

/// [`RegionProbe`] over the canvas crops, each loaded on first use. At most
/// [`CropProbe::MAX_LOADED`] decoded canvases are kept (least recently used out).
pub struct CropProbe {
    sources: HashMap<String, CropSource>,
    loaded: Mutex<Vec<(String, Option<Arc<ProbeCanvas>>)>>,
}

impl std::fmt::Debug for CropProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CropProbe")
            .field("keyframes", &self.sources.len())
            .finish()
    }
}

impl CropProbe {
    /// Probe over `sources` (by keyframe id).
    /// Decoded canvases kept at once. Probes query the keyframe that last read an
    /// item and the one missing it, so a small working set suffices.
    pub const MAX_LOADED: usize = 16;

    pub fn new(sources: HashMap<String, CropSource>) -> Self {
        Self {
            sources,
            loaded: Mutex::new(Vec::new()),
        }
    }

    fn canvas(&self, keyframe_id: &str) -> Option<Arc<ProbeCanvas>> {
        {
            let mut loaded = self.loaded.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(i) = loaded.iter().position(|(k, _)| k == keyframe_id) {
                let hit = loaded.remove(i);
                let c = hit.1.clone();
                loaded.push(hit);
                return c;
            }
        }
        // Decode outside the lock. An unreadable crop gives no pixel evidence
        // (never removal evidence).
        let c = self.sources.get(keyframe_id).and_then(|s| {
            load_canvas(&s.path, s.crop)
                .ok()
                .map(|img| Arc::new(ProbeCanvas::new(img, s.canvas)))
        });
        let mut loaded = self.loaded.lock().unwrap_or_else(PoisonError::into_inner);
        if !loaded.iter().any(|(k, _)| k == keyframe_id) {
            loaded.push((keyframe_id.to_string(), c.clone()));
            if loaded.len() > Self::MAX_LOADED {
                loaded.remove(0);
            }
        }
        c
    }
}

impl RegionProbe for CropProbe {
    fn ink_share(&self, keyframe_id: &str, region: &BBox) -> Option<f64> {
        self.canvas(keyframe_id)?.ink_share(region)
    }
    fn line_cover(
        &self,
        keyframe_id: &str,
        a: (f64, f64),
        b: (f64, f64),
        half_width: f64,
    ) -> Option<f64> {
        self.canvas(keyframe_id)?.line_cover(a, b, half_width)
    }
    fn stroke_between(
        &self,
        keyframe_id: &str,
        a: &BBox,
        b: &BBox,
        region: &BBox,
        masks: &[BBox],
        ring: f64,
    ) -> Option<StrokeTrace> {
        self.canvas(keyframe_id)?
            .stroke_between(a, b, region, masks, ring)
    }
    fn uniform(&self, keyframe_id: &str) -> Option<bool> {
        Some(self.canvas(keyframe_id)?.uniform())
    }
}

/// Consolidates the board keyframes into `glassrip.board_state`, one item per board.
pub struct BoardStateStage {
    params: ConsolidationParams,
    corroborator: Arc<dyn Corroborator>,
    second_reader: Option<Arc<dyn SecondReader>>,
}

impl BoardStateStage {
    /// Stage without transcript corroboration or a second reader pass.
    pub fn new(params: ConsolidationParams) -> Self {
        Self {
            params,
            corroborator: Arc::new(NoCorroboration),
            second_reader: None,
        }
    }

    /// Stage with a corroborator (for example transcript cues).
    pub fn with_corroborator(mut self, corroborator: Arc<dyn Corroborator>) -> Self {
        self.corroborator = corroborator;
        self
    }

    /// Stage with a second reader pass for single sightings.
    pub fn with_second_reader(mut self, reader: Arc<dyn SecondReader>) -> Self {
        self.second_reader = Some(reader);
        self
    }
}

/// OCR spans of one keyframe mapped into the reading's canvas coordinates. Needs the
/// crop box in frame pixels; spans outside the canvas (by region, or by position when
/// no region is recorded) are dropped.
pub fn ocr_anchors(ocr: &OcrView, dirs: Option<&EdgeDirectionItem>) -> Vec<TextAnchor> {
    let Some(d) = dirs else { return Vec::new() };
    let Some(crop) = d.crop_in_frame else {
        return Vec::new();
    };
    if !crop.is_well_formed() || d.canvas.width <= 0.0 || d.canvas.height <= 0.0 {
        return Vec::new();
    }
    let (fx, fy) = (
        d.canvas.width / crop.width(),
        d.canvas.height / crop.height(),
    );
    ocr.spans
        .iter()
        .filter(|s| {
            // Canvas spans, and spans the crop stage left unassigned when they lie
            // inside the crop; chrome and tile names never anchor.
            let inside = s.bbox.x1 >= crop.x1
                && s.bbox.y1 >= crop.y1
                && s.bbox.x2 <= crop.x2
                && s.bbox.y2 <= crop.y2;
            match s.region.as_deref() {
                Some("canvas") => true,
                Some("unassigned") | None => inside,
                Some(_) => false,
            }
        })
        .filter(|s| s.bbox.is_well_formed() && !normalize(&s.text).is_empty())
        .map(|s| TextAnchor {
            text: s.text.clone(),
            bbox: BBox::new(
                (s.bbox.x1 - crop.x1) * fx,
                (s.bbox.y1 - crop.y1) * fy,
                (s.bbox.x2 - crop.x1) * fx,
                (s.bbox.y2 - crop.y1) * fy,
            ),
        })
        .collect()
}

/// Build consolidation frames from upstream items. Board keyframes missing from
/// `keyframes` are dropped (no times). A board keyframe's ink change is its own
/// boundary's only when the previous keyframe is the previous board keyframe and the
/// boundary's alignment was usable ([`crate::artifacts::BoundaryView::measured_ink`]);
/// otherwise the pair's ink is unknown.
pub fn board_frames(
    keyframes: &[KeyframeView],
    boards: Vec<BoardItem>,
    directions: Vec<EdgeDirectionItem>,
    ocr: Vec<OcrView>,
) -> Vec<BoardFrame> {
    let mut kf: Vec<&KeyframeView> = keyframes.iter().collect();
    kf.sort_by(|a, b| a.t_start_s.total_cmp(&b.t_start_s));
    let index: HashMap<&str, usize> = kf
        .iter()
        .enumerate()
        .map(|(i, k)| (k.keyframe_id.as_str(), i))
        .collect();
    let mut dirs: HashMap<String, EdgeDirectionItem> = directions
        .into_iter()
        .map(|d| (d.keyframe_id.clone(), d))
        .collect();
    let ocr: HashMap<String, OcrView> = ocr
        .into_iter()
        .filter_map(|o| o.keyframe_id.clone().map(|k| (k, o)))
        .collect();
    let mut boards: Vec<(usize, BoardItem)> = boards
        .into_iter()
        .filter_map(|b| index.get(b.keyframe_id.as_str()).map(|&i| (i, b)))
        .collect();
    boards.sort_by_key(|b| b.0);
    let mut out = Vec::new();
    let mut prev: Option<usize> = None;
    for (i, b) in boards {
        let own = kf[i].boundary.as_ref().and_then(|x| x.measured_ink());
        let ink_change = match prev {
            None => own,
            Some(p) if p + 1 == i => own,
            Some(_) => None,
        };
        prev = Some(i);
        let d = dirs.remove(&b.keyframe_id);
        let anchors = ocr
            .get(&b.keyframe_id)
            .map(|o| ocr_anchors(o, d.as_ref()))
            .unwrap_or_default();
        // App panel text (board title bar, board list) names the board, not content.
        let title_hints = ocr
            .get(&b.keyframe_id)
            .map(|o| {
                o.spans
                    .iter()
                    .filter(|s| s.chrome_reason.as_deref() == Some("app_panel"))
                    .map(|s| s.text.clone())
                    .collect()
            })
            .unwrap_or_default();
        out.push(BoardFrame {
            keyframe_index: i,
            t_start_s: kf[i].t_start_s,
            t_end_s: kf[i].t_end_s,
            t_rep_s: kf[i].t_rep_s,
            canvas: b.canvas,
            board_title: b.board_title,
            directions: d,
            keyframe_id: b.keyframe_id,
            board: b.board,
            ink_change,
            ocr_anchors: anchors,
            title_hints,
        });
    }
    out
}

impl Stage for BoardStateStage {
    type Params = ConsolidationParams;
    type Work = BoardStateWork;
    type Output = BoardStateItem;

    fn name(&self) -> &'static str {
        "board_state"
    }
    fn version(&self) -> u32 {
        // 3: final state is "observed and not later removed", with coverage evidence.
        // 4: canvas pixels decide removals (connector corridors, emptied boxes), and
        // ink from a failed alignment is unknown.
        // 5: routed connectors traced on the canvas with other elements masked, owner
        // events carry structured targets, blank views chain through flat links.
        5
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: BOARD_STATE,
            version: Version::new(1, 1, 0),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![
            InputDecl {
                schema: BOARD_VALIDATE,
                major: 1,
            },
            InputDecl {
                schema: EDGE_DIRECTION,
                major: 1,
            },
            InputDecl {
                schema: KEYFRAMES,
                major: 1,
            },
            InputDecl {
                schema: OCR,
                major: 1,
            },
            InputDecl {
                schema: CANVAS_CROP,
                major: 1,
            },
        ]
    }
    fn params(&self) -> &ConsolidationParams {
        &self.params
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<BoardStateWork>>, StageError> {
        let keyframes: Vec<KeyframeView> = inputs
            .read_ok::<KeyframeView>(KEYFRAMES)?
            .into_iter()
            .map(|(_, k)| k)
            .collect();
        let boards = inputs
            .read_ok::<ValidateItemView>(BOARD_VALIDATE)?
            .into_iter()
            .map(|(id, v)| v.into_item(&id))
            .filter(|b| !b.board.needs_reclassification)
            .collect();
        let directions = inputs
            .read_ok::<EdgeDirectionBatch>(EDGE_DIRECTION)?
            .into_iter()
            .flat_map(|(_, d)| d.keyframes)
            .collect();
        let ocr = inputs
            .read_ok::<OcrView>(OCR)?
            .into_iter()
            .map(|(id, mut o)| {
                o.keyframe_id.get_or_insert(id);
                o
            })
            .collect();
        let frames = board_frames(&keyframes, boards, directions, ocr);
        let root = run_root(inputs, CANVAS_CROP)?;
        let mut crops: HashMap<String, (PathBuf, Option<BBox>)> = inputs
            .read_ok::<CanvasCropView>(CANVAS_CROP)?
            .into_iter()
            .map(|(id, c)| {
                (
                    c.keyframe_id.unwrap_or(id),
                    (resolve(&root, &c.path), c.crop),
                )
            })
            .collect();
        let sources: HashMap<String, CropSource> = frames
            .iter()
            .filter_map(|f| {
                let (path, crop) = crops.remove(&f.keyframe_id)?;
                let canvas = f
                    .canvas
                    .or_else(|| f.directions.as_ref().map(|d| d.canvas))
                    .filter(|c| c.width > 0.0 && c.height > 0.0);
                Some((f.keyframe_id.clone(), CropSource { path, crop, canvas }))
            })
            .collect();
        let probe = Arc::new(CropProbe::new(sources));
        Ok(split_boards(frames, &self.params)
            .into_iter()
            .enumerate()
            .map(|(i, frames)| WorkItem {
                id: format!("board-{}", i + 1),
                work: BoardStateWork {
                    frames,
                    probe: Some(probe.clone()),
                },
            })
            .collect())
    }
    async fn process(
        &self,
        ctx: &ItemContext,
        work: BoardStateWork,
    ) -> Result<BoardStateItem, ErrorInfo> {
        let hooks = Hooks {
            corroborator: self.corroborator.as_ref(),
            second_reader: self.second_reader.as_deref(),
        };
        Ok(consolidate_with_probe(
            work.frames,
            ctx.item_id(),
            &self.params,
            &hooks,
            work.probe.as_deref().map(|p| p as &dyn RegionProbe),
        ))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::artifacts::{BoundaryView, OcrSpanView};
    use glassrip_vision::board::BoardNode;

    fn board(nodes: &[(f64, f64, f64, f64)]) -> ValidatedBoard {
        ValidatedBoard {
            nodes: nodes
                .iter()
                .enumerate()
                .map(|(i, b)| BoardNode {
                    local_id: format!("n{i}"),
                    text: format!("Node {i}"),
                    bbox: BBox::new(b.0, b.1, b.2, b.3),
                    conf: 0.9,
                })
                .collect(),
            edges: vec![],
            stickies: vec![],
            owner_tags: vec![],
            other_visible_text: vec![],
            confidence: 0.9,
            chrome_rejected: vec![],
            issues: vec![],
            needs_reclassification: false,
        }
    }

    #[test]
    fn coordinate_check_cases() {
        let b = board(&[(10.0, 10.0, 100.0, 60.0)]);
        let dims = |w, h| {
            Some(CanvasDims {
                width: w,
                height: h,
            })
        };
        assert_eq!(
            coordinate_check(&b, dims(800.0, 600.0), 800.0, 600.0).0,
            CoordinateCheck::Verified1to1
        );
        let (c, s) = coordinate_check(&b, dims(400.0, 300.0), 800.0, 600.0);
        assert_eq!((c, s.x, s.y), (CoordinateCheck::Scaled, 2.0, 2.0));
        assert_eq!(
            coordinate_check(&b, None, 800.0, 600.0).0,
            CoordinateCheck::Assumed1to1
        );
        assert_eq!(
            coordinate_check(&b, None, 80.0, 60.0).0,
            CoordinateCheck::Inconsistent
        );
    }

    fn kf(id: &str, t: f64, ink: Option<f64>) -> KeyframeView {
        KeyframeView {
            keyframe_id: id.into(),
            t_start_s: t,
            t_end_s: t + 10.0,
            t_rep_s: t + 5.0,
            boundary: Some(BoundaryView {
                ink_change: ink,
                ..BoundaryView::default()
            }),
        }
    }

    fn item(id: &str) -> BoardItem {
        BoardItem {
            keyframe_id: id.into(),
            canvas: None,
            board_title: None,
            board: board(&[(10.0, 10.0, 100.0, 60.0)]),
        }
    }

    #[test]
    fn ink_is_attributed_per_adjacent_board_pair_only() {
        let kfs = [
            kf("a", 0.0, Some(0.2)),
            kf("b", 10.0, Some(0.01)),
            kf("x", 20.0, Some(0.9)), // not a board keyframe
            kf("c", 30.0, Some(0.02)),
        ];
        let frames = board_frames(&kfs, vec![item("a"), item("b"), item("c")], vec![], vec![]);
        let ink: Vec<Option<f64>> = frames.iter().map(|f| f.ink_change).collect();
        // "c" follows a non-board keyframe: its pair ink with "b" is unknown, and the
        // unrelated change at "x" is not attributed to it.
        assert_eq!(ink, vec![Some(0.2), Some(0.01), None]);
    }

    #[test]
    fn ink_from_a_failed_alignment_is_unknown_not_a_change() {
        // A pan the aligner could not follow reports ink 1.0 so that segmentation
        // cuts there; it measures no ink and must not reach board state as a change.
        let mut kfs = vec![
            kf("a", 0.0, None),
            kf("b", 10.0, Some(1.0)),
            kf("c", 20.0, Some(1.0)),
            kf("d", 30.0, Some(0.2)),
        ];
        kfs[1].boundary.as_mut().unwrap().ink_align_ok = Some(false);
        kfs[2].boundary.as_mut().unwrap().align_ok = Some(false);
        kfs[3].boundary.as_mut().unwrap().ink_align_ok = Some(true);
        let frames = board_frames(
            &kfs,
            vec![item("a"), item("b"), item("c"), item("d")],
            vec![],
            vec![],
        );
        let ink: Vec<Option<f64>> = frames.iter().map(|f| f.ink_change).collect();
        assert_eq!(ink, vec![None, None, None, Some(0.2)]);
    }

    #[test]
    fn ocr_spans_map_into_canvas_coordinates() {
        let d = EdgeDirectionItem {
            keyframe_id: "a".into(),
            canvas: CanvasDims {
                width: 500.0,
                height: 250.0,
            },
            board_to_image: AxisScale { x: 1.0, y: 1.0 },
            coordinates: CoordinateCheck::Verified1to1,
            crop_in_frame: Some(BBox::new(100.0, 50.0, 1100.0, 550.0)),
            sharpness: 1.0,
            zoom: 1.0,
            edges: vec![],
            error: None,
        };
        let span = |text: &str, b: BBox, region: Option<&str>| OcrSpanView {
            text: text.into(),
            bbox: b,
            confidence: Some(0.9),
            region: region.map(String::from),
            chrome_reason: None,
        };
        let ocr = OcrView {
            keyframe_id: Some("a".into()),
            spans: vec![
                span("inside", BBox::new(300.0, 150.0, 400.0, 170.0), None),
                span("outside", BBox::new(0.0, 0.0, 50.0, 20.0), None),
                span(
                    "tile name",
                    BBox::new(300.0, 150.0, 400.0, 170.0),
                    Some("tile"),
                ),
            ],
        };
        let a = ocr_anchors(&ocr, Some(&d));
        assert_eq!(a.len(), 1);
        // Crop 1000x500 frame pixels onto a 500x250 canvas: half scale, origin shifted.
        assert_eq!(a[0].bbox, BBox::new(100.0, 50.0, 150.0, 60.0));
        assert!(ocr_anchors(&ocr, None).is_empty());
    }

    #[test]
    fn probe_canvas_measures_connector_corridors_and_emptied_boxes() {
        // White 400 x 200 canvas at half the reading's scale: two box outlines, a
        // 2 px connector between them, and a filled card.
        let mut img = RgbImage::from_pixel(400, 200, image::Rgb([250, 250, 250]));
        let ink = image::Rgb([40, 40, 40]);
        for x in 20..80 {
            for y in [40u32, 90] {
                img.put_pixel(x, y, ink);
                img.put_pixel(x + 280, y, ink);
            }
        }
        for x in 80..300 {
            img.put_pixel(x, 65, ink);
            img.put_pixel(x, 66, ink);
        }
        for x in 150..230 {
            for y in 130..180 {
                img.put_pixel(x, y, image::Rgb([250, 220, 90]));
            }
        }
        let canvas = Some(CanvasDims {
            width: 800.0,
            height: 400.0,
        });
        let drawn = ProbeCanvas::new(img.clone(), canvas);
        let (a, b) = ((160.0, 130.0), (560.0, 130.0));
        let cover = drawn.line_cover(a, b, 8.0).unwrap();
        assert!(cover > 0.95, "{cover}");
        let card = BBox::new(300.0, 260.0, 460.0, 360.0);
        let full = drawn.ink_share(&card).unwrap();
        assert!(full > 0.9, "{full}");
        // Erase the connector and the card.
        for x in 80..300 {
            img.put_pixel(x, 65, image::Rgb([250, 250, 250]));
            img.put_pixel(x, 66, image::Rgb([250, 250, 250]));
        }
        for x in 150..230 {
            for y in 130..180 {
                img.put_pixel(x, y, image::Rgb([250, 250, 250]));
            }
        }
        let erased = ProbeCanvas::new(img, canvas);
        assert!(erased.line_cover(a, b, 8.0).unwrap() < 0.05);
        assert!(erased.ink_share(&card).unwrap() < 0.01);
        // Degenerate queries give no evidence.
        assert!(erased.line_cover(a, (162.0, 131.0), 8.0).is_none());
        assert!(erased
            .ink_share(&BBox::new(10.0, 10.0, 11.0, 11.0))
            .is_none());
        assert!(erased
            .ink_share(&BBox::new(f64::NAN, 10.0, 200.0, 200.0))
            .is_none());
        assert!(erased.line_cover((f64::NAN, 0.0), b, 8.0).is_none());
    }
}
