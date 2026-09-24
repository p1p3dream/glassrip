//! `edge_direction` and `board_state` as [`glassrip_core::runner::Stage`]s.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    EdgeDirectionItem, EdgeEvidence, KeyframeView, OcrView, ValidateItemView, VlmFallback,
    BOARD_STATE, BOARD_VALIDATE, CANVAS_CROP, EDGE_DIRECTION, KEYFRAMES, OCR,
};
use crate::consolidate::owners::{Corroborator, NoCorroboration};
use crate::consolidate::{
    consolidate, split_boards, BoardFrame, BoardStateItem, ConsolidationParams, Hooks,
    SecondReader, TextAnchor,
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
    let bgr = bgr_from_rgb(image);
    let nodes: Vec<BBox> = board.nodes.iter().map(|n| n.bbox).collect();
    let texts: Vec<BBox> = board
        .stickies
        .iter()
        .map(|s| s.bbox)
        .chain(board.owner_tags.iter().map(|o| o.bbox))
        .chain(board.other_visible_text.iter().map(|t| t.bbox))
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
            label: label_box(board, &e.label),
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
    (out, sharpness(&bgr), zoom)
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
    let image = match image::open(path) {
        Ok(i) => i.to_rgb8(),
        Err(e) => return empty(format!("{}: {e}", path.display())),
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
    let (mut edges, sharp, zoom) = pixel_evidence(&image, &board, params);
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
                let image = match image::open(path) {
                    Ok(i) => i.to_rgb8(),
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
        1
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
        let frames = inputs
            .read_ok::<ValidateItemView>(BOARD_VALIDATE)?
            .into_iter()
            .map(|(id, v)| {
                let item = v.into_item(&id);
                let crop = crops.get(&item.keyframe_id);
                FrameWork {
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
        .filter(|s| match s.region.as_deref() {
            Some(r) => r == "canvas",
            None => {
                s.bbox.x1 >= crop.x1
                    && s.bbox.y1 >= crop.y1
                    && s.bbox.x2 <= crop.x2
                    && s.bbox.y2 <= crop.y2
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
/// boundary's only when the previous keyframe is the previous board keyframe;
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
        let own = kf[i].boundary.as_ref().and_then(|x| x.ink_change);
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
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: BOARD_STATE,
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
        Ok(split_boards(frames, &self.params)
            .into_iter()
            .enumerate()
            .map(|(i, frames)| WorkItem {
                id: format!("board-{}", i + 1),
                work: BoardStateWork { frames },
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
        Ok(consolidate(
            work.frames,
            ctx.item_id(),
            &self.params,
            &hooks,
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
            boundary: Some(BoundaryView { ink_change: ink }),
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
}
