//! `board_read`: one schema-constrained reading per whiteboard canvas (spec 6.10).
//!
//! - One image per request. The canvas is cut from the native keyframe, chrome
//!   masks are painted over, and the size rule applies: a long edge of at least
//!   1280 px is resized to 1920 px; a smaller crop is upscaled 1.5x and marked
//!   `low_res` (tiling cannot add detail the source lacks).
//! - Tiling: when the median canvas text height from OCR is under the threshold
//!   (and the crop is not low resolution), the overview request is joined by
//!   `grid x grid` tile requests with overlap, each resized to 1920 px long edge.
//!   Nodes, stickies, owner tags, and other text come from the tiles, merged in
//!   canvas coordinates by IoU or normalized-text match; edges come from the
//!   overview only.
//! - Requests go through the shared [`crate::placement::PlacementMonitor`]
//!   (bounded concurrency, preflight, mid-run placement checks).
//! - Raw responses are recorded by the backend under their request keys, which
//!   are listed per reading for offline replay.
//! - A complete reply that lists one text at many places (see
//!   [`glassrip_vision::degenerate`], judged against the keyframe's OCR spans)
//!   is retried once with the repeat-penalty retry. When no retry is left, the
//!   retry fails, or its reply is degenerate too, the reply is kept with each
//!   repeated text reduced to its best-supported copies, and the request log
//!   records it as a warning.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use glassrip_core::envelope::ErrorInfo;
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, KeyExtras, Stage, StageError, StageInputs, WorkItem,
};
use glassrip_vision::board::{
    board_read_request, board_read_request_compact, normalize, BoardNode, BoardReadOutput,
    BoardReading, OwnerTag, Sticky, TextItem, BOARD_READ_PROMPT, COMPACT_RETRY_NOTE,
    COMPACT_RETRY_SCALE,
};
use glassrip_vision::degenerate::{self, DegenerateParams};
use glassrip_vision::image_prep::{
    prepare_board_image_with, prepare_long_edge, BoardSizing, BOARD_LONG_EDGE, LOW_RES_THRESHOLD,
    LOW_RES_UPSCALE,
};
use glassrip_vision::repetition::{detect, RepetitionFinding, RepetitionParams};
use glassrip_vision::{BBox, GenerationOptions, PreparedImage, SamplingOverrides, VisionRequest};
use image::{DynamicImage, RgbImage};
use schemars::JsonSchema;
use serde::Serialize;

use crate::artifacts::{
    self, BoardReadingItem, CanvasCropItem, DegenerateLog, ModelRef, OcrKeyframe, OcrSpan,
    RequestLog, RequestRole, TextRegion,
};
use crate::pixels;
use crate::placement::{vision_error_info, InferFailure, PlacementMonitor};
use crate::raw_store::request_key;
use crate::stages::{input, internal, load_rgb, text_hash};

/// Parameters.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BoardReadParams {
    pub seed: u64,
    /// Fixed context size of every request in the run (must match the backend).
    pub num_ctx: u32,
    /// Output budget per request: what `num_ctx` leaves after the image and
    /// prompt, clamped to this range.
    pub min_num_predict: u32,
    pub max_num_predict: u32,
    /// Tokens left unused in `num_ctx`: the prompt size is an estimate, and a
    /// request that overflows would silently lose context.
    pub ctx_headroom: u32,
    pub target_long_edge_px: u32,
    pub min_long_edge_px: u32,
    pub low_res_upscale: f64,
    pub tiling_text_height_px: f64,
    pub tile_grid: u32,
    pub tile_overlap: f64,
    /// Two elements are one when their boxes overlap at least this much...
    pub merge_iou: f64,
    /// ...or their normalized texts are at least this similar and their centers are close.
    pub merge_text_ratio: f64,
    /// A reply cut off at the output limit is retried once with every list
    /// budget (`maxItems`) scaled by this and a compact single-line instruction.
    pub compact_retry_scale: f64,
    /// Repetition-loop detector applied to every reply (streamed and returned).
    pub repetition: RepetitionParams,
    /// A reply that loops is retried once with this `repeat_penalty`: stronger
    /// than the server default (1.1)...
    pub repetition_retry_penalty: f32,
    /// ...over this many recent tokens (`repeat_last_n`; the server default of 64
    /// is shorter than one board list item, which runs 40 to 60 tokens)...
    pub repetition_retry_last_n: u32,
    /// ...and with the output budget scaled by this, so a retry that loops again
    /// costs less.
    pub repetition_retry_predict_share: f64,
    /// A complete reply repeating one text at many places in a list takes the
    /// repeat-penalty retry; a reply still degenerate is collapsed.
    pub degenerate: DegenerateParams,
}

impl Default for BoardReadParams {
    fn default() -> Self {
        Self {
            seed: 0,
            num_ctx: glassrip_vision::ollama::DEFAULT_NUM_CTX,
            // Dense boards (a 3x3 card grid plus stickies) truncated at 2,048
            // output tokens, and dense real boards at 3,072 while still emitting
            // valid JSON; the budget is sized per request to fill num_ctx, so
            // this cap only takes effect when num_ctx leaves room for it.
            min_num_predict: 1024,
            max_num_predict: 6144,
            ctx_headroom: 512,
            target_long_edge_px: BOARD_LONG_EDGE,
            min_long_edge_px: LOW_RES_THRESHOLD,
            low_res_upscale: LOW_RES_UPSCALE,
            tiling_text_height_px: 14.0,
            tile_grid: 2,
            tile_overlap: 0.12,
            merge_iou: 0.5,
            merge_text_ratio: 0.85,
            compact_retry_scale: COMPACT_RETRY_SCALE,
            repetition: RepetitionParams::default(),
            repetition_retry_penalty: 1.2,
            repetition_retry_last_n: 512,
            repetition_retry_predict_share: 0.75,
            degenerate: DegenerateParams::default(),
        }
    }
}

/// The retry of a looping board read: the same request with a repeat penalty over
/// a long window and a smaller output budget (never under `min_num_predict`).
pub fn repetition_retry(p: &BoardReadParams, request: &VisionRequest) -> VisionRequest {
    let mut retry = request.clone();
    retry.sampling = SamplingOverrides {
        repeat_penalty: Some(p.repetition_retry_penalty),
        repeat_last_n: Some(p.repetition_retry_last_n),
    };
    let scaled =
        (f64::from(request.options.num_predict) * p.repetition_retry_predict_share).floor() as u32;
    retry.options.num_predict = scaled.max(p.min_num_predict.min(request.options.num_predict));
    retry.repetition_guard = Some(p.repetition);
    retry
}

/// A failed retry that the raw store records as the model's answer (stopped at
/// the output limit or in a loop), by a stable label; `None` for every other
/// failure, an invalid reply included (the store keeps no failed reply, so a
/// replay could not take the same path).
fn settled_retry_failure(f: &InferFailure) -> Option<&'static str> {
    match f {
        InferFailure::Truncated(_) => Some("output limit"),
        InferFailure::Repetition(..) => Some("repetition loop"),
        InferFailure::Other(_) => None,
    }
}

/// Output tokens left by `num_ctx` after the image, the prompt (estimated at
/// 3 bytes per token), message framing, and `ctx_headroom`.
pub fn output_budget(p: &BoardReadParams, request: &glassrip_vision::VisionRequest) -> u32 {
    let used = request.image.tokens()
        + glassrip_vision::ollama::estimate_text_tokens(&request.prompt)
        + glassrip_vision::ollama::MESSAGE_OVERHEAD_TOKENS
        + p.ctx_headroom;
    p.num_ctx
        .saturating_sub(used)
        .clamp(p.min_num_predict, p.max_num_predict.max(p.min_num_predict))
}

/// OCR spans on the canvas, in the pixels of [`canvas_image`] (the reading's
/// coordinates): spans `canvas_crop` assigned to the canvas (or, without its
/// per-span regions, spans OCR left on the canvas or unassigned) whose center
/// lies inside the crop.
pub fn canvas_anchors(spans: &[OcrSpan], crop: &CanvasCropItem) -> Vec<(String, BBox)> {
    let c = crop.canvas_bbox;
    let (x0, y0) = (c.x1.max(0.0).floor(), c.y1.max(0.0).floor());
    let final_regions = crop.span_regions.len() == spans.len();
    spans
        .iter()
        .enumerate()
        .filter(|(i, s)| {
            if final_regions {
                crop.span_regions[*i] == TextRegion::Canvas
            } else {
                matches!(s.region, TextRegion::Canvas | TextRegion::Unassigned)
            }
        })
        .filter(|(_, s)| {
            let (x, y) = ((s.bbox.x1 + s.bbox.x2) / 2.0, (s.bbox.y1 + s.bbox.y2) / 2.0);
            x >= c.x1 && x <= c.x2 && y >= c.y1 && y <= c.y2
        })
        .map(|(_, s)| (s.text.clone(), offset(&s.bbox, -x0, -y0)))
        .collect()
}

/// Crop the canvas out of the keyframe and paint the chrome masks.
pub fn canvas_image(img: &RgbImage, crop: &CanvasCropItem) -> Option<RgbImage> {
    let b = crop.canvas_bbox;
    let x0 = b.x1.max(0.0) as u32;
    let y0 = b.y1.max(0.0) as u32;
    let x1 = (b.x2.max(0.0) as u32).min(img.width());
    let y1 = (b.y2.max(0.0) as u32).min(img.height());
    if x1 <= x0 + 8 || y1 <= y0 + 8 {
        return None;
    }
    let mut c = image::imageops::crop_imm(img, x0, y0, x1 - x0, y1 - y0).to_image();
    let fill = pixels::border_median(&c);
    let rel: Vec<BBox> = crop
        .masks
        .iter()
        .map(|m| {
            BBox::new(
                m.bbox.x1 - f64::from(x0),
                m.bbox.y1 - f64::from(y0),
                m.bbox.x2 - f64::from(x0),
                m.bbox.y2 - f64::from(y0),
            )
        })
        .collect();
    pixels::paint_masks(&mut c, &rel, fill);
    Some(c)
}

/// Tile rectangles (canvas pixels) for a `grid x grid` split with overlap.
pub fn tile_regions(w: f64, h: f64, grid: u32, overlap: f64) -> Vec<BBox> {
    let g = f64::from(grid.max(1));
    let tw = w / (g - (g - 1.0) * overlap);
    let th = h / (g - (g - 1.0) * overlap);
    let (sx, sy) = (tw * (1.0 - overlap), th * (1.0 - overlap));
    let mut out = Vec::new();
    for r in 0..grid.max(1) {
        for c in 0..grid.max(1) {
            let x1 = (f64::from(c) * sx).floor();
            let y1 = (f64::from(r) * sy).floor();
            out.push(BBox::new(
                x1,
                y1,
                (x1 + tw).ceil().min(w),
                (y1 + th).ceil().min(h),
            ));
        }
    }
    out
}

fn offset(b: &BBox, dx: f64, dy: f64) -> BBox {
    BBox::new(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy)
}

fn same(a: (&str, &BBox), b: (&str, &BBox), iou: f64, ratio: f64) -> bool {
    if a.1.iou(b.1) >= iou {
        return true;
    }
    let sim = strsim::normalized_levenshtein(&normalize(a.0), &normalize(b.0));
    let (ca, cb) = (
        ((a.1.x1 + a.1.x2) / 2.0, (a.1.y1 + a.1.y2) / 2.0),
        ((b.1.x1 + b.1.x2) / 2.0, (b.1.y1 + b.1.y2) / 2.0),
    );
    let reach =
        a.1.width()
            .max(a.1.height())
            .max(b.1.width().max(b.1.height()));
    sim >= ratio && ((ca.0 - cb.0).powi(2) + (ca.1 - cb.1).powi(2)).sqrt() <= reach
}

/// Merge `(key, text, bbox)` elements: the larger box (the more complete read) wins.
fn merge_list<T: Clone>(
    items: Vec<T>,
    key: impl Fn(&T) -> (&str, &BBox),
    iou: f64,
    ratio: f64,
) -> (Vec<T>, Vec<usize>) {
    let mut kept: Vec<T> = Vec::new();
    let mut map = Vec::with_capacity(items.len());
    for it in items {
        match kept.iter().position(|k| same(key(k), key(&it), iou, ratio)) {
            Some(i) => {
                if key(&it).1.area() > key(&kept[i]).1.area() {
                    kept[i] = it;
                }
                map.push(i);
            }
            None => {
                kept.push(it);
                map.push(kept.len() - 1);
            }
        }
    }
    (kept, map)
}

/// Merge tile readings (already in canvas pixels) with the overview.
pub fn merge_tiles(
    overview: BoardReading,
    tiles: Vec<BoardReading>,
    iou: f64,
    ratio: f64,
) -> BoardReading {
    // Nodes: tiles first, then overview nodes not seen in any tile.
    let mut all_nodes: Vec<(Option<usize>, BoardNode)> = Vec::new();
    for (t, r) in tiles.iter().enumerate() {
        all_nodes.extend(r.nodes.iter().cloned().map(|n| (Some(t), n)));
    }
    all_nodes.extend(overview.nodes.iter().cloned().map(|n| (None, n)));
    let (kept, map) = merge_list(
        all_nodes.clone(),
        |(_, n)| (n.text.as_str(), &n.bbox),
        iou,
        ratio,
    );
    let mut id_map: HashMap<(Option<usize>, String), String> = HashMap::new();
    for ((src, n), &k) in all_nodes.iter().zip(&map) {
        id_map.insert((*src, n.local_id.clone()), format!("n{}", k + 1));
    }
    let nodes: Vec<BoardNode> = kept
        .into_iter()
        .enumerate()
        .map(|(i, (_, mut n))| {
            n.local_id = format!("n{}", i + 1);
            n
        })
        .collect();
    let edges = overview
        .edges
        .iter()
        .filter_map(|e| {
            let src = id_map.get(&(None, e.src.clone()))?.clone();
            let dst = id_map.get(&(None, e.dst.clone()))?.clone();
            let mut e = e.clone();
            e.src = src;
            e.dst = dst;
            Some(e)
        })
        .collect();
    let mut stickies_in: Vec<Sticky> = tiles.iter().flat_map(|r| r.stickies.clone()).collect();
    stickies_in.extend(overview.stickies.clone());
    let (stickies, _) = merge_list(stickies_in, |s| (s.text.as_str(), &s.bbox), iou, ratio);
    let mut owners_in: Vec<(Option<usize>, OwnerTag)> = Vec::new();
    for (t, r) in tiles.iter().enumerate() {
        owners_in.extend(r.owner_tags.iter().cloned().map(|o| (Some(t), o)));
    }
    owners_in.extend(overview.owner_tags.iter().cloned().map(|o| (None, o)));
    let owners_in: Vec<OwnerTag> = owners_in
        .into_iter()
        .map(|(src, mut o)| {
            o.near = id_map
                .get(&(src, o.near.clone()))
                .cloned()
                .unwrap_or_default();
            o
        })
        .collect();
    let (owner_tags, _) = merge_list(owners_in, |o| (o.name_raw.as_str(), &o.bbox), iou, ratio);
    let mut other_in: Vec<TextItem> = tiles
        .iter()
        .flat_map(|r| r.other_visible_text.clone())
        .collect();
    other_in.extend(overview.other_visible_text.clone());
    let (other_visible_text, _) = merge_list(other_in, |t| (t.text.as_str(), &t.bbox), iou, ratio);
    BoardReading {
        nodes,
        edges,
        stickies,
        owner_tags,
        other_visible_text,
        confidence: overview.confidence,
    }
}

fn shift_output(mut o: BoardReading, dx: f64, dy: f64) -> BoardReading {
    for n in &mut o.nodes {
        n.bbox = offset(&n.bbox, dx, dy);
    }
    for s in &mut o.stickies {
        s.bbox = offset(&s.bbox, dx, dy);
    }
    for t in &mut o.owner_tags {
        t.bbox = offset(&t.bbox, dx, dy);
    }
    for t in &mut o.other_visible_text {
        t.bbox = offset(&t.bbox, dx, dy);
    }
    for e in &mut o.edges {
        e.label_bbox = e.label_bbox.map(|b| offset(&b, dx, dy));
    }
    o
}

/// Work: the canvas crop and its OCR spans in canvas pixels.
#[derive(Debug, Clone)]
pub struct ReadWork {
    pub crop: CanvasCropItem,
    pub anchors: Vec<(String, BBox)>,
}

/// The stage.
pub struct BoardReadStage {
    params: BoardReadParams,
    monitor: Arc<PlacementMonitor>,
    model: String,
    digest: Option<String>,
    server_version: Option<String>,
}

impl BoardReadStage {
    pub fn new(
        params: BoardReadParams,
        monitor: Arc<PlacementMonitor>,
        model: impl Into<String>,
        digest: Option<String>,
        server_version: Option<String>,
    ) -> Self {
        Self {
            params,
            monitor,
            model: model.into(),
            digest,
            server_version,
        }
    }

    async fn one(
        &self,
        role: RequestRole,
        region: BBox,
        prepared: PreparedImage,
        anchors: &[(String, BBox)],
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<(BoardReading, RequestLog), ErrorInfo> {
        let mut request = board_read_request(
            &prepared,
            GenerationOptions {
                seed: self.params.seed,
                num_predict: self.params.min_num_predict,
            },
        )
        .map_err(|e| vision_error_info(&e))?;
        request.options.num_predict = output_budget(&self.params, &request);
        request.repetition_guard = Some(self.params.repetition);
        let mut key = request_key(&self.model, &request);
        let started = Instant::now();
        let mut compact_retry = false;
        let mut repetition: Option<RepetitionFinding> = None;
        let mut sent = request.clone();
        let first = self
            .monitor
            .infer_typed_detailed::<BoardReadOutput>(request.clone(), cancel.clone())
            .await;
        // A reply cut off at the output limit may be a loop that ran into it (a
        // returned, not streamed, reply): the loop decides the retry.
        let first = match first {
            Err(InferFailure::Truncated(e)) => {
                match e
                    .raw_text
                    .as_deref()
                    .and_then(|t| detect(t, &self.params.repetition))
                {
                    Some(f) => Err(InferFailure::Repetition(e, Box::new(f))),
                    None => Err(InferFailure::Truncated(e)),
                }
            }
            r => r,
        };
        let (out, mut raw) = match first {
            Ok(v) => v,
            Err(InferFailure::Other(e)) => return Err(e),
            Err(InferFailure::Repetition(e, finding)) => {
                let retry = repetition_retry(&self.params, &request);
                tracing::warn!(
                    role = ?role,
                    finding = %finding,
                    repeat_penalty = self.params.repetition_retry_penalty,
                    num_predict = retry.options.num_predict,
                    error = %e.message,
                    "board reading fell into a repetition loop; retrying once with a repeat penalty"
                );
                key = request_key(&self.model, &retry);
                sent = retry.clone();
                let message = format!("after the repetition retry ({finding})");
                repetition = Some(*finding);
                self.monitor
                    .infer_typed_detailed::<BoardReadOutput>(retry, cancel.clone())
                    .await
                    .map_err(|f| {
                        let mut info = f.into_info();
                        info.message = format!("{message}: {}", info.message);
                        info
                    })?
            }
            Err(InferFailure::Truncated(e)) => {
                tracing::warn!(
                    role = ?role,
                    num_predict = request.options.num_predict,
                    error = %e.message,
                    "board reading hit the output limit; retrying once with a compact budget"
                );
                let mut retry = board_read_request_compact(
                    &prepared,
                    request.options,
                    self.params.compact_retry_scale,
                )
                .map_err(|e| vision_error_info(&e))?;
                retry.repetition_guard = Some(self.params.repetition);
                key = request_key(&self.model, &retry);
                sent = retry.clone();
                compact_retry = true;
                self.monitor
                    .infer_typed_detailed::<BoardReadOutput>(retry, cancel.clone())
                    .await
                    .map_err(|f| {
                        let mut info = f.into_info();
                        info.message = format!("after the compact retry: {}", info.message);
                        info
                    })?
            }
        };
        let to_canvas =
            |o: BoardReadOutput| shift_output(o.to_canvas_coords(&prepared), region.x1, region.y1);
        let mut out = to_canvas(out);
        let p = &self.params.degenerate;
        let mut degenerate = None;
        if let Some(finding) = degenerate::detect(&out, anchors, p) {
            let mut log = DegenerateLog {
                finding,
                retried: false,
                retry_error: None,
                collapsed: Vec::new(),
            };
            // A reply that already is a retry gets no second one.
            if !compact_retry && repetition.is_none() {
                let retry = repetition_retry(&self.params, &request);
                tracing::warn!(
                    role = ?role,
                    finding = %log.finding,
                    repeat_penalty = self.params.repetition_retry_penalty,
                    num_predict = retry.options.num_predict,
                    "board reading lists one text at many places; retrying once with a repeat penalty"
                );
                match self
                    .monitor
                    .infer_typed_detailed::<BoardReadOutput>(retry.clone(), cancel.clone())
                    .await
                {
                    Ok((o, r)) => {
                        out = to_canvas(o);
                        raw = r;
                        key = request_key(&self.model, &retry);
                        sent = retry;
                        log.retried = true;
                    }
                    // A retry stopped at the output limit or in a loop (the raw
                    // store records these, so replay reaches the same place)
                    // keeps the first reply. Anything else (an invalid reply, a
                    // timeout, the server, cancellation, a replay without the
                    // record) fails the item, as the other retries do.
                    Err(f) => match settled_retry_failure(&f) {
                        Some(label) => {
                            tracing::warn!(
                                role = ?role,
                                error = %f.clone().into_info().message,
                                "the degenerate-reading retry failed; keeping the first reply"
                            );
                            log.retry_error = Some(label.to_string());
                        }
                        None => {
                            let mut info = f.into_info();
                            info.message = format!(
                                "after the degenerate-reading retry ({}): {}",
                                log.finding, info.message
                            );
                            return Err(info);
                        }
                    },
                }
            }
            if !log.retried || degenerate::detect(&out, anchors, p).is_some() {
                let (kept, collapsed) = degenerate::collapse(out, anchors, p);
                out = kept;
                tracing::warn!(
                    role = ?role,
                    finding = %log.finding,
                    retried = log.retried,
                    retry_error = log.retry_error.as_deref().unwrap_or(""),
                    collapsed = collapsed.len(),
                    "degenerate board reading kept with repeated texts collapsed"
                );
                log.collapsed = collapsed;
            }
            degenerate = Some(log);
        }
        Ok((
            out,
            RequestLog {
                role,
                region,
                sent_width: prepared.image.width(),
                sent_height: prepared.image.height(),
                request_key: key,
                latency_s: started.elapsed().as_secs_f64(),
                attempts: raw.attempts,
                repaired: raw.repaired,
                eval_count: raw.eval_count,
                prompt_eval_count: raw.prompt_eval_count,
                done_reason: raw.done_reason,
                compact_retry,
                repetition,
                sampling: sent.sampling,
                num_predict: Some(sent.options.num_predict),
                degenerate,
            },
        ))
    }
}

impl Stage for BoardReadStage {
    type Params = BoardReadParams;
    type Work = ReadWork;
    type Output = BoardReadingItem;

    fn name(&self) -> &'static str {
        "board_read"
    }
    fn version(&self) -> u32 {
        // 2: repetition guard and retry, and model-reply failures recurrent
        // instead of terminal (outputs of 1 may cache a failure as permanent).
        // 3: final repetition guard and settled-failure rules.
        // 4: output budget up to 6,144 tokens (num_ctx 12,288); outputs of 3
        // may cache a dense board as truncated.
        // 5: degenerate complete replies (one text at many places) retried
        // with a repeat penalty, then collapsed; outputs of 4 may hold them.
        // 6: only texts at the repeat bound count towards the duplicate share,
        // the share needs piled copies (a row of identical cards OCR misses is
        // no copy), an OCR text read twice backs one copy, and an invalid retry
        // reply fails the item; outputs of 5 may
        // hold collapsed legitimate cards.
        6
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: artifacts::BOARD_READING,
            version: artifacts::output_version(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![input(artifacts::CANVAS_CROP), input(artifacts::OCR)]
    }
    fn params(&self) -> &BoardReadParams {
        &self.params
    }
    fn key_extras(&self) -> KeyExtras {
        KeyExtras {
            model_digest: Some(format!(
                "{}@{}",
                self.model,
                self.digest.clone().unwrap_or_default()
            )),
            prompt_hash: Some(text_hash(&format!(
                "{BOARD_READ_PROMPT}\n{COMPACT_RETRY_NOTE}"
            ))),
            tool_versions: self
                .server_version
                .iter()
                .map(|v| ("ollama".to_string(), v.clone()))
                .collect::<BTreeMap<_, _>>(),
            ..KeyExtras::default()
        }
    }
    fn concurrency(&self) -> usize {
        self.monitor.client().max_in_flight()
    }
    fn item_timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(900))
    }

    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<ReadWork>>, StageError> {
        // canvas_crop carries the median canvas text height (the tiling
        // trigger); the OCR spans back list items in the degenerate-reading rule.
        let spans_of: HashMap<String, Vec<OcrSpan>> = inputs
            .read_ok::<OcrKeyframe>(artifacts::OCR)?
            .into_iter()
            .map(|(_, o)| (o.keyframe_id, o.spans))
            .collect();
        Ok(inputs
            .read_ok::<CanvasCropItem>(artifacts::CANVAS_CROP)?
            .into_iter()
            .map(|(id, crop)| {
                let anchors = spans_of
                    .get(&crop.keyframe_id)
                    .map(|spans| canvas_anchors(spans, &crop))
                    .unwrap_or_default();
                WorkItem {
                    id,
                    work: ReadWork { crop, anchors },
                }
            })
            .collect())
    }

    async fn process(
        &self,
        ctx: &ItemContext,
        work: ReadWork,
    ) -> Result<BoardReadingItem, ErrorInfo> {
        let ReadWork { crop, anchors } = work;
        self.monitor.ensure_preflight().await?;
        let started = Instant::now();
        let img = load_rgb(crop.source_image_path.clone().into()).await?;
        let canvas = canvas_image(&img, &crop)
            .ok_or_else(|| crate::stages::invalid("canvas box is empty"))?;
        let (cw, ch) = (f64::from(canvas.width()), f64::from(canvas.height()));
        let overview_img = DynamicImage::ImageRgb8(canvas);
        let sizing = BoardSizing {
            target_long_edge: self.params.target_long_edge_px,
            min_long_edge: self.params.min_long_edge_px,
            low_res_upscale: self.params.low_res_upscale,
        };
        let overview =
            prepare_board_image_with(&overview_img, &sizing).map_err(|e| vision_error_info(&e))?;
        let low_res = overview.plan.low_res;
        let token_capped = overview.plan.token_capped;
        let tiled = !low_res
            && crop
                .canvas_text_height_px
                .is_some_and(|h| h < self.params.tiling_text_height_px);
        let mut jobs = vec![(RequestRole::Overview, BBox::new(0.0, 0.0, cw, ch), overview)];
        if tiled {
            for r in tile_regions(cw, ch, self.params.tile_grid, self.params.tile_overlap) {
                let sub = overview_img.crop_imm(
                    r.x1 as u32,
                    r.y1 as u32,
                    r.width().max(1.0) as u32,
                    r.height().max(1.0) as u32,
                );
                let p = prepare_long_edge(&sub, self.params.target_long_edge_px)
                    .map_err(|e| vision_error_info(&e))?;
                jobs.push((RequestRole::Tile, r, p));
            }
        }
        let cancel = ctx.cancel_token().clone();
        let results = futures_util::future::join_all(
            jobs.into_iter()
                .map(|(role, region, p)| self.one(role, region, p, &anchors, cancel.clone())),
        )
        .await;
        let mut outputs = Vec::new();
        let mut logs = Vec::new();
        for r in results {
            let (o, l) = r?;
            outputs.push(o);
            logs.push(l);
        }
        let mut it = outputs.into_iter();
        let overview_out = it.next().ok_or_else(|| internal("no overview reading"))?;
        let result = if tiled {
            merge_tiles(
                overview_out,
                it.collect(),
                self.params.merge_iou,
                self.params.merge_text_ratio,
            )
        } else {
            overview_out
        };
        Ok(BoardReadingItem {
            keyframe_id: crop.keyframe_id,
            source_frame_id: crop.source_frame_id,
            source_image_path: crop.source_image_path,
            source_image_blake3: crop.source_image_blake3,
            crop_box: crop.canvas_bbox,
            masks: crop.masks,
            tiles: crop.tiles,
            model: ModelRef {
                name: self.model.clone(),
                digest: self.digest.clone(),
            },
            latency_s: started.elapsed().as_secs_f64(),
            low_res,
            token_capped,
            tiled,
            requests: logs,
            participants: crop.participants,
            result,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use glassrip_vision::board::{BoardEdge, EdgeStyle, StickyColor};
    use glassrip_vision::image_prep::prepare_board_image;

    fn node(id: &str, text: &str, b: BBox) -> BoardNode {
        BoardNode {
            local_id: id.into(),
            text: text.into(),
            bbox: b,
            conf: 0.9,
        }
    }

    fn empty() -> BoardReading {
        BoardReading {
            nodes: vec![],
            edges: vec![],
            stickies: vec![],
            owner_tags: vec![],
            other_visible_text: vec![],
            confidence: 0.8,
        }
    }

    #[test]
    fn output_budget_fills_the_context() {
        // A context too small for the cap: the budget is what it leaves.
        let p = BoardReadParams {
            num_ctx: 8192,
            ..BoardReadParams::default()
        };
        let img =
            |w, h| prepare_board_image(&DynamicImage::ImageRgb8(RgbImage::new(w, h))).unwrap();
        let opts = GenerationOptions {
            seed: 0,
            num_predict: 1,
        };
        let big = board_read_request(&img(1600, 1200), opts).unwrap();
        let small = board_read_request(&img(800, 400), opts).unwrap();
        let (b, s) = (output_budget(&p, &big), output_budget(&p, &small));
        assert!(b < s, "{b} {s}");
        assert!(b >= p.min_num_predict && s <= p.max_num_predict);
        let total = big.image.tokens()
            + glassrip_vision::ollama::estimate_text_tokens(&big.prompt)
            + glassrip_vision::ollama::MESSAGE_OVERHEAD_TOKENS
            + b;
        assert!(total + p.ctx_headroom <= p.num_ctx || b == p.min_num_predict);
    }

    fn full_hd_board_request() -> VisionRequest {
        // The canvas shape of the dense boards that truncated at 3,072 tokens.
        let img = prepare_board_image(&DynamicImage::ImageRgb8(RgbImage::new(1920, 1176))).unwrap();
        board_read_request(
            &img,
            GenerationOptions {
                seed: 0,
                num_predict: 1,
            },
        )
        .unwrap()
    }

    fn used_tokens(p: &BoardReadParams, r: &VisionRequest, budget: u32) -> u32 {
        r.image.tokens()
            + glassrip_vision::ollama::estimate_text_tokens(&r.prompt)
            + glassrip_vision::ollama::MESSAGE_OVERHEAD_TOKENS
            + budget
            + p.ctx_headroom
    }

    #[test]
    fn dense_boards_get_the_full_budget_within_the_context() {
        let p = BoardReadParams::default();
        assert_eq!(p.max_num_predict, 6144);
        let r = full_hd_board_request();
        let b = output_budget(&p, &r);
        assert_eq!(
            b, p.max_num_predict,
            "the default context leaves room for the cap"
        );
        assert!(used_tokens(&p, &r, b) <= p.num_ctx);
        // The largest image the sizing sends (square, token-capped) still fits.
        let sq = prepare_board_image(&DynamicImage::ImageRgb8(RgbImage::new(4000, 4000))).unwrap();
        let r = board_read_request(
            &sq,
            GenerationOptions {
                seed: 0,
                num_predict: 1,
            },
        )
        .unwrap();
        let b = output_budget(&p, &r);
        assert!(b >= 3 * p.max_num_predict / 4, "{b}");
        assert!(used_tokens(&p, &r, b) <= p.num_ctx, "{b}");
    }

    #[test]
    fn the_old_context_cannot_reach_the_cap() {
        // num_ctx 8,192 capped a full HD board below the old 3,072 limit (e2e
        // runs showed 2,888): raising the cap alone would change nothing.
        let p = BoardReadParams {
            num_ctx: 8192,
            ..BoardReadParams::default()
        };
        let b = output_budget(&p, &full_hd_board_request());
        assert!(b < 3200, "{b}");
    }

    #[test]
    fn stage_context_matches_the_run_config() {
        assert_eq!(
            BoardReadParams::default().num_ctx,
            glassrip_core::config::Config::default().ollama.num_ctx
        );
        assert_eq!(
            crate::placement::MonitorConfig::default().num_ctx,
            BoardReadParams::default().num_ctx
        );
        // A full reply fits the 7b request timeout at 40 tokens per second
        // (below the 60 to 100 measured with four parallel slots).
        let timeout = glassrip_core::config::Config::default()
            .ollama
            .request_timeout_s;
        assert_eq!(
            timeout,
            glassrip_vision::ollama::DEFAULT_REQUEST_TIMEOUT.as_secs()
        );
        assert!(timeout * 40 >= u64::from(BoardReadParams::default().max_num_predict));
    }

    #[test]
    fn repetition_retry_keeps_the_guard_and_shrinks_the_budget() {
        let p = BoardReadParams::default();
        let mut r = full_hd_board_request();
        r.options.num_predict = output_budget(&p, &r);
        r.repetition_guard = Some(p.repetition);
        let retry = repetition_retry(&p, &r);
        assert_eq!(retry.repetition_guard, Some(p.repetition));
        assert_eq!(retry.options.num_predict, 4608);
        assert_eq!(
            retry.sampling.repeat_penalty,
            Some(p.repetition_retry_penalty)
        );
    }

    #[test]
    fn tiles_cover_canvas_with_overlap() {
        let t = tile_regions(1000.0, 500.0, 2, 0.12);
        assert_eq!(t.len(), 4);
        assert_eq!(t[0].x1, 0.0);
        assert!((t[3].x2 - 1000.0).abs() < 1.0 && (t[3].y2 - 500.0).abs() < 1.0);
        // Neighbors overlap by about 12% of a tile.
        let overlap = t[0].x2 - t[1].x1;
        assert!(overlap > 0.1 * t[0].width() && overlap < 0.14 * t[0].width());
    }

    #[test]
    fn merge_dedupes_nodes_and_remaps_edges() {
        let mut ov = empty();
        ov.nodes = vec![
            node("a", "Order Service", BBox::new(100.0, 100.0, 300.0, 160.0)),
            node("b", "Ledger", BBox::new(600.0, 100.0, 700.0, 160.0)),
        ];
        ov.edges = vec![BoardEdge {
            src: "a".into(),
            dst: "b".into(),
            label: "REST".into(),
            label_bbox: None,
            style: EdgeStyle::Solid,
            conf: 0.9,
        }];
        let mut t0 = empty();
        // Same node, slightly different box and text case.
        t0.nodes = vec![node(
            "x",
            "order service",
            BBox::new(102.0, 98.0, 305.0, 162.0),
        )];
        t0.stickies = vec![Sticky {
            text: "Ship it?".into(),
            color: StickyColor::Yellow,
            bbox: BBox::new(400.0, 300.0, 450.0, 340.0),
        }];
        t0.owner_tags = vec![OwnerTag {
            name_raw: "Ada".into(),
            near: "x".into(),
            bbox: BBox::new(310.0, 100.0, 340.0, 120.0),
        }];
        let mut t1 = empty();
        t1.stickies = t0.stickies.clone();
        let m = merge_tiles(ov, vec![t0, t1], 0.5, 0.85);
        assert_eq!(m.nodes.len(), 2, "{:?}", m.nodes);
        assert_eq!(m.edges.len(), 1);
        let e = &m.edges[0];
        let src = m.nodes.iter().find(|n| n.local_id == e.src).unwrap();
        assert_eq!(normalize(&src.text), "order service");
        assert_eq!(m.stickies.len(), 1);
        assert_eq!(m.owner_tags[0].near, src.local_id);
    }

    #[test]
    fn canvas_anchors_keep_canvas_spans_in_canvas_pixels() {
        let span = |text: &str, b: BBox, region: TextRegion| OcrSpan {
            text: text.into(),
            bbox: b,
            confidence: 0.9,
            region,
            chrome_reason: None,
            bg_luma: 240.0,
        };
        let spans = vec![
            span(
                "Ledger",
                BBox::new(60.0, 30.0, 80.0, 40.0),
                TextRegion::Unassigned,
            ),
            span(
                "Share",
                BBox::new(60.0, 50.0, 80.0, 60.0),
                TextRegion::Chrome,
            ),
            span(
                "Far away",
                BBox::new(170.0, 30.0, 190.0, 40.0),
                TextRegion::Canvas,
            ),
        ];
        let mut crop = CanvasCropItem {
            keyframe_id: "k".into(),
            source_frame_id: "f".into(),
            source_image_path: String::new(),
            source_image_blake3: None,
            image_width: 200,
            image_height: 100,
            canvas_bbox: BBox::new(50.0, 20.0, 150.0, 90.0),
            raw_canvas_bbox: BBox::new(50.0, 20.0, 150.0, 90.0),
            method: crate::artifacts::CanvasMethod::Layout,
            segment: 0,
            stabilized: false,
            share_area: None,
            tiles: vec![],
            masks: vec![],
            span_regions: vec![],
            canvas_text_height_px: None,
            participants: vec![],
        };
        // Without canvas_crop's regions: OCR's own, canvas or unassigned, inside the crop.
        let a = canvas_anchors(&spans, &crop);
        assert_eq!(
            a,
            vec![("Ledger".to_string(), BBox::new(10.0, 10.0, 30.0, 20.0))]
        );
        // canvas_crop's final regions win when present.
        crop.span_regions = vec![TextRegion::Chrome, TextRegion::Canvas, TextRegion::Canvas];
        let a = canvas_anchors(&spans, &crop);
        assert_eq!(
            a,
            vec![("Share".to_string(), BBox::new(10.0, 30.0, 30.0, 40.0))]
        );
    }

    #[test]
    fn only_recorded_retry_failures_keep_the_first_reply() {
        use glassrip_core::envelope::ErrorCode;
        let info = |code| ErrorInfo::new(code, "x");
        assert_eq!(
            settled_retry_failure(&InferFailure::Truncated(info(ErrorCode::ModelRequest))),
            Some("output limit")
        );
        for code in [
            ErrorCode::SchemaParse,
            ErrorCode::Timeout,
            ErrorCode::Cancelled,
            ErrorCode::ModelRequest,
        ] {
            assert_eq!(
                settled_retry_failure(&InferFailure::Other(info(code))),
                None
            );
        }
    }

    #[test]
    fn degenerate_rule_is_on_by_default() {
        let p = BoardReadParams::default();
        assert!(p.degenerate.enabled());
        let c = glassrip_core::config::Config::default().board_read;
        assert_eq!(c.degenerate_min_repeats as usize, p.degenerate.min_repeats);
        assert_eq!(
            c.degenerate_min_duplicate_share,
            p.degenerate.min_duplicate_share
        );
        assert_eq!(
            c.degenerate_strong_repeats as usize,
            p.degenerate.strong_repeats
        );
    }

    #[test]
    fn canvas_image_crops_and_masks() {
        let mut img = RgbImage::from_pixel(200, 100, image::Rgb([240, 240, 240]));
        img.put_pixel(60, 30, image::Rgb([0, 0, 0]));
        let crop = CanvasCropItem {
            keyframe_id: "k".into(),
            source_frame_id: "f".into(),
            source_image_path: String::new(),
            source_image_blake3: None,
            image_width: 200,
            image_height: 100,
            canvas_bbox: BBox::new(50.0, 20.0, 150.0, 90.0),
            raw_canvas_bbox: BBox::new(50.0, 20.0, 150.0, 90.0),
            method: crate::artifacts::CanvasMethod::Layout,
            segment: 0,
            stabilized: false,
            share_area: None,
            tiles: vec![],
            masks: vec![crate::artifacts::ChromeMask {
                bbox: BBox::new(58.0, 28.0, 62.0, 32.0),
                reason: crate::artifacts::ChromeReason::Denylist,
                text: "100%".into(),
            }],
            span_regions: vec![],
            canvas_text_height_px: None,
            participants: vec![],
        };
        let c = canvas_image(&img, &crop).unwrap();
        assert_eq!((c.width(), c.height()), (100, 70));
        assert_eq!(c.get_pixel(10, 10), &image::Rgb([240, 240, 240]));
    }
}
