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
use schemars::JsonSchema;
use semver::Version;
use serde::{Deserialize, Serialize};

use crate::artifacts::{
    CanvasCropView, CanvasDims, EdgeDirectionItem, EdgeEvidence, KeyframeView, ValidateItemView,
    BOARD_STATE, BOARD_VALIDATE, CANVAS_CROP, EDGE_DIRECTION, KEYFRAMES,
};
use crate::consolidate::owners::{Corroborator, NoCorroboration};
use crate::consolidate::{
    consolidate, ink_since_previous_board, BoardFrame, BoardStateItem, ConsolidationParams,
};
use crate::pixel_direction::{
    bgr_from_rgb, sharpness, EdgeQuery, PixelCheckParams, PreparedCanvas,
};
use crate::text::{fuzzy_eq, FUZZY_THRESHOLD};
use crate::vlm_direction::{check_edge, EdgeEnds, VlmCheckParams, ENDPOINT_PROMPT};

/// `edge_direction` parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeDirectionParams {
    /// Pixel check.
    pub pixel: PixelCheckParams,
    /// VLM endpoint check.
    pub vlm: VlmCheckParams,
    /// Whether the VLM check runs (set from the presence of a client).
    pub vlm_enabled: bool,
}

/// Work for one keyframe.
#[derive(Debug, Clone)]
pub struct EdgeDirectionWork {
    keyframe_id: String,
    board: ValidatedBoard,
    image: Option<PathBuf>,
}

/// Pixel check plus optional VLM endpoint check for every edge of each keyframe.
pub struct EdgeDirectionStage {
    params: EdgeDirectionParams,
    client: Option<VisionClient>,
}

impl EdgeDirectionStage {
    /// Stage with a vision client for the VLM check, or `None` for pixels only.
    pub fn new(pixel: PixelCheckParams, vlm: VlmCheckParams, client: Option<VisionClient>) -> Self {
        Self {
            params: EdgeDirectionParams {
                pixel,
                vlm,
                vlm_enabled: client.is_some(),
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

/// Pixel evidence for every edge of one keyframe (CPU only).
pub fn pixel_evidence(
    image: &image::RgbImage,
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

impl Stage for EdgeDirectionStage {
    type Params = EdgeDirectionParams;
    type Work = EdgeDirectionWork;
    type Output = EdgeDirectionItem;

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
                prompt_hash: Some(glassrip_core::blake3_hex(ENDPOINT_PROMPT.as_bytes())),
                ..KeyExtras::default()
            },
            None => KeyExtras::default(),
        }
    }
    fn concurrency(&self) -> usize {
        self.client
            .as_ref()
            .map(|c| c.max_in_flight())
            .unwrap_or(2)
            .max(1)
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<EdgeDirectionWork>>, StageError> {
        let root = run_root(inputs, CANVAS_CROP)?;
        let crops: HashMap<String, PathBuf> = inputs
            .read_ok::<CanvasCropView>(CANVAS_CROP)?
            .into_iter()
            .map(|(id, c)| (c.keyframe_id.unwrap_or(id), resolve(&root, &c.path)))
            .collect();
        Ok(inputs
            .read_ok::<ValidateItemView>(BOARD_VALIDATE)?
            .into_iter()
            .map(|(id, v)| {
                let (keyframe_id, _, board) = v.into_parts(&id);
                WorkItem {
                    id: keyframe_id.clone(),
                    work: EdgeDirectionWork {
                        image: crops.get(&keyframe_id).cloned(),
                        keyframe_id,
                        board,
                    },
                }
            })
            .collect())
    }
    async fn process(
        &self,
        ctx: &ItemContext,
        work: EdgeDirectionWork,
    ) -> Result<EdgeDirectionItem, ErrorInfo> {
        let Some(path) = work.image.clone() else {
            return Err(ErrorInfo::new(
                ErrorCode::InvalidInput,
                format!("no canvas crop for keyframe {}", work.keyframe_id),
            ));
        };
        let params = self.params.pixel.clone();
        let board = work.board.clone();
        let (image, edges, sharp, zoom) = tokio::task::spawn_blocking(move || {
            let image = image::open(&path)
                .map_err(|e| ErrorInfo::new(ErrorCode::Io, format!("{}: {e}", path.display())))?
                .to_rgb8();
            let (edges, sharp, zoom) = pixel_evidence(&image, &board, &params);
            Ok::<_, ErrorInfo>((image, edges, sharp, zoom))
        })
        .await
        .map_err(|e| ErrorInfo::new(ErrorCode::Internal, format!("pixel check task: {e}")))??;
        let mut edges = edges;
        if let Some(client) = &self.client {
            let by_id: HashMap<&str, &glassrip_vision::board::BoardNode> = work
                .board
                .nodes
                .iter()
                .map(|n| (n.local_id.as_str(), n))
                .collect();
            for ev in &mut edges {
                let (Some(s), Some(d)) = (by_id.get(ev.src.as_str()), by_id.get(ev.dst.as_str()))
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
                let vlm = check_edge(
                    client,
                    &image,
                    ends,
                    &self.params.vlm,
                    ctx.cancel_token().clone(),
                )
                .await
                .map_err(|e| ErrorInfo::new(ErrorCode::ModelRequest, e.to_string()))?;
                ev.vlm = Some(vlm);
            }
        }
        let (w, h) = image.dimensions();
        Ok(EdgeDirectionItem {
            keyframe_id: work.keyframe_id,
            canvas: CanvasDims {
                width: f64::from(w),
                height: f64::from(h),
            },
            sharpness: sharp,
            zoom,
            edges,
        })
    }
}

/// Everything `board_state` consumes, gathered in `plan`.
#[derive(Debug, Clone)]
pub struct BoardStateWork {
    frames: Vec<BoardFrame>,
}

/// Consolidates the board keyframes into `glassrip.board_state`.
pub struct BoardStateStage {
    params: ConsolidationParams,
    corroborator: Arc<dyn Corroborator>,
}

impl BoardStateStage {
    /// Stage without transcript corroboration.
    pub fn new(params: ConsolidationParams) -> Self {
        Self {
            params,
            corroborator: Arc::new(NoCorroboration),
        }
    }

    /// Stage with a corroborator (for example transcript cues).
    pub fn with_corroborator(
        params: ConsolidationParams,
        corroborator: Arc<dyn Corroborator>,
    ) -> Self {
        Self {
            params,
            corroborator,
        }
    }
}

/// Build consolidation frames from upstream items. Board keyframes missing from
/// `keyframes` are dropped (no times).
pub fn board_frames(
    keyframes: &[KeyframeView],
    boards: Vec<(String, Option<CanvasDims>, ValidatedBoard)>,
    directions: Vec<EdgeDirectionItem>,
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
    let mut boards: Vec<(usize, String, Option<CanvasDims>, ValidatedBoard)> = boards
        .into_iter()
        .filter_map(|(id, c, b)| index.get(id.as_str()).map(|&i| (i, id, c, b)))
        .collect();
    boards.sort_by_key(|b| b.0);
    let all_ink: Vec<Option<f64>> = kf
        .iter()
        .map(|k| k.boundary.as_ref().and_then(|b| b.ink_change))
        .collect();
    let board_index: Vec<usize> = boards.iter().map(|b| b.0).collect();
    let ink = ink_since_previous_board(&all_ink, &board_index);
    boards
        .into_iter()
        .zip(ink)
        .map(|((i, id, canvas, board), ink_change)| BoardFrame {
            keyframe_index: i,
            t_start_s: kf[i].t_start_s,
            t_end_s: kf[i].t_end_s,
            t_rep_s: kf[i].t_rep_s,
            canvas,
            directions: dirs.remove(&id),
            keyframe_id: id,
            board,
            ink_change,
        })
        .collect()
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
            .map(|(id, v)| v.into_parts(&id))
            .filter(|(_, _, b)| !b.needs_reclassification)
            .collect();
        let directions = inputs
            .read_ok::<EdgeDirectionItem>(EDGE_DIRECTION)?
            .into_iter()
            .map(|(_, d)| d)
            .collect();
        Ok(vec![WorkItem {
            id: "board-1".into(),
            work: BoardStateWork {
                frames: board_frames(&keyframes, boards, directions),
            },
        }])
    }
    async fn process(
        &self,
        _ctx: &ItemContext,
        work: BoardStateWork,
    ) -> Result<BoardStateItem, ErrorInfo> {
        Ok(consolidate(
            work.frames,
            &self.params,
            self.corroborator.as_ref(),
        ))
    }
}
