//! Synthetic suite in pipeline mode: each fixture frame goes through the real
//! vision stages on a core runner, `ocr_harvest`, `ocr_vocabulary`, `classify`,
//! `canvas_crop`, `board_read`, `board_validate`, and `edge_direction`, instead of
//! the reader-only path of [`crate::suite::run_board_suite`] (which classifies the
//! whole frame and reads the predicted crop directly).
//!
//! The scored board is the validated reading (`glassrip.board_validate`) with
//! edge directions from the pixel check (VLM fallback when inconclusive) of
//! `glassrip.edge_direction`. Metric names match reader mode, so reports compare.
//!
//! Replay is deterministic: vision replies come from a raw response store
//! (`glassrip-vision-stages` [`RawStore`], keyed by everything the server sees)
//! and OCR spans from an [`OcrStore`] keyed by image content, so a replay needs
//! neither a model server nor ONNX Runtime. `--rerecord` fills both stores from a
//! live run. Layout under the responses directory: `raw/` and `ocr/`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use glassrip_core::cache::Cache;
use glassrip_core::envelope::Producer;
use glassrip_core::graph::{meeting_mode_stage_decls, Selection, StageGraph};
use glassrip_core::manifest::RunDir;
use glassrip_core::runner::{Runner, RunnerOptions, Stage};
use glassrip_meeting::artifacts::{EdgeDirectionBatch, EDGE_DIRECTION};
use glassrip_meeting::direction::EndVerdict;
use glassrip_meeting::pixel_direction::PixelCheckParams;
use glassrip_meeting::stages::EdgeDirectionStage;
use glassrip_meeting::vlm_direction::VlmCheckParams;
use glassrip_ocr::{OcrConfig, OcrError, RecognizedSpan, TextRecognizer};
use glassrip_vision::{VisionBackend, VisionClient};
use glassrip_vision_stages::artifacts::{
    BoardReadingItem, BoardValidateItem, ScreenClassItem, BOARD_READING, BOARD_VALIDATE,
    SCREEN_CLASS,
};
use glassrip_vision_stages::layout::LayoutConfig;
use glassrip_vision_stages::placement::{MonitorConfig, PlacementMonitor, PlacementProbe};
use glassrip_vision_stages::stages::canvas_crop::CanvasCropParams;
use glassrip_vision_stages::{
    adapter, BoardReadParams, BoardReadStage, BoardValidateParams, BoardValidateStage,
    CanvasCropStage, ClassifyParams, ClassifyStage, OcrHarvestStage, VocabularyParams,
    VocabularyStage,
};
use serde::Serialize;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::error::{EvalError, Result};
use crate::fixture::BoardCase;
use crate::metrics::board::{score_board, Direction, PredBoard};
use crate::suite::{pred_board_from, summarize_board_results, BoardCaseResult, SuiteRun};
use crate::views::RunArtifacts;

pub use glassrip_vision_stages::raw_store::{RawStore, RecordingBackend, ReplayBackend};

/// OCR spans stored by image content: `<dir>/<blake3>.json`, where the hash is
/// over the image size and RGB bytes.
#[derive(Debug, Clone)]
pub struct OcrStore {
    dir: PathBuf,
}

impl OcrStore {
    /// A store in `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Key of an image.
    pub fn key(image: &image::RgbImage) -> String {
        let mut h = blake3::Hasher::new();
        h.update(&image.width().to_le_bytes());
        h.update(&image.height().to_le_bytes());
        h.update(image.as_raw());
        h.finalize().to_hex().to_string()
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.json"))
    }

    /// Stored spans for `image`, if any.
    pub fn get(
        &self,
        image: &image::RgbImage,
    ) -> std::result::Result<Option<Vec<RecognizedSpan>>, OcrError> {
        let p = self.path(&Self::key(image));
        match fs_err::read(&p) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|e| OcrError::Input(format!("{}: {e}", p.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(OcrError::Input(format!("{}: {e}", p.display()))),
        }
    }

    /// Stores spans for `image`.
    pub fn put(
        &self,
        image: &image::RgbImage,
        spans: &[RecognizedSpan],
    ) -> std::result::Result<(), OcrError> {
        let p = self.path(&Self::key(image));
        let io = |e: &dyn std::fmt::Display| OcrError::Input(format!("{}: {e}", p.display()));
        fs_err::create_dir_all(&self.dir).map_err(|e| io(&e))?;
        let bytes = serde_json::to_vec_pretty(spans).map_err(|e| io(&e))?;
        glassrip_core::atomic::write_atomic(&p, &bytes).map_err(|e| io(&e))
    }
}

/// Wraps a live recognizer and stores every result.
pub struct RecordingRecognizer {
    inner: Arc<dyn TextRecognizer>,
    store: OcrStore,
}

impl RecordingRecognizer {
    /// Records `inner` into `store`.
    pub fn new(inner: Arc<dyn TextRecognizer>, store: OcrStore) -> Self {
        Self { inner, store }
    }
}

impl TextRecognizer for RecordingRecognizer {
    fn recognize(
        &self,
        image: &image::RgbImage,
    ) -> std::result::Result<Vec<RecognizedSpan>, OcrError> {
        let spans = self.inner.recognize(image)?;
        self.store.put(image, &spans)?;
        Ok(spans)
    }
    fn execution_provider(&self) -> String {
        self.inner.execution_provider()
    }
    fn model_fingerprint(&self) -> String {
        self.inner.model_fingerprint()
    }
}

/// Serves stored spans; a missing image is an error.
pub struct ReplayRecognizer {
    store: OcrStore,
}

impl ReplayRecognizer {
    /// Replays from `store`.
    pub fn new(store: OcrStore) -> Self {
        Self { store }
    }
}

impl TextRecognizer for ReplayRecognizer {
    fn recognize(
        &self,
        image: &image::RgbImage,
    ) -> std::result::Result<Vec<RecognizedSpan>, OcrError> {
        self.store.get(image)?.ok_or_else(|| {
            OcrError::Input(format!(
                "no recorded OCR spans for image {} (rerun with --rerecord)",
                OcrStore::key(image)
            ))
        })
    }
    fn execution_provider(&self) -> String {
        "replay".into()
    }
    fn model_fingerprint(&self) -> String {
        "replay".into()
    }
}

/// Everything a pipeline-mode run needs.
pub struct PipelineContext {
    /// Vision requests (replay, recording, or scripted).
    pub vision: Arc<dyn VisionBackend>,
    /// Placement checks.
    pub probe: Arc<dyn PlacementProbe>,
    /// OCR (replay, recording, or scripted).
    pub recognizer: Arc<dyn TextRecognizer>,
    /// Vision model name (recorded in the stage cache keys).
    pub model: String,
    /// Client concurrency.
    pub slots: usize,
    /// Generation seed.
    pub seed: u64,
    /// Fixed context size.
    pub num_ctx: u32,
    /// Per-case run directories are created here (replaced on every run).
    pub work_dir: PathBuf,
    /// Cancellation.
    pub cancel: CancellationToken,
}

/// Runs every case through the vision stages (one after the other) and scores
/// them like reader mode.
pub async fn run_pipeline_suite(ctx: &PipelineContext, cases: &[BoardCase]) -> SuiteRun {
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        results.push(run_pipeline_case(ctx, case).await);
    }
    let mut run = SuiteRun::default();
    summarize_board_results(&results, &mut run);
    run
}

struct PipelineOutput {
    pred_screen: crate::metrics::screen::ScreenType,
    method: String,
    board: Option<PredBoard>,
    latency_s: f64,
}

/// Runs and scores one case; errors are recorded in the result.
pub async fn run_pipeline_case(ctx: &PipelineContext, case: &BoardCase) -> BoardCaseResult {
    let gold = &case.expected;
    match case_inner(ctx, case).await {
        Ok(out) => {
            let pred = out.board.clone().unwrap_or_default();
            BoardCaseResult {
                case: case.name.clone(),
                gold_screen: gold.screen_type,
                pred_screen: Some(out.pred_screen),
                classify_method: Some(out.method),
                score: score_board(&gold.board, &pred, &gold.chrome_texts),
                board: out.board,
                latency_s: out.latency_s,
                error: None,
            }
        }
        Err(e) => BoardCaseResult {
            case: case.name.clone(),
            gold_screen: gold.screen_type,
            pred_screen: None,
            classify_method: None,
            board: None,
            score: score_board(&gold.board, &PredBoard::default(), &gold.chrome_texts),
            latency_s: 0.0,
            error: Some(e.to_string()),
        },
    }
}

fn enum_name<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

fn other(e: impl std::fmt::Display) -> EvalError {
    EvalError::Other(e.to_string())
}

/// Resolves an edge's verdict (pixel first, VLM when the pixel check could not
/// decide) relative to the edge as read.
fn verdict_of(pixel: EndVerdict, vlm: Option<EndVerdict>) -> EndVerdict {
    match pixel {
        EndVerdict::Unknown | EndVerdict::NoArrowhead => match vlm {
            Some(v @ (EndVerdict::Forward | EndVerdict::Reverse | EndVerdict::Bidirectional)) => v,
            _ => pixel,
        },
        decided => decided,
    }
}

async fn case_inner(ctx: &PipelineContext, case: &BoardCase) -> Result<PipelineOutput> {
    let run_dir = ctx.work_dir.join(&case.name);
    if run_dir.exists() {
        fs_err::remove_dir_all(&run_dir).map_err(|e| EvalError::io(&run_dir, e))?;
    }
    let input = run_dir.join("input");
    fs_err::create_dir_all(&input).map_err(|e| EvalError::io(&input, e))?;
    let frame = input.join("frame.png");
    fs_err::copy(case.frame_path(), &frame).map_err(|e| EvalError::io(&frame, e))?;
    let index = run_dir.join("index.json");
    crate::error::write_json(
        &index,
        &json!({"keyframes": [{"file": "input/frame.png", "t_start": 0.0, "t_end": 2.0, "t_rep": 0.0}]}),
    )?;
    adapter::build_inputs(&run_dir, &index, None, "eval").map_err(other)?;

    let run = RunDir::open(
        &run_dir,
        "eval",
        Producer::glassrip(env!("CARGO_PKG_VERSION"), None),
    )
    .map_err(other)?;
    let mut runner = Runner::new(
        run,
        StageGraph::new(meeting_mode_stage_decls()).map_err(other)?,
        &Selection::default(),
        Cache::new(run_dir.join(".cache")),
        RunnerOptions::default(),
        ctx.cancel.clone(),
    )
    .map_err(other)?;
    let client = VisionClient::new(Arc::clone(&ctx.vision), ctx.slots.max(1))?;
    let monitor = Arc::new(PlacementMonitor::new(
        Arc::clone(&ctx.probe),
        client.clone(),
        MonitorConfig {
            num_ctx: ctx.num_ctx,
            ..MonitorConfig::default()
        },
    ));
    let participants = case.expected.participants.clone();
    let layout = LayoutConfig {
        participants: participants.clone(),
        ..LayoutConfig::default()
    };
    let digest = ctx.vision.id().digest;
    let ocr = OcrHarvestStage::new(
        Arc::clone(&ctx.recognizer),
        OcrConfig::default(),
        layout.clone(),
    );
    let vocabulary = VocabularyStage::new(VocabularyParams::default());
    let classify = ClassifyStage::new(
        ClassifyParams {
            seed: ctx.seed,
            ..ClassifyParams::default()
        },
        Arc::clone(&monitor),
        &ctx.model,
        digest.clone(),
        None,
    );
    let canvas = CanvasCropStage::new(CanvasCropParams {
        layout,
        ..CanvasCropParams::default()
    });
    let board_read = BoardReadStage::new(
        BoardReadParams {
            seed: ctx.seed,
            num_ctx: ctx.num_ctx,
            ..BoardReadParams::default()
        },
        Arc::clone(&monitor),
        &ctx.model,
        digest,
        None,
    );
    let validate = BoardValidateStage::new({
        let mut p = BoardValidateParams::default();
        p.validation.participant_names = participants;
        p
    });
    let edges = EdgeDirectionStage::new(
        PixelCheckParams::default(),
        VlmCheckParams::default(),
        Some(client),
    );

    async fn model<S: Stage>(
        runner: &mut Runner,
        stage: &S,
        monitor: &PlacementMonitor,
    ) -> Result<()> {
        monitor
            .begin_stage()
            .await
            .map_err(|e| EvalError::Other(e.message))?;
        runner.run_stage(stage).await.map_err(other)?;
        match monitor.abort_error() {
            Some(e) => Err(EvalError::Other(e.message)),
            None => Ok(()),
        }
    }
    runner.run_stage(&ocr).await.map_err(other)?;
    runner.run_stage(&vocabulary).await.map_err(other)?;
    model(&mut runner, &classify, &monitor).await?;
    runner.run_stage(&canvas).await.map_err(other)?;
    model(&mut runner, &board_read, &monitor).await?;
    runner.run_stage(&validate).await.map_err(other)?;
    runner.run_stage(&edges).await.map_err(other)?;

    let art = RunArtifacts::scan(&run_dir)?;
    let class = art
        .items::<ScreenClassItem>(SCREEN_CLASS)?
        .and_then(|v| v.into_iter().next())
        .ok_or_else(|| EvalError::Other("classify produced no item".into()))?;
    let pred_screen = class.screen_type.into();
    let method = enum_name(&class.method);
    let reading = art
        .items::<BoardReadingItem>(BOARD_READING)?
        .and_then(|v| v.into_iter().next());
    let validated = art
        .items::<BoardValidateItem>(BOARD_VALIDATE)?
        .and_then(|v| v.into_iter().next());
    let directions = art
        .items::<EdgeDirectionBatch>(EDGE_DIRECTION)?
        .and_then(|v| v.into_iter().next());
    let board = validated.map(|v| {
        let mut pred = pred_board_from(&v.board, v.crop_box.x1, v.crop_box.y1);
        let evidence = directions
            .as_ref()
            .and_then(|d| d.keyframes.iter().find(|k| k.keyframe_id == v.keyframe_id));
        for (pe, ve) in pred.edges.iter_mut().zip(&v.board.edges) {
            let found =
                evidence.and_then(|k| k.edges.iter().find(|e| e.src == ve.src && e.dst == ve.dst));
            let verdict = found
                .map(|e| {
                    let (pixel, vlm) = e.verdicts();
                    verdict_of(pixel, vlm)
                })
                .unwrap_or(EndVerdict::Unknown);
            match verdict {
                EndVerdict::Forward => pe.direction = Direction::Forward,
                EndVerdict::Reverse => {
                    std::mem::swap(&mut pe.src, &mut pe.dst);
                    pe.direction = Direction::Forward;
                }
                EndVerdict::Bidirectional => pe.direction = Direction::Bidirectional,
                EndVerdict::NoArrowhead | EndVerdict::Unknown => {
                    pe.direction = Direction::Uncertain
                }
            }
        }
        pred
    });
    Ok(PipelineOutput {
        pred_screen,
        method,
        board,
        latency_s: reading.map_or(0.0, |r| r.latency_s),
    })
}

/// Default responses directory of pipeline mode for a model.
pub fn default_responses(model_slug: &str) -> PathBuf {
    crate::cli::public_fixtures_root()
        .join("responses/synthetic_pipeline")
        .join(model_slug)
}

/// The raw and OCR stores under a responses directory.
pub fn stores(responses: &Path) -> (RawStore, OcrStore) {
    (
        RawStore::new(responses.join("raw")),
        OcrStore::new(responses.join("ocr")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdicts_prefer_pixels_then_vlm() {
        use EndVerdict::*;
        assert_eq!(verdict_of(Reverse, Some(Forward)), Reverse);
        assert_eq!(verdict_of(Unknown, Some(Forward)), Forward);
        assert_eq!(verdict_of(NoArrowhead, Some(Unknown)), NoArrowhead);
        assert_eq!(verdict_of(Unknown, None), Unknown);
    }

    #[test]
    fn ocr_store_round_trips_by_content() {
        let dir = tempfile::tempdir().unwrap();
        let store = OcrStore::new(dir.path());
        let img = image::RgbImage::from_pixel(8, 8, image::Rgb([10, 20, 30]));
        assert!(store.get(&img).unwrap().is_none());
        let spans = vec![RecognizedSpan {
            text: "Ledger API".into(),
            bbox: glassrip_ocr::PixelBox {
                x1: 1.0,
                y1: 1.0,
                x2: 5.0,
                y2: 3.0,
            },
            confidence: 0.9,
            det_score: 0.8,
        }];
        store.put(&img, &spans).unwrap();
        assert_eq!(store.get(&img).unwrap(), Some(spans));
        let other = image::RgbImage::from_pixel(8, 8, image::Rgb([10, 20, 31]));
        assert!(ReplayRecognizer::new(store).recognize(&other).is_err());
    }
}
