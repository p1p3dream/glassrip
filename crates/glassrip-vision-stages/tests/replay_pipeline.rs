//! End-to-end vision branch on synthetic keyframes: scripted OCR, a scripted
//! model whose answers are recorded to a raw store, then a second run in a fresh
//! directory that replays the recorded answers without any model.
//!
//! Images, names, and board text are fictional and generated here.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use glassrip_core::cache::Cache;
use glassrip_core::envelope::{Producer, Record, SchemaReq};
use glassrip_core::graph::Selection;
use glassrip_core::manifest::RunDir;
use glassrip_core::runner::{Runner, RunnerOptions};
use glassrip_ocr::{OcrConfig, OcrError, PixelBox, RecognizedSpan, TextRecognizer};
use glassrip_vision::classify::{ClassifyRules, KeywordPattern, KeywordRule, ScreenType};
use glassrip_vision::{
    BackendId, Durations, Placement, RawResponse, VisionBackend, VisionClient, VisionError,
    VisionRequest,
};
use glassrip_vision_stages::artifacts::{
    self, BoardReadingItem, BoardValidateItem, CanvasCropItem, OcrKeyframe, ScreenClassItem,
    Vocabulary,
};
use glassrip_vision_stages::layout::LayoutConfig;
use glassrip_vision_stages::pipeline::{self, VisionBranch};
use glassrip_vision_stages::placement::{MonitorConfig, PlacementMonitor, StaticProbe};
use glassrip_vision_stages::raw_store::{RawStore, RecordingBackend, ReplayBackend};
use glassrip_vision_stages::stages::canvas_crop::CanvasCropParams;
use glassrip_vision_stages::{
    adapter, BoardReadParams, BoardReadStage, BoardValidateParams, BoardValidateStage,
    CanvasCropStage, ClassifyParams, ClassifyStage, ConsensusParams, OcrHarvestStage,
    VocabularyParams, VocabularyStage,
};
use image::{Rgb, RgbImage};
use serde::de::DeserializeOwned;
use serde_json::json;
use tokio_util::sync::CancellationToken;

const W: u32 = 1280;
const H: u32 = 720;

fn fill(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, c: [u8; 3]) {
    for y in y0..y1.min(img.height()) {
        for x in x0..x1.min(img.width()) {
            img.put_pixel(x, y, Rgb(c));
        }
    }
}

/// A conferencing screen: dark window, light shared canvas, three tiles on the
/// right. A top-left block encodes the keyframe index for the scripted OCR.
fn keyframe_image(index: u8) -> RgbImage {
    let mut img = RgbImage::from_pixel(W, H, Rgb([30, 30, 36]));
    fill(&mut img, 90, 70, 900, 640, [250, 250, 250]);
    fill(&mut img, 300, 250, 500, 330, [40, 40, 40]);
    fill(&mut img, 304, 254, 496, 326, [255, 255, 255]);
    fill(&mut img, 600, 400, 700, 480, [250, 232, 110]);
    fill(&mut img, 925, 110, 1265, 300, [70, 90, 120]);
    fill(&mut img, 925, 330, 1090, 560, [90, 70, 60]);
    fill(&mut img, 1100, 330, 1265, 560, [60, 90, 70]);
    let v = index * 100;
    fill(&mut img, 0, 0, 16, 16, [v, v, v]);
    img
}

fn span(text: &str, x1: f64, y1: f64, x2: f64, y2: f64) -> RecognizedSpan {
    RecognizedSpan {
        text: text.into(),
        bbox: PixelBox { x1, y1, x2, y2 },
        confidence: 0.95,
        det_score: 0.9,
    }
}

struct ScriptedOcr;

impl TextRecognizer for ScriptedOcr {
    fn recognize(&self, image: &RgbImage) -> Result<Vec<RecognizedSpan>, OcrError> {
        let index = (f64::from(image.get_pixel(8, 8)[0]) / 100.0).round() as u8;
        let mut v = vec![
            span("3:15 PM | Weekly Sync", 100.0, 30.0, 300.0, 48.0),
            span("Ada Quill (Presenting)", 700.0, 30.0, 880.0, 48.0),
            span("Order Service", 330.0, 280.0, 470.0, 298.0),
            span("Convert to", 780.0, 420.0, 880.0, 436.0),
            span("Ada Quill", 935.0, 282.0, 1010.0, 298.0),
            span("Bo Tran Liu", 935.0, 540.0, 1030.0, 556.0),
            span("Cy Obi Tar", 1110.0, 540.0, 1200.0, 556.0),
        ];
        if index == 2 {
            v.extend([
                span("Quarry Studio", 120.0, 90.0, 240.0, 106.0),
                span("Structure", 400.0, 90.0, 480.0, 106.0),
                span("Vision", 500.0, 90.0, 550.0, 106.0),
            ]);
        }
        Ok(v)
    }
    fn execution_provider(&self) -> String {
        "scripted".into()
    }
    fn model_fingerprint(&self) -> String {
        "scripted-ocr-v1".into()
    }
}

/// Scripted model: every screen is a whiteboard; every board has one node near
/// the image center and one chrome string.
struct ScriptedModel;

#[async_trait]
impl VisionBackend for ScriptedModel {
    fn id(&self) -> BackendId {
        BackendId {
            backend: "scripted".into(),
            model: "scripted-vl".into(),
            digest: Some("sha256:scripted".into()),
            server_version: None,
        }
    }
    async fn preflight(&self) -> glassrip_vision::Result<Placement> {
        Err(VisionError::Config("unused".into()))
    }
    async fn infer(
        &self,
        request: VisionRequest,
        _cancel: CancellationToken,
    ) -> glassrip_vision::Result<RawResponse> {
        let (w, h) = (
            f64::from(request.image.width()),
            f64::from(request.image.height()),
        );
        let value = if request.schema.name().contains("ScreenClass") {
            json!({
                "screen_type": "whiteboard",
                "app_hint": "Miro",
                "bbox_2d": [0.1 * w, 0.1 * h, 0.7 * w, 0.9 * h],
                "confidence": 0.9
            })
        } else {
            json!({
                "nodes": [{"local_id": "n1", "text": "Order Service",
                           "bbox_2d": [0.3 * w, 0.3 * h, 0.5 * w, 0.45 * h],
                           "conf": 0.9}],
                "edges": [],
                "stickies": [],
                "owner_tags": [],
                "other_visible_text": [{"text": "Overview",
                                        "bbox_2d": [0.02 * w, 0.02 * h, 0.1 * w, 0.05 * h]}],
                "confidence": 0.7
            })
        };
        request
            .schema
            .validate(&value)
            .map_err(|errors| VisionError::SchemaInvalid {
                attempts: 1,
                errors,
                raw_text: value.to_string(),
            })?;
        Ok(RawResponse {
            raw_text: value.to_string(),
            json: value,
            prompt_eval_count: Some(request.image.tokens() + 900),
            eval_count: Some(50),
            durations: Durations::default(),
            attempts: 1,
            repaired: false,
            done_reason: Some("stop".into()),
        })
    }
}

/// A keyword rule for a fictional content studio.
fn fictional_rules() -> ClassifyRules {
    ClassifyRules {
        rules: vec![KeywordRule {
            name: "quarry_studio".into(),
            screen_type: ScreenType::Cms,
            app_hint: Some("Quarry Studio".into()),
            patterns: ["Quarry", "Studio", "Structure", "Vision"]
                .iter()
                .map(|t| KeywordPattern::word(t))
                .collect(),
            min_matches: 2,
            confidence: 0.9,
        }],
        high_confidence: 0.8,
        min_model_confidence: 0.5,
    }
}

fn write_inputs(dir: &Path) {
    let kf = dir.join("keyframes");
    std::fs::create_dir_all(&kf).unwrap();
    let mut entries = Vec::new();
    for (i, t) in [2u32, 10, 20].iter().enumerate() {
        let name = format!("t_{t:06}.jpg");
        keyframe_image(i as u8).save(kf.join(&name)).unwrap();
        entries.push(json!({
            "file": format!("keyframes/{name}"),
            "t_start": f64::from(*t) - 1.0,
            "t_end": f64::from(*t) + 7.0,
            "t_rep": f64::from(*t),
        }));
    }
    std::fs::write(
        dir.join("index.json"),
        serde_json::to_vec(&json!({ "keyframes": entries })).unwrap(),
    )
    .unwrap();
}

fn read<T: DeserializeOwned>(run: &Path, schema: &str) -> Vec<(String, T)> {
    let path = run.join("artifacts").join(format!("{schema}.jsonl"));
    glassrip_core::jsonl::read::<Record<T>>(&path, &SchemaReq::new(schema, 1))
        .unwrap()
        .items
        .into_iter()
        .filter_map(|r| r.outcome.result.map(|v| (r.id, v)))
        .collect()
}

/// The branch with one greedy board read per keyframe (the per-read paths).
async fn run_branch(
    root: &Path,
    run: &Path,
    backend: Arc<dyn VisionBackend>,
) -> (Arc<PlacementMonitor>, Runner) {
    let (monitor, runner, reports) = run_branch_with(
        root,
        run,
        backend,
        BoardReadParams {
            consensus: ConsensusParams::single(),
            ..BoardReadParams::default()
        },
    )
    .await;
    let reports = reports.unwrap();
    assert!(reports.iter().all(|r| r.items_error == 0), "{reports:?}");
    (monitor, runner)
}

async fn run_branch_with(
    root: &Path,
    run: &Path,
    backend: Arc<dyn VisionBackend>,
    board_read: BoardReadParams,
) -> (
    Arc<PlacementMonitor>,
    Runner,
    Result<Vec<glassrip_core::runner::StageReport>, pipeline::PipelineError>,
) {
    adapter::build_inputs(run, &root.join("index.json"), None, "test-run").unwrap();
    let rd = RunDir::open(run, "test-run", Producer::glassrip("0.1.0", None)).unwrap();
    let mut runner = Runner::new(
        rd,
        pipeline::graph().unwrap(),
        &Selection::default(),
        Cache::new(run.join("cache")),
        RunnerOptions::default(),
        CancellationToken::new(),
    )
    .unwrap();
    let client = VisionClient::new(backend, 4).unwrap();
    let monitor = Arc::new(PlacementMonitor::new(
        Arc::new(StaticProbe),
        client,
        MonitorConfig {
            check_every: 2,
            ..MonitorConfig::default()
        },
    ));
    let branch = VisionBranch {
        monitor: Arc::clone(&monitor),
        ocr: OcrHarvestStage::new(
            Arc::new(ScriptedOcr),
            OcrConfig::default(),
            LayoutConfig::default(),
        ),
        vocabulary: VocabularyStage::new(VocabularyParams::default()),
        classify: ClassifyStage::new(
            ClassifyParams {
                rules: fictional_rules(),
                ..ClassifyParams::default()
            },
            Arc::clone(&monitor),
            "scripted-vl",
            None,
            None,
        ),
        canvas: CanvasCropStage::new(CanvasCropParams::default()),
        board_read: BoardReadStage::new(
            board_read,
            Arc::clone(&monitor),
            "scripted-vl",
            None,
            None,
        ),
        board_validate: BoardValidateStage::new(BoardValidateParams::default()),
    };
    let reports = branch.run(&mut runner, None).await;
    if let Ok(r) = &reports {
        assert_eq!(r.len(), 6);
    }
    (monitor, runner, reports)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn record_then_replay_vision_branch() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_inputs(root);
    let store = RawStore::new(root.join("raw"));

    // Live run against the scripted model, recording every answer.
    let live = root.join("live");
    let (monitor, _) = run_branch(
        root,
        &live,
        Arc::new(RecordingBackend::new(
            Arc::new(ScriptedModel),
            store.clone(),
        )),
    )
    .await;
    // 3 classify + 2 board requests; checks after every 2nd request.
    assert_eq!(monitor.completed(), 5);
    let checks = monitor.checks();
    let placements = checks.iter().filter(|c| c.fully_on_gpu.is_some()).count();
    assert_eq!(placements, 3, "{checks:?}");
    // Digest: at preflight, before each model stage, and at each periodic check.
    let digests = checks.iter().filter(|c| c.digest.is_some()).count();
    assert_eq!(digests, 5, "{checks:?}");
    assert!(monitor.abort_error().is_none());

    let ocr: Vec<(String, OcrKeyframe)> = read(&live, artifacts::OCR);
    assert_eq!(ocr.len(), 3);
    assert!(ocr[0].1.tile_names.iter().any(|n| n == "Ada Quill"));
    let vocab: Vec<(String, Vocabulary)> = read(&live, artifacts::ASR_VOCABULARY);
    assert_eq!(vocab[0].1.terms[0].text, "Ada Quill");

    let classes: Vec<(String, ScreenClassItem)> = read(&live, artifacts::SCREEN_CLASS);
    let types: Vec<ScreenType> = classes.iter().map(|(_, c)| c.screen_type).collect();
    assert_eq!(
        types,
        vec![
            ScreenType::Whiteboard,
            ScreenType::Whiteboard,
            ScreenType::Cms
        ],
        "the CMS keyframe must not read as a whiteboard"
    );

    let crops: Vec<(String, CanvasCropItem)> = read(&live, artifacts::CANVAS_CROP);
    assert_eq!(crops.len(), 2);
    let c = &crops[0].1;
    assert!(
        c.canvas_bbox.x2 < 925.0,
        "canvas must stop before the tiles: {:?}",
        c.canvas_bbox
    );
    assert!(c.canvas_bbox.y1 > 48.0);
    assert!(c.stabilized);
    assert!(c.masks.iter().any(|m| m.text == "Convert to"));

    let readings: Vec<(String, BoardReadingItem)> = read(&live, artifacts::BOARD_READING);
    assert_eq!(readings.len(), 2);
    for (_, r) in &readings {
        assert_eq!(r.requests.len(), 1);
        assert!(store.get(&r.requests[0].request_key).unwrap().is_some());
        assert_eq!(r.requests[0].degenerate, None, "a sound board is untouched");
    }

    let validated: Vec<(String, BoardValidateItem)> = read(&live, artifacts::BOARD_VALIDATE);
    assert_eq!(validated.len(), 2);
    let v = &validated[0].1.board;
    assert_eq!(v.nodes.len(), 1);
    assert!(v.chrome_rejected.iter().any(|r| r.text == "Overview"));

    // Replay in a fresh run directory: no model, same results.
    let replay = root.join("replay");
    run_branch(
        root,
        &replay,
        Arc::new(ReplayBackend::new("scripted-vl", store)),
    )
    .await;
    let again: Vec<(String, BoardValidateItem)> = read(&replay, artifacts::BOARD_VALIDATE);
    assert_eq!(again, validated);
    let classes_again: Vec<(String, ScreenClassItem)> = read(&replay, artifacts::SCREEN_CLASS);
    assert_eq!(classes_again, classes);
}

/// board_validate output feeds glassrip-meeting's edge_direction and
/// board_state stages unchanged: the canvas crop is found from the source
/// frame and crop box, and the validated node reaches the board state.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn board_validate_feeds_board_state() {
    use glassrip_meeting::artifacts::EdgeDirectionBatch;
    use glassrip_meeting::consolidate::{BoardStateItem, ConsolidationParams};
    use glassrip_meeting::pixel_direction::PixelCheckParams;
    use glassrip_meeting::stages::{BoardStateStage, EdgeDirectionStage};
    use glassrip_meeting::vlm_direction::VlmCheckParams;

    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_inputs(root);
    let store = RawStore::new(root.join("raw"));
    let run = root.join("run");
    let (_, mut runner) = run_branch(
        root,
        &run,
        Arc::new(RecordingBackend::new(Arc::new(ScriptedModel), store)),
    )
    .await;
    let edge =
        EdgeDirectionStage::new(PixelCheckParams::default(), VlmCheckParams::default(), None);
    let r = runner.run_stage(&edge).await.unwrap();
    assert_eq!(r.items_error, 0, "{r:?}");
    let state = BoardStateStage::new(ConsolidationParams::default());
    let r = runner.run_stage(&state).await.unwrap();
    assert_eq!(r.items_error, 0, "{r:?}");

    let batches: Vec<(String, EdgeDirectionBatch)> = read(&run, "glassrip.edge_direction");
    let frames: Vec<_> = batches.iter().flat_map(|(_, b)| &b.keyframes).collect();
    assert_eq!(frames.len(), 2);
    for f in &frames {
        assert!(f.error.is_none(), "{:?}", f.error);
        // The canvas was cut from the full keyframe: its size is the crop's.
        assert!(
            f.canvas.width < 900.0 && f.canvas.width > 600.0,
            "{:?}",
            f.canvas
        );
    }
    let states: Vec<(String, BoardStateItem)> = read(&run, "glassrip.board_state");
    assert!(!states.is_empty());
    assert!(states
        .iter()
        .any(|(_, s)| s.nodes.iter().any(|n| n.text == "Order Service")));
}

/// Scripted model whose full-budget board reads stop at the output limit; the
/// compact retry (halved list budgets) gets the normal answer.
struct TruncatingModel;

#[async_trait]
impl VisionBackend for TruncatingModel {
    fn id(&self) -> BackendId {
        ScriptedModel.id()
    }
    async fn preflight(&self) -> glassrip_vision::Result<Placement> {
        Err(VisionError::Config("unused".into()))
    }
    async fn infer(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> glassrip_vision::Result<RawResponse> {
        let nodes_budget = request.schema.json()["properties"]["nodes"]["maxItems"].as_u64();
        if nodes_budget == Some(60) {
            return Err(VisionError::Truncated {
                num_predict: request.options.num_predict,
                eval_count: Some(request.options.num_predict),
                raw_text: "{\n  \"nodes\": [\n    {\"local_id\": \"n1\",".into(),
            });
        }
        ScriptedModel.infer(request, cancel).await
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn truncated_board_reads_retry_compact_and_replay() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_inputs(root);
    let store = RawStore::new(root.join("raw"));
    let live = root.join("live");
    let (monitor, _) = run_branch(
        root,
        &live,
        Arc::new(RecordingBackend::new(
            Arc::new(TruncatingModel),
            store.clone(),
        )),
    )
    .await;
    // 3 classify + 2 truncated board reads + 2 compact retries.
    assert_eq!(monitor.completed(), 7);
    let readings: Vec<(String, BoardReadingItem)> = read(&live, artifacts::BOARD_READING);
    assert_eq!(readings.len(), 2);
    for (_, r) in &readings {
        assert_eq!(r.requests.len(), 1);
        assert!(r.requests[0].compact_retry, "{:?}", r.requests[0]);
        assert_eq!(r.result.nodes[0].text, "Order Service");
    }

    // Replay reproduces the truncation and the retry without a model.
    let replay = root.join("replay");
    let (_, _) = run_branch(
        root,
        &replay,
        Arc::new(ReplayBackend::new("scripted-vl", store.clone())),
    )
    .await;
    let again: Vec<(String, BoardReadingItem)> = read(&replay, artifacts::BOARD_READING);
    assert_eq!(
        again.iter().map(|(_, r)| &r.result).collect::<Vec<_>>(),
        readings.iter().map(|(_, r)| &r.result).collect::<Vec<_>>()
    );
    assert!(again.iter().all(|(_, r)| r.requests[0].compact_retry));
    assert_eq!(
        again[0].1.requests[0].request_key,
        readings[0].1.requests[0].request_key
    );
}

/// A board read that loops the same stepped edge until it is stopped.
fn looping_text() -> String {
    let mut s = String::from(
        "{\n  \"nodes\": [\n    {\"local_id\": \"n1\", \"text\": \"Order Service\", \"bbox_2d\": [330, 280, 470, 298], \"conf\": 0.9}\n  ],\n  \"edges\": [\n",
    );
    for k in 15..25 {
        s.push_str(&format!(
            "    {{\"src\": \"n{k}\", \"dst\": \"n{}\", \"label\": \"Relay\", \"label_bbox_2d\": [10, 20, 30, 40], \"style\": \"solid\", \"conf\": 0.9}},\n",
            k + 1
        ));
    }
    s
}

/// Scripted model whose board reads loop unless a repeat penalty is set. With
/// `returned`, the loop comes back as a reply cut off at the output limit (a
/// non-streaming server); otherwise the streaming client stopped it.
struct LoopingModel {
    returned: bool,
}

#[async_trait]
impl VisionBackend for LoopingModel {
    fn id(&self) -> BackendId {
        ScriptedModel.id()
    }
    async fn preflight(&self) -> glassrip_vision::Result<Placement> {
        Err(VisionError::Config("unused".into()))
    }
    async fn infer(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> glassrip_vision::Result<RawResponse> {
        let board = request.schema.json()["properties"]["nodes"].is_object();
        if board && request.sampling.repeat_penalty.is_none() {
            assert!(
                request.repetition_guard.is_some(),
                "board reads are guarded"
            );
            let raw_text = looping_text();
            if self.returned {
                return Err(VisionError::Truncated {
                    num_predict: request.options.num_predict,
                    eval_count: Some(request.options.num_predict),
                    raw_text,
                });
            }
            let finding = glassrip_vision::repetition::detect(
                &raw_text,
                request.repetition_guard.as_ref().unwrap(),
            )
            .unwrap();
            return Err(VisionError::Repetition {
                num_predict: request.options.num_predict,
                finding,
                raw_text,
            });
        }
        ScriptedModel.infer(request, cancel).await
    }
}

async fn looping_reads_retry_with_a_repeat_penalty_and_replay(returned: bool) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_inputs(root);
    let store = RawStore::new(root.join("raw"));
    let live = root.join("live");
    let (monitor, _) = run_branch(
        root,
        &live,
        Arc::new(RecordingBackend::new(
            Arc::new(LoopingModel { returned }),
            store.clone(),
        )),
    )
    .await;
    // 3 classify + 2 looping board reads + 2 penalized retries.
    assert_eq!(monitor.completed(), 7);
    let readings: Vec<(String, BoardReadingItem)> = read(&live, artifacts::BOARD_READING);
    assert_eq!(readings.len(), 2);
    let params = BoardReadParams::default();
    for (_, r) in &readings {
        let log = &r.requests[0];
        assert!(!log.compact_retry, "a loop is not a plain truncation");
        let finding = log.repetition.as_ref().expect("the loop is logged");
        assert!(finding.pattern.contains("Relay"), "{finding:?}");
        assert_eq!(
            log.sampling.repeat_penalty,
            Some(params.repetition_retry_penalty)
        );
        assert_eq!(
            log.sampling.repeat_last_n,
            Some(params.repetition_retry_last_n)
        );
        assert!(log.num_predict.unwrap() < params.max_num_predict);
        assert_eq!(r.result.nodes[0].text, "Order Service");
    }

    // Replay rebuilds the same retry from the recorded stop, without a model.
    let replay = root.join("replay");
    let (_, _) = run_branch(
        root,
        &replay,
        Arc::new(ReplayBackend::new("scripted-vl", store.clone())),
    )
    .await;
    let again: Vec<(String, BoardReadingItem)> = read(&replay, artifacts::BOARD_READING);
    assert_eq!(
        again
            .iter()
            .map(|(_, r)| &r.requests)
            .collect::<Vec<_>>()
            .len(),
        2
    );
    for ((_, a), (_, b)) in again.iter().zip(&readings) {
        assert_eq!(a.result, b.result);
        assert_eq!(a.requests[0].request_key, b.requests[0].request_key);
        assert_eq!(a.requests[0].repetition, b.requests[0].repetition);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn streamed_loops_retry_with_a_repeat_penalty_and_replay() {
    looping_reads_retry_with_a_repeat_penalty_and_replay(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn returned_loops_retry_with_a_repeat_penalty_and_replay() {
    looping_reads_retry_with_a_repeat_penalty_and_replay(true).await;
}

/// GLM B1: a board read cut off at the output limit inside a regular row of six
/// identical cards (sequential ids, evenly spaced boxes). The row is not a loop, so
/// the budget (compact) retry runs, not the repeat-penalty retry.
struct CardRowModel;

fn card_row_text() -> String {
    let mut s = String::from("{\n  \"nodes\": [\n");
    for c in 0..6u32 {
        let x = 100 + c * 120;
        s.push_str(&format!(
            "    {{\"local_id\": \"n{}\", \"text\": \"Card\", \"bbox_2d\": [{x}, 400, {}, 440], \"conf\": 0.9}},\n",
            30 + c,
            x + 80
        ));
    }
    s
}

#[async_trait]
impl VisionBackend for CardRowModel {
    fn id(&self) -> BackendId {
        ScriptedModel.id()
    }
    async fn preflight(&self) -> glassrip_vision::Result<Placement> {
        Err(VisionError::Config("unused".into()))
    }
    async fn infer(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> glassrip_vision::Result<RawResponse> {
        let nodes_budget = request.schema.json()["properties"]["nodes"]["maxItems"].as_u64();
        if nodes_budget == Some(60) {
            return Err(VisionError::Truncated {
                num_predict: request.options.num_predict,
                eval_count: Some(request.options.num_predict),
                raw_text: card_row_text(),
            });
        }
        ScriptedModel.infer(request, cancel).await
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_regular_row_cut_at_the_limit_takes_the_budget_retry() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_inputs(root);
    let store = RawStore::new(root.join("raw"));
    let live = root.join("live");
    run_branch(
        root,
        &live,
        Arc::new(RecordingBackend::new(Arc::new(CardRowModel), store.clone())),
    )
    .await;
    let readings: Vec<(String, BoardReadingItem)> = read(&live, artifacts::BOARD_READING);
    assert_eq!(readings.len(), 2);
    for (_, r) in &readings {
        let log = &r.requests[0];
        assert!(log.compact_retry, "{log:?}");
        assert_eq!(log.repetition, None, "a regular row is not a loop");
        assert_eq!(log.sampling.repeat_penalty, None);
    }
    let replay = root.join("replay");
    run_branch(
        root,
        &replay,
        Arc::new(ReplayBackend::new("scripted-vl", store)),
    )
    .await;
    let again: Vec<(String, BoardReadingItem)> = read(&replay, artifacts::BOARD_READING);
    for ((_, a), (_, b)) in again.iter().zip(&readings) {
        assert_eq!(a.result, b.result);
        assert_eq!(a.requests[0].request_key, b.requests[0].request_key);
    }
}

/// Scripted model whose board reads come back complete and valid but list the
/// one real box six times under fresh ids (stepped boxes), with an edge to each
/// copy (or the answer `board` builds). The repeat-penalty retry answers as
/// `retry` says.
struct DegenerateModel {
    retry: RetryReply,
    board: fn(f64, f64) -> serde_json::Value,
}

#[derive(Clone, Copy, PartialEq)]
enum RetryReply {
    /// The normal one-box answer.
    Clean,
    /// The same degenerate answer.
    Degenerate,
    /// Stopped at the output limit.
    Truncated,
}

fn degenerate_board(w: f64, h: f64) -> serde_json::Value {
    let mut nodes = vec![json!({"local_id": "n1", "text": "Ledger",
        "bbox_2d": [0.05 * w, 0.05 * h, 0.2 * w, 0.15 * h], "conf": 0.9})];
    let mut edges = Vec::new();
    for k in 0..6 {
        let dx = f64::from(k) * 4.0;
        nodes.push(
            json!({"local_id": format!("n{}", k + 2), "text": "Order Service",
            "bbox_2d": [0.3 * w + dx, 0.3 * h, 0.5 * w + dx, 0.45 * h], "conf": 0.9}),
        );
        edges.push(
            json!({"src": "n1", "dst": format!("n{}", k + 2), "label": "",
            "label_bbox_2d": [0, 0, 0, 0], "style": "solid", "conf": 0.8}),
        );
    }
    json!({"nodes": nodes, "edges": edges, "stickies": [], "owner_tags": [],
           "other_visible_text": [], "confidence": 0.7})
}

/// Eight separate boxes, each with an "Avery" owner tag under it: one person
/// owning eight things, a name OCR does not read.
fn one_owner_board(w: f64, h: f64) -> serde_json::Value {
    let mut nodes = Vec::new();
    let mut owner_tags = Vec::new();
    for k in 0..8 {
        let x = 0.02 * w + f64::from(k) * 0.12 * w;
        nodes.push(
            json!({"local_id": format!("n{}", k + 1), "text": format!("Step {k}"),
            "bbox_2d": [x, 0.3 * h, x + 0.1 * w, 0.4 * h], "conf": 0.9}),
        );
        owner_tags.push(json!({"name_raw": "Avery", "near": format!("n{}", k + 1),
            "bbox_2d": [x, 0.42 * h, x + 0.05 * w, 0.46 * h]}));
    }
    json!({"nodes": nodes, "edges": [], "stickies": [], "owner_tags": owner_tags,
           "other_visible_text": [], "confidence": 0.7})
}

#[async_trait]
impl VisionBackend for DegenerateModel {
    fn id(&self) -> BackendId {
        ScriptedModel.id()
    }
    async fn preflight(&self) -> glassrip_vision::Result<Placement> {
        Err(VisionError::Config("unused".into()))
    }
    async fn infer(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> glassrip_vision::Result<RawResponse> {
        let board = request.schema.json()["properties"]["nodes"].is_object();
        let retry = request.sampling.repeat_penalty.is_some();
        if board && retry && self.retry == RetryReply::Truncated {
            return Err(VisionError::Truncated {
                num_predict: request.options.num_predict,
                eval_count: Some(request.options.num_predict),
                raw_text: "{\"nodes\": [".into(),
            });
        }
        if board && (!retry || self.retry == RetryReply::Degenerate) {
            let value = (self.board)(
                f64::from(request.image.width()),
                f64::from(request.image.height()),
            );
            request.schema.validate(&value).unwrap();
            return Ok(RawResponse {
                raw_text: value.to_string(),
                json: value,
                prompt_eval_count: Some(request.image.tokens() + 900),
                eval_count: Some(400),
                durations: Durations::default(),
                attempts: 1,
                repaired: false,
                done_reason: Some("stop".into()),
            });
        }
        ScriptedModel.infer(request, cancel).await
    }
}

async fn degenerate_reads_retry_then_collapse_and_replay(retry: RetryReply) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_inputs(root);
    let store = RawStore::new(root.join("raw"));
    let live = root.join("live");
    let (monitor, _) = run_branch(
        root,
        &live,
        Arc::new(RecordingBackend::new(
            Arc::new(DegenerateModel {
                retry,
                board: degenerate_board,
            }),
            store.clone(),
        )),
    )
    .await;
    // 3 classify + 2 degenerate board reads + 2 penalized retries.
    assert_eq!(monitor.completed(), 7);
    let readings: Vec<(String, BoardReadingItem)> = read(&live, artifacts::BOARD_READING);
    assert_eq!(readings.len(), 2);
    let params = BoardReadParams::default();
    for (_, r) in &readings {
        let log = &r.requests[0];
        let d = log
            .degenerate
            .as_ref()
            .expect("the degenerate reply is logged");
        let rep = &d.finding.repeated[0];
        assert_eq!((rep.text.as_str(), rep.count), ("order service", 6));
        if retry == RetryReply::Truncated {
            assert!(!d.retried, "{d:?}");
            assert_eq!(d.retry_error.as_deref(), Some("output limit"));
            assert_eq!(log.sampling.repeat_penalty, None, "the first reply is kept");
        } else {
            assert!(d.retried && d.retry_error.is_none(), "{d:?}");
            assert_eq!(
                log.sampling.repeat_penalty,
                Some(params.repetition_retry_penalty),
                "the reading is the retry's"
            );
        }
        assert_eq!(log.repetition, None);
        assert!(!log.compact_retry);
        let copies = r
            .result
            .nodes
            .iter()
            .filter(|n| n.text == "Order Service")
            .count();
        assert_eq!(copies, 1);
        if retry != RetryReply::Clean {
            assert_eq!(d.collapsed.len(), 1, "{d:?}");
            assert_eq!((d.collapsed[0].before, d.collapsed[0].after), (6, 1));
            // Six edges to the copies are one edge to the kept box.
            assert_eq!(r.result.nodes.len(), 2);
            assert_eq!(r.result.edges.len(), 1);
        } else {
            assert!(d.collapsed.is_empty(), "{d:?}");
            assert_eq!(r.result.nodes.len(), 1);
        }
    }

    // Replay rebuilds the same retry and the same collapse without a model.
    let replay = root.join("replay");
    run_branch(
        root,
        &replay,
        Arc::new(ReplayBackend::new("scripted-vl", store)),
    )
    .await;
    let again_read: Vec<(String, BoardReadingItem)> = read(&replay, artifacts::BOARD_READING);
    assert_eq!(again_read.len(), 2);
    for ((_, a), (_, b)) in again_read.iter().zip(&readings) {
        assert_eq!(a.result, b.result);
        assert_eq!(a.requests[0].request_key, b.requests[0].request_key);
        assert_eq!(a.requests[0].degenerate, b.requests[0].degenerate);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn degenerate_reads_take_the_repeat_penalty_retry_and_replay() {
    degenerate_reads_retry_then_collapse_and_replay(RetryReply::Clean).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn still_degenerate_reads_are_collapsed_and_replay() {
    degenerate_reads_retry_then_collapse_and_replay(RetryReply::Degenerate).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_degenerate_retry_keeps_the_first_reply_and_replays() {
    degenerate_reads_retry_then_collapse_and_replay(RetryReply::Truncated).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn separate_owner_tags_the_retry_repeats_are_kept_and_replay() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_inputs(root);
    let store = RawStore::new(root.join("raw"));
    let live = root.join("live");
    let (monitor, _) = run_branch(
        root,
        &live,
        Arc::new(RecordingBackend::new(
            Arc::new(DegenerateModel {
                retry: RetryReply::Degenerate,
                board: one_owner_board,
            }),
            store.clone(),
        )),
    )
    .await;
    // 3 classify + 2 flagged board reads + 2 penalized retries.
    assert_eq!(monitor.completed(), 7);
    let readings: Vec<(String, BoardReadingItem)> = read(&live, artifacts::BOARD_READING);
    assert_eq!(readings.len(), 2);
    for (_, r) in &readings {
        let d = r.requests[0]
            .degenerate
            .as_ref()
            .expect("the repeated owner is flagged for the retry");
        let rep = &d.finding.repeated[0];
        assert_eq!((rep.text.as_str(), rep.count), ("avery", 8));
        assert!(d.retried && d.retry_error.is_none(), "{d:?}");
        // The retry repeats the tags side by side: every assignment stays.
        assert!(d.collapsed.is_empty(), "{d:?}");
        assert_eq!(r.result.owner_tags.len(), 8);
        let near: Vec<&str> = r
            .result
            .owner_tags
            .iter()
            .map(|o| o.near.as_str())
            .collect();
        assert_eq!(near, ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8"]);
    }

    let replay = root.join("replay");
    run_branch(
        root,
        &replay,
        Arc::new(ReplayBackend::new("scripted-vl", store)),
    )
    .await;
    let again: Vec<(String, BoardReadingItem)> = read(&replay, artifacts::BOARD_READING);
    assert_eq!(again.len(), 2);
    for ((_, a), (_, b)) in again.iter().zip(&readings) {
        assert_eq!(a.result, b.result);
        assert_eq!(a.requests[0].request_key, b.requests[0].request_key);
        assert_eq!(a.requests[0].degenerate, b.requests[0].degenerate);
    }
}

/// Scripted model whose three consensus reads differ the way sampled replies
/// do: every read has the service box (at slightly different places), the
/// greedy read and one sampled read have the ledger, and one sampled read adds
/// a box no other read has. Reads in `fail` answer with a protocol error.
struct SampledModel {
    fail: &'static [u64],
}

impl SampledModel {
    fn board(seed: u64, w: f64, h: f64) -> serde_json::Value {
        let dx = seed as f64 * 0.004 * w;
        let mut nodes = vec![
            json!({"local_id": format!("s{seed}"), "text": "Order Service",
                   "bbox_2d": [0.3 * w + dx, 0.3 * h, 0.5 * w + dx, 0.45 * h], "conf": 0.9}),
        ];
        let mut edges = Vec::new();
        if seed != 1 {
            nodes.push(json!({"local_id": format!("l{seed}"), "text": "Ledger",
                   "bbox_2d": [0.6 * w, 0.6 * h, 0.8 * w, 0.72 * h], "conf": 0.8}));
            edges.push(json!({"src": format!("s{seed}"), "dst": format!("l{seed}"),
                   "label": "posts", "label_bbox_2d": [0.52 * w, 0.5 * h, 0.58 * w, 0.54 * h],
                   "style": "solid", "conf": 0.7}));
        }
        if seed == 1 {
            nodes.push(json!({"local_id": "g1", "text": "Ghost Queue",
                   "bbox_2d": [0.05 * w, 0.7 * h, 0.2 * w, 0.8 * h], "conf": 0.6}));
        }
        json!({"nodes": nodes, "edges": edges, "stickies": [], "owner_tags": [],
               "other_visible_text": [], "confidence": 0.7})
    }
}

#[async_trait]
impl VisionBackend for SampledModel {
    fn id(&self) -> BackendId {
        ScriptedModel.id()
    }
    async fn preflight(&self) -> glassrip_vision::Result<Placement> {
        Err(VisionError::Config("unused".into()))
    }
    async fn infer(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> glassrip_vision::Result<RawResponse> {
        let board = request.schema.json()["properties"]["nodes"].is_object();
        if !board {
            return ScriptedModel.infer(request, cancel).await;
        }
        let seed = request.options.seed;
        // Read 0 is greedy; the sampled reads carry the temperature.
        assert_eq!(request.sampling.temperature.is_some(), seed > 0, "{seed}");
        if self.fail.contains(&seed) {
            return Err(VisionError::Protocol(format!(
                "scripted failure of read {seed}"
            )));
        }
        let value = Self::board(
            seed,
            f64::from(request.image.width()),
            f64::from(request.image.height()),
        );
        request.schema.validate(&value).unwrap();
        Ok(RawResponse {
            raw_text: value.to_string(),
            json: value,
            prompt_eval_count: Some(request.image.tokens() + 900),
            eval_count: Some(120),
            durations: Durations::default(),
            attempts: 1,
            repaired: false,
            done_reason: Some("stop".into()),
        })
    }
}

fn consensus_params() -> BoardReadParams {
    BoardReadParams {
        consensus: ConsensusParams {
            reads: 3,
            min_agree: 2,
            temperature: 0.3,
        },
        ..BoardReadParams::default()
    }
}

/// A consensus run recorded with the sampled model and its replay.
struct ConsensusRuns {
    live: Vec<(String, BoardReadingItem)>,
    replay: Vec<(String, BoardReadingItem)>,
    /// Model requests the live run completed.
    completed: usize,
    /// Items that failed (the same in both runs).
    errors: u64,
}

/// [`read`], or nothing when the stage failed before writing its artifact.
fn read_if_any<T: DeserializeOwned>(run: &Path, schema: &str) -> Vec<(String, T)> {
    if run
        .join("artifacts")
        .join(format!("{schema}.jsonl"))
        .exists()
    {
        read(run, schema)
    } else {
        Vec::new()
    }
}

/// Failed items of a branch run: the reports' count, or the failed board_read
/// items when they exceeded the runner's error rate.
fn failed_items(
    reports: &Result<Vec<glassrip_core::runner::StageReport>, pipeline::PipelineError>,
) -> u64 {
    match reports {
        Ok(r) => r.iter().map(|r| r.items_error).sum(),
        Err(pipeline::PipelineError::Runner(
            glassrip_core::runner::RunnerError::ErrorRateExceeded { stage, errors, .. },
        )) if stage == "board_read" => *errors,
        Err(e) => panic!("{e}"),
    }
}

/// Record with the sampled model, then replay in a fresh directory.
async fn consensus_record_and_replay(fail: &'static [u64]) -> ConsensusRuns {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_inputs(root);
    let store = RawStore::new(root.join("raw"));
    let live = root.join("live");
    let (monitor, _, reports) = run_branch_with(
        root,
        &live,
        Arc::new(RecordingBackend::new(
            Arc::new(SampledModel { fail }),
            store.clone(),
        )),
        consensus_params(),
    )
    .await;
    let errors = failed_items(&reports);
    let completed = monitor.completed();
    let readings: Vec<(String, BoardReadingItem)> = read_if_any(&live, artifacts::BOARD_READING);
    for (_, r) in &readings {
        for log in &r.requests {
            assert!(store.get(&log.request_key).unwrap().is_some(), "{log:?}");
        }
    }
    let replay = root.join("replay");
    let (_, _, again_reports) = run_branch_with(
        root,
        &replay,
        Arc::new(ReplayBackend::new("scripted-vl", store)),
        consensus_params(),
    )
    .await;
    let again_errors = failed_items(&again_reports);
    assert_eq!(again_errors, errors, "replay fails the same items");
    let again: Vec<(String, BoardReadingItem)> = read_if_any(&replay, artifacts::BOARD_READING);
    ConsensusRuns {
        live: readings,
        replay: again,
        completed,
        errors,
    }
}

fn same_reading(a: &BoardReadingItem, b: &BoardReadingItem) {
    assert_eq!(a.result, b.result);
    // A failed read's error text is the live failure in one run and the
    // missing record in the replay (no failed reply is recorded, as for a
    // failed item); which reads failed, and the vote, are the same.
    let votes = |r: &BoardReadingItem| {
        r.consensus.clone().map(|mut c| {
            let failed: Vec<u32> = c.failed.iter().map(|f| f.read).collect();
            c.failed.clear();
            (c, failed)
        })
    };
    assert_eq!(votes(a), votes(b));
    let keys = |r: &BoardReadingItem| {
        r.requests
            .iter()
            .map(|l| (l.read, l.request_key.clone(), l.sampling))
            .collect::<Vec<_>>()
    };
    assert_eq!(keys(a), keys(b));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn consensus_reads_vote_record_three_replies_and_replay() {
    let ConsensusRuns {
        live,
        replay,
        completed,
        errors,
    } = consensus_record_and_replay(&[]).await;
    // 3 classify + 2 keyframes x 3 reads.
    assert_eq!((completed, errors), (9, 0));
    assert_eq!(live.len(), 2);
    for (_, r) in &live {
        let reads: Vec<u32> = r.requests.iter().map(|l| l.read).collect();
        assert_eq!(reads, [0, 1, 2]);
        let keys: std::collections::BTreeSet<&str> =
            r.requests.iter().map(|l| l.request_key.as_str()).collect();
        assert_eq!(keys.len(), 3, "three distinct recorded replies");
        assert_eq!(r.requests[0].sampling.temperature, None);
        assert_eq!(r.requests[1].sampling.temperature, Some(0.3));
        assert_eq!(r.requests[2].sampling.temperature, Some(0.3));
        // The ghost box one sampled read made up is gone; the ledger two reads
        // saw stays, with its edge.
        let texts: Vec<&str> = r.result.nodes.iter().map(|n| n.text.as_str()).collect();
        assert_eq!(texts, ["Order Service", "Ledger"]);
        assert_eq!(r.result.edges.len(), 1);
        assert_eq!(r.result.edges[0].label, "posts");
        let c = r.consensus.as_ref().expect("a consensus reading");
        assert_eq!((c.reads, c.min_agree, c.low_confidence), (3, 2, false));
        assert_eq!(c.answered, [0, 1, 2]);
        let votes: Vec<u32> = c.nodes.iter().map(|v| v.votes).collect();
        assert_eq!(votes, [3, 2]);
        assert_eq!(c.edges[0].votes, 2);
        assert_eq!(c.dropped.nodes, 1);
    }
    assert_eq!(replay.len(), live.len());
    for ((_, a), (_, b)) in replay.iter().zip(&live) {
        same_reading(a, b);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_failed_read_leaves_two_that_must_agree_and_replays() {
    // Read 1 (the one with the ghost and without the ledger) fails: the ledger
    // both others have stays, the service box too.
    let ConsensusRuns {
        live,
        replay,
        completed,
        errors,
    } = consensus_record_and_replay(&[1]).await;
    assert_eq!((completed, errors), (9, 0));
    for (_, r) in &live {
        let c = r.consensus.as_ref().unwrap();
        assert_eq!(c.answered, [0, 2]);
        assert_eq!(c.failed.len(), 1);
        assert_eq!(c.failed[0].read, 1);
        assert_eq!((c.min_agree, c.low_confidence), (2, false));
        let texts: Vec<&str> = r.result.nodes.iter().map(|n| n.text.as_str()).collect();
        assert_eq!(texts, ["Order Service", "Ledger"]);
        assert!(r.requests.iter().all(|l| l.read != 1));
    }
    for ((_, a), (_, b)) in replay.iter().zip(&live) {
        same_reading(a, b);
    }

    // Reads 0 and 2 fail: read 1 is kept as it came, ghost included, and the
    // reading is marked low confidence.
    let ConsensusRuns {
        live,
        replay,
        errors,
        ..
    } = consensus_record_and_replay(&[0, 2]).await;
    assert_eq!(errors, 0);
    assert_eq!(live.len(), 2);
    for (_, r) in &live {
        let c = r.consensus.as_ref().unwrap();
        assert_eq!(c.answered, [1]);
        assert!(c.low_confidence);
        let texts: Vec<&str> = r.result.nodes.iter().map(|n| n.text.as_str()).collect();
        assert_eq!(texts, ["Order Service", "Ghost Queue"]);
    }
    for ((_, a), (_, b)) in replay.iter().zip(&live) {
        same_reading(a, b);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_read_failing_fails_the_item() {
    let runs = consensus_record_and_replay(&[0, 1, 2]).await;
    assert!(runs.live.is_empty() && runs.replay.is_empty());
    assert_eq!(runs.errors, 2, "both keyframes failed");
}
