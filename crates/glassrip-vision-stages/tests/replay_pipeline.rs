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
    CanvasCropStage, ClassifyParams, ClassifyStage, OcrHarvestStage, VocabularyParams,
    VocabularyStage,
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

async fn run_branch(
    root: &Path,
    run: &Path,
    backend: Arc<dyn VisionBackend>,
) -> (Arc<PlacementMonitor>, Runner) {
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
            BoardReadParams::default(),
            Arc::clone(&monitor),
            "scripted-vl",
            None,
            None,
        ),
        board_validate: BoardValidateStage::new(BoardValidateParams::default()),
    };
    let reports = branch.run(&mut runner, None).await.unwrap();
    assert_eq!(reports.len(), 6);
    assert!(reports.iter().all(|r| r.items_error == 0), "{reports:?}");
    (monitor, runner)
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
/// copy. The repeat-penalty retry answers as `retry` says.
struct DegenerateModel {
    retry: RetryReply,
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
            let value = degenerate_board(
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
            Arc::new(DegenerateModel { retry }),
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
