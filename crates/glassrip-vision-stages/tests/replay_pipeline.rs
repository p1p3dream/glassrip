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
            prompt_eval_count: Some(100),
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
    assert_eq!(monitor.checks().len(), 3);

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
