//! Synthetic suite in pipeline mode: a generated two-case suite (a board with an
//! arrow, a chat screen) runs through the real vision stages with a scripted
//! model and scripted OCR, recorded to response stores; a replay from those
//! stores (no model, no OCR engine) must give identical results. Fictional data.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use glassrip_eval::fixture::load_board_suite;
use glassrip_eval::pipeline::{
    run_pipeline_suite, stores, PipelineContext, RecordingBackend, RecordingRecognizer,
    ReplayBackend, ReplayRecognizer,
};
use glassrip_ocr::{OcrError, PixelBox, RecognizedSpan, TextRecognizer};
use glassrip_vision::{
    BackendId, Durations, Placement, RawResponse, VisionBackend, VisionError, VisionRequest,
};
use glassrip_vision_stages::placement::StaticProbe;
use image::{Rgb, RgbImage};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

const W: u32 = 1280;
const H: u32 = 720;
const A: [u32; 4] = [160, 200, 460, 320];
const B: [u32; 4] = [760, 200, 1060, 320];

fn rect(img: &mut RgbImage, b: [u32; 4], c: [u8; 3], t: u32) {
    for y in b[1]..b[3] {
        for x in b[0]..b[2] {
            let edge = x < b[0] + t || x >= b[2] - t || y < b[1] + t || y >= b[3] - t;
            if edge {
                img.put_pixel(x, y, Rgb(c));
            }
        }
    }
}

fn board_frame() -> RgbImage {
    let mut img = RgbImage::from_pixel(W, H, Rgb([250, 250, 250]));
    rect(&mut img, A, [30, 30, 30], 5);
    rect(&mut img, B, [30, 30, 30], 5);
    // Connector with an arrowhead at B. The head (14 px base, 12 px long on a
    // 5 px line) stays within the pixel check's search disk: a head wider than
    // 0.8 of `arrow_radius_px` is treated as a line crossing by design.
    for x in 460..748 {
        for y in 258..263 {
            img.put_pixel(x, y, Rgb([30, 30, 30]));
        }
    }
    for x in 748..760u32 {
        let half = (760 - x) as i64 * 7 / 12;
        for dy in -half..=half {
            img.put_pixel(x, (260 + dy) as u32, Rgb([30, 30, 30]));
        }
    }
    img
}

fn write_case(root: &Path, name: &str, screen: &str, img: &RgbImage, board: Value) {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    img.save(dir.join("frame.png")).unwrap();
    std::fs::write(
        dir.join("meta.toml"),
        format!(
            "case = \"{name}\"\nsuite = \"synthetic\"\ndescription = \"pipeline test\"\n\
             generator = \"test\"\ngenerator_version = 1\nseed = 1\nscreen_type = \"{screen}\"\n"
        ),
    )
    .unwrap();
    let expected = json!({
        "schema": "glassrip.eval.synthetic_board", "schema_version": "1.0.0",
        "screen_type": screen, "frame_width": W, "frame_height": H,
        "board": board, "chrome_texts": [], "participants": []
    });
    std::fs::write(dir.join("expected.json"), expected.to_string()).unwrap();
}

fn mean_luma(img: &RgbImage) -> f64 {
    img.pixels()
        .map(|p| f64::from(p[0]) + f64::from(p[1]) + f64::from(p[2]))
        .sum::<f64>()
        / (3.0 * f64::from(img.width()) * f64::from(img.height()))
}

struct ScriptedOcr;

impl TextRecognizer for ScriptedOcr {
    fn recognize(&self, img: &RgbImage) -> Result<Vec<RecognizedSpan>, OcrError> {
        let span = |t: &str, b: [f64; 4]| RecognizedSpan {
            text: t.into(),
            bbox: PixelBox {
                x1: b[0],
                y1: b[1],
                x2: b[2],
                y2: b[3],
            },
            confidence: 0.95,
            det_score: 0.9,
        };
        Ok(if mean_luma(img) < 100.0 {
            vec![span("general", [100.0, 130.0, 220.0, 160.0])]
        } else {
            vec![
                span("Ledger API", [200.0, 245.0, 380.0, 275.0]),
                span("Orbit Queue", [800.0, 245.0, 990.0, 275.0]),
            ]
        })
    }
    fn execution_provider(&self) -> String {
        "scripted".into()
    }
    fn model_fingerprint(&self) -> String {
        "scripted-ocr".into()
    }
}

struct ScriptedModel;

fn frac(b: [u32; 4], w: f64, h: f64) -> Value {
    json!([
        f64::from(b[0]) / f64::from(W) * w,
        f64::from(b[1]) / f64::from(H) * h,
        f64::from(b[2]) / f64::from(W) * w,
        f64::from(b[3]) / f64::from(H) * h
    ])
}

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
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(request.image.base64())
            .unwrap();
        let img = image::load_from_memory(&bytes).unwrap().to_rgb8();
        let (w, h) = (f64::from(img.width()), f64::from(img.height()));
        let board = mean_luma(&img) >= 100.0;
        let value = if request.schema.name().contains("ScreenClass") {
            json!({"screen_type": if board { "whiteboard" } else { "chat" }, "app_hint": "",
                   "bbox_2d": [0.0, 0.0, w, h], "confidence": 0.95})
        } else if request.schema.name().contains("BoardRead") {
            json!({
                "nodes": [
                    {"local_id": "n1", "text": "Ledger API", "bbox_2d": frac(A, w, h), "conf": 0.95},
                    {"local_id": "n2", "text": "Orbit Queue", "bbox_2d": frac(B, w, h), "conf": 0.95}
                ],
                "edges": [{"src": "n1", "dst": "n2", "label": "REST", "label_bbox_2d": [0, 0, 0, 0],
                           "style": "solid", "conf": 0.9}],
                "stickies": [], "owner_tags": [], "other_visible_text": [], "confidence": 0.9
            })
        } else {
            // Edge-direction fallback: no decision.
            return Err(VisionError::Config(format!(
                "unexpected request {}",
                request.schema.name()
            )));
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
            prompt_eval_count: Some(900),
            eval_count: Some(80),
            durations: Durations::default(),
            attempts: 1,
            repaired: false,
            done_reason: Some("stop".into()),
        })
    }
}

fn context(
    work: &Path,
    vision: Arc<dyn VisionBackend>,
    recognizer: Arc<dyn TextRecognizer>,
) -> PipelineContext {
    PipelineContext {
        vision,
        probe: Arc::new(StaticProbe),
        recognizer,
        model: "scripted-vl".into(),
        slots: 2,
        seed: 0,
        num_ctx: 8192,
        work_dir: work.to_path_buf(),
        cancel: CancellationToken::new(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pipeline_mode_records_then_replays_identically() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let fixtures = root.join("fixtures");
    write_case(
        &fixtures,
        "board_arrow",
        "whiteboard",
        &board_frame(),
        json!({
            "nodes": [{"id": "a", "text": "Ledger API"}, {"id": "b", "text": "Orbit Queue"}],
            "edges": [{"src": "a", "dst": "b", "label": "REST"}]
        }),
    );
    write_case(
        &fixtures,
        "chat_screen",
        "chat",
        &RgbImage::from_pixel(W, H, Rgb([28, 30, 38])),
        json!({}),
    );
    let cases = load_board_suite(&fixtures).unwrap();
    let responses = root.join("responses");
    let (raw, ocr) = stores(&responses);

    let live = run_pipeline_suite(
        &context(
            &root.join("work-live"),
            Arc::new(RecordingBackend::new(Arc::new(ScriptedModel), raw.clone())),
            Arc::new(RecordingRecognizer::new(Arc::new(ScriptedOcr), ocr.clone())),
        ),
        &cases,
    )
    .await;
    assert!(live.errors.is_empty(), "{:?}", live.errors);
    let m = &live.metrics;
    assert_eq!(m["screen.accuracy"], 1.0);
    assert_eq!(m["board.node.recall"], 1.0);
    assert_eq!(m["board.edge.recall"], 1.0);
    assert_eq!(m["cases"], 2.0);
    // The edge direction comes from the pixel check of edge_direction.
    let board_case = live.details["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["case"] == "board_arrow")
        .unwrap()
        .clone();
    let dir = board_case["board"]["edges"][0]["direction"].clone();
    assert_eq!(
        dir,
        json!("forward"),
        "the drawn arrowhead must be detected"
    );

    // Replay: no model, no OCR engine; identical results.
    let replay = run_pipeline_suite(
        &context(
            &root.join("work-replay"),
            Arc::new(ReplayBackend::new("scripted-vl", raw.clone())),
            Arc::new(ReplayRecognizer::new(ocr.clone())),
        ),
        &cases,
    )
    .await;
    assert!(replay.errors.is_empty(), "{:?}", replay.errors);
    assert_eq!(replay.metrics, live.metrics);
    let strip = |v: &Value| -> Value {
        let mut v = v.clone();
        for c in v["cases"].as_array_mut().unwrap() {
            c["latency_s"] = json!(0);
        }
        v
    };
    assert_eq!(strip(&replay.details), strip(&live.details));

    // Replay against empty stores reports case errors instead of failing.
    let (empty_raw, empty_ocr) = stores(&root.join("empty"));
    let missing = run_pipeline_suite(
        &context(
            &root.join("work-missing"),
            Arc::new(ReplayBackend::new("scripted-vl", empty_raw)),
            Arc::new(ReplayRecognizer::new(empty_ocr)),
        ),
        &cases,
    )
    .await;
    assert_eq!(missing.errors.len(), 2, "{:?}", missing.errors);
}
