//! `edge_direction` and `board_state` through the stage runner on rendered synthetic
//! boards.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use common::Canvas;
use glassrip_core::cache::Cache;
use glassrip_core::envelope::{ErrorInfo, Producer, Record, SchemaReq};
use glassrip_core::graph::{Selection, StageDecl, StageGraph};
use glassrip_core::jsonl;
use glassrip_core::manifest::RunDir;
use glassrip_core::runner::{
    stage_decl, ArtifactSpec, InputDecl, ItemContext, Runner, RunnerOptions, Stage, StageError,
    StageInputs, WorkItem,
};
use glassrip_meeting::artifacts::{
    CoordinateCheck, EdgeDirectionBatch, BOARD_STATE, BOARD_VALIDATE, CANVAS_CROP, EDGE_DIRECTION,
    KEYFRAMES, OCR,
};
use glassrip_meeting::consolidate::{BoardStateItem, ConsolidationParams, EdgeOrientation};
use glassrip_meeting::direction::DirectionBasis;
use glassrip_meeting::pixel_direction::PixelCheckParams;
use glassrip_meeting::stages::{BoardStateStage, EdgeDirectionStage};
use glassrip_meeting::vlm_direction::VlmCheckParams;
use glassrip_vision::backend::{
    BackendId, Durations, Placement, RawResponse, VisionBackend, VisionRequest,
};
use glassrip_vision::{BBox, VisionClient};
use schemars::JsonSchema;
use semver::Version;
use serde::Serialize;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Serialize, JsonSchema)]
struct NoParams {}

/// Emits fixed items for one upstream artifact.
struct Fixed {
    name: &'static str,
    schema: &'static str,
    items: Vec<(String, Value)>,
    params: NoParams,
}

impl Stage for Fixed {
    type Params = NoParams;
    type Work = Value;
    type Output = Value;
    fn name(&self) -> &'static str {
        self.name
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: self.schema,
            version: Version::new(1, 0, 0),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        Vec::new()
    }
    fn params(&self) -> &NoParams {
        &self.params
    }
    fn plan(&self, _inputs: &StageInputs) -> Result<Vec<WorkItem<Value>>, StageError> {
        Ok(self
            .items
            .iter()
            .map(|(id, v)| WorkItem {
                id: id.clone(),
                work: v.clone(),
            })
            .collect())
    }
    async fn process(&self, _ctx: &ItemContext, v: Value) -> Result<Value, ErrorInfo> {
        Ok(v)
    }
}

fn fixed(name: &'static str, schema: &'static str, items: Vec<(String, Value)>) -> Fixed {
    Fixed {
        name,
        schema,
        items,
        params: NoParams {},
    }
}

fn bbox(b: &BBox) -> Value {
    json!({"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2})
}

/// Answers every request with a fixed image-space side and counts calls.
struct MockVlm {
    answer: &'static str,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl VisionBackend for MockVlm {
    fn id(&self) -> BackendId {
        BackendId {
            backend: "mock".into(),
            model: "mock".into(),
            digest: Some("mock-digest".into()),
            server_version: None,
        }
    }
    async fn preflight(&self) -> glassrip_vision::Result<Placement> {
        Ok(Placement {
            model: "mock".into(),
            size_bytes: 1,
            size_vram_bytes: 1,
            fully_on_gpu: true,
            context_length: None,
            concurrency_hint: 1,
        })
    }
    async fn infer(
        &self,
        _request: VisionRequest,
        _cancel: CancellationToken,
    ) -> glassrip_vision::Result<RawResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let v = json!({"answer": self.answer});
        Ok(RawResponse {
            raw_text: v.to_string(),
            json: v,
            prompt_eval_count: None,
            eval_count: None,
            durations: Durations::default(),
            attempts: 1,
            repaired: false,
            done_reason: None,
        })
    }
}

/// One rendered board: Alpha -> Beta (labelled, head at Beta), Gamma -> Beta (head at
/// Beta), and Delta - Beta without heads (pixel-inconclusive).
struct Board {
    png_name: &'static str,
    board: Value,
}

fn render(dir: &Path, png_name: &'static str, texts: [&str; 4], scale_reading: f64) -> Board {
    let alpha = BBox::new(80.0, 120.0, 220.0, 190.0);
    let beta = BBox::new(480.0, 120.0, 620.0, 190.0);
    let gamma = BBox::new(480.0, 330.0, 620.0, 400.0);
    let delta = BBox::new(80.0, 330.0, 220.0, 400.0);
    let mut c = Canvas::new(720, 460);
    for b in [alpha, beta, gamma, delta] {
        c.node(b);
    }
    c.connector(&[(alpha.x2, 155.0), (beta.x1, 155.0)], false, true, false);
    c.connector(&[(550.0, gamma.y1), (550.0, beta.y2)], false, true, false);
    c.connector(
        &[
            (delta.x2, 365.0),
            (300.0, 365.0),
            (300.0, 260.0),
            (beta.x1 - 60.0, 260.0),
            (beta.x1 - 60.0, 180.0),
            (beta.x1, 180.0),
        ],
        false,
        false,
        false,
    );
    let label = c.label(350.0, 155.0, 40.0);
    c.img.save(dir.join(png_name)).unwrap();
    let s = |b: &BBox| {
        bbox(&BBox::new(
            b.x1 * scale_reading,
            b.y1 * scale_reading,
            b.x2 * scale_reading,
            b.y2 * scale_reading,
        ))
    };
    let mut board = json!({
        "nodes": [
            {"local_id": "n1", "text": texts[0], "bbox": s(&alpha), "conf": 0.9},
            {"local_id": "n2", "text": texts[1], "bbox": s(&beta), "conf": 0.9},
            {"local_id": "n3", "text": texts[2], "bbox": s(&gamma), "conf": 0.9},
            {"local_id": "n4", "text": texts[3], "bbox": s(&delta), "conf": 0.9},
        ],
        // The reader got the first edge backwards.
        "edges": [
            {"src": "n2", "dst": "n1", "label": "HTTP", "style": "solid", "conf": 0.8},
            {"src": "n3", "dst": "n2", "label": "", "style": "solid", "conf": 0.8},
            {"src": "n4", "dst": "n2", "label": "", "style": "solid", "conf": 0.8},
        ],
        "stickies": [],
        "owner_tags": [],
        "other_visible_text": [{"text": "HTTP", "bbox": s(&label)}],
        "confidence": 0.9,
        "chrome_rejected": [],
        "issues": [],
        "needs_reclassification": false,
    });
    if (scale_reading - 1.0).abs() > 1e-9 {
        board = json!({
            "keyframe_id": "placeholder",
            "canvas": {"width": 720.0 * scale_reading, "height": 460.0 * scale_reading},
            "board": board,
        });
    }
    Board { png_name, board }
}

struct Setup {
    _dir: tempfile::TempDir,
    runner: Runner,
    upstream: Vec<Fixed>,
}

/// `boards[i]` covers keyframes `kf{i}{j}`; a non-board keyframe separates boards.
fn setup(src: &Path, boards: &[(Board, usize)], wrap_ids: bool) -> Setup {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let run_root = root.join("run");
    let mut validate = Vec::new();
    let mut crops = Vec::new();
    let mut kfs = Vec::new();
    let mut ocr = Vec::new();
    let mut t = 0.0;
    for (bi, (b, n)) in boards.iter().enumerate() {
        std::fs::create_dir_all(&run_root).unwrap();
        for j in 0..*n {
            let id = format!("kf{bi}{j}");
            let mut item = b.board.clone();
            if wrap_ids && item.get("keyframe_id").is_some() {
                item["keyframe_id"] = json!(id);
            }
            validate.push((id.clone(), item));
            crops.push((id.clone(), json!({"crop_path": b.png_name})));
            kfs.push((
                id.clone(),
                json!({"keyframe_id": id, "t_start_s": t, "t_end_s": t + 30.0, "t_rep_s": t + 5.0,
                       "boundary": {"ink_change": 0.01}}),
            ));
            ocr.push((id.clone(), json!({"keyframe_id": id, "spans": []})));
            t += 30.0;
        }
        // A non-board keyframe after each board (a screen switch).
        let id = format!("kf{bi}x");
        kfs.push((
            id.clone(),
            json!({"keyframe_id": id, "t_start_s": t, "t_end_s": t + 30.0, "t_rep_s": t + 5.0}),
        ));
        t += 30.0;
    }
    let upstream = vec![
        fixed("board_validate", BOARD_VALIDATE, validate),
        fixed("canvas_crop", CANVAS_CROP, crops),
        fixed("keyframes", KEYFRAMES, kfs),
        fixed("ocr_harvest", OCR, ocr),
    ];
    let edge =
        EdgeDirectionStage::new(PixelCheckParams::default(), VlmCheckParams::default(), None);
    let state = BoardStateStage::new(ConsolidationParams::default());
    let graph = StageGraph::new(vec![
        StageDecl::new("board_validate", BOARD_VALIDATE, &[]),
        StageDecl::new("canvas_crop", CANVAS_CROP, &[]),
        StageDecl::new("keyframes", KEYFRAMES, &[]),
        StageDecl::new("ocr_harvest", OCR, &[]),
        stage_decl(&edge),
        stage_decl(&state),
    ])
    .unwrap();
    let run = RunDir::open(&run_root, "run", Producer::glassrip("0.1.0", None)).unwrap();
    for (b, _) in boards {
        std::fs::copy(src.join(b.png_name), run_root.join(b.png_name)).unwrap();
    }
    let runner = Runner::new(
        run,
        graph,
        &Selection::default(),
        Cache::in_workspace(&root),
        RunnerOptions::default(),
        CancellationToken::new(),
    )
    .unwrap();
    Setup {
        _dir: dir,
        runner,
        upstream,
    }
}

async fn run_all(
    s: &mut Setup,
    edge: &EdgeDirectionStage,
) -> (EdgeDirectionBatch, Vec<BoardStateItem>) {
    for f in &s.upstream {
        s.runner.run_stage(f).await.unwrap();
    }
    let rep = s.runner.run_stage(edge).await.unwrap();
    assert_eq!(rep.items_error, 0, "{rep:?}");
    s.runner
        .run_stage(&BoardStateStage::new(ConsolidationParams::default()))
        .await
        .unwrap();
    let batch = jsonl::read::<Record<EdgeDirectionBatch>>(
        &s.runner.run_dir().artifact_path(EDGE_DIRECTION),
        &SchemaReq::new(EDGE_DIRECTION, 1),
    )
    .unwrap()
    .items[0]
        .outcome
        .result
        .clone()
        .unwrap();
    let states = jsonl::read::<Record<BoardStateItem>>(
        &s.runner.run_dir().artifact_path(BOARD_STATE),
        &SchemaReq::new(BOARD_STATE, 1),
    )
    .unwrap()
    .items
    .into_iter()
    .filter_map(|r| r.outcome.result)
    .collect();
    (batch, states)
}

fn names(s: &BoardStateItem, e: &glassrip_meeting::consolidate::EdgeState) -> (String, String) {
    let n = |id: &str| {
        s.nodes
            .iter()
            .find(|n| n.id == id)
            .map(|n| n.text.clone())
            .unwrap_or_default()
    };
    (n(&e.src), n(&e.dst))
}

const TEXTS: [&str; 4] = ["Alpha Service", "Beta Store", "Gamma Worker", "Delta Cron"];

#[tokio::test]
async fn default_params_decide_directions_from_pixels() {
    let dir = tempfile::tempdir().unwrap();
    let b = render(dir.path(), "crop.png", TEXTS, 1.0);
    let mut s = setup(dir.path(), &[(b, 3)], false);
    let edge =
        EdgeDirectionStage::new(PixelCheckParams::default(), VlmCheckParams::default(), None);
    // The declarations match the meeting-mode graph in glassrip-core.
    let meeting = glassrip_core::graph::meeting_mode_stage_decls();
    for d in [
        stage_decl(&edge),
        stage_decl(&BoardStateStage::new(ConsolidationParams::default())),
    ] {
        assert!(meeting.contains(&d), "{d:?}");
    }
    let (batch, states) = run_all(&mut s, &edge).await;
    assert!(batch.vlm_fallback.is_empty() && !batch.vlm_available);
    assert_eq!(states.len(), 1);
    let st = &states[0];
    assert_eq!(st.nodes.len(), 4);
    let http = st.edges.iter().find(|e| e.label == "HTTP").unwrap();
    assert_eq!(http.direction, EdgeOrientation::Forward);
    assert_eq!(
        names(st, http),
        ("Alpha Service".into(), "Beta Store".into())
    );
    let gamma = st
        .edges
        .iter()
        .find(|e| e.a_text == "Gamma Worker" || e.b_text == "Gamma Worker")
        .unwrap();
    assert_eq!(
        names(st, gamma),
        ("Gamma Worker".into(), "Beta Store".into())
    );
    let delta = st
        .edges
        .iter()
        .find(|e| e.a_text == "Delta Cron" || e.b_text == "Delta Cron")
        .unwrap();
    assert_eq!(delta.direction, EdgeOrientation::Uncertain);
    assert_eq!(delta.direction_basis, DirectionBasis::None);
}

#[tokio::test]
async fn vlm_is_asked_only_for_pixel_inconclusive_edges() {
    let dir = tempfile::tempdir().unwrap();
    let b = render(dir.path(), "crop.png", TEXTS, 1.0);
    let mut s = setup(dir.path(), &[(b, 3)], false);
    let calls = Arc::new(AtomicUsize::new(0));
    // Delta is left of Beta: "right" means the head is at Beta's end.
    let client = VisionClient::new(
        Arc::new(MockVlm {
            answer: "right",
            calls: Arc::clone(&calls),
        }),
        2,
    )
    .unwrap();
    let edge = EdgeDirectionStage::new(
        PixelCheckParams::default(),
        VlmCheckParams::default(),
        Some(client),
    );
    let (batch, states) = run_all(&mut s, &edge).await;
    // Three edges, one inconclusive, one keyframe asked.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(batch.vlm_fallback.len(), 1);
    assert!(
        batch.vlm_fallback[0].ends.0.contains("Delta")
            || batch.vlm_fallback[0].ends.1.contains("Delta")
    );
    let st = &states[0];
    let delta = st
        .edges
        .iter()
        .find(|e| e.a_text == "Delta Cron" || e.b_text == "Delta Cron")
        .unwrap();
    assert_eq!(delta.direction_basis, DirectionBasis::Vlm);
    assert_eq!(names(st, delta), ("Delta Cron".into(), "Beta Store".into()));
    // Pixel-decided edges are untouched by the VLM.
    let http = st.edges.iter().find(|e| e.label == "HTTP").unwrap();
    assert_eq!(http.direction_basis, DirectionBasis::Pixel);
}

#[tokio::test]
async fn scaled_readings_are_mapped_into_crop_pixels() {
    let dir = tempfile::tempdir().unwrap();
    // The reading's canvas is half the crop's size; boxes are scaled accordingly.
    let b = render(dir.path(), "crop.png", TEXTS, 0.5);
    let mut s = setup(dir.path(), &[(b, 3)], true);
    let edge =
        EdgeDirectionStage::new(PixelCheckParams::default(), VlmCheckParams::default(), None);
    let (batch, states) = run_all(&mut s, &edge).await;
    assert!(batch
        .keyframes
        .iter()
        .all(|k| k.coordinates == CoordinateCheck::Scaled));
    assert!((batch.keyframes[0].board_to_image.x - 2.0).abs() < 1e-9);
    // Termini come back in reading coordinates (inside the half-size canvas).
    let e = &batch.keyframes[0].edges[0];
    assert!(
        e.pixel
            .dst_end
            .is_some_and(|t| t.x <= 360.0 && t.y <= 230.0),
        "{e:?}"
    );
    let http = states[0].edges.iter().find(|e| e.label == "HTTP").unwrap();
    assert_eq!(
        names(&states[0], http),
        ("Alpha Service".into(), "Beta Store".into())
    );
}

#[tokio::test]
async fn two_boards_become_two_items() {
    let dir = tempfile::tempdir().unwrap();
    let a = render(dir.path(), "a.png", TEXTS, 1.0);
    let b = render(
        dir.path(),
        "b.png",
        ["North Relay", "South Relay", "East Relay", "West Relay"],
        1.0,
    );
    // A different edge label, so the boards share no anchor text (spec 8.1).
    let b = Board {
        board: serde_json::from_str(&b.board.to_string().replace("HTTP", "AMQP")).unwrap(),
        ..b
    };
    let mut s = setup(dir.path(), &[(a, 3), (b, 3)], false);
    let edge =
        EdgeDirectionStage::new(PixelCheckParams::default(), VlmCheckParams::default(), None);
    let (_, states) = run_all(&mut s, &edge).await;
    assert_eq!(states.len(), 2);
    let ids: Vec<&str> = states.iter().map(|s| s.board_id.as_str()).collect();
    assert_eq!(ids, vec!["board-1", "board-2"]);
    assert!(states[0].nodes.iter().all(|n| !n.text.contains("Relay")));
    assert!(states[1].nodes.iter().all(|n| n.text.contains("Relay")));
}
