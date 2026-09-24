//! `edge_direction` and `board_state` through the stage runner on a rendered
//! synthetic board.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

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
    BOARD_STATE, BOARD_VALIDATE, CANVAS_CROP, EDGE_DIRECTION, KEYFRAMES,
};
use glassrip_meeting::consolidate::{BoardStateItem, ConsolidationParams};
use glassrip_meeting::direction::{DirectionPolicy, EdgeDirection};
use glassrip_meeting::pixel_direction::PixelCheckParams;
use glassrip_meeting::stages::{BoardStateStage, EdgeDirectionStage};
use glassrip_meeting::vlm_direction::VlmCheckParams;
use glassrip_vision::BBox;
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

fn bbox(b: &BBox) -> Value {
    json!({"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2})
}

#[tokio::test]
async fn stages_run_end_to_end_and_fix_a_reversed_reading() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();

    // Synthetic board: Alpha -> Beta (labelled), Gamma -> Beta (unlabelled).
    let alpha = BBox::new(80.0, 120.0, 220.0, 190.0);
    let beta = BBox::new(480.0, 120.0, 620.0, 190.0);
    let gamma = BBox::new(480.0, 330.0, 620.0, 400.0);
    let mut c = Canvas::new(720, 460);
    for b in [alpha, beta, gamma] {
        c.node(b);
    }
    c.connector(&[(alpha.x2, 155.0), (beta.x1, 155.0)], false, true, false);
    c.connector(&[(550.0, gamma.y1), (550.0, beta.y2)], false, true, false);
    let label = c.label(350.0, 155.0, 40.0);
    let png = root.join("crop.png");
    c.img.save(&png).unwrap();

    let board = json!({
        "nodes": [
            {"local_id": "n1", "text": "Alpha Service", "bbox": bbox(&alpha), "conf": 0.9},
            {"local_id": "n2", "text": "Beta Store", "bbox": bbox(&beta), "conf": 0.9},
            {"local_id": "n3", "text": "Gamma Worker", "bbox": bbox(&gamma), "conf": 0.9},
        ],
        // The reader got the first edge backwards.
        "edges": [
            {"src": "n2", "dst": "n1", "label": "HTTP", "style": "solid", "conf": 0.8},
            {"src": "n3", "dst": "n2", "label": "", "style": "solid", "conf": 0.8},
        ],
        "stickies": [],
        "owner_tags": [],
        "other_visible_text": [{"text": "HTTP", "bbox": bbox(&label)}],
        "confidence": 0.9,
        "chrome_rejected": [],
        "issues": [],
        "needs_reclassification": false,
    });
    let ids = ["kf01", "kf02", "kf03"];
    let validate = Fixed {
        name: "board_validate",
        schema: BOARD_VALIDATE,
        items: ids.iter().map(|i| (i.to_string(), board.clone())).collect(),
        params: NoParams {},
    };
    let crops = Fixed {
        name: "canvas_crop",
        schema: CANVAS_CROP,
        items: ids
            .iter()
            .map(|i| (i.to_string(), json!({"crop_path": "crop.png"})))
            .collect(),
        params: NoParams {},
    };
    let keyframes = Fixed {
        name: "keyframes",
        schema: KEYFRAMES,
        items: ids
            .iter()
            .enumerate()
            .map(|(k, i)| {
                let t = k as f64 * 30.0;
                (
                    i.to_string(),
                    json!({"keyframe_id": i, "t_start_s": t, "t_end_s": t + 30.0, "t_rep_s": t + 5.0,
                           "boundary": {"ink_change": 0.01}}),
                )
            })
            .collect(),
        params: NoParams {},
    };
    let edge =
        EdgeDirectionStage::new(PixelCheckParams::default(), VlmCheckParams::default(), None);
    let state = BoardStateStage::new(ConsolidationParams {
        direction_policy: DirectionPolicy::PixelOnly,
        ..ConsolidationParams::default()
    });
    let graph = StageGraph::new(vec![
        StageDecl::new("board_validate", BOARD_VALIDATE, &[]),
        StageDecl::new("canvas_crop", CANVAS_CROP, &[]),
        StageDecl::new("keyframes", KEYFRAMES, &[]),
        stage_decl(&edge),
        stage_decl(&state),
    ])
    .unwrap();
    // The declarations match the meeting-mode graph in glassrip-core.
    let meeting = glassrip_core::graph::meeting_mode_stage_decls();
    for d in [stage_decl(&edge), stage_decl(&state)] {
        assert!(meeting.contains(&d), "{d:?}");
    }
    let run = RunDir::open(&root.join("run"), "run", Producer::glassrip("0.1.0", None)).unwrap();
    let mut runner = Runner::new(
        run,
        graph,
        &Selection::default(),
        Cache::in_workspace(&root),
        RunnerOptions::default(),
        CancellationToken::new(),
    )
    .unwrap();
    // Relative crop paths resolve against the run root.
    std::fs::copy(&png, root.join("run").join("crop.png")).unwrap();
    for r in [
        runner.run_stage(&validate).await,
        runner.run_stage(&crops).await,
        runner.run_stage(&keyframes).await,
    ] {
        r.unwrap();
    }
    let rep = runner.run_stage(&edge).await.unwrap();
    assert_eq!(rep.items_error, 0, "{rep:?}");
    runner.run_stage(&state).await.unwrap();

    let out = jsonl::read::<Record<BoardStateItem>>(
        &runner.run_dir().artifact_path(BOARD_STATE),
        &SchemaReq::new(BOARD_STATE, 1),
    )
    .unwrap()
    .items;
    let s = out[0].outcome.result.clone().expect("board state");
    assert_eq!(s.nodes.len(), 3);
    let name = |id: &Option<String>| {
        s.nodes
            .iter()
            .find(|n| Some(&n.id) == id.as_ref())
            .map(|n| n.text.clone())
            .unwrap_or_default()
    };
    let http = s.edges.iter().find(|e| e.label == "HTTP").unwrap();
    assert_ne!(http.direction, EdgeDirection::Uncertain);
    assert_eq!(
        (name(&http.src), name(&http.dst)),
        ("Alpha Service".into(), "Beta Store".into())
    );
    let other = s.edges.iter().find(|e| e.label.is_empty()).unwrap();
    assert_eq!(
        (name(&other.src), name(&other.dst)),
        ("Gamma Worker".into(), "Beta Store".into())
    );
    assert!(runner.run_dir().artifact_path(EDGE_DIRECTION).exists());
}
