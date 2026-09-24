//! Runs only the `notes` stage on existing artifacts, for notes quality work.
//!
//! Copies `glassrip.{transcript,board_state,speakers,keyframes}.jsonl` from
//! `--artifacts` into a fresh run, merges `--params` (a JSON object of
//! `NotesParams` fields) over the defaults, runs notes against Ollama, and
//! prints the notes report as JSON.
//!
//! ```sh
//! cargo run --release -p glassrip-notes --example notes_live -- \
//!     --artifacts run/artifacts --params variant.json --run out/run
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;
use std::sync::Arc;

use glassrip_core::cache::Cache;
use glassrip_core::envelope::{Producer, Record, SchemaReq};
use glassrip_core::graph::{meeting_mode_stage_decls, Selection, StageGraph};
use glassrip_core::jsonl;
use glassrip_core::manifest::RunDir;
use glassrip_core::runner::{Runner, RunnerOptions};
use glassrip_notes::import;
use glassrip_notes::notes::llm::OllamaText;
use glassrip_notes::notes::{MeetingNotes, NotesParams, NotesStage};
use glassrip_notes::schemas;
use tokio_util::sync::CancellationToken;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main]
async fn main() {
    let artifacts = PathBuf::from(arg("--artifacts").expect("--artifacts"));
    let run_path = PathBuf::from(arg("--run").expect("--run"));
    let host = arg("--host").unwrap_or_else(|| "http://localhost:11434".into());
    let mut params = serde_json::to_value(NotesParams::default()).unwrap();
    if let Some(p) = arg("--params") {
        let extra: serde_json::Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for (k, v) in extra.as_object().expect("params object") {
            params[k] = v.clone();
        }
    }
    let mut params: NotesParams = serde_json::from_value(params).unwrap();
    params.ollama.base_url = host.clone();
    let run = RunDir::open(&run_path, "notes-live", Producer::glassrip("0.1.0", None)).unwrap();
    for schema in [
        schemas::TRANSCRIPT,
        schemas::BOARD_STATE,
        schemas::SPEAKERS,
        schemas::KEYFRAMES,
    ] {
        import::copy_artifact(&run, schema, &artifacts.join(format!("{schema}.jsonl"))).unwrap();
    }
    let graph = StageGraph::new(meeting_mode_stage_decls()).unwrap();
    let cache_root = run_path.parent().unwrap().to_path_buf();
    let mut runner = Runner::new(
        run,
        graph,
        &Selection::default(),
        Cache::in_workspace(&cache_root),
        RunnerOptions::default(),
        CancellationToken::new(),
    )
    .unwrap();
    let backend = Arc::new(OllamaText::new(params.ollama.clone()).unwrap());
    let stage = NotesStage::new(params, backend);
    let rep = runner.run_stage(&stage).await;
    let path = runner.run_dir().artifact_path(schemas::MEETING_NOTES);
    let notes =
        jsonl::read::<Record<MeetingNotes>>(&path, &SchemaReq::new(schemas::MEETING_NOTES, 1))
            .ok()
            .and_then(|r| r.items.into_iter().find_map(|x| x.outcome.result));
    let out = serde_json::json!({
        "stage": rep.map(|r| format!("{:?}", r.status)).map_err(|e| e.to_string()),
        "report": notes.as_ref().map(|n| &n.report),
        "counts": notes.as_ref().map(|n| [n.decisions.len(), n.action_items.len(), n.open_questions.len()]),
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
