//! Runs `name_speakers`, `notes` and `render` on existing inputs.
//!
//! Inputs: a `glassrip.transcript` envelope from the audio crate, board state
//! JSON (one object or a list), optionally a core `glassrip.keyframes` and
//! `glassrip.ocr` artifact, and the source video for on-demand speaker cues
//! (needs the `ocr` feature and PP-OCRv5 models). Prints a JSON summary.
//!
//! ```sh
//! cargo run --release -p glassrip-render --features ocr-cuda --example meeting_notes_render -- \
//!     --transcript transcript.json --board board.json --video meeting.mp4 \
//!     --participants "First Last, Other Person" --run run/ --out out/ --stem meeting
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use glassrip_audio::types::TranscriptArtifact;
use glassrip_core::cache::Cache;
use glassrip_core::envelope::{Producer, Record, SchemaReq};
use glassrip_core::graph::{meeting_mode_stage_decls, Selection, StageGraph};
use glassrip_core::jsonl;
use glassrip_core::manifest::RunDir;
use glassrip_core::runner::{Runner, RunnerOptions};
use glassrip_notes::board::BoardStateItem;
use glassrip_notes::import;
use glassrip_notes::notes::llm::{OllamaText, OllamaTextConfig};
use glassrip_notes::notes::{MeetingNotes, NotesParams, NotesStage};
use glassrip_notes::schemas;
use glassrip_notes::speakers::{
    NameSpeakersParams, NameSpeakersStage, SpeakersDoc, SpeakersRecord,
};
use glassrip_render::markdown::MarkdownMeta;
use glassrip_render::stage::RENDER_SCHEMA;
use glassrip_render::{RenderParams, RenderResult, RenderStage};
use serde_json::json;
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    transcript: PathBuf,
    #[arg(long)]
    board: PathBuf,
    #[arg(long)]
    keyframes: Option<PathBuf>,
    #[arg(long)]
    ocr: Option<PathBuf>,
    #[arg(long)]
    video: Option<PathBuf>,
    #[arg(long, default_value = "")]
    participants: String,
    /// PP-OCRv5 model directory (det.onnx, rec.onnx, dict.txt).
    #[arg(long)]
    ocr_models: Option<PathBuf>,
    #[arg(long)]
    run: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "meeting")]
    stem: String,
    #[arg(long)]
    date: Option<String>,
    #[arg(long, default_value = "http://localhost:11434")]
    host: String,
    #[arg(long, default_value = "qwen3.6:27b")]
    text_model: String,
    #[arg(long, default_value = "qwen2.5vl:7b")]
    vision_model: String,
    #[arg(long, default_value_t = 16384)]
    num_ctx: u32,
    #[arg(long, default_value_t = 6000)]
    window_tokens: usize,
    /// Stop after this stage (name_speakers, notes, render).
    #[arg(long, default_value = "render")]
    until: String,
    /// Force these stages to rerun (comma separated).
    #[arg(long, default_value = "")]
    force: String,
}

/// Samples used GPU memory with nvidia-smi until `stop` is cancelled.
async fn sample_vram(peak: Arc<AtomicU64>, stop: CancellationToken) {
    while !stop.is_cancelled() {
        let out = tokio::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()
            .await;
        if let Ok(o) = out {
            if let Some(v) = String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .and_then(|l| l.trim().parse::<u64>().ok())
            {
                peak.fetch_max(v, Ordering::SeqCst);
            }
        }
        tokio::select! {
            _ = stop.cancelled() => break,
            _ = tokio::time::sleep(std::time::Duration::from_millis(1000)) => {}
        }
    }
}

#[tokio::main]
async fn main() {
    let a = Args::parse();
    let participants: Vec<String> = a
        .participants
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let run = RunDir::open(
        &a.run,
        "notes-render",
        Producer::glassrip(env!("CARGO_PKG_VERSION"), None),
    )
    .unwrap();

    let t: TranscriptArtifact =
        serde_json::from_slice(&std::fs::read(&a.transcript).unwrap()).unwrap();
    import::import_transcript(&run, &t).unwrap();
    let board_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&a.board).unwrap()).unwrap();
    let boards: Vec<BoardStateItem> = match board_json {
        serde_json::Value::Array(_) => serde_json::from_value(board_json).unwrap(),
        v => vec![serde_json::from_value(v).unwrap()],
    };
    import::import_boards(&run, &boards).unwrap();
    match &a.keyframes {
        Some(p) => import::copy_artifact(&run, schemas::KEYFRAMES, p)
            .map(|_| ())
            .unwrap(),
        None => import::write_empty(&run, schemas::KEYFRAMES)
            .map(|_| ())
            .unwrap(),
    }
    match &a.ocr {
        Some(p) => import::copy_artifact(&run, schemas::OCR, p)
            .map(|_| ())
            .unwrap(),
        None => import::write_empty(&run, schemas::OCR).map(|_| ()).unwrap(),
    }

    let graph = StageGraph::new(meeting_mode_stage_decls()).unwrap();
    let sel = Selection {
        from: Some("name_speakers".into()),
        until: Some(a.until.clone()),
        force: a
            .force
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
    };
    let cache_root = a
        .run
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| a.run.clone());
    let mut runner = Runner::new(
        run,
        graph,
        &sel,
        Cache::in_workspace(&cache_root),
        RunnerOptions::default(),
        CancellationToken::new(),
    )
    .unwrap();
    let mut summary = serde_json::Map::new();

    // name_speakers
    let params = NameSpeakersParams {
        participants,
        video: a.video.clone(),
        ..Default::default()
    };
    #[cfg_attr(not(feature = "ocr"), allow(unused_mut))]
    let mut stage = NameSpeakersStage::new(params);
    #[cfg(feature = "ocr")]
    if let (Some(video), Some(models)) = (&a.video, &a.ocr_models) {
        use glassrip_notes::speakers::frames::FfmpegFrameSource;
        use glassrip_notes::speakers::ocr::{OcrParams, PpOcrReader};
        let frames = FfmpegFrameSource::open(video, 1920, true).await.unwrap();
        let reader = PpOcrReader::new(models, OcrParams::default()).unwrap();
        stage = stage.with_visual(Arc::new(frames), Arc::new(reader));
    }
    let t0 = Instant::now();
    let rep = runner.run_stage(&stage).await.unwrap();
    summary.insert("name_speakers".into(), json!({"wall_s": t0.elapsed().as_secs_f64(), "status": format!("{:?}", rep.status), "items_error": rep.items_error}));
    let dir = runner.run_dir();
    let recs = jsonl::read::<Record<SpeakersRecord>>(
        &dir.artifact_path(schemas::SPEAKERS),
        &SchemaReq::new(schemas::SPEAKERS, 1),
    )
    .unwrap()
    .items;
    let doc = SpeakersDoc::from_records(recs.into_iter().filter_map(|r| r.outcome.result));
    summary.insert(
        "labels".into(),
        json!(doc.labels.iter().map(|l| json!({"label": l.label, "person": l.person_id, "status": l.status, "confidence": l.confidence, "votes": l.votes, "talk_time_s": l.talk_time_s})).collect::<Vec<_>>()),
    );
    let relabeled: Vec<_> = doc
        .segments
        .values()
        .filter(|s| {
            matches!(
                s.source,
                glassrip_notes::speakers::SpeakerSource::VisualRelabel
                    | glassrip_notes::speakers::SpeakerSource::EvidenceRelabel
            ) || !s.spans.is_empty()
        })
        .map(|s| json!({"segment": s.segment_id, "t": s.start_s, "label": s.label, "person": s.person_id, "source": s.source, "reason": s.reason, "spans": s.spans.len()}))
        .collect();
    summary.insert("relabeled".into(), json!(relabeled));
    summary.insert("speakers_summary".into(), json!(doc.summary));

    if a.until != "name_speakers" {
        let backend = Arc::new(
            OllamaText::new(OllamaTextConfig {
                base_url: a.host.clone(),
                num_ctx: a.num_ctx,
                ..Default::default()
            })
            .unwrap(),
        );
        let params = NotesParams {
            text_model: a.text_model.clone(),
            vision_model: Some(a.vision_model.clone()),
            ollama: OllamaTextConfig {
                base_url: a.host.clone(),
                num_ctx: a.num_ctx,
                ..Default::default()
            },
            window_tokens: a.window_tokens,
            ..Default::default()
        };
        let peak = Arc::new(AtomicU64::new(0));
        let stop = CancellationToken::new();
        let sampler = tokio::spawn(sample_vram(peak.clone(), stop.clone()));
        let t0 = Instant::now();
        let rep = runner
            .run_stage(&NotesStage::new(params, backend))
            .await
            .unwrap();
        stop.cancel();
        let _ = sampler.await;
        summary.insert(
            "notes".into(),
            json!({"wall_s": t0.elapsed().as_secs_f64(), "status": format!("{:?}", rep.status), "items_error": rep.items_error, "peak_vram_mib": peak.load(Ordering::SeqCst)}),
        );
        let dir = runner.run_dir();
        let notes = jsonl::read::<Record<MeetingNotes>>(
            &dir.artifact_path(schemas::MEETING_NOTES),
            &SchemaReq::new(schemas::MEETING_NOTES, 1),
        )
        .unwrap()
        .items
        .into_iter()
        .find_map(|r| {
            if r.outcome.result.is_none() {
                eprintln!("notes error: {:?}", r.outcome.error);
            }
            r.outcome.result
        });
        if let Some(n) = &notes {
            summary.insert(
                "notes_report".into(),
                json!({"status": n.report.status, "decisions": n.decisions.len(), "action_items": n.action_items.len(), "open_questions": n.open_questions.len(), "timeline": n.timeline.len(), "summary": n.summary.len(), "dropped": n.report.dropped, "repaired": n.report.items_repaired, "calls": n.report.calls, "placement": n.report.placement}),
            );
        }
    }
    if a.until == "render" {
        let t0 = Instant::now();
        let stage = RenderStage::new(RenderParams {
            out_dir: a.out.clone(),
            stem: a.stem.clone(),
            meta: MarkdownMeta {
                date: a.date.clone(),
                source: None,
            },
            ..RenderParams::default()
        });
        let rep = runner.run_stage(&stage).await.unwrap();
        let dir = runner.run_dir();
        let r = jsonl::read::<Record<RenderResult>>(
            &dir.artifact_path(RENDER_SCHEMA),
            &SchemaReq::new(RENDER_SCHEMA, 1),
        )
        .unwrap()
        .items
        .into_iter()
        .find_map(|r| r.outcome.result);
        summary.insert(
            "render".into(),
            json!({"wall_s": t0.elapsed().as_secs_f64(), "status": format!("{:?}", rep.status), "ok": r.as_ref().map(|x| x.ok), "markdown": r.as_ref().map(|x| &x.markdown), "svg": r.as_ref().map(|x| &x.svg), "files": r.as_ref().map(|x| &x.files)}),
        );
    }
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
}
