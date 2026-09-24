//! Synthetic end-to-end run of `name_speakers` and `notes` on the core runner,
//! with a scripted frame source, tile reader and text model (no GPU, no video).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use glassrip_audio::recluster::Source;
use glassrip_audio::types::{TranscriptArtifact, TranscriptSegment, TranscriptWord};
use glassrip_core::cache::Cache;
use glassrip_core::envelope::{Producer, Record, SchemaReq};
use glassrip_core::graph::{meeting_mode_stage_decls, Selection, StageGraph};
use glassrip_core::jsonl;
use glassrip_core::manifest::RunDir;
use glassrip_core::runner::{Runner, RunnerOptions};
use glassrip_notes::board::BoardState;
use glassrip_notes::import;
use glassrip_notes::notes::llm::{ChatRequest, ChatResponse, LlmError, LoadedModel, TextBackend};
use glassrip_notes::notes::{MeetingNotes, NotesParams, NotesStage, NotesStatus, QuestionSource};
use glassrip_notes::schemas;
use glassrip_notes::speakers::cues::{draw_tile, ScreenText, TileReader};
use glassrip_notes::speakers::frames::FrameSource;
use glassrip_notes::speakers::{
    NameSpeakersParams, NameSpeakersStage, SpeakerSource, SpeakersDoc, SpeakersRecord,
};
use image::RgbImage;
use tokio_util::sync::CancellationToken;

pub const BOARD: &str = include_str!("fixtures/synthetic_board.json");
const RESPONSES: &str = include_str!("fixtures/synthetic_llm_responses.json");

fn segment(id: &str, label: &str, start: f64, text: &str, src: Source) -> TranscriptSegment {
    let words: Vec<TranscriptWord> = text
        .split_whitespace()
        .enumerate()
        .map(|(i, w)| TranscriptWord {
            w: w.to_string(),
            w_raw: None,
            start_s: start + 0.4 * i as f64,
            end_s: start + 0.4 * (i + 1) as f64,
            p: 0.9,
            speaker_label: label.into(),
            assign_conf: if src == Source::GapFill { 0.45 } else { 0.9 },
            source: src,
            gap_sim: None,
            gap_margin: None,
        })
        .collect();
    TranscriptSegment {
        segment_id: id.into(),
        start_s: start,
        end_s: words.last().map_or(start, |w| w.end_s),
        speaker_label: label.into(),
        speaker_conf: if id == "seg_00002" { 0.35 } else { 0.85 },
        text: text.into(),
        text_raw: text.into(),
        gap_fill_words: if src == Source::GapFill {
            words.len()
        } else {
            0
        },
        unassigned_words: 0,
        words,
    }
}

/// A fictional planning call. Avery presents; Rohan has a visible tile; Mira is
/// the local participant (no tile). The diarizer put Avery's greeting in Rohan's
/// label and Mira's reply in Avery's.
fn transcript() -> Vec<TranscriptSegment> {
    use Source::{Diarizer as D, GapFill as G};
    vec![
        segment("seg_00000", "L0", 0.0, "Welcome everyone, this is the kiosk relay planning session for the pilot.", D),
        segment("seg_00001", "L1", 6.0, "Hey, Mira, glad you could join us.", D),
        segment("seg_00002", "L0", 9.2, "Hey.", D),
        segment("seg_00003", "L0", 10.0, "So the relay pulls entries from the ledger service over REST and the kiosk app talks GraphQL to the relay.", D),
        segment("seg_00004", "L1", 20.0, "Do we have to change anything in the badge flow, or can we leave it alone?", D),
        segment("seg_00005", "L0", 26.0, "Let's leave the badge flow alone for the pilot and focus on the relay side.", D),
        segment("seg_00006", "L2", 33.0, "I can take the ledger side and map entries to kit widgets.", D),
        segment("seg_00007", "L1", 40.0, "I'll design the relay storage schema this week.", D),
        segment("seg_00008", "L0", 47.0, "Rohan, can you also pair with Mira on the relay tests?", D),
        segment("seg_00009", "L1", 52.0, "Yes, I will pair with Mira on the relay tests.", D),
        segment("seg_00010", "L0", 58.0, "Which widgets do we need for the kiosk? We still have not settled that.", D),
        segment("seg_00011", "L0", 66.0, "and everyone should push work to a branch early even if it is rough", G),
        segment("seg_00012", "L2", 80.0, "Sounds good, I'll see you all later.", D),
    ]
}

fn avery_lit(t: f64) -> bool {
    [
        (0.0, 8.9),
        (10.0, 18.4),
        (26.0, 32.4),
        (47.0, 51.4),
        (58.0, 63.2),
        (66.0, 71.6),
    ]
    .iter()
    .any(|(a, b)| t >= *a && t <= *b)
}

fn rohan_lit(t: f64) -> bool {
    [(20.0, 25.9), (40.0, 43.2), (52.0, 56.0)]
        .iter()
        .any(|(a, b)| t >= *a && t <= *b)
}

struct ScriptedFrames;

#[async_trait]
impl FrameSource for ScriptedFrames {
    async fn frame_at(&self, t: f64) -> Result<RgbImage, String> {
        let mut img = RgbImage::from_pixel(400, 200, image::Rgb([34, 35, 38]));
        draw_tile(&mut img, [10, 10, 190, 120], [70, 90, 150], avery_lit(t));
        draw_tile(&mut img, [210, 10, 390, 120], [90, 120, 80], rohan_lit(t));
        Ok(img)
    }
    fn argv(&self, t: f64) -> Vec<String> {
        vec!["scripted".into(), format!("{t:.3}")]
    }
}

struct FixedTiles;

impl TileReader for FixedTiles {
    fn read(&self, _img: &RgbImage) -> Result<Vec<ScreenText>, String> {
        let st = |text: &str, bbox| ScreenText {
            text: text.into(),
            bbox,
            confidence: 0.95,
        };
        Ok(vec![
            st("Avery Quinn", [20, 92, 120, 108]),
            st("Rohan Dasgu...", [220, 92, 320, 108]),
            st("Avery Quinn (Presenting)", [0, 150, 200, 170]),
        ])
    }
    fn describe(&self) -> String {
        "fixed tiles".into()
    }
}

/// Replays recorded responses by call purpose and logs the phase C sequence.
#[derive(Default)]
struct Replay {
    log: Mutex<Vec<String>>,
    resident: Mutex<Vec<String>>,
}

#[async_trait]
impl TextBackend for Replay {
    async fn chat(&self, model: &str, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        self.log
            .lock()
            .unwrap()
            .push(format!("chat {model} {}", req.purpose));
        let all: serde_json::Value = serde_json::from_str(RESPONSES).unwrap();
        let key = req.purpose.split_whitespace().next().unwrap_or("");
        Ok(ChatResponse {
            content: all[key].to_string(),
            eval_count: Some(100),
            ..Default::default()
        })
    }
    async fn load(&self, model: &str) -> Result<(), LlmError> {
        self.log.lock().unwrap().push(format!("load {model}"));
        self.resident.lock().unwrap().push(model.into());
        Ok(())
    }
    async fn unload(&self, model: &str) -> Result<(), LlmError> {
        self.log.lock().unwrap().push(format!("unload {model}"));
        self.resident.lock().unwrap().retain(|m| m != model);
        Ok(())
    }
    async fn loaded(&self) -> Result<Vec<LoadedModel>, LlmError> {
        Ok(self
            .resident
            .lock()
            .unwrap()
            .iter()
            .map(|m| LoadedModel {
                name: m.clone(),
                size: 100,
                size_vram: 100,
                context_length: Some(16384),
            })
            .collect())
    }
    async fn digest(&self, _model: &str) -> Result<Option<String>, LlmError> {
        Ok(Some("sha256:synthetic".into()))
    }
}

pub struct Outputs {
    pub speakers: SpeakersDoc,
    pub notes: MeetingNotes,
    pub log: Vec<String>,
}

pub async fn run_pipeline(root: &std::path::Path) -> Outputs {
    let run = RunDir::open(
        &root.join("run"),
        "synthetic",
        Producer::glassrip("0.1.0", None),
    )
    .unwrap();
    let t = TranscriptArtifact {
        schema: "glassrip.transcript".into(),
        schema_version: "1.0.0".into(),
        run_id: "synthetic".into(),
        producer: glassrip_audio::types::Producer::current(),
        inputs: vec![],
        params: serde_json::from_value(serde_json::json!({
            "asr_model": "synthetic", "vocabulary": [], "language": "en", "beam_size": 5,
            "vad_model": null, "diarization": null, "num_speakers": 3, "timeline_offset_s": 0.0,
            "correction_max_p": 0.6, "correction_max_p_proper_noun": 0.85, "asr_backend": "cpu", "gap_fill": null
        }))
        .unwrap(),
        items: transcript(),
    };
    import::import_transcript(&run, &t).unwrap();
    import::write_empty(&run, schemas::KEYFRAMES).unwrap();
    import::write_empty(&run, schemas::OCR).unwrap();
    let board: BoardState = serde_json::from_str(BOARD).unwrap();
    import::import_boards(&run, &[board]).unwrap();

    let graph = StageGraph::new(meeting_mode_stage_decls()).unwrap();
    let sel = Selection {
        from: Some("name_speakers".into()),
        until: Some("notes".into()),
        ..Default::default()
    };
    let mut runner = Runner::new(
        run,
        graph,
        &sel,
        Cache::in_workspace(root),
        RunnerOptions::default(),
        CancellationToken::new(),
    )
    .unwrap();

    let params = NameSpeakersParams {
        participants: vec![
            "Avery Quinn".into(),
            "Rohan Dasgupta".into(),
            "Mira Okafor".into(),
        ],
        ..Default::default()
    };
    let speakers_stage =
        NameSpeakersStage::new(params).with_visual(Arc::new(ScriptedFrames), Arc::new(FixedTiles));
    let rep = runner.run_stage(&speakers_stage).await.unwrap();
    assert_eq!(rep.items_error, 0);

    let backend = Arc::new(Replay::default());
    backend.resident.lock().unwrap().push("vision:7b".into());
    let notes_params = NotesParams {
        text_model: "text:27b".into(),
        vision_model: Some("vision:7b".into()),
        window_tokens: 180,
        window_overlap_lines: 1,
        ..Default::default()
    };
    let notes_stage = NotesStage::new(notes_params, backend.clone());
    let rep = runner.run_stage(&notes_stage).await.unwrap();
    assert_eq!(rep.items_error, 0, "{rep:?}");

    let dir = runner.run_dir();
    let recs = jsonl::read::<Record<SpeakersRecord>>(
        &dir.artifact_path(schemas::SPEAKERS),
        &SchemaReq::new(schemas::SPEAKERS, 1),
    )
    .unwrap()
    .items;
    let speakers = SpeakersDoc::from_records(recs.into_iter().filter_map(|r| r.outcome.result));
    let notes = jsonl::read::<Record<MeetingNotes>>(
        &dir.artifact_path(schemas::MEETING_NOTES),
        &SchemaReq::new(schemas::MEETING_NOTES, 1),
    )
    .unwrap()
    .items
    .into_iter()
    .find_map(|r| r.outcome.result)
    .unwrap();
    let log = backend.log.lock().unwrap().clone();
    Outputs {
        speakers,
        notes,
        log,
    }
}

#[tokio::test]
async fn speakers_and_notes_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_pipeline(dir.path()).await;
    // refresh the render crate's synthetic fixture on request
    if let Ok(path) = std::env::var("GLASSRIP_DUMP_SYNTHETIC_NOTES") {
        std::fs::write(path, serde_json::to_string_pretty(&out.notes).unwrap()).unwrap();
    }

    // speaker naming
    let map: BTreeMap<&str, Option<&str>> = out
        .speakers
        .labels
        .iter()
        .map(|l| (l.label.as_str(), l.person_id.as_deref()))
        .collect();
    assert_eq!(map["L0"], Some("avery-quinn"));
    assert_eq!(map["L1"], Some("rohan-dasgupta"));
    assert_eq!(map["L2"], Some("mira-okafor"));
    let greeting = &out.speakers.segments["seg_00001"];
    assert_eq!(
        (greeting.person_id.as_deref(), greeting.source),
        (Some("avery-quinn"), SpeakerSource::VisualRelabel)
    );
    let reply = &out.speakers.segments["seg_00002"];
    assert_eq!(reply.person_id.as_deref(), Some("mira-okafor"));
    let summary = out.speakers.summary.as_ref().unwrap();
    assert_eq!(summary.presenter.as_deref(), Some("avery-quinn"));
    assert_eq!(summary.gap_fill.runs_total, 1);
    assert_eq!(summary.gap_fill.runs_agree, 1);

    // phase C order: vision model unloaded before the text model loads
    assert_eq!(out.log[0], "unload vision:7b");
    assert_eq!(out.log[1], "load text:27b");
    assert!(out
        .log
        .iter()
        .any(|l| l.starts_with("chat text:27b map 2/")));
    assert!(out.log.iter().any(|l| l == "chat text:27b reduce"));
    assert!(out.log.iter().any(|l| l == "chat text:27b repair"));
    assert_eq!(out.log.last().map(String::as_str), Some("unload text:27b"));

    // validation outcomes
    let n = &out.notes;
    assert_eq!(n.report.status, NotesStatus::Ok);
    assert_eq!(n.decisions.len(), 2);
    assert!(!n.decisions[0].text.contains('\u{2014}'));
    assert_eq!(n.action_items.len(), 5);
    assert!(n.action_items.iter().any(|a| a.owner == "Everyone"));
    assert!(n
        .action_items
        .iter()
        .any(|a| a.task.starts_with("Scope the relay side")));
    assert_eq!(n.report.items_repaired, 1);
    let dropped: Vec<&str> = n.report.dropped.iter().map(|d| d.text.as_str()).collect();
    assert_eq!(dropped.len(), 2, "{dropped:?}");
    assert!(dropped.iter().any(|d| d.contains("See you all later")));
    assert!(dropped.iter().any(|d| d.contains("every lobby")));
    assert_eq!(n.open_questions.len(), 2);
    assert!(n
        .open_questions
        .iter()
        .all(|q| q.source == QuestionSource::BoardAndTranscript));
    assert!(n
        .transcript
        .iter()
        .any(|p| p.speaker == "Mira Okafor" && p.text == "Hey."));
    assert!(n.caveats.iter().any(|c| c.kind == "gap_fill"));
}
