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

#[path = "common/synthetic_board.rs"]
mod synthetic_board;
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
    /// Report the text model as only half on the GPU.
    spill: bool,
    /// Every prompt sent: (purpose, user message).
    prompts: Mutex<Vec<(String, String)>>,
    ids: std::collections::BTreeMap<&'static str, String>,
}

#[async_trait]
impl TextBackend for Replay {
    async fn chat(&self, model: &str, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        self.prompts.lock().unwrap().push((
            req.purpose.clone(),
            req.messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default(),
        ));
        self.log
            .lock()
            .unwrap()
            .push(format!("chat {model} {}", req.purpose));
        let all: serde_json::Value = serde_json::from_str(RESPONSES).unwrap();
        let key = req.purpose.split_whitespace().next().unwrap_or("");
        Ok(ChatResponse {
            content: synthetic_board::resolve(&all[key].to_string(), &self.ids),
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
                size_vram: if self.spill && m.starts_with("text") {
                    60
                } else {
                    100
                },
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
    pub prompts: Vec<(String, String)>,
}

/// The plain notes path (quality options off), as the assertions below expect.
pub async fn run_pipeline(root: &std::path::Path) -> Outputs {
    run_pipeline_with(root, |p| {
        p.concise_items = false;
        p.board_candidates = false;
        p.cue_candidates = false;
        p.board_questions_in_reduce = false;
        p.precision_guard = false;
        p.owner_actions = false;
    })
    .await
}

pub async fn run_pipeline_with(
    root: &std::path::Path,
    tune: impl FnOnce(&mut NotesParams),
) -> Outputs {
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
    let (board, keyframes) = synthetic_board::synthetic_board();
    let ids = synthetic_board::symbolic_ids(&board);
    import::write_artifact(
        &run,
        schemas::KEYFRAMES,
        semver::Version::new(1, 0, 0),
        serde_json::json!({}),
        keyframes
            .into_iter()
            .map(|k| (k["keyframe_id"].as_str().unwrap().to_string(), k))
            .collect(),
    )
    .unwrap();
    import::write_empty(&run, schemas::OCR).unwrap();
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

    let backend = Arc::new(Replay {
        ids,
        ..Replay::default()
    });
    backend.resident.lock().unwrap().push("vision:7b".into());
    let mut notes_params = NotesParams {
        text_model: "text:27b".into(),
        vision_model: Some("vision:7b".into()),
        window_tokens: 180,
        window_overlap_lines: 1,
        ..Default::default()
    };
    tune(&mut notes_params);
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
    let prompts = backend.prompts.lock().unwrap().clone();
    Outputs {
        speakers,
        notes,
        log,
        prompts,
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

#[tokio::test]
async fn board_only_notes_when_there_is_no_audio() {
    let dir = tempfile::tempdir().unwrap();
    let run = RunDir::open(
        &dir.path().join("run"),
        "synthetic",
        Producer::glassrip("0.1.0", None),
    )
    .unwrap();
    let (board, keyframes) = synthetic_board::synthetic_board();
    import::write_artifact(
        &run,
        schemas::KEYFRAMES,
        semver::Version::new(1, 0, 0),
        serde_json::json!({}),
        keyframes
            .into_iter()
            .map(|k| (k["keyframe_id"].as_str().unwrap().to_string(), k))
            .collect(),
    )
    .unwrap();
    // the audio branch was skipped: empty transcript and speakers
    import::write_empty(&run, schemas::TRANSCRIPT).unwrap();
    import::write_empty(&run, schemas::SPEAKERS).unwrap();
    import::import_boards(&run, &[board]).unwrap();
    let graph = StageGraph::new(meeting_mode_stage_decls()).unwrap();
    let sel = Selection {
        from: Some("notes".into()),
        until: Some("notes".into()),
        ..Default::default()
    };
    let mut runner = Runner::new(
        run,
        graph,
        &sel,
        Cache::in_workspace(dir.path()),
        RunnerOptions::default(),
        CancellationToken::new(),
    )
    .unwrap();
    let backend = Arc::new(Replay::default());
    let stage = NotesStage::new(NotesParams::default(), backend.clone());
    let rep = runner.run_stage(&stage).await.unwrap();
    assert_eq!(rep.items_error, 0, "{rep:?}");
    let n = jsonl::read::<Record<MeetingNotes>>(
        &runner.run_dir().artifact_path(schemas::MEETING_NOTES),
        &SchemaReq::new(schemas::MEETING_NOTES, 1),
    )
    .unwrap()
    .items
    .into_iter()
    .find_map(|r| r.outcome.result)
    .unwrap();
    assert!(backend.log.lock().unwrap().is_empty(), "no model calls");
    assert_eq!(n.caveats[0].kind, "no_audio");
    assert!(n.decisions.is_empty() && n.action_items.is_empty() && n.transcript.is_empty());
    assert!(!n.timeline.is_empty());
    assert!(n.summary[0].text.contains("Kiosk App"), "{:?}", n.summary);
    assert!(
        n.summary
            .iter()
            .any(|s| s.text.contains("Mira Okafor on Design Kit")),
        "{:?}",
        n.summary
    );
    assert_eq!(n.open_questions.len(), 2);
    assert!(n
        .open_questions
        .iter()
        .all(|q| q.source == QuestionSource::Board && !q.evidence.event_ids.is_empty()));
}

#[tokio::test]
async fn a_spilled_text_model_is_refused_and_unloaded() {
    let dir = tempfile::tempdir().unwrap();
    let run = RunDir::open(
        &dir.path().join("run"),
        "synthetic",
        Producer::glassrip("0.1.0", None),
    )
    .unwrap();
    let (board, keyframes) = synthetic_board::synthetic_board();
    import::write_artifact(
        &run,
        schemas::KEYFRAMES,
        semver::Version::new(1, 0, 0),
        serde_json::json!({}),
        keyframes
            .into_iter()
            .map(|k| (k["keyframe_id"].as_str().unwrap().to_string(), k))
            .collect(),
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
    import::write_empty(&run, schemas::SPEAKERS).unwrap();
    import::import_boards(&run, &[board]).unwrap();
    let graph = StageGraph::new(meeting_mode_stage_decls()).unwrap();
    let sel = Selection {
        from: Some("notes".into()),
        until: Some("notes".into()),
        ..Default::default()
    };
    let mut runner = Runner::new(
        run,
        graph,
        &sel,
        Cache::in_workspace(dir.path()),
        RunnerOptions::default(),
        CancellationToken::new(),
    )
    .unwrap();
    let backend = Arc::new(Replay {
        spill: true,
        ..Replay::default()
    });
    let stage = NotesStage::new(
        NotesParams {
            text_model: "text:27b".into(),
            vision_model: None,
            ..Default::default()
        },
        backend.clone(),
    );
    assert!(runner.run_stage(&stage).await.is_err());
    let log = backend.log.lock().unwrap().clone();
    assert_eq!(log.first().map(String::as_str), Some("load text:27b"));
    assert_eq!(
        log.last().map(String::as_str),
        Some("unload text:27b"),
        "{log:?}"
    );
    assert!(
        !log.iter().any(|l| l.starts_with("chat")),
        "no calls on a spilled model"
    );
    assert!(backend.resident.lock().unwrap().is_empty());
}

#[tokio::test]
async fn quality_options_keep_the_pipeline_sound() {
    let dir = tempfile::tempdir().unwrap();
    // the defaults are the measured quality options
    let out = run_pipeline_with(dir.path(), |_| {}).await;
    let n = &out.notes;
    assert_eq!(n.report.status, NotesStatus::Ok);
    // both committed decisions survive the precision guard
    assert_eq!(n.decisions.len(), 2, "{:#?}", n.decisions);
    // Avery's owner tag on the Kiosk App becomes an action; Mira's and Rohan's
    // targets are already named by their own actions
    let owned: Vec<(&str, &str)> = n
        .action_items
        .iter()
        .filter(|a| a.task.starts_with("Own "))
        .map(|a| (a.owner.as_str(), a.task.as_str()))
        .collect();
    assert_eq!(owned, vec![("Avery Quinn", "Own Kiosk App")]);
    assert_eq!(n.action_items.len(), 6);
    // the map prompts carried the board facts and the cue lines, and the
    // reduce prompt the board's questions
    let map = out
        .prompts
        .iter()
        .find(|(p, _)| p.starts_with("map 1/"))
        .map(|(_, u)| u.as_str())
        .unwrap();
    assert!(map.contains("Board facts from owner tags"), "{map}");
    assert!(
        map.contains("Mira Okafor moved from Ledger Service to Design Kit"),
        "{map}"
    );
    assert!(
        map.contains("Lines with decision or question cues"),
        "{map}"
    );
    assert!(
        map.contains("seg_00005"),
        "the committing line is a cue: {map}"
    );
    let reduce = out
        .prompts
        .iter()
        .find(|(p, _)| p == "reduce")
        .map(|(_, u)| u.as_str())
        .unwrap();
    assert!(reduce.contains("already in the notes from the board's question stickies"));
    assert!(
        reduce.contains("Which widgets do we need for the kiosk?"),
        "{reduce}"
    );
}

// ---- empty inputs: no speech, no participant names, no board, no keyframes

fn transcript_artifact(items: Vec<TranscriptSegment>) -> TranscriptArtifact {
    TranscriptArtifact {
        schema: "glassrip.transcript".into(),
        schema_version: "1.0.0".into(),
        run_id: "synthetic".into(),
        producer: glassrip_audio::types::Producer::current(),
        inputs: vec![],
        params: serde_json::from_value(serde_json::json!({
            "asr_model": "synthetic", "vocabulary": [], "language": "en", "beam_size": 5,
            "vad_model": null, "diarization": null, "num_speakers": null, "timeline_offset_s": 0.0,
            "correction_max_p": 0.6, "correction_max_p_proper_noun": 0.85, "asr_backend": "cpu", "gap_fill": null
        }))
        .unwrap(),
        items,
    }
}

/// A run directory with the given transcript, an empty OCR artifact, and the
/// synthetic board and its keyframes (or none of either).
fn empty_input_run(
    root: &std::path::Path,
    segments: Option<Vec<TranscriptSegment>>,
    with_board: bool,
) -> (RunDir, BTreeMap<&'static str, String>) {
    let run = RunDir::open(
        &root.join("run"),
        "synthetic",
        Producer::glassrip("0.1.0", None),
    )
    .unwrap();
    let (board, keyframes) = synthetic_board::synthetic_board();
    let ids = synthetic_board::symbolic_ids(&board);
    let keyframes: Vec<serde_json::Value> = if with_board { keyframes } else { vec![] };
    import::write_artifact(
        &run,
        schemas::KEYFRAMES,
        semver::Version::new(1, 0, 0),
        serde_json::json!({}),
        keyframes
            .into_iter()
            .map(|k| (k["keyframe_id"].as_str().unwrap().to_string(), k))
            .collect(),
    )
    .unwrap();
    match segments {
        Some(s) => {
            import::import_transcript(&run, &transcript_artifact(s)).unwrap();
        }
        None => {
            import::write_empty(&run, schemas::TRANSCRIPT).unwrap();
        }
    }
    import::write_empty(&run, schemas::OCR).unwrap();
    let boards = if with_board { vec![board] } else { vec![] };
    import::import_boards(&run, &boards).unwrap();
    (run, ids)
}

fn speakers_and_notes_runner(run: RunDir, root: &std::path::Path) -> Runner {
    let sel = Selection {
        from: Some("name_speakers".into()),
        until: Some("notes".into()),
        ..Default::default()
    };
    Runner::new(
        run,
        StageGraph::new(meeting_mode_stage_decls()).unwrap(),
        &sel,
        Cache::in_workspace(root),
        RunnerOptions::default(),
        CancellationToken::new(),
    )
    .unwrap()
}

fn read_speakers(runner: &Runner) -> SpeakersDoc {
    let recs = jsonl::read::<Record<SpeakersRecord>>(
        &runner.run_dir().artifact_path(schemas::SPEAKERS),
        &SchemaReq::new(schemas::SPEAKERS, 1),
    )
    .unwrap()
    .items;
    SpeakersDoc::from_records(recs.into_iter().filter_map(|r| r.outcome.result))
}

fn read_notes(runner: &Runner) -> MeetingNotes {
    jsonl::read::<Record<MeetingNotes>>(
        &runner.run_dir().artifact_path(schemas::MEETING_NOTES),
        &SchemaReq::new(schemas::MEETING_NOTES, 1),
    )
    .unwrap()
    .items
    .into_iter()
    .find_map(|r| r.outcome.result)
    .unwrap()
}

/// No participants given: the stage has visual cues but no names to look for.
fn unnamed_speakers_stage() -> NameSpeakersStage {
    NameSpeakersStage::new(NameSpeakersParams::default())
        .with_visual(Arc::new(ScriptedFrames), Arc::new(FixedTiles))
}

#[tokio::test]
async fn no_speech_and_no_participants_run_through_speakers_and_notes() {
    let dir = tempfile::tempdir().unwrap();
    let (run, _) = empty_input_run(dir.path(), None, true);
    let mut runner = speakers_and_notes_runner(run, dir.path());
    let rep = runner.run_stage(&unnamed_speakers_stage()).await.unwrap();
    assert_eq!(rep.items_error, 0, "{rep:?}");
    let sp = read_speakers(&runner);
    assert!(sp.people.is_empty() && sp.labels.is_empty() && sp.segments.is_empty());
    let summary = sp.summary.expect("a summary record, even with no speech");
    assert!(
        summary.method.starts_with("none: no transcript segments"),
        "{summary:?}"
    );
    assert_eq!(summary.frames_decoded, 0);
    let why = summary.notes.unwrap_or_default();
    assert!(why.contains("no speech"), "{why}");
    assert!(why.contains("no participant names"), "{why}");

    let backend = Arc::new(Replay::default());
    let rep = runner
        .run_stage(&NotesStage::new(NotesParams::default(), backend.clone()))
        .await
        .unwrap();
    assert_eq!(rep.items_error, 0, "{rep:?}");
    let n = read_notes(&runner);
    assert!(backend.log.lock().unwrap().is_empty(), "no model calls");
    assert_eq!(n.caveats[0].kind, "no_audio");
    assert_eq!(
        n.report.status,
        NotesStatus::Ok,
        "the board still gives notes"
    );
    assert!(!n.timeline.is_empty() && !n.summary.is_empty());
}

#[tokio::test]
async fn speech_without_participant_names_is_left_unnamed_not_failed() {
    let dir = tempfile::tempdir().unwrap();
    let (run, ids) = empty_input_run(dir.path(), Some(transcript()), true);
    let mut runner = speakers_and_notes_runner(run, dir.path());
    let rep = runner.run_stage(&unnamed_speakers_stage()).await.unwrap();
    assert_eq!(rep.items_error, 0, "{rep:?}");
    let sp = read_speakers(&runner);
    assert!(sp.people.is_empty());
    assert_eq!(sp.labels.len(), 3);
    assert!(sp.labels.iter().all(|l| l.person_id.is_none()));
    assert_eq!(sp.segments.len(), transcript().len());
    assert!(sp
        .segments
        .values()
        .all(|s| s.person_id.is_none() && s.source == SpeakerSource::Unresolved));
    let summary = sp.summary.unwrap();
    assert!(
        summary.method.starts_with("none: no participant names"),
        "{summary:?}"
    );
    assert_eq!(
        summary.frames_decoded, 0,
        "no names to look for: no frames decoded"
    );
    assert!(summary
        .notes
        .unwrap_or_default()
        .contains("3 diarization labels left unnamed"));

    let backend = Arc::new(Replay {
        ids,
        ..Replay::default()
    });
    let params = NotesParams {
        text_model: "text:27b".into(),
        vision_model: None,
        window_tokens: 180,
        window_overlap_lines: 1,
        ..Default::default()
    };
    let rep = runner
        .run_stage(&NotesStage::new(params, backend.clone()))
        .await
        .unwrap();
    assert_eq!(rep.items_error, 0, "{rep:?}");
    let n = read_notes(&runner);
    assert!(
        n.caveats.iter().any(|c| c.kind == "no_participants"),
        "{:?}",
        n.caveats
    );
    assert!(
        n.transcript
            .iter()
            .all(|p| p.speaker.ends_with("(unresolved)")),
        "{:?}",
        n.transcript
    );
    // with no participant list, the names on the board's owner tags can own
    // action items (with everyone), but they are not listed as participants
    let (board, _) = synthetic_board::synthetic_board();
    let mut owners: Vec<String> = board
        .owner_assignments
        .iter()
        .map(|o| o.display_name.clone())
        .collect();
    owners.sort();
    owners.dedup();
    assert!(!owners.is_empty());
    assert!(n.people.is_empty(), "{:?}", n.people);
    assert!(!n.action_items.is_empty());
    assert!(
        n.action_items
            .iter()
            .all(|a| a.owner == "Everyone" || owners.contains(&a.owner)),
        "{:?}",
        n.action_items
    );
    let prompts = backend.prompts.lock().unwrap().clone();
    let map = prompts
        .iter()
        .find(|(p, _)| p.starts_with("map 1/"))
        .map(|(_, u)| u.as_str())
        .unwrap();
    assert!(map.starts_with("Participants: none known"), "{map}");
}

#[tokio::test]
async fn nothing_heard_or_read_gives_degraded_empty_notes() {
    let dir = tempfile::tempdir().unwrap();
    // one blank segment (the recognizer kept a segment with no words), no board,
    // no keyframes
    let blank = TranscriptSegment {
        segment_id: "seg_00000".into(),
        start_s: 1.0,
        end_s: 2.0,
        speaker_label: "SPEAKER_00".into(),
        speaker_conf: 0.0,
        text: " ".into(),
        text_raw: " ".into(),
        gap_fill_words: 0,
        unassigned_words: 0,
        words: vec![],
    };
    let (run, _) = empty_input_run(dir.path(), Some(vec![blank]), false);
    let mut runner = speakers_and_notes_runner(run, dir.path());
    let rep = runner.run_stage(&unnamed_speakers_stage()).await.unwrap();
    assert_eq!(rep.items_error, 0, "{rep:?}");
    let backend = Arc::new(Replay::default());
    let rep = runner
        .run_stage(&NotesStage::new(NotesParams::default(), backend.clone()))
        .await
        .unwrap();
    assert_eq!(rep.items_error, 0, "{rep:?}");
    let n = read_notes(&runner);
    assert!(
        backend.log.lock().unwrap().is_empty(),
        "a blank transcript is no speech: no model calls"
    );
    assert_eq!(n.report.status, NotesStatus::Degraded);
    let kinds: Vec<&str> = n.caveats.iter().map(|c| c.kind.as_str()).collect();
    assert_eq!(kinds, vec!["no_audio", "no_board"]);
    assert!(n.summary.is_empty() && n.timeline.is_empty() && n.open_questions.is_empty());
    assert!(n.decisions.is_empty() && n.action_items.is_empty() && n.transcript.is_empty());
}

/// Codex final round 3 MAJOR: the text model's digest keys the notes cache, so
/// other weights retagged under the same model name do not restore the old
/// notes, and a server serving other weights than the pinned digest is refused
/// instead of writing its output under the pinned key.
#[tokio::test]
async fn the_text_model_digest_keys_the_notes_cache() {
    let dir = tempfile::tempdir().unwrap();
    let (run, ids) = empty_input_run(dir.path(), Some(transcript()), true);
    let mut runner = speakers_and_notes_runner(run, dir.path());
    runner.run_stage(&unnamed_speakers_stage()).await.unwrap();
    let backend = Arc::new(Replay {
        ids,
        ..Replay::default()
    });
    let notes = |digest: &str| {
        NotesStage::new(NotesParams::default(), backend.clone())
            .with_text_digest(Some(digest.to_string()))
    };
    let rep = runner.run_stage(&notes("sha256:synthetic")).await.unwrap();
    assert_eq!(rep.items_error, 0, "{rep:?}");
    assert!(runner.cache_hit(&notes("sha256:synthetic")));
    assert!(
        !runner.cache_hit(&notes("sha256:retagged")),
        "another digest is another key"
    );
    // The server still serves sha256:synthetic: the retagged key is not filled.
    let calls = backend.log.lock().unwrap().len();
    match runner.run_stage(&notes("sha256:retagged")).await {
        Ok(rep) => {
            assert_ne!(
                rep.status,
                glassrip_core::manifest::StageStatus::Cached,
                "{rep:?}"
            );
            assert_eq!(rep.items_error, 1, "{rep:?}");
        }
        Err(e) => assert!(
            matches!(
                e,
                glassrip_core::runner::RunnerError::ErrorRateExceeded { .. }
            ),
            "{e}"
        ),
    }
    assert!(
        backend.log.lock().unwrap().len() > calls,
        "the server was contacted, not a cache entry restored"
    );
    assert!(!runner.cache_hit(&notes("sha256:retagged")));
}

/// A text server that answers like `Replay` but reports no digest.
struct NoDigest(Arc<Replay>);

#[async_trait]
impl TextBackend for NoDigest {
    async fn chat(&self, model: &str, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        self.0.chat(model, req).await
    }
    async fn load(&self, model: &str) -> Result<(), LlmError> {
        self.0.load(model).await
    }
    async fn unload(&self, model: &str) -> Result<(), LlmError> {
        self.0.unload(model).await
    }
    async fn loaded(&self) -> Result<Vec<LoadedModel>, LlmError> {
        self.0.loaded().await
    }
    async fn digest(&self, _model: &str) -> Result<Option<String>, LlmError> {
        Ok(None)
    }
}

/// Codex final round 3, second pass: a text model whose digest nobody can tell
/// never keys a cache entry (the notes are recomputed), a pinned digest the
/// server cannot confirm is refused, and notes keyed as board-only refuse a
/// transcript with speech.
#[tokio::test]
async fn notes_without_a_known_text_digest_are_not_cached() {
    let dir = tempfile::tempdir().unwrap();
    let (run, ids) = empty_input_run(dir.path(), Some(transcript()), true);
    let mut runner = speakers_and_notes_runner(run, dir.path());
    runner.run_stage(&unnamed_speakers_stage()).await.unwrap();
    let replay = Arc::new(Replay {
        ids,
        ..Replay::default()
    });
    let backend = Arc::new(NoDigest(replay));
    let unknown = NotesStage::new(NotesParams::default(), backend.clone()).with_text_digest(None);
    for _ in 0..2 {
        assert!(!runner.cache_hit(&unknown));
        let rep = runner.run_stage(&unknown).await.unwrap();
        assert_eq!(rep.items_error, 0, "{rep:?}");
        assert_eq!(rep.status, glassrip_core::manifest::StageStatus::Ok);
    }
    for (stage, why) in [
        (
            NotesStage::new(NotesParams::default(), backend.clone())
                .with_text_digest(Some("sha256:synthetic".into())),
            "an unconfirmed pinned digest is refused",
        ),
        (
            NotesStage::new(NotesParams::default(), backend.clone()).without_text_model(),
            "board-only keyed notes refuse speech",
        ),
    ] {
        let failed = match runner.run_stage(&stage).await {
            Ok(rep) => rep.items_error == 1,
            Err(e) => matches!(
                e,
                glassrip_core::runner::RunnerError::ErrorRateExceeded { .. }
            ),
        };
        assert!(failed, "{why}");
    }
}
