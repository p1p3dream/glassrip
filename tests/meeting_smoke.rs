//! End-to-end smoke test of `glassrip meeting` on a synthetic video.
//!
//! ffmpeg's lavfi sources draw a tiny fictional meeting: two whiteboard slides
//! (outlined boxes, a connector, a sticky, an owner tag) and a chat screen, with a
//! sine tone as audio. Every model is scripted (OCR, vision, ASR, diarization,
//! text), so the run is offline and deterministic; everything else is the real
//! pipeline on the core runner: media stages, GPU phases A/B/C, board state,
//! speaker naming, notes, and render. Names and texts are fictional.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use base64::Engine;
use glassrip::meeting::{self, run_meeting, Backends, MeetingOptions, VisionBackends};
use glassrip_audio::asr::{AsrOutput, AsrSegment};
use glassrip_audio::diarize::Diarization;
use glassrip_audio::recluster::Turn;
use glassrip_audio::stages::{AsrEngine, DiarizeEngine};
use glassrip_audio::words::AsrWord;
use glassrip_core::config::Config;
use glassrip_core::envelope::{Record, SchemaReq};
use glassrip_core::graph::meeting_mode_stage_decls;
use glassrip_core::manifest::{RunManifest, StageStatus};
use glassrip_notes::notes::llm::{ChatRequest, ChatResponse, LlmError, LoadedModel, TextBackend};
use glassrip_notes::notes::MeetingNotes;
use glassrip_ocr::{OcrError, PixelBox, RecognizedSpan, TextRecognizer};
use glassrip_vision::{
    BackendId, Durations, Placement, RawResponse, VisionBackend, VisionError, VisionRequest,
};
use glassrip_vision_stages::placement::{PlacementProbe, StaticProbe};
use glassrip_vision_stages::raw_store::{RawStore, RecordingBackend};
use image::RgbImage;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

const W: u32 = 1280;
const H: u32 = 720;
const VISION_MODEL: &str = "scripted-vl";
const TEXT_MODEL: &str = "scripted-text";

/// Boxes drawn on the slides, frame pixels `(x1, y1, x2, y2)`.
const LEDGER: [u32; 4] = [160, 200, 460, 320];
const ORBIT: [u32; 4] = [760, 200, 1060, 320];
const PARCEL: [u32; 4] = [560, 450, 860, 570];
const STICKY: [u32; 4] = [180, 450, 380, 600];
const TAG: [u32; 4] = [470, 330, 560, 370];

fn ffmpeg_ok() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// ffmpeg is required (CI installs it); `GLASSRIP_SKIP_FFMPEG_TESTS=1` opts out.
fn require_ffmpeg() -> bool {
    if ffmpeg_ok() {
        return true;
    }
    let opted_out = std::env::var("GLASSRIP_SKIP_FFMPEG_TESTS").is_ok_and(|v| v == "1");
    assert!(
        opted_out,
        "ffmpeg is not installed; install it or set GLASSRIP_SKIP_FFMPEG_TESTS=1"
    );
    eprintln!("SKIPPED: ffmpeg is not installed (GLASSRIP_SKIP_FFMPEG_TESTS=1)");
    false
}

fn font() -> Option<&'static str> {
    [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    .into_iter()
    .find(|p| Path::new(p).is_file())
}

fn outline(b: [u32; 4]) -> String {
    format!(
        "drawbox=x={}:y={}:w={}:h={}:color=0x222222:t=5",
        b[0],
        b[1],
        b[2] - b[0],
        b[3] - b[1]
    )
}

fn filled(b: [u32; 4], color: &str) -> String {
    format!(
        "drawbox=x={}:y={}:w={}:h={}:color={color}:t=fill",
        b[0],
        b[1],
        b[2] - b[0],
        b[3] - b[1]
    )
}

fn text(t: &str, x: u32, y: u32) -> Option<String> {
    font().map(|f| {
        format!("drawtext=fontfile={f}:text='{t}':x={x}:y={y}:fontsize=30:fontcolor=0x111111")
    })
}

/// A 440 Hz tone as the audio track.
const TONE: &str = "sine=frequency=440:sample_rate=16000:duration=18";
/// A silent audio track (an audio stream with no speech).
const SILENCE: &str = "anullsrc=channel_layout=mono:sample_rate=16000";

/// A three-slide synthetic meeting (6 s each) with an audio track from the
/// given lavfi source, or none.
fn make_video(path: &Path, audio: Option<&str>) {
    let board_a: Vec<String> = [
        Some(outline(LEDGER)),
        Some(outline(ORBIT)),
        // Connector from the right edge of one box to the left edge of the other.
        Some(filled([460, 258, 760, 263], "0x222222")),
        Some(filled(STICKY, "0xF5D547")),
        Some(filled(TAG, "0x5DBB63")),
        text("Ledger API", 200, 245),
        text("Orbit Queue", 800, 245),
        text("REST", 585, 225),
        text("Retries?", 200, 510),
        text("Ada", 485, 335),
    ]
    .into_iter()
    .flatten()
    .collect();
    let mut board_b = board_a.clone();
    board_b.push(outline(PARCEL));
    board_b.extend(text("Parcel Store", 600, 495));
    let chat: Vec<String> = [
        Some(filled([80, 120, 1200, 180], "0x3A3F4B")),
        Some(filled([80, 220, 900, 280], "0x3A3F4B")),
        text("general", 100, 135),
    ]
    .into_iter()
    .flatten()
    .collect();
    let graph = format!(
        "color=c=0xFAFAFA:s={W}x{H}:r=10:d=6,{}[a];\
         color=c=0xFAFAFA:s={W}x{H}:r=10:d=6,{}[b];\
         color=c=0x1E2129:s={W}x{H}:r=10:d=6,{}[c];\
         [a][b][c]concat=n=3:v=1:a=0,format=yuv420p[v]",
        board_a.join(","),
        board_b.join(","),
        chat.join(",")
    );
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-filter_complex",
        &graph,
    ]);
    if let Some(src) = audio {
        cmd.args(["-f", "lavfi", "-i", src]);
        cmd.args(["-map", "[v]", "-map", "0:a", "-c:a", "aac", "-shortest"]);
    } else {
        cmd.args(["-map", "[v]"]);
    }
    cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p"]);
    cmd.arg(path);
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn mean_luma(img: &RgbImage) -> f64 {
    let n = f64::from(img.width()) * f64::from(img.height());
    img.pixels()
        .map(|p| 0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]))
        .sum::<f64>()
        / n.max(1.0)
}

/// True when the pixel at a frame-relative position is dark (a drawn outline).
fn dark_at(img: &RgbImage, fx: f64, fy: f64) -> bool {
    let x = ((fx * f64::from(img.width())) as u32).min(img.width().saturating_sub(1));
    let y = ((fy * f64::from(img.height())) as u32).min(img.height().saturating_sub(1));
    let p = img.get_pixel(x, y);
    u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]) < 200
}

fn has_parcel(img: &RgbImage) -> bool {
    // Left edge of the third box, halfway down.
    let fx = (f64::from(PARCEL[0]) + 2.0) / f64::from(W);
    let fy = (f64::from(PARCEL[1] + PARCEL[3]) / 2.0) / f64::from(H);
    dark_at(img, fx, fy)
}

fn span(t: &str, b: [u32; 4]) -> RecognizedSpan {
    RecognizedSpan {
        text: t.into(),
        bbox: PixelBox {
            x1: f64::from(b[0]),
            y1: f64::from(b[1]),
            x2: f64::from(b[2]),
            y2: f64::from(b[3]),
        },
        confidence: 0.95,
        det_score: 0.9,
    }
}

/// OCR that recognizes the drawn slides by their pixels.
struct ScriptedOcr;

impl TextRecognizer for ScriptedOcr {
    fn recognize(&self, img: &RgbImage) -> Result<Vec<RecognizedSpan>, OcrError> {
        if mean_luma(img) < 100.0 {
            return Ok(vec![span("general", [100, 135, 220, 165])]);
        }
        let mut v = vec![
            span("Ledger API", [200, 245, 380, 275]),
            span("Orbit Queue", [800, 245, 990, 275]),
            span("REST", [585, 225, 660, 252]),
            span("Retries?", [200, 510, 330, 540]),
            span("Ada", [485, 335, 545, 362]),
        ];
        if has_parcel(img) {
            v.push(span("Parcel Store", [600, 495, 800, 525]));
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

fn decode(request: &VisionRequest) -> RgbImage {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(request.image.base64())
        .unwrap();
    image::load_from_memory(&bytes).unwrap().to_rgb8()
}

fn frac(b: [u32; 4], w: f64, h: f64) -> Value {
    json!([
        f64::from(b[0]) / f64::from(W) * w,
        f64::from(b[1]) / f64::from(H) * h,
        f64::from(b[2]) / f64::from(W) * w,
        f64::from(b[3]) / f64::from(H) * h
    ])
}

/// Vision model that reads the drawn slides from their pixels. Boxes are given
/// as shares of the sent image, which is the whole frame (the classifier
/// reports the full frame as the canvas).
struct ScriptedVision;

#[async_trait]
impl VisionBackend for ScriptedVision {
    fn id(&self) -> BackendId {
        BackendId {
            backend: "scripted".into(),
            model: VISION_MODEL.into(),
            digest: Some("sha256:scripted-vision".into()),
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
        let img = decode(&request);
        let (w, h) = (f64::from(img.width()), f64::from(img.height()));
        let board = mean_luma(&img) >= 100.0;
        let value = if request.schema.name().contains("ScreenClass") {
            json!({
                "screen_type": if board { "whiteboard" } else { "chat" },
                "app_hint": if board { "Miro" } else { "" },
                "bbox_2d": [0.0, 0.0, w, h],
                "confidence": 0.93
            })
        } else {
            let mut nodes = vec![
                json!({"local_id": "n1", "text": "Ledger API", "bbox_2d": frac(LEDGER, w, h), "conf": 0.95}),
                json!({"local_id": "n2", "text": "Orbit Queue", "bbox_2d": frac(ORBIT, w, h), "conf": 0.95}),
            ];
            if has_parcel(&img) {
                nodes.push(json!({"local_id": "n3", "text": "Parcel Store", "bbox_2d": frac(PARCEL, w, h), "conf": 0.9}));
            }
            json!({
                "nodes": nodes,
                "edges": [{"src": "n1", "dst": "n2", "label": "REST",
                           "label_bbox_2d": frac([585, 225, 660, 252], w, h),
                           "style": "solid", "conf": 0.9}],
                "stickies": [{"text": "Retries?", "color": "yellow", "bbox_2d": frac(STICKY, w, h)}],
                "owner_tags": [{"name_raw": "Ada", "near": "n1", "bbox_2d": frac(TAG, w, h)}],
                "other_visible_text": [],
                "confidence": 0.9
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
            prompt_eval_count: Some(900),
            eval_count: Some(120),
            durations: Durations::default(),
            attempts: 1,
            repaired: false,
            done_reason: Some("stop".into()),
        })
    }
}

/// Two short utterances on the audio timeline.
struct ScriptedAsr;

fn words(start: f64, text: &str) -> Vec<AsrWord> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, w)| AsrWord {
            w: w.into(),
            start_s: start + 0.4 * i as f64,
            end_s: start + 0.4 * (i + 1) as f64,
            p: 0.92,
        })
        .collect()
}

impl AsrEngine for ScriptedAsr {
    fn describe(&self) -> Value {
        json!({"engine": "scripted-asr"})
    }
    fn transcribe(
        &self,
        samples: &[f32],
        vocabulary: &[String],
    ) -> glassrip_audio::Result<AsrOutput> {
        assert!(!samples.is_empty(), "audio samples reach ASR");
        assert!(
            vocabulary.iter().any(|v| v == "Ledger"),
            "on-screen vocabulary reaches ASR: {vocabulary:?}"
        );
        let a = words(1.0, "We keep the Ledger API on REST for the pilot.");
        let b = words(9.0, "Ada will build the Orbit Queue this week.");
        Ok(AsrOutput {
            segments: vec![
                AsrSegment {
                    start_s: 1.0,
                    end_s: a.last().map_or(1.0, |w| w.end_s),
                    words: a,
                },
                AsrSegment {
                    start_s: 9.0,
                    end_s: b.last().map_or(9.0, |w| w.end_s),
                    words: b,
                },
            ],
            chunk_spans: vec![(0.0, 18.0)],
            speech_regions: 2,
            prompt_tokens: 4,
            prompt_terms: vocabulary.to_vec(),
            backend: "scripted".into(),
        })
    }
}

struct ScriptedDiarizer;

impl DiarizeEngine for ScriptedDiarizer {
    fn describe(&self) -> Value {
        json!({"engine": "scripted-diarizer"})
    }
    fn diarize(&self, _samples: &[f32]) -> glassrip_audio::Result<Diarization> {
        Ok(Diarization {
            turns: vec![Turn::new(0.0, 8.0, 0), Turn::new(8.0, 18.0, 1)],
            labels: vec!["SPEAKER_00".into(), "SPEAKER_01".into()],
            talk_time_s: vec![8.0, 10.0],
            num_clusters_raw: 2,
            active_s: 18.0,
            centroids: vec![None, None],
        })
    }
}

/// ASR on a silent track: voice activity detection finds no speech.
struct SilentAsr;

impl AsrEngine for SilentAsr {
    fn describe(&self) -> Value {
        json!({"engine": "silent-asr"})
    }
    fn transcribe(
        &self,
        samples: &[f32],
        vocabulary: &[String],
    ) -> glassrip_audio::Result<AsrOutput> {
        assert!(!samples.is_empty(), "the silent track still has samples");
        Ok(AsrOutput {
            segments: vec![],
            chunk_spans: vec![],
            speech_regions: 0,
            prompt_tokens: 0,
            prompt_terms: vocabulary.to_vec(),
            backend: "scripted".into(),
        })
    }
}

/// Diarization of a silent track: no turns and no labels.
struct SilentDiarizer;

impl DiarizeEngine for SilentDiarizer {
    fn describe(&self) -> Value {
        json!({"engine": "silent-diarizer"})
    }
    fn diarize(&self, _samples: &[f32]) -> glassrip_audio::Result<Diarization> {
        Ok(Diarization {
            turns: vec![],
            labels: vec![],
            talk_time_s: vec![],
            num_clusters_raw: 0,
            active_s: 0.0,
            centroids: vec![],
        })
    }
}

/// Text model answering every notes call with the same cited draft; logs the
/// phase C model sequence.
struct ScriptedText {
    log: Mutex<Vec<String>>,
    resident: Mutex<Vec<String>>,
}

impl ScriptedText {
    fn new() -> Self {
        Self {
            log: Mutex::new(Vec::new()),
            // The vision model is still loaded when phase C starts.
            resident: Mutex::new(vec![VISION_MODEL.into()]),
        }
    }
}

#[async_trait]
impl TextBackend for ScriptedText {
    async fn chat(&self, model: &str, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        self.log
            .lock()
            .unwrap()
            .push(format!("chat {model} {}", req.purpose));
        let draft = json!({
            "decisions": [{"text": "Keep the Ledger API on REST for the pilot",
                           "segment_ids": ["seg_00000"], "event_ids": [], "keyframe_ids": [],
                           "quote": "keep the Ledger API on REST"}],
            "action_items": [{"owner": "Ada", "task": "Build the Orbit Queue this week",
                              "segment_ids": ["seg_00001"], "event_ids": [], "keyframe_ids": [],
                              "quote": "build the Orbit Queue this week"}],
            "open_questions": [],
            "timeline": [],
            "summary": [{"text": "The team reviewed the ledger design",
                         "segment_ids": ["seg_00000"], "event_ids": [], "keyframe_ids": []}]
        });
        Ok(ChatResponse {
            content: draft.to_string(),
            eval_count: Some(80),
            done_reason: Some("stop".into()),
            ..ChatResponse::default()
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
        Ok(Some("sha256:scripted-text".into()))
    }
}

/// Placement probe for a model that is not loaded yet (`Ok(None)`).
struct NotLoaded;

#[async_trait]
impl PlacementProbe for NotLoaded {
    async fn preflight(&self) -> Result<Placement, VisionError> {
        StaticProbe.preflight().await
    }
    async fn placement(&self) -> Result<Option<Placement>, VisionError> {
        Ok(None)
    }
    async fn digest(&self) -> Result<String, VisionError> {
        StaticProbe.digest().await
    }
    async fn server_up(&self) -> bool {
        true
    }
}

/// A text server that is down.
struct DownText;

#[async_trait]
impl TextBackend for DownText {
    async fn chat(&self, _m: &str, _r: &ChatRequest) -> Result<ChatResponse, LlmError> {
        Err(LlmError::Transport("down".into()))
    }
    async fn load(&self, _m: &str) -> Result<(), LlmError> {
        Err(LlmError::Transport("down".into()))
    }
    async fn unload(&self, _m: &str) -> Result<(), LlmError> {
        Err(LlmError::Transport("down".into()))
    }
    async fn loaded(&self) -> Result<Vec<LoadedModel>, LlmError> {
        Err(LlmError::Transport("down".into()))
    }
    async fn digest(&self, _m: &str) -> Result<Option<String>, LlmError> {
        Err(LlmError::Transport("down".into()))
    }
}

fn backends(raw: &Path, text: Arc<ScriptedText>) -> Backends {
    let mut b = Backends::none("unused");
    b.ocr = Ok(Arc::new(ScriptedOcr));
    b.vision = Ok(VisionBackends {
        backend: Arc::new(RecordingBackend::new(
            Arc::new(ScriptedVision),
            RawStore::new(raw),
        )),
        probe: Arc::new(StaticProbe),
        model: VISION_MODEL.into(),
        digest: Some("sha256:scripted-vision".into()),
        server_version: None,
        concurrency: 2,
        size_bytes: None,
        parameter_size_b: None,
        offline: None,
    });
    b.text = Ok(text);
    b.asr = Ok(Arc::new(ScriptedAsr));
    b.diarize = Ok(Arc::new(ScriptedDiarizer));
    b
}

fn options(video: PathBuf, out: PathBuf, workspace: &Path) -> MeetingOptions {
    let mut config = Config::default();
    config.frames.scale_width = W;
    config.orient.override_rotation_deg = Some(0);
    config.models.vision = VISION_MODEL.into();
    config.models.text = TEXT_MODEL.into();
    config.audio.speakers = Some(2);
    let mut o = MeetingOptions::new(video, out, workspace, config);
    o.participants = vec!["Ada Quill".into(), "Bo Tran".into()];
    o.allow_model_download = false;
    o
}

fn manifest(out: &Path) -> RunManifest {
    serde_json::from_slice(&std::fs::read(out.join("run.lock.json")).unwrap()).unwrap()
}

fn items<T: serde::de::DeserializeOwned>(out: &Path, schema: &str) -> Vec<T> {
    glassrip_core::jsonl::read::<Record<T>>(
        &out.join("artifacts").join(format!("{schema}.jsonl")),
        &SchemaReq::new(schema, 1),
    )
    .unwrap()
    .items
    .into_iter()
    .filter_map(|r| r.outcome.result)
    .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn meeting_mode_end_to_end_on_a_synthetic_video() {
    if !require_ffmpeg() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let video = root.join("weekly-sync.mp4");
    make_video(&video, Some(TONE));
    let out = root.join("weekly-sync.glassrip");
    let log = out.join(meeting::logging::RUN_LOG);

    // ---- full run
    let text = Arc::new(ScriptedText::new());
    let mut opts = options(video.clone(), out.clone(), root);
    opts.log = Some(meeting::logging::init());
    let outcome = run_meeting(
        &opts,
        backends(&out.join(meeting::RAW_RESPONSES_DIR), Arc::clone(&text)),
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert!(outcome.phase_b_concurrent);
    for phase in ["media", "phase_a", "phase_b", "join", "phase_c", "render"] {
        assert!(outcome.phase_wall_s.contains_key(phase), "{phase}");
    }

    // Every stage of the graph ran and wrote its artifact.
    let m = manifest(&out);
    for d in meeting_mode_stage_decls() {
        let rec = m
            .stages
            .get(&d.name)
            .unwrap_or_else(|| panic!("{} not in manifest", d.name));
        assert_eq!(rec.status, StageStatus::Ok, "{}: {rec:?}", d.name);
        assert!(
            out.join("artifacts")
                .join(format!("{}.jsonl", d.output))
                .is_file(),
            "{}",
            d.output
        );
    }
    assert!(m.inputs.iter().any(|i| i.path.ends_with("weekly-sync.mp4")));
    assert!(m.tool_versions.contains_key("ffmpeg"));
    assert!(m.commands.iter().any(|c| c.stage == "audio_extract"));

    // run.log.jsonl holds structured events with stage spans.
    let log_text = std::fs::read_to_string(&log).unwrap();
    let events: Vec<Value> = log_text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(events
        .iter()
        .any(|e| e["fields"]["message"] == "stage finished"));

    // Keyframes, screen types, board, transcript, speakers, notes.
    let keyframes = std::fs::read_dir(out.join("frames/keyframes"))
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .is_ok_and(|e| e.path().extension().is_some_and(|x| x == "jpg"))
        })
        .count();
    assert!(keyframes >= 2, "{keyframes} keyframe images");
    let classes: Vec<Value> = items(&out, "glassrip.screen_class");
    let types: Vec<&str> = classes
        .iter()
        .filter_map(|c| c["screen_type"].as_str())
        .collect();
    assert!(
        types.contains(&"whiteboard") && types.contains(&"chat"),
        "{types:?}"
    );
    let readings: Vec<Value> = items(&out, "glassrip.board_reading");
    assert!(!readings.is_empty());
    let boards: Vec<Value> = items(&out, "glassrip.board_state");
    assert_eq!(boards.len(), 1, "one board");
    let node_texts: Vec<&str> = boards[0]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n["text"].as_str())
        .collect();
    assert!(node_texts.contains(&"Ledger API"), "{node_texts:?}");
    let transcript: Vec<glassrip_audio::types::TranscriptSegment> =
        items(&out, "glassrip.transcript");
    assert_eq!(transcript.len(), 2, "{transcript:?}");
    assert_eq!(transcript[0].speaker_label, "SPEAKER_00");
    assert_eq!(transcript[1].speaker_label, "SPEAKER_01");
    assert!(transcript[0].text.contains("Ledger API"));
    let notes: Vec<MeetingNotes> = items(&out, "glassrip.meeting_notes");
    assert_eq!(notes.len(), 1);
    assert!(
        notes[0]
            .decisions
            .iter()
            .any(|d| d.text.contains("Ledger API")),
        "{:?}",
        notes[0].decisions
    );
    // Phase C: the vision model is unloaded before the text model loads.
    let calls = text.log.lock().unwrap().clone();
    let unload = calls
        .iter()
        .position(|c| c == &format!("unload {VISION_MODEL}"));
    let load = calls
        .iter()
        .position(|c| c == &format!("load {TEXT_MODEL}"));
    assert!(
        matches!((unload, load), (Some(u), Some(l)) if u < l),
        "{calls:?}"
    );

    // Rendered outputs.
    let md = out.join("weekly-sync-meeting-notes.md");
    let svg = out.join("weekly-sync-architecture.svg");
    let png = out.join("weekly-sync-architecture.png");
    for p in [&md, &svg, &png] {
        assert!(outcome.outputs.contains(p), "{} not reported", p.display());
    }
    let md_text = std::fs::read_to_string(&md).unwrap();
    assert!(md_text.contains("Keep the Ledger API on REST"), "{md_text}");
    let svg_text = std::fs::read_to_string(&svg).unwrap();
    assert!(svg_text.trim_start().starts_with("<svg") || svg_text.starts_with("<?xml"));
    assert!(svg_text.contains("Ledger API"));
    let png_bytes = std::fs::read(&png).unwrap();
    assert!(png_bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    // Vision replies were recorded for offline replay.
    assert!(out.join(meeting::RAW_RESPONSES_DIR).is_dir());

    // ---- eval scores the real run: `glassrip eval --suite meeting --artifacts <run dir>`.
    score_run_with_eval(root, &out).await;

    // ---- rerun: everything restores from cache except the forced render.
    let rerun = run_meeting(
        &opts,
        backends(
            &out.join(meeting::RAW_RESPONSES_DIR),
            Arc::new(ScriptedText::new()),
        ),
        CancellationToken::new(),
    )
    .await
    .unwrap();
    for r in &rerun.reports {
        let expected = if r.stage == "render" {
            StageStatus::Ok
        } else {
            StageStatus::Cached
        };
        assert_eq!(r.status, expected, "{}", r.stage);
    }
    // Reports come in graph order, whatever order concurrent chains finished in.
    let graph_order: Vec<String> =
        glassrip_core::graph::StageGraph::new(meeting_mode_stage_decls())
            .unwrap()
            .order()
            .into_iter()
            .map(str::to_string)
            .collect();
    let reported: Vec<String> = rerun.reports.iter().map(|r| r.stage.clone()).collect();
    assert_eq!(reported, graph_order);

    // ---- offline rerun: model servers down, digests from run.lock.json.
    let recorded = glassrip::meeting::backends::recorded_digests(&out);
    let offline = |raw: &Path| {
        let mut b = backends(raw, Arc::new(ScriptedText::new()));
        b.vision = Ok(VisionBackends::offline(
            VISION_MODEL,
            &recorded[VISION_MODEL],
            "server down in this test",
        ));
        b.text = Ok(Arc::new(DownText));
        b.text_offline = Some("server down in this test".into());
        b
    };
    let cached = run_meeting(
        &opts,
        offline(&out.join(meeting::RAW_RESPONSES_DIR)),
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert!(cached
        .reports
        .iter()
        .all(|r| r.stage == "render" || r.status == StageStatus::Cached));
    // A model stage that is not cached fails clearly instead of calling the server.
    let mut forced = opts.clone();
    forced.selection.force.insert("board_read".into());
    let err = run_meeting(
        &forced,
        offline(&out.join(meeting::RAW_RESPONSES_DIR)),
        CancellationToken::new(),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(&err, glassrip::meeting::MeetingError::Offline { stage, .. } if stage == "board_read"),
        "{err}"
    );

    // ---- --from-stage board_read with a large model that is not loaded yet:
    // phase B must run board reading and audio one after the other.
    let mut from = opts.clone();
    from.selection.from = Some("board_read".into());
    let mut large = backends(
        &out.join(meeting::RAW_RESPONSES_DIR),
        Arc::new(ScriptedText::new()),
    );
    if let Ok(v) = &mut large.vision {
        v.probe = Arc::new(NotLoaded);
        v.size_bytes = Some(21_000_000_000);
        assert_eq!(v.slots(14.0), 1, "large models get one slot");
    }
    let seq = run_meeting(&from, large, CancellationToken::new())
        .await
        .unwrap();
    assert!(
        !seq.phase_b_concurrent,
        "large unloaded model must be sequential"
    );
    let ran: Vec<&str> = seq
        .reports
        .iter()
        .filter(|r| r.status != StageStatus::Skipped)
        .map(|r| r.stage.as_str())
        .collect();
    assert_eq!(ran.first(), Some(&"board_read"), "{ran:?}");
    assert!(
        !ran.contains(&"classify") && !ran.contains(&"asr"),
        "{ran:?}"
    );
    // Unknown size is sequential too.
    let mut unknown = backends(
        &out.join(meeting::RAW_RESPONSES_DIR),
        Arc::new(ScriptedText::new()),
    );
    if let Ok(v) = &mut unknown.vision {
        v.probe = Arc::new(NotLoaded);
    }
    let seq = run_meeting(&from, unknown, CancellationToken::new())
        .await
        .unwrap();
    assert!(!seq.phase_b_concurrent, "unknown size must be sequential");

    // ---- a video without an audio stream: empty transcript, board-only notes.
    let silent = root.join("silent-board.mp4");
    make_video(&silent, None);
    let out2 = root.join("silent-board.glassrip");
    let outcome2 = run_meeting(
        &options(silent, out2.clone(), root),
        backends(
            &out2.join(meeting::RAW_RESPONSES_DIR),
            Arc::new(ScriptedText::new()),
        ),
        CancellationToken::new(),
    )
    .await
    .unwrap();
    let audio: Vec<Value> = items(&out2, "glassrip.audio");
    assert_eq!(audio[0]["has_audio"], json!(false));
    let empty: Vec<Value> = items(&out2, "glassrip.transcript");
    assert!(
        empty.is_empty(),
        "no audio stream yields an empty transcript"
    );
    assert!(out2.join("artifacts/glassrip.transcript.jsonl").is_file());
    assert!(outcome2
        .outputs
        .iter()
        .any(|p| p.ends_with("silent-board-meeting-notes.md")));
}

/// Scores the run directory with the meeting suite against a small fictional
/// golden set in a temporary private root.
async fn score_run_with_eval(root: &Path, run: &Path) {
    let private = root.join("private");
    std::fs::create_dir_all(private.join("golden")).unwrap();
    let golden = json!({
        "golden_version": 1,
        "meeting": "fictional weekly sync",
        "participants": [
            {"person_id": "ada-quill", "display_name": "Ada Quill", "aliases": ["Ada"]},
            {"person_id": "bo-tran", "display_name": "Bo Tran", "aliases": []}
        ],
        "screen_types": [
            {"t_rep_s": 2.0, "screen_type": "whiteboard"},
            {"t_rep_s": 14.0, "screen_type": "chat"}
        ],
        "final_board": {
            "nodes": [
                {"id": "ledger", "text": "Ledger API"},
                {"id": "orbit", "text": "Orbit Queue"},
                {"id": "parcel", "text": "Parcel Store"}
            ],
            "edges": [{"src": "ledger", "dst": "orbit", "label": "REST"}],
            "stickies": [{"text": "Retries?"}]
        },
        "owners": {"probes_s": [], "assignments": [], "moves": [], "negatives": []},
        "static_windows": [],
        "chrome_terms": [],
        "transcript": {
            "decisions": [{"text": "Keep the Ledger API on REST for the pilot"}],
            "action_items": [{"text": "Build the Orbit Queue this week", "person_id": "ada-quill"}],
            "open_questions": [],
            "negative_action_items": [],
            "hotwords": [],
            "speaker_count": 2
        }
    });
    std::fs::write(
        private.join("golden/meeting_golden.json"),
        serde_json::to_vec_pretty(&golden).unwrap(),
    )
    .unwrap();
    let cfg = root.join("eval.toml");
    std::fs::write(
        &cfg,
        format!("[eval]\nprivate_fixtures = \"{}\"\n", private.display()),
    )
    .unwrap();
    let report_dir = root.join("eval-report");
    #[derive(clap::Parser)]
    struct Wrap {
        #[command(flatten)]
        args: glassrip_eval::cli::EvalArgs,
    }
    let args = <Wrap as clap::Parser>::try_parse_from([
        "eval",
        "--suite",
        "meeting",
        "--config",
        cfg.to_str().unwrap(),
        "--artifacts",
        run.to_str().unwrap(),
        "--out",
        report_dir.to_str().unwrap(),
    ])
    .unwrap()
    .args;
    let outcome = glassrip_eval::cli::run(args).await.unwrap();
    assert!(outcome.passed, "{:?}", outcome.status);
    let report: Value =
        serde_json::from_slice(&std::fs::read(report_dir.join("eval_report.json")).unwrap())
            .unwrap();
    let metric = |k: &str| {
        report["metrics"][k]["mean"]
            .as_f64()
            .unwrap_or_else(|| panic!("{k}: {report}"))
    };
    assert_eq!(metric("screen.accuracy"), 1.0);
    assert!(metric("board.node.recall") >= 2.0 / 3.0);
    assert_eq!(metric("board.chrome_fp"), 0.0);
    assert_eq!(metric("notes.decision.recall"), 1.0);
    assert_eq!(metric("notes.action.recall"), 1.0);
    assert_eq!(metric("audio.speaker_labels"), 2.0);
    assert_eq!(report["not_run"], json!([]), "{}", report["not_run"]);
}

/// A screencast with a silent audio track and no participant list: the audio
/// chain runs and finds no speech, no participant names are known, and the run
/// still names speakers (none), writes board-only notes and renders.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn meeting_mode_survives_a_silent_track_without_participants() {
    if !require_ffmpeg() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let video = root.join("silent-talk.mp4");
    make_video(&video, Some(SILENCE));
    let out = root.join("silent-talk.glassrip");
    let text = Arc::new(ScriptedText::new());
    let mut b = backends(&out.join(meeting::RAW_RESPONSES_DIR), Arc::clone(&text));
    b.asr = Ok(Arc::new(SilentAsr));
    b.diarize = Ok(Arc::new(SilentDiarizer));
    let mut opts = options(video, out.clone(), root);
    opts.participants = vec![];
    let outcome = run_meeting(&opts, b, CancellationToken::new())
        .await
        .unwrap();

    let audio: Vec<Value> = items(&out, "glassrip.audio");
    assert_eq!(audio[0]["has_audio"], json!(true), "the track is there");
    let transcript: Vec<Value> = items(&out, "glassrip.transcript");
    assert!(
        transcript.is_empty(),
        "silence yields no transcript segments"
    );
    let m = manifest(&out);
    for stage in ["name_speakers", "notes", "render"] {
        assert_eq!(m.stages[stage].status, StageStatus::Ok, "{stage}");
    }
    let speakers: Vec<Value> = items(&out, "glassrip.speakers");
    let kinds: Vec<&str> = speakers.iter().filter_map(|r| r["kind"].as_str()).collect();
    assert_eq!(kinds, vec!["summary"], "{speakers:?}");
    assert!(
        speakers[0]["method"]
            .as_str()
            .is_some_and(|m| m.starts_with("none: no transcript segments")),
        "{speakers:?}"
    );
    assert!(
        speakers[0]["notes"]
            .as_str()
            .is_some_and(|n| n.contains("no participant names")),
        "no names were known either (the failing case): {speakers:?}"
    );
    let notes: Vec<MeetingNotes> = items(&out, "glassrip.meeting_notes");
    assert_eq!(notes[0].caveats[0].kind, "no_audio");
    assert!(notes[0].people.is_empty());
    assert!(
        !text
            .log
            .lock()
            .unwrap()
            .iter()
            .any(|l| l.starts_with("chat")),
        "board-only notes call no model"
    );
    assert!(outcome
        .outputs
        .iter()
        .any(|p| p.ends_with("silent-talk-meeting-notes.md")));
}
