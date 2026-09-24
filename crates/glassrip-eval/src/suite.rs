//! Suite runners: synthetic (boards and screens), synthetic_docs, meeting, docs.
//!
//! Each runner returns a flat metric map (`name -> value`) plus JSON details, so
//! repetitions can be aggregated and compared against a baseline uniformly.
//!
//! - `synthetic` runs the real vision requests (classification on the frame, then
//!   board reading on the predicted canvas crop when the frame is classified as a
//!   whiteboard), validated by glassrip-vision, through a [`Responder`] (recorded
//!   responses in CI, a live backend with `--rerecord`).
//! - `synthetic_docs` and `docs` score `glassrip.documents` predictions from a run
//!   directory against the fixtures (no `doc_read` stage exists yet).
//! - `meeting` scores a full run's artifacts (see [`crate::views`]) against the
//!   private golden set.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use glassrip_vision::board::{
    board_read_request, validate_board, BoardReadOutput, BoardValidationConfig, CanvasSize,
    ValidatedBoard,
};
use glassrip_vision::classify::{
    classify_options, classify_request, combine, ClassifyRules, ScreenClassOutput,
};
use glassrip_vision::image_prep::prepare_board_image;
use glassrip_vision::{BBox, GenerationOptions};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::error::{EvalError, Result};
use crate::fixture::{BoardCase, DocsCase};
use crate::golden::MeetingGolden;
use crate::metrics::audio::{hotword_wer, speaker_label_count, TimedWord};
use crate::metrics::board::{
    is_chrome, score_board, BoardScore, Direction, GoldBoard, GoldSticky, NodeResolver, PredBoard,
    PredEdge, PredNode, PredOwnerTag, PredSticky,
};
use crate::metrics::docs::{score_document, DocScore, PredDocument};
use crate::metrics::events::{false_change_events, PredEvent, EVENT_TOLERANCE_S};
use crate::metrics::notes::{negative_hits, score_items, PredItem};
use crate::metrics::owners::{owner_attribution, owner_move_errors, Assignment, Target};
use crate::metrics::screen::{ScreenScore, ScreenType};
use crate::metrics::{Counts, Tally};
use crate::replay::{RequestId, Responder};
use crate::text::{median, SENTENCE_MATCH_DICE};
use crate::timejoin::{join_time, KeyframeSpan};
use crate::views::{
    self, BoardReadingItem, BoardStateItem, EdgeOrientation, Keyframe, MeetingNotes, NotesStatus,
    OwnerTarget, RunArtifacts, ScreenClassItem, SpeakersRecord, TranscriptSegment,
};

/// Flat metric map.
pub type Metrics = BTreeMap<String, f64>;

/// Output of one suite pass.
#[derive(Debug, Clone, Default)]
pub struct SuiteRun {
    /// Metric values.
    pub metrics: Metrics,
    /// Per-case or per-section details.
    pub details: Value,
    /// Metrics or sections that could not run, with the reason.
    pub not_run: Vec<String>,
    /// Case errors (missing responses, invalid replies, unreadable files).
    pub errors: Vec<String>,
    /// Per-keyframe latencies of gold whiteboard cases, seconds.
    pub latencies_s: Vec<f64>,
    /// Hard-gate failures detected while scoring (for example degraded notes).
    pub gate_failures: Vec<String>,
}

fn put_prf(m: &mut Metrics, prefix: &str, c: Counts) {
    m.insert(format!("{prefix}.precision"), c.precision());
    m.insert(format!("{prefix}.recall"), c.recall());
    m.insert(format!("{prefix}.f1"), c.f1());
}

fn put_tally(m: &mut Metrics, key: &str, t: Tally) {
    if let Some(a) = t.accuracy() {
        m.insert(key.to_string(), a);
    }
}

fn put_board(m: &mut Metrics, s: &BoardScore) {
    put_prf(m, "board.node", s.nodes);
    put_tally(m, "board.node.core_recall", s.core_nodes);
    put_prf(m, "board.edge", s.edges);
    put_tally(m, "board.edge.direction_accuracy", s.edge_direction);
    put_tally(m, "board.edge.label_accuracy", s.edge_label);
    put_tally(m, "board.edge.style_accuracy", s.edge_style);
    m.insert("board.edge.uncertain".into(), s.edge_uncertain as f64);
    put_prf(m, "board.sticky", s.stickies);
    if s.sticky_cer.reference_len > 0 {
        m.insert("board.sticky.cer".into(), s.sticky_cer.rate());
    }
    put_tally(m, "board.owner.accuracy", s.owners);
    m.insert("board.owner.fp".into(), s.owner_fp as f64);
    m.insert("board.chrome_fp".into(), s.chrome_fp as f64);
}

fn put_screen(m: &mut Metrics, s: &ScreenScore) {
    put_tally(m, "screen.accuracy", s.tally);
    m.insert(
        "screen.cms_as_whiteboard".into(),
        s.cms_as_whiteboard as f64,
    );
    m.insert("screen.missed".into(), s.missed as f64);
}

/// Everything a synthetic case needs from the run.
pub struct SuiteContext {
    /// Response source.
    pub responder: Responder,
    /// Vision model name.
    pub model: String,
    /// Generation seed.
    pub seed: u64,
    /// Board-read `num_predict`.
    pub num_predict: u32,
    /// Cancellation.
    pub cancel: CancellationToken,
}

/// Result of one synthetic board or screen case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardCaseResult {
    /// Case name.
    pub case: String,
    /// Gold screen type.
    pub gold_screen: ScreenType,
    /// Predicted screen type (`None` on error).
    pub pred_screen: Option<ScreenType>,
    /// Classification method.
    pub classify_method: Option<String>,
    /// Predicted board, when the frame was read.
    pub board: Option<PredBoard>,
    /// Scores.
    pub score: BoardScore,
    /// Classification plus board-read wall time, seconds.
    pub latency_s: f64,
    /// Error, if the case could not be evaluated.
    pub error: Option<String>,
}

fn crop_rect(b: &BBox, w: u32, h: u32) -> (u32, u32, u32, u32) {
    let x0 = b.x1.floor().clamp(0.0, f64::from(w.saturating_sub(1))) as u32;
    let y0 = b.y1.floor().clamp(0.0, f64::from(h.saturating_sub(1))) as u32;
    let x1 = (b.x2.ceil().clamp(0.0, f64::from(w)) as u32).max(x0 + 1);
    let y1 = (b.y2.ceil().clamp(0.0, f64::from(h)) as u32).max(y0 + 1);
    (x0, y0, x1 - x0, y1 - y0)
}

/// Converts a validated reading (canvas pixels) to a predicted board in frame pixels.
pub fn pred_board_from(v: &ValidatedBoard, ox: f64, oy: f64) -> PredBoard {
    let shift = |b: &BBox| BBox::new(b.x1 + ox, b.y1 + oy, b.x2 + ox, b.y2 + oy);
    let text_of: BTreeMap<&str, &str> = v
        .nodes
        .iter()
        .map(|n| (n.local_id.as_str(), n.text.as_str()))
        .collect();
    PredBoard {
        nodes: v
            .nodes
            .iter()
            .map(|n| PredNode {
                text: n.text.clone(),
                bbox: Some(shift(&n.bbox)),
            })
            .collect(),
        edges: v
            .edges
            .iter()
            .map(|e| PredEdge {
                src: text_of
                    .get(e.src.as_str())
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                dst: text_of
                    .get(e.dst.as_str())
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                label: e.label.clone(),
                style: e.style.into(),
                direction: Direction::Forward,
            })
            .collect(),
        stickies: v
            .stickies
            .iter()
            .map(|s| PredSticky {
                text: s.text.clone(),
                bbox: Some(shift(&s.bbox)),
            })
            .collect(),
        owner_tags: v
            .owner_tags
            .iter()
            .map(|o| PredOwnerTag {
                name: o.name_raw.clone(),
                near: text_of
                    .get(o.near.as_str())
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
            })
            .collect(),
        other_text: v
            .other_visible_text
            .iter()
            .map(|t| t.text.clone())
            .collect(),
    }
}

struct CaseOutput {
    pred_screen: ScreenType,
    method: String,
    board: Option<PredBoard>,
    latency_s: f64,
}

async fn board_case_inner(ctx: &SuiteContext, case: &BoardCase) -> Result<CaseOutput> {
    let frame_path = case.frame_path();
    let source_blake3 =
        glassrip_core::blake3_file(&frame_path).map_err(|e| EvalError::io(&frame_path, e))?;
    let img = image::open(&frame_path).map_err(|e| EvalError::Image {
        path: frame_path.clone(),
        message: e.to_string(),
    })?;
    let (w, h) = (img.width(), img.height());

    let (request, prepared) = classify_request(&img, classify_options(ctx.seed))?;
    let id = RequestId {
        kind: "classify",
        case: &case.name,
        model: &ctx.model,
        source_blake3: &source_blake3,
        extra: json!({}),
    };
    let classified = ctx
        .responder
        .respond(&id, request, ctx.cancel.clone())
        .await?;
    let mut latency_s = classified.wall_s;
    let out: ScreenClassOutput = classified.response.decode()?;
    let source = out.in_source_coords(&prepared);
    let rules = ClassifyRules::spec_examples();
    // No OCR stage runs in the synthetic suite, so no keyword rule fires.
    let class = combine(Some(&source), &rules.evaluate::<&str>(&[]), &rules);
    let method = serde_json::to_value(class.method)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();

    let mut board = None;
    if class.reads_board() {
        let full = BBox::new(0.0, 0.0, f64::from(w), f64::from(h));
        let (x, y, cw, ch) = crop_rect(&class.canvas_bbox.unwrap_or(full), w, h);
        let crop = img.crop_imm(x, y, cw, ch);
        let prepared = prepare_board_image(&crop)?;
        let request = board_read_request(
            &prepared,
            GenerationOptions {
                seed: ctx.seed,
                num_predict: ctx.num_predict,
            },
        )?;
        let id = RequestId {
            kind: "board_read",
            case: &case.name,
            model: &ctx.model,
            source_blake3: &source_blake3,
            extra: json!({ "crop": [x, y, cw, ch] }),
        };
        let read = ctx
            .responder
            .respond(&id, request, ctx.cancel.clone())
            .await?;
        latency_s += read.wall_s;
        let output: BoardReadOutput = read.response.decode()?;
        let output = output.to_canvas_coords(&prepared);
        let cfg = BoardValidationConfig {
            participant_names: case.expected.participants.clone(),
            ..Default::default()
        };
        let validated = validate_board(
            output,
            CanvasSize {
                width: f64::from(cw),
                height: f64::from(ch),
            },
            &cfg,
        );
        board = Some(pred_board_from(&validated, f64::from(x), f64::from(y)));
    }
    Ok(CaseOutput {
        pred_screen: class.screen_type.into(),
        method,
        board,
        latency_s,
    })
}

/// Runs one synthetic case; errors are recorded in the result, never raised.
pub async fn run_board_case(ctx: &SuiteContext, case: &BoardCase) -> BoardCaseResult {
    let gold = &case.expected;
    match board_case_inner(ctx, case).await {
        Ok(out) => {
            let pred = out.board.clone().unwrap_or_default();
            BoardCaseResult {
                case: case.name.clone(),
                gold_screen: gold.screen_type,
                pred_screen: Some(out.pred_screen),
                classify_method: Some(out.method),
                score: score_board(&gold.board, &pred, &gold.chrome_texts),
                board: out.board,
                latency_s: out.latency_s,
                error: None,
            }
        }
        Err(e) => BoardCaseResult {
            case: case.name.clone(),
            gold_screen: gold.screen_type,
            pred_screen: None,
            classify_method: None,
            board: None,
            score: score_board(&gold.board, &PredBoard::default(), &gold.chrome_texts),
            latency_s: 0.0,
            error: Some(e.to_string()),
        },
    }
}

/// Runs every synthetic case concurrently (the responder bounds live concurrency).
pub async fn run_board_suite(ctx: Arc<SuiteContext>, cases: &[BoardCase]) -> SuiteRun {
    let mut set = tokio::task::JoinSet::new();
    for (i, case) in cases.iter().cloned().enumerate() {
        let ctx = ctx.clone();
        set.spawn(async move { (i, run_board_case(&ctx, &case).await) });
    }
    let mut results: Vec<Option<BoardCaseResult>> = vec![None; cases.len()];
    let mut run = SuiteRun::default();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((i, r)) => results[i] = Some(r),
            Err(e) => run.errors.push(format!("case task failed: {e}")),
        }
    }
    let results: Vec<BoardCaseResult> = results.into_iter().flatten().collect();
    summarize_board_results(&results, &mut run);
    run
}

/// Aggregates synthetic case results into metrics and details.
pub fn summarize_board_results(results: &[BoardCaseResult], run: &mut SuiteRun) {
    let mut screen = ScreenScore::default();
    let mut board = BoardScore::default();
    for r in results {
        screen.record(r.gold_screen, r.pred_screen);
        board.add(&r.score);
        if let Some(e) = &r.error {
            run.errors.push(format!("{}: {e}", r.case));
        } else if r.gold_screen == ScreenType::Whiteboard {
            run.latencies_s.push(r.latency_s);
        }
    }
    put_screen(&mut run.metrics, &screen);
    put_board(&mut run.metrics, &board);
    run.metrics.insert("cases".into(), results.len() as f64);
    run.details = json!({
        "screen_confusion": screen.confusion,
        "cases": results,
    });
}

/// Loads a `glassrip.documents` prediction for one page: an envelope (items
/// filtered by `page_id`) or a bare page object. `completeness.coverage` and
/// doc-type `fields.blocks` / `fields.title` are normalized onto the view.
pub fn load_prediction(path: &Path, page_id: &str) -> Result<PredDocument> {
    let text = crate::error::read_to_string(path)?;
    let v: Value = serde_json::from_str(&text).map_err(|e| EvalError::json(path, e))?;
    let items: Vec<Value> = if v.get("schema").is_some() {
        views::load_items(path, views::schema::DOCUMENTS)?
    } else {
        vec![v]
    };
    let raw = items
        .into_iter()
        .find(|i| i.get("page_id").and_then(Value::as_str) == Some(page_id))
        .ok_or_else(|| EvalError::Fixture {
            case: page_id.to_string(),
            message: format!(
                "{} has no document with page_id `{page_id}`",
                path.display()
            ),
        })?;
    serde_json::from_value(normalize_document(raw)).map_err(|e| EvalError::json(path, e))
}

fn normalize_document(mut v: Value) -> Value {
    let Value::Object(m) = &mut v else { return v };
    if !m.contains_key("coverage") {
        if let Some(c) = m
            .get("completeness")
            .and_then(|c| c.get("coverage"))
            .cloned()
        {
            m.insert("coverage".into(), c);
        }
    }
    if m.get("type").and_then(Value::as_str) == Some("doc") {
        if let Some(Value::Object(f)) = m.remove("fields") {
            if !m.contains_key("blocks") {
                if let Some(b) = f.get("blocks") {
                    m.insert("blocks".into(), b.clone());
                }
            }
            if !m.contains_key("title") {
                if let Some(t) = f.get("title") {
                    m.insert("title".into(), t.clone());
                }
            }
        }
    }
    v
}

/// Scores document cases against predictions in `predictions/<case>/documents.json`.
pub fn run_docs_suite(cases: &[DocsCase], predictions: Option<&Path>) -> SuiteRun {
    let mut run = SuiteRun::default();
    run.metrics.insert("cases".into(), cases.len() as f64);
    let Some(root) = predictions else {
        run.not_run.push(
            "docs metrics: no predictions (the doc_read stage is not wired yet; pass --artifacts DIR with <case>/documents.json)"
                .into(),
        );
        run.details = json!({ "cases": cases.iter().map(|c| &c.name).collect::<Vec<_>>() });
        return run;
    };
    let mut body = crate::text::ErrorCount::default();
    let mut page_cer = Vec::new();
    let mut fields = Tally::default();
    let mut issue_cer = Vec::new();
    let mut blocks = Counts::default();
    let mut tables = crate::metrics::docs::TableCells::default();
    let mut coverage: Vec<f64> = Vec::new();
    let mut hallucinated = 0usize;
    let mut missing = 0usize;
    let mut per_case: BTreeMap<String, DocScore> = BTreeMap::new();
    for case in cases {
        let path = root.join(&case.name).join("documents.json");
        if !path.is_file() {
            missing += 1;
            run.errors.push(format!(
                "{}: no prediction at {}",
                case.name,
                path.display()
            ));
            continue;
        }
        match load_prediction(&path, &case.expected.document.page_id) {
            Ok(pred) => {
                let s = score_document(&case.expected.document, &pred);
                body.add(s.body);
                page_cer.push(s.body.rate());
                fields.add(s.fields);
                if let Some(f) = s.free_text {
                    issue_cer.push(f.rate());
                }
                blocks.add(s.blocks);
                tables.add(s.tables);
                if let Some(c) = s.coverage {
                    coverage.push(c);
                }
                hallucinated += s.hallucinated.len();
                per_case.insert(case.name.clone(), s);
            }
            Err(e) => run.errors.push(format!("{}: {e}", case.name)),
        }
    }
    let m = &mut run.metrics;
    if !page_cer.is_empty() {
        m.insert("docs.body_cer".into(), body.rate());
        if let Some(v) = median(&page_cer) {
            m.insert("docs.body_cer_page_median".into(), v);
        }
        m.insert(
            "docs.body_cer_page_max".into(),
            page_cer.iter().copied().fold(0.0, f64::max),
        );
        put_tally(m, "docs.fields.accuracy", fields);
        if let Some(v) = median(&issue_cer) {
            m.insert("docs.free_text_cer_issue_median".into(), v);
        }
        put_prf(m, "docs.blocks", blocks);
        put_tally(m, "docs.tables.cell_accuracy", tables.cells);
        m.insert(
            "docs.tables.truncated_cells".into(),
            tables.truncated_cells as f64,
        );
        if let Some(min) = coverage.iter().copied().reduce(f64::min) {
            m.insert("docs.coverage_min".into(), min);
        }
        m.insert("docs.hallucinated_spans".into(), hallucinated as f64);
    }
    m.insert("docs.pages_missing".into(), missing as f64);
    run.details = json!({ "cases": per_case });
    run
}

fn spans(items: &[Keyframe]) -> Vec<KeyframeSpan> {
    items
        .iter()
        .map(|k| KeyframeSpan {
            keyframe_id: k.keyframe_id.clone(),
            t_start_s: k.t_start_s,
            t_end_s: k.t_end_s,
            t_rep_s: k.t_rep_s,
        })
        .collect()
}

fn gold_board_with_groups(g: &GoldBoard) -> GoldBoard {
    let mut out = g.clone();
    for group in &g.groups {
        let title = Some(&group.title).filter(|t| !t.trim().is_empty());
        for text in title.into_iter().chain(group.items.iter()) {
            out.stickies.push(GoldSticky {
                text: text.clone(),
                aliases: vec![],
                bbox: None,
                kind: Some("card".into()),
            });
        }
    }
    out
}

fn direction_of(d: EdgeOrientation) -> Direction {
    match d {
        EdgeOrientation::Forward => Direction::Forward,
        EdgeOrientation::Uncertain => Direction::Uncertain,
        EdgeOrientation::Bidirectional => Direction::Bidirectional,
    }
}

/// Snake-case name of a serialized enum value (`NodeAdded` stays as serialized).
fn enum_name<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

fn state_to_pred(b: &BoardStateItem) -> PredBoard {
    let text_of: BTreeMap<&str, &str> = b
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), n.text.as_str()))
        .collect();
    PredBoard {
        nodes: b
            .nodes
            .iter()
            .filter(|n| n.in_final)
            .map(|n| PredNode {
                text: n.text.clone(),
                bbox: None,
            })
            .collect(),
        edges: b
            .edges
            .iter()
            .filter(|e| e.in_final)
            .map(|e| PredEdge {
                src: text_of
                    .get(e.src.as_str())
                    .copied()
                    .unwrap_or(e.src.as_str())
                    .to_string(),
                dst: text_of
                    .get(e.dst.as_str())
                    .copied()
                    .unwrap_or(e.dst.as_str())
                    .to_string(),
                label: e.label.clone(),
                style: e.style.into(),
                direction: direction_of(e.direction),
            })
            .collect(),
        stickies: b
            .stickies
            .iter()
            .filter(|s| s.in_final)
            .map(|s| PredSticky {
                text: s.text.clone(),
                bbox: None,
            })
            .collect(),
        owner_tags: Vec::new(),
        other_text: Vec::new(),
    }
}

fn reading_chrome(items: &[BoardReadingItem], chrome: &[String]) -> usize {
    items
        .iter()
        .map(|r| &r.result)
        .map(|r| {
            r.nodes
                .iter()
                .map(|n| n.text.as_str())
                .chain(r.stickies.iter().map(|s| s.text.as_str()))
                .chain(r.owner_tags.iter().map(|o| o.name_raw.as_str()))
                .chain(r.edges.iter().map(|e| e.label.as_str()))
                .chain(r.other_visible_text.iter().map(|t| t.text.as_str()))
                .filter(|t| is_chrome(t, chrome))
                .count()
        })
        .sum()
}

fn resolve_person(g: &MeetingGolden, name: &str) -> Option<String> {
    let n = crate::text::normalize_label(name);
    if ["everyone", "all", "team", "the team"].contains(&n.as_str()) {
        return Some("everyone".into());
    }
    g.resolve_person(name).map(str::to_string)
}

/// Scores a meeting run against the private golden set.
pub fn run_meeting(
    golden: &MeetingGolden,
    run_art: &RunArtifacts,
    tolerance_s: f64,
) -> Result<SuiteRun> {
    let mut run = SuiteRun::default();
    let mut details = serde_json::Map::new();
    let keyframes: Option<Vec<Keyframe>> = run_art.items(views::schema::KEYFRAMES)?;
    let kspans = keyframes.as_deref().map(spans);
    let screen_items: Option<Vec<ScreenClassItem>> = run_art.items(views::schema::SCREEN_CLASS)?;
    let state_items: Option<Vec<BoardStateItem>> = run_art.items(views::schema::BOARD_STATE)?;
    let readings: Option<Vec<BoardReadingItem>> = run_art.items(views::schema::BOARD_READING)?;
    let transcript: Option<Vec<TranscriptSegment>> = run_art.items(views::schema::TRANSCRIPT)?;
    let speakers: Option<Vec<SpeakersRecord>> = run_art.items(views::schema::SPEAKERS)?;
    let notes: Option<Vec<MeetingNotes>> = run_art.items(views::schema::MEETING_NOTES)?;
    // Errored items fail loudly: they are case errors (the no_case_errors gate),
    // never silently left out of the scores.
    let mut notes_errored = false;
    for schema in [
        views::schema::KEYFRAMES,
        views::schema::SCREEN_CLASS,
        views::schema::BOARD_READING,
        views::schema::BOARD_STATE,
        views::schema::TRANSCRIPT,
        views::schema::SPEAKERS,
        views::schema::MEETING_NOTES,
    ] {
        let failed = run_art.failed_ids(schema)?;
        if !failed.is_empty() {
            run.errors.push(format!(
                "{schema}: {} item(s) errored: {}",
                failed.len(),
                failed.join(", ")
            ));
            details.insert(format!("failed_items.{schema}"), json!(failed));
            if schema == views::schema::MEETING_NOTES {
                notes_errored = true;
            }
        }
    }
    let m = &mut run.metrics;

    // Screen types and trap coverage.
    match (&kspans, &screen_items) {
        (Some(ks), Some(sc)) => {
            let by_kf: BTreeMap<&str, Option<ScreenType>> = sc
                .iter()
                .map(|s| (s.keyframe_id.as_str(), Some(s.screen_type.into())))
                .collect();
            let mut score = ScreenScore::default();
            let mut unconfirmed = 0;
            for label in &golden.screen_types {
                if !label.confirmed {
                    unconfirmed += 1;
                    continue;
                }
                let pred = join_time(label.t_rep_s, ks, tolerance_s)
                    .and_then(|i| by_kf.get(ks[i].keyframe_id.as_str()).copied().flatten());
                score.record(label.screen_type, pred);
            }
            put_screen(m, &score);
            m.insert("screen.unconfirmed_labels".into(), f64::from(unconfirmed));
            details.insert("screen_confusion".into(), json!(score.confusion));
        }
        _ => run
            .not_run
            .push("screen metrics: glassrip.keyframes or glassrip.screen_class missing".into()),
    }
    if let Some(ks) = &kspans {
        let mut t = Tally::default();
        for trap in &golden.traps {
            let times: Vec<f64> = match (trap.t_rep_s, trap.t_from_s, trap.t_to_s) {
                (Some(t), _, _) => vec![t],
                (None, Some(a), Some(b)) => vec![a, b],
                _ => vec![],
            };
            for tt in times {
                t.record(join_time(tt, ks, tolerance_s).is_some());
            }
        }
        put_tally(m, "keyframes.trap_coverage", t);
    }

    // Final board, owners, events.
    let gold_board = gold_board_with_groups(&golden.final_board);
    let mut chrome = golden.chrome_terms.clone();
    chrome.extend(golden.owners.negatives.iter().cloned());
    if let Some(states) = &state_items {
        // The final board is the pipeline's own final state, never a gold-selected one.
        if let Some(state) = views::select_final_state(states).map(|i| &states[i]) {
            let pred = state_to_pred(state);
            let score = score_board(&gold_board, &pred, &golden.chrome_terms);
            put_board(m, &score);
            m.remove("board.owner.accuracy");
            m.remove("board.owner.fp");
            details.insert("board_chrome_hits".into(), json!(score.chrome_hits));
            details.insert("final_board_id".into(), json!(state.board_id));
        }

        // Owners and events are timed: use every state item, each resolved
        // through its own node ids.
        let resolver = NodeResolver::text_only(&golden.final_board.nodes);
        let mut pred_assign = Vec::new();
        let mut unresolved = 0usize;
        let mut negatives = 0usize;
        let mut events: Vec<PredEvent> = Vec::new();
        for state in states {
            let text_of: BTreeMap<&str, &str> = state
                .nodes
                .iter()
                .map(|n| (n.id.as_str(), n.text.as_str()))
                .collect();
            let gold_id =
                |node_id: &str| resolver.resolve(text_of.get(node_id).copied().unwrap_or(node_id));
            for o in &state.owner_assignments {
                let raw = o.person_id.clone();
                if golden.owners.negatives.iter().any(|n| {
                    crate::text::labels_match(n, &o.name_raw) || crate::text::labels_match(n, &raw)
                }) {
                    negatives += 1;
                }
                let person =
                    resolve_person(golden, &raw).or_else(|| resolve_person(golden, &o.name_raw));
                let target = match &o.target {
                    OwnerTarget::Node { node_id, .. } => gold_id(node_id).map(|n| Target::Node {
                        node: n.to_string(),
                    }),
                    OwnerTarget::Edge { src, dst, .. } => match (gold_id(src), gold_id(dst)) {
                        (Some(a), Some(b)) => Some(Target::Edge {
                            src: a.to_string(),
                            dst: b.to_string(),
                        }),
                        _ => None,
                    },
                };
                match (person, target) {
                    (Some(person_id), Some(target)) => pred_assign.push(Assignment {
                        person_id,
                        target,
                        valid_from_s: o.valid_from_s,
                        valid_to_s: Some(o.valid_to_s).filter(|t| t.is_finite()),
                    }),
                    _ => unresolved += 1,
                }
            }
            events.extend(state.events.iter().map(|e| PredEvent {
                kind: enum_name(&e.kind),
                t_s: e.t_s,
            }));
        }
        // The same timed assignment or event repeated in several state items counts once.
        let mut seen = std::collections::BTreeSet::new();
        pred_assign.retain(|a| {
            seen.insert(format!(
                "{}|{:?}|{}|{:?}",
                a.person_id, a.target, a.valid_from_s, a.valid_to_s
            ))
        });
        let mut seen = std::collections::BTreeSet::new();
        events.retain(|e| {
            seen.insert(format!(
                "{}|{}",
                crate::metrics::events::kind_key(&e.kind),
                e.t_s
            ))
        });
        let joined = |t: f64| {
            kspans
                .as_ref()
                .is_none_or(|ks| join_time(t, ks, tolerance_s).is_some())
        };
        let attr = owner_attribution(
            &golden.owners.assignments,
            &pred_assign,
            &golden.owners.probes_s,
            joined,
        );
        put_tally(m, "owners.attribution", attr.tally);
        m.insert("owners.extra".into(), attr.extra as f64);
        m.insert("owners.unresolved".into(), unresolved as f64);
        m.insert("owners.negative_hits".into(), negatives as f64);
        let moves = owner_move_errors(&golden.owners.moves, &pred_assign, tolerance_s);
        let missed = moves.iter().filter(|r| r.error_s.is_none()).count();
        m.insert("owners.moves_missed".into(), missed as f64);
        if let Some(max) = moves.iter().filter_map(|r| r.error_s).reduce(f64::max) {
            m.insert("owners.move_error_max_s".into(), max);
        }
        details.insert("owner_probes".into(), json!(attr.probes));
        details.insert("owner_moves".into(), json!(moves));

        let (n, offenders) =
            false_change_events(&golden.static_windows, &events, EVENT_TOLERANCE_S);
        m.insert("events.false_change".into(), n as f64);
        details.insert("false_change_events".into(), json!(offenders));
    } else {
        run.not_run
            .push("board, owner, and event metrics: glassrip.board_state missing".into());
    }
    if let Some(r) = &readings {
        m.insert(
            "board.chrome_fp_readings".into(),
            reading_chrome(r, &chrome) as f64,
        );
    }

    // Notes.
    match notes.as_ref().and_then(|n| n.first()) {
        Some(n) => {
            if n.report.status == NotesStatus::Degraded {
                run.gate_failures
                    .push("meeting_notes status is degraded (6.14 minimum-output alarm)".into());
            }
            let texts = |v: Vec<&str>| -> Vec<PredItem> {
                v.into_iter()
                    .map(|t| PredItem {
                        text: t.to_string(),
                        person_id: None,
                    })
                    .collect()
            };
            let decisions = texts(n.decisions.iter().map(|d| d.text.as_str()).collect());
            let questions = texts(n.open_questions.iter().map(|q| q.text.as_str()).collect());
            let actions: Vec<PredItem> = n
                .action_items
                .iter()
                .map(|a| PredItem {
                    text: a.task.clone(),
                    // "Everyone" has no person id; resolve it by the owner name
                    person_id: a
                        .person_id
                        .as_deref()
                        .or(Some(a.owner.as_str()))
                        .and_then(|p| resolve_person(golden, p)),
                })
                .collect();
            let t = &golden.transcript;
            put_prf(
                m,
                "notes.decision",
                score_items(&t.decisions, &decisions, SENTENCE_MATCH_DICE).0,
            );
            put_prf(
                m,
                "notes.action",
                score_items(&t.action_items, &actions, SENTENCE_MATCH_DICE).0,
            );
            put_prf(
                m,
                "notes.question",
                score_items(&t.open_questions, &questions, SENTENCE_MATCH_DICE).0,
            );
            let neg = negative_hits(&t.negative_action_items, &actions, SENTENCE_MATCH_DICE);
            m.insert("notes.negative_action_hits".into(), neg.len() as f64);
            details.insert("negative_action_hits".into(), json!(neg));
        }
        None if notes_errored => {
            // the notes stage ran and failed: the section fails and its targets
            // are evaluated against zero instead of being skipped
            run.gate_failures
                .push("meeting_notes: every notes item errored; notes section FAIL".into());
            for prefix in ["notes.decision", "notes.action", "notes.question"] {
                for k in ["precision", "recall", "f1"] {
                    m.insert(format!("{prefix}.{k}"), 0.0);
                }
            }
        }
        None => run
            .not_run
            .push("notes metrics: glassrip.meeting_notes missing".into()),
    }

    // Speaker naming.
    if let Some(records) = &speakers {
        let labels: Vec<_> = records
            .iter()
            .filter_map(|r| match r {
                SpeakersRecord::Label(l) => Some(l),
                _ => None,
            })
            .collect();
        let mapped: std::collections::BTreeSet<&str> = labels
            .iter()
            .filter_map(|l| l.person_id.as_deref())
            .collect();
        m.insert("speakers.labels".into(), labels.len() as f64);
        m.insert("speakers.distinct_people".into(), mapped.len() as f64);
        m.insert(
            "speakers.distinct_people_error".into(),
            (mapped.len() as f64 - golden.transcript.speaker_count as f64).abs(),
        );
        let resolved = mapped
            .iter()
            .filter(|p| resolve_person(golden, p).is_some())
            .count();
        m.insert("speakers.people_in_golden".into(), resolved as f64);
    } else {
        run.not_run
            .push("speaker metrics: glassrip.speakers missing".into());
    }

    // Transcript.
    match &transcript {
        Some(segs) => {
            let mut words = Vec::new();
            for s in segs {
                if s.words.is_empty() {
                    words.extend(s.text.split_whitespace().map(|w| TimedWord {
                        w: w.to_string(),
                        t_s: s.start_s,
                    }));
                } else {
                    words.extend(s.words.iter().map(|w| TimedWord {
                        w: w.w.clone(),
                        t_s: w.start_s,
                    }));
                }
            }
            let hw = hotword_wer(&golden.transcript.hotwords, &words, tolerance_s);
            if hw.total.reference_len > 0 {
                m.insert("audio.hotword_wer".into(), hw.total.rate());
            }
            let n = speaker_label_count(segs.iter().map(|s| s.speaker_label.as_str()));
            m.insert("audio.speaker_labels".into(), n as f64);
            m.insert(
                "audio.speaker_label_error".into(),
                (n as f64 - golden.transcript.speaker_count as f64).abs(),
            );
            details.insert("hotwords".into(), json!(hw.per_word));
        }
        None => run
            .not_run
            .push("audio metrics: glassrip.transcript missing".into()),
    }
    run.details = Value::Object(details);
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::board::{GoldEdge, GoldNode, LineStyle};

    #[test]
    fn crop_rect_clamps() {
        assert_eq!(
            crop_rect(&BBox::new(-5.0, 10.2, 100.4, 50.0), 80, 40),
            (0, 10, 80, 30)
        );
        assert_eq!(
            crop_rect(&BBox::new(10.0, 10.0, 10.0, 10.0), 80, 40),
            (10, 10, 1, 1)
        );
    }

    #[test]
    fn state_conversion_uses_node_text_and_final_flag() {
        use crate::synth::run_artifacts as ra;
        let mut gone_edge = ra::edge(("n1", "Ledger API"), ("n3", "Gone"), "", "forward");
        gone_edge["in_final"] = json!(false);
        let v = ra::board(
            "b",
            true,
            Some(60.0),
            vec![
                ra::node("n1", "Ledger API", true),
                ra::node("n2", "Orbit Queue", true),
                ra::node("n3", "Gone", false),
            ],
            vec![
                ra::edge(
                    ("n1", "Ledger API"),
                    ("n2", "Orbit Queue"),
                    "REST",
                    "forward",
                ),
                gone_edge,
            ],
            vec![ra::sticky("s1", "Who owns retries?", true)],
            vec![],
            vec![],
        );
        let b: BoardStateItem = serde_json::from_value(v).unwrap();
        let p = state_to_pred(&b);
        assert_eq!(p.nodes.len(), 2);
        assert_eq!(
            p.edges.len(),
            1,
            "edges outside the final board are not scored"
        );
        assert_eq!(p.edges[0].src, "Ledger API");
        let gold = GoldBoard {
            nodes: vec![
                GoldNode {
                    id: "a".into(),
                    text: "Ledger API".into(),
                    aliases: vec![],
                    bbox: None,
                    core: true,
                },
                GoldNode {
                    id: "b".into(),
                    text: "Orbit Queue".into(),
                    aliases: vec![],
                    bbox: None,
                    core: true,
                },
            ],
            edges: vec![GoldEdge {
                src: "a".into(),
                dst: "b".into(),
                label: "REST".into(),
                label_aliases: vec![],
                style: LineStyle::Solid,
                directed: true,
            }],
            ..Default::default()
        };
        let s = score_board(&gold, &p, &[]);
        assert_eq!(s.edges.tp, 1);
        assert_eq!(s.edge_direction.correct, 1);
    }

    #[test]
    fn prediction_page_id_must_match() {
        let dir = tempfile::tempdir().unwrap();
        let bare = dir.path().join("bare.json");
        fs_err::write(
            &bare,
            json!({"page_id": "other", "type": "doc"}).to_string(),
        )
        .unwrap();
        assert!(load_prediction(&bare, "wanted").is_err());
        assert!(load_prediction(&bare, "other").is_ok());
        let env = dir.path().join("env.json");
        let doc = json!({
            "schema": "glassrip.documents", "schema_version": "1.0.0", "run_id": "r",
            "producer": {"tool": "glassrip", "version": "0", "git_sha": null},
            "inputs": [], "params": {},
            "items": [{"page_id": "first", "type": "doc"}, {"page_id": "second", "type": "doc"}]
        });
        fs_err::write(&env, doc.to_string()).unwrap();
        assert_eq!(
            load_prediction(&env, "second").unwrap().doc.page_id,
            "second"
        );
        // No fallback to the first document.
        assert!(load_prediction(&env, "third").is_err());
    }

    #[test]
    fn document_normalization() {
        let v = json!({
            "page_id": "p", "type": "doc",
            "fields": {"title": "T", "blocks": [{"kind": "paragraph", "content": "x"}]},
            "completeness": {"coverage": 0.9}
        });
        let d: PredDocument = serde_json::from_value(normalize_document(v)).unwrap();
        assert_eq!(d.doc.title, "T");
        assert_eq!(d.doc.blocks.len(), 1);
        assert_eq!(d.coverage, Some(0.9));
    }
}
