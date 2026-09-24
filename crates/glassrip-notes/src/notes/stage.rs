//! The `notes` stage.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use glassrip_audio::types::TranscriptSegment;
use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, Stage, StageError, StageInputs, WorkItem,
};
use schemars::JsonSchema;
use semver::Version;
use serde::{Deserialize, Serialize};

use super::candidates::{board_facts, cue_lines, windows_by_time};
use super::llm::{same_model, ChatRequest, LoadedModel, OllamaTextConfig, TextBackend};
use super::prompt::PromptExtras;
use super::prompt::{
    board_digest, map_request, reduce_request, repair_context, repair_request, windows, RepairCase,
};
use super::validate::{
    assemble, check_with, dropped, merge_board_questions, CheckOptions, Checked, Corpus, Draft,
    Section,
};
use super::{
    CallRecord, Caveat, Evidence, MeetingNotes, NotesReport, NotesStatus, SpeakerLine,
    SummaryPoint, TimelineEntry,
};
use crate::board::{
    event_text, BoardExt, BoardStateItem, KeyframeTimes, KeyframeView, KEYFRAMES_MAJOR,
};
use crate::named::{named_lines, paragraphs, NamedLine};
use crate::people::AliasTable;
use crate::schemas;
use crate::speakers::{SpeakerStatus, SpeakersDoc, SpeakersRecord};
use crate::text::{mmss, sanitize_dashes};

/// Parameters of `notes`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NotesParams {
    /// Text model tag.
    pub text_model: String,
    /// Vision model to unload before loading the text model (phase C).
    pub vision_model: Option<String>,
    /// Ollama settings.
    pub ollama: OllamaTextConfig,
    /// Maximum tokens generated per call.
    pub num_predict: u32,
    /// Transcript tokens per window.
    pub window_tokens: usize,
    /// Lines repeated between consecutive windows.
    pub window_overlap_lines: usize,
    /// Accept a text model partly in system memory.
    pub allow_spill: bool,
    /// Unload the text model when done.
    pub unload_after: bool,
    /// Title for the notes (default: the board title).
    pub title: Option<String>,
    /// Speaker confidence below which a transcript line is flagged.
    pub low_confidence: f32,
    /// Dropped share of drafted items above which the notes are degraded.
    pub max_drop_rate: f64,
    /// Empty key sections degrade the notes when the transcript is longer than this, seconds.
    pub alarm_min_transcript_s: f64,
    /// Merge transcript lines by one speaker separated by at most this, seconds.
    pub paragraph_gap_s: f64,
    /// Transcript lines sent with a repair request, at most.
    pub repair_context_lines: usize,
    /// Ask for short items in the speakers' own words and strip narrative
    /// prefixes from decisions and tasks.
    #[serde(default)]
    pub concise_items: bool,
    /// Give the model owner-tag facts from the board as candidate decisions and
    /// action items, citable by event or keyframe id.
    #[serde(default)]
    pub board_candidates: bool,
    /// List each window's decision and question cue sentences for the model to
    /// accept or reject.
    #[serde(default)]
    pub cue_candidates: bool,
    /// Most cue sentences listed per window.
    #[serde(default = "default_max_cues")]
    pub max_cue_lines: usize,
    /// Tell the reduce call which open questions the board already has.
    #[serde(default)]
    pub board_questions_in_reduce: bool,
    /// Decisions need a speaker commitment or an owner-tag change.
    #[serde(default)]
    pub precision_guard: bool,
    /// Owner tags valid at the end become "Own <target>" action items when the
    /// transcript corroborates them and they pass validation, unless the owner
    /// already has an action naming the target.
    #[serde(default)]
    pub owner_actions: bool,
    /// Transcript lines this close to an owner tag's appearance can corroborate it, seconds.
    #[serde(default = "default_owner_corroboration_s")]
    pub owner_corroboration_s: f64,
    /// Time windows of this length instead of token windows, seconds.
    #[serde(default)]
    pub window_s: Option<f64>,
    /// Overlap between time windows, seconds.
    #[serde(default = "default_window_overlap_s")]
    pub window_overlap_s: f64,
}

fn default_max_cues() -> usize {
    60
}

fn default_owner_corroboration_s() -> f64 {
    120.0
}

fn default_window_overlap_s() -> f64 {
    60.0
}

impl Default for NotesParams {
    fn default() -> Self {
        Self {
            text_model: "qwen3.6:27b".into(),
            vision_model: Some("qwen2.5vl:7b".into()),
            ollama: OllamaTextConfig::default(),
            num_predict: 4096,
            window_tokens: 6000,
            window_overlap_lines: 4,
            allow_spill: false,
            unload_after: true,
            title: None,
            low_confidence: 0.5,
            max_drop_rate: 0.3,
            alarm_min_transcript_s: 300.0,
            paragraph_gap_s: 2.0,
            repair_context_lines: 160,
            // measured on the reference meeting (see candidates.rs); each can be
            // switched off
            concise_items: true,
            board_candidates: true,
            cue_candidates: true,
            max_cue_lines: default_max_cues(),
            board_questions_in_reduce: true,
            precision_guard: true,
            owner_actions: true,
            owner_corroboration_s: default_owner_corroboration_s(),
            window_s: None,
            window_overlap_s: default_window_overlap_s(),
        }
    }
}

/// Inputs gathered by `plan`.
#[derive(Debug)]
pub struct NotesInput {
    segments: Vec<TranscriptSegment>,
    boards: Vec<BoardStateItem>,
    keyframes: KeyframeTimes,
    speakers: SpeakersDoc,
}

/// `notes`: board state, speakers and transcript to `glassrip.meeting_notes`.
pub struct NotesStage {
    params: NotesParams,
    backend: Arc<dyn TextBackend>,
}

impl NotesStage {
    /// A stage using `backend` for the text model.
    pub fn new(params: NotesParams, backend: Arc<dyn TextBackend>) -> Self {
        Self { params, backend }
    }

    fn model_err(e: impl std::fmt::Display) -> ErrorInfo {
        ErrorInfo::new(ErrorCode::ModelRequest, e.to_string())
    }

    /// Phase C: unload the vision model, load the text model, check placement.
    async fn enter_phase_c(&self) -> Result<Option<LoadedModel>, ErrorInfo> {
        let p = &self.params;
        if let Some(vm) = &p.vision_model {
            let resident = self.backend.loaded().await.map_err(Self::model_err)?;
            if resident.iter().any(|m| same_model(&m.name, vm)) {
                self.backend.unload(vm).await.map_err(Self::model_err)?;
                let deadline = Instant::now() + Duration::from_secs(60);
                loop {
                    let resident = self.backend.loaded().await.map_err(Self::model_err)?;
                    if !resident.iter().any(|m| same_model(&m.name, vm)) {
                        break;
                    }
                    if Instant::now() > deadline {
                        return Err(ErrorInfo::new(
                            ErrorCode::ModelRequest,
                            format!("vision model {vm} still loaded 60 s after keep_alive 0"),
                        ));
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }
        self.backend
            .load(&p.text_model)
            .await
            .map_err(Self::model_err)?;
        let placement = self
            .backend
            .loaded()
            .await
            .map_err(Self::model_err)?
            .into_iter()
            .find(|m| same_model(&m.name, &p.text_model));
        let refused = match &placement {
            Some(m) if m.size_vram < m.size && !p.allow_spill => Err(ErrorInfo::new(
                ErrorCode::ModelRequest,
                format!(
                    "text model {} is only {:.0}% on the GPU ({} of {} bytes); free VRAM or pass allow_spill",
                    p.text_model,
                    100.0 * m.size_vram as f64 / m.size.max(1) as f64,
                    m.size_vram,
                    m.size
                ),
            )),
            None => Err(ErrorInfo::new(ErrorCode::ModelRequest, format!("text model {} did not load", p.text_model))),
            _ => return Ok(placement),
        };
        // a model that spilled to system memory only holds memory: unload it
        let _ = self.backend.unload(&p.text_model).await;
        refused
    }

    /// One schema-constrained call, parsed as a [`Draft`], with one retry on a parse failure.
    async fn call(
        &self,
        req: ChatRequest,
        calls: &mut Vec<CallRecord>,
    ) -> Result<Option<Draft>, ErrorInfo> {
        let mut req = req;
        for attempt in 0..2 {
            let t0 = Instant::now();
            let resp = self
                .backend
                .chat(&self.params.text_model, &req)
                .await
                .map_err(Self::model_err)?;
            let parsed = serde_json::from_str::<Draft>(&resp.content);
            calls.push(CallRecord {
                purpose: if attempt == 0 {
                    req.purpose.clone()
                } else {
                    format!("{} (retry)", req.purpose)
                },
                wall_s: t0.elapsed().as_secs_f64(),
                prompt_tokens: resp.prompt_eval_count,
                eval_tokens: resp.eval_count,
                done_reason: resp.done_reason.clone(),
                parse_error: parsed.as_ref().err().map(|e| e.to_string()),
            });
            match parsed {
                Ok(d) => return Ok(Some(d)),
                Err(e) if attempt == 0 => {
                    if let Some(m) = req.messages.last_mut() {
                        m.content.push_str(&format!(
                            "\n\nYour previous reply was not valid JSON for the schema ({e}). Reply with complete, valid JSON only, and keep it shorter."
                        ));
                    }
                }
                Err(_) => {}
            }
        }
        Ok(None)
    }

    async fn run(&self, input: &NotesInput, ctx: &ItemContext) -> Result<MeetingNotes, ErrorInfo> {
        let started = Instant::now();
        let p = &self.params;
        let doc = &input.speakers;
        let lines = named_lines(&input.segments, doc);
        if lines.is_empty() {
            // no audio stream (or no speech): notes from the board alone
            return Ok(self.board_only(input, started));
        }
        let mut table = AliasTable::default();
        for person in &doc.people {
            let i = table.add_person(&person.display_name);
            for a in &person.aliases {
                table.add_alias(i, a);
            }
        }
        let people = table.people().to_vec();
        let corpus = Corpus::new(&lines, &input.boards, &input.keyframes, table);
        let digest = board_digest(&input.boards, &input.keyframes);

        let placement = self.enter_phase_c().await?;
        let model_digest = self.backend.digest(&p.text_model).await.ok().flatten();

        let mut calls = Vec::new();
        let wins = match p.window_s {
            Some(ws) => windows_by_time(&lines, ws, p.window_overlap_s, p.window_tokens),
            None => windows(&lines, p.window_tokens, p.window_overlap_lines),
        };
        let extras = PromptExtras {
            board_facts: if p.board_candidates {
                board_facts(&input.boards)
            } else {
                String::new()
            },
            cues: if p.cue_candidates {
                wins.iter()
                    .map(|w| cue_lines(w, &lines, p.max_cue_lines))
                    .collect()
            } else {
                Vec::new()
            },
            board_questions: if p.board_questions_in_reduce {
                input
                    .boards
                    .iter()
                    .flat_map(|b| b.stickies.iter())
                    .filter(|s| s.kind == crate::board::StickyKind::Question)
                    .map(|s| s.text.clone())
                    .collect()
            } else {
                Vec::new()
            },
            concise: p.concise_items,
        };
        let opts = CheckOptions {
            board_support: p.board_candidates,
            precision_guard: p.precision_guard,
            strip_prefix: p.concise_items,
        };
        let mut drafts = Vec::new();
        for (i, w) in wins.iter().enumerate() {
            if ctx.cancel_token().is_cancelled() {
                return Err(ErrorInfo::new(
                    ErrorCode::Cancelled,
                    "cancelled during notes",
                ));
            }
            let req = map_request(
                (i, wins.len()),
                w,
                &lines,
                &digest,
                &people,
                p.num_predict,
                &extras,
            );
            if let Some(d) = self.call(req, &mut calls).await? {
                drafts.push(d);
            }
        }
        let draft = if drafts.len() > 1 {
            let req = reduce_request(&drafts, &digest, &people, p.num_predict, &extras);
            match self.call(req, &mut calls).await? {
                Some(d) => d,
                None => {
                    // fall back to the window drafts; Rust merges duplicates below
                    let mut all = Draft::default();
                    for d in &drafts {
                        for s in Section::ALL {
                            all.section_mut(s).extend(d.section(s).iter().cloned());
                        }
                    }
                    all
                }
            }
        } else {
            drafts.into_iter().next().unwrap_or_default()
        };

        let mut checked: Vec<Checked> = Vec::new();
        let mut dropped_items = Vec::new();
        let mut cases = Vec::new();
        for s in Section::ALL {
            for item in draft.section(s) {
                match check_with(s, item, &corpus, &opts) {
                    Ok(c) => checked.push(c),
                    Err(f) if f.fatal => dropped_items.push(dropped(s, item, f.reasons)),
                    Err(f) => cases.push(RepairCase {
                        section: s,
                        item: item.clone(),
                        reasons: f.reasons,
                    }),
                }
            }
        }
        let failed_first = cases.len() + dropped_items.len();
        let mut repaired = 0;
        if !cases.is_empty() {
            let ctx_lines = repair_context(&cases, &lines, p.repair_context_lines);
            let req = repair_request(&cases, &lines, &ctx_lines, &digest, &people, p.num_predict);
            let fixed = self.call(req, &mut calls).await?.unwrap_or_default();
            let mut per_section_left: BTreeMap<Section, usize> = BTreeMap::new();
            for c in &cases {
                *per_section_left.entry(c.section).or_default() += 1;
            }
            let mut still: Vec<(Section, String, Vec<String>)> = Vec::new();
            for s in Section::ALL {
                for item in fixed.section(s) {
                    let left = per_section_left.entry(s).or_default();
                    if *left == 0 {
                        break;
                    }
                    match check_with(s, item, &corpus, &opts) {
                        Ok(c) => {
                            *left -= 1;
                            repaired += 1;
                            checked.push(c);
                        }
                        Err(f) => still.push((s, item.main_text(), f.reasons)),
                    }
                }
            }
            // every failed item not replaced by a valid repair is dropped
            let mut remaining: BTreeMap<Section, usize> = per_section_left;
            for c in cases {
                let left = remaining.entry(c.section).or_default();
                if *left == 0 {
                    continue;
                }
                *left -= 1;
                let mut reasons = c.reasons.clone();
                if let Some((_, _, r)) = still.iter().find(|x| x.0 == c.section) {
                    reasons.push(format!("after repair: {}", r.join("; ")));
                } else {
                    reasons.push("after repair: not returned".into());
                }
                dropped_items.push(dropped(c.section, &c.item, reasons));
            }
        }
        let mut sections = assemble(checked);
        let board_added = merge_board_questions(&mut sections.open_questions, &input.boards);
        // the 6.14 empty-section alarm judges the model's own output, before any
        // action is added from owner tags
        let model_actions_empty = sections.action_items.is_empty();
        let owner_added = if p.owner_actions {
            let ctx = super::candidates::OwnerActionContext {
                lines: &lines,
                corpus: &corpus,
                opts,
                near_s: p.owner_corroboration_s,
            };
            let add = super::candidates::owner_actions(&input.boards, &sections.action_items, &ctx);
            let n = add.len();
            sections.action_items.extend(add);
            for (i, a) in sections.action_items.iter_mut().enumerate() {
                a.id = format!("a{}", i + 1);
            }
            n
        } else {
            0
        };

        let duration_s = input
            .segments
            .iter()
            .map(|s| s.end_s)
            .chain(input.boards.iter().map(|b| b.end_s()))
            .fold(0.0, f64::max);
        let drafted = draft.len();
        let drop_rate = dropped_items.len() as f64 / drafted.max(1) as f64;
        let mut alarm = Vec::new();
        if drop_rate > p.max_drop_rate {
            alarm.push(format!(
                "{:.0}% of drafted items failed validation",
                drop_rate * 100.0
            ));
        }
        if duration_s > p.alarm_min_transcript_s {
            for (name, empty) in [
                ("decisions", sections.decisions.is_empty()),
                ("action items", model_actions_empty),
                ("summary", sections.summary.is_empty()),
            ] {
                if empty {
                    alarm.push(format!("no {name}"));
                }
            }
        }
        let status = if alarm.is_empty() {
            NotesStatus::Ok
        } else {
            NotesStatus::Degraded
        };

        let caveats = caveats(
            &input.segments,
            &lines,
            doc,
            &input.boards,
            p,
            &alarm,
            dropped_items.len(),
        );
        let speakers = doc
            .labels
            .iter()
            .map(|l| SpeakerLine {
                label: l.label.clone(),
                name: l
                    .person_id
                    .as_deref()
                    .and_then(|pid| doc.display_name(pid))
                    .map(str::to_string)
                    .unwrap_or_else(|| match l.status {
                        SpeakerStatus::Noise => "noise".into(),
                        _ => "unresolved".into(),
                    }),
                status: match l.status {
                    SpeakerStatus::Mapped => "mapped",
                    SpeakerStatus::Noise => "noise",
                    SpeakerStatus::Unresolved => "unresolved",
                }
                .into(),
                confidence: l.confidence,
                talk_time_s: l.talk_time_s,
            })
            .collect();
        let presenter = doc
            .summary
            .as_ref()
            .and_then(|s| s.presenter.as_deref())
            .and_then(|pid| doc.display_name(pid))
            .map(str::to_string);
        if p.unload_after {
            // best effort: the notes are complete either way
            let _ = self.backend.unload(&p.text_model).await;
        }
        let items_kept = sections.decisions.len()
            + sections.action_items.len()
            + sections.open_questions.len()
            + sections.timeline.len()
            + sections.summary.len();
        Ok(MeetingNotes {
            title: p.title.clone(),
            duration_s,
            people,
            presenter,
            summary: sections.summary,
            decisions: sections.decisions,
            action_items: sections.action_items,
            open_questions: sections.open_questions,
            timeline: sections.timeline,
            caveats,
            speakers,
            transcript: paragraphs(&lines, p.paragraph_gap_s, p.low_confidence),
            report: NotesReport {
                status,
                model: p.text_model.clone(),
                model_digest,
                windows: wins.len(),
                calls,
                items_drafted: drafted + board_added + owner_added,
                items_kept,
                items_failed_first_pass: failed_first,
                items_repaired: repaired,
                dropped: dropped_items,
                drop_rate,
                placement,
                unloaded_vision_model: p.vision_model.clone(),
                wall_s: started.elapsed().as_secs_f64(),
            },
        })
    }
}

impl NotesStage {
    /// Notes from the board alone (no audio stream, or a transcript with no
    /// words). No model is called: the timeline, summary and open questions come
    /// from the board state, and a visible caveat says why the transcript-based
    /// sections are empty.
    fn board_only(&self, input: &NotesInput, started: Instant) -> MeetingNotes {
        let p = &self.params;
        let mut timeline: Vec<TimelineEntry> = Vec::new();
        let mut summary: Vec<SummaryPoint> = Vec::new();
        for b in &input.boards {
            // one timeline entry per keyframe with changes
            let mut by_kf: Vec<(String, Vec<&crate::board::BoardEvent>)> = Vec::new();
            let mut events: Vec<&crate::board::BoardEvent> =
                b.events.iter().filter(|e| !e.baseline).collect();
            events.sort_by(|x, y| x.t_s.total_cmp(&y.t_s));
            for e in events {
                match by_kf.last_mut() {
                    Some((k, v)) if *k == e.keyframe_id => v.push(e),
                    _ => by_kf.push((e.keyframe_id.clone(), vec![e])),
                }
            }
            for (kf, evs) in by_kf {
                let mut texts: Vec<String> = evs.iter().take(4).map(|e| event_text(e)).collect();
                if evs.len() > 4 {
                    texts.push(format!("and {} more changes", evs.len() - 4));
                }
                let t = evs.first().map_or(0.0, |e| e.t_s);
                timeline.push(TimelineEntry {
                    id: String::new(),
                    t_start_s: t,
                    t_end_s: t,
                    text: sanitize_dashes(&texts.join("; ")),
                    evidence: Evidence {
                        segment_ids: vec![],
                        event_ids: evs.iter().map(|e| e.event_id.clone()).collect(),
                        keyframe_ids: vec![kf],
                    },
                });
            }
            let final_kfs: Vec<String> = b
                .final_window
                .as_ref()
                .map(|w| w.keyframe_ids.clone())
                .filter(|v| !v.is_empty())
                .or_else(|| b.board_keyframes.last().map(|k| vec![k.clone()]))
                .unwrap_or_default();
            let end = b.end_s();
            let nodes = b.final_nodes();
            if !nodes.is_empty() {
                let names: Vec<&str> = nodes.iter().map(|n| n.text.as_str()).collect();
                summary.push(SummaryPoint {
                    id: String::new(),
                    text: sanitize_dashes(&format!(
                        "The final board shows {} components: {}",
                        names.len(),
                        names.join(", ")
                    )),
                    t_start_s: end,
                    t_end_s: end,
                    evidence: Evidence {
                        segment_ids: vec![],
                        event_ids: vec![],
                        keyframe_ids: final_kfs.clone(),
                    },
                });
            }
            let owners = b.current_owners();
            if !owners.is_empty() {
                let list: Vec<String> = owners
                    .iter()
                    .map(|o| {
                        format!(
                            "{} on {}",
                            o.display_name,
                            crate::board::target_text(&o.target)
                        )
                    })
                    .collect();
                let event_ids: Vec<String> = b
                    .events
                    .iter()
                    .filter(|e| {
                        matches!(
                            e.kind,
                            crate::board::EventKind::OwnerAssigned
                                | crate::board::EventKind::OwnerMoved
                        )
                    })
                    .map(|e| e.event_id.clone())
                    .collect();
                summary.push(SummaryPoint {
                    id: String::new(),
                    text: sanitize_dashes(&format!(
                        "Owners on the board at the end: {}",
                        list.join("; ")
                    )),
                    t_start_s: end,
                    t_end_s: end,
                    evidence: Evidence {
                        segment_ids: vec![],
                        event_ids,
                        keyframe_ids: final_kfs.clone(),
                    },
                });
            }
        }
        timeline.sort_by(|a, b| a.t_start_s.total_cmp(&b.t_start_s));
        for (i, t) in timeline.iter_mut().enumerate() {
            t.id = format!("t{}", i + 1);
        }
        for (i, s) in summary.iter_mut().enumerate() {
            s.id = format!("s{}", i + 1);
        }
        let mut open_questions = Vec::new();
        let added = merge_board_questions(&mut open_questions, &input.boards);
        let duration_s = input.boards.iter().map(|b| b.end_s()).fold(0.0, f64::max);
        let mut caveats = vec![Caveat {
            kind: "no_audio".into(),
            text: "No speech was transcribed (the recording has no audio stream or no words were recognized). These notes come from the whiteboard only: decisions, action items and the transcript are not available.".into(),
        }];
        if input.boards.is_empty() {
            caveats.push(Caveat {
                kind: "no_board".into(),
                text: "No whiteboard was read either, so these notes are empty.".into(),
            });
        }
        let items_kept = timeline.len() + summary.len() + open_questions.len();
        MeetingNotes {
            title: p.title.clone(),
            duration_s,
            people: input.speakers.people.clone(),
            presenter: None,
            summary,
            decisions: vec![],
            action_items: vec![],
            open_questions,
            timeline,
            caveats,
            speakers: vec![],
            transcript: vec![],
            report: NotesReport {
                status: NotesStatus::Ok,
                model: p.text_model.clone(),
                model_digest: None,
                windows: 0,
                calls: vec![],
                items_drafted: added,
                items_kept,
                items_failed_first_pass: 0,
                items_repaired: 0,
                dropped: vec![],
                drop_rate: 0.0,
                placement: None,
                unloaded_vision_model: None,
                wall_s: started.elapsed().as_secs_f64(),
            },
        }
    }
}

/// Caveats computed from the inputs (never from model text).
fn caveats(
    segments: &[TranscriptSegment],
    lines: &[NamedLine],
    doc: &SpeakersDoc,
    boards: &[BoardStateItem],
    p: &NotesParams,
    alarm: &[String],
    dropped: usize,
) -> Vec<Caveat> {
    let mut out = Vec::new();
    if !alarm.is_empty() {
        out.push(Caveat {
            kind: "degraded".into(),
            text: format!(
                "These notes are incomplete ({}). Check them against the transcript.",
                alarm.join("; ")
            ),
        });
    }
    let unresolved: Vec<String> = doc
        .labels
        .iter()
        .filter(|l| l.status != SpeakerStatus::Mapped)
        .map(|l| format!("{} ({:.0} s)", l.label, l.talk_time_s))
        .collect();
    if !unresolved.is_empty() {
        out.push(Caveat {
            kind: "speakers".into(),
            text: format!(
                "Diarization labels not matched to a participant: {}.",
                unresolved.join(", ")
            ),
        });
    }
    let low: Vec<&NamedLine> = lines
        .iter()
        .filter(|l| l.speaker_confidence < p.low_confidence)
        .collect();
    let relabeled: Vec<&NamedLine> = lines.iter().filter(|l| l.relabeled).collect();
    if !low.is_empty() || !relabeled.is_empty() {
        let examples: Vec<String> = relabeled
            .iter()
            .take(4)
            .map(|l| {
                let words: Vec<&str> = l.text.split_whitespace().take(6).collect();
                format!(
                    "{} \"{}\" ({})",
                    mmss(l.start_s),
                    words.join(" "),
                    l.speaker
                )
            })
            .collect();
        let mut text = format!(
            "{} of {} transcript lines have a low-confidence speaker (below {:.1}); {} were reassigned from the diarizer's label by on-screen speaker or direct-address cues",
            low.len(),
            lines.len(),
            p.low_confidence,
            relabeled.len()
        );
        if !examples.is_empty() {
            text.push_str(&format!(", for example {}", examples.join(", ")));
        }
        text.push_str(". Flagged lines are marked in the transcript.");
        out.push(Caveat {
            kind: "speakers".into(),
            text,
        });
    }
    let (mut words, mut gap_words, mut gap_s) = (0usize, 0usize, 0.0f64);
    let mut corrections: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut low_p = 0usize;
    for s in segments {
        for w in &s.words {
            words += 1;
            if w.source == glassrip_audio::recluster::Source::GapFill {
                gap_words += 1;
                gap_s += (w.end_s - w.start_s).max(0.0);
            }
            if let Some(raw) = &w.w_raw {
                let clean = |x: &str| x.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
                *corrections.entry((clean(raw), clean(&w.w))).or_default() += 1;
            } else if w.p < 0.3 && w.w.chars().next().is_some_and(char::is_uppercase) {
                low_p += 1;
            }
        }
    }
    if gap_words > 0 {
        let mut text = format!(
            "{:.0}% of words ({gap_words} words, {:.0} s of speech) were attributed by gap filling, where the diarizer marked speech as silence; their speaker is less certain.",
            100.0 * gap_words as f64 / words.max(1) as f64,
            gap_s
        );
        if let Some(g) = doc.summary.as_ref().map(|s| &s.gap_fill) {
            if g.runs_total > 0 {
                text.push_str(&format!(
                    " Checked against the video: {} of {} runs had an on-screen speaker cue; {} agreed, {} disagreed ({} relabeled, {} words).",
                    g.runs_with_cue, g.runs_total, g.runs_agree, g.runs_contradict, g.runs_relabeled, g.words_relabeled
                ));
            }
        }
        out.push(Caveat {
            kind: "gap_fill".into(),
            text,
        });
    }
    if !corrections.is_empty() || low_p > 0 {
        let mut list: Vec<((String, String), usize)> = corrections.into_iter().collect();
        list.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let shown: Vec<String> = list
            .iter()
            .take(8)
            .map(|((r, c), n)| format!("\"{r}\" as \"{c}\" ({n})"))
            .collect();
        let mut text = String::from("The transcript is speech recognition output.");
        if !shown.is_empty() {
            text.push_str(&format!(
                " Vocabulary corrections applied: {}.",
                shown.join(", ")
            ));
        }
        if low_p > 0 {
            text.push_str(&format!(" {low_p} capitalized words were recognized with low confidence and may be misheard names or terms."));
        }
        out.push(Caveat {
            kind: "misheard".into(),
            text,
        });
    }
    if dropped > 0 {
        out.push(Caveat {
            kind: "dropped".into(),
            text: format!("{dropped} drafted items were removed because their citations or quotes could not be verified."),
        });
    }
    if boards.is_empty() {
        out.push(Caveat {
            kind: "no_board".into(),
            text: "No whiteboard was read; the notes come from the transcript only.".into(),
        });
    }
    for c in &mut out {
        c.text = sanitize_dashes(&c.text);
    }
    out
}

impl Stage for NotesStage {
    type Params = NotesParams;
    type Work = Arc<NotesInput>;
    type Output = MeetingNotes;

    fn name(&self) -> &'static str {
        "notes"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: schemas::MEETING_NOTES,
            version: Version::new(1, 0, 0),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![
            InputDecl {
                schema: schemas::BOARD_STATE,
                major: crate::board::BOARD_STATE_MAJOR,
            },
            InputDecl {
                schema: schemas::SPEAKERS,
                major: 1,
            },
            InputDecl {
                schema: schemas::TRANSCRIPT,
                major: 1,
            },
            InputDecl {
                schema: schemas::KEYFRAMES,
                major: KEYFRAMES_MAJOR,
            },
        ]
    }
    fn params(&self) -> &NotesParams {
        &self.params
    }
    fn item_timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(3 * 3600))
    }

    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<Self::Work>>, StageError> {
        let boards: Vec<BoardStateItem> = inputs
            .read_ok::<BoardStateItem>(schemas::BOARD_STATE)?
            .into_iter()
            .map(|(_, b)| b)
            .collect();
        let views: Vec<KeyframeView> = inputs
            .read_ok::<KeyframeView>(schemas::KEYFRAMES)?
            .into_iter()
            .map(|(_, k)| k)
            .collect();
        let keyframes = KeyframeTimes::from_views(&views);
        let speakers = SpeakersDoc::from_records(
            inputs
                .read_ok::<SpeakersRecord>(schemas::SPEAKERS)?
                .into_iter()
                .map(|(_, r)| r),
        );
        let mut segments: Vec<TranscriptSegment> = inputs
            .read_ok::<TranscriptSegment>(schemas::TRANSCRIPT)?
            .into_iter()
            .map(|(_, s)| s)
            .collect();
        segments.sort_by(|a, b| {
            a.start_s
                .total_cmp(&b.start_s)
                .then(a.segment_id.cmp(&b.segment_id))
        });
        Ok(vec![WorkItem {
            id: "meeting_notes".into(),
            work: Arc::new(NotesInput {
                segments,
                boards,
                keyframes,
                speakers,
            }),
        }])
    }

    async fn process(
        &self,
        ctx: &ItemContext,
        work: Self::Work,
    ) -> Result<MeetingNotes, ErrorInfo> {
        let out = self.run(&work, ctx).await;
        if out.is_err() && self.params.unload_after {
            // best effort, as on success: do not leave the text model resident
            let _ = self.backend.unload(&self.params.text_model).await;
        }
        out
    }
}
