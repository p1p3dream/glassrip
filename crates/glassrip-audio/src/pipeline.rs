//! End-to-end audio branch: extract, ASR and diarization, gap filling,
//! correction and assignment, artifacts.

use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::asr::{AsrConfig, AsrOutput, Transcriber};
use crate::assign::AssignConfig;
use crate::diarize::{diarize, Diarization, DiarizeConfig};
use crate::error::{AudioError, Result};
use crate::extract::{extract_audio, ExtractOptions, SAMPLE_RATE};
use crate::gapfill::{fill_gaps_with, GapFillConfig, GapFillStats, GapSpan, SpanEmbedder};
use crate::recluster::{Source, Turn};
use crate::transcript::build_segments;
use crate::types::{
    Envelope, GapFillParams, InputRef, Producer, SpeakerItem, SpeakerStatus, SpeakersArtifact,
    SpeakersParams, TranscriptArtifact, TranscriptParams, TranscriptSegment, SCHEMA_VERSION,
    SPEAKERS_SCHEMA, TRANSCRIPT_SCHEMA,
};
use crate::vocab::{CorrectionConfig, Vocabulary};

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// ffmpeg and ffprobe.
    pub extract: ExtractOptions,
    /// ASR settings (vocabulary included).
    pub asr: AsrConfig,
    /// Diarization; `None` labels everything `SPEAKER_00` as unassigned.
    pub diarize: Option<DiarizeConfig>,
    /// Vocabulary correction thresholds.
    pub correction: CorrectionConfig,
    /// Word assignment settings.
    pub assign: AssignConfig,
    /// Label words the diarizer missed by embedding; `None` disables.
    pub gap_fill: Option<GapFillConfig>,
    /// Run ASR and diarization at the same time.
    pub concurrent: bool,
    /// Run id; generated when `None`.
    pub run_id: Option<String>,
}

/// Wall time per stage, seconds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct StageTimings {
    /// Input hashing plus ffprobe and ffmpeg decode.
    pub extract_s: f64,
    /// whisper model verification and load.
    pub asr_load_s: f64,
    /// VAD plus decoding.
    pub asr_s: f64,
    /// Diarization including verification, model load and re-clustering.
    pub diarize_s: f64,
    /// Embedding-based labeling of speech the diarizer missed.
    pub gapfill_s: f64,
    /// Correction pass, word assignment and segment assembly.
    pub assign_s: f64,
    /// Whole pipeline.
    pub total_s: f64,
}

/// Word counts by label source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct SourceCounts {
    /// Labeled from diarizer turns.
    pub diarizer: usize,
    /// Labeled by gap filling.
    pub gap_fill: usize,
    /// Placeholder labels only.
    pub unassigned: usize,
}

/// Counters useful for validation reports.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RunStats {
    /// Decoded audio length, seconds.
    pub audio_duration_s: f64,
    /// VAD speech regions.
    pub speech_regions: usize,
    /// Decode chunks.
    pub asr_chunks: usize,
    /// Prompt tokens used.
    pub prompt_tokens: usize,
    /// Vocabulary terms that fit in the prompt.
    pub prompt_terms: Vec<String>,
    /// whisper backend device actually used.
    pub asr_backend: String,
    /// Diarizer clusters before re-clustering.
    pub num_clusters_raw: usize,
    /// Words changed by the correction pass.
    pub corrected_words: usize,
    /// Seconds where the diarizer marked any speaker active.
    pub diar_active_s: f64,
    /// Seconds covered by exclusive turns (after gap filling).
    pub turn_time_s: f64,
    /// Gap filling counters.
    pub gap_fill: GapFillStats,
    /// Words by label source.
    pub words_by_source: SourceCounts,
    /// Problems worth a human look (also printed to stderr).
    pub warnings: Vec<String>,
}

/// Pipeline result.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// `glassrip.transcript`.
    pub transcript: TranscriptArtifact,
    /// `glassrip.speakers`.
    pub speakers: SpeakersArtifact,
    /// Stage timings.
    pub timings: StageTimings,
    /// Counters.
    pub stats: RunStats,
    /// Exclusive speaker turns (times on the video timeline).
    pub turns: Vec<Turn>,
    /// Every uncovered span considered by gap filling, with its outcome.
    pub gap_spans: Vec<GapSpan>,
}

/// Settings for [`assemble`].
#[derive(Debug, Clone)]
pub struct AssembleConfig {
    /// Vocabulary for the correction pass.
    pub vocabulary: Vec<String>,
    /// Correction thresholds.
    pub correction: CorrectionConfig,
    /// Assignment settings.
    pub assign: AssignConfig,
    /// Gap filling; `None` disables it.
    pub gap_fill: Option<GapFillConfig>,
    /// Seconds added to every output time.
    pub timeline_offset_s: f64,
}

/// Output of [`assemble`].
#[derive(Debug, Clone)]
pub struct Assembled {
    /// Transcript segments.
    pub segments: Vec<TranscriptSegment>,
    /// Diarization after gap filling.
    pub diarization: Option<Diarization>,
    /// Gap filling counters.
    pub gap_stats: GapFillStats,
    /// Gap filling span records (audio timeline).
    pub gap_spans: Vec<GapSpan>,
    /// Gap filling wall time, seconds.
    pub gapfill_s: f64,
    /// Correction and assignment wall time, seconds.
    pub assign_s: f64,
}

/// Gap filling, vocabulary correction and word assignment on finished ASR and
/// diarization results.
pub fn assemble(
    asr: &AsrOutput,
    mut diar: Option<Diarization>,
    embedder: Option<&mut dyn SpanEmbedder>,
    samples: &[f32],
    cfg: &AssembleConfig,
) -> Result<Assembled> {
    let t = Instant::now();
    let (gap_stats, gap_spans) = match (diar.as_mut(), embedder, cfg.gap_fill) {
        (Some(d), Some(e), Some(g)) => {
            let mut words: Vec<(f64, f64)> = asr
                .segments
                .iter()
                .flat_map(|s| s.words.iter().map(|w| (w.start_s, w.end_s)))
                .collect();
            words.sort_by(|a, b| a.0.total_cmp(&b.0));
            fill_gaps_with(e, samples, SAMPLE_RATE, d, &words, &g)?
        }
        _ => (GapFillStats::default(), Vec::new()),
    };
    let gapfill_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let vocab = Vocabulary::new(&cfg.vocabulary);
    let segments = build_segments(
        &asr.segments,
        diar.as_ref(),
        &vocab,
        &cfg.correction,
        &cfg.assign,
        cfg.timeline_offset_s,
    );
    Ok(Assembled {
        segments,
        diarization: diar,
        gap_stats,
        gap_spans,
        gapfill_s,
        assign_s: t.elapsed().as_secs_f64(),
    })
}

fn new_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}-{:x}", std::process::id())
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

async fn blake3_file(path: &Path) -> Result<String> {
    let p = path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<String> {
        let f = std::fs::File::open(&p).map_err(|e| AudioError::io(&p, e))?;
        let mut h = blake3::Hasher::new();
        h.update_reader(f).map_err(|e| AudioError::io(&p, e))?;
        Ok(h.finalize().to_hex().to_string())
    })
    .await
    .map_err(|e| AudioError::Task(e.to_string()))?
}

type AsrResult = Result<(AsrOutput, f64, f64)>;

fn run_asr(cfg: AsrConfig, samples: Arc<Vec<f32>>) -> AsrResult {
    let t = Instant::now();
    let mut tr = Transcriber::new(cfg)?;
    let load = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let out = tr.transcribe(&samples)?;
    Ok((out, load, t.elapsed().as_secs_f64()))
}

fn run_diar(
    cfg: Option<DiarizeConfig>,
    samples: Arc<Vec<f32>>,
) -> Result<(Option<Diarization>, f64)> {
    let Some(cfg) = cfg else {
        return Ok((None, 0.0));
    };
    let t = Instant::now();
    let d = diarize(&samples, &cfg)?;
    Ok((Some(d), t.elapsed().as_secs_f64()))
}

#[cfg(feature = "diarize")]
fn make_embedder(cfg: &DiarizeConfig) -> Result<Option<Box<dyn SpanEmbedder>>> {
    Ok(Some(Box::new(crate::gapfill::SpeakrsEmbedder::new(cfg)?)))
}

#[cfg(not(feature = "diarize"))]
fn make_embedder(_cfg: &DiarizeConfig) -> Result<Option<Box<dyn SpanEmbedder>>> {
    Ok(None)
}

/// Run the audio branch on a media file.
pub async fn run(input: &Path, cfg: &PipelineConfig) -> Result<PipelineOutput> {
    let t_total = Instant::now();
    let t = Instant::now();
    let input_hash = blake3_file(input).await?;
    let audio = extract_audio(&cfg.extract, input).await?;
    let extract_s = t.elapsed().as_secs_f64();
    let duration_s = audio.duration_s();
    let offset = audio.timeline_offset_s;
    let samples = Arc::new(audio.samples);

    let asr_cfg = cfg.asr.clone();
    let diar_cfg = cfg.diarize.clone();
    let join_err = |e: tokio::task::JoinError| AudioError::Task(e.to_string());
    let (asr_res, diar_res) = if cfg.concurrent {
        let s1 = Arc::clone(&samples);
        let s2 = Arc::clone(&samples);
        let a = tokio::task::spawn_blocking(move || run_asr(asr_cfg, s1));
        let d = tokio::task::spawn_blocking(move || run_diar(diar_cfg, s2));
        let (a, d) = tokio::join!(a, d);
        (a.map_err(join_err)?, d.map_err(join_err)?)
    } else {
        let s1 = Arc::clone(&samples);
        let a = tokio::task::spawn_blocking(move || run_asr(asr_cfg, s1))
            .await
            .map_err(join_err)?;
        let s2 = Arc::clone(&samples);
        let d = tokio::task::spawn_blocking(move || run_diar(diar_cfg, s2))
            .await
            .map_err(join_err)?;
        (a, d)
    };
    let (asr, asr_load_s, asr_s) = asr_res?;
    let (diar, diarize_s) = diar_res?;

    let mut warnings = Vec::new();
    if cfg.asr.use_gpu && asr.backend == "cpu" {
        let msg = "GPU requested but whisper.cpp found no GPU device; ASR ran on CPU".to_string();
        eprintln!("WARNING: {msg}");
        warnings.push(msg);
    }

    let acfg = AssembleConfig {
        vocabulary: cfg.asr.vocabulary.clone(),
        correction: cfg.correction,
        assign: cfg.assign,
        gap_fill: cfg.gap_fill,
        timeline_offset_s: offset,
    };
    let dcfg = cfg.diarize.clone();
    let s = Arc::clone(&samples);
    let asr_for_assembly = asr.clone();
    let assembled = tokio::task::spawn_blocking(move || -> Result<Assembled> {
        let mut embedder = match (&dcfg, &diar, acfg.gap_fill) {
            (Some(d), Some(_), Some(_)) => make_embedder(d)?,
            _ => None,
        };
        let e = embedder
            .as_mut()
            .map(|b| b.as_mut() as &mut dyn SpanEmbedder);
        assemble(&asr_for_assembly, diar, e, &s, &acfg)
    })
    .await
    .map_err(join_err)??;
    let Assembled {
        segments,
        diarization: diar,
        gap_stats,
        gap_spans,
        gapfill_s,
        assign_s,
    } = assembled;

    let run_id = cfg.run_id.clone().unwrap_or_else(new_run_id);
    let producer = Producer::current();
    let inputs = vec![InputRef {
        path: input.display().to_string(),
        blake3: input_hash,
        schema: None,
        schema_version: None,
    }];
    let all_words = || segments.iter().flat_map(|s| &s.words);
    let corrected_words = all_words().filter(|w| w.w_raw.is_some()).count();
    let count = |src: Source| all_words().filter(|w| w.source == src).count();
    let words_by_source = SourceCounts {
        diarizer: count(Source::Diarizer),
        gap_fill: count(Source::GapFill),
        unassigned: count(Source::Unassigned),
    };

    let gap_params = cfg.gap_fill.map(|g| GapFillParams {
        min_span_s: g.min_span_s,
        min_similarity: g.min_similarity,
        min_margin: g.min_margin,
        conf_cap: cfg.assign.gap_fill_cap,
    });
    let transcript = Envelope {
        schema: TRANSCRIPT_SCHEMA.into(),
        schema_version: SCHEMA_VERSION.into(),
        run_id: run_id.clone(),
        producer: producer.clone(),
        inputs: inputs.clone(),
        params: TranscriptParams {
            asr_model: file_name(&cfg.asr.model_path),
            vocabulary: cfg.asr.vocabulary.clone(),
            language: cfg.asr.language.clone(),
            beam_size: cfg.asr.beam_size,
            vad_model: cfg.asr.vad.as_ref().map(|v| file_name(&v.model_path)),
            diarization: cfg
                .diarize
                .as_ref()
                .map(|d| format!("speakrs-0.5/{}", d.mode.as_str())),
            num_speakers: cfg.diarize.as_ref().and_then(|d| d.num_speakers),
            timeline_offset_s: offset,
            correction_max_p: cfg.correction.max_p,
            correction_max_p_proper_noun: cfg.correction.max_p_proper_noun,
            asr_backend: asr.backend.clone(),
            gap_fill: gap_params,
        },
        items: segments.clone(),
    };

    let (items, num_clusters_raw) = match &diar {
        Some(d) => (
            d.labels
                .iter()
                .enumerate()
                .map(|(i, label)| {
                    let by = |src: Source| {
                        all_words()
                            .filter(|w| &w.speaker_label == label && w.source == src)
                            .count()
                    };
                    SpeakerItem {
                        label: label.clone(),
                        status: SpeakerStatus::Unresolved,
                        person_id: None,
                        confidence: 0.0,
                        evidence: vec![],
                        talk_time_s: d.talk_time_s.get(i).copied().unwrap_or(0.0),
                        talk_time_gap_fill_s: d
                            .turns
                            .iter()
                            .filter(|t| t.speaker == i && t.source == Source::GapFill)
                            .map(|t| t.end_s - t.start_s)
                            .sum(),
                        words_diarizer: by(Source::Diarizer),
                        words_gap_fill: by(Source::GapFill),
                        words_unassigned: by(Source::Unassigned),
                    }
                })
                .collect(),
            d.num_clusters_raw,
        ),
        None => (vec![], 0),
    };
    let speakers = SpeakersArtifact {
        envelope: Envelope {
            schema: SPEAKERS_SCHEMA.into(),
            schema_version: SCHEMA_VERSION.into(),
            run_id,
            producer,
            inputs,
            params: SpeakersParams {
                method: cfg.diarize.as_ref().map_or_else(
                    || "none".into(),
                    |d| format!("speakrs-0.5/{}", d.mode.as_str()),
                ),
                num_speakers_requested: cfg.diarize.as_ref().and_then(|d| d.num_speakers),
                num_clusters_raw,
                num_speakers_found: items.len(),
            },
            items,
        },
        people: vec![],
        notes: Some("labels come from diarization only; speaker naming has not run".into()),
    };

    let shift = |t: Turn| Turn {
        start_s: t.start_s + offset,
        end_s: t.end_s + offset,
        ..t
    };
    Ok(PipelineOutput {
        transcript,
        speakers,
        timings: StageTimings {
            extract_s,
            asr_load_s,
            asr_s,
            diarize_s,
            gapfill_s,
            assign_s,
            total_s: t_total.elapsed().as_secs_f64(),
        },
        stats: RunStats {
            audio_duration_s: duration_s,
            speech_regions: asr.speech_regions,
            asr_chunks: asr.chunk_spans.len(),
            prompt_tokens: asr.prompt_tokens,
            prompt_terms: asr.prompt_terms.clone(),
            asr_backend: asr.backend.clone(),
            num_clusters_raw,
            corrected_words,
            diar_active_s: diar.as_ref().map_or(0.0, |d| d.active_s),
            turn_time_s: diar
                .as_ref()
                .map_or(0.0, |d| d.turns.iter().map(|t| t.end_s - t.start_s).sum()),
            gap_fill: gap_stats,
            words_by_source,
            warnings,
        },
        turns: diar
            .map(|d| d.turns.into_iter().map(shift).collect())
            .unwrap_or_default(),
        gap_spans: gap_spans
            .into_iter()
            .map(|g| GapSpan {
                start_s: g.start_s + offset,
                end_s: g.end_s + offset,
                ..g
            })
            .collect(),
    })
}
