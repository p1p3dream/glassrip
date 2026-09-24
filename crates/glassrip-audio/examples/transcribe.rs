//! Transcribe and diarize a media file, write artifacts, and optionally compare
//! against a baseline transcript.
//!
//! ```text
//! cargo run --release --features cuda --example transcribe -- IN.flac --out OUT \
//!     --vocab "Term A, Term B" --speakers 3 --baseline other.json \
//!     --hotword "Term A" --probe 28.0
//! ```
//!
//! The baseline is a JSON object with `segments: [{start, end, speaker, text}]`.

use std::collections::BTreeSet;
use std::error::Error;
use std::path::{Path, PathBuf};

use clap::Parser;
use glassrip_audio::asr::{AsrConfig, VadSettings};
use glassrip_audio::assign::AssignConfig;
use glassrip_audio::diarize::{DiarizeConfig, DiarizeMode};
use glassrip_audio::extract::ExtractOptions;
use glassrip_audio::gapfill::GapFillConfig;
use glassrip_audio::metrics::{count_term, wer};
use glassrip_audio::models;
use glassrip_audio::pipeline::{run, PipelineConfig};
use glassrip_audio::types::TranscriptSegment;
use glassrip_audio::vocab::CorrectionConfig;
use serde_json::{json, Value};

#[derive(Parser, Debug)]
struct Args {
    /// Input media file.
    input: PathBuf,
    /// Output directory for transcript.json, speakers.json and report.json.
    #[arg(long)]
    out: PathBuf,
    /// Models directory (default: $GLASSRIP_MODELS_DIR or ~/.glassrip/models).
    #[arg(long)]
    models_dir: Option<PathBuf>,
    /// whisper model file name (in the models dir) or path.
    #[arg(long, default_value = "ggml-large-v3-turbo.bin")]
    model: String,
    /// Silero VAD file name or path.
    #[arg(long, default_value = "ggml-silero-v6.2.0.bin")]
    vad: String,
    /// Disable VAD chunking.
    #[arg(long)]
    no_vad: bool,
    /// Comma-separated vocabulary, names first.
    #[arg(long, default_value = "")]
    vocab: String,
    /// Known speaker count.
    #[arg(long)]
    speakers: Option<usize>,
    /// Skip diarization.
    #[arg(long)]
    no_diarize: bool,
    /// Diarization device: cuda, cuda-fast or cpu.
    #[arg(long)]
    diarize_mode: Option<String>,
    /// Run ASR and diarization at the same time.
    #[arg(long)]
    concurrent: bool,
    /// Do not label speech the diarizer missed.
    #[arg(long)]
    no_gap_fill: bool,
    /// Beam width.
    #[arg(long, default_value_t = 5)]
    beam: u32,
    /// Words of the previous chunk appended to the prompt.
    #[arg(long, default_value_t = 0)]
    carry_words: usize,
    /// Skip checking model hashes against models.toml.
    #[arg(long)]
    skip_verify: bool,
    /// Baseline transcript JSON for comparison.
    #[arg(long)]
    baseline: Option<PathBuf>,
    /// Term to count in both transcripts (repeatable).
    #[arg(long)]
    hotword: Vec<String>,
    /// Time in seconds whose speaker label to report (repeatable).
    #[arg(long)]
    probe: Vec<f64>,
}

fn resolve(dir: &Path, name: &str) -> PathBuf {
    let p = PathBuf::from(name);
    if p.is_absolute() || p.exists() {
        p
    } else {
        dir.join(name)
    }
}

struct BaseSeg {
    start: f64,
    end: f64,
    speaker: String,
    text: String,
}

fn load_baseline(path: &Path) -> Result<Vec<BaseSeg>, Box<dyn Error>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let segs = v
        .get("segments")
        .and_then(Value::as_array)
        .ok_or("baseline has no segments array")?;
    Ok(segs
        .iter()
        .map(|s| BaseSeg {
            start: s.get("start").and_then(Value::as_f64).unwrap_or(0.0),
            end: s.get("end").and_then(Value::as_f64).unwrap_or(0.0),
            speaker: s
                .get("speaker")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            text: s
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        })
        .collect())
}

fn at_time(segs: &[TranscriptSegment], t: f64) -> Option<&TranscriptSegment> {
    segs.iter()
        .find(|s| s.start_s <= t && t < s.end_s)
        .or_else(|| {
            segs.iter()
                .filter(|s| (s.start_s - t).abs() <= 2.0)
                .min_by(|a, b| (a.start_s - t).abs().total_cmp(&(b.start_s - t).abs()))
        })
}

fn snippet(s: &str) -> String {
    s.chars().take(80).collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let dir = args
        .models_dir
        .clone()
        .or_else(models::models_dir)
        .ok_or("no models dir; pass --models-dir")?;
    let model_path = resolve(&dir, &args.model);
    let vad_path = resolve(&dir, &args.vad);

    let vocabulary: Vec<String> = args
        .vocab
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let mut asr = AsrConfig::new(model_path);
    asr.vad = (!args.no_vad).then(|| VadSettings::new(vad_path));
    asr.beam_size = args.beam;
    asr.vocabulary = vocabulary;
    asr.carry_previous_words = args.carry_words;
    asr.verify_models = !args.skip_verify;

    let mode = match args.diarize_mode.as_deref() {
        Some("cpu") => DiarizeMode::Cpu,
        Some("cuda") => DiarizeMode::Cuda,
        Some("cuda-fast") => DiarizeMode::CudaFast,
        Some(other) => return Err(format!("unknown diarize mode {other}").into()),
        None if cfg!(feature = "cuda") => DiarizeMode::Cuda,
        None => DiarizeMode::Cpu,
    };
    if !args.no_diarize && cfg!(feature = "cuda") && std::env::var_os("ORT_DYLIB_PATH").is_none() {
        return Err(
            "set ORT_DYLIB_PATH to libonnxruntime.so from an ONNX Runtime GPU build".into(),
        );
    }
    let diarize = (!args.no_diarize).then(|| DiarizeConfig {
        models_dir: dir.join("speakrs"),
        mode,
        num_speakers: args.speakers,
        verify_models: !args.skip_verify,
    });

    let cfg = PipelineConfig {
        extract: ExtractOptions::default(),
        asr,
        diarize,
        correction: CorrectionConfig::default(),
        assign: AssignConfig::default(),
        gap_fill: (!args.no_gap_fill).then(GapFillConfig::default),
        concurrent: args.concurrent,
        run_id: None,
    };
    let out = run(&args.input, &cfg).await?;

    std::fs::create_dir_all(&args.out)?;
    std::fs::write(
        args.out.join("transcript.json"),
        serde_json::to_string_pretty(&out.transcript)?,
    )?;
    std::fs::write(
        args.out.join("speakers.json"),
        serde_json::to_string_pretty(&out.speakers)?,
    )?;

    std::fs::write(
        args.out.join("turns.json"),
        serde_json::to_string(&out.turns)?,
    )?;
    std::fs::write(
        args.out.join("gap_spans.json"),
        serde_json::to_string(&out.gap_spans)?,
    )?;

    let segs = &out.transcript.items;
    let text: String = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let text_raw: String = segs
        .iter()
        .map(|s| s.text_raw.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let labels: BTreeSet<&str> = segs.iter().map(|s| s.speaker_label.as_str()).collect();
    let short = segs.iter().filter(|s| s.end_s - s.start_s < 1.5).count();

    let baseline = match &args.baseline {
        Some(p) => Some(load_baseline(p)?),
        None => None,
    };
    let base_text = baseline.as_ref().map(|b| {
        b.iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    });

    let hotwords: Vec<Value> = args
        .hotword
        .iter()
        .map(|h| {
            json!({
                "term": h,
                "text": count_term(&text, h),
                "text_raw": count_term(&text_raw, h),
                "baseline": base_text.as_deref().map(|b| count_term(b, h)),
            })
        })
        .collect();
    let probes: Vec<Value> = args
        .probe
        .iter()
        .map(|&t| {
            let ours = at_time(segs, t);
            let base = baseline.as_ref().and_then(|b| {
                b.iter()
                    .find(|s| s.start <= t && t < s.end)
                    .map(|s| (s.speaker.clone(), snippet(&s.text)))
            });
            json!({
                "t_s": t,
                "label": ours.map(|s| s.speaker_label.clone()),
                "sources": ours.map(|s| s.words.iter().map(|w| w.source).collect::<Vec<_>>()),
                "speaker_conf": ours.map(|s| s.speaker_conf),
                "start_s": ours.map(|s| s.start_s),
                "end_s": ours.map(|s| s.end_s),
                "text": ours.map(|s| snippet(&s.text)),
                "baseline_label": base.as_ref().map(|b| b.0.clone()),
                "baseline_text": base.map(|b| b.1),
            })
        })
        .collect();
    let comparison = baseline.as_ref().map(|b| {
        let bt = base_text.clone().unwrap_or_default();
        let w = wer(&bt, &text);
        let wr = wer(&bt, &text_raw);
        let b_labels: BTreeSet<&str> = b.iter().map(|s| s.speaker.as_str()).collect();
        json!({
            "baseline_segments": b.len(),
            "baseline_under_1_5s": b.iter().filter(|s| s.end - s.start < 1.5).count(),
            "baseline_labels": b_labels.len(),
            "wer_text_vs_baseline": w.rate(),
            "wer_text_raw_vs_baseline": wr.rate(),
            "ref_words": w.ref_words,
            "hyp_words": w.hyp_words,
        })
    });
    let talk: Vec<Value> = out
        .speakers
        .envelope
        .items
        .iter()
        .map(|i| {
            json!({
                "label": i.label,
                "talk_time_s": i.talk_time_s,
                "talk_time_gap_fill_s": i.talk_time_gap_fill_s,
                "words_diarizer": i.words_diarizer,
                "words_gap_fill": i.words_gap_fill,
                "words_unassigned": i.words_unassigned,
            })
        })
        .collect();
    let report = json!({
        "timings": out.timings,
        "stats": out.stats,
        "segments": segs.len(),
        "segments_under_1_5s": short,
        "distinct_labels": labels.len(),
        "talk_time": talk,
        "words": segs.iter().map(|s| s.words.len()).sum::<usize>(),
        "hotwords": hotwords,
        "probes": probes,
        "comparison": comparison,
    });
    let pretty = serde_json::to_string_pretty(&report)?;
    std::fs::write(args.out.join("report.json"), &pretty)?;
    println!("{pretty}");
    Ok(())
}
