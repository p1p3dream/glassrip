//! Speech recognition with whisper.cpp through `whisper-rs`.
//!
//! Design notes:
//! - VAD runs here (Silero through whisper.cpp's VAD API), not inside
//!   `whisper_full`. whisper.cpp remaps only segment times after its internal VAD
//!   pass; token and DTW times stay on the compressed timeline. Running VAD
//!   ourselves and decoding each speech chunk separately keeps every token time
//!   on the original timeline.
//! - Chunks are at most about one whisper window long and each is decoded with
//!   `no_context` plus the vocabulary prompt, so the vocabulary conditions every
//!   window (whisper-rs 0.16 does not expose `carry_initial_prompt`).
//! - Flash attention stays off because it disables DTW token timestamps.

use std::path::{Path, PathBuf};

use whisper_rs::{
    DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters, WhisperVadContext, WhisperVadContextParams, WhisperVadParams,
};

use crate::error::{AudioError, Result};
use crate::extract::SAMPLE_RATE;
use crate::words::{tokens_to_words, AsrWord, RawToken};

/// Known whisper model families, used to pick DTW alignment heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrModelKind {
    /// ggml-large-v3-turbo.
    LargeV3Turbo,
    /// ggml-large-v3.
    LargeV3,
    /// Anything else; DTW uses the top text layers.
    Other,
}

impl AsrModelKind {
    /// Guess the model family from a file name.
    pub fn from_path(path: &Path) -> Self {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if name.contains("large-v3-turbo") {
            Self::LargeV3Turbo
        } else if name.contains("large-v3") {
            Self::LargeV3
        } else {
            Self::Other
        }
    }

    fn dtw_mode(self) -> DtwMode<'static> {
        match self {
            Self::LargeV3Turbo => DtwMode::ModelPreset {
                model_preset: DtwModelPreset::LargeV3Turbo,
            },
            Self::LargeV3 => DtwMode::ModelPreset {
                model_preset: DtwModelPreset::LargeV3,
            },
            Self::Other => DtwMode::TopMost { n_top: 2 },
        }
    }
}

/// Silero VAD settings.
#[derive(Debug, Clone)]
pub struct VadSettings {
    /// Path to `ggml-silero-*.bin`.
    pub model_path: PathBuf,
    /// Speech probability threshold.
    pub threshold: f32,
    /// Minimum speech duration, ms.
    pub min_speech_ms: i32,
    /// Minimum silence that splits speech, ms.
    pub min_silence_ms: i32,
    /// Padding added around speech, ms.
    pub speech_pad_ms: i32,
}

impl VadSettings {
    /// Defaults tuned for meeting speech.
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            threshold: 0.5,
            min_speech_ms: 200,
            min_silence_ms: 300,
            speech_pad_ms: 200,
        }
    }
}

/// ASR configuration.
#[derive(Debug, Clone)]
pub struct AsrConfig {
    /// ggml whisper model.
    pub model_path: PathBuf,
    /// Model family; `None` infers it from the file name.
    pub model_kind: Option<AsrModelKind>,
    /// VAD; `None` decodes fixed-length chunks over the whole input.
    pub vad: Option<VadSettings>,
    /// Decoding language.
    pub language: String,
    /// Beam width.
    pub beam_size: u32,
    /// Vocabulary terms (names first), joined into the prompt.
    pub vocabulary: Vec<String>,
    /// Maximum prompt length in tokens.
    pub max_prompt_tokens: usize,
    /// Words of the previous chunk appended to the prompt (0 disables).
    pub carry_previous_words: usize,
    /// CPU threads for whisper.
    pub n_threads: i32,
    /// Use the GPU backend compiled in (CUDA or Metal).
    pub use_gpu: bool,
    /// DTW token timestamps.
    pub dtw: bool,
    /// Maximum chunk length, seconds.
    pub max_chunk_s: f64,
    /// Speech regions closer than this are decoded together, seconds.
    pub max_merge_gap_s: f64,
}

impl AsrConfig {
    /// Defaults for a model path.
    pub fn new(model_path: PathBuf) -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8);
        Self {
            model_path,
            model_kind: None,
            vad: None,
            language: "en".into(),
            beam_size: 5,
            vocabulary: Vec::new(),
            max_prompt_tokens: 200,
            carry_previous_words: 0,
            n_threads: i32::try_from(threads).unwrap_or(4),
            use_gpu: true,
            dtw: true,
            max_chunk_s: 28.0,
            max_merge_gap_s: 2.0,
        }
    }
}

/// One decoded whisper segment with its words, on the input timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct AsrSegment {
    /// Start, seconds.
    pub start_s: f64,
    /// End, seconds.
    pub end_s: f64,
    /// Words.
    pub words: Vec<AsrWord>,
}

/// ASR result.
#[derive(Debug, Clone, PartialEq)]
pub struct AsrOutput {
    /// Segments in time order.
    pub segments: Vec<AsrSegment>,
    /// Number of chunks decoded.
    pub chunks: usize,
    /// Speech regions found by VAD.
    pub speech_regions: usize,
    /// Tokens in the vocabulary prompt actually used.
    pub prompt_tokens: usize,
    /// Vocabulary terms that fit in the prompt.
    pub prompt_terms: Vec<String>,
}

/// Merge speech regions into decode chunks.
///
/// Regions are `(start_s, end_s)` in time order. A chunk grows while the next
/// region starts within `max_gap_s` of the chunk end and the chunk stays within
/// `max_chunk_s`. Regions longer than `max_chunk_s` are split evenly.
pub fn plan_chunks(regions: &[(f64, f64)], max_chunk_s: f64, max_gap_s: f64) -> Vec<(f64, f64)> {
    let max_chunk_s = max_chunk_s.max(1.0);
    let mut pieces = Vec::new();
    for &(s, e) in regions {
        if !(s.is_finite() && e.is_finite()) || e <= s {
            continue;
        }
        let len = e - s;
        if len <= max_chunk_s {
            pieces.push((s, e));
        } else {
            let n = (len / max_chunk_s).ceil();
            let step = len / n;
            let mut t = s;
            while t < e - 1e-9 {
                let end = (t + step).min(e);
                pieces.push((t, end));
                t = end;
            }
        }
    }
    let mut chunks: Vec<(f64, f64)> = Vec::new();
    for (s, e) in pieces {
        match chunks.last_mut() {
            Some(cur) if s - cur.1 <= max_gap_s && e - cur.0 <= max_chunk_s => cur.1 = cur.1.max(e),
            _ => chunks.push((s, e)),
        }
    }
    chunks
}

fn sanitize(text: &str) -> String {
    text.replace('\0', " ")
}

fn path_str(path: &Path) -> Result<String> {
    let s = path
        .to_str()
        .ok_or_else(|| AudioError::InvalidInput(format!("non-UTF-8 path {}", path.display())))?;
    if s.contains('\0') {
        return Err(AudioError::InvalidInput("path contains NUL".into()));
    }
    Ok(s.to_string())
}

/// Loaded whisper model plus configuration.
pub struct Transcriber {
    ctx: WhisperContext,
    cfg: AsrConfig,
    kind: AsrModelKind,
}

impl Transcriber {
    /// Load the model.
    pub fn new(cfg: AsrConfig) -> Result<Self> {
        if !cfg.model_path.is_file() {
            return Err(AudioError::Model {
                name: cfg.model_path.display().to_string(),
                message: "file not found".into(),
            });
        }
        let kind = cfg
            .model_kind
            .unwrap_or_else(|| AsrModelKind::from_path(&cfg.model_path));
        let mut params = WhisperContextParameters::default();
        params.use_gpu(cfg.use_gpu);
        params.flash_attn(false);
        if cfg.dtw {
            params.dtw_parameters(DtwParameters {
                mode: kind.dtw_mode(),
                ..DtwParameters::default()
            });
        }
        let ctx = WhisperContext::new_with_params(&cfg.model_path, params)?;
        Ok(Self { ctx, cfg, kind })
    }

    /// Model family in use.
    pub fn kind(&self) -> AsrModelKind {
        self.kind
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.ctx.tokenize(text, 4096)?.len())
    }

    /// Build the vocabulary prompt: terms in order until the token budget is used.
    pub fn vocabulary_prompt(&self) -> Result<(String, Vec<String>, usize)> {
        let mut used: Vec<String> = Vec::new();
        let mut prompt = String::new();
        let mut n_tokens = 0;
        for term in &self.cfg.vocabulary {
            let term = sanitize(term.trim());
            if term.is_empty() {
                continue;
            }
            let candidate = if prompt.is_empty() {
                term.clone()
            } else {
                format!("{prompt}, {term}")
            };
            let n = self.count_tokens(&candidate)?;
            if n > self.cfg.max_prompt_tokens {
                break;
            }
            prompt = candidate;
            n_tokens = n;
            used.push(term);
        }
        if !prompt.is_empty() {
            prompt.push('.');
        }
        Ok((prompt, used, n_tokens))
    }

    fn speech_regions(&self, samples: &[f32]) -> Result<Vec<(f64, f64)>> {
        let total_s = samples.len() as f64 / f64::from(SAMPLE_RATE);
        let Some(vad) = &self.cfg.vad else {
            return Ok(vec![(0.0, total_s)]);
        };
        let mut ctx_params = WhisperVadContextParams::new();
        ctx_params.set_n_threads(self.cfg.n_threads);
        ctx_params.set_use_gpu(false);
        let mut vctx = WhisperVadContext::new(&path_str(&vad.model_path)?, ctx_params)?;
        let mut vp = WhisperVadParams::new();
        vp.set_threshold(vad.threshold);
        vp.set_min_speech_duration(vad.min_speech_ms);
        vp.set_min_silence_duration(vad.min_silence_ms);
        vp.set_speech_pad(vad.speech_pad_ms);
        vp.set_max_speech_duration(self.cfg.max_chunk_s as f32);
        let segs = vctx.segments_from_samples(vp, samples)?;
        // whisper.cpp reports VAD segment times in centiseconds
        Ok(segs
            .map(|s| {
                (
                    (f64::from(s.start) / 100.0).clamp(0.0, total_s),
                    (f64::from(s.end) / 100.0).clamp(0.0, total_s),
                )
            })
            .filter(|(s, e)| e > s)
            .collect())
    }

    /// Transcribe 16 kHz mono samples.
    pub fn transcribe(&self, samples: &[f32]) -> Result<AsrOutput> {
        let regions = self.speech_regions(samples)?;
        let chunks = plan_chunks(&regions, self.cfg.max_chunk_s, self.cfg.max_merge_gap_s);
        let (vocab_prompt, prompt_terms, prompt_tokens) = self.vocabulary_prompt()?;
        let language = sanitize(&self.cfg.language);
        let beam = i32::try_from(self.cfg.beam_size.max(1)).unwrap_or(5);
        let eot = self.ctx.token_eot();
        let sr = f64::from(SAMPLE_RATE);

        let mut state = self.ctx.create_state()?;
        let mut segments: Vec<AsrSegment> = Vec::new();
        for &(cs, ce) in &chunks {
            let a = ((cs * sr).floor() as usize).min(samples.len());
            let b = ((ce * sr).ceil() as usize).min(samples.len());
            if b <= a {
                continue;
            }
            let mut audio = samples[a..b].to_vec();
            // whisper.cpp skips inputs shorter than one second; pad with silence
            let min_len = (1.1 * sr) as usize;
            if audio.len() < min_len {
                audio.resize(min_len, 0.0);
            }
            let chunk_start = a as f64 / sr;
            let chunk_end = b as f64 / sr;

            let mut prompt = vocab_prompt.clone();
            if self.cfg.carry_previous_words > 0 {
                let tail: Vec<&str> = segments
                    .iter()
                    .rev()
                    .flat_map(|s| s.words.iter().rev())
                    .take(self.cfg.carry_previous_words)
                    .map(|w| w.w.as_str())
                    .collect();
                if !tail.is_empty() {
                    let tail: Vec<&str> = tail.into_iter().rev().collect();
                    let candidate = format!("{prompt} {}", tail.join(" "));
                    if self.count_tokens(&candidate)? <= self.cfg.max_prompt_tokens {
                        prompt = candidate;
                    }
                }
            }
            let prompt = sanitize(prompt.trim());

            let mut params = FullParams::new(SamplingStrategy::BeamSearch {
                beam_size: beam,
                patience: -1.0,
            });
            params.set_language(Some(&language));
            params.set_n_threads(self.cfg.n_threads);
            params.set_no_context(true);
            params.set_token_timestamps(true);
            params.set_split_on_word(true);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_special(false);
            params.set_print_timestamps(false);
            if !prompt.is_empty() {
                params.set_initial_prompt(&prompt);
            }
            state.full(params, &audio)?;

            for seg in state.as_iter() {
                let seg_start = chunk_start + seg.start_timestamp() as f64 / 100.0;
                let seg_end = (chunk_start + seg.end_timestamp() as f64 / 100.0).min(chunk_end);
                if seg_start >= chunk_end {
                    continue;
                }
                let mut tokens = Vec::new();
                for i in 0..seg.n_tokens() {
                    let Some(tok) = seg.get_token(i) else {
                        continue;
                    };
                    let data = tok.token_data();
                    if data.id >= eot {
                        continue;
                    }
                    let bytes = tok.to_bytes()?.to_vec();
                    tokens.push(RawToken {
                        bytes,
                        p: data.p,
                        t0_s: chunk_start + data.t0 as f64 / 100.0,
                        t1_s: chunk_start + data.t1 as f64 / 100.0,
                        t_dtw_s: (data.t_dtw >= 0).then(|| chunk_start + data.t_dtw as f64 / 100.0),
                    });
                }
                let words = tokens_to_words(&tokens, seg_start, seg_end.max(seg_start));
                if words.is_empty() {
                    continue;
                }
                segments.push(AsrSegment {
                    start_s: seg_start,
                    end_s: seg_end.max(seg_start),
                    words,
                });
            }
        }
        Ok(AsrOutput {
            segments,
            chunks: chunks.len(),
            speech_regions: regions.len(),
            prompt_tokens,
            prompt_terms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_kind_from_name() {
        assert_eq!(
            AsrModelKind::from_path(Path::new("/m/ggml-large-v3-turbo.bin")),
            AsrModelKind::LargeV3Turbo
        );
        assert_eq!(
            AsrModelKind::from_path(Path::new("ggml-large-v3.bin")),
            AsrModelKind::LargeV3
        );
        assert_eq!(
            AsrModelKind::from_path(Path::new("ggml-tiny.en.bin")),
            AsrModelKind::Other
        );
    }

    #[test]
    fn chunks_merge_close_regions_within_limit() {
        let r = [(0.0, 5.0), (6.0, 10.0), (20.0, 22.0), (22.5, 40.0)];
        let c = plan_chunks(&r, 28.0, 2.0);
        assert_eq!(c, vec![(0.0, 10.0), (20.0, 40.0)]);
        // a 17.5 s region is split in two at a 15 s cap; the first half joins
        // the preceding short region
        let c = plan_chunks(&r, 15.0, 2.0);
        assert_eq!(c, vec![(0.0, 10.0), (20.0, 31.25), (31.25, 40.0)]);
    }

    #[test]
    fn long_region_is_split_evenly() {
        let c = plan_chunks(&[(0.0, 60.0)], 28.0, 2.0);
        assert_eq!(c.len(), 3);
        assert!(c.iter().all(|(s, e)| e - s <= 28.0 + 1e-9));
        assert!((c[2].1 - 60.0).abs() < 1e-9);
    }

    #[test]
    fn empty_and_degenerate_regions() {
        assert!(plan_chunks(&[], 28.0, 2.0).is_empty());
        assert!(plan_chunks(&[(3.0, 3.0)], 28.0, 2.0).is_empty());
    }
}
