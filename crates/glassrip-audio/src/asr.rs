//! Speech recognition.
//!
//! The chunking core ([`transcribe_with`]) is independent of the recognizer:
//! a [`SpeechDetector`] finds speech, [`plan_chunks`] groups it into decode
//! chunks, a [`ChunkDecoder`] decodes each chunk with the vocabulary prompt, and
//! chunk-relative token times are moved onto the input timeline and grouped into
//! words. [`Transcriber`] is the whisper.cpp decoder (through `whisper-rs`);
//! [`SileroVad`] and [`EnergyVad`] are detectors.
//!
//! Design notes:
//! - VAD runs here, not inside `whisper_full`. whisper.cpp remaps only segment
//!   times after its internal VAD pass; token and DTW times stay on the
//!   compressed timeline. Decoding each speech chunk separately keeps every
//!   token time on the original timeline.
//! - Chunks are at most about one whisper window long and each is decoded with
//!   the vocabulary prompt, so the vocabulary conditions every window (whisper-rs
//!   0.16 does not expose `carry_initial_prompt`).
//! - Flash attention stays off because it disables DTW token timestamps.

use std::ffi::CStr;
use std::path::{Path, PathBuf};

use whisper_rs::{
    DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters, WhisperState, WhisperVadContext, WhisperVadContextParams,
    WhisperVadParams,
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
    /// Check model files against `models.toml` before loading.
    pub verify_models: bool,
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
            verify_models: true,
        }
    }

    /// The recognizer-independent part of the configuration.
    pub fn chunking(&self) -> ChunkingConfig {
        ChunkingConfig {
            vocabulary: self.vocabulary.clone(),
            max_prompt_tokens: self.max_prompt_tokens,
            carry_previous_words: self.carry_previous_words,
            max_chunk_s: self.max_chunk_s,
            max_merge_gap_s: self.max_merge_gap_s,
        }
    }
}

/// Chunking and prompting settings shared by every decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkingConfig {
    /// Vocabulary terms (names first).
    pub vocabulary: Vec<String>,
    /// Maximum prompt length in tokens.
    pub max_prompt_tokens: usize,
    /// Words of the previous chunk appended to the prompt (0 disables).
    pub carry_previous_words: usize,
    /// Maximum chunk length, seconds.
    pub max_chunk_s: f64,
    /// Speech regions closer than this are decoded together, seconds.
    pub max_merge_gap_s: f64,
}

/// One segment decoded from a chunk; times are relative to the chunk start.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedSegment {
    /// Start, seconds from the chunk start.
    pub start_s: f64,
    /// End, seconds from the chunk start.
    pub end_s: f64,
    /// Non-special tokens with chunk-relative times.
    pub tokens: Vec<RawToken>,
}

/// A speech recognizer that decodes one chunk at a time.
pub trait ChunkDecoder {
    /// Token count of `text` in the recognizer's vocabulary.
    fn count_tokens(&self, text: &str) -> Result<usize>;
    /// Decode `audio` (16 kHz mono) conditioned on `prompt`.
    fn decode(&mut self, audio: &[f32], prompt: &str) -> Result<Vec<DecodedSegment>>;
}

/// Finds speech regions `(start_s, end_s)` in 16 kHz mono audio.
pub trait SpeechDetector {
    /// Speech regions in time order.
    fn detect(&mut self, samples: &[f32]) -> Result<Vec<(f64, f64)>>;
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
    /// Decode chunks `(start_s, end_s)`.
    pub chunk_spans: Vec<(f64, f64)>,
    /// Speech regions found by the detector.
    pub speech_regions: usize,
    /// Tokens in the vocabulary prompt actually used.
    pub prompt_tokens: usize,
    /// Vocabulary terms that fit in the prompt.
    pub prompt_terms: Vec<String>,
    /// Backend device the decoder ran on (empty for decoders that do not say).
    pub backend: String,
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

/// Vocabulary prompt: terms in order until the token budget is used.
///
/// Returns the prompt, the terms that fit, and its token count.
pub fn vocabulary_prompt(
    decoder: &dyn ChunkDecoder,
    vocabulary: &[String],
    max_tokens: usize,
) -> Result<(String, Vec<String>, usize)> {
    let mut used: Vec<String> = Vec::new();
    let mut prompt = String::new();
    let mut n_tokens = 0;
    for term in vocabulary {
        let term = sanitize(term.trim());
        if term.is_empty() {
            continue;
        }
        let candidate = if prompt.is_empty() {
            term.clone()
        } else {
            format!("{prompt}, {term}")
        };
        let n = decoder.count_tokens(&format!("{candidate}."))?;
        if n > max_tokens {
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

/// Detect speech, chunk it, decode every chunk and build words on the input
/// timeline.
pub fn transcribe_with(
    decoder: &mut dyn ChunkDecoder,
    detector: Option<&mut dyn SpeechDetector>,
    samples: &[f32],
    cfg: &ChunkingConfig,
) -> Result<AsrOutput> {
    let sr = f64::from(SAMPLE_RATE);
    let total_s = samples.len() as f64 / sr;
    let regions = match detector {
        Some(d) => d.detect(samples)?,
        None => vec![(0.0, total_s)],
    };
    let chunks = plan_chunks(&regions, cfg.max_chunk_s, cfg.max_merge_gap_s);
    let (vocab_prompt, prompt_terms, prompt_tokens) =
        vocabulary_prompt(decoder, &cfg.vocabulary, cfg.max_prompt_tokens)?;

    let mut segments: Vec<AsrSegment> = Vec::new();
    let mut chunk_spans = Vec::with_capacity(chunks.len());
    for &(cs, ce) in &chunks {
        let a = ((cs * sr).round() as usize).min(samples.len());
        let b = ((ce * sr).round() as usize).min(samples.len());
        if b <= a {
            continue;
        }
        let chunk_start = a as f64 / sr;
        let chunk_end = b as f64 / sr;
        chunk_spans.push((chunk_start, chunk_end));

        let mut prompt = vocab_prompt.clone();
        if cfg.carry_previous_words > 0 {
            let mut tail: Vec<&str> = segments
                .iter()
                .rev()
                .flat_map(|s| s.words.iter().rev())
                .take(cfg.carry_previous_words)
                .map(|w| w.w.as_str())
                .collect();
            if !tail.is_empty() {
                tail.reverse();
                let candidate = format!("{prompt} {}", tail.join(" "));
                if decoder.count_tokens(&candidate)? <= cfg.max_prompt_tokens {
                    prompt = candidate;
                }
            }
        }
        let prompt = sanitize(prompt.trim());

        for seg in decoder.decode(&samples[a..b], &prompt)? {
            let seg_start = chunk_start + seg.start_s;
            let seg_end = (chunk_start + seg.end_s).min(chunk_end);
            if seg_start >= chunk_end {
                continue;
            }
            let tokens: Vec<RawToken> = seg
                .tokens
                .into_iter()
                .map(|t| RawToken {
                    t0_s: chunk_start + t.t0_s,
                    t1_s: chunk_start + t.t1_s,
                    t_dtw_s: t.t_dtw_s.map(|x| chunk_start + x),
                    ..t
                })
                .collect();
            let seg_end = seg_end.max(seg_start);
            let words = tokens_to_words(&tokens, seg_start, seg_end);
            if !words.is_empty() {
                segments.push(AsrSegment {
                    start_s: seg_start,
                    end_s: seg_end,
                    words,
                });
            }
        }
    }
    Ok(AsrOutput {
        segments,
        chunk_spans,
        speech_regions: regions.len(),
        prompt_tokens,
        prompt_terms,
        backend: String::new(),
    })
}

/// Energy-based speech detector (frame RMS against a dBFS threshold).
///
/// Useful without a VAD model and for tests with generated audio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyVad {
    /// Frame length, seconds.
    pub frame_s: f64,
    /// Frames louder than this are speech, dBFS.
    pub threshold_db: f32,
    /// Speech runs shorter than this are dropped, seconds.
    pub min_speech_s: f64,
    /// Silences shorter than this are bridged, seconds.
    pub min_silence_s: f64,
    /// Padding added around speech, seconds.
    pub pad_s: f64,
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self {
            frame_s: 0.02,
            threshold_db: -40.0,
            min_speech_s: 0.2,
            min_silence_s: 0.3,
            pad_s: 0.1,
        }
    }
}

impl SpeechDetector for EnergyVad {
    fn detect(&mut self, samples: &[f32]) -> Result<Vec<(f64, f64)>> {
        let sr = f64::from(SAMPLE_RATE);
        let frame = ((self.frame_s * sr).round() as usize).max(1);
        let total = samples.len() as f64 / sr;
        let mut runs: Vec<(f64, f64)> = Vec::new();
        for (i, chunk) in samples.chunks(frame).enumerate() {
            let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
            let db = 20.0 * rms.max(1e-10).log10();
            if db <= self.threshold_db {
                continue;
            }
            let s = (i * frame) as f64 / sr;
            let e = ((i * frame + chunk.len()) as f64 / sr).min(total);
            match runs.last_mut() {
                Some(r) if s - r.1 < self.min_silence_s => r.1 = e,
                _ => runs.push((s, e)),
            }
        }
        runs.retain(|(s, e)| e - s >= self.min_speech_s);
        let mut out: Vec<(f64, f64)> = Vec::new();
        for (s, e) in runs {
            let (s, e) = ((s - self.pad_s).max(0.0), (e + self.pad_s).min(total));
            match out.last_mut() {
                Some(r) if s <= r.1 => r.1 = r.1.max(e),
                _ => out.push((s, e)),
            }
        }
        Ok(out)
    }
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

/// Silero VAD through whisper.cpp.
pub struct SileroVad {
    ctx: WhisperVadContext,
    settings: VadSettings,
    max_speech_s: f32,
}

impl SileroVad {
    /// Load the VAD model.
    pub fn new(settings: &VadSettings, n_threads: i32, max_speech_s: f64) -> Result<Self> {
        let mut ctx_params = WhisperVadContextParams::new();
        ctx_params.set_n_threads(n_threads);
        ctx_params.set_use_gpu(false);
        let ctx = WhisperVadContext::new(&path_str(&settings.model_path)?, ctx_params)?;
        Ok(Self {
            ctx,
            settings: settings.clone(),
            max_speech_s: max_speech_s as f32,
        })
    }
}

impl SpeechDetector for SileroVad {
    fn detect(&mut self, samples: &[f32]) -> Result<Vec<(f64, f64)>> {
        let total_s = samples.len() as f64 / f64::from(SAMPLE_RATE);
        let mut vp = WhisperVadParams::new();
        vp.set_threshold(self.settings.threshold);
        vp.set_min_speech_duration(self.settings.min_speech_ms);
        vp.set_min_silence_duration(self.settings.min_silence_ms);
        vp.set_speech_pad(self.settings.speech_pad_ms);
        vp.set_max_speech_duration(self.max_speech_s);
        let segs = self.ctx.segments_from_samples(vp, samples)?;
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
}

/// Name of the GPU device whisper.cpp selects for `gpu_device`, if any.
///
/// whisper.cpp takes the `gpu_device`-th device whose type is GPU or integrated
/// GPU; this repeats that selection through the ggml device registry.
fn selected_gpu_device(gpu_device: usize) -> Option<String> {
    use whisper_rs::whisper_rs_sys as sys;
    let mut seen = 0usize;
    // SAFETY: the ggml device registry is initialized by the time a whisper
    // context exists; these calls only read registry entries and return
    // pointers owned by ggml (the name is a NUL-terminated static string).
    unsafe {
        for i in 0..sys::ggml_backend_dev_count() {
            let dev = sys::ggml_backend_dev_get(i);
            if dev.is_null() {
                continue;
            }
            let t = sys::ggml_backend_dev_type(dev);
            if t == sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                || t == sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
            {
                if seen == gpu_device {
                    let name = sys::ggml_backend_dev_name(dev);
                    if name.is_null() {
                        return Some("gpu".into());
                    }
                    return Some(CStr::from_ptr(name).to_string_lossy().into_owned());
                }
                seen += 1;
            }
        }
    }
    None
}

/// Loaded whisper model plus configuration.
pub struct Transcriber {
    ctx: WhisperContext,
    state: WhisperState,
    cfg: AsrConfig,
    kind: AsrModelKind,
    backend: String,
}

impl Transcriber {
    /// Verify (unless disabled) and load the model.
    pub fn new(cfg: AsrConfig) -> Result<Self> {
        if !cfg.model_path.is_file() {
            return Err(AudioError::Model {
                name: cfg.model_path.display().to_string(),
                message: "file not found".into(),
            });
        }
        if cfg.verify_models {
            crate::models::verify_model_file(&cfg.model_path)?;
            if let Some(v) = &cfg.vad {
                crate::models::verify_model_file(&v.model_path)?;
            }
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
        let state = ctx.create_state()?;
        let backend = if cfg.use_gpu {
            selected_gpu_device(0).unwrap_or_else(|| "cpu".into())
        } else {
            "cpu".into()
        };
        Ok(Self {
            ctx,
            state,
            cfg,
            kind,
            backend,
        })
    }

    /// Model family in use.
    pub fn kind(&self) -> AsrModelKind {
        self.kind
    }

    /// Backend device in use (`cpu` or the ggml GPU device name).
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// True when the GPU was requested but whisper.cpp found no GPU device.
    pub fn gpu_fallback(&self) -> bool {
        self.cfg.use_gpu && self.backend == "cpu"
    }

    /// Vocabulary prompt for the configured vocabulary.
    pub fn vocabulary_prompt(&self) -> Result<(String, Vec<String>, usize)> {
        vocabulary_prompt(self, &self.cfg.vocabulary, self.cfg.max_prompt_tokens)
    }

    /// Transcribe 16 kHz mono samples with the configured VAD.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<AsrOutput> {
        let chunking = self.cfg.chunking();
        let mut vad = match &self.cfg.vad {
            Some(v) => Some(SileroVad::new(v, self.cfg.n_threads, self.cfg.max_chunk_s)?),
            None => None,
        };
        let detector = vad.as_mut().map(|v| v as &mut dyn SpeechDetector);
        let mut out = transcribe_with(self, detector, samples, &chunking)?;
        out.backend = self.backend.clone();
        Ok(out)
    }
}

impl ChunkDecoder for Transcriber {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.ctx.tokenize(&sanitize(text), 4096)?.len())
    }

    fn decode(&mut self, audio: &[f32], prompt: &str) -> Result<Vec<DecodedSegment>> {
        let sr = f64::from(SAMPLE_RATE);
        let mut buf = audio.to_vec();
        // whisper.cpp skips inputs shorter than one second; pad with silence
        let min_len = (1.1 * sr) as usize;
        if buf.len() < min_len {
            buf.resize(min_len, 0.0);
        }
        let language = sanitize(&self.cfg.language);
        let beam = i32::try_from(self.cfg.beam_size.max(1)).unwrap_or(5);
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
        let prompt = sanitize(prompt);
        if !prompt.is_empty() {
            params.set_initial_prompt(&prompt);
        }
        self.state.full(params, &buf)?;

        let eot = self.ctx.token_eot();
        let mut out = Vec::new();
        for seg in self.state.as_iter() {
            let mut tokens = Vec::new();
            for i in 0..seg.n_tokens() {
                let Some(tok) = seg.get_token(i) else {
                    continue;
                };
                let data = tok.token_data();
                if data.id >= eot {
                    continue;
                }
                tokens.push(RawToken {
                    bytes: tok.to_bytes()?.to_vec(),
                    p: data.p,
                    t0_s: data.t0 as f64 / 100.0,
                    t1_s: data.t1 as f64 / 100.0,
                    t_dtw_s: (data.t_dtw >= 0).then(|| data.t_dtw as f64 / 100.0),
                });
            }
            out.push(DecodedSegment {
                start_s: seg.start_timestamp() as f64 / 100.0,
                end_s: seg.end_timestamp() as f64 / 100.0,
                tokens,
            });
        }
        Ok(out)
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

    fn tone(samples: &mut [f32], start_s: f64, end_s: f64, amp: f32) {
        let sr = f64::from(SAMPLE_RATE);
        let a = (start_s * sr) as usize;
        let b = (end_s * sr) as usize;
        for (i, x) in samples[a..b].iter_mut().enumerate() {
            *x = amp * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / SAMPLE_RATE as f32).sin();
        }
    }

    #[test]
    fn energy_vad_finds_tone_layout() {
        let mut s = vec![0.0f32; SAMPLE_RATE as usize * 10];
        tone(&mut s, 1.0, 3.0, 0.1);
        tone(&mut s, 3.2, 4.0, 0.1); // 0.2 s pause is bridged
        tone(&mut s, 8.0, 9.0, 0.3);
        tone(&mut s, 9.5, 9.6, 0.3); // isolated 0.1 s blip is dropped
        let r = EnergyVad::default().detect(&s).unwrap();
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert_eq!(r.len(), 2, "{r:?}");
        assert!(close(r[0].0, 0.9) && close(r[0].1, 4.1), "{r:?}");
        assert!(close(r[1].0, 7.9) && close(r[1].1, 9.1), "{r:?}");
    }
}
