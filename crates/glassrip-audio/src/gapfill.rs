//! Label speech that the diarizer missed.
//!
//! On far-field recordings the segmentation model can mark long stretches of
//! real speech as silence, which leaves ASR words outside every turn. Those
//! words are grouped into spans (split wherever a diarization turn intervenes).
//! Each span is embedded with a [`SpanEmbedder`] (the diarizer's own WeSpeaker
//! model in production) and joins the most similar speaker centroid only when
//! it passes three gates:
//! 1. the uncovered span (before any widening) is at least `min_span_s` long;
//! 2. the best cosine similarity is at least `min_similarity`;
//! 3. the best similarity beats the second best by at least `min_margin`.
//!
//! Spans that fail a gate are recorded with their status and their words stay
//! unassigned. Spans shorter than `min_embed_s` are widened for embedding, but
//! only into uncovered audio, never into another turn.

use serde::Serialize;

use crate::diarize::Diarization;
use crate::error::Result;
use crate::recluster::{cosine, Source, Turn};

/// Gap filling settings.
#[derive(
    Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct GapFillConfig {
    /// Uncovered words closer than this join one span, seconds.
    pub merge_gap_s: f64,
    /// Maximum span length, seconds.
    pub max_span_s: f64,
    /// Minimum uncovered span length (before widening), seconds.
    pub min_span_s: f64,
    /// Spans shorter than this are widened into uncovered audio for embedding.
    pub min_embed_s: f64,
    /// Minimum cosine similarity to the chosen centroid.
    pub min_similarity: f32,
    /// Minimum similarity margin over the second-best centroid.
    pub min_margin: f32,
}

impl Default for GapFillConfig {
    fn default() -> Self {
        Self {
            merge_gap_s: 0.6,
            max_span_s: 8.0,
            min_span_s: 0.4,
            min_embed_s: 1.5,
            min_similarity: 0.5,
            min_margin: 0.05,
        }
    }
}

/// Outcome for one uncovered span.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SpanStatus {
    /// Joined a speaker.
    Assigned,
    /// Shorter than `min_span_s`.
    TooShort,
    /// Best similarity below `min_similarity`.
    LowSimilarity,
    /// Margin over the second best below `min_margin`.
    LowMargin,
    /// No usable embedding (too little audio, or non-finite output).
    NoEmbedding,
}

/// One uncovered span and what happened to it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct GapSpan {
    /// Span start, seconds (audio timeline).
    pub start_s: f64,
    /// Span end, seconds.
    pub end_s: f64,
    /// Outcome.
    pub status: SpanStatus,
    /// Best-matching speaker index, when an embedding was scored.
    pub speaker: Option<usize>,
    /// Best cosine similarity.
    pub similarity: Option<f32>,
    /// Best minus second-best similarity.
    pub margin: Option<f32>,
}

/// Counters for a gap filling pass.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct GapFillStats {
    /// Uncovered spans found.
    pub spans: usize,
    /// Spans that joined a speaker.
    pub labeled: usize,
    /// Spans rejected as too short.
    pub rejected_short: usize,
    /// Spans rejected for low similarity.
    pub rejected_similarity: usize,
    /// Spans rejected for a low margin.
    pub rejected_margin: usize,
    /// Spans without a usable embedding.
    pub no_embedding: usize,
    /// Seconds of turns added.
    pub filled_s: f64,
}

/// Produces a speaker embedding for a stretch of 16 kHz audio.
pub trait SpanEmbedder {
    /// Embedding of `audio`, or `None` when no usable embedding exists.
    fn embed(&mut self, audio: &[f32]) -> Result<Option<Vec<f32>>>;
}

fn overlaps_any(turns: &[Turn], s: f64, e: f64) -> bool {
    let lo = turns.partition_point(|t| t.end_s <= s);
    turns[lo..]
        .iter()
        .take_while(|t| t.start_s < e)
        .any(|t| t.end_s > s && t.start_s < e)
}

/// Spans of consecutive words that overlap no turn.
///
/// `words` are `(start_s, end_s)` in time order; `turns` are exclusive and sorted.
/// A zero-length word lying inside a turn counts as covered.
pub fn uncovered_spans(
    words: &[(f64, f64)],
    turns: &[Turn],
    cfg: &GapFillConfig,
) -> Vec<(f64, f64)> {
    let mut spans: Vec<(f64, f64)> = Vec::new();
    let mut open = false;
    for &(ws, we) in words {
        let we = we.max(ws);
        let covered =
            overlaps_any(turns, ws, we) || turns.iter().any(|t| ws >= t.start_s && we <= t.end_s);
        if covered {
            open = false;
            continue;
        }
        match spans.last_mut() {
            Some(cur)
                if open
                    && ws - cur.1 <= cfg.merge_gap_s
                    && we - cur.0 <= cfg.max_span_s
                    && !overlaps_any(turns, cur.1, ws.max(cur.1)) =>
            {
                cur.1 = cur.1.max(we);
            }
            _ => spans.push((ws, we)),
        }
        open = true;
    }
    spans
}

/// Best speaker, its similarity, and the margin over the second best.
///
/// With a single centroid the margin equals the similarity.
pub fn score_embedding(emb: &[f32], centroids: &[Option<Vec<f32>>]) -> Option<(usize, f32, f32)> {
    let mut sims: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.as_ref().and_then(|c| cosine(emb, c)).map(|s| (i, s)))
        .filter(|(_, s)| s.is_finite())
        .collect();
    sims.sort_by(|a, b| b.1.total_cmp(&a.1));
    let (best_i, best) = *sims.first()?;
    let margin = sims.get(1).map_or(best, |(_, second)| best - second);
    Some((best_i, best, margin))
}

/// Free interval around `span`: from the end of the previous turn to the start
/// of the next one, within `[0, total]`.
pub fn free_bounds(span: (f64, f64), turns: &[Turn], total: f64) -> (f64, f64) {
    let lo = turns
        .iter()
        .filter(|t| t.end_s <= span.0)
        .map(|t| t.end_s)
        .fold(0.0, f64::max);
    let hi = turns
        .iter()
        .filter(|t| t.start_s >= span.1)
        .map(|t| t.start_s)
        .fold(total, f64::min);
    (lo.min(span.0), hi.max(span.1))
}

/// Widen `span` to at least `min_len`, centered, staying within `[lo, hi]`.
pub fn widen(span: (f64, f64), min_len: f64, lo: f64, hi: f64) -> (f64, f64) {
    let (s, e) = span;
    if e - s >= min_len {
        return (s.max(lo), e.min(hi));
    }
    let mid = (s + e) / 2.0;
    let mut a = mid - min_len / 2.0;
    let mut b = mid + min_len / 2.0;
    if a < lo {
        b += lo - a;
        a = lo;
    }
    if b > hi {
        a -= b - hi;
        b = hi;
    }
    (a.max(lo), b.min(hi))
}

/// Parts of `span` not covered by `turns`.
pub fn subtract_turns(span: (f64, f64), turns: &[Turn]) -> Vec<(f64, f64)> {
    let (mut s, e) = span;
    let mut out = Vec::new();
    let lo = turns.partition_point(|t| t.end_s <= s);
    for t in turns[lo..].iter().take_while(|t| t.start_s < e) {
        if t.start_s > s {
            out.push((s, t.start_s.min(e)));
        }
        s = s.max(t.end_s);
        if s >= e {
            break;
        }
    }
    if s < e {
        out.push((s, e));
    }
    out.retain(|(a, b)| b - a > 0.02);
    out
}

/// Add assigned spans as gap-fill turns and update talk time. Returns seconds added.
pub fn apply_spans(diar: &mut Diarization, spans: &[GapSpan]) -> f64 {
    let original = diar.turns.clone();
    let mut added = Vec::new();
    for sp in spans.iter().filter(|s| s.status == SpanStatus::Assigned) {
        let Some(speaker) = sp.speaker else { continue };
        for (s, e) in subtract_turns((sp.start_s, sp.end_s), &original) {
            added.push(Turn {
                start_s: s,
                end_s: e,
                speaker,
                embedding_sim: sp.similarity,
                source: Source::GapFill,
                gap_margin: sp.margin,
            });
        }
    }
    added.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
    let mut prev_end = f64::MIN;
    added.retain_mut(|t| {
        if t.start_s < prev_end {
            t.start_s = prev_end;
        }
        let keep = t.end_s - t.start_s > 0.02;
        if keep {
            prev_end = t.end_s;
        }
        keep
    });
    let mut filled = 0.0;
    for t in &added {
        let d = t.end_s - t.start_s;
        filled += d;
        if let Some(x) = diar.talk_time_s.get_mut(t.speaker) {
            *x += d;
        }
    }
    diar.turns.extend(added);
    diar.turns.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
    filled
}

/// Find uncovered words, embed and gate their spans, and add turns for the
/// spans that pass. Returns counters and one record per span.
pub fn fill_gaps_with(
    embedder: &mut dyn SpanEmbedder,
    samples: &[f32],
    sample_rate: u32,
    diar: &mut Diarization,
    words: &[(f64, f64)],
    cfg: &GapFillConfig,
) -> Result<(GapFillStats, Vec<GapSpan>)> {
    let sr = f64::from(sample_rate);
    let total = samples.len() as f64 / sr;
    let spans = uncovered_spans(words, &diar.turns, cfg);
    let mut out = Vec::with_capacity(spans.len());
    let mut stats = GapFillStats {
        spans: spans.len(),
        ..GapFillStats::default()
    };
    for &(s, e) in &spans {
        let mut rec = GapSpan {
            start_s: s,
            end_s: e,
            status: SpanStatus::TooShort,
            speaker: None,
            similarity: None,
            margin: None,
        };
        if e - s < cfg.min_span_s {
            stats.rejected_short += 1;
            out.push(rec);
            continue;
        }
        let (lo, hi) = free_bounds((s, e), &diar.turns, total);
        let (a, b) = widen((s, e), cfg.min_embed_s, lo, hi);
        let ia = ((a * sr) as usize).min(samples.len());
        let ib = ((b * sr) as usize).min(samples.len());
        let emb = if ib > ia {
            embedder.embed(&samples[ia..ib])?
        } else {
            None
        };
        let scored = emb.and_then(|v| score_embedding(&v, &diar.centroids));
        let Some((spk, sim, margin)) = scored else {
            rec.status = SpanStatus::NoEmbedding;
            stats.no_embedding += 1;
            out.push(rec);
            continue;
        };
        rec.speaker = Some(spk);
        rec.similarity = Some(sim);
        rec.margin = Some(margin);
        rec.status = if sim < cfg.min_similarity {
            stats.rejected_similarity += 1;
            SpanStatus::LowSimilarity
        } else if margin < cfg.min_margin {
            stats.rejected_margin += 1;
            SpanStatus::LowMargin
        } else {
            stats.labeled += 1;
            SpanStatus::Assigned
        };
        out.push(rec);
    }
    stats.filled_s = apply_spans(diar, &out);
    Ok((stats, out))
}

/// WeSpeaker embedder from the speakrs model bundle.
///
/// The span audio is tiled to fill the model's 10 s window so that zero padding
/// does not dilute the pooled statistics.
#[cfg(feature = "diarize")]
pub struct SpeakrsEmbedder {
    model: speakrs::inference::EmbeddingModel,
    tiled: Vec<f32>,
}

#[cfg(feature = "diarize")]
impl SpeakrsEmbedder {
    /// Load the embedding model from the speakrs model directory.
    pub fn new(cfg: &crate::diarize::DiarizeConfig) -> Result<Self> {
        use crate::error::AudioError;
        use speakrs::pipeline::SEGMENTATION_WINDOW_SECONDS;
        let model = speakrs::inference::EmbeddingModel::with_mode(
            cfg.models_dir.join("wespeaker-voxceleb-resnet34.onnx"),
            crate::diarize::execution_mode(cfg.mode),
        )
        .map_err(|e| AudioError::Diarization(e.to_string()))?;
        let window =
            (SEGMENTATION_WINDOW_SECONDS * f64::from(crate::extract::SAMPLE_RATE)) as usize;
        Ok(Self {
            model,
            tiled: vec![0.0; window],
        })
    }
}

#[cfg(feature = "diarize")]
impl SpanEmbedder for SpeakrsEmbedder {
    fn embed(&mut self, audio: &[f32]) -> Result<Option<Vec<f32>>> {
        // at least 0.1 s of audio
        if audio.len() < (crate::extract::SAMPLE_RATE / 10) as usize {
            return Ok(None);
        }
        for (i, x) in self.tiled.iter_mut().enumerate() {
            *x = audio[i % audio.len()];
        }
        let emb = self
            .model
            .embed(&self.tiled)
            .map_err(|e| crate::error::AudioError::Diarization(e.to_string()))?;
        let v: Vec<f32> = emb.iter().copied().collect();
        Ok(v.iter().all(|x| x.is_finite()).then_some(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(s: f64, e: f64, spk: usize) -> Turn {
        Turn::new(s, e, spk)
    }

    fn diar(turns: Vec<Turn>, centroids: Vec<Option<Vec<f32>>>) -> Diarization {
        let n = centroids.len();
        Diarization {
            turns,
            labels: (0..n).map(crate::diarize::speaker_label).collect(),
            talk_time_s: vec![0.0; n],
            num_clusters_raw: n,
            active_s: 0.0,
            centroids,
        }
    }

    /// Returns a fixed embedding and records the audio length it was given.
    struct Fixed(Option<Vec<f32>>, Vec<usize>);
    impl SpanEmbedder for Fixed {
        fn embed(&mut self, audio: &[f32]) -> Result<Option<Vec<f32>>> {
            self.1.push(audio.len());
            Ok(self.0.clone())
        }
    }

    #[test]
    fn spans_group_uncovered_words_and_stop_at_turns() {
        let turns = vec![turn(0.0, 2.0, 0), turn(6.0, 7.0, 1)];
        let words = vec![
            (1.0, 1.5),
            (2.5, 2.9),
            (3.0, 3.4),
            (5.0, 5.4),
            (6.2, 6.5),
            (7.2, 7.5),
        ];
        let s = uncovered_spans(&words, &turns, &GapFillConfig::default());
        assert_eq!(s, vec![(2.5, 3.4), (5.0, 5.4), (7.2, 7.5)]);
    }

    #[test]
    fn zero_length_word_inside_turn_is_covered() {
        let turns = vec![turn(0.0, 2.0, 0)];
        assert!(uncovered_spans(&[(1.0, 1.0)], &turns, &GapFillConfig::default()).is_empty());
    }

    #[test]
    fn score_reports_similarity_and_margin() {
        let c = vec![Some(vec![1.0, 0.0]), Some(vec![0.0, 1.0])];
        let (spk, sim, margin) = score_embedding(&[1.0, 0.0], &c).unwrap();
        assert_eq!(spk, 0);
        assert!((sim - 1.0).abs() < 1e-6);
        assert!((margin - 1.0).abs() < 1e-6);
        assert!(score_embedding(&[1.0, 0.0], &[None, None]).is_none());
    }

    #[test]
    fn gates_reject_short_dissimilar_and_ambiguous_spans() {
        let cents = vec![Some(vec![1.0, 0.0]), Some(vec![0.0, 1.0])];
        let samples = vec![0.1f32; 16_000 * 20];
        let cfg = GapFillConfig::default();
        // short: 0.3 s of uncovered words
        let mut d = diar(vec![], cents.clone());
        let mut e = Fixed(Some(vec![1.0, 0.0]), vec![]);
        let (st, sp) =
            fill_gaps_with(&mut e, &samples, 16_000, &mut d, &[(1.0, 1.3)], &cfg).unwrap();
        assert_eq!(sp[0].status, SpanStatus::TooShort);
        assert_eq!(st.rejected_short, 1);
        assert!(e.1.is_empty(), "short spans are not embedded");
        assert!(d.turns.is_empty());
        // low similarity: best cosine 0.45
        let mut d = diar(vec![], cents.clone());
        let v = vec![0.45f32, -(1.0f32 - 0.45 * 0.45).sqrt()];
        let mut e = Fixed(Some(v), vec![]);
        let (_, sp) =
            fill_gaps_with(&mut e, &samples, 16_000, &mut d, &[(1.0, 2.0)], &cfg).unwrap();
        assert_eq!(sp[0].status, SpanStatus::LowSimilarity);
        assert!(d.turns.is_empty());
        // low margin: equal similarity to both
        let mut d = diar(vec![], cents.clone());
        let mut e = Fixed(Some(vec![0.7, 0.7]), vec![]);
        let (_, sp) =
            fill_gaps_with(&mut e, &samples, 16_000, &mut d, &[(1.0, 2.0)], &cfg).unwrap();
        assert_eq!(sp[0].status, SpanStatus::LowMargin);
        assert!(d.turns.is_empty());
        // no embedding
        let mut d = diar(vec![], cents.clone());
        let mut e = Fixed(None, vec![]);
        let (st, sp) =
            fill_gaps_with(&mut e, &samples, 16_000, &mut d, &[(1.0, 2.0)], &cfg).unwrap();
        assert_eq!(sp[0].status, SpanStatus::NoEmbedding);
        assert_eq!(st.no_embedding, 1);
        // assigned
        let mut d = diar(vec![], cents);
        let mut e = Fixed(Some(vec![0.1, 0.9]), vec![]);
        let (st, sp) =
            fill_gaps_with(&mut e, &samples, 16_000, &mut d, &[(1.0, 2.0)], &cfg).unwrap();
        assert_eq!(sp[0].status, SpanStatus::Assigned);
        assert_eq!(sp[0].speaker, Some(1));
        assert_eq!(st.labeled, 1);
        assert_eq!(d.turns.len(), 1);
        assert_eq!(d.turns[0].source, Source::GapFill);
        assert!(d.turns[0].gap_margin.is_some());
        assert!((st.filled_s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn widening_stays_inside_uncovered_audio() {
        // 0.5 s span between turns ending at 9.8 and starting at 11.0: the
        // 1.5 s embedding window must stay inside [9.8, 11.0]
        let turns = vec![turn(5.0, 9.8, 0), turn(11.0, 15.0, 1)];
        let (lo, hi) = free_bounds((10.0, 10.5), &turns, 20.0);
        assert_eq!((lo, hi), (9.8, 11.0));
        let (a, b) = widen((10.0, 10.5), 1.5, lo, hi);
        assert!(a >= 9.8 - 1e-9 && b <= 11.0 + 1e-9, "{a}..{b}");

        let mut d = diar(turns, vec![Some(vec![1.0, 0.0]), Some(vec![0.0, 1.0])]);
        let samples = vec![0.1f32; 16_000 * 20];
        let mut e = Fixed(Some(vec![1.0, 0.0]), vec![]);
        fill_gaps_with(
            &mut e,
            &samples,
            16_000,
            &mut d,
            &[(10.0, 10.5)],
            &GapFillConfig::default(),
        )
        .unwrap();
        // embedded audio is at most the 1.2 s free gap
        assert!(e.1[0] <= (1.2 * 16_000.0) as usize + 1, "{}", e.1[0]);
    }

    #[test]
    fn widen_centers_and_clamps() {
        let close =
            |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9;
        assert!(close(widen((1.0, 1.2), 1.0, 0.0, 10.0), (0.6, 1.6)));
        assert!(close(widen((0.0, 0.2), 1.0, 0.0, 10.0), (0.0, 1.0)));
        assert!(close(widen((9.9, 10.0), 1.0, 0.0, 10.0), (9.0, 10.0)));
        assert!(close(widen((2.0, 5.0), 1.0, 0.0, 10.0), (2.0, 5.0)));
        assert!(close(widen((2.0, 2.2), 2.0, 1.9, 2.5), (1.9, 2.5)));
    }

    #[test]
    fn subtract_splits_around_turns() {
        let turns = vec![turn(1.0, 2.0, 0), turn(3.0, 4.0, 1)];
        assert_eq!(
            subtract_turns((0.5, 3.5), &turns),
            vec![(0.5, 1.0), (2.0, 3.0)]
        );
        assert_eq!(subtract_turns((4.5, 5.0), &turns), vec![(4.5, 5.0)]);
        assert!(subtract_turns((1.2, 1.8), &turns).is_empty());
    }
}
