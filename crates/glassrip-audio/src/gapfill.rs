//! Label speech that the diarizer missed.
//!
//! On far-field recordings the segmentation model can mark long stretches of
//! real speech as silence, which leaves ASR words outside every turn. Those
//! words are grouped into spans (split wherever a diarization turn intervenes),
//! each span is embedded with the same WeSpeaker model the diarizer uses, and
//! the span joins the speaker whose centroid is most similar. The span's turn
//! carries a margin-weighted similarity, so the per-word confidence reflects
//! how clearly one speaker won.

use serde::Serialize;

use crate::diarize::{DiarizeConfig, Diarization};
use crate::error::Result;
use crate::recluster::{cosine, Turn};

/// Gap filling settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GapFillConfig {
    /// Uncovered words closer than this join one span, seconds.
    pub merge_gap_s: f64,
    /// Maximum span length, seconds.
    pub max_span_s: f64,
    /// Spans shorter than this are widened (centered) before embedding, seconds.
    pub min_embed_s: f64,
    /// Similarity margin (best minus second best) that earns full confidence.
    pub full_margin: f32,
}

impl Default for GapFillConfig {
    fn default() -> Self {
        Self {
            merge_gap_s: 0.6,
            max_span_s: 8.0,
            min_embed_s: 1.5,
            full_margin: 0.2,
        }
    }
}

/// What gap filling did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct GapFillStats {
    /// Uncovered spans found.
    pub spans: usize,
    /// Spans that received a speaker.
    pub labeled: usize,
    /// Seconds of turns added.
    pub filled_s: f64,
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
pub fn uncovered_spans(words: &[(f64, f64)], turns: &[Turn], cfg: &GapFillConfig) -> Vec<(f64, f64)> {
    let mut spans: Vec<(f64, f64)> = Vec::new();
    let mut open = false;
    for &(ws, we) in words {
        let we = we.max(ws);
        if overlaps_any(turns, ws, we) {
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

/// Pick the speaker for an embedding and a margin-weighted score in [0, 1].
pub fn label_embedding(
    emb: &[f32],
    centroids: &[Option<Vec<f32>>],
    full_margin: f32,
) -> Option<(usize, f32)> {
    let mut sims: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.as_ref().and_then(|c| cosine(emb, c)).map(|s| (i, s)))
        .collect();
    sims.sort_by(|a, b| b.1.total_cmp(&a.1));
    let (best_i, best) = *sims.first()?;
    let margin_factor = match sims.get(1) {
        Some((_, second)) => ((best - second) / full_margin.max(1e-6)).clamp(0.2, 1.0),
        None => 1.0,
    };
    Some((best_i, (best.max(0.0) * margin_factor).clamp(0.0, 1.0)))
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

/// Add labeled spans as turns and update talk time. Returns seconds added.
pub fn apply_labels(diar: &mut Diarization, labeled: &[((f64, f64), usize, f32)]) -> f64 {
    let original = diar.turns.clone();
    let mut added = Vec::new();
    for &(span, speaker, score) in labeled {
        for (s, e) in subtract_turns(span, &original) {
            added.push(Turn {
                start_s: s,
                end_s: e,
                speaker,
                embedding_sim: Some(score),
            });
        }
    }
    added.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
    // spans come from monotonic words but may touch; keep them exclusive
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

/// Widen a span to at least `min_len` seconds, centered, within `[0, total]`.
pub fn widen(span: (f64, f64), min_len: f64, total: f64) -> (f64, f64) {
    let (s, e) = span;
    if e - s >= min_len {
        return (s.max(0.0), e.min(total));
    }
    let mid = (s + e) / 2.0;
    let mut a = mid - min_len / 2.0;
    let mut b = mid + min_len / 2.0;
    if a < 0.0 {
        b -= a;
        a = 0.0;
    }
    if b > total {
        a -= b - total;
        b = total;
    }
    (a.max(0.0), b)
}

/// Embed each span with the diarizer's WeSpeaker model.
///
/// The span audio is tiled to fill the model's 10 s window so that zero padding
/// does not dilute the pooled statistics.
#[cfg(feature = "diarize")]
pub fn embed_spans(
    samples: &[f32],
    spans: &[(f64, f64)],
    cfg: &DiarizeConfig,
    min_embed_s: f64,
) -> Result<Vec<Option<Vec<f32>>>> {
    use crate::error::AudioError;
    use crate::extract::SAMPLE_RATE;
    use speakrs::inference::EmbeddingModel;
    use speakrs::pipeline::SEGMENTATION_WINDOW_SECONDS;

    let sr = f64::from(SAMPLE_RATE);
    let mut model = EmbeddingModel::with_mode(
        cfg.models_dir.join("wespeaker-voxceleb-resnet34.onnx"),
        crate::diarize::execution_mode(cfg.mode),
    )
    .map_err(|e| AudioError::Diarization(e.to_string()))?;
    let window = (SEGMENTATION_WINDOW_SECONDS * sr) as usize;
    let total = samples.len() as f64 / sr;
    let mut out = Vec::with_capacity(spans.len());
    let mut tiled = vec![0.0f32; window];
    for &span in spans {
        let (s, e) = widen(span, min_embed_s, total);
        let a = ((s * sr) as usize).min(samples.len());
        let b = ((e * sr) as usize).min(samples.len());
        if b <= a + (sr * 0.1) as usize {
            out.push(None);
            continue;
        }
        let src = &samples[a..b];
        for (i, x) in tiled.iter_mut().enumerate() {
            *x = src[i % src.len()];
        }
        let emb = model
            .embed(&tiled)
            .map_err(|e| AudioError::Diarization(e.to_string()))?;
        let v: Vec<f32> = emb.iter().copied().collect();
        out.push(v.iter().all(|x| x.is_finite()).then_some(v));
    }
    Ok(out)
}

/// Stub used when the crate is built without the `diarize` feature.
#[cfg(not(feature = "diarize"))]
pub fn embed_spans(
    _samples: &[f32],
    _spans: &[(f64, f64)],
    _cfg: &DiarizeConfig,
    _min_embed_s: f64,
) -> Result<Vec<Option<Vec<f32>>>> {
    Err(crate::error::AudioError::FeatureDisabled("diarize"))
}

/// Find uncovered words, embed their spans and add turns for them.
pub fn fill_gaps(
    samples: &[f32],
    diar: &mut Diarization,
    words: &[(f64, f64)],
    dcfg: &DiarizeConfig,
    gcfg: &GapFillConfig,
) -> Result<GapFillStats> {
    let spans = uncovered_spans(words, &diar.turns, gcfg);
    if spans.is_empty() || diar.centroids.iter().all(Option::is_none) {
        return Ok(GapFillStats {
            spans: spans.len(),
            ..GapFillStats::default()
        });
    }
    let embs = embed_spans(samples, &spans, dcfg, gcfg.min_embed_s)?;
    let labeled: Vec<((f64, f64), usize, f32)> = spans
        .iter()
        .zip(&embs)
        .filter_map(|(&span, e)| {
            e.as_ref()
                .and_then(|e| label_embedding(e, &diar.centroids, gcfg.full_margin))
                .map(|(spk, score)| (span, spk, score))
        })
        .collect();
    let filled_s = apply_labels(diar, &labeled);
    Ok(GapFillStats {
        spans: spans.len(),
        labeled: labeled.len(),
        filled_s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(s: f64, e: f64, spk: usize) -> Turn {
        Turn {
            start_s: s,
            end_s: e,
            speaker: spk,
            embedding_sim: None,
        }
    }

    #[test]
    fn spans_group_uncovered_words_and_stop_at_turns() {
        let turns = vec![turn(0.0, 2.0, 0), turn(6.0, 7.0, 1)];
        let words = vec![
            (1.0, 1.5),  // covered
            (2.5, 2.9),  // uncovered, starts span
            (3.0, 3.4),  // joins
            (5.0, 5.4),  // gap 1.6 s: new span
            (6.2, 6.5),  // covered
            (7.2, 7.5),  // uncovered, new span after the turn
        ];
        let s = uncovered_spans(&words, &turns, &GapFillConfig::default());
        assert_eq!(s, vec![(2.5, 3.4), (5.0, 5.4), (7.2, 7.5)]);
    }

    #[test]
    fn span_length_is_capped() {
        let words: Vec<(f64, f64)> = (0..20).map(|i| (i as f64 * 0.5, i as f64 * 0.5 + 0.4)).collect();
        let cfg = GapFillConfig {
            max_span_s: 3.0,
            ..GapFillConfig::default()
        };
        let s = uncovered_spans(&words, &[], &cfg);
        assert!(s.len() >= 3);
        assert!(s.iter().all(|(a, b)| b - a <= 3.0 + 1e-9));
    }

    #[test]
    fn label_prefers_closest_centroid_and_scales_by_margin() {
        let c = vec![Some(vec![1.0, 0.0]), Some(vec![0.0, 1.0])];
        let (spk, clear) = label_embedding(&[0.9, 0.1], &c, 0.2).unwrap();
        assert_eq!(spk, 0);
        let (_, unclear) = label_embedding(&[0.7, 0.69], &c, 0.2).unwrap();
        assert!(clear > unclear);
        assert!(label_embedding(&[1.0, 0.0], &[None, None], 0.2).is_none());
    }

    #[test]
    fn subtract_splits_around_turns() {
        let turns = vec![turn(1.0, 2.0, 0), turn(3.0, 4.0, 1)];
        assert_eq!(subtract_turns((0.5, 3.5), &turns), vec![(0.5, 1.0), (2.0, 3.0)]);
        assert_eq!(subtract_turns((4.5, 5.0), &turns), vec![(4.5, 5.0)]);
        assert!(subtract_turns((1.2, 1.8), &turns).is_empty());
    }

    #[test]
    fn apply_keeps_turns_exclusive_and_sorted() {
        let mut d = Diarization {
            turns: vec![turn(0.0, 1.0, 0)],
            labels: vec!["SPEAKER_00".into(), "SPEAKER_01".into()],
            talk_time_s: vec![1.0, 0.0],
            num_clusters_raw: 2,
            active_s: 1.0,
            centroids: vec![None, None],
        };
        let filled = apply_labels(&mut d, &[((0.5, 2.0), 1, 0.8), ((1.8, 2.5), 1, 0.7)]);
        assert!((filled - 1.5).abs() < 1e-9, "filled {filled}");
        for w in d.turns.windows(2) {
            assert!(w[0].end_s <= w[1].start_s + 1e-9);
        }
        assert!((d.talk_time_s[1] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn widen_centers_and_clamps() {
        let close = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9;
        assert!(close(widen((1.0, 1.2), 1.0, 10.0), (0.6, 1.6)));
        assert!(close(widen((0.0, 0.2), 1.0, 10.0), (0.0, 1.0)));
        assert!(close(widen((9.9, 10.0), 1.0, 10.0), (9.0, 10.0)));
        assert!(close(widen((2.0, 5.0), 1.0, 10.0), (2.0, 5.0)));
    }
}
