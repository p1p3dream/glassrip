//! Word-level speaker assignment.
//!
//! Each word goes to the exclusive turn it overlaps most (a word lying inside a
//! turn, including a zero-length word, counts as fully overlapping); a word with
//! no overlap goes to the nearest turn within `max_gap_s`, otherwise it stays
//! unassigned. Confidence combines:
//! - overlap fraction (share of the word inside the chosen turn);
//! - boundary distance (word midpoint to the nearest edge of its turn, saturating
//!   at `boundary_scale_s`), so words at speaker changes score lower;
//! - embedding similarity of the turn to its speaker centroid, when available.
//!
//! Words placed by gap-fill turns have their confidence scaled and capped below
//! diarizer confidence. Every confidence is clamped to [0, 1].

use crate::recluster::{Source, Turn};

/// Assignment settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssignConfig {
    /// Maximum distance to a turn for words with no overlap, seconds.
    pub max_gap_s: f64,
    /// Boundary distance at which the boundary term reaches 1, seconds.
    pub boundary_scale_s: f64,
    /// Weight of the overlap term.
    pub w_overlap: f32,
    /// Weight of the boundary term.
    pub w_boundary: f32,
    /// Weight of the embedding term.
    pub w_embedding: f32,
    /// Multiplier for words placed by gap-fill turns.
    pub gap_fill_scale: f32,
    /// Upper bound for words placed by gap-fill turns.
    pub gap_fill_cap: f32,
}

impl Default for AssignConfig {
    fn default() -> Self {
        Self {
            max_gap_s: 0.5,
            boundary_scale_s: 0.5,
            w_overlap: 0.5,
            w_boundary: 0.25,
            w_embedding: 0.25,
            gap_fill_scale: 0.6,
            gap_fill_cap: 0.6,
        }
    }
}

/// Result for one word.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Assignment {
    /// Speaker index, if any turn qualified.
    pub speaker: Option<usize>,
    /// Confidence in [0, 1].
    pub conf: f32,
    /// Fraction of the word inside the chosen turn, in [0, 1].
    pub overlap_frac: f32,
    /// Provenance: the chosen turn's source, or `Unassigned`.
    pub source: Source,
    /// Gap fill similarity of the chosen turn.
    pub gap_sim: Option<f32>,
    /// Gap fill margin of the chosen turn.
    pub gap_margin: Option<f32>,
}

fn unit(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn unit64(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn overlap(a0: f64, a1: f64, b0: f64, b1: f64) -> f64 {
    (a1.min(b1) - a0.max(b0)).max(0.0)
}

fn confidence(cfg: &AssignConfig, overlap_frac: f64, boundary: f64, emb: Option<f32>) -> f32 {
    let o = unit64(overlap_frac) as f32;
    let b = unit64(boundary) as f32;
    let (num, den) = match emb {
        Some(s) => (
            cfg.w_overlap * o + cfg.w_boundary * b + cfg.w_embedding * unit(s),
            cfg.w_overlap + cfg.w_boundary + cfg.w_embedding,
        ),
        None => (
            cfg.w_overlap * o + cfg.w_boundary * b,
            cfg.w_overlap + cfg.w_boundary,
        ),
    };
    if den > 0.0 {
        unit(num / den)
    } else {
        0.0
    }
}

fn finish(cfg: &AssignConfig, t: &Turn, conf: f32, overlap_frac: f64) -> Assignment {
    let conf = match t.source {
        Source::GapFill => unit(conf * cfg.gap_fill_scale).min(unit(cfg.gap_fill_cap)),
        _ => unit(conf),
    };
    Assignment {
        speaker: Some(t.speaker),
        conf,
        overlap_frac: unit64(overlap_frac) as f32,
        source: t.source,
        gap_sim: (t.source == Source::GapFill)
            .then_some(t.embedding_sim)
            .flatten(),
        gap_margin: t.gap_margin,
    }
}

/// Assign one word given as `(start_s, end_s)` to exclusive turns sorted by start.
pub fn assign_word(ws: f64, we: f64, turns: &[Turn], cfg: &AssignConfig) -> Assignment {
    let we = we.max(ws);
    let dur = we - ws;
    let lo = turns.partition_point(|t| t.end_s < ws - cfg.max_gap_s);
    let mut best: Option<(f64, usize)> = None; // (overlap fraction, turn)
    let mut nearest: Option<(f64, usize)> = None; // (gap, turn)
    for (i, t) in turns.iter().enumerate().skip(lo) {
        if t.start_s > we + cfg.max_gap_s {
            break;
        }
        let inside = ws >= t.start_s && we <= t.end_s;
        let ov = overlap(ws, we, t.start_s, t.end_s);
        if inside || ov > 0.0 {
            let frac = if dur > 1e-9 { ov / dur } else { 1.0 };
            let frac = if inside { 1.0 } else { frac };
            if best.is_none_or(|(b, _)| frac > b) {
                best = Some((frac, i));
            }
        } else {
            let gap = if t.end_s <= ws {
                ws - t.end_s
            } else {
                t.start_s - we
            }
            .max(0.0);
            if gap <= cfg.max_gap_s && nearest.is_none_or(|(g, _)| gap < g) {
                nearest = Some((gap, i));
            }
        }
    }
    if let Some((frac, i)) = best {
        let t = &turns[i];
        let mid = (ws + we) / 2.0;
        let edge = (mid - t.start_s).min(t.end_s - mid).max(0.0);
        let boundary = edge / cfg.boundary_scale_s.max(1e-6);
        finish(
            cfg,
            t,
            confidence(cfg, frac, boundary, t.embedding_sim),
            frac,
        )
    } else if let Some((gap, i)) = nearest {
        let t = &turns[i];
        let closeness = unit64(1.0 - gap / cfg.max_gap_s.max(1e-6)) as f32;
        finish(
            cfg,
            t,
            confidence(cfg, 0.0, 0.0, t.embedding_sim) * closeness,
            0.0,
        )
    } else {
        Assignment {
            speaker: None,
            conf: 0.0,
            overlap_frac: 0.0,
            source: Source::Unassigned,
            gap_sim: None,
            gap_margin: None,
        }
    }
}

/// Assign words given as `(start_s, end_s)` to exclusive turns sorted by start.
pub fn assign_words(words: &[(f64, f64)], turns: &[Turn], cfg: &AssignConfig) -> Vec<Assignment> {
    words
        .iter()
        .map(|&(ws, we)| assign_word(ws, we, turns, cfg))
        .collect()
}

/// Speaker of the temporally nearest turn to `t` (any distance).
pub fn nearest_turn(turns: &[Turn], t: f64) -> Option<&Turn> {
    let i = turns.partition_point(|x| x.start_s <= t);
    let dist = |x: &Turn| {
        if t < x.start_s {
            x.start_s - t
        } else if t > x.end_s {
            t - x.end_s
        } else {
            0.0
        }
    };
    [i.checked_sub(1), Some(i)]
        .into_iter()
        .flatten()
        .filter_map(|j| turns.get(j))
        .min_by(|a, b| dist(a).total_cmp(&dist(b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(s: f64, e: f64, spk: usize) -> Turn {
        Turn::new(s, e, spk)
    }

    #[test]
    fn picks_max_overlap() {
        let turns = vec![turn(0.0, 1.0, 0), turn(1.0, 3.0, 1)];
        let a = assign_words(&[(0.8, 1.6)], &turns, &AssignConfig::default());
        assert_eq!(a[0].speaker, Some(1));
        assert!((a[0].overlap_frac - 0.75).abs() < 1e-6);
        assert_eq!(a[0].source, Source::Diarizer);
    }

    #[test]
    fn short_word_inside_short_turn_keeps_its_speaker() {
        let turns = vec![turn(0.0, 10.0, 0), turn(10.2, 10.6, 1), turn(11.0, 20.0, 0)];
        let a = assign_words(&[(10.25, 10.55)], &turns, &AssignConfig::default());
        assert_eq!(a[0].speaker, Some(1));
        assert!(a[0].conf > 0.5);
    }

    #[test]
    fn zero_length_word_inside_turn_is_overlapping() {
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 4.0, 1)];
        let a = assign_word(1.0, 1.0, &turns, &AssignConfig::default());
        assert_eq!(a.speaker, Some(0));
        assert_eq!(a.overlap_frac, 1.0);
        assert!(a.conf <= 1.0);
        // zero-length word exactly on a boundary is inside both; the first wins
        let b = assign_word(2.0, 2.0, &turns, &AssignConfig::default());
        assert!(b.speaker.is_some());
        assert!((0.0..=1.0).contains(&b.conf));
    }

    #[test]
    fn nearest_within_gap_else_none() {
        let turns = vec![turn(0.0, 1.0, 0), turn(5.0, 6.0, 1)];
        let a = assign_words(&[(1.3, 1.5), (3.0, 3.2)], &turns, &AssignConfig::default());
        assert_eq!(a[0].speaker, Some(0));
        assert_eq!(a[0].overlap_frac, 0.0);
        assert!(a[0].conf < 0.3);
        assert_eq!(a[1].speaker, None);
        assert_eq!(a[1].source, Source::Unassigned);
    }

    #[test]
    fn boundary_words_score_lower_than_central_words() {
        let turns = vec![turn(0.0, 4.0, 0), turn(4.0, 8.0, 1)];
        let a = assign_words(&[(1.8, 2.2), (3.8, 4.0)], &turns, &AssignConfig::default());
        assert!(a[0].conf > a[1].conf);
    }

    #[test]
    fn embedding_similarity_raises_confidence() {
        let mut good = turn(0.0, 4.0, 0);
        good.embedding_sim = Some(0.9);
        let mut bad = turn(0.0, 4.0, 0);
        bad.embedding_sim = Some(0.1);
        let cfg = AssignConfig::default();
        let g = assign_words(&[(1.0, 1.5)], &[good], &cfg)[0].conf;
        let b = assign_words(&[(1.0, 1.5)], &[bad], &cfg)[0].conf;
        assert!(g > b);
    }

    #[test]
    fn gap_fill_confidence_is_capped_below_diarizer() {
        let cfg = AssignConfig::default();
        let mut d = turn(0.0, 4.0, 0);
        d.embedding_sim = Some(1.0);
        let mut g = d.clone();
        g.source = Source::GapFill;
        g.gap_margin = Some(0.3);
        let a = assign_word(1.5, 2.5, &[d], &cfg);
        let b = assign_word(1.5, 2.5, &[g], &cfg);
        assert!(a.conf > 0.9);
        assert!(b.conf <= 0.6 + 1e-6);
        assert_eq!(b.source, Source::GapFill);
        assert_eq!(b.gap_sim, Some(1.0));
        assert_eq!(b.gap_margin, Some(0.3));
    }

    #[test]
    fn nearest_turn_prefers_closer_side() {
        let turns = vec![turn(0.0, 1.0, 0), turn(5.0, 6.0, 1)];
        assert_eq!(nearest_turn(&turns, 1.2).map(|t| t.speaker), Some(0));
        assert_eq!(nearest_turn(&turns, 4.5).map(|t| t.speaker), Some(1));
        assert_eq!(nearest_turn(&turns, 5.5).map(|t| t.speaker), Some(1));
        assert!(nearest_turn(&[], 1.0).is_none());
    }

    /// Small deterministic generator for randomized layouts.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn property_confidences_stay_in_unit_interval() {
        let mut rng = Lcg(0x5eed);
        let cfg = AssignConfig::default();
        for _case in 0..2000 {
            // random exclusive turns with random gaps, sources and similarities
            let mut turns = Vec::new();
            let mut t = rng.next_f64() * 2.0;
            let n = (rng.next_f64() * 8.0) as usize;
            for k in 0..n {
                let len = rng.next_f64() * 3.0;
                let mut tr = turn(t, t + len, k % 3);
                let r = rng.next_f64();
                tr.embedding_sim = if r < 0.3 {
                    None
                } else {
                    Some((rng.next_f64() * 3.0 - 1.5) as f32)
                };
                if rng.next_f64() < 0.3 {
                    tr.source = Source::GapFill;
                    tr.gap_margin = Some(rng.next_f64() as f32);
                }
                turns.push(tr);
                t += len + rng.next_f64() * 1.5;
            }
            for _ in 0..20 {
                let ws = rng.next_f64() * (t + 2.0) - 1.0;
                let len = if rng.next_f64() < 0.3 {
                    0.0
                } else {
                    rng.next_f64() * 1.2
                };
                // include reversed words (end before start)
                let we = if rng.next_f64() < 0.05 {
                    ws - 0.1
                } else {
                    ws + len
                };
                let a = assign_word(ws, we, &turns, &cfg);
                assert!(
                    (0.0..=1.0).contains(&a.conf),
                    "conf {} for {ws}..{we}",
                    a.conf
                );
                assert!((0.0..=1.0).contains(&a.overlap_frac));
                if a.source == Source::GapFill {
                    assert!(a.conf <= cfg.gap_fill_cap + 1e-6);
                }
                // a word inside a turn must be assigned
                if let Some(tr) = turns
                    .iter()
                    .find(|x| ws >= x.start_s && we.max(ws) <= x.end_s)
                {
                    assert!(a.speaker.is_some(), "inside {:?} but unassigned", tr);
                    assert_eq!(a.overlap_frac, 1.0);
                }
            }
        }
    }
}
