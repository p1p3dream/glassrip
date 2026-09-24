//! Word-level speaker assignment.
//!
//! Each word goes to the exclusive turn it overlaps most; a word with no overlap
//! goes to the nearest turn within `max_gap_s`, otherwise it stays unassigned.
//! Confidence combines:
//! - overlap fraction (share of the word inside the chosen turn);
//! - boundary distance (word midpoint to the nearest edge of its turn, saturating
//!   at `boundary_scale_s`), so words at speaker changes score lower;
//! - embedding similarity of the turn to its speaker centroid, when available.

use crate::recluster::Turn;

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
}

impl Default for AssignConfig {
    fn default() -> Self {
        Self {
            max_gap_s: 0.5,
            boundary_scale_s: 0.5,
            w_overlap: 0.5,
            w_boundary: 0.25,
            w_embedding: 0.25,
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
    /// Fraction of the word inside the chosen turn.
    pub overlap_frac: f32,
}

fn overlap(a0: f64, a1: f64, b0: f64, b1: f64) -> f64 {
    (a1.min(b1) - a0.max(b0)).max(0.0)
}

fn confidence(cfg: &AssignConfig, overlap_frac: f64, boundary: f64, emb: Option<f32>) -> f32 {
    let o = overlap_frac.clamp(0.0, 1.0) as f32;
    let b = boundary.clamp(0.0, 1.0) as f32;
    match emb {
        Some(s) => {
            let e = s.clamp(0.0, 1.0);
            let w = cfg.w_overlap + cfg.w_boundary + cfg.w_embedding;
            if w <= 0.0 {
                return 0.0;
            }
            ((cfg.w_overlap * o + cfg.w_boundary * b + cfg.w_embedding * e) / w).clamp(0.0, 1.0)
        }
        None => {
            let w = cfg.w_overlap + cfg.w_boundary;
            if w <= 0.0 {
                return 0.0;
            }
            ((cfg.w_overlap * o + cfg.w_boundary * b) / w).clamp(0.0, 1.0)
        }
    }
}

/// Assign words given as `(start_s, end_s)` to exclusive turns sorted by start.
pub fn assign_words(words: &[(f64, f64)], turns: &[Turn], cfg: &AssignConfig) -> Vec<Assignment> {
    words
        .iter()
        .map(|&(ws, we)| {
            let we = we.max(ws);
            let dur = (we - ws).max(1e-3);
            // first turn that could matter
            let lo = turns.partition_point(|t| t.end_s < ws - cfg.max_gap_s);
            let mut best_overlap: Option<(f64, usize)> = None;
            let mut nearest: Option<(f64, usize)> = None;
            for (i, t) in turns.iter().enumerate().skip(lo) {
                if t.start_s > we + cfg.max_gap_s {
                    break;
                }
                let ov = overlap(ws, we, t.start_s, t.end_s);
                if ov > 0.0 {
                    if best_overlap.is_none_or(|(b, _)| ov > b) {
                        best_overlap = Some((ov, i));
                    }
                } else {
                    let gap = if t.end_s <= ws { ws - t.end_s } else { t.start_s - we };
                    if gap <= cfg.max_gap_s && nearest.is_none_or(|(g, _)| gap < g) {
                        nearest = Some((gap, i));
                    }
                }
            }
            if let Some((ov, i)) = best_overlap {
                let t = &turns[i];
                let mid = (ws + we) / 2.0;
                let edge = (mid - t.start_s).min(t.end_s - mid).max(0.0);
                let frac = ov / dur;
                Assignment {
                    speaker: Some(t.speaker),
                    conf: confidence(cfg, frac, edge / cfg.boundary_scale_s, t.embedding_sim),
                    overlap_frac: frac.clamp(0.0, 1.0) as f32,
                }
            } else if let Some((gap, i)) = nearest {
                let t = &turns[i];
                let closeness = 1.0 - gap / cfg.max_gap_s.max(1e-6);
                Assignment {
                    speaker: Some(t.speaker),
                    conf: confidence(cfg, 0.0, 0.0, t.embedding_sim) * closeness as f32,
                    overlap_frac: 0.0,
                }
            } else {
                Assignment {
                    speaker: None,
                    conf: 0.0,
                    overlap_frac: 0.0,
                }
            }
        })
        .collect()
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
    fn picks_max_overlap() {
        let turns = vec![turn(0.0, 1.0, 0), turn(1.0, 3.0, 1)];
        let a = assign_words(&[(0.8, 1.6)], &turns, &AssignConfig::default());
        assert_eq!(a[0].speaker, Some(1));
        assert!((a[0].overlap_frac - 0.75).abs() < 1e-6);
    }

    #[test]
    fn short_word_inside_short_turn_keeps_its_speaker() {
        // a brief interjection inside a pause of a long speaker
        let turns = vec![turn(0.0, 10.0, 0), turn(10.2, 10.6, 1), turn(11.0, 20.0, 0)];
        let a = assign_words(&[(10.25, 10.55)], &turns, &AssignConfig::default());
        assert_eq!(a[0].speaker, Some(1));
        assert!(a[0].conf > 0.5);
    }

    #[test]
    fn nearest_within_gap_else_none() {
        let turns = vec![turn(0.0, 1.0, 0), turn(5.0, 6.0, 1)];
        let a = assign_words(&[(1.3, 1.5), (3.0, 3.2)], &turns, &AssignConfig::default());
        assert_eq!(a[0].speaker, Some(0));
        assert_eq!(a[0].overlap_frac, 0.0);
        assert!(a[0].conf < 0.3);
        assert_eq!(a[1].speaker, None);
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
}
