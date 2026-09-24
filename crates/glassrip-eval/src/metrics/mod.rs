//! Metrics from spec sections 9.3 and 9.4.
//!
//! Every metric works on plain eval-side types (see [`crate::views`] and
//! [`crate::fixture`]), so each is unit-tested with hand-computed expectations.
//!
//! | Module | Metrics |
//! |---|---|
//! | [`screen`] | screen-type confusion, accuracy, CMS-read-as-whiteboard count |
//! | [`board`] | node / edge / sticky P, R, F1; edge direction, label, style accuracy; sticky CER; owner tags; chrome false positives |
//! | [`owners`] | owner attribution at probe times, owner-move time error |
//! | [`events`] | pan/zoom false change events |
//! | [`notes`] | decision, action-item, open-question P/R/F1; negative action hits |
//! | [`audio`] | hotword WER, distinct speaker-label count |
//! | [`docs`] | body CER, structured field exact match, free-text CER, block F1, table cells, coverage, hallucinated spans |
//! | [`bench`] | throughput summary |

pub mod audio;
pub mod bench;
pub mod board;
pub mod docs;
pub mod events;
pub mod notes;
pub mod owners;
pub mod screen;

use serde::{Deserialize, Serialize};

/// Counts behind precision, recall, and F1.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    /// Matched pairs.
    pub tp: usize,
    /// Predictions without a gold match.
    pub fp: usize,
    /// Gold items without a prediction.
    pub fn_: usize,
}

impl Counts {
    /// Builds counts from the number of matches and the two list sizes.
    pub fn from_matches(matched: usize, n_gold: usize, n_pred: usize) -> Self {
        Self {
            tp: matched,
            fp: n_pred.saturating_sub(matched),
            fn_: n_gold.saturating_sub(matched),
        }
    }

    /// Adds another set of counts (pooling across cases).
    pub fn add(&mut self, other: Counts) {
        self.tp += other.tp;
        self.fp += other.fp;
        self.fn_ += other.fn_;
    }

    /// Precision; 1.0 when nothing was predicted.
    pub fn precision(&self) -> f64 {
        ratio_or_one(self.tp, self.tp + self.fp)
    }

    /// Recall; 1.0 when there is no gold item.
    pub fn recall(&self) -> f64 {
        ratio_or_one(self.tp, self.tp + self.fn_)
    }

    /// Harmonic mean of precision and recall; 0.0 when both are 0.
    pub fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Precision, recall, and F1 together.
    pub fn prf(&self) -> Prf {
        Prf {
            counts: *self,
            precision: self.precision(),
            recall: self.recall(),
            f1: self.f1(),
        }
    }
}

/// Precision, recall, F1 with their counts.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Prf {
    /// Underlying counts.
    pub counts: Counts,
    /// Precision.
    pub precision: f64,
    /// Recall.
    pub recall: f64,
    /// F1.
    pub f1: f64,
}

/// `num / den`, or 1.0 when `den` is 0.
pub fn ratio_or_one(num: usize, den: usize) -> f64 {
    if den == 0 {
        1.0
    } else {
        num as f64 / den as f64
    }
}

/// An accuracy over a denominator (for example matched edges).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tally {
    /// Correct items.
    pub correct: usize,
    /// Scored items.
    pub total: usize,
}

impl Tally {
    /// Records one item.
    pub fn record(&mut self, correct: bool) {
        self.total += 1;
        self.correct += usize::from(correct);
    }

    /// Adds another tally.
    pub fn add(&mut self, other: Tally) {
        self.correct += other.correct;
        self.total += other.total;
    }

    /// Accuracy; `None` when nothing was scored.
    pub fn accuracy(&self) -> Option<f64> {
        (self.total > 0).then(|| self.correct as f64 / self.total as f64)
    }
}

/// Deterministic greedy one-to-one matching.
///
/// `score(g, p)` returns `Some(score)` for an admissible pair. Pairs are taken in
/// order of descending score, then ascending gold index, then ascending
/// prediction index; each gold and prediction is used at most once. Returns
/// `(gold_index, pred_index)` pairs sorted by gold index.
pub fn greedy_match(
    n_gold: usize,
    n_pred: usize,
    mut score: impl FnMut(usize, usize) -> Option<f64>,
) -> Vec<(usize, usize)> {
    let mut candidates = Vec::new();
    for g in 0..n_gold {
        for p in 0..n_pred {
            if let Some(s) = score(g, p) {
                candidates.push((s, g, p));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut used_g = vec![false; n_gold];
    let mut used_p = vec![false; n_pred];
    let mut out = Vec::new();
    for (_, g, p) in candidates {
        if !used_g[g] && !used_p[p] {
            used_g[g] = true;
            used_p[p] = true;
            out.push((g, p));
        }
    }
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prf_hand_computed() {
        // 3 gold, 4 pred, 2 matched: P = 2/4, R = 2/3, F1 = 2*(.5*.6667)/(1.1667) = 4/7
        let c = Counts::from_matches(2, 3, 4);
        assert_eq!(
            c,
            Counts {
                tp: 2,
                fp: 2,
                fn_: 1
            }
        );
        assert!((c.precision() - 0.5).abs() < 1e-12);
        assert!((c.recall() - 2.0 / 3.0).abs() < 1e-12);
        assert!((c.f1() - 4.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn prf_edge_cases() {
        let empty = Counts::default();
        assert_eq!(
            (empty.precision(), empty.recall(), empty.f1()),
            (1.0, 1.0, 1.0)
        );
        let no_pred = Counts::from_matches(0, 2, 0);
        assert_eq!(
            (no_pred.precision(), no_pred.recall(), no_pred.f1()),
            (1.0, 0.0, 0.0)
        );
        let no_gold = Counts::from_matches(0, 0, 3);
        assert_eq!(
            (no_gold.precision(), no_gold.recall(), no_gold.f1()),
            (0.0, 1.0, 0.0)
        );
    }

    #[test]
    fn greedy_prefers_higher_scores() {
        // gold 0 scores 0.9 with pred 0 and 0.95 with pred 1; gold 1 only with pred 1 (0.99).
        let s = [[Some(0.9), Some(0.95)], [None, Some(0.99)]];
        let m = greedy_match(2, 2, |g, p| s[g][p]);
        assert_eq!(m, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn greedy_ties_are_deterministic() {
        let m = greedy_match(2, 2, |_, _| Some(1.0));
        assert_eq!(m, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn tally_accuracy() {
        let mut t = Tally::default();
        assert_eq!(t.accuracy(), None);
        t.record(true);
        t.record(false);
        t.record(true);
        assert!((t.accuracy().unwrap() - 2.0 / 3.0).abs() < 1e-12);
    }
}
