//! Throughput summary for `glassrip eval --bench` (spec 9.3).
//!
//! Live runs measure the wall time of each whiteboard keyframe (classification
//! plus board read) and the elapsed time of the whole suite at the configured
//! concurrency; `per_keyframe_s` is the elapsed time divided by the number of
//! keyframes, the figure the 9.3 target (median at most 6 s with 4 concurrent
//! requests) is about. Replay runs report the recorded wall times.

use serde::{Deserialize, Serialize};

use crate::text::median;

/// Throughput summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchSummary {
    /// `live` or `recorded`.
    pub source: String,
    /// Number of timed keyframes.
    pub keyframes: usize,
    /// Median per-keyframe latency, seconds.
    pub median_s: Option<f64>,
    /// 90th percentile (nearest rank), seconds.
    pub p90_s: Option<f64>,
    /// Mean, seconds.
    pub mean_s: Option<f64>,
    /// Concurrency used.
    pub concurrency: usize,
    /// Elapsed wall time of the timed section, seconds (live only).
    pub elapsed_s: Option<f64>,
    /// Elapsed divided by keyframes (live only).
    pub per_keyframe_s: Option<f64>,
}

/// Nearest-rank percentile of `values` (`q` in 0..=1).
pub fn percentile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let rank = (q.clamp(0.0, 1.0) * v.len() as f64).ceil() as usize;
    Some(v[rank.clamp(1, v.len()) - 1])
}

/// Builds a summary from per-keyframe latencies.
pub fn summarize(
    source: &str,
    latencies: &[f64],
    concurrency: usize,
    elapsed_s: Option<f64>,
) -> BenchSummary {
    let n = latencies.len();
    BenchSummary {
        source: source.to_string(),
        keyframes: n,
        median_s: median(latencies),
        p90_s: percentile(latencies, 0.9),
        mean_s: (n > 0).then(|| latencies.iter().sum::<f64>() / n as f64),
        concurrency,
        elapsed_s,
        per_keyframe_s: elapsed_s.filter(|_| n > 0).map(|e| e / n as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_hand_computed() {
        let s = summarize("live", &[4.0, 2.0, 6.0, 8.0], 4, Some(10.0));
        assert_eq!(s.median_s, Some(5.0));
        // nearest rank: ceil(0.9 * 4) = 4th value = 8
        assert_eq!(s.p90_s, Some(8.0));
        assert_eq!(s.mean_s, Some(5.0));
        assert_eq!(s.per_keyframe_s, Some(2.5));
        let empty = summarize("recorded", &[], 1, None);
        assert_eq!((empty.median_s, empty.per_keyframe_s), (None, None));
    }

    #[test]
    fn percentile_bounds() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 0.0), Some(1.0));
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 1.0), Some(3.0));
        assert_eq!(percentile(&[], 0.5), None);
    }
}
