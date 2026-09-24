//! Regression gate against a baseline report (spec 9.1).
//!
//! - Any metric whose name ends in `.f1` fails when it drops by more than
//!   `max_drop_points` (default 2) points below the baseline, plus twice the
//!   observed spread for live runs (the larger of the current and baseline
//!   spreads, in points). An F1 present in the baseline but missing now fails.
//! - Any metric whose name ends in `chrome_fp` fails when it rises at all.
//!
//! In replay mode both spreads are 0, so the gate is deterministic in CI.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{read_json, Result};
use crate::report::{GateResult, MetricStat};

/// The part of a report the gate reads.
#[derive(Debug, Clone, Deserialize)]
pub struct Baseline {
    /// Metrics.
    pub metrics: BTreeMap<String, MetricStat>,
}

/// Loads a baseline (any `eval_report.json`).
pub fn load_baseline(path: &Path) -> Result<Baseline> {
    read_json(path)
}

/// Compares current metrics to the baseline; one gate result per checked metric.
pub fn regression_gate(
    current: &BTreeMap<String, MetricStat>,
    baseline: &Baseline,
    max_drop_points: f64,
) -> Vec<GateResult> {
    let mut out = Vec::new();
    for (name, base) in &baseline.metrics {
        if name.ends_with(".f1") {
            let Some(cur) = current.get(name) else {
                out.push(GateResult {
                    name: format!("regression:{name}"),
                    pass: false,
                    detail: "missing in the current run".into(),
                });
                continue;
            };
            let drop = (base.mean - cur.mean) * 100.0;
            let allowed = max_drop_points + 2.0 * cur.spread.max(base.spread) * 100.0;
            out.push(GateResult {
                name: format!("regression:{name}"),
                pass: drop <= allowed + 1e-9,
                detail: format!(
                    "baseline {:.3}, now {:.3}, drop {drop:.2} points (allowed {allowed:.2})",
                    base.mean, cur.mean
                ),
            });
        } else if name.ends_with("chrome_fp") {
            let Some(cur) = current.get(name) else {
                continue;
            };
            out.push(GateResult {
                name: format!("regression:{name}"),
                pass: cur.mean <= base.mean + 1e-9,
                detail: format!("baseline {:.2}, now {:.2}", base.mean, cur.mean),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(mean: f64, spread: f64) -> MetricStat {
        MetricStat {
            mean,
            spread,
            min: mean,
            max: mean,
            runs: 1,
        }
    }

    fn base(pairs: &[(&str, f64, f64)]) -> Baseline {
        Baseline {
            metrics: pairs
                .iter()
                .map(|(k, m, s)| (k.to_string(), stat(*m, *s)))
                .collect(),
        }
    }

    #[test]
    fn f1_drop_threshold() {
        let b = base(&[("board.node.f1", 0.90, 0.0)]);
        // 1.5 points drop: pass
        let cur = BTreeMap::from([("board.node.f1".to_string(), stat(0.885, 0.0))]);
        assert!(regression_gate(&cur, &b, 2.0)[0].pass);
        // 2.5 points drop: fail
        let cur = BTreeMap::from([("board.node.f1".to_string(), stat(0.875, 0.0))]);
        assert!(!regression_gate(&cur, &b, 2.0)[0].pass);
        // live spread 0.01 widens the allowance to 2 + 2*1 = 4 points
        let cur = BTreeMap::from([("board.node.f1".to_string(), stat(0.865, 0.01))]);
        assert!(regression_gate(&cur, &b, 2.0)[0].pass);
        // an improvement always passes
        let cur = BTreeMap::from([("board.node.f1".to_string(), stat(0.95, 0.0))]);
        assert!(regression_gate(&cur, &b, 2.0)[0].pass);
    }

    #[test]
    fn missing_f1_and_chrome_rise_fail() {
        let b = base(&[
            ("board.edge.f1", 0.8, 0.0),
            ("board.chrome_fp", 1.0, 0.0),
            ("screen.accuracy", 0.9, 0.0),
        ]);
        let cur = BTreeMap::from([("board.chrome_fp".to_string(), stat(2.0, 0.0))]);
        let g = regression_gate(&cur, &b, 2.0);
        assert_eq!(g.len(), 2); // accuracy is not gated
        assert!(g.iter().all(|x| !x.pass));
        let cur = BTreeMap::from([
            ("board.chrome_fp".to_string(), stat(0.0, 0.0)),
            ("board.edge.f1".to_string(), stat(0.8, 0.0)),
        ]);
        assert!(regression_gate(&cur, &b, 2.0).iter().all(|x| x.pass));
    }
}
