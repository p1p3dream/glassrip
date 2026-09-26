//! Regression gate against a baseline report (spec 9.1).
//!
//! Every metric present in the baseline is gated by its [`Rule`]:
//!
//! | Rule | Metrics | Fails when |
//! |---|---|---|
//! | [`Rule::HigherBetter`] | P/R/F1, accuracies, core recall, coverage, attribution | it drops by more than `max_drop_points` (default 2) points, plus twice the spread for live runs |
//! | [`Rule::LowerBetterRate`] | CER, WER | it rises by more than `max_drop_points` points, plus twice the spread |
//! | [`Rule::LowerBetterCount`] | chrome false positives, false change events, misses, hits, and other counts | it rises at all (beyond twice the spread for live runs) |
//! | [`Rule::LowerBetterSeconds`] | owner-move time error | it rises by more than 2 s plus twice the spread |
//! | [`Rule::Info`] | case counts, throughput, raw label counts | never (reported only) |
//!
//! A gated metric present in the baseline but missing from the current run
//! fails, whatever its rule. A metric with no rule fails too, so a new metric
//! must be classified before it can enter a baseline. The spread used is the
//! larger of the current and baseline spreads; in replay mode both are 0, so the
//! gate is deterministic in CI.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{read_json, Result};
use crate::report::{GateResult, MetricStat};

/// Allowed rise of a seconds metric, before spread.
pub const SECONDS_TOLERANCE: f64 = 2.0;

/// How a metric is gated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// Higher is better; drop measured in points.
    HigherBetter,
    /// Lower is better; rise measured in points.
    LowerBetterRate,
    /// Lower is better; any rise fails.
    LowerBetterCount,
    /// Lower is better; seconds with a fixed tolerance.
    LowerBetterSeconds,
    /// Not gated.
    Info,
}

/// The gate rule of a metric name; `None` when the metric is unclassified.
pub fn rule_for(name: &str) -> Option<Rule> {
    // `audio.speaker_label_error` is retired: it meant raw diarizer labels, then
    // mapped identities, so a baseline under that key cannot be compared. Its
    // meanings live on as `audio.diarizer_label_error` and
    // `audio.speaker_identity_error`.
    const INFO: &[&str] = &[
        "cases",
        "screen.unconfirmed_labels",
        "audio.speaker_labels",
        "audio.speaker_identities",
        "audio.speaker_label_error",
    ];
    if INFO.contains(&name) || name.starts_with("bench.") {
        return Some(Rule::Info);
    }
    let ends = |suffixes: &[&str]| suffixes.iter().any(|s| name.ends_with(s));
    if ends(&[
        ".f1",
        ".precision",
        ".recall",
        "accuracy",
        "core_recall",
        "coverage_min",
        "trap_coverage",
        "attribution",
    ]) {
        return Some(Rule::HigherBetter);
    }
    if ends(&[
        ".cer",
        "_cer",
        "cer_page_median",
        "cer_page_max",
        "cer_issue_median",
        "_wer",
    ]) {
        return Some(Rule::LowerBetterRate);
    }
    if ends(&["_s"]) {
        return Some(Rule::LowerBetterSeconds);
    }
    if ends(&[
        "chrome_fp",
        "chrome_fp_readings",
        ".fp",
        "cms_as_whiteboard",
        "missed",
        "false_change",
        "uncertain",
        "unresolved",
        "extra",
        "negative_hits",
        "negative_action_hits",
        "diarizer_label_error",
        "speaker_identity_error",
        "hallucinated_spans",
        "truncated_cells",
        "pages_missing",
    ]) {
        return Some(Rule::LowerBetterCount);
    }
    None
}

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

/// Compares current metrics to the baseline; one gate result per gated metric.
pub fn regression_gate(
    current: &BTreeMap<String, MetricStat>,
    baseline: &Baseline,
    max_drop_points: f64,
) -> Vec<GateResult> {
    let mut out = Vec::new();
    for (name, base) in &baseline.metrics {
        let gate_name = format!("regression:{name}");
        let Some(rule) = rule_for(name) else {
            out.push(GateResult {
                name: gate_name,
                pass: false,
                detail: "metric has no gate rule; classify it in gate::rule_for".into(),
            });
            continue;
        };
        if rule == Rule::Info {
            continue;
        }
        let Some(cur) = current.get(name) else {
            out.push(GateResult {
                name: gate_name,
                pass: false,
                detail: format!("missing in the current run (baseline {:.3})", base.mean),
            });
            continue;
        };
        let spread = cur.spread.max(base.spread);
        let (pass, detail) = match rule {
            Rule::HigherBetter => {
                let drop = (base.mean - cur.mean) * 100.0;
                let allowed = max_drop_points + 2.0 * spread * 100.0;
                (
                    drop <= allowed + 1e-9,
                    format!(
                        "baseline {:.3}, now {:.3}, drop {drop:.2} points (allowed {allowed:.2})",
                        base.mean, cur.mean
                    ),
                )
            }
            Rule::LowerBetterRate => {
                let rise = (cur.mean - base.mean) * 100.0;
                let allowed = max_drop_points + 2.0 * spread * 100.0;
                (
                    rise <= allowed + 1e-9,
                    format!(
                        "baseline {:.3}, now {:.3}, rise {rise:.2} points (allowed {allowed:.2})",
                        base.mean, cur.mean
                    ),
                )
            }
            Rule::LowerBetterCount => {
                let allowed = base.mean + 2.0 * spread;
                (
                    cur.mean <= allowed + 1e-9,
                    format!(
                        "baseline {:.2}, now {:.2} (allowed {allowed:.2})",
                        base.mean, cur.mean
                    ),
                )
            }
            Rule::LowerBetterSeconds => {
                let allowed = base.mean + SECONDS_TOLERANCE + 2.0 * spread;
                (
                    cur.mean <= allowed + 1e-9,
                    format!(
                        "baseline {:.2} s, now {:.2} s (allowed {allowed:.2})",
                        base.mean, cur.mean
                    ),
                )
            }
            Rule::Info => (true, String::new()),
        };
        out.push(GateResult {
            name: gate_name,
            pass,
            detail,
        });
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

    fn cur(pairs: &[(&str, f64, f64)]) -> BTreeMap<String, MetricStat> {
        base(pairs).metrics
    }

    #[test]
    fn f1_drop_threshold() {
        let b = base(&[("board.node.f1", 0.90, 0.0)]);
        // 1.5 points drop: pass
        assert!(regression_gate(&cur(&[("board.node.f1", 0.885, 0.0)]), &b, 2.0)[0].pass);
        // 2.5 points drop: fail
        assert!(!regression_gate(&cur(&[("board.node.f1", 0.875, 0.0)]), &b, 2.0)[0].pass);
        // live spread 0.01 widens the allowance to 2 + 2*1 = 4 points
        assert!(regression_gate(&cur(&[("board.node.f1", 0.865, 0.01)]), &b, 2.0)[0].pass);
        // an improvement always passes
        assert!(regression_gate(&cur(&[("board.node.f1", 0.95, 0.0)]), &b, 2.0)[0].pass);
    }

    #[test]
    fn missing_gated_metric_fails_for_every_rule() {
        let b = base(&[
            ("board.edge.f1", 0.8, 0.0),
            ("board.chrome_fp", 1.0, 0.0),
            ("screen.accuracy", 0.9, 0.0),
            ("board.sticky.cer", 0.01, 0.0),
            ("owners.move_error_max_s", 3.0, 0.0),
            ("cases", 11.0, 0.0),
        ]);
        let g = regression_gate(&BTreeMap::new(), &b, 2.0);
        // five gated metrics, all missing; `cases` is informational
        assert_eq!(g.len(), 5);
        assert!(g.iter().all(|x| !x.pass && x.detail.starts_with("missing")));
    }

    #[test]
    fn accuracies_counts_rates_and_seconds() {
        let b = base(&[
            ("screen.accuracy", 1.0, 0.0),
            ("board.owner.accuracy", 0.5, 0.0),
            ("board.edge.direction_accuracy", 1.0, 0.0),
            ("board.edge.label_accuracy", 0.8, 0.0),
            ("board.chrome_fp", 14.0, 0.0),
            ("board.sticky.cer", 0.02, 0.0),
            ("owners.move_error_max_s", 10.0, 0.0),
        ]);
        let good = cur(&[
            ("screen.accuracy", 0.99, 0.0),
            ("board.owner.accuracy", 0.5, 0.0),
            ("board.edge.direction_accuracy", 1.0, 0.0),
            ("board.edge.label_accuracy", 0.79, 0.0),
            ("board.chrome_fp", 13.0, 0.0),
            ("board.sticky.cer", 0.035, 0.0),
            ("owners.move_error_max_s", 11.5, 0.0),
        ]);
        assert!(regression_gate(&good, &b, 2.0).iter().all(|g| g.pass));
        let bad = cur(&[
            ("screen.accuracy", 0.97, 0.0),              // 3 points
            ("board.owner.accuracy", 0.4, 0.0),          // 10 points
            ("board.edge.direction_accuracy", 0.9, 0.0), // 10 points
            ("board.edge.label_accuracy", 0.7, 0.0),     // 10 points
            ("board.chrome_fp", 15.0, 0.0),              // any rise
            ("board.sticky.cer", 0.05, 0.0),             // 3 points
            ("owners.move_error_max_s", 12.5, 0.0),      // 2.5 s
        ]);
        let g = regression_gate(&bad, &b, 2.0);
        assert_eq!(g.len(), 7);
        assert!(g.iter().all(|x| !x.pass), "{g:#?}");
    }

    #[test]
    fn unclassified_metric_fails() {
        let b = base(&[("mystery.metric", 1.0, 0.0)]);
        let g = regression_gate(&cur(&[("mystery.metric", 1.0, 0.0)]), &b, 2.0);
        assert!(!g[0].pass);
    }

    #[test]
    fn every_emitted_metric_is_classified() {
        for name in [
            "cases",
            "screen.accuracy",
            "screen.cms_as_whiteboard",
            "screen.missed",
            "screen.unconfirmed_labels",
            "keyframes.trap_coverage",
            "board.node.precision",
            "board.node.recall",
            "board.node.f1",
            "board.node.core_recall",
            "board.edge.f1",
            "board.edge.direction_accuracy",
            "board.edge.label_accuracy",
            "board.edge.style_accuracy",
            "board.edge.uncertain",
            "board.sticky.f1",
            "board.sticky.cer",
            "board.owner.accuracy",
            "board.owner.fp",
            "board.chrome_fp",
            "board.chrome_fp_readings",
            "owners.attribution",
            "owners.extra",
            "owners.unresolved",
            "owners.negative_hits",
            "owners.moves_missed",
            "owners.move_error_max_s",
            "events.false_change",
            "notes.decision.f1",
            "notes.action.precision",
            "notes.question.recall",
            "notes.negative_action_hits",
            "audio.hotword_wer",
            "audio.speaker_labels",
            "audio.speaker_label_error",
            "audio.diarizer_label_error",
            "audio.speaker_identities",
            "audio.speaker_identity_error",
            "bench.median_s",
            "docs.body_cer",
            "docs.body_cer_page_median",
            "docs.body_cer_page_max",
            "docs.fields.accuracy",
            "docs.free_text_cer_issue_median",
            "docs.blocks.f1",
            "docs.tables.cell_accuracy",
            "docs.tables.truncated_cells",
            "docs.coverage_min",
            "docs.hallucinated_spans",
            "docs.pages_missing",
        ] {
            assert!(rule_for(name).is_some(), "{name}");
        }
        assert_eq!(rule_for("board.chrome_fp"), Some(Rule::LowerBetterCount));
        assert_eq!(
            rule_for("docs.body_cer_page_max"),
            Some(Rule::LowerBetterRate)
        );
        assert_eq!(
            rule_for("owners.move_error_max_s"),
            Some(Rule::LowerBetterSeconds)
        );
        assert_eq!(rule_for("bench.median_s"), Some(Rule::Info));
        assert_eq!(rule_for("screen.missed"), Some(Rule::LowerBetterCount));
        assert_eq!(
            rule_for("audio.diarizer_label_error"),
            Some(Rule::LowerBetterCount)
        );
        assert_eq!(
            rule_for("audio.speaker_identity_error"),
            Some(Rule::LowerBetterCount)
        );
    }

    /// Codex round-1 M6: a baseline under the retired key never fails a run whose
    /// speaker metric changed meaning.
    #[test]
    fn retired_speaker_key_is_not_compared() {
        let b = base(&[("audio.speaker_label_error", 0.0, 0.0)]);
        let g = regression_gate(
            &cur(&[
                ("audio.diarizer_label_error", 1.0, 0.0),
                ("audio.speaker_identity_error", 0.0, 0.0),
            ]),
            &b,
            2.0,
        );
        assert!(g.is_empty(), "{g:?}");
    }
}
