//! `eval_report.json`, the markdown table, spec targets, and gates.
//!
//! A report aggregates one run (replay) or several repetitions (live, default 3)
//! into a mean and spread (sample standard deviation) per metric. The report JSON
//! is also the baseline format for the regression gate ([`crate::gate`]).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::metrics::bench::BenchSummary;
use crate::suite::Metrics;

/// Report format version.
pub const REPORT_VERSION: u32 = 1;

/// Mean and spread of one metric across repetitions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricStat {
    /// Mean.
    pub mean: f64,
    /// Sample standard deviation (0 for a single run).
    #[serde(default)]
    pub spread: f64,
    /// Minimum.
    #[serde(default)]
    pub min: f64,
    /// Maximum.
    #[serde(default)]
    pub max: f64,
    /// Runs that produced the metric.
    #[serde(default)]
    pub runs: usize,
}

/// Aggregates repetitions: mean, sample standard deviation, min, max.
pub fn aggregate(runs: &[Metrics]) -> BTreeMap<String, MetricStat> {
    let mut values: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for r in runs {
        for (k, v) in r {
            values.entry(k.as_str()).or_default().push(*v);
        }
    }
    values
        .into_iter()
        .map(|(k, v)| {
            let n = v.len() as f64;
            let mean = v.iter().sum::<f64>() / n;
            let spread = if v.len() > 1 {
                (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
            } else {
                0.0
            };
            let min = v.iter().copied().fold(f64::INFINITY, f64::min);
            let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (
                k.to_string(),
                MetricStat {
                    mean,
                    spread,
                    min,
                    max,
                    runs: v.len(),
                },
            )
        })
        .collect()
}

/// Comparison operator of a target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    /// Value must be at least the threshold.
    #[serde(rename = ">=")]
    AtLeast,
    /// Value must be at most the threshold.
    #[serde(rename = "<=")]
    AtMost,
}

/// A spec target checked against a metric mean.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetCheck {
    /// Metric name.
    pub metric: String,
    /// Operator.
    pub op: Op,
    /// Threshold.
    pub threshold: f64,
    /// Observed mean.
    pub value: f64,
    /// Whether it holds.
    pub pass: bool,
    /// Spec reference.
    pub spec: String,
}

/// Spec 9.3 and 9.4 targets (informational unless also a gate).
pub const TARGETS: &[(&str, Op, f64, &str)] = &[
    (
        "screen.accuracy",
        Op::AtLeast,
        0.95,
        "9.3 screen-type accuracy",
    ),
    (
        "screen.cms_as_whiteboard",
        Op::AtMost,
        0.0,
        "9.3 0 CMS frames read as whiteboard (gate)",
    ),
    (
        "board.node.precision",
        Op::AtLeast,
        0.9,
        "9.3 node precision",
    ),
    (
        "board.node.core_recall",
        Op::AtLeast,
        1.0,
        "9.3 node recall (core)",
    ),
    (
        "board.edge.direction_accuracy",
        Op::AtLeast,
        0.9,
        "9.3 edge direction accuracy",
    ),
    (
        "board.edge.label_accuracy",
        Op::AtLeast,
        0.9,
        "9.3 edge label accuracy",
    ),
    ("board.sticky.recall", Op::AtLeast, 0.9, "9.3 sticky recall"),
    ("board.sticky.cer", Op::AtMost, 0.05, "9.3 sticky text CER"),
    (
        "board.chrome_fp",
        Op::AtMost,
        0.0,
        "9.3 UI-chrome false positives",
    ),
    (
        "owners.attribution",
        Op::AtLeast,
        0.9,
        "9.3 owner attribution at probe times",
    ),
    (
        "owners.move_error_max_s",
        Op::AtMost,
        30.0,
        "9.3 owner-move time error",
    ),
    (
        "events.false_change",
        Op::AtMost,
        0.0,
        "9.3 pan/zoom false change events",
    ),
    (
        "notes.decision.recall",
        Op::AtLeast,
        0.75,
        "9.3 decision recall",
    ),
    (
        "notes.action.precision",
        Op::AtLeast,
        0.8,
        "9.3 action-item precision",
    ),
    ("audio.hotword_wer", Op::AtMost, 0.10, "9.3 hotword WER"),
    (
        "audio.speaker_label_error",
        Op::AtMost,
        0.0,
        "9.3 distinct speaker labels equal the people",
    ),
    (
        "bench.median_s",
        Op::AtMost,
        6.0,
        "9.3 throughput per whiteboard keyframe",
    ),
    (
        "docs.body_cer_page_median",
        Op::AtMost,
        0.01,
        "9.4 body CER, per-page median (screen recordings)",
    ),
    (
        "docs.body_cer_page_max",
        Op::AtMost,
        0.03,
        "9.4 body CER, no page above",
    ),
    (
        "docs.fields.accuracy",
        Op::AtLeast,
        0.95,
        "9.4 structured ticket fields",
    ),
    (
        "docs.free_text_cer_issue_median",
        Op::AtMost,
        0.02,
        "9.4 free-text ticket fields",
    ),
    ("docs.blocks.f1", Op::AtLeast, 0.9, "9.4 block-structure F1"),
    (
        "docs.tables.cell_accuracy",
        Op::AtLeast,
        0.95,
        "9.4 table cell accuracy",
    ),
    (
        "docs.coverage_min",
        Op::AtLeast,
        0.95,
        "9.4 coverage per page",
    ),
    (
        "docs.hallucinated_spans",
        Op::AtMost,
        0.0,
        "9.4 hallucinated spans",
    ),
];

/// Checks every target whose metric is present.
pub fn check_targets(metrics: &BTreeMap<String, MetricStat>) -> Vec<TargetCheck> {
    TARGETS
        .iter()
        .filter_map(|(name, op, threshold, spec)| {
            let v = metrics.get(*name)?.mean;
            let pass = match op {
                Op::AtLeast => v >= *threshold - 1e-12,
                Op::AtMost => v <= *threshold + 1e-12,
            };
            Some(TargetCheck {
                metric: name.to_string(),
                op: *op,
                threshold: *threshold,
                value: v,
                pass,
                spec: spec.to_string(),
            })
        })
        .collect()
}

/// A pass/fail gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateResult {
    /// Gate name.
    pub name: String,
    /// Whether it passed.
    pub pass: bool,
    /// Explanation.
    pub detail: String,
}

/// The eval report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    /// Format version.
    pub report_version: u32,
    /// Suite.
    pub suite: String,
    /// `replay` or `live`.
    pub mode: String,
    /// Vision model.
    pub vision_model: String,
    /// Repetitions aggregated.
    pub repetitions: usize,
    /// Metrics.
    pub metrics: BTreeMap<String, MetricStat>,
    /// Spec targets.
    #[serde(default)]
    pub targets: Vec<TargetCheck>,
    /// Gates (all must pass).
    #[serde(default)]
    pub gates: Vec<GateResult>,
    /// Sections that could not run.
    #[serde(default)]
    pub not_run: Vec<String>,
    /// Case errors (first repetition).
    #[serde(default)]
    pub errors: Vec<String>,
    /// Throughput (with `--bench`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench: Option<BenchSummary>,
    /// Details of the first repetition.
    #[serde(default)]
    pub details: Value,
    /// True when every gate passed.
    pub passed: bool,
}

impl EvalReport {
    /// Recomputes `passed` from the gates.
    pub fn finish(&mut self) {
        self.passed = self.gates.iter().all(|g| g.pass);
    }
}

fn fmt_value(name: &str, v: f64) -> String {
    let count_like = name.ends_with("_fp")
        || name.ends_with(".fp")
        || name.ends_with("missed")
        || name.ends_with("cases")
        || name.contains("hits")
        || name.ends_with("false_change")
        || name.ends_with("uncertain")
        || name.ends_with("unresolved")
        || name.ends_with("extra")
        || name.ends_with("labels")
        || name.ends_with("cells")
        || name.ends_with("spans")
        || name.ends_with("missing");
    if count_like && v.fract() == 0.0 {
        format!("{v:.0}")
    } else if name.ends_with("_s") {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

/// Renders the markdown summary.
pub fn render_markdown(r: &EvalReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# glassrip eval: {}\n", r.suite);
    let _ = writeln!(
        s,
        "Mode `{}`, vision model `{}`, repetitions {}. Result: **{}**.\n",
        r.mode,
        r.vision_model,
        r.repetitions,
        if r.passed { "PASS" } else { "FAIL" }
    );
    let _ = writeln!(s, "| Metric | Mean | Spread | Min | Max |");
    let _ = writeln!(s, "|---|---:|---:|---:|---:|");
    for (k, m) in &r.metrics {
        let _ = writeln!(
            s,
            "| {k} | {} | {:.3} | {} | {} |",
            fmt_value(k, m.mean),
            m.spread,
            fmt_value(k, m.min),
            fmt_value(k, m.max)
        );
    }
    if !r.targets.is_empty() {
        let _ = writeln!(s, "\n| Target | Threshold | Value | Status |");
        let _ = writeln!(s, "|---|---:|---:|---|");
        for t in &r.targets {
            let op = match t.op {
                Op::AtLeast => ">=",
                Op::AtMost => "<=",
            };
            let _ = writeln!(
                s,
                "| {} ({}) | {op} {} | {} | {} |",
                t.metric,
                t.spec,
                fmt_value(&t.metric, t.threshold),
                fmt_value(&t.metric, t.value),
                if t.pass { "met" } else { "not met" }
            );
        }
    }
    let _ = writeln!(s, "\n| Gate | Status | Detail |");
    let _ = writeln!(s, "|---|---|---|");
    for g in &r.gates {
        let _ = writeln!(
            s,
            "| {} | {} | {} |",
            g.name,
            if g.pass { "pass" } else { "FAIL" },
            g.detail.replace('|', "/")
        );
    }
    if let Some(b) = &r.bench {
        let _ = writeln!(
            s,
            "\nThroughput ({}): {} keyframes, median {} s, p90 {} s, concurrency {}, per keyframe {} s.",
            b.source,
            b.keyframes,
            b.median_s.map_or("n/a".into(), |v| format!("{v:.2}")),
            b.p90_s.map_or("n/a".into(), |v| format!("{v:.2}")),
            b.concurrency,
            b.per_keyframe_s.map_or("n/a".into(), |v| format!("{v:.2}"))
        );
    }
    if !r.not_run.is_empty() {
        let _ = writeln!(s, "\nNot run:");
        for n in &r.not_run {
            let _ = writeln!(s, "- {n}");
        }
    }
    if !r.errors.is_empty() {
        let _ = writeln!(s, "\nErrors:");
        for e in &r.errors {
            let _ = writeln!(s, "- {}", e.lines().next().unwrap_or(""));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_mean_and_sample_std() {
        let runs: Vec<Metrics> = [0.8, 0.9, 1.0]
            .iter()
            .map(|v| Metrics::from([("board.node.f1".to_string(), *v)]))
            .collect();
        let a = aggregate(&runs);
        let m = a["board.node.f1"];
        assert!((m.mean - 0.9).abs() < 1e-12);
        // sample std of {0.8, 0.9, 1.0} = 0.1
        assert!((m.spread - 0.1).abs() < 1e-12);
        assert_eq!((m.min, m.max, m.runs), (0.8, 1.0, 3));
        let one = aggregate(&runs[..1]);
        assert_eq!(one["board.node.f1"].spread, 0.0);
    }

    #[test]
    fn targets_only_for_present_metrics() {
        let mut m = BTreeMap::new();
        m.insert(
            "screen.cms_as_whiteboard".to_string(),
            MetricStat {
                mean: 1.0,
                spread: 0.0,
                min: 1.0,
                max: 1.0,
                runs: 1,
            },
        );
        m.insert(
            "board.node.precision".to_string(),
            MetricStat {
                mean: 0.95,
                spread: 0.0,
                min: 0.95,
                max: 0.95,
                runs: 1,
            },
        );
        let t = check_targets(&m);
        assert_eq!(t.len(), 2);
        assert!(t
            .iter()
            .any(|x| x.metric == "board.node.precision" && x.pass));
        assert!(t
            .iter()
            .any(|x| x.metric == "screen.cms_as_whiteboard" && !x.pass));
    }

    #[test]
    fn markdown_has_tables() {
        let mut r = EvalReport {
            report_version: REPORT_VERSION,
            suite: "synthetic".into(),
            mode: "replay".into(),
            vision_model: "m".into(),
            repetitions: 1,
            metrics: aggregate(&[Metrics::from([("board.chrome_fp".to_string(), 0.0)])]),
            targets: vec![],
            gates: vec![GateResult {
                name: "g".into(),
                pass: true,
                detail: "ok".into(),
            }],
            not_run: vec!["x".into()],
            errors: vec![],
            bench: None,
            details: Value::Null,
            passed: false,
        };
        r.finish();
        assert!(r.passed);
        let md = render_markdown(&r);
        assert!(md.contains("| board.chrome_fp | 0 |"));
        assert!(md.contains("PASS"));
        assert!(md.contains("Not run:"));
    }
}
