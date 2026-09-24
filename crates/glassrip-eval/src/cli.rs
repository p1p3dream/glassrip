//! `glassrip eval` (spec 4):
//!
//! ```text
//! glassrip eval --suite synthetic|synthetic_docs|meeting|docs
//!     [--vision-model M] [--rerecord] [--baseline FILE] [--bench]
//!     [--mode reader|pipeline] [--config FILE] [--host URL] [--fixtures DIR]
//!     [--responses DIR] [--artifacts DIR] [--out DIR] [--repetitions N]
//! ```
//!
//! The synthetic suite has two modes. `reader` (default, the CI gate) classifies
//! each frame and reads the predicted crop directly, replaying
//! `tests/fixtures/responses/synthetic/<model>/`. `pipeline` runs the vision
//! stages (OCR harvest through edge direction) on each frame, replaying vision
//! replies and OCR spans from `tests/fixtures/responses/synthetic_pipeline/<model>/`
//! (see [`crate::pipeline`]); `--rerecord` fills that store from a live run.
//! `--suite meeting --artifacts DIR` scores a run directory (or its
//! `artifacts/` directory) written by `glassrip meeting`.
//!
//! Defaults:
//! - Public suites read `tests/fixtures/<suite>/`, replay responses from
//!   `tests/fixtures/responses/<suite>/<model>/`, and write
//!   `target/glassrip-eval/<suite>/{eval_report.json, eval_report.md}`.
//! - Private suites need `eval.private_fixtures` (or `GLASSRIP_PRIVATE_FIXTURES`)
//!   and are skipped with a notice when it is unset or missing. They read
//!   `<private>/golden/meeting_golden.json` or `<private>/golden/docs/`, run
//!   artifacts from `<private>/runs/<suite>/`, and write reports to
//!   `<private>/eval_reports/<suite>/`. Nothing private is written into the repo.
//! - The Ollama URL comes from `--host`, then `GLASSRIP_OLLAMA_URL`, then
//!   `OLLAMA_HOST`, then `ollama.host` in the config. `--rerecord` runs live,
//!   records the first repetition, and prunes stale recordings; live runs repeat
//!   3 times by default and report mean and spread.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::{Args, ValueEnum};
use glassrip_core::config::Config;
use glassrip_vision::{OllamaBackend, OllamaConfig, VisionBackend, VisionClient};
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::error::{write_json, write_text, EvalError, Result};
use crate::fixture::{load_board_suite, load_docs_suite};
use crate::gate::{load_baseline, regression_gate};
use crate::metrics::bench::summarize;
use crate::replay::{Responder, ResponseStore};
use crate::report::{
    aggregate, check_targets, render_markdown, EvalReport, GateResult, REPORT_VERSION,
};
use crate::suite::{run_board_suite, run_docs_suite, run_meeting, SuiteContext, SuiteRun};

/// Which suite to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum Suite {
    /// Public synthetic boards and screens.
    Synthetic,
    /// Public synthetic document pages.
    SyntheticDocs,
    /// Private meeting golden set.
    Meeting,
    /// Private document golden set.
    Docs,
}

impl Suite {
    /// snake_case name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::SyntheticDocs => "synthetic_docs",
            Self::Meeting => "meeting",
            Self::Docs => "docs",
        }
    }
}

/// How the synthetic suite reads boards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum SyntheticMode {
    /// Classify the whole frame, read the predicted crop (the CI gate).
    #[default]
    Reader,
    /// Run the vision stages (OCR harvest through edge direction) on each frame.
    Pipeline,
}

/// Arguments of `glassrip eval`.
#[derive(Debug, Clone, Args)]
pub struct EvalArgs {
    /// Suite to run.
    #[arg(long, value_enum)]
    pub suite: Suite,
    /// Vision model (default: `models.vision` from the config).
    #[arg(long)]
    pub vision_model: Option<String>,
    /// Run against the live backend and refresh the recorded responses.
    #[arg(long)]
    pub rerecord: bool,
    /// Baseline report to gate against.
    #[arg(long)]
    pub baseline: Option<PathBuf>,
    /// Report throughput per whiteboard keyframe.
    #[arg(long)]
    pub bench: bool,
    /// Config file (default: ./glassrip.toml when present).
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Ollama URL (overrides GLASSRIP_OLLAMA_URL, OLLAMA_HOST, and the config).
    #[arg(long)]
    pub host: Option<String>,
    /// Fixture root for the suite.
    #[arg(long)]
    pub fixtures: Option<PathBuf>,
    /// Recorded-responses directory.
    #[arg(long)]
    pub responses: Option<PathBuf>,
    /// Run artifacts to score (docs and meeting suites).
    #[arg(long)]
    pub artifacts: Option<PathBuf>,
    /// Output directory for eval_report.json and eval_report.md.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Live repetitions (default 3; replay always runs once).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=20))]
    pub repetitions: Option<u32>,
    /// Synthetic suite mode.
    #[arg(long, value_enum, default_value_t = SyntheticMode::Reader)]
    pub mode: SyntheticMode,
}

/// What the command did.
#[derive(Debug, Clone)]
pub struct EvalOutcome {
    /// Every gate passed, or the suite was skipped (exit status 0 either way).
    pub passed: bool,
    /// Pass, fail, or skipped.
    pub status: crate::report::Status,
    /// Report path, when written.
    pub report_path: Option<PathBuf>,
}

/// Filesystem-safe model name (`qwen2.5vl:7b` becomes `qwen2.5vl-7b`).
pub fn model_slug(model: &str) -> String {
    model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

fn expand_home(p: &Path) -> PathBuf {
    if let Ok(rest) = p.strip_prefix("~") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    p.to_path_buf()
}

/// Private fixtures root: `GLASSRIP_PRIVATE_FIXTURES`, else `eval.private_fixtures`.
pub fn private_root(config: &Config) -> Option<PathBuf> {
    std::env::var_os("GLASSRIP_PRIVATE_FIXTURES")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| config.eval.private_fixtures.clone())
        .map(|p| expand_home(&p))
}

/// Ollama URL resolution order: flag, `GLASSRIP_OLLAMA_URL`, `OLLAMA_HOST`, config.
pub fn resolve_host(flag: Option<&str>, config: &Config) -> String {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    let raw = flag
        .map(str::to_string)
        .or_else(|| env("GLASSRIP_OLLAMA_URL"))
        .or_else(|| env("OLLAMA_HOST"))
        .unwrap_or_else(|| config.ollama.host.clone());
    if raw.starts_with("http://") || raw.starts_with("https://") {
        raw
    } else {
        format!("http://{raw}")
    }
}

fn load_config(path: Option<&Path>) -> Result<Config> {
    let default = PathBuf::from("glassrip.toml");
    let p = match path {
        Some(p) => Some(p.to_path_buf()),
        None => default.is_file().then_some(default),
    };
    match p {
        Some(p) => Config::load(&p).map_err(|e| EvalError::Config(format!("{}: {e}", p.display()))),
        None => Ok(Config::default()),
    }
}

/// Public fixture root: `./tests/fixtures`, else the workspace's (build-time path).
pub fn public_fixtures_root() -> PathBuf {
    let local = PathBuf::from("tests/fixtures");
    if local.is_dir() {
        return local;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

async fn live_client(host: &str, model: &str, config: &Config) -> Result<VisionClient> {
    let mut oc = OllamaConfig::new(host, model, config.ollama.num_ctx);
    oc.keep_alive = config.ollama.keep_alive.clone();
    oc.slots = config.ollama.num_parallel as usize;
    oc.allow_spill = config.gpu.allow_spill;
    oc.request_timeout = std::time::Duration::from_secs(config.ollama.request_timeout_s);
    let backend = Arc::new(OllamaBackend::new(oc)?);
    let placement = backend.preflight().await?;
    eprintln!(
        "eval: live backend {host}, model {model} ({}), concurrency {}",
        backend.id().digest.unwrap_or_default(),
        placement.concurrency_hint
    );
    Ok(VisionClient::new(
        backend,
        placement.concurrency_hint.max(1),
    )?)
}

/// Live vision backend, placement probe, and OCR for pipeline-mode recording.
type LivePipeline = (
    Arc<dyn VisionBackend>,
    Arc<dyn glassrip_vision_stages::placement::PlacementProbe>,
    Arc<dyn glassrip_ocr::TextRecognizer>,
);

async fn live_pipeline(host: &str, model: &str, config: &Config) -> Result<LivePipeline> {
    let mut oc = OllamaConfig::new(host, model, config.ollama.num_ctx);
    oc.keep_alive = config.ollama.keep_alive.clone();
    oc.slots = config.ollama.num_parallel as usize;
    oc.allow_spill = config.gpu.allow_spill;
    oc.request_timeout = std::time::Duration::from_secs(config.ollama.request_timeout_s);
    let backend = Arc::new(OllamaBackend::new(oc)?);
    backend.resolve_digest().await?;
    Ok((backend.clone(), backend, ocr_engine()?))
}

#[cfg(feature = "onnx")]
fn ocr_engine() -> Result<Arc<dyn glassrip_ocr::TextRecognizer>> {
    glassrip_ocr::engine::PpOcrEngine::new(
        &glassrip_ocr::models::default_dir(),
        glassrip_ocr::OcrConfig::default(),
    )
    .map(|e| Arc::new(e) as Arc<dyn glassrip_ocr::TextRecognizer>)
    .map_err(|e| EvalError::Config(format!("PP-OCRv5: {e}")))
}

#[cfg(not(feature = "onnx"))]
fn ocr_engine() -> Result<Arc<dyn glassrip_ocr::TextRecognizer>> {
    Err(EvalError::Config(
        "recording pipeline mode needs PP-OCRv5: build with `--features onnx` or `--features cuda`"
            .into(),
    ))
}

fn base_report(suite: Suite, mode: &str, model: &str, runs: &[SuiteRun]) -> EvalReport {
    let metrics = aggregate(&runs.iter().map(|r| r.metrics.clone()).collect::<Vec<_>>());
    let first = runs.first().cloned().unwrap_or_default();
    EvalReport {
        report_version: REPORT_VERSION,
        suite: suite.name().into(),
        mode: mode.into(),
        vision_model: model.into(),
        repetitions: runs.len(),
        targets: check_targets(&metrics),
        metrics,
        gates: Vec::new(),
        not_run: first.not_run,
        errors: first.errors,
        bench: None,
        details: first.details,
        status: crate::report::Status::Fail,
        skipped_reason: None,
        passed: false,
    }
}

fn standard_gates(report: &mut EvalReport, runs: &[SuiteRun]) {
    let errors: usize = runs.iter().map(|r| r.errors.len()).sum();
    report.gates.push(GateResult {
        name: "no_case_errors".into(),
        pass: errors == 0,
        detail: format!("{errors} case error(s) across repetitions"),
    });
    if let Some(m) = report.metrics.get("screen.cms_as_whiteboard") {
        report.gates.push(GateResult {
            name: "cms_not_read_as_whiteboard".into(),
            pass: m.max == 0.0,
            detail: format!(
                "worst repetition: {} CMS frame(s) read as whiteboard",
                m.max
            ),
        });
    }
    for r in runs {
        for f in &r.gate_failures {
            report.gates.push(GateResult {
                name: "scoring".into(),
                pass: false,
                detail: f.clone(),
            });
        }
    }
}

/// Runs `glassrip eval`.
pub async fn run(args: EvalArgs) -> Result<EvalOutcome> {
    let config = load_config(args.config.as_deref())?;
    let model = args
        .vision_model
        .clone()
        .unwrap_or_else(|| config.models.vision.clone());
    let suite = args.suite;
    let private = private_root(&config);
    let cancel = CancellationToken::new();

    let (report, out_dir) = match suite {
        Suite::Synthetic if args.mode == SyntheticMode::Pipeline => {
            let root = args
                .fixtures
                .clone()
                .unwrap_or_else(|| public_fixtures_root().join("synthetic"));
            let cases = load_board_suite(&root)?;
            let responses = args
                .responses
                .clone()
                .unwrap_or_else(|| crate::pipeline::default_responses(&model_slug(&model)));
            let out = args.out.clone().unwrap_or_else(|| {
                PathBuf::from("target/glassrip-eval").join("synthetic_pipeline")
            });
            let (raw, ocr_store) = crate::pipeline::stores(&responses);
            let reps = if args.rerecord {
                args.repetitions.unwrap_or(3) as usize
            } else {
                1
            };
            let mut runs = Vec::new();
            for rep in 0..reps {
                let ctx = if args.rerecord {
                    let host = resolve_host(args.host.as_deref(), &config);
                    let live = live_pipeline(&host, &model, &config).await?;
                    let record = rep == 0;
                    crate::pipeline::PipelineContext {
                        vision: if record {
                            Arc::new(crate::pipeline::RecordingBackend::new(
                                live.0.clone(),
                                raw.clone(),
                            ))
                        } else {
                            live.0.clone()
                        },
                        probe: live.1,
                        recognizer: if record {
                            Arc::new(crate::pipeline::RecordingRecognizer::new(
                                live.2,
                                ocr_store.clone(),
                            ))
                        } else {
                            live.2
                        },
                        model: model.clone(),
                        slots: config.ollama.num_parallel as usize,
                        seed: config.ollama.seed,
                        num_ctx: config.ollama.num_ctx,
                        work_dir: out.join("runs"),
                        cancel: cancel.clone(),
                    }
                } else {
                    crate::pipeline::PipelineContext {
                        vision: Arc::new(crate::pipeline::ReplayBackend::new(&model, raw.clone())),
                        probe: Arc::new(glassrip_vision_stages::placement::StaticProbe),
                        recognizer: Arc::new(crate::pipeline::ReplayRecognizer::new(
                            ocr_store.clone(),
                        )),
                        model: model.clone(),
                        slots: 1,
                        seed: config.ollama.seed,
                        num_ctx: config.ollama.num_ctx,
                        work_dir: out.join("runs"),
                        cancel: cancel.clone(),
                    }
                };
                let run = crate::pipeline::run_pipeline_suite(&ctx, &cases).await;
                eprintln!(
                    "eval: pipeline repetition {} of {reps} done ({} case errors)",
                    rep + 1,
                    run.errors.len()
                );
                runs.push(run);
            }
            let mode = if args.rerecord {
                "pipeline_live"
            } else {
                "pipeline_replay"
            };
            let mut report = base_report(suite, mode, &model, &runs);
            standard_gates(&mut report, &runs);
            (report, out)
        }
        Suite::Synthetic => {
            let root = args
                .fixtures
                .clone()
                .unwrap_or_else(|| public_fixtures_root().join("synthetic"));
            let cases = load_board_suite(&root)?;
            let responses = args.responses.clone().unwrap_or_else(|| {
                public_fixtures_root()
                    .join("responses/synthetic")
                    .join(model_slug(&model))
            });
            let store = ResponseStore::new(&responses);
            let reps = if args.rerecord {
                args.repetitions.unwrap_or(3) as usize
            } else {
                1
            };
            let mut runs = Vec::new();
            let mut elapsed = None;
            let mut concurrency = 1;
            for rep in 0..reps {
                let responder = if args.rerecord {
                    let host = resolve_host(args.host.as_deref(), &config);
                    let client = live_client(&host, &model, &config).await?;
                    concurrency = client.max_in_flight();
                    Responder::live(client, (rep == 0).then(|| store.clone()))
                } else {
                    Responder::replay(store.clone())
                };
                let ctx = Arc::new(SuiteContext {
                    responder,
                    model: model.clone(),
                    seed: config.ollama.seed,
                    num_predict: config.ollama.num_predict,
                    cancel: cancel.clone(),
                });
                let start = Instant::now();
                let run = run_board_suite(ctx.clone(), &cases).await;
                if rep == 0 {
                    elapsed = Some(start.elapsed().as_secs_f64());
                    let pruned = ctx.responder.prune()?;
                    if pruned > 0 {
                        eprintln!("eval: pruned {pruned} stale recorded response(s)");
                    }
                }
                eprintln!(
                    "eval: repetition {} of {reps} done ({} case errors)",
                    rep + 1,
                    run.errors.len()
                );
                runs.push(run);
            }
            let mode = if args.rerecord { "live" } else { "replay" };
            let mut report = base_report(suite, mode, &model, &runs);
            if args.bench {
                let lat = runs
                    .first()
                    .map(|r| r.latencies_s.clone())
                    .unwrap_or_default();
                let b = if args.rerecord {
                    summarize("live", &lat, concurrency, elapsed)
                } else {
                    summarize("recorded", &lat, 1, None)
                };
                if let Some(m) = b.median_s {
                    report.metrics.insert(
                        "bench.median_s".into(),
                        crate::report::MetricStat {
                            mean: m,
                            spread: 0.0,
                            min: m,
                            max: m,
                            runs: 1,
                        },
                    );
                }
                report.bench = Some(b);
                report.targets = check_targets(&report.metrics);
            }
            standard_gates(&mut report, &runs);
            let out = args
                .out
                .clone()
                .unwrap_or_else(|| PathBuf::from("target/glassrip-eval").join(suite.name()));
            (report, out)
        }
        Suite::SyntheticDocs => {
            let root = args
                .fixtures
                .clone()
                .unwrap_or_else(|| public_fixtures_root().join("synthetic_docs"));
            let cases = load_docs_suite(&root)?;
            if args.rerecord {
                eprintln!("eval: --rerecord has no effect on synthetic_docs until a doc_read stage exists");
            }
            let run = run_docs_suite(&cases, args.artifacts.as_deref());
            let runs = vec![run];
            let mut report = base_report(suite, "artifacts", &model, &runs);
            standard_gates(&mut report, &runs);
            let out = args
                .out
                .clone()
                .unwrap_or_else(|| PathBuf::from("target/glassrip-eval").join(suite.name()));
            (report, out)
        }
        Suite::Meeting | Suite::Docs => {
            let Some(private) = private.filter(|p| p.is_dir()) else {
                let reason = format!(
                    "suite {} needs private fixtures: eval.private_fixtures (or GLASSRIP_PRIVATE_FIXTURES) is not set or does not exist",
                    suite.name()
                );
                eprintln!("eval: SKIPPED: {reason}");
                let mut report = base_report(suite, "skipped", &model, &[]);
                report.skipped_reason = Some(reason);
                report.finish();
                // Nothing private is known here, so the skip report goes to the public output path.
                let out = args
                    .out
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("target/glassrip-eval").join(suite.name()));
                let json_path = out.join("eval_report.json");
                write_json(&json_path, &report)?;
                let md = render_markdown(&report);
                write_text(&out.join("eval_report.md"), &md)?;
                println!("{md}");
                return Ok(EvalOutcome {
                    passed: true,
                    status: report.status,
                    report_path: Some(json_path),
                });
            };
            if args.rerecord {
                eprintln!(
                    "eval: --rerecord is not used by the {} suite yet; it scores run artifacts",
                    suite.name()
                );
            }
            let artifacts = args
                .artifacts
                .clone()
                .unwrap_or_else(|| private.join("runs").join(suite.name()).join("artifacts"));
            let runs = if suite == Suite::Meeting {
                let path = crate::golden::find_golden(&private).ok_or_else(|| {
                    EvalError::Config(format!(
                        "no golden/meeting_golden.json under {}",
                        private.display()
                    ))
                })?;
                let golden = crate::golden::load_golden(&path)?;
                if artifacts.is_dir() {
                    let art = crate::views::RunArtifacts::scan(&artifacts)?;
                    vec![run_meeting(
                        &golden,
                        &art,
                        config.eval.time_join_tolerance_s,
                    )?]
                } else {
                    vec![SuiteRun {
                        not_run: vec![format!(
                            "all metrics: no run artifacts at {}",
                            artifacts.display()
                        )],
                        details: json!({ "golden": path.display().to_string() }),
                        ..Default::default()
                    }]
                }
            } else {
                let root = args
                    .fixtures
                    .clone()
                    .unwrap_or_else(|| private.join("golden").join("docs"));
                let cases = load_docs_suite(&root)?;
                vec![run_docs_suite(
                    &cases,
                    artifacts.is_dir().then_some(artifacts.as_path()),
                )]
            };
            let mut report = base_report(suite, "artifacts", &model, &runs);
            standard_gates(&mut report, &runs);
            let out = args
                .out
                .clone()
                .unwrap_or_else(|| private.join("eval_reports").join(suite.name()));
            (report, out)
        }
    };

    let mut report = report;
    if let Some(b) = &args.baseline {
        let baseline = load_baseline(b)?;
        report.gates.extend(regression_gate(
            &report.metrics,
            &baseline,
            config.eval.max_f1_drop_points,
        ));
    }
    report.finish();
    let json_path = out_dir.join("eval_report.json");
    write_json(&json_path, &report)?;
    let md = render_markdown(&report);
    write_text(&out_dir.join("eval_report.md"), &md)?;
    println!("{md}");
    eprintln!("eval: wrote {}", json_path.display());
    Ok(EvalOutcome {
        passed: report.passed,
        status: report.status,
        report_path: Some(json_path),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        args: EvalArgs,
    }

    #[test]
    fn parses_spec_flags() {
        let w = Wrap::try_parse_from([
            "x",
            "--suite",
            "synthetic_docs",
            "--vision-model",
            "m:1",
            "--rerecord",
            "--baseline",
            "b.json",
            "--bench",
        ])
        .unwrap();
        assert_eq!(w.args.suite, Suite::SyntheticDocs);
        assert_eq!(w.args.vision_model.as_deref(), Some("m:1"));
        assert!(w.args.rerecord && w.args.bench);
        assert!(Wrap::try_parse_from(["x", "--suite", "nope"]).is_err());
        assert!(Wrap::try_parse_from(["x", "--suite", "meeting", "--repetitions", "0"]).is_err());
    }

    #[test]
    fn slug_and_host() {
        assert_eq!(model_slug("qwen2.5vl:7b"), "qwen2.5vl-7b");
        let c = Config::default();
        assert_eq!(
            resolve_host(Some("example.test:11434"), &c),
            "http://example.test:11434"
        );
        assert_eq!(resolve_host(Some("https://h"), &c), "https://h");
    }
}
