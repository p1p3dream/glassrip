//! The `glassrip meeting` command: resolve flags, install logging, connect the
//! backends the selected stages need, run, and print a summary.

use anyhow::Context;
use glassrip_core::graph::{meeting_mode_stage_decls, StageGraph};

use super::preflight::Needs;
use super::{logging, run_meeting, Backends, MeetingArgs};

/// Runs `glassrip meeting`. Nothing is written to `--out` unless preflight
/// passes (the log is buffered until then).
pub async fn run(args: MeetingArgs) -> anyhow::Result<()> {
    let mut opts = args.resolve()?;
    let log = logging::init();
    opts.log = Some(log);
    let plan = StageGraph::new(meeting_mode_stage_decls())?.plan(&opts.selection)?;
    let needs = Needs::from_plan(&plan);
    let backends = Backends::connect(&opts.config, &needs, &opts.out_dir).await;

    let cancel = tokio_util::sync::CancellationToken::new();
    let on_signal = cancel.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!("interrupt: stopping after in-flight items; rerun to resume");
            on_signal.cancel();
        }
    });

    let log_path = opts.out_dir.join(logging::RUN_LOG);
    let outcome = run_meeting(&opts, backends, cancel)
        .await
        .with_context(|| {
            if log_path.is_file() {
                format!("details in {}", log_path.display())
            } else {
                "nothing was written".to_string()
            }
        })?;
    for r in &outcome.reports {
        eprintln!(
            "{:<15} {:<8} items {:>5} ok {:>5} err {:>3} {:>8.2} s",
            r.stage,
            format!("{:?}", r.status).to_lowercase(),
            r.items_total,
            r.items_ok,
            r.items_error,
            r.wall_s
        );
    }
    for (phase, wall) in &outcome.phase_wall_s {
        eprintln!("phase {phase:<8} {wall:>8.2} s");
    }
    for p in &outcome.outputs {
        println!("{}", p.display());
    }
    Ok(())
}
