//! `glassrip-privacy-scan [REPO_ROOT]`: fails (exit 1) on any denylist term,
//! private IPv4 address, or absolute home path in the repository (spec 9.1).
//!
//! The private root comes from `GLASSRIP_PRIVATE_FIXTURES`, else from
//! `eval.private_fixtures` in `./glassrip.toml`; without it the denylist scan is
//! skipped with a notice and only the built-in checks run.

use std::path::PathBuf;
use std::process::ExitCode;

use glassrip_core::config::Config;
use glassrip_eval::privacy::scan;

fn main() -> ExitCode {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let config = match PathBuf::from("glassrip.toml") {
        p if p.is_file() => match Config::load(&p) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("privacy scan: {}: {e}", p.display());
                return ExitCode::from(2);
            }
        },
        _ => Config::default(),
    };
    let private = glassrip_eval::cli::private_root(&config);
    let report = match scan(&root, private.as_deref()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("privacy scan: {e}");
            return ExitCode::from(2);
        }
    };
    for n in &report.notices {
        eprintln!("privacy scan: notice: {n}");
    }
    for f in &report.findings {
        eprintln!("{}:{}: {}", f.path.display(), f.line, f.what);
    }
    eprintln!(
        "privacy scan: {} finding(s); denylist scanned {} file(s), built-in scanned {} tracked file(s)",
        report.findings.len(),
        report.denylist_files,
        report.builtin_files
    );
    if report.findings.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
