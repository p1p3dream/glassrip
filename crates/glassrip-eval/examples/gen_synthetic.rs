//! Writes the public synthetic fixtures deterministically.
//!
//! ```text
//! cargo run -p glassrip-eval --example gen_synthetic [-- OUT_ROOT]
//! ```
//!
//! `OUT_ROOT` defaults to the workspace's `tests/fixtures`. The `synthetic` and
//! `synthetic_docs` directories under it are replaced; recorded responses and
//! baselines next to them are left alone (rerecord responses after changing
//! the fixtures).

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures"));
    for sub in ["synthetic", "synthetic_docs"] {
        let dir = root.join(sub);
        if dir.is_dir() {
            if let Err(e) = fs_err::remove_dir_all(&dir) {
                eprintln!("gen_synthetic: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    match glassrip_eval::synth::generate_all(&root) {
        Ok(g) => {
            println!(
                "gen_synthetic: wrote {} files, {:.1} KiB, under {}",
                g.files.len(),
                g.bytes as f64 / 1024.0,
                root.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("gen_synthetic: {e}");
            ExitCode::FAILURE
        }
    }
}
