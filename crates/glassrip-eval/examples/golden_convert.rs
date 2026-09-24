//! Converts an authored golden file into the JSON the meeting suite reads.
//!
//! ```text
//! cargo run -p glassrip-eval --example golden_convert -- \
//!     --in PRIVATE/golden_src.toml --keyframes PRIVATE/keyframes.json \
//!     --out PRIVATE/golden/meeting_golden.json
//! ```
//!
//! - `--in`: golden source, TOML or JSON (see `glassrip_eval::golden`), which may
//!   use `screen_type_ranges`.
//! - `--keyframes`: reference keyframe list (a bare list of `t_rep` numbers or an
//!   object with `keyframes: [{t_rep}]`), required when ranges are used.
//! - `--out`: output JSON. The converter refuses to write inside a git worktree
//!   (golden files are private and never belong in a repository) unless
//!   `--allow-in-repo` is given.
//!
//! Run it locally against private files only.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use glassrip_eval::golden::{parse_golden, read_keyframe_times};

struct Args {
    input: PathBuf,
    keyframes: Option<PathBuf>,
    out: PathBuf,
    allow_in_repo: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut input = None;
    let mut keyframes = None;
    let mut out = None;
    let mut allow_in_repo = false;
    let mut it = std::env::args_os().skip(1);
    while let Some(a) = it.next() {
        match a.to_str() {
            Some("--in") => input = it.next().map(PathBuf::from),
            Some("--keyframes") => keyframes = it.next().map(PathBuf::from),
            Some("--out") => out = it.next().map(PathBuf::from),
            Some("--allow-in-repo") => allow_in_repo = true,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        input: input.ok_or("--in is required")?,
        keyframes,
        out: out.ok_or("--out is required")?,
        allow_in_repo,
    })
}

fn inside_git_worktree(path: &Path) -> bool {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut probe = dir.to_path_buf();
    while !probe.exists() {
        match probe.parent() {
            Some(p) if !p.as_os_str().is_empty() => probe = p.to_path_buf(),
            _ => return false,
        }
    }
    Command::new("git")
        .arg("-C")
        .arg(&probe)
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "true")
        .unwrap_or(false)
}

fn run() -> Result<String, String> {
    let args = parse_args()?;
    if !args.allow_in_repo && inside_git_worktree(&args.out) {
        return Err(format!(
            "refusing to write {} inside a git worktree; golden files are private (use --allow-in-repo to override)",
            args.out.display()
        ));
    }
    let mut golden = parse_golden(&args.input).map_err(|e| e.to_string())?;
    if !golden.screen_type_ranges.is_empty() {
        let kf = args
            .keyframes
            .as_ref()
            .ok_or("--keyframes is required to expand screen_type_ranges")?;
        let times = read_keyframe_times(kf).map_err(|e| e.to_string())?;
        golden.expand_ranges(&times).map_err(|e| e.to_string())?;
    }
    golden.validate().map_err(|e| e.to_string())?;
    let mut text = serde_json::to_string_pretty(&golden).map_err(|e| e.to_string())?;
    text.push('\n');
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs_err::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs_err::write(&args.out, text).map_err(|e| e.to_string())?;
    Ok(format!(
        "golden_convert: wrote {} ({} screen labels, {} nodes, {} edges, {} owner assignments)",
        args.out.display(),
        golden.screen_types.len(),
        golden.final_board.nodes.len(),
        golden.final_board.edges.len(),
        golden.owners.assignments.len()
    ))
}

fn main() -> ExitCode {
    match run() {
        Ok(msg) => {
            println!("{msg}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("golden_convert: {e}");
            ExitCode::FAILURE
        }
    }
}
