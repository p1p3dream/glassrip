//! Records the git commit for artifact provenance (`Producer::git_sha`).
//!
//! `GLASSRIP_GIT_SHA` from the environment wins; otherwise `git rev-parse HEAD`
//! is used, and the value is empty when git is unavailable.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn main() {
    println!("cargo:rerun-if-env-changed=GLASSRIP_GIT_SHA");
    for p in ["HEAD", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--git-path", p]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&["rev-parse", "--git-path", &r]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    let sha = std::env::var("GLASSRIP_GIT_SHA")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| git(&["rev-parse", "HEAD"]))
        .unwrap_or_default();
    println!("cargo:rustc-env=GLASSRIP_GIT_SHA={sha}");
}
