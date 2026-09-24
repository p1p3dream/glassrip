//! Regenerates the committed hashed denylist from the private plaintext list.
//!
//! ```text
//! cargo run -p glassrip-eval --example hash_denylist -- <private denylist> [--out FILE] [--new-salt]
//! ```
//!
//! `--out` defaults to the workspace's `tests/privacy/denylist.sha256`. The salt
//! already in that file is reused so regeneration is stable; `--new-salt` (or a
//! missing file) draws a fresh one. Terms prefixed with `!` stay plaintext-only.
//! Dictionary words are guessable from salted hashes (the salt is public), so
//! the plaintext list should favor names and specific identifiers.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use glassrip_eval::privacy::{parse_denylist, term_words, HashedDenylist, MAX_HASHED_NGRAM};

fn fresh_salt(input: &Path) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seed = format!("{nanos}:{}:{}", std::process::id(), input.display());
    glassrip_eval::privacy::hash_term("salt", &seed)[..32].to_string()
}

fn run() -> Result<String, String> {
    let mut input = None;
    let mut out =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/privacy/denylist.sha256");
    let mut new_salt = false;
    let mut it = std::env::args_os().skip(1);
    while let Some(a) = it.next() {
        match a.to_str() {
            Some("--out") => out = it.next().map(PathBuf::from).ok_or("--out needs a path")?,
            Some("--new-salt") => new_salt = true,
            _ => input = Some(PathBuf::from(a)),
        }
    }
    let input = input.ok_or("usage: hash_denylist <private denylist> [--out FILE] [--new-salt]")?;
    let text = fs_err::read_to_string(&input).map_err(|e| e.to_string())?;
    let terms = parse_denylist(&text);
    for t in terms.iter().filter(|t| !t.plaintext_only) {
        let words = term_words(&t.text);
        let rejoinable = t
            .text
            .chars()
            .all(|c| c.is_alphanumeric() || c == ' ' || c == '-' || c == '_');
        if words > MAX_HASHED_NGRAM || !rejoinable {
            eprintln!(
                "hash_denylist: warning: a {words}-word term with punctuation or more than {MAX_HASHED_NGRAM} words cannot be matched by the hashed scan; shorten it or mark it with !"
            );
        }
    }
    let existing = fs_err::read_to_string(&out)
        .ok()
        .and_then(|t| HashedDenylist::parse(&t).ok());
    let salt = match existing {
        Some(e) if !new_salt => e.salt,
        _ => fresh_salt(&input),
    };
    let list = HashedDenylist::from_terms(&salt, &terms);
    if let Some(parent) = out.parent() {
        fs_err::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs_err::write(&out, list.render()).map_err(|e| e.to_string())?;
    Ok(format!(
        "hash_denylist: wrote {} hashes ({} plaintext-only terms skipped) to {}",
        list.hashes.len(),
        terms.iter().filter(|t| t.plaintext_only).count(),
        out.display()
    ))
}

fn main() -> ExitCode {
    match run() {
        Ok(m) => {
            println!("{m}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("hash_denylist: {e}");
            ExitCode::FAILURE
        }
    }
}
