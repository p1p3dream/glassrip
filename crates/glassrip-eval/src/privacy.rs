//! Privacy check for the public repository (spec 9.1).
//!
//! Two scans:
//! 1. **Denylist** (whole worktree): every file under the repository root, except
//!    `.git/`, `target/`, and the private fixtures path when it happens to be
//!    inside the checkout, is searched case-insensitively for each term of the
//!    private denylist, matched on word boundaries. The denylist lives in the
//!    private root (`<private>/privacy-denylist.txt`, one term per line, `#`
//!    comments); when it is absent the scan is skipped with a notice.
//! 2. **Built-in** (tracked files, from `git ls-files`): RFC1918 IPv4 addresses
//!    (10/8, 172.16/12, 192.168/16) and absolute user home paths
//!    (`/Users/<name>`, `/home/<name>`), except placeholder names such as `you`
//!    or `user`.
//!
//! Binary files (containing a NUL byte in the first 8 KiB) are skipped.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{EvalError, Result};

/// File name of the denylist inside the private root.
pub const DENYLIST_FILE: &str = "privacy-denylist.txt";

/// Home-directory names treated as placeholders.
pub const PLACEHOLDER_USERS: &[&str] = &[
    "you",
    "user",
    "username",
    "me",
    "example",
    "runner",
    "name",
    "your-name",
    "yourname",
    "someone",
];

/// One finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// File, relative to the scanned root.
    pub path: PathBuf,
    /// 1-based line.
    pub line: usize,
    /// What matched (the denylist term is not echoed, only its index).
    pub what: String,
}

/// Reads a denylist: trimmed non-empty lines, `#` comments removed.
pub fn parse_denylist(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// True when `term` (lowercase) occurs in `line` on word boundaries, case-insensitively.
pub fn contains_term(line: &str, term: &str) -> bool {
    let hay = line.to_lowercase();
    let mut start = 0;
    while let Some(pos) = hay[start..].find(term) {
        let at = start + pos;
        let end = at + term.len();
        let before_ok = hay[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !is_word_char(c));
        let after_ok = hay[end..].chars().next().is_none_or(|c| !is_word_char(c));
        if before_ok && after_ok {
            return true;
        }
        start = at + term.chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// RFC1918 addresses found in `line`.
pub fn private_ipv4(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let boundary = i == 0 || !(chars[i - 1].is_ascii_digit() || chars[i - 1] == '.');
        if chars[i].is_ascii_digit() && boundary {
            let mut j = i;
            while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
                j += 1;
            }
            let token: String = chars[i..j]
                .iter()
                .collect::<String>()
                .trim_end_matches('.')
                .to_string();
            let parts: Vec<&str> = token.split('.').collect();
            if parts.len() == 4 && parts.iter().all(|p| !p.is_empty() && p.len() <= 3) {
                let nums: Vec<u32> = parts.iter().filter_map(|p| p.parse().ok()).collect();
                if nums.len() == 4 && nums.iter().all(|n| *n <= 255) {
                    let private = nums[0] == 10
                        || (nums[0] == 172 && (16..=31).contains(&nums[1]))
                        || (nums[0] == 192 && nums[1] == 168);
                    if private {
                        out.push(token);
                    }
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Absolute user home paths found in `line` (placeholders excluded).
pub fn home_paths(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    for prefix in ["/Users/", "/home/"] {
        let mut start = 0;
        while let Some(pos) = line[start..].find(prefix) {
            let at = start + pos;
            let rest = &line[at + prefix.len()..];
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
                .collect();
            let placeholder = PLACEHOLDER_USERS.contains(&name.to_lowercase().as_str())
                || name.is_empty()
                || name.starts_with('<')
                || name.starts_with('$');
            let preceded_ok = line[..at]
                .chars()
                .next_back()
                .is_none_or(|c| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '~'));
            if !placeholder && preceded_ok {
                out.push(format!("{prefix}{name}"));
            }
            start = at + prefix.len();
        }
    }
    out
}

fn is_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8192).any(|b| *b == 0)
}

fn walk(dir: &Path, skip: &[PathBuf], out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = fs_err::read_dir(dir).map_err(|e| EvalError::io(dir, e))?;
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for p in paths {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == ".git" || name == "target" || skip.iter().any(|s| p.starts_with(s)) {
            continue;
        }
        let meta = fs_err::symlink_metadata(&p).map_err(|e| EvalError::io(&p, e))?;
        if meta.is_dir() {
            walk(&p, skip, out)?;
        } else if meta.is_file() {
            out.push(p);
        }
    }
    Ok(())
}

/// Every file of the worktree (see module docs for exclusions).
pub fn worktree_files(root: &Path, skip: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    walk(root, skip, &mut out)?;
    Ok(out)
}

/// Tracked files from `git ls-files`; `None` when git is unavailable.
pub fn tracked_files(root: &Path) -> Option<Vec<PathBuf>> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        out.stdout
            .split(|b| *b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| root.join(String::from_utf8_lossy(s).as_ref()))
            .filter(|p| p.is_file())
            .collect(),
    )
}

fn read_text(path: &Path) -> Option<String> {
    let bytes = fs_err::read(path).ok()?;
    if is_binary(&bytes) {
        return None;
    }
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

fn rel(root: &Path, p: &Path) -> PathBuf {
    p.strip_prefix(root).unwrap_or(p).to_path_buf()
}

/// Denylist scan over `files`.
pub fn scan_denylist(root: &Path, files: &[PathBuf], terms: &[String]) -> Vec<Finding> {
    let mut out = Vec::new();
    for f in files {
        let Some(text) = read_text(f) else { continue };
        for (i, line) in text.lines().enumerate() {
            for (ti, term) in terms.iter().enumerate() {
                if contains_term(line, term) {
                    out.push(Finding {
                        path: rel(root, f),
                        line: i + 1,
                        what: format!("denylist term #{}", ti + 1),
                    });
                }
            }
        }
    }
    out
}

/// Built-in scan (private addresses, home paths) over `files`.
pub fn scan_builtin(root: &Path, files: &[PathBuf]) -> Vec<Finding> {
    let mut out = Vec::new();
    for f in files {
        let Some(text) = read_text(f) else { continue };
        for (i, line) in text.lines().enumerate() {
            for ip in private_ipv4(line) {
                out.push(Finding {
                    path: rel(root, f),
                    line: i + 1,
                    what: format!("private IPv4 address {ip}"),
                });
            }
            for h in home_paths(line) {
                out.push(Finding {
                    path: rel(root, f),
                    line: i + 1,
                    what: format!("absolute home path {h}"),
                });
            }
        }
    }
    out
}

/// Result of a full scan.
#[derive(Debug, Default)]
pub struct ScanReport {
    /// Findings.
    pub findings: Vec<Finding>,
    /// Notices (skipped scans).
    pub notices: Vec<String>,
    /// Files scanned by the denylist scan.
    pub denylist_files: usize,
    /// Files scanned by the built-in scan.
    pub builtin_files: usize,
}

/// Runs both scans for the repository at `root`.
pub fn scan(root: &Path, private_root: Option<&Path>) -> Result<ScanReport> {
    let mut report = ScanReport::default();
    let root = root.canonicalize().map_err(|e| EvalError::io(root, e))?;
    let mut skip = Vec::new();
    if let Some(p) = private_root.and_then(|p| p.canonicalize().ok()) {
        if p.starts_with(&root) {
            skip.push(p);
        }
    }
    match private_root.map(|p| p.join(DENYLIST_FILE)).filter(|p| p.is_file()) {
        Some(list) => {
            let terms = parse_denylist(&crate::error::read_to_string(&list)?);
            let files = worktree_files(&root, &skip)?;
            report.denylist_files = files.len();
            report.findings.extend(scan_denylist(&root, &files, &terms));
        }
        None => report.notices.push(format!(
            "denylist scan skipped: no {DENYLIST_FILE} in the private fixtures root (set eval.private_fixtures or GLASSRIP_PRIVATE_FIXTURES)"
        )),
    }
    let tracked = match tracked_files(&root) {
        Some(t) => t,
        None => {
            report
                .notices
                .push("git unavailable: built-in scan covers the whole worktree".into());
            worktree_files(&root, &skip)?
        }
    };
    let tracked: Vec<PathBuf> = tracked
        .into_iter()
        .filter(|p| !skip.iter().any(|s| p.starts_with(s)))
        .collect();
    report.builtin_files = tracked.len();
    report.findings.extend(scan_builtin(&root, &tracked));
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(a: u32, b: u32, c: u32, d: u32) -> String {
        format!("{a}.{b}.{c}.{d}")
    }

    #[test]
    fn detects_private_ranges_only() {
        let line = format!(
            "hosts {} {} {} {} {} v{}",
            ip(10, 1, 2, 3),
            ip(172, 20, 0, 1),
            ip(192, 168, 7, 9),
            ip(172, 32, 0, 1),
            ip(8, 8, 8, 8),
            ip(1, 10, 0, 0)
        );
        let found = private_ipv4(&line);
        assert_eq!(
            found,
            vec![ip(10, 1, 2, 3), ip(172, 20, 0, 1), ip(192, 168, 7, 9)]
        );
        // versions and longer dotted tokens are not addresses
        assert!(private_ipv4("version 10.2.3").is_empty());
        assert!(private_ipv4(&format!("{}.5", ip(10, 1, 2, 3))).is_empty());
        assert!(private_ipv4(&format!("{}.", ip(10, 0, 0, 1))).len() == 1);
    }

    #[test]
    fn detects_home_paths_except_placeholders() {
        let real = format!("{}{}", "/Users/", "alice/code");
        assert_eq!(home_paths(&real), vec![format!("{}{}", "/Users/", "alice")]);
        assert!(home_paths(&format!("{}{}", "/home/", "runner/work")).is_empty());
        assert!(home_paths(&format!("{}{}", "/Users/", "you/project")).is_empty());
        assert!(home_paths(&format!("{}{}", "/home/", "$USER")).is_empty());
        assert!(home_paths("~/Documents and /usr/local").is_empty());
        assert_eq!(home_paths(&format!("x={}{}", "/home/", "bob")).len(), 1);
    }

    #[test]
    fn denylist_word_boundaries() {
        let terms = parse_denylist("# comment\n  Quorra Labs \n\nzeta # trailing\n");
        assert_eq!(terms, vec!["quorra labs", "zeta"]);
        assert!(contains_term("Hello quorra labs team", "quorra labs"));
        assert!(contains_term("ZETA!", "zeta"));
        assert!(!contains_term("zetas", "zeta"));
        assert!(!contains_term("alphazeta", "zeta"));
        assert!(contains_term("alphazeta zeta", "zeta"));
    }

    #[test]
    fn scan_walks_and_skips() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs_err::create_dir_all(root.join("src")).unwrap();
        fs_err::create_dir_all(root.join("target")).unwrap();
        fs_err::create_dir_all(root.join("private")).unwrap();
        fs_err::write(root.join("src/a.rs"), "let x = 1; // Quorra here\n").unwrap();
        fs_err::write(root.join("target/b.rs"), "Quorra\n").unwrap();
        fs_err::write(root.join("private/c.txt"), "Quorra\n").unwrap();
        fs_err::write(root.join("private").join(DENYLIST_FILE), "quorra\n").unwrap();
        fs_err::write(root.join("bin.dat"), [0u8, 81, 117, 111, 114, 114, 97]).unwrap();
        let report = scan(root, Some(&root.join("private"))).unwrap();
        let denied: Vec<_> = report
            .findings
            .iter()
            .filter(|f| f.what.starts_with("denylist"))
            .map(|f| f.path.clone())
            .collect();
        assert_eq!(denied, vec![PathBuf::from("src/a.rs")]);
        // without a denylist the scan is skipped with a notice
        let report = scan(root, None).unwrap();
        assert!(report.notices.iter().any(|n| n.contains("skipped")));
    }
}
