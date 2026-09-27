//! Privacy check for the public repository (spec 9.1).
//!
//! Three scans:
//!
//! 1. **Hashed denylist** (tracked files; always runs, including public CI):
//!    `tests/privacy/denylist.sha256` holds a salt and the salted SHA-256 of each
//!    private term, normalized with [`normalize_term`] (NFKC, lowercase,
//!    collapsed whitespace). Each tracked text file is split into paragraphs
//!    (runs of non-blank lines, so a term broken across a line wrap is still
//!    found), tokenized into alphanumeric words, and every 1 to 5 word n-gram is
//!    hashed joined by a space, a hyphen, and an underscore; any hit fails.
//!    Findings never echo the term.
//! 2. **Plaintext denylist** (whole worktree; only when the private root is
//!    present): the same n-gram matching against
//!    `<private>/privacy-denylist.txt`, with no n-gram length limit.
//! 3. **Built-in** (tracked files): RFC1918 IPv4 addresses and absolute user
//!    home paths (`/Users/<name>`, `/home/<name>`, any case), except placeholder
//!    names such as `you` or `user`.
//!
//! Excluded everywhere: `.git/`, `target/`, the private root when it is inside
//! the checkout, the scanner's own configuration under `tests/privacy/`, and
//! binary files (a NUL byte in the first 8 KiB).
//!
//! `tests/privacy/allowlist.txt` lists n-grams that are known to be generic in a
//! specific file (`path: phrase`) or everywhere (`phrase`).
//!
//! **Hashes are not secrecy for dictionary words.** Anyone can hash a word list
//! with the committed salt and learn which common words are on the list. The
//! plaintext list should therefore favor names and specific identifiers; mark
//! a guessable term with a leading `!` to keep it in the plaintext scan only
//! (the `hash_denylist` example skips it).

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

use crate::error::{EvalError, Result};

/// File name of the plaintext denylist inside the private root.
pub const DENYLIST_FILE: &str = "privacy-denylist.txt";
/// Scanner configuration directory, relative to the repository root.
pub const CONFIG_DIR: &str = "tests/privacy";
/// Hashed denylist, relative to the repository root.
pub const HASHED_DENYLIST: &str = "tests/privacy/denylist.sha256";
/// Allowlist, relative to the repository root.
pub const ALLOWLIST: &str = "tests/privacy/allowlist.txt";
/// Longest n-gram checked by the hashed scan.
pub const MAX_HASHED_NGRAM: usize = 5;
/// Joiners tried for every n-gram.
pub const JOINERS: [&str; 3] = [" ", "-", "_"];

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
    /// What matched (denylist terms are not echoed).
    pub what: String,
}

/// NFKC, lowercase, whitespace collapsed.
pub fn normalize_term(s: &str) -> String {
    let n: String = s.nfkc().collect::<String>().to_lowercase();
    n.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A plaintext denylist entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term {
    /// Normalized term.
    pub text: String,
    /// Plaintext scan only (marked `!`): excluded from the hashed list.
    pub plaintext_only: bool,
}

/// Reads a plaintext denylist: `#` comments, blank lines skipped, `!` prefix
/// marks a plaintext-only term.
pub fn parse_denylist(text: &str) -> Vec<Term> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(|l| match l.strip_prefix('!') {
            Some(rest) => Term {
                text: normalize_term(rest),
                plaintext_only: true,
            },
            None => Term {
                text: normalize_term(l),
                plaintext_only: false,
            },
        })
        .filter(|t| !t.text.is_empty())
        .collect()
}

/// Forms a term is matched in: its normalized text and its canonical token form.
pub fn term_forms(normalized: &str) -> Vec<String> {
    let canonical = tokenize_line(normalized).join(" ");
    let mut out = vec![normalized.to_string()];
    if !canonical.is_empty() && canonical != normalized {
        out.push(canonical);
    }
    out
}

/// Words of a term as the tokenizer would split them.
pub fn term_words(term: &str) -> usize {
    tokenize_line(term).len()
}

/// Salted hash of a normalized term: hex SHA-256 of `salt ":" term`.
pub fn hash_term(salt: &str, normalized: &str) -> String {
    let mut h = Sha256::new();
    h.update(salt.as_bytes());
    h.update(b":");
    h.update(normalized.as_bytes());
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The committed hashed denylist.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HashedDenylist {
    /// Salt (hex).
    pub salt: String,
    /// Hex hashes.
    pub hashes: BTreeSet<String>,
}

impl HashedDenylist {
    /// Parses the file format written by [`HashedDenylist::render`].
    pub fn parse(text: &str) -> std::result::Result<Self, String> {
        let mut out = HashedDenylist::default();
        for line in text.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(salt) = line.strip_prefix("salt:") {
                out.salt = salt.trim().to_string();
            } else if line.len() == 64 && line.bytes().all(|b| b.is_ascii_hexdigit()) {
                out.hashes.insert(line.to_ascii_lowercase());
            } else {
                return Err(format!("unexpected line in hashed denylist: {line}"));
            }
        }
        if out.salt.is_empty() {
            return Err("hashed denylist has no salt line".into());
        }
        Ok(out)
    }

    /// Builds the list from plaintext terms (plaintext-only terms skipped). Each
    /// term is stored as its normalized text and as its canonical token form
    /// (words joined by single spaces), so terms mixing joiners still match.
    pub fn from_terms(salt: &str, terms: &[Term]) -> Self {
        Self {
            salt: salt.to_string(),
            hashes: terms
                .iter()
                .filter(|t| !t.plaintext_only)
                .flat_map(|t| term_forms(&t.text))
                .map(|f| hash_term(salt, &f))
                .collect(),
        }
    }

    /// File contents, deterministic (sorted hashes).
    pub fn render(&self) -> String {
        let mut s = String::from(
            "# glassrip privacy denylist: salted SHA-256 of normalized private terms\n\
             # (NFKC, lowercase, collapsed whitespace), hash = sha256(salt \":\" term).\n\
             # Regenerate from the private plaintext list:\n\
             #   cargo run -p glassrip-eval --example hash_denylist -- <private denylist>\n\
             # Dictionary words are guessable from these hashes; keep generic words out\n\
             # of the hashed list (prefix them with ! in the plaintext list).\n",
        );
        s.push_str(&format!("salt: {}\n", self.salt));
        for h in &self.hashes {
            s.push_str(h);
            s.push('\n');
        }
        s
    }
}

/// Allowlisted n-grams: global and per file (relative path).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Allowlist {
    global: BTreeSet<String>,
    per_file: BTreeMap<PathBuf, BTreeSet<String>>,
}

impl Allowlist {
    /// Parses `phrase` or `path: phrase` lines (`#` comments).
    pub fn parse(text: &str) -> Self {
        let mut out = Allowlist::default();
        for line in text
            .lines()
            .map(|l| l.split('#').next().unwrap_or("").trim())
        {
            if line.is_empty() {
                continue;
            }
            match line.split_once(": ") {
                Some((path, phrase)) => {
                    out.per_file
                        .entry(PathBuf::from(path.trim()))
                        .or_default()
                        .insert(canonical(phrase));
                }
                None => {
                    out.global.insert(canonical(line));
                }
            }
        }
        out
    }

    /// True when the space-joined n-gram is allowed in `path`.
    pub fn allows(&self, path: &Path, gram: &str) -> bool {
        self.global.contains(gram) || self.per_file.get(path).is_some_and(|s| s.contains(gram))
    }
}

/// Space-joined token form of a phrase.
fn canonical(phrase: &str) -> String {
    tokenize_line(phrase).join(" ")
}

/// Alphanumeric tokens of a line after NFKC and lowercasing.
pub fn tokenize_line(line: &str) -> Vec<String> {
    let n: String = line.nfkc().collect::<String>().to_lowercase();
    n.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// Paragraphs of a text as token lists with the 1-based line of each token.
pub fn paragraphs(text: &str) -> Vec<Vec<(String, usize)>> {
    let mut out = Vec::new();
    let mut cur: Vec<(String, usize)> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        cur.extend(tokenize_line(line).into_iter().map(|t| (t, i + 1)));
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Every n-gram (1..=`max_n` words) of a paragraph: (line, space-joined form,
/// joined variants).
fn ngrams(par: &[(String, usize)], max_n: usize) -> Vec<(usize, String, Vec<String>)> {
    let mut out = Vec::new();
    for i in 0..par.len() {
        for n in 1..=max_n.min(par.len() - i) {
            let words: Vec<&str> = par[i..i + n].iter().map(|(t, _)| t.as_str()).collect();
            let variants: Vec<String> = if n == 1 {
                vec![words[0].to_string()]
            } else {
                JOINERS.iter().map(|j| words.join(j)).collect()
            };
            out.push((par[i].1, words.join(" "), variants));
        }
    }
    out
}

/// How n-grams are matched.
pub enum Matcher<'a> {
    /// Plaintext terms (normalized).
    Plain(&'a HashSet<String>, usize),
    /// Salted hashes.
    Hashed(&'a HashedDenylist),
}

impl Matcher<'_> {
    fn max_n(&self) -> usize {
        match self {
            Matcher::Plain(_, n) => (*n).max(1),
            Matcher::Hashed(_) => MAX_HASHED_NGRAM,
        }
    }

    fn hit(&self, variants: &[String]) -> bool {
        match self {
            Matcher::Plain(set, _) => variants.iter().any(|v| set.contains(v)),
            Matcher::Hashed(list) => variants
                .iter()
                .any(|v| list.hashes.contains(&hash_term(&list.salt, v))),
        }
    }
}

/// Denylist scan of one text.
pub fn scan_text(
    rel: &Path,
    text: &str,
    matcher: &Matcher<'_>,
    allow: &Allowlist,
    label: &str,
) -> Vec<Finding> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for par in paragraphs(text) {
        for (line, gram, variants) in ngrams(&par, matcher.max_n()) {
            if allow.allows(rel, &gram) {
                continue;
            }
            if matcher.hit(&variants) && seen.insert((line, gram.split(' ').count())) {
                out.push(Finding {
                    path: rel.to_path_buf(),
                    line,
                    what: format!("{label} term ({} word n-gram)", gram.split(' ').count()),
                });
            }
        }
    }
    out
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

/// Absolute user home paths found in `line`, any case (placeholders excluded).
pub fn home_paths(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    // ASCII lowercasing keeps byte offsets, so matches index the original line.
    let lower = line.to_ascii_lowercase();
    for prefix in ["/users/", "/home/"] {
        let mut start = 0;
        while let Some(pos) = lower[start..].find(prefix) {
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
                out.push(format!("{}{name}", &line[at..at + prefix.len()]));
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
pub fn scan_files(
    root: &Path,
    files: &[PathBuf],
    matcher: &Matcher<'_>,
    allow: &Allowlist,
    label: &str,
) -> Vec<Finding> {
    let mut out = Vec::new();
    for f in files {
        let Some(text) = read_text(f) else { continue };
        out.extend(scan_text(&rel(root, f), &text, matcher, allow, label));
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
    /// Files scanned by the hashed denylist scan.
    pub hashed_files: usize,
    /// Hashes in the committed list.
    pub hashed_terms: usize,
    /// Files scanned by the plaintext denylist scan.
    pub denylist_files: usize,
    /// Files scanned by the built-in scan.
    pub builtin_files: usize,
}

/// Runs every scan for the repository at `root`.
pub fn scan(root: &Path, private_root: Option<&Path>) -> Result<ScanReport> {
    let mut report = ScanReport::default();
    let root = root.canonicalize().map_err(|e| EvalError::io(root, e))?;
    let mut skip = vec![root.join(CONFIG_DIR)];
    if let Some(p) = private_root.and_then(|p| p.canonicalize().ok()) {
        if p.starts_with(&root) {
            skip.push(p);
        }
    }
    let allow = match read_text(&root.join(ALLOWLIST)) {
        Some(t) => Allowlist::parse(&t),
        None => Allowlist::default(),
    };
    let tracked = match tracked_files(&root) {
        Some(t) => t,
        None => {
            report
                .notices
                .push("git unavailable: tracked-file scans cover the whole worktree".into());
            worktree_files(&root, &skip)?
        }
    };
    let tracked: Vec<PathBuf> = tracked
        .into_iter()
        .filter(|p| !skip.iter().any(|s| p.starts_with(s)))
        .collect();

    // 1. Hashed denylist (always).
    match read_text(&root.join(HASHED_DENYLIST)) {
        Some(text) => {
            let list = HashedDenylist::parse(&text)
                .map_err(|e| EvalError::Config(format!("{HASHED_DENYLIST}: {e}")))?;
            report.hashed_files = tracked.len();
            report.hashed_terms = list.hashes.len();
            report.findings.extend(scan_files(
                &root,
                &tracked,
                &Matcher::Hashed(&list),
                &allow,
                "hashed denylist",
            ));
        }
        None => report.notices.push(format!(
            "hashed denylist scan skipped: {HASHED_DENYLIST} not found"
        )),
    }

    // 2. Plaintext denylist (private root only).
    match private_root.map(|p| p.join(DENYLIST_FILE)).filter(|p| p.is_file()) {
        Some(list) => {
            let terms = parse_denylist(&crate::error::read_to_string(&list)?);
            let set: HashSet<String> = terms.iter().flat_map(|t| term_forms(&t.text)).collect();
            let max_n = terms.iter().map(|t| term_words(&t.text)).max().unwrap_or(1);
            let files = worktree_files(&root, &skip)?;
            report.denylist_files = files.len();
            report.findings.extend(scan_files(&root, &files, &Matcher::Plain(&set, max_n), &allow, "plaintext denylist"));
        }
        None => report.notices.push(format!(
            "plaintext denylist scan skipped: no {DENYLIST_FILE} in the private fixtures root (set eval.private_fixtures or GLASSRIP_PRIVATE_FIXTURES)"
        )),
    }

    // 3. Built-in checks.
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
        assert!(private_ipv4("version 10.2.3").is_empty());
        assert!(private_ipv4(&format!("{}.5", ip(10, 1, 2, 3))).is_empty());
        assert!(private_ipv4(&format!("{}.", ip(10, 0, 0, 1))).len() == 1);
    }

    #[test]
    fn detects_home_paths_any_case_except_placeholders() {
        let real = format!("{}{}", "/Users/", "alice/code");
        assert_eq!(home_paths(&real), vec![format!("{}{}", "/Users/", "alice")]);
        let upper = format!("{}{}", "/USERS/", "bob");
        assert_eq!(home_paths(&upper), vec![upper.clone()]);
        let mixed = format!("{}{}", "/Home/", "carol/x");
        assert_eq!(home_paths(&mixed).len(), 1);
        assert!(home_paths(&format!("{}{}", "/home/", "runner/work")).is_empty());
        assert!(home_paths(&format!("{}{}", "/Users/", "you/project")).is_empty());
        assert!(home_paths(&format!("{}{}", "/home/", "$USER")).is_empty());
        assert!(home_paths("~/Documents and /usr/local").is_empty());
    }

    #[test]
    fn normalization_and_parsing() {
        // NFKC folds the full-width letters; case and whitespace are normalized.
        assert_eq!(normalize_term("  Ｑuorra   LABS "), "quorra labs");
        let terms = parse_denylist("# c\n Quorra Labs \n!zeta # generic\n\n");
        assert_eq!(
            terms,
            vec![
                Term {
                    text: "quorra labs".into(),
                    plaintext_only: false
                },
                Term {
                    text: "zeta".into(),
                    plaintext_only: true
                }
            ]
        );
        assert_eq!(term_words("quorra_labs-api"), 3);
    }

    #[test]
    fn hashed_list_round_trip_and_skips_plaintext_only() {
        let terms = parse_denylist("Quorra Labs\n!zeta\nnimbus_core\n");
        let list = HashedDenylist::from_terms("00ff", &terms);
        // "quorra labs" (one form) + "nimbus_core" and "nimbus core" (two forms)
        assert_eq!(list.hashes.len(), 3);
        assert!(list.hashes.contains(&hash_term("00ff", "quorra labs")));
        let back = HashedDenylist::parse(&list.render()).unwrap();
        assert_eq!(back, list);
        assert!(!list.render().contains("quorra"));
        assert!(HashedDenylist::parse("abc\n").is_err());
        assert!(HashedDenylist::parse(&"a".repeat(64)).is_err()); // no salt
    }

    fn hashed(terms: &str) -> HashedDenylist {
        HashedDenylist::from_terms("5a17", &parse_denylist(terms))
    }

    #[test]
    fn hashed_ngrams_variants_and_paragraph_joins() {
        let list = hashed("Quorra Labs\nnimbus_core\nsky-hook relay\n");
        let m = Matcher::Hashed(&list);
        let allow = Allowlist::default();
        let p = Path::new("f.md");
        // case-insensitive, punctuation between words
        assert_eq!(
            scan_text(p, "See QUORRA, labs here", &m, &allow, "h").len(),
            1
        );
        // underscore-joined term found from its tokens
        assert_eq!(
            scan_text(p, "let x = nimbus_core::run();", &m, &allow, "h").len(),
            1
        );
        // a term split by a line wrap inside one paragraph is found, on the first line
        let wrapped = scan_text(p, "the sky-hook\nrelay goes", &m, &allow, "h");
        assert_eq!(wrapped.len(), 1);
        assert_eq!(wrapped[0].line, 1);
        // a blank line ends the paragraph
        assert!(scan_text(p, "quorra\n\nlabs", &m, &allow, "h").is_empty());
        // whole words only
        assert!(scan_text(p, "quorralabs nimbus_cores", &m, &allow, "h").is_empty());
        // findings never echo the term
        assert!(!wrapped[0].what.contains("sky"));
    }

    #[test]
    fn hashed_scan_matches_phrases_up_to_the_ngram_limit() {
        let list = hashed("alpha bravo charlie delta echo\nkilo lima mike november oscar papa\n");
        let m = Matcher::Hashed(&list);
        let allow = Allowlist::default();
        let p = Path::new("f.md");
        assert_eq!(MAX_HASHED_NGRAM, 5);
        // a five-word phrase is found inside a longer sentence, across a wrap
        let hits = scan_text(
            p,
            "so Alpha, bravo charlie\ndelta-echo foxtrot",
            &m,
            &allow,
            "h",
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 1);
        assert!(hits[0].what.contains("5 word"));
        // a missing word breaks the phrase
        assert!(scan_text(p, "alpha bravo delta echo", &m, &allow, "h").is_empty());
        // a six-word phrase is past the hashed limit (the plaintext scan has none)
        assert!(scan_text(p, "kilo lima mike november oscar papa", &m, &allow, "h").is_empty());
    }

    #[test]
    fn allowlist_per_file_and_global() {
        let list = hashed("zeta\nquorra\n");
        let m = Matcher::Hashed(&list);
        let allow = Allowlist::parse("words/common.txt: zeta\n# comment\nquorra\n");
        assert!(scan_text(
            Path::new("words/common.txt"),
            "alpha\nzeta\n",
            &m,
            &allow,
            "h"
        )
        .is_empty());
        assert_eq!(
            scan_text(Path::new("src/a.rs"), "zeta", &m, &allow, "h").len(),
            1
        );
        assert!(scan_text(Path::new("src/a.rs"), "Quorra", &m, &allow, "h").is_empty());
    }

    #[test]
    fn plaintext_matcher_has_no_length_limit() {
        let terms: HashSet<String> = ["one two three four five".to_string()].into();
        let m = Matcher::Plain(&terms, 5);
        let hits = scan_text(
            Path::new("x"),
            "One two three\nfour five!",
            &m,
            &Allowlist::default(),
            "p",
        );
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn scan_walks_skips_and_reports() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for d in ["src", "target", "private", CONFIG_DIR] {
            fs_err::create_dir_all(root.join(d)).unwrap();
        }
        fs_err::write(root.join("src/a.rs"), "let x = 1; // Quorra here\n").unwrap();
        fs_err::write(root.join("src/b.rs"), "zeta only\n").unwrap();
        fs_err::write(root.join("target/b.rs"), "Quorra\n").unwrap();
        fs_err::write(root.join("private/c.txt"), "Quorra\n").unwrap();
        fs_err::write(root.join("private").join(DENYLIST_FILE), "quorra\n!zeta\n").unwrap();
        fs_err::write(root.join(ALLOWLIST), "quorra-free\nzeta\n").unwrap();
        fs_err::write(root.join("bin.dat"), [0u8, 81, 117, 111, 114, 114, 97]).unwrap();
        fs_err::write(root.join(HASHED_DENYLIST), hashed("quorra\n").render()).unwrap();
        let report = scan(root, Some(&root.join("private"))).unwrap();
        let mut paths: Vec<(String, PathBuf)> = report
            .findings
            .iter()
            .map(|f| (f.what.clone(), f.path.clone()))
            .collect();
        paths.sort();
        // hashed and plaintext scans both find src/a.rs; zeta is allowlisted;
        // target/, the private root, the config dir, and binaries are skipped
        assert_eq!(
            paths,
            vec![
                (
                    "hashed denylist term (1 word n-gram)".to_string(),
                    PathBuf::from("src/a.rs")
                ),
                (
                    "plaintext denylist term (1 word n-gram)".to_string(),
                    PathBuf::from("src/a.rs")
                ),
            ]
        );
        assert_eq!(report.hashed_terms, 1);
        // without the private root only the hashed scan and built-ins run
        // (the private folder is then ordinary worktree content and is flagged too)
        let report = scan(root, None).unwrap();
        let mut hit: Vec<PathBuf> = report.findings.iter().map(|f| f.path.clone()).collect();
        hit.sort();
        assert_eq!(
            hit,
            vec![
                PathBuf::from("private/c.txt"),
                PathBuf::from("private").join(DENYLIST_FILE),
                PathBuf::from("src/a.rs")
            ]
        );
        assert!(report.findings.iter().all(|f| f.what.starts_with("hashed")));
        assert!(report
            .notices
            .iter()
            .any(|n| n.contains("plaintext denylist scan skipped")));
    }
}
