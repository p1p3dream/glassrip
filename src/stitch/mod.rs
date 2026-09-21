pub mod clean;
pub mod dedup;
pub mod refine;
pub mod scroll;

pub use clean::{clean_hallucinations, is_hallucinated};
pub use dedup::{dedup_blocks, dedup_passages, dedup_sections};
pub use scroll::{compute_diff, stitch_all_revisions, stitch_scroll_sequence};

use md5::{Digest, Md5};

pub(crate) fn line_hash(line: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(line.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)[..12].to_string()
}

pub(crate) fn normalize_line(line: &str) -> String {
    let s = line.trim_start_matches(|c: char| c == '#' || c == '*' || c == '-' || c == '>' || c == '@' || c == '+' || c.is_whitespace());
    s.replace("**", "").trim().to_lowercase()
}

pub(crate) fn normalized_line_hash(line: &str) -> String {
    let norm = normalize_line(line);
    let mut hasher = Md5::new();
    hasher.update(norm.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)[..12].to_string()
}
