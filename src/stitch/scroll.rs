use std::collections::HashSet;

use super::dedup::block_similarity_from_hashes;
use super::{line_hash, normalized_line_hash};

pub fn stitch_scroll_sequence(code_blocks: &[String]) -> String {
    if code_blocks.is_empty() {
        return String::new();
    }
    if code_blocks.len() == 1 {
        return code_blocks[0].clone();
    }

    let mut full_lines: Vec<&str> = code_blocks[0].split('\n').collect();

    for i in 1..code_blocks.len() {
        let curr_lines: Vec<&str> = code_blocks[i].split('\n').collect();
        let tail_start = full_lines.len().saturating_sub(curr_lines.len());
        let tail = &full_lines[tail_start..];
        let overlap = find_overlap(tail, &curr_lines, 3);

        if overlap > 0 {
            full_lines.extend_from_slice(&curr_lines[overlap..]);
        } else {
            let reverse = find_reverse_overlap(&full_lines, &curr_lines);
            if reverse < 0 {
                full_lines.extend_from_slice(&curr_lines);
            }
        }
    }

    full_lines.join("\n")
}

pub fn stitch_all_revisions(revisions: &[String]) -> String {
    let mut accumulated: Vec<String> = Vec::new();
    let mut accumulated_norm: Vec<String> = Vec::new();
    let mut accepted_hashes: Vec<Vec<String>> = Vec::new();

    for revision in revisions {
        let lines: Vec<&str> = revision.lines().collect();
        let non_empty = lines.iter().filter(|l| !l.trim().is_empty()).count();
        if non_empty < 3 {
            continue;
        }

        let hashes: Vec<String> = lines.iter().map(|line| normalized_line_hash(line)).collect();
        if accepted_hashes
            .iter()
            .any(|prior| block_similarity_from_hashes(prior, &hashes) > 0.75)
        {
            continue;
        }
        accepted_hashes.push(hashes);

        if accumulated.is_empty() {
            accumulated = lines.iter().map(|l| l.to_string()).collect();
            accumulated_norm = accumulated.iter().map(|l| normalized_line_hash(l)).collect();
            continue;
        }

        let tail_start = accumulated_norm.len().saturating_sub(200);
        let tail_norm = &accumulated_norm[tail_start..];
        let rev_norm: Vec<String> = lines.iter().map(|l| normalized_line_hash(l)).collect();
        let overlap = find_overlap_from_hashes(tail_norm, &rev_norm, 3);

        if overlap > 0 {
            accumulated.extend(lines[overlap..].iter().map(|l| l.to_string()));
            accumulated_norm.extend(rev_norm[overlap..].iter().cloned());
        } else {
            let reverse = find_reverse_overlap_from_hashes(&accumulated_norm, &rev_norm);
            if reverse >= 0 {
                let pos = reverse as usize;
                let acc_from_pos = &accumulated_norm[pos..];
                let match_len = rev_norm
                    .iter()
                    .zip(acc_from_pos.iter())
                    .take_while(|(c, a)| c == a)
                    .count();
                if match_len < lines.len() {
                    accumulated.extend(lines[match_len..].iter().map(|l| l.to_string()));
                    accumulated_norm.extend(rev_norm[match_len..].iter().cloned());
                }
            } else {
                accumulated.extend(lines.iter().map(|l| l.to_string()));
                accumulated_norm.extend(rev_norm);
            }
        }
    }

    accumulated.join("\n")
}

pub fn compute_diff(old_code: Option<&str>, new_code: &str) -> String {
    let old = match old_code {
        Some(s) => s,
        None => return String::new(),
    };

    let diff = similar::TextDiff::from_lines(old, new_code);
    let unified = diff.unified_diff().context_radius(1).to_string();

    unified.lines().skip(2).collect::<Vec<_>>().join("\n")
}

fn find_overlap<S1: AsRef<str>, S2: AsRef<str>>(
    prev_lines: &[S1],
    curr_lines: &[S2],
    min_overlap: usize,
) -> usize {
    if prev_lines.is_empty() || curr_lines.is_empty() {
        return 0;
    }

    let prev_hashes: Vec<String> = prev_lines.iter().map(|l| line_hash(l.as_ref())).collect();
    let curr_hashes: Vec<String> = curr_lines.iter().map(|l| line_hash(l.as_ref())).collect();
    let max_check = prev_lines.len().min(curr_lines.len());

    for n in (min_overlap..=max_check).rev() {
        let prev_tail = &prev_hashes[prev_hashes.len() - n..];
        let curr_head = &curr_hashes[..n];
        if prev_tail == curr_head {
            return n;
        }
    }

    let fuzzy_max = max_check.min(15);
    let blank_hash = line_hash("");

    for n in (min_overlap..=fuzzy_max).rev() {
        let prev_tail = &prev_hashes[prev_hashes.len() - n..];
        let curr_head = &curr_hashes[..n];

        let match_count = prev_tail
            .iter()
            .zip(curr_head.iter())
            .filter(|(a, b)| a == b)
            .count();

        let non_blank = prev_tail.iter().filter(|h| **h != blank_hash).count();

        if match_count as f64 >= n as f64 * 0.85 && non_blank >= min_overlap {
            return n;
        }
    }

    0
}

fn find_reverse_overlap<S1: AsRef<str>, S2: AsRef<str>>(
    full_lines: &[S1],
    curr_lines: &[S2],
) -> i64 {
    if curr_lines.len() < 3 {
        return -1;
    }

    let window_size = curr_lines.len().min(10);
    let curr_hashes: Vec<String> = curr_lines[..window_size]
        .iter()
        .map(|l| line_hash(l.as_ref()))
        .collect();

    let blank_hash = line_hash("");
    let distinct: HashSet<&String> = curr_hashes.iter().filter(|h| **h != blank_hash).collect();
    if distinct.len() < 2 {
        return -1;
    }

    if full_lines.len() < window_size {
        return -1;
    }

    for i in 0..=full_lines.len() - window_size {
        let window: Vec<String> = full_lines[i..i + window_size]
            .iter()
            .map(|l| line_hash(l.as_ref()))
            .collect();
        if window == curr_hashes {
            return i as i64;
        }
    }

    -1
}

fn find_overlap_from_hashes(prev_hashes: &[String], curr_hashes: &[String], min_overlap: usize) -> usize {
    if prev_hashes.is_empty() || curr_hashes.is_empty() {
        return 0;
    }

    let max_check = prev_hashes.len().min(curr_hashes.len());
    for n in (min_overlap..=max_check).rev() {
        let prev_tail = &prev_hashes[prev_hashes.len() - n..];
        let curr_head = &curr_hashes[..n];
        if prev_tail == curr_head {
            return n;
        }
    }

    let blank_hash = normalized_line_hash("");
    let fuzzy_max = max_check.min(15);
    for n in (min_overlap..=fuzzy_max).rev() {
        let prev_tail = &prev_hashes[prev_hashes.len() - n..];
        let curr_head = &curr_hashes[..n];
        let match_count = prev_tail.iter().zip(curr_head.iter()).filter(|(a, b)| a == b).count();
        let non_blank = prev_tail.iter().filter(|h| **h != blank_hash).count();
        if match_count as f64 >= n as f64 * 0.85 && non_blank >= min_overlap {
            return n;
        }
    }

    0
}

fn find_reverse_overlap_from_hashes(full_hashes: &[String], curr_hashes: &[String]) -> i64 {
    if curr_hashes.len() < 3 {
        return -1;
    }

    let window_size = curr_hashes.len().min(10);
    let curr_window = &curr_hashes[..window_size];

    let blank_hash = normalized_line_hash("");
    let distinct: HashSet<&String> = curr_window.iter().filter(|h| **h != blank_hash).collect();
    if distinct.len() < 2 {
        return -1;
    }

    if full_hashes.len() < window_size {
        return -1;
    }

    for i in 0..=full_hashes.len() - window_size {
        if full_hashes[i..i + window_size] == *curr_window {
            return i as i64;
        }
    }

    -1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_blocks() {
        assert_eq!(stitch_scroll_sequence(&[]), "");
    }

    #[test]
    fn single_block() {
        let blocks = vec!["line1\nline2\nline3".to_string()];
        assert_eq!(stitch_scroll_sequence(&blocks), "line1\nline2\nline3");
    }

    #[test]
    fn overlapping_blocks() {
        let blocks = vec![
            "a\nb\nc\nd\ne".to_string(),
            "c\nd\ne\nf\ng".to_string(),
        ];
        assert_eq!(stitch_scroll_sequence(&blocks), "a\nb\nc\nd\ne\nf\ng");
    }

    #[test]
    fn no_overlap_appends() {
        let blocks = vec!["a\nb\nc".to_string(), "x\ny\nz".to_string()];
        assert_eq!(stitch_scroll_sequence(&blocks), "a\nb\nc\nx\ny\nz");
    }

    #[test]
    fn compute_diff_none_returns_empty() {
        assert_eq!(compute_diff(None, "hello"), "");
    }

    #[test]
    fn compute_diff_identical() {
        assert_eq!(compute_diff(Some("a\nb"), "a\nb"), "");
    }

    #[test]
    fn compute_diff_changed() {
        let result = compute_diff(Some("a\nb\nc"), "a\nB\nc");
        assert!(result.contains("-b"));
        assert!(result.contains("+B"));
    }

    #[test]
    fn compute_diff_preserves_triple_dash_content() {
        let result = compute_diff(Some("a\n---\nc"), "a\n---\nd");
        assert!(result.contains("-c"));
        assert!(result.contains("+d"));
    }

    #[test]
    fn line_hash_deterministic() {
        let h1 = line_hash("test");
        let h2 = line_hash("test");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 12);
    }

    #[test]
    fn line_hash_differs() {
        assert_ne!(line_hash("foo"), line_hash("bar"));
    }

    #[test]
    fn all_revisions_skips_empty_and_short_blocks() {
        let blocks = vec![
            "fn main() {\n    let x = 1;\n    let y = 2;\n}".to_string(),
            "".to_string(),
            "garbage".to_string(),
            "one\ntwo".to_string(),
            "fn helper() {\n    return 42;\n    // done\n}".to_string(),
        ];
        let result = stitch_all_revisions(&blocks);
        assert!(result.contains("fn main()"));
        assert!(result.contains("fn helper()"));
        assert!(!result.contains("garbage"));
        assert!(!result.contains("one\ntwo"));
    }

    #[test]
    fn all_revisions_overlap_against_accumulated_tail() {
        let blocks = vec![
            "line1\nline2\nline3\nline4\nline5".to_string(),
            "line3\nline4\nline5\nline6\nline7".to_string(),
            "line5\nline6\nline7\nline8\nline9".to_string(),
        ];
        let result = stitch_all_revisions(&blocks);
        let result_lines: Vec<&str> = result.lines().collect();
        assert_eq!(
            result_lines,
            vec![
                "line1", "line2", "line3", "line4", "line5", "line6", "line7", "line8", "line9"
            ]
        );
    }

    #[test]
    fn all_revisions_skips_single_line_blocks() {
        let blocks = vec![
            "alpha\nbeta\ngamma".to_string(),
            "just one line".to_string(),
            "delta\nepsilon\nzeta".to_string(),
        ];
        let result = stitch_all_revisions(&blocks);
        assert!(result.contains("alpha"));
        assert!(result.contains("delta"));
        assert!(!result.contains("just one line"));
    }

    #[test]
    fn all_revisions_skips_near_duplicate_revision() {
        let first = (1..=20)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let second = first.replacen("line1", "changed", 1);
        assert_eq!(stitch_all_revisions(&[first.clone(), second]), first);
    }

    #[test]
    fn all_revisions_keeps_distinct_revisions() {
        let blocks = vec![
            "alpha\nbeta\ngamma".to_string(),
            "delta\nepsilon\nzeta".to_string(),
        ];
        assert_eq!(stitch_all_revisions(&blocks), blocks.join("\n"));
    }

    #[test]
    fn all_revisions_checks_full_history() {
        let first = "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta".to_string();
        let middle = (1..=250)
            .map(|i| format!("middle{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let last = first.replacen("alpha", "changed", 1);
        let expected = format!("{first}\n{middle}");
        assert_eq!(stitch_all_revisions(&[first, middle, last]), expected);
    }

    #[test]
    fn all_revisions_keeps_dissimilar_revisions() {
        let blocks = vec![
            "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\neta\ntheta".to_string(),
            "changed1\nchanged2\nchanged3\ndelta\nepsilon\nzeta\neta\ntheta".to_string(),
        ];
        assert_eq!(stitch_all_revisions(&blocks), blocks.join("\n"));
    }

    #[test]
    fn reverse_overlap_appends_new_suffix() {
        let blocks = vec![
            "line1\nline2\nline3\nline4\nline5".to_string(),
            "line6\nline7\nline8\nline9\nline10".to_string(),
            "line1\nline2\nline3\nline4\nline5\nnew_line\nanother_new_line".to_string(),
        ];
        let result = stitch_all_revisions(&blocks);
        assert!(result.contains("new_line"));
        assert!(result.contains("another_new_line"));
    }

    #[test]
    fn reverse_overlap_rejects_structural_false_match() {
        let blocks = vec![
            "fn a() {\n    x\n    y\n}\nfn b() {\n    z\n}".to_string(),
            "}\n}\n}\n}\n}\n}\n}".to_string(),
        ];
        let result = stitch_all_revisions(&blocks);
        assert!(result.contains("fn a()"));
    }
}
