use std::collections::{HashMap, HashSet};

use super::normalized_line_hash;

pub fn dedup_blocks(text: &str) -> String {
    let blocks = split_into_blocks(text);
    if blocks.len() <= 1 {
        return text.to_string();
    }

    let hashed: Vec<Vec<String>> = blocks
        .iter()
        .map(|block| block.iter().map(|line| normalized_line_hash(line)).collect())
        .collect();

    let mut removed: HashSet<usize> = HashSet::new();

    for i in 0..blocks.len() {
        if removed.contains(&i) {
            continue;
        }
        for j in (i + 1)..blocks.len() {
            if removed.contains(&j) {
                continue;
            }
            let sim = block_similarity_from_hashes(&hashed[i], &hashed[j]);
            if sim > 0.85 {
                if blocks[j].len() > blocks[i].len() {
                    removed.insert(i);
                    break;
                } else {
                    removed.insert(j);
                }
            }
        }
    }

    let mut result = String::new();
    let mut first = true;
    for (idx, block) in blocks.iter().enumerate() {
        if removed.contains(&idx) {
            continue;
        }
        if !first {
            result.push_str("\n\n");
        }
        result.push_str(&block.join("\n"));
        first = false;
    }

    result
}

pub(crate) fn block_similarity_from_hashes(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let (shorter, longer) = if a.len() <= b.len() {
        (a, b)
    } else {
        (b, a)
    };

    let mut longer_counts: HashMap<&String, usize> = HashMap::new();
    for h in longer {
        *longer_counts.entry(h).or_insert(0) += 1;
    }

    let mut matches = 0usize;
    let mut used_counts: HashMap<&String, usize> = HashMap::new();
    for h in shorter {
        let used = used_counts.entry(h).or_insert(0);
        let available = longer_counts.get(h).copied().unwrap_or(0);
        if *used < available {
            matches += 1;
            *used += 1;
        }
    }

    matches as f64 / longer.len() as f64
}

pub fn dedup_passages(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < PASSAGE_WINDOW {
        return text.to_string();
    }

    let norm_hashes: Vec<String> = lines.iter().map(|l| normalized_line_hash(l)).collect();
    let blank_hash = normalized_line_hash("");
    let mut remove = vec![false; lines.len()];

    let mut i = PASSAGE_WINDOW;
    while i < lines.len().saturating_sub(PASSAGE_WINDOW - 1) {
        if remove[i] || norm_hashes[i] == blank_hash {
            i += 1;
            continue;
        }

        let window = &norm_hashes[i..i + PASSAGE_WINDOW];
        let distinct: HashSet<&String> = window.iter().filter(|h| **h != blank_hash).collect();
        if distinct.len() < 2 {
            i += 1;
            continue;
        }

        let mut found = false;
        let search_end = i.saturating_sub(PASSAGE_WINDOW) + 1;
        for j in 0..search_end {
            if norm_hashes[j..j + PASSAGE_WINDOW] == *window {
                let mut end = i + PASSAGE_WINDOW;
                while end < lines.len()
                    && j + (end - i) < lines.len()
                    && j + (end - i) < i
                    && norm_hashes[end] == norm_hashes[j + (end - i)]
                {
                    end += 1;
                }
                for k in i..end {
                    remove[k] = true;
                }
                i = end;
                found = true;
                break;
            }
        }

        if !found {
            i += 1;
        }
    }

    lines
        .iter()
        .enumerate()
        .filter(|(idx, _)| !remove[*idx])
        .map(|(_, line)| *line)
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn dedup_sections(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let sections = split_into_sections(&lines);
    if sections.len() <= 1 {
        return text.to_string();
    }

    let mut kept: Vec<&Section> = Vec::new();
    for section in &sections {
        if let Some(ref key) = section.norm_heading {
            if let Some(existing) = kept.iter_mut().find(|s| s.norm_heading.as_ref() == Some(key)) {
                if section.body_len > existing.body_len {
                    *existing = section;
                }
                continue;
            }
        }
        kept.push(section);
    }

    let mut result = String::new();
    for (i, section) in kept.iter().enumerate() {
        if i > 0 && !result.ends_with('\n') {
            result.push('\n');
        }
        for &line in &lines[section.start..section.end] {
            result.push_str(line);
            result.push('\n');
        }
    }

    result.trim_end_matches('\n').to_string()
}

struct Section {
    start: usize,
    end: usize,
    norm_heading: Option<String>,
    body_len: usize,
}

fn is_heading(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.starts_with('#') || trimmed.starts_with("@@") {
        return true;
    }
    let stripped = trimmed.trim_start_matches(|c: char| c == '*' || c == '-' || c == '>' || c.is_whitespace());
    stripped.starts_with("v") && stripped.len() > 1 && stripped.as_bytes()[1].is_ascii_digit()
}

fn split_into_sections<'a>(lines: &[&'a str]) -> Vec<Section> {
    use super::normalize_line;

    let mut sections = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if is_heading(lines[i]) {
            let norm = normalize_line(lines[i]);
            let start = i;
            i += 1;
            while i < lines.len() && !is_heading(lines[i]) {
                i += 1;
            }
            let body_len = i - start - 1;
            sections.push(Section {
                start,
                end: i,
                norm_heading: Some(norm),
                body_len,
            });
        } else {
            let start = i;
            i += 1;
            while i < lines.len() && !is_heading(lines[i]) {
                i += 1;
            }
            sections.push(Section {
                start,
                end: i,
                norm_heading: None,
                body_len: i - start,
            });
        }
    }
    sections
}

const PASSAGE_WINDOW: usize = 3;

fn split_into_blocks(text: &str) -> Vec<Vec<&str>> {
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    let mut current_block: Vec<&str> = Vec::new();

    for line in text.lines() {
        if line.trim().is_empty() {
            if !current_block.is_empty() {
                blocks.push(std::mem::take(&mut current_block));
            }
        } else {
            current_block.push(line);
        }
    }

    if !current_block.is_empty() {
        blocks.push(current_block);
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_duplicates_unchanged() {
        let input = "block one\nline two\n\nblock two\nline three";
        let result = dedup_blocks(input);
        assert_eq!(result, input);
    }

    #[test]
    fn exact_duplicate_removed() {
        let input = "## v0.21.0\n- feature A\n- feature B\n\n## v0.20.0\n- fix C\n\n## v0.21.0\n- feature A\n- feature B";
        let result = dedup_blocks(input);
        assert!(result.contains("## v0.21.0"));
        assert!(result.contains("## v0.20.0"));
        assert_eq!(
            result.matches("## v0.21.0").count(),
            1,
            "duplicate block should be removed"
        );
    }

    #[test]
    fn similar_blocks_keep_longer() {
        let block_a = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6";
        let block_b = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nextra line";
        let input = format!("{}\n\n{}", block_a, block_b);
        let result = dedup_blocks(&input);
        assert!(
            result.contains("extra line"),
            "longer block should be kept"
        );
        let line1_count = result.matches("line 1").count();
        assert_eq!(line1_count, 1, "shorter duplicate should be removed");
    }

    #[test]
    fn below_threshold_kept() {
        let block_a = "alpha\nbeta\ngamma";
        let block_b = "alpha\nbeta\ndelta\nepsilon\nzeta";
        let input = format!("{}\n\n{}", block_a, block_b);
        let result = dedup_blocks(&input);
        assert!(result.contains("gamma"));
        assert!(result.contains("zeta"));
    }

    #[test]
    fn normalized_blocks_dedup_markdown_variants() {
        let block_a = "## v0.36.0\n## Features\n- CMMC2 update: containers";
        let block_b = "@@ v0.36.0\n@@ Features\n- **CMMC2 update**: containers";
        let input = format!("{}\n\n{}", block_a, block_b);
        let result = dedup_blocks(&input);
        assert_eq!(result.matches("v0.36.0").count(), 1);
    }

    #[test]
    fn passage_dedup_removes_duplicate_passage() {
        let lines: Vec<String> = (1..=10).map(|i| format!("unique line {i}")).collect();
        let passage: Vec<String> = (1..=6).map(|i| format!("repeated {i}")).collect();
        let mut all = lines.clone();
        all.extend(passage.clone());
        all.push("middle content".to_string());
        all.extend(passage.clone());
        all.push("end content".to_string());
        let input = all.join("\n");
        let result = dedup_passages(&input);
        assert_eq!(result.matches("repeated 1").count(), 1);
        assert!(result.contains("end content"));
        assert!(result.contains("middle content"));
    }

    #[test]
    fn passage_dedup_keeps_non_duplicate() {
        let input = "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\neta\ntheta";
        assert_eq!(dedup_passages(input), input);
    }

    #[test]
    fn passage_dedup_ignores_short_matches() {
        let input = "a\nb\nx\ny\nz\na\nb\nw";
        let result = dedup_passages(input);
        assert_eq!(result.matches("\na\n").count() + result.starts_with("a\n") as usize, 2);
    }

    #[test]
    fn section_dedup_keeps_longest() {
        let input = "## v0.35.0\n- feature A\n\n## v0.34.0\n- fix B\n- fix C\n\n## v0.35.0\n- feature A\n- feature D\n- feature E";
        let result = dedup_sections(input);
        assert_eq!(result.matches("v0.35.0").count(), 1);
        assert!(result.contains("feature E"), "longer section should be kept");
        assert!(result.contains("v0.34.0"), "different section untouched");
    }

    #[test]
    fn section_dedup_normalized_headings() {
        let input = "## v0.35.0\n- feature A\n\n@@ v0.35.0\n- feature A\n- feature B\n- feature C";
        let result = dedup_sections(input);
        assert_eq!(result.matches("v0.35.0").count(), 1);
        assert!(result.contains("feature C"));
    }

    #[test]
    fn section_dedup_no_headings_unchanged() {
        let input = "line one\nline two\nline three";
        assert_eq!(dedup_sections(input), input);
    }

    #[test]
    fn section_dedup_version_pattern_heading() {
        let input = "v1.2.3\n- change A\nv2.0.0\n- other stuff\nv1.2.3\n- change A\n- change B\n- change C";
        let result = dedup_sections(input);
        assert_eq!(result.matches("v1.2.3").count(), 1);
        assert!(result.contains("change C"));
        assert!(result.contains("v2.0.0"));
    }

    #[test]
    fn repeated_lines_multiset_semantics() {
        let block_a = "}\n}\n}\n}\n}\n}\n}\n}\n}\n}";
        let block_b = "}\nalpha\nbeta\ngamma\ndelta\nepsilon\nzeta\neta\ntheta\niota\nkappa";
        let input = format!("{}\n\n{}", block_a, block_b);
        let result = dedup_blocks(&input);
        assert!(
            result.contains("alpha"),
            "block with one shared line should not be removed"
        );
    }
}
