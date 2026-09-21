const GARBAGE_THRESHOLD: f64 = 0.3;
const HALLUCINATION_LINE_RATIO: f64 = 0.5;
const LENGTH_MULTIPLIER: usize = 5;
const LONG_WORD_THRESHOLD: usize = 15;
const NEUTRAL_SCORE: f64 = 0.5;

pub fn clean_hallucinations(text: &str) -> String {
    collapse_repetitions(text)
        .into_iter()
        .filter(|line| line_quality(line) >= GARBAGE_THRESHOLD)
        .collect::<Vec<_>>()
        .join("\n")
}

fn collapse_repetitions(text: &str) -> Vec<&str> {
    let lines: Vec<&str> = text.lines().collect();
    let mut collapsed = Vec::new();

    for run in lines.chunk_by(|a, b| a == b) {
        if run.len() >= 5 {
            collapsed.push(run[0]);
        } else {
            collapsed.extend_from_slice(run);
        }
    }

    collapsed
}

pub fn is_hallucinated(text: &str, expected_line_count: Option<usize>) -> bool {
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();

    if total == 0 {
        return false;
    }

    if let Some(expected) = expected_line_count {
        if expected > 0 && total > expected * LENGTH_MULTIPLIER {
            return true;
        }
    }

    let garbage_count = lines
        .iter()
        .filter(|line| line_quality(line) < GARBAGE_THRESHOLD)
        .count();

    (garbage_count as f64 / total as f64) > HALLUCINATION_LINE_RATIO
}

fn line_quality(line: &str) -> f64 {
    let trimmed = line.trim();

    if trimmed.is_empty() {
        return NEUTRAL_SCORE;
    }

    if trimmed.len() < 3 {
        return NEUTRAL_SCORE;
    }

    let mut score = 0.5;

    score += pattern_bonus(trimmed);
    score -= long_word_penalty(trimmed);
    score -= char_class_penalty(trimmed);

    score.clamp(0.0, 1.0)
}

fn pattern_bonus(line: &str) -> f64 {
    let mut bonus = 0.0;

    if is_markdown_table_row(line) {
        return 0.5;
    }

    if line.starts_with('#') || line.starts_with("```") {
        bonus += 0.4;
    }

    if line.starts_with("- ")
        || line.starts_with("* ")
        || line.starts_with("> ")
        || line.starts_with("//")
        || line.starts_with("///")
    {
        bonus += 0.3;
    }

    let code_tokens = [
        "import ", "from ", "def ", "fn ", "let ", "const ", "pub ", "use ", "struct ", "enum ",
        "impl ", "trait ", "class ", "return ", "if ", "else ", "for ", "while ", "match ",
        "async ", "await ",
    ];
    for token in &code_tokens {
        if line.contains(token) {
            bonus += 0.3;
            break;
        }
    }

    let punctuation_chars = ['=', '{', '}', '(', ')', '[', ']', ';', ':', ',', '.'];
    let punct_count = line.chars().filter(|c| punctuation_chars.contains(c)).count();
    if punct_count > 0 {
        bonus += (punct_count as f64 * 0.05).min(0.2);
    }

    bonus.min(0.5)
}

fn is_markdown_table_row(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
        return false;
    }
    trimmed.matches('|').count() >= 3
}

fn long_word_penalty(line: &str) -> f64 {
    let words: Vec<&str> = line.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }

    let long_words = words
        .iter()
        .filter(|w| {
            let stripped: String = w
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '.' || *c == ':')
                .collect();
            stripped.len() > LONG_WORD_THRESHOLD && !is_identifier(&stripped)
        })
        .count();

    if long_words == 0 {
        return 0.0;
    }

    let ratio = long_words as f64 / words.len() as f64;
    (ratio * 0.6).min(0.5)
}

fn is_identifier(word: &str) -> bool {
    if word.contains('_') || word.contains('.') || word.contains("::") {
        return true;
    }
    let has_lower = word.chars().any(|c| c.is_lowercase());
    let has_upper = word.chars().any(|c| c.is_uppercase());
    if has_lower && has_upper {
        let transitions = word
            .chars()
            .zip(word.chars().skip(1))
            .filter(|(a, b)| a.is_lowercase() && b.is_uppercase())
            .count();
        if transitions >= 1 {
            return true;
        }
    }
    false
}

fn char_class_penalty(line: &str) -> f64 {
    let alpha_only: String = line.chars().filter(|c| c.is_alphabetic()).collect();
    if alpha_only.len() < 5 {
        return 0.0;
    }

    let upper = alpha_only.chars().filter(|c| c.is_uppercase()).count() as f64;
    let total = alpha_only.len() as f64;
    let upper_ratio = upper / total;

    if upper_ratio > 0.3 && upper_ratio < 0.7 && total > 8.0 {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.iter().any(|w| is_identifier(w)) {
            return 0.0;
        }
        let deviation = 0.5 - (upper_ratio - 0.5).abs();
        return (deviation * 0.8).min(0.4);
    }

    let max_alpha_run = longest_alpha_run(line);
    if max_alpha_run > 20 {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.iter().any(|w| is_identifier(w)) {
            return 0.0;
        }
        return 0.3;
    }

    0.0
}

fn longest_alpha_run(line: &str) -> usize {
    let mut max_run = 0;
    let mut current_run = 0;

    for ch in line.chars() {
        if ch.is_alphabetic() {
            current_run += 1;
            max_run = max_run.max(current_run);
        } else {
            current_run = 0;
        }
    }

    max_run
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_repetitions_removes_long_runs() {
        let line = "# Added edge case handling for database schema integration with the system";
        let input = format!("{line}\n").repeat(50);
        assert_eq!(collapse_repetitions(&input), vec![line]);
        assert_eq!(clean_hallucinations(&input), line);
    }

    #[test]
    fn collapse_repetitions_preserves_short_runs() {
        let input = "# comment\n# comment\n# comment";
        assert_eq!(collapse_repetitions(input), vec!["# comment"; 3]);
    }

    #[test]
    fn collapse_repetitions_preserves_non_consecutive() {
        let input = "# comment\nfirst\n# comment\nsecond\n# comment\nthird\n# comment\nfourth\n# comment";
        assert_eq!(collapse_repetitions(input), input.lines().collect::<Vec<_>>());
    }

    #[test]
    fn collapse_repetitions_threshold() {
        let input = "a\na\na\na\nb\nb\nb\nb\nb\nc";
        assert_eq!(collapse_repetitions(input), vec!["a", "a", "a", "a", "b", "c"]);
        assert!(collapse_repetitions("").is_empty());
    }

    #[test]
    fn clean_preserves_trailing_blank_line() {
        assert_eq!(clean_hallucinations("# comment\n\n"), "# comment\n");
    }

    #[test]
    fn clean_collapses_repetitions_before_filtering_garbage() {
        let input = "# comment\n# comment\n# comment\nasdkjhaslkdjhqweoiruqwlekrj\n# comment\n# comment";
        assert_eq!(clean_hallucinations(input), ["# comment"; 5].join("\n"));
    }

    #[test]
    fn real_code_scores_high() {
        assert!(line_quality("fn main() {") > 0.5);
        assert!(line_quality("let x = 42;") > 0.5);
        assert!(line_quality("import os") > 0.5);
        assert!(line_quality("# Header") > 0.5);
    }

    #[test]
    fn table_rows_score_high() {
        assert!(line_quality("| Name | Value | Description |") > 0.5);
        assert!(line_quality("|---|---|---|") > 0.5);
    }

    #[test]
    fn empty_lines_are_neutral() {
        assert!((line_quality("") - NEUTRAL_SCORE).abs() < f64::EPSILON);
        assert!((line_quality("  ") - NEUTRAL_SCORE).abs() < f64::EPSILON);
    }

    #[test]
    fn short_lines_are_neutral() {
        assert!((line_quality("x") - NEUTRAL_SCORE).abs() < f64::EPSILON);
        assert!((line_quality("{}") - NEUTRAL_SCORE).abs() < f64::EPSILON);
    }

    #[test]
    fn gibberish_scores_low() {
        assert!(line_quality("Meldipgodequicedtransfomburg") < GARBAGE_THRESHOLD);
        assert!(line_quality("asdkjhaslkdjhqweoiruqwlekrj") < GARBAGE_THRESHOLD);
    }

    #[test]
    fn clean_removes_garbage() {
        let input = "fn main() {\nMeldipgodequicedtransfomburg\n    println!(\"hi\");\n}";
        let cleaned = clean_hallucinations(input);
        assert!(!cleaned.contains("Meldipgodequiced"));
        assert!(cleaned.contains("fn main()"));
        assert!(cleaned.contains("println!"));
    }

    #[test]
    fn hallucination_detection_by_ratio() {
        let garbage =
            "Meldipgodequicedtransfomburg\nasdkjhaslkdjhqweoiruqwlekrj\nxyzabcdefghijklmnopqrstuvw\n";
        assert!(is_hallucinated(garbage, None));
    }

    #[test]
    fn hallucination_detection_by_length() {
        let text = "line\n".repeat(60);
        assert!(is_hallucinated(&text, Some(10)));
    }

    #[test]
    fn real_content_not_flagged() {
        let real = "# Module\n\nfn process(x: i32) -> i32 {\n    x + 1\n}\n";
        assert!(!is_hallucinated(real, None));
        assert!(!is_hallucinated(real, Some(5)));
    }

    #[test]
    fn camel_case_identifiers_not_penalized() {
        assert!(
            line_quality("databaseConnectionPoolSize:") >= GARBAGE_THRESHOLD,
            "camelCase identifier should not be filtered"
        );
        assert!(
            line_quality("x = authenticationServiceConfiguration") >= GARBAGE_THRESHOLD,
            "camelCase assignment should not be filtered"
        );
    }

    #[test]
    fn snake_case_identifiers_not_penalized() {
        assert!(
            line_quality("database_connection_pool_size = 10") >= GARBAGE_THRESHOLD,
            "snake_case identifier should not be filtered"
        );
    }

    #[test]
    fn dotted_paths_not_penalized() {
        assert!(
            line_quality("com.example.authentication.service") >= GARBAGE_THRESHOLD,
            "dotted path should not be filtered"
        );
    }

    #[test]
    fn is_identifier_detects_patterns() {
        assert!(is_identifier("snake_case_name"));
        assert!(is_identifier("camelCaseName"));
        assert!(is_identifier("PascalCaseName"));
        assert!(is_identifier("com.example.path"));
        assert!(is_identifier("std::collections::HashMap"));
        assert!(!is_identifier("meldipgodequiced"));
        assert!(!is_identifier("ALLCAPS"));
    }
}
