//! Transcript metrics used by validation: WER and term counts.

use crate::vocab::levenshtein;

/// Lowercase words with punctuation removed (apostrophes kept inside words).
pub fn normalize_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .flat_map(char::to_lowercase)
                .collect::<String>()
                .trim_matches('\'')
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Word error rate result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wer {
    /// Substitutions plus insertions plus deletions.
    pub errors: usize,
    /// Reference word count.
    pub ref_words: usize,
    /// Hypothesis word count.
    pub hyp_words: usize,
}

impl Wer {
    /// errors / reference words (0 when the reference is empty).
    pub fn rate(&self) -> f64 {
        if self.ref_words == 0 {
            0.0
        } else {
            self.errors as f64 / self.ref_words as f64
        }
    }
}

/// WER of `hyp` against `reference` after normalization.
pub fn wer(reference: &str, hyp: &str) -> Wer {
    let r = normalize_words(reference);
    let h = normalize_words(hyp);
    Wer {
        errors: levenshtein(&r, &h),
        ref_words: r.len(),
        hyp_words: h.len(),
    }
}

/// Case-insensitive whole-word occurrences of `term` (may be several words).
pub fn count_term(text: &str, term: &str) -> usize {
    let words = normalize_words(text);
    let needle = normalize_words(term);
    if needle.is_empty() || words.len() < needle.len() {
        return 0;
    }
    words
        .windows(needle.len())
        .filter(|w| w.iter().zip(&needle).all(|(a, b)| a == b))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_punctuation_and_case() {
        assert_eq!(normalize_words("Hello, World! It's 'fine'."), vec!["hello", "world", "it's", "fine"]);
    }

    #[test]
    fn wer_counts_edits() {
        let w = wer("the cat sat on the mat", "the cat sat on a mat today");
        assert_eq!(w.errors, 2);
        assert_eq!(w.ref_words, 6);
        assert!((w.rate() - 2.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn counts_whole_words_and_phrases() {
        let t = "Blue server, blue servers and the blue server.";
        assert_eq!(count_term(t, "blue server"), 2);
        assert_eq!(count_term(t, "BLUE"), 3);
        assert_eq!(count_term(t, ""), 0);
    }
}
