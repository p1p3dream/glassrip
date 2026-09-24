//! Vocabulary correction: replace low-probability words that sound like a term.
//!
//! A word (or a run of words for multi-word terms) is replaced only when all of
//! these hold:
//! 1. its mean token probability is below `max_p`, or below `max_p_proper_noun`
//!    when every word of the span and of the term is capitalized (whisper tends
//!    to emit a known proper noun with fair confidence for an unfamiliar name).
//!    That exception never applies to a sentence-initial word or to a word in
//!    the embedded common-English list (`common_words.txt`);
//! 2. its phonetic key equals the term's key;
//! 3. the normalized letter edit distance is at most `max_norm_edit`.
//!
//! The phonetic key is a small Soundex-style code: common digraphs are folded
//! (`th`→`t`, `ph`→`f`, ...), voiced and unvoiced stops and fricatives are merged
//! (`d`→`t`, `b`→`p`, `g`→`k`, `v`→`f`, `z`→`s`), vowels after the first letter
//! and `h`, `w`, `y` are dropped, and repeats collapse.

use std::collections::HashSet;
use std::sync::OnceLock;

const COMMON_WORDS: &str = include_str!("../common_words.txt");

fn common_words() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        COMMON_WORDS
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect()
    })
}

/// True when the lowercase form of `word` is in the embedded common-word list.
pub fn is_common_word(word: &str) -> bool {
    common_words().contains(letters(word).as_str())
}

/// Correction thresholds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrectionConfig {
    /// Words with probability at or above this are never changed.
    pub max_p: f32,
    /// Probability limit when both the span and the term are capitalized.
    pub max_p_proper_noun: f32,
    /// Maximum edit distance divided by the longer letter count.
    pub max_norm_edit: f64,
    /// Words with fewer letters are never changed.
    pub min_letters: usize,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self {
            max_p: 0.6,
            max_p_proper_noun: 0.85,
            max_norm_edit: 0.4,
            min_letters: 3,
        }
    }
}

/// Lowercase letters and digits of `s`.
pub fn letters(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Soundex-style phonetic key (see module docs).
pub fn phonetic_key(s: &str) -> String {
    let l = letters(s);
    let mut folded = String::with_capacity(l.len());
    let chars: Vec<char> = l.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let pair = next.map(|n| (c, n));
        let (out, adv): (&str, usize) = match pair {
            Some(('t', 'h')) => ("t", 2),
            Some(('p', 'h')) => ("f", 2),
            Some(('s', 'h')) => ("s", 2),
            Some(('c', 'h')) => ("k", 2),
            Some(('c', 'k')) => ("k", 2),
            Some(('g', 'h')) => ("k", 2),
            Some(('q', 'u')) => ("k", 2),
            Some(('w', 'h')) => ("w", 2),
            Some(('k', 'n')) if i == 0 => ("n", 2),
            _ => {
                let o = match c {
                    'c' => {
                        if matches!(next, Some('e' | 'i' | 'y')) {
                            "s"
                        } else {
                            "k"
                        }
                    }
                    'q' | 'g' => "k",
                    'x' => "ks",
                    'z' => "s",
                    'd' => "t",
                    'b' => "p",
                    'v' => "f",
                    _ => "",
                };
                if o.is_empty() {
                    folded.push(c);
                    i += 1;
                    continue;
                }
                (o, 1)
            }
        };
        folded.push_str(out);
        i += adv;
    }
    let mut key = String::new();
    for (idx, c) in folded.chars().enumerate() {
        let is_vowel = matches!(c, 'a' | 'e' | 'i' | 'o' | 'u');
        let mapped = if idx == 0 {
            if is_vowel {
                'a'
            } else {
                c
            }
        } else if is_vowel || matches!(c, 'h' | 'w' | 'y') {
            continue;
        } else {
            c
        };
        if !key.ends_with(mapped) {
            key.push(mapped);
        }
    }
    key
}

/// Levenshtein distance over any comparable items.
pub fn levenshtein<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, x) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(x != y);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Edit distance divided by the longer length.
pub fn normalized_edit(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let n = a.len().max(b.len());
    if n == 0 {
        return 0.0;
    }
    levenshtein(&a, &b) as f64 / n as f64
}

#[derive(Debug, Clone)]
struct Term {
    words: Vec<String>,
    letters: String,
    key: String,
}

/// Vocabulary prepared for matching.
#[derive(Debug, Clone, Default)]
pub struct Vocabulary {
    terms: Vec<Term>,
}

impl Vocabulary {
    /// Build from terms such as `["Kethra", "blue server"]`.
    pub fn new<S: AsRef<str>>(terms: &[S]) -> Self {
        let terms = terms
            .iter()
            .filter_map(|t| {
                let words: Vec<String> =
                    t.as_ref().split_whitespace().map(str::to_string).collect();
                if words.is_empty() {
                    return None;
                }
                let joined = words.join(" ");
                Some(Term {
                    letters: letters(&joined),
                    key: phonetic_key(&joined),
                    words,
                })
            })
            .collect();
        Self { terms }
    }

    /// True when no terms are present.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    fn is_term_word(&self, core_letters: &str) -> bool {
        self.terms
            .iter()
            .any(|t| t.words.iter().any(|w| letters(w) == core_letters))
    }
}

fn starts_upper(w: &str) -> bool {
    w.chars().next().is_some_and(char::is_uppercase)
}

/// Split a word into leading punctuation, core and trailing punctuation.
pub fn split_punct(w: &str) -> (&str, &str, &str) {
    let start = w
        .char_indices()
        .find(|(_, c)| c.is_alphanumeric())
        .map_or(w.len(), |(i, _)| i);
    let end = w
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map_or(start, |(i, c)| i + c.len_utf8());
    (&w[..start], &w[start..end.max(start)], &w[end.max(start)..])
}

/// Compute replacements for `words` given as `(text, probability)`.
///
/// Returns one entry per word: `Some(new_text)` when the word changes.
pub fn correct(
    words: &[(&str, f32)],
    vocab: &Vocabulary,
    cfg: &CorrectionConfig,
) -> Vec<Option<String>> {
    let mut out: Vec<Option<String>> = vec![None; words.len()];
    if vocab.is_empty() {
        return out;
    }
    let mut i = 0;
    while i < words.len() {
        let mut best: Option<(f64, usize, Vec<String>)> = None;
        for term in &vocab.terms {
            let n = term.words.len();
            if i + n > words.len() || out[i..i + n].iter().any(Option::is_some) {
                continue;
            }
            let span = &words[i..i + n];
            let cores: Vec<&str> = span.iter().map(|(w, _)| split_punct(w).1).collect();
            let joined = cores.join(" ");
            let span_letters = letters(&joined);
            if span_letters.chars().count() < cfg.min_letters {
                continue;
            }
            if span_letters == term.letters {
                continue;
            }
            if n == 1 && vocab.is_term_word(&span_letters) {
                continue;
            }
            let mean_p = span.iter().map(|(_, p)| *p).sum::<f32>() / n as f32;
            let sentence_initial = i == 0 || words[i - 1].0.trim_end().ends_with(['.', '?', '!']);
            let proper = !sentence_initial
                && cores.iter().all(|c| starts_upper(c))
                && term.words.iter().all(|w| starts_upper(w))
                && !cores.iter().any(|c| is_common_word(c));
            let limit = if proper {
                cfg.max_p.max(cfg.max_p_proper_noun)
            } else {
                cfg.max_p
            };
            if mean_p >= limit {
                continue;
            }
            if phonetic_key(&joined) != term.key {
                continue;
            }
            let d = normalized_edit(&span_letters, &term.letters);
            if d > cfg.max_norm_edit {
                continue;
            }
            if best.as_ref().is_none_or(|(bd, _, _)| d < *bd) {
                best = Some((d, n, term.words.clone()));
            }
        }
        if let Some((_, n, term_words)) = best {
            for (k, tw) in term_words.iter().enumerate() {
                let (lead, _, trail) = split_punct(words[i + k].0);
                out[i + k] = Some(format!("{lead}{tw}{trail}"));
            }
            i += n;
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_merge_voicing_and_digraphs() {
        assert_eq!(phonetic_key("Kethra"), phonetic_key("Ketra"));
        assert_eq!(phonetic_key("Dalvik"), phonetic_key("Talvic"));
        assert_eq!(phonetic_key("Zorbin"), phonetic_key("Sorpin"));
        assert_ne!(phonetic_key("Dox"), phonetic_key("Pox"));
        assert_eq!(phonetic_key("Aurel"), "arl");
    }

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein(&['a', 'b', 'c'], &['a', 'b', 'c']), 0);
        assert_eq!(levenshtein(&['a'], &[]), 1);
        assert_eq!(levenshtein::<char>(&[], &['x', 'y']), 2);
        assert_eq!(
            levenshtein(
                &"kitten".chars().collect::<Vec<_>>(),
                &"sitting".chars().collect::<Vec<_>>()
            ),
            3
        );
    }

    #[test]
    fn replaces_low_probability_sound_alike_and_keeps_punctuation() {
        let vocab = Vocabulary::new(&["Kethra", "Zorbin"]);
        let words = [
            ("Hi", 0.9),
            ("Ketra,", 0.3),
            ("and", 0.95),
            ("Sorpin.", 0.4),
        ];
        let out = correct(&words, &vocab, &CorrectionConfig::default());
        assert_eq!(out[0], None);
        assert_eq!(out[1].as_deref(), Some("Kethra,"));
        assert_eq!(out[2], None);
        assert_eq!(out[3].as_deref(), Some("Zorbin."));
    }

    #[test]
    fn keeps_confident_words() {
        let vocab = Vocabulary::new(&["Kethra"]);
        let out = correct(&[("Ketra", 0.95)], &vocab, &CorrectionConfig::default());
        assert_eq!(out[0], None);
    }

    #[test]
    fn proper_nouns_use_the_higher_limit() {
        let vocab = Vocabulary::new(&["Kethra", "zorbin"]);
        let cfg = CorrectionConfig::default();
        // capitalized word and term: 0.7 is below the proper-noun limit
        assert_eq!(
            correct(&[("Hi", 0.9), ("Ketra,", 0.7)], &vocab, &cfg)[1].as_deref(),
            Some("Kethra,")
        );
        // lowercase word: the normal limit applies
        assert_eq!(
            correct(&[("hi", 0.9), ("ketra", 0.7)], &vocab, &cfg)[1],
            None
        );
        // lowercase term: the normal limit applies
        assert_eq!(
            correct(&[("Hi", 0.9), ("Sorpin", 0.7)], &vocab, &cfg)[1],
            None
        );
    }

    #[test]
    fn proper_noun_exception_skips_sentence_starts() {
        let vocab = Vocabulary::new(&["Kethra"]);
        let cfg = CorrectionConfig::default();
        assert_eq!(correct(&[("Ketra", 0.7)], &vocab, &cfg)[0], None);
        assert_eq!(
            correct(&[("Done.", 0.9), ("Ketra", 0.7)], &vocab, &cfg)[1],
            None
        );
        // below the normal limit it still applies at a sentence start
        assert_eq!(
            correct(&[("Ketra", 0.5)], &vocab, &cfg)[0].as_deref(),
            Some("Kethra")
        );
    }

    #[test]
    fn proper_noun_exception_skips_common_words() {
        assert!(is_common_word("Time"));
        assert!(!is_common_word("Kethra"));
        let vocab = Vocabulary::new(&["Thyme"]);
        let cfg = CorrectionConfig::default();
        // "Time" sounds like the term but is a common word: no exception
        assert_eq!(
            correct(&[("Well,", 0.9), ("Time", 0.7)], &vocab, &cfg)[1],
            None
        );
        // the normal limit still applies
        assert_eq!(
            correct(&[("Well,", 0.9), ("Time", 0.5)], &vocab, &cfg)[1].as_deref(),
            Some("Thyme")
        );
    }

    #[test]
    fn keeps_phonetically_distant_words() {
        let vocab = Vocabulary::new(&["Kethra"]);
        let out = correct(&[("camera", 0.1)], &vocab, &CorrectionConfig::default());
        assert_eq!(out[0], None);
    }

    #[test]
    fn multi_word_terms_replace_word_for_word() {
        let vocab = Vocabulary::new(&["blue server"]);
        let out = correct(
            &[("the", 0.9), ("blew", 0.3), ("server", 0.5)],
            &vocab,
            &CorrectionConfig::default(),
        );
        assert_eq!(out[1].as_deref(), Some("blue"));
        assert_eq!(out[2].as_deref(), Some("server"));
    }

    #[test]
    fn split_punct_handles_edges() {
        assert_eq!(split_punct("\"Hey,\""), ("\"", "Hey", ",\""));
        assert_eq!(split_punct("..."), ("...", "", ""));
        assert_eq!(split_punct("ok"), ("", "ok", ""));
    }
}
