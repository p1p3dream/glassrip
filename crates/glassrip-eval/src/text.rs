//! Text normalization and similarity measures used by every metric.
//!
//! - [`normalize_label`]: case-folded, punctuation mapped to spaces, whitespace
//!   collapsed. Used for label matching (nodes, stickies, owners, chrome).
//! - [`label_similarity`]: Jaro-Winkler on normalized labels. Labels match at
//!   [`LABEL_MATCH_JW`] (0.9) or above.
//! - [`cer`] / [`wer`]: character and word error rates (Levenshtein distance
//!   divided by reference length) after whitespace normalization.
//! - [`token_dice`]: Dice coefficient over normalized word multisets, used for
//!   sentence-level items (decisions, action items, questions).
//! - [`containment`]: share of a gold phrasing's content words stated by a
//!   prediction (paraphrases inside longer sentences), gated by
//!   [`CONTAINMENT_MIN_PRECISION`]; [`key_terms_present`] requires the names in the
//!   gold phrasing to appear in the prediction.

use std::collections::BTreeMap;

/// Jaro-Winkler threshold at which two labels are the same element.
pub const LABEL_MATCH_JW: f64 = 0.9;

/// Default content-word Dice threshold for sentence-level matches (notes items).
pub const SENTENCE_MATCH_DICE: f64 = 0.6;

/// Function words ignored by [`content_dice`].
pub const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "of", "to", "for", "in", "on", "at", "by", "with",
    "from", "into", "as", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "these", "those", "we", "i", "you", "he", "she", "they", "our", "us", "your", "my", "me",
    "will", "would", "should", "can", "could", "do", "does", "did", "so", "just", "now", "then",
    "what", "how", "which", "who", "ll", "s", "re", "ve", "d", "m", "t", "not", "if", "there",
    "here", "about",
];

/// Lowercases, maps every non-alphanumeric character to a space, and collapses
/// whitespace. `ledger_api.py` and `Ledger API py` normalize equally.
pub fn normalize_label(text: &str) -> String {
    let mapped: String = text
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_lowercase().next().unwrap_or(c)
            } else {
                ' '
            }
        })
        .collect();
    collapse_ws(&mapped)
}

/// Collapses runs of whitespace to one space and trims.
pub fn collapse_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Jaro-Winkler similarity of two labels after [`normalize_label`]. Two empty
/// labels are identical (1.0); one empty label matches nothing (0.0).
pub fn label_similarity(a: &str, b: &str) -> f64 {
    let (a, b) = (normalize_label(a), normalize_label(b));
    match (a.is_empty(), b.is_empty()) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.0,
        _ => strsim::jaro_winkler(&a, &b),
    }
}

/// Best [`label_similarity`] of `text` against `candidates`.
pub fn best_label_similarity<'a>(text: &str, candidates: impl IntoIterator<Item = &'a str>) -> f64 {
    candidates
        .into_iter()
        .map(|c| label_similarity(text, c))
        .fold(0.0, f64::max)
}

/// True when two labels match at [`LABEL_MATCH_JW`].
pub fn labels_match(a: &str, b: &str) -> bool {
    label_similarity(a, b) >= LABEL_MATCH_JW
}

/// Edit distance between two sequences.
pub fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
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

/// Error counts behind a rate: `errors / reference_len`.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ErrorCount {
    /// Edit operations.
    pub errors: usize,
    /// Reference length (characters or words).
    pub reference_len: usize,
}

impl ErrorCount {
    /// The rate. Empty reference: 0.0 when there are no errors, else 1.0.
    pub fn rate(&self) -> f64 {
        if self.reference_len == 0 {
            if self.errors == 0 {
                0.0
            } else {
                1.0
            }
        } else {
            self.errors as f64 / self.reference_len as f64
        }
    }

    /// Adds another count (pooling).
    pub fn add(&mut self, other: ErrorCount) {
        self.errors += other.errors;
        self.reference_len += other.reference_len;
    }
}

/// Character error counts of `hyp` against `reference` after whitespace
/// normalization (case preserved: case errors count).
pub fn cer_count(hyp: &str, reference: &str) -> ErrorCount {
    let h: Vec<char> = collapse_ws(hyp).chars().collect();
    let r: Vec<char> = collapse_ws(reference).chars().collect();
    ErrorCount {
        errors: edit_distance(&h, &r),
        reference_len: r.len(),
    }
}

/// Character error rate (see [`cer_count`]).
pub fn cer(hyp: &str, reference: &str) -> f64 {
    cer_count(hyp, reference).rate()
}

/// Lowercased words with surrounding punctuation stripped.
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Word error counts of `hyp` against `reference` (case and edge punctuation ignored).
pub fn wer_count(hyp: &str, reference: &str) -> ErrorCount {
    let h = words(hyp);
    let r = words(reference);
    ErrorCount {
        errors: edit_distance(&h, &r),
        reference_len: r.len(),
    }
}

/// Word error rate (see [`wer_count`]).
pub fn wer(hyp: &str, reference: &str) -> f64 {
    wer_count(hyp, reference).rate()
}

/// Dice coefficient over word multisets of the normalized texts:
/// `2 * |A ∩ B| / (|A| + |B|)`. Two empty texts score 1.0.
pub fn token_dice(a: &str, b: &str) -> f64 {
    let count = |t: &str| {
        let mut m: BTreeMap<String, usize> = BTreeMap::new();
        for w in normalize_label(t).split(' ').filter(|w| !w.is_empty()) {
            *m.entry(w.to_string()).or_default() += 1;
        }
        m
    };
    let (ca, cb) = (count(a), count(b));
    let (na, nb): (usize, usize) = (ca.values().sum(), cb.values().sum());
    if na + nb == 0 {
        return 1.0;
    }
    let inter: usize = ca
        .iter()
        .map(|(w, n)| (*n).min(cb.get(w).copied().unwrap_or(0)))
        .sum();
    2.0 * inter as f64 / (na + nb) as f64
}

/// [`token_dice`] after removing [`STOPWORDS`]; when either side has no content
/// word left, falls back to [`token_dice`] on the full texts.
pub fn content_dice(a: &str, b: &str) -> f64 {
    let strip = |t: &str| {
        normalize_label(t)
            .split(' ')
            .filter(|w| !w.is_empty() && !STOPWORDS.contains(w))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let (sa, sb) = (strip(a), strip(b));
    if sa.is_empty() || sb.is_empty() {
        token_dice(a, b)
    } else {
        token_dice(&sa, &sb)
    }
}

/// Smallest share of a prediction's content words that must come from the gold
/// phrasing for [`containment`] to count. A prediction may carry up to ten times the
/// gold's content (context, names, reasons) and still match, but a kitchen-sink
/// sentence that happens to contain a short gold phrase does not.
pub const CONTAINMENT_MIN_PRECISION: f64 = 0.1;

/// Light suffix stripping so inflections of one word compare equal (`decided` and
/// `decide`, `moves`, `moved`, `moving` and `move`, `skipped` and `skip`). Words of
/// three letters or fewer are kept whole, as are `-ss`, `-us`, and `-is` endings
/// (`class`, `focus`, `analysis`).
pub fn stem(word: &str) -> String {
    let mut w = word.to_lowercase();
    let n = w.chars().count();
    let strip = |w: &mut String, k: usize| w.truncate(w.len() - k);
    let mut undouble = false;
    if n >= 6 && w.ends_with("ing") {
        strip(&mut w, 3);
        undouble = true;
    } else if n >= 5 && w.ends_with("ed") {
        strip(&mut w, 2);
        undouble = true;
    } else if n >= 5 && w.ends_with("es") && !w.ends_with("ses") {
        strip(&mut w, 2);
    } else if n >= 4
        && w.ends_with('s')
        && !(w.ends_with("ss") || w.ends_with("us") || w.ends_with("is"))
    {
        strip(&mut w, 1);
    }
    let chars: Vec<char> = w.chars().collect();
    if undouble && chars.len() >= 4 {
        let (a, b) = (chars[chars.len() - 2], chars[chars.len() - 1]);
        if a == b && !"lsz".contains(b) && !"aeiou".contains(b) {
            w.pop();
        }
    }
    if w.chars().count() > 3 && w.ends_with('e') {
        w.pop();
    }
    w
}

/// Stemmed content words (normalized, [`STOPWORDS`] removed), deduplicated.
pub fn content_stems(text: &str) -> std::collections::BTreeSet<String> {
    normalize_label(text)
        .split(' ')
        .filter(|w| !w.is_empty() && !STOPWORDS.contains(w))
        .map(stem)
        .collect()
}

/// Key terms of a gold phrasing: words that name something specific, which a
/// matching prediction must also name. A word is a key term when it is capitalized
/// anywhere but the first position, is an acronym (two or more capitals), or holds a
/// digit or underscore. Returned stemmed and lowercased. A sentence-initial word is
/// capitalized by grammar, so a name in first position is not recognized; gold
/// phrasings that hinge on a name should carry an alias that names it mid-sentence.
pub fn key_terms(text: &str) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    for (i, raw) in text.split_whitespace().enumerate() {
        let w = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
        if w.is_empty() {
            continue;
        }
        let upper = w.chars().filter(char::is_ascii_uppercase).count();
        let capitalized = w.chars().next().is_some_and(char::is_uppercase);
        let special = w.chars().any(|c| c.is_ascii_digit() || c == '_');
        if (capitalized && i > 0) || upper >= 2 || special {
            for part in normalize_label(w).split(' ').filter(|p| !p.is_empty()) {
                if !STOPWORDS.contains(&part) {
                    out.insert(stem(part));
                }
            }
        }
    }
    out
}

/// Every key term of `gold` ([`key_terms`]) appears among the content words of `pred`.
pub fn key_terms_present(gold: &str, pred: &str) -> bool {
    let p = content_stems(pred);
    key_terms(gold).iter().all(|k| p.contains(k))
}

/// How much of `gold` a prediction states: the share of the gold's stemmed content
/// words found in `pred`. A paraphrase that restates the gold claim inside a longer
/// sentence scores high here while its Dice coefficient stays low. Zero when the gold
/// has fewer than two content words (too little to contain), or when under
/// [`CONTAINMENT_MIN_PRECISION`] of the prediction's content words come from the gold.
pub fn containment(gold: &str, pred: &str) -> f64 {
    let (g, p) = (content_stems(gold), content_stems(pred));
    if g.len() < 2 || p.is_empty() {
        return 0.0;
    }
    let inter = g.intersection(&p).count() as f64;
    if inter / (p.len() as f64) < CONTAINMENT_MIN_PRECISION {
        return 0.0;
    }
    inter / g.len() as f64
}

/// Median of a slice (average of the two middle values for even counts); `None` when empty.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems_join_inflections() {
        for group in [
            &["move", "moves", "moved", "moving"][..],
            &["decide", "decided", "decides"],
            &["skip", "skipped", "skipping"],
            &["build", "builds"],
            &["release", "releases"],
        ] {
            let stems: std::collections::BTreeSet<String> = group.iter().map(|w| stem(w)).collect();
            assert_eq!(stems.len(), 1, "{group:?} -> {stems:?}");
        }
        assert_eq!(stem("focus"), "focus");
        assert_eq!(stem("class"), "class");
        assert_eq!(stem("use"), "use");
    }

    #[test]
    fn key_terms_and_containment() {
        let k = key_terms("Skip the Ledger step and ask QA about order_svc v2");
        let want: std::collections::BTreeSet<String> = ["ledger", "qa", "order_svc", "v2"]
            .iter()
            .map(|s| stem(s))
            .collect();
        // order_svc normalizes to two words; both are kept.
        assert!(
            k.contains("ledger") && k.contains("qa") && k.contains("v2"),
            "{k:?}"
        );
        assert!(
            k.contains("order") && k.contains("svc"),
            "{k:?} vs {want:?}"
        );
        assert!(!k.contains("skip"), "the first word is not a key term");
        assert!(key_terms_present("ask Tamsin", "we will ask tamsin today"));
        assert!(!key_terms_present("ask Tamsin", "we will ask Quill today"));
        assert!(
            (containment("skip importer", "we skip the importer this week") - 1.0).abs() < 1e-12
        );
        assert_eq!(
            containment("importer", "skip the importer"),
            0.0,
            "one word is too short"
        );
    }

    #[test]
    fn normalize_maps_punctuation() {
        assert_eq!(normalize_label("  Ledger_API.py "), "ledger api py");
        assert_eq!(normalize_label("/web  design-kit"), "web design kit");
        assert_eq!(normalize_label("???"), "");
    }

    #[test]
    fn label_similarity_bounds() {
        assert_eq!(label_similarity("Orbit Queue", "orbit  queue"), 1.0);
        assert_eq!(label_similarity("", ""), 1.0);
        assert_eq!(label_similarity("", "x"), 0.0);
        // Jaro-Winkler of "martha" / "marhta" is 0.9611 (textbook value).
        assert!((label_similarity("martha", "marhta") - 0.961_111).abs() < 1e-5);
        assert!(labels_match("Ledger API", "Ledger AP1"));
        assert!(!labels_match("Ledger API", "Orbit Queue"));
    }

    #[test]
    fn cer_hand_computed() {
        // one substitution over 5 reference chars
        assert!((cer("hellp", "hello") - 0.2).abs() < 1e-12);
        // whitespace collapsed before comparing
        assert_eq!(cer("a  b\n c", "a b c"), 0.0);
        // one deletion, one insertion: "abcd" -> "abd" is 1 edit of 4
        assert!((cer("abd", "abcd") - 0.25).abs() < 1e-12);
        assert_eq!(cer("", ""), 0.0);
        assert_eq!(cer("x", ""), 1.0);
    }

    #[test]
    fn wer_hand_computed() {
        // "the cat sat" vs "the bat sat down": 1 sub + 1 del = 2 of 4
        assert!((wer("the cat sat", "the bat sat down") - 0.5).abs() < 1e-12);
        assert_eq!(wer("Hello, World!", "hello world"), 0.0);
    }

    #[test]
    fn dice_hand_computed() {
        // A = {skip, the, step}, B = {skip, step, now}: 2*2/(3+3)
        assert!((token_dice("skip the step", "skip step now") - 4.0 / 6.0).abs() < 1e-12);
        assert_eq!(token_dice("", ""), 1.0);
        assert_eq!(token_dice("a", ""), 0.0);
    }

    #[test]
    fn content_dice_hand_computed() {
        // {defer, importer} vs {defer, importer}: stopwords "the", "now" removed
        assert_eq!(
            content_dice("defer the importer", "defer importer now"),
            1.0
        );
        // {skip, sanity, step} vs {skip, step}: 2*2/5
        assert!((content_dice("skip the sanity step", "skip that step") - 0.8).abs() < 1e-12);
        // only stopwords on one side: full-text Dice fallback
        assert_eq!(content_dice("it is", "it is"), 1.0);
    }

    #[test]
    fn median_even_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn error_count_pooling() {
        let mut a = ErrorCount {
            errors: 1,
            reference_len: 10,
        };
        a.add(ErrorCount {
            errors: 3,
            reference_len: 10,
        });
        assert!((a.rate() - 0.2).abs() < 1e-12);
    }
}
