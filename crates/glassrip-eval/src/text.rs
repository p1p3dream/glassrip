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
//! - [`claim_words`]: stemmed content words with their polarity (a negation governs
//!   the rest of its clause), so a claim and its negation share no word.
//! - [`claim_dice`] and [`containment`]: sentence similarity over claim words; the
//!   second finds a paraphrase inside a longer sentence, bounded by
//!   [`CONTAINMENT_MAX_EXPANSION`] and [`CONTAINMENT_MAX_SPREAD`]. [`covers`] requires
//!   the gold claim words to be stated ([`allowed_missing`]), and
//!   [`key_terms_present`] requires the names in the gold phrasing (known participants
//!   and entities from a [`Vocabulary`], at any position) to appear in the prediction.

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

/// Most content words a prediction may carry (repeats counted) for [`containment`]
/// to count, as a multiple of the gold phrasing's distinct content words, plus
/// [`CONTAINMENT_EXTRA_WORDS`]: the claim plus room for context (names, reasons,
/// qualifiers). A longer sentence that happens to contain the gold words is a
/// ramble, not a restatement.
pub const CONTAINMENT_MAX_EXPANSION: usize = 4;

/// Fixed room added to [`CONTAINMENT_MAX_EXPANSION`], so a two-word gold can sit in
/// an ordinary sentence ("after a long discussion the team decided to skip the
/// importer").
pub const CONTAINMENT_EXTRA_WORDS: usize = 4;

/// Widest span, in content words, in which [`containment`] must find the gold words
/// in the prediction: this multiple of the gold's length plus
/// [`CONTAINMENT_SPREAD_SLACK`]. A restatement keeps its words together (modifiers
/// between them are fine); gold words scattered across a long prediction do not
/// state the claim.
pub const CONTAINMENT_MAX_SPREAD: usize = 2;

/// Fixed room added to [`CONTAINMENT_MAX_SPREAD`] ("skip the old broken legacy
/// importer").
pub const CONTAINMENT_SPREAD_SLACK: usize = 2;

/// Light suffix stripping so inflections of one word compare equal (`decided` and
/// `decide`, `moves`, `moved`, `moving` and `move`, `skipped` and `skip`, `used` and
/// `use`, `retries`, `retried`, and `retry`). Suffixes are not stripped from words of three letters, nor from `-ss`,
/// `-us`, and `-is` endings (`class`, `focus`, `analysis`).
pub fn stem(word: &str) -> String {
    let mut w = word.to_lowercase();
    let n = w.chars().count();
    let strip = |w: &mut String, k: usize| w.truncate(w.len() - k);
    let mut undouble = false;
    if n >= 5 && (w.ends_with("ies") || w.ends_with("ied")) {
        strip(&mut w, 3);
        w.push('y');
    } else if n >= 6 && w.ends_with("ing") {
        strip(&mut w, 3);
        undouble = true;
    } else if w.ends_with("ed") && (n >= 5 || (n == 4 && !w.ends_with("eed"))) {
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
    if w.chars().count() > 2 && w.ends_with('e') {
        w.pop();
    }
    w
}

/// Words that negate the rest of their clause ("do not ship", "we won't ship",
/// "no weekly builds", "never ship"). Compared after removing apostrophes; any word
/// ending in `n't` negates as well.
pub const NEGATIONS: &[&str] = &[
    "not", "no", "never", "cannot", "cant", "dont", "doesnt", "didnt", "wont", "wouldnt",
    "shouldnt", "couldnt", "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
    "mustnt", "neednt", "without", "nor", "neither", "none", "nothing", "nobody",
];

/// Words that start a new clause and so end a negation's scope ("ship weekly builds
/// but not nightly ones").
const CLAUSE_WORDS: &[&str] = &[
    "but", "however", "although", "though", "whereas", "while", "yet", "instead", "unless",
];

/// True for a negation word ([`NEGATIONS`], or any word ending in `n't`).
fn is_negation(word: &str) -> bool {
    let lower = word.to_lowercase();
    lower.ends_with("n't") || NEGATIONS.contains(&lower.replace('\'', "").as_str())
}

/// A token of a phrasing: a word (original case, inner apostrophes kept) or a clause
/// boundary.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Word(String),
    Boundary(char),
}

/// Splits text into words and clause boundaries. A word is a run of letters, digits,
/// `_`, and apostrophes (`don't`, `order_svc`); a `.` or `,` between two digits stays
/// inside it (`2.5`, `1,000`). A `.` before a lower-case letter or a digit separates
/// two words without ending the clause (`ledger.py`, `builds.not`); before a capital
/// it ends a sentence whose space was dropped (`builds.We`). Every other
/// character separates words, so `don't-ship` is `don't` and `ship`; each of
/// `, ; : . ! ?` outside a word is a clause boundary (`builds,not` ends a clause
/// before `not`). Curly apostrophes count as `'`; quotes around a word are dropped.
fn tokenize(text: &str) -> Vec<Token> {
    let chars: Vec<char> = text
        .chars()
        .map(|c| {
            if c == '\u{2019}' || c == '\u{2018}' {
                '\''
            } else {
                c
            }
        })
        .collect();
    let at = |i: Option<usize>, f: fn(&char) -> bool| i.and_then(|i| chars.get(i)).is_some_and(f);
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<Token>| {
        let w = cur.trim_matches('\'');
        if !w.is_empty() {
            out.push(Token::Word(w.to_string()));
        }
        cur.clear();
    };
    for (i, &c) in chars.iter().enumerate() {
        let (prev, next) = (i.checked_sub(1), Some(i + 1));
        let word_char = c.is_alphanumeric() || c == '_' || c == '\'';
        let numeric_mark = (c == '.' || c == ',')
            && at(prev, char::is_ascii_digit)
            && at(next, char::is_ascii_digit);
        let joining_dot = c == '.'
            && at(prev, |c| c.is_alphanumeric())
            && at(next, |c| c.is_lowercase() || c.is_ascii_digit());
        if word_char || numeric_mark {
            cur.push(c);
        } else {
            flush(&mut cur, &mut out);
            if !joining_dot && ",;:.!?".contains(c) {
                out.push(Token::Boundary(c));
            }
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// Words that are also names ("Will", "May"): stopwords and modal verbs.
fn is_function_word(lower: &str) -> bool {
    STOPWORDS.contains(&lower) || ["may", "might", "must", "shall"].contains(&lower)
}

/// Whether the word at `i` may name a participant. It must be capitalized; a word
/// that is also a function word ("Will", "May") is read as the function word where
/// it opens a sentence or clause and another function word follows ("Will we
/// ship?", "yes, Will we ship?"), and as a name otherwise ("Will moves the dashboard", "Will Park ships").
fn may_name(tokens: &[Token], i: usize) -> bool {
    let Some(Token::Word(w)) = tokens.get(i) else {
        return false;
    };
    if !w.chars().next().is_some_and(char::is_uppercase) {
        return false;
    }
    if !is_function_word(&w.to_lowercase()) {
        return true;
    }
    let opens = i == 0 || matches!(tokens[i - 1], Token::Boundary(_));
    let next_function = match tokens.get(i + 1) {
        Some(Token::Word(n)) => is_function_word(&n.to_lowercase()) || is_negation(n),
        _ => true,
    };
    !(opens && next_function)
}

/// Names the golden set knows: participants (display names, first names, aliases) and
/// entities (board node labels, hotwords). A known name is a key term wherever it
/// stands in a sentence, first position included.
///
/// A participant name word counts as that participant only where it is capitalized,
/// so "May moves the dashboard" names a participant May while "we may move the
/// dashboard" does not, and a name that is also a function word ("Will") is still a
/// name. Every capitalized spelling of one participant is one term, so "Tam" in a
/// prediction states "Tamsin". Entity words are key terms in any case.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Vocabulary {
    /// Normalized participant name word to the canonical ids (`<person_id>`) of the
    /// participants it names; more than one when two participants share it.
    people: BTreeMap<String, std::collections::BTreeSet<String>>,
    /// Stemmed words of each entity name.
    entities: Vec<std::collections::BTreeSet<String>>,
}

impl Vocabulary {
    /// Adds a participant: every word of two or more characters of each name names
    /// `person_id`. A word two participants share ("Tam" for Tam Ly and for Tamsin)
    /// becomes one ambiguous term that names neither alone ([`Self::ambiguous_names`]).
    pub fn add_person<'a>(&mut self, person_id: &str, names: impl IntoIterator<Item = &'a str>) {
        let id = normalize_label(person_id).replace(' ', "_");
        for name in names {
            for w in normalize_label(name).split(' ') {
                if w.chars().count() >= 2 {
                    self.people
                        .entry(w.to_string())
                        .or_default()
                        .insert(id.clone());
                }
            }
        }
    }

    /// Name words shared by two or more participants, with the participants.
    pub fn ambiguous_names(&self) -> Vec<String> {
        self.people
            .iter()
            .filter(|(_, ids)| ids.len() > 1)
            .map(|(w, ids)| {
                format!(
                    "name word `{w}` belongs to {}; alone it names none of them",
                    ids.iter().cloned().collect::<Vec<_>>().join(" and ")
                )
            })
            .collect()
    }

    /// Adds an entity name (a board label, a hotword): each of its content words is a
    /// key term.
    pub fn add_entity(&mut self, name: &str) {
        let words: std::collections::BTreeSet<String> = normalize_label(name)
            .split(' ')
            .filter(|w| !w.is_empty() && !STOPWORDS.contains(w))
            .map(stem)
            .collect();
        if !words.is_empty() {
            self.entities.push(words);
        }
    }

    /// The participant term of a normalized word (`@<id>`, or `@<id>|<id>` for a
    /// shared word), when the word may name someone ([`may_name`]).
    fn person(&self, word: &str, name_ok: bool) -> Option<String> {
        let ids = self.people.get(word).filter(|_| name_ok)?;
        Some(format!(
            "@{}",
            ids.iter().cloned().collect::<Vec<_>>().join("|")
        ))
    }

    /// True when the term names a known participant or entity.
    pub fn is_name(&self, term: &str) -> bool {
        term.starts_with('@') || self.entities.iter().any(|e| e.contains(term))
    }

    /// True when `terms` name what the key term `term` names: they hold `term` itself,
    /// or a word of an entity containing `term` that no other entity shares ("ledger"
    /// names "Ledger API"; "queue" does not name "Ledger Queue" when "Orbit Queue"
    /// exists too).
    pub fn names(&self, term: &str, terms: &std::collections::BTreeSet<String>) -> bool {
        if terms.contains(term) {
            return true;
        }
        let owners: Vec<&std::collections::BTreeSet<String>> =
            self.entities.iter().filter(|e| e.contains(term)).collect();
        owners.iter().flat_map(|e| e.iter()).any(|w| {
            terms.contains(w)
                && self
                    .entities
                    .iter()
                    .filter(|e| e.contains(w))
                    .all(|e| e.contains(term))
        })
    }
}

/// One content word of a claim: its comparison term (a participant's `@<id>` or the
/// word's [`stem`]) and whether a negation governs it. "ship weekly builds" and "do
/// not ship weekly builds" share no claim word.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClaimWord {
    /// Stem or canonical participant term.
    pub term: String,
    /// Inside a negation's scope.
    pub negated: bool,
}

/// Number words folded to their digits, so "retry thirty times" and "retry 30 times"
/// share a term. "one" is left out: it is as often a pronoun ("the new one") as a
/// count. Compounds ("twenty five") stay two terms and do not match "25".
const NUMBER_WORDS: &[(&str, &str)] = &[
    ("zero", "0"),
    ("two", "2"),
    ("three", "3"),
    ("four", "4"),
    ("five", "5"),
    ("six", "6"),
    ("seven", "7"),
    ("eight", "8"),
    ("nine", "9"),
    ("ten", "10"),
    ("eleven", "11"),
    ("twelve", "12"),
    ("thirteen", "13"),
    ("fourteen", "14"),
    ("fifteen", "15"),
    ("sixteen", "16"),
    ("seventeen", "17"),
    ("eighteen", "18"),
    ("nineteen", "19"),
    ("twenty", "20"),
    ("thirty", "30"),
    ("forty", "40"),
    ("fifty", "50"),
    ("sixty", "60"),
    ("seventy", "70"),
    ("eighty", "80"),
    ("ninety", "90"),
];

/// The digits of a normalized number word ([`NUMBER_WORDS`]).
fn number_word(lower: &str) -> Option<&'static str> {
    NUMBER_WORDS
        .iter()
        .find(|(w, _)| *w == lower)
        .map(|(_, d)| *d)
}

/// The comparison terms of one word: a numeric literal whole (`2.5` stays `2.5`,
/// `1,000` is `1000`, so "two or five" never states "2.5"), participant terms where
/// [`Vocabulary`] knows the name and the word may name someone ([`may_name`]),
/// digits of number words ([`NUMBER_WORDS`]), stems of the other non-stopword parts.
fn word_terms(word: &str, name_ok: bool, vocab: &Vocabulary) -> Vec<String> {
    if word.chars().any(|c| c.is_ascii_digit())
        && word
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == ',')
    {
        return vec![word.replace(',', "")];
    }
    normalize_label(word)
        .split(' ')
        .filter(|p| !p.is_empty())
        .filter_map(|p| match vocab.person(p, name_ok) {
            Some(t) => Some(t),
            None => match number_word(p) {
                Some(d) => Some(d.to_string()),
                None => (!STOPWORDS.contains(&p)).then(|| stem(p)),
            },
        })
        .collect()
}

/// Content words of a claim in order ([`tokenize`]), each marked with its polarity.
/// A negation ([`NEGATIONS`]) flips the polarity of the words after it up to the end
/// of its clause: a clause boundary or a clause word ("but", "instead"), so "we
/// can't not ship" affirms. Stopwords and clause words are dropped.
pub fn claim_words(text: &str, vocab: &Vocabulary) -> Vec<ClaimWord> {
    claim_clauses(text, vocab).concat()
}

/// [`claim_words`] split into clauses (at clause boundaries and clause words).
pub fn claim_clauses(text: &str, vocab: &Vocabulary) -> Vec<Vec<ClaimWord>> {
    let tokens = tokenize(text);
    let mut clauses: Vec<Vec<ClaimWord>> = vec![Vec::new()];
    let mut negated = false;
    for (i, token) in tokens.iter().enumerate() {
        let new_clause = match token {
            Token::Boundary(_) => true,
            Token::Word(w) => CLAUSE_WORDS.contains(&w.to_lowercase().as_str()),
        };
        if new_clause {
            negated = false;
            if clauses.last().is_some_and(|c| !c.is_empty()) {
                clauses.push(Vec::new());
            }
            continue;
        }
        let Token::Word(w) = token else { continue };
        if is_negation(w) {
            negated = !negated;
            continue;
        }
        if let Some(clause) = clauses.last_mut() {
            clause.extend(
                word_terms(w, may_name(&tokens, i), vocab)
                    .into_iter()
                    .map(|term| ClaimWord { term, negated }),
            );
        }
    }
    clauses.retain(|c| !c.is_empty());
    clauses
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
/// matching prediction must also name. A word is a key term when it is a known
/// participant or entity name ([`Vocabulary`], at any position, first word
/// included), is capitalized anywhere but the first word, is an acronym (two or
/// more capitals), holds a digit or underscore, or is a number word
/// ([`NUMBER_WORDS`], so "thirty" keys like "30"). Negation words never are.
/// Returned as comparison terms.
pub fn key_terms(text: &str, vocab: &Vocabulary) -> std::collections::BTreeSet<String> {
    let tokens = tokenize(text);
    let mut out = std::collections::BTreeSet::new();
    let mut first = true;
    for (i, token) in tokens.iter().enumerate() {
        let Token::Word(w) = token else { continue };
        let upper = w.chars().filter(char::is_ascii_uppercase).count();
        let capitalized = w.chars().next().is_some_and(char::is_uppercase);
        let special = w.chars().any(|c| c.is_ascii_digit() || c == '_')
            || number_word(&w.to_lowercase()).is_some();
        let shaped = (capitalized && !first) || upper >= 2 || special;
        first = false;
        if is_negation(w) {
            continue;
        }
        for term in word_terms(w, may_name(&tokens, i), vocab) {
            if shaped || vocab.is_name(&term) {
                out.insert(term);
            }
        }
    }
    out
}

/// Every key term of `gold` ([`key_terms`]) is named by the content words of `pred`:
/// a participant by the same term, an entity word by itself or by a word of the same
/// entity that no other entity shares ([`Vocabulary::names`]).
pub fn key_terms_present(gold: &str, pred: &str, vocab: &Vocabulary) -> bool {
    let p: std::collections::BTreeSet<String> = claim_words(pred, vocab)
        .into_iter()
        .map(|w| w.term)
        .collect();
    key_terms(gold, vocab).iter().all(|k| {
        if k.starts_with('@') {
            p.contains(k)
        } else {
            vocab.names(k, &p)
        }
    })
}

/// Gold claim words a prediction may leave out and still state the claim: none for a
/// phrasing of up to five distinct content words or one without key terms, one for
/// six to ten, two for eleven to fifteen, and so on. A short claim is its words
/// ("skip the Ledger step" is not stated by "the Ledger step was hard"), and a
/// phrasing without key terms has nothing else to pin its subject. Key terms can
/// never be left out ([`key_terms_present`]), nor can the phrasing's head (its first
/// content word) or its predicate (its first content word that is not a name), and a
/// word stated with the opposite polarity is a contradiction, not an omission
/// ([`contradicts`]).
pub fn allowed_missing(gold_words: usize, has_key_terms: bool) -> usize {
    if has_key_terms {
        gold_words.saturating_sub(1) / 5
    } else {
        0
    }
}

/// Distinct gold claim words (same term and polarity) found in the prediction, and
/// the distinct counts of both sides.
fn polar_overlap(g: &[ClaimWord], p: &[ClaimWord]) -> (usize, usize, usize) {
    let gs: std::collections::BTreeSet<&ClaimWord> = g.iter().collect();
    let ps: std::collections::BTreeSet<&ClaimWord> = p.iter().collect();
    (gs.intersection(&ps).count(), gs.len(), ps.len())
}

/// The same claim word with the opposite polarity.
fn flip(w: &ClaimWord) -> ClaimWord {
    ClaimWord {
        term: w.term.clone(),
        negated: !w.negated,
    }
}

/// The protected words of a gold phrasing's claim words `g`: its head (the first
/// content word) and its predicate (the first content word that is not a name or a
/// key term in `keys`). Both are the same word when the phrasing opens with a verb.
fn protected_words<'a>(
    g: &'a [ClaimWord],
    keys: &std::collections::BTreeSet<String>,
    vocab: &Vocabulary,
) -> Vec<&'a ClaimWord> {
    let head = g.first();
    let predicate = g
        .iter()
        .find(|w| !vocab.is_name(&w.term) && !keys.contains(&w.term));
    head.into_iter().chain(predicate).collect()
}

/// True when `pred` states some gold claim word only with the opposite polarity
/// (gold "ship weekly builds" and "we will not ship; weekly builds stay" disagree on
/// "ship", whatever else they share), or when one of its clauses restates the gold
/// claim retracted.
///
/// In one clause of a prediction, a gold word is *flipped* when the clause states it
/// with the opposite polarity, and *kept* when the clause states it with the gold's
/// polarity before its first negated word (the arguments a negation follows:
/// "Tamsin" in "Tamsin does not ship weekly builds", "Tamsin's weekly builds" in
/// "Tamsin's weekly builds do not ship"). The clause retracts the claim when some
/// gold word is flipped, the claim's protected words (head and predicate,
/// [`covers`]) are flipped or kept, and
///
/// - when no gold word is both flipped and kept, the flipped and kept words together
///   number as many as [`covers`] needs, in any order ("weekly builds do not ship on
///   Friday" retracts "weekly builds ship on Friday");
/// - when some gold word is both (the clause affirms and then negates it), every gold
///   word from the first flipped one in gold order onward is flipped and every one
///   before it is kept, with nothing left out ("ship weekly builds.not ship weekly
///   builds" retracts "ship weekly builds"; "Tamsin ships weekly builds after review
///   and not after audit" and "Tamsin ships weekly builds and doesn't ship nightly
///   builds" do not).
///
/// A clause that negates the predicate about something else does not retract ("we
/// don't ship nightly builds; we ship weekly builds"). Entity words match through
/// [`Vocabulary::names`].
fn clause_retracts(
    gw: &[ClaimWord],
    need: usize,
    anchors: &[&ClaimWord],
    clause: &[ClaimWord],
    vocab: &Vocabulary,
) -> bool {
    use std::collections::BTreeSet;
    let all: BTreeSet<ClaimWord> = clause.iter().cloned().collect();
    let before_negation = clause
        .iter()
        .position(|w| w.negated)
        .unwrap_or(clause.len());
    let prefix: BTreeSet<ClaimWord> = clause[..before_negation].iter().cloned().collect();
    let flipped = |w: &ClaimWord| stated(&flip(w), &all, vocab);
    let kept = |w: &ClaimWord| stated(w, &prefix, vocab);
    let Some(first) = gw.iter().position(flipped) else {
        return false;
    };
    let g: BTreeSet<&ClaimWord> = gw.iter().collect();
    if g.iter().any(|w| flipped(w) && kept(w)) {
        let subject: BTreeSet<&ClaimWord> = gw[..first].iter().collect();
        return g.iter().all(|w| {
            if subject.contains(w) {
                kept(w)
            } else {
                flipped(w)
            }
        });
    }
    let retracted = |w: &ClaimWord| flipped(w) || kept(w);
    anchors.iter().all(|w| retracted(w)) && g.iter().filter(|w| retracted(w)).count() >= need
}

/// True when `pred` states some gold claim word only with the opposite polarity
/// (gold "ship weekly builds" and "we will not ship; weekly builds stay" disagree on
/// "ship", whatever else they share), or when one of its clauses restates the gold
/// claim retracted ([`clause_retracts`]), whichever word the gold opens with:
/// "Tamsin does not ship weekly builds" retracts "Tamsin ships weekly builds".
pub fn contradicts(gold: &str, pred: &str, vocab: &Vocabulary) -> bool {
    use std::collections::BTreeSet;
    let gw = claim_words(gold, vocab);
    let g: BTreeSet<ClaimWord> = gw.iter().cloned().collect();
    if g.is_empty() {
        return false;
    }
    let p: BTreeSet<ClaimWord> = claim_words(pred, vocab).into_iter().collect();
    if g.iter()
        .any(|w| stated(&flip(w), &p, vocab) && !stated(w, &p, vocab))
    {
        return true;
    }
    let keys = key_terms(gold, vocab);
    let need = g.len() - allowed_missing(g.len(), !keys.is_empty());
    let anchors = protected_words(&gw, &keys, vocab);
    claim_clauses(pred, vocab)
        .iter()
        .any(|clause| clause_retracts(&gw, need, &anchors, clause, vocab))
}

/// True when `pred` states a gold claim word: the same term with the same polarity,
/// or, for an entity word, a word of the same entity that no other entity shares
/// ([`Vocabulary::names`]) with the same polarity.
fn stated(w: &ClaimWord, p: &std::collections::BTreeSet<ClaimWord>, vocab: &Vocabulary) -> bool {
    if p.contains(w) {
        return true;
    }
    if w.term.starts_with('@') || !vocab.is_name(&w.term) {
        return false;
    }
    let same: std::collections::BTreeSet<String> = p
        .iter()
        .filter(|x| x.negated == w.negated)
        .map(|x| x.term.clone())
        .collect();
    vocab.names(&w.term, &same)
}

/// True when `pred` states the protected words of `gold` (its head, the first
/// content word, and its predicate, the first content word that is not a name or
/// key term) and enough of its other claim words with the same polarity
/// ([`allowed_missing`]), and contradicts none of them ([`contradicts`]).
pub fn covers(gold: &str, pred: &str, vocab: &Vocabulary) -> bool {
    let g = claim_words(gold, vocab);
    let p: std::collections::BTreeSet<ClaimWord> = claim_words(pred, vocab).into_iter().collect();
    let gs: std::collections::BTreeSet<&ClaimWord> = g.iter().collect();
    let keys = key_terms(gold, vocab);
    let protected_stated = protected_words(&g, &keys, vocab)
        .iter()
        .all(|w| stated(w, &p, vocab));
    let inter = gs.iter().filter(|w| stated(w, &p, vocab)).count();
    let ng = gs.len();
    ng > 0
        && protected_stated
        && ng - inter <= allowed_missing(ng, !keys.is_empty())
        && !contradicts(gold, pred, vocab)
}

/// Dice coefficient of polar claim words ([`claim_words`]): shared distinct words over
/// the gold's distinct words plus every content word of the prediction, so padding a
/// prediction with repeats lowers it. When either side has no content word,
/// [`token_dice`] on the full texts.
pub fn claim_dice(gold: &str, pred: &str, vocab: &Vocabulary) -> f64 {
    let (g, p) = (claim_words(gold, vocab), claim_words(pred, vocab));
    if g.is_empty() || p.is_empty() {
        return token_dice(gold, pred);
    }
    let (inter, ng, _) = polar_overlap(&g, &p);
    2.0 * inter as f64 / (ng + p.len()) as f64
}

/// How much of `gold` a prediction states inside a longer sentence: the share of the
/// gold's claim words (same term, same polarity) found in `pred`. A paraphrase that
/// restates the gold claim with context scores high here while its Dice coefficient
/// stays low. Zero when the gold has fewer than two content words (too little to
/// contain), when the prediction is longer than [`CONTAINMENT_MAX_EXPANSION`] allows
/// (repeats count), or when the gold words found are spread wider than
/// [`CONTAINMENT_MAX_SPREAD`] allows. Coverage itself is checked by [`covers`].
pub fn containment(gold: &str, pred: &str, vocab: &Vocabulary) -> f64 {
    let (g, p) = (claim_words(gold, vocab), claim_words(pred, vocab));
    let (inter, ng, _) = polar_overlap(&g, &p);
    if ng < 2 || inter == 0 || p.len() > CONTAINMENT_MAX_EXPANSION * ng + CONTAINMENT_EXTRA_WORDS {
        return 0.0;
    }
    let wanted: std::collections::BTreeSet<&ClaimWord> =
        g.iter().filter(|w| p.contains(w)).collect();
    if min_window(&p, &wanted) > CONTAINMENT_MAX_SPREAD * ng + CONTAINMENT_SPREAD_SLACK {
        return 0.0;
    }
    inter as f64 / ng as f64
}

/// Length of the shortest run of `seq` holding every word of `wanted`.
fn min_window(seq: &[ClaimWord], wanted: &std::collections::BTreeSet<&ClaimWord>) -> usize {
    let mut best = usize::MAX;
    for start in 0..seq.len() {
        if !wanted.contains(&seq[start]) {
            continue;
        }
        let mut seen = std::collections::BTreeSet::new();
        for (end, w) in seq.iter().enumerate().skip(start) {
            if wanted.contains(w) {
                seen.insert(w);
                if seen.len() == wanted.len() {
                    best = best.min(end - start + 1);
                    break;
                }
            }
        }
    }
    best
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
            &["use", "used", "uses"],
            &["retry", "retries", "retried"],
            &["need", "needs"],
        ] {
            let stems: std::collections::BTreeSet<String> = group.iter().map(|w| stem(w)).collect();
            assert_eq!(stems.len(), 1, "{group:?} -> {stems:?}");
        }
        assert_eq!(stem("focus"), "focus");
        assert_eq!(stem("class"), "class");
    }

    fn names() -> Vocabulary {
        let mut v = Vocabulary::default();
        v.add_person("tamsin", ["Tamsin Reed", "Tam"]);
        v.add_person("quill", ["Quill Obi"]);
        v.add_entity("Ledger API");
        v
    }

    #[test]
    fn key_terms_and_containment() {
        let none = Vocabulary::default();
        let k = key_terms("Skip the Ledger step and ask QA about order_svc v2", &none);
        // order_svc normalizes to two words; both are kept.
        for want in ["ledger", "qa", "order", "svc", "v2"] {
            assert!(k.contains(want), "{want} in {k:?}");
        }
        assert!(
            !k.contains("skip"),
            "a capitalized first word is not a key term by shape"
        );
        assert!(key_terms_present(
            "ask Tamsin",
            "we will ask tamsin today",
            &none
        ));
        assert!(!key_terms_present(
            "ask Tamsin",
            "we will ask Quill today",
            &none
        ));
        assert!(
            (containment("skip importer", "we skip the importer this week", &none) - 1.0).abs()
                < 1e-12
        );
        assert_eq!(
            containment("importer", "skip the importer", &none),
            0.0,
            "one word is too short"
        );
    }

    #[test]
    fn known_names_are_key_terms_at_the_start_of_a_sentence() {
        let v = names();
        let gold = "Tamsin moves to dashboard work";
        let k = key_terms(gold, &v);
        assert!(k.contains("@tamsin"), "{k:?}");
        assert!(!key_terms_present(
            gold,
            "Quill moves to dashboard work",
            &v
        ));
        // any spelling of the same participant states the name
        assert!(key_terms_present(
            gold,
            "Tam moved over to the dashboard work",
            &v
        ));
        // an entity word is a key term even lower-case and first
        assert!(key_terms("ledger is deferred", &v).contains("ledger"));
        // a negation word is never a key term
        assert!(key_terms("We Don't ship", &Vocabulary::default()).is_empty());
    }

    #[test]
    fn negation_governs_the_rest_of_its_clause() {
        let v = Vocabulary::default();
        let polar = |t: &str| -> Vec<(String, bool)> {
            claim_words(t, &v)
                .into_iter()
                .map(|w| (w.term, w.negated))
                .collect()
        };
        assert!(polar("do not ship weekly builds").iter().all(|(_, n)| *n));
        assert!(polar("we won't ship weekly builds").iter().all(|(_, n)| *n));
        assert!(polar("ship weekly builds").iter().all(|(_, n)| !*n));
        // the negation ends with its clause
        let p = polar("ship weekly builds, not nightly ones");
        assert_eq!(p[0], ("ship".into(), false));
        assert!(p.last().unwrap().1);
        let p = polar("No, we ship weekly builds");
        assert!(p.iter().all(|(_, n)| !*n), "{p:?}");
        let p = polar("never the importer but ship the dashboard");
        assert_eq!(p[0], ("importer".into(), true));
        assert!(p[1..].iter().all(|(_, n)| !*n), "{p:?}");
    }

    #[test]
    fn a_claim_and_its_negation_do_not_overlap() {
        let v = Vocabulary::default();
        assert!(!covers(
            "ship weekly builds",
            "do not ship weekly builds",
            &v
        ));
        assert!(!covers(
            "do not ship weekly builds",
            "ship weekly builds",
            &v
        ));
        assert_eq!(
            containment("ship weekly builds", "we will not ship weekly builds", &v),
            0.0
        );
        assert_eq!(
            claim_dice("ship weekly builds", "do not ship weekly builds", &v),
            0.0
        );
        assert!(covers(
            "do not ship weekly builds",
            "We decided we won't ship weekly builds.",
            &v
        ));
        assert!(covers(
            "ship weekly builds",
            "ship weekly builds, not nightly ones",
            &v
        ));
    }

    #[test]
    fn short_golds_need_every_claim_word() {
        let v = names();
        // Kimi reproducer: two of three gold words plus the name, no predicate.
        let gold = "Skip the Ledger step for now";
        let pred = "Ledger told us the step was hard";
        assert!(!covers(gold, pred, &v));
        assert!(!covers(
            "Tamsin moves from the importer to the dashboard work",
            "Tamsin complained about the importer and the dashboard work",
            &v
        ));
        assert_eq!(allowed_missing(3, true), 0);
        assert_eq!(allowed_missing(5, true), 0);
        assert_eq!(allowed_missing(6, true), 1);
        assert_eq!(
            allowed_missing(6, false),
            0,
            "no key terms: nothing else pins the subject"
        );
        // a long keyed claim may drop one qualifier
        assert!(covers(
            "Tamsin moves the importer retries to the dashboard backlog this sprint",
            "Tamsin moves the importer retries to the dashboard backlog",
            &v
        ));
    }

    /// Codex round-1 B2: a contraction joined to the next word still negates it.
    #[test]
    fn punctuation_joined_contractions_still_negate() {
        let v = Vocabulary::default();
        for pred in [
            "don't-ship weekly builds",
            "(don't) ship weekly builds",
            "we \u{2018}won\u{2019}t\u{2019} ship weekly builds",
            "we won't-ship weekly builds",
        ] {
            assert!(!covers("ship weekly builds", pred, &v), "{pred}");
        }
        // inner dots and commas stay inside a word; they do not end a clause
        let p = claim_words("do not ship v2.5 weekly builds", &v);
        assert!(p.iter().all(|w| w.negated), "{p:?}");
    }

    /// Kimi round-1 M6: the head of a long gold (a sentence-initial name the golden
    /// does not know) is never the omitted word.
    #[test]
    fn the_head_of_a_gold_is_never_the_omitted_word() {
        let v = Vocabulary::default();
        let gold = "Tamsin moves the importer retries to the API backlog this sprint";
        assert!(covers(
            gold,
            "Tamsin moves the importer retries to the API backlog",
            &v
        ));
        assert!(!covers(
            gold,
            "Quill moves the importer retries to the API backlog this sprint",
            &v
        ));
    }

    /// Kimi round-1 M8 and M9: a second negation affirms again, and clause words are
    /// not claim words.
    #[test]
    fn double_negation_and_clause_words() {
        let v = Vocabulary::default();
        assert!(covers(
            "ship weekly builds",
            "we can't not ship weekly builds",
            &v
        ));
        assert!(covers("ship builds instead", "ship builds", &v));
        assert!(claim_words("however we ship", &v)
            .iter()
            .all(|w| w.term != "however"));
    }

    /// Codex round-2 B2 and M3: a retraction of the claim's head, or a negation
    /// behind a comma with no space, is caught; a negated side word is not a
    /// retraction.
    #[test]
    fn retractions_and_tight_commas_are_caught() {
        let v = Vocabulary::default();
        let gold = "ship weekly builds";
        assert!(!covers(
            gold,
            "ship weekly builds; do not ship weekly builds",
            &v
        ));
        assert!(!covers(
            gold,
            "ship weekly builds,not ship weekly builds",
            &v
        ));
        assert!(!covers(
            gold,
            "ship weekly builds.not ship weekly builds",
            &v
        ));
        assert!(covers(gold, "ship weekly builds, not nightly builds", &v));
        // a comma between digits stays inside the number
        assert!(key_terms("keep 1,000 builds", &v).contains("1000"));
    }

    /// Codex round-2 B1: padding a prediction with repeats lowers its Dice score.
    #[test]
    fn repeats_lower_dice() {
        let v = Vocabulary::default();
        let padded = format!("ship builds {}", "lunch ".repeat(18));
        assert!(claim_dice("ship builds", &padded, &v) < 0.6);
        assert!((claim_dice("ship builds", "ship builds lunch", &v) - 0.8).abs() < 1e-12);
    }

    /// Codex round-2 M5: a function-word name opening a question is the function
    /// word.
    #[test]
    fn function_word_names_follow_context() {
        let mut v = Vocabulary::default();
        v.add_person("will", ["Will Park"]);
        assert!(key_terms("Will we ship weekly builds?", &v).is_empty());
        assert!(covers(
            "Will we ship weekly builds?",
            "Are we shipping weekly builds?",
            &v
        ));
        assert!(key_terms("Will Park ships weekly builds", &v).contains("@will"));
        assert!(key_terms("ask Will about the builds", &v).contains("@will"));
        // Kimi round-2 minor: after a comma too.
        assert!(key_terms("yes, Will we ask about the builds?", &v).is_empty());
    }

    /// Kimi round-2 M1 with Codex round-2 B2: a clause that restates the negated
    /// claim retracts it; a clause that negates the head about something else does
    /// not.
    #[test]
    fn retraction_is_a_clause_that_negates_the_claim() {
        let v = Vocabulary::default();
        assert!(!covers(
            "ship weekly builds",
            "ship weekly builds; do not ship weekly builds",
            &v
        ));
        assert!(covers(
            "ship weekly builds",
            "we don't ship nightly builds; we ship weekly builds",
            &v
        ));
        assert!(covers(
            "do not ship weekly builds",
            "we do not ship weekly builds; we ship nightly builds",
            &v
        ));
        // Kimi round-2 minor: a sentence whose space was dropped still ends.
        assert!(covers(
            "ship nightly builds",
            "We will not ship weekly builds.We ship nightly builds",
            &v
        ));
    }

    /// Kimi round-2 M2: the predicate of a long gold that opens with a name is never
    /// the omitted word.
    #[test]
    fn the_predicate_is_never_the_omitted_word() {
        let v = names();
        let gold = "Tamsin moves the importer retries to the dashboard backlog this sprint";
        assert!(!covers(
            gold,
            "Tamsin complained about the importer retries, the dashboard backlog, and this sprint",
            &v
        ));
        assert!(covers(
            gold,
            "Tamsin moves the importer retries to the dashboard backlog",
            &v
        ));
    }

    /// Kimi round-2 M3: a name word two participants share names neither alone.
    #[test]
    fn shared_name_words_name_neither_participant() {
        let mut v = Vocabulary::default();
        v.add_person("tam_ly", ["Tam Ly"]);
        v.add_person("tamsin", ["Tamsin Reed", "Tam"]);
        assert_eq!(v.ambiguous_names().len(), 1, "{:?}", v.ambiguous_names());
        let gold = "Tamsin argued for weekly builds";
        assert!(!key_terms_present(
            gold,
            "Tam Ly argued for weekly builds",
            &v
        ));
        assert!(key_terms_present(
            gold,
            "Tamsin Reed argued for weekly builds",
            &v
        ));
    }

    /// Kimi round-2 M4: a distinctive word of an entity names it in a prediction,
    /// as it does in an alias.
    #[test]
    fn a_distinctive_entity_word_names_the_entity() {
        let mut v = Vocabulary::default();
        v.add_entity("Ledger API");
        assert!(key_terms_present(
            "fix the api retries",
            "fix the Ledger retries",
            &v
        ));
        assert!(covers(
            "fix the api retries",
            "we will fix the Ledger retries",
            &v
        ));
        v.add_entity("Ledger Queue");
        assert!(!key_terms_present(
            "fix the api retries",
            "fix the Ledger retries",
            &v
        ));
    }

    /// Codex round-1 M3: the one-word allowance of a long gold never admits the gold's
    /// own word stated with the opposite polarity.
    #[test]
    fn a_contradicted_word_is_not_an_omission() {
        let v = names();
        let gold = "Tamsin ships weekly builds after quarterly review";
        assert!(covers(
            gold,
            "Tamsin ships weekly builds after the review",
            &v
        ));
        let pred = "Tamsin will not ship; weekly builds remain after quarterly review";
        assert!(contradicts(gold, pred, &v));
        assert!(!covers(gold, pred, &v));
    }

    /// Final review BLOCKER (Codex 1, Kimi M1): a retraction is anchored on the
    /// claim's predicate and arguments, not its first word, so a name-headed or
    /// noun-headed claim stated and then retracted does not match.
    #[test]
    fn a_retraction_is_anchored_on_the_predicate_not_the_first_word() {
        let v = names();
        // name-headed: the name comes before the negation and is never flipped
        let gold = "Tamsin ships weekly builds";
        let pred = "Tamsin ships weekly builds; Tamsin does not ship weekly builds";
        assert!(contradicts(gold, pred, &v));
        assert!(!covers(gold, pred, &v));
        assert!(!covers(
            gold,
            "Tamsin ships weekly builds. Later Tam said she won't ship weekly builds",
            &v
        ));
        // noun-headed: the subject noun phrase comes before the negation
        let none = Vocabulary::default();
        let gold = "weekly builds ship on Friday";
        assert!(covers(gold, "weekly builds ship on Friday", &none));
        assert!(!covers(
            gold,
            "weekly builds ship on Friday; weekly builds do not ship on Friday",
            &none
        ));
        // entity-headed: an entity sibling word states the entity when retracted too
        let gold = "Ledger API ships weekly builds";
        assert!(!covers(
            gold,
            "Ledger API ships weekly builds; the Ledger does not ship weekly builds",
            &v
        ));
        // verb-headed still holds
        assert!(!covers(
            "ship weekly builds",
            "ship weekly builds; we never ship weekly builds",
            &none
        ));
        // negated gold, name-headed: an affirming clause retracts it
        let gold = "Tamsin does not ship weekly builds";
        assert!(!covers(
            gold,
            "Tamsin does not ship weekly builds; Tamsin ships weekly builds",
            &v
        ));
        assert!(covers(gold, "Tamsin won't ship weekly builds", &v));
        // Codex r2eval round 1 BLOCKER: the retraction may reorder the arguments
        let gold = "Tamsin ships weekly builds";
        assert!(!covers(
            gold,
            "Tamsin ships weekly builds; Tamsin's weekly builds do not ship",
            &v
        ));
        // stated and retracted within one clause
        assert!(!covers(
            gold,
            "Tamsin ships weekly builds and then Tamsin does not ship weekly builds",
            &v
        ));
    }

    /// The anchored retraction rule stays narrow: a clause that negates the
    /// predicate about something else, states a gold word with both polarities, or
    /// negates a side word after the full claim, is not a retraction.
    #[test]
    fn a_contrast_about_something_else_is_not_a_retraction() {
        let v = names();
        let gold = "Tamsin ships weekly builds";
        for pred in [
            "Tamsin ships weekly builds, not nightly builds",
            "Tamsin ships weekly builds and doesn't ship nightly builds",
            "Tamsin ships weekly builds that never break",
            "Quill does not ship nightly builds; Tamsin ships weekly builds",
            "Tamsin doesn't ship nightly builds; Tamsin ships weekly builds",
        ] {
            assert!(!contradicts(gold, pred, &v), "{pred}");
            assert!(covers(gold, pred, &v), "{pred}");
        }
        let none = Vocabulary::default();
        assert!(covers(
            "weekly builds ship on Friday",
            "weekly builds ship on Friday; nightly builds do not ship on Friday",
            &none
        ));
        // Codex r2eval round 1 MAJOR: a negated qualifier after the affirmed claim
        // does not retract the claim, even when the gold's allowance is one word.
        let gold = "Tamsin ships weekly builds after review";
        let pred = "Tamsin ships weekly builds after review and not after audit";
        assert!(!contradicts(gold, pred, &v));
        assert!(covers(gold, pred, &v));
    }

    /// GLM final M11: number words and digits compare equal and both key the claim.
    #[test]
    fn number_words_match_digits() {
        let v = Vocabulary::default();
        assert!(key_terms("retry thirty times", &v).contains("30"));
        assert!(key_terms("retry 30 times", &v).contains("30"));
        assert!(covers("retry 30 times", "retry thirty times", &v));
        assert!(covers("retry thirty times", "we retry 30 times", &v));
        assert!(key_terms_present(
            "retry 30 times",
            "retry thirty times",
            &v
        ));
        assert!(!covers("retry 30 times", "retry forty times", &v));
        assert!(!key_terms_present(
            "retry thirty times",
            "retry 40 times",
            &v
        ));
        // "one" is not folded: it is as often a pronoun
        assert!(key_terms("ship the new one", &v).is_empty());
        // Codex r2eval round 1 MAJOR: a numeric literal is one term, so "two or
        // five" does not state "2.5"; grouping commas are dropped.
        assert!(!covers("wait 2.5 seconds", "wait two or five seconds", &v));
        assert!(!key_terms_present(
            "wait 2.5 seconds",
            "wait 2 or 5 seconds",
            &v
        ));
        assert!(covers("wait 2.5 seconds", "we wait 2.5 seconds", &v));
        assert!(covers("keep 1,000 builds", "keep 1000 builds", &v));
    }

    /// Codex round-1 M4: a participant name is a name where it is capitalized, even
    /// when it is also a function word, and never where it is lower-case.
    #[test]
    fn participant_names_follow_capitalization() {
        let mut v = Vocabulary::default();
        v.add_person("will", ["Will Park"]);
        v.add_person("may", ["May Ito"]);
        v.add_person("quill", ["Quill Obi"]);
        assert!(key_terms("Will moves dashboard", &v).contains("@will"));
        assert!(!key_terms_present(
            "Will moves dashboard",
            "Quill moves dashboard",
            &v
        ));
        assert!(key_terms("May moves dashboard", &v).contains("@may"));
        assert!(!key_terms_present(
            "May moves dashboard",
            "We may move dashboard",
            &v
        ));
        assert!(key_terms("we may move the dashboard", &v).is_empty());
        assert!(key_terms_present(
            "May moves dashboard",
            "May Ito moved the dashboard",
            &v
        ));
    }

    #[test]
    fn long_predictions_cannot_game_containment() {
        let v = Vocabulary::default();
        // 2-word gold inside a 20-word ramble (exactly the old 0.1 floor): no match.
        let ramble = "ship builds hiring plans office seating travel budgets lunch vendors parking passes badge printers laptop refresh cycles conference talks onboarding";
        assert_eq!(containment("ship builds", ramble, &v), 0.0);
        // Up to four times the gold's content words, together: a restatement.
        assert_eq!(
            containment("ship builds", "we ship builds after the review", &v),
            1.0
        );
        // Codex round-1 M9: repeating a few distractors does not stay under the bound.
        let repeated = format!(
            "ship builds; {}",
            "hiring plans office seating travel budgets ".repeat(4)
        );
        assert_eq!(containment("ship builds", &repeated, &v), 0.0);
        // The gold words scattered across the prediction: not a restatement.
        assert_eq!(
            containment(
                "skip importer",
                "skip lunch vendors travel parking badges printers importer",
                &v
            ),
            0.0
        );
        // Kimi round-1 M5: ordinary sentences around a short gold still restate it.
        for pred in [
            "we decided to skip the old broken legacy importer",
            "after a long discussion about priorities and staffing the team decided to skip the importer",
        ] {
            assert_eq!(containment("skip importer", pred, &v), 1.0, "{pred}");
        }
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
