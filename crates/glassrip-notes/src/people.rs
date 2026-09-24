//! Participants and the alias table used to match names in speech and on screen.
//!
//! Names come from `--participants` and from conferencing tile names read by OCR.
//! A transcript word matches a name by exact letters, by the audio crate's
//! Soundex-style phonetic key plus a bounded edit distance (so an ASR spelling
//! such as a homophone of a first name still maps), or by Jaro-Winkler similarity.
//! Tile text matches full names, including names truncated with an ellipsis.

use glassrip_audio::vocab::{is_common_word, letters, normalized_edit, phonetic_key};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A known participant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Person {
    /// Stable person id (slug of the display name).
    pub person_id: String,
    /// Display name.
    pub display_name: String,
    /// Other spellings (first name, ASR variants, tile names).
    pub aliases: Vec<String>,
}

impl Person {
    /// First word of the display name.
    pub fn first_name(&self) -> &str {
        self.display_name
            .split_whitespace()
            .next()
            .unwrap_or(&self.display_name)
    }
}

/// Slug used as a person id: lowercase letters and digits joined by `-`.
pub fn slug(name: &str) -> String {
    let mut out = String::new();
    for word in name.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('-');
        }
        out.extend(word.chars().flat_map(char::to_lowercase));
    }
    out
}

/// How a name matched.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum MatchKind {
    /// Same letters as an alias.
    Exact,
    /// Truncated tile name that is a prefix of a full name.
    Prefix,
    /// Same phonetic key and a small edit distance.
    Phonetic,
    /// High Jaro-Winkler similarity.
    Fuzzy,
}

/// A name match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NameMatch {
    /// Index into [`AliasTable::people`].
    pub person: usize,
    /// Match strength in [0, 1].
    pub score: f64,
    /// How it matched.
    pub kind: MatchKind,
}

#[derive(Debug, Clone)]
struct Entry {
    person: usize,
    letters: String,
    key: String,
    /// Single-word alias (a first or last name) as opposed to a full name.
    single: bool,
}

/// Participants plus every spelling that maps to each of them.
#[derive(Debug, Clone, Default)]
pub struct AliasTable {
    people: Vec<Person>,
    entries: Vec<Entry>,
}

/// Minimum Jaro-Winkler similarity for a fuzzy match.
const FUZZY_MIN: f64 = 0.9;
/// Largest normalized edit distance accepted with a phonetic-key match.
const PHONETIC_MAX_EDIT: f64 = 0.5;
/// Shortest truncated tile name accepted as a prefix match (letters).
const PREFIX_MIN_LETTERS: usize = 6;

impl AliasTable {
    /// Builds a table from display names (`"First Last"`, ...). Duplicates are merged.
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Self {
        let mut t = Self::default();
        for n in names {
            t.add_person(n.as_ref());
        }
        t
    }

    /// Adds a participant (or returns the existing one with the same slug).
    pub fn add_person(&mut self, display_name: &str) -> usize {
        let display_name = display_name.trim();
        let id = slug(display_name);
        if let Some(i) = self.people.iter().position(|p| p.person_id == id) {
            return i;
        }
        let idx = self.people.len();
        self.people.push(Person {
            person_id: id,
            display_name: display_name.to_string(),
            aliases: Vec::new(),
        });
        self.push_entry(idx, display_name);
        let words: Vec<&str> = display_name.split_whitespace().collect();
        if words.len() > 1 {
            if let Some(first) = words.first() {
                self.add_alias(idx, first);
            }
            if let Some(last) = words.last() {
                self.add_alias(idx, last);
            }
        }
        idx
    }

    /// Adds another spelling for a participant.
    pub fn add_alias(&mut self, person: usize, alias: &str) {
        let alias = alias.trim();
        let Some(p) = self.people.get_mut(person) else {
            return;
        };
        if letters(alias).is_empty() || p.aliases.iter().any(|a| letters(a) == letters(alias)) {
            return;
        }
        if letters(&p.display_name) != letters(alias) {
            p.aliases.push(alias.to_string());
        }
        self.push_entry(person, alias);
    }

    fn push_entry(&mut self, person: usize, text: &str) {
        let l = letters(text);
        if l.is_empty()
            || self
                .entries
                .iter()
                .any(|e| e.person == person && e.letters == l)
        {
            return;
        }
        self.entries.push(Entry {
            person,
            key: phonetic_key(text),
            single: !text.trim().contains(char::is_whitespace),
            letters: l,
        });
    }

    /// Participants in insertion order.
    pub fn people(&self) -> &[Person] {
        &self.people
    }

    /// Index of a person id.
    pub fn index_of(&self, person_id: &str) -> Option<usize> {
        self.people.iter().position(|p| p.person_id == person_id)
    }

    /// Person by index.
    pub fn person(&self, idx: usize) -> Option<&Person> {
        self.people.get(idx)
    }

    /// Matches one transcript word (punctuation ignored) against single-word aliases.
    ///
    /// Lowercase common words never match phonetically or fuzzily, so ordinary words
    /// that happen to sound like a name are not treated as names.
    pub fn match_word(&self, word: &str) -> Option<NameMatch> {
        let l = letters(word);
        if l.len() < 3 {
            return None;
        }
        let capitalized = word
            .trim_start_matches(|c: char| !c.is_alphanumeric())
            .chars()
            .next()
            .is_some_and(char::is_uppercase);
        let loose_ok = capitalized || !is_common_word(&l);
        let key = phonetic_key(word);
        let mut best: Option<NameMatch> = None;
        for e in self.entries.iter().filter(|e| e.single) {
            let m = if e.letters == l {
                Some((1.0, MatchKind::Exact))
            } else if !loose_ok {
                None
            } else if e.key == key
                && e.letters.chars().next() == l.chars().next()
                && normalized_edit(&e.letters, &l) <= PHONETIC_MAX_EDIT
            {
                Some((
                    0.85 - 0.2 * normalized_edit(&e.letters, &l),
                    MatchKind::Phonetic,
                ))
            } else {
                let jw = strsim::jaro_winkler(&e.letters, &l);
                (jw >= FUZZY_MIN).then_some((jw * 0.8, MatchKind::Fuzzy))
            };
            if let Some((score, kind)) = m {
                if best.is_none_or(|b| score > b.score) {
                    best = Some(NameMatch {
                        person: e.person,
                        score,
                        kind,
                    });
                }
            }
        }
        best
    }

    /// Matches on-screen text (a tile name, possibly truncated with `...`).
    pub fn match_screen_text(&self, text: &str) -> Option<NameMatch> {
        let trimmed = text.trim().trim_end_matches(['.', '\u{2026}']).trim();
        let truncated = trimmed.len() + 1 < text.trim().len() || text.contains('\u{2026}');
        let l = letters(trimmed);
        if l.len() < 3 {
            return None;
        }
        let mut best: Option<NameMatch> = None;
        let mut consider = |m: NameMatch| {
            if best.is_none_or(|b| m.score > b.score) {
                best = Some(m);
            }
        };
        for e in &self.entries {
            if e.letters == l {
                consider(NameMatch {
                    person: e.person,
                    score: if e.single { 0.8 } else { 1.0 },
                    kind: MatchKind::Exact,
                });
            } else if truncated && l.len() >= PREFIX_MIN_LETTERS && e.letters.starts_with(&l) {
                consider(NameMatch {
                    person: e.person,
                    score: 0.9,
                    kind: MatchKind::Prefix,
                });
            } else if !e.single {
                let jw = strsim::jaro_winkler(&e.letters, &l);
                if jw >= FUZZY_MIN {
                    consider(NameMatch {
                        person: e.person,
                        score: jw * 0.9,
                        kind: MatchKind::Fuzzy,
                    });
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> AliasTable {
        AliasTable::from_names(&["Avery Quinn", "Rohan Dasgupta", "Mira Okafor"])
    }

    #[test]
    fn slugs_and_first_names() {
        let t = table();
        assert_eq!(t.people()[1].person_id, "rohan-dasgupta");
        assert_eq!(t.people()[1].first_name(), "Rohan");
        assert_eq!(t.people()[1].aliases, vec!["Rohan", "Dasgupta"]);
    }

    #[test]
    fn exact_and_phonetic_words() {
        let t = table();
        let m = t.match_word("Mira,").unwrap();
        assert_eq!((m.person, m.kind), (2, MatchKind::Exact));
        // an ASR respelling with the same phonetic key
        let m = t.match_word("Rowan").unwrap();
        assert_eq!(m.person, 1);
        assert_ne!(m.kind, MatchKind::Exact);
        assert!(t.match_word("the").is_none());
        assert!(
            t.match_word("every").is_none(),
            "common lowercase word must not match"
        );
    }

    #[test]
    fn screen_text_handles_truncation() {
        let t = table();
        let m = t.match_screen_text("Rohan Dasgu...").unwrap();
        assert_eq!((m.person, m.kind), (1, MatchKind::Prefix));
        let m = t.match_screen_text("Avery Quinn").unwrap();
        assert_eq!((m.person, m.kind), (0, MatchKind::Exact));
        assert!(t.match_screen_text("Overview").is_none());
    }

    #[test]
    fn aliases_are_deduplicated() {
        let mut t = table();
        t.add_alias(0, "avery");
        t.add_alias(0, "Aves");
        assert_eq!(t.people()[0].aliases, vec!["Avery", "Quinn", "Aves"]);
        assert_eq!(t.add_person("Avery Quinn"), 0);
    }
}
