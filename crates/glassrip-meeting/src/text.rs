//! Text normalization, fuzzy matching, and the participant alias table.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::difflib;

/// Fuzzy match threshold on [`difflib::ratio`] (spec 6.11).
pub const FUZZY_THRESHOLD: f64 = 0.85;

/// Lowercase, trim, and collapse internal whitespace (the same rule the validator uses).
pub fn normalize(text: &str) -> String {
    glassrip_vision::board::normalize(text)
}

/// Similarity of two texts after normalization, in `[0, 1]`.
pub fn similarity(a: &str, b: &str) -> f64 {
    difflib::ratio(&normalize(a), &normalize(b))
}

/// True when the normalized texts are equal or their ratio reaches `threshold`.
pub fn fuzzy_eq(a: &str, b: &str, threshold: f64) -> bool {
    let (na, nb) = (normalize(a), normalize(b));
    na == nb || difflib::ratio(&na, &nb) >= threshold
}

/// True when the text is an illegible or elided reading that must not anchor anything.
pub fn is_unreliable(text: &str) -> bool {
    let n = normalize(text);
    n.is_empty() || n.contains("[illegible]") || n.ends_with("...") || n.ends_with('\u{2026}')
}

/// True for arrow glyphs and similar connector marks.
fn is_arrow(c: char) -> bool {
    matches!(u32::from(c), 0x2190..=0x21FF | 0x27F0..=0x27FF | 0x2900..=0x297F | 0x2B00..=0x2B2F)
        || matches!(c, '>' | '<' | '-' | '=' | '~' | '\u{2014}' | '\u{2013}')
}

/// An edge label with arrow glyphs trimmed from both ends; empty when nothing
/// alphanumeric is left (a label of only glyphs or punctuation is not a label).
pub fn clean_label(label: &str) -> String {
    let t = label.trim_matches(|c: char| c.is_whitespace() || is_arrow(c));
    // Line style words written into the label slot describe the line, not a label.
    let style_word = matches!(
        normalize(t).as_str(),
        "solid" | "dashed" | "dotted" | "line" | "arrow" | "none" | "null"
    );
    if !style_word && t.chars().any(char::is_alphanumeric) {
        t.to_string()
    } else {
        String::new()
    }
}

/// Alphanumeric tokens of the normalized text.
pub fn tokens(text: &str) -> Vec<String> {
    normalize(text)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// True when `short` reads as a piece of `long`: its tokens are a strict subset of
/// `long`'s, or its normalized text is contained in `long`'s starting at a word
/// boundary (a cut-off reading).
pub fn is_fragment_of(short: &str, long: &str) -> bool {
    let (s, l) = (normalize(short), normalize(long));
    if s.chars().filter(|c| c.is_alphanumeric()).count() < 3 || s.len() >= l.len() {
        return false;
    }
    let (ts, tl) = (tokens(short), tokens(long));
    let subset = !ts.is_empty() && ts.len() < tl.len() && ts.iter().all(|t| tl.contains(t));
    // Contained text must start at a word boundary ("updates" in "shared updates", or
    // a cut-off "ledger_wr"), not inside a word ("eta" in "theta").
    let at_boundary = l.match_indices(s.as_str()).any(|(i, _)| {
        l[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric())
    });
    subset || at_boundary
}

/// One meeting participant and the names they may appear under.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Participant {
    /// Stable id used in outputs.
    pub person_id: String,
    /// Preferred display name.
    pub display_name: String,
    /// Other spellings (first name, nickname, common misreadings).
    #[serde(default)]
    pub aliases: Vec<String>,
}

/// Resolves raw names from owner tags to participants.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AliasTable {
    /// Known participants.
    pub participants: Vec<Participant>,
    /// Fuzzy threshold for names; defaults to [`FUZZY_THRESHOLD`] when absent.
    #[serde(default)]
    pub threshold: Option<f64>,
}

impl AliasTable {
    /// Table from participants with the default threshold.
    pub fn new(participants: Vec<Participant>) -> Self {
        Self {
            participants,
            threshold: None,
        }
    }

    fn names(p: &Participant) -> impl Iterator<Item = &str> {
        std::iter::once(p.display_name.as_str()).chain(p.aliases.iter().map(String::as_str))
    }

    /// Every normalized name in the table (for the validator's participant list).
    pub fn all_names(&self) -> Vec<String> {
        self.participants
            .iter()
            .flat_map(Self::names)
            .map(normalize)
            .collect()
    }

    /// The participant a raw name refers to: an exact normalized match first, then the
    /// unique best fuzzy match at or above the threshold. Ambiguous or unknown names
    /// resolve to `None`.
    pub fn resolve(&self, raw: &str) -> Option<&Participant> {
        let n = normalize(raw);
        if n.is_empty() {
            return None;
        }
        if let Some(p) = self
            .participants
            .iter()
            .find(|p| Self::names(p).any(|a| normalize(a) == n))
        {
            return Some(p);
        }
        let thr = self.threshold.unwrap_or(FUZZY_THRESHOLD);
        let mut best: Option<(&Participant, f64)> = None;
        let mut tied = false;
        for p in &self.participants {
            let score = Self::names(p)
                .map(|a| difflib::ratio(&normalize(a), &n))
                .fold(0.0, f64::max);
            if score < thr {
                continue;
            }
            match best {
                Some((_, s)) if score < s => {}
                Some((bp, s)) if score == s && bp.person_id != p.person_id => tied = true,
                _ => {
                    best = Some((p, score));
                    tied = false;
                }
            }
        }
        if tied {
            None
        } else {
            best.map(|(p, _)| p)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> AliasTable {
        AliasTable::new(vec![
            Participant {
                person_id: "p1".into(),
                display_name: "Avery".into(),
                aliases: vec!["Avery Stone".into()],
            },
            Participant {
                person_id: "p2".into(),
                display_name: "Jordan".into(),
                aliases: vec![],
            },
        ])
    }

    #[test]
    fn resolves_exact_fuzzy_and_rejects_unknown() {
        let t = table();
        assert_eq!(
            t.resolve(" avery ").map(|p| p.person_id.as_str()),
            Some("p1")
        );
        assert_eq!(
            t.resolve("AVERY STONE").map(|p| p.person_id.as_str()),
            Some("p1")
        );
        // ratio("jordan", "jordon") = 0.833 is below 0.85; "jordann" is 0.923.
        assert!(t.resolve("Jordon").is_none());
        assert_eq!(
            t.resolve("Jordann").map(|p| p.person_id.as_str()),
            Some("p2")
        );
        assert!(t.resolve("Morgan").is_none());
        assert!(t.resolve("").is_none());
    }

    #[test]
    fn labels_and_fragments() {
        assert_eq!(clean_label("\u{2192}"), "");
        assert_eq!(clean_label(" -> "), "");
        assert_eq!(clean_label("\u{2192} REST"), "REST");
        assert_eq!(clean_label("gRPC"), "gRPC");
        assert_eq!(clean_label("solid"), "");
        assert_eq!(clean_label(" Dashed "), "");
        assert_eq!(clean_label("..."), "");
        assert!(is_fragment_of("updates", "Shared updates"));
        assert!(is_fragment_of("Store", "Invoice Ledger Store"));
        assert!(is_fragment_of(
            "ledger_wr",
            "Ledger worker ledger_writer_py"
        ));
        assert!(!is_fragment_of("Queue", "Queue"));
        assert!(!is_fragment_of("ab", "abc def"));
        assert!(!is_fragment_of("Ledger", "Queue worker"));
        assert!(!is_fragment_of("Eta card", "Theta card"));
    }

    #[test]
    fn fuzzy_and_unreliable_text() {
        assert!(fuzzy_eq("Queue  Worker", "queue worker", 0.85));
        assert!(fuzzy_eq("/cache layer", "Cache Layer", 0.85));
        assert!(!fuzzy_eq("Cache", "Queue", 0.85));
        assert!(is_unreliable("Product Pl..."));
        assert!(is_unreliable("[illegible] service"));
        assert!(!is_unreliable("Widget Service"));
    }
}
