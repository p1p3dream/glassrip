//! Notes metrics: decision, action-item, and open-question precision and recall
//! by text similarity (spec 9.3), plus negative action items (farewells) that
//! must not appear.
//!
//! A predicted item matches a gold item when its similarity to the gold text or any
//! alias is at least the threshold (default [`crate::text::SENTENCE_MATCH_DICE`],
//! 0.6). The similarity to one gold phrasing is the larger of:
//!
//! - the content-word Dice coefficient ([`crate::text::content_dice`], stopwords
//!   removed), which rewards the same wording at the same length, and
//! - token containment ([`crate::text::containment`]): the share of the gold
//!   phrasing's stemmed content words that the prediction states, so a paraphrase
//!   that restates the claim inside a longer sentence ("The team decided to skip
//!   the importer step for now to focus on ...") matches a short gold phrasing
//!   ("skip the importer for now"). A prediction that is mostly unrelated content
//!   (under [`crate::text::CONTAINMENT_MIN_PRECISION`] of its words from the gold)
//!   does not count.
//!
//! Either way, every key term of the gold phrasing (a proper noun, acronym, or
//! identifier; [`crate::text::key_terms`]) must appear in the prediction: a decision
//! about another person or system is another decision, however similar the wording.
//! When the gold item names a person, the prediction must name the same person.
//! Matching is one-to-one and greedy by score.

use serde::{Deserialize, Serialize};

use super::{greedy_match, Counts};
use crate::text::{containment, content_dice, key_terms_present};

/// A gold notes item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldItem {
    /// Canonical text.
    pub text: String,
    /// Accepted paraphrases (short phrasings a correct item would use).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// Owner (action items), a gold person id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub person_id: Option<String>,
    /// Approximate time, seconds (informational).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_s: Option<f64>,
}

/// A predicted notes item, person already resolved to a gold person id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredItem {
    /// Text.
    pub text: String,
    /// Resolved person id, if any.
    #[serde(default)]
    pub person_id: Option<String>,
}

/// Similarity of a prediction to one gold phrasing (see the module docs).
pub fn phrasing_similarity(gold: &str, pred: &str) -> f64 {
    if !key_terms_present(gold, pred) {
        return 0.0;
    }
    content_dice(gold, pred).max(containment(gold, pred))
}

/// Best similarity of a prediction to a gold item (its text or any alias).
pub fn item_similarity(gold: &GoldItem, pred: &str) -> f64 {
    std::iter::once(gold.text.as_str())
        .chain(gold.aliases.iter().map(String::as_str))
        .map(|g| phrasing_similarity(g, pred))
        .fold(0.0, f64::max)
}

/// Matches items and returns counts plus `(gold, pred)` pairs.
pub fn score_items(
    gold: &[GoldItem],
    pred: &[PredItem],
    threshold: f64,
) -> (Counts, Vec<(usize, usize)>) {
    let m = greedy_match(gold.len(), pred.len(), |g, p| {
        if let Some(person) = &gold[g].person_id {
            if pred[p].person_id.as_deref() != Some(person.as_str()) {
                return None;
            }
        }
        let s = item_similarity(&gold[g], &pred[p].text);
        (s >= threshold).then_some(s)
    });
    (Counts::from_matches(m.len(), gold.len(), pred.len()), m)
}

/// Predictions matching any negative text at the threshold.
pub fn negative_hits(negatives: &[String], pred: &[PredItem], threshold: f64) -> Vec<String> {
    pred.iter()
        .filter(|p| {
            negatives
                .iter()
                .any(|n| content_dice(n, &p.text) >= threshold)
        })
        .map(|p| p.text.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(text: &str, person: Option<&str>) -> GoldItem {
        GoldItem {
            text: text.into(),
            aliases: vec![],
            person_id: person.map(String::from),
            t_s: None,
        }
    }

    fn p(text: &str, person: Option<&str>) -> PredItem {
        PredItem {
            text: text.into(),
            person_id: person.map(String::from),
        }
    }

    #[test]
    fn decisions_hand_computed() {
        let gold = vec![g("defer the importer", None), g("ship weekly builds", None)];
        let pred = vec![
            // content words {defer, importer} on both sides: 1.0
            p("defer importer now", None),
            // unrelated
            p("lunch at noon", None),
        ];
        let (c, m) = score_items(&gold, &pred, 0.6);
        assert_eq!(
            c,
            Counts {
                tp: 1,
                fp: 1,
                fn_: 1
            }
        );
        assert_eq!(m, vec![(0, 0)]);
    }

    #[test]
    fn action_items_need_person() {
        let gold = vec![g("write the parser", Some("p1"))];
        let wrong_person = vec![p("write the parser", Some("p2"))];
        assert_eq!(score_items(&gold, &wrong_person, 0.6).0.tp, 0);
        let no_person = vec![p("write the parser", None)];
        assert_eq!(score_items(&gold, &no_person, 0.6).0.tp, 0);
        let right = vec![p("write the parser", Some("p1"))];
        assert_eq!(score_items(&gold, &right, 0.6).0.tp, 1);
    }

    fn ga(text: &str, aliases: &[&str]) -> GoldItem {
        GoldItem {
            text: text.into(),
            aliases: aliases.iter().map(|a| a.to_string()).collect(),
            person_id: None,
            t_s: None,
        }
    }

    #[test]
    fn paraphrases_inside_longer_sentences_match() {
        let gold = vec![
            ga("Skip the Ledger step for now", &["Ledger is deferred"]),
            ga(
                "Tamsin moves from the importer to the dashboard work",
                &["Tamsin works on the dashboard instead"],
            ),
            ga("Ordering can be random for the demo", &[]),
        ];
        let pred = vec![
            // Dice against the gold text is 0.5; containment is 1.0.
            p(
                "The team decided to skip the Ledger integration step for now to focus on the API shape.",
                None,
            ),
            // Inflections differ (moves / moved); every content word is covered.
            p(
                "Tamsin moved off the importer and onto the dashboard work.",
                None,
            ),
            p(
                "For the demo, the ordering of cards can simply be random.",
                None,
            ),
        ];
        assert!(
            content_dice(&gold[0].text, &pred[0].text) < 0.6,
            "Dice alone misses this paraphrase"
        );
        let (c, m) = score_items(&gold, &pred, 0.6);
        assert_eq!(
            c,
            Counts {
                tp: 3,
                fp: 0,
                fn_: 0
            },
            "{m:?}"
        );
    }

    #[test]
    fn different_claims_do_not_match() {
        let gold = vec![
            ga("Skip the Ledger step for now", &[]),
            ga("Move Tamsin to the dashboard work", &[]),
            ga("Ship weekly builds", &[]),
        ];
        let pred = vec![
            // Same topic words, other system: the key term Ledger is missing.
            p("Skip the Orbit step for now", None),
            // Dice is 0.75, but another person is named.
            p("Move Quill to the dashboard work", None),
            // One shared word out of three.
            p("Builds are slow this week", None),
            // Contains every gold word, but nearly all of it is unrelated content.
            p(
                "We reviewed hiring plans, office seating, travel budgets, lunch vendors, parking passes, badge printers, laptop refresh cycles, conference talks, onboarding checklists, ship weekly builds, holiday calendars, desk moves, printer toner, coffee orders, standup times, team photos, and retrospectives notes archives",
                None,
            ),
        ];
        let (c, _) = score_items(&gold, &pred, 0.6);
        assert_eq!(c.tp, 0, "{c:?}");
        assert_eq!(c.fp, 4);
    }

    #[test]
    fn aliases_and_negatives() {
        let gold = [GoldItem {
            text: "defer the importer for the prototype".into(),
            aliases: vec!["skip importer".into()],
            person_id: None,
            t_s: None,
        }];
        // Content Dice with alias {skip, importer} vs {skip, importer, step} is
        // 4/5 = 0.8; the prediction states the whole alias, so containment is 1.0.
        assert!((content_dice("skip importer", "skip importer step") - 0.8).abs() < 1e-12);
        assert!((item_similarity(&gold[0], "skip importer step") - 1.0).abs() < 1e-12);
        let hits = negative_hits(
            &["see you later".into()],
            &[p("I'll see you guys later", None), p("write tests", None)],
            0.6,
        );
        // content words {see, later} vs {see, guys, later}: 4/5 = 0.8
        assert_eq!(hits, vec!["I'll see you guys later".to_string()]);
    }
}
