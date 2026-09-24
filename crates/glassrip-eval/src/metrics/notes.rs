//! Notes metrics: decision, action-item, and open-question precision and recall
//! by text similarity (spec 9.3), plus negative action items (farewells) that
//! must not appear.
//!
//! A predicted item matches a gold item when the token Dice coefficient
//! ([`crate::text::token_dice`]) against the gold text or any alias is at least
//! the threshold (default [`crate::text::SENTENCE_MATCH_DICE`], 0.5). When the
//! gold item names a person, the prediction must name the same person.
//! Matching is one-to-one and greedy by score.

use serde::{Deserialize, Serialize};

use super::{greedy_match, Counts};
use crate::text::token_dice;

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

/// Best similarity of a prediction to a gold item.
pub fn item_similarity(gold: &GoldItem, pred: &str) -> f64 {
    std::iter::once(gold.text.as_str())
        .chain(gold.aliases.iter().map(String::as_str))
        .map(|g| token_dice(g, pred))
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
                .any(|n| token_dice(n, &p.text) >= threshold)
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
            // Dice({defer, the, importer}, {defer, importer, now}) = 4/6 = 0.667
            p("defer importer now", None),
            // unrelated
            p("lunch at noon", None),
        ];
        let (c, m) = score_items(&gold, &pred, 0.5);
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
        assert_eq!(score_items(&gold, &wrong_person, 0.5).0.tp, 0);
        let no_person = vec![p("write the parser", None)];
        assert_eq!(score_items(&gold, &no_person, 0.5).0.tp, 0);
        let right = vec![p("write the parser", Some("p1"))];
        assert_eq!(score_items(&gold, &right, 0.5).0.tp, 1);
    }

    #[test]
    fn aliases_and_negatives() {
        let gold = [GoldItem {
            text: "defer the importer for the prototype".into(),
            aliases: vec!["skip importer".into()],
            person_id: None,
            t_s: None,
        }];
        // Dice with alias {skip, importer} vs {skip, importer, step}: 4/5 = 0.8
        assert!((item_similarity(&gold[0], "skip importer step") - 0.8).abs() < 1e-12);
        let hits = negative_hits(
            &["see you later".into()],
            &[p("I'll see you guys later", None), p("write tests", None)],
            0.5,
        );
        // Dice({see, you, later}, {i, ll, see, you, guys, later}) = 6/9 = 0.667
        assert_eq!(hits, vec!["I'll see you guys later".to_string()]);
    }
}
