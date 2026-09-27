//! Notes metrics: decision, action-item, and open-question precision and recall
//! by text similarity (spec 9.3), plus negative action items (farewells) that
//! must not appear.
//!
//! A predicted item matches a gold item when its similarity to the gold text or any
//! alias is at least the threshold (default [`crate::text::SENTENCE_MATCH_DICE`],
//! 0.6). Words are compared as claim words ([`crate::text::claim_words`]): stemmed,
//! stopwords removed, capitalized participant spellings folded to one term, and
//! marked with their polarity (a negation governs the rest of its clause), so "ship
//! weekly builds" and "do not ship weekly builds" share no word. A prediction scores
//! against one gold phrasing only when
//!
//! - every key term of the phrasing (a capitalized participant name or an entity
//!   name at any position, a capitalized word after the first, an acronym, an
//!   identifier, or a number in digits or as a single number word ("thirty" is
//!   "30"; compounds such as "twenty five" are not folded);
//!   [`crate::text::key_terms`]) appears in it: a decision about another
//!   person or system is another decision, however similar the wording;
//! - it states the phrasing's claim words with the same polarity
//!   ([`crate::text::covers`]): all of them for a phrasing of up to five content
//!   words or one without key terms, all but one per five words otherwise, and in
//!   either case it may also drop one low-content word per three gold words (a light
//!   verb such as the "to use" of a purpose clause, a placeholder noun such as
//!   "step" or "side", a hedge such as "also"; [`crate::text::is_light`]) when it puts
//!   no word of its own in that word's place and the word is neither the head, the
//!   predicate, nor a key term; and
//! - it states none of them only with the opposite polarity, and no clause of it
//!   restates the claim retracted, anchored on the claim's head and predicate
//!   rather than its first word ([`crate::text::contradicts`]).
//!
//! The similarity is then the larger of the claim-word Dice coefficient
//! ([`crate::text::claim_dice`]), which rewards the same wording at the same length,
//! and containment ([`crate::text::containment`]), which accepts a restatement
//! inside a longer sentence ("The team decided to archive the draft on Friday
//! after review" for "archive the draft on Friday") when the prediction carries at
//! most four times as many content words as the gold (repeats counted) and keeps
//! the gold words together.
//! When the gold item names a person, the prediction must name the same person.
//! Matching is one-to-one and greedy by score.

use serde::{Deserialize, Serialize};

use super::{greedy_match, Counts};
use crate::text::{
    claim_dice, claim_words, containment, content_dice, covers, key_terms_present, token_dice,
    Vocabulary,
};

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

/// Similarity of a prediction to one gold phrasing (see the module docs). A gold
/// phrasing without any content word falls back to [`token_dice`].
pub fn phrasing_similarity(gold: &str, pred: &str, vocab: &Vocabulary) -> f64 {
    if claim_words(gold, vocab).is_empty() {
        return token_dice(gold, pred);
    }
    if !key_terms_present(gold, pred, vocab) || !covers(gold, pred, vocab) {
        return 0.0;
    }
    claim_dice(gold, pred, vocab).max(containment(gold, pred, vocab))
}

/// Best similarity of a prediction to a gold item (its text or any alias).
pub fn item_similarity(gold: &GoldItem, pred: &str, vocab: &Vocabulary) -> f64 {
    std::iter::once(gold.text.as_str())
        .chain(gold.aliases.iter().map(String::as_str))
        .map(|g| phrasing_similarity(g, pred, vocab))
        .fold(0.0, f64::max)
}

/// Matches items and returns counts plus `(gold, pred)` pairs. `vocab` holds the
/// golden set's participant and entity names ([`Vocabulary`]).
pub fn score_items(
    gold: &[GoldItem],
    pred: &[PredItem],
    threshold: f64,
    vocab: &Vocabulary,
) -> (Counts, Vec<(usize, usize)>) {
    let m = greedy_match(gold.len(), pred.len(), |g, p| {
        if let Some(person) = &gold[g].person_id {
            if pred[p].person_id.as_deref() != Some(person.as_str()) {
                return None;
            }
        }
        let s = item_similarity(&gold[g], &pred[p].text, vocab);
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

    fn none() -> Vocabulary {
        Vocabulary::default()
    }

    /// Fictional participants and one board entity.
    fn team() -> Vocabulary {
        let mut v = Vocabulary::default();
        v.add_person("tamsin", ["Tamsin Reed"]);
        v.add_person("quill", ["Quill Obi"]);
        v.add_entity("Ledger API");
        v
    }

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
        let (c, m) = score_items(&gold, &pred, 0.6, &none());
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
        assert_eq!(score_items(&gold, &wrong_person, 0.6, &none()).0.tp, 0);
        let no_person = vec![p("write the parser", None)];
        assert_eq!(score_items(&gold, &no_person, 0.6, &none()).0.tp, 0);
        let right = vec![p("write the parser", Some("p1"))];
        assert_eq!(score_items(&gold, &right, 0.6, &none()).0.tp, 1);
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
            ga("Seating can be random for the offsite", &[]),
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
                "For the offsite, the seating of guests can simply be random.",
                None,
            ),
        ];
        assert!(
            content_dice(&gold[0].text, &pred[0].text) < 0.6,
            "Dice alone misses this paraphrase"
        );
        let (c, m) = score_items(&gold, &pred, 0.6, &team());
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
        let (c, _) = score_items(&gold, &pred, 0.6, &team());
        assert_eq!(c.tp, 0, "{c:?}");
        assert_eq!(c.fp, 4);
    }

    /// Codex finding 8: the opposite decision is not the decision.
    #[test]
    fn negated_claims_do_not_match() {
        let gold = vec![
            ga("ship weekly builds", &[]),
            ga("Do not ship the importer", &[]),
        ];
        let pred = vec![
            p("do not ship weekly builds", None),
            p("We will ship the importer this week", None),
        ];
        let (c, m) = score_items(&gold, &pred, 0.6, &none());
        assert_eq!(c.tp, 0, "{m:?}");
        // the same polarity still matches, negation words vary
        let pred = vec![
            p("Ship weekly builds, not nightly ones", None),
            p("We decided we won't ship the importer.", None),
        ];
        assert_eq!(score_items(&gold, &pred, 0.6, &none()).0.tp, 2);
    }

    /// Kimi finding 1: a short gold is not stated by two of its three words.
    #[test]
    fn short_gold_partial_overlap_does_not_match() {
        let gold = vec![ga("Skip the Ledger step for now", &[])];
        let pred = vec![p("Ledger told us the step was hard", None)];
        assert_eq!(score_items(&gold, &pred, 0.6, &team()).0.tp, 0);
        // A prediction that drops the claim's verb from a five-word gold.
        let gold = vec![ga(
            "Tamsin moves from the importer to the dashboard work",
            &[],
        )];
        let pred = vec![p(
            "Tamsin complained about the importer and the dashboard work",
            None,
        )];
        assert_eq!(score_items(&gold, &pred, 0.6, &team()).0.tp, 0);
    }

    /// Kimi finding 2 and Codex finding 9: a name in first position is still a key term.
    #[test]
    fn sentence_initial_names_are_key_terms() {
        let gold = vec![ga(
            "Tamsin moves from the importer to the dashboard work",
            &["Tamsin works on the dashboard instead"],
        )];
        let other = vec![p(
            "Quill moved off the importer and onto the dashboard work",
            None,
        )];
        assert_eq!(score_items(&gold, &other, 0.6, &team()).0.tp, 0);
        let short = vec![ga("Tamsin moves to dashboard work", &[])];
        let other = vec![p("Quill moves to dashboard work", None)];
        assert_eq!(score_items(&short, &other, 0.6, &team()).0.tp, 0);
        let same = vec![p("Tamsin Reed moves to dashboard work", None)];
        assert_eq!(score_items(&short, &same, 0.6, &team()).0.tp, 1);
    }

    /// Kimi finding 5: a two-word gold inside a ten-times-longer ramble.
    #[test]
    fn two_word_golds_do_not_match_rambles() {
        let gold = vec![ga("ship builds", &[])];
        let pred = vec![p(
            "Ship builds, hiring plans, office seating, travel budgets, lunch vendors, parking passes, badge printers, laptop refresh cycles, conference talks, onboarding",
            None,
        )];
        assert_eq!(score_items(&gold, &pred, 0.6, &none()).0.tp, 0);
        let pred = vec![p("We ship builds after the review", None)];
        assert_eq!(score_items(&gold, &pred, 0.6, &none()).0.tp, 1);
        // Codex round-2 B1: repeating one distractor does not score through Dice.
        let pred = vec![p(&format!("ship builds {}", "lunch ".repeat(18)), None)];
        assert_eq!(score_items(&gold, &pred, 0.6, &none()).0.tp, 0);
    }

    /// Dropped low-content words at the item level: the drop matches, while another
    /// participant, a substituted word, a negation, and a ramble still do not.
    #[test]
    fn dropped_light_words_match_and_guards_hold() {
        let gold = vec![
            g("Post the notes and slides for the crew to use", Some("p1")),
            ga("Skip the Ledger step for now", &[]),
            ga("Tamsin posts the notes for the crew to use", &[]),
        ];
        let pred = vec![
            p("Post the notes and slides for the crew", Some("p1")),
            p("Skip Ledger for now", None),
            p("Tamsin posts the notes for the crew", None),
        ];
        assert_eq!(score_items(&gold, &pred, 0.6, &team()).0.tp, 3);
        let pred = vec![
            // the right words, the wrong owner
            p("Post the notes and slides for the crew", Some("p2")),
            // a substituted noun in the placeholder's slot
            p("Skip the Ledger rollout for now", None),
            // another participant
            p("Quill posts the notes for the crew", None),
            // the opposite claim
            p("Do not post the notes and slides for the crew", Some("p1")),
            // a ramble holding the words
            p(
                "We reviewed hiring plans, office seating, travel budgets, lunch vendors, parking passes, badge printers, laptop refresh cycles, conference talks, onboarding checklists, post the notes and slides for the crew, holiday calendars, desk moves, printer toner, coffee orders",
                Some("p1"),
            ),
        ];
        let (c, m) = score_items(&gold, &pred, 0.6, &team());
        assert_eq!(c.tp, 0, "{m:?}");
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
        assert!((item_similarity(&gold[0], "skip importer step", &none()) - 1.0).abs() < 1e-12);
        let hits = negative_hits(
            &["see you tomorrow".into()],
            &[
                p("I'll see you folks tomorrow", None),
                p("write tests", None),
            ],
            0.6,
        );
        // content words {see, tomorrow} vs {see, folks, tomorrow}: 4/5 = 0.8
        assert_eq!(hits, vec!["I'll see you folks tomorrow".to_string()]);
    }
}
