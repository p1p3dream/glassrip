//! Direct address in the transcript ("Hey, Name." / "Name, you want to ...?").
//!
//! A name is vocative when it follows a greeting, opens a sentence (after
//! optional fillers) followed by a comma, or closes a question after a comma or
//! after "you". The person addressed is not the speaker; the next segment, when it
//! starts soon after the addressing sentence ends, is a likely answer by them.

use glassrip_audio::types::TranscriptSegment;
use glassrip_audio::vocab::letters;

use crate::people::AliasTable;

/// How a name was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressKind {
    /// After a greeting or thanks.
    Greeting,
    /// Opening a sentence, followed by a comma.
    Opening,
    /// Closing a question.
    Closing,
}

/// One direct address.
#[derive(Debug, Clone, PartialEq)]
pub struct AddressEvent {
    /// Index of the addressing segment.
    pub segment: usize,
    /// Index of the name word in the segment.
    pub word: usize,
    /// End of the name word, seconds.
    pub t_s: f64,
    /// End of the addressing sentence, seconds.
    pub sentence_end_s: f64,
    /// Addressed participant (index into the alias table).
    pub person: usize,
    /// Match strength of the name.
    pub name_score: f64,
    /// How the name was used.
    pub kind: AddressKind,
    /// The addressing sentence.
    pub sentence: String,
    /// Segment that likely answers (the next segment, when it starts in time).
    pub response_segment: Option<usize>,
}

const GREETINGS: &[&str] = &[
    "hey", "hi", "hello", "thanks", "welcome", "bye", "morning", "sorry",
];
const FILLERS: &[&str] = &[
    "um", "uh", "so", "okay", "ok", "and", "but", "well", "oh", "yeah", "alright", "right", "hey",
];

fn ends_sentence(w: &str) -> bool {
    let t = w.trim_end_matches(['"', '\'', ')']);
    t.ends_with('.') || t.ends_with('?') || t.ends_with('!')
}

/// Finds direct addresses. `segments` must be sorted by start time.
pub fn find_addresses(
    segments: &[TranscriptSegment],
    table: &AliasTable,
    response_window_s: f64,
    min_name_score: f64,
) -> Vec<AddressEvent> {
    let mut out = Vec::new();
    for (si, seg) in segments.iter().enumerate() {
        let words = &seg.words;
        let mut start = 0;
        while start < words.len() {
            let mut end = start;
            while end < words.len() && !ends_sentence(&words[end].w) {
                end += 1;
            }
            let last = end.min(words.len().saturating_sub(1));
            let sentence: Vec<&str> = words[start..=last].iter().map(|w| w.w.as_str()).collect();
            let is_last_sentence = last + 1 >= words.len();
            for (k, w) in words[start..=last].iter().enumerate() {
                let Some(m) = table.match_word(&w.w) else {
                    continue;
                };
                if m.score < min_name_score {
                    continue;
                }
                let prev = k.checked_sub(1).map(|p| letters(sentence[p]));
                let prev_raw = k.checked_sub(1).map(|p| sentence[p]);
                let prev2 = k.checked_sub(2).map(|p| letters(sentence[p]));
                let only_fillers_before = sentence[..k]
                    .iter()
                    .all(|p| FILLERS.contains(&letters(p).as_str()));
                let kind = if prev.as_deref().is_some_and(|p| GREETINGS.contains(&p))
                    || (prev.as_deref() == Some("you") && prev2.as_deref() == Some("thank"))
                {
                    Some(AddressKind::Greeting)
                } else if only_fillers_before && w.w.trim_end().ends_with(',') {
                    Some(AddressKind::Opening)
                } else if k + 1 == sentence.len()
                    && w.w.trim_end().ends_with('?')
                    && (prev_raw.is_some_and(|p| p.trim_end().ends_with(','))
                        || prev.as_deref() == Some("you"))
                {
                    Some(AddressKind::Closing)
                } else {
                    None
                };
                let Some(kind) = kind else {
                    continue;
                };
                let sentence_end_s = words[last].end_s;
                let response_segment = if is_last_sentence {
                    segments
                        .get(si + 1)
                        .filter(|n| n.start_s - sentence_end_s <= response_window_s)
                        .map(|_| si + 1)
                } else {
                    None
                };
                out.push(AddressEvent {
                    segment: si,
                    word: start + k,
                    t_s: w.end_s,
                    sentence_end_s,
                    person: m.person,
                    name_score: m.score,
                    kind,
                    sentence: sentence.join(" "),
                    response_segment,
                });
            }
            start = last + 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speakers::vote::test_support::seg;

    #[test]
    fn greeting_opening_and_closing_forms() {
        let table = AliasTable::from_names(&["Avery Quinn", "Rohan Dasgupta"]);
        let segs = vec![
            seg(
                "s0",
                "L0",
                10.0,
                &["Oh,", "is", "that", "Rohan?", "Hey,", "Rohan."],
            ),
            seg("s1", "L1", 13.2, &["Hey."]),
            seg(
                "s2",
                "L1",
                20.0,
                &["Um,", "Avery,", "you", "want", "to", "chime", "in?"],
            ),
            seg("s3", "L0", 40.0, &["what", "about", "you", "Rowan?"]),
            seg("s4", "L0", 60.0, &["We", "told", "Avery", "already."]),
        ];
        let ev = find_addresses(&segs, &table, 4.0, 0.6);
        let got: Vec<(usize, usize, AddressKind, Option<usize>)> = ev
            .iter()
            .map(|e| (e.segment, e.person, e.kind, e.response_segment))
            .collect();
        assert_eq!(
            got,
            vec![
                (0, 1, AddressKind::Greeting, Some(1)),
                (2, 0, AddressKind::Opening, None),
                (3, 1, AddressKind::Closing, None),
            ]
        );
    }
}
