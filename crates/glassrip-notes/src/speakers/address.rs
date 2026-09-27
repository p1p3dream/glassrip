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
    /// The answer is a short greeting or acknowledgment echoing a greeting
    /// ("Hey, Name." then "Hey."), which is strong evidence it is Name's.
    pub echo: bool,
}

/// Greetings that echo a greeting. Generic acknowledgments ("yes", "yeah",
/// "thanks") answer anyone and are not echo evidence.
const ECHO_WORDS: &[&str] = &["hey", "hi", "hello", "yo", "morning", "hiya", "howdy"];

/// Words that introduce reported speech: a name after them is quoted, not said
/// to someone in the room ("it says, hey Avery, your build is red").
const REPORTING: &[&str] = &[
    "said", "says", "say", "saying", "told", "tells", "asked", "asks", "wrote", "writes", "reads",
    "read", "quote", "quoting", "goes",
];

fn quote_mark(w: &str) -> bool {
    w.contains(['"', '\u{201c}', '\u{201d}'])
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
                // reported speech: a reporting word or an opening quote mark earlier
                // in the sentence
                let reported = sentence[..k]
                    .iter()
                    .any(|p| quote_mark(p) || REPORTING.contains(&letters(p).as_str()))
                    || quote_mark(&w.w);
                if reported {
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
                let echo = kind == AddressKind::Greeting
                    && response_segment
                        .and_then(|r| segments.get(r))
                        .is_some_and(|r| {
                            r.words.len() <= 3
                                && r.words
                                    .first()
                                    .is_some_and(|w| ECHO_WORDS.contains(&letters(&w.w).as_str()))
                        });
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
                    echo,
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
                &["Um,", "Avery,", "any", "thoughts", "on", "this?"],
            ),
            seg("s3", "L0", 40.0, &["what", "about", "you", "Rowan?"]),
            seg("s4", "L0", 60.0, &["We", "told", "Avery", "already."]),
        ];
        let ev = find_addresses(&segs, &table, 4.0, 0.6);
        let got: Vec<(usize, usize, AddressKind, Option<usize>)> = ev
            .iter()
            .map(|e| (e.segment, e.person, e.kind, e.response_segment))
            .collect();
        assert!(ev[0].echo, "\"Hey.\" answering \"Hey, Rohan.\" is an echo");
        assert!(!ev[1].echo);
        assert_eq!(
            got,
            vec![
                (0, 1, AddressKind::Greeting, Some(1)),
                (2, 0, AddressKind::Opening, None),
                (3, 1, AddressKind::Closing, None),
            ]
        );
    }

    #[test]
    fn reported_speech_and_generic_acks_are_not_address_evidence() {
        let table = AliasTable::from_names(&["Avery Quinn", "Rohan Dasgupta"]);
        let segs = vec![
            // read aloud: the name is quoted, not said to Avery
            seg(
                "s0",
                "L0",
                0.0,
                &[
                    "The", "bot", "says,", "hey", "Avery,", "your", "build", "is", "red.",
                ],
            ),
            seg(
                "s1",
                "L0",
                5.0,
                &["He", "said", "\"hi", "Rohan\"", "earlier."],
            ),
            // a greeting answered by a generic acknowledgment
            seg("s2", "L0", 10.0, &["Bye,", "Rohan."]),
            seg("s3", "L1", 11.2, &["Yeah,", "thanks."]),
        ];
        let ev = find_addresses(&segs, &table, 4.0, 0.6);
        assert_eq!(ev.len(), 1, "{ev:?}");
        assert_eq!((ev[0].segment, ev[0].kind), (2, AddressKind::Greeting));
        assert!(!ev[0].echo, "\"Yeah, thanks.\" answers anyone");
    }
}
