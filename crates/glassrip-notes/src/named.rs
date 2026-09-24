//! Transcript lines with resolved speaker names.

use glassrip_audio::recluster::Source;
use glassrip_audio::types::TranscriptSegment;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::speakers::{SpeakerSource, SpeakersDoc};

/// One transcript line (a segment, or part of one when a word range has its own speaker).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NamedLine {
    /// Transcript segment id (shared by the parts of a split segment).
    pub segment_id: String,
    /// Start, seconds.
    pub start_s: f64,
    /// End, seconds.
    pub end_s: f64,
    /// Speaker, when resolved.
    pub person_id: Option<String>,
    /// Display name, or the diarization label when unresolved.
    pub speaker: String,
    /// Corrected text.
    pub text: String,
    /// Verbatim ASR text.
    pub text_raw: String,
    /// Confidence of the speaker in [0, 1].
    pub speaker_confidence: f32,
    /// Speaker differs from the label's mapping.
    pub relabeled: bool,
    /// Share of words labeled by gap filling.
    pub gap_fill_share: f32,
}

/// Builds named lines from segments (sorted by time) and the speakers artifact.
pub fn named_lines(segments: &[TranscriptSegment], doc: &SpeakersDoc) -> Vec<NamedLine> {
    let name = |pid: Option<&str>, label: &str| -> String {
        pid.and_then(|p| doc.display_name(p))
            .map(str::to_string)
            .unwrap_or_else(|| format!("{label} (unresolved)"))
    };
    let mut out = Vec::new();
    for s in segments {
        let decided = doc.segments.get(&s.segment_id);
        let seg_pid = decided.and_then(|d| d.person_id.clone()).or_else(|| {
            doc.labels
                .iter()
                .find(|l| l.label == s.speaker_label)
                .and_then(|l| l.person_id.clone())
        });
        let seg_conf = decided.map(|d| d.confidence).unwrap_or(0.0);
        let seg_relabeled = decided.is_some_and(|d| d.source != SpeakerSource::LabelMap);
        // word ranges and their speakers, in order
        let mut ranges: Vec<(usize, usize, Option<String>, f32, bool)> = Vec::new();
        let mut spans: Vec<_> = decided.map(|d| d.spans.clone()).unwrap_or_default();
        spans.sort_by_key(|sp| sp.word_start);
        let mut i = 0;
        for sp in spans
            .iter()
            .filter(|sp| sp.word_end <= s.words.len() && sp.word_start < sp.word_end)
        {
            if sp.word_start < i {
                continue;
            }
            if sp.word_start > i {
                ranges.push((i, sp.word_start, seg_pid.clone(), seg_conf, seg_relabeled));
            }
            ranges.push((
                sp.word_start,
                sp.word_end,
                sp.person_id.clone(),
                sp.confidence,
                true,
            ));
            i = sp.word_end;
        }
        if i < s.words.len() || s.words.is_empty() {
            ranges.push((i, s.words.len(), seg_pid.clone(), seg_conf, seg_relabeled));
        }
        // merge neighbouring ranges with the same speaker
        let mut merged: Vec<(usize, usize, Option<String>, f32, bool)> = Vec::new();
        for r in ranges {
            match merged.last_mut() {
                Some(last) if last.2 == r.2 => {
                    last.1 = r.1;
                    last.3 = last.3.min(r.3);
                    last.4 |= r.4;
                }
                _ => merged.push(r),
            }
        }
        let whole = merged.len() == 1;
        for (a, b, pid, conf, relabeled) in merged {
            let words = &s.words[a..b.min(s.words.len())];
            let (text, text_raw, start, end) = if whole {
                (s.text.clone(), s.text_raw.clone(), s.start_s, s.end_s)
            } else {
                (
                    words
                        .iter()
                        .map(|w| w.w.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    words
                        .iter()
                        .map(|w| w.w_raw.as_deref().unwrap_or(&w.w))
                        .collect::<Vec<_>>()
                        .join(" "),
                    words.first().map_or(s.start_s, |w| w.start_s),
                    words.last().map_or(s.end_s, |w| w.end_s),
                )
            };
            let gap = words.iter().filter(|w| w.source == Source::GapFill).count();
            out.push(NamedLine {
                segment_id: s.segment_id.clone(),
                start_s: start,
                end_s: end,
                speaker: name(pid.as_deref(), &s.speaker_label),
                person_id: pid,
                text,
                text_raw,
                speaker_confidence: conf,
                relabeled,
                gap_fill_share: if words.is_empty() {
                    0.0
                } else {
                    gap as f32 / words.len() as f32
                },
            });
        }
    }
    out
}

/// Consecutive lines by one speaker, merged for reading.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Paragraph {
    /// Start, seconds.
    pub start_s: f64,
    /// End, seconds.
    pub end_s: f64,
    /// Speaker display name.
    pub speaker: String,
    /// Speaker id.
    pub person_id: Option<String>,
    /// Text.
    pub text: String,
    /// Segment ids in the paragraph.
    pub segment_ids: Vec<String>,
    /// Some line in the paragraph has a low-confidence or relabeled speaker.
    pub flagged: bool,
}

/// Merges lines by the same speaker separated by at most `max_gap_s`.
pub fn paragraphs(lines: &[NamedLine], max_gap_s: f64, low_conf: f32) -> Vec<Paragraph> {
    let mut out: Vec<Paragraph> = Vec::new();
    for l in lines {
        let flagged = l.relabeled || l.speaker_confidence < low_conf;
        match out.last_mut() {
            Some(p) if p.speaker == l.speaker && l.start_s - p.end_s <= max_gap_s => {
                p.text.push(' ');
                p.text.push_str(l.text.trim());
                p.end_s = p.end_s.max(l.end_s);
                if p.segment_ids.last() != Some(&l.segment_id) {
                    p.segment_ids.push(l.segment_id.clone());
                }
                p.flagged |= flagged;
            }
            _ => out.push(Paragraph {
                start_s: l.start_s,
                end_s: l.end_s,
                speaker: l.speaker.clone(),
                person_id: l.person_id.clone(),
                text: l.text.trim().to_string(),
                segment_ids: vec![l.segment_id.clone()],
                flagged,
            }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::people::AliasTable;
    use crate::speakers::vote::test_support::seg;
    use crate::speakers::{SegmentSpeaker, WordSpan};

    #[test]
    fn spans_split_lines_and_paragraphs_merge() {
        let table = AliasTable::from_names(&["Avery Quinn", "Rohan Dasgupta"]);
        let segs = vec![
            seg(
                "s0",
                "L0",
                0.0,
                &["We", "ship", "Friday.", "I", "can", "test."],
            ),
            seg("s1", "L0", 2.6, &["Great."]),
        ];
        let mut doc = SpeakersDoc {
            people: table.people().to_vec(),
            ..Default::default()
        };
        let mk = |id: &str, spans: Vec<WordSpan>| SegmentSpeaker {
            segment_id: id.into(),
            start_s: 0.0,
            end_s: 0.0,
            label: "L0".into(),
            person_id: Some("avery-quinn".into()),
            confidence: 0.9,
            source: SpeakerSource::LabelMap,
            reason: None,
            scores: Default::default(),
            observations: vec![],
            spans,
        };
        doc.segments.insert(
            "s0".into(),
            mk(
                "s0",
                vec![WordSpan {
                    word_start: 3,
                    word_end: 6,
                    person_id: Some("rohan-dasgupta".into()),
                    confidence: 0.7,
                    source: SpeakerSource::VisualRelabel,
                    reason: "lit".into(),
                }],
            ),
        );
        doc.segments.insert("s1".into(), mk("s1", vec![]));
        let lines = named_lines(&segs, &doc);
        let got: Vec<(&str, &str)> = lines
            .iter()
            .map(|l| (l.speaker.as_str(), l.text.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("Avery Quinn", "We ship Friday."),
                ("Rohan Dasgupta", "I can test."),
                ("Avery Quinn", "Great.")
            ]
        );
        assert!(lines[1].relabeled);
        let paras = paragraphs(&lines, 2.0, 0.5);
        assert_eq!(paras.len(), 3);
    }
}
