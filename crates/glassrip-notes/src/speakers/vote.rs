//! The evidence vote: label mapping, per-segment relabeling, gap-fill re-check.
//!
//! Pure functions over the transcript, the alias table, direct-address events
//! and frame observations, so the whole decision is testable without video.

use std::collections::{BTreeMap, BTreeSet};

use glassrip_audio::recluster::Source;
use glassrip_audio::types::TranscriptSegment;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::address::AddressEvent;
use super::{
    CueKind, FrameObservation, GapFillCheck, LabelEvidence, LabelItem, LabelStatus, SegmentSpeaker,
    SpeakerSource, WordSpan,
};
use crate::people::{AliasTable, Person};

/// Vote weights and thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VoteParams {
    /// Seconds after a segment start for the early sample.
    pub early_offset_s: f64,
    /// The early sample never goes later than this past the segment end.
    pub early_max_past_end_s: f64,
    /// Weight of a highlighted tile (times the fraction of samples showing it).
    pub w_highlight: f64,
    /// Weight of the absent-tile inference.
    pub w_absent: f64,
    /// Weight of answering a direct address.
    pub w_response: f64,
    /// Negative weight of addressing someone by name.
    pub w_addressed_not_speaker: f64,
    /// Weight of the presenter cue on the label with the most talk time.
    pub w_role: f64,
    /// Weight of the diarizer's own label for a segment (times its confidence).
    pub w_diarizer: f64,
    /// Minimum score for a relabel.
    pub relabel_min: f64,
    /// Margin a relabel needs over the diarizer's choice.
    pub relabel_margin: f64,
    /// Longest gap between an address and the answering segment, seconds.
    pub response_window_s: f64,
    /// Minimum name-match score for a direct address.
    pub min_name_score: f64,
    /// Gap-fill runs at least this long get their own sample, seconds.
    pub gap_run_min_s: f64,
    /// Unmapped labels with at most this much talk time are noise, seconds.
    pub noise_max_talk_s: f64,
    /// Minimum label confidence for status `mapped`.
    pub min_mapped_confidence: f64,
    /// Turns shorter than this get a sample before they start (ring hold check), seconds.
    pub short_turn_s: f64,
    /// How long before a short turn the hold sample is taken, seconds.
    pub pre_offset_s: f64,
    /// Weight of a short greeting answering a greeting by name.
    pub w_response_echo: f64,
}

impl Default for VoteParams {
    fn default() -> Self {
        Self {
            early_offset_s: 0.5,
            early_max_past_end_s: 0.3,
            w_highlight: 1.0,
            w_absent: 0.4,
            w_response: 0.6,
            w_addressed_not_speaker: 0.8,
            w_role: 2.0,
            w_diarizer: 0.8,
            relabel_min: 0.7,
            relabel_margin: 0.25,
            response_window_s: 4.0,
            min_name_score: 0.6,
            gap_run_min_s: 0.8,
            noise_max_talk_s: 2.0,
            min_mapped_confidence: 0.2,
            short_turn_s: 1.5,
            pre_offset_s: 0.4,
            w_response_echo: 1.2,
        }
    }
}

/// Time key on a 0.1 s grid (frames closer than that are shared).
pub fn time_key(t_s: f64) -> i64 {
    (t_s * 10.0).round() as i64
}

/// A run of consecutive gap-filled words.
#[derive(Debug, Clone, PartialEq)]
pub struct GapRun {
    /// Segment index.
    pub segment: usize,
    /// First word (inclusive).
    pub word_start: usize,
    /// Last word (exclusive).
    pub word_end: usize,
    /// Start, seconds.
    pub start_s: f64,
    /// End, seconds.
    pub end_s: f64,
    /// Own sample times (empty: use the segment's samples).
    pub times: Vec<f64>,
    /// True when the run is the whole segment.
    pub whole_segment: bool,
}

/// Frame times to decode.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplePlan {
    /// Sample times per segment.
    pub segment_times: Vec<Vec<f64>>,
    /// Hold-check sample before each short segment.
    pub pre_times: Vec<Option<f64>>,
    /// Gap-fill runs.
    pub runs: Vec<GapRun>,
}

impl SamplePlan {
    /// Every distinct sample time, sorted.
    pub fn all_times(&self) -> Vec<f64> {
        let mut m: BTreeMap<i64, f64> = BTreeMap::new();
        for t in self
            .segment_times
            .iter()
            .flatten()
            .chain(self.runs.iter().flat_map(|r| r.times.iter()))
            .chain(self.pre_times.iter().flatten())
        {
            m.entry(time_key(*t)).or_insert(*t);
        }
        m.into_values().collect()
    }
}

/// Plans frame samples: each segment's midpoint and early point, and the
/// midpoint of each long gap-fill run.
pub fn plan_samples(segments: &[TranscriptSegment], p: &VoteParams) -> SamplePlan {
    let mut plan = SamplePlan::default();
    for (si, s) in segments.iter().enumerate() {
        let mid = (s.start_s + s.end_s) / 2.0;
        let early = (s.start_s + p.early_offset_s).min(s.end_s + p.early_max_past_end_s);
        let mut times = vec![mid];
        if time_key(early) != time_key(mid) {
            times.push(early);
        }
        plan.segment_times.push(times);
        // Conferencing apps light the ring with a delay and keep it lit for a
        // moment after speech stops, so a short turn right after someone else's
        // can show the previous speaker's ring. A sample just before the turn
        // tells a held ring from a new one.
        plan.pre_times.push(
            (s.end_s - s.start_s < p.short_turn_s).then(|| (s.start_s - p.pre_offset_s).max(0.0)),
        );

        let mut i = 0;
        while i < s.words.len() {
            if s.words[i].source != Source::GapFill {
                i += 1;
                continue;
            }
            let start = i;
            while i < s.words.len() && s.words[i].source == Source::GapFill {
                i += 1;
            }
            let (a, b) = (s.words[start].start_s, s.words[i - 1].end_s);
            let whole = start == 0 && i == s.words.len();
            let times = if !whole && b - a >= p.gap_run_min_s {
                vec![(a + b) / 2.0]
            } else {
                Vec::new()
            };
            plan.runs.push(GapRun {
                segment: si,
                word_start: start,
                word_end: i,
                start_s: a,
                end_s: b,
                times,
                whole_segment: whole,
            });
        }
    }
    plan
}

/// A visual verdict over a set of samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    /// Participant.
    pub person_id: String,
    /// Fraction of usable samples supporting it.
    pub strength: f64,
    /// Highlight or absent tile.
    pub kind: CueKind,
}

/// Combines samples into one verdict (None when there is no usable or a tied cue).
pub fn visual_verdict(obs: &[&FrameObservation], people: &[Person]) -> Option<Verdict> {
    let usable: Vec<&&FrameObservation> = obs
        .iter()
        .filter(|o| o.error.is_none() && !o.tiles.is_empty())
        .collect();
    if usable.is_empty() {
        return None;
    }
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for o in &usable {
        for p in o.highlighted() {
            *counts.entry(p).or_default() += 1;
        }
    }
    if !counts.is_empty() {
        let max = counts.values().copied().max().unwrap_or(0);
        let top: Vec<&str> = counts
            .iter()
            .filter(|(_, c)| **c == max)
            .map(|(p, _)| *p)
            .collect();
        return match top.as_slice() {
            [one] => Some(Verdict {
                person_id: (*one).to_string(),
                strength: max as f64 / usable.len() as f64,
                kind: CueKind::ActiveSpeakerHighlight,
            }),
            _ => None,
        };
    }
    if usable.len() != obs.len() {
        return None;
    }
    let visible: BTreeSet<&str> = usable.iter().flat_map(|o| o.visible()).collect();
    let missing: Vec<&Person> = people
        .iter()
        .filter(|p| !visible.contains(p.person_id.as_str()))
        .collect();
    match missing.as_slice() {
        [one] if !visible.is_empty() => Some(Verdict {
            person_id: one.person_id.clone(),
            strength: 1.0,
            kind: CueKind::AbsentTile,
        }),
        _ => None,
    }
}

/// Everything the vote produces.
#[derive(Debug, Clone, PartialEq)]
pub struct VoteOutput {
    /// Labels.
    pub labels: Vec<LabelItem>,
    /// Segments.
    pub segments: Vec<SegmentSpeaker>,
    /// Gap-fill re-check.
    pub gap_fill: GapFillCheck,
    /// Presenter (majority over frames).
    pub presenter: Option<String>,
}

fn word_secs(s: &TranscriptSegment, src: Source) -> f64 {
    s.words
        .iter()
        .filter(|w| w.source == src)
        .map(|w| (w.end_s - w.start_s).max(0.0))
        .sum()
}

fn kind_weight(k: CueKind, p: &VoteParams) -> f64 {
    match k {
        CueKind::ActiveSpeakerHighlight => p.w_highlight,
        CueKind::AbsentTile => p.w_absent,
        CueKind::AddressResponse => p.w_response,
        CueKind::AddressedNotSpeaker => -p.w_addressed_not_speaker,
        CueKind::RoleCue => p.w_role,
    }
}

fn describe(o: &[&FrameObservation]) -> String {
    o.iter()
        .map(|x| {
            let lit = x.highlighted();
            if lit.is_empty() {
                format!("{:.1}s: visible {}", x.t_s, x.visible().join("+"))
            } else {
                format!("{:.1}s: lit {}", x.t_s, lit.join("+"))
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Runs the vote.
pub fn vote(
    segments: &[TranscriptSegment],
    table: &AliasTable,
    addresses: &[AddressEvent],
    plan: &SamplePlan,
    frames: &BTreeMap<i64, FrameObservation>,
    p: &VoteParams,
) -> VoteOutput {
    let people = table.people();
    let obs_for = |times: &[f64]| -> Vec<&FrameObservation> {
        times
            .iter()
            .filter_map(|t| frames.get(&time_key(*t)))
            .collect()
    };
    let seg_obs: Vec<Vec<&FrameObservation>> =
        plan.segment_times.iter().map(|t| obs_for(t)).collect();
    let pre_obs: Vec<Option<&FrameObservation>> = (0..segments.len())
        .map(|si| {
            plan.pre_times
                .get(si)
                .copied()
                .flatten()
                .and_then(|t| frames.get(&time_key(t)))
        })
        .collect();
    let seg_verdict: Vec<Option<Verdict>> = seg_obs
        .iter()
        .zip(&pre_obs)
        .map(|(o, pre)| {
            let v = visual_verdict(o, people)?;
            let held = v.kind == CueKind::ActiveSpeakerHighlight
                && pre.is_some_and(|pre| pre.highlighted().contains(&v.person_id.as_str()));
            // a ring already lit before a short turn belongs to the previous turn
            (!held).then_some(v)
        })
        .collect();
    let response_weight = |a: &AddressEvent| {
        if a.echo {
            p.w_response_echo
        } else {
            p.w_response
        }
    };

    // label votes
    let mut labels: BTreeSet<&str> = BTreeSet::new();
    let mut talk: BTreeMap<&str, (f64, f64, f64)> = BTreeMap::new(); // total, gap, diarizer
    for s in segments {
        labels.insert(s.speaker_label.as_str());
        let e = talk.entry(s.speaker_label.as_str()).or_default();
        let (d, g) = (
            word_secs(s, Source::Diarizer),
            word_secs(s, Source::GapFill),
        );
        e.0 += d + g;
        e.1 += g;
        e.2 += d;
    }
    let mut votes: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    let mut evidence: BTreeMap<String, Vec<LabelEvidence>> = BTreeMap::new();
    let mut add =
        |label: &str, ev: LabelEvidence, votes: &mut BTreeMap<String, BTreeMap<String, f64>>| {
            if !labels.contains(label) {
                return;
            }
            *votes
                .entry(label.to_string())
                .or_default()
                .entry(ev.person_id.clone())
                .or_default() += ev.weight;
            evidence.entry(label.to_string()).or_default().push(ev);
        };
    for (si, s) in segments.iter().enumerate() {
        let Some(v) = &seg_verdict[si] else {
            continue;
        };
        let dur = word_secs(s, Source::Diarizer) + 0.5 * word_secs(s, Source::GapFill);
        if dur <= 0.0 {
            continue;
        }
        let seg_w = 0.3 + 0.7 * dur.min(8.0) / 8.0;
        add(
            &s.speaker_label,
            LabelEvidence {
                t_s: (s.start_s + s.end_s) / 2.0,
                kind: v.kind,
                person_id: v.person_id.clone(),
                weight: kind_weight(v.kind, p) * v.strength * seg_w,
                segment_id: Some(s.segment_id.clone()),
                text: describe(&seg_obs[si]),
            },
            &mut votes,
        );
    }
    for a in addresses {
        let (Some(seg), Some(person)) = (segments.get(a.segment), people.get(a.person)) else {
            continue;
        };
        add(
            &seg.speaker_label,
            LabelEvidence {
                t_s: a.t_s,
                kind: CueKind::AddressedNotSpeaker,
                person_id: person.person_id.clone(),
                weight: kind_weight(CueKind::AddressedNotSpeaker, p),
                segment_id: Some(seg.segment_id.clone()),
                text: a.sentence.clone(),
            },
            &mut votes,
        );
        if let Some(r) = a.response_segment.and_then(|r| segments.get(r)) {
            if r.speaker_label != seg.speaker_label {
                add(
                    &r.speaker_label,
                    LabelEvidence {
                        t_s: r.start_s,
                        kind: CueKind::AddressResponse,
                        person_id: person.person_id.clone(),
                        weight: response_weight(a),
                        segment_id: Some(r.segment_id.clone()),
                        text: format!("answers \"{}\" with \"{}\"", a.sentence, r.text),
                    },
                    &mut votes,
                );
            }
        }
    }
    // presenter cue
    let mut presenter_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for o in frames.values() {
        if let Some(pr) = &o.presenter {
            *presenter_counts.entry(pr.as_str()).or_default() += 1;
        }
    }
    let presenter = presenter_counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
        .map(|(p, _)| (*p).to_string());
    if let Some(pr) = &presenter {
        let top = talk
            .iter()
            .max_by(|a, b| a.1 .2.total_cmp(&b.1 .2).then(b.0.cmp(a.0)))
            .map(|(l, t)| (*l, t.2));
        if let Some((label, secs)) = top {
            add(
                label,
                LabelEvidence {
                    t_s: 0.0,
                    kind: CueKind::RoleCue,
                    person_id: pr.clone(),
                    weight: p.w_role,
                    segment_id: None,
                    text: format!(
                        "presenter banner in {} frames; label has the most diarized talk time ({secs:.0} s)",
                        presenter_counts.get(pr.as_str()).copied().unwrap_or(0)
                    ),
                },
                &mut votes,
            );
        }
    }

    // assignment: greedy by weight, one participant per label, then allow a
    // second label on a participant when its own evidence is one-sided
    let mut cands: Vec<(&str, &String, f64)> = votes
        .iter()
        .flat_map(|(l, m)| m.iter().map(move |(pid, v)| (l.as_str(), pid, *v)))
        .filter(|c| c.2 > 0.0)
        .collect();
    cands.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(b.0)).then(a.1.cmp(b.1)));
    let mut mapped: BTreeMap<&str, String> = BTreeMap::new();
    let mut taken: BTreeSet<&String> = BTreeSet::new();
    for (l, pid, _) in &cands {
        if !mapped.contains_key(l) && !taken.contains(pid) {
            mapped.insert(l, (*pid).clone());
            taken.insert(pid);
        }
    }
    let positive_sum = |l: &str| -> f64 {
        votes
            .get(l)
            .map(|m| m.values().filter(|v| **v > 0.0).sum())
            .unwrap_or(0.0)
    };
    // a second label on one participant only when the diarizer produced more
    // labels than there are participants (a split cluster); otherwise a tile that
    // looks lit for other reasons could pull two labels onto one person
    if labels.len() > people.len() {
        for (l, pid, v) in &cands {
            if !mapped.contains_key(l) && *v >= 1.0 && *v / positive_sum(l) >= 0.7 {
                mapped.insert(l, (*pid).clone());
            }
        }
    }
    let confidence = |l: &str, pid: &str| -> f64 {
        let v = votes
            .get(l)
            .and_then(|m| m.get(pid))
            .copied()
            .unwrap_or(0.0);
        let total = positive_sum(l);
        if v <= 0.0 || total <= 0.0 {
            return 0.0;
        }
        // share of the positive evidence, times a saturating amount term
        (v / total) * (1.0 - (-v / 0.75).exp())
    };
    let mut label_items = Vec::new();
    let mut label_person: BTreeMap<&str, (String, f64)> = BTreeMap::new();
    for l in &labels {
        let (total, gap, _) = talk.get(l).copied().unwrap_or_default();
        let (status, person, conf) = match mapped.get(l) {
            Some(pid) if confidence(l, pid) >= p.min_mapped_confidence => {
                (LabelStatus::Mapped, Some(pid.clone()), confidence(l, pid))
            }
            _ if total <= p.noise_max_talk_s => (LabelStatus::Noise, None, 0.0),
            _ => (LabelStatus::Unresolved, None, 0.0),
        };
        if let Some(pid) = &person {
            label_person.insert(l, (pid.clone(), conf));
        }
        let mut ev = evidence.remove(*l).unwrap_or_default();
        let evidence_total = ev.len();
        ev.sort_by(|a, b| {
            b.weight
                .abs()
                .total_cmp(&a.weight.abs())
                .then(a.t_s.total_cmp(&b.t_s))
        });
        ev.truncate(40);
        ev.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
        label_items.push(LabelItem {
            label: (*l).to_string(),
            status,
            person_id: person,
            confidence: conf.clamp(0.0, 1.0) as f32,
            votes: votes.get(*l).cloned().unwrap_or_default(),
            evidence: ev,
            evidence_total,
            talk_time_s: total,
            talk_time_gap_fill_s: gap,
        });
    }

    // per-segment decisions
    let mut responses: BTreeMap<usize, Vec<&AddressEvent>> = BTreeMap::new();
    let mut addressed: BTreeMap<usize, Vec<&AddressEvent>> = BTreeMap::new();
    for a in addresses {
        addressed.entry(a.segment).or_default().push(a);
        if let Some(r) = a.response_segment {
            responses.entry(r).or_default().push(a);
        }
    }
    let mut out_segments = Vec::with_capacity(segments.len());
    for (si, s) in segments.iter().enumerate() {
        let base = label_person.get(s.speaker_label.as_str()).cloned();
        let mut scores: BTreeMap<String, f64> = BTreeMap::new();
        if let Some((pid, lconf)) = &base {
            *scores.entry(pid.clone()).or_default() +=
                p.w_diarizer * (0.5 * f64::from(s.speaker_conf) + 0.5 * lconf);
        }
        let verdict = &seg_verdict[si];
        if let Some(v) = verdict {
            *scores.entry(v.person_id.clone()).or_default() += kind_weight(v.kind, p) * v.strength;
        }
        for a in responses.get(&si).into_iter().flatten() {
            if let Some(person) = people.get(a.person) {
                *scores.entry(person.person_id.clone()).or_default() += response_weight(a);
            }
        }
        for a in addressed.get(&si).into_iter().flatten() {
            if let Some(person) = people.get(a.person) {
                *scores.entry(person.person_id.clone()).or_default() -= p.w_addressed_not_speaker;
            }
        }
        let best = scores
            .iter()
            .filter(|(pid, _)| base.as_ref().is_none_or(|(b, _)| b != *pid))
            .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(a.0)))
            .map(|(pid, v)| (pid.clone(), *v));
        let base_score = base
            .as_ref()
            .map(|(b, _)| scores.get(b).copied().unwrap_or(0.0));
        let relabel = match (&best, base_score) {
            (Some((_, v)), Some(bs)) => *v >= p.relabel_min && *v >= bs + p.relabel_margin,
            (Some((_, v)), None) => *v >= p.relabel_min,
            (None, _) => false,
        };
        let (person, source, reason) = if relabel {
            let (pid, _) = best.clone().unwrap_or_default();
            let visual = verdict
                .as_ref()
                .is_some_and(|v| v.person_id == pid && v.kind == CueKind::ActiveSpeakerHighlight);
            let mut why = Vec::new();
            if let Some(v) = verdict.as_ref().filter(|v| v.person_id == pid) {
                why.push(match v.kind {
                    CueKind::ActiveSpeakerHighlight => {
                        format!("tile lit in {:.0}% of samples", v.strength * 100.0)
                    }
                    _ => "no visible tile lit; only participant without a visible tile".to_string(),
                });
            }
            if responses.get(&si).is_some_and(|r| {
                r.iter()
                    .any(|a| people.get(a.person).is_some_and(|x| x.person_id == pid))
            }) {
                why.push("answers a direct address".into());
            }
            let src = if visual {
                SpeakerSource::VisualRelabel
            } else {
                SpeakerSource::EvidenceRelabel
            };
            let reason = match &base {
                Some((b, _)) => format!("diarizer label maps to {b}; {}", why.join("; ")),
                None => why.join("; "),
            };
            (Some(pid), src, Some(reason))
        } else if let Some((b, _)) = &base {
            (Some(b.clone()), SpeakerSource::LabelMap, None)
        } else {
            (None, SpeakerSource::Unresolved, None)
        };
        let positive: f64 = scores.values().filter(|v| **v > 0.0).sum();
        let chosen = person
            .as_ref()
            .and_then(|pid| scores.get(pid))
            .copied()
            .unwrap_or(0.0);
        let conf = if positive > 0.0 && chosen > 0.0 {
            (chosen / positive) * chosen.min(1.0)
        } else {
            0.0
        };
        out_segments.push(SegmentSpeaker {
            segment_id: s.segment_id.clone(),
            start_s: s.start_s,
            end_s: s.end_s,
            label: s.speaker_label.clone(),
            person_id: person,
            confidence: conf.clamp(0.0, 1.0) as f32,
            source,
            reason,
            scores,
            observations: pre_obs[si]
                .into_iter()
                .chain(seg_obs[si].iter().copied())
                .cloned()
                .collect(),
            spans: Vec::new(),
        });
    }

    // gap-fill re-check
    let mut gf = GapFillCheck {
        words_total: segments
            .iter()
            .map(|s| {
                s.words
                    .iter()
                    .filter(|w| w.source == Source::GapFill)
                    .count()
            })
            .sum(),
        runs_total: plan.runs.len(),
        ..GapFillCheck::default()
    };
    for run in &plan.runs {
        let (Some(s), Some(decided)) = (segments.get(run.segment), out_segments.get(run.segment))
        else {
            continue;
        };
        let n = run.word_end - run.word_start;
        let obs: Vec<&FrameObservation> = if run.whole_segment {
            seg_obs[run.segment].clone()
        } else if !run.times.is_empty() {
            obs_for(&run.times)
        } else {
            seg_obs[run.segment]
                .iter()
                .copied()
                .filter(|o| o.t_s >= run.start_s - 0.5 && o.t_s <= run.end_s + 0.5)
                .collect()
        };
        let Some(v) = visual_verdict(&obs, people) else {
            gf.words_no_cue += n;
            continue;
        };
        gf.runs_with_cue += 1;
        let label_pid = label_person
            .get(s.speaker_label.as_str())
            .map(|x| x.0.clone());
        let reference = if run.whole_segment {
            label_pid.clone()
        } else {
            decided.person_id.clone()
        };
        if reference.as_deref() == Some(v.person_id.as_str()) {
            gf.runs_agree += 1;
            gf.words_agree += n;
            continue;
        }
        gf.runs_contradict += 1;
        gf.words_contradict += n;
        if run.whole_segment {
            if decided.person_id.as_deref() == Some(v.person_id.as_str())
                && decided.source != SpeakerSource::LabelMap
            {
                gf.runs_relabeled += 1;
                gf.words_relabeled += n;
            }
            continue;
        }
        let words = &s.words[run.word_start..run.word_end];
        let mean_conf =
            words.iter().map(|w| f64::from(w.assign_conf)).sum::<f64>() / n.max(1) as f64;
        let base = p.w_diarizer * mean_conf;
        let score = kind_weight(v.kind, p) * v.strength;
        if v.kind == CueKind::ActiveSpeakerHighlight
            && score >= p.relabel_min
            && score >= base + p.relabel_margin
        {
            gf.runs_relabeled += 1;
            gf.words_relabeled += n;
            if let Some(d) = out_segments.get_mut(run.segment) {
                d.spans.push(WordSpan {
                    word_start: run.word_start,
                    word_end: run.word_end,
                    person_id: Some(v.person_id.clone()),
                    confidence: (score / (score + base)).clamp(0.0, 1.0) as f32,
                    source: SpeakerSource::VisualRelabel,
                    reason: format!(
                        "gap-filled words; tile of {} lit in {:.0}% of samples",
                        v.person_id,
                        v.strength * 100.0
                    ),
                });
            }
        }
    }

    VoteOutput {
        labels: label_items,
        segments: out_segments,
        gap_fill: gf,
        presenter,
    }
}

/// Builders for synthetic transcripts (tests only).
#[cfg(test)]
pub mod test_support {
    use glassrip_audio::recluster::Source;
    use glassrip_audio::types::{TranscriptSegment, TranscriptWord};

    /// A segment of 0.4 s words starting at `start`.
    pub fn seg(id: &str, label: &str, start: f64, words: &[&str]) -> TranscriptSegment {
        seg_src(id, label, start, words, Source::Diarizer)
    }

    /// Like [`seg`] with every word from `src`.
    pub fn seg_src(
        id: &str,
        label: &str,
        start: f64,
        words: &[&str],
        src: Source,
    ) -> TranscriptSegment {
        let ws: Vec<TranscriptWord> = words
            .iter()
            .enumerate()
            .map(|(i, w)| TranscriptWord {
                w: (*w).to_string(),
                w_raw: None,
                start_s: start + 0.4 * i as f64,
                end_s: start + 0.4 * (i + 1) as f64,
                p: 0.9,
                speaker_label: label.to_string(),
                assign_conf: if src == Source::GapFill { 0.4 } else { 0.9 },
                source: src,
                gap_sim: None,
                gap_margin: None,
            })
            .collect();
        let text = words.join(" ");
        TranscriptSegment {
            segment_id: id.to_string(),
            start_s: start,
            end_s: start + 0.4 * words.len() as f64,
            speaker_label: label.to_string(),
            speaker_conf: 0.9,
            text: text.clone(),
            text_raw: text,
            gap_fill_words: if src == Source::GapFill {
                words.len()
            } else {
                0
            },
            unassigned_words: 0,
            words: ws,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{seg, seg_src};
    use super::*;
    use crate::speakers::address::find_addresses;
    use crate::speakers::TileSeen;

    fn tile(pid: &str, lit: bool) -> TileSeen {
        TileSeen {
            person_id: pid.into(),
            text: pid.into(),
            name_bbox: [0, 0, 10, 10],
            ring_score: if lit { 0.9 } else { 0.0 },
            highlighted: lit,
        }
    }

    fn obs(t: f64, tiles: Vec<TileSeen>, presenter: Option<&str>) -> FrameObservation {
        FrameObservation {
            t_s: t,
            tiles,
            presenter: presenter.map(str::to_string),
            error: None,
        }
    }

    /// Observations for every planned time, driven by who is lit at time `t`.
    fn frames_for(
        plan: &SamplePlan,
        lit_at: impl Fn(f64) -> Option<&'static str>,
    ) -> BTreeMap<i64, FrameObservation> {
        plan.all_times()
            .into_iter()
            .map(|t| {
                // the local participant has no visible tile
                let lit = lit_at(t);
                let tiles = vec![
                    tile("avery-quinn", lit == Some("avery-quinn")),
                    tile("rohan-dasgupta", lit == Some("rohan-dasgupta")),
                ];
                (time_key(t), obs(t, tiles, Some("avery-quinn")))
            })
            .collect()
    }

    #[test]
    fn maps_labels_and_fixes_a_short_reply_hidden_in_the_presenter_label() {
        let table = AliasTable::from_names(&["Avery Quinn", "Rohan Dasgupta", "Mira Okafor"]);
        let segs = vec![
            seg(
                "s0",
                "L0",
                0.0,
                &[
                    "Welcome", "everyone", "to", "the", "kiosk", "sync", "today", "folks.",
                ],
            ),
            // Avery's greeting, put in Rohan's label by the diarizer
            seg(
                "s1",
                "L1",
                3.6,
                &["Hey,", "Mira,", "glad", "you", "could", "join", "us."],
            ),
            // Mira's reply, put in the presenter's label
            seg("s2", "L0", 6.8, &["Hey."]),
            seg(
                "s3",
                "L0",
                8.0,
                &[
                    "So", "the", "relay", "pulls", "from", "the", "ledger", "service", "nightly.",
                ],
            ),
            seg(
                "s4",
                "L1",
                13.0,
                &[
                    "I", "can", "take", "the", "badge", "printer", "work", "this", "week.",
                ],
            ),
            seg(
                "s5",
                "L2",
                18.0,
                &["I", "will", "handle", "the", "ledger", "side", "then."],
            ),
        ];
        let p = VoteParams::default();
        let plan = plan_samples(&segs, &p);
        // Avery lit during s0, s1 and s3; Rohan during s4; nobody lit during s2
        // and s5 (Mira has no visible tile)
        let frames = frames_for(&plan, |t| {
            if (0.0..=6.5).contains(&t) || (8.0..=11.6).contains(&t) {
                Some("avery-quinn")
            } else if (13.0..=16.6).contains(&t) {
                Some("rohan-dasgupta")
            } else {
                None
            }
        });
        let addresses = find_addresses(&segs, &table, p.response_window_s, p.min_name_score);
        let out = vote(&segs, &table, &addresses, &plan, &frames, &p);
        let map: Vec<(&str, Option<&str>, LabelStatus)> = out
            .labels
            .iter()
            .map(|l| (l.label.as_str(), l.person_id.as_deref(), l.status))
            .collect();
        assert_eq!(
            map,
            vec![
                ("L0", Some("avery-quinn"), LabelStatus::Mapped),
                ("L1", Some("rohan-dasgupta"), LabelStatus::Mapped),
                ("L2", Some("mira-okafor"), LabelStatus::Mapped),
            ]
        );
        let who: Vec<(&str, Option<&str>, SpeakerSource)> = out
            .segments
            .iter()
            .map(|s| (s.segment_id.as_str(), s.person_id.as_deref(), s.source))
            .collect();
        assert_eq!(
            who[1],
            ("s1", Some("avery-quinn"), SpeakerSource::VisualRelabel)
        );
        assert_eq!(
            who[2],
            ("s2", Some("mira-okafor"), SpeakerSource::EvidenceRelabel)
        );
        assert_eq!(who[3].2, SpeakerSource::LabelMap);
        assert_eq!(out.presenter.as_deref(), Some("avery-quinn"));
        for s in &out.segments {
            assert!((0.0..=1.0).contains(&s.confidence));
        }
    }

    #[test]
    fn a_held_ring_does_not_claim_a_short_greeting_reply() {
        let table = AliasTable::from_names(&["Avery Quinn", "Rohan Dasgupta", "Mira Okafor"]);
        let segs = vec![
            seg(
                "s0",
                "L0",
                0.0,
                &["Welcome", "to", "the", "relay", "sync", "everyone", "today"],
            ),
            seg("s1", "L0", 3.0, &["Hey,", "Mira."]),
            // Mira's reply, put in the presenter's label by the diarizer
            seg("s2", "L0", 4.2, &["Hey."]),
            seg(
                "s3",
                "L0",
                6.0,
                &[
                    "So", "the", "relay", "pulls", "from", "the", "ledger", "nightly",
                ],
            ),
            seg(
                "s4",
                "L1",
                12.0,
                &["I", "will", "check", "the", "printer", "queue", "then"],
            ),
        ];
        let p = VoteParams::default();
        let plan = plan_samples(&segs, &p);
        assert!(plan.pre_times[2].is_some() && plan.pre_times[0].is_none());
        // Avery's ring stays lit through Mira's reply (hold), then Rohan speaks
        let frames = frames_for(&plan, |t| {
            if t < 10.0 {
                Some("avery-quinn")
            } else {
                Some("rohan-dasgupta")
            }
        });
        let addresses = find_addresses(&segs, &table, p.response_window_s, p.min_name_score);
        assert!(addresses[0].echo);
        let out = vote(&segs, &table, &addresses, &plan, &frames, &p);
        let reply = &out.segments[2];
        assert_eq!(reply.person_id.as_deref(), Some("mira-okafor"), "{reply:?}");
        assert_eq!(reply.source, SpeakerSource::EvidenceRelabel);
        assert_eq!(
            reply.observations.len(),
            3,
            "hold sample plus two turn samples"
        );
        // without the hold rule the lit ring would have kept it
        let no_hold = VoteParams {
            short_turn_s: 0.0,
            ..VoteParams::default()
        };
        let plan2 = plan_samples(&segs, &no_hold);
        let out2 = vote(&segs, &table, &addresses, &plan2, &frames, &no_hold);
        assert_eq!(out2.segments[2].person_id.as_deref(), Some("avery-quinn"));
    }

    #[test]
    fn gap_fill_runs_are_rechecked_and_relabeled_on_a_lit_tile() {
        let table = AliasTable::from_names(&["Avery Quinn", "Rohan Dasgupta", "Mira Okafor"]);
        let mut s0 = seg(
            "s0",
            "L0",
            0.0,
            &["The", "ledger", "sync", "runs", "at", "night"],
        );
        let gap = seg_src(
            "x",
            "L0",
            2.4,
            &["and", "I", "will", "own", "the", "printer"],
            Source::GapFill,
        );
        s0.words.extend(gap.words);
        s0.end_s = 4.8;
        let segs = vec![
            s0,
            seg("s1", "L1", 6.0, &["Sounds", "good", "to", "me", "then"]),
        ];
        let p = VoteParams::default();
        let plan = plan_samples(&segs, &p);
        assert_eq!(plan.runs.len(), 1);
        assert_eq!(plan.runs[0].times.len(), 1);
        let frames = frames_for(&plan, |t| {
            if t < 2.4 {
                Some("avery-quinn")
            } else {
                Some("rohan-dasgupta")
            }
        });
        let out = vote(&segs, &table, &[], &plan, &frames, &p);
        assert_eq!(out.gap_fill.words_total, 6);
        assert_eq!(out.gap_fill.runs_contradict, 1);
        assert_eq!(out.gap_fill.words_relabeled, 6);
        let spans = &out.segments[0].spans;
        assert_eq!(spans.len(), 1);
        assert_eq!((spans[0].word_start, spans[0].word_end), (6, 12));
        assert_eq!(spans[0].person_id.as_deref(), Some("rohan-dasgupta"));
    }

    #[test]
    fn no_video_leaves_labels_to_address_and_role_cues() {
        let table = AliasTable::from_names(&["Avery Quinn", "Rohan Dasgupta"]);
        let segs = vec![
            seg(
                "s0",
                "L0",
                0.0,
                &["Rohan,", "can", "you", "check", "the", "relay?"],
            ),
            seg("s1", "L1", 3.0, &["Yes,", "I", "will", "check", "it."]),
            seg("x2", "L2", 9.0, &["Bye."]),
        ];
        let p = VoteParams::default();
        let plan = plan_samples(&segs, &p);
        let addresses = find_addresses(&segs, &table, p.response_window_s, p.min_name_score);
        let out = vote(&segs, &table, &addresses, &plan, &BTreeMap::new(), &p);
        assert_eq!(out.labels[1].person_id.as_deref(), Some("rohan-dasgupta"));
        assert_eq!(out.labels[0].status, LabelStatus::Unresolved);
        assert_eq!(out.labels[2].status, LabelStatus::Noise);
        assert!(out.labels[0].votes["rohan-dasgupta"] < 0.0);
    }
}
