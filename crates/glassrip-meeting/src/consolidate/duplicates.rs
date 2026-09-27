//! Duplicate node tracks: a same-text reading away from an established node.
//!
//! Tracking keeps a same-text reading at another place of the canvas apart from the
//! element already there (two real boxes may share a label). The reader, though,
//! sometimes misplaces a box (a made-up grid, a collapsed strip, one box thrown
//! elsewhere) or reads one box twice; those readings open a new track that is the
//! established element read badly. Each sighting of the newer track is judged
//! against the established one ([`Sighting`]):
//!
//! - **Near**: its registered box is within the same-place tolerance of the
//!   established track's box.
//! - **Apart**: OCR reads the text at its own place (and not at the established
//!   track's), on a trusted registration that is not off around it and a reliable
//!   reading: a second element.
//! - **Explained**: OCR does not read the text at its own place, and its box is no
//!   evidence of a second place: the keyframe's reading is unreliable (the OCR
//!   geometry check found most boxes displaced, or the boxes coincide), the
//!   keyframe's registration is off around it (no other placed element read near it
//!   where its track is, while the keyframe places some other element away from
//!   where it is), or OCR reads the text at the established track's place.
//! - **Unknown**: none of these (no OCR of the text anywhere, text-only
//!   registration, or OCR at its own place where the reading or registration cannot
//!   be trusted).
//!
//! The newer track is merged into the established one when no sighting is apart
//! and at least half of its sightings are near or explained. A target that is
//! itself merged passes the duplicate on only to a final target that accepts it.
//! Two real boxes with one label stay two elements when OCR confirms the second
//! place on a reliable reading; a second box OCR never confirms and the reader
//! places consistently stays too (its sightings are unknown).

use std::collections::BTreeMap;

use crate::difflib;
use crate::text::normalize;

use super::tracks::{ObsList, Track};

/// How one sighting of a candidate duplicate relates to the established track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sighting {
    /// Within the same-place tolerance of the established track.
    Near,
    /// The reading's box is no evidence of another place.
    Explained,
    /// OCR confirms a second place on a reliable reading.
    Apart,
    /// No evidence either way.
    Unknown,
}

/// Same text: equal after normalization, or fuzzy ratio at least `fuzzy`.
pub(crate) fn same_text(a: &str, b: &str, fuzzy: f64) -> bool {
    let (a, b) = (normalize(a), normalize(b));
    !a.is_empty() && (a == b || difflib::ratio(&a, &b) >= fuzzy)
}

/// Decide merges. `judge(t, i, e)` classifies sighting `i` of track `t` against
/// established track `e`. Only node tracks take part; the target is a node track
/// seen in at least `min_support` keyframes and in more keyframes than the
/// candidate. Returns candidate track to target track (never chained).
pub(crate) fn duplicate_merges(
    tracks: &[Track],
    min_support: usize,
    fuzzy: f64,
    judge: &dyn Fn(usize, usize, usize) -> Sighting,
) -> BTreeMap<usize, usize> {
    let frames: Vec<usize> = tracks.iter().map(|t| t.frames().len()).collect();
    let is_node = |t: &Track| !t.obs.is_empty() && t.kind() == ObsList::Node;
    let texts: Vec<String> = tracks.iter().map(Track::text).collect();
    // Track `ti` may merge into track `ei`.
    let accepts = |ti: usize, ei: usize| -> bool {
        if ei == ti
            || !is_node(&tracks[ei])
            || frames[ei] < min_support.max(1)
            || frames[ei] <= frames[ti]
            || !same_text(&texts[ti], &texts[ei], fuzzy)
        {
            return false;
        }
        let verdicts: Vec<Sighting> = (0..tracks[ti].obs.len())
            .map(|i| judge(ti, i, ei))
            .collect();
        let supporting = verdicts
            .iter()
            .filter(|v| matches!(v, Sighting::Near | Sighting::Explained))
            .count();
        !verdicts.contains(&Sighting::Apart) && supporting > 0 && 2 * supporting >= verdicts.len()
    };
    let mut out: BTreeMap<usize, usize> = BTreeMap::new();
    for (ti, t) in tracks.iter().enumerate() {
        if !is_node(t) {
            continue;
        }
        let best = (0..tracks.len())
            .filter(|&ei| accepts(ti, ei))
            .max_by_key(|&ei| (frames[ei], std::cmp::Reverse(ei)));
        if let Some(ei) = best {
            out.insert(ti, ei);
        }
    }
    // A target that is itself merged passes its duplicates on, but only to a final
    // target the duplicate is itself accepted by; otherwise the duplicate stays apart
    // (and its own duplicates stop at it). Chains never loop: every step goes to a
    // track seen in strictly more keyframes.
    let resolve = |out: &BTreeMap<usize, usize>, t: usize| {
        let mut u = out[&t];
        while let Some(&v) = out.get(&u) {
            u = v;
        }
        u
    };
    while let Some(t) = out
        .keys()
        .copied()
        .find(|&t| resolve(&out, t) != out[&t] && !accepts(t, resolve(&out, t)))
    {
        out.remove(&t);
    }
    out.keys().map(|&t| (t, resolve(&out, t))).collect()
}

/// Move the sightings of every merged track into its target. A sighting that is not
/// near the target (`keep_box(t, i)` false) loses its boxes: its place is no
/// geometry of the target. A keyframe the target already has keeps the target's own
/// sighting. The merged tracks are left empty.
pub(crate) fn apply_merges(
    tracks: &mut [Track],
    merges: &BTreeMap<usize, usize>,
    keep_box: &dyn Fn(usize, usize) -> bool,
) {
    let mut moved: Vec<(usize, usize, usize)> = Vec::new();
    for (&t, &e) in merges {
        for i in 0..tracks[t].obs.len() {
            moved.push((t, i, e));
        }
    }
    let boxes: Vec<bool> = moved.iter().map(|&(t, i, _)| keep_box(t, i)).collect();
    let mut incoming: BTreeMap<usize, Vec<super::tracks::Obs>> = BTreeMap::new();
    for (&(t, i, e), keep) in moved.iter().zip(boxes) {
        let mut o = tracks[t].obs[i].clone();
        if !keep {
            o.bbox = None;
            o.raw_bbox = None;
        }
        incoming.entry(e).or_default().push(o);
    }
    for &t in merges.keys() {
        tracks[t].obs.clear();
    }
    for (e, obs) in incoming {
        for o in obs {
            if tracks[e].obs.iter().any(|x| x.frame == o.frame) {
                continue;
            }
            tracks[e].obs.push(o);
        }
        tracks[e].obs.sort_by_key(|o| o.frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidate::tracks::Obs;

    fn track(text: &str, frames: &[usize]) -> Track {
        Track {
            obs: frames
                .iter()
                .map(|&f| Obs {
                    frame: f,
                    lists: vec![ObsList::Node],
                    text: text.into(),
                    bbox: None,
                    raw_bbox: None,
                    cluster: 0,
                    local_id: None,
                    color: None,
                    single_ok: false,
                    weight: 1.0,
                })
                .collect(),
            kind_override: None,
        }
    }

    #[test]
    fn merges_need_no_apart_sighting_and_a_supporting_half() {
        let tracks = vec![
            track("Ledger Store", &[0, 1, 2, 3, 4]),
            track("Ledger Store", &[5, 6]),
        ];
        let run = |v: [Sighting; 2]| {
            duplicate_merges(&tracks, 2, 0.85, &|t, i, e| {
                assert_eq!((t, e), (1, 0));
                v[i]
            })
        };
        assert_eq!(
            run([Sighting::Explained, Sighting::Unknown]).get(&1),
            Some(&0)
        );
        assert!(run([Sighting::Near, Sighting::Apart]).is_empty());
        assert!(run([Sighting::Unknown, Sighting::Unknown]).is_empty());
    }

    #[test]
    fn only_node_tracks_with_the_same_text_and_fewer_sightings_merge() {
        let mut sticky = track("Ledger Store", &[5]);
        sticky.obs[0].lists = vec![ObsList::Sticky];
        let tracks = vec![
            track("Ledger Store", &[0, 1, 2]),
            track("Report Builder", &[3]),
            sticky,
            track("Ledger Store", &[6, 7, 8]),
        ];
        let m = duplicate_merges(&tracks, 2, 0.85, &|_, _, _| Sighting::Near);
        assert!(m.is_empty(), "{m:?}");
    }

    #[test]
    fn a_chain_ends_at_a_target_the_duplicate_is_accepted_by() {
        // Track 2 is near track 1, which merges into track 0; track 2 is apart from
        // track 0, so it stays apart instead of following the chain.
        let tracks = vec![
            track("Ledger Store", &[0, 1, 2, 3, 4]),
            track("Ledger Store", &[5, 6]),
            track("Ledger Store", &[7]),
        ];
        let judge = |t: usize, _: usize, e: usize| match (t, e) {
            (1, 0) | (2, 1) => Sighting::Near,
            _ => Sighting::Apart,
        };
        let m = duplicate_merges(&tracks, 2, 0.85, &judge);
        assert_eq!(m, BTreeMap::from([(1, 0)]));
        let judge = |t: usize, _: usize, e: usize| match (t, e) {
            (1, 0) | (2, 1) | (2, 0) => Sighting::Near,
            _ => Sighting::Apart,
        };
        let m = duplicate_merges(&tracks, 2, 0.85, &judge);
        assert_eq!(m, BTreeMap::from([(1, 0), (2, 0)]));
    }

    #[test]
    fn merged_sightings_join_the_target_once_per_keyframe() {
        let mut tracks = vec![track("Queue", &[0, 1, 3]), track("Queue", &[1, 2])];
        let merges = BTreeMap::from([(1, 0)]);
        apply_merges(&mut tracks, &merges, &|_, _| false);
        assert!(tracks[1].obs.is_empty());
        assert_eq!(tracks[0].frames(), vec![0, 1, 2, 3]);
        assert_eq!(tracks[0].obs.len(), 4);
    }
}
