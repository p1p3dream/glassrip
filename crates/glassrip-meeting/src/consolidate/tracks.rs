//! Element tracks across keyframes: matching, list votes, lifetimes, support.

use std::collections::BTreeMap;

use glassrip_vision::board::StickyColor;
use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::difflib;
use crate::text::normalize;

/// Which output list an observation came from.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ObsList {
    /// A node box.
    Node,
    /// A sticky note or card.
    Sticky,
    /// Other canvas text.
    Other,
    /// The label written on an edge.
    EdgeLabel,
}

/// One sighting of a text element in one keyframe.
#[derive(Debug, Clone, PartialEq)]
pub struct Obs {
    /// Board keyframe index (time order).
    pub frame: usize,
    /// Lists this text was read in within the keyframe (usually one).
    pub lists: Vec<ObsList>,
    /// Text as read.
    pub text: String,
    /// Box in the cluster reference frame, when the keyframe is registered.
    pub bbox: Option<BBox>,
    /// Box in the keyframe's own canvas coordinates (none for edge labels).
    pub raw_bbox: Option<BBox>,
    /// Registration cluster (only meaningful with `bbox`).
    pub cluster: usize,
    /// Node `local_id` in the keyframe, for node observations.
    pub local_id: Option<String>,
    /// Sticky color, for sticky observations.
    pub color: Option<StickyColor>,
    /// This sighting alone may support the element (pixel check and reading
    /// confidence agree, plus the second reader pass when one is configured).
    pub single_ok: bool,
}

/// Votes over lists across all sightings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListVotes {
    /// Read as a node.
    pub node: u32,
    /// Read as a sticky.
    pub sticky: u32,
    /// Read as other text.
    pub other: u32,
    /// Read as an edge label.
    pub edge_label: u32,
}

impl ListVotes {
    /// The majority list; ties prefer node, then sticky, then edge label, then other.
    pub fn kind(&self) -> ObsList {
        let order = [
            (ObsList::Node, self.node),
            (ObsList::Sticky, self.sticky),
            (ObsList::EdgeLabel, self.edge_label),
            (ObsList::Other, self.other),
        ];
        let mut best = order[0];
        for o in &order[1..] {
            if o.1 > best.1 {
                best = *o;
            }
        }
        best.0
    }
}

/// A text element followed across keyframes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Track {
    /// Sightings in time order (at most one per keyframe).
    pub obs: Vec<Obs>,
}

impl Track {
    /// Normalized variant to (display text, count), in first-seen order of keys.
    pub fn variants(&self) -> BTreeMap<String, (String, u32)> {
        let mut m: BTreeMap<String, (String, u32)> = BTreeMap::new();
        for o in &self.obs {
            let e = m.entry(normalize(&o.text)).or_insert((o.text.clone(), 0));
            e.1 += 1;
            e.0.clone_from(&o.text);
        }
        m
    }

    /// Most frequent variant as last written (ties go to the most recent sighting).
    pub fn text(&self) -> String {
        let v = self.variants();
        let last_pos = |k: &str| {
            self.obs
                .iter()
                .rposition(|o| normalize(&o.text) == k)
                .unwrap_or(0)
        };
        v.iter()
            .max_by(|a, b| a.1 .1.cmp(&b.1 .1).then(last_pos(a.0).cmp(&last_pos(b.0))))
            .map(|(_, (t, _))| t.clone())
            .unwrap_or_default()
    }

    /// Label history: `(sighting index, normalized text)` for the initial label and
    /// every change. A change is a variant that differs from the established label
    /// (ratio below `fuzzy`) and holds for two consecutive sightings.
    pub fn label_changes(&self, fuzzy: f64) -> Vec<(usize, String)> {
        let Some(first) = self.obs.first() else {
            return Vec::new();
        };
        let mut out = vec![(0, normalize(&first.text))];
        for (i, w) in self.obs.windows(2).enumerate() {
            let est = out.last().map(|e| e.1.clone()).unwrap_or_default();
            let (n0, n1) = (normalize(&w[0].text), normalize(&w[1].text));
            let differs = n0 != est && difflib::ratio(&n0, &est) < fuzzy;
            if differs && (n0 == n1 || difflib::ratio(&n0, &n1) >= fuzzy) {
                out.push((i, n0));
            }
        }
        out
    }

    /// Current label: the most frequent variant (latest spelling) among sightings
    /// since the last label change that match the established label.
    pub fn current_text(&self, fuzzy: f64) -> String {
        let hist = self.label_changes(fuzzy);
        let Some((start, est)) = hist.last().cloned() else {
            return String::new();
        };
        let mut counts: Vec<(String, String, u32, usize)> = Vec::new();
        for (i, o) in self.obs.iter().enumerate().skip(start) {
            let n = normalize(&o.text);
            if n != est && difflib::ratio(&n, &est) < fuzzy {
                continue;
            }
            match counts.iter_mut().find(|c| c.0 == n) {
                Some(c) => {
                    c.1.clone_from(&o.text);
                    c.2 += 1;
                    c.3 = i;
                }
                None => counts.push((n, o.text.clone(), 1, i)),
            }
        }
        counts
            .into_iter()
            .max_by(|a, b| a.2.cmp(&b.2).then(a.3.cmp(&b.3)))
            .map(|c| c.1)
            .unwrap_or_default()
    }

    /// Votes over lists.
    pub fn votes(&self) -> ListVotes {
        let mut v = ListVotes::default();
        for o in &self.obs {
            for l in &o.lists {
                match l {
                    ObsList::Node => v.node += 1,
                    ObsList::Sticky => v.sticky += 1,
                    ObsList::Other => v.other += 1,
                    ObsList::EdgeLabel => v.edge_label += 1,
                }
            }
        }
        v
    }

    /// Sorted keyframe indices.
    pub fn frames(&self) -> Vec<usize> {
        let mut f: Vec<usize> = self.obs.iter().map(|o| o.frame).collect();
        f.sort_unstable();
        f.dedup();
        f
    }

    /// Median box of the sightings in `cluster`.
    pub fn bbox_in(&self, cluster: usize) -> Option<BBox> {
        let boxes: Vec<BBox> = self
            .obs
            .iter()
            .filter(|o| o.cluster == cluster)
            .filter_map(|o| o.bbox)
            .collect();
        if boxes.is_empty() {
            return None;
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        Some(BBox::new(
            med(boxes.iter().map(|b| b.x1).collect()),
            med(boxes.iter().map(|b| b.y1).collect()),
            med(boxes.iter().map(|b| b.x2).collect()),
            med(boxes.iter().map(|b| b.y2).collect()),
        ))
    }

    /// Most frequent sticky color.
    pub fn color(&self) -> Option<StickyColor> {
        let mut counts: Vec<(StickyColor, u32)> = Vec::new();
        for c in self.obs.iter().filter_map(|o| o.color) {
            match counts.iter_mut().find(|e| e.0 == c) {
                Some(e) => e.1 += 1,
                None => counts.push((c, 1)),
            }
        }
        counts.into_iter().max_by_key(|e| e.1).map(|e| e.0)
    }
}

/// Matching settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchParams {
    /// Fuzzy text threshold.
    pub fuzzy: f64,
    /// Same-place tolerance in reference pixels.
    pub position_tolerance_px: f64,
    /// IoU for a same-place node with different text to count as a label change.
    pub label_change_min_iou: f64,
    /// Largest offset, in reference pixels, at which a same-text reading may still
    /// join a track when the view moved (imprecise boxes and registration).
    pub off_position_max_px: f64,
}

fn center(b: &BBox) -> (f64, f64) {
    ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
}

/// Assign each observation of one keyframe to a track (one-to-one, best score first),
/// creating new tracks for the rest. Returns the track index of every observation.
/// `view_moved(a, b)` says whether the registered view changed between keyframes `a`
/// and `b`.
pub fn assign_frame(
    tracks: &mut Vec<Track>,
    obs: Vec<Obs>,
    p: &MatchParams,
    view_moved: &dyn Fn(usize, usize) -> bool,
) -> Vec<usize> {
    let mut cands: Vec<(f64, usize, usize)> = Vec::new();
    for (oi, o) in obs.iter().enumerate() {
        let on = normalize(&o.text);
        for (ti, t) in tracks.iter().enumerate() {
            let text_score = t
                .variants()
                .keys()
                .map(|k| {
                    if *k == on {
                        1.0
                    } else {
                        difflib::ratio(k, &on)
                    }
                })
                .fold(0.0, f64::max);
            let place = o
                .bbox
                .and_then(|ob| t.bbox_in(o.cluster).map(|tb| (ob, tb)));
            let score = match place {
                Some((ob, tb)) => {
                    let (c1, c2) = (center(&ob), center(&tb));
                    let d = ((c1.0 - c2.0).powi(2) + (c1.1 - c2.1).powi(2)).sqrt();
                    let iou = ob.iou(&tb);
                    let same_place = d <= p.position_tolerance_px || iou >= 0.3;
                    if text_score >= p.fuzzy && same_place {
                        Some(2.0 + text_score)
                    } else if text_score >= p.fuzzy {
                        // Same text in another place of the same canvas is another
                        // element, except when a non-sticky reading is only a little
                        // off and the view moved since the track was last seen: then
                        // the offset is more likely an imprecise box or registration
                        // than a second element. It still ranks below any same-place
                        // match; duplicates within one keyframe stay apart through the
                        // one-to-one assignment.
                        let sticky_pair =
                            o.lists.contains(&ObsList::Sticky) && t.votes().sticky > 0;
                        let moved = t.obs.last().is_some_and(|l| view_moved(l.frame, o.frame));
                        (!sticky_pair && moved && d <= p.off_position_max_px)
                            .then_some(1.2 + text_score)
                    } else if iou >= p.label_change_min_iou
                        && o.lists.contains(&ObsList::Node)
                        && t.votes().node > 0
                    {
                        Some(0.5 + iou)
                    } else {
                        None
                    }
                }
                None => (text_score >= p.fuzzy).then_some(1.0 + text_score),
            };
            if let Some(s) = score {
                cands.push((s, oi, ti));
            }
        }
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut assigned: Vec<Option<usize>> = vec![None; obs.len()];
    let mut used = vec![false; tracks.len()];
    for (_, oi, ti) in cands {
        if assigned[oi].is_none() && !used[ti] {
            assigned[oi] = Some(ti);
            used[ti] = true;
        }
    }
    let mut out = Vec::with_capacity(obs.len());
    for (oi, o) in obs.into_iter().enumerate() {
        let ti = match assigned[oi] {
            Some(ti) => ti,
            None => {
                tracks.push(Track::default());
                tracks.len() - 1
            }
        };
        tracks[ti].obs.push(o);
        out.push(ti);
    }
    out
}

/// Visibility of an element in one keyframe (registered keyframes only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// The element's place is inside the keyframe's view.
    Visible,
    /// Outside the view, or unknown (text-only registration).
    Unknown,
}

/// One supported lifetime interval, as keyframe indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    /// First sighting.
    pub first: usize,
    /// Last sighting.
    pub last: usize,
    /// Sightings in the interval.
    pub count: usize,
    /// First keyframe of the absence that ended it, when it was removed.
    pub removed_at: Option<usize>,
}

/// Support settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SupportParams {
    /// Minimum sightings.
    pub min_keyframes: usize,
    /// Minimum share of board keyframes within the interval.
    pub min_density: f64,
    /// Consecutive visible-but-absent keyframes that end an interval.
    pub removal_absent: usize,
}

/// Build lifetime intervals from sightings and keep the supported ones.
///
/// Sightings are merged left to right while no removal (a run of `removal_absent`
/// visible-but-absent keyframes) lies between them and the merged interval keeps at
/// least `min_density` of the board keyframes it spans. An interval is supported with
/// at least `min_keyframes` sightings; single sightings survive only when
/// `confirm_single` says so (pixel check plus a second reader pass).
pub fn intervals(
    seen: &[usize],
    visibility: impl Fn(usize) -> Visibility,
    n_frames: usize,
    p: &SupportParams,
    confirm_single: impl Fn(usize) -> bool,
) -> Vec<Interval> {
    let removal_between = |from: usize, to: usize| -> Option<usize> {
        let mut run = 0usize;
        let mut start = None;
        for f in from..to {
            if visibility(f) == Visibility::Visible {
                if run == 0 {
                    start = Some(f);
                }
                run += 1;
                if run >= p.removal_absent.max(1) {
                    return start;
                }
            } else {
                run = 0;
            }
        }
        None
    };
    let mut out: Vec<Interval> = Vec::new();
    let mut cur: Option<Interval> = None;
    for &s in seen {
        cur = match cur {
            None => Some(Interval {
                first: s,
                last: s,
                count: 1,
                removed_at: None,
            }),
            Some(c) => {
                let removed = removal_between(c.last + 1, s);
                let span = s - c.first + 1;
                let dense = (c.count + 1) as f64 / span as f64 >= p.min_density;
                if removed.is_none() && dense {
                    Some(Interval {
                        last: s,
                        count: c.count + 1,
                        ..c
                    })
                } else {
                    out.push(Interval {
                        removed_at: removed,
                        ..c
                    });
                    Some(Interval {
                        first: s,
                        last: s,
                        count: 1,
                        removed_at: None,
                    })
                }
            }
        };
    }
    if let Some(c) = cur {
        out.push(Interval {
            removed_at: removal_between(c.last + 1, n_frames),
            ..c
        });
    }
    out.into_iter()
        .filter(|i| i.count >= p.min_keyframes || (i.count == 1 && confirm_single(i.first)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp() -> SupportParams {
        SupportParams {
            min_keyframes: 2,
            min_density: 0.1,
            removal_absent: 2,
        }
    }

    #[test]
    fn singles_are_dropped_and_sparse_sightings_split() {
        let iv = intervals(&[3], |_| Visibility::Unknown, 50, &sp(), |_| false);
        assert!(iv.is_empty());
        // Two sightings 40 keyframes apart: density 2/41 < 10%, both singles.
        let iv = intervals(&[2, 42], |_| Visibility::Unknown, 50, &sp(), |_| false);
        assert!(iv.is_empty());
        let iv = intervals(&[2, 4, 5, 9], |_| Visibility::Unknown, 50, &sp(), |_| false);
        assert_eq!(iv.len(), 1);
        assert_eq!((iv[0].first, iv[0].last, iv[0].count), (2, 9, 4));
    }

    #[test]
    fn visible_absence_removes() {
        let vis = |f: usize| {
            if (6..9).contains(&f) {
                Visibility::Visible
            } else {
                Visibility::Unknown
            }
        };
        let iv = intervals(&[1, 2, 3, 10, 11], vis, 20, &sp(), |_| false);
        assert_eq!(iv.len(), 2);
        assert_eq!(iv[0].removed_at, Some(6));
        assert_eq!(iv[1].first, 10);
        // One visible-absent keyframe is not a removal.
        let one = |f: usize| {
            if f == 6 {
                Visibility::Visible
            } else {
                Visibility::Unknown
            }
        };
        assert_eq!(
            intervals(&[1, 2, 3, 10, 11], one, 20, &sp(), |_| false).len(),
            1
        );
    }

    #[test]
    fn list_votes_and_variants() {
        let mut t = Track::default();
        for (f, text, l) in [
            (0, "Cache Layer", ObsList::Node),
            (1, "cache layer", ObsList::Other),
            (2, "Cache Layer", ObsList::Node),
        ] {
            t.obs.push(Obs {
                frame: f,
                lists: vec![l],
                text: text.into(),
                bbox: None,
                raw_bbox: None,
                cluster: usize::MAX,
                local_id: None,
                color: None,
                single_ok: false,
            });
        }
        assert_eq!(t.votes().kind(), ObsList::Node);
        assert_eq!(t.text(), "Cache Layer");
        assert_eq!(t.frames(), vec![0, 1, 2]);
    }
}
