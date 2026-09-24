//! Board cleanup: title bar text, fragment folding, and sticky group derivation.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use glassrip_vision::BBox;

use super::anchor::box_distance;
use super::tracks::{ObsList, Track};
use super::{canvas_of, BoardFrame, ConsolidationParams, StickyGroup};
use crate::difflib;
use crate::text::{is_fragment_of, normalize, tokens};

fn elided(n: &str) -> Option<String> {
    let t = n.trim_end_matches(['.', '\u{2026}', ' ']);
    (t.len() < n.len()).then(|| t.to_string())
}

/// The board's title as read from the title bar, the app panel, or the producer.
#[derive(Debug, Clone, Default)]
pub(crate) struct Titles {
    /// Normalized full titles with how often they were seen, and a display form.
    exact: BTreeMap<String, (usize, String)>,
    /// Normalized prefixes of elided titles ("Board name...").
    prefixes: Vec<String>,
    /// Normalized app panel texts (board list and title bar entries).
    panel: Vec<String>,
}

fn title_like(n: &str) -> bool {
    tokens(n).len() >= 2 && n.len() >= 8
}

impl Titles {
    fn add(&mut self, text: &str, counted: bool) {
        let n = normalize(text);
        if !title_like(&n) {
            return;
        }
        match elided(&n) {
            Some(p) if p.len() >= 8 => self.prefixes.push(p),
            Some(_) => {}
            None if counted => {
                let e = self.exact.entry(n).or_insert((0, text.trim().to_string()));
                e.0 += 1;
            }
            None => self.panel.push(n),
        }
    }

    /// Titles from every keyframe: the producer's board title, text entirely inside
    /// the title band, and app panel text (matching only, never the board title).
    pub(crate) fn collect(frames: &[BoardFrame], p: &ConsolidationParams) -> Self {
        let mut t = Self::default();
        for f in frames {
            if let Some(bt) = &f.board_title {
                t.add(bt, true);
            }
            for h in &f.title_hints {
                t.add(h, false);
            }
            let Some(c) = canvas_of(f).0 else { continue };
            let band = p.title_band_share * c.height;
            let b = &f.board;
            let items = b
                .nodes
                .iter()
                .map(|x| (&x.text, &x.bbox))
                .chain(b.stickies.iter().map(|x| (&x.text, &x.bbox)))
                .chain(b.other_visible_text.iter().map(|x| (&x.text, &x.bbox)));
            for (text, bb) in items {
                if bb.y2 <= band {
                    t.add(text, true);
                }
            }
        }
        t
    }

    /// True when an element is title bar text: inside the title band, or reading as
    /// a known title (fuzzy, or by an elided prefix).
    pub(crate) fn is_title(
        &self,
        f: &BoardFrame,
        text: &str,
        b: &BBox,
        p: &ConsolidationParams,
    ) -> bool {
        if let Some(c) = canvas_of(f).0 {
            if b.y2 <= p.title_band_share * c.height {
                return true;
            }
        }
        let n = normalize(text);
        if !title_like(&n) {
            return false;
        }
        let near = |t: &String| difflib::ratio(&n, t) >= p.fuzzy_threshold;
        if self.exact.keys().any(near) || self.panel.iter().any(near) {
            return true;
        }
        let own_prefix = elided(&n);
        self.prefixes.iter().any(|pre| n.starts_with(pre.as_str()))
            || own_prefix.is_some_and(|op| {
                op.len() >= 8
                    && self
                        .exact
                        .keys()
                        .chain(self.panel.iter())
                        .any(|t| t.starts_with(&op))
            })
    }

    /// The most frequent full title, as written.
    pub(crate) fn board_title(&self) -> Option<String> {
        self.exact
            .values()
            .max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))
            .map(|v| v.1.clone())
    }
}

/// Fold fragments: a node or sticky whose text is a piece of a longer element of the
/// same kind at the same place (a cut-off or split reading). Where the two share a
/// registered cluster their positions must be close; otherwise the fragment must never
/// be seen apart from the longer element in one keyframe. The longer element must be
/// seen at least as often. Returns fragment track to target track.
pub(crate) fn fold_fragments(tracks: &[Track]) -> HashMap<usize, usize> {
    let kind = |t: &Track| t.votes().kind();
    let mut out = HashMap::new();
    for (ti, t) in tracks.iter().enumerate() {
        let k = kind(t);
        if !matches!(k, ObsList::Node | ObsList::Sticky) {
            continue;
        }
        let (text, frames_t) = (t.text(), t.frames());
        let mut best: Option<(usize, usize)> = None;
        for (ui, u) in tracks.iter().enumerate() {
            if ui == ti || kind(u) != k {
                continue;
            }
            let frames_u = u.frames();
            if frames_u.len() < frames_t.len() || !is_fragment_of(&text, &u.text()) {
                continue;
            }
            // Seen apart in one keyframe: a different element.
            let apart = t.obs.iter().any(|a| {
                u.obs.iter().any(|b| {
                    a.frame == b.frame
                        && match (a.raw_bbox, b.raw_bbox) {
                            (Some(x), Some(y)) => box_distance(&x, &y) > y.width().max(y.height()),
                            _ => false,
                        }
                })
            });
            if apart {
                continue;
            }
            // Registered positions, where both have them.
            let clusters: BTreeSet<usize> = t
                .obs
                .iter()
                .filter(|o| o.bbox.is_some())
                .map(|o| o.cluster)
                .collect();
            let placed: Vec<bool> = clusters
                .iter()
                .filter_map(|&c| {
                    let (x, y) = (t.bbox_in(c)?, u.bbox_in(c)?);
                    Some(box_distance(&x, &y) <= 0.5 * y.width().max(y.height()))
                })
                .collect();
            if placed.iter().any(|v| !v) {
                continue;
            }
            if best.is_none_or(|(_, n)| frames_u.len() > n) {
                best = Some((ui, frames_u.len()));
            }
        }
        if let Some((ui, _)) = best {
            out.insert(ti, ui);
        }
    }
    out
}

fn center(b: &BBox) -> (f64, f64) {
    ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
}

fn similar(a: &BBox, b: &BBox) -> bool {
    let r = |x: f64, y: f64| {
        let q = x / y.max(1e-9);
        (0.6..=1.67).contains(&q)
    };
    r(a.width(), b.width()) && r(a.height(), b.height())
}

fn count_lines(values: &mut [f64], tol: f64) -> u32 {
    values.sort_by(f64::total_cmp);
    let mut n = 0u32;
    let mut last: Option<f64> = None;
    for &v in values.iter() {
        if last.is_none_or(|l| v - l > tol) {
            n += 1;
        }
        last = Some(v);
    }
    n
}

/// Derive sticky groups: in each keyframe, final stickies of similar size aligned in
/// rows or columns with gaps under one card size form a group of at least
/// `min_cards`. Groups are taken from the keyframe that shows the most cards, merging
/// repeats. A text just above a group (within one card height, overlapping it
/// horizontally) is its heading. Returns the groups and the heading tracks.
pub(crate) fn derive_groups(
    tracks: &[Track],
    frames: &[BoardFrame],
    final_sticky: &HashMap<usize, String>,
    min_cards: usize,
) -> (Vec<StickyGroup>, Vec<usize>) {
    struct Cand {
        members: Vec<(usize, BBox)>,
        frame: usize,
    }
    let mut cands: Vec<Cand> = Vec::new();
    for fi in 0..frames.len() {
        let mut items: Vec<(usize, BBox)> = Vec::new();
        for (&ti, _) in final_sticky.iter() {
            if let Some(b) = tracks[ti]
                .obs
                .iter()
                .find(|o| o.frame == fi)
                .and_then(|o| o.raw_bbox)
            {
                items.push((ti, b));
            }
        }
        items.sort_by_key(|x| x.0);
        let n = items.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for i in 0..n {
            for j in i + 1..n {
                let (a, b) = (&items[i].1, &items[j].1);
                if !similar(a, b) {
                    continue;
                }
                let (w, h) = (
                    (a.width() + b.width()) / 2.0,
                    (a.height() + b.height()) / 2.0,
                );
                let (ca, cb) = (center(a), center(b));
                let hgap = a.x1.max(b.x1) - a.x2.min(b.x2);
                let vgap = a.y1.max(b.y1) - a.y2.min(b.y2);
                let row = (ca.1 - cb.1).abs() <= 0.35 * h && hgap <= w;
                let col = (ca.0 - cb.0).abs() <= 0.35 * w && vgap <= h;
                if row || col {
                    let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                    parent[ri.max(rj)] = ri.min(rj);
                }
            }
        }
        let mut comps: BTreeMap<usize, Vec<(usize, BBox)>> = BTreeMap::new();
        for (i, it) in items.iter().enumerate() {
            let r = find(&mut parent, i);
            comps.entry(r).or_default().push(*it);
        }
        cands.extend(
            comps
                .into_values()
                .filter(|c| c.len() >= min_cards.max(2))
                .map(|members| Cand { members, frame: fi }),
        );
    }
    cands.sort_by(|a, b| {
        b.members
            .len()
            .cmp(&a.members.len())
            .then(a.frame.cmp(&b.frame))
    });
    let mut accepted: Vec<Cand> = Vec::new();
    for c in cands {
        let set: BTreeSet<usize> = c.members.iter().map(|m| m.0).collect();
        let overlaps = accepted.iter().any(|a| {
            let s: BTreeSet<usize> = a.members.iter().map(|m| m.0).collect();
            2 * s.intersection(&set).count() >= set.len()
        });
        if !overlaps {
            accepted.push(c);
        }
    }
    let mut groups = Vec::new();
    let mut headings = Vec::new();
    for (gi, c) in accepted.into_iter().enumerate() {
        let boxes: Vec<BBox> = c.members.iter().map(|m| m.1).collect();
        let k = boxes.len() as f64;
        let (mw, mh) = (
            boxes.iter().map(BBox::width).sum::<f64>() / k,
            boxes.iter().map(BBox::height).sum::<f64>() / k,
        );
        let ext = BBox::new(
            boxes.iter().map(|b| b.x1).fold(f64::INFINITY, f64::min),
            boxes.iter().map(|b| b.y1).fold(f64::INFINITY, f64::min),
            boxes.iter().map(|b| b.x2).fold(f64::NEG_INFINITY, f64::max),
            boxes.iter().map(|b| b.y2).fold(f64::NEG_INFINITY, f64::max),
        );
        let rows = count_lines(
            &mut boxes.iter().map(|b| center(b).1).collect::<Vec<_>>(),
            0.5 * mh,
        );
        let cols = count_lines(
            &mut boxes.iter().map(|b| center(b).0).collect::<Vec<_>>(),
            0.5 * mw,
        );
        let mut members = c.members.clone();
        members.sort_by(|a, b| {
            let (ca, cb) = (center(&a.1), center(&b.1));
            ((ca.1 / (0.5 * mh)).round())
                .total_cmp(&(cb.1 / (0.5 * mh)).round())
                .then(ca.0.total_cmp(&cb.0))
        });
        let member_set: BTreeSet<usize> = members.iter().map(|m| m.0).collect();
        // Heading: nearest text just above the group, overlapping it horizontally.
        let heading = tracks
            .iter()
            .enumerate()
            .filter(|(ti, t)| !member_set.contains(ti) && t.votes().kind() != ObsList::EdgeLabel)
            .filter_map(|(ti, t)| {
                let b = t.obs.iter().find(|o| o.frame == c.frame)?.raw_bbox?;
                let overlap = (b.x2.min(ext.x2) - b.x1.max(ext.x1)).max(0.0);
                let above = ext.y1 - b.y2;
                (overlap >= 0.5 * b.width() && above >= -0.25 * mh && above <= mh)
                    .then_some((ti, above.abs()))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|x| x.0);
        if let Some(h) = heading {
            headings.push(h);
        }
        groups.push(StickyGroup {
            id: format!("group-{}", gi + 1),
            title: heading.map(|h| tracks[h].text()),
            sticky_ids: members
                .iter()
                .filter_map(|m| final_sticky.get(&m.0).cloned())
                .collect(),
            rows,
            cols,
            keyframe_id: frames[c.frame].keyframe_id.clone(),
            bbox: ext,
        });
    }
    (groups, headings)
}
