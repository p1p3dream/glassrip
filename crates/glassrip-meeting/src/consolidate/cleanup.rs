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

/// The board's title, from corroborated title bar text, the app panel, or the producer.
///
/// Text inside the title band of the canvas is only treated as a title when it looks
/// like one (two or more words, eight or more characters) and is corroborated: it
/// matches the producer's board title or app panel text, or it persists (read inside
/// the band in at least two keyframes, and no more often outside it than inside). Outside the band, only
/// chrome context removes text: app panel entries and elided title prefixes.
#[derive(Debug, Clone, Default)]
pub(crate) struct Titles {
    /// Corroborated normalized titles with how often they were seen, and a display form.
    exact: BTreeMap<String, (usize, String)>,
    /// Normalized prefixes of elided titles ("Board name...").
    prefixes: Vec<String>,
    /// Normalized app panel texts (chrome context).
    panel: Vec<String>,
    /// Normalized producer board titles (corroborate band text only).
    producer: Vec<String>,
}

fn title_like(n: &str) -> bool {
    tokens(n).len() >= 2 && n.len() >= 8
}

impl Titles {
    fn chrome_match(&self, n: &str, p: &ConsolidationParams) -> bool {
        let near = |t: &String| difflib::ratio(n, t) >= p.fuzzy_threshold;
        if self.panel.iter().any(near)
            || self.prefixes.iter().any(|pre| n.starts_with(pre.as_str()))
        {
            return true;
        }
        elided(n).is_some_and(|op| op.len() >= 8 && self.panel.iter().any(|t| t.starts_with(&op)))
    }

    /// Titles from every keyframe (see the type docs).
    pub(crate) fn collect(frames: &[BoardFrame], p: &ConsolidationParams) -> Self {
        let mut t = Self::default();
        let mut producer: Vec<(String, String)> = Vec::new();
        for f in frames {
            for h in &f.title_hints {
                let n = normalize(h);
                if !title_like(&n) {
                    continue;
                }
                match elided(&n) {
                    Some(pre) if pre.len() >= 8 => t.prefixes.push(pre),
                    Some(_) => {}
                    None => t.panel.push(n.clone()),
                }
            }
            if let Some(bt) = &f.board_title {
                producer.push((normalize(bt), bt.trim().to_string()));
                t.producer.push(normalize(bt));
            }
        }
        // Band text: counts inside the band, and whether it is ever read outside it.
        let mut band: BTreeMap<String, (usize, usize, String)> = BTreeMap::new();
        for f in frames {
            let Some(c) = canvas_of(f).0 else { continue };
            let limit = p.title_band_share * c.height;
            let b = &f.board;
            let items = b
                .nodes
                .iter()
                .map(|x| (&x.text, &x.bbox))
                .chain(b.stickies.iter().map(|x| (&x.text, &x.bbox)))
                .chain(b.other_visible_text.iter().map(|x| (&x.text, &x.bbox)));
            for (text, bb) in items {
                let n = normalize(text);
                if !title_like(&n) || elided(&n).is_some() {
                    continue;
                }
                let e = band.entry(n).or_insert((0, 0, text.trim().to_string()));
                if bb.y2 <= limit {
                    e.0 += 1;
                } else {
                    e.1 += 1;
                }
            }
        }
        for (n, (inside, outside, display)) in band {
            let persistent = inside >= 2 && inside >= outside;
            let by_producer = t
                .producer
                .iter()
                .any(|x| difflib::ratio(&n, x) >= p.fuzzy_threshold);
            let corroborated = inside >= 1 && (by_producer || t.chrome_match(&n, p));
            if persistent || corroborated {
                t.exact.insert(n, (inside, display));
            }
        }
        for (n, display) in producer {
            if title_like(&n) {
                let e = t.exact.entry(n).or_insert((0, display));
                e.0 += 1;
            }
        }
        t
    }

    /// True when an element is title bar text: a corroborated title read inside the
    /// title band, or chrome context (app panel text, an elided title prefix) anywhere.
    pub(crate) fn is_title(
        &self,
        f: &BoardFrame,
        text: &str,
        b: &BBox,
        p: &ConsolidationParams,
    ) -> bool {
        let n = normalize(text);
        if !title_like(&n) {
            return false;
        }
        let in_band = canvas_of(f)
            .0
            .is_some_and(|c| b.y2 <= p.title_band_share * c.height);
        if in_band && self.exact.contains_key(&n) {
            return true;
        }
        self.chrome_match(&n, p)
    }

    /// The most frequent corroborated title, as written.
    pub(crate) fn board_title(&self) -> Option<String> {
        self.exact
            .values()
            .max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))
            .map(|v| v.1.clone())
    }
}

/// Fold fragments: a node or sticky whose text is a piece of a longer element of the
/// same kind at the same place (a cut-off or split reading). Folding needs positive
/// geometric evidence (near each other in a keyframe that shows both, or in a shared
/// registered cluster) and no evidence of being apart; pairs never compared are not
/// folded. The longer element must be seen at least as often. Returns fragment track
/// to target track.
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
            // Positive geometric evidence is required: the two are near each other
            // in a keyframe that shows both, or in a registered cluster where both are
            // placed. Seen apart anywhere, or never compared, they are different.
            let (mut near, mut apart) = (false, false);
            let mut judge = |x: &BBox, y: &BBox| {
                let size = y.width().max(y.height());
                let d = box_distance(x, y);
                if d > size {
                    apart = true;
                } else if d <= 0.5 * size {
                    near = true;
                }
            };
            for a in &t.obs {
                for b in u.obs.iter().filter(|b| b.frame == a.frame) {
                    if let (Some(x), Some(y)) = (a.raw_bbox, b.raw_bbox) {
                        judge(&x, &y);
                    }
                }
            }
            let clusters: BTreeSet<usize> = t
                .obs
                .iter()
                .filter(|o| o.bbox.is_some())
                .map(|o| o.cluster)
                .collect();
            for c in clusters {
                if let (Some(x), Some(y)) = (t.bbox_in(c), u.bbox_in(c)) {
                    judge(&x, &y);
                }
            }
            if apart || !near {
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

/// Line index of each value: values sorted, a new line where the gap exceeds `tol`.
fn line_index(values: &[f64], tol: f64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut out = vec![0usize; values.len()];
    let mut line = 0usize;
    let mut last: Option<f64> = None;
    for i in order {
        if last.is_some_and(|l| values[i] - l > tol) {
            line += 1;
        }
        out[i] = line;
        last = Some(values[i]);
    }
    out
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
    excluded: &dyn Fn(usize) -> bool,
) -> (Vec<StickyGroup>, Vec<(usize, String)>) {
    struct Cand {
        members: Vec<(usize, BBox)>,
        frame: usize,
    }
    let mut cands: Vec<Cand> = Vec::new();
    for fi in 0..frames.len() {
        let mut items: Vec<(usize, BBox)> = Vec::new();
        for &ti in final_sticky.keys() {
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
    // Components within one keyframe are disjoint, so candidates share cards only
    // when two keyframes saw the same cards: those shared cards stay with the larger
    // view already taken, and a candidate stands only on the cards no group holds
    // yet (a new row or column seen later becomes its own group).
    let mut accepted: Vec<Cand> = Vec::new();
    let mut taken: BTreeSet<usize> = BTreeSet::new();
    for mut c in cands {
        c.members.retain(|m| !taken.contains(&m.0));
        if c.members.len() >= min_cards.max(2) {
            taken.extend(c.members.iter().map(|m| m.0));
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
        // One clustering gives both the row and column counts and the member order.
        let row_of = line_index(
            &boxes.iter().map(|b| center(b).1).collect::<Vec<_>>(),
            0.5 * mh,
        );
        let col_of = line_index(
            &boxes.iter().map(|b| center(b).0).collect::<Vec<_>>(),
            0.5 * mw,
        );
        let rows = row_of.iter().max().map_or(0, |m| m + 1) as u32;
        let cols = col_of.iter().max().map_or(0, |m| m + 1) as u32;
        let mut order: Vec<usize> = (0..c.members.len()).collect();
        order.sort_by_key(|&i| (row_of[i], col_of[i]));
        let members: Vec<(usize, BBox)> = order.iter().map(|&i| c.members[i]).collect();
        let member_set: BTreeSet<usize> = members.iter().map(|m| m.0).collect();
        // Heading: nearest text just above the group, overlapping it horizontally.
        let heading = tracks
            .iter()
            .enumerate()
            .filter(|(ti, t)| {
                !member_set.contains(ti) && !excluded(*ti) && t.votes().kind() != ObsList::EdgeLabel
            })
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
            headings.push((h, format!("group-{}", gi + 1)));
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
