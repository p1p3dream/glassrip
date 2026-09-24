//! Text-anchor registration of board keyframes into shared canvas frames.
//!
//! Anchors are texts that occur exactly once in a keyframe (nodes, stickies, other
//! canvas text; owner tags are excluded because they move). Two keyframes share an
//! anchor when the normalized texts are equal, or when they are the unique best fuzzy
//! match of each other at [`crate::text::FUZZY_THRESHOLD`]. A similarity transform is
//! fit over the shared anchors' box centers with [`crate::similarity::ransac`]
//! (at least 3 inliers, residual below a share of the canvas diagonal).
//!
//! Chaining: a cluster starts from the unregistered keyframe with the most shared
//! anchors; the cluster then repeatedly absorbs the unregistered keyframe with the
//! most shared anchors to any member whose fit succeeds, composing transforms back to
//! the cluster reference. A pan or zoom long after the reference still registers
//! through any member it shares labels with. Keyframes left alone form singleton
//! clusters and are merged elsewhere by text only (`registration: text_only`).

use std::collections::{BTreeMap, HashMap};

use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::difflib;
use crate::similarity::{ransac, Point, RansacParams, Similarity};
use crate::text::{is_unreliable, normalize, FUZZY_THRESHOLD};

/// Unique texts of one keyframe with their box centers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Anchors {
    /// Normalized text to box center.
    pub points: BTreeMap<String, Point>,
}

impl Anchors {
    /// Keep texts that occur once, are reliable, and have well-formed boxes.
    pub fn from_items<'a>(items: impl IntoIterator<Item = (&'a str, &'a BBox)>) -> Self {
        let mut seen: HashMap<String, (usize, Point)> = HashMap::new();
        for (text, b) in items {
            if is_unreliable(text) || !b.is_well_formed() {
                continue;
            }
            let c = ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0);
            seen.entry(normalize(text))
                .and_modify(|e| e.0 += 1)
                .or_insert((1, c));
        }
        Self {
            points: seen
                .into_iter()
                .filter(|(_, (n, _))| *n == 1)
                .map(|(k, (_, c))| (k, c))
                .collect(),
        }
    }
}

/// Correspondences `(point in a, point in b)` between two anchor sets.
pub fn shared_anchors(a: &Anchors, b: &Anchors) -> Vec<(Point, Point)> {
    let mut out = Vec::new();
    let mut used_b: Vec<&str> = Vec::new();
    let mut fuzzy_a = Vec::new();
    for (k, pa) in &a.points {
        match b.points.get(k) {
            Some(pb) => {
                out.push((*pa, *pb));
                used_b.push(k);
            }
            None => fuzzy_a.push((k, *pa)),
        }
    }
    // Fuzzy pass: mutual unique best matches among the leftovers.
    let rest_b: Vec<(&String, &Point)> = b
        .points
        .iter()
        .filter(|(k, _)| !used_b.contains(&k.as_str()) && !a.points.contains_key(*k))
        .collect();
    let best_in = |key: &str, pool: &[(&String, Point)]| -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        let mut tie = false;
        for (i, (k, _)) in pool.iter().enumerate() {
            let r = difflib::ratio(key, k);
            if r < FUZZY_THRESHOLD {
                continue;
            }
            match best {
                Some((_, s)) if r < s => {}
                Some((_, s)) if r == s => tie = true,
                _ => {
                    best = Some((i, r));
                    tie = false;
                }
            }
        }
        if tie {
            None
        } else {
            best.map(|b| b.0)
        }
    };
    let pool_b: Vec<(&String, Point)> = rest_b.iter().map(|(k, p)| (*k, **p)).collect();
    let pool_a: Vec<(&String, Point)> = fuzzy_a.iter().map(|(k, p)| (*k, *p)).collect();
    for (ia, (ka, pa)) in pool_a.iter().enumerate() {
        if let Some(ib) = best_in(ka, &pool_b) {
            if best_in(pool_b[ib].0, &pool_a) == Some(ia) {
                out.push((*pa, pool_b[ib].1));
            }
        }
    }
    out
}

/// Registration settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RegistrationParams {
    /// Inlier residual as a share of the destination canvas diagonal.
    pub residual_share_of_diagonal: f64,
    /// Minimum RANSAC inliers.
    pub min_inliers: usize,
}

impl Default for RegistrationParams {
    fn default() -> Self {
        Self {
            residual_share_of_diagonal: 0.02,
            min_inliers: 3,
        }
    }
}

/// How a keyframe entered the shared frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RegistrationMode {
    /// Reference of its cluster.
    Reference,
    /// Registered to its cluster through text anchors.
    Registered,
    /// Could not register; merged by text only.
    TextOnly,
}

/// Registration of one keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameRegistration {
    /// Cluster index.
    pub cluster: usize,
    /// Mode.
    pub mode: RegistrationMode,
    /// Transform from this keyframe's canvas to the cluster reference.
    pub to_reference: Similarity,
    /// Inliers of the fit that attached this keyframe (0 for references).
    pub inliers: usize,
    /// RMS residual of that fit in pixels.
    pub rms_px: f64,
    /// Keyframe index it was attached through.
    pub via: Option<usize>,
}

/// Transform, inlier count and RMS residual of one pairwise fit.
type Fit = (Similarity, usize, f64);

/// Register `anchors[i]` (keyframes in time order). `diagonals[i]` is keyframe `i`'s
/// canvas diagonal in pixels.
pub fn register(
    anchors: &[Anchors],
    diagonals: &[f64],
    params: &RegistrationParams,
) -> Vec<FrameRegistration> {
    let n = anchors.len();
    let shared: Vec<Vec<Vec<(Point, Point)>>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        Vec::new()
                    } else {
                        shared_anchors(&anchors[i], &anchors[j])
                    }
                })
                .collect()
        })
        .collect();
    let mut fit_cache: HashMap<(usize, usize), Option<Fit>> = HashMap::new();
    let mut fit = |k: usize, j: usize| -> Option<Fit> {
        *fit_cache.entry((k, j)).or_insert_with(|| {
            let pairs = &shared[k][j];
            if pairs.len() < params.min_inliers {
                return None;
            }
            let src: Vec<Point> = pairs.iter().map(|p| p.0).collect();
            let dst: Vec<Point> = pairs.iter().map(|p| p.1).collect();
            let rp = RansacParams {
                threshold_px: params.residual_share_of_diagonal
                    * diagonals.get(j).copied().unwrap_or(0.0),
                min_inliers: params.min_inliers,
                ..RansacParams::default()
            };
            ransac(&src, &dst, &rp).map(|f| (f.transform, f.inliers.len(), f.rms_px))
        })
    };
    let mut out: Vec<Option<FrameRegistration>> = vec![None; n];
    let mut cluster = 0usize;
    loop {
        let unreg: Vec<usize> = (0..n).filter(|&i| out[i].is_none()).collect();
        let Some(&reference) = unreg.iter().max_by(|&&a, &&b| {
            let sa: usize = unreg.iter().map(|&o| shared[a][o].len()).sum();
            let sb: usize = unreg.iter().map(|&o| shared[b][o].len()).sum();
            sa.cmp(&sb).then(b.cmp(&a))
        }) else {
            break;
        };
        out[reference] = Some(FrameRegistration {
            cluster,
            mode: RegistrationMode::Reference,
            to_reference: Similarity::IDENTITY,
            inliers: 0,
            rms_px: 0.0,
            via: None,
        });
        let mut members = vec![reference];
        loop {
            let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
            for k in (0..n).filter(|&i| out[i].is_none()) {
                for &j in &members {
                    let s = shared[k][j].len();
                    if s >= params.min_inliers {
                        candidates.push((s, k, j));
                    }
                }
            }
            candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            let mut attached = false;
            for (_, k, j) in candidates {
                if let Some((t, inl, rms)) = fit(k, j) {
                    let base = out[j]
                        .as_ref()
                        .map(|r| r.to_reference)
                        .unwrap_or(Similarity::IDENTITY);
                    out[k] = Some(FrameRegistration {
                        cluster,
                        mode: RegistrationMode::Registered,
                        to_reference: base.compose(&t),
                        inliers: inl,
                        rms_px: rms,
                        via: Some(j),
                    });
                    members.push(k);
                    attached = true;
                    break;
                }
            }
            if !attached {
                break;
            }
        }
        if members.len() == 1 {
            if let Some(r) = out[reference].as_mut() {
                r.mode = RegistrationMode::TextOnly;
            }
        }
        cluster += 1;
    }
    // Every keyframe is assigned: each outer iteration registers at least its
    // reference, and the loop ends only when none is left unregistered.
    debug_assert!(out.iter().all(Option::is_some));
    out.into_iter().flatten().collect()
}

/// Map a box through a similarity (bounding box of the mapped corners).
pub fn map_bbox(t: &Similarity, b: &BBox) -> BBox {
    let corners = [(b.x1, b.y1), (b.x2, b.y1), (b.x1, b.y2), (b.x2, b.y2)].map(|p| t.apply(p));
    let xs = corners.map(|p| p.0);
    let ys = corners.map(|p| p.1);
    BBox::new(
        xs.iter().copied().fold(f64::INFINITY, f64::min),
        ys.iter().copied().fold(f64::INFINITY, f64::min),
        xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn anchors(items: &[(&str, (f64, f64))], t: &Similarity) -> Anchors {
        let boxes: Vec<(String, BBox)> = items
            .iter()
            .map(|(s, c)| {
                let p = t.apply(*c);
                (
                    s.to_string(),
                    BBox::new(p.0 - 20.0, p.1 - 10.0, p.0 + 20.0, p.1 + 10.0),
                )
            })
            .collect();
        Anchors::from_items(boxes.iter().map(|(s, b)| (s.as_str(), b)))
    }

    const BOARD: [(&str, (f64, f64)); 6] = [
        ("Ingest Gateway", (100.0, 100.0)),
        ("Queue", (400.0, 120.0)),
        ("Ledger Store", (700.0, 110.0)),
        ("Report Builder", (380.0, 420.0)),
        ("Is the queue durable?", (650.0, 400.0)),
        ("Archive", (120.0, 430.0)),
    ];

    #[test]
    fn chains_a_late_zoom_through_shared_labels() {
        let id = Similarity::IDENTITY;
        let pan = Similarity {
            scale: 1.0,
            angle: 0.0,
            tx: -150.0,
            ty: 40.0,
        };
        let zoom = Similarity {
            scale: 1.6,
            angle: 0.0,
            tx: -300.0,
            ty: -120.0,
        };
        // Frame 2 sees only the right part of the board, zoomed.
        let a0 = anchors(&BOARD, &id);
        let a1 = anchors(&BOARD, &pan);
        let a2 = anchors(&BOARD[1..5], &zoom);
        // A distinct board with no shared labels.
        let other = [
            ("Alpha Panel", (100.0, 100.0)),
            ("Beta Panel", (300.0, 100.0)),
            ("Gamma Panel", (500.0, 300.0)),
        ];
        let a3 = anchors(&other, &id);
        let regs = register(
            &[a0, a1, a2, a3],
            &[1000.0; 4],
            &RegistrationParams::default(),
        );
        assert_eq!(regs[0].cluster, regs[1].cluster);
        assert_eq!(regs[0].cluster, regs[2].cluster);
        assert_eq!(regs[3].mode, RegistrationMode::TextOnly);
        assert_ne!(regs[3].cluster, regs[0].cluster);
        // Frame 2 maps back onto frame 0's coordinates (reference may be 0 or 1).
        let to0 = |i: usize, p: Point| {
            let r = regs[0].to_reference.inverse().unwrap();
            r.apply(regs[i].to_reference.apply(p))
        };
        let q = zoom.apply((400.0, 120.0));
        let back = to0(2, q);
        assert!(
            (back.0 - 400.0).abs() < 1e-6 && (back.1 - 120.0).abs() < 1e-6,
            "{back:?}"
        );
    }

    #[test]
    fn two_shared_labels_are_not_enough() {
        let id = Similarity::IDENTITY;
        let a0 = anchors(&BOARD[..2], &id);
        let a1 = anchors(&BOARD[..2], &id);
        let regs = register(&[a0, a1], &[1000.0; 2], &RegistrationParams::default());
        assert!(regs.iter().all(|r| r.mode == RegistrationMode::TextOnly));
    }

    #[test]
    fn duplicate_and_illegible_texts_are_not_anchors() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        let a = Anchors::from_items([
            ("Todo", &b),
            ("todo ", &b),
            ("[illegible]", &b),
            ("Queue", &b),
        ]);
        assert_eq!(a.points.keys().collect::<Vec<_>>(), vec!["queue"]);
    }

    #[test]
    fn identical_centers_cannot_register() {
        // Readings without geometry carry the same box for every element.
        let b = BBox::new(0.0, 0.0, 1920.0, 1080.0);
        let items = [
            ("Queue", &b),
            ("Ledger", &b),
            ("Gateway", &b),
            ("Archive", &b),
        ];
        let a0 = Anchors::from_items(items);
        let a1 = Anchors::from_items(items);
        let regs = register(&[a0, a1], &[2203.0; 2], &RegistrationParams::default());
        assert!(regs.iter().all(|r| r.mode == RegistrationMode::TextOnly));
    }

    #[test]
    fn rotated_and_scaled_views_chain_to_one_reference() {
        let t1 = Similarity {
            scale: 1.4,
            angle: 0.3,
            tx: -200.0,
            ty: 80.0,
        };
        let t2 = Similarity {
            scale: 0.7,
            angle: -0.25,
            tx: 150.0,
            ty: 210.0,
        };
        let a0 = anchors(&BOARD, &Similarity::IDENTITY);
        let a1 = anchors(&BOARD, &t1);
        let a2 = anchors(&BOARD[2..], &t2);
        let regs = register(&[a0, a1, a2], &[1000.0; 3], &RegistrationParams::default());
        assert!(regs.iter().all(|r| r.cluster == regs[0].cluster));
        let inv0 = regs[0].to_reference.inverse().unwrap();
        for (i, t) in [(1usize, t1), (2, t2)] {
            let p = t.apply((650.0, 400.0));
            let back = inv0.apply(regs[i].to_reference.apply(p));
            assert!(
                (back.0 - 650.0).abs() < 1e-6 && (back.1 - 400.0).abs() < 1e-6,
                "{i}: {back:?}"
            );
        }
    }
}
