//! `keyframes` (spec 6.6): segmentation into static runs plus singleton merge.
//!
//! Segmentation is global, so it runs in [`Stage::plan`]: consecutive-pair scores come from
//! the `features` artifact; anchor comparisons and merge scores are computed on demand with
//! the same scorer, loading per-frame features lazily through a bounded LRU (the frame
//! paths and hashes travel in the `features` items). Each resulting keyframe is one work
//! item. The segmentation and merge rules are `glassrip-media`'s exact ports.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use glassrip_core::config::KeyframesConfig;
use glassrip_core::envelope::ErrorInfo;
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, KeyExtras, Stage, StageError, StageInputs, WorkItem,
};
use glassrip_media::features::FrameFeatures as MediaFeatures;
use glassrip_media::segment::{
    self, DiffCache, MergeParams, PairOracle, Reason, Thresholds, merge_singletons, segment,
};
use schemars::JsonSchema;
use serde::Serialize;

use crate::schema::{
    Boundary, BoundaryReason, FEATURES, FrameFeatures, KEYFRAMES, Keyframe, MEDIA_PROBE,
    MediaProbe, MergedSingleton, v1,
};
use crate::scoring::{Scorer, ScoringParams};

/// Segmentation parameters.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct KeyframesParams {
    /// `differs` SSIM threshold.
    pub ssim_threshold: f64,
    /// `differs` changed-fraction threshold.
    pub changed_frac_threshold: f64,
    /// `differs` ink threshold.
    pub ink_threshold: f64,
    /// Persistence in samples (only 2 is implemented, as in the prototype).
    pub persistence: u32,
    /// Merge: minimum SSIM against the neighbor representative.
    pub merge_min_ssim: f64,
    /// Merge: maximum changed fraction.
    pub merge_max_changed_frac: f64,
    /// Merge: blur ratio against the median sharpness.
    pub merge_blur_ratio: f64,
    /// `production`: representative sharpness tolerance (see [`central_representative`]).
    pub rep_sharpness_tolerance: f64,
    /// Scoring (must match `features`).
    pub scoring: ScoringParams,
}

impl KeyframesParams {
    /// From config sections.
    pub fn from_config(k: &KeyframesConfig, scoring: ScoringParams) -> Self {
        Self {
            ssim_threshold: k.ssim_threshold,
            changed_frac_threshold: k.changed_frac_threshold,
            ink_threshold: k.ink_threshold,
            persistence: k.persistence,
            merge_min_ssim: k.merge_min_ssim,
            merge_max_changed_frac: k.merge_max_changed_frac,
            merge_blur_ratio: k.merge_blur_ratio,
            rep_sharpness_tolerance: k.production_rep_sharpness_tolerance,
            scoring,
        }
    }
}

/// The `keyframes` stage.
#[derive(Debug, Clone)]
pub struct KeyframesStage {
    params: KeyframesParams,
    scorer: Scorer,
    run_root: PathBuf,
    /// Upcoming frames scored speculatively in parallel (speed only).
    pub segment_batch: usize,
    /// Frames whose features are kept in memory (speed only).
    pub lru_frames: usize,
}

impl KeyframesStage {
    /// Stage reading frames under `run_root`.
    pub fn new(params: KeyframesParams, run_root: PathBuf) -> Result<Self, String> {
        if params.persistence != 2 {
            return Err(format!(
                "keyframes.persistence = {} is not supported; the ported rule uses 2 samples",
                params.persistence
            ));
        }
        Ok(Self {
            scorer: Scorer::new(&params.scoring)?,
            params,
            run_root,
            segment_batch: crate::util::cpus().clamp(1, 16),
            lru_frames: 96,
        })
    }
}

/// LRU state: a tick counter and `index -> (features, last use)`.
type Lru = (u64, HashMap<usize, (Arc<MediaFeatures>, u64)>);

/// Lazily loaded per-frame features with a bounded LRU.
struct LazyOracle<'a> {
    scorer: &'a Scorer,
    frames: &'a [FrameFeatures],
    root: &'a std::path::Path,
    cap: usize,
    cache: Mutex<Lru>,
    error: Mutex<Option<ErrorInfo>>,
}

impl LazyOracle<'_> {
    fn get(&self, i: usize) -> Option<Arc<MediaFeatures>> {
        {
            let mut c = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
            c.0 += 1;
            let tick = c.0;
            if let Some(e) = c.1.get_mut(&i) {
                e.1 = tick;
                return Some(Arc::clone(&e.0));
            }
        }
        let f = &self.frames[i];
        match self
            .scorer
            .features(&self.root.join(&f.frame_path), Some(&f.frame_blake3))
        {
            Ok(feat) => {
                let feat = Arc::new(feat);
                let mut c = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
                c.0 += 1;
                let tick = c.0;
                c.1.insert(i, (Arc::clone(&feat), tick));
                while c.1.len() > self.cap {
                    let Some(old) = c.1.iter().min_by_key(|(_, v)| v.1).map(|(k, _)| *k) else {
                        break;
                    };
                    c.1.remove(&old);
                }
                Some(feat)
            }
            Err(e) => {
                self.error
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .get_or_insert(e);
                None
            }
        }
    }
}

impl PairOracle for LazyOracle<'_> {
    fn score(&self, a: usize, b: usize) -> (f64, f64, bool) {
        match (self.get(a), self.get(b)) {
            (Some(x), Some(y)) => {
                let s = self.scorer.score(&x, &y);
                (s.ssim, s.changed_frac, s.align_ok)
            }
            _ => (0.0, 1.0, false),
        }
    }
    fn ink(&self, a: usize, b: usize) -> f64 {
        self.ink_checked(a, b).0
    }
    fn ink_checked(&self, a: usize, b: usize) -> (f64, bool) {
        match (self.get(a), self.get(b)) {
            (Some(x), Some(y)) => self.scorer.ink(&x, &y),
            _ => (1.0, false),
        }
    }
}

fn reason(c: &segment::Comparison, production: bool) -> BoundaryReason {
    if production && !c.align_ok {
        return BoundaryReason::AlignFailed;
    }
    match c.reason {
        Some(Reason::Ssim) => BoundaryReason::Ssim,
        Some(Reason::Frac) => BoundaryReason::Frac,
        Some(Reason::Ink) => BoundaryReason::Ink,
        None => BoundaryReason::Start,
    }
}

impl KeyframesStage {
    /// Runs segmentation and merge over ordered features. `end_s` is the last keyframe's end.
    pub fn build(&self, frames: &[FrameFeatures], end_s: f64) -> Result<Vec<Keyframe>, StageError> {
        let n = frames.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        for (i, f) in frames.iter().enumerate() {
            if f.index != i as u64 {
                return Err(StageError::Invalid(format!(
                    "features are not contiguous: {} has index {} at position {i}",
                    f.frame_id, f.index
                )));
            }
        }
        let oracle = LazyOracle {
            scorer: &self.scorer,
            frames,
            root: &self.run_root,
            cap: self.lru_frames.max(8),
            cache: Mutex::new((0, HashMap::new())),
            error: Mutex::new(None),
        };
        let thr = Thresholds {
            ssim: self.params.ssim_threshold,
            frac: self.params.changed_frac_threshold,
            ink: self.params.ink_threshold,
        };
        let cache = DiffCache::new(&oracle, thr);
        for (i, f) in frames.iter().enumerate().skip(1) {
            let Some(p) = &f.prev else {
                return Err(StageError::Invalid(format!(
                    "{} has no previous-frame scores",
                    f.frame_id
                )));
            };
            if p.prev_frame_id != frames[i - 1].frame_id {
                return Err(StageError::Invalid(format!(
                    "{} was scored against {}, expected {}",
                    f.frame_id,
                    p.prev_frame_id,
                    frames[i - 1].frame_id
                )));
            }
            cache.seed_checked(
                i - 1,
                i,
                p.ssim,
                p.changed_frac,
                (p.ink_change, p.ink_align_ok),
                p.align_ok,
            );
        }
        let sharp: Vec<f64> = frames.iter().map(|f| f.sharpness_lapvar).collect();
        let times: Vec<f64> = frames.iter().map(|f| f.pts_s).collect();
        let (runs, mut bounds) = segment(n, &cache, self.segment_batch);
        let production = self.scorer.is_production();
        // Production only: one-sample states that differ from both neighbors become their
        // own runs (the persistence rule cannot start a run for them); the unchanged merge
        // below re-absorbs noise, and a host run split by an island that merged back is
        // rejoined afterwards.
        let islands: Vec<usize> = if production {
            (1..n.saturating_sub(1))
                .filter(|&i| cache.get(i - 1, i).differs() && cache.get(i, i + 1).differs())
                .collect()
        } else {
            Vec::new()
        };
        let (mut runs, origin) = split_islands(runs, &islands, n);
        let merges = merge_singletons(
            &mut runs,
            &sharp,
            MergeParams {
                ssim: self.params.merge_min_ssim,
                frac: self.params.merge_max_changed_frac,
                blur: self.params.merge_blur_ratio,
            },
            &cache,
        );
        let runs = if islands.is_empty() {
            runs
        } else {
            rejoin_islands(runs, &origin, &islands)
        };
        // Runs created by island splits start without a segmentation boundary: record the
        // consecutive comparison that separates them.
        let known: std::collections::HashSet<usize> = bounds.iter().map(|b| b.frame).collect();
        for r in runs.iter().skip(1) {
            let first = r[0];
            if first > 0 && !known.contains(&first) {
                bounds.push(segment::Boundary {
                    frame: first,
                    anchor: first - 1,
                    comparison: cache.get(first - 1, first),
                });
            }
        }
        if let Some(e) = oracle
            .error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            return Err(StageError::Invalid(format!("cannot score frames: {e}")));
        }
        let end = end_s.max(times[n - 1]);
        let mut kfs = segment::keyframes(&runs, &times, &sharp, end, &bounds);
        if production {
            let merged_frames: std::collections::HashSet<usize> =
                merges.iter().map(|m| m.frame).collect();
            for kf in &mut kfs {
                let r = central_representative(
                    &kf.frames,
                    &sharp,
                    &times,
                    &merged_frames,
                    self.params.rep_sharpness_tolerance,
                );
                kf.rep_frame = r;
                kf.t_rep = times[r];
                kf.sharpness = sharp[r];
            }
        }
        let run_of: HashMap<usize, usize> = runs
            .iter()
            .enumerate()
            .flat_map(|(k, r)| r.iter().map(move |&i| (i, k)))
            .collect();
        let mut merged: HashMap<usize, Vec<MergedSingleton>> = HashMap::new();
        for m in &merges {
            if let Some(&k) = run_of.get(&m.frame) {
                merged.entry(k).or_default().push(MergedSingleton {
                    frame_id: frames[m.frame].frame_id.clone(),
                    ssim: m.ssim,
                    changed_frac: m.frac,
                    blurry: m.blurry,
                });
            }
        }
        Ok(kfs
            .into_iter()
            .enumerate()
            .map(|(k, kf)| {
                let boundary = match (k, kf.boundary) {
                    (_, Some(b)) => Boundary {
                        reason: reason(&b.comparison, production),
                        anchor_frame_id: Some(frames[b.anchor].frame_id.clone()),
                        ssim: Some(b.comparison.ssim),
                        changed_frac: Some(b.comparison.frac),
                        ink_change: b.comparison.ink,
                        align_ok: Some(b.comparison.align_ok),
                        ink_align_ok: b.comparison.ink.map(|_| b.comparison.ink_align_ok),
                    },
                    _ => Boundary {
                        reason: BoundaryReason::Start,
                        anchor_frame_id: None,
                        ssim: None,
                        changed_frac: None,
                        ink_change: None,
                        align_ok: None,
                        ink_align_ok: None,
                    },
                };
                Keyframe {
                    keyframe_id: format!("k{k:04}"),
                    index: k as u32,
                    rep_frame_id: frames[kf.rep_frame].frame_id.clone(),
                    frame_ids: kf
                        .frames
                        .iter()
                        .map(|&i| frames[i].frame_id.clone())
                        .collect(),
                    t_start_s: kf.t_start,
                    t_end_s: kf.t_end,
                    t_rep_s: kf.t_rep,
                    n_frames: kf.frames.len() as u32,
                    sharpness_lapvar: kf.sharpness,
                    boundary,
                    merged: merged.remove(&k).unwrap_or_default(),
                }
            })
            .collect())
    }
}

/// Production representative: among the run's own frames (merged singletons excluded
/// unless nothing else is left) whose sharpness is within `tol` of the best, the one
/// nearest the midpoint of those frames' time span (ties: sharper, then earlier). Avoids
/// picking a transition frame at a run edge or a merged singleton from a neighboring state.
pub fn central_representative(
    run: &[usize],
    sharp: &[f64],
    times: &[f64],
    merged: &std::collections::HashSet<usize>,
    tol: f64,
) -> usize {
    let own: Vec<usize> = run
        .iter()
        .copied()
        .filter(|i| !merged.contains(i))
        .collect();
    let pool = if own.is_empty() { run.to_vec() } else { own };
    let best = pool
        .iter()
        .map(|&i| sharp[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let lo = pool.iter().map(|&i| times[i]).fold(f64::INFINITY, f64::min);
    let hi = pool
        .iter()
        .map(|&i| times[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let mid = (lo + hi) / 2.0;
    let mut cands: Vec<usize> = pool
        .into_iter()
        .filter(|&i| sharp[i] >= (1.0 - tol) * best)
        .collect();
    cands.sort_by(|&a, &b| {
        (times[a] - mid)
            .abs()
            .total_cmp(&(times[b] - mid).abs())
            .then(sharp[b].total_cmp(&sharp[a]))
            .then(a.cmp(&b))
    });
    cands.first().copied().unwrap_or(run[0])
}

/// Splits each island frame out of its run into a run of its own. Returns the runs (in
/// frame order) and each frame's original run index.
pub fn split_islands(
    runs: Vec<Vec<usize>>,
    islands: &[usize],
    n: usize,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut origin = vec![0usize; n];
    for (r, run) in runs.iter().enumerate() {
        for &i in run {
            origin[i] = r;
        }
    }
    let is_island: std::collections::HashSet<usize> = islands.iter().copied().collect();
    let mut out = Vec::with_capacity(runs.len() + islands.len());
    for run in runs {
        if run.len() == 1 {
            out.push(run);
            continue;
        }
        let mut cur = Vec::new();
        for i in run {
            if is_island.contains(&i) {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                out.push(vec![i]);
            } else {
                cur.push(i);
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    (out, origin)
}

/// After the singleton merge: joins adjacent runs that came from the same original run,
/// unless one of them is a surviving island.
pub fn rejoin_islands(
    runs: Vec<Vec<usize>>,
    origin: &[usize],
    islands: &[usize],
) -> Vec<Vec<usize>> {
    let is_island: std::collections::HashSet<usize> = islands.iter().copied().collect();
    let surviving = |r: &Vec<usize>| r.len() == 1 && is_island.contains(&r[0]);
    let same_origin = |r: &Vec<usize>| -> Option<usize> {
        let o = origin[r[0]];
        r.iter().all(|&i| origin[i] == o).then_some(o)
    };
    let mut out: Vec<Vec<usize>> = Vec::with_capacity(runs.len());
    for run in runs {
        if let Some(prev) = out.last_mut() {
            let joinable = !surviving(prev)
                && !surviving(&run)
                && same_origin(prev).is_some()
                && same_origin(prev) == same_origin(&run);
            if joinable {
                prev.extend(run);
                prev.sort_unstable();
                continue;
            }
        }
        out.push(run);
    }
    out
}

impl Stage for KeyframesStage {
    type Params = KeyframesParams;
    type Work = Keyframe;
    type Output = Keyframe;

    fn name(&self) -> &'static str {
        "keyframes"
    }
    fn version(&self) -> u32 {
        // 4: boundaries record whether their ink alignment was usable.
        // 5: two flat frames align as the identity and measure no change.
        5
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: KEYFRAMES,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![
            InputDecl {
                schema: FEATURES,
                major: 1,
            },
            InputDecl {
                schema: MEDIA_PROBE,
                major: 1,
            },
        ]
    }
    fn params(&self) -> &KeyframesParams {
        &self.params
    }
    fn key_extras(&self) -> KeyExtras {
        KeyExtras {
            features_mode: Some(self.params.scoring.mode_name().into()),
            ..KeyExtras::default()
        }
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<Keyframe>>, StageError> {
        let probe = inputs
            .read_ok::<MediaProbe>(MEDIA_PROBE)?
            .into_iter()
            .next()
            .ok_or_else(|| StageError::Invalid("media_probe has no ok item".into()))?
            .1;
        let mut frames: Vec<FrameFeatures> = inputs
            .read_ok::<FrameFeatures>(FEATURES)?
            .into_iter()
            .map(|(_, f)| f)
            .collect();
        frames.sort_by_key(|f| f.index);
        // Plan is synchronous; segmentation parallelizes internally on rayon.
        let kfs = self.build(&frames, probe.end_s)?;
        Ok(kfs
            .into_iter()
            .map(|k| WorkItem {
                id: k.keyframe_id.clone(),
                work: k,
            })
            .collect())
    }
    async fn process(&self, _ctx: &ItemContext, k: Keyframe) -> Result<Keyframe, ErrorInfo> {
        Ok(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn island_is_split_out_and_rejoined_when_merged_back() {
        // Run 0..5 hosts island 2; run 5..7 is separate.
        let runs = vec![vec![0, 1, 2, 3, 4], vec![5, 6]];
        let (split, origin) = split_islands(runs, &[2], 7);
        assert_eq!(split, vec![vec![0, 1], vec![2], vec![3, 4], vec![5, 6]]);
        // Island survives the merge: it stays a keyframe between the two halves.
        assert_eq!(
            rejoin_islands(split.clone(), &origin, &[2]),
            vec![vec![0, 1], vec![2], vec![3, 4], vec![5, 6]]
        );
        // Island merged into the left half: the host run is whole again, the next run
        // (different origin) is untouched.
        let merged = vec![vec![0, 1, 2], vec![3, 4], vec![5, 6]];
        assert_eq!(
            rejoin_islands(merged, &origin, &[2]),
            vec![vec![0, 1, 2, 3, 4], vec![5, 6]]
        );
    }

    #[test]
    fn representative_is_central_sharp_and_not_merged() {
        let times: Vec<f64> = (0..6).map(|i| 2.0 * i as f64).collect();
        // Frame 5 is the sharpest but a merged singleton; frames 0..5 are near-equal.
        let sharp = [257.0, 240.0, 253.0, 255.0, 100.0, 300.0];
        let merged = [5usize].into_iter().collect();
        let r = central_representative(&[0, 1, 2, 3, 4, 5], &sharp, &times, &merged, 0.05);
        // Own frames 0..4 span 0..8 s (mid 4 s); within 5% of 257: frames 0, 2, 3.
        assert_eq!(r, 2);
        // Only merged frames left: they are used.
        assert_eq!(
            central_representative(&[5], &sharp, &times, &merged, 0.05),
            5
        );
        // Zero tolerance is the sharpest own frame.
        assert_eq!(
            central_representative(&[0, 1, 2, 3, 4, 5], &sharp, &times, &merged, 0.0),
            0
        );
    }

    #[test]
    fn island_at_run_start_and_singletons() {
        let runs = vec![vec![0, 1], vec![2, 3, 4], vec![5]];
        let (split, origin) = split_islands(runs, &[2, 5], 6);
        assert_eq!(split, vec![vec![0, 1], vec![2], vec![3, 4], vec![5]]);
        // A surviving island is never joined with its origin neighbor.
        assert_eq!(
            rejoin_islands(split, &origin, &[2, 5]),
            vec![vec![0, 1], vec![2], vec![3, 4], vec![5]]
        );
    }
}
