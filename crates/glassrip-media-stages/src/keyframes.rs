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
        match (self.get(a), self.get(b)) {
            (Some(x), Some(y)) => self.scorer.ink(&x, &y).0,
            _ => 1.0,
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
            cache.seed(i - 1, i, p.ssim, p.changed_frac, p.ink_change, p.align_ok);
        }
        let sharp: Vec<f64> = frames.iter().map(|f| f.sharpness_lapvar).collect();
        let times: Vec<f64> = frames.iter().map(|f| f.pts_s).collect();
        let (mut runs, bounds) = segment(n, &cache, self.segment_batch);
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
        if let Some(e) = oracle
            .error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            return Err(StageError::Invalid(format!("cannot score frames: {e}")));
        }
        let end = end_s.max(times[n - 1]);
        let kfs = segment::keyframes(&runs, &times, &sharp, end, &bounds);
        let production = self.scorer.is_production();
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
                    },
                    _ => Boundary {
                        reason: BoundaryReason::Start,
                        anchor_frame_id: None,
                        ssim: None,
                        changed_frac: None,
                        ink_change: None,
                        align_ok: None,
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

impl Stage for KeyframesStage {
    type Params = KeyframesParams;
    type Work = Keyframe;
    type Output = Keyframe;

    fn name(&self) -> &'static str {
        "keyframes"
    }
    fn version(&self) -> u32 {
        1
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
