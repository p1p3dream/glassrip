//! `features` (spec 6.5): per-frame sharpness and pair scores against the previous frame.
//!
//! One item per frame. Each item computes its own frame's features and the previous
//! frame's (recomputed rather than shared, so items stay independent and resumable), then
//! scores the pair (previous frame as ECC template) in the configured mode.

use std::path::PathBuf;

use glassrip_core::envelope::ErrorInfo;
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, KeyExtras, Stage, StageError, StageInputs, WorkItem,
};

use crate::schema::{FEATURES, FRAMES, FrameFeatures, FrameRecord, PairToPrev, v1};
use crate::scoring::{Scorer, ScoringParams};

/// Per-item work.
#[derive(Debug, Clone)]
pub struct FeatureWork {
    index: u64,
    cur: FrameRecord,
    prev: Option<FrameRecord>,
}

/// The `features` stage.
#[derive(Debug, Clone)]
pub struct FeaturesStage {
    params: ScoringParams,
    scorer: Scorer,
    run_root: PathBuf,
}

impl FeaturesStage {
    /// Stage reading frames under `run_root`.
    pub fn new(params: ScoringParams, run_root: PathBuf) -> Result<Self, String> {
        Ok(Self {
            scorer: Scorer::new(&params)?,
            params,
            run_root,
        })
    }
}

/// Frames sorted by PTS (ties by id), failing on duplicate PTS.
pub fn ordered_frames(mut frames: Vec<FrameRecord>) -> Result<Vec<FrameRecord>, StageError> {
    frames.sort_by(|a, b| a.pts.cmp(&b.pts).then_with(|| a.frame_id.cmp(&b.frame_id)));
    if let Some(w) = frames.windows(2).find(|w| w[0].pts == w[1].pts) {
        return Err(StageError::Invalid(format!(
            "frames {} and {} share pts {}",
            w[0].frame_id, w[1].frame_id, w[0].pts
        )));
    }
    Ok(frames)
}

impl Stage for FeaturesStage {
    type Params = ScoringParams;
    type Work = FeatureWork;
    type Output = FrameFeatures;

    fn name(&self) -> &'static str {
        "features"
    }
    fn version(&self) -> u32 {
        2
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: FEATURES,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        // Features run on unrectified frames exactly as the prototype did (spec 6.4).
        vec![InputDecl {
            schema: FRAMES,
            major: 1,
        }]
    }
    fn params(&self) -> &ScoringParams {
        &self.params
    }
    fn key_extras(&self) -> KeyExtras {
        KeyExtras {
            features_mode: Some(self.params.mode_name().into()),
            ..KeyExtras::default()
        }
    }
    fn concurrency(&self) -> usize {
        crate::util::cpus() * 2
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<FeatureWork>>, StageError> {
        let frames = ordered_frames(
            inputs
                .read_ok::<FrameRecord>(FRAMES)?
                .into_iter()
                .map(|(_, f)| f)
                .collect(),
        )?;
        let mut items = Vec::with_capacity(frames.len());
        let mut prev: Option<FrameRecord> = None;
        for (i, f) in frames.into_iter().enumerate() {
            items.push(WorkItem {
                id: f.frame_id.clone(),
                work: FeatureWork {
                    index: i as u64,
                    cur: f.clone(),
                    prev: prev.replace(f),
                },
            });
        }
        Ok(items)
    }
    async fn process(
        &self,
        _ctx: &ItemContext,
        w: FeatureWork,
    ) -> Result<FrameFeatures, ErrorInfo> {
        let (scorer, root) = (self.scorer.clone(), self.run_root.clone());
        crate::util::on_rayon(move || {
            let load = |f: &FrameRecord| scorer.features(&root.join(&f.path), Some(&f.blake3));
            let (cur, prev) = rayon::join(|| load(&w.cur), || w.prev.as_ref().map(load));
            let cur = cur?;
            let prev_scores = match (prev, &w.prev) {
                (Some(p), Some(pf)) => {
                    let p = p?;
                    let s = scorer.pair(&p, &cur);
                    Some(PairToPrev {
                        prev_frame_id: pf.frame_id.clone(),
                        ssim: s.ssim,
                        changed_frac: s.changed_frac,
                        ink_change: s.ink_change,
                        align_ok: s.align_ok,
                        align_method: s.align_method,
                        ink_align_ok: s.ink_align_ok,
                        shift_px: s.shift,
                        valid_frac: s.valid_frac,
                    })
                }
                _ => None,
            };
            Ok(FrameFeatures {
                frame_id: w.cur.frame_id.clone(),
                index: w.index,
                pts_s: w.cur.pts_s,
                frame_path: w.cur.path.clone(),
                frame_blake3: w.cur.blake3.clone(),
                sharpness_lapvar: cur.sharpness,
                prev: prev_scores,
            })
        })
        .await
    }
}
