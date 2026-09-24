//! Run the vision-branch stages in order on a [`Runner`].

use glassrip_core::graph::{meeting_mode_stage_decls, GraphError, StageGraph};
use glassrip_core::runner::{Runner, RunnerError, StageReport};

use crate::stages::board_read::BoardReadStage;
use crate::stages::board_validate::BoardValidateStage;
use crate::stages::canvas_crop::CanvasCropStage;
use crate::stages::classify::ClassifyStage;
use crate::stages::ocr_harvest::OcrHarvestStage;
use crate::stages::vocabulary::VocabularyStage;

/// Stage names in run order.
pub const STAGES: [&str; 6] = [
    "ocr_harvest",
    "ocr_vocabulary",
    "classify",
    "canvas_crop",
    "board_read",
    "board_validate",
];

/// The meeting-mode stage graph.
pub fn graph() -> Result<StageGraph, GraphError> {
    StageGraph::new(meeting_mode_stage_decls())
}

/// The six vision-branch stages.
pub struct VisionBranch {
    pub ocr: OcrHarvestStage,
    pub vocabulary: VocabularyStage,
    pub classify: ClassifyStage,
    pub canvas: CanvasCropStage,
    pub board_read: BoardReadStage,
    pub board_validate: BoardValidateStage,
}

impl VisionBranch {
    /// Run every stage up to and including `until` (all when `None`).
    pub async fn run(
        &self,
        runner: &mut Runner,
        until: Option<&str>,
    ) -> Result<Vec<StageReport>, RunnerError> {
        let mut reports = Vec::new();
        for name in STAGES {
            let report = match name {
                "ocr_harvest" => runner.run_stage(&self.ocr).await?,
                "ocr_vocabulary" => runner.run_stage(&self.vocabulary).await?,
                "classify" => runner.run_stage(&self.classify).await?,
                "canvas_crop" => runner.run_stage(&self.canvas).await?,
                "board_read" => runner.run_stage(&self.board_read).await?,
                _ => runner.run_stage(&self.board_validate).await?,
            };
            tracing::info!(
                stage = name,
                status = ?report.status,
                ok = report.items_ok,
                errors = report.items_error,
                wall_s = report.wall_s,
                "stage done"
            );
            reports.push(report);
            if until == Some(name) {
                break;
            }
        }
        Ok(reports)
    }
}
