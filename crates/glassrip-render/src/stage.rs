//! The `render` stage.

use std::path::PathBuf;

use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, Stage, StageError, StageInputs, WorkItem,
};
use glassrip_notes::board::{BoardStateItem, BOARD_STATE_MAJOR};
use glassrip_notes::notes::MeetingNotes;
use glassrip_notes::schemas;
use schemars::JsonSchema;
use semver::Version;
use serde::{Deserialize, Serialize};

use crate::markdown::MarkdownMeta;
use crate::svg::FontConfig;
use crate::{render_all, RenderResult};

/// Schema of the render artifact.
pub const RENDER_SCHEMA: &str = "glassrip.render";

/// Parameters of `render`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RenderParams {
    /// Output directory for the markdown, SVG and PNG files.
    pub out_dir: PathBuf,
    /// File name stem (usually the video stem).
    pub stem: String,
    /// Header facts.
    pub meta: MarkdownMeta,
    /// Fail the item when a validation check fails (default: true).
    pub strict: bool,
    /// Fonts for the validation render.
    pub fonts: FontConfig,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("out"),
            stem: "meeting".into(),
            meta: MarkdownMeta::default(),
            strict: true,
            fonts: FontConfig::default(),
        }
    }
}

/// `render`: meeting notes and board state to markdown and SVG.
pub struct RenderStage {
    params: RenderParams,
}

impl RenderStage {
    /// A stage writing into `params.out_dir`.
    pub fn new(params: RenderParams) -> Self {
        Self { params }
    }
}

/// Inputs gathered by `plan`.
#[derive(Debug)]
pub struct RenderInput {
    notes: MeetingNotes,
    boards: Vec<BoardStateItem>,
}

impl Stage for RenderStage {
    type Params = RenderParams;
    type Work = Box<RenderInput>;
    type Output = RenderResult;

    fn name(&self) -> &'static str {
        "render"
    }
    fn version(&self) -> u32 {
        // 2: edge labels avoid zone titles, slide along their path, and wrap.
        // 3: changed markdown and scene output (and never restored from the
        // cache, see `cacheable`).
        3
    }
    /// The markdown, SVG and PNG files are the stage's output and live outside
    /// the run directory: a cache hit would restore the artifact without
    /// rewriting them (old or deleted files reported as current), so the stage
    /// renders whenever it is selected.
    fn cacheable(&self) -> bool {
        false
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: RENDER_SCHEMA,
            version: Version::new(1, 0, 0),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![
            InputDecl {
                schema: schemas::MEETING_NOTES,
                major: 1,
            },
            InputDecl {
                schema: schemas::BOARD_STATE,
                major: BOARD_STATE_MAJOR,
            },
        ]
    }
    fn params(&self) -> &RenderParams {
        &self.params
    }

    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<Self::Work>>, StageError> {
        let notes = inputs
            .read_ok::<MeetingNotes>(schemas::MEETING_NOTES)?
            .into_iter()
            .map(|(_, n)| n)
            .next()
            .ok_or_else(|| StageError::Invalid("no meeting notes".into()))?;
        let boards = inputs
            .read_ok::<BoardStateItem>(schemas::BOARD_STATE)?
            .into_iter()
            .map(|(_, b)| b)
            .collect();
        Ok(vec![WorkItem {
            id: "render".into(),
            work: Box::new(RenderInput { notes, boards }),
        }])
    }

    async fn process(
        &self,
        _ctx: &ItemContext,
        work: Self::Work,
    ) -> Result<RenderResult, ErrorInfo> {
        let p = &self.params;
        let result = render_all(&work.notes, &work.boards, p)
            .map_err(|e| ErrorInfo::new(ErrorCode::Io, e.to_string()))?;
        if p.strict && !result.ok {
            let detail =
                serde_json::to_string(&(&result.markdown, &result.svg)).unwrap_or_default();
            return Err(
                ErrorInfo::new(ErrorCode::Validation, "render validation failed")
                    .with_raw_text(detail),
            );
        }
        Ok(result)
    }
}
