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

/// Failures listed in a validation error message before the rest are counted.
pub const MAX_LISTED_FAILURES: usize = 12;

/// The validation error message: every failed check (the first
/// [`MAX_LISTED_FAILURES`], then a count), so the run log says what failed.
pub fn failure_message(failures: &[String]) -> String {
    let mut msg = String::from("render validation failed");
    if failures.is_empty() {
        return msg;
    }
    msg.push_str(": ");
    msg.push_str(
        &failures
            .iter()
            .take(MAX_LISTED_FAILURES)
            .cloned()
            .collect::<Vec<_>>()
            .join("; "),
    );
    if failures.len() > MAX_LISTED_FAILURES {
        msg.push_str(&format!(
            "; and {} more",
            failures.len() - MAX_LISTED_FAILURES
        ));
    }
    msg
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
        // 4: edge owners slide along their edge; validation errors list the
        // failed checks.
        // 5: an annotation with no room on the board (owner, note, badge, edge
        // label) is listed below it with a numbered marker and a warning
        // instead of failing validation; more spots and leader lines first.
        // 6: edges are routed orthogonally around the cards; labels sit beside
        // a straight segment of their edge; history notes sit next to their
        // owner pill; validation fails an edge through a card or a label an
        // edge runs through. Text is sized and fitted by its measured glyph
        // widths, and any glyph past the canvas fails validation.
        // 7: an edge the grid cannot route takes a cheap detour (L, Z, or around
        // the architecture) clear of the cards before falling back to a straight
        // line, and validation fails a fallback edge through another card
        // (outputs of 6 may report ok with a straight edge over a card).
        7
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
                ErrorInfo::new(ErrorCode::Validation, failure_message(&result.failures()))
                    .with_raw_text(detail),
            );
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_message_lists_checks_and_counts_the_rest() {
        assert_eq!(failure_message(&[]), "render validation failed");
        let one = failure_message(&["svg b1: layout: card a overlaps card b".into()]);
        assert_eq!(
            one,
            "render validation failed: svg b1: layout: card a overlaps card b"
        );
        let many: Vec<String> = (0..MAX_LISTED_FAILURES + 3)
            .map(|i| format!("f{i}"))
            .collect();
        let m = failure_message(&many);
        assert!(m.contains("f0; f1") && m.ends_with("; and 3 more"), "{m}");
        assert!(!m.contains(&format!("f{MAX_LISTED_FAILURES}")));
    }
}
