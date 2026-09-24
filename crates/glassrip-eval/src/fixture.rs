//! Public fixture model (spec 9.1).
//!
//! ```text
//! tests/fixtures/synthetic/<case>/{frame.png, expected.json, meta.toml}
//! tests/fixtures/synthetic_docs/<case>/{frames/*.png, expected.json, meta.toml}
//! ```
//!
//! The private document suite uses the same layout under
//! `<eval.private_fixtures>/golden/docs/<case>/`. Every case is validated when
//! loaded: `meta.case` equals the directory name, ids are unique, and every
//! reference resolves.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use glassrip_vision::BBox;
use serde::{Deserialize, Serialize};

use crate::error::{read_json, read_toml, EvalError, Result};
use crate::metrics::board::GoldBoard;
use crate::metrics::docs::Document;
use crate::metrics::screen::ScreenType;

/// Schema name of a synthetic board `expected.json`.
pub const BOARD_EXPECTED_SCHEMA: &str = "glassrip.eval.synthetic_board";
/// Schema name of a synthetic document `expected.json`.
pub const DOCS_EXPECTED_SCHEMA: &str = "glassrip.eval.synthetic_document";
/// Major version understood by this crate.
pub const EXPECTED_MAJOR: &str = "1.";

/// `meta.toml` of one case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseMeta {
    /// Case name (equals the directory name).
    pub case: String,
    /// Suite (`synthetic` or `synthetic_docs`, or a private suite name).
    pub suite: String,
    /// What the case exercises.
    pub description: String,
    /// Generator that wrote the case (`gen_synthetic` for public fixtures).
    pub generator: String,
    /// Generator version.
    pub generator_version: u32,
    /// Generator seed.
    pub seed: u64,
    /// Gold screen type.
    pub screen_type: ScreenType,
    /// Feature tags (for example `dashed_edge`, `owner_tags`, `sticky_header`).
    #[serde(default)]
    pub tags: Vec<String>,
}

/// `expected.json` of a board or screen case (meeting mode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoardExpected {
    /// [`BOARD_EXPECTED_SCHEMA`].
    pub schema: String,
    /// Semver, major 1.
    pub schema_version: String,
    /// Gold screen type.
    pub screen_type: ScreenType,
    /// Frame width, pixels.
    pub frame_width: u32,
    /// Frame height, pixels.
    pub frame_height: u32,
    /// Whiteboard canvas in frame pixels (whiteboards only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canvas_bbox: Option<BBox>,
    /// Board content in frame pixels (empty for non-whiteboards).
    pub board: GoldBoard,
    /// UI strings drawn outside the canvas; any element reading one is a chrome false positive.
    pub chrome_texts: Vec<String>,
    /// Fictional participants (names on video tiles and owner tags).
    pub participants: Vec<String>,
}

/// One frame of a scrolled document fixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameRef {
    /// Path relative to the case directory.
    pub file: String,
    /// Page offset of the frame's content top, pixels.
    pub scroll_y: u32,
}

/// `expected.json` of a document case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocsExpected {
    /// [`DOCS_EXPECTED_SCHEMA`].
    pub schema: String,
    /// Semver, major 1.
    pub schema_version: String,
    /// Gold screen type (`ticket` or `doc`).
    pub screen_type: ScreenType,
    /// Gold page content.
    pub document: Document,
    /// Full page height, pixels.
    pub page_height_px: u32,
    /// Viewport width, pixels.
    pub viewport_width: u32,
    /// Viewport height, pixels.
    pub viewport_height: u32,
    /// Height of the fixed header drawn on every frame, pixels.
    pub fixed_header_px: u32,
    /// Frames in capture order.
    pub frames: Vec<FrameRef>,
}

/// A loaded board or screen case.
#[derive(Debug, Clone)]
pub struct BoardCase {
    /// Case name.
    pub name: String,
    /// Case directory.
    pub dir: PathBuf,
    /// Metadata.
    pub meta: CaseMeta,
    /// Expected output.
    pub expected: BoardExpected,
}

impl BoardCase {
    /// Path of `frame.png`.
    pub fn frame_path(&self) -> PathBuf {
        self.dir.join("frame.png")
    }
}

/// A loaded document case.
#[derive(Debug, Clone)]
pub struct DocsCase {
    /// Case name.
    pub name: String,
    /// Case directory.
    pub dir: PathBuf,
    /// Metadata.
    pub meta: CaseMeta,
    /// Expected output.
    pub expected: DocsExpected,
}

fn fixture_err(case: &str, message: impl Into<String>) -> EvalError {
    EvalError::Fixture {
        case: case.to_string(),
        message: message.into(),
    }
}

/// Case directories under `root`, sorted by name (hidden entries skipped).
pub fn case_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    let entries = fs_err::read_dir(root).map_err(|e| EvalError::io(root, e))?;
    let mut dirs = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| EvalError::io(root, e))?;
        let path = entry.path();
        let hidden = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with('.'));
        if path.is_dir() && !hidden {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}

fn dir_name(dir: &Path) -> String {
    dir.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn check_meta(name: &str, meta: &CaseMeta) -> Result<()> {
    if meta.case != name {
        return Err(fixture_err(
            name,
            format!("meta.case is `{}` but the directory is `{name}`", meta.case),
        ));
    }
    Ok(())
}

/// Validates ids and references of a gold board.
pub fn validate_board(case: &str, board: &GoldBoard) -> Result<()> {
    let mut ids = BTreeSet::new();
    for n in &board.nodes {
        if !ids.insert(n.id.as_str()) {
            return Err(fixture_err(case, format!("duplicate node id `{}`", n.id)));
        }
    }
    for e in &board.edges {
        for end in [&e.src, &e.dst] {
            if !ids.contains(end.as_str()) {
                return Err(fixture_err(
                    case,
                    format!("edge endpoint `{end}` is not a node id"),
                ));
            }
        }
    }
    for o in &board.owners {
        if !ids.contains(o.near.as_str()) {
            return Err(fixture_err(
                case,
                format!("owner `{}` is near unknown node `{}`", o.name, o.near),
            ));
        }
    }
    Ok(())
}

/// Loads and validates every board case under `root`.
pub fn load_board_suite(root: &Path) -> Result<Vec<BoardCase>> {
    let mut out = Vec::new();
    for dir in case_dirs(root)? {
        let name = dir_name(&dir);
        let meta: CaseMeta = read_toml(&dir.join("meta.toml"))?;
        check_meta(&name, &meta)?;
        let expected: BoardExpected = read_json(&dir.join("expected.json"))?;
        if expected.schema != BOARD_EXPECTED_SCHEMA
            || !expected.schema_version.starts_with(EXPECTED_MAJOR)
        {
            return Err(fixture_err(
                &name,
                format!(
                    "expected.json has schema `{}` {}, want `{BOARD_EXPECTED_SCHEMA}` 1.x",
                    expected.schema, expected.schema_version
                ),
            ));
        }
        if expected.screen_type != meta.screen_type {
            return Err(fixture_err(
                &name,
                "meta.toml and expected.json disagree on screen_type",
            ));
        }
        validate_board(&name, &expected.board)?;
        if !dir.join("frame.png").is_file() {
            return Err(fixture_err(&name, "frame.png is missing"));
        }
        out.push(BoardCase {
            name,
            dir,
            meta,
            expected,
        });
    }
    Ok(out)
}

/// Loads and validates every document case under `root`.
pub fn load_docs_suite(root: &Path) -> Result<Vec<DocsCase>> {
    let mut out = Vec::new();
    for dir in case_dirs(root)? {
        let name = dir_name(&dir);
        let meta: CaseMeta = read_toml(&dir.join("meta.toml"))?;
        check_meta(&name, &meta)?;
        let expected: DocsExpected = read_json(&dir.join("expected.json"))?;
        if expected.schema != DOCS_EXPECTED_SCHEMA
            || !expected.schema_version.starts_with(EXPECTED_MAJOR)
        {
            return Err(fixture_err(
                &name,
                format!(
                    "expected.json has schema `{}` {}, want `{DOCS_EXPECTED_SCHEMA}` 1.x",
                    expected.schema, expected.schema_version
                ),
            ));
        }
        if expected.frames.is_empty() {
            return Err(fixture_err(&name, "no frames listed"));
        }
        for f in &expected.frames {
            if !dir.join(&f.file).is_file() {
                return Err(fixture_err(&name, format!("frame {} is missing", f.file)));
            }
        }
        out.push(DocsCase {
            name,
            dir,
            meta,
            expected,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::board::{GoldEdge, GoldNode, GoldOwnerTag, LineStyle};

    fn node(id: &str) -> GoldNode {
        GoldNode {
            id: id.into(),
            text: id.into(),
            aliases: vec![],
            bbox: None,
            core: true,
        }
    }

    #[test]
    fn board_validation_catches_bad_references() {
        let mut b = GoldBoard {
            nodes: vec![node("a"), node("b")],
            ..Default::default()
        };
        assert!(validate_board("c", &b).is_ok());
        b.edges.push(GoldEdge {
            src: "a".into(),
            dst: "zz".into(),
            label: String::new(),
            label_aliases: vec![],
            style: LineStyle::Solid,
            directed: true,
        });
        assert!(validate_board("c", &b).is_err());
        b.edges.clear();
        b.owners.push(GoldOwnerTag {
            name: "Avery".into(),
            near: "nope".into(),
        });
        assert!(validate_board("c", &b).is_err());
        b.owners.clear();
        b.nodes.push(node("a"));
        assert!(validate_board("c", &b).is_err());
    }

    #[test]
    fn meta_name_must_match_dir() {
        let meta = CaseMeta {
            case: "x".into(),
            suite: "synthetic".into(),
            description: String::new(),
            generator: "gen_synthetic".into(),
            generator_version: 1,
            seed: 1,
            screen_type: ScreenType::Whiteboard,
            tags: vec![],
        };
        assert!(check_meta("x", &meta).is_ok());
        assert!(check_meta("y", &meta).is_err());
    }
}
