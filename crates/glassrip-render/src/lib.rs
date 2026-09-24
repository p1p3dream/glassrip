//! glassrip-render: markdown notes and house-style SVG for meeting mode.
//!
//! - [`markdown`]: `glassrip.meeting_notes` to markdown through a minijinja
//!   template, then parsed back with pulldown-cmark to catch broken tables and links.
//! - [`scene`] and [`svg`]: `glassrip.board_state` plus the validated decisions to a
//!   house-style SVG (layout from the board's canvas positions; Sugiyama fallback),
//!   validated by a strict usvg parse and a resvg PNG render.
//! - [`stage::RenderStage`]: the `render` stage (`glassrip.render`).
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod facts;
pub mod markdown;
pub mod scene;
pub mod stage;
pub mod style;
pub mod svg;

use std::collections::BTreeSet;
use std::path::Path;

use glassrip_notes::board::BoardStateItem;
use glassrip_notes::notes::MeetingNotes;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use stage::{RenderParams, RenderStage};

/// Render errors.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// A template failed.
    #[error("template: {0}")]
    Template(String),
    /// Writing an output failed.
    #[error("writing {path}: {message}")]
    Write {
        /// Path.
        path: String,
        /// Message.
        message: String,
    },
}

/// One written file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RenderedFile {
    /// File name relative to the output directory.
    pub name: String,
    /// Kind (`markdown`, `svg`, `png`).
    pub kind: String,
    /// Board id (SVG and PNG).
    pub board_id: Option<String>,
    /// Bytes.
    pub bytes: usize,
    /// blake3 of the contents.
    pub blake3: String,
}

/// Everything `render` produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RenderResult {
    /// Files written.
    pub files: Vec<RenderedFile>,
    /// Markdown checks.
    pub markdown: markdown::MarkdownChecks,
    /// SVG checks per board.
    pub svg: Vec<(String, svg::SvgChecks)>,
    /// Markdown text (so a cached result can be written again).
    pub markdown_text: String,
    /// SVG text per board.
    pub svg_text: Vec<(String, String)>,
    /// All checks passed.
    pub ok: bool,
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> Result<(), RenderError> {
    let path = dir.join(name);
    let err = |e: std::io::Error| RenderError::Write {
        path: path.display().to_string(),
        message: e.to_string(),
    };
    std::fs::create_dir_all(dir).map_err(err)?;
    let tmp = dir.join(format!(".{name}.tmp"));
    std::fs::write(&tmp, bytes).map_err(err)?;
    std::fs::rename(&tmp, &path).map_err(err)
}

/// Renders markdown and one SVG (plus PNG preview) per board into `out_dir`.
pub fn render_all(
    notes: &MeetingNotes,
    boards: &[BoardStateItem],
    params: &RenderParams,
) -> Result<RenderResult, RenderError> {
    let (out_dir, stem, meta): (&Path, &str, _) = (&params.out_dir, &params.stem, &params.meta);
    let env = svg::environment();
    let mut files = Vec::new();
    let mut links = markdown::Links::default();
    let mut svg_checks = Vec::new();
    let mut svg_text = Vec::new();
    let multi = boards.len() > 1;
    for b in boards {
        let scene = scene::build_scene(b, notes);
        let text = svg::render_svg(&env, &scene)?;
        let (checks, png) = svg::validate_svg(&text, &scene, &params.fonts);
        let base = if multi {
            format!("{stem}-{}-architecture", b.board_id)
        } else {
            format!("{stem}-architecture")
        };
        let svg_name = format!("{base}.svg");
        write(out_dir, &svg_name, text.as_bytes())?;
        files.push(RenderedFile {
            name: svg_name.clone(),
            kind: "svg".into(),
            board_id: Some(b.board_id.clone()),
            bytes: text.len(),
            blake3: glassrip_core::blake3_hex(text.as_bytes()),
        });
        links.svg.push((b.board_id.clone(), svg_name));
        if let Some(png) = png {
            let png_name = format!("{base}.png");
            write(out_dir, &png_name, &png)?;
            files.push(RenderedFile {
                name: png_name.clone(),
                kind: "png".into(),
                board_id: Some(b.board_id.clone()),
                bytes: png.len(),
                blake3: glassrip_core::blake3_hex(&png),
            });
            links.png.push((b.board_id.clone(), png_name));
        }
        svg_checks.push((b.board_id.clone(), checks));
        svg_text.push((b.board_id.clone(), text));
    }
    let md = markdown::render_markdown(&env, notes, boards, &links, meta)?;
    let md_name = format!("{stem}-meeting-notes.md");
    write(out_dir, &md_name, md.as_bytes())?;
    let present: BTreeSet<String> = files.iter().map(|f| f.name.clone()).collect();
    let md_checks = markdown::check_markdown(&md, &present);
    files.insert(
        0,
        RenderedFile {
            name: md_name,
            kind: "markdown".into(),
            board_id: None,
            bytes: md.len(),
            blake3: glassrip_core::blake3_hex(md.as_bytes()),
        },
    );
    let ok = md_checks.ok && svg_checks.iter().all(|(_, c)| c.ok);
    Ok(RenderResult {
        files,
        markdown: md_checks,
        svg: svg_checks,
        markdown_text: md,
        svg_text,
        ok,
    })
}
