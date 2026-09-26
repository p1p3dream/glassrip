//! House-style SVG rendering and validation.

use minijinja::{AutoEscape, Environment};
use resvg::{tiny_skia, usvg};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::scene::{overlaps, Scene};
use crate::RenderError;

/// Formats a number for SVG attributes (integers without a decimal point).
fn num(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v:.1}")
    }
}

/// Makes text safe inside an XML comment (no double hyphens, no markup).
fn comment_safe(s: String) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c == '-' && out.ends_with('-') {
            continue;
        }
        if c == '<' || c == '>' {
            continue;
        }
        out.push(c);
    }
    out.trim_end_matches('-').to_string()
}

/// Template environment shared by the SVG and markdown renderers.
pub fn environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_auto_escape_callback(|name| {
        if name.ends_with(".svg") {
            AutoEscape::Html
        } else {
            AutoEscape::None
        }
    });
    env.add_filter("n", num);
    env.add_filter("cmt", comment_safe);
    env.add_filter("mmss", glassrip_notes::text::mmss);
    env.add_filter("cell", crate::markdown::cell);
    // the templates are compiled into the binary, so a failure here is a bug
    // caught by the tests; it surfaces as a render error, not a panic
    let _ = env.add_template("board.svg", include_str!("../templates/board.svg.j2"));
    let _ = env.add_template("notes.md", include_str!("../templates/notes.md.j2"));
    env
}

/// Renders a scene to SVG text.
pub fn render_svg(env: &Environment<'_>, scene: &Scene) -> Result<String, RenderError> {
    let t = env
        .get_template("board.svg")
        .map_err(|e| RenderError::Template(format!("{e:#}")))?;
    t.render(scene)
        .map_err(|e| RenderError::Template(format!("{e:#}")))
}

/// SVG validation results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SvgChecks {
    /// usvg parsed the document.
    pub parsed: bool,
    /// Parse error, if any.
    pub parse_error: Option<String>,
    /// Canvas size.
    pub width: u32,
    /// Canvas size.
    pub height: u32,
    /// PNG bytes written.
    pub png_bytes: usize,
    /// Luma standard deviation of the rendered PNG.
    pub luma_std: f64,
    /// The rendering is not blank.
    pub nonblank: bool,
    /// Text elements found.
    pub text_nodes: usize,
    /// Text elements that produced glyph outlines.
    pub text_rendered: usize,
    /// Font faces available to the renderer.
    pub font_faces: usize,
    /// Overlapping boxes (names).
    pub overlaps: Vec<String>,
    /// House-style violations (markers, `font:` shorthand, dashes).
    pub style_violations: Vec<String>,
    /// Layout method.
    pub layout_method: String,
    /// Annotations with no room on the board, listed below it instead
    /// (warnings: they do not fail validation).
    #[serde(default)]
    pub warnings: Vec<String>,
    /// All checks passed.
    pub ok: bool,
}

impl SvgChecks {
    /// Every failed check, naming the elements involved. `ok` is exactly
    /// "this list is empty".
    pub fn failures(&self) -> Vec<String> {
        let mut out = Vec::new();
        let layout = self.overlaps.iter().map(|o| format!("layout: {o}"));
        let style = self.style_violations.iter().map(|v| format!("style: {v}"));
        if !self.parsed {
            // the render checks need a parsed tree; layout and style do not
            out.push(format!(
                "svg did not parse: {}",
                self.parse_error.as_deref().unwrap_or("unknown error")
            ));
            out.extend(layout);
            out.extend(style);
            return out;
        }
        if let Some(e) = &self.parse_error {
            out.push(format!("svg render: {e}"));
        }
        if !self.nonblank {
            out.push(format!(
                "rendering is blank (luma std {:.1} <= 5)",
                self.luma_std
            ));
        }
        if self.text_nodes == 0 {
            out.push("no text elements".into());
        } else if self.text_rendered * 10 < self.text_nodes * 9 {
            out.push(format!(
                "only {} of {} text elements rendered glyphs ({} font faces)",
                self.text_rendered, self.text_nodes, self.font_faces
            ));
        }
        out.extend(layout);
        out.extend(style);
        if self.png_bytes == 0 {
            out.push("png preview was not encoded".into());
        }
        out
    }
}

fn count_text(g: &usvg::Group, total: &mut usize, rendered: &mut usize) {
    for n in g.children() {
        match n {
            usvg::Node::Group(g) => count_text(g, total, rendered),
            usvg::Node::Text(t) => {
                *total += 1;
                let b = t.flattened().bounding_box();
                if b.width() > 0.0 && b.height() > 0.0 {
                    *rendered += 1;
                }
            }
            _ => {}
        }
    }
}

/// Fonts available to the SVG validation render.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FontConfig {
    /// Load the host's system fonts.
    pub system: bool,
    /// Extra font directories, loaded first (tests ship an OFL font here).
    pub dirs: Vec<std::path::PathBuf>,
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            system: true,
            dirs: Vec::new(),
        }
    }
}

/// Loads fonts and maps the generic families to installed ones (Inter first,
/// then common sans fonts, then any loaded family). Returns the sans-serif
/// family chosen.
pub fn configure_fonts(db: &mut usvg::fontdb::Database, fonts: &FontConfig) -> Option<String> {
    for d in &fonts.dirs {
        db.load_fonts_dir(d);
    }
    if fonts.system {
        db.load_system_fonts();
    }
    let families: std::collections::BTreeSet<String> = db
        .faces()
        .flat_map(|f| f.families.iter().map(|(n, _)| n.clone()))
        .collect();
    let pick = |cands: &[&str]| {
        cands
            .iter()
            .find(|c| families.contains(**c))
            .map(|c| (*c).to_string())
    };
    let sans = pick(&[
        "Inter",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Noto Sans",
    ])
    .or_else(|| families.iter().find(|f| f.starts_with("Inter")).cloned())
    .or_else(|| families.iter().next().cloned());
    if let Some(f) = &sans {
        db.set_sans_serif_family(f.clone());
    }
    if let Some(f) = pick(&[
        "Menlo",
        "SF Mono",
        "DejaVu Sans Mono",
        "Liberation Mono",
        "Noto Sans Mono",
        "Courier New",
    ])
    .or_else(|| sans.clone())
    {
        db.set_monospace_family(f);
    }
    sans
}

/// CSS in an SVG: `<style>` element bodies and `style` attribute values. Text
/// content is not CSS (and the template escapes it, so it cannot fake either).
fn css_parts(svg: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = svg;
    while let Some(i) = rest.find("<style") {
        let after = &rest[i + 6..];
        let Some(open_end) = after.find('>') else {
            break;
        };
        let body = &after[open_end + 1..];
        let end = body.find("</style>").unwrap_or(body.len());
        out.push(&body[..end]);
        rest = &body[end..];
    }
    for quote in ['"', '\''] {
        let key = format!("style={quote}");
        let mut rest = svg;
        while let Some(i) = rest.find(&key) {
            let after = &rest[i + key.len()..];
            let end = after.find(quote).unwrap_or(after.len());
            out.push(&after[..end]);
            rest = &after[end..];
        }
    }
    out
}

/// True when CSS uses the `font` shorthand property (not `font-size` etc.).
fn has_font_shorthand(css: &str) -> bool {
    let b = css.as_bytes();
    css.match_indices("font").any(|(i, _)| {
        let before_ok = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'-');
        before_ok && css[i + 4..].trim_start().starts_with(':')
    })
}

/// Style rules that do not need a parser.
pub fn style_violations(svg: &str) -> Vec<String> {
    let mut v = Vec::new();
    if svg.contains("<marker") {
        v.push("uses <marker> (arrowheads must be polygons)".into());
    }
    if css_parts(svg).iter().any(|css| has_font_shorthand(css)) {
        v.push("uses the font: shorthand".into());
    }
    if svg.contains('\u{2014}') || svg.contains('\u{2013}') {
        v.push("contains an em or en dash".into());
    }
    let mut rest = svg;
    while let Some(i) = rest.find("<!--") {
        let after = &rest[i + 4..];
        let end = after.find("-->").unwrap_or(after.len());
        if after[..end].contains("--") {
            v.push("XML comment contains a double hyphen".into());
            break;
        }
        rest = &after[end.min(after.len())..];
    }
    v
}

/// Parses, renders and checks an SVG. Returns the checks and the PNG bytes.
pub fn validate_svg(svg: &str, scene: &Scene, fonts: &FontConfig) -> (SvgChecks, Option<Vec<u8>>) {
    let mut opt = usvg::Options::default();
    let sans = configure_fonts(opt.fontdb_mut(), fonts);
    if let Some(f) = sans {
        opt.font_family = f;
    }
    let font_faces = opt.fontdb.len();
    let mut checks = SvgChecks {
        parsed: false,
        parse_error: None,
        width: 0,
        height: 0,
        png_bytes: 0,
        luma_std: 0.0,
        nonblank: false,
        text_nodes: 0,
        text_rendered: 0,
        font_faces,
        overlaps: overlaps(scene),
        style_violations: style_violations(svg),
        layout_method: scene.layout_method.to_string(),
        warnings: scene.degraded.clone(),
        ok: false,
    };
    let tree = match usvg::Tree::from_str(svg, &opt) {
        Ok(t) => t,
        Err(e) => {
            checks.parse_error = Some(e.to_string());
            return (checks, None);
        }
    };
    checks.parsed = true;
    let size = tree.size().to_int_size();
    checks.width = size.width();
    checks.height = size.height();
    count_text(
        tree.root(),
        &mut checks.text_nodes,
        &mut checks.text_rendered,
    );
    let Some(mut pixmap) = tiny_skia::Pixmap::new(size.width(), size.height()) else {
        checks.parse_error = Some("zero-sized canvas".into());
        return (checks, None);
    };
    resvg::render(&tree, tiny_skia::Transform::default(), &mut pixmap.as_mut());
    let (mut sum, mut sq, mut n) = (0.0f64, 0.0f64, 0.0f64);
    for px in pixmap.pixels().iter().step_by(7) {
        let c = px.demultiply();
        let l =
            0.299 * f64::from(c.red()) + 0.587 * f64::from(c.green()) + 0.114 * f64::from(c.blue());
        sum += l;
        sq += l * l;
        n += 1.0;
    }
    let mean = sum / n.max(1.0);
    checks.luma_std = (sq / n.max(1.0) - mean * mean).max(0.0).sqrt();
    checks.nonblank = checks.luma_std > 5.0;
    let png = pixmap.encode_png().ok();
    checks.png_bytes = png.as_ref().map_or(0, Vec::len);
    checks.ok = checks.failures().is_empty();
    (checks, png)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_and_comments() {
        assert_eq!(num(248.0), "248");
        assert_eq!(num(1.25), "1.2");
        assert_eq!(comment_safe("n--x->".into()), "n-x");
    }

    #[test]
    fn an_unparsed_svg_still_reports_layout_and_style() {
        let c = SvgChecks {
            parsed: false,
            parse_error: Some("bad xml".into()),
            width: 0,
            height: 0,
            png_bytes: 0,
            luma_std: 0.0,
            nonblank: false,
            text_nodes: 0,
            text_rendered: 0,
            font_faces: 1,
            overlaps: vec!["card a overlaps card b".into()],
            style_violations: vec!["uses <marker> (arrowheads must be polygons)".into()],
            layout_method: "grid".into(),
            warnings: vec!["no room on the board, listed below it as note 1: x".into()],
            ok: false,
        };
        assert_eq!(
            c.failures(),
            vec![
                "svg did not parse: bad xml".to_string(),
                "layout: card a overlaps card b".into(),
                "style: uses <marker> (arrowheads must be polygons)".into(),
            ]
        );
    }

    #[test]
    fn style_rules() {
        assert!(style_violations("<svg><!-- a--b --></svg>")
            .iter()
            .any(|v| v.contains("double hyphen")));
        assert!(style_violations("<svg><marker/></svg>")
            .iter()
            .any(|v| v.contains("marker")));
        assert!(
            style_violations("<svg><!-- ok --><text>a \u{2014} b</text></svg>")
                .iter()
                .any(|v| v.contains("dash"))
        );
        assert!(style_violations("<svg><!-- ok - fine --></svg>").is_empty());
        // the shorthand counts in CSS only, never in text content
        assert!(style_violations("<svg><text>font: bold</text></svg>").is_empty());
        assert!(style_violations(
            "<svg><style>.a { font-size: 12px; font-family: Inter; }</style></svg>"
        )
        .is_empty());
        assert!(!style_violations("<svg><style>.a { font: 12px Inter; }</style></svg>").is_empty());
        assert!(!style_violations("<svg><text style=\"font : 12px x\">a</text></svg>").is_empty());
    }
}
