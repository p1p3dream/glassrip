//! Screen layout from OCR text: conferencing tiles, banner, shared area, and
//! whiteboard application panels (spec 6.9).
//!
//! Layout comes first; pixels are not used here. Participant tiles are found
//! from their name labels (bottom-left text in a tile) and the grid those labels
//! form: labels in one column share a left edge, labels in one row share a
//! baseline, and the pitch between them gives the tile size. The shared area is
//! what remains inside the conferencing window: below the title bar, left of a
//! tile column (or above a tile strip). Inside it, a whiteboard's sidebar and top
//! bar are recognized from their UI strings and cut away.

use glassrip_vision::board::{normalize, ChromeDenylist};
use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::artifacts::{ChromeReason, TextRegion, TileBox};

/// Layout parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LayoutConfig {
    pub denylist: ChromeDenylist,
    /// Substrings of conferencing banners (for example `(Presenting`).
    pub banner_patterns: Vec<String>,
    /// Known participant names (optional; banners add more per keyframe).
    pub participants: Vec<String>,
    /// UI strings of a whiteboard application's sidebar.
    pub sidebar_terms: Vec<String>,
    /// UI strings of a whiteboard application's top bar (logo).
    pub top_bar_terms: Vec<String>,
    /// Tallest tile label, as a fraction of image height.
    pub max_label_height: f64,
    /// Alignment tolerance for label columns and rows, fraction of the image side.
    pub align_tolerance: f64,
    /// Smallest distance between neighboring tile labels, fraction of the image side.
    pub min_tile_pitch: f64,
    /// Minimum OCR confidence for a tile label.
    pub min_label_confidence: f64,
    /// Tile aspect ratio (width / height) when only one grid direction is seen.
    pub tile_aspect: f64,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            denylist: ChromeDenylist::miro_meet_defaults(),
            banner_patterns: vec!["(Presenting".into(), "is presenting".into()],
            participants: Vec::new(),
            sidebar_terms: ["Overview", "Browse", "Create section", "Starred boards"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            top_bar_terms: vec!["miro".into()],
            max_label_height: 0.05,
            align_tolerance: 0.025,
            min_tile_pitch: 0.08,
            min_label_confidence: 0.7,
            tile_aspect: 16.0 / 9.0,
        }
    }
}

/// Text span as layout analysis sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub text: String,
    pub bbox: BBox,
    pub confidence: f64,
}

/// Result of layout analysis on one image.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Layout {
    pub width: f64,
    pub height: f64,
    pub tiles: Vec<TileBox>,
    /// Union of the tiles.
    pub tile_region: Option<BBox>,
    /// A lone name label that could be a tile but has no grid support.
    pub ambiguous_tile: Option<TileBox>,
    /// Bottom of the conferencing title bar.
    pub banner_bottom: Option<f64>,
    /// Left edge of the conferencing window (from the title bar).
    pub window_left: Option<f64>,
    pub share_area: Option<BBox>,
    /// Right edge of the whiteboard sidebar plus toolbar.
    pub panel_left: Option<f64>,
    /// Bottom of the whiteboard top bar.
    pub panel_top: Option<f64>,
    /// Canvas from layout alone (share area minus panels), when any evidence was found.
    pub canvas: Option<BBox>,
    /// Names from banners and tile labels.
    pub names: Vec<String>,
    /// Per-span chrome reason (same order as the input).
    pub reasons: Vec<Option<ChromeReason>>,
    /// Indices of spans that are tile labels.
    pub label_spans: Vec<usize>,
}

fn center(b: &BBox) -> (f64, f64) {
    ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
}

fn contains_point(b: &BBox, (x, y): (f64, f64)) -> bool {
    x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2
}

fn union(a: BBox, b: BBox) -> BBox {
    BBox::new(
        a.x1.min(b.x1),
        a.y1.min(b.y1),
        a.x2.max(b.x2),
        a.y2.max(b.y2),
    )
}

/// Strip a truncation marker (`...`, `…`) and surrounding space.
pub fn strip_ellipsis(text: &str) -> &str {
    text.trim()
        .trim_end_matches('…')
        .trim_end_matches("...")
        .trim_end_matches("..")
        .trim()
}

fn is_name_token(t: &str) -> bool {
    let letters = t.chars().filter(|c| c.is_alphabetic()).count();
    letters >= 1
        && t.chars()
            .all(|c| c.is_alphabetic() || c == '-' || c == '\'' || c == '.')
}

fn capitalized(t: &str) -> bool {
    t.chars().next().is_some_and(char::is_uppercase)
}

/// Text shaped like a person's name: 2 to 4 alphabetic words whose first and
/// last words are capitalized (the last may be cut off by a truncation marker).
pub fn name_shaped(text: &str) -> Option<String> {
    let s = strip_ellipsis(text);
    let tokens: Vec<&str> = s.split_whitespace().collect();
    if !(2..=4).contains(&tokens.len()) || !tokens.iter().all(|t| is_name_token(t)) {
        return None;
    }
    let letters: usize = tokens
        .iter()
        .map(|t| t.chars().filter(|c| c.is_alphabetic()).count())
        .sum();
    let first_ok = tokens.first().is_some_and(|t| capitalized(t));
    let last_ok = tokens.last().is_some_and(|t| capitalized(t));
    (letters >= 5 && first_ok && last_ok).then(|| s.to_string())
}

/// Fuzzy name match that tolerates truncated labels ("Jane Smi" vs "Jane Smith").
pub fn names_match(a: &str, b: &str) -> bool {
    let (a, b) = (normalize(strip_ellipsis(a)), normalize(strip_ellipsis(b)));
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let (short, long) = if a.len() <= b.len() {
        (&a, &b)
    } else {
        (&b, &a)
    };
    (short.len() >= 5 && long.starts_with(short.as_str()))
        || strsim::normalized_levenshtein(&a, &b) >= 0.8
}

/// Name before a banner pattern, for example "Jane Smith (Presenting)".
fn banner_name(text: &str, patterns: &[String]) -> Option<String> {
    let lower = text.to_lowercase();
    patterns.iter().find_map(|p| {
        let i = lower.find(&p.to_lowercase())?;
        let name = text.get(..i)?.trim();
        name_shaped(name)
    })
}

fn is_clock(text: &str) -> bool {
    // "12:58 PM", "9:05", "12:58 PM | Title"
    let t = text.trim();
    let mut chars = t.chars().peekable();
    let mut digits = 0;
    while let Some(c) = chars.peek() {
        if c.is_ascii_digit() {
            digits += 1;
            chars.next();
        } else {
            break;
        }
    }
    if !(1..=2).contains(&digits) || chars.next() != Some(':') {
        return false;
    }
    let rest: String = chars.collect();
    rest.len() >= 2 && rest.chars().take(2).all(|c| c.is_ascii_digit())
}

fn same_line(a: &BBox, b: &BBox) -> bool {
    let h = a.height().max(b.height());
    (center(a).1 - center(b).1).abs() <= h * 0.6
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

/// Group values that lie within `tol` of the group's first value.
fn clusters(values: &[(usize, f64)], tol: f64) -> Vec<Vec<usize>> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut out: Vec<(f64, Vec<usize>)> = Vec::new();
    for (i, v) in sorted {
        match out.last_mut() {
            Some((start, members)) if (v - *start).abs() <= tol => members.push(i),
            _ => out.push((v, vec![i])),
        }
    }
    out.into_iter().map(|(_, m)| m).collect()
}

/// Analyze one image's OCR spans.
pub fn analyze(spans: &[Span], width: f64, height: f64, cfg: &LayoutConfig) -> Layout {
    let mut layout = Layout {
        width,
        height,
        reasons: vec![None; spans.len()],
        ..Layout::default()
    };
    let full = BBox::new(0.0, 0.0, width, height);

    // Banner and title bar.
    let mut names: Vec<String> = Vec::new();
    let mut banner_idx = Vec::new();
    for (i, s) in spans.iter().enumerate() {
        let lower = s.text.to_lowercase();
        let banner = cfg
            .banner_patterns
            .iter()
            .any(|p| lower.contains(&p.to_lowercase()));
        if banner || (is_clock(&s.text) && center(&s.bbox).1 < height * 0.3) {
            banner_idx.push(i);
        }
        if let Some(n) = banner_name(&s.text, &cfg.banner_patterns) {
            names.push(n);
        }
    }
    // Spans on the same line as a banner span in the top part are title bar text.
    let title_line: Vec<usize> = spans
        .iter()
        .enumerate()
        .filter(|(i, s)| {
            banner_idx.contains(i)
                || banner_idx.iter().any(|&b| {
                    same_line(&spans[b].bbox, &s.bbox) && center(&s.bbox).1 < height * 0.3
                })
        })
        .map(|(i, _)| i)
        .collect();
    for &i in &title_line {
        layout.reasons[i] = Some(ChromeReason::Banner);
    }
    if !title_line.is_empty() {
        layout.banner_bottom = title_line
            .iter()
            .map(|&i| spans[i].bbox.y2)
            .fold(None, |a: Option<f64>, v| Some(a.map_or(v, |a| a.max(v))));
        layout.window_left = spans
            .iter()
            .enumerate()
            .filter(|(i, s)| title_line.contains(i) && is_clock(&s.text))
            .map(|(_, s)| s.bbox.x1)
            .fold(None, |a: Option<f64>, v| Some(a.map_or(v, |a| a.min(v))));
    }
    let known: Vec<String> = cfg
        .participants
        .iter()
        .cloned()
        .chain(names.clone())
        .collect();

    // Tile label candidates.
    let cands: Vec<usize> = spans
        .iter()
        .enumerate()
        .filter(|(i, s)| {
            layout.reasons[*i].is_none()
                && s.confidence >= cfg.min_label_confidence
                && s.bbox.height() <= height * cfg.max_label_height
                && !cfg.denylist.matches(&s.text)
                && (name_shaped(&s.text).is_some() || known.iter().any(|k| names_match(k, &s.text)))
        })
        .map(|(i, _)| i)
        .collect();
    let is_known = |i: usize| known.iter().any(|k| names_match(k, &spans[i].text));
    let col_groups: Vec<Vec<usize>> = clusters(
        &cands
            .iter()
            .map(|&i| (i, spans[i].bbox.x1))
            .collect::<Vec<_>>(),
        width * cfg.align_tolerance,
    );
    let row_groups: Vec<Vec<usize>> = clusters(
        &cands
            .iter()
            .map(|&i| (i, spans[i].bbox.y2))
            .collect::<Vec<_>>(),
        height * cfg.align_tolerance,
    );
    let pitch = |g: &[usize], f: &dyn Fn(usize) -> f64| -> Option<f64> {
        let mut v: Vec<f64> = g.iter().map(|&i| f(i)).collect();
        v.sort_by(f64::total_cmp);
        median(v.windows(2).map(|w| w[1] - w[0]).collect())
    };
    let mut labels: Vec<usize> = Vec::new();
    let mut pitch_y: Option<f64> = None;
    let mut pitch_x: Option<f64> = None;
    let edge_hugging = |g: &[usize], col: bool| {
        g.iter().all(|&i| {
            let b = &spans[i].bbox;
            if col {
                b.x1 >= width * 0.8
            } else {
                b.y1 >= height * 0.8
            }
        })
    };
    for g in col_groups.iter().filter(|g| g.len() >= 2) {
        let p = pitch(g, &|i| spans[i].bbox.y2);
        let spaced = p.is_some_and(|p| p >= height * cfg.min_tile_pitch);
        let right = g.iter().all(|&i| spans[i].bbox.x1 >= width * 0.55);
        if spaced && right && (g.iter().any(|&i| is_known(i)) || edge_hugging(g, true)) {
            labels.extend(g);
            pitch_y = p;
        }
    }
    for g in row_groups.iter().filter(|g| g.len() >= 2) {
        let p = pitch(g, &|i| spans[i].bbox.x1);
        let spaced = p.is_some_and(|p| p >= width * cfg.min_tile_pitch);
        let low_or_right = g
            .iter()
            .all(|&i| spans[i].bbox.y1 >= height * 0.65 || spans[i].bbox.x1 >= width * 0.55);
        let supported =
            g.iter().any(|&i| is_known(i) || labels.contains(&i)) || edge_hugging(g, false);
        if spaced && low_or_right && supported {
            labels.extend(g);
            pitch_x = p;
        }
    }
    labels.sort_unstable();
    labels.dedup();
    let tile_h = pitch_y.or_else(|| pitch_x.map(|p| p / cfg.tile_aspect));
    let tile_w = pitch_x.or_else(|| pitch_y.map(|p| p * cfg.tile_aspect));
    if let (Some(th), Some(tw)) = (tile_h, tile_w) {
        let mut region: Option<BBox> = None;
        for &i in &labels {
            let b = spans[i].bbox;
            let x1 = (b.x1 - width * 0.015).max(0.0);
            let y2 = (b.y2 + height * 0.012).min(height);
            let tile = BBox::new(x1, (y2 - th).max(0.0), (x1 + tw).min(width), y2);
            region = Some(region.map_or(tile, |r| union(r, tile)));
            layout.tiles.push(TileBox {
                name: strip_ellipsis(&spans[i].text).to_string(),
                bbox: tile,
            });
            layout.reasons[i] = Some(ChromeReason::TileName);
            names.push(strip_ellipsis(&spans[i].text).to_string());
        }
        layout.tile_region = region;
        layout.label_spans = labels.clone();
    } else if let Some(&i) = cands.iter().find(|&&i| {
        is_known(i) && (spans[i].bbox.x1 >= width * 0.55 || spans[i].bbox.y1 >= height * 0.65)
    }) {
        // One known name with no grid: a candidate for the variance tiebreak.
        let b = spans[i].bbox;
        let tw = width * 0.22;
        let th = tw / cfg.tile_aspect;
        let x1 = (b.x1 - width * 0.015).max(0.0);
        let y2 = (b.y2 + height * 0.012).min(height);
        layout.ambiguous_tile = Some(TileBox {
            name: strip_ellipsis(&spans[i].text).to_string(),
            bbox: BBox::new(x1, (y2 - th).max(0.0), (x1 + tw).min(width), y2),
        });
    }
    names.sort();
    names.dedup_by(|a, b| names_match(a, b));
    layout.names = names;

    compute_share_and_canvas(&mut layout, spans, cfg);
    let _ = full;
    layout
}

/// Accept the ambiguous tile (after a tiebreak) and recompute the areas.
pub fn accept_ambiguous_tile(layout: &mut Layout, spans: &[Span], cfg: &LayoutConfig) {
    if let Some(t) = layout.ambiguous_tile.take() {
        layout.tile_region = Some(t.bbox);
        if let Some(i) = spans.iter().position(|s| {
            strip_ellipsis(&s.text) == t.name && contains_point(&t.bbox, center(&s.bbox))
        }) {
            layout.reasons[i] = Some(ChromeReason::TileName);
            layout.label_spans.push(i);
        }
        layout.names.push(t.name.clone());
        layout.tiles.push(t);
        compute_share_and_canvas(layout, spans, cfg);
    }
}

fn compute_share_and_canvas(layout: &mut Layout, spans: &[Span], cfg: &LayoutConfig) {
    let (width, height) = (layout.width, layout.height);
    let mut share: Option<BBox> = None;
    if let Some(t) = layout.tile_region {
        let column = t.x1 >= width * 0.45;
        let strip = !column && t.y1 >= height * 0.55;
        let top = layout.banner_bottom.map_or(0.0, |b| b + height * 0.01);
        let left = layout
            .window_left
            .map_or(0.0, |l| (l - width * 0.01).max(0.0));
        if column {
            share = Some(BBox::new(
                left,
                top,
                (t.x1 - width * 0.008).max(left + 1.0),
                t.y2.max(top + 1.0),
            ));
        } else if strip {
            share = Some(BBox::new(
                left,
                top,
                width,
                (t.y1 - height * 0.008).max(top + 1.0),
            ));
        }
    } else if let Some(b) = layout.banner_bottom {
        let left = layout
            .window_left
            .map_or(0.0, |l| (l - width * 0.01).max(0.0));
        share = Some(BBox::new(left, b + height * 0.01, width, height));
    }
    layout.share_area = share.filter(BBox::is_well_formed);
    let area = layout
        .share_area
        .unwrap_or(BBox::new(0.0, 0.0, width, height));

    // Whiteboard panels inside the shared area.
    let in_area = |s: &Span| contains_point(&area, center(&s.bbox));
    let sidebar: Vec<&Span> = spans
        .iter()
        .filter(|s| {
            in_area(s)
                && cfg
                    .sidebar_terms
                    .iter()
                    .any(|t| normalize(&s.text) == normalize(t))
                && center(&s.bbox).0 < area.x1 + area.width() * 0.35
        })
        .collect();
    layout.panel_left = if sidebar.is_empty() {
        None
    } else {
        let right = sidebar.iter().map(|s| s.bbox.x2).fold(f64::MIN, f64::max);
        let h = median(sidebar.iter().map(|s| s.bbox.height()).collect()).unwrap_or(0.0);
        Some(right + (width * 0.035).max(3.0 * h))
    };
    let logo: Vec<&Span> = spans
        .iter()
        .filter(|s| {
            in_area(s)
                && cfg
                    .top_bar_terms
                    .iter()
                    .any(|t| normalize(&s.text) == normalize(t))
                && center(&s.bbox).1 < area.y1 + area.height() * 0.2
        })
        .collect();
    layout.panel_top = if logo.is_empty() {
        None
    } else {
        let line_bottom = spans
            .iter()
            .filter(|s| logo.iter().any(|l| same_line(&l.bbox, &s.bbox)))
            .map(|s| s.bbox.y2)
            .fold(f64::MIN, f64::max);
        Some(line_bottom + height * 0.012)
    };

    let evidence =
        layout.share_area.is_some() || layout.panel_left.is_some() || layout.panel_top.is_some();
    layout.canvas = evidence
        .then(|| {
            BBox::new(
                layout.panel_left.map_or(area.x1, |p| p.max(area.x1)),
                layout.panel_top.map_or(area.y1, |p| p.max(area.y1)),
                area.x2,
                area.y2,
            )
        })
        .filter(|c| c.is_well_formed() && c.area() >= width * height * 0.08);

    // Per-span chrome reasons (banner and tile labels already set).
    for (i, s) in spans.iter().enumerate() {
        if matches!(
            layout.reasons[i],
            Some(ChromeReason::Banner | ChromeReason::TileName)
        ) {
            continue;
        }
        let c = center(&s.bbox);
        layout.reasons[i] = if cfg.denylist.matches(&s.text) {
            Some(ChromeReason::Denylist)
        } else if layout.share_area.is_some_and(|a| !contains_point(&a, c)) {
            Some(ChromeReason::OutsideShare)
        } else if layout.panel_left.is_some_and(|p| c.0 < p)
            || layout.panel_top.is_some_and(|p| c.1 < p)
        {
            Some(ChromeReason::AppPanel)
        } else {
            None
        };
    }
}

/// Region of a span given the layout (before or after the canvas is known).
pub fn region_of(
    layout: &Layout,
    index: usize,
    span: &Span,
    canvas: Option<&BBox>,
) -> (TextRegion, Option<ChromeReason>) {
    let c = center(&span.bbox);
    if layout.label_spans.contains(&index)
        || layout.tiles.iter().any(|t| contains_point(&t.bbox, c))
    {
        return (TextRegion::Tile, Some(ChromeReason::TileName));
    }
    if let Some(reason) = layout.reasons.get(index).copied().flatten() {
        return (TextRegion::Chrome, Some(reason));
    }
    match canvas {
        Some(cv) if contains_point(cv, c) => (TextRegion::Canvas, None),
        Some(_) => (TextRegion::Chrome, Some(ChromeReason::OutsideCanvas)),
        None => (TextRegion::Unassigned, None),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn span(text: &str, x1: f64, y1: f64, x2: f64, y2: f64) -> Span {
        Span {
            text: text.into(),
            bbox: BBox::new(x1, y1, x2, y2),
            confidence: 0.95,
        }
    }

    /// Synthetic conferencing layout on a 1000 x 600 image: title bar, a board
    /// on the left, and a tile column on the right (fictional names).
    fn meeting_spans() -> Vec<Span> {
        vec![
            span("3:15 PM | Weekly Sync", 100.0, 40.0, 260.0, 55.0),
            span("Ada Quill (Presenting)", 700.0, 40.0, 860.0, 55.0),
            span("miro", 120.0, 75.0, 150.0, 88.0),
            span("Board title", 160.0, 75.0, 260.0, 88.0),
            span("Overview", 110.0, 130.0, 170.0, 142.0),
            span("Browse", 110.0, 150.0, 160.0, 162.0),
            span("Order Service", 350.0, 250.0, 450.0, 265.0),
            span("Payments", 500.0, 300.0, 570.0, 315.0),
            span("100%", 580.0, 440.0, 610.0, 452.0),
            span("Ada Quill", 690.0, 200.0, 760.0, 214.0),
            span("Bo Tran Liu", 690.0, 340.0, 780.0, 354.0),
            span("Cy Obi Tar", 830.0, 340.0, 910.0, 354.0),
        ]
    }

    #[test]
    fn finds_tile_grid_share_area_and_panels() {
        let spans = meeting_spans();
        let l = analyze(&spans, 1000.0, 600.0, &LayoutConfig::default());
        assert_eq!(l.tiles.len(), 3, "{:?}", l.tiles);
        let share = l.share_area.unwrap();
        assert!(share.x2 < 690.0 && share.x2 > 640.0, "{share:?}");
        assert!(share.y1 > 55.0 && share.y1 < 70.0);
        assert!((share.x1 - 90.0).abs() < 1.0);
        let canvas = l.canvas.unwrap();
        assert!(canvas.x1 > 170.0, "{canvas:?}");
        assert!(canvas.y1 > 88.0);
        assert!(l.names.iter().any(|n| n == "Ada Quill"));
        assert_eq!(l.reasons[0], Some(ChromeReason::Banner));
        assert_eq!(l.reasons[3], Some(ChromeReason::AppPanel));
        assert_eq!(l.reasons[4], Some(ChromeReason::Denylist));
        assert_eq!(l.reasons[8], Some(ChromeReason::Denylist));
        assert_eq!(l.reasons[6], None);
        let (r, _) = region_of(&l, 6, &spans[6], Some(&canvas));
        assert_eq!(r, TextRegion::Canvas);
        let (r, _) = region_of(&l, 10, &spans[10], Some(&canvas));
        assert_eq!(r, TextRegion::Tile);
    }

    #[test]
    fn board_labels_are_not_tiles() {
        // Name-shaped board labels in the left half, no banner: no tiles.
        let spans = vec![
            span("Mobile Client", 100.0, 100.0, 200.0, 112.0),
            span("Edge Gateway", 100.0, 300.0, 200.0, 312.0),
            span("Ledger Store", 300.0, 300.0, 400.0, 312.0),
        ];
        let l = analyze(&spans, 1000.0, 600.0, &LayoutConfig::default());
        assert!(l.tiles.is_empty());
        assert!(l.share_area.is_none());
        assert!(l.canvas.is_none());
    }

    #[test]
    fn edge_column_without_known_names() {
        let spans = vec![
            span("Rex Hale", 910.0, 120.0, 980.0, 132.0),
            span("Ivy Moss Park", 905.0, 250.0, 990.0, 262.0),
            span("Queue", 300.0, 300.0, 350.0, 312.0),
        ];
        let l = analyze(&spans, 1000.0, 600.0, &LayoutConfig::default());
        assert_eq!(l.tiles.len(), 2);
        let c = l.canvas.unwrap();
        assert!(c.x2 < 905.0 && c.x1 == 0.0);
    }

    #[test]
    fn single_known_label_is_ambiguous_until_accepted() {
        let spans = vec![
            span("Ada Quill (Presenting)", 300.0, 20.0, 460.0, 34.0),
            span("Ada Quill", 800.0, 400.0, 870.0, 414.0),
        ];
        let cfg = LayoutConfig::default();
        let mut l = analyze(&spans, 1000.0, 600.0, &cfg);
        assert!(l.tiles.is_empty());
        assert!(l.ambiguous_tile.is_some());
        accept_ambiguous_tile(&mut l, &spans, &cfg);
        assert_eq!(l.tiles.len(), 1);
        assert!(l.share_area.unwrap().x2 < 800.0);
    }

    #[test]
    fn name_shape_rules() {
        assert!(name_shaped("Jane Smith").is_some());
        assert!(name_shaped("Jane van Smi...").is_some());
        assert!(name_shaped("Unread chats").is_none());
        assert!(name_shaped("Set of cool").is_none());
        assert!(name_shaped("Jane").is_none());
        assert!(name_shaped("Order 66 Service").is_none());
        assert!(names_match("Jane van Smi...", "Jane van Smith"));
        assert!(!names_match("Jane Smith", "John Doe"));
        assert!(is_clock("12:58 PM | Title"));
        assert!(!is_clock("Q3: plan"));
    }
}
