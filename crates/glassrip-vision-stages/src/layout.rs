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
    /// Tile labels sit on a dark overlay: median luma inside the label box must be below this.
    pub max_label_bg_luma: f64,
}

/// The spec's Miro and Meet denylist plus Miro's shape menu and text toolbar
/// strings, which float over the canvas while someone edits.
pub fn meeting_denylist() -> ChromeDenylist {
    let mut d = ChromeDenylist::miro_meet_defaults();
    d.exact.extend(
        [
            "Line",
            "Arrow",
            "Elbow arrow",
            "Block arrow",
            "Rectangle",
            "Oval",
            "Rhombus",
            "Triangle",
            "Divider",
            "More shapes",
            "Diagram",
            "Noto Sans",
            "Auto",
            "Aa",
            "Privacy Policy",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    d.contains.push("Tidy up your Space".into());
    // The text toolbar is often read as one merged line ("Convert to Aa Auto ...").
    d.contains.push("Convert to".into());
    d
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            denylist: meeting_denylist(),
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
            max_label_bg_luma: 140.0,
        }
    }
}

/// Text span as layout analysis sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub text: String,
    pub bbox: BBox,
    pub confidence: f64,
    /// Median luma inside the box, when measured.
    pub bg_luma: Option<f64>,
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
    /// The whiteboard top bar (logo line). It floats over the canvas, so it is
    /// masked as chrome rather than cut off: board content can sit beside it.
    pub top_bar: Option<BBox>,
    /// Canvas from layout alone (share area minus panels), when any evidence was found.
    pub canvas: Option<BBox>,
    /// Top of the whiteboard zoom control, which marks the canvas bottom.
    pub zoom_top: Option<f64>,
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

/// A looser name shape for labels cut by the image edge: 2 to 4 alphabetic
/// words, the first capitalized ("Jane van", "Jane S").
pub fn loose_name(text: &str) -> bool {
    let s = strip_ellipsis(text);
    let tokens: Vec<&str> = s.split_whitespace().collect();
    let letters: usize = tokens
        .iter()
        .map(|t| t.chars().filter(|c| c.is_alphabetic()).count())
        .sum();
    (2..=4).contains(&tokens.len())
        && tokens.iter().all(|t| is_name_token(t))
        && tokens.first().is_some_and(|t| capitalized(t))
        && letters >= 5
}

/// One capitalized alphabetic word of at least four letters ("Jane").
pub fn single_name_word(text: &str) -> bool {
    let s = strip_ellipsis(text);
    !s.contains(char::is_whitespace)
        && s.chars().count() >= 4
        && s.chars().all(char::is_alphabetic)
        && capitalized(s)
}

/// Fuzzy name match that tolerates truncated labels ("Jane Smi" vs "Jane Smith").
pub fn names_match(a: &str, b: &str) -> bool {
    let (a, b) = (normalize(strip_ellipsis(a)), normalize(strip_ellipsis(b)));
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if strsim::normalized_levenshtein(&a, &b) >= 0.8 {
        return true;
    }
    // A label cut by the image edge: compare against the same-length prefix,
    // ignoring spaces ("Janevan Smi" vs "Jane van Smith").
    let squash = |s: &str| -> Vec<char> { s.chars().filter(|c| !c.is_whitespace()).collect() };
    let (a, b) = (squash(&a), squash(&b));
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.len() < 5 {
        return false;
    }
    let prefix: String = long[..short.len()].iter().collect();
    let short: String = short.iter().collect();
    strsim::normalized_levenshtein(&short, &prefix) >= 0.8
}

/// Fuzzy match of a UI string, tolerant of OCR misreads and leading icons
/// ("+ Create secton", "8 Browse", "Overveiw").
pub fn ui_term_match(text: &str, term: &str) -> bool {
    let clean = |s: &str| -> String {
        s.chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .filter(|w| w.chars().any(char::is_alphabetic))
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let (t, u) = (clean(text), clean(term));
    !t.is_empty() && !u.is_empty() && strsim::normalized_levenshtein(&t, &u) >= 0.7
}

/// A whiteboard zoom control, for example "53%", "-53% +", or "100 %".
fn is_zoom_control(text: &str) -> bool {
    let t: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let t = t.trim_matches(|c| c == '-' || c == '+' || c == '−');
    match t.strip_suffix('%') {
        Some(num) => !num.is_empty() && num.len() <= 3 && num.chars().all(|c| c.is_ascii_digit()),
        None => false,
    }
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
                && s.bg_luma.is_none_or(|l| l < cfg.max_label_bg_luma)
                && (loose_name(&s.text)
                    || single_name_word(&s.text)
                    || known.iter().any(|k| names_match(k, &s.text)))
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
    // Geometric candidates: columns in the right part, rows low or right.
    let cols: Vec<(&Vec<usize>, Option<f64>)> = col_groups
        .iter()
        .filter(|g| g.len() >= 2)
        .map(|g| (g, pitch(g, &|i| spans[i].bbox.y2)))
        .filter(|(g, p)| {
            p.is_some_and(|p| p >= height * cfg.min_tile_pitch)
                && g.iter().all(|&i| spans[i].bbox.x1 >= width * 0.55)
        })
        .collect();
    let rows: Vec<(&Vec<usize>, Option<f64>)> = row_groups
        .iter()
        .filter(|g| g.len() >= 2)
        .map(|g| (g, pitch(g, &|i| spans[i].bbox.x1)))
        .filter(|(g, p)| {
            p.is_some_and(|p| p >= width * cfg.min_tile_pitch)
                && g.iter()
                    .all(|&i| spans[i].bbox.y1 >= height * 0.65 || spans[i].bbox.x1 >= width * 0.55)
        })
        .collect();
    // A column and a row sharing a label form a grid: strong evidence even
    // when no name is known.
    let strict = |g: &[usize]| g.iter().all(|&i| name_shaped(&spans[i].text).is_some());
    let in_grid = |g: &[usize], others: &[(&Vec<usize>, Option<f64>)]| {
        strict(g)
            && others
                .iter()
                .any(|(o, _)| strict(o) && o.iter().any(|i| g.contains(i)))
    };
    for (g, p) in &cols {
        if g.iter().any(|&i| is_known(i)) || edge_hugging(g, true) || in_grid(g, &rows) {
            labels.extend(g.iter());
            pitch_y = *p;
        }
    }
    for (g, p) in &rows {
        let supported = g.iter().any(|&i| is_known(i) || labels.contains(&i))
            || edge_hugging(g, false)
            || in_grid(g, &cols);
        if supported {
            labels.extend(g.iter());
            pitch_x = *p;
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
    // Share bottom: the whiteboard zoom control sits at the canvas bottom; else
    // a tall tile column ends with the share area; else the image bottom.
    let zoom_bottom = spans
        .iter()
        .filter(|s| is_zoom_control(&s.text) && center(&s.bbox).1 > height * 0.3)
        .filter(|s| {
            share.is_none_or(|a| {
                center(&s.bbox).0 >= a.x1 && center(&s.bbox).0 <= a.x2 + width * 0.02
            })
        })
        .map(|s| s.bbox.y1)
        .fold(None, |a: Option<f64>, v| Some(a.map_or(v, |a| a.max(v))));
    layout.zoom_top = zoom_bottom;
    if let Some(a) = share.as_mut() {
        let tall = layout
            .tile_region
            .is_some_and(|t| t.height() >= height * 0.4 && t.x1 >= width * 0.45);
        a.y2 = match zoom_bottom {
            Some(z) => (z - height * 0.005).clamp(a.y1 + 1.0, height),
            None if tall => a.y2,
            None => height,
        };
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
                && cfg.sidebar_terms.iter().any(|t| ui_term_match(&s.text, t))
                && center(&s.bbox).0 < area.x1 + area.width() * 0.35
        })
        .collect();
    // No known sidebar strings: a column of at least four short lines hugging
    // the left edge of the shared area is an application sidebar.
    let sidebar: Vec<&Span> = if sidebar.is_empty() {
        let near_edge: Vec<&Span> = spans
            .iter()
            .filter(|s| {
                in_area(s)
                    && s.text.split_whitespace().count() <= 3
                    && s.bbox.x1 <= area.x1 + width * 0.05
                    && center(&s.bbox).0 < area.x1 + area.width() * 0.2
            })
            .collect();
        let mut lines: Vec<f64> = near_edge.iter().map(|s| center(&s.bbox).1).collect();
        lines.sort_by(f64::total_cmp);
        lines.dedup_by(|a, b| (*a - *b).abs() < height * 0.01);
        if lines.len() >= 4 {
            near_edge
        } else {
            Vec::new()
        }
    } else {
        sidebar
    };
    layout.panel_left = if sidebar.is_empty() {
        None
    } else {
        let right = sidebar.iter().map(|s| s.bbox.x2).fold(f64::MIN, f64::max);
        let h = median(sidebar.iter().map(|s| s.bbox.height()).collect()).unwrap_or(0.0);
        // Toolbar icons read as one or two characters in a column right of the sidebar.
        let reach = right + (width * 0.08).max(6.0 * h);
        let icons: Vec<f64> = spans
            .iter()
            .filter(|s| {
                let c = center(&s.bbox);
                s.text.trim().chars().count() <= 2 && c.0 > right && c.0 < reach && in_area(s)
            })
            .map(|s| s.bbox.x2)
            .collect();
        Some(if icons.len() >= 2 {
            icons.iter().copied().fold(f64::MIN, f64::max) + width * 0.01
        } else {
            right + (width * 0.035).max(4.0 * h)
        })
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
    layout.top_bar = spans
        .iter()
        .filter(|s| {
            logo.iter()
                .any(|l| same_line(&l.bbox, &s.bbox) && s.bbox.x1 <= l.bbox.x2 + width * 0.3)
        })
        .map(|s| s.bbox)
        .reduce(union)
        .map(|b| {
            BBox::new(
                b.x1 - width * 0.005,
                b.y1 - height * 0.012,
                b.x2 + width * 0.005,
                b.y2 + height * 0.012,
            )
        });

    let evidence =
        layout.share_area.is_some() || layout.panel_left.is_some() || layout.top_bar.is_some();
    let bottom = match layout.zoom_top {
        Some(z) if z > area.y1 + area.height() * 0.5 => area.y2.min(z - height * 0.005),
        _ => area.y2,
    };
    let evidence = evidence || layout.zoom_top.is_some();
    layout.canvas = evidence
        .then(|| {
            BBox::new(
                layout.panel_left.map_or(area.x1, |p| p.max(area.x1)),
                area.y1,
                area.x2,
                bottom,
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
            || layout.top_bar.is_some_and(|t| contains_point(&t, c))
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
            bg_luma: None,
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
        // The top bar floats over the canvas: masked, not cut off.
        assert!(canvas.y1 < 70.0, "{canvas:?}");
        assert!(l.top_bar.is_some_and(|t| t.x2 < 300.0));
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
    fn generic_sidebar_and_zoom_bound_the_canvas() {
        let spans = vec![
            span("Ada Quill (Presenting)", 300.0, 20.0, 460.0, 34.0),
            span("Folders", 10.0, 60.0, 70.0, 72.0),
            span("Drafts", 10.0, 90.0, 60.0, 102.0),
            span("Recent", 10.0, 120.0, 60.0, 132.0),
            span("New frame", 10.0, 150.0, 90.0, 162.0),
            span("Order Service", 300.0, 200.0, 420.0, 214.0),
            span("100%", 700.0, 560.0, 740.0, 572.0),
        ];
        let l = analyze(&spans, 1000.0, 600.0, &LayoutConfig::default());
        let c = l.canvas.unwrap();
        assert!(c.x1 > 90.0 && c.x1 < 300.0, "{c:?}");
        assert!(c.y2 < 560.0 && c.y2 > 500.0, "{c:?}");
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
    fn cut_labels_at_the_edge_form_a_column() {
        let spans = vec![
            span("Rexa", 960.0, 60.0, 1000.0, 72.0),
            span("Ivy Mo", 962.0, 200.0, 1000.0, 212.0),
            span("Ivy", 400.0, 200.0, 430.0, 212.0),
            span("Rexa", 400.0, 300.0, 440.0, 312.0),
        ];
        let l = analyze(&spans, 1000.0, 600.0, &LayoutConfig::default());
        assert_eq!(l.tiles.len(), 2, "{:?}", l.tiles);
        assert!(l.share_area.unwrap().x2 < 960.0);
        // Board words in the middle stay content.
        assert_eq!(l.reasons[2], None);
        assert_eq!(l.reasons[3], None);
    }

    #[test]
    fn labels_on_light_background_are_not_tiles() {
        let mut spans = vec![
            span("Rexa Holt", 910.0, 120.0, 980.0, 132.0),
            span("Ivy Moss Park", 905.0, 250.0, 990.0, 262.0),
        ];
        let l = analyze(&spans, 1000.0, 600.0, &LayoutConfig::default());
        assert_eq!(l.tiles.len(), 2);
        for s in &mut spans {
            s.bg_luma = Some(235.0);
        }
        let l = analyze(&spans, 1000.0, 600.0, &LayoutConfig::default());
        assert!(l.tiles.is_empty());
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
        assert!(loose_name("Jane van"));
        assert!(!loose_name("Jane"));
        assert!(single_name_word("Janet"));
        assert!(!single_name_word("Jo"));
        assert!(!single_name_word("12 members"));
        assert!(name_shaped("Jane Smith").is_some());
        assert!(name_shaped("Jane van Smi...").is_some());
        assert!(name_shaped("Pending tasks").is_none());
        assert!(name_shaped("Pile of old").is_none());
        assert!(name_shaped("Jane").is_none());
        assert!(name_shaped("Order 66 Service").is_none());
        assert!(names_match("Jane van Smi...", "Jane van Smith"));
        assert!(names_match("Janevan Smi", "Jane van Smith"));
        assert!(!names_match("Jonc", "Jonas Berg"));
        assert!(names_match("Jonas Bery", "Jonas Berg"));
        assert!(!names_match("Jane Smith", "John Doe"));
        assert!(ui_term_match("Overveiw", "Overview"));
        assert!(ui_term_match("+ Create secton", "Create section"));
        assert!(ui_term_match("8 Browse", "Browse"));
        assert!(!ui_term_match("Order Service", "Overview"));
        assert!(is_zoom_control("-53% +"));
        assert!(is_zoom_control("100 %"));
        assert!(!is_zoom_control("50% done"));
        assert!(is_clock("12:58 PM | Title"));
        assert!(!is_clock("Q3: plan"));
    }
}
