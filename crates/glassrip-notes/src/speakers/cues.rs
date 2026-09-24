//! Visual active-speaker cues from conferencing tiles.
//!
//! A tile is located by its name label (read by a [`TileReader`]). Conferencing
//! apps draw the active speaker's tile with a thin bright border ("speaking
//! ring"); under the name label the ring shows as a short run of bright, bluish
//! pixels at the tile's bottom edge followed by a sharp drop into the darker gap
//! between tiles. [`ring_score`] measures the fraction of columns under the name
//! that show that profile. Plain white text is not bluish, and a tile edge
//! without a ring drops from ordinary content, so neither counts.

use image::RgbImage;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{FrameObservation, TileSeen};
use crate::people::AliasTable;

/// A line of text read from a frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenText {
    /// Text as read.
    pub text: String,
    /// Box `[x0, y0, x1, y1]` in frame pixels (x1, y1 exclusive).
    pub bbox: [u32; 4],
    /// Recognition confidence in [0, 1].
    pub confidence: f32,
}

/// Reads text lines from a frame (OCR).
pub trait TileReader: Send + Sync {
    /// Text lines with boxes.
    fn read(&self, img: &RgbImage) -> Result<Vec<ScreenText>, String>;
    /// Short description for manifests (engine, provider).
    fn describe(&self) -> String;
}

/// Speaking-ring detector thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RingParams {
    /// Minimum luma of a ring pixel.
    pub min_luma: f32,
    /// Minimum blue minus red of a ring pixel (rings are bluish, text is neutral).
    pub min_blue_minus_red: f32,
    /// Minimum luma drop within `drop_within_px` below the ring pixel.
    pub min_drop: f32,
    /// Pixels below the ring pixel in which the drop must occur.
    pub drop_within_px: u32,
    /// Search depth below the name box, in name-box heights.
    pub search_heights: f32,
    /// Fraction of columns with a ring profile needed for a highlight.
    pub min_column_fraction: f32,
}

impl Default for RingParams {
    fn default() -> Self {
        Self {
            min_luma: 170.0,
            min_blue_minus_red: 12.0,
            min_drop: 70.0,
            drop_within_px: 4,
            search_heights: 2.5,
            min_column_fraction: 0.45,
        }
    }
}

fn luma(p: &image::Rgb<u8>) -> f32 {
    0.299 * f32::from(p[0]) + 0.587 * f32::from(p[1]) + 0.114 * f32::from(p[2])
}

/// Fraction of columns under `name_bbox` that show a speaking-ring profile.
pub fn ring_score(img: &RgbImage, name_bbox: [u32; 4], p: &RingParams) -> f32 {
    let [x0, y0, x1, y1] = name_bbox;
    let (w, h) = img.dimensions();
    if x1 <= x0 || y1 <= y0 || y1 >= h || x0 >= w {
        return 0.0;
    }
    let box_h = (y1 - y0) as f32;
    let depth = (box_h * p.search_heights).max(8.0) as u32;
    let y_end = (y1 + depth).min(h.saturating_sub(p.drop_within_px + 1));
    let mut cols = 0u32;
    let mut hits = 0u32;
    for x in x0..x1.min(w) {
        cols += 1;
        let mut hit = false;
        for y in y1..y_end {
            let px = img.get_pixel(x, y);
            let l = luma(px);
            if l < p.min_luma || f32::from(px[2]) - f32::from(px[0]) < p.min_blue_minus_red {
                continue;
            }
            let dropped =
                (1..=p.drop_within_px).any(|k| luma(img.get_pixel(x, y + k)) <= l - p.min_drop);
            if dropped {
                hit = true;
                break;
            }
        }
        hits += u32::from(hit);
    }
    if cols == 0 {
        0.0
    } else {
        hits as f32 / cols as f32
    }
}

/// Tile matching settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TileParams {
    /// Minimum name-match score for a tile label.
    pub min_match: f64,
    /// Minimum OCR confidence for a tile label.
    pub min_confidence: f32,
    /// Case-insensitive marker of the presenter banner.
    pub presenting_marker: String,
    /// Ring detector.
    pub ring: RingParams,
}

impl Default for TileParams {
    fn default() -> Self {
        Self {
            min_match: 0.85,
            min_confidence: 0.5,
            presenting_marker: "presenting".into(),
            ring: RingParams::default(),
        }
    }
}

/// Builds the observation for one frame from its text lines.
pub fn observe(
    t_s: f64,
    img: &RgbImage,
    texts: &[ScreenText],
    table: &AliasTable,
    p: &TileParams,
) -> FrameObservation {
    let marker = p.presenting_marker.to_lowercase();
    let mut tiles: Vec<TileSeen> = Vec::new();
    let mut presenter = None;
    for t in texts.iter().filter(|t| t.confidence >= p.min_confidence) {
        let lower = t.text.to_lowercase();
        if let Some(pos) = lower.find(&marker) {
            // `pos` indexes the lowercased copy; `get` avoids splitting a character
            // when lowercasing changed byte lengths.
            let name = t
                .text
                .get(..pos)
                .unwrap_or("")
                .trim()
                .trim_end_matches('(')
                .trim();
            if let Some(m) = table.match_screen_text(name) {
                if m.score >= p.min_match {
                    presenter = table.person(m.person).map(|x| x.person_id.clone());
                }
            }
            continue;
        }
        let Some(m) = table.match_screen_text(&t.text) else {
            continue;
        };
        if m.score < p.min_match {
            continue;
        }
        let Some(person) = table.person(m.person) else {
            continue;
        };
        let score = ring_score(img, t.bbox, &p.ring);
        let seen = TileSeen {
            person_id: person.person_id.clone(),
            text: t.text.clone(),
            name_bbox: t.bbox,
            ring_score: score,
            highlighted: score >= p.ring.min_column_fraction,
        };
        // one tile per person: keep the strongest ring
        match tiles.iter_mut().find(|x| x.person_id == seen.person_id) {
            Some(existing) if existing.ring_score >= seen.ring_score => {}
            Some(existing) => *existing = seen,
            None => tiles.push(seen),
        }
    }
    tiles.sort_by(|a, b| a.person_id.cmp(&b.person_id));
    FrameObservation {
        t_s,
        tiles,
        presenter,
        error: None,
    }
}

/// Draws a synthetic tile (for tests and fixtures): a filled rectangle, an
/// optional speaking ring along its border, and a light name bar.
pub fn draw_tile(img: &mut RgbImage, rect: [u32; 4], fill: [u8; 3], ring: bool) {
    let [x0, y0, x1, y1] = rect;
    for y in y0..y1.min(img.height()) {
        for x in x0..x1.min(img.width()) {
            let on_border = ring && (y < y0 + 3 || y + 3 >= y1 || x < x0 + 3 || x + 3 >= x1);
            let c = if on_border { [210, 222, 248] } else { fill };
            img.put_pixel(x, y, image::Rgb(c));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> RgbImage {
        let mut img = RgbImage::from_pixel(400, 200, image::Rgb([34, 35, 38]));
        draw_tile(&mut img, [10, 10, 190, 120], [70, 90, 150], true);
        draw_tile(&mut img, [210, 10, 390, 120], [90, 120, 80], false);
        img
    }

    #[test]
    fn ring_detected_only_on_the_lit_tile() {
        let img = frame();
        let p = RingParams::default();
        // name boxes sit near the bottom-left of each tile
        assert!(ring_score(&img, [20, 92, 120, 108], &p) > 0.9);
        assert!(ring_score(&img, [220, 92, 320, 108], &p) < 0.05);
    }

    #[test]
    fn white_text_is_not_a_ring() {
        let mut img = RgbImage::from_pixel(200, 100, image::Rgb([34, 35, 38]));
        draw_tile(&mut img, [0, 0, 200, 80], [60, 60, 60], false);
        for x in 20..120 {
            img.put_pixel(x, 70, image::Rgb([245, 245, 245]));
        }
        assert!(ring_score(&img, [20, 50, 120, 66], &RingParams::default()) < 0.05);
    }

    #[test]
    fn observe_matches_names_banner_and_highlight() {
        let img = frame();
        let table = AliasTable::from_names(&["Avery Quinn", "Rohan Dasgupta"]);
        let texts = vec![
            ScreenText {
                text: "Avery Quinn".into(),
                bbox: [20, 92, 120, 108],
                confidence: 0.95,
            },
            ScreenText {
                text: "Rohan Dasgu...".into(),
                bbox: [220, 92, 320, 108],
                confidence: 0.9,
            },
            ScreenText {
                text: "Rohan Dasgupta (Presenting)".into(),
                bbox: [0, 150, 200, 170],
                confidence: 0.9,
            },
            ScreenText {
                text: "Overview".into(),
                bbox: [0, 180, 80, 195],
                confidence: 0.99,
            },
        ];
        let obs = observe(12.5, &img, &texts, &table, &TileParams::default());
        assert_eq!(obs.visible(), vec!["avery-quinn", "rohan-dasgupta"]);
        assert_eq!(obs.highlighted(), vec!["avery-quinn"]);
        assert_eq!(obs.presenter.as_deref(), Some("rohan-dasgupta"));
    }
}
