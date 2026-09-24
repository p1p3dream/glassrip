//! DB (differentiable binarization) post-processing: text boxes from the
//! detector's probability map.
//!
//! The map is thresholded, split into 8-connected components, and each
//! component becomes an axis-aligned box scored by its mean probability. Boxes
//! are expanded ("unclipped") by `area * ratio / perimeter` like PaddleOCR, then
//! mapped from detector pixels to source pixels.

use crate::{OcrConfig, PixelBox};

/// A detected text region in source pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetBox {
    pub bbox: PixelBox,
    pub score: f64,
}

/// Extract text boxes from a `map_w` x `map_h` probability map.
///
/// `scale_x` and `scale_y` map detector pixels to source pixels; the result is
/// clamped to `source_w` x `source_h`. Boxes are sorted top to bottom, then
/// left to right.
#[allow(clippy::too_many_arguments)]
pub fn boxes_from_map(
    prob: &[f32],
    map_w: usize,
    map_h: usize,
    scale_x: f64,
    scale_y: f64,
    source_w: u32,
    source_h: u32,
    cfg: &OcrConfig,
) -> Vec<DetBox> {
    if map_w == 0 || map_h == 0 || prob.len() < map_w * map_h {
        return Vec::new();
    }
    let mut label = vec![0u32; map_w * map_h];
    let mut next = 0u32;
    let mut out = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..map_w * map_h {
        if label[start] != 0 || prob[start] <= cfg.det_threshold {
            continue;
        }
        next += 1;
        label[start] = next;
        stack.clear();
        stack.push(start);
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        let mut sum = 0f64;
        let mut count = 0usize;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % map_w, i / map_w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            sum += f64::from(prob[i]);
            count += 1;
            let xs = x.saturating_sub(1)..=(x + 1).min(map_w - 1);
            for ny in y.saturating_sub(1)..=(y + 1).min(map_h - 1) {
                for nx in xs.clone() {
                    let j = ny * map_w + nx;
                    if label[j] == 0 && prob[j] > cfg.det_threshold {
                        label[j] = next;
                        stack.push(j);
                    }
                }
            }
        }
        let w = (x1 - x0 + 1) as f64;
        let h = (y1 - y0 + 1) as f64;
        if w.min(h) < f64::from(cfg.min_box_side) {
            continue;
        }
        let score = sum / count as f64;
        if score < f64::from(cfg.box_threshold) {
            continue;
        }
        let d = (w * h * cfg.unclip_ratio) / (2.0 * (w + h));
        let (sw, sh) = (f64::from(source_w), f64::from(source_h));
        let bbox = PixelBox {
            x1: ((x0 as f64 - d) * scale_x).clamp(0.0, sw),
            y1: ((y0 as f64 - d) * scale_y).clamp(0.0, sh),
            x2: ((x1 as f64 + 1.0 + d) * scale_x).clamp(0.0, sw),
            y2: ((y1 as f64 + 1.0 + d) * scale_y).clamp(0.0, sh),
        };
        if bbox.width() >= 1.0 && bbox.height() >= 1.0 {
            out.push(DetBox { bbox, score });
        }
    }
    reading_order(&mut out, |b| b.bbox);
    out
}

/// Sort boxes into reading order: lines top to bottom, left to right within a
/// line. A box joins the current line when its vertical center lies within half
/// the line's first box height of that box's center.
pub fn reading_order<T>(items: &mut Vec<T>, bbox: impl Fn(&T) -> PixelBox) {
    let cy = |b: &PixelBox| (b.y1 + b.y2) / 2.0;
    items.sort_by(|a, b| cy(&bbox(a)).total_cmp(&cy(&bbox(b))));
    let mut lines: Vec<Vec<T>> = Vec::new();
    let mut anchor: Option<PixelBox> = None;
    for item in items.drain(..) {
        let b = bbox(&item);
        match (anchor, lines.last_mut()) {
            (Some(a), Some(line)) if (cy(&b) - cy(&a)).abs() <= a.height() / 2.0 => line.push(item),
            _ => {
                anchor = Some(b);
                lines.push(vec![item]);
            }
        }
    }
    for mut line in lines {
        line.sort_by(|a, b| bbox(a).x1.total_cmp(&bbox(b).x1));
        items.extend(line);
    }
}

/// Join same-line fragments separated by less than `max_gap` times the text
/// height (a word split by wide letter spacing). Input and output are in
/// reading order; merged spans keep the lower confidence.
pub fn merge_line_fragments(
    spans: Vec<crate::RecognizedSpan>,
    max_gap: f64,
) -> Vec<crate::RecognizedSpan> {
    let mut out: Vec<crate::RecognizedSpan> = Vec::with_capacity(spans.len());
    for s in spans {
        if let Some(prev) = out.last_mut() {
            let (h1, h2) = (prev.bbox.height(), s.bbox.height());
            let cy = |b: &PixelBox| (b.y1 + b.y2) / 2.0;
            let same_line = (cy(&prev.bbox) - cy(&s.bbox)).abs() <= h1.min(h2) / 2.0;
            let similar = h1.max(h2) <= 1.5 * h1.min(h2).max(1.0);
            let gap = s.bbox.x1 - prev.bbox.x2;
            if same_line && similar && gap >= -h1.min(h2) && gap <= max_gap * h1.max(h2) {
                prev.text = format!("{} {}", prev.text, s.text);
                prev.bbox = PixelBox {
                    x1: prev.bbox.x1.min(s.bbox.x1),
                    y1: prev.bbox.y1.min(s.bbox.y1),
                    x2: prev.bbox.x2.max(s.bbox.x2),
                    y2: prev.bbox.y2.max(s.bbox.y2),
                };
                prev.confidence = prev.confidence.min(s.confidence);
                prev.det_score = prev.det_score.min(s.det_score);
                continue;
            }
        }
        out.push(s);
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn map_with(rects: &[(usize, usize, usize, usize, f32)], w: usize, h: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; w * h];
        for &(x0, y0, x1, y1, v) in rects {
            for y in y0..y1 {
                for x in x0..x1 {
                    m[y * w + x] = v;
                }
            }
        }
        m
    }

    #[test]
    fn finds_separate_components_in_reading_order() {
        let m = map_with(
            &[
                (40, 30, 80, 40, 0.9),
                (5, 10, 30, 18, 0.95),
                (5, 30, 20, 38, 0.8),
            ],
            100,
            50,
        );
        let cfg = OcrConfig::default();
        let boxes = boxes_from_map(&m, 100, 50, 2.0, 2.0, 200, 100, &cfg);
        assert_eq!(boxes.len(), 3);
        assert!(boxes[0].bbox.y1 < boxes[1].bbox.y1);
        assert!(boxes[1].bbox.x1 < boxes[2].bbox.x1);
        // Unclip grows the box beyond the raw component (10..18 rows -> 20..36 px).
        assert!(boxes[0].bbox.y1 < 20.0 && boxes[0].bbox.y2 > 36.0);
    }

    #[test]
    fn merges_close_fragments_on_one_line_only() {
        let s = |t: &str, x1: f64, x2: f64, y1: f64| crate::RecognizedSpan {
            text: t.into(),
            bbox: PixelBox {
                x1,
                y1,
                x2,
                y2: y1 + 20.0,
            },
            confidence: 0.9,
            det_score: 0.9,
        };
        let out = merge_line_fragments(
            vec![
                s("Riley", 100.0, 160.0, 50.0),
                s("Park", 168.0, 210.0, 51.0),
                s("REST", 400.0, 450.0, 50.0),
                s("Next", 100.0, 150.0, 90.0),
            ],
            0.5,
        );
        let texts: Vec<&str> = out.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, vec!["Riley Park", "REST", "Next"]);
        assert_eq!(out[0].bbox.x2, 210.0);
    }

    #[test]
    fn drops_weak_and_tiny_regions() {
        let m = map_with(&[(10, 10, 40, 20, 0.45), (60, 10, 62, 30, 0.99)], 100, 50);
        let boxes = boxes_from_map(&m, 100, 50, 1.0, 1.0, 100, 50, &OcrConfig::default());
        assert!(boxes.is_empty(), "{boxes:?}");
    }

    #[test]
    fn clamps_to_source_and_handles_empty_map() {
        let m = map_with(&[(0, 0, 20, 10, 0.9)], 20, 10);
        let boxes = boxes_from_map(&m, 20, 10, 1.0, 1.0, 20, 10, &OcrConfig::default());
        assert_eq!(boxes.len(), 1);
        let b = boxes[0].bbox;
        assert!(b.x1 >= 0.0 && b.y1 >= 0.0 && b.x2 <= 20.0 && b.y2 <= 10.0);
        assert!(boxes_from_map(&[], 0, 0, 1.0, 1.0, 1, 1, &OcrConfig::default()).is_empty());
    }
}
