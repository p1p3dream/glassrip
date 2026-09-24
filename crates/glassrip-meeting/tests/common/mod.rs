//! Synthetic whiteboard rendering for tests: fictional boxes, connectors, arrowheads,
//! and label blocks drawn with imageproc. No real frames or text.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use glassrip_vision::BBox;
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_line_segment_mut, draw_polygon_mut};
use imageproc::point::Point;
use imageproc::rect::Rect;

pub const INK: Rgb<u8> = Rgb([40, 40, 48]);
pub const PAPER: Rgb<u8> = Rgb([250, 250, 246]);

pub struct Canvas {
    pub img: RgbImage,
}

impl Canvas {
    pub fn new(w: u32, h: u32) -> Self {
        Self {
            img: RgbImage::from_pixel(w, h, PAPER),
        }
    }

    /// Outlined node box (2 px border).
    pub fn node(&mut self, b: BBox) {
        for k in 0..2 {
            let k = k as f64;
            self.line((b.x1 + k, b.y1 + k), (b.x2 - k, b.y1 + k), 1);
            self.line((b.x1 + k, b.y2 - k), (b.x2 - k, b.y2 - k), 1);
            self.line((b.x1 + k, b.y1 + k), (b.x1 + k, b.y2 - k), 1);
            self.line((b.x2 - k, b.y1 + k), (b.x2 - k, b.y2 - k), 1);
        }
        // Text-like glyph blocks inside the box.
        let cx = (b.x1 + b.x2) / 2.0;
        let cy = (b.y1 + b.y2) / 2.0;
        self.glyphs(cx - 30.0, cy - 5.0, 60.0);
    }

    /// Short dark blocks that look like a word to a threshold.
    pub fn glyphs(&mut self, x: f64, y: f64, len: f64) {
        let mut gx = x;
        while gx < x + len {
            draw_filled_rect_mut(
                &mut self.img,
                Rect::at(gx as i32, y as i32).of_size(5, 9),
                INK,
            );
            gx += 8.0;
        }
    }

    /// Line of the given thickness.
    pub fn line(&mut self, a: (f64, f64), b: (f64, f64), thickness: u32) {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = (dx * dx + dy * dy).sqrt().max(1e-9);
        let (nx, ny) = (-dy / len, dx / len);
        let t = thickness as f64;
        let mut o = -(t - 1.0) / 2.0;
        while o <= (t - 1.0) / 2.0 + 1e-9 {
            draw_line_segment_mut(
                &mut self.img,
                ((a.0 + nx * o) as f32, (a.1 + ny * o) as f32),
                ((b.0 + nx * o) as f32, (b.1 + ny * o) as f32),
                INK,
            );
            o += 1.0;
        }
    }

    /// Dashed line.
    pub fn dashed(&mut self, a: (f64, f64), b: (f64, f64), dash: f64, gap: f64) {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = (dx * dx + dy * dy).sqrt();
        let (ux, uy) = (dx / len, dy / len);
        let mut s = 0.0;
        while s < len {
            let e = (s + dash).min(len);
            self.line(
                (a.0 + ux * s, a.1 + uy * s),
                (a.0 + ux * e, a.1 + uy * e),
                2,
            );
            s += dash + gap;
        }
    }

    /// Filled triangular arrowhead with its tip at `tip`, pointing along `from -> tip`.
    pub fn arrowhead(&mut self, from: (f64, f64), tip: (f64, f64), len: f64, half: f64) {
        let (dx, dy) = (tip.0 - from.0, tip.1 - from.1);
        let l = (dx * dx + dy * dy).sqrt();
        let (ux, uy) = (dx / l, dy / l);
        let base = (tip.0 - ux * len, tip.1 - uy * len);
        let (nx, ny) = (-uy, ux);
        let pts = [
            Point::new(tip.0.round() as i32, tip.1.round() as i32),
            Point::new(
                (base.0 + nx * half).round() as i32,
                (base.1 + ny * half).round() as i32,
            ),
            Point::new(
                (base.0 - nx * half).round() as i32,
                (base.1 - ny * half).round() as i32,
            ),
        ];
        draw_polygon_mut(&mut self.img, &pts, INK);
    }

    /// A label: paper background over the line plus glyph blocks; returns its box.
    pub fn label(&mut self, cx: f64, cy: f64, len: f64) -> BBox {
        let b = BBox::new(
            cx - len / 2.0 - 3.0,
            cy - 8.0,
            cx + len / 2.0 + 3.0,
            cy + 8.0,
        );
        draw_filled_rect_mut(
            &mut self.img,
            Rect::at(b.x1 as i32, b.y1 as i32).of_size(b.width() as u32, b.height() as u32),
            PAPER,
        );
        self.glyphs(cx - len / 2.0, cy - 4.0, len);
        b
    }

    /// Filled colored square (sticky or owner tag).
    pub fn sticky(&mut self, b: BBox, color: Rgb<u8>) {
        draw_filled_rect_mut(
            &mut self.img,
            Rect::at(b.x1 as i32, b.y1 as i32).of_size(b.width() as u32, b.height() as u32),
            color,
        );
    }

    /// Connector along a polyline with optional arrowheads at either end. The line
    /// stops at the arrowhead base so the head is a solid triangle.
    pub fn connector(
        &mut self,
        pts: &[(f64, f64)],
        head_at_start: bool,
        head_at_end: bool,
        dashed: bool,
    ) {
        let mut p = pts.to_vec();
        let n = p.len();
        let shorten = |a: (f64, f64), b: (f64, f64), by: f64| {
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let l = (dx * dx + dy * dy).sqrt();
            (b.0 - dx / l * by, b.1 - dy / l * by)
        };
        if head_at_end {
            self.arrowhead(pts[n - 2], pts[n - 1], 13.0, 6.0);
            p[n - 1] = shorten(pts[n - 2], pts[n - 1], 10.0);
        }
        if head_at_start {
            self.arrowhead(pts[1], pts[0], 13.0, 6.0);
            p[0] = shorten(pts[1], pts[0], 10.0);
        }
        for w in p.windows(2) {
            if dashed {
                self.dashed(w[0], w[1], 9.0, 6.0);
            } else {
                self.line(w[0], w[1], 2);
            }
        }
    }

    /// Cubic Bezier connector from `p0` to `p3` with a filled head of `(len, half)` at
    /// `p3`, pointing along the final tangent.
    pub fn bezier(&mut self, p: [(f64, f64); 4], head: (f64, f64)) {
        let at = |t: f64| {
            let u = 1.0 - t;
            let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
            (
                a * p[0].0 + b * p[1].0 + c * p[2].0 + d * p[3].0,
                a * p[0].1 + b * p[1].1 + c * p[2].1 + d * p[3].1,
            )
        };
        let pts: Vec<(f64, f64)> = (0..=200).map(|i| at(f64::from(i) / 200.0)).collect();
        // Stop the line where the head's base begins.
        let mut cut = pts.len() - 1;
        while cut > 0 {
            let q = pts[cut];
            if ((q.0 - p[3].0).powi(2) + (q.1 - p[3].1).powi(2)).sqrt() >= head.0 - 2.0 {
                break;
            }
            cut -= 1;
        }
        for w in pts[..=cut].windows(2) {
            self.line(w[0], w[1], 2);
        }
        self.arrowhead(pts[cut.saturating_sub(3)], p[3], head.0, head.1);
    }
}
