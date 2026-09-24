//! Bounding boxes shared by classification and board reading.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::image_prep::PreparedImage;

/// Axis-aligned box in pixels: top-left `(x1, y1)`, bottom-right `(x2, y2)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BBox {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl BBox {
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self { x1, y1, x2, y2 }
    }

    pub fn width(&self) -> f64 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f64 {
        self.y2 - self.y1
    }

    pub fn area(&self) -> f64 {
        self.width().max(0.0) * self.height().max(0.0)
    }

    /// Finite coordinates with positive width and height.
    pub fn is_well_formed(&self) -> bool {
        [self.x1, self.y1, self.x2, self.y2]
            .iter()
            .all(|v| v.is_finite())
            && self.x2 > self.x1
            && self.y2 > self.y1
    }

    /// True when the box lies within `[0, width] x [0, height]`, allowing `tolerance` px of slack.
    pub fn is_inside(&self, width: f64, height: f64, tolerance: f64) -> bool {
        self.x1 >= -tolerance
            && self.y1 >= -tolerance
            && self.x2 <= width + tolerance
            && self.y2 <= height + tolerance
    }

    /// Scale every coordinate.
    pub fn scaled(&self, sx: f64, sy: f64) -> Self {
        Self::new(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)
    }

    /// Clamp every coordinate into `[0, width] x [0, height]`.
    pub fn clamped(&self, width: f64, height: f64) -> Self {
        Self::new(
            self.x1.clamp(0.0, width),
            self.y1.clamp(0.0, height),
            self.x2.clamp(0.0, width),
            self.y2.clamp(0.0, height),
        )
    }

    /// Map a box in sent-image pixels back to source (canvas or frame) pixels.
    pub fn to_source(&self, prepared: &PreparedImage) -> Self {
        let (x1, y1) = prepared.to_source(self.x1, self.y1);
        let (x2, y2) = prepared.to_source(self.x2, self.y2);
        Self::new(x1, y1, x2, y2)
    }

    /// Intersection over union.
    pub fn iou(&self, other: &Self) -> f64 {
        let ix = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let iy = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        let inter = ix * iy;
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_and_well_formed() {
        let b = BBox::new(10.0, 10.0, 100.0, 50.0);
        assert!(b.is_well_formed());
        assert!(b.is_inside(100.0, 50.0, 0.0));
        assert!(!b.is_inside(99.0, 50.0, 0.0));
        assert!(b.is_inside(99.0, 50.0, 2.0));
        assert!(!BBox::new(5.0, 5.0, 5.0, 9.0).is_well_formed());
        assert!(!BBox::new(0.0, f64::NAN, 1.0, 1.0).is_well_formed());
        let c = BBox::new(-5.0, 10.0, 120.0, 90.0).clamped(100.0, 50.0);
        assert_eq!(c, BBox::new(0.0, 10.0, 100.0, 50.0));
    }

    #[test]
    fn iou_basics() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((a.iou(&a) - 1.0).abs() < 1e-9);
        let b = BBox::new(5.0, 0.0, 15.0, 10.0);
        assert!((a.iou(&b) - 50.0 / 150.0).abs() < 1e-9);
        assert_eq!(a.iou(&BBox::new(20.0, 20.0, 30.0, 30.0)), 0.0);
    }
}
