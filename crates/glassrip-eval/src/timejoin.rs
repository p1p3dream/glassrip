//! Joining golden times to predicted keyframes (spec 9.2).
//!
//! Golden `t_rep` values sit on the prototype's nominal 2 s grid, while Rust
//! keyframe times come from PTS on a VFR stream. A golden time joins to the
//! keyframe whose `[t_start_s, t_end_s)` contains it; if none does, to the
//! keyframe with the nearest `t_rep_s` within the tolerance (2.0 s by default,
//! `eval.time_join_tolerance_s`); otherwise it is missed. Ties on distance go to
//! the earlier keyframe.

use serde::{Deserialize, Serialize};

/// The time fields of one predicted keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyframeSpan {
    /// Stable keyframe id.
    pub keyframe_id: String,
    /// Segment start (inclusive), seconds.
    pub t_start_s: f64,
    /// Segment end (exclusive), seconds.
    pub t_end_s: f64,
    /// Representative frame time, seconds.
    pub t_rep_s: f64,
}

/// Default join tolerance, seconds.
pub const DEFAULT_JOIN_TOLERANCE_S: f64 = 2.0;

/// Index of the keyframe a golden time joins to, per the 9.2 rule.
pub fn join_time(t: f64, keyframes: &[KeyframeSpan], tolerance_s: f64) -> Option<usize> {
    if let Some(i) = keyframes
        .iter()
        .position(|k| k.t_start_s <= t && t < k.t_end_s)
    {
        return Some(i);
    }
    let mut best: Option<(f64, usize)> = None;
    for (i, k) in keyframes.iter().enumerate() {
        let d = (k.t_rep_s - t).abs();
        if d <= tolerance_s && best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, i));
        }
    }
    best.map(|(_, i)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kf(id: &str, s: f64, e: f64, r: f64) -> KeyframeSpan {
        KeyframeSpan {
            keyframe_id: id.into(),
            t_start_s: s,
            t_end_s: e,
            t_rep_s: r,
        }
    }

    #[test]
    fn containing_segment_wins() {
        let k = vec![kf("a", 0.0, 4.0, 2.0), kf("b", 4.0, 24.0, 6.0)];
        assert_eq!(join_time(0.0, &k, 2.0), Some(0));
        assert_eq!(join_time(4.0, &k, 2.0), Some(1)); // start inclusive, end exclusive
        assert_eq!(join_time(23.9, &k, 2.0), Some(1));
    }

    #[test]
    fn nearest_t_rep_within_tolerance() {
        // gap between 10 and 20; t_rep of "b" is 21.
        let k = vec![kf("a", 0.0, 10.0, 9.0), kf("b", 20.0, 30.0, 21.0)];
        assert_eq!(join_time(10.5, &k, 2.0), Some(0)); // |9-10.5| = 1.5
        assert_eq!(join_time(19.5, &k, 2.0), Some(1)); // |21-19.5| = 1.5
        assert_eq!(join_time(15.0, &k, 2.0), None);
        // exactly at tolerance joins; tie goes to the earlier keyframe
        let t = vec![kf("a", 0.0, 1.0, 5.0), kf("b", 20.0, 30.0, 9.0)];
        assert_eq!(join_time(7.0, &t, 2.0), Some(0));
    }

    #[test]
    fn empty_keyframes() {
        assert_eq!(join_time(1.0, &[], 2.0), None);
    }
}
