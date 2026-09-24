//! Pan/zoom false change events (spec 9.2, 9.3).
//!
//! A static window is a time range in which the board content does not change
//! (only pans and zooms, or nothing), with an optional list of allowed real
//! events (for example an owner move). A predicted change event inside the
//! window, `t_start_s < t_s <= t_end_s + tolerance`, that does not match an
//! allowed event of the same kind within `tolerance` is a false change event.
//! The lower bound is exclusive so the change *into* the window's first
//! keyframe is not counted. Each allowed event absorbs at most one prediction.

use serde::{Deserialize, Serialize};

/// Event tolerance: the metric's own 2.0 s plus the 9.2 join tolerance of 2.0 s.
pub const EVENT_TOLERANCE_S: f64 = 4.0;

/// A predicted change event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredEvent {
    /// Event kind in snake_case (`node_added`, `owner_moved`, ...).
    pub kind: String,
    /// Time, seconds.
    pub t_s: f64,
}

/// An event allowed inside a static window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AllowedEvent {
    /// Event kind in snake_case.
    pub kind: String,
    /// Time, seconds.
    pub t_s: f64,
}

/// A window with no content change except the allowed events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticWindow {
    /// Start, seconds (exclusive for events).
    pub t_start_s: f64,
    /// End, seconds.
    pub t_end_s: f64,
    /// Real events that may occur inside.
    #[serde(default)]
    pub allowed_events: Vec<AllowedEvent>,
    /// Free-text note.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// snake_case key for an event kind: `OwnerMoved`, `owner-moved`, and
/// `owner_moved` are the same kind.
pub fn kind_key(k: &str) -> String {
    let mut out = String::new();
    let mut prev_lower = false;
    for c in k.trim().chars() {
        if c == '-' || c == ' ' || c == '_' {
            if !out.ends_with('_') {
                out.push('_');
            }
            prev_lower = false;
        } else if c.is_uppercase() {
            if prev_lower {
                out.push('_');
            }
            out.extend(c.to_lowercase());
            prev_lower = false;
        } else {
            out.push(c);
            prev_lower = c.is_lowercase() || c.is_ascii_digit();
        }
    }
    out
}

/// Counts false change events and returns the offending events.
pub fn false_change_events(
    windows: &[StaticWindow],
    events: &[PredEvent],
    tolerance_s: f64,
) -> (usize, Vec<PredEvent>) {
    let mut offenders = Vec::new();
    for w in windows {
        let mut used = vec![false; w.allowed_events.len()];
        let mut inside: Vec<&PredEvent> = events
            .iter()
            .filter(|e| e.t_s > w.t_start_s && e.t_s <= w.t_end_s + tolerance_s)
            .collect();
        inside.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
        for e in inside {
            let slot = w.allowed_events.iter().enumerate().position(|(i, a)| {
                !used[i]
                    && kind_key(&a.kind) == kind_key(&e.kind)
                    && (a.t_s - e.t_s).abs() <= tolerance_s
            });
            match slot {
                Some(i) => used[i] = true,
                None => offenders.push(e.clone()),
            }
        }
    }
    (offenders.len(), offenders)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(k: &str, t: f64) -> PredEvent {
        PredEvent {
            kind: k.into(),
            t_s: t,
        }
    }

    #[test]
    fn counts_hand_computed() {
        let windows = vec![
            StaticWindow {
                t_start_s: 342.0,
                t_end_s: 662.0,
                allowed_events: vec![],
                note: None,
            },
            StaticWindow {
                t_start_s: 1808.0,
                t_end_s: 1848.0,
                allowed_events: vec![AllowedEvent {
                    kind: "owner_moved".into(),
                    t_s: 1847.0,
                }],
                note: None,
            },
        ];
        let events = vec![
            ev("node_added", 342.0),    // at the exclusive start: not counted
            ev("node_added", 350.0),    // inside: false
            ev("label_changed", 665.0), // within end + 4: false
            ev("edge_added", 700.0),    // outside
            ev("OwnerMoved", 1849.0),   // same kind as owner_moved; slot already used -> false
            ev("owner_moved", 1848.0),  // allowed (earliest, takes the slot)
            ev("owner_moved", 1850.0),  // allowed slot used -> false
        ];
        let (n, offenders) = false_change_events(&windows, &events, EVENT_TOLERANCE_S);
        assert_eq!(n, 4);
        assert_eq!(kind_key("OwnerMoved"), "owner_moved");
        assert_eq!(kind_key("edge-reversed"), "edge_reversed");
        assert_eq!(offenders[0], ev("node_added", 350.0));
        assert_eq!(offenders[1], ev("label_changed", 665.0));
    }
}
