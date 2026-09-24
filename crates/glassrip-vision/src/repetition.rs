//! Repetition-loop detection on model output (streamed or returned).
//!
//! A vision model at temperature 0 sometimes falls into a loop: it keeps emitting
//! list items that differ only in an id or coordinate stepped by the same amount
//! each time (`{"src": "n15", "dst": "n16", ...}`, `{"src": "n16", "dst": "n17",
//! ...}`), or restates one item verbatim, until the output limit cuts it off. The
//! detector reads the (possibly incomplete) JSON text and reports a loop when one
//! array holds either
//!
//! - **a templated run**: at least [`RepetitionParams::min_templated_run`]
//!   consecutive items that are identical once their numbers are removed, and
//!   whose numbers change by the same non-zero step from each item to the next; or
//! - **verbatim repeats**: one item, byte for byte, at least
//!   [`RepetitionParams::min_exact_repeats`] times.
//!
//! Both are cheap to check on partial text, so a streaming client can stop the
//! generation early. The defaults are deliberately loose: a real board rarely lists
//! six consecutive elements that differ only by evenly stepped ids and positions,
//! and never lists the same element three times, while a detected loop only costs
//! one retry with a repeat penalty (see `board_read`).

use serde::{Deserialize, Serialize};

/// Detector thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RepetitionParams {
    /// Consecutive templated items (same text apart from evenly stepped numbers)
    /// that make a loop.
    pub min_templated_run: usize,
    /// Verbatim copies of one item in one array that make a loop.
    pub min_exact_repeats: usize,
    /// A streaming client re-checks after at least this many new bytes.
    pub check_every_bytes: usize,
}

impl Default for RepetitionParams {
    fn default() -> Self {
        Self {
            min_templated_run: 6,
            min_exact_repeats: 3,
            check_every_bytes: 256,
        }
    }
}

/// Which rule fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RepetitionKind {
    /// Consecutive items with evenly stepped numbers.
    Templated,
    /// One item repeated verbatim.
    Exact,
}

/// A detected loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RepetitionFinding {
    /// Rule that fired.
    pub kind: RepetitionKind,
    /// Items in the run (templated) or copies of the item (exact).
    pub repeats: usize,
    /// The repeated item with its numbers replaced by `#`, at most 160 characters.
    pub pattern: String,
    /// Output length, in bytes, when the loop was detected.
    pub at_bytes: usize,
}

impl std::fmt::Display for RepetitionFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.kind {
            RepetitionKind::Templated => "templated items",
            RepetitionKind::Exact => "verbatim copies",
        };
        write!(
            f,
            "{} {kind} after {} bytes: {}",
            self.repeats, self.at_bytes, self.pattern
        )
    }
}

/// One complete object item of an array, as source text.
#[derive(Debug, Clone)]
struct Item<'a> {
    /// Start offset of the enclosing array (items of one array share it).
    array: usize,
    text: &'a str,
}

/// Complete objects whose parent is an array, in order. Tolerates a cut-off tail.
fn array_items(text: &str) -> Vec<Item<'_>> {
    #[derive(Clone, Copy, PartialEq)]
    enum C {
        Obj(usize),
        Arr(usize),
    }
    let bytes = text.as_bytes();
    let mut stack: Vec<C> = Vec::new();
    let mut out = Vec::new();
    let (mut in_str, mut esc) = (false, false);
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => stack.push(C::Obj(i)),
            b'[' => stack.push(C::Arr(i)),
            b'}' => {
                if let Some(C::Obj(start)) = stack.pop() {
                    if let Some(C::Arr(a)) = stack.last() {
                        out.push(Item {
                            array: *a,
                            text: &text[start..=i],
                        });
                    }
                }
            }
            b']' => {
                stack.pop();
            }
            _ => {}
        }
    }
    out
}

/// The item with every number replaced by `#`, whitespace removed, and the numbers
/// in order. Digits inside words (`n15`, `v2`) count as numbers too: loops step ids.
fn split_numbers(item: &str) -> (String, Vec<f64>) {
    let mut shape = String::with_capacity(item.len());
    let mut nums = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, shape: &mut String, nums: &mut Vec<f64>| {
        if !cur.is_empty() {
            nums.push(cur.parse::<f64>().unwrap_or(0.0));
            shape.push('#');
            cur.clear();
        }
    };
    let mut prev: Option<char> = None;
    for c in item.chars() {
        let starts_negative =
            c == '-' && cur.is_empty() && !prev.is_some_and(|p| p.is_alphanumeric());
        if c.is_ascii_digit() || (c == '.' && !cur.is_empty()) || starts_negative {
            cur.push(c);
        } else {
            flush(&mut cur, &mut shape, &mut nums);
            if !c.is_whitespace() {
                shape.push(c);
            }
        }
        prev = Some(c);
    }
    flush(&mut cur, &mut shape, &mut nums);
    (shape, nums)
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n).collect();
        t.push_str("...");
        t
    }
}

/// Look for a loop in `text` (complete or cut-off JSON).
pub fn detect(text: &str, p: &RepetitionParams) -> Option<RepetitionFinding> {
    let items = array_items(text);
    let at_bytes = text.len();
    // Verbatim repeats, per array.
    if p.min_exact_repeats >= 2 {
        let mut counts: std::collections::HashMap<(usize, String), usize> =
            std::collections::HashMap::new();
        for it in &items {
            let key: String = it.text.chars().filter(|c| !c.is_whitespace()).collect();
            let n = counts.entry((it.array, key.clone())).or_default();
            *n += 1;
            if *n >= p.min_exact_repeats {
                return Some(RepetitionFinding {
                    kind: RepetitionKind::Exact,
                    repeats: *n,
                    pattern: truncate(&split_numbers(&key).0, 160),
                    at_bytes,
                });
            }
        }
    }
    // Templated runs: consecutive items of one array with one shape and a constant,
    // non-zero step between their number vectors.
    if p.min_templated_run >= 2 {
        let parsed: Vec<(usize, String, Vec<f64>)> = items
            .iter()
            .map(|it| {
                let (shape, nums) = split_numbers(it.text);
                (it.array, shape, nums)
            })
            .collect();
        let mut run = 1usize;
        let mut step: Option<Vec<f64>> = None;
        for w in parsed.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let same = a.0 == b.0 && a.1 == b.1 && a.2.len() == b.2.len();
            let d: Option<Vec<f64>> =
                same.then(|| b.2.iter().zip(&a.2).map(|(x, y)| x - y).collect());
            let nonzero = d.as_ref().is_some_and(|d| d.iter().any(|v| v.abs() > 1e-9));
            let continues = match (&d, &step) {
                (Some(d), Some(s)) if nonzero => d.iter().zip(s).all(|(x, y)| (x - y).abs() < 1e-6),
                _ => false,
            };
            if continues {
                run += 1;
            } else if nonzero {
                run = 2;
                step = d;
            } else {
                run = 1;
                step = None;
            }
            if run >= p.min_templated_run {
                return Some(RepetitionFinding {
                    kind: RepetitionKind::Templated,
                    repeats: run,
                    pattern: truncate(&b.1, 160),
                    at_bytes,
                });
            }
        }
    }
    None
}

/// Incremental check for a streaming client: re-runs [`detect`] once at least
/// `check_every_bytes` new bytes arrived since the last check.
#[derive(Debug, Clone)]
pub struct StreamGuard {
    params: RepetitionParams,
    checked_at: usize,
}

impl StreamGuard {
    /// Guard with `params`.
    pub fn new(params: RepetitionParams) -> Self {
        Self {
            params,
            checked_at: 0,
        }
    }

    /// Check `text` (the whole output so far) when enough new bytes arrived.
    pub fn feed(&mut self, text: &str) -> Option<RepetitionFinding> {
        if text.len() < self.checked_at + self.params.check_every_bytes.max(1) {
            return None;
        }
        self.checked_at = text.len();
        detect(text, &self.params)
    }

    /// Final check on the complete output.
    pub fn finish(&mut self, text: &str) -> Option<RepetitionFinding> {
        self.checked_at = text.len();
        detect(text, &self.params)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn edge(src: u32, dst: u32, label: &str, bbox: [u32; 4]) -> String {
        format!(
            "{{\"src\": \"n{src}\", \"dst\": \"n{dst}\", \"label\": \"{label}\", \"label_bbox_2d\": [{}, {}, {}, {}], \"style\": \"solid\", \"conf\": 0.9}}",
            bbox[0], bbox[1], bbox[2], bbox[3]
        )
    }

    fn board(nodes: &[String], edges: &[String], closed: bool) -> String {
        let mut s = format!(
            "{{\n  \"nodes\": [\n    {}\n  ],\n  \"edges\": [\n    {}",
            nodes.join(",\n    "),
            edges.join(",\n    ")
        );
        if closed {
            s.push_str("\n  ],\n  \"stickies\": []\n}");
        }
        s
    }

    fn node(i: u32, text: &str, x: u32, y: u32) -> String {
        format!(
            "{{\"local_id\": \"n{i}\", \"text\": \"{text}\", \"bbox_2d\": [{x}, {y}, {}, {}], \"conf\": 0.9}}",
            x + 80,
            y + 30
        )
    }

    fn real_nodes() -> Vec<String> {
        vec![
            node(1, "Ledger API", 100, 40),
            node(2, "Orbit Queue", 400, 40),
            node(3, "Quill Store", 100, 300),
            node(4, "Web Shell", 400, 300),
        ]
    }

    #[test]
    fn stepped_id_loop_in_a_cut_off_reply_is_found() {
        let mut edges = vec![edge(1, 2, "REST", [240, 50, 300, 70])];
        for k in 15..30 {
            edges.push(edge(k, k + 1, "Relay", [1340, 320, 1420, 390]));
        }
        let text = board(&real_nodes(), &edges, false);
        let f = detect(&text, &RepetitionParams::default()).unwrap();
        assert_eq!(f.kind, RepetitionKind::Templated);
        assert_eq!(f.repeats, 6);
        assert!(f.pattern.contains("Relay"), "{f:?}");
    }

    #[test]
    fn fan_out_loop_is_found() {
        // One source, destination ids counting up, empty labels.
        let mut edges = Vec::new();
        for k in 20..30 {
            edges.push(edge(8, k, "", [0, 0, 0, 0]));
        }
        let text = board(&real_nodes(), &edges, false);
        assert!(detect(&text, &RepetitionParams::default()).is_some());
    }

    #[test]
    fn verbatim_copies_are_found() {
        let e = edge(1, 2, "REST", [240, 50, 300, 70]);
        let text = board(
            &real_nodes(),
            &[e.clone(), edge(2, 3, "", [0; 4]), e.clone(), e],
            false,
        );
        let f = detect(&text, &RepetitionParams::default()).unwrap();
        assert_eq!((f.kind, f.repeats), (RepetitionKind::Exact, 3));
    }

    #[test]
    fn a_real_board_is_not_a_loop() {
        // Unlabelled edges with the same style and irregular ids, and a card grid
        // whose rows wrap: neither is a constant-step run.
        let edges: Vec<String> = [
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (2, 6),
            (6, 7),
            (5, 7),
        ]
        .iter()
        .map(|&(a, b)| edge(a, b, "", [0, 0, 0, 0]))
        .collect();
        let mut nodes = real_nodes();
        for r in 0..3 {
            for c in 0..3 {
                nodes.push(node(10 + r * 3 + c, "Card", 100 + c * 120, 500 + r * 80));
            }
        }
        let text = board(&nodes, &edges, true);
        assert!(detect(&text, &RepetitionParams::default()).is_none());
        // Five evenly spaced identical cards are still under the run threshold.
        let row: Vec<String> = (0..5)
            .map(|c| node(30 + c, "Card", 100 + c * 120, 900))
            .collect();
        assert!(detect(&board(&row, &[], true), &RepetitionParams::default()).is_none());
    }

    #[test]
    fn strings_with_brackets_do_not_confuse_the_scanner() {
        let tricky = "{\"text\": \"a } ] [ { b\", \"local_id\": \"n1\", \"bbox_2d\": [1, 2, 3, 4], \"conf\": 0.9}";
        let text = format!("{{\"nodes\": [{tricky}, {tricky}]}}");
        let items = array_items(&text);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].text, tricky);
    }

    #[test]
    fn stream_guard_checks_in_steps_and_on_finish() {
        let mut edges = Vec::new();
        for k in 15..30 {
            edges.push(edge(k, k + 1, "Relay", [1, 2, 3, 4]));
        }
        let text = board(&real_nodes(), &edges, false);
        let mut g = StreamGuard::new(RepetitionParams::default());
        // Too little new text: no check yet.
        assert!(g.feed(&text[..100]).is_none());
        let mut found = None;
        for end in (0..=text.len()).step_by(64) {
            if let Some(f) = g.feed(&text[..end]) {
                found = Some(f);
                break;
            }
        }
        let f = found.unwrap();
        assert!(f.at_bytes < text.len(), "stopped before the end: {f:?}");
        let mut g = StreamGuard::new(RepetitionParams::default());
        assert!(g.finish(&text).is_some());
    }
}
