//! Repetition-loop detection on model output (streamed or returned).
//!
//! A vision model at temperature 0 sometimes falls into a loop: it keeps emitting
//! list items for the same element under fresh ids (`{"src": "n15", "dst": "n16",
//! ...}`, `{"src": "n16", "dst": "n17", ...}` with one label box), or restates one
//! item verbatim, until the output limit cuts it off. The detector reads the
//! (possibly incomplete) JSON text and reports a loop when one array holds
//!
//! - **a templated run**: at least [`RepetitionParams::min_templated_run`]
//!   consecutive items that are identical once their numbers are removed, whose
//!   numbers change by the same non-zero step from item to item, whose *place*
//!   (the JSON numbers: boxes, confidences) does not move, and whose changed ids
//!   mostly name nothing that another array declares. That is one element
//!   restated under new labels, not a new element;
//! - **a runaway run**: at least [`RepetitionParams::min_spatial_run`] such items
//!   whose place does step evenly. A real row of evenly spaced cards is regular
//!   too, so coordinates in arithmetic progression are never a loop by themselves;
//!   only a run longer than any plausible single row of a board is; or
//! - **verbatim repeats**: one item, byte for byte, at least
//!   [`RepetitionParams::min_exact_repeats`] times.
//!
//! Numbers inside strings (`n15`, `Step 3`) are *labels*; JSON numbers outside
//! strings are the item's *place*. A grid of cards with sequential ids and evenly
//! stepped boxes moves its place, and a chain or fan-out of edges steps ids that
//! the node list declares: neither is a templated loop.
//!
//! Both are cheap to check on partial text, so a streaming client can stop the
//! generation early. A complete reply that validates is accepted whatever the
//! detector says (see `ollama`): the guard exists to stop a runaway generation, not
//! to second-guess a finished answer.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Version of the detection rules. Recorded replies stopped by the guard are only
/// valid under the policy that stopped them (see `raw_store` in the stages crate).
pub const POLICY_VERSION: u32 = 2;

/// Detector thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RepetitionParams {
    /// Consecutive templated items (one element restated under evenly stepped,
    /// undeclared ids, its place unchanged) that make a loop.
    pub min_templated_run: usize,
    /// Consecutive evenly stepped items whose place steps too that make a loop:
    /// longer than any single row or column of a real board (`0` disables).
    #[serde(default = "default_min_spatial_run")]
    pub min_spatial_run: usize,
    /// Verbatim copies of one item in one array that make a loop.
    pub min_exact_repeats: usize,
    /// A streaming client re-checks after at least this many new bytes.
    pub check_every_bytes: usize,
}

fn default_min_spatial_run() -> usize {
    24
}

impl Default for RepetitionParams {
    fn default() -> Self {
        Self {
            min_templated_run: 6,
            min_spatial_run: default_min_spatial_run(),
            min_exact_repeats: 3,
            check_every_bytes: 256,
        }
    }
}

impl RepetitionParams {
    /// Identifies the rules and thresholds: two guards with the same id stop the
    /// same replies at the same byte.
    pub fn policy_id(&self) -> String {
        format!(
            "repetition.v{POLICY_VERSION}:templated={}:spatial={}:exact={}:every={}",
            self.min_templated_run,
            self.min_spatial_run,
            self.min_exact_repeats,
            self.check_every_bytes
        )
    }
}

/// Which rule fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RepetitionKind {
    /// One element restated under evenly stepped, undeclared ids.
    Templated,
    /// Evenly stepped items, place included, past any plausible board row.
    Runaway,
    /// One item repeated verbatim.
    Exact,
}

/// A detected loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RepetitionFinding {
    /// Rule that fired.
    pub kind: RepetitionKind,
    /// Items in the run (templated, runaway) or copies of the item (exact).
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
            RepetitionKind::Runaway => "evenly stepped items",
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

/// An item split into its shape and numbers.
#[derive(Debug, Clone, PartialEq)]
struct Scan {
    /// The item with every number replaced by `#` and whitespace removed.
    shape: String,
    /// JSON numbers outside strings (boxes, confidences), in order.
    place: Vec<f64>,
    /// Numbers inside strings (`n15`, `Step 3`), in order.
    label: Vec<f64>,
    /// Words of string values (not keys) that hold a digit: ids and numbered text.
    words: HashSet<String>,
}

/// Split an item. Digits inside words (`n15`, `v2`) count as numbers too: loops
/// step ids.
fn scan(item: &str) -> Scan {
    let mut shape = String::with_capacity(item.len());
    let (mut place, mut label) = (Vec::new(), Vec::new());
    let mut cur = String::new();
    let mut cur_in_str = false;
    let flush = |cur: &mut String,
                 in_str: bool,
                 shape: &mut String,
                 place: &mut Vec<f64>,
                 label: &mut Vec<f64>| {
        if !cur.is_empty() {
            let v = cur.parse::<f64>().unwrap_or(0.0);
            if in_str {
                label.push(v);
            } else {
                place.push(v);
            }
            shape.push('#');
            cur.clear();
        }
    };
    let mut prev: Option<char> = None;
    let (mut in_str, mut esc) = (false, false);
    for c in item.chars() {
        let starts_negative =
            c == '-' && cur.is_empty() && !prev.is_some_and(|p| p.is_alphanumeric());
        if c.is_ascii_digit() || (c == '.' && !cur.is_empty()) || starts_negative {
            if cur.is_empty() {
                cur_in_str = in_str;
            }
            cur.push(c);
        } else {
            flush(&mut cur, cur_in_str, &mut shape, &mut place, &mut label);
            if !c.is_whitespace() {
                shape.push(c);
            }
        }
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
        } else if c == '"' {
            in_str = true;
        }
        prev = Some(c);
    }
    flush(&mut cur, cur_in_str, &mut shape, &mut place, &mut label);
    Scan {
        shape,
        place,
        label,
        words: value_words(item),
    }
}

/// Words (runs of alphanumerics and `_`) with a digit, from string values only;
/// a string followed by `:` is a key and is skipped.
fn value_words(item: &str) -> HashSet<String> {
    let bytes = item.as_bytes();
    let mut out = HashSet::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut j = start;
        let mut esc = false;
        while j < bytes.len() {
            if esc {
                esc = false;
            } else if bytes[j] == b'\\' {
                esc = true;
            } else if bytes[j] == b'"' {
                break;
            }
            j += 1;
        }
        let content = &item[start..j.min(bytes.len())];
        let mut k = j + 1;
        while k < bytes.len() && bytes[k].is_ascii_whitespace() {
            k += 1;
        }
        let is_key = k < bytes.len() && bytes[k] == b':';
        if !is_key {
            for w in content.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
                if w.chars().any(|c| c.is_ascii_digit()) {
                    out.insert(w.to_string());
                }
            }
        }
        i = j + 1;
    }
    out
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

fn same_step(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)
}

fn diff(b: &[f64], a: &[f64]) -> Vec<f64> {
    b.iter().zip(a).map(|(x, y)| x - y).collect()
}

fn nonzero(d: &[f64]) -> bool {
    d.iter().any(|v| v.abs() > 1e-9)
}

/// Look for a loop in `text` (complete or cut-off JSON).
pub fn detect(text: &str, p: &RepetitionParams) -> Option<RepetitionFinding> {
    let items = array_items(text);
    let at_bytes = text.len();
    // Verbatim repeats, per array.
    if p.min_exact_repeats >= 2 {
        let mut counts: HashMap<(usize, String), usize> = HashMap::new();
        for it in &items {
            let key: String = it.text.chars().filter(|c| !c.is_whitespace()).collect();
            let n = counts.entry((it.array, key.clone())).or_default();
            *n += 1;
            if *n >= p.min_exact_repeats {
                return Some(RepetitionFinding {
                    kind: RepetitionKind::Exact,
                    repeats: *n,
                    pattern: truncate(&scan(&key).shape, 160),
                    at_bytes,
                });
            }
        }
    }
    if p.min_templated_run < 2 && p.min_spatial_run < 2 {
        return None;
    }
    let parsed: Vec<(usize, Scan)> = items.iter().map(|it| (it.array, scan(it.text))).collect();
    // Which arrays mention each id-like word: an id another array declares (an
    // edge endpoint the node list holds) is a reference, not a fabrication.
    let mut arrays_of: HashMap<&str, HashSet<usize>> = HashMap::new();
    for (arr, s) in &parsed {
        for w in &s.words {
            arrays_of.entry(w.as_str()).or_default().insert(*arr);
        }
    }
    let declared_elsewhere = |w: &str, arr: usize| {
        arrays_of
            .get(w)
            .is_some_and(|s| s.iter().any(|a| *a != arr))
    };
    // Consecutive items of one array with one shape and a constant, non-zero step
    // between their number vectors.
    let mut run = 1usize;
    let mut step: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut ungrounded = 0usize;
    for w in parsed.windows(2) {
        let ((aa, a), (ba, b)) = (&w[0], &w[1]);
        let same = aa == ba
            && a.shape == b.shape
            && a.place.len() == b.place.len()
            && a.label.len() == b.label.len();
        let d = same.then(|| (diff(&b.place, &a.place), diff(&b.label, &a.label)));
        let moved = d
            .as_ref()
            .is_some_and(|(dp, dl)| nonzero(dp) || nonzero(dl));
        let continues = match (&d, &step) {
            (Some((dp, dl)), Some((sp, sl))) if moved => same_step(dp, sp) && same_step(dl, sl),
            _ => false,
        };
        // New ids in this item that no other array declares.
        let fabricated = b
            .words
            .iter()
            .any(|x| !a.words.contains(x) && !declared_elsewhere(x, *ba));
        if continues {
            run += 1;
            ungrounded += usize::from(fabricated);
        } else if moved {
            run = 2;
            step = d;
            ungrounded = usize::from(fabricated);
        } else {
            run = 1;
            step = None;
            ungrounded = 0;
        }
        let Some((sp, _)) = &step else { continue };
        let stays = !nonzero(sp);
        let transitions = run - 1;
        let kind = if stays
            && p.min_templated_run >= 2
            && run >= p.min_templated_run
            && ungrounded * 2 >= transitions
        {
            Some(RepetitionKind::Templated)
        } else if !stays && p.min_spatial_run >= 2 && run >= p.min_spatial_run {
            Some(RepetitionKind::Runaway)
        } else {
            None
        };
        if let Some(kind) = kind {
            return Some(RepetitionFinding {
                kind,
                repeats: run,
                pattern: truncate(&b.shape, 160),
                at_bytes,
            });
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

    /// Codex 5 / GLM B1: six identical cards in a straight row with sequential ids
    /// and evenly spaced boxes, and a "Step 1".."Step 6" flow on a snapped grid.
    #[test]
    fn evenly_spaced_rows_of_regular_elements_are_not_a_loop() {
        let p = RepetitionParams::default();
        let cards: Vec<String> = (0..6)
            .map(|c| node(30 + c, "Card", 100 + c * 120, 900))
            .collect();
        assert_eq!(detect(&board(&cards, &[], true), &p), None);
        // Cut off mid-row (the stream guard's view) is not a loop either.
        assert_eq!(detect(&board(&cards, &[], false), &p), None);
        let steps: Vec<String> = (1..=6)
            .map(|k| node(k, &format!("Step {k}"), 60 + (k - 1) * 200, 400))
            .collect();
        let flow: Vec<String> = (1..6).map(|k| edge(k, k + 1, "", [0; 4])).collect();
        assert_eq!(detect(&board(&steps, &flow, true), &p), None);
        // A long timeline row is still under the runaway limit.
        let row: Vec<String> = (0..20)
            .map(|c| node(50 + c, "Week", 40 + c * 90, 200))
            .collect();
        assert_eq!(detect(&board(&row, &[], false), &p), None);
    }

    #[test]
    fn a_chain_and_a_fan_out_over_declared_nodes_are_not_a_loop() {
        let p = RepetitionParams::default();
        let nodes: Vec<String> = (1..=8)
            .map(|k| {
                node(
                    k,
                    &format!(
                        "Service {}",
                        ["A", "B", "C", "D", "E", "F", "G", "H"][k as usize - 1]
                    ),
                    100 * k,
                    50,
                )
            })
            .collect();
        let chain: Vec<String> = (1..8).map(|k| edge(k, k + 1, "", [0; 4])).collect();
        assert_eq!(detect(&board(&nodes, &chain, true), &p), None);
        let fan: Vec<String> = (2..=8).map(|k| edge(1, k, "", [0; 4])).collect();
        assert_eq!(detect(&board(&nodes, &fan, true), &p), None);
    }

    #[test]
    fn one_box_restated_under_new_ids_is_a_loop() {
        let nodes: Vec<String> = (5..14).map(|k| node(k, "Card", 300, 300)).collect();
        let f = detect(&board(&nodes, &[], false), &RepetitionParams::default()).unwrap();
        assert_eq!((f.kind, f.repeats), (RepetitionKind::Templated, 6));
        // Numbered text in one place is the same element too.
        let nodes: Vec<String> = (1..10)
            .map(|k| node(1, &format!("Item {k}"), 300, 300))
            .collect();
        let f = detect(&board(&nodes, &[], false), &RepetitionParams::default()).unwrap();
        assert_eq!(f.kind, RepetitionKind::Templated);
    }

    #[test]
    fn a_stepped_run_past_any_board_row_is_a_runaway() {
        let row: Vec<String> = (0..40)
            .map(|c| node(100 + c, "Card", 10 + c * 60, 900))
            .collect();
        let f = detect(&board(&row, &[], false), &RepetitionParams::default()).unwrap();
        assert_eq!((f.kind, f.repeats), (RepetitionKind::Runaway, 24));
        let off = RepetitionParams {
            min_spatial_run: 0,
            ..RepetitionParams::default()
        };
        assert_eq!(detect(&board(&row, &[], false), &off), None);
    }

    #[test]
    fn policy_id_names_every_threshold() {
        let a = RepetitionParams::default();
        assert!(a
            .policy_id()
            .starts_with(&format!("repetition.v{POLICY_VERSION}:")));
        for b in [
            RepetitionParams {
                min_templated_run: 7,
                ..a
            },
            RepetitionParams {
                min_spatial_run: 30,
                ..a
            },
            RepetitionParams {
                min_exact_repeats: 4,
                ..a
            },
            RepetitionParams {
                check_every_bytes: 512,
                ..a
            },
        ] {
            assert_ne!(a.policy_id(), b.policy_id());
        }
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
