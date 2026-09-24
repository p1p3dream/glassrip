//! Exact port of CPython's `difflib.SequenceMatcher` (no `isjunk`, `autojunk=True`).
//!
//! The port follows `Lib/difflib.py` line by line so that [`ratio`] returns the same
//! float as Python for the same pair of strings (compared as sequences of Unicode
//! code points, like Python `str`):
//!
//! - `__chain_b`: `b2j` maps each element of `b` to its ascending indices. With
//!   `autojunk` and `len(b) >= 200`, every element occurring more than
//!   `len(b) // 100 + 1` times is "popular" and removed from `b2j` (it is not junk, so
//!   the extension loops may still cross it).
//! - `find_longest_match`: dynamic programming over `j2len`, then extension of the best
//!   match by equal non-junk elements on both sides (the junk extension passes are
//!   no-ops without `isjunk`, and are kept for fidelity).
//! - `get_matching_blocks`: the explicit stack (`queue.pop()` from the end), sort, and
//!   collapse of adjacent blocks, followed by the `(la, lb, 0)` sentinel.
//! - `ratio`: `2 * matches / (la + lb)`, or `1.0` when both are empty.

use std::collections::{HashMap, HashSet};

/// A matching block `a[a..a+size] == b[b..b+size]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Match {
    /// Start in `a`.
    pub a: usize,
    /// Start in `b`.
    pub b: usize,
    /// Length.
    pub size: usize,
}

/// `SequenceMatcher(None, a, b)` over code points.
#[derive(Debug, Clone)]
pub struct SequenceMatcher {
    a: Vec<char>,
    b: Vec<char>,
    b2j: HashMap<char, Vec<usize>>,
    /// Always empty (no `isjunk`); kept so the extension loops mirror CPython.
    bjunk: HashSet<char>,
    autojunk: bool,
}

impl SequenceMatcher {
    /// Matcher with CPython's default `autojunk=True`.
    pub fn new(a: &str, b: &str) -> Self {
        Self::with_autojunk(a, b, true)
    }

    /// Matcher with an explicit `autojunk` flag.
    pub fn with_autojunk(a: &str, b: &str, autojunk: bool) -> Self {
        let mut m = Self {
            a: a.chars().collect(),
            b: b.chars().collect(),
            b2j: HashMap::new(),
            bjunk: HashSet::new(),
            autojunk,
        };
        m.chain_b();
        m
    }

    fn chain_b(&mut self) {
        let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
        for (i, &elt) in self.b.iter().enumerate() {
            b2j.entry(elt).or_default().push(i);
        }
        let n = self.b.len();
        if self.autojunk && n >= 200 {
            let ntest = n / 100 + 1;
            b2j.retain(|_, idxs| idxs.len() <= ntest);
        }
        self.b2j = b2j;
    }

    fn is_bjunk(&self, c: char) -> bool {
        self.bjunk.contains(&c)
    }

    /// Longest matching block in `a[alo..ahi]` and `b[blo..bhi]`.
    pub fn find_longest_match(&self, alo: usize, ahi: usize, blo: usize, bhi: usize) -> Match {
        let (a, b) = (&self.a, &self.b);
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for (i, ai) in a.iter().enumerate().take(ahi).skip(alo) {
            let mut newj2len: HashMap<usize, usize> = HashMap::new();
            if let Some(js) = self.b2j.get(ai) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let prev = if j == 0 {
                        0
                    } else {
                        j2len.get(&(j - 1)).copied().unwrap_or(0)
                    };
                    let k = prev + 1;
                    newj2len.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = newj2len;
        }
        while besti > alo
            && bestj > blo
            && !self.is_bjunk(b[bestj - 1])
            && a[besti - 1] == b[bestj - 1]
        {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && !self.is_bjunk(b[bestj + bestsize])
            && a[besti + bestsize] == b[bestj + bestsize]
        {
            bestsize += 1;
        }
        while besti > alo
            && bestj > blo
            && self.is_bjunk(b[bestj - 1])
            && a[besti - 1] == b[bestj - 1]
        {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && self.is_bjunk(b[bestj + bestsize])
            && a[besti + bestsize] == b[bestj + bestsize]
        {
            bestsize += 1;
        }
        Match {
            a: besti,
            b: bestj,
            size: bestsize,
        }
    }

    /// Matching blocks, ending with the `(len(a), len(b), 0)` sentinel.
    pub fn get_matching_blocks(&self) -> Vec<Match> {
        let (la, lb) = (self.a.len(), self.b.len());
        let mut queue = vec![(0usize, la, 0usize, lb)];
        let mut blocks: Vec<Match> = Vec::new();
        while let Some((alo, ahi, blo, bhi)) = queue.pop() {
            let x = self.find_longest_match(alo, ahi, blo, bhi);
            let (i, j, k) = (x.a, x.b, x.size);
            if k > 0 {
                blocks.push(x);
                if alo < i && blo < j {
                    queue.push((alo, i, blo, j));
                }
                if i + k < ahi && j + k < bhi {
                    queue.push((i + k, ahi, j + k, bhi));
                }
            }
        }
        blocks.sort();
        let (mut i1, mut j1, mut k1) = (0usize, 0usize, 0usize);
        let mut out = Vec::new();
        for m in blocks {
            if i1 + k1 == m.a && j1 + k1 == m.b {
                k1 += m.size;
            } else {
                if k1 > 0 {
                    out.push(Match {
                        a: i1,
                        b: j1,
                        size: k1,
                    });
                }
                i1 = m.a;
                j1 = m.b;
                k1 = m.size;
            }
        }
        if k1 > 0 {
            out.push(Match {
                a: i1,
                b: j1,
                size: k1,
            });
        }
        out.push(Match {
            a: la,
            b: lb,
            size: 0,
        });
        out
    }

    /// `SequenceMatcher.ratio()`.
    pub fn ratio(&self) -> f64 {
        let matches: usize = self.get_matching_blocks().iter().map(|m| m.size).sum();
        let length = self.a.len() + self.b.len();
        if length > 0 {
            2.0 * matches as f64 / length as f64
        } else {
            1.0
        }
    }
}

/// `SequenceMatcher(None, a, b).ratio()`.
pub fn ratio(a: &str, b: &str) -> f64 {
    SequenceMatcher::new(a, b).ratio()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn m(a: usize, b: usize, size: usize) -> Match {
        Match { a, b, size }
    }

    /// Values produced by CPython 3.14 `difflib` for the same synthetic strings.
    #[test]
    fn ratios_match_cpython() {
        let cases: &[(&str, &str, f64)] = &[
            ("abcd", "bcde", 0.75),
            ("Widget Service", "Widgit Service", 0.9285714285714286),
            ("queue worker", "worker queue", 0.5),
            (
                "frontend design system",
                "/frontend design system",
                0.9777777777777777,
            ),
            ("Avery", "Averi", 0.8),
            ("", "", 1.0),
            ("abc", "", 0.0),
            ("private ledger", "ledger", 0.6),
            ("GRPC", "gRPC", 0.75),
            ("aaaa", "aa", 0.6666666666666666),
            (
                "the quick brown fox",
                "the quack brown fax",
                0.8947368421052632,
            ),
            ("Relay Node Alpha", "Relay Node Beta", 0.7741935483870968),
            ("Jordan", "Jordon", 0.8333333333333334),
        ];
        for &(a, b, want) in cases {
            assert_eq!(ratio(a, b), want, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn matching_blocks_match_cpython() {
        let sm = SequenceMatcher::new("the quick brown fox", "the quack brown fax");
        assert_eq!(
            sm.get_matching_blocks(),
            vec![m(0, 0, 6), m(7, 7, 10), m(18, 18, 1), m(19, 19, 0)]
        );
        assert_eq!(sm.find_longest_match(0, 19, 0, 19), m(7, 7, 10));
        let sm = SequenceMatcher::new("abc", "");
        assert_eq!(sm.get_matching_blocks(), vec![m(3, 0, 0)]);
        assert_eq!(sm.find_longest_match(0, 3, 0, 0), m(0, 0, 0));
        let sm = SequenceMatcher::new("private ledger", "ledger");
        assert_eq!(sm.get_matching_blocks(), vec![m(8, 0, 6), m(14, 6, 0)]);
    }

    /// Strings of 200+ code points trigger the autojunk rule; CPython values with and
    /// without it differ widely, so these pin the popular-element purge.
    #[test]
    fn autojunk_matches_cpython() {
        let a1 = format!("{}{}", "ab".repeat(60), "xyz".repeat(40));
        let b1 = "aaabcc".repeat(40);
        assert_eq!((a1.chars().count(), b1.chars().count()), (240, 240));
        assert_eq!(ratio(&a1, &b1), 0.004166666666666667);
        assert_eq!(
            SequenceMatcher::with_autojunk(&a1, &b1, false).ratio(),
            0.3333333333333333
        );

        let a2: String = (0..250u32)
            .filter_map(|i| char::from_u32(97 + (i * 7) % 26))
            .collect();
        let b2: String = (0..230u32)
            .filter_map(|i| char::from_u32(97 + (i * 11) % 26))
            .collect();
        assert_eq!(ratio(&a2, &b2), 0.004166666666666667);
        assert_eq!(
            SequenceMatcher::with_autojunk(&a2, &b2, false).ratio(),
            0.325
        );

        let a3 = "lorem ipsum dolor sit amet ".repeat(9);
        let b3 = "lorem ipsum dolor sit amet consectetur ".repeat(6);
        assert_eq!(ratio(&a3, &b3), 0.11320754716981132);
        assert_eq!(
            SequenceMatcher::with_autojunk(&a3, &b3, false).ratio(),
            0.7547169811320755
        );
    }

    #[test]
    fn autojunk_threshold_is_len_200() {
        // CPython values. At 200 elements 'a' is purged from b2j, but the non-junk
        // extension loop still grows an empty match at (0, 0) over equal prefixes.
        let s199 = "a".repeat(199);
        assert_eq!(ratio(&s199, &s199), 1.0);
        let s200 = "a".repeat(200);
        assert_eq!(ratio(&s200, &s200), 1.0);
        let a = format!("b{}", "a".repeat(200));
        let b = format!("{}b", "a".repeat(200));
        assert_eq!(ratio(&a, &b), 0.004975124378109453);
        assert_eq!(ratio(&"xa".repeat(100), &"ya".repeat(100)), 0.0);
    }

    proptest! {
        #[test]
        fn ratio_bounds_and_identity(a in "[a-e ]{0,40}", b in "[a-e ]{0,40}") {
            let r = ratio(&a, &b);
            prop_assert!((0.0..=1.0).contains(&r));
            prop_assert_eq!(ratio(&a, &a), 1.0);
            let blocks = SequenceMatcher::new(&a, &b).get_matching_blocks();
            let ac: Vec<char> = a.chars().collect();
            let bc: Vec<char> = b.chars().collect();
            for w in blocks.windows(2) {
                prop_assert!(w[0].a + w[0].size <= w[1].a && w[0].b + w[0].size <= w[1].b);
            }
            for blk in &blocks {
                prop_assert_eq!(&ac[blk.a..blk.a + blk.size], &bc[blk.b..blk.b + blk.size]);
            }
        }
    }
}
