//! Keyframe segmentation (`segment_runs.py` and `build_keyframes.py`).
//!
//! Frames are grouped into runs of visually static content. A run's anchor is its first
//! frame; frame `i` starts a new run only when it differs from the anchor, frame `i + 1`
//! also differs from the anchor, and `i` and `i + 1` agree with each other (two-sample
//! persistence; the last frame needs only the first condition). Singleton runs are then
//! merged into a neighbor when they match it or are blurry.

use std::collections::HashMap;
use std::sync::Mutex;

use rayon::prelude::*;
use serde::Serialize;

/// Scores for one ordered frame pair `(template, input)`.
pub trait PairOracle: Sync {
    /// `(ssim, changed_frac, align_ok)` of `b` aligned to `a` (`score()` in the prototype).
    fn score(&self, a: usize, b: usize) -> (f64, f64, bool);
    /// Ink change of `b` relative to `a`.
    fn ink(&self, a: usize, b: usize) -> f64;
}

/// Thresholds for `differs(x, y)`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Thresholds {
    /// A pair differs when SSIM is below this.
    pub ssim: f64,
    /// ... or when the changed fraction is above this.
    pub frac: f64,
    /// ... or when the ink change is above this.
    pub ink: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            ssim: 0.80,
            frac: 0.10,
            ink: 0.05,
        }
    }
}

/// Singleton merge parameters.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct MergeParams {
    /// Merge when SSIM against the neighbor's representative is at least this ...
    pub ssim: f64,
    /// ... and the changed fraction is at most this.
    pub frac: f64,
    /// Or merge regardless when sharpness is below this multiple of the median sharpness.
    pub blur: f64,
}

impl Default for MergeParams {
    fn default() -> Self {
        Self {
            ssim: 0.70,
            frac: 0.20,
            blur: 0.35,
        }
    }
}

/// Which criterion made a pair differ (the first one that fired, in evaluation order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    /// SSIM below threshold.
    Ssim,
    /// Changed fraction above threshold.
    Frac,
    /// Ink change above threshold.
    Ink,
}

/// Memoized comparison of an ordered pair.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Comparison {
    /// SSIM after alignment.
    pub ssim: f64,
    /// Changed fraction after alignment.
    pub frac: f64,
    /// Ink change; `None` when SSIM or frac already decided the pair.
    pub ink: Option<f64>,
    /// Whether the pair-score ECC converged.
    pub align_ok: bool,
    /// Why the pair differs, or `None` when it does not.
    pub reason: Option<Reason>,
}

impl Comparison {
    /// True when the pair counts as different.
    pub fn differs(&self) -> bool {
        self.reason.is_some()
    }
}

/// Lazily evaluates and caches `differs` for ordered pairs.
pub struct DiffCache<'a, O: PairOracle> {
    oracle: &'a O,
    thr: Thresholds,
    cache: Mutex<HashMap<(usize, usize), Comparison>>,
}

impl<'a, O: PairOracle> DiffCache<'a, O> {
    /// Creates an empty cache.
    pub fn new(oracle: &'a O, thr: Thresholds) -> Self {
        Self {
            oracle,
            thr,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Inserts a precomputed comparison (for example the consecutive pairs).
    pub fn seed(&self, a: usize, b: usize, ssim: f64, frac: f64, ink: f64, align_ok: bool) {
        let c = self.classify(ssim, frac, Some(ink), align_ok);
        if let Ok(mut m) = self.cache.lock() {
            m.insert((a, b), c);
        }
    }

    fn classify(&self, ssim: f64, frac: f64, ink: Option<f64>, align_ok: bool) -> Comparison {
        let reason = if ssim < self.thr.ssim {
            Some(Reason::Ssim)
        } else if frac > self.thr.frac {
            Some(Reason::Frac)
        } else if ink.is_some_and(|v| v > self.thr.ink) {
            Some(Reason::Ink)
        } else {
            None
        };
        Comparison {
            ssim,
            frac,
            ink,
            align_ok,
            reason,
        }
    }

    fn cached(&self, a: usize, b: usize) -> Option<Comparison> {
        self.cache.lock().ok().and_then(|m| m.get(&(a, b)).copied())
    }

    /// `differs(a, b)` with the prototype's short-circuit: ink is only computed when SSIM
    /// and frac do not already decide.
    pub fn get(&self, a: usize, b: usize) -> Comparison {
        if let Some(c) = self.cached(a, b) {
            return c;
        }
        let (ssim, frac, align_ok) = self.oracle.score(a, b);
        let decided = ssim < self.thr.ssim || frac > self.thr.frac;
        let ink = if decided {
            None
        } else {
            Some(self.oracle.ink(a, b))
        };
        let c = self.classify(ssim, frac, ink, align_ok);
        if let Ok(mut m) = self.cache.lock() {
            m.insert((a, b), c);
        }
        c
    }

    /// Evaluates several pairs in parallel, filling the cache.
    pub fn prefetch(&self, pairs: &[(usize, usize)]) {
        let todo: Vec<(usize, usize)> = pairs
            .iter()
            .copied()
            .filter(|&(a, b)| self.cached(a, b).is_none())
            .collect();
        todo.par_iter().for_each(|&(a, b)| {
            self.get(a, b);
        });
    }

    /// All cached comparisons.
    pub fn snapshot(&self) -> HashMap<(usize, usize), Comparison> {
        self.cache.lock().map(|m| m.clone()).unwrap_or_default()
    }
}

/// Boundary decision recorded when a run starts.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Boundary {
    /// Frame that starts the run.
    pub frame: usize,
    /// Anchor it was compared with.
    pub anchor: usize,
    /// Comparison of anchor and frame.
    pub comparison: Comparison,
}

/// `segment(S, F, T, persist=True)`. Returns the runs (sorted frame indices) and the
/// boundary decisions. Anchor comparisons are evaluated speculatively in parallel batches of
/// `batch` upcoming frames.
pub fn segment<O: PairOracle>(
    n: usize,
    cache: &DiffCache<'_, O>,
    batch: usize,
) -> (Vec<Vec<usize>>, Vec<Boundary>) {
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut bounds = Vec::new();
    if n == 0 {
        return (runs, bounds);
    }
    runs.push(vec![0]);
    let mut anchor = 0usize;
    let batch = batch.max(1);
    let mut i = 1usize;
    while i < n {
        if cache.cached(anchor, i).is_none() {
            let end = (i + batch + 1).min(n);
            let mut pairs: Vec<(usize, usize)> = (i..end).map(|j| (anchor, j)).collect();
            pairs.extend((i..end.saturating_sub(1)).map(|j| (j, j + 1)));
            cache.prefetch(&pairs);
        }
        let d = cache.get(anchor, i);
        let starts = d.differs()
            && (i + 1 >= n
                || (cache.get(anchor, i + 1).differs() && !cache.get(i, i + 1).differs()));
        if starts {
            runs.push(vec![i]);
            bounds.push(Boundary {
                frame: i,
                anchor,
                comparison: d,
            });
            anchor = i;
        } else if let Some(last) = runs.last_mut() {
            last.push(i);
        }
        i += 1;
    }
    (runs, bounds)
}

/// numpy `median`: the middle value, or the mean of the two middle values.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Representative frame of a run: the sharpest, first one on ties (Python `max` semantics).
pub fn representative(run: &[usize], sharp: &[f64]) -> usize {
    let mut best = run[0];
    for &i in &run[1..] {
        if sharp[i] > sharp[best] {
            best = i;
        }
    }
    best
}

/// One singleton merge performed by [`merge_singletons`].
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Merge {
    /// The singleton frame.
    pub frame: usize,
    /// Representative frame of the chosen neighbor run.
    pub neighbor_rep: usize,
    /// SSIM against the neighbor's representative.
    pub ssim: f64,
    /// Changed fraction against the neighbor's representative.
    pub frac: f64,
    /// True when the merge happened only because the frame is blurry.
    pub blurry: bool,
}

fn tuple_gt(a: (f64, usize, f64, f64), b: (f64, usize, f64, f64)) -> bool {
    if a.0 != b.0 {
        return a.0 > b.0;
    }
    if a.1 != b.1 {
        return a.1 > b.1;
    }
    if a.2 != b.2 {
        return a.2 > b.2;
    }
    a.3 > b.3
}

/// The singleton merge loop of `build_keyframes.py`, with its exact semantics: scan from the
/// start, take the first singleton, score both neighbors' representatives against it, pick
/// the neighbor maximizing `(s - f, j, s, f)`, merge only into that neighbor when it matches
/// or the frame is blurry, and restart the scan after every merge.
pub fn merge_singletons<O: PairOracle>(
    runs: &mut Vec<Vec<usize>>,
    sharp: &[f64],
    params: MergeParams,
    cache: &DiffCache<'_, O>,
) -> Vec<Merge> {
    let med = median(sharp);
    let mut merges = Vec::new();
    loop {
        let mut changed = false;
        for k in 0..runs.len() {
            if runs[k].len() != 1 {
                continue;
            }
            let i = runs[k][0];
            let mut neighbors = Vec::with_capacity(2);
            if k >= 1 {
                neighbors.push(k - 1);
            }
            if k + 1 < runs.len() {
                neighbors.push(k + 1);
            }
            let reps: Vec<(usize, usize)> = neighbors
                .iter()
                .map(|&j| (j, representative(&runs[j], sharp)))
                .collect();
            let scored: Vec<(f64, usize, f64, f64, usize)> = reps
                .par_iter()
                .map(|&(j, r)| {
                    let (s, f, _) = cache.oracle.score(r, i);
                    (s - f, j, s, f, r)
                })
                .collect();
            let mut best: Option<(f64, usize, f64, f64, usize)> = None;
            for c in scored {
                best = match best {
                    Some(b) if !tuple_gt((c.0, c.1, c.2, c.3), (b.0, b.1, b.2, b.3)) => Some(b),
                    _ => Some(c),
                };
            }
            let Some((_, j, s, f, r)) = best else {
                continue;
            };
            let matches = s >= params.ssim && f <= params.frac;
            let blurry = sharp[i] < params.blur * med;
            if matches || blurry {
                let mut merged = runs[j].clone();
                merged.push(i);
                merged.sort_unstable();
                runs[j] = merged;
                runs.remove(k);
                merges.push(Merge {
                    frame: i,
                    neighbor_rep: r,
                    ssim: s,
                    frac: f,
                    blurry: !matches,
                });
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
    merges
}

/// One emitted keyframe (`build_keyframes.py` output record).
#[derive(Debug, Clone, Serialize)]
pub struct Keyframe {
    /// Index of the representative (sharpest) frame.
    pub rep_frame: usize,
    /// Frame indices in the run.
    pub frames: Vec<usize>,
    /// Time of the run's first frame.
    pub t_start: f64,
    /// Time of the next run's first frame, or the duration for the last run.
    pub t_end: f64,
    /// Time of the representative frame.
    pub t_rep: f64,
    /// Sharpness of the representative frame.
    pub sharpness: f64,
    /// Boundary that started the run (`None` for the first run).
    pub boundary: Option<Boundary>,
}

/// Emits keyframes for final runs. `duration` becomes the last keyframe's `t_end`.
pub fn keyframes(
    runs: &[Vec<usize>],
    times: &[f64],
    sharp: &[f64],
    duration: f64,
    bounds: &[Boundary],
) -> Vec<Keyframe> {
    let by_frame: HashMap<usize, Boundary> = bounds.iter().map(|b| (b.frame, *b)).collect();
    runs.iter()
        .enumerate()
        .map(|(k, run)| {
            let r = representative(run, sharp);
            let t_end = runs.get(k + 1).map_or(duration, |next| times[next[0]]);
            Keyframe {
                rep_frame: r,
                frames: run.clone(),
                t_start: times[run[0]],
                t_end,
                t_rep: times[r],
                sharpness: sharp[r],
                boundary: by_frame.get(&run[0]).copied(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames carry a scene label; pairs in the same scene score 0.95/0.01, different scenes
    /// 0.3/0.5. Ink never fires.
    struct Scenes(Vec<u32>);

    impl PairOracle for Scenes {
        fn score(&self, a: usize, b: usize) -> (f64, f64, bool) {
            if self.0[a] == self.0[b] {
                (0.95, 0.01, true)
            } else {
                (0.3, 0.5, true)
            }
        }
        fn ink(&self, _a: usize, _b: usize) -> f64 {
            0.0
        }
    }

    fn run_all(scenes: Vec<u32>, sharp: Vec<f64>) -> Vec<Vec<usize>> {
        let o = Scenes(scenes);
        let cache = DiffCache::new(&o, Thresholds::default());
        let (mut runs, _) = segment(o.0.len(), &cache, 3);
        merge_singletons(&mut runs, &sharp, MergeParams::default(), &cache);
        runs
    }

    #[test]
    fn persistence_ignores_one_frame_glitch() {
        let o = Scenes(vec![0, 0, 1, 0, 0, 2, 2, 2]);
        let cache = DiffCache::new(&o, Thresholds::default());
        let (runs, bounds) = segment(8, &cache, 2);
        assert_eq!(runs, vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7]]);
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds[0].comparison.reason, Some(Reason::Ssim));
    }

    #[test]
    fn last_frame_can_start_a_run() {
        let o = Scenes(vec![0, 0, 0, 1]);
        let cache = DiffCache::new(&o, Thresholds::default());
        let (runs, _) = segment(4, &cache, 4);
        assert_eq!(runs, vec![vec![0, 1, 2], vec![3]]);
    }

    #[test]
    fn blurry_singleton_merges_into_best_neighbor() {
        // Final singleton (scene 1) is blurry: it merges into the only neighbor.
        let sharp = vec![100.0, 100.0, 100.0, 10.0];
        assert_eq!(run_all(vec![0, 0, 0, 1], sharp), vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn sharp_distinct_singleton_is_kept() {
        let sharp = vec![100.0, 100.0, 100.0, 100.0];
        assert_eq!(
            run_all(vec![0, 0, 0, 1], sharp),
            vec![vec![0, 1, 2], vec![3]]
        );
    }

    #[test]
    fn tie_prefers_later_neighbor() {
        assert!(tuple_gt((0.5, 2, 0.6, 0.1), (0.5, 0, 0.6, 0.1)));
        assert!(!tuple_gt((0.4, 9, 0.6, 0.1), (0.5, 0, 0.6, 0.1)));
    }

    #[test]
    fn numpy_median() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn representative_takes_first_maximum() {
        assert_eq!(
            representative(&[4, 5, 6], &[0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 3.0]),
            5
        );
    }

    #[test]
    fn keyframe_times_use_next_start_and_duration() {
        let runs = vec![vec![0, 1], vec![2, 3]];
        let times = vec![0.0, 2.0, 4.0, 6.0];
        let sharp = vec![1.0, 2.0, 5.0, 4.0];
        let k = keyframes(&runs, &times, &sharp, 7.5, &[]);
        assert_eq!((k[0].t_start, k[0].t_end, k[0].t_rep), (0.0, 4.0, 2.0));
        assert_eq!((k[1].t_start, k[1].t_end, k[1].t_rep), (4.0, 7.5, 4.0));
    }
}
