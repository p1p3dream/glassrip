//! Re-clustering diarizer output to a known speaker count, and exclusive turns.
//!
//! The diarizer gives, per segmentation chunk and local speaker, an embedding and
//! a global cluster id, plus a frame-level activation matrix whose columns are
//! cluster ids. With a known count K, cluster centroids (mean of L2-normalized
//! embeddings) are merged by agglomerative clustering with cosine distance and
//! size-weighted centroid linkage until K remain. Activation columns of merged
//! clusters are combined with `max`, then every frame keeps a single speaker so
//! turns never overlap.

/// One chunk-level speaker embedding with its cluster id.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingSample {
    /// Segmentation chunk index.
    pub chunk: usize,
    /// Cluster id from the diarizer.
    pub cluster: usize,
    /// Embedding vector.
    pub embedding: Vec<f32>,
}

/// An exclusive speaker turn.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Turn {
    /// Start, seconds.
    pub start_s: f64,
    /// End, seconds.
    pub end_s: f64,
    /// Speaker index (0-based, ordered by first appearance).
    pub speaker: usize,
    /// Cosine similarity between the turn's embedding and its speaker centroid.
    pub embedding_sim: Option<f32>,
}

/// L2-normalize a vector; `None` for zero or non-finite input.
pub fn normalize(v: &[f32]) -> Option<Vec<f32>> {
    if v.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n <= f32::EPSILON {
        return None;
    }
    Some(v.iter().map(|x| x / n).collect())
}

/// Cosine similarity of two vectors (normalized internally).
pub fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let a = normalize(a)?;
    let b = normalize(b)?;
    Some(a.iter().zip(&b).map(|(x, y)| x * y).sum())
}

/// Per-cluster centroid (normalized mean of normalized embeddings) and count.
pub fn centroids(samples: &[EmbeddingSample], n_clusters: usize) -> Vec<Option<(Vec<f32>, usize)>> {
    let mut sums: Vec<Option<(Vec<f32>, usize)>> = vec![None; n_clusters];
    for s in samples {
        if s.cluster >= n_clusters {
            continue;
        }
        let Some(e) = normalize(&s.embedding) else {
            continue;
        };
        match &mut sums[s.cluster] {
            Some((acc, n)) if acc.len() == e.len() => {
                for (a, x) in acc.iter_mut().zip(&e) {
                    *a += x;
                }
                *n += 1;
            }
            Some(_) => {}
            slot @ None => *slot = Some((e, 1)),
        }
    }
    sums.into_iter()
        .map(|s| s.and_then(|(v, n)| normalize(&v).map(|c| (c, n))))
        .collect()
}

/// Merge clusters down to `k` and return `old cluster -> new cluster`.
///
/// Clusters without a centroid join the new cluster with the most talk time.
/// New ids are ordered by the earliest old id they contain. If there are already
/// `k` or fewer clusters with centroids, only centroid-less clusters are folded.
pub fn ahc_to_k(cents: &[Option<(Vec<f32>, usize)>], talk_s: &[f64], k: usize) -> Vec<usize> {
    let k = k.max(1);
    // groups: (members, centroid sum weighted by count, count)
    let mut groups: Vec<(Vec<usize>, Vec<f32>, usize)> = cents
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            c.as_ref()
                .map(|(v, n)| (vec![i], v.iter().map(|x| x * *n as f32).collect(), *n))
        })
        .collect();
    while groups.len() > k {
        let mut best: Option<(f32, usize, usize)> = None;
        for a in 0..groups.len() {
            for b in (a + 1)..groups.len() {
                let sim = cosine(&groups[a].1, &groups[b].1).unwrap_or(-1.0);
                if best.is_none_or(|(bs, _, _)| sim > bs) {
                    best = Some((sim, a, b));
                }
            }
        }
        let Some((_, a, b)) = best else { break };
        let (members, sum, n) = groups.remove(b);
        let g = &mut groups[a];
        g.0.extend(members);
        for (x, y) in g.1.iter_mut().zip(&sum) {
            *x += y;
        }
        g.2 += n;
    }
    for g in &mut groups {
        g.0.sort_unstable();
    }
    groups.sort_by_key(|g| g.0.first().copied().unwrap_or(usize::MAX));

    let mut map = vec![usize::MAX; cents.len()];
    for (new, g) in groups.iter().enumerate() {
        for &m in &g.0 {
            map[m] = new;
        }
    }
    if groups.is_empty() {
        // nothing had an embedding: keep ids but cap at k
        return (0..cents.len()).map(|i| i.min(k - 1)).collect();
    }
    let mut group_talk = vec![0.0; groups.len()];
    for (old, &new) in map.iter().enumerate() {
        if new != usize::MAX {
            group_talk[new] += talk_s.get(old).copied().unwrap_or(0.0);
        }
    }
    let busiest = group_talk
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i);
    for m in &mut map {
        if *m == usize::MAX {
            *m = busiest;
        }
    }
    map
}

/// Merge activation columns with `max` according to `map`.
pub fn merge_columns(acts: &[Vec<f32>], map: &[usize], n_new: usize) -> Vec<Vec<f32>> {
    acts.iter()
        .map(|row| {
            let mut out = vec![0.0f32; n_new];
            for (old, &v) in row.iter().enumerate() {
                if let Some(&new) = map.get(old) {
                    if new < n_new && v > out[new] {
                        out[new] = v;
                    }
                }
            }
            out
        })
        .collect()
}

/// Talk time per column, seconds.
pub fn talk_time(acts: &[Vec<f32>], n_cols: usize, frame_step_s: f64) -> Vec<f64> {
    let mut t = vec![0.0; n_cols];
    for row in acts {
        for (c, &v) in row.iter().enumerate().take(n_cols) {
            if v > 0.5 {
                t[c] += frame_step_s;
            }
        }
    }
    t
}

/// Exclusive turns from a binary activation matrix.
///
/// Per frame, one active speaker is kept: the previous frame's speaker if still
/// active, otherwise the active speaker with the most total talk time. Frame `i`
/// is centered at `i * step + duration / 2`. Speakers are relabeled by first
/// appearance; the returned vector maps new index to column.
pub fn exclusive_turns(
    acts: &[Vec<f32>],
    frame_step_s: f64,
    frame_duration_s: f64,
) -> (Vec<Turn>, Vec<usize>) {
    let n_cols = acts.iter().map(Vec::len).max().unwrap_or(0);
    let talk = talk_time(acts, n_cols, frame_step_s);
    let mid = |i: usize| i as f64 * frame_step_s + frame_duration_s / 2.0;
    let mut prev: Option<usize> = None;
    let mut runs: Vec<(usize, usize, usize)> = Vec::new(); // (col, first frame, end frame)
    for (i, row) in acts.iter().enumerate() {
        let active = |c: usize| row.get(c).is_some_and(|v| *v > 0.5);
        let pick = match prev {
            Some(p) if active(p) => Some(p),
            _ => (0..n_cols)
                .filter(|&c| active(c))
                .max_by(|&a, &b| talk[a].total_cmp(&talk[b]).then(b.cmp(&a))),
        };
        match (pick, runs.last_mut()) {
            (Some(c), Some(last)) if last.0 == c && last.2 == i => last.2 = i + 1,
            (Some(c), _) => runs.push((c, i, i + 1)),
            (None, _) => {}
        }
        prev = pick;
    }
    let mut order: Vec<usize> = Vec::new();
    let mut turns = Vec::with_capacity(runs.len());
    for (col, a, b) in runs {
        let speaker = match order.iter().position(|&c| c == col) {
            Some(p) => p,
            None => {
                order.push(col);
                order.len() - 1
            }
        };
        turns.push(Turn {
            start_s: mid(a),
            end_s: mid(b),
            speaker,
            embedding_sim: None,
        });
    }
    (turns, order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(chunk: usize, cluster: usize, v: &[f32]) -> EmbeddingSample {
        EmbeddingSample {
            chunk,
            cluster,
            embedding: v.to_vec(),
        }
    }

    #[test]
    fn centroids_skip_nan_and_normalize() {
        let s = vec![
            emb(0, 0, &[1.0, 0.0]),
            emb(1, 0, &[2.0, 0.0]),
            emb(2, 1, &[f32::NAN, 1.0]),
        ];
        let c = centroids(&s, 2);
        let (v, n) = c[0].clone().unwrap();
        assert_eq!(n, 2);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!(c[1].is_none());
    }

    #[test]
    fn ahc_merges_most_similar_pair() {
        // clusters 0 and 2 point the same way; 1 is orthogonal
        let s = vec![
            emb(0, 0, &[1.0, 0.0, 0.0]),
            emb(1, 1, &[0.0, 1.0, 0.0]),
            emb(2, 2, &[0.95, 0.05, 0.0]),
            emb(3, 3, &[0.0, 0.0, 1.0]),
        ];
        let c = centroids(&s, 4);
        let map = ahc_to_k(&c, &[10.0, 5.0, 1.0, 3.0], 3);
        assert_eq!(map[0], map[2]);
        assert_ne!(map[0], map[1]);
        assert_ne!(map[1], map[3]);
        assert_eq!(map, vec![0, 1, 0, 2]);
    }

    #[test]
    fn ahc_folds_clusters_without_embeddings_into_busiest() {
        let c = vec![Some((vec![1.0, 0.0], 1)), None, Some((vec![0.0, 1.0], 1))];
        let map = ahc_to_k(&c, &[1.0, 0.5, 9.0], 2);
        assert_eq!(map, vec![0, 1, 1]);
    }

    #[test]
    fn exclusive_turns_keep_previous_speaker_in_overlap() {
        let a = |x: f32, y: f32| vec![x, y];
        let acts = vec![
            a(1.0, 0.0),
            a(1.0, 0.0),
            a(1.0, 1.0),
            a(0.0, 1.0),
            a(0.0, 0.0),
            a(0.0, 1.0),
        ];
        let (turns, order) = exclusive_turns(&acts, 1.0, 0.0);
        assert_eq!(order, vec![0, 1]);
        assert_eq!(turns.len(), 3);
        assert_eq!((turns[0].start_s, turns[0].end_s, turns[0].speaker), (0.0, 3.0, 0));
        assert_eq!((turns[1].start_s, turns[1].end_s, turns[1].speaker), (3.0, 4.0, 1));
        assert_eq!((turns[2].start_s, turns[2].end_s, turns[2].speaker), (5.0, 6.0, 1));
    }

    #[test]
    fn merge_columns_takes_max() {
        let acts = vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
        let m = merge_columns(&acts, &[0, 0, 1], 2);
        assert_eq!(m, vec![vec![1.0, 0.0], vec![1.0, 0.0]]);
    }
}
