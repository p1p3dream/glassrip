//! Parity gate for `prototype_compat` mode.
//!
//! Recomputes per-frame sharpness, consecutive-pair SSIM / changed fraction / ink change and
//! the final keyframe runs from the prototype's input frames, and compares them with the
//! prototype's saved outputs. All data, including the SHA-256 digests that pin it, is supplied
//! at runtime; nothing derived from the reference data lives in this repository.
//!
//! ```text
//! parity --frames DIR --reference DIR --keyframes FILE
//!        [--checksums FILE] [--duration SECS] [--report FILE] [--segment-batch N]
//! ```
//!
//! `--reference` holds `pair_scores.json` and `ink_pairs.json`; `--keyframes` is the
//! prototype's `keyframes.json`. `--checksums` (default `REFERENCE/parity_checksums.sha256`)
//! is a `sha256sum`-style file with one `<hex digest>  <name>` line for each of
//! `pair_scores.json`, `ink_pairs.json` and `frames` (all frames concatenated in filename
//! order). Exits 0 when every criterion passes, 1 on a parity failure and 2 on a usage or
//! input error, including any checksum mismatch. `--segment-batch` sets how many upcoming
//! anchor comparisons are evaluated speculatively in parallel (default: threads, at most 8);
//! it changes speed only, never results.

use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use glassrip_media::features::{frames_features, pair_features, FeatureOracle};
use glassrip_media::segment::{
    keyframes, merge_singletons, segment, DiffCache, MergeParams, Thresholds,
};

const TOL_SHARP_REL: f64 = 1e-3;
const TOL_PAIR: f64 = 0.005;
const TOL_INK: f64 = 0.005;
const MAX_BOUNDARY_DIFFS: usize = 2;
const EXPLAIN_MARGIN: f64 = 0.005;

type BoxError = Box<dyn std::error::Error>;

struct Args {
    frames: PathBuf,
    reference: PathBuf,
    keyframes: PathBuf,
    checksums: Option<PathBuf>,
    duration: Option<f64>,
    report: Option<PathBuf>,
    segment_batch: Option<usize>,
}

fn parse_args() -> Result<Args, BoxError> {
    let mut frames = None;
    let mut reference = None;
    let mut keyframes = None;
    let mut checksums = None;
    let mut duration = None;
    let mut report = None;
    let mut segment_batch = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("missing value for {a}"));
        match a.as_str() {
            "--frames" => frames = Some(PathBuf::from(val()?)),
            "--reference" => reference = Some(PathBuf::from(val()?)),
            "--keyframes" => keyframes = Some(PathBuf::from(val()?)),
            "--checksums" => checksums = Some(PathBuf::from(val()?)),
            "--duration" => duration = Some(val()?.parse::<f64>()?),
            "--report" => report = Some(PathBuf::from(val()?)),
            "--segment-batch" => segment_batch = Some(val()?.parse::<usize>()?),
            "-h" | "--help" => {
                return Err(
                    "usage: parity --frames DIR --reference DIR --keyframes FILE \
                            [--checksums FILE] [--duration SECS] [--report FILE] \
                            [--segment-batch N]"
                        .into(),
                )
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(Args {
        frames: frames.ok_or("--frames is required")?,
        reference: reference.ok_or("--reference is required")?,
        keyframes: keyframes.ok_or("--keyframes is required")?,
        checksums,
        duration,
        report,
        segment_batch,
    })
}

#[derive(Deserialize)]
struct PairRecord {
    file: String,
    t: f64,
    sharp: f64,
    ssim: Option<f64>,
    frac: Option<f64>,
    #[allow(dead_code)]
    shift: Option<f64>,
}

#[derive(Deserialize)]
struct KeyframeRecord {
    t_start: f64,
    t_end: f64,
    t_rep: f64,
    n_merged: usize,
}

#[derive(Deserialize)]
struct KeyframesFile {
    keyframes: Vec<KeyframeRecord>,
}

fn sha256_file(path: &Path) -> Result<String, BoxError> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(hex(&Sha256::digest(&bytes)))
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_concat(paths: &[PathBuf]) -> Result<String, BoxError> {
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    for p in paths {
        let mut f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
    }
    Ok(hex(&h.finalize()))
}

/// Reads a `sha256sum`-style file into `name -> lowercase hex digest`.
fn read_checksums(path: &Path) -> Result<HashMap<String, String>, BoxError> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(digest), Some(name)) = (parts.next(), parts.next()) else {
            return Err(format!("bad checksum line: {line}").into());
        };
        let name = name.trim_start_matches('*');
        if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("bad SHA-256 digest for {name}").into());
        }
        out.insert(name.to_string(), digest.to_ascii_lowercase());
    }
    Ok(out)
}

/// Frames named `t_NNNNNN.jpg`, sorted by name, with the time parsed from the name.
fn list_frames(dir: &Path) -> Result<Vec<(PathBuf, String, f64)>, BoxError> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let e = e?;
        let name = e.file_name().to_string_lossy().to_string();
        if let Some(stem) = name.strip_prefix("t_").and_then(|s| s.strip_suffix(".jpg")) {
            let t: f64 = stem
                .parse::<u64>()
                .map_err(|_| format!("bad frame name {name}"))? as f64;
            out.push((e.path(), name, t));
        }
    }
    out.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(out)
}

#[derive(Default)]
struct Stat {
    max_err: f64,
    worst: usize,
    exact: usize,
    over: Vec<usize>,
    n: usize,
}

impl Stat {
    fn add(&mut self, idx: usize, err: f64, tol: f64) {
        self.n += 1;
        if err == 0.0 {
            self.exact += 1;
        }
        if err > self.max_err || err.is_nan() {
            self.max_err = err;
            self.worst = idx;
        }
        if err > tol || err.is_nan() {
            self.over.push(idx);
        }
    }

    fn pass(&self) -> bool {
        self.over.is_empty()
    }

    fn json(&self) -> serde_json::Value {
        json!({"n": self.n, "max_err": self.max_err, "worst_index": self.worst,
               "bit_identical": self.exact, "over_tolerance": self.over})
    }
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

fn run() -> Result<bool, BoxError> {
    let args = parse_args()?;
    let t_total = Instant::now();

    // 1. Pin the inputs.
    let frames = list_frames(&args.frames)?;
    let paths: Vec<PathBuf> = frames.iter().map(|f| f.0.clone()).collect();
    let pair_path = args.reference.join("pair_scores.json");
    let ink_path = args.reference.join("ink_pairs.json");
    let sums_path = args
        .checksums
        .clone()
        .unwrap_or_else(|| args.reference.join("parity_checksums.sha256"));
    let sums = read_checksums(&sums_path)?;
    let want = |name: &str| -> Result<String, BoxError> {
        sums.get(name)
            .cloned()
            .ok_or_else(|| format!("{} has no digest for {name}", sums_path.display()).into())
    };
    let checks = [
        (
            "pair_scores.json",
            sha256_file(&pair_path)?,
            want("pair_scores.json")?,
        ),
        (
            "ink_pairs.json",
            sha256_file(&ink_path)?,
            want("ink_pairs.json")?,
        ),
        ("frames", sha256_concat(&paths)?, want("frames")?),
    ];
    let mut inputs_ok = true;
    println!(
        "== input checksums (SHA-256, pinned by {})",
        sums_path.display()
    );
    for (name, got, want) in &checks {
        let ok = got == want;
        inputs_ok &= ok;
        println!("  {:<22} {}  {}", name, verdict(ok), got);
    }
    println!("  keyframes file         {}", sha256_file(&args.keyframes)?);
    if !inputs_ok {
        return Err(
            "input checksums do not match the pinned reference; refusing to compare".into(),
        );
    }

    let pairs_ref: Vec<PairRecord> = serde_json::from_slice(&std::fs::read(&pair_path)?)?;
    let ink_ref: Vec<f64> = serde_json::from_slice(&std::fs::read(&ink_path)?)?;
    let kf_ref: KeyframesFile = serde_json::from_slice(&std::fs::read(&args.keyframes)?)?;
    let n = frames.len();
    if pairs_ref.len() != n || ink_ref.len() + 1 != n {
        return Err(format!(
            "reference sizes do not match: {n} frames, {} pair records, {} ink values",
            pairs_ref.len(),
            ink_ref.len()
        )
        .into());
    }
    for (f, r) in frames.iter().zip(&pairs_ref) {
        if f.1 != r.file || f.2 != r.t {
            return Err(format!("frame {} does not match reference entry {}", f.1, r.file).into());
        }
    }
    let times: Vec<f64> = frames.iter().map(|f| f.2).collect();
    let duration = args
        .duration
        .or_else(|| kf_ref.keyframes.last().map(|k| k.t_end))
        .ok_or("no --duration and no reference keyframes to take it from")?;

    // 2. Per-frame features.
    println!(
        "== computing ({} rayon threads)",
        rayon::current_num_threads()
    );
    let t0 = Instant::now();
    let feats = frames_features(&paths)?;
    let t_feat = t0.elapsed().as_secs_f64();
    println!("  per-frame features   {n} frames in {t_feat:.1} s");

    // 3. Consecutive pairs.
    let t0 = Instant::now();
    let pairs: Vec<_> = (1..n)
        .into_par_iter()
        .map(|i| pair_features(&feats[i - 1], &feats[i]))
        .collect();
    let t_pairs = t0.elapsed().as_secs_f64();
    println!("  consecutive pairs    {} pairs in {t_pairs:.1} s", n - 1);

    // 4. Segmentation and merge.
    let t0 = Instant::now();
    let oracle = FeatureOracle { frames: &feats };
    let cache = DiffCache::new(&oracle, Thresholds::default());
    for (k, p) in pairs.iter().enumerate() {
        cache.seed(
            k,
            k + 1,
            p.score.ssim,
            p.score.changed_frac,
            p.ink_change,
            p.score.ecc.ok(),
        );
    }
    let sharp: Vec<f64> = feats.iter().map(|f| f.sharpness).collect();
    let batch = args
        .segment_batch
        .unwrap_or_else(|| rayon::current_num_threads().min(8));
    let (mut runs, bounds) = segment(n, &cache, batch);
    let n_segment_runs = runs.len();
    let merges = merge_singletons(&mut runs, &sharp, MergeParams::default(), &cache);
    let kfs = keyframes(&runs, &times, &sharp, duration, &bounds);
    let t_seg = t0.elapsed().as_secs_f64();
    println!(
        "  segmentation         {n_segment_runs} runs, {} merges, {} keyframes in {t_seg:.1} s \
         (speculative batch {batch})",
        merges.len(),
        kfs.len()
    );

    // 5. Compare per-frame and per-pair values.
    let mut s_sharp = Stat::default();
    let mut s_ssim = Stat::default();
    let mut s_frac = Stat::default();
    let mut s_ink = Stat::default();
    let mut align_fail_pairs = Vec::new();
    for i in 0..n {
        let r = &pairs_ref[i];
        s_sharp.add(
            i,
            (feats[i].sharpness - r.sharp).abs() / r.sharp.abs().max(1e-12),
            TOL_SHARP_REL,
        );
        if i == 0 {
            continue;
        }
        let p = &pairs[i - 1];
        let (rs, rf) = (r.ssim.unwrap_or(f64::NAN), r.frac.unwrap_or(f64::NAN));
        s_ssim.add(i, (p.score.ssim - rs).abs(), TOL_PAIR);
        s_frac.add(i, (p.score.changed_frac - rf).abs(), TOL_PAIR);
        s_ink.add(i, (p.ink_change - ink_ref[i - 1]).abs(), TOL_INK);
        if !p.score.ecc.ok() || !p.ink_align_ok {
            align_fail_pairs.push(i);
        }
    }

    // 6. Compare boundaries.
    let ref_starts: BTreeSet<i64> = kf_ref.keyframes.iter().map(|k| k.t_start as i64).collect();
    let got_starts: BTreeSet<i64> = kfs.iter().map(|k| k.t_start as i64).collect();
    let only_ref: Vec<i64> = ref_starts.difference(&got_starts).copied().collect();
    let only_got: Vec<i64> = got_starts.difference(&ref_starts).copied().collect();
    let idx_of: HashMap<i64, usize> = times
        .iter()
        .enumerate()
        .map(|(i, t)| (*t as i64, i))
        .collect();
    let snapshot = cache.snapshot();
    let thr = Thresholds::default();
    let mp = MergeParams::default();
    let mut explanations = Vec::new();
    let mut all_explained = true;
    for (side, t) in only_ref
        .iter()
        .map(|t| ("reference_only", *t))
        .chain(only_got.iter().map(|t| ("computed_only", *t)))
    {
        let i = idx_of.get(&t).copied().unwrap_or(usize::MAX);
        let mut best: Option<(f64, String)> = None;
        let mut consider = |margin: f64, what: String| {
            if best.as_ref().is_none_or(|b| margin < b.0) {
                best = Some((margin, what));
            }
        };
        for (&(a, b), c) in &snapshot {
            let involved = b == i || b == i + 1 || a == i || (i > 0 && b == i - 1);
            if !involved {
                continue;
            }
            consider(
                (c.ssim - thr.ssim).abs(),
                format!("ssim({a},{b})={:.4}", c.ssim),
            );
            consider(
                (c.frac - thr.frac).abs(),
                format!("frac({a},{b})={:.4}", c.frac),
            );
            if let Some(v) = c.ink {
                consider((v - thr.ink).abs(), format!("ink({a},{b})={v:.4}"));
            }
        }
        for m in merges
            .iter()
            .filter(|m| m.frame == i || m.frame + 1 == i || m.frame == i + 1)
        {
            consider(
                (m.ssim - mp.ssim).abs(),
                format!("merge ssim({},{})={:.4}", m.neighbor_rep, m.frame, m.ssim),
            );
            consider(
                (m.frac - mp.frac).abs(),
                format!("merge frac({},{})={:.4}", m.neighbor_rep, m.frame, m.frac),
            );
        }
        let (margin, what) = best.unwrap_or((f64::INFINITY, "no score recorded".to_string()));
        let explained = margin <= EXPLAIN_MARGIN;
        all_explained &= explained;
        explanations.push(json!({"side": side, "t_start": t, "frame": i,
            "nearest_threshold_score": what, "margin": margin, "explained": explained}));
    }
    let n_diffs = only_ref.len() + only_got.len();
    let boundaries_ok = n_diffs <= MAX_BOUNDARY_DIFFS && all_explained;

    // Informational: for runs present in both, compare t_rep and run length.
    let got_by_start: HashMap<i64, &glassrip_media::segment::Keyframe> =
        kfs.iter().map(|k| (k.t_start as i64, k)).collect();
    let mut rep_mismatch = Vec::new();
    for r in &kf_ref.keyframes {
        if let Some(g) = got_by_start.get(&(r.t_start as i64)) {
            if g.t_rep != r.t_rep || g.frames.len() != r.n_merged || g.t_end != r.t_end {
                rep_mismatch.push(json!({"t_start": r.t_start,
                    "ref": {"t_rep": r.t_rep, "n": r.n_merged, "t_end": r.t_end},
                    "got": {"t_rep": g.t_rep, "n": g.frames.len(), "t_end": g.t_end}}));
            }
        }
    }

    let total = t_total.elapsed().as_secs_f64();
    let pass = s_sharp.pass() && s_ssim.pass() && s_frac.pass() && s_ink.pass() && boundaries_ok;

    println!("== per-frame and consecutive-pair values");
    let line = |name: &str, s: &Stat, tol: &str| {
        println!(
            "  {:<10} {}  max err {:.3e} (index {})  bit-identical {}/{}  over tolerance {}  [tol {}]",
            name,
            verdict(s.pass()),
            s.max_err,
            s.worst,
            s.exact,
            s.n,
            s.over.len(),
            tol
        );
    };
    line("sharpness", &s_sharp, "0.1% relative");
    line("ssim", &s_ssim, "0.005");
    line("frac", &s_frac, "0.005");
    line("ink", &s_ink, "0.005");
    println!(
        "  ECC non-convergence on consecutive pairs (pair or ink path): {}",
        align_fail_pairs.len()
    );
    println!("== keyframe boundaries");
    println!(
        "  {}  reference {} runs, computed {} runs, differences {} (allowed {}, each within {} of a threshold)",
        verdict(boundaries_ok),
        ref_starts.len(),
        got_starts.len(),
        n_diffs,
        MAX_BOUNDARY_DIFFS,
        EXPLAIN_MARGIN
    );
    for e in &explanations {
        println!("    {e}");
    }
    println!(
        "  runs with identical start but different t_rep / length / t_end: {}",
        rep_mismatch.len()
    );
    for m in &rep_mismatch {
        println!("    {m}");
    }
    println!("== timing");
    println!("  features {t_feat:.1} s, pairs {t_pairs:.1} s, segmentation {t_seg:.1} s, total {total:.1} s");
    println!("== RESULT: {}", verdict(pass));

    if let Some(path) = &args.report {
        let report = json!({
            "result": verdict(pass),
            "mode": "prototype_compat",
            "inputs": checks.iter().map(|(n, g, w)| json!({"name": n, "sha256": g, "expected": w})).collect::<Vec<_>>(),
            "keyframes_sha256": sha256_file(&args.keyframes)?,
            "n_frames": n,
            "duration": duration,
            "sharpness": s_sharp.json(),
            "ssim": s_ssim.json(),
            "frac": s_frac.json(),
            "ink": s_ink.json(),
            "ecc_nonconvergence_pairs": align_fail_pairs,
            "boundaries": {"pass": boundaries_ok, "reference_runs": ref_starts.len(),
                "computed_runs": got_starts.len(), "differences": explanations,
                "segment_runs_before_merge": n_segment_runs, "merges": merges},
            "run_detail_mismatches": rep_mismatch,
            "timing_s": {"features": t_feat, "pairs": t_pairs, "segmentation": t_seg, "total": total,
                "threads": rayon::current_num_threads()},
        });
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
        println!("report written to {}", path.display());
    }
    Ok(pass)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("parity: {e}");
            ExitCode::from(2)
        }
    }
}
