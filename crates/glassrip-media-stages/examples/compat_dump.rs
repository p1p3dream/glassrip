//! Dumps every `prototype_compat` value for a directory of 1920x1080 JPEG frames (sorted by
//! name) as JSON with f64 values written as raw bit patterns, so two builds can be compared
//! byte for byte. Uses only `glassrip-media`'s compat API.
//!
//! ```text
//! cargo run --release -p glassrip-media-stages --example compat_dump -- FRAMES_DIR OUT.json
//! ```

use anyhow::{Context, Result};
use glassrip_media::features::{FeatureOracle, frames_features, pair_features};
use glassrip_media::segment::{DiffCache, MergeParams, Thresholds, merge_singletons, segment};
use rayon::prelude::*;

fn bits(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(
        args.next()
            .context("usage: compat_dump FRAMES_DIR OUT.json")?,
    );
    let out = std::path::PathBuf::from(args.next().context("missing OUT.json")?);
    let mut paths: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jpg"))
        .collect();
    paths.sort();
    let feats = frames_features(&paths)?;
    let n = feats.len();
    let pairs: Vec<_> = (1..n)
        .into_par_iter()
        .map(|i| pair_features(&feats[i - 1], &feats[i]))
        .collect();
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
    let (mut runs, bounds) = segment(n, &cache, 8);
    let merges = merge_singletons(&mut runs, &sharp, MergeParams::default(), &cache);
    let json = serde_json::json!({
        "frames": n,
        "sharpness": sharp.iter().map(|v| bits(*v)).collect::<Vec<_>>(),
        "pairs": pairs.iter().map(|p| serde_json::json!([
            bits(p.score.ssim), bits(p.score.changed_frac), bits(p.score.shift),
            bits(p.ink_change), p.score.ecc.ok(), p.ink_align_ok
        ])).collect::<Vec<_>>(),
        "boundaries": bounds.iter().map(|b| b.frame).collect::<Vec<_>>(),
        "merges": merges.iter().map(|m| (m.frame, m.neighbor_rep)).collect::<Vec<_>>(),
        "runs": runs,
    });
    std::fs::write(&out, serde_json::to_vec(&json)?)?;
    println!(
        "{n} frames, {} runs after merge -> {}",
        runs.len(),
        out.display()
    );
    Ok(())
}
