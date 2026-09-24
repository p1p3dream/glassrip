//! Scores frame pairs (template, input) in both features modes.
//!
//! ```text
//! cargo run --release -p glassrip-media-stages --example pair_debug -- A.jpg B.jpg [C.jpg ...]
//! ```
//! Every file after the first is scored against the first.

use anyhow::{Context, Result};
use glassrip_core::config::{FeaturesConfig, FeaturesMode};
use glassrip_media_stages::scoring::{Scorer, ScoringParams};

fn main() -> Result<()> {
    let files: Vec<String> = std::env::args().skip(1).collect();
    let (first, rest) = files
        .split_first()
        .context("usage: pair_debug TEMPLATE INPUT...")?;
    for mode in [FeaturesMode::Production, FeaturesMode::PrototypeCompat] {
        let cfg = FeaturesConfig {
            mode,
            ..FeaturesConfig::default()
        };
        let scorer = Scorer::new(&ScoringParams::from_config(&cfg)).map_err(anyhow::Error::msg)?;
        let a = scorer
            .features(std::path::Path::new(first), None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        for f in rest {
            let b = scorer
                .features(std::path::Path::new(f), None)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let p = scorer.pair(&a, &b);
            println!(
                "{mode:?} {first} -> {f}: ssim {:.3} frac {:.3} ink {:.3} align {:?} valid {:?}",
                p.ssim, p.changed_frac, p.ink_change, p.align_method, p.valid_frac
            );
        }
    }
    Ok(())
}
