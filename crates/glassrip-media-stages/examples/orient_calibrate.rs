//! Prints the orientation model's decision for an image turned 0, 90, 180 and 270 degrees
//! clockwise, for one or more tile scales. The correction reported for a turn of `d` should
//! be `(360 - d) % 360`.
//!
//! ```text
//! cargo run --release -p glassrip-media-stages --features onnx --example orient_calibrate -- \
//!     IMAGE [SHORT_SIDE...]
//! ```

use anyhow::{Context, Result};
use glassrip_media_stages::models;
use glassrip_media_stages::orient::{ClassifyOptions, classify_all};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: orient_calibrate IMAGE [SHORT_SIDE...]")?;
    let sides: Vec<u32> = std::env::args()
        .skip(2)
        .filter_map(|a| a.parse().ok())
        .collect();
    let sides = if sides.is_empty() { vec![896] } else { sides };
    let img = image::open(&path)?.to_rgb8();
    let entry = models::entry(models::ORIENT_MODEL)?;
    let model = models::ensure(&models::default_dir(), &entry, true)?;
    let turns = [
        (0u32, img.clone()),
        (90, image::imageops::rotate90(&img)),
        (180, image::imageops::rotate180(&img)),
        (270, image::imageops::rotate270(&img)),
    ];
    for side in sides {
        for (deg, im) in &turns {
            let opts = ClassifyOptions {
                tile_short_side: side,
                tile_min_conf: 0.6,
                min_conf: 0.0,
            };
            let (s, ep) = classify_all(&model, &[(0.0, im.clone())], opts)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let s = &s[0];
            println!(
                "side {side:>4} turned {deg:>3} cw -> correction {:>3} (expected {:>3}) probs {:?} [{ep}]",
                s.predicted_deg,
                (360 - deg) % 360,
                s.probs.map(|p| (p * 1000.0).round() / 1000.0)
            );
        }
    }
    Ok(())
}
