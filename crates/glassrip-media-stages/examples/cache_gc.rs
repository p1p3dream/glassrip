//! `cache gc` for the stage cache plus the media blob store.
//!
//! ```text
//! cargo run -p glassrip-media-stages --example cache_gc -- [--workspace DIR] [--older-than 30d]
//! ```

use anyhow::Result;
use clap::Parser;
use glassrip_core::cache::{Cache, DEFAULT_CACHE_DIR, parse_age};
use glassrip_media_stages::blobs::{BlobStore, gc_cache_and_blobs};

#[derive(Parser)]
struct Args {
    /// Workspace holding `.glassrip/cache` and `.glassrip/blobs`.
    #[arg(long, default_value = ".")]
    workspace: std::path::PathBuf,
    /// Remove entries and unreferenced blobs older than this.
    #[arg(long, default_value = "30d")]
    older_than: String,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let cache = Cache::new(a.workspace.join(DEFAULT_CACHE_DIR));
    let blobs = BlobStore::new(a.workspace.join(".glassrip/blobs"));
    let r = gc_cache_and_blobs(
        &cache,
        &blobs,
        parse_age(&a.older_than)?,
        std::time::SystemTime::now(),
    )?;
    println!(
        "cache: {} entries removed ({} bytes); blobs: {} removed ({} bytes), {} still referenced",
        r.cache.removed.len(),
        r.cache.bytes_freed,
        r.blobs_removed.len(),
        r.blob_bytes_freed,
        r.blobs_referenced
    );
    Ok(())
}
