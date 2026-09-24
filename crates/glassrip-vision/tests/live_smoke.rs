//! Live smoke test against a real Ollama server. Not part of the normal test run.
//!
//! Run with:
//!   GLASSRIP_OLLAMA_URL=http://<ollama-host>:11434 \
//!     cargo test -p glassrip-vision --test live_smoke -- --ignored --nocapture
//!
//! The test is skipped (with a message) when `GLASSRIP_OLLAMA_URL` is unset.
//! Uses only a synthetic generated image (boxes, lines, colored squares; no text).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use glassrip_vision::board::{
    board_read_request, validate_board, BoardReadOutput, BoardValidationConfig, CanvasSize,
};
use glassrip_vision::classify::{
    classify_options, classify_request, combine, ClassifyRules, ScreenClassOutput,
};
use glassrip_vision::image_prep::prepare_board_image;
use glassrip_vision::{
    GenerationOptions, OllamaBackend, OllamaConfig, VisionBackend, VisionClient,
};
use image::{DynamicImage, Rgb, RgbImage};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

const MODEL: &str = "qwen2.5vl:7b";

fn fill(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, c: Rgb<u8>) {
    for y in y0..y1.min(img.height()) {
        for x in x0..x1.min(img.width()) {
            img.put_pixel(x, y, c);
        }
    }
}

fn outline(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, t: u32, c: Rgb<u8>) {
    fill(img, x0, y0, x1, y0 + t, c);
    fill(img, x0, y1 - t, x1, y1, c);
    fill(img, x0, y0, x0 + t, y1, c);
    fill(img, x1 - t, y0, x1, y1, c);
}

fn line(img: &mut RgbImage, (x0, y0): (i64, i64), (x1, y1): (i64, i64), c: Rgb<u8>) {
    let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1);
    for i in 0..=steps {
        let x = x0 + (x1 - x0) * i / steps;
        let y = y0 + (y1 - y0) * i / steps;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (px, py) = (x + dx, y + dy);
                if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                    img.put_pixel(px as u32, py as u32, c);
                }
            }
        }
    }
}

/// A synthetic whiteboard canvas: three outlined boxes, two connectors, two stickies.
fn synthetic_board() -> DynamicImage {
    let mut img = RgbImage::from_pixel(1600, 900, Rgb([250, 250, 248]));
    let ink = Rgb([40, 40, 40]);
    outline(&mut img, 150, 200, 450, 350, 4, ink);
    outline(&mut img, 650, 200, 950, 350, 4, ink);
    outline(&mut img, 650, 550, 950, 700, 4, ink);
    line(&mut img, (450, 275), (650, 275), ink);
    line(&mut img, (800, 350), (800, 550), ink);
    fill(&mut img, 1150, 200, 1350, 400, Rgb([255, 225, 90]));
    fill(&mut img, 1150, 480, 1350, 680, Rgb([140, 200, 255]));
    DynamicImage::ImageRgb8(img)
}

async fn wait_reachable(backend: &OllamaBackend) -> String {
    for attempt in 1..=18 {
        match backend.server_version().await {
            Ok(v) => return v,
            Err(e) => {
                println!(
                    "attempt {attempt}: server not reachable yet ({e}); polling again in 10 s"
                );
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        }
    }
    panic!("Ollama server never became reachable");
}

#[tokio::test]
#[ignore = "needs a live Ollama server (GLASSRIP_OLLAMA_URL)"]
async fn live_self_test_and_classification() {
    let Ok(url) = std::env::var("GLASSRIP_OLLAMA_URL") else {
        println!("skipping live smoke test: set GLASSRIP_OLLAMA_URL to an Ollama server URL");
        return;
    };
    let mut cfg = OllamaConfig::new(&url, MODEL, 8192);
    cfg.max_attempts = 2;
    cfg.backoff_max = Duration::from_secs(10);
    let backend = Arc::new(OllamaBackend::new(cfg).unwrap());

    let version = wait_reachable(&backend).await;
    println!("server {url} version {version}");

    let t = Instant::now();
    let placement = backend.preflight().await.unwrap();
    println!("preflight ({:.2?}): {placement:?}", t.elapsed());
    println!("backend id: {:?}", backend.id());

    let report = backend.self_test(CancellationToken::new()).await.unwrap();
    println!(
        "self_test: answer {:?}, latency {:.2?}, server total {:?}, prompt_eval_count {:?}, eval_count {:?}",
        report.answer,
        report.latency,
        report.response.durations.total,
        report.response.prompt_eval_count,
        report.response.eval_count
    );

    let client = VisionClient::new(backend.clone(), placement.concurrency_hint).unwrap();
    let frame = synthetic_board();
    let (request, prepared) = classify_request(&frame, classify_options(11)).unwrap();
    println!(
        "classify thumbnail {}x{} ({} image tokens)",
        prepared.image.width(),
        prepared.image.height(),
        prepared.image.tokens()
    );
    let t = Instant::now();
    let (answer, raw): (ScreenClassOutput, _) = client
        .infer_typed(request, CancellationToken::new())
        .await
        .unwrap();
    println!(
        "classify: latency {:.2?}, server total {:?}, prompt_eval_count {:?}, eval_count {:?}, repaired {}",
        t.elapsed(),
        raw.durations.total,
        raw.prompt_eval_count,
        raw.eval_count,
        raw.repaired
    );
    println!("classify raw: {}", raw.raw_text);
    let answer = answer.in_source_coords(&prepared);
    let rules = ClassifyRules::spec_examples();
    let combined = combine(Some(&answer), &rules.evaluate::<&str>(&[]), &rules);
    println!("classify combined: {combined:?}");

    let prepared = prepare_board_image(&frame).unwrap();
    let request = board_read_request(
        &prepared,
        GenerationOptions {
            seed: 11,
            num_predict: 2048,
        },
    )
    .unwrap();
    println!(
        "board image {}x{} low_res={} ({} image tokens)",
        prepared.image.width(),
        prepared.image.height(),
        prepared.plan.low_res,
        prepared.image.tokens()
    );
    let t = Instant::now();
    let (board, raw): (BoardReadOutput, _) = client
        .infer_typed(request, CancellationToken::new())
        .await
        .unwrap();
    println!(
        "board_read: latency {:.2?}, prompt_eval_count {:?}, eval_count {:?}, repaired {}",
        t.elapsed(),
        raw.prompt_eval_count,
        raw.eval_count,
        raw.repaired
    );
    let board = board.to_canvas_coords(&prepared);
    let validated = validate_board(
        board,
        CanvasSize {
            width: 1600.0,
            height: 900.0,
        },
        &BoardValidationConfig::default(),
    );
    println!(
        "board validated: {} nodes, {} edges, {} stickies, {} owner tags, {} rejected, {} issues, reclassify={}",
        validated.nodes.len(),
        validated.edges.len(),
        validated.stickies.len(),
        validated.owner_tags.len(),
        validated.chrome_rejected.len(),
        validated.issues.len(),
        validated.needs_reclassification
    );

    println!("drawn boxes (truth): (150,200)-(450,350), (650,200)-(950,350), (650,550)-(950,700)");
    for n in &validated.nodes {
        println!("  node {} {:?} bbox {:?}", n.local_id, n.text, n.bbox);
    }
    for e in &validated.edges {
        println!(
            "  edge {} -> {} label {:?} style {:?}",
            e.src, e.dst, e.label, e.style
        );
    }
    for r in &validated.chrome_rejected {
        println!("  rejected {:?} {:?} {:?}", r.list, r.text, r.reason);
    }

    let after = backend.placement().await.unwrap();
    println!("/api/ps after run: {after:?}");
}
