//! Model-server supervision against a mocked Ollama: a digest change mid-run
//! aborts (spec 8.1), and a server that goes down and comes back is waited
//! for, re-preflighted, and resumed. Synthetic data only.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use glassrip_vision::{
    EncodedImage, GenerationOptions, OllamaBackend, OllamaConfig, VisionClient, VisionRequest,
};
use glassrip_vision_stages::placement::{CheckKind, MonitorConfig, PlacementMonitor};
use image::{DynamicImage, RgbImage};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const MODEL: &str = "fixture-vl:7b";

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Answer {
    n: u32,
}

fn request() -> VisionRequest {
    let img = EncodedImage::encode(&DynamicImage::ImageRgb8(RgbImage::new(28, 28))).unwrap();
    VisionRequest::for_output::<Answer>(
        "count",
        img,
        GenerationOptions {
            seed: 1,
            num_predict: 32,
        },
    )
    .unwrap()
}

fn chat_ok() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "model": MODEL,
        "message": {"role": "assistant", "content": "{\"n\": 3}"},
        "done": true,
        "done_reason": "stop",
        "prompt_eval_count": 120,
        "eval_count": 8
    }))
}

fn show(digest: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({"digest": digest}))
}

async fn mount_ps(server: &MockServer) {
    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": [
            {"name": MODEL, "model": MODEL, "size": 100, "size_vram": 100,
             "digest": "sha256-one", "context_length": 8192}
        ]})))
        .mount(server)
        .await;
}

fn monitor(server: &MockServer, check_every: usize, server_wait: Duration) -> PlacementMonitor {
    let mut cfg = OllamaConfig::new(server.uri(), MODEL, 8192);
    cfg.backoff_min = Duration::from_millis(2);
    cfg.backoff_max = Duration::from_millis(5);
    cfg.max_attempts = 3;
    cfg.request_timeout = Duration::from_secs(5);
    let backend = Arc::new(OllamaBackend::new(cfg).unwrap());
    let client = VisionClient::new(backend.clone(), 2).unwrap();
    PlacementMonitor::new(
        backend,
        client,
        MonitorConfig {
            check_every,
            poll_interval: Duration::from_millis(10),
            server_wait,
            ..MonitorConfig::default()
        },
    )
}

#[tokio::test]
async fn digest_change_mid_run_aborts() {
    let server = MockServer::start().await;
    mount_ps(&server).await;
    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"version": "0.0.1"})))
        .mount(&server)
        .await;
    // Preflight resolves the digest twice (backend preflight, monitor record).
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(show("sha256-one"))
        .up_to_n_times(2)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(show("sha256-two"))
        .with_priority(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_ok())
        .mount(&server)
        .await;

    let m = monitor(&server, 2, Duration::from_secs(1));
    let first = m
        .infer_typed::<Answer>(request(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(first.0.n, 3);
    // The periodic check after the second request sees the new digest.
    let second = m
        .infer_typed::<Answer>(request(), CancellationToken::new())
        .await;
    let e = second.unwrap_err();
    assert!(e.message.contains("digest changed"), "{e}");
    // Sticky: later requests and stage starts fail without asking the model.
    assert!(m
        .infer_typed::<Answer>(request(), CancellationToken::new())
        .await
        .is_err());
    assert!(m.begin_stage().await.is_err());
    assert!(m.abort_error().is_some());
}

#[tokio::test]
async fn server_down_then_up_is_waited_for_and_resumed() {
    let server = MockServer::start().await;
    mount_ps(&server).await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(show("sha256-one"))
        .mount(&server)
        .await;
    // Preflight asks for the version once (1 call); then the server is down
    // for the chat attempts (3) and the first wait polls (4 calls), then up.
    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"version": "0.0.1"})))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(4)
        .with_priority(2)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"version": "0.0.1"})))
        .with_priority(3)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(3)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_ok())
        .with_priority(2)
        .mount(&server)
        .await;

    let m = monitor(&server, 10, Duration::from_secs(5));
    let (answer, _) = m
        .infer_typed::<Answer>(request(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(answer.n, 3);
    let recovery: Vec<_> = m
        .checks()
        .into_iter()
        .filter(|c| c.kind == CheckKind::Recovery)
        .collect();
    assert!(
        recovery.iter().any(|c| c.fully_on_gpu == Some(true)),
        "{recovery:?}"
    );
    assert!(m.abort_error().is_none());
}

#[tokio::test]
async fn server_that_never_returns_fails_the_item() {
    let server = MockServer::start().await;
    mount_ps(&server).await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(show("sha256-one"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"version": "0.0.1"})))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(503))
        .with_priority(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let m = monitor(&server, 10, Duration::from_millis(60));
    let e = m
        .infer_typed::<Answer>(request(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(e.message.contains("did not answer"), "{e}");
}
