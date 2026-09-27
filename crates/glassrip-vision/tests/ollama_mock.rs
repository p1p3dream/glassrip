//! Ollama backend behavior against a mocked HTTP server. Synthetic data only.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use glassrip_vision::ollama::{self_test_image, SelfTestAnswer, SelfTestColor, YesNo};
use glassrip_vision::{
    GenerationOptions, OllamaBackend, OllamaConfig, VisionBackend, VisionError, VisionRequest,
};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{body_partial_json, body_string_contains, method, path};
use wiremock::{Match, Mock, MockServer, Request, ResponseTemplate};

const MODEL: &str = "qwen2.5vl:7b";
const DIGEST: &str = "sha256-synthetic-digest-0001";

fn config(server: &MockServer) -> OllamaConfig {
    let mut c = OllamaConfig::new(server.uri(), MODEL, 8192);
    c.backoff_min = Duration::from_millis(5);
    c.backoff_max = Duration::from_millis(20);
    c.request_timeout = Duration::from_secs(5);
    c.max_attempts = 4;
    c
}

fn backend(server: &MockServer) -> OllamaBackend {
    OllamaBackend::new(config(server)).unwrap()
}

fn chat_reply(content: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "model": MODEL,
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": true,
        "done_reason": "stop",
        "total_duration": 2_000_000_000u64,
        "load_duration": 10_000_000u64,
        "prompt_eval_count": 612,
        "prompt_eval_duration": 900_000_000u64,
        "eval_count": 17,
        "eval_duration": 300_000_000u64
    }))
}

const GOOD: &str = r#"{"dominant_color":"red","contains_text":"no"}"#;
const BAD_ENUM: &str = r#"{"dominant_color":"crimson","contains_text":"no"}"#;
const BAD_EXTRA: &str = r#"{"dominant_color":"red","contains_text":"no","note":"x"}"#;

fn request() -> VisionRequest {
    VisionRequest::for_output::<SelfTestAnswer>(
        "Describe the synthetic image.",
        self_test_image().unwrap(),
        GenerationOptions {
            seed: 7,
            num_predict: 64,
        },
    )
    .unwrap()
}

async fn chat_requests(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r.url.path() == "/api/chat")
        .map(|r| serde_json::from_slice(&r.body).unwrap())
        .collect()
}

#[tokio::test]
async fn success_sends_expected_request_shape() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_partial_json(json!({
            "model": MODEL,
            "stream": false,
            "think": false,
            "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 7, "num_ctx": 8192, "num_predict": 64}
        })))
        .respond_with(chat_reply(GOOD))
        .expect(1)
        .mount(&server)
        .await;

    let b = backend(&server);
    let raw = b.infer(request(), CancellationToken::new()).await.unwrap();
    assert_eq!(raw.attempts, 1);
    assert!(!raw.repaired);
    assert_eq!(raw.prompt_eval_count, Some(612));
    assert_eq!(raw.eval_count, Some(17));
    assert_eq!(raw.durations.total, Some(Duration::from_secs(2)));
    assert_eq!(raw.done_reason.as_deref(), Some("stop"));
    let answer: SelfTestAnswer = raw.decode().unwrap();
    assert_eq!(answer.dominant_color, SelfTestColor::Red);
    assert_eq!(answer.contains_text, YesNo::No);

    let body = &chat_requests(&server).await[0];
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["images"].as_array().unwrap().len(), 1);
    let prompt = messages[0]["content"].as_str().unwrap();
    assert!(
        prompt.contains("\"dominant_color\""),
        "schema text must be in the prompt"
    );
    assert_eq!(
        body["format"]["properties"]["contains_text"]["enum"],
        json!(["yes", "no"])
    );
    assert!(body["format"].get("$schema").is_none());
}

#[tokio::test]
async fn server_error_then_success() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(503).set_body_string("busy"))
        .up_to_n_times(2)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(GOOD))
        .with_priority(2)
        .mount(&server)
        .await;

    let raw = backend(&server)
        .infer(request(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(raw.attempts, 3);
}

#[tokio::test]
async fn server_errors_exhaust_retries() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(500).set_body_string("runner crashed"))
        .expect(4)
        .mount(&server)
        .await;
    let err = backend(&server)
        .infer(request(), CancellationToken::new())
        .await
        .unwrap_err();
    match err {
        VisionError::RetriesExhausted { attempts, last } => {
            assert_eq!(attempts, 4);
            assert!(last.contains("500"), "{last}");
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn too_many_requests_honors_retry_after() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "1"))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(GOOD))
        .with_priority(2)
        .mount(&server)
        .await;

    let start = Instant::now();
    let raw = backend(&server)
        .infer(request(), CancellationToken::new())
        .await
        .unwrap();
    // Backoff alone is at most 20 ms here, so a wait near 1 s proves Retry-After was used.
    assert!(
        start.elapsed() >= Duration::from_millis(950),
        "{:?}",
        start.elapsed()
    );
    assert_eq!(raw.attempts, 2);
}

#[tokio::test]
async fn client_error_is_not_retried() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(400).set_body_string(r#"{"error":"bad request"}"#))
        .expect(1)
        .mount(&server)
        .await;
    let err = backend(&server)
        .infer(request(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(err, VisionError::Http { status: 400, .. }),
        "{err:?}"
    );
}

#[tokio::test]
async fn timeout_is_retried_then_reported() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(GOOD).set_delay(Duration::from_millis(600)))
        .expect(2)
        .mount(&server)
        .await;
    let mut c = config(&server);
    c.request_timeout = Duration::from_millis(100);
    c.max_attempts = 2;
    let err = OllamaBackend::new(c)
        .unwrap()
        .infer(request(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(err, VisionError::Timeout { attempts: 2, .. }),
        "{err:?}"
    );
}

#[tokio::test]
async fn timeout_then_server_errors_reports_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(GOOD).set_delay(Duration::from_millis(600)))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(503).set_body_string("busy"))
        .with_priority(2)
        .mount(&server)
        .await;
    let mut c = config(&server);
    c.request_timeout = Duration::from_millis(100);
    c.max_attempts = 3;
    let err = OllamaBackend::new(c)
        .unwrap()
        .infer(request(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(err, VisionError::Timeout { attempts: 3, .. }),
        "{err:?}"
    );
}

#[tokio::test]
async fn not_found_on_non_model_endpoint_is_generic() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(ResponseTemplate::new(404).set_body_string("404 page not found"))
        .mount(&server)
        .await;
    let err = backend(&server).placement().await.unwrap_err();
    match err {
        VisionError::Http { path, status, .. } => {
            assert_eq!(path, "/api/ps");
            assert_eq!(status, 404);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn schema_invalid_then_repaired() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_string_contains("failed validation"))
        .respond_with(chat_reply(GOOD))
        .with_priority(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(BAD_ENUM))
        .with_priority(2)
        .expect(1)
        .mount(&server)
        .await;

    let raw = backend(&server)
        .infer(request(), CancellationToken::new())
        .await
        .unwrap();
    assert!(raw.repaired);
    assert_eq!(raw.json["dominant_color"], "red");

    let bodies = chat_requests(&server).await;
    let repair = &bodies[1]["messages"];
    assert_eq!(repair.as_array().unwrap().len(), 3);
    assert_eq!(repair[1]["content"], BAD_ENUM);
    let hint = repair[2]["content"].as_str().unwrap();
    assert!(
        hint.contains("/dominant_color"),
        "repair hint must name the path: {hint}"
    );
    // The single image stays attached to the original user turn only.
    assert!(repair[2].get("images").is_none());
}

/// Independent budget check on a chat request: every message's content bytes / 3,
/// plus the image tokens and `num_predict`, must fit in `num_ctx`.
struct WithinBudget {
    image_tokens: u64,
}

impl Match for WithinBudget {
    fn matches(&self, request: &Request) -> bool {
        let Ok(body) = serde_json::from_slice::<Value>(&request.body) else {
            return false;
        };
        let content_bytes: u64 = body["messages"]
            .as_array()
            .map(|ms| {
                ms.iter()
                    .map(|m| m["content"].as_str().map_or(0, |c| c.len() as u64))
                    .sum()
            })
            .unwrap_or(u64::MAX / 2);
        let num_ctx = body["options"]["num_ctx"].as_u64().unwrap_or(0);
        let num_predict = body["options"]["num_predict"]
            .as_u64()
            .unwrap_or(u64::MAX / 2);
        content_bytes.div_ceil(3) + self.image_tokens + num_predict <= num_ctx
    }
}

#[tokio::test]
async fn huge_invalid_output_repair_stays_within_num_ctx() {
    let server = MockServer::start().await;
    let image_tokens = u64::from(self_test_image().unwrap().tokens());
    // About 10k tokens of invalid output against a 2048-token context.
    let huge = format!("{{\"dominant_color\":\"{}\"}}", "crimson ".repeat(4000));
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_string_contains("failed validation"))
        .and(WithinBudget { image_tokens })
        .respond_with(chat_reply(GOOD))
        .with_priority(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(&huge))
        .with_priority(2)
        .expect(1)
        .mount(&server)
        .await;
    let mut c = OllamaConfig::new(server.uri(), MODEL, 2048);
    c.backoff_min = Duration::from_millis(5);
    let raw = OllamaBackend::new(c)
        .unwrap()
        .infer(request(), CancellationToken::new())
        .await
        .unwrap();
    assert!(raw.repaired);
    let bodies = chat_requests(&server).await;
    let echo = bodies[1]["messages"][1]["content"].as_str().unwrap();
    assert!(echo.len() < huge.len());
    assert!(echo.starts_with("{\"dominant_color\":\"crimson"));
    assert!(echo.contains("bytes omitted"));
}

#[tokio::test]
async fn normal_repair_echoes_output_verbatim() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_string_contains("failed validation"))
        .respond_with(chat_reply(GOOD))
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(BAD_EXTRA))
        .with_priority(2)
        .mount(&server)
        .await;
    let raw = backend(&server)
        .infer(request(), CancellationToken::new())
        .await
        .unwrap();
    assert!(raw.repaired);
    let bodies = chat_requests(&server).await;
    assert_eq!(bodies[1]["messages"][1]["role"], "assistant");
    assert_eq!(bodies[1]["messages"][1]["content"], BAD_EXTRA);
    assert!(!bodies[1]["messages"][2]["content"]
        .as_str()
        .unwrap()
        .contains("not repeated"));
}

#[tokio::test]
async fn schema_invalid_twice_is_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(BAD_EXTRA))
        .expect(2)
        .mount(&server)
        .await;
    let err = backend(&server)
        .infer(request(), CancellationToken::new())
        .await
        .unwrap_err();
    match err {
        VisionError::SchemaInvalid {
            attempts,
            errors,
            raw_text,
        } => {
            assert_eq!(attempts, 2);
            assert_eq!(raw_text, BAD_EXTRA);
            assert!(!errors.is_empty());
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn non_json_output_is_schema_invalid() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply("the image is red"))
        .expect(2)
        .mount(&server)
        .await;
    let err = backend(&server)
        .infer(request(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(err, VisionError::SchemaInvalid { attempts: 2, .. }),
        "{err:?}"
    );
}

#[tokio::test]
async fn cancellation_stops_a_slow_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(GOOD).set_delay(Duration::from_secs(3)))
        .mount(&server)
        .await;
    let b = backend(&server);
    let cancel = CancellationToken::new();
    let c2 = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        c2.cancel();
    });
    let start = Instant::now();
    let err = b.infer(request(), cancel).await.unwrap_err();
    assert!(matches!(err, VisionError::Cancelled));
    assert!(start.elapsed() < Duration::from_secs(1));
}

async fn mount_metadata(server: &MockServer, ps_models: Value) {
    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"version": "0.0.0-test"})))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "details": {"family": "qwen25vl"}, "capabilities": ["completion", "vision"]
        })))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": [
            {"name": "other:1b", "model": "other:1b", "digest": "sha256-other"},
            {"name": MODEL, "model": MODEL, "digest": DIGEST}
        ]})))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": ps_models})))
        .mount(server)
        .await;
}

fn ps_entry(size: u64, size_vram: u64) -> Value {
    json!({"name": MODEL, "model": MODEL, "size": size, "size_vram": size_vram,
           "digest": DIGEST, "context_length": 8192})
}

#[tokio::test]
async fn digest_from_tags_when_show_has_none() {
    let server = MockServer::start().await;
    mount_metadata(&server, json!([])).await;
    let b = backend(&server);
    assert_eq!(b.id().digest, None);
    assert_eq!(b.resolve_digest().await.unwrap(), DIGEST);
    let id = b.id();
    assert_eq!(id.backend, "ollama");
    assert_eq!(id.model, MODEL);
    assert_eq!(id.digest.as_deref(), Some(DIGEST));
}

#[tokio::test]
async fn digest_from_show_when_present_and_change_detected() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"digest": "sha256-first"})))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"digest": "sha256-second"})))
        .with_priority(2)
        .mount(&server)
        .await;
    let b = backend(&server);
    assert_eq!(b.resolve_digest().await.unwrap(), "sha256-first");
    let err = b.resolve_digest().await.unwrap_err();
    assert!(matches!(err, VisionError::DigestChanged { .. }), "{err:?}");
}

#[tokio::test]
async fn missing_model_reports_pull_command() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(
            ResponseTemplate::new(404)
                .set_body_string(r#"{"error":"model 'qwen2.5vl:7b' not found"}"#),
        )
        .mount(&server)
        .await;
    let err = backend(&server).resolve_digest().await.unwrap_err();
    assert!(matches!(err, VisionError::ModelNotFound { .. }), "{err:?}");
    assert!(err.to_string().contains("ollama pull qwen2.5vl:7b"));
}

#[tokio::test]
async fn preflight_fully_on_gpu() {
    let server = MockServer::start().await;
    mount_metadata(&server, json!([ps_entry(6_400_000_000, 6_400_000_000)])).await;
    let b = backend(&server);
    let p = b.preflight().await.unwrap();
    assert!(p.fully_on_gpu);
    assert_eq!(p.concurrency_hint, 4);
    assert_eq!(p.context_length, Some(8192));
    assert_eq!(b.id().server_version.as_deref(), Some("0.0.0-test"));
    assert_eq!(b.id().digest.as_deref(), Some(DIGEST));
}

#[tokio::test]
async fn preflight_detects_spill() {
    let server = MockServer::start().await;
    mount_metadata(&server, json!([ps_entry(9_000_000_000, 7_000_000_000)])).await;
    let err = backend(&server).preflight().await.unwrap_err();
    match err {
        VisionError::Spill {
            size, size_vram, ..
        } => {
            assert_eq!(size, 9_000_000_000);
            assert_eq!(size_vram, 7_000_000_000);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn preflight_spill_allowed_reduces_concurrency() {
    let server = MockServer::start().await;
    mount_metadata(&server, json!([ps_entry(9_000_000_000, 7_000_000_000)])).await;
    let mut c = config(&server);
    c.allow_spill = true;
    let p = OllamaBackend::new(c.clone())
        .unwrap()
        .preflight()
        .await
        .unwrap();
    assert!(!p.fully_on_gpu);
    // Spill allowed: slots halved (4 -> 2), never below 1.
    assert_eq!(p.concurrency_hint, 2);
    c.slots = 1;
    let p = OllamaBackend::new(c).unwrap().preflight().await.unwrap();
    assert_eq!(p.concurrency_hint, 1);
}

#[tokio::test]
async fn preflight_loads_model_when_not_resident() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": []})))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    mount_metadata(&server, json!([ps_entry(6_000, 6_000)])).await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_partial_json(
            json!({"messages": [], "keep_alive": "30m", "options": {"num_ctx": 8192}}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"done": true})))
        .expect(1)
        .mount(&server)
        .await;
    let p = backend(&server).preflight().await.unwrap();
    assert!(p.fully_on_gpu);
}

#[tokio::test]
async fn unload_sends_keep_alive_zero() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_partial_json(json!({"model": MODEL, "keep_alive": 0})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"done": true})))
        .expect(1)
        .mount(&server)
        .await;
    backend(&server).unload().await.unwrap();
}

#[tokio::test]
async fn self_test_passes_and_fails_without_repair() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(GOOD))
        .mount(&server)
        .await;
    let report = backend(&server)
        .self_test(CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.answer.dominant_color, SelfTestColor::Red);

    let bad = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(BAD_ENUM))
        .expect(1)
        .mount(&bad)
        .await;
    let err = backend(&bad)
        .self_test(CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(err, VisionError::SelfTestFailed(_)), "{err:?}");
}

fn length_reply(content: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "model": MODEL,
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": true,
        "done_reason": "length",
        "prompt_eval_count": 612,
        "eval_count": 64
    }))
}

#[tokio::test]
async fn length_stop_is_truncated_without_a_repair_request() {
    let server = MockServer::start().await;
    // Indented output that ran out of tokens mid-document (synthetic).
    let cut = "{\n  \"dominant_color\": \"red\",\n  \"contains_";
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(length_reply(cut))
        .expect(1)
        .mount(&server)
        .await;
    let err = backend(&server)
        .infer(request(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(err.is_truncated(), "{err:?}");
    match err {
        VisionError::Truncated {
            num_predict,
            eval_count,
            raw_text,
        } => {
            assert_eq!(num_predict, 64);
            assert_eq!(eval_count, Some(64));
            assert_eq!(raw_text, cut);
        }
        other => panic!("unexpected {other:?}"),
    }
    // One request only: echoing a cut-off reply back cannot fix it.
    assert_eq!(chat_requests(&server).await.len(), 1);
}

#[tokio::test]
async fn length_stop_with_valid_json_is_still_truncated() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(length_reply(GOOD))
        .expect(1)
        .mount(&server)
        .await;
    let err = backend(&server)
        .infer(request(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(err.is_truncated(), "{err:?}");
}

#[tokio::test]
async fn format_is_the_structural_schema() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_reply(GOOD))
        .mount(&server)
        .await;
    let req = request();
    let expected = req.schema.grammar_value();
    backend(&server)
        .infer(req, CancellationToken::new())
        .await
        .unwrap();
    let body = &chat_requests(&server).await[0];
    assert_eq!(body["format"], expected);
    let text = body["format"].to_string();
    assert!(
        !text.contains("\"description\"") && !text.contains("\"title\""),
        "{text}"
    );
    assert_eq!(body["format"]["additionalProperties"], json!(false));
}

#[tokio::test]
async fn model_size_comes_from_tags_and_show() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [
                {"name": "other:1b", "model": "other:1b", "digest": "x", "size": 1},
                {"name": MODEL, "model": MODEL, "digest": DIGEST, "size": 21_000_000_000u64}
            ]
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "digest": DIGEST,
            "details": {"parameter_size": "32.8B", "quantization_level": "Q4_K_M"}
        })))
        .mount(&server)
        .await;
    let size = backend(&server).model_size().await.unwrap();
    assert_eq!(size.file_bytes, Some(21_000_000_000));
    assert_eq!(size.parameter_size_b, Some(32.8));
}

/// NDJSON body of a streamed chat reply: one line per piece, then the final line.
fn stream_body(pieces: &[&str], done: bool) -> String {
    let mut s = String::new();
    for p in pieces {
        s.push_str(
            &json!({"model": MODEL, "message": {"role": "assistant", "content": p}, "done": false})
                .to_string(),
        );
        s.push('\n');
    }
    if done {
        s.push_str(
            &json!({
                "model": MODEL, "message": {"role": "assistant", "content": ""}, "done": true,
                "done_reason": "stop", "prompt_eval_count": 612, "eval_count": 17,
                "total_duration": 2_000_000_000u64
            })
            .to_string(),
        );
        s.push('\n');
    }
    s
}

fn guarded() -> VisionRequest {
    let mut r = request();
    r.repetition_guard = Some(glassrip_vision::repetition::RepetitionParams::default());
    r
}

#[tokio::test]
async fn guarded_requests_stream_and_forward_sampling_overrides() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_partial_json(json!({
            "stream": true,
            "options": {"repeat_penalty": 1.25, "repeat_last_n": 512, "temperature": 0.25, "seed": 8}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(stream_body(
            &[
                r#"{"dominant_color":"#,
                r#""red","contains"#,
                r#"_text":"no"}"#,
            ],
            true,
        )))
        .expect(1)
        .mount(&server)
        .await;
    let mut r = guarded();
    r.sampling.repeat_penalty = Some(1.25);
    r.sampling.repeat_last_n = Some(512);
    r.sampling.temperature = Some(0.25);
    r.options.seed = 8;
    let raw = backend(&server)
        .infer(r, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(raw.raw_text, GOOD);
    assert_eq!(raw.done_reason.as_deref(), Some("stop"));
    assert_eq!(raw.eval_count, Some(17));
    let answer: SelfTestAnswer = raw.decode().unwrap();
    assert_eq!(answer.dominant_color, SelfTestColor::Red);
}

#[tokio::test]
async fn a_looping_stream_is_stopped_before_it_ends() {
    let server = MockServer::start().await;
    // A node list, then edges stepping into ids it never declared.
    let mut pieces: Vec<String> =
        vec!["{\"nodes\": [{\"local_id\": \"n1\", \"text\": \"Api\"}], \"edges\": [".into()];
    for k in 0..40 {
        pieces.push(format!(
            "{{\"src\": \"n{k}\", \"dst\": \"n{}\", \"label\": \"Relay\", \"style\": \"solid\"}},",
            k + 1
        ));
    }
    let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let total: usize = refs.iter().map(|p| p.len()).sum();
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_string(stream_body(&refs, false)))
        .expect(1)
        .mount(&server)
        .await;
    match backend(&server)
        .infer(guarded(), CancellationToken::new())
        .await
    {
        Err(VisionError::Repetition {
            finding, raw_text, ..
        }) => {
            assert!(
                raw_text.len() < total,
                "stopped early: {} of {total}",
                raw_text.len()
            );
            assert!(finding.repeats >= 6, "{finding:?}");
            assert!(finding.pattern.contains("Relay"));
        }
        other => panic!("expected a repetition stop, got {other:?}"),
    }
    assert_eq!(
        chat_requests(&server).await.len(),
        1,
        "a loop is not retried here"
    );
}

#[tokio::test]
async fn unguarded_requests_do_not_stream_or_send_overrides() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_partial_json(json!({"stream": false})))
        .respond_with(chat_reply(GOOD))
        .expect(1)
        .mount(&server)
        .await;
    let body = &{
        backend(&server)
            .infer(request(), CancellationToken::new())
            .await
            .unwrap();
        chat_requests(&server).await
    }[0];
    let opts = body["options"].as_object().unwrap();
    assert!(
        !opts.contains_key("repeat_penalty"),
        "no override, server default"
    );
}

fn guarded_board() -> VisionRequest {
    let mut r = VisionRequest::for_output::<glassrip_vision::board::BoardReadOutput>(
        "Read the synthetic board.",
        self_test_image().unwrap(),
        GenerationOptions {
            seed: 7,
            num_predict: 2048,
        },
    )
    .unwrap();
    r.repetition_guard = Some(glassrip_vision::repetition::RepetitionParams::default());
    r
}

/// Six identical cards in one row, sequential ids, evenly spaced boxes.
fn card_row_pieces() -> Vec<String> {
    let mut pieces = vec!["{\"nodes\": [".to_string()];
    for c in 0..6u32 {
        let x = 100 + c * 120;
        pieces.push(format!(
            "{}{{\"local_id\": \"n{}\", \"text\": \"Card\", \"bbox_2d\": [{x}, 900, {}, 930], \"conf\": 0.9}}",
            if c == 0 { "" } else { ", " },
            30 + c,
            x + 80
        ));
    }
    pieces.push(
        "], \"edges\": [], \"stickies\": [], \"owner_tags\": [], \"other_visible_text\": [], \"confidence\": 0.9}"
            .to_string(),
    );
    pieces
}

fn final_line(done_reason: &str) -> String {
    let mut s = json!({
        "model": MODEL, "message": {"role": "assistant", "content": ""}, "done": true,
        "done_reason": done_reason, "prompt_eval_count": 612, "eval_count": 300,
        "total_duration": 2_000_000_000u64
    })
    .to_string();
    s.push('\n');
    s
}

/// Codex 5 / GLM B1: a complete, valid reading of a regular board streams to the
/// end and is accepted.
#[tokio::test]
async fn a_complete_regular_board_is_accepted() {
    let server = MockServer::start().await;
    let pieces = card_row_pieces();
    let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_string(stream_body(&refs, true)))
        .expect(1)
        .mount(&server)
        .await;
    let raw = backend(&server)
        .infer(guarded_board(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(raw.done_reason.as_deref(), Some("stop"));
    assert_eq!(raw.json["nodes"].as_array().unwrap().len(), 6);
}

/// A complete reply that validates is accepted even when the detector would
/// flag it: the guard stops runaway generations, it does not judge answers.
#[tokio::test]
async fn a_complete_valid_reply_is_never_rejected_for_repetition() {
    let server = MockServer::start().await;
    let edge = "{\"src\": \"n1\", \"dst\": \"n2\", \"label\": \"\", \"label_bbox_2d\": [0, 0, 0, 0], \"style\": \"solid\", \"conf\": 0.9}";
    let body = format!(
        "{{\"nodes\": [{{\"local_id\": \"n1\", \"text\": \"Api\", \"bbox_2d\": [10, 10, 90, 40], \"conf\": 0.9}}, {{\"local_id\": \"n2\", \"text\": \"Db\", \"bbox_2d\": [200, 10, 290, 40], \"conf\": 0.9}}], \"edges\": [{edge}, {edge}, {edge}], \"stickies\": [], \"owner_tags\": [], \"other_visible_text\": [], \"confidence\": 0.8}}"
    );
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_string(stream_body(&[&body], true)))
        .expect(1)
        .mount(&server)
        .await;
    let raw = backend(&server)
        .infer(guarded_board(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(raw.json["edges"].as_array().unwrap().len(), 3);
}

/// GLM B1: a reply cut off at the output limit inside a regular run is a
/// truncation (the budget retry), not a repetition (the penalty retry).
#[tokio::test]
async fn a_regular_run_cut_at_the_limit_is_a_truncation() {
    let server = MockServer::start().await;
    let pieces = card_row_pieces();
    let mut body = String::new();
    for p in &pieces[..pieces.len() - 1] {
        body.push_str(
            &json!({"model": MODEL, "message": {"role": "assistant", "content": p}, "done": false})
                .to_string(),
        );
        body.push('\n');
    }
    body.push_str(&final_line("length"));
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(1)
        .mount(&server)
        .await;
    match backend(&server)
        .infer(guarded_board(), CancellationToken::new())
        .await
    {
        Err(VisionError::Truncated { .. }) => {}
        other => panic!("expected a truncation, got {other:?}"),
    }
}
