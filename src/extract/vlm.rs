use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use reqwest::Client;
use serde::Deserialize;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};

use crate::retry::{honors_retry_after, is_retryable_status, parse_retry_after, RetryPolicy};

const EXTRACT_PROMPT: &str = "\
Extract the exact source code visible in this image. \
Output ONLY the code, no explanations or markdown fences. \
Preserve exact indentation. \
Ignore line numbers, file tabs, sidebar, minimap, terminal chrome.";

const EXTRACT_WITH_CONTEXT_TEMPLATE: &str = "\
The presenter said: \"{narration}\"\n\n\
Extract the exact source code visible in this image. \
Output ONLY the code, no explanations or markdown fences. \
Preserve exact indentation. \
Ignore line numbers, file tabs, sidebar, minimap, terminal chrome.";

const DIFF_TEMPLATE: &str = "\
Previous frame's code ended with:\n```\n{previous_tail}\n```\n\n\
Extract the code in this new frame exactly as shown. Output ONLY the code.";

/// Default per-request timeout for VLM calls, in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

#[derive(Deserialize)]
struct OllamaResponse {
    #[serde(default)]
    response: String,
}

/// Settings shared by every request in a VLM batch.
#[derive(Debug, Clone)]
pub struct VlmOptions {
    /// Per-request timeout applied to the shared HTTP client.
    pub timeout: Duration,
    pub retry: RetryPolicy,
}

impl Default for VlmOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            retry: RetryPolicy::default(),
        }
    }
}

pub async fn extract_code_from_frame(
    frame_path: &Path,
    ollama_host: &str,
    model: &str,
    client: Option<&Client>,
    narration: Option<&str>,
    previous_code: Option<&str>,
    timeout_secs: u64,
) -> Result<String> {
    extract_with_policy(
        frame_path,
        ollama_host,
        model,
        client,
        narration,
        previous_code,
        Duration::from_secs(timeout_secs),
        &RetryPolicy::default(),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn extract_with_policy(
    frame_path: &Path,
    ollama_host: &str,
    model: &str,
    client: Option<&Client>,
    narration: Option<&str>,
    previous_code: Option<&str>,
    timeout: Duration,
    policy: &RetryPolicy,
) -> Result<String> {
    let image_b64 = encode_image(frame_path).await?;

    let prompt = if let Some(prev) = previous_code {
        let lines: Vec<&str> = prev.lines().collect();
        let start = lines.len().saturating_sub(40);
        let tail = lines[start..].join("\n");
        DIFF_TEMPLATE.replace("{previous_tail}", &tail)
    } else if let Some(narr) = narration {
        let truncated = if narr.len() > 500 {
            let end = narr
                .char_indices()
                .take_while(|(i, _)| *i < 500)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0);
            &narr[..end]
        } else {
            narr
        };
        EXTRACT_WITH_CONTEXT_TEMPLATE.replace("{narration}", truncated)
    } else {
        EXTRACT_PROMPT.to_string()
    };

    let payload = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": false,
        "options": {
            "temperature": 0.1,
            "num_predict": 4096,
            "num_ctx": 8192
        }
    });

    let owned_client;
    let client = match client {
        Some(c) => c,
        None => {
            owned_client = Client::builder().timeout(timeout).build()?;
            &owned_client
        }
    };

    let url = format!("{ollama_host}/api/generate");
    let max_attempts = policy.max_attempts.max(1);
    let mut attempt: u32 = 0;

    loop {
        attempt += 1;
        let (err, retry_after) = match client.post(&url).json(&payload).send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    let data: OllamaResponse = resp
                        .json()
                        .await
                        .context("failed to parse Ollama response")?;
                    let cleaned = clean_response(&data.response);
                    if cleaned.trim().is_empty() {
                        eprintln!("  Warning: empty VLM response for {}", frame_path.display());
                    }
                    return Ok(cleaned);
                }
                if !is_retryable_status(status) {
                    let body = resp.text().await.unwrap_or_default();
                    anyhow::bail!("Ollama API returned {status}: {body}");
                }
                let retry_after = if honors_retry_after(status) {
                    parse_retry_after(resp.headers())
                } else {
                    None
                };
                (anyhow::anyhow!("Ollama API returned {status}"), retry_after)
            }
            Err(e) => (anyhow::Error::new(e).context("Ollama request failed"), None),
        };

        if attempt >= max_attempts {
            return Err(err.context(format!(
                "giving up on {} after {attempt} attempt(s)",
                frame_path.display()
            )));
        }
        let wait = policy.delay_for(attempt - 1, retry_after);
        eprintln!(
            "  Retry {attempt}/{} for {} after {err:#}, waiting {wait:.2?}",
            max_attempts - 1,
            frame_path.display()
        );
        sleep(wait).await;
    }
}

/// Extract every frame concurrently. The outer `Result` covers setup only
/// (building the HTTP client); each frame gets its own `Result` so one bad
/// frame never aborts the batch. Output order matches `frame_paths`.
pub async fn extract_batch(
    frame_paths: &[&Path],
    ollama_host: &str,
    model: &str,
    max_concurrent: usize,
    options: &VlmOptions,
) -> Result<Vec<Result<String>>> {
    let sem = Arc::new(Semaphore::new(max_concurrent.max(1)));
    let client = Client::builder().timeout(options.timeout).build()?;

    let mut handles = Vec::with_capacity(frame_paths.len());

    for path in frame_paths {
        let permit = sem.clone();
        let client = client.clone();
        let host = ollama_host.to_string();
        let model = model.to_string();
        let path = path.to_path_buf();
        let timeout = options.timeout;
        let policy = options.retry.clone();

        handles.push(tokio::spawn(async move {
            let _permit = permit
                .acquire()
                .await
                .context("semaphore closed unexpectedly")?;
            extract_with_policy(
                &path,
                &host,
                &model,
                Some(&client),
                None,
                None,
                timeout,
                &policy,
            )
            .await
        }));
    }

    let mut results = Vec::with_capacity(handles.len());
    for (i, handle) in handles.into_iter().enumerate() {
        results.push(match handle.await {
            Ok(r) => r,
            Err(e) => Err(anyhow::anyhow!(
                "VLM worker for frame {i} failed to complete: {e}"
            )),
        });
    }

    Ok(results)
}

async fn encode_image(path: &Path) -> Result<String> {
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("failed to read image: {}", path.display()))?;
    Ok(STANDARD.encode(bytes))
}

fn clean_response(text: &str) -> String {
    let text = text.trim();
    if text.starts_with("```") {
        let mut lines: Vec<&str> = text.lines().collect();
        // Remove opening fence
        lines.remove(0);
        // Remove closing fence
        if let Some(last) = lines.last() {
            if last.trim() == "```" {
                lines.pop();
            }
        }
        lines.join("\n")
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{serve, Reply};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    fn fast_policy(max_attempts: u32) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(50),
            max_retry_after: Duration::from_secs(5),
        }
    }

    fn opts(timeout: Duration, max_attempts: u32) -> VlmOptions {
        VlmOptions {
            timeout,
            retry: fast_policy(max_attempts),
        }
    }

    fn fake_frame(dir: &tempfile::TempDir, name: &str) -> PathBuf {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"not really a png").unwrap();
        path
    }

    fn ok_reply(text: &str) -> Reply {
        Reply::json(200, serde_json::json!({ "response": text }))
    }

    #[tokio::test]
    async fn batch_isolates_failed_frames() {
        let server = serve(vec![ok_reply("let x = 1;")]).await;
        let dir = tempfile::tempdir().unwrap();
        let good_a = fake_frame(&dir, "a.png");
        let missing = dir.path().join("missing.png");
        let good_b = fake_frame(&dir, "b.png");
        let paths = [good_a.as_path(), missing.as_path(), good_b.as_path()];

        let results = extract_batch(
            &paths,
            &server.url,
            "m",
            2,
            &opts(Duration::from_secs(5), 1),
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap(), "let x = 1;");
        let err = results[1].as_ref().unwrap_err();
        assert!(
            format!("{err:#}").contains("failed to read image"),
            "{err:#}"
        );
        assert_eq!(results[2].as_ref().unwrap(), "let x = 1;");
    }

    #[tokio::test]
    async fn batch_uses_configured_timeout() {
        let server = serve(vec![ok_reply("slow").delay(Duration::from_secs(3))]).await;
        let dir = tempfile::tempdir().unwrap();
        let frame = fake_frame(&dir, "a.png");

        let started = Instant::now();
        let results = extract_batch(
            &[frame.as_path()],
            &server.url,
            "m",
            1,
            &opts(Duration::from_millis(300), 1),
        )
        .await
        .unwrap();

        let err = results[0].as_ref().unwrap_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("timed out"),
            "{err:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "timeout not applied"
        );
    }

    #[tokio::test]
    async fn retries_429_and_honors_retry_after() {
        let server = serve(vec![
            Reply::json(429, serde_json::json!({})).header("Retry-After", "1"),
            ok_reply("done"),
        ])
        .await;
        let dir = tempfile::tempdir().unwrap();
        let frame = fake_frame(&dir, "a.png");

        let started = Instant::now();
        let out = extract_with_policy(
            &frame,
            &server.url,
            "m",
            None,
            None,
            None,
            Duration::from_secs(5),
            &fast_policy(3),
        )
        .await
        .unwrap();

        assert_eq!(out, "done");
        assert_eq!(server.hits(), 2);
        assert!(
            started.elapsed() >= Duration::from_millis(950),
            "Retry-After ignored"
        );
    }

    #[tokio::test]
    async fn retries_429_without_retry_after_using_backoff() {
        let server = serve(vec![
            Reply::json(429, serde_json::json!({})),
            ok_reply("done"),
        ])
        .await;
        let dir = tempfile::tempdir().unwrap();
        let frame = fake_frame(&dir, "a.png");

        let out = extract_with_policy(
            &frame,
            &server.url,
            "m",
            None,
            None,
            None,
            Duration::from_secs(5),
            &fast_policy(3),
        )
        .await
        .unwrap();
        assert_eq!(out, "done");
        assert_eq!(server.hits(), 2);
    }

    #[tokio::test]
    async fn retries_server_errors_then_gives_up() {
        let server = serve(vec![Reply::json(503, serde_json::json!({}))]).await;
        let dir = tempfile::tempdir().unwrap();
        let frame = fake_frame(&dir, "a.png");

        let err = extract_with_policy(
            &frame,
            &server.url,
            "m",
            None,
            None,
            None,
            Duration::from_secs(5),
            &fast_policy(3),
        )
        .await
        .unwrap_err();
        assert_eq!(server.hits(), 3);
        assert!(format!("{err:#}").contains("after 3 attempt"), "{err:#}");
    }

    #[tokio::test]
    async fn retries_dropped_connections() {
        let server = serve(vec![Reply::Drop, ok_reply("recovered")]).await;
        let dir = tempfile::tempdir().unwrap();
        let frame = fake_frame(&dir, "a.png");

        let out = extract_with_policy(
            &frame,
            &server.url,
            "m",
            None,
            None,
            None,
            Duration::from_secs(5),
            &fast_policy(3),
        )
        .await
        .unwrap();
        assert_eq!(out, "recovered");
        assert_eq!(server.hits(), 2);
    }

    #[tokio::test]
    async fn does_not_retry_client_errors() {
        let server = serve(vec![Reply::json(
            400,
            serde_json::json!({"error": "bad model"}),
        )])
        .await;
        let dir = tempfile::tempdir().unwrap();
        let frame = fake_frame(&dir, "a.png");

        let err = extract_with_policy(
            &frame,
            &server.url,
            "m",
            None,
            None,
            None,
            Duration::from_secs(5),
            &fast_policy(3),
        )
        .await
        .unwrap_err();
        assert_eq!(server.hits(), 1);
        assert!(err.to_string().contains("400"), "{err}");
    }

    #[test]
    fn default_timeout_is_120s() {
        assert_eq!(VlmOptions::default().timeout, Duration::from_secs(120));
    }
}
