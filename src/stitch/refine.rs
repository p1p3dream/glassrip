use anyhow::{bail, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use tokio::time::Duration;

use crate::retry::{honors_retry_after, is_retryable_status, parse_retry_after, RetryPolicy};

/// Default model for `--refine`.
pub const DEFAULT_REFINE_MODEL: &str = "claude-opus-5-5";

const API_BASE_URL: &str = "https://api.anthropic.com";

const REFINE_SYSTEM: &str = r#"You are cleaning up a chunk of VLM-extracted text from a video recording. The text was extracted frame-by-frame and mechanically deduplicated, but quality issues remain.

Fix these issues in this chunk:

1. Remove diff artifacts: lines starting with @@, ---, +++, diff --git. For -/+ paired lines with identical content, keep one copy without the prefix.
2. Remove duplicate lines: consecutive identical or near-identical lines, keep the better version.
3. Fix garbled text: obvious OCR errors, gibberish sequences, broken words.
4. Remove within-line repetitions: collapse patterns repeated 3+ times in a single line.
5. Remove duplicate sections within this chunk.
6. Fix structural issues: empty headers, broken markdown, stray code fence markers.
7. Preserve all legitimate content.

Output ONLY the cleaned text. No commentary, no explanations, no markdown fences wrapping the output."#;

#[derive(Deserialize)]
struct ClaudeResponse {
    content: Vec<ContentBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(default)]
    text: String,
}

pub async fn refine_text(text: &str, model: &str, num_agents: usize) -> Result<String> {
    if text.trim().is_empty() {
        return Ok(text.to_string());
    }

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY not set (required for --refine)")?;

    let chunks = split_into_chunks(text, num_agents);
    let total = chunks.len();
    let line_count = text.lines().count();
    println!("Refining {line_count} lines via {total} Claude agents ({model})...");

    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    let policy = RetryPolicy {
        base_delay: Duration::from_secs(2),
        ..RetryPolicy::default()
    };
    let mut handles = Vec::with_capacity(total);
    for (i, chunk) in chunks.into_iter().enumerate() {
        let client = client.clone();
        let api_key = api_key.clone();
        let model = model.to_string();
        let policy = policy.clone();
        handles.push(tokio::spawn(async move {
            refine_chunk(&client, API_BASE_URL, &api_key, &model, &chunk, i + 1, total, &policy)
                .await
        }));
    }

    let mut results = Vec::with_capacity(total);
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle
            .await
            .with_context(|| format!("Agent {} panicked", i + 1))?
            .with_context(|| format!("Agent {} failed", i + 1))?;
        let chunk_lines = result.lines().count();
        println!("  Agent {}/{}: {chunk_lines} lines", i + 1, total);
        results.push(result);
    }

    let final_text = results.join("\n");
    let refined_count = final_text.lines().count();
    println!("  Refined to {refined_count} lines (from {line_count})");

    Ok(final_text)
}

#[allow(clippy::too_many_arguments)]
async fn refine_chunk(
    client: &Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    chunk: &str,
    chunk_num: usize,
    total_chunks: usize,
    policy: &RetryPolicy,
) -> Result<String> {
    let payload = serde_json::json!({
        "model": model,
        "max_tokens": 32768,
        "system": REFINE_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": format!(
                    "Clean up this text (chunk {chunk_num} of {total_chunks}):\n\n{chunk}"
                )
            }
        ]
    });

    let url = format!("{base_url}/v1/messages");
    let max_attempts = policy.max_attempts.max(1);
    let mut last_err = String::from("unknown");
    let mut retry_after = None;

    for attempt in 0..max_attempts {
        if attempt > 0 {
            let wait = policy.delay_for(attempt - 1, retry_after.take());
            println!(
                "  Agent {chunk_num}: retry {attempt}/{}, waiting {wait:.2?}",
                max_attempts - 1
            );
            tokio::time::sleep(wait).await;
        }

        let resp = match client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&payload)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                last_err = format!("request to Claude API failed: {e}");
                continue;
            }
        };

        let status = resp.status();
        if is_retryable_status(status) {
            if honors_retry_after(status) {
                retry_after = parse_retry_after(resp.headers());
            }
            let body = resp.text().await.unwrap_or_default();
            last_err = format!("Claude API {status}: {body}");
            continue;
        }

        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("Claude API {status}: {body}");
        }

        let data: ClaudeResponse = resp
            .json()
            .await
            .context("Failed to parse Claude response")?;

        if let Some(ref reason) = data.stop_reason {
            if reason == "max_tokens" {
                bail!("Agent {chunk_num}: response truncated (hit max_tokens). Input chunk may be too large");
            }
            if reason == "refusal" {
                bail!("Agent {chunk_num}: model refused to process chunk");
            }
        }

        let text = data
            .content
            .into_iter()
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");

        let cleaned = strip_fences(text.trim());
        if cleaned.trim().is_empty() && !chunk.trim().is_empty() {
            // Accepting this would silently delete the whole chunk.
            last_err = "Claude API returned empty content for a non-empty chunk".into();
            continue;
        }
        return Ok(cleaned);
    }

    bail!("Agent {chunk_num} failed after {max_attempts} attempts: {last_err}");
}

fn strip_fences(text: &str) -> String {
    if !text.starts_with("```") {
        return text.to_string();
    }
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 2 {
        return text.to_string();
    }
    let start = 1;
    let end = if lines.last().is_some_and(|l| l.trim() == "```") {
        lines.len() - 1
    } else {
        lines.len()
    };
    lines[start..end].join("\n")
}

fn split_into_chunks(text: &str, n: usize) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() || n == 0 {
        return vec![text.to_string()];
    }
    let n = n.min(lines.len());
    let chunk_size = lines.len() / n;
    let remainder = lines.len() % n;

    let mut chunks = Vec::with_capacity(n);
    let mut start = 0;
    for i in 0..n {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + chunk_size + extra;
        chunks.push(lines[start..end].join("\n"));
        start = end;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::{serve, Reply};

    fn fast_policy() -> RetryPolicy {
        RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(50),
            max_retry_after: Duration::from_secs(5),
        }
    }

    fn text_reply(text: &str) -> Reply {
        Reply::json(
            200,
            serde_json::json!({
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn"
            }),
        )
    }

    async fn run_chunk(url: &str) -> Result<String> {
        let client = Client::new();
        refine_chunk(&client, url, "key", "m", "some text", 1, 1, &fast_policy()).await
    }

    #[tokio::test]
    async fn refine_retries_transport_errors() {
        let server = serve(vec![Reply::Drop, text_reply("clean")]).await;
        assert_eq!(run_chunk(&server.url).await.unwrap(), "clean");
        assert_eq!(server.hits(), 2);
    }

    #[tokio::test]
    async fn refine_honors_retry_after_on_429() {
        let server = serve(vec![
            Reply::json(429, serde_json::json!({})).header("Retry-After", "1"),
            text_reply("clean"),
        ])
        .await;
        let started = std::time::Instant::now();
        assert_eq!(run_chunk(&server.url).await.unwrap(), "clean");
        assert!(started.elapsed() >= Duration::from_millis(950), "Retry-After ignored");
    }

    #[tokio::test]
    async fn refine_rejects_empty_content() {
        let server = serve(vec![text_reply("")]).await;
        let err = run_chunk(&server.url).await.unwrap_err();
        assert!(err.to_string().contains("empty content"), "{err}");
        assert_eq!(server.hits(), 3);
    }

    #[tokio::test]
    async fn refine_recovers_after_empty_content() {
        let server = serve(vec![text_reply("   "), text_reply("clean")]).await;
        assert_eq!(run_chunk(&server.url).await.unwrap(), "clean");
    }

    #[tokio::test]
    async fn refine_does_not_retry_client_errors() {
        let server = serve(vec![Reply::json(401, serde_json::json!({"error": "auth"}))]).await;
        assert!(run_chunk(&server.url).await.is_err());
        assert_eq!(server.hits(), 1);
    }

    #[test]
    fn default_refine_model_is_current() {
        assert_eq!(DEFAULT_REFINE_MODEL, "claude-opus-5-5");
    }

    #[test]
    fn split_even() {
        let text = "a\nb\nc\nd";
        let chunks = split_into_chunks(text, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "a\nb");
        assert_eq!(chunks[1], "c\nd");
    }

    #[test]
    fn split_uneven() {
        let text = "a\nb\nc\nd\ne";
        let chunks = split_into_chunks(text, 3);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "a\nb");
        assert_eq!(chunks[1], "c\nd");
        assert_eq!(chunks[2], "e");
    }

    #[test]
    fn split_more_agents_than_lines() {
        let text = "a\nb";
        let chunks = split_into_chunks(text, 5);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn strip_fences_normal() {
        let text = "```markdown\nline 1\nline 2\n```";
        assert_eq!(strip_fences(text), "line 1\nline 2");
    }

    #[test]
    fn strip_fences_no_closing() {
        let text = "```\nline 1\nline 2";
        assert_eq!(strip_fences(text), "line 1\nline 2");
    }

    #[test]
    fn strip_fences_lone_fence() {
        let text = "```";
        assert_eq!(strip_fences(text), "```");
    }

    #[test]
    fn strip_fences_not_fenced() {
        let text = "just normal text";
        assert_eq!(strip_fences(text), "just normal text");
    }
}
