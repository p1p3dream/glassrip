use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use reqwest::Client;
use serde::Deserialize;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};

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

const MAX_RETRIES: u32 = 3;

#[derive(Deserialize)]
struct OllamaResponse {
    #[serde(default)]
    response: String,
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
    let image_b64 = encode_image(frame_path)?;

    let prompt = if let Some(prev) = previous_code {
        let lines: Vec<&str> = prev.lines().collect();
        let start = lines.len().saturating_sub(40);
        let tail = lines[start..].join("\n");
        DIFF_TEMPLATE.replace("{previous_tail}", &tail)
    } else if let Some(narr) = narration {
        let truncated = if narr.len() > 500 {
            let end = narr.char_indices().take_while(|(i, _)| *i < 500).last().map(|(i, c)| i + c.len_utf8()).unwrap_or(0);
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
            owned_client = Client::builder()
                .timeout(Duration::from_secs(timeout_secs))
                .build()?;
            &owned_client
        }
    };

    let url = format!("{ollama_host}/api/generate");

    for attempt in 0..MAX_RETRIES {
        let result = client.post(&url).json(&payload).send().await;
        let should_retry = |e: &dyn std::fmt::Display, attempt: u32| -> bool {
            if attempt >= MAX_RETRIES - 1 {
                return false;
            }
            let wait = 2u64.pow(attempt);
            eprintln!("  Retry {}/{MAX_RETRIES} after {e}, waiting {wait}s", attempt + 1);
            true
        };

        match result {
            Ok(resp) => {
                if resp.status().is_server_error() {
                    let status = resp.status();
                    if should_retry(&status, attempt) {
                        sleep(Duration::from_secs(2u64.pow(attempt))).await;
                        continue;
                    }
                    anyhow::bail!("Ollama API returned {status} after {MAX_RETRIES} retries");
                }
                let resp = resp
                    .error_for_status()
                    .context("Ollama API returned error status")?;
                let data: OllamaResponse = resp.json().await?;
                let cleaned = clean_response(&data.response);
                if cleaned.trim().is_empty() {
                    eprintln!("  Warning: empty VLM response for {}", frame_path.display());
                }
                return Ok(cleaned);
            }
            Err(e) => {
                if !should_retry(&e, attempt) {
                    return Err(e).context("Ollama request failed after all retries");
                }
                sleep(Duration::from_secs(2u64.pow(attempt))).await;
            }
        }
    }

    unreachable!("retry loop exhausted without returning")
}

pub async fn extract_batch(
    frame_paths: &[&Path],
    ollama_host: &str,
    model: &str,
    max_concurrent: usize,
) -> Result<Vec<String>> {
    let sem = Arc::new(Semaphore::new(max_concurrent));
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    let mut handles = Vec::with_capacity(frame_paths.len());

    for path in frame_paths {
        let permit = sem.clone();
        let client = client.clone();
        let host = ollama_host.to_string();
        let model = model.to_string();
        let path = path.to_path_buf();

        handles.push(tokio::spawn(async move {
            let _permit = permit
                .acquire()
                .await
                .context("semaphore closed unexpectedly")?;
            extract_code_from_frame(&path, &host, &model, Some(&client), None, None, 60).await
        }));
    }

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        results.push(handle.await??);
    }

    Ok(results)
}

fn encode_image(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)
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
