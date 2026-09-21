use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use tokio::time::Duration;

const REFINE_PROMPT: &str = r#"You are cleaning up VLM-extracted text from a video recording. The text was extracted frame-by-frame and mechanically deduplicated, but quality issues remain.

Clean up the text by fixing ALL of these issues:

1. Remove diff artifacts: Strip lines starting with @@, ---, +++, diff --git. For -/+ paired lines with identical content, keep only one copy without the prefix. Remove # Changelog literal headings from diff views.

2. Remove duplicate lines: If consecutive lines are identical or near-identical (same content, different markdown like ### X followed by - **X**), keep only the better version. If the same content appears multiple times in the document, keep only the first or best occurrence.

3. Fix garbled text: Fix obvious OCR errors, remove gibberish sequences, fix broken words.

4. Remove within-line repetitions: Collapse any pattern repeated 3+ times within a single line.

5. Remove duplicate sections: If the same section appears multiple times with different formatting, keep only the most complete version. Same for diagrams.

6. Fix structural issues: Remove empty headers with no content. Fix broken markdown. Remove stray code fence markers that appear as content.

7. Preserve legitimate content: Don't remove genuinely different content. The document may contain multiple document types (changelogs, design docs, requirements). All are legitimate.

Output ONLY the cleaned text, no commentary or explanations."#;

#[derive(Deserialize)]
struct OllamaResponse {
    #[serde(default)]
    response: String,
}

pub async fn refine_text(text: &str, ollama_host: &str, model: &str) -> Result<String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;

    let line_count = text.lines().count();
    println!("Refining {line_count} lines via LLM ({model})...");

    let payload = serde_json::json!({
        "model": model,
        "prompt": format!("{REFINE_PROMPT}\n\n---\n\n{text}"),
        "stream": false,
        "options": {
            "temperature": 0.1,
            "num_predict": 32768,
            "num_ctx": 65536
        }
    });

    let url = format!("{ollama_host}/api/generate");
    let resp = client
        .post(&url)
        .json(&payload)
        .send()
        .await
        .context("Failed to send refine request to Ollama")?
        .error_for_status()
        .context("Ollama refine request returned error")?;

    let data: OllamaResponse = resp.json().await.context("Failed to parse Ollama response")?;
    let cleaned = data.response.trim().to_string();

    let cleaned = if cleaned.starts_with("```") {
        let lines: Vec<&str> = cleaned.lines().collect();
        let start = 1;
        let end = if lines.last().map_or(false, |l| l.trim() == "```") {
            lines.len() - 1
        } else {
            lines.len()
        };
        lines[start..end].join("\n")
    } else {
        cleaned
    };

    let refined_count = cleaned.lines().count();
    println!("  Refined to {refined_count} lines (removed {})", line_count - refined_count);

    Ok(cleaned)
}
