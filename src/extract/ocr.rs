use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};

pub fn is_available() -> bool {
    Command::new("tesseract")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub fn extract_code_from_frame(frame_path: &Path) -> Result<String> {
    let output = Command::new("tesseract")
        .arg(frame_path)
        .arg("stdout")
        .args(["--psm", "6"])
        .args(["-c", "preserve_interword_spaces=1"])
        .output()
        .context("tesseract not found")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tesseract failed: {stderr}");
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}
