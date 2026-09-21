use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::types::PipelineOutput;

pub fn write_output(output: &PipelineOutput, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir)?;

    let mut used_names: HashSet<String> = HashSet::new();
    used_names.insert("extracted.json".to_string());

    for (i, ef) in output.files.iter().enumerate() {
        let mut safe_name = ef.filename.replace(['/', '\\'], "_");
        if safe_name.is_empty() || safe_name == "." || safe_name == ".." {
            safe_name = format!("file_{i}.txt");
        }

        while used_names.contains(&safe_name) {
            let (stem, ext) = match safe_name.rpartition('.') {
                (s, e) if !s.is_empty() => (s.to_string(), e.to_string()),
                _ => (safe_name.clone(), "txt".to_string()),
            };
            safe_name = format!("{stem}_{i}.{ext}");
        }

        used_names.insert(safe_name.clone());
        let file_path = output_dir.join(&safe_name);
        fs::write(&file_path, &ef.final_content)?;
    }

    let out_path = output_dir.join("extracted.json");
    let json = serde_json::to_string_pretty(output)?;
    fs::write(&out_path, json)?;

    Ok(out_path)
}

trait RPartitionExt {
    fn rpartition(&self, sep: char) -> (&str, &str);
}

impl RPartitionExt for str {
    fn rpartition(&self, sep: char) -> (&str, &str) {
        match self.rfind(sep) {
            Some(pos) => (&self[..pos], &self[pos + sep.len_utf8()..]),
            None => ("", self),
        }
    }
}
