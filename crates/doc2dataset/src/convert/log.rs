use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

pub fn convert_log(path: &Path) -> Result<String> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read log file {}", path.display()))?;
    let mut out = String::new();
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        out.push_str(&format!("# {} (log)\n\n", stem));
    }
    out.push_str("```text\n");
    out.push_str(&content);
    if !content.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("```\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn log_is_wrapped_in_code_block() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "WARN first line").unwrap();
        writeln!(file, "INFO second line").unwrap();
        let markdown = convert_log(file.path()).unwrap();
        assert!(markdown.contains("```text"));
        assert!(markdown.contains("WARN first line"));
    }
}
