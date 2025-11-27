use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

pub fn convert(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read HTML file {}", path.display()))?;
    let mut markdown = String::new();
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        markdown.push_str(&format!("# {}\n\n", stem));
    }
    let rendered = html2md::parse_html(&raw);
    markdown.push_str(rendered.trim());
    markdown.push('\n');
    Ok(markdown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn html_is_rendered_to_markdown() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "<html><body><h1>Heading</h1><p>Body</p></body></html>"
        )
        .unwrap();
        let markdown = convert(file.path()).unwrap();
        assert!(markdown.contains("# "));
        assert!(markdown.to_lowercase().contains("heading"));
        assert!(markdown.to_lowercase().contains("body"));
    }
}
