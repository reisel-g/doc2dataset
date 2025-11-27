use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

pub fn convert_bib(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read bib file {}", path.display()))?;
    let mut out = String::new();
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        out.push_str(&format!("# {} (BibTeX)\n\n", stem));
    }
    let mut current_title: Option<String> = None;
    let mut body: Vec<String> = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('@') {
            finalize_entry(&mut out, current_title.take(), &mut body);
            current_title = Some(render_bib_heading(trimmed));
            body.clear();
        } else if trimmed == "}" {
            finalize_entry(&mut out, current_title.take(), &mut body);
            body.clear();
        } else {
            body.push(trimmed.to_string());
        }
    }
    finalize_entry(&mut out, current_title.take(), &mut body);
    Ok(out)
}

pub fn convert_bbl(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read bbl file {}", path.display()))?;
    let mut out = String::new();
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        out.push_str(&format!("# {} (BibLaTeX)\n\n", stem));
    }
    let mut current_title: Option<String> = None;
    let mut body: Vec<String> = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("\\bibitem") {
            finalize_entry(&mut out, current_title.take(), &mut body);
            body.clear();
            let (heading, remainder) = render_bibitem_heading(trimmed);
            current_title = Some(heading);
            if let Some(rest) = remainder {
                if !rest.is_empty() {
                    body.push(rest.to_string());
                }
            }
        } else {
            body.push(trimmed.to_string());
        }
    }
    finalize_entry(&mut out, current_title.take(), &mut body);
    Ok(out)
}

fn finalize_entry(out: &mut String, title: Option<String>, body: &mut Vec<String>) {
    if let Some(title) = title {
        out.push_str(&format!("### {}\n\n", title));
        for line in body.iter() {
            let clean = line.trim().trim_start_matches(',');
            if clean.is_empty() {
                continue;
            }
            if let Some((field, value)) = clean.split_once('=') {
                let field = field.trim().trim_matches('{').trim_matches('}');
                let value = value
                    .trim()
                    .trim_start_matches('{')
                    .trim_end_matches('}')
                    .trim_end_matches(',')
                    .trim();
                out.push_str(&format!("- **{}**: {}\n", field, value));
            } else {
                out.push_str(&format!("> {}\n", clean));
            }
        }
        out.push('\n');
    }
    body.clear();
}

fn render_bib_heading(line: &str) -> String {
    let content = line.trim_start_matches('@');
    if let Some((entry_type, tail)) = content.split_once('{') {
        if let Some((key, _)) = tail.split_once(',') {
            return format!("[{}] {}", key.trim(), entry_type.trim());
        }
    }
    content.trim().to_string()
}

fn render_bibitem_heading(line: &str) -> (String, Option<String>) {
    let rest = line.trim_start_matches("\\bibitem").trim();
    if let Some(stripped) = rest.strip_prefix('{') {
        if let Some(end) = stripped.find('}') {
            let key = stripped[..end].trim();
            let leftover = stripped[end + 1..].trim();
            let tail = if leftover.is_empty() {
                None
            } else {
                Some(leftover.to_string())
            };
            return (format!("[{}]", key), tail);
        }
    }
    (rest.to_string(), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn bib_entries_are_rendered() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"@article{key,\nauthor={Doe},\ntitle={Test}\n}")
            .unwrap();
        let markdown = convert_bib(file.path()).unwrap();
        assert!(markdown.contains("key"));
        assert!(markdown.contains("author"));
    }

    #[test]
    fn bbl_entries_are_rendered() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"\\bibitem{ref}Text line").unwrap();
        let markdown = convert_bbl(file.path()).unwrap();
        assert!(markdown.contains("ref"));
        assert!(markdown.contains("Text line"));
    }
}
