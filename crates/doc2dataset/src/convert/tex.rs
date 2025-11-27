use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

pub fn convert_tex(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read TeX file {}", path.display()))?;
    let mut out = String::new();
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        out.push_str(&format!("# {} (TeX)\n\n", stem));
    }
    let mut paragraph = String::new();
    let mut list_depth = 0usize;
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut in_table = false;
    for raw_line in raw.lines() {
        let line = strip_comments(raw_line);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(title) = extract_command(trimmed, "\\section") {
            flush_paragraph(&mut paragraph, &mut out);
            flush_table(&mut table_rows, &mut in_table, &mut out);
            out.push_str(&format!("## {}\n\n", title));
            continue;
        }
        if let Some(title) = extract_command(trimmed, "\\subsection") {
            flush_paragraph(&mut paragraph, &mut out);
            flush_table(&mut table_rows, &mut in_table, &mut out);
            out.push_str(&format!("### {}\n\n", title));
            continue;
        }
        if let Some(title) = extract_command(trimmed, "\\subsubsection") {
            flush_paragraph(&mut paragraph, &mut out);
            flush_table(&mut table_rows, &mut in_table, &mut out);
            out.push_str(&format!("#### {}\n\n", title));
            continue;
        }
        if trimmed.starts_with("\\begin{itemize}") || trimmed.starts_with("\\begin{enumerate}") {
            list_depth += 1;
            flush_paragraph(&mut paragraph, &mut out);
            continue;
        }
        if trimmed.starts_with("\\end{itemize}") || trimmed.starts_with("\\end{enumerate}") {
            if list_depth > 0 {
                list_depth -= 1;
            }
            continue;
        }
        if trimmed.starts_with("\\item") {
            flush_table(&mut table_rows, &mut in_table, &mut out);
            let bullet = trimmed.trim_start_matches("\\item").trim();
            let indent = "  ".repeat(list_depth.saturating_sub(1));
            out.push_str(&format!("{}- {}\n", indent, clean_tex_fragment(bullet)));
            continue;
        }
        if trimmed.starts_with("\\begin{tabular}") {
            flush_paragraph(&mut paragraph, &mut out);
            table_rows.clear();
            in_table = true;
            continue;
        }
        if trimmed.starts_with("\\end{tabular}") {
            flush_table(&mut table_rows, &mut in_table, &mut out);
            continue;
        }
        if in_table {
            for row in trimmed.split("\\\\") {
                let cells: Vec<String> = row
                    .split('&')
                    .map(|cell| clean_tex_fragment(cell))
                    .filter(|cell| !cell.is_empty())
                    .collect();
                if !cells.is_empty() {
                    table_rows.push(cells);
                }
            }
            continue;
        }
        paragraph.push_str(trimmed);
        paragraph.push(' ');
    }
    flush_paragraph(&mut paragraph, &mut out);
    flush_table(&mut table_rows, &mut in_table, &mut out);
    Ok(out)
}

fn flush_paragraph(buffer: &mut String, out: &mut String) {
    let text = buffer.trim();
    if !text.is_empty() {
        out.push_str(text);
        out.push_str("\n\n");
    }
    buffer.clear();
}

fn flush_table(rows: &mut Vec<Vec<String>>, in_table: &mut bool, out: &mut String) {
    if !*in_table {
        return;
    }
    if rows.is_empty() {
        *in_table = false;
        return;
    }
    let column_count = rows.iter().map(|row| row.len()).max().unwrap_or(0);
    if column_count == 0 {
        rows.clear();
        *in_table = false;
        return;
    }
    let header = rows.first().cloned().unwrap_or_else(|| {
        (0..column_count)
            .map(|idx| format!("Col {}", idx + 1))
            .collect()
    });
    out.push('|');
    for (idx, cell) in header.iter().enumerate() {
        out.push(' ');
        let title = if cell.is_empty() {
            format!("Col {}", idx + 1)
        } else {
            cell.clone()
        };
        out.push_str(&title);
        out.push(' ');
        out.push('|');
    }
    out.push('\n');
    out.push('|');
    for _ in 0..column_count {
        out.push_str(" --- |");
    }
    out.push('\n');
    for row in rows.iter().skip(1) {
        out.push('|');
        for col in 0..column_count {
            let cell = row.get(col).map(|s| s.as_str()).unwrap_or("");
            out.push(' ');
            out.push_str(cell);
            out.push(' ');
            out.push('|');
        }
        out.push('\n');
    }
    out.push('\n');
    rows.clear();
    *in_table = false;
}

fn extract_command(line: &str, command: &str) -> Option<String> {
    if line.starts_with(command) {
        let rest = line[command.len()..].trim();
        if let Some(stripped) = rest.strip_prefix('{') {
            if let Some(end) = stripped.find('}') {
                return Some(clean_tex_fragment(&stripped[..end]));
            }
        }
    }
    None
}

fn clean_tex_fragment(value: &str) -> String {
    value
        .replace('{', "")
        .replace('}', "")
        .replace("\\", "")
        .trim()
        .to_string()
}

fn strip_comments(line: &str) -> &str {
    if let Some(idx) = line.find('%') {
        &line[..idx]
    } else {
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn tex_conversion_adds_headings_and_lists() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(
            b"\\section{Intro}\nThis is text.\\begin{itemize}\\item Bullet\\end{itemize}",
        )
        .unwrap();
        let markdown = convert_tex(file.path()).unwrap();
        assert!(markdown.contains("## Intro"));
        assert!(markdown.contains("Bullet"));
    }
}
