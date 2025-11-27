use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use configparser::ini::Ini;
use serde_json::{Map, Value};

const TABLE_CHUNK: usize = 50;

pub fn convert_json(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read JSON file {}", path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("invalid JSON payload in {}", path.display()))?;
    Ok(render_root(path, value))
}

pub fn convert_yaml(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read YAML file {}", path.display()))?;
    let yaml_value: serde_yaml::Value = serde_yaml::from_str(&raw)
        .with_context(|| format!("invalid YAML payload in {}", path.display()))?;
    let value = serde_json::to_value(yaml_value).context("failed to normalize YAML")?;
    Ok(render_root(path, value))
}

pub fn convert_toml(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read TOML file {}", path.display()))?;
    let toml_value: toml::Value = toml::from_str(&raw)
        .with_context(|| format!("invalid TOML payload in {}", path.display()))?;
    let value = serde_json::to_value(toml_value).context("failed to normalize TOML")?;
    Ok(render_root(path, value))
}

pub fn convert_ini(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read INI file {}", path.display()))?;
    let mut parser = Ini::new();
    let sections = parser
        .read(raw)
        .map_err(|e| anyhow!("invalid INI payload in {}: {}", path.display(), e))?;
    let mut root = Map::new();
    for (section, props) in sections {
        let mut section_map = Map::new();
        let section_name = if section.is_empty() {
            "default".to_string()
        } else {
            section
        };
        for (key, value) in props {
            let val = value.unwrap_or_default();
            section_map.insert(key, Value::String(val));
        }
        root.insert(section_name, Value::Object(section_map));
    }
    Ok(render_root(path, Value::Object(root)))
}

fn render_root(path: &Path, value: Value) -> String {
    let mut out = String::new();
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        out.push_str(&format!("# {}\n\n", stem));
    }
    render_value(None, &value, 2, &mut out);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn render_value(name: Option<&str>, value: &Value, depth: usize, out: &mut String) {
    match value {
        Value::Object(map) => render_object(name, map, depth, out),
        Value::Array(items) => render_array(name, items, depth, out),
        other => {
            if let Some(title) = name {
                write_heading(depth, title, out);
            }
            out.push_str(&format!("{}\n\n", format_scalar(other)));
        }
    }
}

fn render_object(name: Option<&str>, map: &Map<String, Value>, depth: usize, out: &mut String) {
    if let Some(title) = name {
        write_heading(depth, title, out);
    }
    for key in sorted_keys(map) {
        if let Some(value) = map.get(&key) {
            render_value(Some(&key), value, depth + 1, out);
        }
    }
}

fn render_array(name: Option<&str>, items: &[Value], depth: usize, out: &mut String) {
    if let Some(title) = name {
        write_heading(depth, title, out);
    }
    if items.is_empty() {
        out.push_str("(empty array)\n\n");
        return;
    }
    if try_render_table(items, depth + 1, out) {
        return;
    }
    for (idx, item) in items.iter().enumerate() {
        let indent = "  ".repeat(depth.saturating_sub(1));
        out.push_str(&format!("{}- {}\n", indent, format_scalar_label(idx, item)));
        if item.is_object() {
            render_value(None, item, depth + 1, out);
        }
    }
    out.push('\n');
}

fn try_render_table(items: &[Value], depth: usize, out: &mut String) -> bool {
    let mut headers = BTreeSet::new();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for value in items {
        match value {
            Value::Object(map) => {
                headers.extend(map.keys().cloned());
            }
            _ => return false,
        }
    }
    if headers.is_empty() {
        return false;
    }
    let headers: Vec<String> = headers.into_iter().collect();
    rows.clear();
    for value in items {
        if let Value::Object(map) = value {
            let mut row = Vec::new();
            for key in &headers {
                row.push(format_scalar(map.get(key).unwrap_or(&Value::Null)));
            }
            rows.push(row);
        }
    }
    for (chunk_idx, chunk) in rows.chunks(TABLE_CHUNK).enumerate() {
        let title = format!("Table chunk {}", chunk_idx + 1);
        write_heading(depth, &title, out);
        render_table(&headers, chunk, out);
    }
    true
}

fn render_table(headers: &[String], rows: &[Vec<String>], out: &mut String) {
    out.push('|');
    for header in headers {
        out.push(' ');
        out.push_str(&sanitize_cell(header));
        out.push(' ');
        out.push('|');
    }
    out.push('\n');
    out.push('|');
    for _ in headers {
        out.push_str(" --- |");
    }
    out.push('\n');
    for row in rows {
        out.push('|');
        for cell in row {
            out.push(' ');
            out.push_str(&sanitize_cell(cell));
            out.push(' ');
            out.push('|');
        }
        out.push('\n');
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn json_conversion_emits_headings_and_table() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "{{\"meta\": {{\"name\": \"demo\"}}, \"rows\": [{{\"a\": 1, \"b\": 2}}, {{\"a\": 3, \"b\": 4}}]}}"
        )
        .unwrap();
        let markdown = convert_json(file.path()).unwrap();
        assert!(markdown.contains("rows"));
        assert!(markdown.contains("| a"));
        assert!(markdown.contains("3"));
    }

    #[test]
    fn yaml_conversion_round_trips() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "service:\n  name: demo\n  enabled: true").unwrap();
        let markdown = convert_yaml(file.path()).unwrap();
        assert!(markdown.contains("service"));
        assert!(markdown.contains("demo"));
    }

    #[test]
    fn toml_conversion_supported() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "[app]\nname = 'demo'\nports = [80, 8080]").unwrap();
        let markdown = convert_toml(file.path()).unwrap();
        assert!(markdown.contains("app"));
        assert!(markdown.contains("ports"));
    }

    #[test]
    fn ini_conversion_supported() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "[section]\nkey=value").unwrap();
        let markdown = convert_ini(file.path()).unwrap();
        assert!(markdown.contains("section"));
        assert!(markdown.contains("key"));
    }
}

fn sorted_keys(map: &Map<String, Value>) -> Vec<String> {
    let mut keys: Vec<String> = map.keys().cloned().collect();
    keys.sort();
    keys
}

fn write_heading(level: usize, title: &str, out: &mut String) {
    let capped = level.min(6);
    let hashes = "#".repeat(capped.max(1));
    out.push_str(&format!("{} {}\n\n", hashes, title));
}

fn format_scalar(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(v) => v.to_string(),
        Value::Number(num) => num.to_string(),
        Value::String(s) => s.trim().to_string(),
        Value::Array(_) => "[...]".to_string(),
        Value::Object(_) => "{...}".to_string(),
    }
}

fn format_scalar_label(idx: usize, value: &Value) -> String {
    match value {
        Value::Object(_) | Value::Array(_) => format!("item {}", idx + 1),
        other => format_scalar(other),
    }
}

fn sanitize_cell(value: &str) -> String {
    value
        .replace('|', "\\|")
        .replace('\n', " ")
        .trim()
        .to_string()
}
