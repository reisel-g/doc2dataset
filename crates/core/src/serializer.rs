use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::document::{CellType, Document};
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableMode {
    Auto,
    Csv,
    Dims,
}

impl Default for TableMode {
    fn default() -> Self {
        TableMode::Auto
    }
}

#[derive(Debug, Clone)]
pub struct TextSerializerConfig {
    pub include_header: bool,
    pub include_grammar: bool,
    pub max_preview_chars: usize,
    pub table_mode: TableMode,
    pub preset_label: Option<String>,
    pub budget_label: Option<String>,
}

impl Default for TextSerializerConfig {
    fn default() -> Self {
        Self {
            include_header: true,
            include_grammar: true,
            max_preview_chars: 64,
            table_mode: TableMode::Auto,
            preset_label: None,
            budget_label: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextSerializer {
    config: TextSerializerConfig,
}

impl TextSerializer {
    pub fn new() -> Self {
        Self {
            config: TextSerializerConfig::default(),
        }
    }

    pub fn with_config(config: TextSerializerConfig) -> Self {
        Self { config }
    }

    pub fn to_string(&self, document: &Document) -> Result<String> {
        let mut out = String::new();
        if self.config.include_header {
            let preset = self
                .config
                .preset_label
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            let budget = self
                .config
                .budget_label
                .clone()
                .unwrap_or_else(|| "auto".to_string());
            out.push_str(&format!(
                "<ctx3d grid={} codeset={} preset={} budget={}>\n",
                document.header.grid, document.header.codeset, preset, budget
            ));
        }
        for cell in document.ordered_cells() {
            let code_hex = hex::encode(cell.code_id);
            let preview = document
                .payload_for(&cell.code_id)
                .map(|payload| match cell.cell_type {
                    CellType::Table => render_table_preview(payload, &self.config),
                    _ => preview(payload, self.config.max_preview_chars),
                })
                .unwrap_or_else(|| "<missing>".to_string());
            let code_short = &code_hex[..16];
            let preview_escaped = escape_preview(&preview);
            out.push_str(&format!(
                "(z={z},x={x},y={y},w={w},h={h},code={code},rle={rle},imp={imp},type={typ}) \"{preview}\"\n",
                z = cell.z,
                x = cell.x,
                y = cell.y,
                w = cell.w,
                h = cell.h,
                code = code_short,
                rle = cell.rle,
                imp = cell.importance,
                typ = format!("{:?}", cell.cell_type).to_uppercase(),
                preview = preview_escaped
            ));
        }
        if self.config.include_grammar {
            out.push_str("\ngrammar: --select \"z=0,x=0..1024,y=0..4096\"\n");
        }
        if self.config.include_header {
            out.push_str("</ctx3d>\n");
        }
        Ok(out)
    }

    pub fn write_textual<P: AsRef<Path>>(&self, document: &Document, path: P) -> Result<()> {
        let txt = self.to_string(document)?;
        let mut file = File::create(path)?;
        file.write_all(txt.as_bytes())?;
        Ok(())
    }
}

fn preview(payload: &str, limit: usize) -> String {
    if payload.len() <= limit {
        return payload.to_string();
    }
    let mut truncated = payload.chars().take(limit).collect::<String>();
    truncated.push_str("...");
    truncated
}

fn estimate_table_columns(payload: &str) -> usize {
    if payload.contains('|') {
        payload
            .split('|')
            .filter(|c| !c.trim().is_empty())
            .count()
            .max(1)
    } else {
        payload.split_whitespace().count().max(1)
    }
}

fn render_table_preview(payload: &str, config: &TextSerializerConfig) -> String {
    match config.table_mode {
        TableMode::Csv => csv_preview(payload, config.max_preview_chars),
        TableMode::Dims => dims_preview(payload),
        TableMode::Auto => {
            if payload.len() <= config.max_preview_chars * 2 {
                csv_preview(payload, config.max_preview_chars)
            } else {
                dims_preview(payload)
            }
        }
    }
}

fn dims_preview(payload: &str) -> String {
    let rows = payload
        .lines()
        .filter(|l| !l.trim().is_empty())
        .count()
        .max(1);
    let cols = estimate_table_columns(payload);
    format!("[table rows={rows} cols={cols}]")
}

fn csv_preview(payload: &str, limit: usize) -> String {
    let mut rows = Vec::new();
    for line in payload.lines().filter(|l| !l.trim().is_empty()) {
        let normalized = line
            .replace('|', ",")
            .replace('\t', ",")
            .split(',')
            .map(|cell| cell.trim())
            .filter(|cell| !cell.is_empty())
            .collect::<Vec<_>>()
            .join(", ");
        if normalized.is_empty() {
            continue;
        }
        rows.push(normalized);
        if rows.len() >= 4 {
            break;
        }
    }
    let mut combined = rows.join(" | ");
    if combined.len() > limit {
        combined.truncate(limit);
        combined.push_str("...");
    }
    if combined.is_empty() {
        dims_preview(payload)
    } else {
        format!("[csv {combined}]")
    }
}

fn escape_preview(payload: &str) -> String {
    payload.replace('\"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{CellRecord, CellType, Document, Header, PageInfo};

    fn sample_document() -> Document {
        let mut doc = Document::new(Header::default());
        doc.add_page(PageInfo {
            z: 0,
            width_px: 800,
            height_px: 1000,
        });
        let text_code = [1u8; 32];
        doc.dict.insert(text_code, "Hello world".to_string());
        doc.cells.push(CellRecord {
            z: 0,
            x: 10,
            y: 20,
            w: 700,
            h: 20,
            code_id: text_code,
            rle: 0,
            cell_type: CellType::Text,
            importance: 100,
        });

        let table_code = [2u8; 32];
        doc.dict.insert(
            table_code,
            "Quarter | Revenue | Cost\nQ1 | 10 | 5\nQ2 | 12 | 6".to_string(),
        );
        doc.cells.push(CellRecord {
            z: 0,
            x: 10,
            y: 60,
            w: 700,
            h: 40,
            code_id: table_code,
            rle: 0,
            cell_type: CellType::Table,
            importance: 120,
        });

        doc
    }

    #[test]
    fn snapshot_serialization() {
        let doc = sample_document();
        let serializer = TextSerializer::new();
        let rendered = serializer.to_string(&doc).unwrap();
        insta::assert_snapshot!(rendered);
    }

    #[test]
    fn snapshot_serialization_csv_mode() {
        let doc = sample_document();
        let serializer = TextSerializer::with_config(TextSerializerConfig {
            table_mode: TableMode::Csv,
            ..Default::default()
        });
        let rendered = serializer.to_string(&doc).unwrap();
        insta::assert_snapshot!(rendered);
    }

    #[test]
    fn snapshot_serialization_dims_mode() {
        let doc = sample_document();
        let serializer = TextSerializer::with_config(TextSerializerConfig {
            table_mode: TableMode::Dims,
            ..Default::default()
        });
        let rendered = serializer.to_string(&doc).unwrap();
        insta::assert_snapshot!(rendered);
    }
}
