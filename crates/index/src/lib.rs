use std::io::Write;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentRecord {
    pub doc_id: String,
    pub title: Option<String>,
    pub source_type: String,
    pub source_format: String,
    pub source_ref: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PageRecord {
    pub page_id: String,
    pub doc_id: String,
    pub page_number: u32,
    pub approx_tokens: Option<u32>,
    #[serde(default)]
    pub meta: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CellRecord {
    pub cell_id: String,
    pub doc_id: String,
    pub page_id: String,
    pub kind: String,
    pub text: String,
    pub importance: f32,
    pub bbox: Option<[f32; 4]>,
    pub numguard: Option<Value>,
    #[serde(default)]
    pub meta: Value,
}

pub struct JsonlWriter<W> {
    writer: W,
}

impl<W: Write> JsonlWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub fn write_record<T: Serialize>(&mut self, record: &T) -> Result<()> {
        let mut buf = serde_json::to_vec(record)?;
        buf.push(b'\n');
        self.writer.write_all(&buf)?;
        Ok(())
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_writer_roundtrips_records() {
        let record = DocumentRecord {
            doc_id: "doc_1".to_string(),
            title: Some("Test".to_string()),
            source_type: "files".to_string(),
            source_format: "pdf".to_string(),
            source_ref: "/tmp/input.pdf".to_string(),
            tags: vec!["tag1".to_string()],
        };
        let writer = Vec::new();
        let mut writer = JsonlWriter::new(writer);
        writer.write_record(&record).unwrap();
        let buf = writer.into_inner();
        assert!(buf.ends_with(b"\n"));
        let parsed: DocumentRecord = serde_json::from_slice(&buf).unwrap();
        assert_eq!(parsed.doc_id, "doc_1");
        assert_eq!(parsed.title.unwrap(), "Test");
    }
}
