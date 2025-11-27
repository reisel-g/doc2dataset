use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};

use crate::model::{FileFormat, RawDocument};

mod bib;
mod html;
mod log;
mod rtf;
mod structured;
mod tabular;
mod tex;

static TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
pub enum PreparedDocument {
    Original(PathBuf),
    Converted(TempConverted),
}

impl PreparedDocument {
    pub fn path(&self) -> &Path {
        match self {
            PreparedDocument::Original(path) => path.as_path(),
            PreparedDocument::Converted(temp) => temp.path(),
        }
    }
}

pub fn prepare_document(doc: &RawDocument) -> Result<PreparedDocument> {
    match doc.format {
        FileFormat::Html => convert_and_write(doc, html::convert(&doc.path)?),
        FileFormat::Xml => convert_and_write(doc, html::convert(&doc.path)?),
        FileFormat::Json => convert_and_write(doc, structured::convert_json(&doc.path)?),
        FileFormat::Yaml => convert_and_write(doc, structured::convert_yaml(&doc.path)?),
        FileFormat::Csv => convert_and_write(doc, tabular::convert_csv(&doc.path)?),
        FileFormat::Tsv => convert_and_write(doc, tabular::convert_tsv(&doc.path)?),
        FileFormat::CsvGz => convert_and_write(doc, tabular::convert_csv_gz(&doc.path)?),
        FileFormat::TsvGz => convert_and_write(doc, tabular::convert_tsv_gz(&doc.path)?),
        FileFormat::Tex => convert_and_write(doc, tex::convert_tex(&doc.path)?),
        FileFormat::Bib => convert_and_write(doc, bib::convert_bib(&doc.path)?),
        FileFormat::Bbl => convert_and_write(doc, bib::convert_bbl(&doc.path)?),
        FileFormat::Ini => convert_and_write(doc, structured::convert_ini(&doc.path)?),
        FileFormat::Toml => convert_and_write(doc, structured::convert_toml(&doc.path)?),
        FileFormat::Log => convert_and_write(doc, log::convert_log(&doc.path)?),
        FileFormat::Rtf => convert_and_write(doc, rtf::convert_rtf(&doc.path)?),
        _ => Ok(PreparedDocument::Original(doc.path.clone())),
    }
}

fn convert_and_write(doc: &RawDocument, markdown: String) -> Result<PreparedDocument> {
    let temp = TempConverted::write(doc, markdown.as_bytes())?;
    Ok(PreparedDocument::Converted(temp))
}

#[derive(Debug)]
pub(super) struct TempConverted {
    path: PathBuf,
}

impl TempConverted {
    fn write(doc: &RawDocument, contents: &[u8]) -> Result<Self> {
        let dir = tmp_root();
        fs::create_dir_all(&dir).context("failed to create tmp-dataset directory")?;
        let filename = format!(
            "{}_{}.md",
            doc.id,
            TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = dir.join(filename);
        fs::write(&path, contents)
            .with_context(|| format!("failed to write converted file {}", path.display()))?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        self.path.as_path()
    }
}

impl Drop for TempConverted {
    fn drop(&mut self) {
        if let Err(err) = fs::remove_file(&self.path) {
            eprintln!(
                "[doc2dataset::convert] failed to remove temp file {}: {}",
                self.path.display(),
                err
            );
        }
    }
}

fn tmp_root() -> PathBuf {
    std::env::temp_dir().join("doc2dataset-convert")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FileFormat, SourceType};

    #[test]
    fn prepared_document_path_resolves() {
        let doc = RawDocument {
            id: "raw_0001".into(),
            source_type: SourceType::Files,
            format: FileFormat::Markdown,
            path: PathBuf::from("docs/sample.md"),
        };
        let prepared = PreparedDocument::Original(doc.path.clone());
        assert_eq!(prepared.path(), doc.path);
    }
}
