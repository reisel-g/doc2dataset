use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use glob::Pattern;
use three_dcf_core::{ingest_to_index_with_opts, IngestOptions};
use walkdir::WalkDir;

use crate::convert;
use crate::logging;
use crate::model::{FileFormat, RawDocument, SourceType};

pub fn run(
    input: String,
    output: String,
    pattern: String,
    preset: String,
    enable_ocr: bool,
    force_ocr: bool,
    ocr_langs: String,
) -> Result<()> {
    run_with(
        input,
        output,
        pattern,
        preset,
        enable_ocr,
        force_ocr,
        ocr_langs,
        |doc, out, opts| ingest_to_index_with_opts(doc, out, opts),
    )
}

fn run_with<F>(
    input: String,
    output: String,
    pattern: String,
    preset: String,
    enable_ocr: bool,
    force_ocr: bool,
    ocr_langs: String,
    ingest_fn: F,
) -> Result<()>
where
    F: Fn(&Path, &Path, &IngestOptions) -> Result<()>,
{
    let input_path = PathBuf::from(&input);
    if !input_path.exists() {
        return Err(anyhow!(format!(
            "input path {} does not exist",
            input_path.display()
        )));
    }
    let output_dir = PathBuf::from(output);
    let docs = discover_files(&input_path, &pattern)?;
    if docs.is_empty() {
        println!(
            "[doc2dataset] no documents matched pattern '{}' under {}",
            pattern,
            input_path.display()
        );
        return Ok(());
    }
    println!(
        "[doc2dataset] discovered {} files under {}",
        docs.len(),
        input_path.display()
    );
    let lang_list: Vec<String> = ocr_langs
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let base_opts = IngestOptions {
        preset,
        enable_ocr,
        force_ocr,
        ocr_languages: if lang_list.is_empty() {
            vec!["eng".to_string()]
        } else {
            lang_list
        },
        source_override: None,
    };
    for doc in docs {
        println!(
            "[doc2dataset] ingesting {} ({}) source={} format={}",
            doc.path.display(),
            doc.id,
            doc.source_type.as_str(),
            doc.format.as_str()
        );
        let prepared = convert::prepare_document(&doc)?;
        let mut doc_opts = base_opts.clone();
        doc_opts.source_override = Some(doc.path.clone());
        if let Err(err) = ingest_fn(prepared.path(), &output_dir, &doc_opts)
            .with_context(|| format!("failed to ingest {}", doc.path.display()))
        {
            let mut reasons = Vec::new();
            let mut unsupported = false;
            for source in err.chain() {
                let text = source.to_string();
                let lower = text.to_lowercase();
                if lower.contains("unsupported input format")
                    || lower.contains("unsupported format")
                    || lower.contains("ocr support not enabled")
                {
                    unsupported = true;
                }
                reasons.push(text);
            }
            logging::stage(
                "ingest",
                format!(
                    "ingest error for {}: {}",
                    doc.path.display(),
                    reasons.join(" | ")
                ),
            );
            if unsupported {
                logging::stage(
                    "ingest",
                    format!("skipping {} due to unsupported format", doc.path.display()),
                );
                continue;
            }
            return Err(err);
        }
    }
    println!(
        "[doc2dataset] index ready under {}",
        output_dir.join("index").display()
    );
    Ok(())
}

fn discover_files(root: &Path, pattern: &str) -> Result<Vec<RawDocument>> {
    let patterns = build_patterns(pattern)?;
    let mut docs = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(root)
            .unwrap_or_else(|_| entry.path());
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let rel_norm = rel_str.to_lowercase();
        if !patterns.is_empty() && !patterns.iter().any(|pat| pat.matches(&rel_norm)) {
            continue;
        }
        let id = format!("raw_{:04}", docs.len() + 1);
        docs.push(RawDocument {
            id,
            source_type: SourceType::Files,
            format: FileFormat::from_path(entry.path()),
            path: entry.path().to_path_buf(),
        });
    }
    docs.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(docs)
}

fn build_patterns(pattern: &str) -> Result<Vec<Pattern>> {
    let mut patterns = Vec::new();
    for raw in pattern.split(',') {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let normalized = if trimmed.contains('/') {
            trimmed.to_lowercase()
        } else {
            format!("**/{}", trimmed.to_lowercase())
        };
        patterns.push(Pattern::new(&normalized).map_err(|e| anyhow!(e.msg))?);
    }
    Ok(patterns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn build_patterns_expands_simple_globs() {
        let patterns = build_patterns("*.pdf,sub/*.md").unwrap();
        let rendered: Vec<String> = patterns.iter().map(|p| p.as_str().to_string()).collect();
        assert!(rendered.contains(&"**/*.pdf".to_string()));
        assert!(rendered.contains(&"sub/*.md".to_string()));
    }

    #[test]
    fn discover_files_filters_by_patterns_and_formats() {
        let dir = tempdir().unwrap();
        let pdf = dir.path().join("paper.pdf");
        let md = dir.path().join("notes.md");
        let txt = dir.path().join("readme.txt");
        std::fs::write(&pdf, "fake").unwrap();
        std::fs::write(&md, "fake").unwrap();
        std::fs::write(&txt, "fake").unwrap();

        let docs = discover_files(dir.path(), "*.pdf,*.md").unwrap();
        assert_eq!(docs.len(), 2);
        let formats: Vec<_> = docs.iter().map(|d| d.format.as_str()).collect();
        assert!(formats.contains(&"pdf"));
        assert!(formats.contains(&"md"));
        // ensure deterministic ordering
        assert!(docs[0].path.ends_with("notes.md") || docs[0].path.ends_with("paper.pdf"));
    }

    #[test]
    fn run_passes_ingest_options() {
        let dir = tempdir().unwrap();
        let input_dir = dir.path().join("docs");
        std::fs::create_dir_all(&input_dir).unwrap();
        let file = input_dir.join("sample.md");
        std::fs::write(&file, "hello").unwrap();
        let output = dir.path().join("out");
        let recorded: RefCell<Vec<IngestOptions>> = RefCell::new(Vec::new());
        run_with(
            input_dir.to_string_lossy().into_owned(),
            output.to_string_lossy().into_owned(),
            "*.md".to_string(),
            "slides".to_string(),
            true,
            true,
            "eng,pol".to_string(),
            |_, _, opts| {
                recorded.borrow_mut().push(opts.clone());
                Ok(())
            },
        )
        .unwrap();
        let opts = recorded.borrow();
        assert_eq!(opts.len(), 1);
        assert_eq!(opts[0].preset, "slides");
        assert!(opts[0].enable_ocr);
        assert!(opts[0].force_ocr);
        assert_eq!(
            opts[0].ocr_languages,
            vec!["eng".to_string(), "pol".to_string()]
        );
    }

    #[test]
    fn run_converts_html_before_ingest() {
        let dir = tempdir().unwrap();
        let input_dir = dir.path().join("docs");
        fs::create_dir_all(&input_dir).unwrap();
        let html = input_dir.join("sample.html");
        std::fs::write(&html, "<html><body><h1>Title</h1></body></html>").unwrap();
        let output = dir.path().join("dataset");
        let recorded: RefCell<Vec<PathBuf>> = RefCell::new(Vec::new());
        run_with(
            input_dir.to_string_lossy().into_owned(),
            output.to_string_lossy().into_owned(),
            "*.html".to_string(),
            "reports".to_string(),
            false,
            false,
            "eng".to_string(),
            |path, _, _| {
                recorded.borrow_mut().push(path.to_path_buf());
                Ok(())
            },
        )
        .unwrap();
        let paths = recorded.borrow();
        assert_eq!(paths.len(), 1);
        assert_eq!(
            paths[0].extension().and_then(|ext| ext.to_str()),
            Some("md")
        );
    }
}
