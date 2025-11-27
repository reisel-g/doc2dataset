use std::fs;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use csv::ReaderBuilder;
use flate2::read::MultiGzDecoder;

const ROW_CHUNK: usize = 50;

pub fn convert_csv(path: &Path) -> Result<String> {
    fs::metadata(path).with_context(|| format!("failed to read CSV file {}", path.display()))?;
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open CSV file {}", path.display()))?;
    convert_tabular_from_reader(path, file, b',', "CSV")
}

pub fn convert_tsv(path: &Path) -> Result<String> {
    fs::metadata(path).with_context(|| format!("failed to read TSV file {}", path.display()))?;
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open TSV file {}", path.display()))?;
    convert_tabular_from_reader(path, file, b'\t', "TSV")
}

pub fn convert_csv_gz(path: &Path) -> Result<String> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open CSV file {}", path.display()))?;
    let decoder = MultiGzDecoder::new(file);
    convert_tabular_from_reader(path, decoder, b',', "CSV (gzip)")
}

pub fn convert_tsv_gz(path: &Path) -> Result<String> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open TSV file {}", path.display()))?;
    let decoder = MultiGzDecoder::new(file);
    convert_tabular_from_reader(path, decoder, b'\t', "TSV (gzip)")
}

fn convert_tabular_from_reader<R: Read>(
    path: &Path,
    reader: R,
    delimiter: u8,
    label: &str,
) -> Result<String> {
    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter)
        .from_reader(reader);
    let headers = reader
        .headers()
        .map(|h| h.iter().map(|cell| cell.to_string()).collect::<Vec<_>>())
        .with_context(|| format!("missing headers in {}", path.display()))?;
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.with_context(|| format!("invalid row in {}", path.display()))?;
        rows.push(
            record
                .iter()
                .map(|cell| cell.to_string())
                .collect::<Vec<_>>(),
        );
    }
    let mut out = String::new();
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        out.push_str(&format!("# {} ({})\n\n", stem, label));
    }
    if rows.is_empty() {
        out.push_str("(no rows)\n\n");
        return Ok(out);
    }
    for (chunk_idx, chunk) in rows.chunks(ROW_CHUNK).enumerate() {
        let start = chunk_idx * ROW_CHUNK + 1;
        let end = start + chunk.len() - 1;
        let title = format!("Rows {}-{}", start, end);
        out.push_str(&format!("## {}\n\n", title));
        render_table(&headers, chunk, &mut out);
    }
    Ok(out)
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

fn sanitize_cell(value: &str) -> String {
    value
        .replace('|', "\\|")
        .replace('\n', " ")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn csv_conversion_outputs_table() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "name,value").unwrap();
        writeln!(file, "foo,1").unwrap();
        writeln!(file, "bar,2").unwrap();
        let markdown = convert_csv(file.path()).unwrap();
        assert!(markdown.contains("| name"));
        assert!(markdown.contains("foo"));
    }

    #[test]
    fn tsv_conversion_outputs_table() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "name\tvalue").unwrap();
        writeln!(file, "foo\t1").unwrap();
        let markdown = convert_tsv(file.path()).unwrap();
        assert!(markdown.contains("| name"));
        assert!(markdown.contains("foo"));
    }

    #[test]
    fn csv_gz_conversion_outputs_table() {
        let file = NamedTempFile::new().unwrap();
        let mut encoder = GzEncoder::new(
            fs::File::create(file.path()).unwrap(),
            Compression::default(),
        );
        writeln!(encoder, "name,value").unwrap();
        writeln!(encoder, "alpha,10").unwrap();
        encoder.finish().unwrap();
        let markdown = convert_csv_gz(file.path()).unwrap();
        assert!(markdown.contains("alpha"));
    }

    #[test]
    fn tsv_gz_conversion_outputs_table() {
        let file = NamedTempFile::new().unwrap();
        let mut encoder = GzEncoder::new(
            fs::File::create(file.path()).unwrap(),
            Compression::default(),
        );
        writeln!(encoder, "name\tvalue").unwrap();
        writeln!(encoder, "beta\t20").unwrap();
        encoder.finish().unwrap();
        let markdown = convert_tsv_gz(file.path()).unwrap();
        assert!(markdown.contains("beta"));
    }
}
