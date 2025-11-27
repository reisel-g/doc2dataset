use std::path::{Path, PathBuf};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use thiserror::Error;
use three_dcf_core::{
    estimate_tokens, CellRecord, Decoder, Document, Encoder, Metrics, Stats as CoreStats,
    TextSerializer, TextSerializerConfig, TokenizerKind,
};

#[derive(Error, Debug)]
pub enum FfiError {
    #[error("{0}")]
    Core(#[from] three_dcf_core::DcfError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("unknown tokenizer '{0}'")]
    UnknownTokenizer(String),
}

impl From<FfiError> for Error {
    fn from(err: FfiError) -> Self {
        Error::new(Status::GenericFailure, err.to_string())
    }
}

fn default_preset(preset: Option<&str>) -> &str {
    preset.unwrap_or("reports")
}

fn encode_internal(
    input: &Path,
    output: &Path,
    preset: Option<&str>,
    budget: Option<u32>,
    json_out: Option<&Path>,
    text_out: Option<&Path>,
) -> Result<(), FfiError> {
    let mut builder = Encoder::builder(default_preset(preset))?;
    if let Some(b) = budget {
        builder = builder.budget(Some(b as usize));
    }
    let encoder = builder.build();
    let (document, _) = encoder.encode_path(input)?;
    document.save_bin(output)?;
    if let Some(json_path) = json_out {
        document.save_json(json_path)?;
    }
    if let Some(text_path) = text_out {
        TextSerializer::new().write_textual(&document, text_path)?;
    }
    Ok(())
}

fn tokenizer_from(spec: Option<&str>) -> Result<TokenizerKind, FfiError> {
    match spec.map(|s| s.to_lowercase()) {
        None => Ok(TokenizerKind::Cl100k),
        Some(ref name) if name == "cl100k_base" || name == "cl100k" => Ok(TokenizerKind::Cl100k),
        Some(ref name) if name == "gpt2" || name == "p50k" => Ok(TokenizerKind::Gpt2),
        Some(ref name) if name == "o200k" || name == "o200k_base" => Ok(TokenizerKind::O200k),
        Some(ref name) if name == "anthropic" => Ok(TokenizerKind::Anthropic),
        Some(name) => {
            let path = PathBuf::from(&name);
            if path.exists() {
                Ok(TokenizerKind::Custom(path))
            } else {
                Err(FfiError::UnknownTokenizer(name))
            }
        }
    }
}

fn decode_internal(document_path: &Path, page: Option<u32>) -> Result<String, FfiError> {
    let document = Document::load_bin(document_path)?;
    if let Some(z) = page {
        Ok(document.decode_page_to_text(z))
    } else {
        let decoder = Decoder::new();
        Ok(decoder.to_text(&document)?)
    }
}

fn stats_internal(document_path: &Path, tokenizer: Option<&str>) -> Result<CoreStats, FfiError> {
    let document = Document::load_bin(document_path)?;
    let kind = tokenizer_from(tokenizer)?;
    let stats = CoreStats::measure(&document, kind)?;
    Ok(stats)
}

fn context_metrics_from(metrics: &Metrics) -> ContextMetrics {
    ContextMetrics {
        pages: metrics.pages,
        lines_total: metrics.lines_total,
        cells_total: metrics.cells_total,
        cells_kept: metrics.cells_kept,
        dedup_ratio: metrics.dedup_ratio,
        numguard_count: metrics.numguard_count,
        raw_tokens_estimate: metrics.raw_tokens_estimate,
        compressed_tokens_estimate: metrics.compressed_tokens_estimate,
        compression_factor: metrics.compression_factor,
    }
}

#[napi(object)]
pub struct StatsSummary {
    pub tokens_raw: u32,
    pub tokens_3dcf: u32,
    pub cells: u32,
    pub unique_payloads: u32,
    pub savings_ratio: f32,
}

#[napi(object)]
pub struct ContextMetrics {
    pub pages: u32,
    pub lines_total: u32,
    pub cells_total: u32,
    pub cells_kept: u32,
    pub dedup_ratio: f32,
    pub numguard_count: u32,
    pub raw_tokens_estimate: Option<u32>,
    pub compressed_tokens_estimate: Option<u32>,
    pub compression_factor: Option<f32>,
}

#[napi(object)]
pub struct ContextResult {
    pub text: String,
    pub metrics: ContextMetrics,
}

#[napi(object)]
pub struct ContextOptions {
    pub preset: Option<String>,
    pub budget: Option<u32>,
    pub tokenizer: Option<String>,
}

#[napi(object)]
pub struct CellOptions {
    pub preset: Option<String>,
}

#[napi(object)]
pub struct CellBbox {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

#[napi(object)]
pub struct EncodedCell {
    pub text: String,
    pub importance: u8,
    pub page: u32,
    pub bbox: CellBbox,
}

impl From<CoreStats> for StatsSummary {
    fn from(value: CoreStats) -> Self {
        Self {
            tokens_raw: value.tokens_raw as u32,
            tokens_3dcf: value.tokens_3dcf as u32,
            cells: value.cells as u32,
            unique_payloads: value.unique_payloads as u32,
            savings_ratio: value.savings_ratio,
        }
    }
}

#[napi]
pub fn encode_file(
    input: String,
    output: String,
    preset: Option<String>,
    budget: Option<u32>,
    json_out: Option<String>,
    text_out: Option<String>,
) -> napi::Result<()> {
    encode_internal(
        Path::new(&input),
        Path::new(&output),
        preset.as_deref(),
        budget,
        json_out.as_deref().map(Path::new),
        text_out.as_deref().map(Path::new),
    )
    .map_err(Error::from)
}

#[napi]
pub fn decode_text(document: String, page: Option<u32>) -> napi::Result<String> {
    decode_internal(Path::new(&document), page).map_err(Error::from)
}

#[napi]
pub fn stats(document: String, tokenizer: Option<String>) -> napi::Result<StatsSummary> {
    stats_internal(Path::new(&document), tokenizer.as_deref())
        .map(StatsSummary::from)
        .map_err(Error::from)
}

#[napi]
pub fn encode_to_context(
    file_path: String,
    options: Option<ContextOptions>,
) -> napi::Result<ContextResult> {
    let opts = options.unwrap_or(ContextOptions {
        preset: None,
        budget: None,
        tokenizer: None,
    });
    let mut builder = Encoder::builder(default_preset(opts.preset.as_deref()))
        .map_err(FfiError::from)
        .map_err(Error::from)?;
    if let Some(b) = opts.budget {
        builder = builder.budget(Some(b as usize));
    }
    let encoder = builder.build();
    let (doc, mut metrics, raw_text) = encoder
        .encode_path_with_plaintext(Path::new(&file_path))
        .map_err(FfiError::from)
        .map_err(Error::from)?;
    let serializer = TextSerializer::with_config(TextSerializerConfig::default());
    let context_text = serializer
        .to_string(&doc)
        .map_err(FfiError::from)
        .map_err(Error::from)?;
    let tokenizer = tokenizer_from(opts.tokenizer.as_deref())?;
    let raw_tokens = estimate_tokens(&raw_text, &tokenizer)
        .map_err(FfiError::from)
        .map_err(Error::from)? as u32;
    let compressed_tokens = estimate_tokens(context_text.as_str(), &tokenizer)
        .map_err(FfiError::from)
        .map_err(Error::from)? as u32;
    metrics.record_tokens(Some(raw_tokens), Some(compressed_tokens));
    Ok(ContextResult {
        text: context_text,
        metrics: context_metrics_from(&metrics),
    })
}

#[napi]
pub fn encode_to_cells(
    file_path: String,
    options: Option<CellOptions>,
) -> napi::Result<Vec<EncodedCell>> {
    let preset = options
        .and_then(|o| o.preset)
        .map(|s| s)
        .unwrap_or_else(|| "reports".to_string());
    let encoder = Encoder::builder(preset.as_str())
        .map_err(FfiError::from)
        .map_err(Error::from)?
        .build();
    let (doc, _) = encoder
        .encode_path(Path::new(&file_path))
        .map_err(FfiError::from)
        .map_err(Error::from)?;
    let mut rows = Vec::new();
    for cell in doc.ordered_cells() {
        if let Some(payload) = doc.payload_for(&cell.code_id) {
            rows.push(EncodedCell {
                text: payload.to_string(),
                importance: cell.importance,
                page: cell.z,
                bbox: CellBbox {
                    x: cell.x,
                    y: cell.y,
                    w: cell.w,
                    h: cell.h,
                },
            });
        }
    }
    Ok(rows)
}
