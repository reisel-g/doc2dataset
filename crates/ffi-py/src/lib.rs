use std::path::{Path, PathBuf};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use thiserror::Error;
use three_dcf_core::{
    estimate_tokens, CellRecord, Decoder, Document, Encoder, Metrics, Stats as CoreStats,
    TextSerializer, TextSerializerConfig, TokenizerKind,
};

#[derive(Error, Debug)]
pub enum FfiError {
    #[error("{0}")]
    Core(#[from] three_dcf_core::DcfError),
    #[error("unknown tokenizer '{0}'")]
    UnknownTokenizer(String),
}

impl From<FfiError> for PyErr {
    fn from(err: FfiError) -> Self {
        PyRuntimeError::new_err(err.to_string())
    }
}

fn default_preset<'a>(preset: Option<&'a str>) -> &'a str {
    preset.unwrap_or("reports")
}

fn encode_internal(
    input: &Path,
    output: &Path,
    preset: Option<&str>,
    budget: Option<usize>,
    json_out: Option<&Path>,
    text_out: Option<&Path>,
) -> Result<(), FfiError> {
    let mut builder = Encoder::builder(default_preset(preset))?;
    if let Some(b) = budget {
        builder = builder.budget(Some(b));
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

fn decode_internal(document: &Path, page: Option<u32>) -> Result<String, FfiError> {
    let document = Document::load_bin(document)?;
    if let Some(z) = page {
        Ok(document.decode_page_to_text(z))
    } else {
        let decoder = Decoder::new();
        Ok(decoder.to_text(&document)?)
    }
}

fn stats_internal(document: &Path, tokenizer: Option<&str>) -> Result<CoreStats, FfiError> {
    let document = Document::load_bin(document)?;
    let kind = tokenizer_from(tokenizer)?;
    Ok(CoreStats::measure(&document, kind)?)
}

#[pyclass]
pub struct StatsResult {
    tokens_raw: usize,
    tokens_3dcf: usize,
    cells: usize,
    unique_payloads: usize,
    savings_ratio: f32,
}

#[pyclass(name = "ContextResult")]
pub struct PyContextResult {
    text: String,
    metrics: Py<PyDict>,
}

#[pymethods]
impl PyContextResult {
    #[getter]
    fn text(&self) -> &str {
        &self.text
    }

    #[getter]
    fn metrics<'py>(&self, py: Python<'py>) -> Py<PyDict> {
        self.metrics.clone_ref(py)
    }
}

#[pyclass(name = "Cell")]
pub struct PyCell {
    text: String,
    importance: u8,
    page: u32,
    bbox: Py<PyDict>,
}

#[pymethods]
impl PyCell {
    #[getter]
    fn text(&self) -> &str {
        &self.text
    }

    #[getter]
    fn importance(&self) -> u8 {
        self.importance
    }

    #[getter]
    fn page(&self) -> u32 {
        self.page
    }

    #[getter]
    fn bbox<'py>(&self, py: Python<'py>) -> Py<PyDict> {
        self.bbox.clone_ref(py)
    }
}

impl From<CoreStats> for StatsResult {
    fn from(value: CoreStats) -> Self {
        Self {
            tokens_raw: value.tokens_raw,
            tokens_3dcf: value.tokens_3dcf,
            cells: value.cells,
            unique_payloads: value.unique_payloads,
            savings_ratio: value.savings_ratio,
        }
    }
}

#[pymethods]
impl StatsResult {
    #[getter]
    fn tokens_raw(&self) -> usize {
        self.tokens_raw
    }

    #[getter]
    fn tokens_3dcf(&self) -> usize {
        self.tokens_3dcf
    }

    #[getter]
    fn cells(&self) -> usize {
        self.cells
    }

    #[getter]
    fn unique_payloads(&self) -> usize {
        self.unique_payloads
    }

    #[getter]
    fn savings_ratio(&self) -> f32 {
        self.savings_ratio
    }
}

#[pyfunction]
fn encode(
    input: &str,
    output: &str,
    preset: Option<&str>,
    budget: Option<usize>,
    json_out: Option<&str>,
    text_out: Option<&str>,
) -> PyResult<()> {
    encode_internal(
        Path::new(input),
        Path::new(output),
        preset,
        budget,
        json_out.map(Path::new),
        text_out.map(Path::new),
    )
    .map_err(PyErr::from)
}

#[pyfunction]
fn decode_text(document: &str, page: Option<u32>) -> PyResult<String> {
    decode_internal(Path::new(document), page).map_err(PyErr::from)
}

#[pyfunction]
fn stats(document: &str, tokenizer: Option<&str>) -> PyResult<StatsResult> {
    stats_internal(Path::new(document), tokenizer)
        .map(StatsResult::from)
        .map_err(PyErr::from)
}

#[pyfunction]
fn encode_to_context(
    py: Python<'_>,
    input: &str,
    preset: Option<&str>,
    budget: Option<usize>,
    tokenizer: Option<&str>,
) -> PyResult<PyContextResult> {
    let mut builder = Encoder::builder(default_preset(preset))
        .map_err(FfiError::from)
        .map_err(PyErr::from)?;
    if let Some(b) = budget {
        builder = builder.budget(Some(b));
    }
    let encoder = builder.build();
    let (doc, mut metrics, raw_text) = encoder
        .encode_path_with_plaintext(Path::new(input))
        .map_err(FfiError::from)
        .map_err(PyErr::from)?;
    let serializer = TextSerializer::with_config(TextSerializerConfig::default());
    let context_text = serializer
        .to_string(&doc)
        .map_err(FfiError::from)
        .map_err(PyErr::from)?;
    let tokenizer = tokenizer_from(tokenizer)?;
    let raw_tokens = estimate_tokens(&raw_text, &tokenizer)
        .map_err(FfiError::from)
        .map_err(PyErr::from)? as u32;
    let compressed_tokens = estimate_tokens(context_text.as_str(), &tokenizer)
        .map_err(FfiError::from)
        .map_err(PyErr::from)? as u32;
    metrics.record_tokens(Some(raw_tokens), Some(compressed_tokens));
    let metrics_dict = metrics_to_dict(py, &metrics)?;
    Ok(PyContextResult {
        text: context_text,
        metrics: metrics_dict,
    })
}

#[pyfunction]
fn encode_to_cells(
    py: Python<'_>,
    input: &str,
    preset: Option<&str>,
) -> PyResult<Vec<PyCell>> {
    let encoder = Encoder::builder(default_preset(preset))
        .map_err(FfiError::from)
        .map_err(PyErr::from)?
        .build();
    let (doc, _) = encoder
        .encode_path(Path::new(input))
        .map_err(FfiError::from)
        .map_err(PyErr::from)?;
    let mut cells = Vec::new();
    for cell in doc.ordered_cells() {
        if let Some(payload) = doc.payload_for(&cell.code_id) {
            let bbox = cell_bbox_dict(py, &cell)?;
            cells.push(PyCell {
                text: payload.to_string(),
                importance: cell.importance,
                page: cell.z,
                bbox,
            });
        }
    }
    Ok(cells)
}

fn metrics_to_dict(py: Python<'_>, metrics: &Metrics) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("pages", metrics.pages)?;
    dict.set_item("lines_total", metrics.lines_total)?;
    dict.set_item("cells_total", metrics.cells_total)?;
    dict.set_item("cells_kept", metrics.cells_kept)?;
    dict.set_item("dedup_ratio", metrics.dedup_ratio)?;
    dict.set_item("numguard_count", metrics.numguard_count)?;
    dict.set_item("raw_tokens_estimate", metrics.raw_tokens_estimate)?;
    dict.set_item(
        "compressed_tokens_estimate",
        metrics.compressed_tokens_estimate,
    )?;
    dict.set_item("compression_factor", metrics.compression_factor)?;
    Ok(dict.into())
}

fn cell_bbox_dict(py: Python<'_>, cell: &CellRecord) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("x", cell.x)?;
    dict.set_item("y", cell.y)?;
    dict.set_item("w", cell.w)?;
    dict.set_item("h", cell.h)?;
    Ok(dict.into())
}

#[pymodule]
fn three_dcf_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode_text, m)?)?;
    m.add_function(wrap_pyfunction!(stats, m)?)?;
    m.add_function(wrap_pyfunction!(encode_to_context, m)?)?;
    m.add_function(wrap_pyfunction!(encode_to_cells, m)?)?;
    m.add_class::<StatsResult>()?;
    m.add_class::<PyContextResult>()?;
    m.add_class::<PyCell>()?;
    Ok(())
}
