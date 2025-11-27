use std::cmp::Reverse;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use html2text::from_read;
use image::{self, DynamicImage};
use pulldown_cmark::{html, Parser};
use rayon::prelude::*;

use crate::document::{
    hash_payload, CellRecord, CellType, CodeHash, Document, Header, NumGuard, PageInfo,
};
use crate::error::{DcfError, Result};
use crate::metrics::Metrics;
use crate::normalization::{
    classify_cell_type, importance_score, looks_like_table_with_tolerance, normalize_lines,
    HyphenationMode, ImportanceTuning,
};
use crate::numguard;

#[cfg(feature = "pdfium")]
use pdfium_render::prelude::*;

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub preset: EncoderPreset,
    pub grid: String,
    pub codeset: String,
    pub page_width_px: u32,
    pub page_height_px: u32,
    pub margin_left_px: i32,
    pub margin_top_px: i32,
    pub line_height_px: u32,
    pub line_gap_px: u32,
    pub budget: Option<usize>,
    pub drop_footers: bool,
    pub dedup_window_pages: u32,
    pub hyphenation: HyphenationMode,
    pub table_column_tolerance: u32,
    pub enable_ocr: bool,
    pub force_ocr: bool,
    pub ocr_languages: Vec<String>,
    pub importance: ImportanceTuning,
}

impl EncoderConfig {
    fn new(preset: EncoderPreset) -> Self {
        let (page_width_px, page_height_px, line_height_px, line_gap_px) = match preset {
            EncoderPreset::Slides => (1920, 1080, 42, 12),
            EncoderPreset::News => (1100, 1600, 28, 8),
            EncoderPreset::Scans => (1400, 2000, 30, 8),
            _ => (1024, 1400, 24, 6),
        };
        Self {
            preset,
            grid: "coarse".to_string(),
            codeset: "HASH256".to_string(),
            page_width_px,
            page_height_px,
            margin_left_px: 64,
            margin_top_px: 64,
            line_height_px,
            line_gap_px,
            budget: None,
            drop_footers: false,
            dedup_window_pages: 0,
            hyphenation: HyphenationMode::Merge,
            table_column_tolerance: 24,
            enable_ocr: false,
            force_ocr: false,
            ocr_languages: vec!["eng".to_string()],
            importance: ImportanceTuning::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncoderBuilder {
    config: EncoderConfig,
}

impl EncoderBuilder {
    pub fn new<S: AsRef<str>>(preset: S) -> Result<Self> {
        Ok(Self {
            config: EncoderConfig::new(EncoderPreset::from_str(preset.as_ref())?),
        })
    }

    pub fn budget(mut self, budget: Option<usize>) -> Self {
        self.config.budget = budget;
        self
    }

    pub fn drop_footers(mut self, drop: bool) -> Self {
        self.config.drop_footers = drop;
        self
    }

    pub fn dedup_window(mut self, window: u32) -> Self {
        self.config.dedup_window_pages = window;
        self
    }

    pub fn hyphenation(mut self, mode: HyphenationMode) -> Self {
        self.config.hyphenation = mode;
        self
    }

    pub fn table_tolerance(mut self, tolerance: u32) -> Self {
        self.config.table_column_tolerance = tolerance;
        self
    }

    pub fn enable_ocr(mut self, enable: bool) -> Self {
        self.config.enable_ocr = enable;
        self
    }

    pub fn force_ocr(mut self, force: bool) -> Self {
        self.config.force_ocr = force;
        self
    }

    pub fn ocr_languages(mut self, langs: Vec<String>) -> Self {
        if !langs.is_empty() {
            self.config.ocr_languages = langs;
        }
        self
    }

    pub fn importance_tuning(mut self, tuning: ImportanceTuning) -> Self {
        self.config.importance = tuning;
        self
    }

    pub fn build(self) -> Encoder {
        Encoder {
            config: self.config,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Encoder {
    config: EncoderConfig,
}

impl Encoder {
    pub fn builder<S: AsRef<str>>(preset: S) -> Result<EncoderBuilder> {
        EncoderBuilder::new(preset)
    }

    pub fn from_preset<S: AsRef<str>>(preset: S) -> Result<Self> {
        Ok(Self::builder(preset)?.build())
    }

    pub fn with_budget(mut self, budget: usize) -> Self {
        self.config.budget = Some(budget);
        self
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    pub fn encode_path<P: AsRef<Path>>(&self, path: P) -> Result<(Document, Metrics)> {
        let input = EncodeInput::from_path(path.as_ref(), &self.config)?;
        self.encode(input)
    }

    pub fn encode_path_with_plaintext<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(Document, Metrics, String)> {
        let input = EncodeInput::from_path(path.as_ref(), &self.config)?;
        self.encode_with_plaintext(input)
    }

    pub fn encode_with_plaintext(&self, input: EncodeInput) -> Result<(Document, Metrics, String)> {
        let raw_text = input.to_plaintext();
        let (document, metrics) = self.encode(input)?;
        Ok((document, metrics, raw_text))
    }

    pub fn encode(&self, input: EncodeInput) -> Result<(Document, Metrics)> {
        let mut document = Document::new(Header {
            version: 1,
            grid: self.config.grid.clone(),
            codeset: self.config.codeset.clone(),
        });
        let mut metrics = Metrics::default();
        metrics.pages = clamp_usize_to_u32(input.pages.len());

        for page in &input.pages {
            document.add_page(PageInfo {
                z: page.index,
                width_px: page.width_px,
                height_px: page.height_px,
            });
        }

        let processed_pages = input
            .pages
            .into_par_iter()
            .map(|page| self.encode_page(page))
            .collect::<Vec<_>>();

        let mut cells_total = 0usize;
        let mut lines_total = 0usize;
        for page_output in processed_pages {
            let page_output = page_output?;
            cells_total += page_output.cells.len();
            lines_total += page_output.line_count;
            document.cells.extend(page_output.cells);
            for guard in page_output.numguards {
                document.add_numguard(guard);
            }
            for (code, payload) in page_output.dict_entries {
                document.dict.entry(code).or_insert(payload);
            }
        }
        metrics.cells_total = clamp_usize_to_u32(cells_total);
        metrics.lines_total = clamp_usize_to_u32(lines_total);

        let unique_payloads = document.dict.len();

        self.apply_budget(&mut document);
        self.post_filters(&mut document);
        self.annotate_rle(&mut document.cells);
        metrics.cells_kept = clamp_usize_to_u32(document.cells.len());
        metrics.numguard_count = clamp_usize_to_u32(document.numguards.len());
        metrics.dedup_ratio = if unique_payloads == 0 {
            0.0
        } else {
            metrics.cells_total as f32 / unique_payloads as f32
        };

        Ok((document, metrics))
    }

    fn encode_page(&self, page: PageBuffer) -> Result<PageResult> {
        let normalized = normalize_lines(&page.lines, self.config.hyphenation);
        let mut y = self.config.margin_top_px;
        let mut cells = Vec::with_capacity(normalized.len());
        let mut dict_entries = Vec::new();
        let mut numguards_acc = Vec::new();
        for (line_index, line) in normalized.iter().enumerate() {
            let mut cell_type: CellType = classify_cell_type(line);
            if cell_type == CellType::Text
                && looks_like_table_with_tolerance(line, self.config.table_column_tolerance)
            {
                cell_type = CellType::Table;
            }
            let importance = importance_score(line, cell_type, line_index, &self.config.importance);
            let code_id = hash_payload(line);
            let w = (page.width_px as i32 - self.config.margin_left_px * 2).max(0) as u32;
            let cell = CellRecord {
                z: page.index,
                x: self.config.margin_left_px,
                y,
                w,
                h: self.config.line_height_px,
                code_id,
                rle: 0,
                cell_type,
                importance,
            };
            cells.push(cell);
            dict_entries.push((code_id, line.clone()));
            let guards = numguard::extract_guards(
                line,
                page.index,
                self.config.margin_left_px as u32,
                y as u32,
            );
            numguards_acc.extend(guards);
            y += (self.config.line_height_px + self.config.line_gap_px) as i32;
        }
        Ok(PageResult {
            cells,
            dict_entries,
            numguards: numguards_acc,
            line_count: normalized.len(),
        })
    }

    fn apply_budget(&self, doc: &mut Document) {
        if let Some(limit) = self.config.budget {
            if doc.cells.len() <= limit {
                return;
            }
            doc.cells.sort_by_key(|c| (Reverse(c.importance), c.key()));
            doc.cells.truncate(limit);
            doc.cells.sort_by_key(|c| c.key());
            doc.retain_dict_for_cells();
        }
    }

    fn post_filters(&self, doc: &mut Document) {
        if self.config.drop_footers {
            doc.cells.retain(|c| c.cell_type != CellType::Footer);
        }
        if self.config.dedup_window_pages > 0 {
            let mut seen: HashMap<CodeHash, Vec<u32>> = HashMap::new();
            doc.cells.retain(|cell| {
                let entry = seen.entry(cell.code_id).or_insert_with(Vec::new);
                if entry
                    .iter()
                    .any(|z| cell.z.abs_diff(*z) <= self.config.dedup_window_pages)
                {
                    false
                } else {
                    entry.push(cell.z);
                    true
                }
            });
        }
        doc.cells.sort_by_key(|c| c.key());
        doc.retain_dict_for_cells();
    }

    fn annotate_rle(&self, cells: &mut [CellRecord]) {
        if cells.is_empty() {
            return;
        }
        let mut i = 0;
        while i < cells.len() {
            let mut run = 1;
            while i + run < cells.len() && cells[i + run].code_id == cells[i].code_id {
                run += 1;
            }
            cells[i].rle = (run - 1) as u32;
            for j in 1..run {
                cells[i + j].rle = 0;
            }
            i += run;
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncodeInput {
    pub pages: Vec<PageBuffer>,
}

impl EncodeInput {
    pub fn from_path(path: &Path, config: &EncoderConfig) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());

        match ext.as_deref() {
            Some("txt") | Some("text") => {
                let content = read_text_lossy(path)?;
                Ok(Self {
                    pages: text_to_pages(&content, config),
                })
            }
            Some("md") | Some("markdown") => {
                let content = read_text_lossy(path)?;
                let markdown = markdown_to_text(&content);
                Ok(Self {
                    pages: text_to_pages(&markdown, config),
                })
            }
            Some("html") | Some("htm") => {
                let content = read_text_lossy(path)?;
                let flattened = html_to_plaintext(&content);
                Ok(Self {
                    pages: text_to_pages(&flattened, config),
                })
            }
            Some("tex") | Some("json") | Some("bib") => {
                let content = read_text_lossy(path)?;
                Ok(Self {
                    pages: text_to_pages(&content, config),
                })
            }
            Some("pdf") => Self::from_pdf(path, config),
            Some(ext) if is_image_ext(ext) => Self::from_image(path, config),
            None => {
                let content = read_text_lossy(path)?;
                Ok(Self {
                    pages: text_to_pages(&content, config),
                })
            }
            _ => Err(DcfError::UnsupportedInput(path.to_path_buf())),
        }
    }

    fn from_pdf(path: &Path, config: &EncoderConfig) -> Result<Self> {
        #[cfg(feature = "pdfium")]
        {
            match pdfium_pdf_to_pages(path, config) {
                Ok(pages) => return Ok(Self { pages }),
                Err(err) => {
                    tracing::warn!("pdfium read failed: {err}");
                }
            }
        }
        let pages = fallback_pdf_to_pages(path, config)?;
        Ok(Self { pages })
    }

    fn from_image(path: &Path, config: &EncoderConfig) -> Result<Self> {
        let image = image::open(path).map_err(|e| {
            DcfError::Other(format!("failed to open image {}: {e}", path.display()))
        })?;
        let pages = ocr_image_to_pages(image, config)?;
        Ok(Self { pages })
    }

    pub fn to_plaintext(&self) -> String {
        let mut buffer = String::new();
        for (idx, page) in self.pages.iter().enumerate() {
            if idx > 0 {
                buffer.push_str("\n\n");
            }
            for line in &page.lines {
                buffer.push_str(line);
                buffer.push('\n');
            }
        }
        buffer
    }
}

#[derive(Debug, Clone)]
pub struct PageBuffer {
    pub index: u32,
    pub width_px: u32,
    pub height_px: u32,
    pub lines: Vec<String>,
}

impl PageBuffer {
    fn from_text(index: u32, text: &str, config: &EncoderConfig) -> Self {
        let wrap_width = (config.page_width_px / 10).max(40) as usize;
        let mut lines = Vec::new();
        for raw_line in text.lines() {
            if raw_line.trim().is_empty() {
                lines.push(String::new());
                continue;
            }
            for chunk in wrap_line(raw_line, wrap_width) {
                lines.push(chunk);
            }
        }
        if lines.is_empty() {
            lines.push(String::new());
        }
        Self {
            index,
            width_px: config.page_width_px,
            height_px: config.page_height_px,
            lines,
        }
    }
}

#[derive(Debug, Clone)]
struct PageResult {
    cells: Vec<CellRecord>,
    dict_entries: Vec<(CodeHash, String)>,
    numguards: Vec<NumGuard>,
    line_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum EncoderPreset {
    Reports,
    Slides,
    News,
    Scans,
    Custom,
}

impl EncoderPreset {
    pub fn from_str(name: &str) -> Result<Self> {
        match name.to_lowercase().as_str() {
            "reports" => Ok(Self::Reports),
            "slides" => Ok(Self::Slides),
            "news" => Ok(Self::News),
            "scans" => Ok(Self::Scans),
            "custom" => Ok(Self::Custom),
            other => Err(DcfError::UnknownPreset(other.to_string())),
        }
    }
}

fn text_to_pages(text: &str, config: &EncoderConfig) -> Vec<PageBuffer> {
    text.split('\u{c}')
        .enumerate()
        .map(|(idx, chunk)| PageBuffer::from_text(idx as u32, chunk, config))
        .collect()
}

fn fallback_pdf_to_pages(path: &Path, config: &EncoderConfig) -> Result<Vec<PageBuffer>> {
    let pages = pdf_extract::extract_text_by_pages(path)
        .map_err(|e| DcfError::Other(format!("pdf extract failed: {e}")))?;
    Ok(pages
        .into_iter()
        .enumerate()
        .map(|(idx, txt)| PageBuffer::from_text(idx as u32, &txt, config))
        .collect())
}

#[cfg(feature = "pdfium")]
fn pdfium_pdf_to_pages(path: &Path, config: &EncoderConfig) -> Result<Vec<PageBuffer>> {
    let bindings = Pdfium::bind_to_system_library()
        .map_err(|e| DcfError::Other(format!("pdfium binding failed: {e}")))?;
    let pdfium = Pdfium::new(bindings);
    let document = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| DcfError::Other(format!("pdfium load failed: {e}")))?;
    let mut buffers = Vec::new();
    for (idx, page) in document.pages().iter().enumerate() {
        let mut page_text = page.text().ok().map(|t| t.all()).unwrap_or_default();
        let mut should_ocr = config.force_ocr;
        if !should_ocr {
            let trimmed = page_text.trim();
            if trimmed.is_empty() || trimmed.len() < 16 {
                should_ocr = config.enable_ocr;
            }
        }
        if should_ocr {
            if !config.enable_ocr {
                return Err(DcfError::OcrSupportDisabled);
            }
            let target_width: i32 = config
                .page_width_px
                .try_into()
                .map_err(|_| DcfError::Other("page width exceeds i32::MAX".to_string()))?;
            let target_height: i32 = config
                .page_height_px
                .try_into()
                .map_err(|_| DcfError::Other("page height exceeds i32::MAX".to_string()))?;
            let render_config = PdfRenderConfig::new()
                .set_target_width(target_width)
                .set_target_height(target_height);
            let render = page
                .render_with_config(&render_config)
                .map_err(|e| DcfError::Other(format!("pdf render failed: {e}")))?;
            let image = render.as_image();
            page_text = crate::ocr::image_to_text(&image, &config.ocr_languages)?;
        }
        buffers.push(PageBuffer::from_text(idx as u32, &page_text, config));
    }
    Ok(buffers)
}

fn markdown_to_text(md: &str) -> String {
    let mut html_buf = String::new();
    html::push_html(&mut html_buf, Parser::new(md));
    html_to_plaintext(&html_buf)
}

fn html_to_plaintext(html_src: &str) -> String {
    from_read(html_src.as_bytes(), 80)
}

fn is_image_ext(ext: &str) -> bool {
    matches!(
        ext,
        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp" | "webp" | "gif"
    )
}

fn clamp_usize_to_u32(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

fn wrap_line(line: &str, width: usize) -> Vec<String> {
    if line.len() <= width {
        return vec![line.trim().to_string()];
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for word in line.split_whitespace() {
        if current.len() + word.len() + 1 > width && !current.is_empty() {
            out.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        out.push(current.trim().to_string());
    }
    if out.is_empty() {
        out.push(line.trim().to_string());
    }
    out
}

fn read_text_lossy(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

#[cfg(feature = "ocr")]
fn ocr_image_to_pages(image: DynamicImage, config: &EncoderConfig) -> Result<Vec<PageBuffer>> {
    if !config.enable_ocr {
        return Err(DcfError::OcrSupportDisabled);
    }
    let text = crate::ocr::image_to_text(&image, &config.ocr_languages)?;
    Ok(text_to_pages(&text, config))
}

#[cfg(not(feature = "ocr"))]
fn ocr_image_to_pages(_image: DynamicImage, _config: &EncoderConfig) -> Result<Vec<PageBuffer>> {
    Err(DcfError::OcrSupportDisabled)
}
