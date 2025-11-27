use hex;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tiktoken_rs::CoreBPE;

use crate::document::{CellRecord, CellType, Document};

static TOKENIZER: Lazy<CoreBPE> = Lazy::new(|| tiktoken_rs::cl100k_base().expect("tokenizer"));
const CHUNK_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChunkMode {
    Cells,
    Tokens,
    Headings,
    TableRows,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChunkConfig {
    pub mode: ChunkMode,
    pub cells_per_chunk: usize,
    pub overlap_cells: usize,
    pub max_tokens: usize,
    pub overlap_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            mode: ChunkMode::Cells,
            cells_per_chunk: 200,
            overlap_cells: 20,
            max_tokens: 512,
            overlap_tokens: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub chunk_id: String,
    pub doc: String,
    pub chunk_index: usize,
    pub z_start: u32,
    pub z_end: u32,
    pub cell_start: usize,
    pub cell_end: usize,
    pub text: String,
    #[serde(default)]
    pub token_count: usize,
    #[serde(default = "default_cell_type")]
    pub dominant_type: CellType,
    #[serde(default)]
    pub importance_mean: f32,
}

pub struct Chunker {
    config: ChunkConfig,
}

impl Chunker {
    pub fn new(config: ChunkConfig) -> Self {
        Self { config }
    }

    pub fn chunk_document(&self, document: &Document, doc_id: &str) -> Vec<ChunkRecord> {
        let ordered = document.ordered_cells();
        if ordered.is_empty() {
            return Vec::new();
        }
        match self.config.mode {
            ChunkMode::Cells => self.chunk_by_cells(document, doc_id, &ordered),
            ChunkMode::Tokens => self.chunk_by_tokens(document, doc_id, &ordered),
            ChunkMode::Headings => self.chunk_by_headings(document, doc_id, &ordered),
            ChunkMode::TableRows => self.chunk_table_blocks(document, doc_id, &ordered),
        }
    }

    fn chunk_by_cells(
        &self,
        document: &Document,
        doc_id: &str,
        ordered: &[CellRecord],
    ) -> Vec<ChunkRecord> {
        let chunk_size = self.config.cells_per_chunk.max(1);
        let overlap = self.config.overlap_cells.min(chunk_size.saturating_sub(1));
        let mut start = 0usize;
        let mut chunk_index = 0usize;
        let mut chunks = Vec::new();
        while start < ordered.len() {
            let end = (start + chunk_size).min(ordered.len());
            if let Some(record) =
                self.build_chunk(document, doc_id, chunk_index, start, end, ordered)
            {
                chunks.push(record);
                chunk_index += 1;
            }
            if end == ordered.len() {
                break;
            }
            start = if overlap == 0 {
                end
            } else {
                end.saturating_sub(overlap)
            };
        }
        chunks
    }

    fn chunk_by_tokens(
        &self,
        document: &Document,
        doc_id: &str,
        ordered: &[CellRecord],
    ) -> Vec<ChunkRecord> {
        let max_tokens = self.config.max_tokens.max(1);
        let overlap_tokens = self.config.overlap_tokens.min(max_tokens.saturating_sub(1));
        let tokens_per_cell = token_counts(document, ordered);
        let mut start = 0usize;
        let mut chunk_index = 0usize;
        let mut chunks = Vec::new();
        while start < ordered.len() {
            let mut end = start;
            let mut used_tokens = 0usize;
            while end < ordered.len() {
                let cell_tokens = tokens_per_cell[end].max(1);
                if end > start && used_tokens + cell_tokens > max_tokens {
                    break;
                }
                used_tokens += cell_tokens;
                end += 1;
            }
            if end == start {
                end += 1;
            }
            if let Some(record) =
                self.build_chunk(document, doc_id, chunk_index, start, end, ordered)
            {
                chunks.push(record);
                chunk_index += 1;
            }
            if end == ordered.len() {
                break;
            }
            if overlap_tokens == 0 {
                start = end;
            } else {
                let mut back_tokens = 0usize;
                let mut new_start = end;
                while new_start > start {
                    new_start -= 1;
                    back_tokens += tokens_per_cell[new_start].max(1);
                    if back_tokens >= overlap_tokens {
                        break;
                    }
                }
                start = new_start;
            }
        }
        chunks
    }

    fn chunk_by_headings(
        &self,
        document: &Document,
        doc_id: &str,
        ordered: &[CellRecord],
    ) -> Vec<ChunkRecord> {
        let mut chunks = Vec::new();
        let tokens_per_cell = token_counts(document, ordered);
        let mut chunk_index = 0usize;
        let mut idx = 0usize;
        while idx < ordered.len() {
            if ordered[idx].cell_type != CellType::Header {
                idx += 1;
                continue;
            }
            let start = idx;
            let mut end = idx;
            let mut tokens = 0usize;
            while end < ordered.len() {
                if end > start && ordered[end].cell_type == CellType::Header {
                    break;
                }
                tokens += tokens_per_cell[end];
                if self.config.max_tokens > 0 && tokens >= self.config.max_tokens {
                    end += 1;
                    break;
                }
                end += 1;
            }
            if let Some(record) =
                self.build_chunk(document, doc_id, chunk_index, start, end, ordered)
            {
                chunks.push(record);
                chunk_index += 1;
            }
            idx = end;
        }
        chunks
    }

    fn chunk_table_blocks(
        &self,
        document: &Document,
        doc_id: &str,
        ordered: &[CellRecord],
    ) -> Vec<ChunkRecord> {
        let mut chunks = Vec::new();
        let mut idx = 0usize;
        let mut chunk_index = 0usize;
        while idx < ordered.len() {
            if ordered[idx].cell_type != CellType::Table {
                idx += 1;
                continue;
            }
            let mut block_end = idx;
            while block_end < ordered.len() && ordered[block_end].cell_type == CellType::Table {
                block_end += 1;
            }
            let mut start = idx;
            while start < block_end {
                let end = (start + self.config.cells_per_chunk.max(1)).min(block_end);
                if let Some(record) =
                    self.build_chunk(document, doc_id, chunk_index, start, end, ordered)
                {
                    chunks.push(record);
                    chunk_index += 1;
                }
                start = end;
            }
            idx = block_end;
        }
        chunks
    }

    fn build_chunk(
        &self,
        document: &Document,
        doc_id: &str,
        chunk_index: usize,
        start: usize,
        end: usize,
        ordered: &[CellRecord],
    ) -> Option<ChunkRecord> {
        if start >= end || start >= ordered.len() {
            return None;
        }
        let slice = &ordered[start..end];
        let mut parts = Vec::with_capacity(slice.len());
        let mut token_total = 0usize;
        let mut importance_sum = 0usize;
        let mut type_hist = [0usize; 5];
        for cell in slice {
            if let Some(payload) = document.payload_for(&cell.code_id) {
                if !payload.trim().is_empty() {
                    parts.push(payload.to_string());
                }
                token_total += count_tokens(payload);
            }
            importance_sum += cell.importance as usize;
            increment_histogram(&mut type_hist, cell.cell_type);
        }
        let text = parts.join("\n");
        if text.trim().is_empty() {
            return None;
        }
        let z_start = slice.first().map(|c| c.z).unwrap_or(0);
        let z_end = slice.last().map(|c| c.z).unwrap_or(z_start);
        let chunk_id = stable_chunk_id(
            doc_id,
            chunk_index,
            start,
            end.saturating_sub(1),
            self.config.mode,
            CHUNK_VERSION,
        );
        let dominant_type = dominant_cell_type(&type_hist);
        let importance_mean = if slice.is_empty() {
            0.0
        } else {
            importance_sum as f32 / (slice.len() as f32 * 255.0)
        };
        Some(ChunkRecord {
            chunk_id,
            doc: doc_id.to_string(),
            chunk_index,
            z_start,
            z_end,
            cell_start: start,
            cell_end: end.saturating_sub(1),
            text,
            token_count: token_total,
            dominant_type,
            importance_mean,
        })
    }
}

fn increment_histogram(hist: &mut [usize; 5], cell_type: CellType) {
    match cell_type {
        CellType::Text => hist[0] += 1,
        CellType::Table => hist[1] += 1,
        CellType::Figure => hist[2] += 1,
        CellType::Footer => hist[3] += 1,
        CellType::Header => hist[4] += 1,
    }
}

fn dominant_cell_type(hist: &[usize; 5]) -> CellType {
    let mut max_idx = 0usize;
    let mut max_val = 0usize;
    for (idx, val) in hist.iter().enumerate() {
        if *val > max_val {
            max_val = *val;
            max_idx = idx;
        }
    }
    match max_idx {
        0 => CellType::Text,
        1 => CellType::Table,
        2 => CellType::Figure,
        3 => CellType::Footer,
        _ => CellType::Header,
    }
}

fn default_cell_type() -> CellType {
    CellType::Text
}

fn token_counts(document: &Document, cells: &[CellRecord]) -> Vec<usize> {
    cells
        .iter()
        .map(|cell| {
            document
                .payload_for(&cell.code_id)
                .map(count_tokens)
                .unwrap_or(0)
        })
        .collect()
}

fn count_tokens(text: &str) -> usize {
    TOKENIZER.encode_with_special_tokens(text).len()
}

fn stable_chunk_id(
    doc_id: &str,
    chunk_index: usize,
    cell_start: usize,
    cell_end: usize,
    mode: ChunkMode,
    version: u32,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(doc_id.as_bytes());
    hasher.update(&version.to_be_bytes());
    hasher.update(&(mode_discriminant(mode)).to_be_bytes());
    hasher.update(&chunk_index.to_be_bytes());
    hasher.update(&cell_start.to_be_bytes());
    hasher.update(&cell_end.to_be_bytes());
    hex::encode(hasher.finalize())
}

fn mode_discriminant(mode: ChunkMode) -> u32 {
    match mode {
        ChunkMode::Cells => 0,
        ChunkMode::Tokens => 1,
        ChunkMode::Headings => 2,
        ChunkMode::TableRows => 3,
    }
}
