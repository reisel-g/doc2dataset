use std::cell::RefCell;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use sysinfo::{Pid, System};
use walkdir::WalkDir;

use crate::document::Document;
use crate::encoder::Encoder;
use crate::error::Result;
use crate::metrics::{cer, numeric_stats, wer};
use crate::stats::{Stats, TokenizerKind};

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchMode {
    Encode,
    Decode,
    Full,
}

#[derive(Debug, Clone, Copy)]
enum BenchStage {
    Encode,
    Decode,
}

impl BenchStage {
    fn as_mode(self) -> BenchMode {
        match self {
            BenchStage::Encode => BenchMode::Encode,
            BenchStage::Decode => BenchMode::Decode,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub mode: BenchMode,
    pub root: PathBuf,
    pub gold_root: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub preset: String,
    pub tokenizer: TokenizerKind,
    pub budgets: Vec<Option<usize>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub row_type: &'static str,
    pub run_id: String,
    pub mode: BenchMode,
    pub doc: String,
    pub preset: String,
    pub encode_ms: u128,
    pub decode_ms: u128,
    pub cer: Option<f64>,
    pub wer: Option<f64>,
    pub numguard_f1: Option<f64>,
    pub units_ok: Option<f64>,
    pub tokens_raw: usize,
    pub tokens_3dcf: usize,
    pub savings_ratio: f64,
    pub avg_cells_kept_per_page: f64,
    pub pages: usize,
    pub budget: Option<usize>,
    pub numguard_mismatches: usize,
    pub encode_pages_per_s: f64,
    pub decode_pages_per_s: f64,
    pub mem_peak_mb: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchPageRow {
    pub row_type: &'static str,
    pub run_id: String,
    pub doc: String,
    pub preset: String,
    pub page_idx: u32,
    pub cer_page: f64,
    pub precision_page: f64,
    pub tokens_gold_page: usize,
    pub tokens_3dcf_page: usize,
    pub compression_ratio: f64,
    pub budget: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorpusMetrics {
    pub results: Vec<BenchResult>,
    pub mean_savings: f64,
    pub median_savings: f64,
    pub encode_p50_ms: f64,
    pub encode_p95_ms: f64,
    pub decode_p50_ms: f64,
    pub decode_p95_ms: f64,
    pub mean_encode_pages_per_s: f64,
    pub mean_decode_pages_per_s: f64,
    pub max_mem_mb: f64,
}

pub struct BenchRunner {
    config: BenchConfig,
    tokenizer: tiktoken_rs::CoreBPE,
    sys: RefCell<System>,
    pid: Pid,
    mem_peak_mb: RefCell<f64>,
}

impl BenchRunner {
    pub fn new(config: BenchConfig) -> Result<Self> {
        let tokenizer = config.tokenizer.build()?;
        let pid =
            sysinfo::get_current_pid().map_err(|e| crate::error::DcfError::Other(e.to_string()))?;
        let mut sys = System::new();
        sys.refresh_process(pid);
        Ok(Self {
            config,
            tokenizer,
            sys: RefCell::new(sys),
            pid,
            mem_peak_mb: RefCell::new(0.0),
        })
    }

    pub fn run(&self) -> Result<CorpusMetrics> {
        let budgets = if self.config.budgets.is_empty() {
            vec![None]
        } else {
            self.config.budgets.clone()
        };

        let mut doc_rows = Vec::new();
        *self.mem_peak_mb.borrow_mut() = 0.0;
        match self.config.mode {
            BenchMode::Encode => {
                for budget in budgets {
                    doc_rows.extend(self.run_encode_cycle(budget)?);
                }
            }
            BenchMode::Decode => {
                doc_rows.extend(self.run_decode_cycle()?);
            }
            BenchMode::Full => {
                for budget in budgets {
                    doc_rows.extend(self.run_encode_cycle(budget)?);
                }
                doc_rows.extend(self.run_decode_cycle()?);
            }
        }

        let mean = if doc_rows.is_empty() {
            0.0
        } else {
            doc_rows.iter().map(|r| r.savings_ratio).sum::<f64>() / doc_rows.len() as f64
        };
        let mut ordered = doc_rows.iter().map(|r| r.savings_ratio).collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if ordered.is_empty() {
            0.0
        } else {
            let mid = ordered.len() / 2;
            ordered[mid]
        };
        let encode_ms_vals = doc_rows
            .iter()
            .map(|r| r.encode_ms as f64)
            .collect::<Vec<_>>();
        let decode_ms_vals = doc_rows
            .iter()
            .map(|r| r.decode_ms as f64)
            .collect::<Vec<_>>();
        let encode_p50 = percentile(&encode_ms_vals, 0.5);
        let encode_p95 = percentile(&encode_ms_vals, 0.95);
        let decode_p50 = percentile(&decode_ms_vals, 0.5);
        let decode_p95 = percentile(&decode_ms_vals, 0.95);
        let mean_encode_pages = if doc_rows.is_empty() {
            0.0
        } else {
            doc_rows.iter().map(|r| r.encode_pages_per_s).sum::<f64>() / doc_rows.len() as f64
        };
        let mean_decode_pages = if doc_rows.is_empty() {
            0.0
        } else {
            doc_rows.iter().map(|r| r.decode_pages_per_s).sum::<f64>() / doc_rows.len() as f64
        };
        Ok(CorpusMetrics {
            results: doc_rows,
            mean_savings: mean,
            median_savings: median,
            encode_p50_ms: encode_p50,
            encode_p95_ms: encode_p95,
            decode_p50_ms: decode_p50,
            decode_p95_ms: decode_p95,
            mean_encode_pages_per_s: mean_encode_pages,
            mean_decode_pages_per_s: mean_decode_pages,
            max_mem_mb: *self.mem_peak_mb.borrow(),
        })
    }

    fn run_encode_cycle(&self, budget: Option<usize>) -> Result<Vec<BenchResult>> {
        let encoder = self.build_encoder(budget)?;
        let mut rows = Vec::new();
        for entry in WalkDir::new(&self.config.root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            if !is_supported_source(path) {
                continue;
            }
            let (row, pages) = self.process_encode_doc(&encoder, path, budget)?;
            self.append_row(&row)?;
            for page in pages {
                self.append_page_row(&page)?;
            }
            rows.push(row);
        }
        Ok(rows)
    }

    fn run_decode_cycle(&self) -> Result<Vec<BenchResult>> {
        let mut rows = Vec::new();
        for entry in WalkDir::new(&self.config.root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("3dcf") {
                continue;
            }
            let (row, pages) = self.process_decode_doc(path)?;
            self.append_row(&row)?;
            for page in pages {
                self.append_page_row(&page)?;
            }
            rows.push(row);
        }
        Ok(rows)
    }

    fn process_encode_doc(
        &self,
        encoder: &Encoder,
        path: &Path,
        budget: Option<usize>,
    ) -> Result<(BenchResult, Vec<BenchPageRow>)> {
        let encode_start = Instant::now();
        let (doc, _) = encoder.encode_path(path)?;
        let encode_ms = encode_start.elapsed().as_millis();
        self.measure_doc(path, &doc, encode_ms, BenchStage::Encode, budget)
    }

    fn process_decode_doc(&self, path: &Path) -> Result<(BenchResult, Vec<BenchPageRow>)> {
        let load_start = Instant::now();
        let doc = Document::load_bin(path)?;
        let _load_ms = load_start.elapsed().as_millis();
        let (row, pages) = self.measure_doc(path, &doc, 0, BenchStage::Decode, None)?;
        Ok((row, pages))
    }

    fn measure_doc(
        &self,
        path: &Path,
        doc: &Document,
        encode_ms: u128,
        stage: BenchStage,
        budget: Option<usize>,
    ) -> Result<(BenchResult, Vec<BenchPageRow>)> {
        let decode_start = Instant::now();
        let decoded = doc.decode_to_text();
        let decode_ms = decode_start.elapsed().as_millis();
        let stats = Stats::measure_with_bpe(doc, &self.tokenizer)?;
        let rel = self.relative_path(path);
        let run_id = self.run_id(stage, budget);
        let gold = self.load_gold(&rel, doc.total_pages())?;

        let (cer_doc, wer_doc, num_stats) = if let Some(gold_doc) = &gold {
            let gold_text = gold_doc
                .doc
                .clone()
                .unwrap_or_else(|| gold_doc.joined_pages());
            (
                Some(cer(&decoded, &gold_text)),
                Some(wer(&decoded, &gold_text)),
                Some(numeric_stats(&decoded, &gold_text)),
            )
        } else {
            (None, None, None)
        };

        let avg_cells = if doc.total_pages() == 0 {
            0.0
        } else {
            doc.total_cells() as f64 / doc.total_pages() as f64
        };

        let numguard_alerts = doc.numguard_mismatches();
        let mem_mb = self.observe_memory_mb();

        let pages_f = doc.total_pages().max(1) as f64;

        let row = BenchResult {
            row_type: "doc",
            run_id: run_id.clone(),
            mode: stage.as_mode(),
            doc: rel.clone(),
            preset: self.config.preset.clone(),
            encode_ms,
            decode_ms,
            cer: cer_doc,
            wer: wer_doc,
            numguard_f1: num_stats.map(|n| n.f1),
            units_ok: num_stats.map(|n| n.units_ok),
            tokens_raw: stats.tokens_raw,
            tokens_3dcf: stats.tokens_3dcf,
            savings_ratio: stats.savings_ratio as f64,
            avg_cells_kept_per_page: avg_cells,
            pages: doc.total_pages(),
            budget,
            numguard_mismatches: numguard_alerts.len(),
            encode_pages_per_s: if encode_ms == 0 {
                0.0
            } else {
                pages_f / (encode_ms as f64 / 1000.0)
            },
            decode_pages_per_s: if decode_ms == 0 {
                0.0
            } else {
                pages_f / (decode_ms as f64 / 1000.0)
            },
            mem_peak_mb: mem_mb,
        };

        let page_rows = match gold {
            Some(gold_doc) => self.page_metrics(&run_id, &rel, doc, &gold_doc, budget)?,
            None => Vec::new(),
        };

        Ok((row, page_rows))
    }

    fn page_metrics(
        &self,
        run_id: &str,
        rel: &str,
        doc: &Document,
        gold: &GoldDoc,
        budget: Option<usize>,
    ) -> Result<Vec<BenchPageRow>> {
        let mut rows = Vec::new();
        for (idx, gold_page) in gold.pages.iter().enumerate() {
            let gold_text = match gold_page {
                Some(text) => text,
                None => continue,
            };
            let pred = doc.decode_page_to_text(idx as u32);
            let cer_page = cer(&pred, gold_text);
            let precision = (1.0 - cer_page).clamp(0.0, 1.0);
            let tokens_gold = self.tokenizer.encode_with_special_tokens(gold_text).len();
            let tokens_pred = self
                .tokenizer
                .encode_with_special_tokens(pred.as_str())
                .len();
            let compression = if tokens_pred == 0 {
                0.0
            } else {
                tokens_gold as f64 / tokens_pred as f64
            };
            rows.push(BenchPageRow {
                row_type: "page",
                run_id: run_id.to_string(),
                doc: rel.to_string(),
                preset: self.config.preset.clone(),
                page_idx: idx as u32,
                cer_page,
                precision_page: precision,
                tokens_gold_page: tokens_gold,
                tokens_3dcf_page: tokens_pred,
                compression_ratio: compression,
                budget,
            });
        }
        Ok(rows)
    }

    fn load_gold(&self, rel: &str, page_count: usize) -> Result<Option<GoldDoc>> {
        let root = match &self.config.gold_root {
            Some(path) => path,
            None => return Ok(None),
        };
        let rel_path = Path::new(rel);
        let mut doc_path = root.join(rel_path);
        doc_path.set_extension("txt");
        let doc_text = fs::read_to_string(&doc_path).ok();
        let mut base = doc_path.clone();
        base.set_extension("");
        let mut pages = Vec::with_capacity(page_count);
        for idx in 0..page_count {
            let page_path = base.join(format!("page_{idx:04}.txt"));
            pages.push(fs::read_to_string(&page_path).ok());
        }
        if doc_text.is_none() && pages.iter().all(|p| p.is_none()) {
            return Ok(None);
        }
        Ok(Some(GoldDoc {
            doc: doc_text,
            pages,
        }))
    }

    fn append_row(&self, row: &BenchResult) -> Result<()> {
        if let Some(out) = &self.config.output {
            append_json_line(out, row)?;
        }
        Ok(())
    }

    fn append_page_row(&self, row: &BenchPageRow) -> Result<()> {
        if let Some(out) = &self.config.output {
            append_json_line(out, row)?;
        }
        Ok(())
    }

    fn relative_path(&self, path: &Path) -> String {
        path.strip_prefix(&self.config.root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string()
    }

    fn run_id(&self, stage: BenchStage, budget: Option<usize>) -> String {
        match stage {
            BenchStage::Encode => match budget {
                Some(b) => format!("{}-{}", self.config.preset, b),
                None => format!("{}-auto", self.config.preset),
            },
            BenchStage::Decode => format!("{}-decode", self.config.preset),
        }
    }

    fn build_encoder(&self, budget: Option<usize>) -> Result<Encoder> {
        let mut builder = Encoder::builder(&self.config.preset)?;
        if let Some(b) = budget {
            builder = builder.budget(Some(b));
        }
        Ok(builder.build())
    }

    fn observe_memory_mb(&self) -> f64 {
        let mem = self.current_memory_mb();
        let mut peak = self.mem_peak_mb.borrow_mut();
        if mem > *peak {
            *peak = mem;
        }
        mem
    }

    fn current_memory_mb(&self) -> f64 {
        let mut sys = self.sys.borrow_mut();
        sys.refresh_process(self.pid);
        if let Some(proc) = sys.process(self.pid) {
            proc.memory() as f64 / 1024.0
        } else {
            0.0
        }
    }
}
struct GoldDoc {
    doc: Option<String>,
    pages: Vec<Option<String>>,
}

impl GoldDoc {
    fn joined_pages(&self) -> String {
        self.pages
            .iter()
            .filter_map(|p| p.as_ref())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn is_supported_source(path: &Path) -> bool {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        Some("pdf") | Some("txt") | Some("text") | Some("md") | Some("markdown") | Some("html")
        | Some("htm") | Some("json") | Some("tex") | Some("bib") => true,
        None => true,
        _ => false,
    }
}

fn append_json_line<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut file = File::options().append(true).create(true).open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted[idx]
}
