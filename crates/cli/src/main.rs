use hex;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use bincode;
use clap::{ArgAction, Args, Parser, Subcommand};
use glob::glob;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::thread::sleep;
use tracing_subscriber::EnvFilter;
use walkdir::WalkDir;

use reqwest::blocking::Client as HttpClient;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::StatusCode;

use three_dcf_core::{
    estimate_tokens, BenchConfig, BenchMode, BenchRunner, CellType, ChunkConfig, ChunkMode,
    ChunkRecord, Chunker, CorpusMetrics, Decoder, Document, EmbeddingRecord, Encoder, HashEmbedder,
    HashEmbedderConfig, HyphenationMode, ImportanceTuning, Metrics, NumGuardAlert, NumGuardIssue,
    Stats, TableMode, TextSerializer, TextSerializerConfig, TokenizerKind,
};
use three_dcf_rag::{
    encryption, execute_rag_query, normalize_level, CellInsert, DocumentInsert, EmbeddingClient,
    LlmClient, LlmProvider, LlmRequest, LlmResponse, PricingConfig, PricingRate, RagAnswer,
    RagPolicy, RagQuery, RagStore,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const VERSION_LONG: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (features: ",
    env!("THREE_DCF_FEATURES"),
    ")"
);

const DEFAULT_CONFIG: &str = "3dcf.toml";

#[derive(Parser, Debug)]
#[command(name = "3dcf", version = VERSION, long_version = VERSION_LONG, about = "3DCF-Lite prototype CLI")]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long = "rag-db", global = true, default_value = "rag.sqlite")]
    rag_db: PathBuf,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Encode {
        input: PathBuf,
        #[command(flatten)]
        encode: EncodeArgs,
        #[arg(long, default_value = "tokens.3dcf")]
        out: PathBuf,
        #[arg(long = "json-out")]
        json_out: Option<PathBuf>,
        #[arg(long = "text-out")]
        text_out: Option<PathBuf>,
        #[arg(long = "cells-out")]
        cells_out: Option<PathBuf>,
        #[arg(long, action = ArgAction::SetTrue)]
        quiet: bool,
    },
    Context {
        input: PathBuf,
        #[command(flatten)]
        encode: EncodeArgs,
        #[arg(short = 'o', long = "out")]
        out: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<String>,
        #[arg(long = "tokenizer-file")]
        tokenizer_file: Option<PathBuf>,
        #[arg(long, action = ArgAction::SetTrue)]
        quiet: bool,
    },
    RagInit,
    RagCreateCollection {
        name: String,
    },
    RagIndex {
        collection: String,
        inputs: Vec<PathBuf>,
        #[command(flatten)]
        encode: EncodeArgs,
        #[arg(long, default_value = "public")]
        sensitivity: String,
        #[arg(long, action = ArgAction::SetTrue)]
        encrypt: bool,
    },
    RagAsk {
        collection: String,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[command(flatten)]
        ask: AskOptions,
        #[arg(long, default_value = "public")]
        sensitivity_threshold: String,
        #[arg(long, default_value = "external")]
        policy: String,
        #[arg(long, default_value = "openai")]
        provider: String,
        #[arg(long)]
        model: Option<String>,
    },
    AskOpenai {
        input: PathBuf,
        #[command(flatten)]
        encode: EncodeArgs,
        #[command(flatten)]
        ask: AskOptions,
        #[arg(long, default_value = "gpt-4.1-mini")]
        model: String,
    },
    AskAnthropic {
        input: PathBuf,
        #[command(flatten)]
        encode: EncodeArgs,
        #[command(flatten)]
        ask: AskOptions,
        #[arg(long, default_value = "claude-3-5-sonnet")]
        model: String,
    },
    AskGemini {
        input: PathBuf,
        #[command(flatten)]
        encode: EncodeArgs,
        #[command(flatten)]
        ask: AskOptions,
        #[arg(long, default_value = "gemini-1.5-flash")]
        model: String,
    },
    AskDeepseek {
        input: PathBuf,
        #[command(flatten)]
        encode: EncodeArgs,
        #[command(flatten)]
        ask: AskOptions,
        #[arg(long, default_value = "deepseek-chat")]
        model: String,
    },
    Decode {
        input: PathBuf,
        #[arg(long = "text-out")]
        text_out: Option<PathBuf>,
        #[arg(long = "json-out")]
        json_out: Option<PathBuf>,
        #[arg(long = "page")]
        page: Option<u32>,
        #[arg(long = "select")]
        select: Option<String>,
        #[arg(long = "strict-numguard", action = ArgAction::SetTrue)]
        strict_numguard: bool,
        #[arg(long = "numguard-units")]
        numguard_units: Option<PathBuf>,
    },
    Serialize {
        input: PathBuf,
        out: PathBuf,
        #[arg(long, default_value_t = 64)]
        preview: usize,
        #[arg(long = "table-mode")]
        table_mode: Option<String>,
        #[arg(long = "preset-label")]
        preset_label: Option<String>,
        #[arg(long = "budget-label")]
        budget_label: Option<String>,
    },
    Stats {
        input: PathBuf,
        #[arg(long)]
        tokenizer: Option<String>,
        #[arg(long = "tokenizer-file")]
        tokenizer_file: Option<PathBuf>,
    },
    Bench {
        root: PathBuf,
        #[arg(long)]
        preset: Option<String>,
        #[arg(long)]
        tokenizer: Option<String>,
        #[arg(long = "tokenizer-file")]
        tokenizer_file: Option<PathBuf>,
        #[arg(long)]
        budget: Option<usize>,
        #[arg(long = "budgets")]
        budgets: Option<String>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long = "gold")]
        gold_root: Option<PathBuf>,
        #[arg(long, default_value = "encode")]
        mode: String,
        #[arg(long = "cer-threshold")]
        cer_threshold: Option<f64>,
        #[arg(long = "wer-threshold")]
        wer_threshold: Option<f64>,
        #[arg(long = "numguard-max")]
        numguard_max: Option<usize>,
        #[arg(long = "encode-p95-max")]
        encode_p95_max: Option<f64>,
        #[arg(long = "decode-p95-max")]
        decode_p95_max: Option<f64>,
    },
    Report {
        input: PathBuf,
        out: PathBuf,
    },
    Encrypt {
        input: PathBuf,
        out: PathBuf,
        #[arg(long)]
        recipient: String,
        #[arg(long = "redact", value_delimiter = ',')]
        redact_types: Vec<String>,
    },
    Decrypt {
        input: PathBuf,
        out: PathBuf,
        #[arg(long)]
        identity: PathBuf,
    },
    Synth {
        out_dir: PathBuf,
        #[arg(long, default_value_t = 5)]
        count: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    Chunk {
        input: PathBuf,
        out: PathBuf,
        #[arg(long, default_value_t = 200)]
        cells: usize,
        #[arg(long, default_value_t = 20)]
        overlap: usize,
        #[arg(long, default_value = "cells")]
        mode: String,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        #[arg(long, default_value_t = 64)]
        overlap_tokens: usize,
    },
    Embed {
        chunks: PathBuf,
        out: PathBuf,
        #[arg(long, default_value = "hash")]
        backend: String,
        #[arg(long, default_value_t = 64)]
        dimensions: usize,
        #[arg(long, default_value_t = 1337)]
        seed: u64,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        cache: Option<PathBuf>,
        #[arg(long, default_value_t = 4)]
        max_concurrency: usize,
        #[arg(long, default_value_t = 5)]
        retry_limit: usize,
        #[arg(long = "retry-base-ms", default_value_t = 500)]
        retry_base_ms: u64,
        #[arg(long = "openai-model")]
        openai_model: Option<String>,
        #[arg(long = "openai-api-key")]
        openai_api_key: Option<String>,
        #[arg(long = "openai-base-url", default_value = "https://api.openai.com/v1")]
        openai_base_url: String,
        #[arg(long = "cohere-model", default_value = "embed-multilingual-v3.0")]
        cohere_model: String,
        #[arg(long = "cohere-api-key")]
        cohere_api_key: Option<String>,
        #[arg(long = "cohere-base-url", default_value = "https://api.cohere.com/v1")]
        cohere_base_url: String,
    },
    Search {
        #[arg(long)]
        embeddings: Option<PathBuf>,
        #[arg(long)]
        index: Option<PathBuf>,
        query: String,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long)]
        filters: Option<String>,
        #[arg(long, action = ArgAction::SetTrue)]
        hybrid: bool,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long)]
        dimensions: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long = "openai-model")]
        openai_model: Option<String>,
        #[arg(long = "openai-api-key")]
        openai_api_key: Option<String>,
        #[arg(long = "openai-base-url", default_value = "https://api.openai.com/v1")]
        openai_base_url: String,
        #[arg(long = "cohere-model", default_value = "embed-multilingual-v3.0")]
        cohere_model: String,
        #[arg(long = "cohere-api-key")]
        cohere_api_key: Option<String>,
        #[arg(long = "cohere-base-url", default_value = "https://api.cohere.com/v1")]
        cohere_base_url: String,
    },
    Index {
        embeddings: PathBuf,
        out: PathBuf,
    },
    QdrantPush {
        #[arg(long)]
        embeddings: Option<PathBuf>,
        #[arg(long)]
        index: Option<PathBuf>,
        url: String,
        collection: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long, default_value_t = 64)]
        batch: usize,
        #[arg(long, action = ArgAction::SetTrue)]
        wait: bool,
    },
    QdrantSearch {
        #[arg(long)]
        embeddings: Option<PathBuf>,
        #[arg(long)]
        index: Option<PathBuf>,
        url: String,
        collection: String,
        query: String,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long)]
        filters: Option<String>,
        #[arg(long, action = ArgAction::SetTrue)]
        hybrid: bool,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long = "openai-model")]
        openai_model: Option<String>,
        #[arg(long = "openai-api-key")]
        openai_api_key: Option<String>,
        #[arg(long = "openai-base-url", default_value = "https://api.openai.com/v1")]
        openai_base_url: String,
        #[arg(long = "cohere-model", default_value = "embed-multilingual-v3.0")]
        cohere_model: String,
        #[arg(long = "cohere-api-key")]
        cohere_api_key: Option<String>,
        #[arg(long = "cohere-base-url", default_value = "https://api.cohere.com/v1")]
        cohere_base_url: String,
    },
}

#[derive(Args, Debug, Clone)]
struct EncodeArgs {
    #[arg(long)]
    preset: Option<String>,
    #[arg(long)]
    budget: Option<usize>,
    #[arg(long, action = ArgAction::SetTrue)]
    drop_footers: bool,
    #[arg(long)]
    dedup_window: Option<u32>,
    #[arg(long)]
    hyphenation: Option<String>,
    #[arg(long)]
    table_column_tolerance: Option<u32>,
    #[arg(long, action = ArgAction::SetTrue)]
    enable_ocr: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    force_ocr: bool,
    #[arg(long, value_delimiter = ',')]
    ocr_langs: Vec<String>,
    #[arg(long = "heading-boost")]
    heading_boost: Option<f32>,
    #[arg(long = "number-boost")]
    number_boost: Option<f32>,
    #[arg(long = "footer-penalty")]
    footer_penalty: Option<f32>,
    #[arg(long = "early-line-bonus")]
    early_line_bonus: Option<f32>,
    #[arg(long = "table-mode")]
    table_mode: Option<String>,
    #[arg(long = "preset-label")]
    preset_label: Option<String>,
    #[arg(long = "budget-label")]
    budget_label: Option<String>,
    #[arg(long = "strict-numguard", action = ArgAction::SetTrue)]
    strict_numguard: bool,
    #[arg(long = "numguard-units")]
    numguard_units: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
struct AskOptions {
    #[arg(long)]
    question: String,
    #[arg(long)]
    tokenizer: Option<String>,
    #[arg(long = "tokenizer-file")]
    tokenizer_file: Option<PathBuf>,
    #[arg(long, action = ArgAction::SetTrue)]
    quiet: bool,
}

fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let config_path = cli
        .config
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG));
    let file_config = load_config(&config_path)?;

    match cli.command {
        Commands::Encode {
            input,
            encode,
            out,
            json_out,
            text_out,
            cells_out,
            quiet,
        } => {
            let defaults = file_config.defaults.encode.as_ref();
            let resolved = resolve_encode_config(&encode, defaults)?;
            let encoder = build_encoder_from_resolved(&resolved)?;
            let (doc, metrics) = encoder.encode_path(&input)?;
            let unit_whitelist = load_unit_whitelist(resolved.numguard_units.clone())?;
            let alerts = doc.numguard_mismatches_with_units(unit_whitelist.as_ref());
            if !alerts.is_empty() {
                eprintln!(
                    "warning: detected {} numeric guard issues ({})",
                    alerts.len(),
                    summarize_numguard(&alerts)
                );
            }
            if resolved.strict_numguard && !alerts.is_empty() {
                return Err(anyhow!(format!(
                    "strict numguard mode failed ({} mismatches)",
                    alerts.len()
                )));
            }
            doc.save_bin(&out)?;
            if let Some(json) = json_out {
                doc.save_json(json)?;
            }
            if let Some(text) = text_out {
                let serializer = TextSerializer::with_config(TextSerializerConfig {
                    table_mode: resolved.table_mode,
                    preset_label: Some(resolved.preset_label.clone()),
                    budget_label: Some(resolved.budget_label.clone()),
                    ..Default::default()
                });
                serializer.write_textual(&doc, text)?;
            }
            if let Some(manifest) = cells_out {
                write_cell_manifest(&doc, &input, &manifest, 96)?;
            }
            if !quiet {
                print_encode_summary(&input, &metrics);
            }
        }
        Commands::RagInit => {
            let store = RagStore::open(&cli.rag_db)?;
            store.init()?;
            println!("[3DCF RAG] Initialized store at {}", cli.rag_db.display());
        }
        Commands::RagCreateCollection { name } => {
            let store = RagStore::open(&cli.rag_db)?;
            let id = store.ensure_collection(&name)?;
            println!("[3DCF RAG] Collection '{}' ready (id={})", name, id);
        }
        Commands::RagIndex {
            collection,
            inputs,
            encode,
            sensitivity,
            encrypt,
        } => {
            let files = gather_input_files(&inputs)?;
            if files.is_empty() {
                return Err(anyhow!("no files matched for indexing"));
            }
            let defaults = file_config.defaults.encode.as_ref();
            let resolved = resolve_encode_config(&encode, defaults)?;
            let store = RagStore::open(&cli.rag_db)?;
            let collection_id = store.ensure_collection(&collection)?;
            let embed_client = EmbeddingClient::from_env()?;
            let normalized_sensitivity = normalize_level(&sensitivity);
            let recipient = if encrypt {
                Some(env::var("RAG_ENCRYPTION_RECIPIENT").map_err(|_| {
                    anyhow!("RAG_ENCRYPTION_RECIPIENT must be set when --encrypt is used")
                })?)
            } else {
                None
            };
            let mut total_cells = 0usize;
            let mut total_docs = 0usize;
            for path in files {
                let stats = rag_index_file(
                    &store,
                    collection_id,
                    &path,
                    &resolved,
                    &embed_client,
                    &normalized_sensitivity,
                    recipient.as_deref(),
                )?;
                total_cells += stats.cells;
                total_docs += 1;
                println!(
                    "[3DCF RAG] Indexed {} cells ({} pages) from {}",
                    stats.cells,
                    stats.pages,
                    path.display()
                );
            }
            println!(
                "[3DCF RAG] Collection '{}' updated ({} documents, {} cells)",
                collection, total_docs, total_cells
            );
        }
        Commands::RagAsk {
            collection,
            top_k,
            ask,
            sensitivity_threshold,
            policy,
            provider,
            model,
        } => {
            let store = RagStore::open(&cli.rag_db)?;
            let embed_client = EmbeddingClient::from_env()?;
            let tokenizer_name = ask
                .tokenizer
                .clone()
                .or_else(|| file_config.defaults.stats_tokenizer())
                .unwrap_or_else(|| "cl100k_base".to_string());
            let tokenizer_file = ask
                .tokenizer_file
                .clone()
                .or_else(|| file_config.defaults.stats_tokenizer_file());
            let tokenizer_kind = resolve_tokenizer(&tokenizer_name, tokenizer_file)?;
            let policy_value = parse_rag_policy(&policy);
            let decrypt_identity = if matches!(policy_value, RagPolicy::Internal) {
                Some(
                    env::var("RAG_DECRYPT_IDENTITY")
                        .map(PathBuf::from)
                        .map_err(|_| {
                            anyhow!("RAG_DECRYPT_IDENTITY must be set for internal policy")
                        })?,
                )
            } else {
                None
            };
            let llm_client = build_llm_client(&provider, model)?;
            let rag_query = RagQuery {
                collection: collection.clone(),
                question: ask.question.clone(),
                top_k,
                sensitivity_threshold: normalize_level(&sensitivity_threshold),
                policy: policy_value,
                tokenizer: tokenizer_kind.clone(),
                tokenizer_name: tokenizer_name.clone(),
            };
            let answer = execute_rag_query(
                &store,
                &embed_client,
                &llm_client,
                &rag_query,
                decrypt_identity.as_deref(),
            )?;
            print_rag_answer(
                &collection,
                &answer,
                ask.quiet,
                &llm_client,
                &tokenizer_name,
                &file_config.pricing,
            );
        }
        Commands::Context {
            input,
            encode,
            out,
            tokenizer,
            tokenizer_file,
            quiet,
        } => {
            let defaults = file_config.defaults.encode.as_ref();
            let (artifacts, alerts) = build_context_artifacts(&input, &encode, defaults)?;
            let ContextArtifacts {
                resolved,
                mut metrics,
                raw_text,
                context_text,
            } = artifacts;
            if !alerts.is_empty() {
                eprintln!(
                    "warning: detected {} numeric guard issues ({})",
                    alerts.len(),
                    summarize_numguard(&alerts)
                );
            }
            if resolved.strict_numguard && !alerts.is_empty() {
                return Err(anyhow!(format!(
                    "strict numguard mode failed ({} mismatches)",
                    alerts.len()
                )));
            }
            let tokenizer_name = tokenizer
                .or_else(|| file_config.defaults.stats_tokenizer())
                .unwrap_or_else(|| "cl100k_base".to_string());
            let tokenizer_file =
                tokenizer_file.or_else(|| file_config.defaults.stats_tokenizer_file());
            let tokenizer_kind = resolve_tokenizer(&tokenizer_name, tokenizer_file)?;
            let raw_tokens = estimate_tokens(&raw_text, &tokenizer_kind)? as u32;
            let compressed_tokens = estimate_tokens(context_text.as_str(), &tokenizer_kind)? as u32;
            metrics.record_tokens(Some(raw_tokens), Some(compressed_tokens));
            if let Some(path) = &out {
                fs::write(path, context_text.as_bytes())
                    .with_context(|| format!("failed to write {}", path.display()))?;
            } else {
                println!("{}", context_text);
            }
            if !quiet {
                emit_context_metrics(out.as_deref(), &metrics, &tokenizer_name, out.is_none());
            }
        }
        Commands::AskOpenai {
            input,
            encode,
            ask,
            model,
        } => {
            let client = LlmClient::new(LlmProvider::OpenAi, model)?;
            run_ask_flow(&file_config, &input, &encode, &ask, client)?;
        }
        Commands::AskAnthropic {
            input,
            encode,
            ask,
            model,
        } => {
            let client = LlmClient::new(LlmProvider::Anthropic, model)?;
            run_ask_flow(&file_config, &input, &encode, &ask, client)?;
        }
        Commands::AskGemini {
            input,
            encode,
            ask,
            model,
        } => {
            let client = LlmClient::new(LlmProvider::Gemini, model)?;
            run_ask_flow(&file_config, &input, &encode, &ask, client)?;
        }
        Commands::AskDeepseek {
            input,
            encode,
            ask,
            model,
        } => {
            let client = LlmClient::new(LlmProvider::Deepseek, model)?;
            run_ask_flow(&file_config, &input, &encode, &ask, client)?;
        }
        Commands::Decode {
            input,
            text_out,
            json_out,
            page,
            select,
            strict_numguard,
            numguard_units,
        } => {
            let doc = load_document(&input)?;
            let defaults = file_config.defaults.encode.as_ref();
            let strict_numguard =
                strict_numguard || defaults.and_then(|d| d.strict_numguard).unwrap_or(false);
            let units_path =
                numguard_units.or_else(|| defaults.and_then(|d| d.numguard_units.clone()));
            let unit_whitelist = load_unit_whitelist(units_path)?;
            let selection = parse_selection(page, select.as_deref())?;
            let rendered = render_selection(&doc, &selection)?;
            if let Some(path) = text_out {
                fs::write(path, &rendered)?;
            } else {
                println!("{rendered}");
            }
            let alerts = doc.numguard_mismatches_with_units(unit_whitelist.as_ref());
            if !alerts.is_empty() {
                eprintln!(
                    "warning: detected {} numeric guard issues ({})",
                    alerts.len(),
                    summarize_numguard(&alerts)
                );
            }
            if strict_numguard && !alerts.is_empty() {
                return Err(anyhow!(format!(
                    "strict numguard mode failed ({} mismatches)",
                    alerts.len()
                )));
            }
            if let Some(json) = json_out {
                doc.save_json(json)?;
            }
        }
        Commands::Serialize {
            input,
            out,
            preview,
            table_mode,
            preset_label,
            budget_label,
        } => {
            let doc = load_document(&input)?;
            let mode = parse_table_mode(table_mode.as_deref())?;
            let serializer = TextSerializer::with_config(TextSerializerConfig {
                max_preview_chars: preview,
                table_mode: mode,
                preset_label,
                budget_label,
                ..Default::default()
            });
            serializer.write_textual(&doc, out)?;
        }
        Commands::Stats {
            input,
            tokenizer,
            tokenizer_file,
        } => {
            let doc = load_document(&input)?;
            let tokenizer = tokenizer
                .or_else(|| file_config.defaults.stats_tokenizer())
                .unwrap_or_else(|| "cl100k_base".to_string());
            let tokenizer_file =
                tokenizer_file.or_else(|| file_config.defaults.stats_tokenizer_file());
            let tokenizer = resolve_tokenizer(&tokenizer, tokenizer_file)?;
            let stats = Stats::measure(&doc, tokenizer)?;
            println!(
                "tokens_raw={} tokens_3dcf={} savings={:.2}x cells={}",
                stats.tokens_raw, stats.tokens_3dcf, stats.savings_ratio, stats.cells
            );
        }
        Commands::Bench {
            root,
            preset,
            tokenizer,
            tokenizer_file,
            budget,
            budgets,
            output,
            gold_root,
            mode: mode_arg,
            cer_threshold,
            wer_threshold,
            numguard_max,
            encode_p95_max,
            decode_p95_max,
        } => {
            let defaults = file_config.defaults.bench.as_ref();
            let preset = preset
                .or_else(|| defaults.and_then(|d| d.preset.clone()))
                .unwrap_or_else(|| "reports".to_string());
            let tokenizer = tokenizer
                .or_else(|| defaults.and_then(|d| d.tokenizer.clone()))
                .unwrap_or_else(|| "cl100k_base".to_string());
            let tokenizer_file =
                tokenizer_file.or_else(|| defaults.and_then(|d| d.tokenizer_file.clone()));
            let mode = Some(mode_arg)
                .or_else(|| defaults.and_then(|d| d.mode.clone()))
                .unwrap_or_else(|| "encode".to_string());
            let budgets = budgets.or_else(|| defaults.and_then(|d| d.budgets.clone()));
            let budget = budget.or_else(|| defaults.and_then(|d| d.budget));
            let tokenizer = resolve_tokenizer(&tokenizer, tokenizer_file)?;
            let bench_mode = parse_bench_mode(&mode)?;
            let budget_list = parse_budgets(budgets.as_deref(), budget)?;
            let config = BenchConfig {
                mode: bench_mode,
                root,
                gold_root,
                output,
                preset,
                tokenizer,
                budgets: budget_list,
            };
            let runner = BenchRunner::new(config)?;
            let metrics = runner.run()?;
            println!(
                "bench: files={} mean_savings={:.2}x median_savings={:.2}x enc_p50={:.0}ms enc_p95={:.0}ms max_mem={:.1}MB",
                metrics.results.len(),
                metrics.mean_savings,
                metrics.median_savings,
                metrics.encode_p50_ms,
                metrics.encode_p95_ms,
                metrics.max_mem_mb
            );
            enforce_bench_thresholds(
                &metrics,
                cer_threshold,
                wer_threshold,
                numguard_max,
                encode_p95_max,
                decode_p95_max,
            )?;
        }
        Commands::Report { input, out } => {
            write_report(&input, &out)?;
        }
        Commands::Encrypt {
            input,
            out,
            recipient,
            redact_types,
        } => {
            encrypt_file(&input, &out, &recipient, &redact_types)?;
        }
        Commands::Decrypt {
            input,
            out,
            identity,
        } => {
            decrypt_file(&input, &out, &identity)?;
        }
        Commands::Synth {
            out_dir,
            count,
            seed,
        } => {
            generate_synthetic(&out_dir, count, seed)?;
        }
        Commands::Chunk {
            input,
            out,
            cells,
            overlap,
            mode,
            max_tokens,
            overlap_tokens,
        } => {
            let doc = load_document(&input)?;
            let chunk_mode = parse_chunk_mode(&mode)?;
            let chunker = Chunker::new(ChunkConfig {
                mode: chunk_mode,
                cells_per_chunk: cells,
                overlap_cells: overlap,
                max_tokens,
                overlap_tokens,
            });
            let doc_id = input.to_string_lossy().to_string();
            let chunks = chunker.chunk_document(&doc, &doc_id);
            let mut writer = BufWriter::new(File::create(&out)?);
            let meta = ChunkMetadataRow {
                row_type: "chunk_meta".to_string(),
                doc: doc_id.clone(),
                cells_per_chunk: cells,
                overlap_cells: overlap,
                mode: chunk_mode_string(chunk_mode),
                max_tokens,
                overlap_tokens,
                chunk_version: 1,
                chunk_count: chunks.len(),
            };
            serde_json::to_writer(&mut writer, &meta)?;
            writer.write_all(b"\n")?;
            for chunk in &chunks {
                serde_json::to_writer(&mut writer, chunk)?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
            println!(
                "chunk: wrote {} chunks (mode={} cells={} overlap={} max_tokens={} overlap_tokens={}) to {}",
                chunks.len(),
                chunk_mode_string(chunk_mode),
                cells,
                overlap,
                max_tokens,
                overlap_tokens,
                out.display()
            );
        }
        Commands::Embed {
            chunks,
            out,
            backend,
            dimensions,
            seed,
            limit,
            cache,
            max_concurrency,
            retry_limit,
            retry_base_ms,
            openai_model,
            openai_api_key,
            openai_base_url,
            cohere_model,
            cohere_api_key,
            cohere_base_url,
        } => {
            let backend_handle = build_embed_backend(
                &backend,
                dimensions,
                seed,
                openai_model.as_deref(),
                openai_api_key.as_deref(),
                &openai_base_url,
                Some(cohere_model.as_str()),
                cohere_api_key.as_deref(),
                &cohere_base_url,
            )?;
            let namespace = format!(
                "{}:{}",
                backend_handle.name(),
                backend_handle.model().unwrap_or("default")
            );
            let retry_policy = RetryPolicy::new(retry_limit, Duration::from_millis(retry_base_ms));
            let mut cache = EmbedCache::new(cache.as_ref(), &namespace)?;
            let reader = BufReader::new(
                File::open(&chunks)
                    .with_context(|| format!("failed to open chunks {}", chunks.display()))?,
            );
            let mut records = Vec::new();
            let mut cached_records = Vec::new();
            let mut pending = Vec::new();
            for (idx, line) in reader.lines().enumerate() {
                let line = line?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(meta) = serde_json::from_str::<ChunkMetadataRow>(line) {
                    if meta.row_type == "chunk_meta" {
                        continue;
                    }
                }
                if let Some(max) = limit {
                    if cached_records.len() + pending.len() >= max {
                        break;
                    }
                }
                let chunk: ChunkRecord = serde_json::from_str(line).with_context(|| {
                    format!("invalid chunk row {} in {}", idx + 1, chunks.display())
                })?;
                let text_hash = hash_text(&chunk.text);
                if let Some(hit) = cache.get(&chunk.chunk_id, &text_hash) {
                    cached_records.push(hit);
                    continue;
                }
                pending.push((chunk, text_hash));
            }

            let max_workers = max_concurrency.max(1);
            let backend_clone = backend_handle.clone();
            let pending_results: Vec<EmbeddingRecord> = if pending.is_empty() {
                Vec::new()
            } else {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(max_workers)
                    .build()?;
                pool.install(|| -> Result<Vec<EmbeddingRecord>> {
                    pending
                        .par_iter()
                        .map(|(chunk, _text_hash)| {
                            embed_chunk_with_retry(&backend_clone, chunk, &retry_policy)
                        })
                        .collect()
                })?
            };

            for record in &pending_results {
                let text_hash = hash_text(&record.text);
                cache.insert(record.clone(), &text_hash);
            }

            records.extend(cached_records.into_iter());
            records.extend(pending_results.into_iter());
            records.sort_by_key(|r| (r.chunk_index, r.cell_start));
            cache.flush()?;
            let mut writer = BufWriter::new(
                File::create(&out)
                    .with_context(|| format!("failed to create embeddings {}", out.display()))?,
            );
            let resolved_dims = records
                .first()
                .map(|r| r.embedding.len())
                .or_else(|| backend_handle.dims_hint())
                .unwrap_or(dimensions);
            let meta = EmbeddingMetadataRow {
                row_type: "embed_meta".to_string(),
                chunk_source: chunks.to_string_lossy().to_string(),
                backend: backend_handle.name().to_string(),
                model: backend_handle.model().map(|s| s.to_string()),
                dimensions: resolved_dims,
                seed: backend_handle.seed(),
                vectors: records.len(),
                normalized: true,
            };
            serde_json::to_writer(&mut writer, &meta)?;
            writer.write_all(b"\n")?;
            for record in &records {
                serde_json::to_writer(&mut writer, record)?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
            println!(
                "embed: wrote {} vectors (dim={}) to {}",
                records.len(),
                resolved_dims,
                out.display()
            );
        }
        Commands::Search {
            embeddings,
            index,
            query,
            top_k,
            backend,
            dimensions,
            seed,
            filters,
            hybrid,
            openai_model,
            openai_api_key,
            openai_base_url,
            cohere_model,
            cohere_api_key,
            cohere_base_url,
        } => {
            let source = load_embeddings_source(embeddings.as_ref(), index.as_ref())?;
            let filter_predicate = filters.map(parse_filters).transpose()?;
            let backend_handle = build_backend_from_metadata(
                &source.metadata,
                backend.as_deref(),
                dimensions,
                seed,
                openai_model.as_deref(),
                openai_api_key.as_deref(),
                &openai_base_url,
                Some(cohere_model.as_str()),
                cohere_api_key.as_deref(),
                &cohere_base_url,
            )?;
            let mut query_vec = backend_handle.embed(&query, VectorKind::Query)?;
            normalize_vector(&mut query_vec);
            if source.records.is_empty() {
                println!("search: no candidates");
            } else {
                let hits = if hybrid {
                    search_hybrid(
                        &query,
                        &query_vec,
                        &source.records,
                        filter_predicate.as_ref(),
                    )
                } else {
                    search_dense_only(&query_vec, &source.records, filter_predicate.as_ref())
                };
                if hits.is_empty() {
                    println!("search: no candidates");
                } else {
                    for hit in hits.into_iter().take(top_k) {
                        println!(
                            "score={:.3} doc={} chunk={} {}",
                            hit.score, hit.doc, hit.chunk_index, hit.preview
                        );
                    }
                }
            }
        }
        Commands::Index { embeddings, out } => {
            let source = load_embeddings_from_jsonl(&embeddings)?;
            save_flat_index(&out, &source)?;
            println!(
                "index: wrote {} vectors to {}",
                source.records.len(),
                out.display()
            );
        }
        Commands::QdrantPush {
            embeddings,
            index,
            url,
            collection,
            api_key,
            batch,
            wait,
        } => {
            let source = load_embeddings_source(embeddings.as_ref(), index.as_ref())?;
            let client = QdrantClient::new(&url, api_key.as_deref())?;
            client.ensure_collection(&collection, source.metadata.dimensions)?;
            let chunk_size = batch.max(1);
            for chunk in source.records.chunks(chunk_size) {
                let mut points = Vec::with_capacity(chunk.len());
                for record in chunk {
                    let payload = json!({
                        "doc": record.doc,
                        "chunk_id": record.chunk_id,
                        "chunk_index": record.chunk_index,
                        "z_start": record.z_start,
                        "z_end": record.z_end,
                        "cell_start": record.cell_start,
                        "cell_end": record.cell_end,
                        "token_count": record.token_count,
                        "dominant_type": format!("{:?}", record.dominant_type),
                        "importance_mean": record.importance_mean,
                        "backend": source.metadata.backend,
                        "model": source.metadata.model,
                        "text": record.text,
                    });
                    points.push(QdrantPoint {
                        id: record.chunk_id.clone(),
                        vector: record.embedding.clone(),
                        payload,
                    });
                }
                client.upsert_points(&collection, points, wait)?;
            }
            println!(
                "qdrant push: uploaded {} vectors to {}",
                source.records.len(),
                collection
            );
        }
        Commands::QdrantSearch {
            embeddings,
            index,
            url,
            collection,
            query,
            top_k,
            filters,
            hybrid,
            api_key,
            openai_model,
            openai_api_key,
            openai_base_url,
            cohere_model,
            cohere_api_key,
            cohere_base_url,
        } => {
            let source = load_embeddings_source(embeddings.as_ref(), index.as_ref())?;
            let filter_predicate = filters.map(parse_filters).transpose()?;
            let backend_handle = build_backend_from_metadata(
                &source.metadata,
                None,
                None,
                None,
                openai_model.as_deref(),
                openai_api_key.as_deref(),
                &openai_base_url,
                Some(cohere_model.as_str()),
                cohere_api_key.as_deref(),
                &cohere_base_url,
            )?;
            let mut query_vec = backend_handle.embed(&query, VectorKind::Query)?;
            normalize_vector(&mut query_vec);
            let client = QdrantClient::new(&url, api_key.as_deref())?;
            let hits = client.search(
                &collection,
                &query,
                &query_vec,
                top_k,
                filter_predicate.as_ref(),
                hybrid,
                &source.records,
            )?;
            if hits.is_empty() {
                println!("qdrant search: no matches");
            } else {
                for hit in hits {
                    println!(
                        "score={:.3} doc={} chunk={} {}",
                        hit.score, hit.doc, hit.chunk_index, hit.preview
                    );
                }
            }
        }
    }
    Ok(())
}

fn load_document(path: &Path) -> Result<Document> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        Some("json") => Document::load_json(path).context("loading json document"),
        _ => Document::load_bin(path).context("loading binary document"),
    }
}

fn enforce_bench_thresholds(
    metrics: &CorpusMetrics,
    cer: Option<f64>,
    wer: Option<f64>,
    numguard_max: Option<usize>,
    encode_p95_max: Option<f64>,
    decode_p95_max: Option<f64>,
) -> Result<()> {
    let mut failures = Vec::new();
    if let Some(threshold) = cer {
        if let Some(worst) = metrics.results.iter().flat_map(|r| r.cer).reduce(f64::max) {
            if worst > threshold {
                failures.push(format!("CER {:.4} > threshold {:.4}", worst, threshold));
            }
        }
    }
    if let Some(threshold) = wer {
        if let Some(worst) = metrics.results.iter().flat_map(|r| r.wer).reduce(f64::max) {
            if worst > threshold {
                failures.push(format!("WER {:.4} > threshold {:.4}", worst, threshold));
            }
        }
    }
    if let Some(max_allowed) = numguard_max {
        let worst = metrics
            .results
            .iter()
            .map(|r| r.numguard_mismatches)
            .max()
            .unwrap_or(0);
        if worst > max_allowed {
            failures.push(format!(
                "numguard mismatches {} > threshold {}",
                worst, max_allowed
            ));
        }
    }
    if let Some(max_ms) = encode_p95_max {
        if metrics.encode_p95_ms > max_ms {
            failures.push(format!(
                "encode p95 {:.1}ms > threshold {:.1}ms",
                metrics.encode_p95_ms, max_ms
            ));
        }
    }
    if let Some(max_ms) = decode_p95_max {
        if metrics.decode_p95_ms > max_ms {
            failures.push(format!(
                "decode p95 {:.1}ms > threshold {:.1}ms",
                metrics.decode_p95_ms, max_ms
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(format!(
            "bench thresholds failed: {}",
            failures.join(", ")
        )))
    }
}

fn parse_filters(expr: String) -> Result<FilterPredicate> {
    let mut predicate = FilterPredicate {
        doc_ids: HashSet::new(),
        cell_types: HashSet::new(),
        min_importance: None,
    };
    for raw in expr.split(',') {
        let part = raw.trim();
        if part.is_empty() {
            continue;
        }
        if let Some(value) = part.strip_prefix("doc_id=") {
            predicate.doc_ids.insert(value.trim().to_string());
        } else if let Some(value) = part.strip_prefix("type=") {
            let cell = match value.trim().to_uppercase().as_str() {
                "TEXT" => CellType::Text,
                "TABLE" => CellType::Table,
                "FIGURE" => CellType::Figure,
                "FOOTER" => CellType::Footer,
                "HEADER" => CellType::Header,
                other => {
                    return Err(anyhow!(
                        "unknown cell type '{}' in --filters (valid: TEXT|TABLE|FIGURE|FOOTER|HEADER)",
                        other
                    ));
                }
            };
            predicate.cell_types.insert(cell);
        } else if let Some(value) = part.strip_prefix("min_importance=") {
            let raw_val: f32 = value.trim().parse()?;
            predicate.min_importance = Some((raw_val / 255.0).clamp(0.0, 1.0));
        } else {
            return Err(anyhow!(
                "unrecognized filter '{}'. Use doc_id=..., type=..., min_importance=...",
                part
            ));
        }
    }
    Ok(predicate)
}

impl FilterPredicate {
    fn matches(&self, record: &EmbeddingRecord) -> bool {
        if !self.doc_ids.is_empty() && !self.doc_ids.contains(&record.doc) {
            return false;
        }
        if !self.cell_types.is_empty() && !self.cell_types.contains(&record.dominant_type) {
            return false;
        }
        if let Some(min_imp) = self.min_importance {
            if record.importance_mean < min_imp {
                return false;
            }
        }
        true
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

#[derive(Debug)]
enum DecodeSelection {
    All,
    Page(u32),
    Bbox {
        z: u32,
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
    },
}

fn parse_selection(page: Option<u32>, select: Option<&str>) -> Result<DecodeSelection> {
    if let Some(expr) = select {
        if page.is_some() {
            return Err(anyhow!("cannot combine --page and --select"));
        }
        return parse_bbox(expr);
    }
    if let Some(z) = page {
        Ok(DecodeSelection::Page(z))
    } else {
        Ok(DecodeSelection::All)
    }
}

fn parse_chunk_mode(value: &str) -> Result<ChunkMode> {
    match value.to_lowercase().as_str() {
        "cells" => Ok(ChunkMode::Cells),
        "tokens" => Ok(ChunkMode::Tokens),
        "headings" => Ok(ChunkMode::Headings),
        "table-rows" | "tablerows" | "tables" => Ok(ChunkMode::TableRows),
        other => Err(anyhow!(
            "unknown chunk mode '{}'. choose cells|tokens|headings|table-rows",
            other
        )),
    }
}

fn chunk_mode_string(mode: ChunkMode) -> String {
    match mode {
        ChunkMode::Cells => "cells",
        ChunkMode::Tokens => "tokens",
        ChunkMode::Headings => "headings",
        ChunkMode::TableRows => "table_rows",
    }
    .to_string()
}

fn default_chunk_mode_str() -> String {
    "cells".to_string()
}

fn parse_bbox(expr: &str) -> Result<DecodeSelection> {
    let mut z = None;
    let mut x = None;
    let mut y = None;
    for part in expr.split(',') {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| anyhow!("invalid selector segment: {part}"))?;
        match key.trim().to_lowercase().as_str() {
            "z" => {
                z = Some(
                    value
                        .trim()
                        .parse::<u32>()
                        .map_err(|_| anyhow!("invalid z"))?,
                );
            }
            "x" => x = Some(parse_range(value.trim())?),
            "y" => y = Some(parse_range(value.trim())?),
            other => return Err(anyhow!("unknown selector key {other}")),
        }
    }
    let z = z.ok_or_else(|| anyhow!("selector requires z=<page>"))?;
    let (x0, x1) = x.ok_or_else(|| anyhow!("selector requires x=<start..end>"))?;
    let (y0, y1) = y.ok_or_else(|| anyhow!("selector requires y=<start..end>"))?;
    Ok(DecodeSelection::Bbox { z, x0, y0, x1, y1 })
}

fn parse_range(value: &str) -> Result<(i32, i32)> {
    if let Some((start, end)) = value.split_once("..") {
        let a = start
            .trim()
            .parse::<i32>()
            .map_err(|_| anyhow!("invalid range"))?;
        let b = end
            .trim()
            .parse::<i32>()
            .map_err(|_| anyhow!("invalid range"))?;
        Ok((a, b))
    } else {
        let v = value
            .trim()
            .parse::<i32>()
            .map_err(|_| anyhow!("invalid value"))?;
        Ok((v, v))
    }
}

fn render_selection(doc: &Document, selection: &DecodeSelection) -> Result<String> {
    let decoder = Decoder::new();
    match selection {
        DecodeSelection::All => decoder.to_text(doc).map_err(|e| anyhow!(e.to_string())),
        DecodeSelection::Page(z) => decoder
            .page_to_text(doc, *z)
            .map_err(|e| anyhow!(e.to_string())),
        DecodeSelection::Bbox { z, x0, y0, x1, y1 } => decoder
            .bbox_to_text(doc, *z, *x0, *y0, *x1, *y1)
            .map_err(|e| anyhow!(e.to_string())),
    }
}

fn load_config(path: &Path) -> Result<AppConfig> {
    if !path.exists() {
        return Ok(AppConfig::default());
    }
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read config {}", path.display()))?;
    toml::from_str(&contents).map_err(|e| anyhow!("invalid config: {e}"))
}

fn hyphenation_from_str(value: &str) -> HyphenationMode {
    match value.to_lowercase().as_str() {
        "preserve" => HyphenationMode::Preserve,
        _ => HyphenationMode::Merge,
    }
}

fn parse_bench_mode(value: &str) -> Result<BenchMode> {
    match value.to_lowercase().as_str() {
        "encode" => Ok(BenchMode::Encode),
        "decode" => Ok(BenchMode::Decode),
        "full" => Ok(BenchMode::Full),
        other => Err(anyhow!(format!("unknown bench mode {other}"))),
    }
}

fn parse_budgets(spec: Option<&str>, single: Option<usize>) -> Result<Vec<Option<usize>>> {
    if let Some(raw) = spec {
        let mut out = Vec::new();
        for part in raw.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
                out.push(None);
            } else {
                let value = trimmed
                    .parse::<usize>()
                    .map_err(|_| anyhow!("invalid budget value {trimmed}"))?;
                out.push(Some(value));
            }
        }
        if out.is_empty() {
            out.push(None);
        }
        return Ok(out);
    }
    if let Some(b) = single {
        Ok(vec![Some(b)])
    } else {
        Ok(vec![None])
    }
}

fn resolve_tokenizer(name: &str, file: Option<PathBuf>) -> Result<TokenizerKind> {
    match name.to_lowercase().as_str() {
        "cl100k" | "cl100k_base" => Ok(TokenizerKind::Cl100k),
        "o200k" | "o200k_base" => Ok(TokenizerKind::O200k),
        "gpt2" | "p50k" | "p50k_base" => Ok(TokenizerKind::Gpt2),
        "anthropic" => Ok(TokenizerKind::Anthropic),
        "custom" => {
            let path =
                file.ok_or_else(|| anyhow!("--tokenizer-file is required when tokenizer=custom"))?;
            Ok(TokenizerKind::Custom(path))
        }
        other => Err(anyhow!(format!("unknown tokenizer {other}"))),
    }
}

fn parse_table_mode(value: Option<&str>) -> Result<TableMode> {
    let mode = value.unwrap_or("auto").to_lowercase();
    match mode.as_str() {
        "auto" => Ok(TableMode::Auto),
        "csv" => Ok(TableMode::Csv),
        "dims" => Ok(TableMode::Dims),
        other => Err(anyhow!(format!("unknown table mode {other}"))),
    }
}

fn load_unit_whitelist(path: Option<PathBuf>) -> Result<Option<HashSet<String>>> {
    if let Some(p) = path {
        let data = fs::read_to_string(&p)
            .with_context(|| format!("failed to read whitelist {}", p.display()))?;
        let set = data
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| line.to_lowercase())
            .collect::<HashSet<_>>();
        return Ok(Some(set));
    }
    Ok(None)
}

fn summarize_numguard(alerts: &[NumGuardAlert]) -> String {
    alerts
        .iter()
        .take(3)
        .map(|alert| {
            let issue = match alert.issue {
                NumGuardIssue::MissingCell => "missing_cell",
                NumGuardIssue::MissingPayload => "missing_payload",
                NumGuardIssue::HashMismatch => "hash_mismatch",
                NumGuardIssue::UnitNotAllowed => "unit_not_allowed",
            };
            format!(
                "{}@z{}:{}",
                issue,
                alert.guard.z,
                alert.guard.units.as_str()
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn gather_input_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        let input_str = input.to_string_lossy();
        if input_str.contains('*') || input_str.contains('?') || input_str.contains('[') {
            for entry in glob(&input_str)? {
                let path = entry?;
                if path.is_file() {
                    files.push(path);
                }
            }
            continue;
        }
        let meta = fs::metadata(input)
            .with_context(|| format!("failed to inspect {}", input.display()))?;
        if meta.is_dir() {
            for entry in WalkDir::new(input)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                files.push(entry.path().to_path_buf());
            }
        } else if meta.is_file() {
            files.push(input.clone());
        }
    }
    Ok(files)
}

struct RagIndexStats {
    cells: usize,
    pages: usize,
}

fn rag_index_file(
    store: &RagStore,
    collection_id: i64,
    path: &Path,
    resolved: &ResolvedEncodeConfig,
    embed_client: &EmbeddingClient,
    sensitivity: &str,
    recipient: Option<&str>,
) -> Result<RagIndexStats> {
    let encoder = build_encoder_from_resolved(resolved)?;
    let (doc, _) = encoder.encode_path(path)?;
    let doc_record = store.add_document(
        collection_id,
        &DocumentInsert {
            source_path: path.to_string_lossy().to_string(),
            dcf_path: None,
            title: None,
        },
    )?;
    let ordered = doc.ordered_cells();
    if ordered.is_empty() {
        return Ok(RagIndexStats {
            cells: 0,
            pages: doc.total_pages(),
        });
    }
    let mut texts = Vec::with_capacity(ordered.len());
    for cell in &ordered {
        texts.push(doc.payload_for(&cell.code_id).unwrap_or("").to_string());
    }
    let embeddings = embed_client.embed_batch(&texts)?;
    if embeddings.len() != ordered.len() {
        return Err(anyhow!(
            "embedding backend returned mismatched vector count"
        ));
    }
    let mut inserts = Vec::with_capacity(ordered.len());
    for ((cell, text), embedding) in ordered.iter().zip(texts.iter()).zip(embeddings.into_iter()) {
        let (plain_text, encrypted, encryption_label) = if let Some(recipient) = recipient {
            if text.is_empty() {
                (None, None, None)
            } else {
                let cipher = encryption::encrypt_text(text, recipient)?;
                (None, Some(cipher), Some("age".to_string()))
            }
        } else {
            (
                if text.is_empty() {
                    None
                } else {
                    Some(text.clone())
                },
                None,
                None,
            )
        };
        inserts.push(CellInsert {
            page: cell.z,
            importance: cell.importance,
            sensitivity: sensitivity.to_string(),
            text: plain_text,
            text_encrypted: encrypted,
            encryption: encryption_label,
            embedding,
            bbox_x: cell.x,
            bbox_y: cell.y,
            bbox_w: cell.w,
            bbox_h: cell.h,
        });
    }
    store.add_cells(doc_record.id, &inserts)?;
    Ok(RagIndexStats {
        cells: inserts.len(),
        pages: doc.total_pages(),
    })
}

fn build_llm_client(name: &str, model: Option<String>) -> Result<LlmClient> {
    let provider =
        LlmProvider::from_str(name).ok_or_else(|| anyhow!(format!("unknown provider {name}")))?;
    let model_name = model.unwrap_or_else(|| default_llm_model(provider));
    LlmClient::new(provider, model_name)
}

fn default_llm_model(provider: LlmProvider) -> String {
    match provider {
        LlmProvider::OpenAi => "gpt-4.1-mini",
        LlmProvider::Anthropic => "claude-3-5-sonnet",
        LlmProvider::Gemini => "gemini-1.5-flash",
        LlmProvider::Deepseek => "deepseek-chat",
        LlmProvider::Local => "local",
    }
    .to_string()
}

fn parse_rag_policy(value: &str) -> RagPolicy {
    match value.to_lowercase().as_str() {
        "internal" => RagPolicy::Internal,
        _ => RagPolicy::External,
    }
}

fn print_rag_answer(
    collection: &str,
    answer: &RagAnswer,
    quiet: bool,
    llm: &LlmClient,
    tokenizer: &str,
    pricing: &PricingConfig,
) {
    let (provider_name, provider_key) = provider_labels(llm.provider());
    if !quiet {
        let doc_count = answer
            .used_cells
            .iter()
            .map(|cell| cell.document_id)
            .collect::<HashSet<_>>()
            .len();
        println!("[3DCF RAG] Collection:          {}", collection);
        println!(
            "[3DCF RAG] Retrieved cells:     {} (from {} documents)",
            answer.used_cells.len(),
            doc_count
        );
        println!(
            "[3DCF RAG] LLM:                 {} {}",
            provider_name,
            llm.model()
        );
        println!(
            "[3DCF RAG] Prompt tokens:       {}",
            format_number(answer.response.prompt_tokens as u64)
        );
        println!(
            "[3DCF RAG] Completion tokens:   {}",
            format_number(answer.response.completion_tokens as u64)
        );
        println!(
            "[3DCF RAG] Compression factor:  {}",
            format_ratio(answer.metrics.compression_factor)
        );
        if let Some(rate) = pricing.lookup(provider_key, llm.model()) {
            let prompt_cost = answer.response.prompt_tokens as f64 / 1000.0 * rate.prompt_per_1k;
            let completion_cost =
                answer.response.completion_tokens as f64 / 1000.0 * rate.completion_per_1k;
            let raw_prompt = answer.metrics.raw_tokens_estimate as f64;
            let raw_cost = raw_prompt / 1000.0 * rate.prompt_per_1k;
            let saving = (raw_cost - prompt_cost).max(0.0);
            println!(
                "[3DCF RAG] Cost (3DCF prompt): {}",
                format_currency(prompt_cost)
            );
            println!(
                "[3DCF RAG] Cost (raw prompt):  {}",
                format_currency(raw_cost)
            );
            if raw_cost > 0.0 {
                println!(
                    "[3DCF RAG] Est. saving:        {} (~{:.0}%)",
                    format_currency(saving),
                    saving / raw_cost * 100.0
                );
            }
            println!(
                "[3DCF RAG] Total cost:         {}",
                format_currency(prompt_cost + completion_cost)
            );
        }
        println!("[3DCF RAG] Tokenizer:          {}", tokenizer);
        println!("\nSources:");
        for (idx, cell) in answer.used_cells.iter().enumerate() {
            println!(
                "[Source {}] {}, page {}, score={:.2}",
                idx + 1,
                cell.document_source,
                cell.page,
                cell.score
            );
            println!("{}\n", cell.text.trim());
        }
    }
    println!("\n{}", answer.answer.trim());
}

fn provider_labels(provider: LlmProvider) -> (&'static str, &'static str) {
    match provider {
        LlmProvider::OpenAi => ("OpenAI", "openai"),
        LlmProvider::Anthropic => ("Anthropic", "anthropic"),
        LlmProvider::Gemini => ("Gemini", "gemini"),
        LlmProvider::Deepseek => ("DeepSeek", "deepseek"),
        LlmProvider::Local => ("Local", "local"),
    }
}

fn print_encode_summary(input: &Path, metrics: &Metrics) {
    println!("[3DCF] Parsed:          {}", input.display());
    println!(
        "[3DCF] Pages:           {}",
        format_number(metrics.pages as u64)
    );
    println!(
        "[3DCF] Lines:           {}",
        format_number(metrics.lines_total as u64)
    );
    println!(
        "[3DCF] Cells total:     {}",
        format_number(metrics.cells_total as u64)
    );
    let removal = removal_note(metrics.cells_total, metrics.cells_kept);
    println!(
        "[3DCF] Cells kept:      {}{}",
        format_number(metrics.cells_kept as u64),
        removal
    );
    println!(
        "[3DCF] Dedup ratio:     {}",
        format_ratio(metrics.dedup_ratio)
    );
    println!(
        "[3DCF] NumGuard fields: {}",
        format_number(metrics.numguard_count as u64)
    );
}

fn removal_note(total: u32, kept: u32) -> String {
    if total == 0 || kept >= total {
        String::new()
    } else {
        let removed = total.saturating_sub(kept);
        let pct = removed as f32 / total as f32 * 100.0;
        format!(" ({:.0}% removed as low importance/duplicates)", pct)
    }
}

fn format_number(value: u64) -> String {
    let s = value.to_string();
    let mut out = String::new();
    for (idx, ch) in s.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            out.push(' ');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn format_ratio(value: f32) -> String {
    if value <= 0.0 {
        "0.0x".to_string()
    } else if value >= 100.0 {
        format!("{:.0}x", value)
    } else if value >= 10.0 {
        format!("{:.1}x", value)
    } else {
        format!("{:.2}x", value)
    }
}

fn format_currency(value: f64) -> String {
    if value >= 1.0 {
        format!("~${:.2}", value)
    } else if value >= 0.01 {
        format!("~${:.3}", value)
    } else {
        format!("~${:.4}", value)
    }
}

fn emit_context_metrics(
    out_path: Option<&Path>,
    metrics: &Metrics,
    tokenizer: &str,
    log_to_stderr: bool,
) {
    let log_line = |line: String| {
        if log_to_stderr {
            eprintln!("{}", line);
        } else {
            println!("{}", line);
        }
    };
    if let Some(path) = out_path {
        log_line(format!("[3DCF] Context created: {}", path.display()));
    } else {
        log_line("[3DCF] Context emitted to stdout".to_string());
    }
    log_line(format!(
        "[3DCF] Pages:                   {}",
        format_number(metrics.pages as u64)
    ));
    log_line(format!(
        "[3DCF] Cells total / kept:      {} / {}",
        format_number(metrics.cells_total as u64),
        format_number(metrics.cells_kept as u64)
    ));
    log_line(format!(
        "[3DCF] Dedup ratio:             {}",
        format_ratio(metrics.dedup_ratio)
    ));
    log_line(format!(
        "[3DCF] NumGuard fields:         {}",
        format_number(metrics.numguard_count as u64)
    ));
    if let Some(raw) = metrics.raw_tokens_estimate {
        log_line(format!(
            "[3DCF] Estimated raw tokens:    {}",
            format_number(raw as u64)
        ));
    }
    if let Some(compressed) = metrics.compressed_tokens_estimate {
        log_line(format!(
            "[3DCF] Compressed tokens:       {}",
            format_number(compressed as u64)
        ));
    }
    if let Some(factor) = metrics.compression_factor {
        log_line(format!(
            "[3DCF] Compression factor:      {}",
            format_ratio(factor)
        ));
    }
    log_line(format!("[3DCF] Tokenizer:               {}", tokenizer));
}

fn run_ask_flow(
    config: &AppConfig,
    input: &Path,
    encode: &EncodeArgs,
    ask: &AskOptions,
    client: LlmClient,
) -> Result<()> {
    let defaults = config.defaults.encode.as_ref();
    let (artifacts, alerts) = build_context_artifacts(input, encode, defaults)?;
    let ContextArtifacts {
        resolved,
        mut metrics,
        raw_text,
        context_text,
    } = artifacts;
    if !alerts.is_empty() {
        eprintln!(
            "warning: detected {} numeric guard issues ({})",
            alerts.len(),
            summarize_numguard(&alerts)
        );
    }
    if resolved.strict_numguard && !alerts.is_empty() {
        return Err(anyhow!(format!(
            "strict numguard mode failed ({} mismatches)",
            alerts.len()
        )));
    }
    let tokenizer_name = ask
        .tokenizer
        .clone()
        .or_else(|| config.defaults.stats_tokenizer())
        .unwrap_or_else(|| "cl100k_base".to_string());
    let tokenizer_file = ask
        .tokenizer_file
        .clone()
        .or_else(|| config.defaults.stats_tokenizer_file());
    let tokenizer_kind = resolve_tokenizer(&tokenizer_name, tokenizer_file)?;
    let raw_tokens = estimate_tokens(&raw_text, &tokenizer_kind)? as u32;
    let compressed_tokens = estimate_tokens(context_text.as_str(), &tokenizer_kind)? as u32;
    let question_tokens = estimate_tokens(ask.question.as_str(), &tokenizer_kind)? as u32;
    metrics.record_tokens(
        Some(raw_tokens.saturating_add(question_tokens)),
        Some(compressed_tokens.saturating_add(question_tokens)),
    );
    let prompt = format!(
        "{}\n\nQuestion: {}\n\nAnswer:",
        context_text.trim(),
        ask.question.trim()
    );
    let response = client.chat_blocking(&LlmRequest {
        system: None,
        user: prompt,
    })?;
    if !ask.quiet {
        let (_, provider_key) = provider_labels(client.provider());
        let pricing = config.pricing.lookup(provider_key, client.model());
        print_ask_metrics(&client, &metrics, &tokenizer_name, &response, pricing);
    }
    println!("\n{}", response.content.trim());
    Ok(())
}

fn print_ask_metrics(
    client: &LlmClient,
    metrics: &Metrics,
    tokenizer: &str,
    response: &LlmResponse,
    pricing: Option<PricingRate>,
) {
    let (provider_name, provider_key) = provider_labels(client.provider());
    println!(
        "[3DCF] LLM:                     {} {}",
        provider_name,
        client.model()
    );
    println!(
        "[3DCF] Prompt tokens (3DCF):    {}",
        format_number(response.prompt_tokens as u64)
    );
    println!(
        "[3DCF] Completion tokens:       {}",
        format_number(response.completion_tokens as u64)
    );
    println!(
        "[3DCF] Total tokens:            {}",
        format_number(response.total_tokens() as u64)
    );
    if let Some(raw) = metrics.raw_tokens_estimate {
        println!(
            "[3DCF] Est. raw prompt tokens: {}",
            format_number(raw as u64)
        );
    }
    if let Some(factor) = metrics.compression_factor {
        println!("[3DCF] Compression factor:      {}", format_ratio(factor));
    }
    if let Some(rate) = pricing {
        let prompt_cost = response.prompt_tokens as f64 / 1000.0 * rate.prompt_per_1k;
        let completion_cost = response.completion_tokens as f64 / 1000.0 * rate.completion_per_1k;
        let raw_prompt = metrics.raw_tokens_estimate.unwrap_or(0) as f64;
        let raw_cost = raw_prompt / 1000.0 * rate.prompt_per_1k;
        let saving = (raw_cost - prompt_cost).max(0.0);
        let saving_pct = if raw_cost > 0.0 {
            saving / raw_cost * 100.0
        } else {
            0.0
        };
        println!(
            "[3DCF] Cost (3DCF prompt):      {}",
            format_currency(prompt_cost)
        );
        println!(
            "[3DCF] Cost (raw prompt):       {}",
            format_currency(raw_cost)
        );
        println!(
            "[3DCF] Est. saving per query:   {} (~{:.0}%)",
            format_currency(saving),
            saving_pct
        );
        println!(
            "[3DCF] Total cost (with cmp.):  {}",
            format_currency(prompt_cost + completion_cost)
        );
    } else {
        println!(
            "[3DCF] Pricing:                 configure pricing.{} in 3dcf.toml",
            provider_key
        );
    }
    println!(
        "[3DCF] NumGuard fields:         {}",
        format_number(metrics.numguard_count as u64)
    );
    println!("[3DCF] Tokenizer:               {}", tokenizer);
}

fn encrypt_file(input: &Path, out: &Path, recipient: &str, redact_types: &[String]) -> Result<()> {
    let mut data =
        fs::read(input).with_context(|| format!("failed to read {}", input.display()))?;
    if !redact_types.is_empty() {
        let doc = load_document(input).context("redacting cells")?;
        let rows = redact_cells(&doc, redact_types, input);
        data = serde_json::to_vec(&rows)?;
    }
    let recipient = age::x25519::Recipient::from_str(recipient.trim())
        .map_err(|e| anyhow!(format!("invalid recipient: {e}")))?;
    let mut out_file = std::fs::File::create(out)
        .with_context(|| format!("failed to create {}", out.display()))?;
    let encryptor = age::Encryptor::with_recipients(vec![Box::new(recipient)])
        .ok_or_else(|| anyhow!("no recipients provided"))?;
    let mut writer = encryptor.wrap_output(&mut out_file)?;
    writer.write_all(&data)?;
    writer.finish()?;
    Ok(())
}

fn decrypt_file(input: &Path, out: &Path, identity_path: &Path) -> Result<()> {
    let data = fs::read(input).with_context(|| format!("failed to read {}", input.display()))?;
    let identity_content = fs::read_to_string(identity_path)
        .with_context(|| format!("failed to read identity {}", identity_path.display()))?;
    let identity_line = identity_content
        .lines()
        .map(|l| l.trim())
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .ok_or_else(|| anyhow!("identity file is empty"))?;
    let identity = age::x25519::Identity::from_str(identity_line)
        .map_err(|e| anyhow!(format!("invalid identity: {e}")))?;
    let decryptor = age::Decryptor::new(&data[..])?;
    let mut out_file = std::fs::File::create(out)
        .with_context(|| format!("failed to create {}", out.display()))?;
    match decryptor {
        age::Decryptor::Recipients(d) => {
            let identities: Vec<Box<dyn age::Identity>> = vec![Box::new(identity)];
            let mut reader = d.decrypt(identities.iter().map(|id| id.as_ref()))?;
            io::copy(&mut reader, &mut out_file)?;
        }
        _ => return Err(anyhow!("unsupported passphrase-encrypted payload")),
    }
    Ok(())
}

fn redact_cells(doc: &Document, types: &[String], source: &Path) -> Vec<RedactedCell> {
    let allow = types
        .iter()
        .map(|t| t.trim().to_uppercase())
        .collect::<std::collections::HashSet<_>>();
    let doc_path = source.to_string_lossy().to_string();
    doc.ordered_cells()
        .into_iter()
        .enumerate()
        .map(|(idx, cell)| {
            let ctype = format!("{:?}", cell.cell_type).to_uppercase();
            let keep = allow.is_empty() || allow.contains(&ctype);
            let payload = doc.payload_for(&cell.code_id).unwrap_or("<missing>");
            RedactedCell {
                doc: doc_path.clone(),
                cell_index: idx,
                code_hash: hex::encode(cell.code_id),
                z: cell.z,
                x: cell.x,
                y: cell.y,
                w: cell.w,
                h: cell.h,
                importance: cell.importance,
                cell_type: ctype,
                text: if keep {
                    payload.to_string()
                } else {
                    "<redacted>".to_string()
                },
                preview: short_preview(payload, 96),
            }
        })
        .collect()
}

fn write_cell_manifest(
    doc: &Document,
    source: &Path,
    out: &Path,
    preview_limit: usize,
) -> Result<()> {
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(out)
        .with_context(|| format!("failed to open {}", out.display()))?;
    let doc_path = source.to_string_lossy().to_string();
    for (index, cell) in doc.ordered_cells().into_iter().enumerate() {
        let payload = doc.payload_for(&cell.code_id).unwrap_or("");
        let row = CellManifestRow {
            doc: doc_path.clone(),
            cell_index: index,
            code_hash: hex::encode(cell.code_id),
            z: cell.z,
            x: cell.x,
            y: cell.y,
            w: cell.w,
            h: cell.h,
            importance: cell.importance,
            cell_type: format!("{:?}", cell.cell_type),
            text: payload.to_string(),
            preview: short_preview(payload, preview_limit),
        };
        serde_json::to_writer(&mut file, &row)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn short_preview(payload: &str, limit: usize) -> String {
    if payload.len() <= limit {
        payload.to_string()
    } else {
        let mut s = payload.chars().take(limit).collect::<String>();
        s.push_str("...");
        s
    }
}

#[derive(Debug, Serialize)]
struct SearchHit {
    doc: String,
    chunk_id: String,
    chunk_index: usize,
    score: f32,
    preview: String,
}

#[derive(Debug, Clone)]
struct FilterPredicate {
    doc_ids: HashSet<String>,
    cell_types: HashSet<CellType>,
    min_importance: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChunkMetadataRow {
    row_type: String,
    doc: String,
    #[serde(default)]
    cells_per_chunk: usize,
    #[serde(default)]
    overlap_cells: usize,
    #[serde(default = "default_chunk_mode_str")]
    mode: String,
    #[serde(default)]
    max_tokens: usize,
    #[serde(default)]
    overlap_tokens: usize,
    #[serde(default)]
    chunk_version: u32,
    #[serde(default)]
    chunk_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EmbeddingMetadataRow {
    row_type: String,
    chunk_source: String,
    backend: String,
    model: Option<String>,
    dimensions: usize,
    seed: u64,
    vectors: usize,
    #[serde(default)]
    normalized: bool,
}

#[derive(Clone, Copy)]
enum VectorKind {
    Document,
    Query,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let len = a.len().min(b.len());
    let dot = a
        .iter()
        .zip(b.iter())
        .take(len)
        .map(|(x, y)| x * y)
        .sum::<f32>();
    dot
}

fn normalize_vector(vec: &mut [f32]) {
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return;
    }
    for value in vec.iter_mut() {
        *value /= norm;
    }
}

#[derive(Clone, Copy)]
struct RetryPolicy {
    limit: usize,
    base_delay_ms: u64,
}

impl RetryPolicy {
    fn new(limit: usize, base_delay: Duration) -> Self {
        let base = base_delay.as_millis().max(1) as u64;
        Self {
            limit: limit.max(1),
            base_delay_ms: base,
        }
    }

    fn delay_for_attempt(&self, attempt: usize) -> Duration {
        let exponent = attempt.saturating_sub(1).min(16);
        let backoff = self.base_delay_ms.saturating_mul(1u64 << exponent);
        Duration::from_millis(backoff.max(self.base_delay_ms))
    }
}

fn embed_chunk_with_retry(
    backend: &BackendHandle,
    chunk: &ChunkRecord,
    policy: &RetryPolicy,
) -> Result<EmbeddingRecord> {
    let mut vector = embed_with_retry(backend, &chunk.text, VectorKind::Document, policy)?;
    normalize_vector(&mut vector);
    Ok(EmbeddingRecord {
        chunk_id: chunk.chunk_id.clone(),
        doc: chunk.doc.clone(),
        chunk_index: chunk.chunk_index,
        z_start: chunk.z_start,
        z_end: chunk.z_end,
        cell_start: chunk.cell_start,
        cell_end: chunk.cell_end,
        token_count: chunk.token_count,
        dominant_type: chunk.dominant_type,
        importance_mean: chunk.importance_mean,
        embedding: vector,
        text: chunk.text.clone(),
    })
}

fn embed_with_retry(
    backend: &BackendHandle,
    text: &str,
    kind: VectorKind,
    policy: &RetryPolicy,
) -> Result<Vec<f32>> {
    let mut attempt = 0usize;
    let mut rng = thread_rng();
    loop {
        match backend.embed(text, kind) {
            Ok(vec) => return Ok(vec),
            Err(err) => {
                attempt += 1;
                if attempt > policy.limit {
                    return Err(err);
                }
                let backoff = policy.delay_for_attempt(attempt);
                let jitter = rng.gen_range(0..=policy.base_delay_ms);
                sleep(backoff + Duration::from_millis(jitter));
            }
        }
    }
}

fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

struct LoadedEmbeddings {
    metadata: EmbeddingMetadataRow,
    records: Vec<EmbeddingRecord>,
}

#[derive(Serialize, Deserialize)]
struct FlatIndex {
    metadata: EmbeddingMetadataRow,
    records: Vec<EmbeddingRecord>,
}

struct EmbedCache {
    path: Option<PathBuf>,
    namespace: String,
    entries: HashMap<String, EmbeddingRecord>,
    dirty: bool,
}

impl EmbedCache {
    fn new(path: Option<&PathBuf>, namespace: &str) -> Result<Self> {
        let (entries, path_owned) = if let Some(p) = path {
            if p.exists() {
                let data = fs::read_to_string(p)
                    .with_context(|| format!("failed to read cache {}", p.display()))?;
                let map: HashMap<String, EmbeddingRecord> = serde_json::from_str(&data)
                    .with_context(|| format!("failed to parse cache {}", p.display()))?;
                (map, Some(p.clone()))
            } else {
                (HashMap::new(), Some(p.clone()))
            }
        } else {
            (HashMap::new(), None)
        };
        Ok(Self {
            path: path_owned,
            namespace: namespace.to_string(),
            entries,
            dirty: false,
        })
    }

    fn make_key(&self, chunk_id: &str, text_hash: &str) -> String {
        format!("{}::{}::{}", self.namespace, chunk_id, text_hash)
    }

    fn get(&self, chunk_id: &str, text_hash: &str) -> Option<EmbeddingRecord> {
        let key = self.make_key(chunk_id, text_hash);
        self.entries
            .get(&key)
            .or_else(|| self.entries.get(chunk_id))
            .cloned()
    }

    fn insert(&mut self, record: EmbeddingRecord, text_hash: &str) {
        let key = self.make_key(&record.chunk_id, text_hash);
        self.entries.insert(key, record);
        if self.path.is_some() {
            self.dirty = true;
        }
    }

    fn flush(&mut self) -> Result<()> {
        if self.dirty {
            if let Some(path) = &self.path {
                let mut file = BufWriter::new(
                    File::create(path)
                        .with_context(|| format!("failed to write cache {}", path.display()))?,
                );
                serde_json::to_writer(&mut file, &self.entries)?;
                file.flush()?;
            }
            self.dirty = false;
        }
        Ok(())
    }
}

#[derive(Clone)]
enum BackendHandle {
    Hash {
        embedder: HashEmbedder,
        dims: usize,
        seed: u64,
    },
    OpenAi {
        client: OpenAiBackend,
    },
    Cohere {
        client: CohereBackend,
    },
}

impl BackendHandle {
    fn new_hash(dimensions: usize, seed: u64) -> Self {
        let embedder = HashEmbedder::new(HashEmbedderConfig { dimensions, seed });
        BackendHandle::Hash {
            embedder,
            dims: dimensions,
            seed,
        }
    }

    fn new_openai(model: &str, api_key: &str, base_url: &str) -> Result<Self> {
        let client = OpenAiBackend::new(model, api_key, base_url)?;
        Ok(BackendHandle::OpenAi { client })
    }

    fn new_cohere(model: &str, api_key: &str, base_url: &str) -> Result<Self> {
        let client = CohereBackend::new(model, api_key, base_url)?;
        Ok(BackendHandle::Cohere { client })
    }

    fn embed(&self, text: &str, kind: VectorKind) -> Result<Vec<f32>> {
        match self {
            BackendHandle::Hash { embedder, .. } => Ok(embedder.embed_text(text)),
            BackendHandle::OpenAi { client } => client.embed(text),
            BackendHandle::Cohere { client } => client.embed(text, kind),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            BackendHandle::Hash { .. } => "hash",
            BackendHandle::OpenAi { .. } => "openai",
            BackendHandle::Cohere { .. } => "cohere",
        }
    }

    fn dims_hint(&self) -> Option<usize> {
        match self {
            BackendHandle::Hash { dims, .. } => Some(*dims),
            BackendHandle::OpenAi { .. } => None,
            BackendHandle::Cohere { .. } => None,
        }
    }

    fn seed(&self) -> u64 {
        match self {
            BackendHandle::Hash { seed, .. } => *seed,
            BackendHandle::OpenAi { .. } | BackendHandle::Cohere { .. } => 0,
        }
    }

    fn model(&self) -> Option<&str> {
        match self {
            BackendHandle::Hash { .. } => None,
            BackendHandle::OpenAi { client } => Some(client.model()),
            BackendHandle::Cohere { client } => Some(client.model()),
        }
    }
}

fn build_embed_backend(
    backend: &str,
    dimensions: usize,
    seed: u64,
    openai_model: Option<&str>,
    openai_api_key: Option<&str>,
    openai_base_url: &str,
    cohere_model: Option<&str>,
    cohere_api_key: Option<&str>,
    cohere_base_url: &str,
) -> Result<BackendHandle> {
    let openai_model_env = std::env::var("OPENAI_EMBEDDING_MODEL").ok();
    let openai_key_env = std::env::var("OPENAI_API_KEY").ok();
    let cohere_model_env = std::env::var("COHERE_EMBEDDING_MODEL").ok();
    let cohere_key_env = std::env::var("COHERE_API_KEY").ok();
    match backend.to_lowercase().as_str() {
        "hash" => Ok(BackendHandle::new_hash(dimensions, seed)),
        "openai" => {
            let model = openai_model
                .or_else(|| openai_model_env.as_deref())
                .ok_or_else(|| anyhow!("--openai-model or OPENAI_EMBEDDING_MODEL required"))?;
            let api_key = openai_api_key
                .or_else(|| openai_key_env.as_deref())
                .ok_or_else(|| anyhow!("--openai-api-key or OPENAI_API_KEY required"))?;
            BackendHandle::new_openai(model, api_key, openai_base_url)
        }
        "cohere" => {
            let model = cohere_model
                .or_else(|| cohere_model_env.as_deref())
                .unwrap_or("embed-multilingual-v3.0");
            let api_key = cohere_api_key
                .or_else(|| cohere_key_env.as_deref())
                .ok_or_else(|| anyhow!("--cohere-api-key or COHERE_API_KEY required"))?;
            BackendHandle::new_cohere(model, api_key, cohere_base_url)
        }
        other => bail!("unknown embedding backend: {}", other),
    }
}

fn build_backend_from_metadata(
    meta: &EmbeddingMetadataRow,
    backend_override: Option<&str>,
    dims_override: Option<usize>,
    seed_override: Option<u64>,
    openai_model: Option<&str>,
    openai_api_key: Option<&str>,
    openai_base_url: &str,
    cohere_model: Option<&str>,
    cohere_api_key: Option<&str>,
    cohere_base_url: &str,
) -> Result<BackendHandle> {
    let openai_key_env = std::env::var("OPENAI_API_KEY").ok();
    let cohere_key_env = std::env::var("COHERE_API_KEY").ok();
    let backend_name = backend_override.unwrap_or(meta.backend.as_str());
    match backend_name.to_lowercase().as_str() {
        "hash" => {
            let dims = dims_override.unwrap_or(meta.dimensions);
            let seed = seed_override.unwrap_or(meta.seed);
            Ok(BackendHandle::new_hash(dims, seed))
        }
        "openai" => {
            let model = openai_model
                .or(meta.model.as_deref())
                .ok_or_else(|| anyhow!("embedding metadata missing model; pass --openai-model"))?;
            let api_key = openai_api_key
                .or_else(|| openai_key_env.as_deref())
                .ok_or_else(|| anyhow!("--openai-api-key or OPENAI_API_KEY required"))?;
            BackendHandle::new_openai(model, api_key, openai_base_url)
        }
        "cohere" => {
            let model = cohere_model
                .or(meta.model.as_deref())
                .unwrap_or("embed-multilingual-v3.0");
            let api_key = cohere_api_key
                .or_else(|| cohere_key_env.as_deref())
                .ok_or_else(|| anyhow!("--cohere-api-key or COHERE_API_KEY required"))?;
            BackendHandle::new_cohere(model, api_key, cohere_base_url)
        }
        other => bail!("unsupported backend '{}' in metadata", other),
    }
}

fn load_embeddings_from_jsonl(path: &Path) -> Result<LoadedEmbeddings> {
    let reader = BufReader::new(
        File::open(path)
            .with_context(|| format!("failed to open embeddings {}", path.display()))?,
    );
    let mut metadata: Option<EmbeddingMetadataRow> = None;
    let mut records = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(meta) = serde_json::from_str::<EmbeddingMetadataRow>(line) {
            if meta.row_type == "embed_meta" {
                metadata = Some(meta);
                continue;
            }
        }
        let record: EmbeddingRecord = serde_json::from_str(line)
            .with_context(|| format!("invalid embedding row {} in {}", idx + 1, path.display()))?;
        records.push(record);
    }
    let metadata =
        metadata.ok_or_else(|| anyhow!("{} missing embed_meta header", path.display()))?;
    Ok(LoadedEmbeddings { metadata, records })
}

fn save_flat_index(path: &Path, data: &LoadedEmbeddings) -> Result<()> {
    let mut file = BufWriter::new(
        File::create(path).with_context(|| format!("failed to create index {}", path.display()))?,
    );
    let index = FlatIndex {
        metadata: data.metadata.clone(),
        records: data.records.clone(),
    };
    bincode::serialize_into(&mut file, &index)?;
    file.flush()?;
    Ok(())
}

fn load_flat_index(path: &Path) -> Result<LoadedEmbeddings> {
    let mut file = BufReader::new(
        File::open(path).with_context(|| format!("failed to open index {}", path.display()))?,
    );
    let index: FlatIndex = bincode::deserialize_from(&mut file)?;
    Ok(LoadedEmbeddings {
        metadata: index.metadata,
        records: index.records,
    })
}

fn load_embeddings_source(
    embeddings: Option<&PathBuf>,
    index: Option<&PathBuf>,
) -> Result<LoadedEmbeddings> {
    match (embeddings, index) {
        (Some(path), None) => load_embeddings_from_jsonl(path),
        (None, Some(path)) => load_flat_index(path),
        (Some(_), Some(_)) => bail!("provide either --embeddings or --index, not both"),
        (None, None) => bail!("provide --embeddings or --index"),
    }
}

#[derive(Clone)]
struct OpenAiBackend {
    client: HttpClient,
    model: String,
    api_key: String,
    base_url: String,
}

impl OpenAiBackend {
    fn new(model: &str, api_key: &str, base_url: &str) -> Result<Self> {
        let client = HttpClient::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self {
            client,
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct Request<'a> {
            model: &'a str,
            input: &'a str,
        }

        #[derive(Deserialize)]
        struct Response {
            data: Vec<ResponseItem>,
        }

        #[derive(Deserialize)]
        struct ResponseItem {
            embedding: Vec<f64>,
        }

        #[derive(Deserialize)]
        struct ErrorResponse {
            error: ErrorBody,
        }

        #[derive(Deserialize)]
        struct ErrorBody {
            message: String,
        }

        let url = format!("{}/embeddings", self.base_url);
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))?,
        );
        let req = Request {
            model: &self.model,
            input: text,
        };
        let response = self.client.post(url).headers(headers).json(&req).send()?;
        if !response.status().is_success() {
            let status = response.status();
            let body: ErrorResponse = response.json().unwrap_or_else(|_| ErrorResponse {
                error: ErrorBody {
                    message: "unknown error".to_string(),
                },
            });
            bail!(
                "openai embeddings failed ({}): {}",
                status,
                body.error.message
            );
        }
        let body: Response = response.json()?;
        let first = body
            .data
            .first()
            .ok_or_else(|| anyhow!("missing embedding data"))?;
        Ok(first.embedding.iter().map(|v| *v as f32).collect())
    }
}

#[derive(Clone)]
struct CohereBackend {
    client: HttpClient,
    model: String,
    api_key: String,
    base_url: String,
}

struct QdrantClient {
    http: HttpClient,
    base_url: String,
    api_key: Option<String>,
}

impl QdrantClient {
    fn new(url: &str, api_key: Option<&str>) -> Result<Self> {
        let client = HttpClient::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self {
            http: client,
            base_url: url.trim_end_matches('/').to_string(),
            api_key: api_key.map(|s| s.to_string()),
        })
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(key) = &self.api_key {
            headers.insert(
                HeaderName::from_static("api-key"),
                HeaderValue::from_str(key)?,
            );
        }
        Ok(headers)
    }

    fn ensure_collection(&self, collection: &str, size: usize) -> Result<()> {
        let url = format!("{}/collections/{}", self.base_url, collection);
        let body = json!({
            "vectors": {
                "size": size,
                "distance": "Cosine"
            }
        });
        let response = self
            .http
            .put(url)
            .headers(self.headers()?)
            .json(&body)
            .send()?;
        match response.status() {
            StatusCode::OK | StatusCode::CREATED => Ok(()),
            StatusCode::CONFLICT => Ok(()),
            other => bail!("qdrant collection error ({}): {}", other, response.text()?),
        }
    }

    fn upsert_points(&self, collection: &str, points: Vec<QdrantPoint>, wait: bool) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let url = format!("{}/collections/{}/points", self.base_url, collection);
        let query = if wait {
            vec![("wait", "true")]
        } else {
            Vec::new()
        };
        let body = json!({ "points": points });
        let response = self
            .http
            .post(url)
            .headers(self.headers()?)
            .query(&query)
            .json(&body)
            .send()?;
        if response.status().is_success() {
            Ok(())
        } else {
            bail!(
                "qdrant upsert failed ({}): {}",
                response.status(),
                response.text()?
            );
        }
    }

    fn search(
        &self,
        collection: &str,
        query_text: &str,
        vector: &[f32],
        limit: usize,
        filter: Option<&FilterPredicate>,
        hybrid: bool,
        records: &[EmbeddingRecord],
    ) -> Result<Vec<SearchHit>> {
        #[derive(Serialize)]
        struct SearchRequest {
            vector: Vec<f32>,
            limit: usize,
            #[serde(rename = "with_payload")]
            with_payload: bool,
        }

        #[derive(Deserialize)]
        struct SearchResponse {
            result: Vec<SearchEntry>,
        }

        #[derive(Deserialize)]
        struct SearchEntry {
            score: f32,
            payload: Option<serde_json::Value>,
        }

        let fetch_limit = if filter.is_some() || hybrid {
            limit.saturating_mul(3)
        } else {
            limit
        };
        let url = format!("{}/collections/{}/points/search", self.base_url, collection);
        let body = SearchRequest {
            vector: vector.to_vec(),
            limit: fetch_limit.max(limit),
            with_payload: true,
        };
        let response = self
            .http
            .post(url)
            .headers(self.headers()?)
            .json(&body)
            .send()?;
        if !response.status().is_success() {
            bail!(
                "qdrant search failed ({}): {}",
                response.status(),
                response.text()?
            );
        }
        let body: SearchResponse = response.json()?;
        let mut record_map = HashMap::new();
        let mut doc_tokens = HashMap::new();
        for record in records {
            record_map.insert(record.chunk_id.clone(), record);
            doc_tokens.insert(record.chunk_id.clone(), tokenize(&record.text));
        }
        let avg_len = if doc_tokens.is_empty() {
            1.0
        } else {
            doc_tokens
                .values()
                .map(|tokens| tokens.len() as f32)
                .sum::<f32>()
                / doc_tokens.len() as f32
        };
        let mut df = HashMap::<String, usize>::new();
        for tokens in doc_tokens.values() {
            let mut seen = HashSet::new();
            for token in tokens {
                if seen.insert(token) {
                    *df.entry(token.clone()).or_insert(0) += 1;
                }
            }
        }
        let query_tokens = if hybrid {
            Some(tokenize(query_text))
        } else {
            None
        };
        let mut hits = Vec::new();
        for entry in body.result {
            let payload = match entry.payload {
                Some(p) => p,
                None => continue,
            };
            let chunk_id = match payload.get("chunk_id").and_then(|v| v.as_str()) {
                Some(id) => id,
                None => continue,
            };
            let record = match record_map.get(chunk_id) {
                Some(r) => *r,
                None => continue,
            };
            if let Some(pred) = filter {
                if !pred.matches(record) {
                    continue;
                }
            }
            let mut score = entry.score;
            if let (true, Some(qt)) = (hybrid, query_tokens.as_ref()) {
                if let Some(doc_tok) = doc_tokens.get(chunk_id) {
                    let sparse = bm25_score(qt, doc_tok, &df, avg_len);
                    score = score * 0.7 + sparse * 0.3;
                }
            }
            hits.push(SearchHit {
                doc: record.doc.clone(),
                chunk_id: record.chunk_id.clone(),
                chunk_index: record.chunk_index,
                score,
                preview: short_preview(&record.text, 160),
            });
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }
}

#[derive(Serialize)]
struct QdrantPoint {
    id: String,
    vector: Vec<f32>,
    payload: serde_json::Value,
}

impl CohereBackend {
    fn new(model: &str, api_key: &str, base_url: &str) -> Result<Self> {
        let client = HttpClient::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self {
            client,
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn embed(&self, text: &str, kind: VectorKind) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct Request<'a> {
            model: &'a str,
            texts: Vec<&'a str>,
            #[serde(rename = "input_type")]
            input_type: &'a str,
        }

        #[derive(Deserialize)]
        struct Response {
            embeddings: Vec<Vec<f64>>,
        }

        #[derive(Deserialize)]
        struct ErrorResponse {
            message: String,
        }

        let url = format!("{}/embed", self.base_url);
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))?,
        );
        headers.insert(
            HeaderName::from_static("cohere-version"),
            HeaderValue::from_static("2022-12-06"),
        );
        let input_type = match kind {
            VectorKind::Document => "search_document",
            VectorKind::Query => "search_query",
        };
        let req = Request {
            model: &self.model,
            texts: vec![text],
            input_type,
        };
        let response = self.client.post(url).headers(headers).json(&req).send()?;
        if !response.status().is_success() {
            let status = response.status();
            let body: ErrorResponse = response.json().unwrap_or(ErrorResponse {
                message: "unknown Cohere error".to_string(),
            });
            bail!("cohere embeddings failed ({}): {}", status, body.message);
        }
        let body: Response = response.json()?;
        let first = body
            .embeddings
            .first()
            .ok_or_else(|| anyhow!("missing embedding data"))?;
        Ok(first.iter().map(|v| *v as f32).collect())
    }
}

fn write_report(input: &Path, out: &Path) -> Result<()> {
    let contents = fs::read_to_string(input)
        .with_context(|| format!("failed to read bench results {}", input.display()))?;
    let mut rows = Vec::new();
    for (idx, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("failed to parse row {}", idx + 1))?;
        if value.get("row_type").and_then(|v| v.as_str()).unwrap_or("") != "doc" {
            continue;
        }
        let row = ReportRow {
            doc: value
                .get("doc")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            tokens_raw: value
                .get("tokens_raw")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            tokens_3dcf: value
                .get("tokens_3dcf")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            savings_ratio: value
                .get("savings_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            encode_ms: value.get("encode_ms").and_then(|v| v.as_u64()).unwrap_or(0) as u128,
            decode_ms: value.get("decode_ms").and_then(|v| v.as_u64()).unwrap_or(0) as u128,
            encode_pages_per_s: value
                .get("encode_pages_per_s")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            decode_pages_per_s: value
                .get("decode_pages_per_s")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            numguard_mismatches: value
                .get("numguard_mismatches")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            mem_peak_mb: value
                .get("mem_peak_mb")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
        };
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(anyhow!("no benchmark rows found"));
    }
    let total = rows.len() as f64;
    let mean = rows.iter().map(|r| r.savings_ratio as f64).sum::<f64>() / total;
    let median = percentile(
        &rows
            .iter()
            .map(|r| r.savings_ratio as f64)
            .collect::<Vec<_>>(),
        0.5,
    );
    let encode_vals = rows.iter().map(|r| r.encode_ms as f64).collect::<Vec<_>>();
    let decode_vals = rows.iter().map(|r| r.decode_ms as f64).collect::<Vec<_>>();
    let encode_p50 = percentile(&encode_vals, 0.5);
    let encode_p95 = percentile(&encode_vals, 0.95);
    let decode_p50 = percentile(&decode_vals, 0.5);
    let decode_p95 = percentile(&decode_vals, 0.95);
    let mean_encode_pages = rows.iter().map(|r| r.encode_pages_per_s).sum::<f64>() / total;
    let mean_decode_pages = rows.iter().map(|r| r.decode_pages_per_s).sum::<f64>() / total;
    let max_mem = rows.iter().map(|r| r.mem_peak_mb).fold(0.0f64, f64::max);
    let mut html = String::new();
    html.push_str("<html><head><meta charset='utf-8'><title>3DCF Bench Report</title><style>body{font-family:system-ui,monospace;margin:2rem;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ccc;padding:0.35rem;text-align:left;}tbody tr:nth-child(odd){background:#fafafa;}</style></head><body>");
    html.push_str(&format!(
        "<h1>3DCF Benchmark</h1><p>documents: {}<br>mean savings: {:.2}x<br>median savings: {:.2}x<br>encode p50: {:.0} ms &nbsp; encode p95: {:.0} ms<br>decode p50: {:.0} ms &nbsp; decode p95: {:.0} ms<br>mean encode throughput: {:.2} pages/s &nbsp; mean decode throughput: {:.2} pages/s<br>max RSS: {:.1} MB</p>",
        rows.len(),
        mean,
        median,
        encode_p50,
        encode_p95,
        decode_p50,
        decode_p95,
        mean_encode_pages,
        mean_decode_pages,
        max_mem
    ));
    html.push_str("<table><thead><tr><th>Document</th><th>Raw Tokens</th><th>3DCF Tokens</th><th>Savings</th><th>Encode (ms / pps)</th><th>Decode (ms / pps)</th><th>NumGuard</th><th>Peak MB</th></tr></thead><tbody>");
    for row in &rows {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2}x</td><td>{} ({:.2})</td><td>{} ({:.2})</td><td>{}</td><td>{:.1}</td></tr>",
            row.doc,
            row.tokens_raw,
            row.tokens_3dcf,
            row.savings_ratio,
            row.encode_ms,
            row.encode_pages_per_s,
            row.decode_ms,
            row.decode_pages_per_s,
            row.numguard_mismatches,
            row.mem_peak_mb
        ));
    }
    html.push_str("</tbody></table></body></html>");
    fs::write(out, html).with_context(|| format!("failed to write {}", out.display()))?;
    Ok(())
}

struct ReportRow {
    doc: String,
    tokens_raw: usize,
    tokens_3dcf: usize,
    savings_ratio: f64,
    encode_ms: u128,
    decode_ms: u128,
    encode_pages_per_s: f64,
    decode_pages_per_s: f64,
    numguard_mismatches: usize,
    mem_peak_mb: f64,
}

#[derive(Serialize)]
struct CellManifestRow {
    doc: String,
    cell_index: usize,
    code_hash: String,
    z: u32,
    x: i32,
    y: i32,
    w: u32,
    h: u32,
    importance: u8,
    cell_type: String,
    text: String,
    preview: String,
}

#[derive(Serialize)]
struct RedactedCell {
    doc: String,
    cell_index: usize,
    code_hash: String,
    z: u32,
    x: i32,
    y: i32,
    w: u32,
    h: u32,
    importance: u8,
    cell_type: String,
    text: String,
    preview: String,
}

fn generate_synthetic(out_dir: &Path, count: usize, seed: u64) -> Result<()> {
    fs::create_dir_all(out_dir)?;
    let mut rng = StdRng::seed_from_u64(seed);
    for idx in 0..count {
        let revenue: i32 = rng.gen_range(50_000..500_000);
        let expenses: i32 = rng.gen_range(10_000..200_000);
        let profit = revenue - expenses;
        let idx_name = format!("synthetic_{idx:03}.md");
        let path = out_dir.join(idx_name);
        let variance = rng.gen_range(3..15);
        let attrition = rng.gen_range(2..9);
        let fx: i32 = rng.gen_range(1_000..9_000);
        let content = format!(
            "## Synthetic Report {0}\n\n| Metric | Value |\n| --- | --- |\n| Revenue | ${1} |\n| Expenses | ${2} |\n| Profit | ${3} |\n\nTop risks:\n1. Supply chain variance {4}%\n2. Attrition {5}%\n3. FX impact ${6}\n",
            idx, revenue, expenses, profit, variance, attrition, fx
        );
        fs::write(&path, content)?;
    }
    Ok(())
}

#[derive(Debug, Default, Deserialize)]
struct AppConfig {
    #[serde(default)]
    defaults: DefaultSections,
    #[serde(default)]
    pricing: PricingConfig,
}

#[derive(Debug, Default, Deserialize)]
struct DefaultSections {
    #[serde(default)]
    encode: Option<EncodeDefaults>,
    #[serde(default)]
    bench: Option<BenchDefaults>,
    #[serde(default)]
    stats: Option<StatsDefaults>,
}

impl DefaultSections {
    fn stats_tokenizer(&self) -> Option<String> {
        self.stats.as_ref().and_then(|s| s.tokenizer.clone())
    }

    fn stats_tokenizer_file(&self) -> Option<PathBuf> {
        self.stats.as_ref().and_then(|s| s.tokenizer_file.clone())
    }
}

#[derive(Debug, Default, Deserialize)]
struct EncodeDefaults {
    preset: Option<String>,
    budget: Option<usize>,
    drop_footers: Option<bool>,
    dedup_window: Option<u32>,
    hyphenation: Option<String>,
    table_column_tolerance: Option<u32>,
    enable_ocr: Option<bool>,
    force_ocr: Option<bool>,
    ocr_langs: Option<Vec<String>>,
    heading_boost: Option<f32>,
    number_boost: Option<f32>,
    footer_penalty: Option<f32>,
    early_line_bonus: Option<f32>,
    table_mode: Option<String>,
    preset_label: Option<String>,
    budget_label: Option<String>,
    strict_numguard: Option<bool>,
    numguard_units: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct ResolvedEncodeConfig {
    preset: String,
    budget: Option<usize>,
    drop_footers: bool,
    dedup_window: u32,
    hyphenation: HyphenationMode,
    table_column_tolerance: u32,
    enable_ocr: bool,
    force_ocr: bool,
    ocr_langs: Vec<String>,
    importance: ImportanceTuning,
    table_mode: TableMode,
    preset_label: String,
    budget_label: String,
    strict_numguard: bool,
    numguard_units: Option<PathBuf>,
}

struct ContextArtifacts {
    resolved: ResolvedEncodeConfig,
    metrics: Metrics,
    raw_text: String,
    context_text: String,
}

fn resolve_encode_config(
    cli: &EncodeArgs,
    defaults: Option<&EncodeDefaults>,
) -> Result<ResolvedEncodeConfig> {
    let preset = cli
        .preset
        .clone()
        .or_else(|| defaults.and_then(|d| d.preset.clone()))
        .unwrap_or_else(|| "reports".to_string());
    let budget = cli.budget.or_else(|| defaults.and_then(|d| d.budget));
    let dedup_window = cli
        .dedup_window
        .or_else(|| defaults.and_then(|d| d.dedup_window))
        .unwrap_or(0);
    let hyphenation = cli
        .hyphenation
        .as_deref()
        .or_else(|| defaults.and_then(|d| d.hyphenation.as_deref()))
        .map(hyphenation_from_str)
        .unwrap_or(HyphenationMode::Merge);
    let table_column_tolerance = cli
        .table_column_tolerance
        .or_else(|| defaults.and_then(|d| d.table_column_tolerance))
        .unwrap_or(24);
    let mut ocr_langs = if !cli.ocr_langs.is_empty() {
        cli.ocr_langs.clone()
    } else {
        defaults
            .and_then(|d| d.ocr_langs.clone())
            .unwrap_or_else(|| vec!["eng".to_string()])
    };
    if ocr_langs.is_empty() {
        ocr_langs.push("eng".to_string());
    }
    let force_ocr = cli.force_ocr || defaults.and_then(|d| d.force_ocr).unwrap_or(false);
    let enable_ocr = cli.enable_ocr
        || force_ocr
        || defaults.and_then(|d| d.enable_ocr).unwrap_or(false)
        || !ocr_langs.is_empty();
    let drop_footers = cli.drop_footers || defaults.and_then(|d| d.drop_footers).unwrap_or(false);
    let strict_numguard =
        cli.strict_numguard || defaults.and_then(|d| d.strict_numguard).unwrap_or(false);
    let numguard_units = cli
        .numguard_units
        .clone()
        .or_else(|| defaults.and_then(|d| d.numguard_units.clone()));
    let mut importance = ImportanceTuning::default();
    if let Some(d) = defaults {
        if let Some(v) = d.heading_boost {
            importance.heading_boost = v;
        }
        if let Some(v) = d.number_boost {
            importance.number_boost = v;
        }
        if let Some(v) = d.footer_penalty {
            importance.footer_penalty = v;
        }
        if let Some(v) = d.early_line_bonus {
            importance.early_line_bonus = v;
        }
    }
    if let Some(v) = cli.heading_boost {
        importance.heading_boost = v;
    }
    if let Some(v) = cli.number_boost {
        importance.number_boost = v;
    }
    if let Some(v) = cli.footer_penalty {
        importance.footer_penalty = v;
    }
    if let Some(v) = cli.early_line_bonus {
        importance.early_line_bonus = v;
    }
    let table_mode = parse_table_mode(
        cli.table_mode
            .as_deref()
            .or_else(|| defaults.and_then(|d| d.table_mode.as_deref())),
    )?;
    let preset_label = cli
        .preset_label
        .clone()
        .or_else(|| defaults.and_then(|d| d.preset_label.clone()))
        .unwrap_or_else(|| preset.clone());
    let budget_label = cli
        .budget_label
        .clone()
        .or_else(|| defaults.and_then(|d| d.budget_label.clone()))
        .or_else(|| budget.map(|b| b.to_string()))
        .unwrap_or_else(|| "auto".to_string());
    Ok(ResolvedEncodeConfig {
        preset,
        budget,
        drop_footers,
        dedup_window,
        hyphenation,
        table_column_tolerance,
        enable_ocr,
        force_ocr,
        ocr_langs,
        importance,
        table_mode,
        preset_label,
        budget_label,
        strict_numguard,
        numguard_units,
    })
}

fn build_encoder_from_resolved(resolved: &ResolvedEncodeConfig) -> Result<Encoder> {
    let builder = Encoder::builder(&resolved.preset)?
        .budget(resolved.budget)
        .drop_footers(resolved.drop_footers)
        .dedup_window(resolved.dedup_window)
        .hyphenation(resolved.hyphenation)
        .table_tolerance(resolved.table_column_tolerance)
        .enable_ocr(resolved.enable_ocr)
        .force_ocr(resolved.force_ocr)
        .ocr_languages(resolved.ocr_langs.clone())
        .importance_tuning(resolved.importance.clone());
    Ok(builder.build())
}

fn build_context_artifacts(
    input: &Path,
    encode: &EncodeArgs,
    defaults: Option<&EncodeDefaults>,
) -> Result<(ContextArtifacts, Vec<NumGuardAlert>)> {
    let resolved = resolve_encode_config(encode, defaults)?;
    let encoder = build_encoder_from_resolved(&resolved)?;
    let (doc, metrics, raw_text) = encoder.encode_path_with_plaintext(input)?;
    let serializer = TextSerializer::with_config(TextSerializerConfig {
        table_mode: resolved.table_mode,
        preset_label: Some(resolved.preset_label.clone()),
        budget_label: Some(resolved.budget_label.clone()),
        ..Default::default()
    });
    let context_text = serializer.to_string(&doc)?;
    let unit_whitelist = load_unit_whitelist(resolved.numguard_units.clone())?;
    let alerts = doc.numguard_mismatches_with_units(unit_whitelist.as_ref());
    Ok((
        ContextArtifacts {
            resolved,
            metrics,
            raw_text,
            context_text,
        },
        alerts,
    ))
}

#[derive(Debug, Default, Deserialize)]
struct BenchDefaults {
    preset: Option<String>,
    tokenizer: Option<String>,
    tokenizer_file: Option<PathBuf>,
    budget: Option<usize>,
    mode: Option<String>,
    budgets: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct StatsDefaults {
    tokenizer: Option<String>,
    tokenizer_file: Option<PathBuf>,
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
fn search_dense_only(
    query_vec: &[f32],
    records: &[EmbeddingRecord],
    filter: Option<&FilterPredicate>,
) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    for record in records {
        if let Some(pred) = filter {
            if !pred.matches(record) {
                continue;
            }
        }
        let score = cosine_similarity(query_vec, &record.embedding);
        hits.push(SearchHit {
            doc: record.doc.clone(),
            chunk_id: record.chunk_id.clone(),
            chunk_index: record.chunk_index,
            score,
            preview: short_preview(&record.text, 160),
        });
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits
}

fn search_hybrid(
    query_text: &str,
    query_vec: &[f32],
    records: &[EmbeddingRecord],
    filter: Option<&FilterPredicate>,
) -> Vec<SearchHit> {
    let query_tokens = tokenize(query_text);
    if query_tokens.is_empty() {
        return search_dense_only(query_vec, records, filter);
    }
    let doc_tokens: Vec<Vec<String>> = records.iter().map(|r| tokenize(&r.text)).collect();
    if doc_tokens.is_empty() {
        return search_dense_only(query_vec, records, filter);
    }
    let avg_len = doc_tokens
        .iter()
        .map(|tokens| tokens.len() as f32)
        .sum::<f32>()
        / doc_tokens.len().max(1) as f32;
    let mut df = HashMap::<String, usize>::new();
    for tokens in &doc_tokens {
        let mut seen = HashSet::new();
        for token in tokens {
            if seen.insert(token) {
                *df.entry(token.clone()).or_insert(0) += 1;
            }
        }
    }
    let mut hits = Vec::new();
    for (idx, record) in records.iter().enumerate() {
        if let Some(pred) = filter {
            if !pred.matches(record) {
                continue;
            }
        }
        let dense = cosine_similarity(query_vec, &record.embedding);
        let sparse = bm25_score(&query_tokens, &doc_tokens[idx], &df, avg_len);
        let score = dense * 0.7 + sparse * 0.3;
        hits.push(SearchHit {
            doc: record.doc.clone(),
            chunk_id: record.chunk_id.clone(),
            chunk_index: record.chunk_index,
            score,
            preview: short_preview(&record.text, 160),
        });
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits
}

fn bm25_score(
    query_tokens: &[String],
    doc_tokens: &[String],
    df: &HashMap<String, usize>,
    avg_len: f32,
) -> f32 {
    const K1: f32 = 1.2;
    const B: f32 = 0.75;
    if doc_tokens.is_empty() {
        return 0.0;
    }
    let doc_len = doc_tokens.len() as f32;
    let mut tf = HashMap::<&str, usize>::new();
    for token in doc_tokens {
        *tf.entry(token).or_insert(0) += 1;
    }
    let mut score = 0.0;
    let total_docs = df.values().copied().max().unwrap_or(1) as f32;
    for token in query_tokens {
        if let Some(freq) = tf.get(token.as_str()) {
            let df_token = df.get(token).copied().unwrap_or(1) as f32;
            let idf = ((total_docs - df_token + 0.5) / (df_token + 0.5))
                .ln()
                .max(0.0);
            let numerator = (*freq as f32) * (K1 + 1.0);
            let denominator = (*freq as f32) + K1 * (1.0 - B + B * (doc_len / avg_len.max(1e-3)));
            score += idf * (numerator / denominator.max(1e-6));
        }
    }
    score
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_lowercase())
        .collect()
}
