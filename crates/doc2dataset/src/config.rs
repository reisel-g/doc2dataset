use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::env;
use three_dcf_llm::LlmProvider;

#[derive(Debug, Clone)]
pub struct Doc2DatasetConfig {
    pub provider: LlmProvider,
    pub model: String,
    pub lang: String,
    pub llm_delay_ms: u64,
    pub qa_max_per_doc: usize,
    pub summary_max_per_doc: usize,
}

impl Doc2DatasetConfig {
    pub fn from_env() -> Result<Self> {
        let provider_name =
            env::var("DOC2DATASET_PROVIDER").unwrap_or_else(|_| "openai".to_string());
        let provider = LlmProvider::from_str(&provider_name)
            .ok_or_else(|| anyhow!(format!("unknown provider {provider_name}")))?;
        let default_model = default_model(provider);
        let model = env::var("DOC2DATASET_MODEL").unwrap_or_else(|_| default_model.to_string());
        let lang = env::var("DOC2DATASET_LANG").unwrap_or_else(|_| "en".to_string());
        let llm_delay_ms = env::var("DOC2DATASET_THROTTLE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let qa_max_per_doc = env::var("DOC2DATASET_QA_MAX_PER_DOC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_QA_MAX_PER_DOC);
        let summary_max_per_doc = env::var("DOC2DATASET_SUMMARY_MAX_PER_DOC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_SUMMARY_MAX_PER_DOC);
        Ok(Self {
            provider,
            model,
            lang,
            llm_delay_ms,
            qa_max_per_doc,
            summary_max_per_doc,
        })
    }
}

fn default_model(provider: LlmProvider) -> &'static str {
    match provider {
        LlmProvider::OpenAi => "gpt-4.1-mini",
        LlmProvider::Anthropic => "claude-3-5-sonnet",
        LlmProvider::Gemini => "gemini-1.5-flash",
        LlmProvider::Deepseek => "deepseek-chat",
        LlmProvider::Local => "local",
    }
}

#[derive(Debug, Deserialize)]
pub struct SourceConfig {
    pub path: String,
    #[serde(default = "default_pattern")]
    pub pattern: String,
}

fn default_pattern() -> String {
    "*.pdf,*.md,*.txt,*.html,*.htm,*.xml,*.xhtml,*.rss,*.atom,*.json,*.yaml,*.yml,*.csv,*.tsv,*.csv.gz,*.tsv.gz,*.tex,*.bib,*.bbl,*.ini,*.cfg,*.conf,*.toml,*.log,*.rtf,*.png,*.jpg,*.jpeg"
        .to_string()
}

#[derive(Debug, Deserialize, Default)]
pub struct ExportsConfig {
    #[serde(default)]
    pub hf: bool,
    #[serde(default)]
    pub llama_factory: Option<LlamaFactoryExport>,
    #[serde(default)]
    pub openai: bool,
    #[serde(default)]
    pub axolotl: Option<AxolotlExport>,
    #[serde(default)]
    pub rag_jsonl: bool,
}

#[derive(Debug, Deserialize)]
pub struct LlamaFactoryExport {
    #[serde(default = "default_lf_format")]
    pub format: String,
}

fn default_lf_format() -> String {
    "alpaca".to_string()
}

#[derive(Debug, Deserialize)]
pub struct AxolotlExport {
    #[serde(default = "default_ax_mode")]
    pub mode: String,
}

fn default_ax_mode() -> String {
    "chat".to_string()
}

#[derive(Debug, Deserialize, Default)]
pub struct IngestConfig {
    #[serde(default = "default_preset")]
    pub preset: String,
    #[serde(default)]
    pub enable_ocr: bool,
    #[serde(default)]
    pub force_ocr: bool,
    #[serde(default = "default_ocr_langs")]
    pub ocr_langs: Vec<String>,
}

fn default_preset() -> String {
    "reports".to_string()
}

fn default_ocr_langs() -> Vec<String> {
    vec!["eng".to_string()]
}

#[derive(Debug, Deserialize)]
pub struct RunConfig {
    pub dataset_root: String,
    pub sources: Vec<SourceConfig>,
    pub tasks: Vec<String>,
    #[serde(default)]
    pub exports: ExportsConfig,
    #[serde(default)]
    pub ingest: IngestConfig,
}
pub const DEFAULT_QA_MAX_PER_DOC: usize = 4;
pub const DEFAULT_SUMMARY_MAX_PER_DOC: usize = 3;
