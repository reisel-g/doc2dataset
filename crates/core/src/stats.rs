use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use base64::{engine::general_purpose, Engine as _};
use rustc_hash::FxHashMap;
use serde::Deserialize;

use crate::decoder::Decoder;
use crate::document::Document;
use crate::error::{DcfError, Result};
use crate::serializer::TextSerializer;

#[derive(Debug, Clone)]
pub enum TokenizerKind {
    Cl100k,
    Gpt2,
    O200k,
    Anthropic,
    Custom(PathBuf),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stats {
    pub tokens_raw: usize,
    pub tokens_3dcf: usize,
    pub cells: usize,
    pub unique_payloads: usize,
    pub savings_ratio: f32,
}

impl Stats {
    pub fn measure(document: &Document, tokenizer: TokenizerKind) -> Result<Self> {
        let encoder = tokenizer.build()?;
        Self::measure_with_bpe(document, &encoder)
    }

    pub fn measure_with_bpe(document: &Document, tokenizer: &tiktoken_rs::CoreBPE) -> Result<Self> {
        let decoder = Decoder::new();
        let raw_text = decoder.to_text(document)?;
        let textual = TextSerializer::new().to_string(document)?;
        let tokens_raw = tokenizer
            .encode_with_special_tokens(raw_text.as_str())
            .len();
        let tokens_3dcf = tokenizer.encode_with_special_tokens(textual.as_str()).len();
        let savings_ratio = if tokens_3dcf == 0 {
            0.0
        } else {
            tokens_raw as f32 / tokens_3dcf as f32
        };
        Ok(Self {
            tokens_raw,
            tokens_3dcf,
            cells: document.total_cells(),
            unique_payloads: document.dict.len(),
            savings_ratio,
        })
    }
}

pub fn estimate_tokens(text: &str, tokenizer: &TokenizerKind) -> Result<usize> {
    let encoder = tokenizer.build()?;
    Ok(estimate_tokens_with_bpe(text, &encoder))
}

pub fn estimate_tokens_with_bpe(text: &str, tokenizer: &tiktoken_rs::CoreBPE) -> usize {
    tokenizer.encode_with_special_tokens(text).len()
}

impl TokenizerKind {
    pub fn build(&self) -> Result<tiktoken_rs::CoreBPE> {
        match self {
            TokenizerKind::Cl100k => {
                tiktoken_rs::cl100k_base().map_err(|e| DcfError::Tokenizer(e.to_string()))
            }
            TokenizerKind::Gpt2 => {
                tiktoken_rs::p50k_base().map_err(|e| DcfError::Tokenizer(e.to_string()))
            }
            TokenizerKind::O200k => {
                tiktoken_rs::o200k_base().map_err(|e| DcfError::Tokenizer(e.to_string()))
            }
            TokenizerKind::Anthropic => anthropic_base(),
            TokenizerKind::Custom(path) => load_custom_tokenizer(path),
        }
    }
}

fn anthropic_base() -> Result<tiktoken_rs::CoreBPE> {
    // Placeholder: Anthropic tokenization aligns closely with cl100k defaults.
    tiktoken_rs::cl100k_base().map_err(|e| DcfError::Tokenizer(e.to_string()))
}

fn load_custom_tokenizer(path: &Path) -> Result<tiktoken_rs::CoreBPE> {
    let data = fs::read_to_string(path).map_err(|e| {
        DcfError::Tokenizer(format!(
            "failed to read tokenizer file {}: {e}",
            path.display()
        ))
    })?;
    let spec: CustomTokenizerSpec = serde_json::from_str(&data)
        .map_err(|e| DcfError::Tokenizer(format!("invalid tokenizer json: {e}")))?;
    let mut encoder: FxHashMap<Vec<u8>, usize> = FxHashMap::default();
    for (token, rank) in spec.mergeable_ranks {
        encoder.insert(decode_token_key(&token), rank);
    }
    let mut special_tokens: FxHashMap<String, usize> = FxHashMap::default();
    special_tokens.extend(spec.special_tokens.into_iter());
    tiktoken_rs::CoreBPE::new(encoder, special_tokens, &spec.pat_str)
        .map_err(|e| DcfError::Tokenizer(format!("failed to build tokenizer: {e}")))
}

#[derive(Deserialize)]
struct CustomTokenizerSpec {
    pat_str: String,
    mergeable_ranks: HashMap<String, usize>,
    #[serde(default)]
    special_tokens: HashMap<String, usize>,
}

fn decode_token_key(key: &str) -> Vec<u8> {
    general_purpose::STANDARD
        .decode(key)
        .unwrap_or_else(|_| key.as_bytes().to_vec())
}
