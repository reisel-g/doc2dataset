use crate::embedding::EmbeddingClient;
use crate::encryption::decrypt_text;
use crate::store::{RagPolicy, RagStore, ScoredCell, SearchFilters};
use crate::{LlmClient, LlmRequest, LlmResponse};
use anyhow::{anyhow, Result};
use std::path::Path;
use three_dcf_core::{estimate_tokens, TokenizerKind};

pub struct RagQuery {
    pub collection: String,
    pub question: String,
    pub top_k: usize,
    pub sensitivity_threshold: String,
    pub policy: RagPolicy,
    pub tokenizer: TokenizerKind,
    pub tokenizer_name: String,
}

pub struct RagAnswer {
    pub answer: String,
    pub used_cells: Vec<RagUsedCell>,
    pub metrics: RagMetrics,
    pub response: LlmResponse,
    pub context_snippet: String,
}

pub struct RagUsedCell {
    pub document_id: i64,
    pub document_source: String,
    pub page: i64,
    pub score: f32,
    pub text: String,
    pub sensitivity: String,
}

pub struct RagMetrics {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub raw_tokens_estimate: u32,
    pub compressed_tokens_estimate: u32,
    pub compression_factor: f32,
}

pub fn execute_rag_query(
    store: &RagStore,
    embeddings: &EmbeddingClient,
    client: &LlmClient,
    query: &RagQuery,
    decrypt_identity: Option<&Path>,
) -> Result<RagAnswer> {
    let question_embedding = embeddings.embed(&query.question)?;
    let filters = SearchFilters {
        top_k: query.top_k,
        sensitivity_threshold: query.sensitivity_threshold.clone(),
        policy: query.policy,
    };
    let cells = store.search_cells(&query.collection, &question_embedding, &filters)?;
    if cells.is_empty() {
        return Err(anyhow!("no cells matched the query"));
    }
    let mut used_cells = Vec::new();
    let mut context = String::new();
    context.push_str("You are a helpful assistant. Answer only with the information in CONTEXT.\n\n=== CONTEXT START ===\n");
    for cell in &cells {
        if let Some(text) = resolve_cell_text(cell, decrypt_identity)? {
            context.push_str(&format!(
                "[DOC: {}, page {}]\n{}\n\n",
                cell.document_source, cell.page, text
            ));
            used_cells.push(RagUsedCell {
                document_id: cell.document_id,
                document_source: cell.document_source.clone(),
                page: cell.page,
                score: cell.score,
                text: text.clone(),
                sensitivity: cell.sensitivity.clone(),
            });
        }
    }
    context.push_str("=== CONTEXT END ===\n\n");
    let prompt = format!("{}Question: {}\nAnswer:", context, query.question.trim());
    let response = client.chat_blocking(&LlmRequest {
        system: None,
        user: prompt.clone(),
    })?;
    let tokenizer = query
        .tokenizer
        .build()
        .map_err(|e| anyhow!(e.to_string()))?;
    let raw_tokens = estimate_tokens(
        &format!("{}\n{}", context, query.question),
        &query.tokenizer,
    )? as u32;
    let compressed_tokens = tokenizer.encode_with_special_tokens(prompt.as_str()).len() as u32;
    let metrics = RagMetrics {
        prompt_tokens: response.prompt_tokens,
        completion_tokens: response.completion_tokens,
        raw_tokens_estimate: raw_tokens,
        compressed_tokens_estimate: compressed_tokens,
        compression_factor: if compressed_tokens == 0 {
            0.0
        } else {
            raw_tokens as f32 / compressed_tokens as f32
        },
    };
    Ok(RagAnswer {
        answer: response.content.clone(),
        used_cells,
        metrics,
        response,
        context_snippet: context,
    })
}

fn resolve_cell_text(cell: &ScoredCell, identity: Option<&Path>) -> Result<Option<String>> {
    if let Some(text) = &cell.text {
        return Ok(Some(text.clone()));
    }
    if let Some(cipher) = &cell.text_encrypted {
        if let Some(identity_path) = identity {
            return decrypt_text(cipher, identity_path).map(Some);
        } else {
            return Ok(None);
        }
    }
    Ok(None)
}
