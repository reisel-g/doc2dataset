use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{Multipart, Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::{routing::get, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use thiserror::Error;
use tokio::task;
use tracing::{error, info};

use three_dcf_core::{
    estimate_tokens, Encoder, Metrics, TableMode, TextSerializer, TextSerializerConfig,
    TokenizerKind,
};
use three_dcf_rag::{
    encryption, execute_rag_query, normalize_level, CellInsert, DocumentInsert, EmbeddingClient,
    LlmClient, LlmProvider, PricingConfig, RagPolicy, RagQuery, RagStore,
};

#[derive(Clone)]
struct AppState {
    store: RagStore,
    embed_client: EmbeddingClient,
    pricing: PricingConfig,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt::init();
    let store_path = std::env::var("RAG_DB").unwrap_or_else(|_| "rag.sqlite".to_string());
    let store = RagStore::open(&store_path)?;
    let embed_client = EmbeddingClient::from_env().unwrap_or_else(|_| EmbeddingClient::hash());
    let pricing = load_pricing_config();
    let state = Arc::new(AppState {
        store,
        embed_client,
        pricing,
    });
    let app = Router::new()
        .route("/", get(serve_ui))
        .route("/encode", post(handle_encode))
        .route(
            "/rag/collections/:name/documents",
            post(handle_rag_document),
        )
        .route("/rag/query", post(handle_rag_query))
        .with_state(state);
    let addr: SocketAddr = std::env::var("BIND_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8000".to_string())
        .parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("listening" = %addr);
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct EncodeParams {
    preset: Option<String>,
    budget: Option<usize>,
    tokenizer: Option<String>,
}

#[derive(Debug, Serialize)]
struct EncodeResponse {
    context_text: String,
    metrics: ServiceMetrics,
}

#[derive(Debug, Serialize)]
struct ServiceMetrics {
    pages: u32,
    lines_total: u32,
    cells_total: u32,
    cells_kept: u32,
    dedup_ratio: f32,
    numguard_count: u32,
    raw_tokens_estimate: Option<u32>,
    compressed_tokens_estimate: Option<u32>,
    compression_factor: Option<f32>,
}

#[derive(Debug, Deserialize, Clone)]
struct RagDocumentParams {
    preset: Option<String>,
    budget: Option<usize>,
    sensitivity: Option<String>,
    encrypt: Option<bool>,
}

#[derive(Debug, Serialize)]
struct RagDocumentResponse {
    collection: String,
    document_id: i64,
    pages: usize,
    cells_indexed: usize,
}

#[derive(Debug, Deserialize)]
struct RagQueryRequest {
    collection: String,
    question: String,
    top_k: Option<usize>,
    provider: Option<String>,
    model: Option<String>,
    sensitivity_threshold: Option<String>,
    policy: Option<String>,
    tokenizer: Option<String>,
}

#[derive(Debug, Serialize)]
struct RagQueryResponse {
    answer: String,
    used_cells: Vec<RagCellResponse>,
    metrics: RagQueryMetrics,
}

#[derive(Debug, Serialize)]
struct RagCellResponse {
    document_id: i64,
    document_source: String,
    page: i64,
    score: f32,
    text: String,
    sensitivity: String,
}

#[derive(Debug, Serialize)]
struct RagQueryMetrics {
    prompt_tokens: u32,
    completion_tokens: u32,
    raw_tokens_estimate: u32,
    compressed_tokens_estimate: u32,
    compression_factor: f32,
    estimated_saving_usd: Option<f64>,
}

impl From<Metrics> for ServiceMetrics {
    fn from(value: Metrics) -> Self {
        Self {
            pages: value.pages,
            lines_total: value.lines_total,
            cells_total: value.cells_total,
            cells_kept: value.cells_kept,
            dedup_ratio: value.dedup_ratio,
            numguard_count: value.numguard_count,
            raw_tokens_estimate: value.raw_tokens_estimate,
            compressed_tokens_estimate: value.compressed_tokens_estimate,
            compression_factor: value.compression_factor,
        }
    }
}

async fn handle_encode(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<EncodeParams>,
    mut multipart: Multipart,
) -> Result<Json<EncodeResponse>, AppError> {
    let UploadedFile {
        data: file_bytes, ..
    } = extract_file(&mut multipart).await?;
    let preset = params.preset.as_deref().unwrap_or("reports").to_string();
    let tokenizer_name = params
        .tokenizer
        .clone()
        .unwrap_or_else(|| "cl100k_base".to_string());
    let result = task::spawn_blocking(move || {
        run_encode(&file_bytes, &preset, params.budget, &tokenizer_name)
    })
    .await
    .map_err(AppError::internal)??;
    Ok(Json(result))
}

async fn handle_rag_document(
    State(state): State<Arc<AppState>>,
    AxumPath(collection): AxumPath<String>,
    Query(params): Query<RagDocumentParams>,
    mut multipart: Multipart,
) -> Result<Json<RagDocumentResponse>, AppError> {
    let uploaded = extract_file(&mut multipart).await?;
    let state = state.clone();
    let collection_clone = collection.clone();
    let params_clone = params.clone();
    let result = task::spawn_blocking(move || {
        ingest_document(&state, &collection_clone, uploaded, &params_clone)
    })
    .await
    .map_err(AppError::internal)??;
    Ok(Json(result))
}

async fn handle_rag_query(
    State(state): State<Arc<AppState>>,
    Json(body): Json<RagQueryRequest>,
) -> Result<Json<RagQueryResponse>, AppError> {
    let state = state.clone();
    let response = task::spawn_blocking(move || execute_rag_request(&state, body))
        .await
        .map_err(AppError::internal)??;
    Ok(Json(response))
}

fn run_encode(
    bytes: &[u8],
    preset: &str,
    budget: Option<usize>,
    tokenizer_name: &str,
) -> Result<EncodeResponse, AppError> {
    let mut tmp = NamedTempFile::new().map_err(AppError::internal)?;
    std::io::Write::write_all(&mut tmp, bytes).map_err(AppError::internal)?;
    let encoder = build_service_encoder(preset, budget)?;
    let (doc, mut metrics, raw_text) = encoder
        .encode_path_with_plaintext(tmp.path())
        .map_err(AppError::internal)?;
    let serializer = TextSerializer::with_config(TextSerializerConfig {
        table_mode: TableMode::Auto,
        ..Default::default()
    });
    let context_text = serializer.to_string(&doc).map_err(AppError::internal)?;
    let tokenizer = resolve_tokenizer(tokenizer_name)?;
    let raw_tokens = estimate_tokens(&raw_text, &tokenizer).map_err(AppError::internal)? as u32;
    let compressed_tokens =
        estimate_tokens(context_text.as_str(), &tokenizer).map_err(AppError::internal)? as u32;
    metrics.record_tokens(Some(raw_tokens), Some(compressed_tokens));
    Ok(EncodeResponse {
        context_text,
        metrics: metrics.into(),
    })
}

fn build_service_encoder(preset: &str, budget: Option<usize>) -> Result<Encoder, AppError> {
    let mut builder = Encoder::builder(preset).map_err(AppError::internal)?;
    if let Some(b) = budget {
        builder = builder.budget(Some(b));
    }
    Ok(builder.build())
}

fn resolve_tokenizer(name: &str) -> Result<TokenizerKind, AppError> {
    match name.to_lowercase().as_str() {
        "cl100k" | "cl100k_base" => Ok(TokenizerKind::Cl100k),
        "o200k" | "o200k_base" => Ok(TokenizerKind::O200k),
        "gpt2" | "p50k" | "p50k_base" => Ok(TokenizerKind::Gpt2),
        "anthropic" => Ok(TokenizerKind::Anthropic),
        other => Err(AppError::bad_request(format!("unknown tokenizer {other}"))),
    }
}

struct UploadedFile {
    data: Vec<u8>,
    filename: Option<String>,
}

async fn extract_file(multipart: &mut Multipart) -> Result<UploadedFile, AppError> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(AppError::bad_request)?
    {
        if let Some(name) = field.name() {
            if name == "file" {
                let filename = field.file_name().map(|s| s.to_string());
                let data = field.bytes().await.map_err(AppError::bad_request)?;
                return Ok(UploadedFile {
                    data: data.to_vec(),
                    filename,
                });
            }
        }
    }
    Err(AppError::bad_request("missing file"))
}

fn ingest_document(
    state: &AppState,
    collection: &str,
    upload: UploadedFile,
    params: &RagDocumentParams,
) -> Result<RagDocumentResponse, AppError> {
    let collection_id = state
        .store
        .ensure_collection(collection)
        .map_err(AppError::internal)?;
    let mut tmp = NamedTempFile::new().map_err(AppError::internal)?;
    std::io::Write::write_all(&mut tmp, &upload.data).map_err(AppError::internal)?;
    let encoder =
        build_service_encoder(params.preset.as_deref().unwrap_or("reports"), params.budget)?;
    let (doc, _) = encoder
        .encode_path(tmp.path())
        .map_err(AppError::internal)?;
    let ordered = doc.ordered_cells();
    let doc_record = state
        .store
        .add_document(
            collection_id,
            &DocumentInsert {
                source_path: upload
                    .filename
                    .clone()
                    .unwrap_or_else(|| format!("{collection}/upload")),
                dcf_path: None,
                title: None,
            },
        )
        .map_err(AppError::internal)?;
    let sensitivity = normalize_level(params.sensitivity.as_deref().unwrap_or("public"));
    let recipient = if params.encrypt.unwrap_or(false) {
        Some(std::env::var("RAG_ENCRYPTION_RECIPIENT").map_err(|_| {
            AppError::bad_request("RAG_ENCRYPTION_RECIPIENT must be set when encrypt=true")
        })?)
    } else {
        None
    };
    let mut texts = Vec::with_capacity(ordered.len());
    for cell in &ordered {
        texts.push(doc.payload_for(&cell.code_id).unwrap_or("").to_string());
    }
    let embeddings = state
        .embed_client
        .embed_batch(&texts)
        .map_err(AppError::internal)?;
    let mut cells = Vec::with_capacity(ordered.len());
    for ((cell, text), embedding) in ordered.iter().zip(texts.iter()).zip(embeddings.into_iter()) {
        let (plain, encrypted, enc_label) = if let Some(ref recipient) = recipient {
            if text.is_empty() {
                (None, None, None)
            } else {
                let cipher =
                    encryption::encrypt_text(text, recipient).map_err(AppError::internal)?;
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
        cells.push(CellInsert {
            page: cell.z,
            importance: cell.importance,
            sensitivity: sensitivity.clone(),
            text: plain,
            text_encrypted: encrypted,
            encryption: enc_label,
            embedding,
            bbox_x: cell.x,
            bbox_y: cell.y,
            bbox_w: cell.w,
            bbox_h: cell.h,
        });
    }
    state
        .store
        .add_cells(doc_record.id, &cells)
        .map_err(AppError::internal)?;
    Ok(RagDocumentResponse {
        collection: collection.to_string(),
        document_id: doc_record.id,
        pages: doc.total_pages(),
        cells_indexed: cells.len(),
    })
}

fn execute_rag_request(
    state: &AppState,
    body: RagQueryRequest,
) -> Result<RagQueryResponse, AppError> {
    let llm_client = build_provider(
        body.provider.unwrap_or_else(|| "openai".to_string()),
        body.model,
    )?;
    let tokenizer_name = body
        .tokenizer
        .clone()
        .unwrap_or_else(|| "cl100k_base".to_string());
    let tokenizer_kind = resolve_tokenizer(&tokenizer_name)?;
    let policy = parse_policy(body.policy.as_deref());
    let sensitivity = normalize_level(body.sensitivity_threshold.as_deref().unwrap_or("public"));
    let decrypt_identity = if matches!(policy, RagPolicy::Internal) {
        Some(
            std::env::var("RAG_DECRYPT_IDENTITY")
                .map(PathBuf::from)
                .map_err(|_| {
                    AppError::bad_request("RAG_DECRYPT_IDENTITY must be set for internal policy")
                })?,
        )
    } else {
        None
    };
    let rag_query = RagQuery {
        collection: body.collection.clone(),
        question: body.question.clone(),
        top_k: body.top_k.unwrap_or(10),
        sensitivity_threshold: sensitivity,
        policy,
        tokenizer: tokenizer_kind.clone(),
        tokenizer_name: tokenizer_name.clone(),
    };
    let answer = execute_rag_query(
        &state.store,
        &state.embed_client,
        &llm_client,
        &rag_query,
        decrypt_identity.as_deref(),
    )
    .map_err(AppError::internal)?;
    let provider_key = provider_key(llm_client.provider());
    let rate = state.pricing.lookup(provider_key, llm_client.model());
    let estimated_saving = rate.map(|pricing| {
        let prompt_cost = answer.response.prompt_tokens as f64 / 1000.0 * pricing.prompt_per_1k;
        let raw_cost = answer.metrics.raw_tokens_estimate as f64 / 1000.0 * pricing.prompt_per_1k;
        (raw_cost - prompt_cost).max(0.0)
    });
    let used_cells = answer
        .used_cells
        .iter()
        .map(|cell| RagCellResponse {
            document_id: cell.document_id,
            document_source: cell.document_source.clone(),
            page: cell.page,
            score: cell.score,
            text: cell.text.clone(),
            sensitivity: cell.sensitivity.clone(),
        })
        .collect();
    let metrics = RagQueryMetrics {
        prompt_tokens: answer.response.prompt_tokens,
        completion_tokens: answer.response.completion_tokens,
        raw_tokens_estimate: answer.metrics.raw_tokens_estimate,
        compressed_tokens_estimate: answer.metrics.compressed_tokens_estimate,
        compression_factor: answer.metrics.compression_factor,
        estimated_saving_usd: estimated_saving,
    };
    Ok(RagQueryResponse {
        answer: answer.answer,
        used_cells,
        metrics,
    })
}

fn build_provider(name: String, model: Option<String>) -> Result<LlmClient, AppError> {
    let provider = LlmProvider::from_str(&name)
        .ok_or_else(|| AppError::bad_request(format!("unknown provider {name}")))?;
    let model_name = model.unwrap_or_else(|| default_llm_model(provider).to_string());
    LlmClient::new(provider, model_name).map_err(AppError::internal)
}

fn parse_policy(value: Option<&str>) -> RagPolicy {
    match value.unwrap_or("external").to_lowercase().as_str() {
        "internal" => RagPolicy::Internal,
        _ => RagPolicy::External,
    }
}

fn provider_key(provider: LlmProvider) -> &'static str {
    match provider {
        LlmProvider::OpenAi => "openai",
        LlmProvider::Anthropic => "anthropic",
        LlmProvider::Gemini => "gemini",
        LlmProvider::Deepseek => "deepseek",
        LlmProvider::Local => "local",
    }
}

fn default_llm_model(provider: LlmProvider) -> &'static str {
    match provider {
        LlmProvider::OpenAi => "gpt-4.1-mini",
        LlmProvider::Anthropic => "claude-3-5-sonnet",
        LlmProvider::Gemini => "gemini-1.5-flash",
        LlmProvider::Deepseek => "deepseek-chat",
        LlmProvider::Local => "local",
    }
}

fn load_pricing_config() -> PricingConfig {
    let config_path = std::env::var("THREE_DCF_CONFIG").unwrap_or_else(|_| "3dcf.toml".to_string());
    let path = Path::new(&config_path);
    if !path.exists() {
        return PricingConfig::default();
    }
    match fs::read_to_string(path) {
        Ok(contents) => toml::from_str::<ServiceConfig>(&contents)
            .map(|cfg| cfg.pricing)
            .unwrap_or_default(),
        Err(_) => PricingConfig::default(),
    }
}

#[derive(Debug, Default, Deserialize)]
struct ServiceConfig {
    #[serde(default)]
    pricing: PricingConfig,
}

async fn serve_ui() -> Html<&'static str> {
    Html(include_str!("../../../ui/index.html"))
}

#[derive(Debug, Error)]
enum AppError {
    #[error("{0}")]
    BadRequest(String),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl AppError {
    fn bad_request<E: ToString>(msg: E) -> Self {
        Self::BadRequest(msg.to_string())
    }

    fn internal<E: Into<anyhow::Error>>(err: E) -> Self {
        Self::Internal(err.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        match self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg).into_response(),
            AppError::Internal(err) => {
                error!("internal_error" = %err);
                (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
            }
        }
    }
}
