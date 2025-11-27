use anyhow::{anyhow, Context, Result};
use reqwest::{header::HeaderValue, Client, StatusCode};
use serde::Deserialize;
use serde_json::{json, Value};
use std::env;
use tokio::runtime::Runtime;
use tokio::time::{sleep, Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmProvider {
    OpenAi,
    Anthropic,
    Gemini,
    Deepseek,
    Local,
}

impl LlmProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            LlmProvider::OpenAi => "openai",
            LlmProvider::Anthropic => "anthropic",
            LlmProvider::Gemini => "gemini",
            LlmProvider::Deepseek => "deepseek",
            LlmProvider::Local => "local",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.to_lowercase().as_str() {
            "openai" => Some(LlmProvider::OpenAi),
            "anthropic" => Some(LlmProvider::Anthropic),
            "gemini" => Some(LlmProvider::Gemini),
            "deepseek" => Some(LlmProvider::Deepseek),
            "local" => Some(LlmProvider::Local),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LlmRequest {
    pub system: Option<String>,
    pub user: String,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl LlmResponse {
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

#[derive(Clone)]
pub struct LlmClient {
    http: Client,
    provider: LlmProvider,
    model: String,
    config: ProviderConfig,
}

#[derive(Clone)]
enum ProviderConfig {
    OpenAi(OpenAiConfig),
    Anthropic(AnthropicConfig),
    Gemini(GeminiConfig),
    Deepseek(DeepseekConfig),
    Local,
}

#[derive(Clone)]
struct OpenAiConfig {
    api_key: String,
    base_url: String,
}

#[derive(Clone)]
struct AnthropicConfig {
    api_key: String,
    max_tokens: u32,
}

#[derive(Clone)]
struct GeminiConfig {
    api_key: String,
}

#[derive(Clone)]
struct DeepseekConfig {
    api_key: String,
}

impl LlmClient {
    pub fn new(provider: LlmProvider, model: impl Into<String>) -> Result<Self> {
        let model = model.into();
        let http = Client::new();
        let config = match provider {
            LlmProvider::OpenAi => ProviderConfig::OpenAi(OpenAiConfig {
                api_key: read_api_key("OPENAI_API_KEY")?,
                base_url: env::var("OPENAI_BASE_URL")
                    .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            }),
            LlmProvider::Anthropic => ProviderConfig::Anthropic(AnthropicConfig {
                api_key: read_api_key("ANTHROPIC_API_KEY")?,
                max_tokens: env::var("ANTHROPIC_MAX_TOKENS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(512),
            }),
            LlmProvider::Gemini => ProviderConfig::Gemini(GeminiConfig {
                api_key: read_api_key("GEMINI_API_KEY")?,
            }),
            LlmProvider::Deepseek => ProviderConfig::Deepseek(DeepseekConfig {
                api_key: read_api_key("DEEPSEEK_API_KEY")?,
            }),
            LlmProvider::Local => ProviderConfig::Local,
        };
        Ok(Self {
            http,
            provider,
            model,
            config,
        })
    }

    pub fn provider(&self) -> LlmProvider {
        self.provider
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn chat(&self, req: &LlmRequest) -> Result<LlmResponse> {
        match &self.config {
            ProviderConfig::OpenAi(cfg) => self.chat_openai(cfg, req).await,
            ProviderConfig::Anthropic(cfg) => self.chat_anthropic(cfg, req).await,
            ProviderConfig::Gemini(cfg) => self.chat_gemini(cfg, req).await,
            ProviderConfig::Deepseek(cfg) => self.chat_deepseek(cfg, req).await,
            ProviderConfig::Local => Ok(self.chat_local(req)),
        }
    }

    pub fn chat_blocking(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let rt = Runtime::new().context("failed to create tokio runtime")?;
        rt.block_on(self.chat(req))
    }

    async fn chat_openai(&self, cfg: &OpenAiConfig, req: &LlmRequest) -> Result<LlmResponse> {
        if openai_uses_responses(&self.model) {
            return self.chat_openai_responses(cfg, req).await;
        }
        self.chat_openai_chat(cfg, req).await
    }

    async fn chat_openai_chat(&self, cfg: &OpenAiConfig, req: &LlmRequest) -> Result<LlmResponse> {
        const MAX_RETRIES: usize = 6;
        let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
        let mut messages = Vec::new();
        if let Some(system) = &req.system {
            messages.push(json!({"role": "system", "content": system }));
        }
        messages.push(json!({"role": "user", "content": req.user }));
        let payload = json!({
            "model": self.model,
            "messages": messages,
        });
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            let response = match self
                .http
                .post(&url)
                .bearer_auth(&cfg.api_key)
                .json(&payload)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(err) => {
                    if attempt > MAX_RETRIES {
                        return Err(err).with_context(|| "openai request failed");
                    }
                    sleep(backoff_delay(attempt, None)).await;
                    continue;
                }
            };
            if response.status() == StatusCode::TOO_MANY_REQUESTS {
                if attempt > MAX_RETRIES {
                    return Err(anyhow!("openai rate limited after {MAX_RETRIES} retries"));
                }
                let wait = backoff_delay(attempt, response.headers().get("retry-after"));
                sleep(wait).await;
                continue;
            }
            let value = decode_openai_body(response).await?;
            let content = extract_openai_text(&value)
                .ok_or_else(|| anyhow!("missing text in OpenAI response"))?;
            let usage: OpenAiUsage = value
                .get("usage")
                .and_then(|value| serde_json::from_value(value.clone()).ok())
                .unwrap_or_default();
            return Ok(LlmResponse {
                content,
                prompt_tokens: usage.prompt_tokens.unwrap_or(0),
                completion_tokens: usage.completion_tokens.unwrap_or(0),
            });
        }
    }

    async fn chat_openai_responses(
        &self,
        cfg: &OpenAiConfig,
        req: &LlmRequest,
    ) -> Result<LlmResponse> {
        const MAX_RETRIES: usize = 6;
        let url = format!("{}/responses", cfg.base_url.trim_end_matches('/'));
        let mut input = Vec::new();
        if let Some(system) = &req.system {
            input.push(json!({
                "role": "system",
                "content": [{ "type": "input_text", "text": system }],
            }));
        }
        input.push(json!({
            "role": "user",
            "content": [{ "type": "input_text", "text": req.user }],
        }));
        let payload = json!({
            "model": self.model,
            "input": input,
        });
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            let response = match self
                .http
                .post(&url)
                .bearer_auth(&cfg.api_key)
                .json(&payload)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(err) => {
                    if attempt > MAX_RETRIES {
                        return Err(err).with_context(|| "openai request failed");
                    }
                    sleep(backoff_delay(attempt, None)).await;
                    continue;
                }
            };
            if response.status() == StatusCode::TOO_MANY_REQUESTS {
                if attempt > MAX_RETRIES {
                    return Err(anyhow!("openai rate limited after {MAX_RETRIES} retries"));
                }
                let wait = backoff_delay(attempt, response.headers().get("retry-after"));
                sleep(wait).await;
                continue;
            }
            let value = decode_openai_body(response).await?;
            let content = extract_openai_text(&value)
                .ok_or_else(|| anyhow!("missing text in OpenAI response"))?;
            let (prompt_tokens, completion_tokens) = parse_responses_usage(&value);
            return Ok(LlmResponse {
                content,
                prompt_tokens,
                completion_tokens,
            });
        }
    }

    async fn chat_anthropic(&self, cfg: &AnthropicConfig, req: &LlmRequest) -> Result<LlmResponse> {
        let mut payload = json!({
            "model": self.model,
            "max_tokens": cfg.max_tokens,
            "messages": [ { "role": "user", "content": req.user } ],
        });
        if let Some(system) = &req.system {
            payload["system"] = json!(system);
        }
        let response = self
            .http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &cfg.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&payload)
            .send()
            .await
            .with_context(|| "anthropic request failed")?
            .error_for_status()
            .context("anthropic returned an error")?
            .json::<AnthropicResponse>()
            .await
            .context("failed to decode anthropic response")?;
        let text = response
            .content
            .into_iter()
            .find_map(|part| part.text)
            .ok_or_else(|| anyhow!("missing text in Anthropic response"))?;
        let usage = response.usage.unwrap_or_default();
        Ok(LlmResponse {
            content: text,
            prompt_tokens: usage.input_tokens.unwrap_or(0),
            completion_tokens: usage.output_tokens.unwrap_or(0),
        })
    }

    async fn chat_gemini(&self, cfg: &GeminiConfig, req: &LlmRequest) -> Result<LlmResponse> {
        let mut prompt = String::new();
        if let Some(system) = &req.system {
            prompt.push_str("[SYSTEM]\n");
            prompt.push_str(system.trim());
            prompt.push_str("\n\n");
        }
        prompt.push_str(&req.user);
        let payload = json!({
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        });
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, cfg.api_key
        );
        let response = self
            .http
            .post(url)
            .json(&payload)
            .send()
            .await
            .with_context(|| "gemini request failed")?
            .error_for_status()
            .context("gemini returned an error")?
            .json::<GeminiResponse>()
            .await
            .context("failed to decode gemini response")?;
        let text = response
            .candidates
            .and_then(|mut c| c.pop())
            .and_then(|candidate| {
                candidate
                    .content
                    .parts
                    .into_iter()
                    .find_map(|part| part.text)
            })
            .ok_or_else(|| anyhow!("missing text in Gemini response"))?;
        let usage = response.usage.unwrap_or_default();
        Ok(LlmResponse {
            content: text,
            prompt_tokens: usage.prompt_tokens.unwrap_or(0),
            completion_tokens: usage.completion_tokens.unwrap_or(0),
        })
    }

    async fn chat_deepseek(&self, cfg: &DeepseekConfig, req: &LlmRequest) -> Result<LlmResponse> {
        let mut messages = Vec::new();
        if let Some(system) = &req.system {
            messages.push(json!({ "role": "system", "content": system }));
        }
        messages.push(json!({ "role": "user", "content": req.user }));
        let payload = json!({
            "model": self.model,
            "messages": messages,
        });
        let response = self
            .http
            .post("https://api.deepseek.com/v1/chat/completions")
            .bearer_auth(&cfg.api_key)
            .json(&payload)
            .send()
            .await
            .with_context(|| "deepseek request failed")?
            .error_for_status()
            .context("deepseek returned an error")?
            .json::<ChatResponse>()
            .await
            .context("failed to decode deepseek response")?;
        let text = response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or_else(|| anyhow!("missing text in DeepSeek response"))?;
        let usage = response.usage.unwrap_or_default();
        Ok(LlmResponse {
            content: text,
            prompt_tokens: usage.prompt_tokens.unwrap_or(0),
            completion_tokens: usage.completion_tokens.unwrap_or(0),
        })
    }

    fn chat_local(&self, req: &LlmRequest) -> LlmResponse {
        let content = synthesize_local_response(req);
        LlmResponse {
            content,
            prompt_tokens: 0,
            completion_tokens: 0,
        }
    }
}

fn backoff_delay(attempt: usize, retry_after: Option<&HeaderValue>) -> Duration {
    if let Some(value) = retry_after {
        if let Ok(text) = value.to_str() {
            if let Ok(secs) = text.parse::<u64>() {
                return Duration::from_secs(secs.max(1));
            }
        }
    }
    let capped = attempt.min(6) as u32;
    Duration::from_secs(1u64 << capped)
}

fn synthesize_local_response(req: &LlmRequest) -> String {
    let user_lower = req.user.to_lowercase();
    if user_lower.contains("generate a helpful question") {
        let context = extract_context_block(
            &req.user,
            "Here is a fragment of a document:",
            "Generate a helpful question",
        );
        let snippet = summarize_text(&context, 60);
        return format!(
            "Question: What key facts are stated in the excerpt?\nAnswer: {}",
            snippet
        );
    }
    if user_lower.contains("write a concise summary") {
        let body = extract_context_block(&req.user, "Heading:", "Language:");
        return summarize_text(&body, 80);
    }
    summarize_text(&req.user, 40)
}

fn openai_uses_responses(model: &str) -> bool {
    let lower = model.to_lowercase();
    lower.starts_with("gpt-4.1") || lower.starts_with("gpt-4o") || lower.starts_with("o1")
}

fn parse_responses_usage(value: &Value) -> (u32, u32) {
    if let Some(usage) = value.get("usage") {
        let prompt = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let completion = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        return (prompt, completion);
    }
    (0, 0)
}

async fn decode_openai_body(response: reqwest::Response) -> Result<Value> {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!(format!(
            "openai returned error (status {}): {}",
            status, body
        )));
    }
    serde_json::from_str(&body).context("failed to decode openai response")
}

fn extract_context_block(text: &str, start_marker: &str, stop_marker: &str) -> String {
    if let Some(start_idx) = text.find(start_marker) {
        let after = &text[start_idx + start_marker.len()..];
        let after_lower = after.to_lowercase();
        let stop_lower = stop_marker.to_lowercase();
        if let Some(end_idx) = after_lower.find(&stop_lower) {
            let (segment, _) = after.split_at(end_idx);
            return segment.trim().to_string();
        }
        return after.trim().to_string();
    }
    text.trim().to_string()
}

fn summarize_text(text: &str, max_words: usize) -> String {
    if max_words == 0 {
        return String::new();
    }
    let cleaned = text
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<&str>>()
        .join(" ");
    cleaned
        .split_whitespace()
        .take(max_words)
        .collect::<Vec<&str>>()
        .join(" ")
}

fn read_api_key(var: &str) -> Result<String> {
    let value = env::var(var).map_err(|_| anyhow!(format!("{var} is not set")))?;
    validate_api_key(var, &value)?;
    Ok(value)
}

fn validate_api_key(var: &str, value: &str) -> Result<()> {
    if var.contains("OPENAI") && !value.starts_with("sk-") {
        return Err(anyhow!(format!(
            "{} must start with 'sk-' (see https://platform.openai.com/)",
            var
        )));
    }
    if var.contains("ANTHROPIC") && !value.starts_with("sk-ant-") {
        return Err(anyhow!(format!("{} must start with 'sk-ant-'", var)));
    }
    if var.contains("DEEPSEEK") && !value.starts_with("sk-") {
        return Err(anyhow!(format!("{} must start with 'sk-'", var)));
    }
    if var.contains("GEMINI") && !value.starts_with("AI") {
        return Err(anyhow!(format!(
            "{} must be a valid Gemini API key (starts with 'AI...')",
            var
        )));
    }
    Ok(())
}

fn extract_openai_text(value: &Value) -> Option<String> {
    if let Some(outputs) = value.get("output").and_then(|v| v.as_array()) {
        for output in outputs {
            if let Some(content) = output.get("content").and_then(|v| v.as_array()) {
                for block in content {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        return Some(text.to_string());
                    }
                }
            }
        }
    }
    if let Some(texts) = value.get("output_text").and_then(|v| v.as_array()) {
        if let Some(text) = texts.first().and_then(|t| t.as_str()) {
            return Some(text.to_string());
        }
    }
    if let Some(choices) = value.get("choices").and_then(|v| v.as_array()) {
        if let Some(choice) = choices.first() {
            if let Some(text) = choice.get("text").and_then(|t| t.as_str()) {
                return Some(text.to_string());
            }
            if let Some(message) = choice.get("message") {
                if let Some(content) = message.get("content") {
                    if let Some(text) = content.as_str() {
                        return Some(text.to_string());
                    }
                    if let Some(parts) = content.as_array() {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                return Some(text.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

#[derive(Default, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: Option<String>,
}

#[derive(Default, Deserialize)]
struct AnthropicUsage {
    #[serde(rename = "input_tokens")]
    input_tokens: Option<u32>,
    #[serde(rename = "output_tokens")]
    output_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage: Option<GeminiUsage>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Deserialize)]
struct GeminiPart {
    text: Option<String>,
}

#[derive(Default, Deserialize)]
struct GeminiUsage {
    #[serde(rename = "promptTokenCount")]
    prompt_tokens: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    completion_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: String,
}
