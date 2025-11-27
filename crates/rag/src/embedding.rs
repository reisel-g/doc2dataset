use anyhow::{anyhow, Result};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::env;

use three_dcf_core::{HashEmbedder, HashEmbedderConfig};

#[derive(Clone)]
pub enum EmbeddingBackend {
    Hash(HashEmbedder),
    OpenAi(OpenAiEmbeddingClient),
}

#[derive(Clone)]
pub struct EmbeddingClient {
    backend: EmbeddingBackend,
}

impl EmbeddingClient {
    pub fn from_env() -> Result<Self> {
        match env::var("EMBEDDING_PROVIDER")
            .unwrap_or_else(|_| "hash".to_string())
            .to_lowercase()
            .as_str()
        {
            "openai" => {
                let model = env::var("EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string());
                Ok(Self {
                    backend: EmbeddingBackend::OpenAi(OpenAiEmbeddingClient::new(&model)?),
                })
            }
            _ => {
                let dims = env::var("HASH_EMBED_DIMENSIONS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(64);
                Ok(Self {
                    backend: EmbeddingBackend::Hash(HashEmbedder::new(HashEmbedderConfig {
                        dimensions: dims,
                        seed: 1337,
                    })),
                })
            }
        }
    }

    pub fn hash() -> Self {
        Self {
            backend: EmbeddingBackend::Hash(HashEmbedder::new(HashEmbedderConfig::default())),
        }
    }

    pub fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        match &self.backend {
            EmbeddingBackend::Hash(embedder) => Ok(inputs
                .iter()
                .map(|text| embedder.embed_text(text))
                .collect()),
            EmbeddingBackend::OpenAi(client) => client.embed_batch(inputs),
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let inputs = vec![text.to_string()];
        let mut output = self.embed_batch(&inputs)?;
        Ok(output.pop().unwrap_or_default())
    }
}

#[derive(Clone)]
pub struct OpenAiEmbeddingClient {
    http: Client,
    model: String,
    api_key: String,
}

impl OpenAiEmbeddingClient {
    pub fn new(model: &str) -> Result<Self> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow!("OPENAI_API_KEY is required for openai embeddings"))?;
        Ok(Self {
            http: Client::new(),
            model: model.to_string(),
            api_key,
        })
    }

    pub fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let url = "https://api.openai.com/v1/embeddings";
        let payload = serde_json::json!({
            "model": self.model,
            "input": inputs,
        });
        let response = self
            .http
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "openai embeddings request failed: {}",
                response.status()
            ));
        }
        let parsed: OpenAiEmbeddingResponse = response.json()?;
        let mut out = Vec::new();
        for data in parsed.data {
            out.push(data.embedding);
        }
        Ok(out)
    }
}

#[derive(Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f32>,
}
