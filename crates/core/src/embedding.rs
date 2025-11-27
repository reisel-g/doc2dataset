use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::document::CellType;

#[derive(Debug, Clone, Copy)]
pub struct HashEmbedderConfig {
    pub dimensions: usize,
    pub seed: u64,
}

impl Default for HashEmbedderConfig {
    fn default() -> Self {
        Self {
            dimensions: 64,
            seed: 1337,
        }
    }
}

#[derive(Clone)]
pub struct HashEmbedder {
    config: HashEmbedderConfig,
}

impl HashEmbedder {
    pub fn new(config: HashEmbedderConfig) -> Self {
        Self { config }
    }

    pub fn embed_text(&self, text: &str) -> Vec<f32> {
        let dims = self.config.dimensions.max(1);
        let mut vector = vec![0f32; dims];
        for token in text.split_whitespace() {
            let bucket = self.bucket_for(token);
            vector[bucket] += 1.0;
        }
        normalize(&mut vector);
        vector
    }

    fn bucket_for(&self, token: &str) -> usize {
        let mut hasher = DefaultHasher::new();
        hasher.write_u64(self.config.seed);
        token.to_lowercase().hash(&mut hasher);
        (hasher.finish() as usize) % self.config.dimensions.max(1)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRecord {
    pub chunk_id: String,
    pub doc: String,
    pub chunk_index: usize,
    #[serde(default)]
    pub z_start: u32,
    #[serde(default)]
    pub z_end: u32,
    #[serde(default)]
    pub cell_start: usize,
    #[serde(default)]
    pub cell_end: usize,
    #[serde(default)]
    pub token_count: usize,
    #[serde(default = "default_cell_type")]
    pub dominant_type: CellType,
    #[serde(default)]
    pub importance_mean: f32,
    pub embedding: Vec<f32>,
    pub text: String,
}

fn default_cell_type() -> CellType {
    CellType::Text
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return;
    }
    for value in vector.iter_mut() {
        *value /= norm;
    }
}
