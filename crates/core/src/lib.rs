pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/dcf.v1.rs"));
}

mod bench;
mod chunk;
mod decoder;
mod document;
mod embedding;
mod encoder;
mod error;
mod ingest;
mod metrics;
mod normalization;
mod numguard;
mod ocr;
mod serializer;
mod stats;

pub use bench::{BenchConfig, BenchMode, BenchResult, BenchRunner, CorpusMetrics};
pub use chunk::{ChunkConfig, ChunkMode, ChunkRecord, Chunker};
pub use decoder::Decoder;
pub use document::{
    hash_payload, CellRecord, CellType, CodeHash, Document, Header, NumGuard, NumGuardAlert,
    NumGuardIssue, PageInfo,
};
pub use embedding::{EmbeddingRecord, HashEmbedder, HashEmbedderConfig};
pub use encoder::{EncodeInput, Encoder, EncoderBuilder, EncoderPreset};
pub use error::{DcfError, Result};
pub use ingest::{ingest_to_index, ingest_to_index_with_opts, IngestOptions};
pub use metrics::{cer, numeric_stats, wer, Metrics, NumStats, TokenMetrics};
pub use normalization::{HyphenationMode, ImportanceTuning};
pub use serializer::{TableMode, TextSerializer, TextSerializerConfig};
pub use stats::{estimate_tokens, Stats, TokenizerKind};
