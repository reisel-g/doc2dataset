use std::path::PathBuf;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DcfError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde json error: {0}")]
    SerdeJson(#[from] serde_json::Error),
    #[error("prost encode error: {0}")]
    ProtoEncode(#[from] prost::EncodeError),
    #[error("prost decode error: {0}")]
    ProtoDecode(#[from] prost::DecodeError),
    #[error("pdf support not enabled: {0:?}")]
    PdfSupportDisabled(PathBuf),
    #[error("ocr support not enabled")]
    OcrSupportDisabled,
    #[error("unsupported input format: {0:?}")]
    UnsupportedInput(PathBuf),
    #[error("invalid document: {0}")]
    InvalidDocument(&'static str),
    #[error("unknown preset: {0}")]
    UnknownPreset(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("benchmark error: {0}")]
    Bench(String),
    #[error("other: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, DcfError>;

impl From<anyhow::Error> for DcfError {
    fn from(value: anyhow::Error) -> Self {
        Self::Other(value.to_string())
    }
}
