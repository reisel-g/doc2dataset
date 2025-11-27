use std::path::Path;

use anyhow::Result;

use crate::cli::ExportCommand;
use crate::model::{read_jsonl, DatasetIndex, QaSample, SummarySample};

mod axolotl;
mod hf;
mod llama_factory;
mod openai;
mod rag;

pub fn run(command: ExportCommand) -> Result<()> {
    match command {
        ExportCommand::Hf { dataset_root } => run_hf(Path::new(&dataset_root)),
        ExportCommand::LlamaFactory {
            dataset_root,
            format,
        } => run_llama_factory(Path::new(&dataset_root), &format),
        ExportCommand::Openai { dataset_root } => run_openai(Path::new(&dataset_root)),
        ExportCommand::Axolotl { dataset_root, mode } => {
            run_axolotl(Path::new(&dataset_root), &mode)
        }
        ExportCommand::RagJsonl { dataset_root } => run_rag_jsonl(Path::new(&dataset_root)),
    }
}

pub fn run_hf(root: &Path) -> Result<()> {
    let index = DatasetIndex::load(root)?;
    let qa = load_qa_samples(root)?;
    let summary = load_summary_samples(root)?;
    hf::export(root, &index, &qa, &summary)
}

pub fn run_llama_factory(root: &Path, format: &str) -> Result<()> {
    let index = DatasetIndex::load(root)?;
    let qa = load_qa_samples(root)?;
    llama_factory::export(root, &index, &qa, format)
}

pub fn run_openai(root: &Path) -> Result<()> {
    let index = DatasetIndex::load(root)?;
    let qa = load_qa_samples(root)?;
    openai::export(root, &index, &qa)
}

pub fn run_axolotl(root: &Path, mode: &str) -> Result<()> {
    let index = DatasetIndex::load(root)?;
    let qa = load_qa_samples(root)?;
    match mode {
        "chat" => axolotl::export_chat(root, &index, &qa),
        "text" => axolotl::export_text(root, &index, &qa),
        other => Err(anyhow::anyhow!(format!(
            "unsupported axolotl mode: {other}"
        ))),
    }
}

pub fn run_rag_jsonl(root: &Path) -> Result<()> {
    rag::export(root)
}

fn load_qa_samples(root: &Path) -> Result<Vec<QaSample>> {
    let path = root.join("samples/qa.jsonl");
    read_jsonl(&path)
}

fn load_summary_samples(root: &Path) -> Result<Vec<SummarySample>> {
    let path = root.join("samples/summary.jsonl");
    read_jsonl(&path)
}
