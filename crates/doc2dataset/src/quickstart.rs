use std::path::PathBuf;

use anyhow::Result;

use crate::{export, ingest, tasks};

pub fn run(input: String) -> Result<()> {
    let output = PathBuf::from("datasets/default");
    ingest::run(
        input.clone(),
        output.to_string_lossy().into_owned(),
        "*.pdf,*.md,*.txt".to_string(),
        "reports".to_string(),
        false,
        false,
        "eng".to_string(),
    )?;
    tasks::run(
        output.to_string_lossy().into_owned(),
        "qa,summary".to_string(),
    )?;
    export::run_hf(&output)?;
    export::run_openai(&output)?;
    println!(
        "✅ Done. Check {}/exports for HF + OpenAI payloads",
        output.display()
    );
    Ok(())
}
