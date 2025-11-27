mod cli;
mod config;
mod convert;
mod export;
mod ingest;
mod logging;
mod model;
mod quickstart;
mod run;
mod tasks;

use anyhow::Result;
use clap::Parser;

use crate::cli::{Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let verbose = if cli.verbose {
        true
    } else {
        logging::env_flag()
    };
    logging::init(verbose);
    match cli.command {
        Command::Ingest {
            input,
            output,
            pattern,
            preset,
            enable_ocr,
            force_ocr,
            ocr_langs,
        } => ingest::run(
            input, output, pattern, preset, enable_ocr, force_ocr, ocr_langs,
        ),
        Command::Tasks {
            dataset_root,
            tasks,
        } => tasks::run(dataset_root, tasks),
        Command::Export { target } => export::run(target),
        Command::Quickstart { input } => quickstart::run(input),
        Command::Run { config } => run::run_from_config(&config),
    }
}
