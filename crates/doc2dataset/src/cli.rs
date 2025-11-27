use clap::{ArgAction, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "doc2dataset", about = "3DCF doc2dataset CLI")]
pub struct Cli {
    #[arg(long, global = true, action = ArgAction::SetTrue)]
    pub verbose: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    Ingest {
        input: String,
        #[arg(long)]
        output: String,
        #[arg(
            long,
            default_value = "*.pdf,*.md,*.txt,*.html,*.htm,*.xml,*.xhtml,*.rss,*.atom,*.json,*.yaml,*.yml,*.csv,*.tsv,*.csv.gz,*.tsv.gz,*.tex,*.bib,*.bbl,*.ini,*.cfg,*.conf,*.toml,*.log,*.rtf,*.png,*.jpg,*.jpeg"
        )]
        pattern: String,
        #[arg(long, default_value = "reports")]
        preset: String,
        #[arg(long, default_value_t = false)]
        enable_ocr: bool,
        #[arg(long, default_value_t = false)]
        force_ocr: bool,
        #[arg(long, default_value = "eng")]
        ocr_langs: String,
    },
    Tasks {
        dataset_root: String,
        #[arg(long, default_value = "qa,summary")]
        tasks: String,
    },
    Export {
        #[command(subcommand)]
        target: ExportCommand,
    },
    Quickstart {
        input: String,
    },
    Run {
        #[arg(long, default_value = "doc2dataset.yaml")]
        config: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum ExportCommand {
    Hf {
        dataset_root: String,
    },
    LlamaFactory {
        dataset_root: String,
        #[arg(long, default_value = "alpaca")]
        format: String,
    },
    Openai {
        dataset_root: String,
    },
    Axolotl {
        dataset_root: String,
        #[arg(long, default_value = "chat")]
        mode: String,
    },
    RagJsonl {
        dataset_root: String,
    },
}
