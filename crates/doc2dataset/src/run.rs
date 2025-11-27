use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde_yaml::from_str;

use crate::cli::ExportCommand;
use crate::config::RunConfig;
use crate::{export, ingest, tasks};

pub fn run_from_config(path: &str) -> Result<()> {
    let raw = fs::read_to_string(path).with_context(|| format!("failed to read config {path}"))?;
    let cfg: RunConfig = from_str(&raw).context("invalid doc2dataset config")?;
    run_pipeline(cfg, ingest::run, tasks::run, export::run)
}

fn run_pipeline<FIngest, FTasks, FExport>(
    cfg: RunConfig,
    ingest_fn: FIngest,
    tasks_fn: FTasks,
    export_fn: FExport,
) -> Result<()>
where
    FIngest: Fn(String, String, String, String, bool, bool, String) -> Result<()>,
    FTasks: Fn(String, String) -> Result<()>,
    FExport: Fn(ExportCommand) -> Result<()>,
{
    if cfg.sources.is_empty() {
        return Err(anyhow!("run config must declare at least one source"));
    }
    fs::create_dir_all(Path::new(&cfg.dataset_root))?;
    for (idx, source) in cfg.sources.iter().enumerate() {
        eprintln!(
            "[doc2dataset] ingest source {}: path={} pattern={}",
            idx + 1,
            source.path,
            source.pattern
        );
        ingest_fn(
            source.path.clone(),
            cfg.dataset_root.clone(),
            source.pattern.clone(),
            cfg.ingest.preset.clone(),
            cfg.ingest.enable_ocr,
            cfg.ingest.force_ocr,
            cfg.ingest.ocr_langs.join(","),
        )?;
    }
    if !cfg.tasks.is_empty() {
        tasks_fn(cfg.dataset_root.clone(), cfg.tasks.join(","))?;
    }
    let root = cfg.dataset_root.clone();
    if cfg.exports.hf {
        export_fn(ExportCommand::Hf {
            dataset_root: root.clone(),
        })?;
    }
    if let Some(lf) = &cfg.exports.llama_factory {
        export_fn(ExportCommand::LlamaFactory {
            dataset_root: root.clone(),
            format: lf.format.clone(),
        })?;
    }
    if cfg.exports.openai {
        export_fn(ExportCommand::Openai {
            dataset_root: root.clone(),
        })?;
    }
    if let Some(ax) = &cfg.exports.axolotl {
        export_fn(ExportCommand::Axolotl {
            dataset_root: root.clone(),
            mode: ax.mode.clone(),
        })?;
    }
    if cfg.exports.rag_jsonl {
        export_fn(ExportCommand::RagJsonl {
            dataset_root: root.clone(),
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn pipeline_invokes_all_hooks() {
        let mut cfg = RunConfig {
            dataset_root: "./tmp-dataset".to_string(),
            sources: vec![crate::config::SourceConfig {
                path: "./input".to_string(),
                pattern: "*.pdf".to_string(),
            }],
            tasks: vec!["qa".to_string()],
            exports: crate::config::ExportsConfig {
                hf: true,
                llama_factory: Some(crate::config::LlamaFactoryExport {
                    format: "alpaca".to_string(),
                }),
                openai: true,
                axolotl: Some(crate::config::AxolotlExport {
                    mode: "chat".to_string(),
                }),
                rag_jsonl: true,
            },
            ingest: crate::config::IngestConfig {
                preset: "slides".to_string(),
                enable_ocr: true,
                force_ocr: false,
                ocr_langs: vec!["eng".to_string(), "deu".to_string()],
            },
        };
        cfg.sources.push(crate::config::SourceConfig {
            path: "./input2".to_string(),
            pattern: "*.md".to_string(),
        });
        let ingest_calls: RefCell<Vec<(String, String)>> = RefCell::new(Vec::new());
        let task_calls: RefCell<Vec<String>> = RefCell::new(Vec::new());
        let export_calls: RefCell<Vec<String>> = RefCell::new(Vec::new());
        run_pipeline(
            cfg,
            |input, output, pattern, preset, enable, force, langs| {
                ingest_calls
                    .borrow_mut()
                    .push((input.clone(), preset.clone()));
                assert_eq!(output, "./tmp-dataset");
                assert!(pattern == "*.pdf" || pattern == "*.md");
                assert!(enable);
                assert!(!force);
                assert_eq!(langs, "eng,deu");
                Ok(())
            },
            |dataset, tasks| {
                task_calls.borrow_mut().push(dataset.clone());
                assert_eq!(tasks, "qa");
                Ok(())
            },
            |cmd| {
                export_calls.borrow_mut().push(match cmd {
                    ExportCommand::Hf { .. } => "hf".into(),
                    ExportCommand::LlamaFactory { .. } => "lf".into(),
                    ExportCommand::Openai { .. } => "openai".into(),
                    ExportCommand::Axolotl { .. } => "ax".into(),
                    ExportCommand::RagJsonl { .. } => "rag".into(),
                });
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(ingest_calls.borrow().len(), 2);
        assert_eq!(task_calls.borrow().len(), 1);
        assert_eq!(export_calls.borrow().len(), 5);
    }

    #[test]
    fn pipeline_skips_tasks_when_none_requested() {
        let cfg = RunConfig {
            dataset_root: "./tmp-dataset".to_string(),
            sources: vec![crate::config::SourceConfig {
                path: "./input".to_string(),
                pattern: "*.pdf".to_string(),
            }],
            tasks: vec![],
            exports: crate::config::ExportsConfig {
                hf: false,
                llama_factory: None,
                openai: false,
                axolotl: None,
                rag_jsonl: false,
            },
            ingest: crate::config::IngestConfig {
                preset: "reports".to_string(),
                enable_ocr: false,
                force_ocr: false,
                ocr_langs: vec!["eng".to_string()],
            },
        };
        let task_calls: RefCell<Vec<String>> = RefCell::new(Vec::new());
        run_pipeline(
            cfg,
            |_, _, _, _, _, _, _| Ok(()),
            |dataset, _| {
                task_calls.borrow_mut().push(dataset.clone());
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();
        assert!(task_calls.borrow().is_empty());
    }
}
