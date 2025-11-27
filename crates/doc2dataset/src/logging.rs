use std::env;
use std::sync::atomic::{AtomicBool, Ordering};

static VERBOSE: AtomicBool = AtomicBool::new(false);

pub fn init(enabled: bool) {
    VERBOSE.store(enabled, Ordering::Relaxed);
    if enabled {
        info("verbose logging enabled");
    }
}

pub fn verbose_enabled() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

pub fn info(message: impl AsRef<str>) {
    eprintln!("[doc2dataset] {}", message.as_ref());
}

pub fn stage(stage: &str, message: impl AsRef<str>) {
    eprintln!("[doc2dataset::{}] {}", stage, message.as_ref());
}

pub fn verbose(message: impl AsRef<str>) {
    if verbose_enabled() {
        eprintln!("[doc2dataset::verbose] {}", message.as_ref());
    }
}

pub fn env_flag() -> bool {
    env::var("DOC2DATASET_VERBOSE")
        .map(|value| parse_bool(value.trim()))
        .unwrap_or(false)
}

fn parse_bool(raw: &str) -> bool {
    matches!(
        raw.trim().to_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}
