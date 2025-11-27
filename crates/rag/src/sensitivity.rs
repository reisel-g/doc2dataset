use once_cell::sync::Lazy;
use std::collections::HashMap;

static LEVELS: [&str; 4] = ["public", "internal", "confidential", "restricted"];

static RANKS: Lazy<HashMap<&'static str, u8>> = Lazy::new(|| {
    LEVELS
        .iter()
        .enumerate()
        .map(|(idx, level)| (*level, idx as u8))
        .collect()
});

pub fn normalize_level(value: &str) -> String {
    let lower = value.trim().to_lowercase();
    if LEVELS.contains(&lower.as_str()) {
        lower
    } else {
        "public".to_string()
    }
}

pub fn sensitivity_rank(value: &str) -> u8 {
    let lower = value.trim().to_lowercase();
    *RANKS.get(lower.as_str()).unwrap_or(&0)
}

pub fn allowed(level: &str, threshold: &str) -> bool {
    sensitivity_rank(level) <= sensitivity_rank(threshold)
}

pub fn levels() -> &'static [&'static str] {
    &LEVELS
}
