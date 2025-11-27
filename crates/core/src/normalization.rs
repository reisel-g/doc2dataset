use once_cell::sync::Lazy;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

use crate::document::CellType;

#[derive(Debug, Clone, Copy)]
pub struct ImportanceTuning {
    pub heading_boost: f32,
    pub number_boost: f32,
    pub footer_penalty: f32,
    pub early_line_bonus: f32,
}

impl Default for ImportanceTuning {
    fn default() -> Self {
        Self {
            heading_boost: 1.0,
            number_boost: 1.0,
            footer_penalty: 0.5,
            early_line_bonus: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyphenationMode {
    Merge,
    Preserve,
}

pub fn normalize_line(line: &str) -> String {
    let trimmed = line.trim_matches(|c: char| c.is_control() || c.is_whitespace());
    let nfkc = trimmed.nfkc().collect::<String>();
    let mut result = String::with_capacity(nfkc.len());
    let mut prev_space = false;
    for ch in nfkc.chars() {
        if ch.is_control() {
            continue;
        }
        if ch.is_whitespace() {
            if !prev_space {
                result.push(' ');
                prev_space = true;
            }
        } else {
            result.push(ch);
            prev_space = false;
        }
    }
    result.trim().to_string()
}

pub fn normalize_lines(lines: &[String], mode: HyphenationMode) -> Vec<String> {
    let mut merged = match mode {
        HyphenationMode::Merge => merge_hyphenation(lines),
        HyphenationMode::Preserve => lines.to_vec(),
    };
    merged
        .drain(..)
        .map(|line| normalize_line(&line))
        .filter(|line| !line.is_empty())
        .collect()
}

fn merge_hyphenation(lines: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(lines.len());
    let mut carry = String::new();
    for line in lines {
        let current = if carry.is_empty() {
            line.clone()
        } else {
            let mut combined = carry.clone();
            combined.push_str(line.trim_start());
            combined
        };
        let trimmed = current.trim_end().to_string();
        if trimmed.ends_with('-') && trimmed.len() > 1 {
            carry = trimmed.trim_end_matches('-').to_string();
            continue;
        }
        out.push(current);
        carry.clear();
    }
    if !carry.is_empty() {
        out.push(carry);
    }
    out
}

pub fn classify_cell_type(line: &str) -> CellType {
    if looks_like_table(line) {
        CellType::Table
    } else if looks_like_header(line) {
        CellType::Header
    } else if looks_like_footer(line) {
        CellType::Footer
    } else {
        CellType::Text
    }
}

pub fn importance_score(
    line: &str,
    cell_type: CellType,
    line_index: usize,
    tuning: &ImportanceTuning,
) -> u8 {
    let base = match cell_type {
        CellType::Header => 220,
        CellType::Footer => (40.0 * tuning.footer_penalty) as i32,
        CellType::Table => 160,
        _ => 100,
    };
    let heading_bonus = if is_all_caps(line) {
        (35.0 * tuning.heading_boost) as i32
    } else {
        0
    };
    let number_bonus = if contains_numbers(line) {
        (20.0 * tuning.number_boost) as i32
    } else {
        0
    };
    let early_bonus = if line_index < 5 {
        (15.0 * tuning.early_line_bonus) as i32
    } else {
        0
    };
    let length_penalty = (line.len() / 120) as i32 * -10;
    let score = base + heading_bonus + number_bonus + early_bonus + length_penalty;
    score.clamp(0, 255) as u8
}

fn looks_like_table(line: &str) -> bool {
    static TABLE_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"\b(total|subtotal|amount)\b.*\b(usd|eur|%)\b").unwrap());
    line.contains('|') || line.contains('\t') || TABLE_RE.is_match(&line.to_lowercase())
}

pub fn looks_like_table_with_tolerance(line: &str, tolerance_px: u32) -> bool {
    if looks_like_table(line) {
        return true;
    }
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() < 3 {
        return false;
    }
    let tolerance_chars = ((tolerance_px / 8).max(2)) as usize;
    longest_space_run(line) >= tolerance_chars
}

fn looks_like_header(line: &str) -> bool {
    line.chars().filter(|c| c.is_alphabetic()).count() > 3 && is_all_caps(line)
}

fn looks_like_footer(line: &str) -> bool {
    let lower = line.to_lowercase();
    lower.contains("page ") || lower.contains("confidential")
}

fn contains_numbers(line: &str) -> bool {
    line.chars().any(|c| c.is_ascii_digit())
}

fn is_all_caps(line: &str) -> bool {
    let letters: Vec<char> = line.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.is_empty() {
        return false;
    }
    letters.iter().all(|c| c.is_uppercase())
}

fn longest_space_run(line: &str) -> usize {
    let mut current = 0;
    let mut best = 0;
    for ch in line.chars() {
        if ch == ' ' {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_whitespace() {
        let line = "  H e l l o   —  WORLD  ";
        assert_eq!(normalize_line(line), "H e l l o — WORLD");
    }

    #[test]
    fn detects_tables() {
        assert_eq!(classify_cell_type("| Col |"), CellType::Table);
        assert_eq!(classify_cell_type("TOTAL AMOUNT USD"), CellType::Table);
    }

    #[test]
    fn tolerance_detects_layout_tables() {
        assert!(looks_like_table_with_tolerance("Q1      Q2      Q3", 24));
        assert!(!looks_like_table_with_tolerance("Short line", 32));
    }
}
