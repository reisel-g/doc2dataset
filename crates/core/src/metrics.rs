use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use strsim::levenshtein;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub pages: u32,
    pub lines_total: u32,
    pub cells_total: u32,
    pub cells_kept: u32,
    pub dedup_ratio: f32,
    pub numguard_count: u32,
    pub raw_tokens_estimate: Option<u32>,
    pub compressed_tokens_estimate: Option<u32>,
    pub compression_factor: Option<f32>,
}

impl Metrics {
    pub fn with_token_metrics(mut self, raw: Option<u32>, compressed: Option<u32>) -> Self {
        self.record_tokens(raw, compressed);
        self
    }

    pub fn record_tokens(&mut self, raw: Option<u32>, compressed: Option<u32>) {
        if let Some(raw) = raw {
            self.raw_tokens_estimate = Some(raw);
        }
        if let Some(compressed) = compressed {
            self.compressed_tokens_estimate = Some(compressed);
        }
        self.compression_factor = match (self.raw_tokens_estimate, self.compressed_tokens_estimate)
        {
            (Some(raw), Some(comp)) if comp > 0 => Some(raw as f32 / comp as f32),
            _ => None,
        };
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenMetrics {
    pub raw: u32,
    pub compressed: u32,
    pub factor: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NumStats {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub units_ok: f64,
}

pub fn cer(pred: &str, gold: &str) -> f64 {
    normalized_distance(pred, gold)
}

pub fn wer(pred: &str, gold: &str) -> f64 {
    let pred_words: Vec<&str> = pred.split_whitespace().collect();
    let gold_words: Vec<&str> = gold.split_whitespace().collect();
    if gold_words.is_empty() {
        return if pred_words.is_empty() { 0.0 } else { 1.0 };
    }
    let dist = levenshtein_words(&pred_words, &gold_words);
    dist as f64 / gold_words.len() as f64
}

pub fn numeric_stats(pred: &str, gold: &str) -> NumStats {
    let pred_vals = extract_numbers(pred);
    let gold_vals = extract_numbers(gold);
    if gold_vals.is_empty() && pred_vals.is_empty() {
        return NumStats {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
            units_ok: 1.0,
        };
    }
    let mut gold_used = vec![false; gold_vals.len()];
    let mut matches = 0usize;
    let mut units_match = 0usize;
    for pred in &pred_vals {
        if let Some((idx, gold_item)) = gold_vals
            .iter()
            .enumerate()
            .find(|(i, g)| !gold_used[*i] && g.value == pred.value)
        {
            gold_used[idx] = true;
            matches += 1;
            if gold_item
                .unit
                .as_ref()
                .map(|u| pred.unit.as_ref() == Some(u))
                .unwrap_or(true)
            {
                units_match += 1;
            }
        }
    }
    let precision = if pred_vals.is_empty() {
        1.0
    } else {
        matches as f64 / pred_vals.len() as f64
    };
    let recall = if gold_vals.is_empty() {
        1.0
    } else {
        matches as f64 / gold_vals.len() as f64
    };
    let f1 = if precision == 0.0 || recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    let units_ok = if matches == 0 {
        1.0
    } else {
        units_match as f64 / matches as f64
    };
    NumStats {
        precision,
        recall,
        f1,
        units_ok,
    }
}

fn normalized_distance(pred: &str, gold: &str) -> f64 {
    let pred_norm = normalize(pred);
    let gold_norm = normalize(gold);
    if gold_norm.is_empty() {
        return if pred_norm.is_empty() { 0.0 } else { 1.0 };
    }
    let dist = levenshtein(&pred_norm, &gold_norm);
    dist as f64 / gold_norm.chars().count() as f64
}

fn levenshtein_words(pred: &[&str], gold: &[&str]) -> usize {
    let m = gold.len();
    let n = pred.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if gold[i - 1] == pred[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[m][n]
}

fn normalize(text: &str) -> String {
    text.chars().filter(|c| !c.is_control()).collect::<String>()
}

#[derive(Debug, Clone)]
struct NumberToken {
    value: String,
    unit: Option<String>,
}

static NUM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?P<num>[+-]?\d{1,3}(?:[\d,.]*))(?:\s*(?P<unit>[a-z%$]{1,8}))?")
        .expect("valid regex")
});

fn extract_numbers(text: &str) -> Vec<NumberToken> {
    let mut out = Vec::new();
    for caps in NUM_RE.captures_iter(text) {
        let raw = caps.name("num").map(|m| m.as_str()).unwrap_or("");
        let mut digits = raw.replace(',', "");
        digits.retain(|c| c.is_ascii_digit() || c == '.' || c == '-');
        if digits.is_empty() {
            continue;
        }
        let unit = caps.name("unit").map(|m| m.as_str().trim().to_lowercase());
        out.push(NumberToken {
            value: digits,
            unit,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cer_handles_exact_match() {
        assert_eq!(cer("hello", "hello"), 0.0);
    }

    #[test]
    fn numeric_stats_counts_matches() {
        let stats = numeric_stats("Revenue $123", "Revenue $123");
        assert_eq!(stats.precision, 1.0);
        assert_eq!(stats.recall, 1.0);
    }
}
