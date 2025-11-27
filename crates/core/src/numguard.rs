use once_cell::sync::Lazy;
use regex::Regex;
use sha1::{Digest, Sha1};

use crate::document::NumGuard;

static NUM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?P<num>\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?)\s*(?P<unit>%|mmhg|mm|cm|mg|kg|usd|eur|bpm)?",
    )
    .expect("valid regex")
});

pub fn extract_guards(line: &str, z: u32, x: u32, y: u32) -> Vec<NumGuard> {
    let mut guards = Vec::new();
    for caps in NUM_RE.captures_iter(&line.to_lowercase()) {
        let mut digits = caps
            .name("num")
            .map(|m| m.as_str())
            .unwrap_or("")
            .to_string();
        digits.retain(|c| c.is_ascii_digit());
        if digits.is_empty() {
            continue;
        }
        let units = caps.name("unit").map(|m| m.as_str()).unwrap_or("");
        guards.push(NumGuard {
            z,
            x,
            y,
            units: units.to_string(),
            sha1: sha1_from_digits(&digits),
        });
    }
    guards
}

pub fn hash_digits_from_payload(payload: &str) -> Option<[u8; 20]> {
    let digits = payload
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    Some(sha1_from_digits(&digits))
}

fn sha1_from_digits(digits: &str) -> [u8; 20] {
    let mut hasher = Sha1::new();
    hasher.update(digits.as_bytes());
    let hash = hasher.finalize();
    let mut sha = [0u8; 20];
    sha.copy_from_slice(&hash[..]);
    sha
}
