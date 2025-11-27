use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Default, Deserialize, Clone)]
pub struct PricingConfig {
    #[serde(default)]
    pub openai: HashMap<String, PricingEntry>,
    #[serde(default)]
    pub anthropic: HashMap<String, PricingEntry>,
    #[serde(default)]
    pub gemini: HashMap<String, PricingEntry>,
    #[serde(default)]
    pub deepseek: HashMap<String, PricingEntry>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PricingEntry {
    pub prompt_per_1k: Option<f64>,
    pub completion_per_1k: Option<f64>,
    pub prompt_per_1m: Option<f64>,
    pub completion_per_1m: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct PricingRate {
    pub prompt_per_1k: f64,
    pub completion_per_1k: f64,
}

impl PricingConfig {
    pub fn lookup(&self, provider: &str, model: &str) -> Option<PricingRate> {
        let model_key = model.to_string();
        match provider {
            "openai" => Self::lookup_entry(&self.openai, &model_key),
            "anthropic" => Self::lookup_entry(&self.anthropic, &model_key),
            "gemini" => Self::lookup_entry(&self.gemini, &model_key),
            "deepseek" => Self::lookup_entry(&self.deepseek, &model_key),
            _ => None,
        }
    }

    fn lookup_entry(map: &HashMap<String, PricingEntry>, model: &str) -> Option<PricingRate> {
        map.get(model)
            .or_else(|| map.get(&model.to_lowercase()))
            .and_then(|entry| entry.normalized())
    }
}

impl PricingEntry {
    pub fn normalized(&self) -> Option<PricingRate> {
        let prompt = self
            .prompt_per_1k
            .or_else(|| self.prompt_per_1m.map(|value| value / 1000.0));
        let completion = self
            .completion_per_1k
            .or_else(|| self.completion_per_1m.map(|value| value / 1000.0));
        if prompt.is_none() && completion.is_none() {
            return None;
        }
        Some(PricingRate {
            prompt_per_1k: prompt.unwrap_or(0.0),
            completion_per_1k: completion.unwrap_or(0.0),
        })
    }
}
