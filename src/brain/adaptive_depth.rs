//! Adaptive depth with heterogeneous expert tiers for the MoE decoder
//! (docs/ADAPTIVE_DEPTH.md). Layers are grouped into tiers (easy, medium,
//! hard, ...); every tier has its own expert sizes and top-k. All tokens run the
//! first tier. Before each later tier a per-token gate decides, causally and
//! identically in training and inference, whether a position continues; exited
//! positions keep their hidden state, still serve as attention keys, and skip
//! the deeper experts. A budget loss holds the fraction reaching each tier near
//! its `capacity`.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// `count` experts of SwiGLU hidden width `ff_dim`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertGroup {
    pub count: usize,
    pub ff_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DepthTier {
    pub name: String,
    pub layers: usize,
    /// Target fraction of non-padding tokens that reach this tier. The first
    /// tier is always 1; later tiers must not exceed the previous one.
    #[serde(default = "full_capacity")]
    pub capacity: f64,
    pub top_k: usize,
    /// Mixed sizes are allowed; one router picks among all of the tier's experts.
    pub experts: Vec<ExpertGroup>,
}

fn full_capacity() -> f64 { 1.0 }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveDepthConfig {
    pub tiers: Vec<DepthTier>,
    /// Weight of sum over gated tiers of (reached fraction - capacity)^2.
    #[serde(default = "default_budget_weight")]
    pub budget_loss_weight: f64,
}

fn default_budget_weight() -> f64 { 0.1 }

impl DepthTier {
    /// Expert hidden widths in router order.
    pub fn expert_widths(&self) -> Vec<usize> {
        self.experts.iter().flat_map(|g| std::iter::repeat(g.ff_dim).take(g.count)).collect()
    }
    pub fn num_experts(&self) -> usize { self.experts.iter().map(|g| g.count).sum() }
    /// Run-name fragment, e.g. `8l16x512k2c100`.
    fn tag(&self) -> String {
        let groups: Vec<String> = self.experts.iter().map(|g| format!("{}x{}", g.count, g.ff_dim)).collect();
        format!("{}l{}k{}c{}", self.layers, groups.join("+"), self.top_k, (self.capacity * 100.0).round() as u32)
    }
}

impl AdaptiveDepthConfig {
    pub fn total_layers(&self) -> usize { self.tiers.iter().map(|t| t.layers).sum() }

    /// Tier index of every layer.
    pub fn layer_tiers(&self) -> Vec<usize> {
        self.tiers.iter().enumerate().flat_map(|(i, t)| std::iter::repeat(i).take(t.layers)).collect()
    }

    pub fn tag(&self) -> String {
        format!("MoeDepth_{}", self.tiers.iter().map(DepthTier::tag).collect::<Vec<_>>().join("_"))
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.tiers.len() >= 1, "adaptive_depth needs at least one tier");
        ensure!(self.budget_loss_weight.is_finite() && self.budget_loss_weight >= 0.0,
            "adaptive_depth.budget_loss_weight must be finite and non-negative");
        let mut previous = 1.0;
        for (i, tier) in self.tiers.iter().enumerate() {
            let name = &tier.name;
            ensure!(tier.layers > 0, "adaptive_depth tier {name}: layers must be positive");
            ensure!(!tier.experts.is_empty() && tier.experts.iter().all(|g| g.count > 0 && g.ff_dim > 0),
                "adaptive_depth tier {name}: expert groups need positive count and ff_dim");
            ensure!(tier.top_k > 0 && tier.top_k <= tier.num_experts(),
                "adaptive_depth tier {name}: require 1 <= top_k <= {} experts", tier.num_experts());
            ensure!(tier.capacity.is_finite() && tier.capacity > 0.0 && tier.capacity <= previous,
                "adaptive_depth tier {name}: capacity must be in (0, {previous}]");
            ensure!(i > 0 || tier.capacity == 1.0, "adaptive_depth: the first tier ({name}) runs every token; capacity must be 1");
            previous = tier.capacity;
        }
        Ok(())
    }

    /// FFN parameters per layer for each tier: (total, expected active per token
    /// that reaches it, using the mean expert width), excluding routers and gates.
    pub fn ffn_params(&self, width: usize) -> Vec<(usize, usize)> {
        self.tiers.iter().map(|t| {
            let widths = t.expert_widths();
            let total: usize = widths.iter().map(|w| 3 * width * w).sum();
            (total, total * t.top_k / widths.len())
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AdaptiveDepthConfig {
        serde_json::from_str(r#"{"tiers": [
            {"name": "easy", "layers": 2, "top_k": 2, "experts": [{"count": 4, "ff_dim": 8}]},
            {"name": "medium", "layers": 1, "capacity": 0.5, "top_k": 1,
             "experts": [{"count": 2, "ff_dim": 8}, {"count": 1, "ff_dim": 16}]},
            {"name": "hard", "layers": 1, "capacity": 0.25, "top_k": 1, "experts": [{"count": 2, "ff_dim": 32}]}
        ]}"#).unwrap()
    }

    #[test]
    fn adaptive_depth_config_layout_and_tag() {
        let c = config();
        c.validate().unwrap();
        assert_eq!(c.budget_loss_weight, 0.1);
        assert_eq!(c.total_layers(), 4);
        assert_eq!(c.layer_tiers(), vec![0, 0, 1, 2]);
        assert_eq!(c.tiers[1].expert_widths(), vec![8, 8, 16]);
        assert_eq!(c.tag(), "MoeDepth_2l4x8k2c100_1l2x8+1x16k1c50_1l2x32k1c25");
    }

    #[test]
    fn adaptive_depth_config_rejects_bad_tiers() {
        let mut c = config();
        c.tiers[0].capacity = 0.9;
        assert!(c.validate().is_err(), "first tier must run every token");
        let mut c = config();
        c.tiers[2].capacity = 0.75;
        assert!(c.validate().is_err(), "capacity cannot grow with depth");
        let mut c = config();
        c.tiers[1].top_k = 4;
        assert!(c.validate().is_err(), "top_k above expert count");
        assert!(serde_json::from_str::<AdaptiveDepthConfig>(r#"{"tiers": [], "typo": 1}"#).is_err());
    }
}
