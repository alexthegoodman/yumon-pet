//! Johnson-Lindenstrauss projections for the dense decoder (docs/JL.md).
//!
//! A JL map is a fixed random matrix R [from, to] with N(0, 1/to) entries, so for
//! any x, y: |xR| ~ |x| and (xR).(yR) ~ x.y, with error shrinking like 1/sqrt(to).
//! The decoder can put one at three sites:
//! - `residual`: the residual stream goes d -> dim before block `from_layer`
//!   (`n_layers` = just before the final norm and token head); later blocks,
//!   the norm and the head run at width dim.
//! - `attention`: in blocks from `from_layer` on, each head's q and k (after
//!   RoPE) and v go head_dim -> dim through one shared R. Scores keep the
//!   1/sqrt(head_dim) scale, so they estimate the unprojected q.k. The output
//!   projection reads heads * dim, and the KV cache stores dim per head.
//! - `ffn`: in blocks from `from_layer` on, the normed MLP input goes d -> dim.
//! `none` adds nothing; it names the baseline arm of an experiment config so
//! all arms share seeding and run-name layout.
//! R is frozen unless `trainable` (the learned-projection control).
use burn::{module::Param, prelude::*};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JlSite {
    None,
    Residual,
    Attention,
    Ffn,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JlConfig {
    pub site: JlSite,
    /// Target width: model width for residual/ffn, per-head width for attention.
    #[serde(default)]
    pub dim: usize,
    /// First block affected (residual: projection sits before this block).
    #[serde(default)]
    pub from_layer: usize,
    /// Residual only: MLP hidden width of the narrow blocks. Absent = ff_dim
    /// scaled by dim / embed_dim (same ratio as the wide blocks).
    #[serde(default)]
    pub ff_dim: Option<usize>,
    /// Seed of R; block i uses seed + i.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Train R with the rest of the model instead of keeping it fixed.
    #[serde(default)]
    pub trainable: bool,
}
fn default_seed() -> u64 { 1 }

impl JlConfig {
    pub fn active(&self) -> bool { self.site != JlSite::None }

    /// Parses `none`, `SITE:DIM` or `SITE:DIM@LAYER`, with an optional
    /// `+learned` suffix (e.g. `residual:64@3`, `attention:16+learned`).
    pub fn parse(spec: &str) -> anyhow::Result<Self> {
        let (spec, trainable) = match spec.strip_suffix("+learned") {
            Some(s) => (s, true),
            None => (spec, false),
        };
        let (site, rest) = spec.split_once(':').unwrap_or((spec, ""));
        let site: JlSite = serde_json::from_value(serde_json::Value::String(site.to_owned()))
            .map_err(|_| anyhow::anyhow!("unknown JL site {site:?} (none, residual, attention, ffn)"))?;
        let (dim, layer) = rest.split_once('@').unwrap_or((rest, "0"));
        let dim = if site == JlSite::None { 0 } else { dim.parse().map_err(|_| anyhow::anyhow!("bad JL dim {dim:?}"))? };
        let from_layer = layer.parse().map_err(|_| anyhow::anyhow!("bad JL layer {layer:?}"))?;
        Ok(Self { site, dim, from_layer, ff_dim: None, seed: default_seed(), trainable })
    }

    pub fn validate(&self, embed_dim: usize, attn_heads: usize, n_layers: usize) -> anyhow::Result<()> {
        use anyhow::ensure;
        if !self.active() { return Ok(()); }
        ensure!(self.dim > 0, "jl.dim must be positive");
        let head_dim = embed_dim / attn_heads;
        match self.site {
            JlSite::None => {}
            JlSite::Residual => {
                ensure!(self.from_layer <= n_layers, "jl.from_layer {} is past n_layers {n_layers}", self.from_layer);
                ensure!(self.dim < embed_dim, "residual jl.dim {} must be below embed_dim {embed_dim}", self.dim);
                if self.from_layer < n_layers {
                    ensure!(self.dim % attn_heads == 0 && (self.dim / attn_heads) % 2 == 0,
                        "residual jl.dim must split into attn_heads even-width heads (RoPE)");
                }
                ensure!(self.ff_dim != Some(0), "jl.ff_dim must be positive");
            }
            JlSite::Attention | JlSite::Ffn => {
                ensure!(self.from_layer < n_layers, "jl.from_layer {} must be below n_layers {n_layers}", self.from_layer);
                let full = if self.site == JlSite::Attention { head_dim } else { embed_dim };
                ensure!(self.dim < full, "jl.dim {} must be below the projected width {full}", self.dim);
            }
        }
        ensure!(self.site == JlSite::Residual || self.ff_dim.is_none(), "jl.ff_dim is for site \"residual\"");
        Ok(())
    }

    /// Run-name suffix, e.g. `_jl_residual64_at3`.
    pub fn tag(&self) -> String {
        let site = match self.site {
            JlSite::None => return "_jl_none".into(),
            JlSite::Residual => "residual",
            JlSite::Attention => "attn",
            JlSite::Ffn => "ffn",
        };
        let mut tag = format!("_jl_{site}{}_at{}", self.dim, self.from_layer);
        if let Some(ff) = self.ff_dim { tag += &format!("_ff{ff}"); }
        if self.seed != default_seed() { tag += &format!("_s{}", self.seed); }
        if self.trainable { tag += "_learned"; }
        tag
    }

    /// MLP width of the residual site's narrow blocks.
    pub fn narrow_ff_dim(&self, embed_dim: usize, ff_dim: usize) -> usize {
        self.ff_dim.unwrap_or_else(|| (ff_dim * self.dim / embed_dim).max(1))
    }

    pub fn projection<B: Backend>(&self, from: usize, layer: usize, device: &B::Device) -> JlProjection<B> {
        JlProjection::new(from, self.dim, self.seed.wrapping_add(layer as u64), self.trainable, device)
    }
}

/// x -> x R over the last dimension. R is a Param so checkpoints carry it,
/// but it gets no gradient unless trainable.
#[derive(Module, Debug)]
pub struct JlProjection<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
}

impl<B: Backend> JlProjection<B> {
    pub fn new(from: usize, to: usize, seed: u64, trainable: bool, device: &B::Device) -> Self {
        let weight = Tensor::from_data(TensorData::new(gaussian_matrix(from, to, seed), [from, to]), device);
        Self { weight: Param::from_tensor(weight).set_require_grad(trainable) }
    }
    pub fn out_dim(&self) -> usize { self.weight.dims()[1] }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dims = x.dims();
        let from = dims[D - 1];
        let rows = dims[..D - 1].iter().product();
        let mut out = dims;
        out[D - 1] = self.out_dim();
        x.reshape([rows, from]).matmul(self.weight.val()).reshape(out)
    }
}

/// [from, to] row-major entries drawn from N(0, 1/to) (Box-Muller over StdRng).
pub fn gaussian_matrix(from: usize, to: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let std = 1.0 / (to as f64).sqrt();
    (0..from * to)
        .map(|_| {
            let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
            let u2: f64 = rng.r#gen();
            (std * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_specs() {
        let c = JlConfig::parse("residual:64@3").unwrap();
        assert_eq!((c.site, c.dim, c.from_layer, c.trainable), (JlSite::Residual, 64, 3, false));
        assert_eq!(c.tag(), "_jl_residual64_at3");
        let c = JlConfig::parse("attention:16+learned").unwrap();
        assert_eq!((c.site, c.dim, c.from_layer, c.trainable), (JlSite::Attention, 16, 0, true));
        assert_eq!(c.tag(), "_jl_attn16_at0_learned");
        assert_eq!(JlConfig::parse("none").unwrap().tag(), "_jl_none");
        assert!(JlConfig::parse("head:64").is_err());
        assert!(JlConfig::parse("ffn:x").is_err());
    }

    #[test]
    fn validate_limits() {
        let ok = |s: &str| JlConfig::parse(s).unwrap().validate(128, 4, 6);
        assert!(ok("residual:64@6").is_ok());
        assert!(ok("residual:64@0").is_ok());
        assert!(ok("residual:60@3").is_err(), "60 / 4 heads = 15, odd");
        assert!(ok("residual:128@3").is_err());
        assert!(ok("residual:64@7").is_err());
        assert!(ok("attention:16").is_ok());
        assert!(ok("attention:32").is_err(), "head_dim is 32");
        assert!(ok("ffn:64@5").is_ok());
        assert!(ok("ffn:64@6").is_err());
    }

    /// Inner products are kept on average: JL is unbiased, error ~ |x||y| sqrt(2/to).
    #[test]
    fn gaussian_matrix_keeps_inner_products() {
        let (from, to) = (128, 64);
        let r = gaussian_matrix(from, to, 7);
        let mean = r.iter().map(|&v| v as f64).sum::<f64>() / r.len() as f64;
        let var = r.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / r.len() as f64;
        assert!(mean.abs() < 0.01 && (var * to as f64 - 1.0).abs() < 0.05, "mean {mean} var {var}");
        let mut rng = StdRng::seed_from_u64(3);
        let (mut err, n) = (0.0, 200);
        for _ in 0..n {
            let x: Vec<f64> = (0..from).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let y: Vec<f64> = (0..from).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let proj = |v: &[f64]| (0..to).map(|j| (0..from).map(|i| v[i] * r[i * to + j] as f64).sum::<f64>()).collect::<Vec<_>>();
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
            let norm = (dot(&x, &x) * dot(&y, &y)).sqrt();
            err += (dot(&proj(&x), &proj(&y)) - dot(&x, &y)) / norm;
        }
        // Mean signed error over 200 pairs ~ N(0, sqrt(2/64)/sqrt(200)) = 0.0125.
        assert!((err / n as f64).abs() < 0.05, "mean relative error {}", err / n as f64);
    }
}
