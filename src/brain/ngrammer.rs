//! N-Grammer layers for the dense decoder (after lucidrains/n-grammer-pytorch,
//! Roy et al. 2022, "N-Grammer: Augmenting Transformers with latent n-grams").
//!
//! Before each block listed in `layers`, the residual stream x [batch, seq, width]
//! is split into `heads` sub-vectors. Each sub-vector is rms-normalised and snapped
//! to its nearest of `num_clusters` centroids (product quantization), giving a
//! latent token id per head. The bigram (previous id, this id) indexes a learned
//! embedding table per head; the rms-normed bigram embeddings are projected and
//! added to x. The projection starts at zero, so a fresh layer is a no-op.
//!
//! Position 0 uses a BOS state as its previous id (id `num_clusters`), and cached
//! generation carries each layer's last ids in the `KvCache`.
//!
//! The upstream module updates its centroids with an EMA k-means step. Here they are
//! ordinary parameters trained by an auxiliary loss, `loss_weight * mean |c[id] - h|^2`
//! with h detached (the assignment itself has no gradient), which the trainer adds
//! to the LM loss; the bigram table and projection learn from the LM loss.
//! Specs for `--ngram`: `none` or `K[xH][@L1,L2..]`, e.g. `64x4@0`.
use super::model::{RMSNorm, RMSNormConfig};
use burn::{
    module::Param,
    nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig},
    prelude::*,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NgramConfig {
    /// Blocks each N-Grammer layer sits before (0 = on the token embeddings).
    #[serde(default = "default_layers")]
    pub layers: Vec<usize>,
    /// Centroids per head.
    #[serde(default = "default_clusters")]
    pub num_clusters: usize,
    /// Product-quantization heads; the width at each layer must divide by it.
    #[serde(default = "default_heads")]
    pub heads: usize,
    /// Weight of the centroid (k-means) loss added to the LM loss.
    #[serde(default = "default_loss_weight")]
    pub loss_weight: f64,
    /// Seed of the initial centroids; layer i uses seed + i.
    #[serde(default = "default_seed")]
    pub seed: u64,
}
fn default_layers() -> Vec<usize> { vec![0] }
fn default_clusters() -> usize { 64 }
fn default_heads() -> usize { 4 }
fn default_loss_weight() -> f64 { 0.25 }
fn default_seed() -> u64 { 1 }

impl NgramConfig {
    /// `K`, `KxH` or either with `@L1,L2`; `none` yields None.
    pub fn parse(spec: &str) -> anyhow::Result<Option<Self>> {
        if spec.trim() == "none" { return Ok(None); }
        let (shape, layers) = spec.split_once('@').unwrap_or((spec, "0"));
        let (k, h) = shape.split_once('x').unwrap_or((shape, "4"));
        let num = |s: &str, what: &str| s.trim().parse::<usize>().map_err(|_| anyhow::anyhow!("bad ngram {what} {s:?}"));
        let layers = layers.split(',').map(|l| num(l, "layer")).collect::<anyhow::Result<_>>()?;
        Ok(Some(Self { layers, num_clusters: num(k, "clusters")?, heads: num(h, "heads")?,
            loss_weight: default_loss_weight(), seed: default_seed() }))
    }

    /// Width of the residual stream before block `layer`.
    pub fn validate(&self, n_layers: usize, width_at: impl Fn(usize) -> usize) -> anyhow::Result<()> {
        use anyhow::ensure;
        ensure!(!self.layers.is_empty(), "ngram.layers is empty");
        ensure!(self.heads > 0, "ngram.heads must be positive");
        ensure!((2..=256).contains(&self.num_clusters), "ngram.num_clusters must be 2..=256");
        ensure!(self.loss_weight >= 0.0, "ngram.loss_weight must not be negative");
        for (i, &l) in self.layers.iter().enumerate() {
            ensure!(l < n_layers, "ngram layer {l} must be below n_layers {n_layers}");
            ensure!(!self.layers[..i].contains(&l), "ngram layer {l} is listed twice");
            ensure!(width_at(l) % self.heads == 0, "width {} before block {l} does not split into {} ngram heads", width_at(l), self.heads);
        }
        Ok(())
    }

    /// Run-name suffix, e.g. `_ngram64x4_at0-3`.
    pub fn tag(&self) -> String {
        let layers: Vec<String> = self.layers.iter().map(|l| l.to_string()).collect();
        let mut tag = format!("_ngram{}x{}_at{}", self.num_clusters, self.heads, layers.join("-"));
        if self.seed != default_seed() { tag += &format!("_s{}", self.seed); }
        tag
    }

    /// Bigram table rows per head: (previous state incl. BOS) x current id.
    pub fn table_rows(&self) -> usize { self.heads * (self.num_clusters + 1) * self.num_clusters }

    /// Weights at a layer of the given width: centroids, bigram table, projection.
    pub fn param_count(&self, width: usize) -> usize {
        let head_dim = width / self.heads;
        self.heads * self.num_clusters * head_dim + self.table_rows() * head_dim + head_dim + width * width
    }
}

#[derive(Module, Debug)]
pub struct NgramLayer<B: Backend> {
    /// [heads, clusters, head_dim]
    centroids: Param<Tensor<B, 3>>,
    /// Rows `(h * (K + 1) + prev) * K + id`.
    bigrams: Embedding<B>,
    norm: RMSNorm<B>,
    out: Linear<B>,
    clusters: usize,
    heads: usize,
}

/// What a layer hands back: the new residual, its last ids per head ([batch, heads],
/// the carry for cached decoding) and the centroid loss (weighted, [1]).
pub struct NgramOut<B: Backend> {
    pub x: Tensor<B, 3>,
    pub last: Tensor<B, 2, Int>,
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> NgramLayer<B> {
    pub fn new(config: &NgramConfig, width: usize, layer: usize, device: &B::Device) -> Self {
        let (k, h) = (config.num_clusters, config.heads);
        let dh = width / h;
        let data = super::jl::gaussian_matrix(h * k, dh, config.seed.wrapping_add(layer as u64));
        // gaussian_matrix is N(0, 1/dh); rescale to unit variance like the normed inputs.
        let scale = (dh as f32).sqrt();
        let centroids = Tensor::from_data(TensorData::new(data.into_iter().map(|v| v * scale).collect(), [h, k, dh]), device);
        Self {
            centroids: Param::from_tensor(centroids),
            bigrams: EmbeddingConfig::new(config.table_rows(), dh)
                .with_initializer(Initializer::Normal { mean: 0.0, std: 0.02 }).init(device),
            norm: RMSNormConfig::new(dh).init(device),
            out: LinearConfig::new(width, width).with_bias(false).with_initializer(Initializer::Zeros).init(device),
            clusters: k,
            heads: h,
        }
    }

    #[cfg(test)]
    pub(crate) fn centroids(&self) -> Tensor<B, 3> { self.centroids.val() }
    #[cfg(test)]
    pub(crate) fn bigrams_weight(&self) -> Tensor<B, 2> { self.bigrams.weight.val() }

    /// Test helper: gives the output projection non-zero weights so the layer has an effect.
    #[cfg(test)]
    pub(crate) fn randomize_out(&mut self, seed: u64) {
        let [a, b] = self.out.weight.dims();
        let data = super::jl::gaussian_matrix(a, b, seed);
        self.out.weight = Param::from_tensor(Tensor::from_data(TensorData::new(data, [a, b]), &self.out.weight.device()));
    }

    /// `prev`: ids before this chunk ([batch, heads]); None at the start of a sequence.
    /// `loss_weight` scales the returned centroid loss.
    pub fn forward(&self, x: Tensor<B, 3>, prev: Option<Tensor<B, 2, Int>>, loss_weight: f64) -> NgramOut<B> {
        let [batch, seq, width] = x.dims();
        let (heads, k) = (self.heads, self.clusters);
        let dh = width / heads;
        let device = x.device();

        // [heads, batch * seq, dh], rms-normalised, detached: assignment only.
        let rows = batch * seq;
        let h = x.clone().detach().reshape([batch, seq, heads, dh]);
        let rms = (h.clone().powf_scalar(2.0).mean_dim(3) + 1e-6).sqrt();
        let h = (h / rms).swap_dims(0, 2).swap_dims(1, 2).reshape([heads, rows, dh]);

        let c = self.centroids.val();
        // argmin |h - c|^2 = argmin (|c|^2 - 2 h.c). The transpose is copied contiguous
        // first (strided-transpose matmul is wrong on WGPU for some widths, see tied_logits).
        let c_t = c.clone().detach().swap_dims(1, 2).reshape([heads * dh * k]).reshape([heads, dh, k]);
        let dist = c.clone().detach().powf_scalar(2.0).sum_dim(2).reshape([heads, 1, k]) - h.clone().matmul(c_t) * 2.0;
        let ids = dist.argmin(2).reshape([heads, rows]); // [heads, rows] Int

        // Centroid loss: pull each used centroid toward the vectors assigned to it.
        let head_base = Tensor::<B, 1, Int>::from_ints((0..heads as i32).map(|i| i * k as i32).collect::<Vec<_>>().as_slice(), &device)
            .reshape([heads, 1]);
        let flat = (ids.clone() + head_base).reshape([heads * rows]);
        let q = c.reshape([heads * k, dh]).select(0, flat).reshape([heads, rows, dh]);
        let loss = (q - h).powf_scalar(2.0).mean().reshape([1]) * loss_weight;

        // [batch, seq, heads] ids, previous ids shifted in (BOS state = k).
        let ids = ids.swap_dims(0, 1).reshape([batch, seq, heads]);
        let last = ids.clone().narrow(1, seq - 1, 1).reshape([batch, heads]);
        let first = match prev {
            Some(p) => p.reshape([batch, 1, heads]),
            None => Tensor::<B, 3, Int>::full([batch, 1, heads], k as i32, &device),
        };
        let prev_ids = if seq > 1 { Tensor::cat(vec![first, ids.clone().narrow(1, 0, seq - 1)], 1) } else { first };
        let head_off = Tensor::<B, 1, Int>::from_ints(
            (0..heads as i32).map(|i| i * ((k + 1) * k) as i32).collect::<Vec<_>>().as_slice(), &device)
            .reshape([1, 1, heads]);
        let rows_idx = (prev_ids * (k as i32) + ids + head_off).reshape([batch, seq * heads]);

        let emb = self.bigrams.forward(rows_idx).reshape([batch, seq, heads, dh]);
        let emb = self.norm.forward(emb).reshape([batch, seq, width]);
        NgramOut { x: x + self.out.forward(emb), last, loss }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_specs() {
        let c = NgramConfig::parse("64x4@0,3").unwrap().unwrap();
        assert_eq!((c.num_clusters, c.heads, c.layers.clone()), (64, 4, vec![0, 3]));
        assert_eq!(c.tag(), "_ngram64x4_at0-3");
        let c = NgramConfig::parse("32").unwrap().unwrap();
        assert_eq!((c.num_clusters, c.heads, c.layers), (32, 4, vec![0]));
        assert!(NgramConfig::parse("none").unwrap().is_none());
        assert!(NgramConfig::parse("x").is_err());
    }

    #[test]
    fn validate_limits() {
        let ok = |s: &str| NgramConfig::parse(s).unwrap().unwrap().validate(6, |_| 128);
        assert!(ok("64x4@0").is_ok());
        assert!(ok("64x4@0,5").is_ok());
        assert!(ok("64x4@6").is_err());
        assert!(ok("64x4@1,1").is_err());
        assert!(ok("64x3").is_err(), "128 / 3");
        assert!(ok("1x4").is_err());
        let one: NgramConfig = serde_json::from_str(r#"{"num_clusters": 16}"#).unwrap();
        assert_eq!((one.heads, one.layers), (4, vec![0]));
    }
}
