//! Pass 1 weight forensics for Yumon checkpoints.
//!
//! Usage:
//!   analyze_weights_pass1 <config.json | run dir | out dir | model.bin> [options]
//!
//! Options:
//!   --run <substr>        pick the run subdirectory whose name contains <substr>
//!   --compare <target>    second checkpoint (same forms as above); per-tensor drift
//!   --compare-run <substr> run filter for the --compare target
//!   --raw                 force the byte scanner (no model load, no names)
//!   --all                 print every tensor, not just the flagged/top ones
//!   --top <n>             rows in the "worst" lists (default 12)
//!   --csv <path>          write per-tensor stats as CSV
//!   --embed-dim/--ff-dim/--experts/--layers/--vocab/--shared-experts <n>
//!                         raw-mode shape overrides
//!
//! Named mode (default) loads the MoE checkpoint through `YumonMoeBrain::load` and
//! walks every parameter with its module path. Raw mode scans model.bin for runs of
//! plausible f32 values and is the fallback for checkpoints that do not load.

use anyhow::{bail, Context, Result};
use burn::module::{Module, ModuleVisitor, Param};
use burn::prelude::*;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use yumon_pet::brain::moe_model::YumonMoeBrain;

type BackendType = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub ff_dim: usize,
    pub num_experts: usize,
    pub num_shared_experts: usize,
    pub n_layers: usize,
    pub attn_heads: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 4096,
            embed_dim: 64,
            ff_dim: 256,
            num_experts: 4,
            num_shared_experts: 0,
            n_layers: 4,
            attn_heads: 4,
        }
    }
}

impl ModelConfig {
    pub fn update_from_json(&mut self, val: &serde_json::Value) {
        let extract = |k: &str| -> Option<usize> {
            val.get(k)
                .and_then(|x| x.as_u64())
                .or_else(|| val.get("config").and_then(|c| c.get(k)).and_then(|x| x.as_u64()))
                .map(|x| x as usize)
        };
        if let Some(v) = extract("vocab_size") { self.vocab_size = v; }
        if let Some(v) = extract("embed_dim") { self.embed_dim = v; }
        if let Some(v) = extract("ff_dim") { self.ff_dim = v; }
        if let Some(v) = extract("num_experts") { self.num_experts = v; }
        if let Some(v) = extract("num_shared_experts") { self.num_shared_experts = v; }
        if let Some(v) = extract("n_layers") { self.n_layers = v; }
        if let Some(v) = extract("attn_heads") { self.attn_heads = v; }
    }
}

struct Args {
    target: Option<PathBuf>,
    run: Option<String>,
    compare: Option<PathBuf>,
    compare_run: Option<String>,
    raw: bool,
    all: bool,
    top: usize,
    csv: Option<PathBuf>,
    overrides: Vec<(String, usize)>,
}

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Args {
        target: None,
        run: None,
        compare: None,
        compare_run: None,
        raw: false,
        all: false,
        top: 12,
        csv: None,
        overrides: Vec::new(),
    };
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        let next = || argv.get(i + 1).cloned().with_context(|| format!("{a} needs a value"));
        match a {
            "--run" => { args.run = Some(next()?); i += 2; }
            "--compare" => { args.compare = Some(PathBuf::from(next()?)); i += 2; }
            "--compare-run" => { args.compare_run = Some(next()?); i += 2; }
            "--raw" => { args.raw = true; i += 1; }
            "--all" => { args.all = true; i += 1; }
            "--top" => { args.top = next()?.parse().context("--top")?; i += 2; }
            "--csv" => { args.csv = Some(PathBuf::from(next()?)); i += 2; }
            // Kept for compatibility: the config is now read from the target itself.
            "--config" => { if args.target.is_none() { args.target = Some(PathBuf::from(next()?)); } i += 2; }
            "--embed-dim" | "--ff-dim" | "--experts" | "--num-experts" | "--layers" | "--n-layers"
            | "--vocab" | "--shared-experts" | "--num-shared-experts" => {
                let key = match a {
                    "--embed-dim" => "embed_dim",
                    "--ff-dim" => "ff_dim",
                    "--experts" | "--num-experts" => "num_experts",
                    "--layers" | "--n-layers" => "n_layers",
                    "--vocab" => "vocab_size",
                    _ => "num_shared_experts",
                };
                args.overrides.push((key.to_string(), next()?.parse().with_context(|| a.to_string())?));
                i += 2;
            }
            _ if a.starts_with("--") => bail!("unknown option {a}"),
            _ => {
                if args.target.is_some() {
                    bail!("unexpected extra argument {a}");
                }
                args.target = Some(PathBuf::from(a));
                i += 1;
            }
        }
    }
    Ok(args)
}

// ---------------------------------------------------------------------------
// Checkpoint resolution
// ---------------------------------------------------------------------------

/// A resolved checkpoint: the model.bin and the run directory holding it.
struct Target {
    model_bin: PathBuf,
    run_dir: PathBuf,
    /// The training config the user pointed at, if any.
    config_json: Option<serde_json::Value>,
}

fn mtime(p: &Path) -> SystemTime {
    std::fs::metadata(p).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH)
}

fn age(p: &Path) -> String {
    match SystemTime::now().duration_since(mtime(p)) {
        Ok(d) if d.as_secs() < 120 => format!("{}s ago", d.as_secs()),
        Ok(d) if d.as_secs() < 7200 => format!("{}m ago", d.as_secs() / 60),
        Ok(d) if d.as_secs() < 172_800 => format!("{}h ago", d.as_secs() / 3600),
        Ok(d) => format!("{}d ago", d.as_secs() / 86_400),
        Err(_) => "in the future".to_string(),
    }
}

/// Run directories one level below `dir` (or `dir` itself) that hold a model.bin.
fn runs_with_model(dir: &Path) -> Vec<PathBuf> {
    let mut runs = Vec::new();
    if dir.join("model.bin").is_file() {
        runs.push(dir.to_path_buf());
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() && p.join("model.bin").is_file() {
                runs.push(p);
            }
        }
    }
    runs
}

/// Subdirectories that look like runs (have code-config.json) but no model.bin yet.
fn runs_without_model(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() && !p.join("model.bin").is_file() && p.join("code-config.json").is_file() {
                out.push(p);
            }
        }
    }
    out
}

/// The `_opt_<kind>` token a config's run names carry, or None for AdamW.
fn optimizer_tag(config: &serde_json::Value) -> Option<String> {
    config
        .get("optimizer")
        .and_then(|o| o.get("kind"))
        .and_then(|k| k.as_str())
        .map(|k| format!("_opt_{k}"))
}

fn pick_run(dir: &Path, filter: Option<&str>, config: Option<&serde_json::Value>) -> Result<PathBuf> {
    let mut runs = runs_with_model(dir);
    if let Some(f) = filter {
        runs.retain(|r| r.file_name().map_or(false, |n| n.to_string_lossy().contains(f)));
    }
    if runs.is_empty() {
        let mut msg = format!("no model.bin found in {} or its run subdirectories", dir.display());
        if let Some(f) = filter {
            msg.push_str(&format!(" matching --run {f:?}"));
        }
        for p in runs_without_model(dir) {
            msg.push_str(&format!(
                "\n  run without a checkpoint yet (training has not saved model.bin): {} (created {})",
                p.display(),
                age(&p)
            ));
        }
        // Sibling out_dirs sharing the prefix, e.g. `-beta` vs `-beta-hilr`.
        if let (Some(parent), Some(name)) = (dir.parent(), dir.file_name()) {
            let name = name.to_string_lossy().to_string();
            if let Ok(entries) = std::fs::read_dir(parent) {
                let mut siblings: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.is_dir()
                            && p.as_path() != dir
                            && p.file_name().map_or(false, |n| n.to_string_lossy().starts_with(&name))
                    })
                    .flat_map(|p| runs_with_model(&p))
                    .collect();
                siblings.sort_by_key(|p| std::cmp::Reverse(mtime(&p.join("model.bin"))));
                for s in siblings {
                    msg.push_str(&format!(
                        "\n  sibling checkpoint: {} (model.bin {})",
                        s.display(),
                        age(&s.join("model.bin"))
                    ));
                }
            }
        }
        bail!(msg);
    }
    // Prefer runs trained with the config's optimizer, then the newest model.bin.
    let tag = config.map(optimizer_tag);
    runs.sort_by_key(|r| {
        let name = r.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let matches = match &tag {
            Some(Some(t)) => name.contains(t.as_str()),
            Some(None) => !name.contains("_opt_"),
            None => true,
        };
        (std::cmp::Reverse(matches), std::cmp::Reverse(mtime(&r.join("model.bin"))))
    });
    if runs.len() > 1 {
        println!("  {} runs with a model.bin; using the first (pass --run <substr> to choose):", runs.len());
        for r in &runs {
            println!("    {} (model.bin {})", r.display(), age(&r.join("model.bin")));
        }
    }
    Ok(runs.remove(0))
}

fn resolve_target(input: Option<&Path>, filter: Option<&str>) -> Result<Target> {
    let input = input.map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    if input.is_file() && input.extension().map_or(false, |e| e == "json") {
        let text = std::fs::read_to_string(&input)?;
        let config: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", input.display()))?;
        let out_dir = config
            .get("out_dir")
            .or_else(|| config.get("config").and_then(|c| c.get("out_dir")))
            .and_then(|s| s.as_str())
            .with_context(|| format!("{} has no out_dir; pass the checkpoint directory instead", input.display()))?;
        let run_dir = pick_run(Path::new(out_dir), filter, Some(&config))?;
        return Ok(Target { model_bin: run_dir.join("model.bin"), run_dir, config_json: Some(config) });
    }
    if input.is_file() {
        let run_dir = input.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        return Ok(Target { model_bin: input, run_dir, config_json: None });
    }
    if input.is_dir() {
        let run_dir = pick_run(&input, filter, None)?;
        return Ok(Target { model_bin: run_dir.join("model.bin"), run_dir, config_json: None });
    }
    bail!("{} does not exist", input.display())
}

fn load_model_config(target: &Target, overrides: &[(String, usize)]) -> ModelConfig {
    let mut config = ModelConfig::default();
    if let Some(c) = &target.config_json {
        config.update_from_json(c);
    }
    for name in ["code-config.json", "metadata.json"] {
        if let Ok(text) = std::fs::read_to_string(target.run_dir.join(name)) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                config.update_from_json(&v);
            }
        }
    }
    let as_json = serde_json::Value::Object(
        overrides.iter().map(|(k, v)| (k.clone(), serde_json::json!(v))).collect(),
    );
    config.update_from_json(&as_json);
    config
}

// ---------------------------------------------------------------------------
// Run context: metadata, optimizer.jsonl
// ---------------------------------------------------------------------------

/// code-config.json stores the f32 config values, so 0.85 reads back as 0.8500000238.
fn round_floats(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Number(n) if n.is_f64() => {
            let x = n.as_f64().unwrap();
            serde_json::json!((x as f32).to_string().parse::<f64>().unwrap_or(x))
        }
        serde_json::Value::Object(m) => {
            serde_json::Value::Object(m.iter().map(|(k, v)| (k.clone(), round_floats(v))).collect())
        }
        serde_json::Value::Array(a) => serde_json::Value::Array(a.iter().map(round_floats).collect()),
        other => other.clone(),
    }
}

/// `saved` may carry defaults the user's config omits; only keys in `user` are checked.
fn json_close(saved: &serde_json::Value, user: &serde_json::Value) -> bool {
    use serde_json::Value::*;
    match (saved, user) {
        (Number(x), Number(y)) => {
            let (x, y) = (x.as_f64().unwrap_or(f64::NAN), y.as_f64().unwrap_or(f64::NAN));
            (x - y).abs() <= 1e-6 * x.abs().max(y.abs()).max(1e-12)
        }
        (Object(x), Object(y)) => {
            y.iter().all(|(k, w)| x.get(k).map_or(w.is_null(), |v| json_close(v, w)))
        }
        (Array(x), Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(v, w)| json_close(v, w)),
        _ => saved == user,
    }
}

fn print_run_context(target: &Target) {
    let model_age = age(&target.model_bin);
    println!("   Run dir:            {}", target.run_dir.display());
    println!("   model.bin:          {:.1} MiB, written {}",
        std::fs::metadata(&target.model_bin).map(|m| m.len() as f64 / 1048576.0).unwrap_or(0.0),
        model_age);

    if let Ok(text) = std::fs::read_to_string(target.run_dir.join("metadata.json")) {
        if let Ok(m) = serde_json::from_str::<serde_json::Value>(&text) {
            let get = |k: &str| m.get(k).map(|v| v.to_string()).unwrap_or_else(|| "-".into());
            println!(
                "   Metadata:           epochs_trained={} final_loss={} val_loss={} tied={} kv_heads={}",
                get("epochs_trained"), get("final_loss"), get("val_loss"), get("tie_embeddings"), get("kv_heads")
            );
        }
    }

    let cfg_path = target.run_dir.join("code-config.json");
    let cfg_text = std::fs::read_to_string(&cfg_path).ok();
    // code-config.json nests the training config under "config".
    let saved_optimizer = cfg_text
        .as_deref()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(t).ok())
        .map(|v| v.get("config").cloned().unwrap_or(v))
        .map(|c| c.get("optimizer").cloned().unwrap_or(serde_json::Value::Null));
    let user_optimizer = target
        .config_json
        .as_ref()
        .map(|c| c.get("optimizer").cloned().unwrap_or(serde_json::Value::Null));
    match saved_optimizer.as_ref().or(user_optimizer.as_ref()) {
        Some(serde_json::Value::Null) | None => println!("   Optimizer:          AdamW (no optimizer section)"),
        Some(o) => println!("   Optimizer:          {}", round_floats(o)),
    }
    // The run directory was written by the config the run started with; if the
    // config passed here has changed since, say so rather than analyse silently.
    if let (Some(saved), Some(user)) = (&saved_optimizer, &user_optimizer) {
        if !json_close(saved, user) {
            println!(
                "   [!] the run's code-config.json optimizer differs from the config you passed:\n       run:    {}\n       config: {}",
                round_floats(saved),
                round_floats(user)
            );
        }
    }

    let log = target.run_dir.join("optimizer.jsonl");
    let Ok(text) = std::fs::read_to_string(&log) else { return };
    let rows: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    if rows.is_empty() {
        return;
    }
    println!("   optimizer.jsonl:    {} entries, last written {}", rows.len(), age(&log));
    if mtime(&log) > mtime(&target.model_bin) {
        println!("   [i] optimizer.jsonl is newer than model.bin: the weights are from an earlier step than the last stats row.");
    }
    let fmt = |r: &serde_json::Value| {
        let stats = r.get("stats").map(|s| s.to_string()).unwrap_or_default();
        format!(
            "step {} epoch {} lr {:.3e} {}",
            r.get("step").unwrap_or(&serde_json::Value::Null),
            r.get("epoch").unwrap_or(&serde_json::Value::Null),
            r.get("lr").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
            stats
        )
    };
    println!("     first: {}", fmt(&rows[0]));
    if rows.len() > 1 {
        println!("     last:  {}", fmt(rows.last().unwrap()));
    }
}

// ---------------------------------------------------------------------------
// Named tensors
// ---------------------------------------------------------------------------

struct NamedTensor {
    name: String,
    dims: Vec<usize>,
    data: Vec<f32>,
}

struct Collector {
    path: Vec<String>,
    tensors: Vec<NamedTensor>,
    error: Option<String>,
}

impl<B: Backend> ModuleVisitor<B> for Collector {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let t = param.val();
        let dims = t.dims().to_vec();
        match t.into_data().convert::<f32>().to_vec::<f32>() {
            Ok(data) => self.tensors.push(NamedTensor { name: self.path.join("."), dims, data }),
            Err(e) => self.error = Some(format!("{}: {e:?}", self.path.join("."))),
        }
    }
}

fn load_named(target: &Target) -> Result<Vec<NamedTensor>> {
    for need in ["metadata.json", "tokenizer.json"] {
        if !target.run_dir.join(need).is_file() {
            bail!("{} has no {need} (needed to load the model)", target.run_dir.display());
        }
    }
    let device: <BackendType as Backend>::Device = Default::default();
    let dir = target.run_dir.to_str().context("non-UTF-8 run dir")?;
    let (model, _, _) = YumonMoeBrain::<BackendType>::load(dir, &device)?;
    let mut collector = Collector { path: Vec::new(), tensors: Vec::new(), error: None };
    model.visit(&mut collector);
    if let Some(e) = collector.error {
        bail!("reading tensor {e}");
    }
    Ok(collector.tensors)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Category {
    Embedding,
    Head,
    Attention,
    Expert,
    SharedExpert,
    Router,
    ExitGate,
    Norm,
    Other,
}

impl Category {
    fn of(name: &str) -> Self {
        let has = |s: &str| name.split('.').any(|p| p == s);
        if name.contains("norm") && name.ends_with("weight") && !has("score") {
            Category::Norm
        } else if has("router") {
            Category::Router
        } else if has("exit_gates") {
            Category::ExitGate
        } else if has("shared_experts") {
            Category::SharedExpert
        } else if has("experts") {
            Category::Expert
        } else if has("q") || has("k") || has("v") || has("o") {
            Category::Attention
        } else if has("embedding") {
            Category::Embedding
        } else if has("token_head") {
            Category::Head
        } else {
            Category::Other
        }
    }
    fn label(self) -> &'static str {
        match self {
            Category::Embedding => "embedding",
            Category::Head => "token_head",
            Category::Attention => "attention",
            Category::Expert => "experts",
            Category::SharedExpert => "shared_experts",
            Category::Router => "router",
            Category::ExitGate => "exit_gates",
            Category::Norm => "norm gains",
            Category::Other => "other",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    NonFinite,
    Zero,
    Collapsed,
    RandomLike,
    Learned,
    GainUntouched,
    GainDrifted,
    GainOk,
    Vector,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::NonFinite => "NONFINITE",
            Status::Zero => "ALL-ZERO",
            Status::Collapsed => "COLLAPSED",
            Status::RandomLike => "RANDOM-LIKE",
            Status::Learned => "learned",
            Status::GainUntouched => "UNTOUCHED (=1.0)",
            Status::GainDrifted => "GAIN DRIFT",
            Status::GainOk => "ok",
            Status::Vector => "-",
        }
    }
    fn flagged(self) -> bool {
        !matches!(self, Status::Learned | Status::GainOk | Status::Vector)
    }
}

/// Matrix health thresholds.
/// - rank1 share above this AND above COLLAPSE_OVER_RANDOM x the share a random
///   matrix of the same shape has = most energy in one direction (anisotropic
///   collapse). The relative test keeps narrow matrices (256x8 routers sit near 17%
///   at init) from being flagged for their shape alone.
const COLLAPSE_RANK1_PCT: f64 = 25.0;
const COLLAPSE_OVER_RANDOM: f64 = 3.0;
/// - top singular value within this factor of the Marchenko-Pastur edge of a random
///   matrix with the same RMS = no structure beyond the init's noise.
const RANDOM_MP_RATIO: f64 = 1.10;

struct Stats {
    name: String,
    dims: Vec<usize>,
    category: Category,
    rms: f64,
    mean: f64,
    max_abs: f64,
    zero_frac: f64,
    nonfinite: usize,
    /// 2D only.
    s1: f64,
    s2: f64,
    stable_rank: f64,
    rank1_pct: f64,
    mp_ratio: f64,
    /// Share of the energy in the mean row (a common offset shared by every row).
    mean_share: f64,
    /// Norm gains only: max |g - 1| and mean gain.
    gain_dev: f64,
    gain_mean: f64,
    status: Status,
}

fn matrix_shape(dims: &[usize]) -> Option<(usize, usize)> {
    if dims.len() < 2 {
        return None;
    }
    let rows = dims[0];
    let cols: usize = dims[1..].iter().product();
    (rows.min(cols) >= 4).then_some((rows, cols))
}

fn compute_stats(name: &str, dims: &[usize], data: &[f32]) -> Stats {
    let category = Category::of(name);
    let numel = data.len();
    let mut sum = 0.0f64;
    let mut sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut zeros = 0usize;
    let mut nonfinite = 0usize;
    for &x in data {
        if !x.is_finite() {
            nonfinite += 1;
            continue;
        }
        let x = x as f64;
        sum += x;
        sq += x * x;
        max_abs = max_abs.max(x.abs());
        if x == 0.0 {
            zeros += 1;
        }
    }
    let n = numel.max(1) as f64;
    let mut s = Stats {
        name: name.to_string(),
        dims: dims.to_vec(),
        category,

        rms: (sq / n).sqrt(),
        mean: sum / n,
        max_abs,
        zero_frac: zeros as f64 / n,
        nonfinite,
        s1: 0.0,
        s2: 0.0,
        stable_rank: 0.0,
        rank1_pct: 0.0,
        mp_ratio: 0.0,
        mean_share: 0.0,
        gain_dev: 0.0,
        gain_mean: 0.0,
        status: Status::Vector,
    };

    if nonfinite > 0 {
        s.status = Status::NonFinite;
        return s;
    }
    if category == Category::Norm {
        s.gain_mean = s.mean;
        s.gain_dev = data.iter().map(|&g| (g as f64 - 1.0).abs()).fold(0.0, f64::max);
        s.status = if s.gain_dev == 0.0 {
            Status::GainUntouched
        } else if !(0.2..=5.0).contains(&s.gain_mean) {
            Status::GainDrifted
        } else {
            Status::GainOk
        };
        return s;
    }
    if sq == 0.0 {
        s.status = Status::Zero;
        return s;
    }
    let Some((rows, cols)) = matrix_shape(dims) else { return s };

    let (s1, s2) = top2_singular_values(data, rows, cols);
    s.s1 = s1;
    s.s2 = s2;
    s.stable_rank = sq / (s1 * s1).max(1e-30);
    s.rank1_pct = s1 * s1 / sq * 100.0;
    let sigma = s.rms;
    let mp_edge = sigma * ((rows as f64).sqrt() + (cols as f64).sqrt());
    s.mp_ratio = s1 / mp_edge.max(1e-30);
    let mut mean_row = vec![0.0f64; cols];
    for r in 0..rows {
        for (m, &x) in mean_row.iter_mut().zip(&data[r * cols..(r + 1) * cols]) {
            *m += x as f64;
        }
    }
    let mean_sq: f64 = mean_row.iter().map(|m| (m / rows as f64).powi(2)).sum();
    s.mean_share = mean_sq * rows as f64 / sq;

    let random_rank1_pct = mp_edge * mp_edge / sq * 100.0;
    s.status = if s.rank1_pct > COLLAPSE_RANK1_PCT.max(COLLAPSE_OVER_RANDOM * random_rank1_pct) {
        Status::Collapsed
    } else if s.mp_ratio < RANDOM_MP_RATIO {
        Status::RandomLike
    } else {
        Status::Learned
    };
    s
}

/// Top two singular values by power iteration with deflation (f64 accumulation).
fn top2_singular_values(w: &[f32], rows: usize, cols: usize) -> (f64, f64) {
    let (s1, u1, v1) = power_iter(w, rows, cols, None, 40);
    let (s2, _, _) = power_iter(w, rows, cols, Some((s1, &u1, &v1)), 40);
    (s1, s2)
}

fn power_iter(
    w: &[f32],
    rows: usize,
    cols: usize,
    deflate: Option<(f64, &[f64], &[f64])>,
    iters: usize,
) -> (f64, Vec<f64>, Vec<f64>) {
    let mut v: Vec<f64> = (0..cols).map(|j| (j as f64 * 1.618).sin() + 0.1).collect();
    norm_vec(&mut v);
    let mut u = vec![0.0f64; rows];
    let mut sigma = 0.0;

    for _ in 0..iters {
        // u = (W - s1 u1 v1^T) v
        let vd = deflate.map_or(0.0, |(_, _, v1)| dot(v1, &v));
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let mut acc = 0.0f64;
            for c in 0..cols {
                acc += row[c] as f64 * v[c];
            }
            if let Some((s1, u1, _)) = deflate {
                acc -= s1 * u1[r] * vd;
            }
            u[r] = acc;
        }
        norm_vec(&mut u);
        // v = (W - s1 u1 v1^T)^T u
        let ud = deflate.map_or(0.0, |(_, u1, _)| dot(u1, &u));
        v.iter_mut().for_each(|x| *x = 0.0);
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let ur = u[r];
            for c in 0..cols {
                v[c] += row[c] as f64 * ur;
            }
        }
        if let Some((s1, _, v1)) = deflate {
            for c in 0..cols {
                v[c] -= s1 * v1[c] * ud;
            }
        }
        sigma = norm_vec(&mut v);
    }
    (sigma, u, v)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm_vec(v: &mut [f64]) -> f64 {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 1e-30 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    n
}

fn dims_str(d: &[usize]) -> String {
    d.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x")
}

fn print_stats_row(s: &Stats) {
    if s.category == Category::Norm {
        println!(
            "  {:<44} {:>10} | gain mean {:6.3} max|g-1| {:9.3e} | {}",
            s.name, dims_str(&s.dims), s.gain_mean, s.gain_dev, s.status.label()
        );
    } else if s.s1 > 0.0 {
        println!(
            "  {:<44} {:>10} | rms {:8.4} | s1 {:7.3} s1/s2 {:5.2} | srank {:7.2} | rank1 {:5.1}% | s1/MP {:5.2} | mean {:4.1}% | {}",
            s.name, dims_str(&s.dims), s.rms, s.s1, s.s1 / s.s2.max(1e-30), s.stable_rank,
            s.rank1_pct, s.mp_ratio, s.mean_share * 100.0, s.status.label()
        );
    } else {
        println!(
            "  {:<44} {:>10} | rms {:8.4} | mean {:9.2e} | max {:7.3} | zeros {:4.1}% | {}",
            s.name, dims_str(&s.dims), s.rms, s.mean, s.max_abs, s.zero_frac * 100.0, s.status.label()
        );
    }
}

// ---------------------------------------------------------------------------
// MoE structure
// ---------------------------------------------------------------------------

/// `blocks.<i>.…` -> i
fn block_index(name: &str) -> Option<usize> {
    let mut parts = name.split('.');
    (parts.next()? == "blocks").then(|| parts.next()?.parse().ok()).flatten()
}

/// `….experts.<j>.…` -> j (routed experts only)
fn expert_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    let pos = parts.iter().position(|p| *p == "experts")?;
    parts.get(pos + 1)?.parse().ok()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

struct LayerSummary {
    layer: usize,
    attn_srank: f64,
    attn_mp: f64,
    expert_norm_min: f64,
    expert_norm_max: f64,
    expert_cos_mean: f64,
    expert_cos_max: f64,
    expert_mp: f64,
    router_col_min: f64,
    router_col_max: f64,
    router_bias_range: f64,
    gains: Vec<f64>,
    /// Routed experts whose matrices still look like random init (mean s1/MP).
    starved: Vec<(usize, f64)>,
    /// Routed experts with most of their matrices COLLAPSED (mean rank-1 share).
    collapsed: Vec<(usize, f64)>,
}

/// An expert whose mean s1/MP stays below this has learned little beyond its init.
const STARVED_EXPERT_MP: f64 = 1.25;

fn layer_summaries(tensors: &[NamedTensor], stats: &[Stats]) -> Vec<LayerSummary> {
    let mut by_layer: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, t) in tensors.iter().enumerate() {
        if let Some(l) = block_index(&t.name) {
            by_layer.entry(l).or_default().push(i);
        }
    }
    by_layer
        .into_par_iter()
        .map(|(layer, idx)| {
            let mean = |v: &[f64]| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
            let attn: Vec<&Stats> = idx
                .iter()
                .map(|&i| &stats[i])
                .filter(|s| s.category == Category::Attention && s.s1 > 0.0)
                .collect();
            let attn_srank = mean(&attn.iter().map(|s| s.stable_rank).collect::<Vec<_>>());
            let attn_mp = mean(&attn.iter().map(|s| s.mp_ratio).collect::<Vec<_>>());

            // Concatenate each routed expert's weight matrices (biases excluded).
            let mut experts: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
            let mut expert_mps = Vec::new();
            let mut per_expert: BTreeMap<usize, Vec<&Stats>> = BTreeMap::new();
            for &i in &idx {
                let t = &tensors[i];
                if stats[i].category == Category::Expert && t.dims.len() == 2 {
                    if let Some(e) = expert_index(&t.name) {
                        experts.entry(e).or_default().extend_from_slice(&t.data);
                        expert_mps.push(stats[i].mp_ratio);
                        per_expert.entry(e).or_default().push(&stats[i]);
                    }
                }
            }
            let mut starved = Vec::new();
            let mut collapsed = Vec::new();
            for (e, ss) in &per_expert {
                let n = ss.len() as f64;
                let mp = ss.iter().map(|s| s.mp_ratio).sum::<f64>() / n;
                if mp < STARVED_EXPERT_MP {
                    starved.push((*e, mp));
                }
                if ss.iter().filter(|s| s.status == Status::Collapsed).count() * 2 > ss.len() {
                    collapsed.push((*e, ss.iter().map(|s| s.rank1_pct).sum::<f64>() / n));
                }
            }
            let flat: Vec<&Vec<f32>> = experts.values().collect();
            let norms: Vec<f64> = flat
                .iter()
                .map(|v| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt())
                .collect();
            let mut cos = Vec::new();
            for a in 0..flat.len() {
                for b in a + 1..flat.len() {
                    if flat[a].len() == flat[b].len() {
                        cos.push(cosine(flat[a], flat[b]));
                    }
                }
            }

            // Router weight is [width, experts]; column e scores expert e.
            let mut router_col_min = f64::NAN;
            let mut router_col_max = f64::NAN;
            let mut router_bias_range = f64::NAN;
            for &i in &idx {
                let t = &tensors[i];
                if stats[i].category != Category::Router {
                    continue;
                }
                if t.name.ends_with("weight") && t.dims.len() == 2 {
                    let (rows, cols) = (t.dims[0], t.dims[1]);
                    let cols_norm: Vec<f64> = (0..cols)
                        .map(|c| (0..rows).map(|r| (t.data[r * cols + c] as f64).powi(2)).sum::<f64>().sqrt())
                        .collect();
                    router_col_min = cols_norm.iter().cloned().fold(f64::INFINITY, f64::min);
                    router_col_max = cols_norm.iter().cloned().fold(0.0, f64::max);
                } else if t.name.ends_with("bias") {
                    let lo = t.data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let hi = t.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    router_bias_range = (hi - lo) as f64;
                }
            }

            let gains = idx
                .iter()
                .map(|&i| &stats[i])
                .filter(|s| s.category == Category::Norm)
                .map(|s| s.gain_mean)
                .collect();

            LayerSummary {
                layer,
                attn_srank,
                attn_mp,
                expert_norm_min: norms.iter().cloned().fold(f64::INFINITY, f64::min),
                expert_norm_max: norms.iter().cloned().fold(0.0, f64::max),
                expert_cos_mean: mean(&cos),
                expert_cos_max: cos.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                expert_mp: mean(&expert_mps),
                router_col_min,
                router_col_max,
                router_bias_range,
                gains,
                starved,
                collapsed,
            }
        })
        .collect()
}

fn print_layer_table(layers: &[LayerSummary]) {
    if layers.is_empty() {
        return;
    }
    println!("\n=== PER-LAYER SUMMARY ===");
    println!("  attn: mean stable rank / mean s1-over-MP-edge of q,k,v,o (MP ~1.0 = random init)");
    println!("  experts: L2 of each routed expert's weights (min-max), pairwise cosine (mean / max), mean s1/MP");
    println!("  router: per-expert column L2 (min-max), bias range; gains: mean RMSNorm gain per norm in the block");
    println!("  starved: experts with mean s1/MP < {STARVED_EXPERT_MP}; collapsed: experts with most matrices COLLAPSED");
    println!(
        "  {:>5} | {:>6} {:>5} | {:>15} {:>13} {:>5} | {:>13} {:>7} | {:>9} | {:<10} {:<10}",
        "layer", "srank", "s1/MP", "expert L2", "cos mean/max", "s1/MP", "router col", "bias rng", "gains",
        "starved", "collapsed"
    );
    let ids = |v: &[(usize, f64)]| {
        if v.is_empty() { "-".to_string() } else { v.iter().map(|(e, _)| e.to_string()).collect::<Vec<_>>().join(",") }
    };
    for l in layers {
        let gains = l.gains.iter().map(|g| format!("{g:.2}")).collect::<Vec<_>>().join(" ");
        let bias = if l.router_bias_range.is_nan() { "-".to_string() } else { format!("{:.3}", l.router_bias_range) };
        println!(
            "  {:>5} | {:>6.1} {:>5.2} | {:>7.2}-{:<7.2} {:>6.3}/{:<6.3} {:>5.2} | {:>6.3}-{:<6.3} {:>7} | {:>9} | {:<10} {:<10}",
            l.layer,
            l.attn_srank,
            l.attn_mp,
            l.expert_norm_min,
            l.expert_norm_max,
            l.expert_cos_mean,
            l.expert_cos_max,
            l.expert_mp,
            l.router_col_min,
            l.router_col_max,
            bias,
            gains,
            ids(&l.starved),
            ids(&l.collapsed)
        );
    }
}

fn layer_flags(layers: &[LayerSummary]) -> Vec<String> {
    let mut flags = Vec::new();
    for l in layers {
        if l.expert_cos_max > 0.5 {
            flags.push(format!(
                "layer {}: experts converging (max pairwise cosine {:.2}); routing may be redundant",
                l.layer, l.expert_cos_max
            ));
        }
        if l.expert_norm_max > 0.0 && l.expert_norm_min / l.expert_norm_max < 0.6 {
            flags.push(format!(
                "layer {}: expert L2 spread {:.2}-{:.2}; the small ones may be starved by the router",
                l.layer, l.expert_norm_min, l.expert_norm_max
            ));
        }
        for (e, mp) in &l.starved {
            flags.push(format!(
                "layer {} expert {}: still near random init (mean s1/MP {:.2}); likely starved by the router",
                l.layer, e, mp
            ));
        }
        for (e, r1) in &l.collapsed {
            flags.push(format!(
                "layer {} expert {}: collapsed (mean rank-1 share {:.1}%)",
                l.layer, e, r1
            ));
        }
        if l.router_col_min > 0.0 && l.router_col_max / l.router_col_min > 3.0 {
            flags.push(format!(
                "layer {}: router columns unbalanced ({:.3}-{:.3})",
                l.layer, l.router_col_min, l.router_col_max
            ));
        }
    }
    flags
}

// ---------------------------------------------------------------------------
// Embedding rows
// ---------------------------------------------------------------------------

fn embedding_report(tensors: &[NamedTensor]) {
    let Some(t) = tensors.iter().find(|t| Category::of(&t.name) == Category::Embedding && t.dims.len() == 2)
    else {
        return;
    };
    let (rows, cols) = (t.dims[0], t.dims[1]);
    let mut norms: Vec<f64> = (0..rows)
        .map(|r| t.data[r * cols..(r + 1) * cols].iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt())
        .collect();
    // Cosine of each row with the mean row: how much every token shares one direction.
    let mut mean_row = vec![0.0f64; cols];
    for r in 0..rows {
        for (m, &x) in mean_row.iter_mut().zip(&t.data[r * cols..(r + 1) * cols]) {
            *m += x as f64 / rows as f64;
        }
    }
    let mean_norm = mean_row.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut cos_mean: Vec<f64> = (0..rows)
        .map(|r| {
            let row = &t.data[r * cols..(r + 1) * cols];
            let d: f64 = row.iter().zip(&mean_row).map(|(&x, m)| x as f64 * m).sum();
            d / (norms[r] * mean_norm).max(1e-30)
        })
        .collect();
    norms.sort_by(|a, b| a.total_cmp(b));
    cos_mean.sort_by(|a, b| a.total_cmp(b));
    let pct = |v: &[f64], p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    println!("\n=== EMBEDDING ROWS ({}) ===", t.name);
    println!(
        "  row L2:            p1 {:.3}  p10 {:.3}  p50 {:.3}  p90 {:.3}  p99 {:.3}  max {:.3}",
        pct(&norms, 0.01), pct(&norms, 0.10), pct(&norms, 0.5), pct(&norms, 0.9), pct(&norms, 0.99), norms[rows - 1]
    );
    println!(
        "  cos(row, mean):    p10 {:.3}  p50 {:.3}  p90 {:.3}   (near 0 = isotropic; near 1 = every token points the same way)",
        pct(&cos_mean, 0.10), pct(&cos_mean, 0.5), pct(&cos_mean, 0.9)
    );
    println!("  mean-row L2:       {:.3}", mean_norm);
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Per group: (count, stable rank, s1/MP, rank-1 % or mean gain for norms, rms).
fn group_means(stats: &[Stats]) -> BTreeMap<Category, (usize, f64, f64, f64, f64)> {
    let mut groups: BTreeMap<Category, Vec<&Stats>> = BTreeMap::new();
    for s in stats {
        groups.entry(s.category).or_default().push(s);
    }
    let mean = |v: &[&Stats], f: fn(&Stats) -> f64| {
        if v.is_empty() { f64::NAN } else { v.iter().map(|s| f(s)).sum::<f64>() / v.len() as f64 }
    };
    groups
        .into_iter()
        .map(|(cat, ss)| {
            let mats: Vec<&Stats> = ss.iter().copied().filter(|s| s.s1 > 0.0).collect();
            let row = if cat == Category::Norm {
                (ss.len(), f64::NAN, f64::NAN, mean(&ss, |s| s.gain_mean), mean(&ss, |s| s.rms))
            } else {
                (
                    ss.len(),
                    mean(&mats, |s| s.stable_rank),
                    mean(&mats, |s| s.mp_ratio),
                    mean(&mats, |s| s.rank1_pct),
                    mean(&ss, |s| s.rms),
                )
            };
            (cat, row)
        })
        .collect()
}

fn compare_report(a: &[NamedTensor], a_stats: &[Stats], b: &[NamedTensor], top: usize) {
    let b_stats: Vec<Stats> = b.par_iter().map(|t| compute_stats(&t.name, &t.dims, &t.data)).collect();
    let (ga, gb) = (group_means(a_stats), group_means(&b_stats));
    println!("\n=== COMPARISON BY GROUP (A = analysed checkpoint, B = --compare) ===");
    println!("  matrices: mean stable rank, s1/MP, rank-1 %, rms; norm gains: the r1% column holds the mean gain");
    println!(
        "  {:<15} | {:>7} {:>7} | {:>6} {:>6} | {:>7} {:>7} | {:>8} {:>8}",
        "group", "srank A", "srank B", "MP A", "MP B", "r1% A", "r1% B", "rms A", "rms B"
    );
    let f = |x: f64, p: usize| if x.is_nan() { "-".to_string() } else { format!("{x:.p$}") };
    for (cat, ra) in &ga {
        let rb = gb.get(cat).copied().unwrap_or((0, f64::NAN, f64::NAN, f64::NAN, f64::NAN));
        println!(
            "  {:<15} | {:>7} {:>7} | {:>6} {:>6} | {:>7} {:>7} | {:>8} {:>8}",
            cat.label(), f(ra.1, 1), f(rb.1, 1), f(ra.2, 2), f(rb.2, 2), f(ra.3, 1), f(rb.3, 1), f(ra.4, 4), f(rb.4, 4)
        );
    }
    let count = |ss: &[Stats], st: Status| ss.iter().filter(|s| s.status == st).count();
    for st in [Status::Collapsed, Status::RandomLike, Status::GainUntouched, Status::NonFinite] {
        let (na, nb) = (count(a_stats, st), count(&b_stats, st));
        if na + nb > 0 {
            println!("  {:<15} A {:>4}  B {:>4}", st.label(), na, nb);
        }
    }

    let b_map: HashMap<&str, &NamedTensor> = b.iter().map(|t| (t.name.as_str(), t)).collect();
    struct Diff {
        name: String,
        category: Category,
        rel: f64,
        cos: f64,
        norm_ratio: f64,
    }
    let mut missing = Vec::new();
    let mut diffs: Vec<Diff> = a
        .par_iter()
        .filter_map(|ta| {
            let tb = b_map.get(ta.name.as_str())?;
            if ta.dims != tb.dims {
                return None;
            }
            let (mut dd, mut aa, mut bb, mut ab) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (&x, &y) in ta.data.iter().zip(&tb.data) {
                let (x, y) = (x as f64, y as f64);
                dd += (x - y).powi(2);
                aa += x * x;
                bb += y * y;
                ab += x * y;
            }
            Some(Diff {
                name: ta.name.clone(),
                category: Category::of(&ta.name),
                rel: dd.sqrt() / bb.sqrt().max(1e-30),
                cos: ab / (aa.sqrt() * bb.sqrt()).max(1e-30),
                norm_ratio: aa.sqrt() / bb.sqrt().max(1e-30),
            })
        })
        .collect();
    for ta in a {
        match b_map.get(ta.name.as_str()) {
            None => missing.push(format!("{} (only in A)", ta.name)),
            Some(tb) if tb.dims != ta.dims => {
                missing.push(format!("{} (A {} vs B {})", ta.name, dims_str(&ta.dims), dims_str(&tb.dims)))
            }
            _ => {}
        }
    }
    let a_names: std::collections::HashSet<&str> = a.iter().map(|t| t.name.as_str()).collect();
    for tb in b {
        if !a_names.contains(tb.name.as_str()) {
            missing.push(format!("{} (only in B)", tb.name));
        }
    }

    // Elementwise drift only means something when both runs started from one init.
    let mat_cos: Vec<f64> = diffs
        .iter()
        .filter(|d| matches!(d.category, Category::Attention | Category::Expert))
        .map(|d| d.cos)
        .collect();
    let mean_cos = mat_cos.iter().sum::<f64>() / mat_cos.len().max(1) as f64;
    if mean_cos < 0.2 {
        println!(
            "\n  Elementwise drift skipped: attention/expert weights have mean cosine {:.3} between A and B,\n  so the runs do not share an init (or have moved far from it). Use the group table above.",
            mean_cos
        );
        if !missing.is_empty() {
            println!("  Not compared ({}): {}", missing.len(), missing.iter().take(top).cloned().collect::<Vec<_>>().join("; "));
        }
        return;
    }
    println!("\n=== ELEMENTWISE DRIFT (shared init; mean matrix cosine {:.3}) ===", mean_cos);
    println!("  rel = ||A-B|| / ||B||, cos = cosine(A, B), L2 A/B = ||A|| / ||B||");
    println!("  {:<15} {:>6} {:>9} {:>9} {:>9}", "group", "count", "rel", "cos", "L2 A/B");
    let mut groups: BTreeMap<Category, Vec<&Diff>> = BTreeMap::new();
    for d in &diffs {
        groups.entry(d.category).or_default().push(d);
    }
    for (cat, ds) in &groups {
        let n = ds.len() as f64;
        println!(
            "  {:<15} {:>6} {:>9.4} {:>9.4} {:>9.3}",
            cat.label(),
            ds.len(),
            ds.iter().map(|d| d.rel).sum::<f64>() / n,
            ds.iter().map(|d| d.cos).sum::<f64>() / n,
            ds.iter().map(|d| d.norm_ratio).sum::<f64>() / n
        );
    }
    diffs.sort_by(|x, y| y.rel.total_cmp(&x.rel));
    println!("  Largest relative differences:");
    for d in diffs.iter().take(top) {
        println!("    {:<44} rel {:8.4}  cos {:7.4}  L2 A/B {:6.3}", d.name, d.rel, d.cos, d.norm_ratio);
    }
    if !missing.is_empty() {
        println!("  Not compared ({}):", missing.len());
        for m in missing.iter().take(top) {
            println!("    {m}");
        }
    }
}

// ---------------------------------------------------------------------------
// Named report
// ---------------------------------------------------------------------------

fn named_report(tensors: &[NamedTensor], args: &Args) -> Result<Vec<Stats>> {
    let total: usize = tensors.iter().map(|t| t.data.len()).sum();
    println!("\n=== PARAMETERS ===");
    println!("  {} tensors, {} floats ({:.1} MiB)", tensors.len(), total, total as f64 * 4.0 / 1048576.0);

    let stats: Vec<Stats> = tensors
        .par_iter()
        .map(|t| compute_stats(&t.name, &t.dims, &t.data))
        .collect();

    // Category x status counts.
    println!("\n=== HEALTH BY GROUP ===");
    println!(
        "  matrices: COLLAPSED = rank-1 share > max({COLLAPSE_RANK1_PCT}%, {COLLAPSE_OVER_RANDOM}x a random matrix of that shape), RANDOM-LIKE = s1 < {RANDOM_MP_RATIO}x the MP edge for its RMS"
    );
    println!("  norm gains start at exactly 1.0, so UNTOUCHED means no update ever reached them");
    let mut groups: BTreeMap<Category, Vec<&Stats>> = BTreeMap::new();
    for s in &stats {
        groups.entry(s.category).or_default().push(s);
    }
    for (cat, ss) in &groups {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for s in ss {
            *counts.entry(s.status.label()).or_default() += 1;
        }
        let mats: Vec<&&Stats> = ss.iter().filter(|s| s.s1 > 0.0).collect();
        let extra = if mats.is_empty() {
            String::new()
        } else {
            let n = mats.len() as f64;
            format!(
                " | mean srank {:.1}, s1/MP {:.2}, rank1 {:.1}%, rms {:.4}",
                mats.iter().map(|s| s.stable_rank).sum::<f64>() / n,
                mats.iter().map(|s| s.mp_ratio).sum::<f64>() / n,
                mats.iter().map(|s| s.rank1_pct).sum::<f64>() / n,
                mats.iter().map(|s| s.rms).sum::<f64>() / n
            )
        };
        let counts = counts.iter().map(|(k, v)| format!("{v} {k}")).collect::<Vec<_>>().join(", ");
        println!("  {:<15} {:>4} tensors: {}{}", cat.label(), ss.len(), counts, extra);
    }

    // Issues first, so a broken checkpoint is obvious at a glance.
    let mut issues: Vec<String> = Vec::new();
    for s in &stats {
        match s.status {
            Status::NonFinite => issues.push(format!("{}: {} non-finite values", s.name, s.nonfinite)),
            Status::Zero => issues.push(format!("{}: all zeros", s.name)),
            Status::GainUntouched => issues.push(format!("{}: still exactly 1.0 (never updated)", s.name)),
            Status::GainDrifted => issues.push(format!("{}: mean gain {:.3}", s.name, s.gain_mean)),
            _ => {}
        }
    }
    let untouched = stats.iter().filter(|s| s.status == Status::GainUntouched).count();
    let gains = stats.iter().filter(|s| s.category == Category::Norm).count();
    if untouched > 0 && untouched == gains {
        issues.insert(0, format!(
            "ALL {gains} norm gains are exactly 1.0: the optimizer is not updating 1-D parameters (or the checkpoint is the init)"
        ));
    }

    for s in stats.iter().filter(|s| s.category == Category::Embedding && s.s1 > 0.0) {
        if s.mean_share > 0.5 {
            issues.push(format!(
                "{}: {:.0}% of its energy is the mean row (every token shares one direction; s1/s2 {:.1})",
                s.name, s.mean_share * 100.0, s.s1 / s.s2.max(1e-30)
            ));
        }
    }

    let layers = layer_summaries(tensors, &stats);
    issues.extend(layer_flags(&layers));

    println!("\n=== ISSUES ===");
    if issues.is_empty() {
        println!("  none (no non-finite, all-zero or untouched tensors; no expert/router imbalance)");
    } else {
        for i in issues.iter().take(args.top.max(40)) {
            println!("  [!] {i}");
        }
        if issues.len() > args.top.max(40) {
            println!("  ... {} more", issues.len() - args.top.max(40));
        }
    }

    let mut mats: Vec<&Stats> = stats.iter().filter(|s| s.s1 > 0.0).collect();
    mats.sort_by(|a, b| b.rank1_pct.total_cmp(&a.rank1_pct));
    println!("\n=== MOST ANISOTROPIC MATRICES (highest rank-1 share) ===");
    for s in mats.iter().take(args.top) {
        print_stats_row(s);
    }
    mats.sort_by(|a, b| a.mp_ratio.total_cmp(&b.mp_ratio));
    println!("\n=== CLOSEST TO RANDOM INIT (lowest s1/MP) ===");
    for s in mats.iter().take(args.top) {
        print_stats_row(s);
    }

    print_layer_table(&layers);
    embedding_report(tensors);

    if args.all {
        println!("\n=== ALL TENSORS ===");
        for s in &stats {
            print_stats_row(s);
        }
    }

    if let Some(path) = &args.csv {
        let mut f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        writeln!(f, "name,category,dims,status,rms,mean,max_abs,zero_frac,nonfinite,s1,s2,stable_rank,rank1_pct,s1_over_mp,mean_share,gain_mean,gain_dev")?;
        for s in &stats {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                s.name, s.category.label(), dims_str(&s.dims), s.status.label(), s.rms, s.mean, s.max_abs,
                s.zero_frac, s.nonfinite, s.s1, s.s2, s.stable_rank, s.rank1_pct, s.mp_ratio, s.mean_share,
                s.gain_mean, s.gain_dev
            )?;
        }
        println!("\n  wrote {}", path.display());
    }
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Raw byte scanner (fallback)
// ---------------------------------------------------------------------------

fn is_plausible_weight(val: f32) -> bool {
    val.is_finite() && val.abs() < 50.0
}

/// Inspects the first bytes of the file for magic headers, strings, or bincode struct fields
fn inspect_file_header(bytes: &[u8]) {
    println!("=== 1. FILE HEADER FORENSICS (First 128 bytes) ===");
    let preview_len = bytes.len().min(128);
    print!("  ASCII preview: \"");
    for &b in &bytes[0..preview_len] {
        if (32..=126).contains(&b) {
            print!("{}", b as char);
        } else {
            print!(".");
        }
    }
    println!("\"");
    print!("  First u64s (little-endian): ");
    for offset in (0..32.min(bytes.len().saturating_sub(8))).step_by(8) {
        let v = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        print!("[+{}]={}  ", offset, v);
    }
    println!();
}

/// Checks which byte alignment (0, 1, 2, or 3) contains the valid f32 weights.
fn scan_byte_alignments(bytes: &[u8]) {
    println!("\n=== 2. BYTE-ALIGNMENT DETECTOR ===");
    let mut best_align = 0;
    let mut best_valid_count = 0;
    for align in 0..4 {
        let slice = &bytes[align.min(bytes.len())..];
        let total_chunks = (slice.len() / 4).max(1);
        let (mut valid, mut zero, mut bad) = (0usize, 0usize, 0usize);
        for chunk in slice.chunks_exact(4) {
            let v = f32::from_le_bytes(chunk.try_into().unwrap());
            if v == 0.0 {
                zero += 1;
            } else if is_plausible_weight(v) && v.abs() > 1e-7 {
                valid += 1;
            } else {
                bad += 1;
            }
        }
        println!(
            "  Alignment offset +{}: {:8} valid weights ({:5.1}%), {:7} zeros, {:8} corrupt/header floats",
            align, valid, (valid + zero) as f64 * 100.0 / total_chunks as f64, zero, bad
        );
        if valid > best_valid_count {
            best_valid_count = valid;
            best_align = align;
        }
    }
    println!("  --> Dominant f32 byte alignment is +{}", best_align);
}

struct ContiguousTensor {
    byte_offset: usize,
    floats: Vec<f32>,
}

/// Contiguous runs of plausible f32 values, checked at every byte offset.
fn extract_all_valid_tensors(bytes: &[u8]) -> Vec<ContiguousTensor> {
    let mut tensors = Vec::new();
    let min_tensor_floats = 64;
    let mut pos = 0;
    while pos + 4 <= bytes.len() {
        let f0 = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        if is_plausible_weight(f0) && (f0 == 0.0 || f0.abs() > 1e-6) {
            let start = pos;
            let mut run = Vec::new();
            while pos + 4 <= bytes.len() {
                let v = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                if v.is_finite() && v.abs() < 30.0 && (v == 0.0 || v.abs() > 1e-9) {
                    run.push(v);
                    pos += 4;
                } else {
                    break;
                }
            }
            if run.len() >= min_tensor_floats {
                tensors.push(ContiguousTensor { byte_offset: start, floats: run });
                continue;
            }
        }
        pos += 1;
    }
    tensors
}

fn guess_shape(len: usize, config: &ModelConfig) -> String {
    let (v, e, f) = (config.vocab_size, config.embed_dim, config.ff_dim);
    if len == v * e {
        return format!("{v}x{e} (Embedding/LM_Head)");
    }
    if len == e * e {
        return format!("{e}x{e} (Attn Proj)");
    }
    if len == 3 * e * e {
        return format!("{}x{e} (Fused QKV)", 3 * e);
    }
    if len == e * f {
        return format!("{e}x{f} (Expert FFN)");
    }
    if len == config.num_experts * e {
        return format!("{}x{e} (MoE Router)", config.num_experts);
    }
    if len == e {
        return format!("{e} (Norm/Bias)");
    }
    if len == f {
        return format!("{f} (FFN Bias)");
    }
    if len % e == 0 {
        return format!("{}x{e}", len / e);
    }
    format!("{len} floats")
}

fn raw_report(model_bin: &Path, config: &ModelConfig, args: &Args) -> Result<()> {
    let mut raw = Vec::new();
    BufReader::new(File::open(model_bin)?).read_to_end(&mut raw)?;
    if raw.first() == Some(&b'{') {
        bail!("{} looks like JSON, not a checkpoint", model_bin.display());
    }
    inspect_file_header(&raw);
    scan_byte_alignments(&raw);

    println!("\n=== 3. RAW TENSOR EXTRACTION & SPECTRAL HEALTH ===");
    println!("  (raw mode: blocks are runs of plausible floats, not exact tensors; names are guesses from shape)");
    let tensors = extract_all_valid_tensors(&raw);
    let total: usize = tensors.iter().map(|t| t.floats.len()).sum();
    println!(
        "  Discovered {} contiguous weight blocks totaling {} floats ({:.1}% of file bytes)",
        tensors.len(), total, (total * 4) as f64 * 100.0 / raw.len() as f64
    );
    let e = config.embed_dim;
    let rows: Vec<(usize, &ContiguousTensor, Stats)> = tensors
        .par_iter()
        .enumerate()
        .filter(|(_, t)| t.floats.len() >= e * 4 && t.floats.len() % e == 0)
        .map(|(i, t)| {
            let dims = vec![t.floats.len() / e, e];
            (i, t, compute_stats(&guess_shape(t.floats.len(), config), &dims, &t.floats))
        })
        .collect();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, _, s) in &rows {
        *counts.entry(s.status.label()).or_default() += 1;
    }
    println!(
        "  {} matrix-shaped blocks: {}",
        rows.len(),
        counts.iter().map(|(k, v)| format!("{v} {k}")).collect::<Vec<_>>().join(", ")
    );
    for (i, t, s) in rows.iter().filter(|(_, _, s)| args.all || s.status.flagged()) {
        print!("  #{:03} byte {:9} ", i, t.byte_offset);
        print_stats_row(s);
    }
    if !args.all {
        println!("  (only flagged blocks shown; --all prints every block)");
    }
    Ok(())
}

// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = parse_args()?;
    println!("==================================================================");
    println!(" PASS 1: MODEL WEIGHT FORENSICS");
    let target = resolve_target(args.target.as_deref(), args.run.as_deref())?;
    let config = load_model_config(&target, &args.overrides);
    println!("   Model File:         {}", target.model_bin.display());
    println!(
        "   Shape:              vocab {} | embed {} | ff {} | {} routed experts{} | {} layers | {} heads",
        config.vocab_size, config.embed_dim, config.ff_dim, config.num_experts,
        if config.num_shared_experts > 0 { format!(" (+{} shared)", config.num_shared_experts) } else { String::new() },
        config.n_layers, config.attn_heads
    );
    print_run_context(&target);
    println!("==================================================================");

    if args.raw {
        return raw_report(&target.model_bin, &config, &args);
    }
    let tensors = match load_named(&target) {
        Ok(t) => t,
        Err(e) => {
            println!("\n  [!] named load failed ({e:#}); falling back to the raw byte scanner\n");
            return raw_report(&target.model_bin, &config, &args);
        }
    };
    let stats = named_report(&tensors, &args)?;

    if let Some(other) = &args.compare {
        println!("\n==================================================================");
        let b = resolve_target(Some(other), args.compare_run.as_deref())?;
        println!(" COMPARE WITH: {}", b.model_bin.display());
        print_run_context(&b);
        let b_tensors = load_named(&b).context("loading --compare checkpoint")?;
        compare_report(&tensors, &stats, &b_tensors, args.top);
    }
    Ok(())
}
