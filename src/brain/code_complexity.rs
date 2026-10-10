//! Versioned, syntax-only Rust complexity. Counting rules: docs/code-complexity.md.
use anyhow::{Result, ensure};
use proc_macro2::{TokenStream, TokenTree};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use syn::visit::{self, Visit};

pub const ANALYZER_VERSION: &str = "rust-syntax-1.0.0";
pub const DEFAULT_WEIGHTS: [f64; 5] = [0.25, 0.30, 0.20, 0.15, 0.10];
pub const FEATURES: [&str; 9] = [
    "cyclomatic_complexity",
    "cognitive_complexity",
    "control_flow_depth",
    "state_mutation_complexity",
    "halstead_volume",
    "token_count",
    "ast_node_count",
    "expression_depth",
    "dependency_count",
];

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Metrics {
    pub cyclomatic_complexity: Option<f64>,
    pub cognitive_complexity: Option<f64>,
    pub control_flow_depth: Option<f64>,
    pub state_mutation_complexity: Option<f64>,
    pub halstead_volume: Option<f64>,
    pub token_count: Option<f64>,
    pub ast_node_count: Option<f64>,
    pub expression_depth: Option<f64>,
    pub dependency_count: Option<f64>,
}
impl Metrics {
    pub fn values(&self) -> [Option<f64>; 9] {
        [
            self.cyclomatic_complexity,
            self.cognitive_complexity,
            self.control_flow_depth,
            self.state_mutation_complexity,
            self.halstead_volume,
            self.token_count,
            self.ast_node_count,
            self.expression_depth,
            self.dependency_count,
        ]
    }
    pub fn primary(&self) -> Option<[f64; 5]> {
        Some([
            self.cyclomatic_complexity?,
            self.cognitive_complexity?,
            self.control_flow_depth?,
            self.state_mutation_complexity?,
            self.halstead_volume?,
        ])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub sample_id: String,
    pub language: String,
    pub analysis_unit: String,
    pub metrics: Metrics,
    pub structural_complexity_score: Option<f64>,
    pub complexity_tier: Option<u8>,
    pub scoring_version: Option<String>,
    pub reference_population_version: Option<String>,
    pub reference_sha256: Option<String>,
    pub analysis_status: String,
    pub analysis_errors: Vec<String>,
    pub parser_version: String,
    pub analyzer_version: String,
    pub source_sha256: String,
    pub source_path: String,
    pub source_location: String,
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Default)]
struct Counter {
    cc: usize,
    cognitive: usize,
    depth: usize,
    max_depth: usize,
    mutations: usize,
    expr_depth: usize,
    max_expr: usize,
    nodes: usize,
    functions: usize,
    opaque: bool,
    calls: usize,
}
impl<'ast> Visit<'ast> for Counter {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.functions += 1;
        self.cc += 1;
        visit::visit_item_fn(self, node);
    }
    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        self.functions += 1;
        self.cc += 1;
        visit::visit_impl_item_fn(self, node);
    }
    fn visit_trait_item_fn(&mut self, node: &'ast syn::TraitItemFn) {
        if node.default.is_some() {
            self.functions += 1;
            self.cc += 1;
        }
        visit::visit_trait_item_fn(self, node);
    }
    fn visit_expr(&mut self, node: &'ast syn::Expr) {
        self.nodes += 1;
        self.expr_depth += 1;
        self.max_expr = self.max_expr.max(self.expr_depth);
        let control = matches!(
            node,
            syn::Expr::If(_)
                | syn::Expr::ForLoop(_)
                | syn::Expr::While(_)
                | syn::Expr::Loop(_)
                | syn::Expr::Match(_)
        );
        if control {
            self.cognitive += 1 + self.depth;
            self.depth += 1;
            self.max_depth = self.max_depth.max(self.depth);
        }
        match node {
            syn::Expr::If(n) => {
                self.cc += 1;
                if n.else_branch.is_some() {
                    self.cognitive += 1;
                }
            }
            syn::Expr::While(_) | syn::Expr::ForLoop(_) => self.cc += 1,
            syn::Expr::Match(n) => {
                self.cc += n.arms.len().saturating_sub(1);
            }
            syn::Expr::Binary(n) => {
                if matches!(n.op, syn::BinOp::And(_) | syn::BinOp::Or(_)) {
                    self.cc += 1;
                    self.cognitive += 1;
                }
                if matches!(
                    n.op,
                    syn::BinOp::AddAssign(_)
                        | syn::BinOp::SubAssign(_)
                        | syn::BinOp::MulAssign(_)
                        | syn::BinOp::DivAssign(_)
                        | syn::BinOp::RemAssign(_)
                        | syn::BinOp::BitXorAssign(_)
                        | syn::BinOp::BitAndAssign(_)
                        | syn::BinOp::BitOrAssign(_)
                        | syn::BinOp::ShlAssign(_)
                        | syn::BinOp::ShrAssign(_)
                ) {
                    self.mutations += 1;
                }
            }
            syn::Expr::Assign(_) => self.mutations += 1,
            syn::Expr::Try(_) => {
                self.cc += 1;
                self.cognitive += 1;
            }
            syn::Expr::Break(_) | syn::Expr::Continue(_) | syn::Expr::Return(_) => {
                self.cognitive += 1
            }
            syn::Expr::Verbatim(_) => self.opaque = true,
            _ => {}
        }
        visit::visit_expr(self, node);
        if control {
            self.depth -= 1;
        }
        self.expr_depth -= 1;
    }
    fn visit_expr_call(&mut self, n: &'ast syn::ExprCall) {
        // Count call sites, without claiming name resolution or external dependencies.
        self.calls += 1;
        visit::visit_expr_call(self, n);
    }
    fn visit_expr_method_call(&mut self, n: &'ast syn::ExprMethodCall) {
        self.calls += 1;
        visit::visit_expr_method_call(self, n);
    }
    fn visit_local(&mut self, n: &'ast syn::Local) {
        self.nodes += 1;
        if n.init.is_some() {
            self.mutations += 1;
        }
        visit::visit_local(self, n);
    }
    fn visit_item(&mut self, n: &'ast syn::Item) {
        self.nodes += 1;
        visit::visit_item(self, n);
    }
    fn visit_pat(&mut self, n: &'ast syn::Pat) {
        self.nodes += 1;
        visit::visit_pat(self, n);
    }
    fn visit_type(&mut self, n: &'ast syn::Type) {
        self.nodes += 1;
        visit::visit_type(self, n);
    }
    fn visit_macro(&mut self, _: &'ast syn::Macro) {
        self.opaque = true;
    }
}

fn lexical(stream: TokenStream, operators: &mut Vec<String>, operands: &mut Vec<String>) {
    const KEYWORDS: &[&str] = &[
        "fn", "pub", "let", "mut", "if", "else", "while", "for", "in", "loop", "match", "return",
        "break", "continue", "impl", "trait", "struct", "enum", "use", "as", "async", "await",
        "move", "unsafe", "const", "static", "where", "ref", "dyn",
    ];
    for token in stream {
        match token {
            TokenTree::Group(g) => {
                if g.delimiter() != proc_macro2::Delimiter::None {
                    operators.push(format!("open:{:?}", g.delimiter()));
                    operators.push(format!("close:{:?}", g.delimiter()));
                }
                lexical(g.stream(), operators, operands);
            }
            TokenTree::Punct(p) => operators.push(p.as_char().to_string()),
            TokenTree::Ident(i) if KEYWORDS.contains(&i.to_string().as_str()) => {
                operators.push(i.to_string())
            }
            other => operands.push(other.to_string()),
        }
    }
}

/// One cached unit; method wrappers are parsed as a file. No recovery or inference.
pub fn analyze(id: &str, language: &str, source: &str, path: &str, location: &str) -> Record {
    let mut record = Record {
        sample_id: id.into(),
        language: language.into(),
        analysis_unit: "function".into(),
        metrics: Metrics::default(),
        structural_complexity_score: None,
        complexity_tier: None,
        scoring_version: None,
        reference_population_version: None,
        reference_sha256: None,
        analysis_status: "success".into(),
        analysis_errors: vec![],
        parser_version: "syn-2.0.117/proc-macro2-1.0.106".into(),
        analyzer_version: ANALYZER_VERSION.into(),
        source_sha256: digest(source.as_bytes()),
        source_path: path.into(),
        source_location: location.into(),
    };
    if language != "rust" {
        record.analysis_status = "unsupported".into();
        return record;
    }
    let parsed = match syn::parse_file(source) {
        Ok(parsed) => parsed,
        Err(error) => {
            record.analysis_status = "parse_error".into();
            record.analysis_errors.push(error.to_string());
            return record;
        }
    };
    let mut counter = Counter::default();
    counter.visit_file(&parsed);
    record.analysis_unit = if counter.functions == 1 {
        "function"
    } else {
        "item"
    }
    .into();
    let mut operators = vec![];
    let mut operands = vec![];
    // File parser already validated the source. Remove a possible shebang for tokenization.
    let source = source.strip_prefix('\u{feff}').unwrap_or(source);
    let source = parsed
        .shebang
        .as_ref()
        .map_or(source, |s| &source[s.len()..]);
    let stream = source.parse::<TokenStream>();
    let stream = match stream {
        Ok(s) => s,
        Err(e) => {
            record.analysis_status = "parse_error".into();
            record.analysis_errors.push(e.to_string());
            return record;
        }
    };
    lexical(stream, &mut operators, &mut operands);
    let n = operators.len() + operands.len();
    let vocabulary = operators.iter().collect::<BTreeSet<_>>().len()
        + operands.iter().collect::<BTreeSet<_>>().len();
    record.metrics = Metrics {
        cyclomatic_complexity: Some(counter.cc as f64),
        cognitive_complexity: Some(counter.cognitive as f64),
        control_flow_depth: Some(counter.max_depth as f64),
        state_mutation_complexity: Some(counter.mutations as f64),
        halstead_volume: Some(if vocabulary > 0 {
            n as f64 * (vocabulary as f64).log2()
        } else {
            0.0
        }),
        token_count: Some(n as f64),
        ast_node_count: Some(counter.nodes as f64),
        expression_depth: Some(counter.max_expr as f64),
        dependency_count: Some(counter.calls as f64),
    };
    if counter.opaque {
        record.analysis_status = "missing_metrics".into();
        record.analysis_errors.push(
            "Unexpanded macro/verbatim syntax: control flow, mutation and calls unavailable".into(),
        );
        record.metrics.cyclomatic_complexity = None;
        record.metrics.cognitive_complexity = None;
        record.metrics.control_flow_depth = None;
        record.metrics.state_mutation_complexity = None;
        record.metrics.dependency_count = None;
    }
    record
}

/// Immutable calibration artifact. Sorted empirical metric populations and score quartiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub scoring_version: String,
    pub reference_population_version: String,
    pub analyzer_version: String,
    pub parser_version: String,
    pub language: String,
    pub source_cache_sha256: String,
    pub weights: [f64; 5],
    pub missing_metric_policy: String,
    pub preprocessing: Preprocessing,
    pub populations: [Vec<f64>; 5],
    pub boundaries: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Preprocessing {
    pub excluded_path_components: Vec<String>,
    pub generated_markers: Vec<String>,
    pub max_line_length: Option<usize>,
}
impl Default for Preprocessing {
    fn default() -> Self {
        Self {
            excluded_path_components: vec!["vendor".into(), "vendored".into(), "generated".into()],
            generated_markers: vec!["@generated".into(), "automatically generated".into()],
            max_line_length: Some(2000),
        }
    }
}
impl Preprocessing {
    pub fn excluded(&self, source: &str, path: &str) -> bool {
        path.split(['/', '\\'])
            .any(|p| self.excluded_path_components.iter().any(|x| x == p))
            || self.generated_markers.iter().any(|m| source.contains(m))
            || self
                .max_line_length
                .is_some_and(|max| source.lines().any(|l| l.chars().count() > max))
    }
}

/// Midrank: ties get the midpoint of their occupied reference ranks. Outside values clamp.
pub fn percentile(sorted: &[f64], value: f64) -> f64 {
    let lower = sorted.partition_point(|&x| x < value);
    let upper = sorted.partition_point(|&x| x <= value);
    (lower + upper) as f64 / (2.0 * sorted.len() as f64)
}
pub fn tier(score: f64, boundaries: &[f64; 3]) -> u8 {
    1 + boundaries
        .iter()
        .filter(|&&boundary| score > boundary)
        .count() as u8
}
fn validate_weights(weights: &[f64; 5]) -> Result<()> {
    ensure!(
        weights.iter().all(|w| w.is_finite() && *w >= 0.0)
            && (weights.iter().sum::<f64>() - 1.0).abs() < 1e-10,
        "weights must be finite, nonnegative and sum to 1"
    );
    Ok(())
}
impl Reference {
    pub fn fit(
        records: &[Record],
        version: &str,
        cache_digest: String,
        weights: [f64; 5],
        preprocessing: Preprocessing,
    ) -> Result<Self> {
        validate_weights(&weights)?;
        ensure!(
            !version.trim().is_empty(),
            "calibration version must not be empty"
        );
        let values: Vec<_> = records
            .iter()
            .filter(|r| r.analysis_status == "success")
            .filter_map(|r| r.metrics.primary())
            .collect();
        ensure!(
            !values.is_empty(),
            "reference population contains no complete samples"
        );
        let populations = std::array::from_fn(|i| {
            let mut v: Vec<_> = values.iter().map(|row| row[i]).collect();
            v.sort_by(f64::total_cmp);
            v
        });
        let mut reference = Self {
            scoring_version: version.into(),
            reference_population_version: version.into(),
            analyzer_version: ANALYZER_VERSION.into(),
            parser_version: "syn-2.0.117/proc-macro2-1.0.106".into(),
            language: "rust".into(),
            source_cache_sha256: cache_digest,
            weights,
            missing_metric_policy: "reject-incomplete".into(),
            preprocessing,
            populations,
            boundaries: [0.0; 3],
        };
        let mut scores: Vec<_> = values.iter().map(|v| reference.score(v)).collect();
        scores.sort_by(f64::total_cmp);
        reference.boundaries =
            [0.25, 0.5, 0.75].map(|q| scores[(q * scores.len() as f64).ceil() as usize - 1]);
        reference.validate()?;
        Ok(reference)
    }
    pub fn validate(&self) -> Result<()> {
        validate_weights(&self.weights)?;
        ensure!(
            self.analyzer_version == ANALYZER_VERSION
                && self.parser_version == "syn-2.0.117/proc-macro2-1.0.106"
                && self.language == "rust"
                && self.missing_metric_policy == "reject-incomplete",
            "incompatible reference analyzer, language, parser or missing-metric policy"
        );
        ensure!(
            !self.scoring_version.trim().is_empty()
                && !self.reference_population_version.trim().is_empty()
                && self.source_cache_sha256.len() == 64,
            "invalid reference provenance"
        );
        let n = self.populations[0].len();
        ensure!(
            n > 0
                && self.populations.iter().all(|p| p.len() == n
                    && p.iter().all(|v| v.is_finite() && *v >= 0.0)
                    && p.windows(2).all(|w| w[0] <= w[1])),
            "invalid empirical populations"
        );
        ensure!(
            self.boundaries
                .iter()
                .all(|b| b.is_finite() && (0.0..=1.0).contains(b))
                && self.boundaries.windows(2).all(|w| w[0] <= w[1]),
            "invalid tier boundaries"
        );
        Ok(())
    }
    pub fn score(&self, values: &[f64; 5]) -> f64 {
        (0..5)
            .map(|i| self.weights[i] * percentile(&self.populations[i], values[i]))
            .sum()
    }
    pub fn classify(&self, record: &mut Record, reference_digest: &str) {
        record.structural_complexity_score = None;
        record.complexity_tier = None;
        record.scoring_version = Some(self.scoring_version.clone());
        record.reference_population_version = Some(self.reference_population_version.clone());
        record.reference_sha256 = Some(reference_digest.into());
        if record.analysis_status == "success" && record.language == self.language {
            if let Some(values) = record.metrics.primary() {
                let score = self.score(&values);
                record.structural_complexity_score = Some(score);
                record.complexity_tier = Some(tier(score, &self.boundaries));
            }
        }
    }
}

pub fn correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }
    let mx = x.iter().sum::<f64>() / x.len() as f64;
    let my = y.iter().sum::<f64>() / y.len() as f64;
    let covariance: f64 = x.iter().zip(y).map(|(x, y)| (x - mx) * (y - my)).sum();
    let variance = x.iter().map(|x| (x - mx).powi(2)).sum::<f64>()
        * y.iter().map(|y| (y - my).powi(2)).sum::<f64>();
    (variance > 0.0).then(|| covariance / variance.sqrt())
}

#[cfg(test)]
#[path = "../../tests/code_complexity/mod.rs"]
mod tests;
