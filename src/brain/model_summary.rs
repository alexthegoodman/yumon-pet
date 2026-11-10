//! Parameter totals and a module tree for any Burn model, printed at the start
//! of a training run. Repeated layers (Vec entries) are folded into one row
//! with their count.

use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{Tensor, backend::Backend};
use std::collections::{HashMap, HashSet};

/// Named levels shown below the model root.
const DEPTH: usize = 2;

#[derive(Default)]
struct Row {
    label: String,
    level: usize,
    params: usize,
    instances: HashSet<String>,
    shape: Option<Vec<usize>>,
}

#[derive(Default)]
struct Collector {
    /// (name, is a Vec/Array index) per entered module.
    stack: Vec<(String, bool)>,
    rows: Vec<Row>,
    index: HashMap<String, usize>,
    total: usize,
    frozen: usize,
    tensors: usize,
}

impl<B: Backend> ModuleVisitor<B> for Collector {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.stack.push((name.to_string(), matches!(container_type, "Vec" | "Array")));
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) { self.stack.pop(); }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let shape = param.lazy_shape().dims.to_vec();
        let n: usize = shape.iter().product();
        self.total += n;
        self.tensors += 1;
        if !param.val().is_require_grad() { self.frozen += n; }

        let mut label = String::new();
        let mut concrete = String::new();
        let mut level = 0;
        for (i, (name, is_index)) in self.stack.iter().enumerate() {
            concrete.push('.');
            concrete.push_str(name);
            if *is_index {
                label.push_str("[]");
            } else {
                if level == DEPTH { break; }
                if !label.is_empty() { label.push('.'); }
                label.push_str(name);
                level += 1;
            }
            // Fold index levels that directly follow into this row.
            if self.stack.get(i + 1).is_some_and(|(_, idx)| *idx) { continue; }
            let leaf = i + 1 == self.stack.len();
            let key = label.clone();
            let row = match self.index.get(&key) {
                Some(&r) => r,
                None => {
                    self.index.insert(key.clone(), self.rows.len());
                    self.rows.push(Row { label: key, level, ..Default::default() });
                    self.rows.len() - 1
                }
            };
            let row = &mut self.rows[row];
            row.params += n;
            row.instances.insert(concrete.clone());
            if leaf { row.shape = Some(shape.clone()); }
        }
    }
}

fn count(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 { out.push(','); }
        out.push(c);
    }
    out
}

/// Prints total, trainable and frozen parameters, fp32 size and a per-module
/// breakdown (two levels deep, repeated layers folded as `name[] x N`).
pub fn print_model_summary<B: Backend, M: Module<B>>(model: &M) {
    println!("\n{}", model_summary::<B, M>(model));
}

pub fn model_summary<B: Backend, M: Module<B>>(model: &M) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let mut c = Collector::default();
    model.visit(&mut c);
    let name = std::any::type_name::<M>().split('<').next().unwrap_or("").rsplit("::").next().unwrap_or("");
    writeln!(out, "Model summary: {name}").unwrap();
    writeln!(out, "  Parameters: {} ({:.2}M) in {} tensors, {:.1} MB as f32", count(c.total), c.total as f64 / 1e6,
        c.tensors, c.total as f64 * 4.0 / (1024.0 * 1024.0)).unwrap();
    if c.frozen > 0 && c.frozen < c.total {
        writeln!(out, "  Trainable: {}, frozen: {}", count(c.total - c.frozen), count(c.frozen)).unwrap();
    }
    let shown = |r: &Row| format!("{}{}", "  ".repeat(r.level - 1), r.label.rsplit('.').next().unwrap_or(&r.label));
    let width = c.rows.iter().map(|r| shown(r).len()).max().unwrap_or(0) + 2;
    for row in &c.rows {
        let label = shown(row);
        let copies = row.instances.len();
        let share = if c.total > 0 { 100.0 * row.params as f64 / c.total as f64 } else { 0.0 };
        let detail = match (&row.shape, copies) {
            (Some(shape), 1) => format!("{shape:?}"),
            (Some(shape), n) => format!("{n} x {shape:?}"),
            (None, n) if row.label.contains("[]") && n > 1 => format!("{n} x {}", count(row.params / n)),
            _ => String::new(),
        };
        writeln!(out, "  {label:<width$} {:>14} {share:>6.2}%  {detail}", count(row.params)).unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::decoder_model::YumonDecBrainConfig;
    use crate::brain::samples::TrainingStage;

    #[test]
    fn summary_matches_num_params() {
        type B = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
        let config = YumonDecBrainConfig::new(64, TrainingStage::Structured).with_embed_dim(16)
            .with_n_layers(3).with_attn_heads(2).with_ff_dim(32).with_max_seq_len(16);
        let model = config.init::<B>(&Default::default());
        let summary = model_summary::<B, _>(&model);
        println!("{summary}");
        assert!(summary.contains(&format!("Parameters: {} ", count(model.num_params()))));
        assert!(summary.contains("3 x "));
    }
}
