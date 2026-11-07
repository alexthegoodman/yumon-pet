//! Coded vectors: `{word word ...} / q`.
//!
//! A codebook maps each word to `dim` numbers; a coded vector is the words'
//! entries laid end to end, divided by `q`. The fixed codebooks in `codebooks/`
//! (built by `scripts/make_codebook.py`) hold k-means centroids of standard normal
//! samples, so they suit any weight row scaled to unit RMS, and each word is one
//! token for the Qwen2.5 tokenizer: `g4` (4,096 words x 4 numbers, 3 bits per
//! weight, four weights per token) and `g2` (256 x 2, 4 bits, two per token).
//!
//! `encode` turns the vectors of a program into coded ones. With calibration
//! inputs it rounds like GPTQ: unit directions column group by column group, each
//! group's rounding error pushed onto the columns not yet coded through the inverse
//! input covariance, then the readouts the same way over the coded units' outputs.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::model::{dot, Program};
use crate::num::{format_num, scale_text, NumFormat};

pub struct Codebook {
    pub name: &'static str,
    pub dim: usize,
    pub words: Vec<&'static str>,
    entries: Vec<f64>,
    lookup: HashMap<&'static str, usize>,
}

impl Codebook {
    fn load(name: &'static str, text: &'static str) -> Self {
        let mut words = Vec::new();
        let mut entries = Vec::new();
        let mut dim = 0;
        for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let mut it = line.split_whitespace();
            words.push(it.next().unwrap());
            let row: Vec<f64> = it.map(|x| x.parse().unwrap()).collect();
            dim = row.len();
            entries.extend(row);
        }
        let lookup = words.iter().enumerate().map(|(i, w)| (*w, i)).collect();
        Self { name, dim, words, entries, lookup }
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    pub fn entry(&self, i: usize) -> &[f64] {
        &self.entries[i * self.dim..(i + 1) * self.dim]
    }

    pub fn index(&self, word: &str) -> Option<usize> {
        self.lookup.get(word).copied()
    }

    /// The entry closest to `v` (Euclidean) on the first `v.len()` numbers, so a
    /// short final group matches on the entries it uses.
    pub fn nearest(&self, v: &[f64]) -> usize {
        let mut best = (f64::INFINITY, 0);
        for i in 0..self.len() {
            let d: f64 = self.entry(i).iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
            if d < best.0 {
                best = (d, i);
            }
        }
        best.1
    }

    /// The vector `{words} / q` denotes, computed exactly as the parser does.
    pub fn expand(&self, idx: &[usize], q: f64) -> Vec<f64> {
        idx.iter().flat_map(|&i| self.entry(i).iter().map(move |x| x / q)).collect()
    }
}

pub const NAMES: &[&str] = &["g4", "g2"];

pub fn codebook(name: &str) -> Option<&'static Codebook> {
    static G4: OnceLock<Codebook> = OnceLock::new();
    static G2: OnceLock<Codebook> = OnceLock::new();
    match name {
        "g4" => Some(G4.get_or_init(|| Codebook::load("g4", include_str!("../codebooks/g4.txt")))),
        "g2" => Some(G2.get_or_init(|| Codebook::load("g2", include_str!("../codebooks/g2.txt")))),
        _ => None,
    }
}

/// The coded vectors of a program, keyed by their exact values, so the printer can
/// write them back as words.
#[derive(Clone)]
pub struct Codes {
    pub book: &'static Codebook,
    /// Exact values -> (q, word indices).
    pub vecs: HashMap<Vec<u64>, (f64, Vec<usize>)>,
}

impl Codes {
    pub fn new(book: &'static Codebook) -> Self {
        Self { book, vecs: HashMap::new() }
    }

    pub fn insert(&mut self, values: &[f64], q: f64, idx: Vec<usize>) {
        self.vecs.insert(key(values), (q, idx));
    }

    /// `v` cut to `len` entries if it is a coded vector whose last word overhangs
    /// that length (fewer than `dim` extra entries); the cut vector stays coded.
    pub fn truncate(&mut self, v: &[f64], len: usize) -> Option<Vec<f64>> {
        if v.len() <= len || v.len() - len >= self.book.dim {
            return None;
        }
        let entry = self.vecs.get(&key(v))?.clone();
        let cut = v[..len].to_vec();
        self.vecs.insert(key(&cut), entry);
        Some(cut)
    }

    /// `{words} / q` if `v` is one of the coded vectors.
    pub fn text(&self, v: &[f64]) -> Option<String> {
        let (q, idx) = self.vecs.get(&key(v))?;
        let words: Vec<&str> = idx.iter().map(|&i| self.book.words[i]).collect();
        let words = format!("{{{}}}", words.join(" "));
        Some(if *q == 1.0 { words } else { format!("{words} / {}", q_text(*q)) })
    }
}

/// Two-significant-digit form (`31`, `4.1e4`) when it reads back exactly, else exact.
pub fn q_text(q: f64) -> String {
    let (m, e) = two_digits(q);
    let t = scale_text(m, e);
    if t.parse::<f64>() == Ok(q) { t } else { format_num(q, NumFormat::EXACT) }
}

fn key(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

impl PartialEq for Codes {
    fn eq(&self, other: &Self) -> bool {
        self.book.name == other.book.name && self.vecs == other.vecs
    }
}

impl std::fmt::Debug for Codes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Codes({}, {} vectors)", self.book.name, self.vecs.len())
    }
}

/// Coded copy of `p`: every unit direction (and gate up factor) and every readout.
/// A vector whose length is not a multiple of the codebook dimension ends in a word
/// that overhangs it (the extra entries are dropped). Biases stay numbers.
/// `samples` (inputs the program will see) make the rounding error-aware.
pub fn encode(p: &Program, book: &'static Codebook, samples: Option<&[Vec<f64>]>) -> Program {
    let mut out = p.clone();
    let mut codes = Codes::new(book);

    let ux = samples.and_then(|xs| inverse_cholesky(&gram(xs.iter().map(|x| x.as_slice()), p.dim), p.dim));
    for u in &mut out.units {
        u.w = code_row(&u.w, book, ux.as_deref(), &mut codes);
        if let Some(up) = &mut u.up {
            up.u = code_row(&up.u, book, ux.as_deref(), &mut codes);
        }
    }

    let n = out.units.len();
    if n > 0 {
        let uh = samples.and_then(|xs| {
            let hs: Vec<Vec<f64>> = xs.iter().map(|x| out.units.iter().map(|u| u.eval(x)).collect()).collect();
            inverse_cholesky(&gram(hs.iter().map(|h| h.as_slice()), n), n)
        });
        for o in 0..out.outputs.len() {
            let row: Vec<f64> = out.units.iter().map(|u| u.coeffs[o]).collect();
            let new = code_row(&row, book, uh.as_deref(), &mut codes);
            for (u, c) in out.units.iter_mut().zip(new) {
                u.coeffs[o] = c;
            }
        }
    }
    out.codes = Some(codes);
    out
}

/// sum of x x^T.
fn gram<'a>(xs: impl Iterator<Item = &'a [f64]>, n: usize) -> Vec<f64> {
    let mut h = vec![0.0; n * n];
    for x in xs {
        for i in 0..n {
            if x[i] == 0.0 {
                continue;
            }
            for j in 0..n {
                h[i * n + j] += x[i] * x[j];
            }
        }
    }
    h
}

/// `v` as words. The scale q (two significant digits, as printed) maps it to unit
/// RMS. With `u`, the upper Cholesky factor of the inverse Gram matrix of the
/// inputs, each group's rounding error is pushed onto the entries after it (GPTQ).
fn code_row(v: &[f64], book: &Codebook, u: Option<&[f64]>, codes: &mut Codes) -> Vec<f64> {
    let n = v.len();
    let k = book.dim;
    let rms = (dot(v, v) / n as f64).sqrt();
    if rms == 0.0 || !rms.is_finite() {
        return v.to_vec();
    }
    let (m, e) = two_digits(1.0 / rms);
    // The value the lexer will read back, so expanded entries match bit for bit.
    let q: f64 = scale_text(m, e).parse().expect("scale text is a number");
    let mut w = v.to_vec();
    let mut idx = Vec::with_capacity(n.div_ceil(k));
    for g in (0..n).step_by(k) {
        let end = (g + k).min(n);
        let scaled: Vec<f64> = w[g..end].iter().map(|x| x * q).collect();
        let i = book.nearest(&scaled);
        idx.push(i);
        if let Some(u) = u {
            for j in g..end {
                let err = (w[j] - book.entry(i)[j - g] / q) / u[j * n + j];
                for t in j + 1..n {
                    w[t] -= err * u[j * n + t];
                }
            }
        }
    }
    let full = book.expand(&idx, q);
    codes.insert(&full, q, idx);
    codes.truncate(&full, n).unwrap_or(full)
}

/// m * 10^e with m in 10..=99 closest to x.
fn two_digits(x: f64) -> (i64, i32) {
    let mut e = x.log10().floor() as i32 - 1;
    let mut m = (x / 10f64.powi(e)).round() as i64;
    if m >= 100 {
        m = (m as f64 / 10.0).round() as i64;
        e += 1;
    }
    (m, e)
}

/// U with U^T U = (H + damp I)^-1, row-major n x n; None if H is not usable.
fn inverse_cholesky(h: &[f64], n: usize) -> Option<Vec<f64>> {
    let mean_diag = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    if !(mean_diag > 0.0) {
        return None;
    }
    let mut a = h.to_vec();
    for i in 0..n {
        a[i * n + i] += 0.01 * mean_diag;
    }
    let l = cholesky(&a, n)?;
    // inverse of L (lower), then Hinv = L^-T L^-1.
    let mut li = vec![0.0; n * n];
    for i in 0..n {
        li[i * n + i] = 1.0 / l[i * n + i];
        for j in 0..i {
            let s: f64 = (j..i).map(|t| l[i * n + t] * li[t * n + j]).sum();
            li[i * n + j] = -s / l[i * n + i];
        }
    }
    let mut hinv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (i..n).map(|t| li[t * n + i] * li[t * n + j]).sum();
            hinv[i * n + j] = s;
            hinv[j * n + i] = s;
        }
    }
    // Upper factor: Hinv = L L^T = U^T U with U = L^T.
    let l = cholesky(&hinv, n)?;
    let mut u = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            u[i * n + j] = l[j * n + i];
        }
    }
    Some(u)
}

fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = a[i * n + j] - (0..j).map(|t| l[i * n + t] * l[j * n + t]).sum::<f64>();
            if i == j {
                if !(s > 0.0) {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebooks_load() {
        for name in NAMES {
            let b = codebook(name).unwrap();
            assert!(b.len() >= 256 && b.dim >= 2);
            assert_eq!(b.index(b.words[7]), Some(7));
            assert_eq!(b.nearest(b.entry(42)), 42);
        }
    }

    #[test]
    fn inverse_cholesky_factors_the_inverse() {
        let n = 3;
        let h = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let u = inverse_cholesky(&h, n).unwrap();
        // (H + damp I) * U^T U = I
        let damp = 0.01 * 3.0;
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for t in 0..n {
                    let hit = h[i * n + t] + if i == t { damp } else { 0.0 };
                    let utu: f64 = (0..n).map(|r| u[r * n + t] * u[r * n + j]).sum();
                    s += hit * utu;
                }
                assert!((s - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12, "{i},{j}: {s}");
            }
            for j in 0..i {
                assert_eq!(u[i * n + j], 0.0, "U is upper triangular");
            }
        }
    }
}
