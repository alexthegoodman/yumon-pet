#![allow(dead_code)]

use wir::{Mat, Weights};

/// xorshift64*: deterministic, dependency-free test randomness.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

/// A random one-hidden-layer network with some deliberately degenerate units:
/// constant units (w = 0), dead readouts (c = 0), positively scaled duplicates
/// (which must merge) and opposite pairs (which must stay distinct).
pub fn random_network(rng: &mut Rng, d: usize, h: usize, o: usize) -> Weights {
    let mut rows: Vec<(Vec<f64>, f64, Vec<f64>)> = Vec::new();
    for _ in 0..h {
        let roll = rng.unit();
        let unit = if roll < 0.05 {
            (vec![0.0; d], rng.uniform(-2.0, 2.0), (0..o).map(|_| rng.uniform(-2.0, 2.0)).collect())
        } else if roll < 0.10 {
            ((0..d).map(|_| rng.uniform(-2.0, 2.0)).collect(), rng.uniform(-2.0, 2.0), vec![0.0; o])
        } else if roll < 0.20 && !rows.is_empty() {
            let (w, b, _) = rows[rng.below(rows.len())].clone();
            let a = rng.uniform(0.1, 10.0);
            (w.iter().map(|x| a * x).collect(), a * b, (0..o).map(|_| rng.uniform(-2.0, 2.0)).collect())
        } else if roll < 0.25 && !rows.is_empty() {
            let (w, b, _) = rows[rng.below(rows.len())].clone();
            (w.iter().map(|x| -x).collect(), -b, (0..o).map(|_| rng.uniform(-2.0, 2.0)).collect())
        } else {
            (
                (0..d).map(|_| rng.uniform(-2.0, 2.0)).collect(),
                rng.uniform(-2.0, 2.0),
                (0..o).map(|_| rng.uniform(-2.0, 2.0)).collect(),
            )
        };
        rows.push(unit);
    }
    let w1 = Mat::from_rows(&rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>()).unwrap();
    let w1 = if h == 0 { Mat::zeros(0, d) } else { w1 };
    let mut w2 = Mat::zeros(o, h);
    for (i, r) in rows.iter().enumerate() {
        for k in 0..o {
            w2.set(k, i, r.2[k]);
        }
    }
    Weights {
        w1,
        b1: rows.iter().map(|r| r.1).collect(),
        w2,
        b2: (0..o).map(|_| rng.uniform(-1.0, 1.0)).collect(),
        input: None,
        outputs: None,
    }
}

pub fn random_point(rng: &mut Rng, d: usize, r: f64) -> Vec<f64> {
    (0..d).map(|_| rng.uniform(-r, r)).collect()
}

/// Largest |a(x) - b(x)| over `n` points in [-r, r]^D.
pub fn max_diff(rng: &mut Rng, a: &Weights, b: &Weights, n: usize, r: f64) -> f64 {
    let d = a.input_dim();
    let mut worst = 0.0f64;
    for _ in 0..n {
        let x = random_point(rng, d, r);
        for (ya, yb) in a.eval(&x).iter().zip(b.eval(&x)) {
            worst = worst.max((ya - yb).abs());
        }
    }
    worst
}

/// Scale every hidden unit by a random positive factor and permute the units.
/// The function is unchanged.
pub fn scale_and_permute(rng: &mut Rng, w: &Weights) -> Weights {
    let h = w.hidden();
    let mut order: Vec<usize> = (0..h).collect();
    rng.shuffle(&mut order);
    let mut out = w.clone();
    for (new, &old) in order.iter().enumerate() {
        let a = rng.uniform(0.05, 20.0);
        for (dst, src) in out.w1.row_mut(new).iter_mut().zip(w.w1.row(old)) {
            *dst = a * src;
        }
        out.b1[new] = a * w.b1[old];
        for k in 0..w.output_dim() {
            out.w2.set(k, new, w.w2.get(k, old) / a);
        }
    }
    out
}
