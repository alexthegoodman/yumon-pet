//! Minimum-norm least squares via one-sided Jacobi SVD, matching the
//! semantics of `numpy.linalg.lstsq` (default rcond = eps * max(m, n)).
//! `solve` systems are tiny (points x creases), so clarity beats speed.

use crate::model::Mat;

pub struct LstsqResult {
    pub x: Vec<f64>,
    pub rank: usize,
    pub singular_values: Vec<f64>,
}

pub fn lstsq(a: &Mat, b: &[f64]) -> LstsqResult {
    let (m, n) = (a.rows, a.cols);
    assert_eq!(b.len(), m);
    // Columns of U are rotated until mutually orthogonal: A V = U, |U_j| = sigma_j.
    let mut u: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a.get(i, j)).collect()).collect();
    let mut v: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    for _sweep in 0..100 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let alpha: f64 = u[p].iter().map(|x| x * x).sum();
                let beta: f64 = u[q].iter().map(|x| x * x).sum();
                let gamma: f64 = u[p].iter().zip(&u[q]).map(|(x, y)| x * y).sum();
                if gamma == 0.0 || gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                rotate(&mut u, p, q, c, s);
                rotate(&mut v, p, q, c, s);
            }
        }
        if !rotated {
            break;
        }
    }
    let sigma: Vec<f64> = u.iter().map(|col| col.iter().map(|x| x * x).sum::<f64>().sqrt()).collect();
    let smax = sigma.iter().cloned().fold(0.0, f64::max);
    let cutoff = f64::EPSILON * m.max(n) as f64 * smax;
    let mut x = vec![0.0; n];
    let mut rank = 0;
    for j in 0..n {
        if sigma[j] > cutoff && sigma[j] > 0.0 {
            rank += 1;
            let coef = u[j].iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / (sigma[j] * sigma[j]);
            for (xi, vij) in x.iter_mut().zip(&v[j]) {
                *xi += coef * vij;
            }
        }
    }
    let mut singular_values = sigma;
    singular_values.sort_by(|a, b| b.total_cmp(a));
    LstsqResult { x, rank, singular_values }
}

fn rotate(cols: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64) {
    let (lo, hi) = cols.split_at_mut(q);
    for (xp, xq) in lo[p].iter_mut().zip(hi[0].iter_mut()) {
        let (a, b) = (*xp, *xq);
        *xp = c * a - s * b;
        *xq = s * a + c * b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overdetermined_exact() {
        // y = 2 + 3t sampled at 4 points.
        let a = Mat::from_rows(&[vec![1.0, 0.0], vec![1.0, 1.0], vec![1.0, 2.0], vec![1.0, 3.0]]).unwrap();
        let r = lstsq(&a, &[2.0, 5.0, 8.0, 11.0]);
        assert_eq!(r.rank, 2);
        assert!((r.x[0] - 2.0).abs() < 1e-12 && (r.x[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rank_deficient_min_norm() {
        // Two identical columns: min-norm splits the weight evenly.
        let a = Mat::from_rows(&[vec![1.0, 1.0], vec![2.0, 2.0]]).unwrap();
        let r = lstsq(&a, &[2.0, 4.0]);
        assert_eq!(r.rank, 1);
        assert!((r.x[0] - 1.0).abs() < 1e-12 && (r.x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn underdetermined_min_norm() {
        let a = Mat::from_rows(&[vec![1.0, 2.0, 2.0]]).unwrap();
        let r = lstsq(&a, &[9.0]);
        assert_eq!(r.rank, 1);
        for (x, e) in r.x.iter().zip([1.0, 2.0, 2.0]) {
            assert!((x - e).abs() < 1e-12);
        }
    }
}
