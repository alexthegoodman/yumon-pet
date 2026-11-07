//! Number formatting for the printer.
//!
//! Canonical text uses 6 significant digits after snapping values that sit
//! within 1e-9 (relative) of a simple rational p/q, q <= 16. Terminating
//! rationals print as decimals (0.5, 0.0625), the rest as `p/q`, which the
//! lexer reads back exactly. `Exact` prints the shortest string that parses
//! back to the same f64, for lossless weight round-trips.
//!
//! Vectors (`[...]` literals, used by the compact high-dimensional syntax) print as
//! integers over one scale with two significant digits, `[-881 -415 107] / 990` or
//! `[...] / 4.1e4`. The scale is the largest such number that keeps every integer
//! within `vector_digits` digits, so the largest entry lands near 10^d (900..999 for
//! d = 3): d digits of resolution relative to the vector's largest entry. Integers
//! of up to three digits are one token for common LLM tokenizers, against ~8 for
//! `-0.881469`.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    /// Round to this many significant digits.
    Significant(u32),
    /// Shortest representation that round-trips the f64 exactly.
    Exact,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumFormat {
    pub precision: Precision,
    pub snap_rationals: bool,
    /// Digits kept by vector literals in rounded modes (largest |entry| -> 10^(d-1)..10^d).
    pub vector_digits: u32,
}

impl NumFormat {
    /// The v0 canonical display: 6 significant digits with rational snapping.
    pub const CANONICAL: NumFormat =
        NumFormat { precision: Precision::Significant(6), snap_rationals: true, vector_digits: 3 };
    /// Lossless: every printed number parses back to the identical f64.
    pub const EXACT: NumFormat = NumFormat { precision: Precision::Exact, snap_rationals: false, vector_digits: 17 };
}

impl Default for NumFormat {
    fn default() -> Self {
        Self::CANONICAL
    }
}

pub const MAX_DENOMINATOR: i64 = 16;
/// Only values up to this magnitude snap (larger ones would print more digits than the display).
pub const MAX_SNAP: f64 = 1e6;
/// Relative distance within which a value snaps to a rational.
pub const RATIONAL_TOL: f64 = 1e-9;
/// Values below this print as 0 in rounded (non-exact) modes.
pub const ZERO_SNAP: f64 = 1e-12;

/// Smallest-denominator rational p/q (q <= MAX_DENOMINATOR) within
/// RATIONAL_TOL of `x`, relative to max(1, |x|).
pub fn snap_rational(x: f64) -> Option<(i64, i64)> {
    if !x.is_finite() || x.abs() > MAX_SNAP {
        return None;
    }
    let tol = RATIONAL_TOL * x.abs().max(1.0);
    for q in 1..=MAX_DENOMINATOR {
        let p = (x * q as f64).round();
        if (x - p / q as f64).abs() <= tol {
            return Some((p as i64, q));
        }
    }
    None
}

fn terminating(mut q: i64) -> bool {
    for f in [2, 5] {
        while q % f == 0 {
            q /= f;
        }
    }
    q == 1
}

pub fn format_num(x: f64, fmt: NumFormat) -> String {
    if x == 0.0 {
        return "0".into();
    }
    if !x.is_finite() {
        return format!("{x}");
    }
    if fmt.snap_rationals
        && let Some((p, q)) = snap_rational(x) {
            if p == 0 {
                return "0".into();
            }
            if q == 1 {
                return format!("{p}");
            }
            if terminating(q) {
                return format_exact(p as f64 / q as f64);
            }
            return format!("{p}/{q}");
        }
    match fmt.precision {
        Precision::Exact => format_exact(x),
        Precision::Significant(n) => format_significant(x, n.max(1)),
    }
}

/// A vector literal. Exact mode lists exact numbers; rounded modes print integers
/// over a power of ten chosen so the largest |entry| becomes 10^(d-1)..10^d.
pub fn format_vector(v: &[f64], fmt: NumFormat) -> String {
    let list = |items: Vec<String>| format!("[{}]", items.join(" "));
    let big = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if fmt.precision == Precision::Exact || big == 0.0 || !big.is_finite() {
        return list(v.iter().map(|x| if *x == 0.0 { "0".into() } else { format_exact(*x) }).collect());
    }
    let (m, e) = vector_scale(big, fmt.vector_digits);
    let q = scale_value(m, e);
    let ints = list(v.iter().map(|x| format!("{}", (x * q).round() as i64)).collect());
    if q == 1.0 { ints } else { format!("{ints} / {}", scale_text(m, e)) }
}

/// Scale m * 10^e (m in 10..=99): the largest with big * scale <= 10^d - 0.5.
pub fn vector_scale(big: f64, digits: u32) -> (i64, i32) {
    let limit = 10f64.powi(digits as i32) - 0.5;
    let target = limit / big;
    let mut e = target.log10().floor() as i32 - 1;
    let mut m = (target / 10f64.powi(e)).floor() as i64;
    // Float noise at the edges of a decade.
    while m >= 100 {
        m /= 10;
        e += 1;
    }
    while m < 10 {
        m = m * 10 + 9;
        e -= 1;
    }
    while m > 10 && big * scale_value(m, e) > limit {
        m -= 1;
    }
    (m, e)
}

/// m * 10^e as an f64 (division for negative exponents keeps it exact-ish).
pub fn scale_value(m: i64, e: i32) -> f64 {
    if e >= 0 { m as f64 * 10f64.powi(e) } else { m as f64 / 10f64.powi(-e) }
}

/// `990`, `4.1e4`, `0.12`: integers up to 4 digits plainly, otherwise mantissa form.
fn scale_text(m: i64, e: i32) -> String {
    match e {
        0..=2 => format!("{}", m * 10i64.pow(e as u32)),
        -2..=-1 => format_exact(scale_value(m, e)),
        _ if m % 10 == 0 => format!("{}e{}", m / 10, e + 1),
        _ => format!("{}.{}e{}", m / 10, m % 10, e + 1),
    }
}

fn format_exact(x: f64) -> String {
    let a = x.abs();
    if (1e-5..1e15).contains(&a) { format!("{x}") } else { format!("{x:e}") }
}

fn format_significant(x: f64, digits: u32) -> String {
    if x.abs() < ZERO_SNAP {
        return "0".into();
    }
    // Round via scientific notation so the exponent reflects carries (9.9999996 -> 1e1).
    let sci = format!("{:.*e}", (digits - 1) as usize, x);
    let (mantissa, exp) = sci.split_once('e').expect("scientific format");
    let exp: i32 = exp.parse().expect("exponent");
    if (-5..15).contains(&exp) {
        let rounded: f64 = sci.parse().expect("rounded value");
        let decimals = (digits as i32 - 1 - exp).max(0) as usize;
        trim_zeros(&format!("{rounded:.decimals$}"))
    } else {
        format!("{}e{exp}", trim_zeros(mantissa))
    }
}

fn trim_zeros(s: &str) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0').trim_end_matches('.');
        if t == "-0" { "0".into() } else { t.into() }
    } else {
        s.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_formatting() {
        let c = NumFormat::CANONICAL;
        assert_eq!(format_num(0.5, c), "0.5");
        assert_eq!(format_num(-2.0, c), "-2");
        assert_eq!(format_num(-0.0, c), "0");
        assert_eq!(format_num(1.0 / 3.0, c), "1/3");
        assert_eq!(format_num(-7.0 / 12.0, c), "-7/12");
        assert_eq!(format_num(0.0625, c), "0.0625");
        assert_eq!(format_num(std::f64::consts::SQRT_2, c), "1.41421");
        assert_eq!(format_num(9.9999996, c), "10");
        assert_eq!(format_num(123456789.0, c), "123457000"); // beyond MAX_SNAP: no snapping
        assert_eq!(format_num(250000.0, c), "250000");
        assert_eq!(format_num(123456789.4, c), "123457000");
        assert_eq!(format_num(1.234567e-7, c), "1.23457e-7");
        assert_eq!(format_num(1e-13, c), "0");
    }

    #[test]
    fn vector_formatting() {
        let c = NumFormat::CANONICAL;
        assert_eq!(format_vector(&[1.0, -0.5, 0.1234], c), "[990 -495 122] / 990");
        assert_eq!(format_vector(&[0.024, -0.0001], c), "[984 -4] / 4.1e4");
        assert_eq!(format_vector(&[0.0047, -0.0001], c), "[987 -21] / 2.1e5");
        assert_eq!(format_vector(&[250.0, 3.0], c), "[975 12] / 3.9");
        assert_eq!(format_vector(&[0.0, 0.0], c), "[0 0]");
        assert_eq!(format_vector(&[0.5, 0.0], NumFormat::EXACT), "[0.5 0]");
    }

    #[test]
    fn exact_formatting_round_trips() {
        for x in [std::f64::consts::PI, -1e-300, 0.1 + 0.2, 1e20, 123.456] {
            let s = format_num(x, NumFormat::EXACT);
            assert_eq!(s.parse::<f64>().unwrap(), x, "{s}");
        }
    }
}
