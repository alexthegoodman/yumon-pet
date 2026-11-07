//! Number formatting for the printer.
//!
//! Canonical text uses 6 significant digits after snapping values that sit
//! within 1e-9 (relative) of a simple rational p/q, q <= 16. Terminating
//! rationals print as decimals (0.5, 0.0625), the rest as `p/q`, which the
//! lexer reads back exactly. `Exact` prints the shortest string that parses
//! back to the same f64, for lossless weight round-trips.

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
}

impl NumFormat {
    /// The v0 canonical display: 6 significant digits with rational snapping.
    pub const CANONICAL: NumFormat =
        NumFormat { precision: Precision::Significant(6), snap_rationals: true };
    /// Lossless: every printed number parses back to the identical f64.
    pub const EXACT: NumFormat = NumFormat { precision: Precision::Exact, snap_rationals: false };
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
    fn exact_formatting_round_trips() {
        for x in [std::f64::consts::PI, -1e-300, 0.1 + 0.2, 1e20, 123.456] {
            let s = format_num(x, NumFormat::EXACT);
            assert_eq!(s.parse::<f64>().unwrap(), x, "{s}");
        }
    }
}
