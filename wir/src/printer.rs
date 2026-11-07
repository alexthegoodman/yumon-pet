//! `Program` -> WIR text.
//!
//! 1D single-output units print as inline hinges, 2D creases as
//! `zero on line P -> Q`, and other dimensions as
//! `zero on plane normal m offset d`. See `crease_decl` for how the points are
//! chosen so that every printed number survives a parse unchanged.
//!
//! `print` also re-reads its own output and re-renders until the text is a
//! fixed point, so `print(parse(print(p))) == print(p)` holds by construction
//! even in corner cases the point choice does not cover. In practice the
//! first render is already the fixed point; the round count is capped.

use crate::ast::PlaneSpec;
use crate::compiler::{lower_str, plane_equation};
use crate::canon::max_abs;
use crate::model::{dot, Program, Unit, UnitKind};
use crate::num::{format_num, NumFormat, Precision};

const MAX_ROUNDS: usize = 8;
const LINE_WIDTH: usize = 80;

pub fn print(p: &Program, fmt: NumFormat) -> String {
    let mut text = render(p, fmt);
    for _ in 0..MAX_ROUNDS {
        let Ok(reread) = lower_str(&text) else { break };
        let next = render(&reread, fmt);
        if next == text {
            break;
        }
        text = next;
    }
    text
}

/// Render once, without the fixed-point pass.
pub fn render(p: &Program, fmt: NumFormat) -> String {
    let num = |x: f64| format_num(x, fmt);
    let point = |v: &[f64]| format!("({})", v.iter().map(|x| num(*x)).collect::<Vec<_>>().join(", "));
    let inline = |u: &Unit| u.kind == UnitKind::Hinge && p.dim == 1 && p.outputs.len() == 1;

    let mut out = String::from("wir 0.1\n");
    out += &format!("input  {}: R{}\n", p.input, p.dim);
    for o in &p.outputs {
        out += &format!("output {}: R1\n", o.name);
    }

    let decls: Vec<&Unit> = p.units.iter().filter(|u| !inline(u)).collect();
    if !decls.is_empty() {
        out.push('\n');
    }
    for u in decls {
        out += &crease_decl(u, p.dim, fmt);
        out.push('\n');
    }

    out.push('\n');
    for (k, o) in p.outputs.iter().enumerate() {
        // (separator, body) pairs; the separator of the first term is folded in.
        let mut parts: Vec<(&str, String)> = Vec::new();
        for u in &p.units {
            let c = u.coeffs[k];
            if inline(u) {
                let s = u.slope();
                let d = u.offset();
                let turn = num(c * s);
                let body = if u.w[0] > 0.0 {
                    format!("hinge at {} turn {turn}", num(d))
                } else {
                    format!("hinge left at {} turn {turn}", num(-d))
                };
                parts.push(("+", body));
            } else {
                let mag = num(c.abs());
                if mag == "0" {
                    continue;
                }
                let body = if mag == "1" { u.name.clone() } else { format!("{mag}*{}", u.name) };
                parts.push((if c < 0.0 { "-" } else { "+" }, body));
            }
        }
        let bias = num(o.bias.abs());
        if bias != "0" {
            parts.push((if o.bias < 0.0 { "-" } else { "+" }, bias));
        }
        let head = format!("{} = ", o.name);
        let first = |(sep, body): &(&str, String)| if *sep == "-" { format!("-{body}") } else { body.clone() };
        let line = match parts.split_first() {
            None => format!("{head}0"),
            Some((f, rest)) => {
                let flat = rest.iter().fold(head.clone() + &first(f), |acc, (s, b)| format!("{acc} {s} {b}"));
                if flat.len() <= LINE_WIDTH {
                    flat
                } else {
                    let indent = " ".repeat(head.len().saturating_sub(2));
                    rest.iter().fold(head.clone() + &first(f), |acc, (s, b)| format!("{acc}\n{indent}{s} {b}"))
                }
            }
        };
        out += &line;
        out.push('\n');
    }

    if !p.asserts.is_empty() {
        out.push('\n');
    }
    for a in &p.asserts {
        out += &format!("assert {}{} == {}\n", p.outputs[a.output].name, point(&a.point), num(a.value));
    }
    out
}

/// Format a number and read the printed text back, so later quantities are
/// computed from exactly what the parser will see.
fn printed(x: f64, fmt: NumFormat) -> (String, f64) {
    let s = format_num(x, fmt);
    let v = match s.split_once('/') {
        Some((p, q)) => p.parse::<f64>().unwrap() / q.parse::<f64>().unwrap(),
        None => s.parse().unwrap(),
    };
    (s, v)
}

/// One `crease` declaration whose numbers survive a parse unchanged:
///
/// - the normal m = w / max|w_i| has a component that is exactly +-1 (axis k),
///   so renormalization in the compiler cannot disturb it;
/// - the plane is pinned by the point P where it crosses axis k, so
///   re-deriving P from the parsed plane gives the same coordinate (a foot of
///   the perpendicular would move by a rounding error every pass);
/// - the anchor is P stepped one unit along axis k toward the rising side, so
///   its value is exactly max|w_i|, which is 1 in canonical form.
///
/// 2D uses `zero on line P -> Q` with Q = P + (-m2, m1); other dimensions use
/// `zero on plane normal m through P`. Sums of printed numbers (Q and the
/// anchor) use 12 significant digits so they stay exact decimals of the
/// 6-digit inputs; the anchor value is measured against the printed plane.
fn crease_decl(u: &Unit, dim: usize, fmt: NumFormat) -> String {
    let derived = match fmt.precision {
        Precision::Exact => fmt,
        Precision::Significant(_) => NumFormat { precision: Precision::Significant(12), ..fmt },
    };
    let pt = |txt: &[String]| format!("({})", txt.join(", "));
    let big = max_abs(&u.w);
    let (m_txt, m): (Vec<String>, Vec<f64>) = u.w.iter().map(|x| printed(x / big, fmt)).unzip();
    let k = m.iter().position(|x| x.abs() == 1.0).unwrap_or_else(|| {
        (0..dim).max_by(|&a, &b| m[a].abs().total_cmp(&m[b].abs())).unwrap()
    });

    let mut p = vec![0.0; dim];
    let mut p_txt = vec!["0".to_string(); dim];
    (p_txt[k], p[k]) = printed(-u.b / u.w[k], fmt);
    let (plane, plane_txt) = if dim == 2 {
        let (q_txt, q): (Vec<String>, Vec<f64>) =
            [p[0] - m[1], p[1] + m[0]].iter().map(|x| printed(*x, derived)).unzip();
        let txt = format!("zero on line {} -> {}", pt(&p_txt), pt(&q_txt));
        (PlaneSpec::Line { p: p.clone(), q }, txt)
    } else {
        let txt = format!("zero on plane normal {} through {}", pt(&m_txt), pt(&p_txt));
        (PlaneSpec::NormalThrough { v: m.clone(), p: p.clone() }, txt)
    };
    let mut a = p;
    a[k] += m[k].signum();
    let (a_txt, a): (Vec<String>, Vec<f64>) = a.iter().map(|x| printed(*x, derived)).unzip();
    let (pn, pe) = plane_equation(&plane);
    let value = big * (dot(&pn, &a) - pe).abs() / max_abs(&pn);
    let (value_txt, _) = printed(value, fmt);
    format!("crease {}: {plane_txt}, equals {value_txt} at {}", u.name, pt(&a_txt))
}
