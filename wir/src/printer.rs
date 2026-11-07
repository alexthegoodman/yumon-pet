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
//!
//! Compact syntax: when D > 3 or the program has gates, points stop being a
//! readable way to pin a plane (a point in R576 is 576 numbers), so creases print
//! as `ramp along V above E` (relu(V . x - E)), gates as `silu along V, times
//! along U`, every vector as integers over a power of ten, and each readout as one
//! coefficient vector dotted with the unit list: `y = [..] / 1e4 . (c1..c64) + b`.
//! `R<c>x3x3` inputs print directions as `patch` components (see `ast::PATCH_BASIS`).

use crate::ast::{patch_profile, PlaneSpec, PATCH_BASIS};
use crate::compiler::{lower_str, plane_equation};
use crate::canon::max_abs;
use crate::model::{dot, Program, Unit, UnitKind};
use crate::num::{format_num, format_vector, NumFormat, Precision};

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

    let compact = p.dim > 3 || p.units.iter().any(|u| u.up.is_some());
    let mut out = String::from("wir 0.1\n");
    let space = match &p.shape {
        Some(s) => s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"),
        None => p.dim.to_string(),
    };
    let domain = p.domain.map_or(String::new(), |(lo, hi)| format!(" in [{}, {}]", num(lo), num(hi)));
    out += &format!("input  {}: R{space}{domain}\n", p.input);
    for o in &p.outputs {
        out += &format!("output {}: R1\n", o.name);
    }

    let decls: Vec<&Unit> = p.units.iter().filter(|u| !inline(u)).collect();
    if !decls.is_empty() {
        out.push('\n');
    }
    for u in &decls {
        out += &match (&u.up, compact) {
            (Some(up), _) => format!(
                "gate {}: silu {}, times {}",
                u.name,
                ramp_text(&u.w, u.b, p, fmt),
                ramp_text(&up.u, up.c, p, fmt)
            ),
            (None, true) => format!("crease {}: ramp {}", u.name, ramp_text(&u.w, u.b, p, fmt)),
            (None, false) => crease_decl(u, p.dim, fmt),
        };
        out.push('\n');
    }

    out.push('\n');
    for (k, o) in p.outputs.iter().enumerate() {
        if compact && !decls.is_empty() {
            let coefs: Vec<f64> = decls.iter().map(|u| u.coeffs[k]).collect();
            let names: Vec<&str> = decls.iter().map(|u| u.name.as_str()).collect();
            let bias = num(o.bias.abs());
            let bias = if bias == "0" { String::new() } else { format!(" {} {bias}", if o.bias < 0.0 { "-" } else { "+" }) };
            out += &format!("{} = {} . ({}){bias}\n", o.name, format_vector(&coefs, fmt), name_list(&names));
            continue;
        }
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

/// `along DIR [above E]` for the unit relu/silu(w . x + b).
fn ramp_text(w: &[f64], b: f64, p: &Program, fmt: NumFormat) -> String {
    let above = format_num(-b, fmt);
    let above = if above == "0" { String::new() } else { format!(" above {above}") };
    format!("along {}{above}", direction_text(w, p, fmt))
}

/// A vector, or for `R<c>x3x3` inputs its `patch` components (zero ones omitted).
fn direction_text(w: &[f64], p: &Program, fmt: NumFormat) -> String {
    let Some(&[c, 3, 3]) = p.shape.as_deref() else {
        return format_vector(w, fmt);
    };
    let mut parts = Vec::new();
    let comps: Vec<Vec<f64>> = PATCH_BASIS
        .iter()
        .map(|&(_, i, j)| {
            let basis = |r: usize, col: usize| patch_profile(i, col) * patch_profile(j, r);
            let den: f64 = (0..9).map(|t| basis(t / 3, t % 3).powi(2)).sum();
            (0..c).map(|ch| (0..9).map(|t| w[ch * 9 + t] * basis(t / 3, t % 3)).sum::<f64>() / den).collect()
        })
        .collect();
    let big = comps.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
    for (&(name, _, _), v) in PATCH_BASIS.iter().zip(&comps) {
        if v.iter().any(|x| x.abs() > 1e-12 * big) {
            parts.push(format!("{name} {}", format_vector(v, fmt)));
        }
    }
    if parts.is_empty() {
        parts.push(format!("level {}", format_vector(&comps[0], fmt)));
    }
    format!("patch {}", parts.join(" "))
}

/// `c1..c64` when the names are one prefix with consecutive numbers, else a comma list.
fn name_list(names: &[&str]) -> String {
    let split = |n: &str| {
        let i = n.trim_end_matches(|c: char| c.is_ascii_digit()).len();
        (n[..i].to_string(), n[i..].parse::<usize>().ok())
    };
    let parsed: Vec<_> = names.iter().map(|n| split(n)).collect();
    // `c01` would not survive the range, so require names to equal prefix + number.
    let consecutive = names.len() > 2
        && parsed.iter().zip(names).all(|((p, k), n)| !p.is_empty() && p == &parsed[0].0 && k.is_some_and(|k| format!("{p}{k}") == *n))
        && parsed.windows(2).all(|w| w[1].1 == w[0].1.map(|k| k + 1));
    if consecutive {
        format!("{}..{}", names[0], names[names.len() - 1])
    } else {
        names.join(", ")
    }
}
