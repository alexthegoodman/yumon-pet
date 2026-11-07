//! Source round-trips: printer idempotence and decompile(compile(src)) == canonicalize(src).

mod common;

use common::Rng;
use wir::{canonical_text, canonicalize, compile, decompile, lower_str, print, DecompileOptions, NumFormat};

fn example(name: &str) -> String {
    std::fs::read_to_string(format!("{}/examples/{name}.wir", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

fn fmt(src: &str, f: NumFormat) -> String {
    print(&lower_str(src).unwrap_or_else(|e| panic!("{e}\n{src}")), f)
}

fn decompiled_text(src: &str) -> String {
    let c = compile(src, Default::default()).unwrap_or_else(|e| panic!("{e}\n{src}"));
    print(&decompile(&c.weights, &DecompileOptions::default()).unwrap(), NumFormat::CANONICAL)
}

/// Small integers and halves keep generated sources readable.
fn coord_value(rng: &mut Rng) -> f64 {
    rng.below(9) as f64 / 2.0 - 2.0
}

fn point_value(rng: &mut Rng, d: usize) -> Vec<f64> {
    (0..d).map(|_| coord_value(rng)).collect()
}

fn show(p: &[f64]) -> String {
    format!("({})", p.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", "))
}

/// A random, valid WIR program using every plane form, hinges, constants,
/// at most one solve, and sometimes two outputs.
fn random_source(rng: &mut Rng, d: usize) -> String {
    loop {
        let n_out = if rng.chance(0.2) { 2 } else { 1 };
        let mut head = format!("wir 0.1\ninput x: R{d}\n");
        for k in 0..n_out {
            head += &format!("output y{k}: R1\n");
        }
        let n_creases = rng.below(5) + usize::from(d > 1);
        let names: Vec<String> = (0..n_creases).map(|i| format!("k{i}")).collect();
        for n in &names {
            let plane = match rng.below(3) {
                0 if d == 2 => format!("zero on line {} -> {}", show(&point_value(rng, d)), show(&point_value(rng, d))),
                1 => format!("between {} and {}", show(&point_value(rng, d)), show(&point_value(rng, d))),
                _ => format!("zero on plane normal {} offset {}", show(&point_value(rng, d)), coord_value(rng)),
            };
            let value = rng.below(8) + 1;
            head += &format!("crease {n}: {plane}, equals {value}/2 at {}\n", show(&point_value(rng, d)));
        }
        // Creases alone must lower before a solve can be fitted against them.
        let Ok(creases_only) = lower_str(&format!("{head}{}", (0..n_out).map(|k| format!("y{k} = {}\n", names.join(" + ") + " + 0")).collect::<String>().replace("=  + 0", "= 0"))) else {
            continue;
        };
        let mut body = String::new();
        let solve_output = (!names.is_empty() && rng.chance(0.3)).then(|| rng.below(n_out));
        for k in 0..n_out {
            if solve_output == Some(k) {
                let mut set = names.clone();
                rng.shuffle(&mut set);
                set.truncate(1 + rng.below(names.len()));
                // Targets come from a combination of the solve set itself, so the fit is exact.
                let pairs: Vec<String> = (0..set.len() + 3)
                    .map(|_| {
                        let x = point_value(rng, d);
                        let v: f64 = creases_only.units.iter().filter(|u| set.contains(&u.name)).map(|u| u.eval(&x)).sum::<f64>() + 0.5;
                        format!("{}->{v}", show(&x))
                    })
                    .collect();
                body += &format!("y{k} = solve over {{{}}}: {}\n", set.join(", "), pairs.join(", "));
                continue;
            }
            let mut terms = Vec::new();
            for n in &names {
                if rng.chance(0.7) {
                    terms.push(format!("{}*{n}", coord_value(rng)));
                }
            }
            if d == 1 {
                for _ in 0..rng.below(4) {
                    let left = if rng.chance(0.3) { "left " } else { "" };
                    terms.push(format!("hinge {left}at {} turn {}", coord_value(rng), coord_value(rng)));
                }
            }
            if rng.chance(0.5) {
                terms.push(coord_value(rng).to_string());
            }
            if terms.is_empty() {
                terms.push("0".into());
            }
            body += &format!("y{k} = {}\n", terms.join("\n  + "));
        }
        let src = head + &body;
        if compile(&src, Default::default()).is_ok() {
            return src;
        }
    }
}

#[test]
fn print_is_idempotent_on_examples() {
    for name in ["tent", "xor", "xor_solve"] {
        let src = example(name);
        for f in [NumFormat::CANONICAL, NumFormat::EXACT] {
            let once = fmt(&src, f);
            let twice = fmt(&once, f);
            assert_eq!(once, twice, "{name}");
            let canon = print(&canonicalize(&lower_str(&src).unwrap(), Default::default()), f);
            assert_eq!(fmt(&canon, f), canon, "{name} canonical");
        }
    }
}

#[test]
fn decompile_of_compile_equals_canonical_source_on_examples() {
    for name in ["tent", "xor", "xor_solve"] {
        let src = example(name);
        assert_eq!(decompiled_text(&src), canonical_text(&src).unwrap(), "{name}");
    }
    // XOR written two ways has one canonical form.
    assert_eq!(canonical_text(&example("xor")).unwrap(), canonical_text(&example("xor_solve")).unwrap());
}

#[test]
fn random_sources_round_trip() {
    let mut rng = Rng::new(42);
    for case in 0..600 {
        let d = 1 + case % 3;
        let src = random_source(&mut rng, d);
        // Printer idempotence, both precisions.
        for f in [NumFormat::CANONICAL, NumFormat::EXACT] {
            let once = fmt(&src, f);
            assert_eq!(fmt(&once, f), once, "case {case} not idempotent:\n{src}\n---\n{once}");
        }
        // Weights path and source path meet in the same canonical text.
        let canon = canonical_text(&src).unwrap();
        assert_eq!(decompiled_text(&src), canon, "case {case}:\n{src}");
        assert_eq!(canonical_text(&canon).unwrap(), canon, "case {case}: canonical form not stable");
        // The exactly printed program is the same function.
        let a = compile(&src, Default::default()).unwrap().weights;
        let b = compile(&fmt(&src, NumFormat::EXACT), Default::default()).unwrap().weights;
        let diff = common::max_diff(&mut rng, &a, &b, 200, 3.0);
        assert!(diff < 1e-9, "case {case}: exact print changed the function by {diff}\n{src}");
    }
}
