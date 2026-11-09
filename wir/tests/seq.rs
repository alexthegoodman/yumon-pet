//! WIR 0.3 sequence programs: attention heads compiled to a norm-free transformer.

use wir::seq::{self, all_sequences, check, compile_str, random_sequences};

fn example(name: &str) -> String {
    std::fs::read_to_string(format!("{}/examples/{name}.wir", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// Program and network both match the task on every sequence up to `max_len`, plus a
/// sample at full length.
fn exact(file: &str, task: &str, max_len: usize) {
    let (p, w, _) = compile_str(&example(file)).unwrap();
    let r = check(&p, &w, task, 0, all_sequences(p.ctx.vocab, max_len)).unwrap();
    assert_eq!((r.program_wrong, r.network_wrong), (0, 0), "{file}: {r:?}");
    assert!(r.worst < 1e-6, "{file}: worst {}", r.worst);
    let r = check(&p, &w, task, 0, random_sequences(p.ctx.vocab, max_len + 1, p.ctx.len, 2000, 7)).unwrap();
    assert_eq!((r.program_wrong, r.network_wrong), (0, 0), "{file} sampled: {r:?}");
}

#[test]
fn reverse_histogram_sort_and_causal_histogram_are_exact() {
    exact("reverse", "reverse", 4);
    exact("hist", "hist", 4);
    exact("sort", "sort", 4);
    exact("prefix_hist", "prefix_hist", 4);
}

#[test]
fn layout_of_the_primitives() {
    let (_, w, c) = compile_str(&example("hist")).unwrap();
    assert_eq!((c.blocks, c.heads), (1, 1));
    // 1 + 1 + (1 + 8) + (1 + 8) embedding dims, the head and its count decoder
    assert_eq!(c.d_model, w.dims.len());
    assert_eq!(c.d_model, 22);
    let (_, _, c) = compile_str(&example("reverse")).unwrap();
    assert_eq!((c.blocks, c.heads), (3, 2));
    let (_, _, c) = compile_str(&example("sort")).unwrap();
    assert_eq!((c.blocks, c.heads), (4, 2));
}

#[test]
fn length_and_empty_mean() {
    let src = "wir 0.3\nsequence len 5\ninput t: token in [0, 2]\noutput n = count(true)\noutput none = mean(t + 1, k(t) == q(t + 3))\n";
    let (p, w, _) = compile_str(src).unwrap();
    let r = check(&p, &w, "length", 0, all_sequences(p.ctx.vocab, 5)).unwrap();
    assert_eq!((r.program_wrong, r.network_wrong), (0, 0));
    for s in all_sequences(p.ctx.vocab, 5) {
        let net = w.eval(&s);
        assert!(net.iter().all(|row| row[1].abs() < 1e-6), "empty mean must read 0: {s:?} {net:?}");
    }
}

#[test]
fn sharpness_sets_the_leak() {
    let mut worst = vec![];
    for b in [10.0, 20.0, 30.0] {
        let mut p = seq::lower(&example("reverse")).unwrap();
        p.ctx.sharpness = b;
        let (w, _) = seq::compile(&p).unwrap();
        worst.push(check(&p, &w, "reverse", 0, all_sequences(p.ctx.vocab, 3)).unwrap().worst);
    }
    assert!(worst[0] > worst[1] && worst[1] > worst[2] && worst[2] > 0.0, "{worst:?}");
}

#[test]
fn errors() {
    let bad = |src: &str, msg: &str| {
        let e = seq::lower(src).unwrap_err().to_string();
        assert!(e.contains(msg), "{e}");
    };
    let head = "wir 0.3\nsequence len 4\ninput t: token in [0, 3]\n";
    bad(&format!("{head}output y = count(k(t) == k(t))\n"), "one `k(..)` side with one `q(..)`");
    bad(&format!("{head}output y = count(k(t / 2) == q(t))\n"), "integer intervals");
    bad(&format!("{head}output y = mean(t, k(t) < q(t))\n"), "expected");
    bad("wir 0.3\ninput t: token in [0, 3]\noutput y = t\n", "sequence len N");
    bad("wir 0.3\nsequence len 4\noutput y = 1\n", "token input");
    assert!(seq::is_seq_source(head) && !seq::is_seq_source("wir 0.2\ninput x: R1\noutput y = x\n"));
}

#[test]
fn the_bound_holds_on_the_examples() {
    for (file, task) in [("reverse", "reverse"), ("hist", "hist"), ("sort", "sort"), ("prefix_hist", "prefix_hist")] {
        let (p, w, _) = compile_str(&example(file)).unwrap();
        let b = seq::error_bound(&p).outputs[0];
        assert!(b < 1e-6, "{file}: bound {b}");
        let r = check(&p, &w, task, 0, random_sequences(p.ctx.vocab, 1, p.ctx.len, 3000, 3)).unwrap();
        assert!(r.worst <= b, "{file}: observed {} above bound {b}", r.worst);
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }
}

/// A random 0.3 program: integer values (token, position, counts, clips) feed
/// predicate sides; means read affine values and become outputs.
fn random_program(r: &mut Rng) -> String {
    let len = r.int(2, 6);
    let vlo = r.int(-2, 1);
    let vhi = vlo + r.int(1, 4);
    let causal = if r.int(0, 1) == 1 { " causal" } else { "" };
    let beta = [20, 30, 40][r.int(0, 2) as usize];
    let mut src = format!("wir 0.3\nsequence len {len}{causal} sharpness {beta}\ninput t: token in [{vlo}, {vhi}]\n");
    let mut ints = vec!["t".to_string(), "pos".to_string()];
    let mut vals: Vec<String> = vec![];
    let affine = |r: &mut Rng, pool: &[String]| -> String {
        let mut e = format!("{}", r.int(-2, 2));
        for _ in 0..r.int(1, 2) {
            let k = [-2, -1, 1, 2][r.int(0, 3) as usize];
            e.push_str(&format!(" + {k} * {}", pool[r.int(0, pool.len() as i64 - 1) as usize]));
        }
        e
    };
    let cmps = ["eq", "ne", "lt", "le", "gt", "ge"];
    for i in 0..r.int(2, 5) {
        let pred = if r.int(0, 4) == 0 {
            "true".to_string()
        } else {
            let (a, b) = (affine(r, &ints), affine(r, &ints));
            let (ka, kb) = if r.int(0, 1) == 0 { ("k", "q") } else { ("q", "k") };
            format!("{}({ka}({a}), {kb}({b}))", cmps[r.int(0, 5) as usize])
        };
        let name = format!("v{i}");
        match r.int(0, 2) {
            0 => {
                src.push_str(&format!("let {name} = count({pred})\n"));
                ints.push(name);
            }
            1 => {
                let v = if vals.is_empty() || r.int(0, 1) == 0 { affine(r, &ints) } else { format!("{} + {}", affine(r, &ints), vals[0]) };
                src.push_str(&format!("let {name} = mean({v}, {pred})\n"));
                vals.push(name);
            }
            _ => {
                let lo = r.int(-3, 2);
                src.push_str(&format!("let {name} = clip({}, {lo}, {})\n", affine(r, &ints), lo + r.int(0, 3)));
                ints.push(name);
            }
        }
    }
    for (o, x) in ints.iter().skip(2).chain(&vals).enumerate() {
        src.push_str(&format!("output o{o} = {x}\n"));
    }
    if ints.len() == 2 && vals.is_empty() {
        src.push_str("output o0 = t\n");
    }
    src
}

#[test]
fn fuzz_random_programs_stay_within_the_bound() {
    // WIR_SEQ_FUZZ=N runs N programs (default 300) and prints statistics.
    let total: usize = std::env::var("WIR_SEQ_FUZZ").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let mut r = Rng(0x5eed_a77e_47104);
    let (mut programs, mut rejected, mut tight, mut ratio, mut causal, mut checked) = (0, 0, 0, 0.0f64, 0, 0u64);
    while programs < total {
        let src = random_program(&mut r);
        let Ok((p, w, _)) = compile_str(&src) else {
            rejected += 1; // e.g. a side wider than MAX_ONEHOT
            assert!(rejected < 1000, "generator produces mostly invalid programs:\n{src}");
            continue;
        };
        programs += 1;
        causal += p.ctx.causal as usize;
        let bound = seq::error_bound(&p);
        tight += bound.outputs.iter().all(|b| *b < 0.5) as usize;
        for s in random_sequences(p.ctx.vocab, 1, p.ctx.len, 60, programs as u64) {
            let (prog, net) = (p.eval(&s), w.eval(&s));
            for i in 0..s.len() {
                for (o, b) in bound.outputs.iter().enumerate() {
                    let d = (net[i][o] - prog[i][o]).abs();
                    checked += 1;
                    if *b > 1e-9 && *b < 0.5 && d / b > ratio {
                        ratio = d / b;
                        if std::env::var("WIR_SEQ_FUZZ_SHOW").is_ok() {
                            eprintln!("ratio {ratio:.6} (d {d:.3e}, bound {b:.3e}) o{o} at {i} on {s:?}
{src}");
                        }
                    }
                    assert!(d <= b + 1e-9, "|network - program| {d} above bound {b} for o{o} on {s:?}\n{src}");
                }
            }
        }
    }
    eprintln!(
        "fuzz: {programs} programs ({causal} causal, {rejected} rejected), {checked} output values, {tight} with every bound < 0.5, largest observed/bound (bounds above f64 noise, 1e-9) {ratio:.3}"
    );
    assert!(tight > total / 2, "only {tight} of {total} programs have a useful bound");
}
