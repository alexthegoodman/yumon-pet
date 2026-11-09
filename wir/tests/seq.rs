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
