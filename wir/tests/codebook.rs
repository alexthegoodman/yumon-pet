//! Coded vectors (`codebook g4`, `{word ...} / q`) and `encode`.

mod common;

use common::*;
use wir::{codebook, compile, decompile, encode, lower_str, print, DecompileOptions, Mat, NumFormat, Weights};

fn swiglu(rng: &mut Rng, d: usize, h: usize, o: usize) -> Weights {
    let m = |rng: &mut Rng, r: usize, c: usize| Mat { rows: r, cols: c, data: (0..r * c).map(|_| rng.uniform(-1.0, 1.0)).collect() };
    let v = |rng: &mut Rng, n: usize| (0..n).map(|_| rng.uniform(-0.5, 0.5)).collect::<Vec<_>>();
    Weights {
        w1: m(rng, h, d),
        b1: v(rng, h),
        w2: m(rng, o, h),
        b2: v(rng, o),
        up: Some((m(rng, h, d), v(rng, h))),
        input: None,
        outputs: None,
        shape: None,
    }
}

fn rel_err(a: &Weights, b: &Weights, xs: &[Vec<f64>]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for x in xs {
        for (ya, yb) in a.eval(x).iter().zip(b.eval(x)) {
            num += (ya - yb) * (ya - yb);
            den += ya * ya;
        }
    }
    (num / den).sqrt()
}

#[test]
fn coded_literals_expand_to_codebook_entries() {
    let book = codebook("g4").unwrap();
    let (a, b) = (book.words[0], book.words[5]);
    let src = format!("wir 0.1\ncodebook g4\ninput x: R8\noutput y: R1\ncrease c: ramp along {{{a} {b}}} / 2\ny = c\n");
    let w = compile(&src, Default::default()).unwrap().weights;
    let want: Vec<f64> = book.entry(0).iter().chain(book.entry(5)).map(|x| x / 2.0).collect();
    assert_eq!(w.w1.row(0), want.as_slice());

    let err = |s: &str| compile(s, Default::default()).unwrap_err().to_string();
    assert!(err(&src.replace(b, "zzzzqq")).contains("not a word of codebook g4"));
    assert!(err(&src.replace("codebook g4\n", "")).contains("need a `codebook` line"));
    assert!(err(&src.replace("codebook g4", "codebook g9")).contains("unknown codebook `g9`"));
}

/// Encoded programs print as words, re-read to the identical weights, and the text
/// is a print fixed point.
#[test]
fn encoded_text_round_trips_exactly() {
    let mut rng = Rng::new(11);
    for (name, case) in ["g4", "g2"].iter().flat_map(|n| (0..20).map(move |c| (*n, c))) {
        let d = 5 + rng.below(20);
        let h = 1 + rng.below(16);
        let w = if case % 2 == 0 { swiglu(&mut rng, d, h, 2) } else { random_network(&mut rng, d, h, 2) };
        let p = decompile(&w, &DecompileOptions::default()).unwrap();
        let coded = encode(&p, codebook(name).unwrap(), None);
        let text = print(&coded, NumFormat::CANONICAL);
        assert!(text.starts_with(&format!("wir 0.1\ncodebook {name}\n")), "{text}");
        // Every vector codes; a last word may overhang the length.
        assert!(!text.contains('['), "{text}");
        let reread = lower_str(&text).unwrap();
        assert_eq!(print(&reread, NumFormat::CANONICAL), text);
        let (a, b) = (coded.to_weights(), reread.to_weights());
        assert_eq!(a.w1.data, b.w1.data);
        assert_eq!(a.w2.data, b.w2.data);
        assert_eq!(a.up.as_ref().map(|u| &u.0.data), b.up.as_ref().map(|u| &u.0.data));
    }
}

/// Error-aware rounding on calibration inputs beats plain nearest-word rounding there.
#[test]
fn calibration_reduces_error() {
    let mut rng = Rng::new(23);
    let w = swiglu(&mut rng, 32, 16, 4);
    let xs: Vec<Vec<f64>> = (0..400)
        .map(|_| {
            // correlated inputs, where error feedback has something to work with
            let z = random_point(&mut rng, 32, 1.0);
            (0..32).map(|i| z[i] + 0.8 * z[(i + 1) % 32]).collect()
        })
        .collect();
    let p = decompile(&w, &DecompileOptions::default()).unwrap();
    let book = codebook("g4").unwrap();
    let plain = rel_err(&w, &encode(&p, book, None).to_weights(), &xs);
    let tuned = rel_err(&w, &encode(&p, book, Some(&xs)).to_weights(), &xs);
    assert!(tuned < plain, "calibrated {tuned:e} vs plain {plain:e}");
    assert!(tuned < 0.35, "calibrated error {tuned:e}");
}

/// A last word may overhang a vector by fewer than `dim` entries; more is an error,
/// and number vectors must still match exactly.
#[test]
fn last_word_may_overhang() {
    let book = codebook("g4").unwrap();
    let w = |i: usize| book.words[i];
    let src = format!(
        "wir 0.1\ncodebook g4\ninput x: R6\noutput y: R1\n\
         crease a: ramp along {{{} {}}} / 2\ncrease b: ramp along [1 0 0 0 0 0]\ny = {{{}}} . (a, b)\n",
        w(1),
        w(2),
        w(3)
    );
    let c = compile(&src, Default::default()).unwrap();
    let want: Vec<f64> = book.entry(1).iter().chain(&book.entry(2)[..2]).map(|x| x / 2.0).collect();
    assert_eq!(c.weights.w1.row(0), want.as_slice());
    assert_eq!(c.weights.w2.row(0), &book.entry(3)[..2]);
    let text = print(&c.program, NumFormat::CANONICAL);
    assert_eq!(print(&lower_str(&text).unwrap(), NumFormat::CANONICAL), text);

    let err = |s: String| compile(&s, Default::default()).unwrap_err().to_string();
    let three = src.replace(&format!("{{{} {}}}", w(1), w(2)), &format!("{{{} {} {}}}", w(1), w(2), w(4)));
    assert!(err(three).contains("entries"), "an overhang of a whole word is an error");
    assert!(err(src.replace("[1 0 0 0 0 0]", "[1 0 0 0 0 0 0]")).contains("entries"));
}
