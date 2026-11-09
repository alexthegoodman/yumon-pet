use wir::superpose::{choice_layer, sample_states, ChoiceLayerSpec, Preserve, State};

const CHOICE: &str = "wir 0.2
input a, b, c: R1 in [0, 3]
let fa = 2*a
let fb = b + relu(c - 1)
let fc = min(a, c)
let half = 3*fb
let k0 = 3*fa + 2
let k1 = half + 1
let k2 = 3*fc + 2*b
output w0 = and(gt(k0, k1), gt(k0, k2))
output w1 = and(ge(k1, k0), gt(k1, k2))
output w2 = and(ge(k2, k0), ge(k2, k1))
output plain = fb
";

fn grid() -> Vec<State> {
    let mut s = vec![];
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                s.push((vec![a as f64, b as f64, c as f64], 1.0));
            }
        }
    }
    s
}

fn spec(neurons: usize, preserve: Preserve, digits: usize) -> ChoiceLayerSpec {
    ChoiceLayerSpec {
        features: vec!["fa".into(), "fb".into(), "fc".into()],
        groups: vec![vec!["k0".into(), "k1".into(), "k2".into()]],
        neurons,
        preserve,
        digits,
        weights: vec![],
        fidelity: 0.001,
    }
}

#[test]
fn full_layer_keeps_every_value() {
    let r = choice_layer(CHOICE, &spec(3, Preserve::Values, 12), &grid()).unwrap();
    assert!((r.variance_kept - 1.0).abs() < 1e-9);
    assert!(r.groups[0].max_abs < 1e-6, "max |key diff| {}", r.groups[0].max_abs);
    assert_eq!(r.groups[0].winner_kept, 1.0);
    // the copy of `half` reads the decoded feature; `plain` still reads the exact one
    assert!(r.source.contains("let sp_half = 3*sp_fb"), "{}", r.source);
    assert!(r.source.contains("let k1 = sp_half + 1"), "{}", r.source);
    assert!(r.source.contains("output plain = fb"), "{}", r.source);
    assert!(r.source.contains("let sp_z2 = unit("), "{}", r.source);
}

#[test]
fn squeezed_layer_reports_a_certificate() {
    let r = choice_layer(CHOICE, &spec(2, Preserve::Winner { margin: 1.0 }, 4), &grid()).unwrap();
    assert_eq!(r.states, 64);
    assert!(r.groups[0].winner_kept > 0.8, "winner kept {}", r.groups[0].winner_kept);
    assert!((0.0..=1.0).contains(&r.certified));
    assert!((0.0..=1.0).contains(&r.outputs_kept));
    assert_eq!(r.after.0.len(), wir::deep::compile(&r.source, 1e-6).unwrap().weights.hidden().len());
}

#[test]
fn keys_must_read_features_affinely() {
    let src = CHOICE.replace("let k2 = 3*fc + 2*b", "let k2 = relu(3*fc - 2) + 2*b");
    let e = choice_layer(&src, &spec(2, Preserve::Values, 4), &grid()).unwrap_err().to_string();
    assert!(e.contains("nonlinearity"), "{e}");
}

#[test]
fn features_come_before_keys() {
    let src = CHOICE.replace("let fc = min(a, c)\n", "").replace("let k2 = 3*fc + 2*b", "let fc = min(a, c)\nlet k2 = 3*fc + 2*b");
    let e = choice_layer(&src, &spec(2, Preserve::Values, 4), &grid()).unwrap_err().to_string();
    assert!(e.contains("before"), "{e}");
}

#[test]
fn sampled_states_stay_in_their_intervals() {
    let s = sample_states(CHOICE, 500, 3).unwrap();
    assert_eq!(s.len(), 500);
    assert!(s.iter().all(|(x, w)| *w == 1.0 && x.iter().all(|v| v.fract() == 0.0 && (0.0..=3.0).contains(v))));
}
