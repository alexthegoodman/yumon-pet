mod common;

use std::path::PathBuf;

use common::*;
use wir::io::{self, DType, Format, LoadOptions};

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("wir-io-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn sample() -> wir::Weights {
    let mut w = random_network(&mut Rng::new(3), 3, 7, 2);
    w.input = Some("p".into());
    w.outputs = Some(vec!["u".into(), "v".into()]);
    w
}

#[test]
fn every_format_round_trips_f64_exactly() {
    let w = sample();
    for ext in ["npz", "safetensors", "json"] {
        let path = tmp(&format!("net.{ext}"));
        io::save(&w, &path, DType::F64).unwrap();
        let back = io::load(&path, &LoadOptions::default()).unwrap();
        assert_eq!(back, w, "{ext}");
    }
}

#[test]
fn f32_is_close() {
    let w = sample();
    for ext in ["npz", "safetensors"] {
        let path = tmp(&format!("net32.{ext}"));
        io::save(&w, &path, DType::F32).unwrap();
        let back = io::load(&path, &LoadOptions::default()).unwrap();
        for (a, b) in w.w1.data.iter().zip(&back.w1.data) {
            assert!((a - b).abs() <= 1e-6 * a.abs().max(1.0), "{ext}");
        }
        assert_eq!(back.outputs, w.outputs);
    }
}

#[test]
fn safetensors_uses_torch_sequential_keys() {
    let bytes = io::to_safetensors(&sample(), DType::F64);
    let file = io::read_tensors(&bytes, Format::Safetensors).unwrap();
    let keys: Vec<&String> = file.tensors.keys().collect();
    assert_eq!(keys, ["0.bias", "0.weight", "2.bias", "2.weight"]);
    assert_eq!(file.tensors["0.weight"].shape, vec![7, 3]);
    assert_eq!(file.tensors["2.weight"].shape, vec![2, 7]);
}

#[test]
fn explicit_keys_and_transpose() {
    // A checkpoint slice with its own names, stored [in, out] like Burn's Linear.
    let w = sample();
    let t = |m: &wir::Mat| {
        let mut out = wir::Mat::zeros(m.cols, m.rows);
        for r in 0..m.rows {
            for c in 0..m.cols {
                out.set(c, r, m.get(r, c));
            }
        }
        out
    };
    let json = serde_json::json!({
        "mlp.fc1.weight": t(&w.w1).to_rows(), "mlp.fc1.bias": w.b1,
        "mlp.fc2.weight": t(&w.w2).to_rows(), "mlp.fc2.bias": w.b2,
        "other.weight": [[1.0]],
    });
    let path = tmp("slice.json");
    std::fs::write(&path, json.to_string()).unwrap();
    let keys = ["mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"].map(String::from);
    let back = io::load(&path, &LoadOptions { keys: Some(keys), transpose: true }).unwrap();
    assert_eq!(back.w1, w.w1);
    assert_eq!(back.w2, w.w2);
    // Without keys the error lists what is available.
    let e = io::load(&path, &LoadOptions::default()).unwrap_err().to_string();
    assert!(e.contains("mlp.fc1.weight[3, 7]"), "{e}");
}

#[test]
fn rejects_bad_files() {
    let path = tmp("bad.npz");
    std::fs::write(&path, b"not a zip").unwrap();
    assert!(io::load(&path, &LoadOptions::default()).unwrap_err().to_string().contains("zip"));
    assert!(io::save(&sample(), &tmp("x.bin"), DType::F64).is_err());
}
