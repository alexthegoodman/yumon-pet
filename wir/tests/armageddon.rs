//! Armageddon 2070 networks: every program compiles and its network matches the program on
//! random in-domain inputs.
use wir::armageddon;

#[test]
fn every_network_matches_its_program() {
    let (json, reports) = armageddon::export_json(2_000, 3).unwrap();
    assert_eq!(reports.len(), 23);
    for r in &reports {
        assert_eq!(r.mismatches, 0, "{} disagrees with its program ({} cases)", r.name, r.cases);
    }
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["format"], "wir-armageddon");
    assert_eq!(v["nets"].as_object().unwrap().len(), 22);
}

#[test]
fn infantry_brain_reads_squad_modes_and_senses() {
    let n = armageddon::infantry_program(&armageddon::Infantry::all()[0]);
    for name in ["m0", "m6", "call", "dl", "supp", "go0", "gl7", "gcs"] {
        assert!(n.inputs.iter().any(|(i, _, _)| i == name), "missing input {name}");
    }
    assert_eq!(armageddon::sense_names().len(), 58);
}
