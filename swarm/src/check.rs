//! `cargo check` on assembled chunks: one scratch crate, one module per candidate, so a
//! single run checks many files and each error maps back to its candidate and byte span.
use anyhow::{Context, Result};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path, process::Command};

#[derive(Debug, Clone)]
pub struct CheckError {
    pub code: String,
    pub message: String,
    /// Byte range inside the candidate's own text.
    pub start: usize,
    pub end: usize,
}

/// Errors per candidate index. Candidates with no entry compiled clean.
pub type Report = BTreeMap<usize, Vec<CheckError>>;

const HEADER: &str = "#![allow(warnings)]\n";

/// Write candidates as modules `m0000`.. into `dir` and run `cargo check` there.
/// Candidates must already parse (`syn`); a syntax error anywhere would hide the rest.
///
/// rustc stops after unresolved macros, derives and attributes (serde, bevy, ...) and
/// hides every later error in every module. Candidates that hit one are reported as
/// `external` and dropped from the batch, and the rest are rechecked until none do.
pub fn check_all(dir: &Path, texts: &[String]) -> Result<Report> {
    let mut skip: Vec<bool> = vec![false; texts.len()];
    loop {
        let r = run_once(dir, texts, &skip)?;
        let hit: Vec<usize> = r
            .iter()
            .filter(|(_, es)| es.iter().any(|e| e.code == "syntax"))
            .map(|(i, _)| *i)
            .filter(|i| !skip[*i])
            .collect();
        if hit.is_empty() {
            let mut r = r;
            for (i, s) in skip.iter().enumerate() {
                if *s {
                    r.insert(i, vec![CheckError { code: "external".into(), message: "unresolved macro or attribute".into(), start: 0, end: 0 }]);
                }
            }
            return Ok(r);
        }
        for i in hit {
            skip[i] = true;
        }
    }
}

fn run_once(dir: &Path, texts: &[String], skip: &[bool]) -> Result<Report> {
    let src = dir.join("src");
    let _ = std::fs::remove_dir_all(&src);
    std::fs::create_dir_all(&src)?;
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"scratch\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[workspace]\n\n[lib]\npath = \"src/lib.rs\"\n",
    )?;
    let mut lib = String::from(HEADER);
    for (i, t) in texts.iter().enumerate() {
        lib.push_str(&format!("mod m{i:04};\n"));
        std::fs::write(src.join(format!("m{i:04}.rs")), if skip[i] { "" } else { t })?;
    }
    std::fs::write(src.join("lib.rs"), lib)?;
    let out = Command::new("cargo")
        .args(["check", "--message-format=json", "--offline", "--quiet"])
        .current_dir(dir)
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .output()
        .context("running cargo check")?;
    let mut report = Report::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
        if v["reason"] != "compiler-message" || v["message"]["level"] != "error" {
            continue;
        }
        let m = &v["message"];
        let code = m["code"]["code"].as_str().unwrap_or("syntax").to_owned();
        let message = m["message"].as_str().unwrap_or("").to_owned();
        let Some(span) = m["spans"].as_array().and_then(|s| s.iter().find(|s| s["is_primary"] == true)) else {
            continue;
        };
        let file = span["file_name"].as_str().unwrap_or("");
        let Some(idx) = file
            .rsplit(['/', '\\'])
            .next()
            .and_then(|f| f.strip_prefix('m'))
            .and_then(|f| f.strip_suffix(".rs"))
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        let start = span["byte_start"].as_u64().unwrap_or(0) as usize;
        let end = span["byte_end"].as_u64().unwrap_or(0) as usize;
        report.entry(idx).or_default().push(CheckError { code, message, start, end });
    }
    Ok(report)
}
