//! `wir` command-line tool.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use wir::io::{self, DType, LoadOptions};
use wir::num::format_num;
use wir::{canonicalize, compile, decompile, lower_str, print, CompileOptions, DecompileOptions, NumFormat, Weights};

const USAGE: &str = "\
wir - WIR v0 compiler and decompiler

usage:
  wir compile <file.wir> [-o weights.npz|.safetensors|.json] [--dtype f64|f32] [--tol 1e-6] [--quiet]
  wir decompile <weights> [-o out.wir] [--exact] [--keys W1,b1,W2,b2] [--transpose]
                [--input x] [--outputs y,z] [--assert-at \"0,0;1,1\"]
  wir fmt <file.wir> [--exact]          print the resolved program (names and asserts kept)
  wir canon <file.wir> [--exact]        print the canonical form (what decompile would emit)
  wir eval <file.wir|weights> <x1,x2,...> [more points]
  wir verify <file.wir|weights> <probe.json> [--tol 1e-4]
                                        compare against reference inputs/outputs

Weights use PyTorch nn.Linear layout: W1 [H, D], b1 [H], W2 [O, H], b2 [O].
.safetensors files use the nn.Sequential(Linear, ReLU, Linear) keys 0.weight ... 2.bias.
";

struct Args {
    positional: Vec<String>,
    flags: Vec<(String, Option<String>)>,
}

impl Args {
    fn parse(raw: &[String], valued: &[&str]) -> Result<Self, String> {
        let mut positional = Vec::new();
        let mut flags = Vec::new();
        let mut it = raw.iter();
        while let Some(a) = it.next() {
            if a.starts_with('-') && a.len() > 1 && !a[1..].starts_with(|c: char| c.is_ascii_digit() || c == '.') {
                if valued.contains(&a.as_str()) {
                    let v = it.next().ok_or(format!("{a} needs a value"))?;
                    flags.push((a.clone(), Some(v.clone())));
                } else {
                    flags.push((a.clone(), None));
                }
            } else {
                positional.push(a.clone());
            }
        }
        Ok(Self { positional, flags })
    }

    fn has(&self, f: &str) -> bool {
        self.flags.iter().any(|(k, _)| k == f)
    }

    fn get(&self, f: &str) -> Option<&str> {
        self.flags.iter().rev().find(|(k, _)| k == f).and_then(|(_, v)| v.as_deref())
    }

    fn check(&self, allowed: &[&str]) -> Result<(), String> {
        match self.flags.iter().find(|(k, _)| !allowed.contains(&k.as_str())) {
            Some((k, _)) => Err(format!("unknown option {k}")),
            None => Ok(()),
        }
    }

    fn file(&self, what: &str) -> Result<PathBuf, String> {
        self.positional.first().map(PathBuf::from).ok_or(format!("missing {what}"))
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = argv.split_first() else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let result = match cmd.as_str() {
        "compile" => cmd_compile(rest),
        "decompile" => cmd_decompile(rest),
        "fmt" => cmd_print(rest, false),
        "canon" => cmd_print(rest, true),
        "eval" => cmd_eval(rest),
        "verify" => cmd_verify(rest),
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command `{other}`\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn read(path: &PathBuf) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn number_format(args: &Args) -> NumFormat {
    if args.has("--exact") { NumFormat::EXACT } else { NumFormat::CANONICAL }
}

fn cmd_compile(raw: &[String]) -> Result<(), String> {
    let args = Args::parse(raw, &["-o", "--dtype", "--tol"])?;
    args.check(&["-o", "--dtype", "--tol", "--quiet"])?;
    let path = args.file("input .wir file")?;
    let mut opts = CompileOptions::default();
    if let Some(t) = args.get("--tol") {
        opts.tolerance = t.parse().map_err(|_| format!("bad --tol `{t}`"))?;
    }
    let dtype = match args.get("--dtype") {
        None | Some("f64") => DType::F64,
        Some("f32") => DType::F32,
        Some(d) => return Err(format!("bad --dtype `{d}` (f64 or f32)")),
    };
    let c = compile(&read(&path)?, opts).map_err(|e| format!("{}: {e}", path.display()))?;
    for w in &c.warnings {
        eprintln!("{w}");
    }
    let (d, h, o) = (c.weights.input_dim(), c.weights.hidden(), c.weights.output_dim());
    println!("cost report: input R{d}, {o} output(s), {h} hidden unit(s)");
    println!("  {:>5}  {:>5}  {:>6}  statement", "line", "units", "params");
    for line in &c.costs {
        println!("  {:>5}  {:>5}  {:>6}  {}", line.line, line.units, line.params, line.text);
    }
    let total_units: usize = c.costs.iter().map(|l| l.units).sum();
    let total_params: usize = c.costs.iter().map(|l| l.params).sum();
    debug_assert_eq!(total_params, c.weights.param_count());
    println!(
        "  {:>5}  {:>5}  {:>6}  W1 {h}x{d}, b1 {h}, W2 {o}x{h}, b2 {o}",
        "total", total_units, total_params
    );
    for s in &c.solves {
        println!(
            "solve `{}` (line {}): max residual {:.3e}, rank {}/{}",
            s.output, s.line, s.max_residual, s.rank, s.columns
        );
    }
    if !c.program.asserts.is_empty() {
        println!("asserts: {} passed (tolerance {:.0e})", c.program.asserts.len(), opts.tolerance);
    }
    if !args.has("--quiet") {
        print_weights(&c.weights);
    }
    if let Some(out) = args.get("-o") {
        io::save(&c.weights, &PathBuf::from(out), dtype).map_err(|e| e.to_string())?;
        println!("wrote {out}");
    }
    Ok(())
}

fn print_weights(w: &Weights) {
    let f = |x: &f64| format_num(*x, NumFormat::EXACT);
    let row = |r: &[f64]| format!("[{}]", r.iter().map(f).collect::<Vec<_>>().join(", "));
    if w.param_count() > 400 {
        println!("({} parameters; pass -o to save them)", w.param_count());
        return;
    }
    println!("W1 = [{}]", w.w1.to_rows().iter().map(|r| row(r)).collect::<Vec<_>>().join(", "));
    println!("b1 = {}", row(&w.b1));
    println!("W2 = [{}]", w.w2.to_rows().iter().map(|r| row(r)).collect::<Vec<_>>().join(", "));
    println!("b2 = {}", row(&w.b2));
}

fn load_weights(path: &Path, args: &Args) -> Result<Weights, String> {
    let mut opts = LoadOptions { transpose: args.has("--transpose"), ..Default::default() };
    if let Some(k) = args.get("--keys") {
        let keys: Vec<String> = k.split(',').map(|s| s.trim().to_string()).collect();
        opts.keys = Some(keys.try_into().map_err(|_| "--keys needs four names: W1,b1,W2,b2".to_string())?);
    }
    io::load(path, &opts).map_err(|e| e.to_string())
}

fn parse_points(s: &str) -> Result<Vec<Vec<f64>>, String> {
    s.split(';')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            p.trim()
                .trim_matches(|c| c == '(' || c == ')')
                .split(',')
                .map(|x| x.trim().parse::<f64>().map_err(|_| format!("bad coordinate `{x}` in `{p}`")))
                .collect()
        })
        .collect()
}

fn cmd_decompile(raw: &[String]) -> Result<(), String> {
    let valued = ["-o", "--keys", "--input", "--outputs", "--assert-at"];
    let args = Args::parse(raw, &valued)?;
    let mut allowed = valued.to_vec();
    allowed.extend(["--exact", "--transpose"]);
    args.check(&allowed)?;
    let path = args.file("weights file")?;
    let w = load_weights(&path, &args)?;
    let opts = DecompileOptions {
        input: args.get("--input").map(String::from),
        outputs: args.get("--outputs").map(|s| s.split(',').map(|n| n.trim().to_string()).collect()),
        assert_at: args.get("--assert-at").map(parse_points).transpose()?.unwrap_or_default(),
        ..Default::default()
    };
    let p = decompile(&w, &opts).map_err(|e| e.to_string())?;
    let text = print(&p, number_format(&args));
    eprintln!(
        "decompiled {} hidden unit(s) into {} crease(s)",
        w.hidden(),
        p.units.len()
    );
    match args.get("-o") {
        Some(out) => std::fs::write(out, &text).map_err(|e| format!("{out}: {e}"))?,
        None => print!("{text}"),
    }
    Ok(())
}

fn cmd_print(raw: &[String], canonical: bool) -> Result<(), String> {
    let args = Args::parse(raw, &[])?;
    args.check(&["--exact"])?;
    let path = args.file("input .wir file")?;
    let p = lower_str(&read(&path)?).map_err(|e| format!("{}: {e}", path.display()))?;
    let p = if canonical { canonicalize(&p, Default::default()) } else { p };
    print!("{}", print(&p, number_format(&args)));
    Ok(())
}

fn cmd_eval(raw: &[String]) -> Result<(), String> {
    let valued = ["--keys"];
    let args = Args::parse(raw, &valued)?;
    args.check(&["--keys", "--transpose"])?;
    let path = args.file("program or weights file")?;
    let w = if path.extension().is_some_and(|e| e == "wir") {
        compile(&read(&path)?, CompileOptions::default()).map_err(|e| format!("{}: {e}", path.display()))?.weights
    } else {
        load_weights(&path, &args)?
    };
    let names = w.outputs.clone().unwrap_or_else(|| (1..=w.output_dim()).map(|k| format!("y{k}")).collect());
    for spec in &args.positional[1..] {
        for p in parse_points(spec)? {
            if p.len() != w.input_dim() {
                return Err(format!("point has {} coordinates but the input is R{}", p.len(), w.input_dim()));
            }
            let y = w.eval(&p);
            let pt = p.iter().map(|x| format_num(*x, NumFormat::EXACT)).collect::<Vec<_>>().join(", ");
            for (n, v) in names.iter().zip(y) {
                println!("{n}({pt}) = {}", format_num(v, NumFormat::EXACT));
            }
        }
    }
    Ok(())
}

/// `probe.json` holds `inputs` [N][D] and reference `outputs` [N][O], e.g. activations
/// and logits recorded from the model a weights file was taken from.
fn cmd_verify(raw: &[String]) -> Result<(), String> {
    let valued = ["--keys", "--tol"];
    let args = Args::parse(raw, &valued)?;
    args.check(&["--keys", "--tol", "--transpose"])?;
    let path = args.file("program or weights file")?;
    let probe_path = args.positional.get(1).map(PathBuf::from).ok_or("missing probe .json")?;
    let tol: f64 = match args.get("--tol") {
        Some(t) => t.parse().map_err(|_| format!("bad --tol `{t}`"))?,
        None => 1e-4,
    };
    let w = if path.extension().is_some_and(|e| e == "wir") {
        compile(&read(&path)?, CompileOptions::default()).map_err(|e| format!("{}: {e}", path.display()))?.weights
    } else {
        load_weights(&path, &args)?
    };
    let probe: serde_json::Value =
        serde_json::from_str(&read(&probe_path)?).map_err(|e| format!("{}: {e}", probe_path.display()))?;
    let matrix = |key: &str| -> Result<Vec<Vec<f64>>, String> {
        probe[key]
            .as_array()
            .ok_or(format!("probe has no `{key}` array"))?
            .iter()
            .map(|r| r.as_array().and_then(|r| r.iter().map(|x| x.as_f64()).collect::<Option<Vec<_>>>()))
            .collect::<Option<_>>()
            .ok_or(format!("probe `{key}` must be an array of number arrays"))
    };
    let (xs, ys) = (matrix("inputs")?, matrix("outputs")?);
    if xs.len() != ys.len() {
        return Err(format!("probe has {} inputs but {} outputs", xs.len(), ys.len()));
    }
    let (mut max_abs, mut max_ref, mut argmax_same) = (0f64, 0f64, 0usize);
    let (mut sq_diff, mut sq_ref) = (0f64, 0f64);
    let argmax = |v: &[f64]| v.iter().enumerate().fold(0, |b, (i, x)| if *x > v[b] { i } else { b });
    for (x, y) in xs.iter().zip(&ys) {
        if x.len() != w.input_dim() || y.len() != w.output_dim() {
            return Err(format!(
                "probe point is R{} -> R{}, the network is R{} -> R{}",
                x.len(), y.len(), w.input_dim(), w.output_dim()
            ));
        }
        let got = w.eval(x);
        for (g, e) in got.iter().zip(y) {
            max_abs = max_abs.max((g - e).abs());
            max_ref = max_ref.max(e.abs());
            sq_diff += (g - e) * (g - e);
            sq_ref += e * e;
        }
        argmax_same += (argmax(&got) == argmax(y)) as usize;
    }
    println!(
        "{} points, {} outputs each: max |diff| {:.3e} (reference max |y| {:.3e}), argmax agrees on {}/{}",
        xs.len(), w.output_dim(), max_abs, max_ref, argmax_same, xs.len()
    );
    if sq_ref > 0.0 {
        println!("relative RMS error |diff| / |y|: {:.4e}", (sq_diff / sq_ref).sqrt());
    }
    if max_abs > tol {
        return Err(format!("max |diff| {max_abs:.3e} exceeds tolerance {tol:.0e}"));
    }
    println!("ok (tolerance {tol:.0e})");
    Ok(())
}
