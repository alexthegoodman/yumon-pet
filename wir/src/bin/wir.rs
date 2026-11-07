//! `wir` command-line tool.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use wir::io::{self, DType, LoadOptions};
use wir::deep;
use wir::games::{self, Pong};
use wir::num::format_num;
use wir::{
    canonicalize, codebook, compile, decompile, encode, lower_str, print, CompileOptions, DecompileOptions, NumFormat, Weights,
};

const USAGE: &str = "\
wir - WIR v0 / 0.2 compiler and decompiler

usage:
  wir compile <file.wir> [-o weights.npz|.safetensors|.json] [--dtype f64|f32] [--tol 1e-6] [--quiet]
  wir decompile <weights> [-o out.wir] [--exact] [--digits 3] [--keys W1,b1,W2,b2[,W3,b3]]
                [--transpose] [--input x] [--outputs y,z] [--assert-at \"0,0;1,1\"]
                [--domain probe.json] [--shape 64x3x3] [--kernel-terms 3]
                [--codebook g4|g2] [--calibrate probe.json]
  wir fmt <file.wir> [--exact]          print the resolved program (names and asserts kept)
  wir canon <file.wir> [--exact]        print the canonical form (what decompile would emit)
  wir eval <file.wir|weights> <x1,x2,...> [more points]
  wir verify <file.wir|weights> <probe.json> [--tol 1e-4]
                                        compare against reference inputs/outputs
  wir pong [--width 16] [--height 12] [--paddle 3] [--speed 1] [--paddle-speed 1]
           [-o pong.wir] [--check 2000000]
  wir pong --variants N --out DIR [--seed 1] [--check 200000]
                                        generate Pong step programs (0.2), each checked
                                        against a reference simulator

Weights use PyTorch nn.Linear layout: W1 [H, D], b1 [H], W2 [O, H], b2 [O]; SwiGLU
networks add W3 [H, D], b3 [H]. --domain drops creases never active on the probe's
`inputs`; --kernel-terms keeps the first N of 9 patch components (lossy); --digits
sets vector precision (default 3). --codebook writes vectors as codebook words
(lossy, about 4 weights per token for g4), rounding error-aware on the
--calibrate inputs (default: the --domain probe).
.safetensors files use the nn.Sequential(Linear, ReLU, Linear) keys 0.weight ... 2.bias.

`wir 0.2` programs (named scalar inputs, let/output expressions, combinators) compile
to deep ReLU networks: .json (`layers`) or .safetensors (nn.Sequential keys 0, 2, 4, ...).
compile, decompile and eval accept them; eval takes one value per input in order.
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
        "pong" => cmd_pong(rest),
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
    let mut f = if args.has("--exact") { NumFormat::EXACT } else { NumFormat::CANONICAL };
    if let Some(d) = args.get("--digits").and_then(|d| d.parse::<u32>().ok()).filter(|d| (1..=15).contains(d)) {
        f.vector_digits = d;
    }
    f
}

/// `inputs` of a probe JSON file (as written for `verify`).
fn probe_inputs(path: &Path) -> Result<Vec<Vec<f64>>, String> {
    let v: serde_json::Value = serde_json::from_str(&read(&path.to_path_buf())?).map_err(|e| format!("{}: {e}", path.display()))?;
    v["inputs"]
        .as_array()
        .and_then(|rows| rows.iter().map(|r| r.as_array()?.iter().map(|x| x.as_f64()).collect()).collect())
        .ok_or(format!("{}: expected `inputs` as an array of number arrays", path.display()))
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
    let src = read(&path)?;
    if deep::is_deep_source(&src) {
        return compile_deep(&src, &path, &args, opts.tolerance, dtype);
    }
    let c = compile(&src, opts).map_err(|e| format!("{}: {e}", path.display()))?;
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

fn compile_deep(src: &str, path: &Path, args: &Args, tol: f64, dtype: DType) -> Result<(), String> {
    let c = deep::compile(src, tol).map_err(|e| format!("{}: {e}", path.display()))?;
    let w = &c.weights;
    let hidden = w.hidden();
    println!(
        "cost report: {} input(s), {} output(s), {} hidden layer(s) {:?}, {} parameters",
        w.input_dim(), w.output_dim(), hidden.len(), hidden, w.param_count()
    );
    let lines: Vec<&str> = src.lines().collect();
    println!("  {:>5}  {:>5}  {:>5}  statement", "layer", "line", "units");
    for cost in &c.costs {
        let (line, text) = match cost.line {
            Some(l) => (l.to_string(), lines.get(l - 1).map_or("", |t| t.split('#').next().unwrap().trim())),
            None => ("-".into(), "(carries values to a later layer)"),
        };
        println!("  {:>5}  {:>5}  {:>5}  {}", cost.layer, line, cost.units, text);
    }
    if !c.program.asserts.is_empty() {
        println!("asserts: {} passed (tolerance {tol:.0e})", c.program.asserts.len());
    }
    if let Some(out) = args.get("-o") {
        deep::save(w, Path::new(out), dtype).map_err(|e| e.to_string())?;
        println!("wrote {out}");
    }
    Ok(())
}

fn cmd_pong(raw: &[String]) -> Result<(), String> {
    let valued = ["--width", "--height", "--paddle", "--speed", "--paddle-speed", "-o", "--check", "--variants", "--out", "--seed"];
    let args = Args::parse(raw, &valued)?;
    args.check(&valued)?;
    let int = |k: &str, d: i64| -> Result<i64, String> { args.get(k).map_or(Ok(d), |v| v.parse().map_err(|_| format!("bad {k} `{v}`"))) };
    let check = |g: &Pong, src: &str, limit: u64, seed: u64| -> Result<wir::games::PongCheck, String> {
        let c = games::pong_check(g, src, limit, seed).map_err(|e| format!("{}: {e}", g.tag()))?;
        if c.mismatches > 0 {
            return Err(format!("{}: {} of {} states disagree with the simulator", g.tag(), c.mismatches, c.states));
        }
        Ok(c)
    };
    let Some(n) = args.get("--variants") else {
        let g = Pong {
            width: int("--width", 16)?,
            height: int("--height", 12)?,
            paddle: int("--paddle", 3)?,
            speed: int("--speed", 1)?,
            paddle_speed: int("--paddle-speed", 1)?,
        };
        let src = games::pong_program(&g).map_err(|e| e.to_string())?;
        let c = check(&g, &src, int("--check", 2_000_000)? as u64, 1)?;
        eprintln!(
            "{}: hidden {:?}, {} parameters; {} {} states match the simulator (worst |diff| {:.1e})",
            g.tag(), c.hidden, c.params, if c.exhaustive { "all" } else { "a sample of" }, c.states, c.worst
        );
        return match args.get("-o") {
            Some(out) => std::fs::write(out, &src).map_err(|e| format!("{out}: {e}")),
            None => {
                print!("{src}");
                Ok(())
            }
        };
    };
    let n: usize = n.parse().map_err(|_| format!("bad --variants `{n}`"))?;
    let dir = PathBuf::from(args.get("--out").ok_or("--variants needs --out DIR")?);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let limit = int("--check", 200_000)? as u64;
    let mut rng = (int("--seed", 1)? as u64).max(1);
    let mut below = |lo: i64, hi: i64| -> i64 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        lo + (rng % (hi - lo + 1) as u64) as i64
    };
    let mut seen = std::collections::HashSet::new();
    let mut manifest = String::new();
    let mut tries = 0;
    while seen.len() < n {
        tries += 1;
        if tries > 100 * n + 1000 {
            return Err(format!("only {} distinct variants found", seen.len()));
        }
        let (width, height) = (below(8, 32), below(6, 24));
        let paddle = below(1, 6.min(height - 1));
        let speed = below(1, 3.min(height - 1).min(width - 3));
        let paddle_speed = below(1, 2.min(height - paddle));
        let g = Pong { width, height, paddle, speed, paddle_speed };
        if !seen.insert(g.tag()) {
            continue;
        }
        let src = games::pong_program(&g).map_err(|e| e.to_string())?;
        let c = check(&g, &src, limit, seen.len() as u64)?;
        let file = format!("{}.wir", g.tag());
        std::fs::write(dir.join(&file), &src).map_err(|e| format!("{file}: {e}"))?;
        let line = serde_json::json!({
            "file": file, "game": "pong",
            "params": { "width": width, "height": height, "paddle": paddle, "speed": speed, "paddle_speed": paddle_speed },
            "inputs": games::PONG_INPUTS, "outputs": games::PONG_OUTPUTS,
            "hidden": c.hidden, "parameters": c.params, "bytes": src.len(),
            "states_checked": c.states, "exhaustive": c.exhaustive, "worst_diff": c.worst,
        });
        manifest.push_str(&format!("{line}
"));
        eprintln!("{file}: hidden {:?}, {} states checked{}", c.hidden, c.states, if c.exhaustive { " (all)" } else { "" });
    }
    std::fs::write(dir.join("manifest.jsonl"), manifest).map_err(|e| e.to_string())?;
    eprintln!("wrote {n} programs and manifest.jsonl to {}", dir.display());
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
        if keys.len() != 4 && keys.len() != 6 {
            return Err("--keys needs W1,b1,W2,b2 or W1,b1,W2,b2,W3,b3".into());
        }
        opts.keys = Some(keys);
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
    let valued = [
        "-o", "--keys", "--input", "--outputs", "--assert-at", "--domain", "--shape", "--kernel-terms", "--digits",
        "--codebook", "--calibrate",
    ];
    let args = Args::parse(raw, &valued)?;
    let mut allowed = valued.to_vec();
    allowed.extend(["--exact", "--transpose"]);
    args.check(&allowed)?;
    let path = args.file("weights file")?;
    if let Some(d) = deep::load(&path).map_err(|e| e.to_string())? {
        let text = deep::decompile(&d, number_format(&args)).map_err(|e| e.to_string())?;
        eprintln!("decompiled a deep network, hidden layers {:?}", d.hidden());
        return match args.get("-o") {
            Some(out) => std::fs::write(out, &text).map_err(|e| format!("{out}: {e}")),
            None => {
                print!("{text}");
                Ok(())
            }
        };
    }
    let w = load_weights(&path, &args)?;
    let opts = DecompileOptions {
        input: args.get("--input").map(String::from),
        outputs: args.get("--outputs").map(|s| s.split(',').map(|n| n.trim().to_string()).collect()),
        assert_at: args.get("--assert-at").map(parse_points).transpose()?.unwrap_or_default(),
        domain_samples: args.get("--domain").map(|p| probe_inputs(Path::new(p))).transpose()?,
        shape: args
            .get("--shape")
            .map(|s| s.split('x').map(|d| d.parse::<usize>().map_err(|_| format!("bad --shape `{s}`"))).collect())
            .transpose()?,
        kernel_terms: args
            .get("--kernel-terms")
            .map(|n| n.parse::<usize>().ok().filter(|n| (1..=9).contains(n)).ok_or(format!("bad --kernel-terms `{n}` (1-9)")))
            .transpose()?,
        ..Default::default()
    };
    let mut p = decompile(&w, &opts).map_err(|e| e.to_string())?;
    if let Some(name) = args.get("--codebook") {
        let book = codebook(name).ok_or(format!("unknown codebook `{name}` (known: {})", wir::codebook::NAMES.join(", ")))?;
        let samples = match args.get("--calibrate") {
            Some(f) => Some(probe_inputs(Path::new(f))?),
            None => opts.domain_samples.clone(),
        };
        p = encode(&p, book, samples.as_deref());
    }
    let text = print(&p, number_format(&args));
    eprintln!(
        "decompiled {} hidden unit(s) into {} unit(s)",
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
    let args = Args::parse(raw, &["--digits"])?;
    args.check(&["--exact", "--digits"])?;
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
    let is_wir = path.extension().is_some_and(|e| e == "wir");
    let deep_net = if is_wir {
        let src = read(&path)?;
        deep::is_deep_source(&src)
            .then(|| deep::compile(&src, 1e-6).map(|c| c.weights).map_err(|e| format!("{}: {e}", path.display())))
            .transpose()?
    } else {
        deep::load(&path).map_err(|e| e.to_string())?
    };
    if let Some(d) = deep_net {
        for spec in &args.positional[1..] {
            for p in parse_points(spec)? {
                if p.len() != d.input_dim() {
                    return Err(format!("point has {} values but the program has {} inputs", p.len(), d.input_dim()));
                }
                let y = d.eval(&p);
                let pt = p.iter().map(|x| format_num(*x, NumFormat::EXACT)).collect::<Vec<_>>().join(", ");
                let ys: Vec<String> = d.outputs.iter().zip(y).map(|(n, v)| format!("{n} = {}", format_num(v, NumFormat::EXACT))).collect();
                println!("({pt}): {}", ys.join(", "));
            }
        }
        return Ok(());
    }
    let w = if is_wir {
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
