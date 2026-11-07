//! Weight files: `.json`, `.safetensors` and `.npz`.
//!
//! Tensor names on write:
//! - `.npz` / `.json`: `W1`, `b1`, `W2`, `b2` (NumPy-first, as in the spec),
//!   plus a `wir_meta` uint8 array holding the input/output names as JSON.
//! - `.safetensors`: `0.weight`, `0.bias`, `2.weight`, `2.bias`, the state-dict
//!   keys of `nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, O))`;
//!   names go in `__metadata__`.
//!
//! On read, any of those spellings (or `w1`/`w2`) is accepted, or explicit
//! keys can be given for slices of larger checkpoints. `.npz` must be stored
//! (`np.savez`), not deflated (`np.savez_compressed`).

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::error::{Result, WirError};
use crate::model::{Mat, Weights};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Json,
    Safetensors,
    Npz,
}

impl Format {
    pub fn from_path(path: &Path) -> Result<Format> {
        match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
            Some("json") => Ok(Format::Json),
            Some("safetensors") => Ok(Format::Safetensors),
            Some("npz") => Ok(Format::Npz),
            _ => Err(WirError::new(format!(
                "{}: unknown weight format; use .npz, .safetensors or .json",
                path.display()
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    F32,
    #[default]
    F64,
}

/// Which tensors to read, for checkpoints that do not use the default names.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Explicit names for W1, b1, W2, b2.
    pub keys: Option<[String; 4]>,
    /// Weights are stored [in, out] (Burn `Linear` layout) rather than [out, in].
    pub transpose: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct TensorFile {
    pub tensors: BTreeMap<String, Tensor>,
    pub input: Option<String>,
    pub outputs: Option<Vec<String>>,
}

pub fn save(w: &Weights, path: &Path, dtype: DType) -> Result<()> {
    let bytes = match Format::from_path(path)? {
        Format::Json => to_json(w).into_bytes(),
        Format::Safetensors => to_safetensors(w, dtype),
        Format::Npz => to_npz(w, dtype),
    };
    std::fs::write(path, bytes).map_err(|e| WirError::new(format!("{}: {e}", path.display())))
}

pub fn load(path: &Path, opts: &LoadOptions) -> Result<Weights> {
    let bytes = std::fs::read(path).map_err(|e| WirError::new(format!("{}: {e}", path.display())))?;
    let file = read_tensors(&bytes, Format::from_path(path)?)
        .map_err(|e| WirError::new(format!("{}: {}", path.display(), e.message)))?;
    assemble(&file, opts)
}

pub fn read_tensors(bytes: &[u8], format: Format) -> Result<TensorFile> {
    match format {
        Format::Json => from_json(bytes),
        Format::Safetensors => from_safetensors(bytes),
        Format::Npz => from_npz(bytes),
    }
}

fn layer_tensors(w: &Weights) -> [(&'static str, Vec<usize>, &[f64]); 4] {
    [
        ("W1", vec![w.w1.rows, w.w1.cols], &w.w1.data),
        ("b1", vec![w.b1.len()], &w.b1),
        ("W2", vec![w.w2.rows, w.w2.cols], &w.w2.data),
        ("b2", vec![w.b2.len()], &w.b2),
    ]
}

fn meta_json(w: &Weights) -> Value {
    json!({ "input": w.input, "outputs": w.outputs })
}

/// Pick W1, b1, W2, b2 out of a tensor file and check their shapes.
pub fn assemble(file: &TensorFile, opts: &LoadOptions) -> Result<Weights> {
    const CANDIDATES: [&[&str]; 4] =
        [&["W1", "w1", "0.weight"], &["b1", "0.bias"], &["W2", "w2", "2.weight"], &["b2", "2.bias"]];
    let mut picked = Vec::new();
    for (i, cands) in CANDIDATES.iter().enumerate() {
        let t = match &opts.keys {
            Some(keys) => file.tensors.get(&keys[i]).ok_or_else(|| {
                WirError::new(format!("tensor `{}` not found; available: {}", keys[i], available(file)))
            })?,
            None => cands.iter().find_map(|k| file.tensors.get(*k)).ok_or_else(|| {
                WirError::new(format!(
                    "none of {} found; available: {} (pass explicit keys for other names)",
                    cands.join("/"),
                    available(file)
                ))
            })?,
        };
        picked.push(t.clone());
    }
    let [w1, b1, w2, b2]: [Tensor; 4] = picked.try_into().unwrap();
    let matrix = |t: Tensor, name: &str| -> Result<Mat> {
        let (r, c) = match t.shape.as_slice() {
            [r, c] => (*r, *c),
            [n] if name == "W2" => (1, *n),
            s => return Err(WirError::new(format!("{name} must be 2-D, got shape {s:?}"))),
        };
        let m = Mat { rows: r, cols: c, data: t.data };
        Ok(if opts.transpose && t.shape.len() == 2 { transpose(&m) } else { m })
    };
    let vector = |t: Tensor, name: &str| -> Result<Vec<f64>> {
        match t.shape.len() {
            0 | 1 => Ok(t.data),
            _ => Err(WirError::new(format!("{name} must be 1-D, got shape {:?}", t.shape))),
        }
    };
    let weights = Weights {
        w1: matrix(w1, "W1")?,
        b1: vector(b1, "b1")?,
        w2: matrix(w2, "W2")?,
        b2: vector(b2, "b2")?,
        input: file.input.clone(),
        outputs: file.outputs.clone(),
    };
    weights.check_shapes()?;
    Ok(weights)
}

fn available(file: &TensorFile) -> String {
    let names: Vec<String> = file.tensors.iter().map(|(k, t)| format!("{k}{:?}", t.shape)).take(40).collect();
    if names.is_empty() { "(no tensors)".into() } else { names.join(", ") }
}

fn transpose(m: &Mat) -> Mat {
    let mut t = Mat::zeros(m.cols, m.rows);
    for r in 0..m.rows {
        for c in 0..m.cols {
            t.set(c, r, m.get(r, c));
        }
    }
    t
}

fn read_meta(v: &Value) -> (Option<String>, Option<Vec<String>>) {
    let input = v.get("input").and_then(|x| x.as_str()).map(String::from);
    let outputs = v
        .get("outputs")
        .and_then(|x| x.as_array())
        .and_then(|a| a.iter().map(|s| s.as_str().map(String::from)).collect::<Option<Vec<_>>>());
    (input, outputs)
}

// ---------------------------------------------------------------- JSON

pub fn to_json(w: &Weights) -> String {
    let mut obj = serde_json::Map::new();
    obj.insert("format".into(), json!("wir-weights"));
    obj.insert("input".into(), json!(w.input));
    obj.insert("outputs".into(), json!(w.outputs));
    obj.insert("W1".into(), json!(w.w1.to_rows()));
    obj.insert("b1".into(), json!(w.b1));
    obj.insert("W2".into(), json!(w.w2.to_rows()));
    obj.insert("b2".into(), json!(w.b2));
    serde_json::to_string_pretty(&Value::Object(obj)).unwrap() + "\n"
}

fn from_json(bytes: &[u8]) -> Result<TensorFile> {
    let v: Value = serde_json::from_slice(bytes).map_err(|e| WirError::new(format!("invalid JSON: {e}")))?;
    let obj = v.as_object().ok_or_else(|| WirError::new("expected a JSON object of tensors"))?;
    let mut file = TensorFile::default();
    (file.input, file.outputs) = read_meta(&v);
    for (k, val) in obj {
        if let Some(t) = json_tensor(val) {
            file.tensors.insert(k.clone(), t);
        }
    }
    Ok(file)
}

fn json_tensor(v: &Value) -> Option<Tensor> {
    match v {
        Value::Number(n) => Some(Tensor { shape: vec![], data: vec![n.as_f64()?] }),
        Value::Array(items) => {
            let parts: Vec<Tensor> = items.iter().map(json_tensor).collect::<Option<_>>()?;
            let inner = parts.first().map_or(vec![], |t| t.shape.clone());
            if parts.iter().any(|t| t.shape != inner) {
                return None;
            }
            let mut shape = vec![parts.len()];
            shape.extend(inner);
            Some(Tensor { shape, data: parts.into_iter().flat_map(|t| t.data).collect() })
        }
        _ => None,
    }
}

// ---------------------------------------------------------------- safetensors

pub fn to_safetensors(w: &Weights, dtype: DType) -> Vec<u8> {
    let torch_names = ["0.weight", "0.bias", "2.weight", "2.bias"];
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for ((_, shape, values), name) in layer_tensors(w).into_iter().zip(torch_names) {
        let start = data.len();
        encode(values, dtype, &mut data);
        header.insert(
            name.into(),
            json!({
                "dtype": if dtype == DType::F64 { "F64" } else { "F32" },
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let mut meta = serde_json::Map::new();
    if let Some(i) = &w.input {
        meta.insert("wir.input".into(), json!(i));
    }
    if let Some(o) = &w.outputs {
        meta.insert("wir.outputs".into(), json!(serde_json::to_string(o).unwrap()));
    }
    header.insert("__metadata__".into(), Value::Object(meta));
    let mut h = serde_json::to_string(&Value::Object(header)).unwrap().into_bytes();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    out
}

fn from_safetensors(bytes: &[u8]) -> Result<TensorFile> {
    let bad = |m: &str| WirError::new(format!("invalid safetensors: {m}"));
    if bytes.len() < 8 {
        return Err(bad("file too short"));
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8usize.checked_add(n).filter(|e| *e <= bytes.len()).ok_or_else(|| bad("header length out of range"))?;
    let header: Value = serde_json::from_slice(&bytes[8..header_end]).map_err(|e| bad(&e.to_string()))?;
    let obj = header.as_object().ok_or_else(|| bad("header is not an object"))?;
    let body = &bytes[header_end..];
    let mut file = TensorFile::default();
    if let Some(meta) = obj.get("__metadata__") {
        file.input = meta.get("wir.input").and_then(|v| v.as_str()).map(String::from);
        file.outputs = meta
            .get("wir.outputs")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok());
    }
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info["dtype"].as_str().ok_or_else(|| bad("missing dtype"))?;
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .and_then(|a| a.iter().map(|x| x.as_u64().map(|v| v as usize)).collect())
            .ok_or_else(|| bad("missing shape"))?;
        let offs = info["data_offsets"].as_array().ok_or_else(|| bad("missing data_offsets"))?;
        let (start, end) = match offs.as_slice() {
            [a, b] => (a.as_u64().unwrap_or(u64::MAX) as usize, b.as_u64().unwrap_or(u64::MAX) as usize),
            _ => return Err(bad("data_offsets must have two entries")),
        };
        if start > end || end > body.len() {
            return Err(bad(&format!("tensor `{name}` points outside the file")));
        }
        let Some(data) = decode(&body[start..end], dtype) else {
            // Integer and other tensors are not weights; skip rather than fail.
            continue;
        };
        if data.len() != shape.iter().product::<usize>() {
            return Err(bad(&format!("tensor `{name}` has {} values for shape {shape:?}", data.len())));
        }
        file.tensors.insert(name.clone(), Tensor { shape, data });
    }
    Ok(file)
}

fn encode(values: &[f64], dtype: DType, out: &mut Vec<u8>) {
    for v in values {
        match dtype {
            DType::F64 => out.extend(v.to_le_bytes()),
            DType::F32 => out.extend((*v as f32).to_le_bytes()),
        }
    }
}

/// Little-endian decode; `None` for dtypes that are not floats.
fn decode(bytes: &[u8], dtype: &str) -> Option<Vec<f64>> {
    let chunks = |n: usize| bytes.chunks_exact(n).filter(move |_| bytes.len().is_multiple_of(n));
    Some(match dtype {
        "F64" => chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
        "F32" => chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        "F16" => chunks(2).map(|c| f16_to_f64(u16::from_le_bytes(c.try_into().unwrap()))).collect(),
        "BF16" => chunks(2)
            .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16) as f64)
            .collect(),
        _ => return None,
    })
}

fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f64;
    sign * match exp {
        0 => frac * 2f64.powi(-24),
        31 => {
            if frac == 0.0 { f64::INFINITY } else { f64::NAN }
        }
        _ => (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    }
}

// ---------------------------------------------------------------- npy / npz

fn npy(shape: &[usize], descr: &str, data: &[u8]) -> Vec<u8> {
    let dims = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {dims}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header.as_bytes());
    out.extend(data);
    out
}

pub fn to_npz(w: &Weights, dtype: DType) -> Vec<u8> {
    let descr = if dtype == DType::F64 { "<f8" } else { "<f4" };
    let mut entries = Vec::new();
    for (name, shape, values) in layer_tensors(w) {
        let mut data = Vec::new();
        encode(values, dtype, &mut data);
        entries.push((format!("{name}.npy"), npy(&shape, descr, &data)));
    }
    let meta = serde_json::to_vec(&meta_json(w)).unwrap();
    entries.push(("wir_meta.npy".into(), npy(&[meta.len()], "|u1", &meta)));
    zip_stored(&entries)
}

fn from_npz(bytes: &[u8]) -> Result<TensorFile> {
    let mut file = TensorFile::default();
    for (name, data) in unzip_stored(bytes)? {
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        let (shape, descr, payload, fortran) = parse_npy(&data).map_err(|m| WirError::new(format!("{name}: {m}")))?;
        if key == "wir_meta" && descr == "|u1" {
            if let Ok(v) = serde_json::from_slice::<Value>(payload) {
                (file.input, file.outputs) = read_meta(&v);
            }
            continue;
        }
        let Some(mut values) = decode_npy(payload, &descr) else { continue };
        if values.len() != shape.iter().product::<usize>() {
            return Err(WirError::new(format!("{name}: {} values for shape {shape:?}", values.len())));
        }
        if fortran && shape.len() == 2 {
            let m = Mat { rows: shape[1], cols: shape[0], data: values };
            values = transpose(&m).data;
        }
        file.tensors.insert(key, Tensor { shape, data: values });
    }
    Ok(file)
}

fn decode_npy(bytes: &[u8], descr: &str) -> Option<Vec<f64>> {
    match descr {
        "<f8" => decode(bytes, "F64"),
        "<f4" => decode(bytes, "F32"),
        "<f2" => decode(bytes, "F16"),
        _ => None,
    }
}

fn parse_npy(b: &[u8]) -> std::result::Result<(Vec<usize>, String, &[u8], bool), String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("not a .npy array".into());
    }
    let (hlen, start) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 | 3 if b.len() >= 12 => (u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize, 12),
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let header = std::str::from_utf8(b.get(start..start + hlen).ok_or("truncated header")?).map_err(|_| "header is not UTF-8")?;
    let field = |key: &str| -> std::result::Result<&str, String> {
        let i = header.find(&format!("'{key}':")).ok_or(format!("header has no '{key}'"))?;
        Ok(header[i + key.len() + 3..].trim_start())
    };
    let descr_s = field("descr")?;
    let q = descr_s.chars().next().ok_or("bad descr")?;
    let descr = descr_s[1..].split(q).next().ok_or("bad descr")?.to_string();
    let fortran = field("fortran_order")?.starts_with("True");
    let shape_s = field("shape")?;
    let inner = &shape_s[1..shape_s.find(')').ok_or("bad shape")?];
    let shape = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('L').parse::<usize>().map_err(|_| format!("bad shape entry `{s}`")))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((shape, descr, &b[start + hlen..], fortran))
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

fn zip_stored(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
    const DOS_DATE: u16 = (1 << 5) | 1; // 1980-01-01
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let offset = out.len() as u32;
        let crc = crc32(data);
        let (size, nlen) = (data.len() as u32, name.len() as u16);
        let common = |v: &mut Vec<u8>| {
            v.extend(20u16.to_le_bytes()); // version needed
            v.extend(0u16.to_le_bytes()); // flags
            v.extend(0u16.to_le_bytes()); // method: stored
            v.extend(0u16.to_le_bytes()); // time
            v.extend(DOS_DATE.to_le_bytes());
            v.extend(crc.to_le_bytes());
            v.extend(size.to_le_bytes());
            v.extend(size.to_le_bytes());
            v.extend(nlen.to_le_bytes());
            v.extend(0u16.to_le_bytes()); // extra length
        };
        out.extend(0x0403_4b50u32.to_le_bytes());
        common(&mut out);
        out.extend(name.as_bytes());
        out.extend(data);

        central.extend(0x0201_4b50u32.to_le_bytes());
        central.extend(20u16.to_le_bytes()); // version made by
        common(&mut central);
        central.extend([0u8; 8]); // comment len, disk, internal attr, external attr (part)
        central.extend(0u16.to_le_bytes()); // external attr (rest)
        central.extend(offset.to_le_bytes());
        central.extend(name.as_bytes());
    }
    let cd_offset = out.len() as u32;
    let cd_size = central.len() as u32;
    out.extend(central);
    out.extend(0x0605_4b50u32.to_le_bytes());
    out.extend([0u8; 4]); // disk numbers
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend(cd_size.to_le_bytes());
    out.extend(cd_offset.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out
}

fn unzip_stored(b: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    let bad = |m: &str| WirError::new(format!("invalid npz: {m}"));
    let u16_at = |i: usize| -> Result<u16> {
        b.get(i..i + 2).map(|s| u16::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| bad("truncated"))
    };
    let u32_at = |i: usize| -> Result<u32> {
        b.get(i..i + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| bad("truncated"))
    };
    let u64_at = |i: usize| -> Result<u64> {
        b.get(i..i + 8).map(|s| u64::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| bad("truncated"))
    };
    let min = b.len().saturating_sub(22 + 65535);
    let eocd = (min..b.len().saturating_sub(21))
        .rev()
        .find(|&i| b[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or_else(|| bad("no end-of-central-directory record (is this a zip file?)"))?;
    let mut count = u16_at(eocd + 10)? as u64;
    let mut cd = u32_at(eocd + 16)? as u64;
    // Zip64 end record, via its locator just before the classic one.
    if (count == 0xFFFF || cd == 0xFFFF_FFFF) && eocd >= 20 && u32_at(eocd - 20)? == 0x0706_4b50 {
        let z = u64_at(eocd - 12)? as usize;
        count = u64_at(z + 32)?;
        cd = u64_at(z + 48)?;
    }
    let mut pos = cd as usize;
    let mut out = Vec::new();
    for _ in 0..count {
        if u32_at(pos)? != 0x0201_4b50 {
            return Err(bad("bad central directory entry"));
        }
        let method = u16_at(pos + 10)?;
        let crc = u32_at(pos + 16)?;
        let mut csize = u32_at(pos + 20)? as u64;
        let mut usize_ = u32_at(pos + 24)? as u64;
        let nlen = u16_at(pos + 28)? as usize;
        let elen = u16_at(pos + 30)? as usize;
        let clen = u16_at(pos + 32)? as usize;
        let mut local = u32_at(pos + 42)? as u64;
        let name = String::from_utf8_lossy(b.get(pos + 46..pos + 46 + nlen).ok_or_else(|| bad("truncated name"))?).into_owned();
        // Zip64 extra field: replaces whichever of the 32-bit fields are saturated, in order.
        let mut e = pos + 46 + nlen;
        let extra_end = e + elen;
        while e + 4 <= extra_end {
            let (id, len) = (u16_at(e)?, u16_at(e + 2)? as usize);
            if id == 0x0001 {
                let mut f = e + 4;
                for field in [&mut usize_, &mut csize, &mut local] {
                    if *field == 0xFFFF_FFFF && f + 8 <= e + 4 + len {
                        *field = u64_at(f)?;
                        f += 8;
                    }
                }
            }
            e += 4 + len;
        }
        if method != 0 {
            return Err(bad(&format!(
                "`{name}` is compressed (method {method}); save with np.savez, not np.savez_compressed"
            )));
        }
        let l = local as usize;
        if u32_at(l)? != 0x0403_4b50 {
            return Err(bad("bad local header"));
        }
        let data_start = l + 30 + u16_at(l + 26)? as usize + u16_at(l + 28)? as usize;
        let data = b.get(data_start..data_start + csize as usize).ok_or_else(|| bad("truncated entry"))?;
        if crc32(data) != crc {
            return Err(bad(&format!("CRC mismatch in `{name}`")));
        }
        out.push((name, data.to_vec()));
        pos += 46 + nlen + elen + clen;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_reference() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn f16_decoding() {
        assert_eq!(f16_to_f64(0x3c00), 1.0);
        assert_eq!(f16_to_f64(0xc000), -2.0);
        assert_eq!(f16_to_f64(0x3555), 0.333251953125);
    }
}
