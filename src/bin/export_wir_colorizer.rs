//! Export the CNN colorizer's classification head as a WIR weights file.
//!
//! `out_conv(relu(head_conv(f)))` with a 3x3 `head_conv` and a 1x1 `out_conv` is, at
//! every pixel, a one-hidden-layer ReLU network over the 3x3 feature patch: input
//! R^(c*9), c hidden units, 313 outputs. That is exactly WIR v0, so `wir decompile`
//! applies with no approximation.
//!
//! Writes (in `--out-dir`):
//! - `head.json`: W1 [c, c*9], b1 [c], W2 [313, c], b2 [313] (nn.Linear layout,
//!   patch index = channel*9 + ky*3 + kx; zero padding at the border).
//! - `probe.json`: real backbone patches from `--image` and the model's own logits at
//!   those pixels, for `wir verify`.

use anyhow::{Context, Result};
use burn::{
    module::Module,
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::TensorData,
};
use clap::Parser;
use serde_json::json;
use std::path::PathBuf;

use yumon_pet::vision::colorizer::{srgb_to_lab, ColorizerModel, ColorizerModelConfig, INPUT_SIZE, NUM_COLOR_BINS};

type ExportBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

#[derive(Parser, Debug)]
#[command(about = "Export the CNN colorizer head (conv3x3 -> ReLU -> conv1x1) for WIR")]
struct Args {
    /// Checkpoint directory containing model.bin (CNN colorizer, model_type "cnn").
    #[arg(long, default_value = "checkpoints/old/yumon-colorizer-256-a")]
    checkpoint: PathBuf,

    #[arg(long, default_value_t = 64)]
    channels: usize,

    #[arg(long, default_value_t = 5)]
    num_res_blocks: usize,

    /// Image whose backbone activations become the probe set.
    #[arg(long, default_value = "gt_sample0.png")]
    image: PathBuf,

    /// Number of probe pixels (the four corners are always included).
    #[arg(long, default_value_t = 512)]
    samples: usize,

    #[arg(long, default_value = "tmp/wir-colorizer")]
    out_dir: PathBuf,
}

fn floats<const D: usize>(t: Tensor<ExportBackend, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec().unwrap()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device: <ExportBackend as Backend>::Device = Default::default();
    let c = args.channels;

    let model: ColorizerModel<ExportBackend> = ColorizerModelConfig::new()
        .with_scales_from(INPUT_SIZE)
        .with_channels(c)
        .with_num_res_blocks(args.num_res_blocks)
        .init(&device);
    let model_path = args.checkpoint.join("model");
    let model = model
        .load_file(&model_path, &BinFileRecorder::<FullPrecisionSettings>::default(), &device)
        .with_context(|| format!("loading {}", model_path.display()))?;
    println!("loaded {}", model_path.display());

    // Head weights. Conv2d weight is [out, in, kh, kw], so a row-major reshape to
    // [out, in*9] is already nn.Linear layout with patch index in*9 + ky*3 + kx.
    let (head, out) = model.head();
    let w1 = floats(head.weight.val());
    let b1 = floats(head.bias.as_ref().context("head_conv has no bias")?.val());
    let w2 = floats(out.weight.val());
    let b2 = floats(out.bias.as_ref().context("out_conv has no bias")?.val());
    let d = c * 9;
    anyhow::ensure!(w1.len() == c * d && w2.len() == NUM_COLOR_BINS * c, "unexpected head shapes");
    let rows = |v: &[f32], n: usize| -> Vec<Vec<f64>> { v.chunks(n).map(|r| r.iter().map(|&x| x as f64).collect()).collect() };
    let vec64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
    let outputs: Vec<String> = (0..NUM_COLOR_BINS).map(|q| format!("bin{q}")).collect();

    std::fs::create_dir_all(&args.out_dir)?;
    let head_json = json!({
        "format": "wir-weights",
        "source": format!("{} head_conv -> relu -> out_conv", args.checkpoint.display()),
        "input": "f",
        "outputs": outputs,
        "W1": rows(&w1, d), "b1": vec64(&b1),
        "W2": rows(&w2, c), "b2": vec64(&b2),
    });
    let head_path = args.out_dir.join("head.json");
    std::fs::write(&head_path, serde_json::to_string(&head_json)?)?;
    println!("wrote {} (W1 {c}x{d}, W2 {NUM_COLOR_BINS}x{c})", head_path.display());

    // Probe: run the real model on an image, keep backbone patches and logits.
    let img = image::open(&args.image).with_context(|| format!("opening {}", args.image.display()))?.to_rgb8();
    let img = image::imageops::resize(&img, INPUT_SIZE as u32, INPUT_SIZE as u32, image::imageops::FilterType::Triangle);
    let flat: Vec<f32> = img.pixels().map(|p| (srgb_to_lab(p[0], p[1], p[2]).0 - 50.0) / 100.0).collect();
    let x = Tensor::<ExportBackend, 4>::from_floats(TensorData::new(flat, [1, 1, INPUT_SIZE, INPUT_SIZE]), &device);

    let f = model.features(x.clone());
    let [_, fc, h, w] = f.dims();
    let logits = floats(model.forward(x));
    let f = floats(f);

    let mut pixels = vec![(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)];
    let mut state = 0x9e3779b97f4a7c15u64;
    while pixels.len() < args.samples.min(h * w) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let p = ((state % (h * w) as u64) as usize / w, (state % (h * w) as u64) as usize % w);
        if !pixels.contains(&p) {
            pixels.push(p);
        }
    }

    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    let mut active = vec![0usize; c];
    for &(py, px) in &pixels {
        let mut patch = vec![0f64; d];
        for ch in 0..fc {
            for ky in 0..3 {
                for kx in 0..3 {
                    let (yy, xx) = (py as isize + ky as isize - 1, px as isize + kx as isize - 1);
                    if yy >= 0 && xx >= 0 && (yy as usize) < h && (xx as usize) < w {
                        patch[ch * 9 + ky * 3 + kx] = f[(ch * h + yy as usize) * w + xx as usize] as f64;
                    }
                }
            }
        }
        for (u, a) in active.iter_mut().enumerate() {
            let z: f64 = b1[u] as f64 + (0..d).map(|i| w1[u * d + i] as f64 * patch[i]).sum::<f64>();
            if z > 0.0 {
                *a += 1;
            }
        }
        inputs.push(patch);
        expected.push((0..NUM_COLOR_BINS).map(|q| logits[(q * h + py) * w + px] as f64).collect::<Vec<_>>());
    }
    let never = active.iter().filter(|&&a| a == 0).count();
    println!("probe: {} pixels of {h}x{w}; {never}/{c} hidden units never active on them", pixels.len());

    let probe = json!({
        "source": format!("{} on {}", args.checkpoint.display(), args.image.display()),
        "pixels": pixels.iter().map(|&(y, x)| [y, x]).collect::<Vec<_>>(),
        "inputs": inputs,
        "outputs": expected,
    });
    let probe_path = args.out_dir.join("probe.json");
    std::fs::write(&probe_path, serde_json::to_string(&probe)?)?;
    println!("wrote {}", probe_path.display());
    Ok(())
}
