// dataset.rs - 64x64 Image Dataset loader and image I/O with [-1.0, 1.0] normalization

use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::TensorData;

/// Scans a directory for .png images.
pub fn discover_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if !dir.exists() {
        return Ok(paths);
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext.eq_ignore_ascii_case("png")) {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// Loads a single 64x64 PNG image as a [3, 64, 64] float tensor in [-1.0, 1.0].
pub fn load_png_as_tensor<B: Backend>(path: &Path, device: &B::Device) -> Result<Tensor<B, 3>> {
    let img = image::open(path)
        .with_context(|| format!("failed to open image: {:?}", path))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    anyhow::ensure!(w == 64 && h == 64, "image {:?} dimensions {}x{} != 64x64", path, w, h);

    let raw = img.into_raw(); // [64 * 64 * 3] interleaved RGB
    let area = (w * h) as usize;
    let mut ch0 = Vec::with_capacity(area);
    let mut ch1 = Vec::with_capacity(area);
    let mut ch2 = Vec::with_capacity(area);

    for pixel in raw.chunks_exact(3) {
        ch0.push((pixel[0] as f32 / 127.5) - 1.0);
        ch1.push((pixel[1] as f32 / 127.5) - 1.0);
        ch2.push((pixel[2] as f32 / 127.5) - 1.0);
    }

    let mut planar = Vec::with_capacity(3 * area);
    planar.extend(ch0);
    planar.extend(ch1);
    planar.extend(ch2);

    Ok(Tensor::from_data(
        TensorData::new(planar, [3, 64, 64]),
        device,
    ))
}

/// Loads a batch of images into a [B, 3, 64, 64] tensor.
pub fn load_image_batch<B: Backend>(paths: &[PathBuf], device: &B::Device) -> Result<Tensor<B, 4>> {
    anyhow::ensure!(!paths.is_empty(), "cannot load empty image batch");
    let tensors: Result<Vec<Tensor<B, 3>>> = paths
        .iter()
        .map(|p| load_png_as_tensor(p, device))
        .collect();
    let tensors = tensors?;
    Ok(Tensor::stack(tensors, 0))
}

/// Saves a single [3, 64, 64] float tensor (values in [-1.0, 1.0]) as a 64x64 PNG file.
pub fn save_tensor_as_png<B: Backend>(tensor: Tensor<B, 3>, path: &Path) -> Result<()> {
    let [c, h, w] = tensor.dims();
    anyhow::ensure!(c == 3 && h == 64 && w == 64, "expected [3, 64, 64] tensor, got [{}, {}, {}]", c, h, w);
    let data = tensor.into_data().convert::<f32>().to_vec::<f32>().unwrap();

    let area = h * w;
    let ch0 = &data[0..area];
    let ch1 = &data[area..2 * area];
    let ch2 = &data[2 * area..3 * area];

    let mut img = image::RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let r = ((ch0[idx] + 1.0) * 127.5).clamp(0.0, 255.0).round() as u8;
            let g = ((ch1[idx] + 1.0) * 127.5).clamp(0.0, 255.0).round() as u8;
            let b = ((ch2[idx] + 1.0) * 127.5).clamp(0.0, 255.0).round() as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    img.save(path).with_context(|| format!("failed to save image to {:?}", path))?;
    Ok(())
}

/// Saves a batch of [B, 3, 64, 64] images to `dir` named `{prefix}_{idx}.png`.
pub fn save_batch_as_pngs<B: Backend>(batch: Tensor<B, 4>, dir: &Path, prefix: &str) -> Result<Vec<PathBuf>> {
    let [b, _, _, _] = batch.dims();
    let mut saved_paths = Vec::with_capacity(b);
    for i in 0..b {
        let img_tensor = batch.clone().slice([i..i + 1]).squeeze::<3>();
        let filename = format!("{}_{:03}.png", prefix, i);
        let path = dir.join(filename);
        save_tensor_as_png(img_tensor, &path)?;
        saved_paths.push(path);
    }
    Ok(saved_paths)
}
