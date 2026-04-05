/// Image loading utility for inference-time use.
/// Loads any JPEG/PNG, resizes to IMG_SIZE, normalizes to [-1, 1].

use anyhow::Result;
use burn::{prelude::*, tensor::TensorData};
#[cfg(target_os = "windows")]
use image::imageops::FilterType;

use crate::vision::IMG_SIZE;

/// Load a user-provided image file and return a [1, 3, IMG_SIZE, IMG_SIZE] tensor.
#[cfg(target_os = "windows")]
pub fn load_image_tensor<B: Backend>(path: &str, device: &B::Device) -> Result<Tensor<B, 4>> {
    let img = image::open(path)?
        .resize_exact(IMG_SIZE as u32, IMG_SIZE as u32, FilterType::Lanczos3)
        .to_rgb8();

    let mut flat = vec![0.0f32; 3 * IMG_SIZE * IMG_SIZE];
    for (i, pixel) in img.pixels().enumerate() {
        let [r, g, b] = pixel.0;
        flat[0 * IMG_SIZE * IMG_SIZE + i] = r as f32 / 255.0 * 2.0 - 1.0;
        flat[1 * IMG_SIZE * IMG_SIZE + i] = g as f32 / 255.0 * 2.0 - 1.0;
        flat[2 * IMG_SIZE * IMG_SIZE + i] = b as f32 / 255.0 * 2.0 - 1.0;
    }

    let t = Tensor::<B, 4>::from_floats(
        TensorData::new(flat, [1, 3, IMG_SIZE, IMG_SIZE]),
        device,
    );
    Ok(t)
}
