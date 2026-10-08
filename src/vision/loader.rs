/// Image loading utility for inference-time use.
/// Loads any JPEG/PNG, resizes to IMG_SIZE, normalizes to [-1, 1].

use anyhow::Result;
use burn::{prelude::*, tensor::TensorData};
#[cfg(not(target_arch = "wasm32"))]
use image::imageops::FilterType;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::vision::IMG_SIZE;

/// Load a user-provided image file and return a [1, 3, IMG_SIZE, IMG_SIZE] tensor.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_image_tensor<B: Backend>(path: &str, device: &B::Device) -> Result<Tensor<B, 4>> {
    let img = image::open(path)?
        .resize_exact(IMG_SIZE as u32, IMG_SIZE as u32, FilterType::Lanczos3)
        .to_rgb8();

    let mut flat = vec![0.0f32; 3 * IMG_SIZE * IMG_SIZE];
    let plane_size = IMG_SIZE * IMG_SIZE;
    flat.par_chunks_mut(plane_size)
        .enumerate()
        .for_each(|(channel, plane)| {
            for (value, pixel) in plane.iter_mut().zip(img.pixels()) {
                *value = pixel.0[channel] as f32 / 255.0 * 2.0 - 1.0;
            }
        });

    let t = Tensor::<B, 4>::from_floats(TensorData::new(flat, [1, 3, IMG_SIZE, IMG_SIZE]), device);
    Ok(t)
}
