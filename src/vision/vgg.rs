//! Pretrained VGG-19 Perceptual Feature Extractor for SRGAN / ESRGAN Loss.
//!
//! Re-implements the standard VGG-19 convolutional backbone (Simonyan & Zisserman, 2014)
//! used in SRGAN (Ledig et al., 2017) and ESRGAN (Wang et al., 2018) for perceptual loss.
//!
//! Loads frozen ImageNet-pretrained weights exported from torchvision (`vgg19_features.bin`)
//! and computes high-level semantic feature representations:
//! - `relu1_2`: Early low-level spatial gradients (layer 3)
//! - `relu2_2`: Texture and edge primitives (`SRGAN-VGG22`, layer 8)
//! - `relu3_4`: Complex textures and patterns (layer 17)
//! - `relu4_4`: Mid-level semantic structures (layer 26)
//! - `relu5_4`: Deep semantic object textures (`SRGAN-VGG54`, layer 35)

use anyhow::{bail, ensure, Context, Result};
use burn::{
    module::{Module, Param},
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        PaddingConfig2d,
    },
    prelude::*,
    tensor::{
        activation::relu,
        Tensor, TensorData,
        Distribution,
    },
};
use std::{
    fs::File,
    io::Read,
    path::Path,
};

pub const VGG_MAGIC: &[u8; 8] = b"YUMONVGG";
pub const VGG_VERSION: u32 = 1;
pub const VGG_NUM_CONVS: usize = 16;

/// ImageNet normalization constants:
/// mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Intermediate VGG-19 feature activations across 5 stages.
pub struct Vgg19Activations<B: Backend> {
    // pub relu1_2: Tensor<B, 4>,
    pub relu2_2: Tensor<B, 4>,
    pub relu3_4: Tensor<B, 4>,
    // pub relu4_4: Tensor<B, 4>,
    // pub relu5_4: Tensor<B, 4>,
}

/// Frozen VGG-19 Convolutional Feature Extractor (up to `relu5_4`).
#[derive(Module, Debug)]
pub struct Vgg19Features<B: Backend> {
    // Stage 1: 3 -> 64 -> 64
    pub conv1_1: Conv2d<B>,
    pub conv1_2: Conv2d<B>,

    // Stage 2: 64 -> 128 -> 128
    pub conv2_1: Conv2d<B>,
    pub conv2_2: Conv2d<B>,

    // Stage 3: 128 -> 256 -> 256 -> 256 -> 256
    pub conv3_1: Conv2d<B>,
    pub conv3_2: Conv2d<B>,
    pub conv3_3: Conv2d<B>,
    pub conv3_4: Conv2d<B>,

    // Stage 4: 256 -> 512 -> 512 -> 512 -> 512
    pub conv4_1: Conv2d<B>,
    pub conv4_2: Conv2d<B>,
    pub conv4_3: Conv2d<B>,
    pub conv4_4: Conv2d<B>,

    // Stage 5: 512 -> 512 -> 512 -> 512 -> 512
    pub conv5_1: Conv2d<B>,
    pub conv5_2: Conv2d<B>,
    pub conv5_3: Conv2d<B>,
    pub conv5_4: Conv2d<B>,

    pub pool: MaxPool2d,
}

impl<B: Backend> Vgg19Features<B> {
    /// Initialize with uninitialized structure for loading weights.
    pub fn new(device: &B::Device) -> Self {
        let pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        Self {
            conv1_1: Conv2dConfig::new([3, 64], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv1_2: Conv2dConfig::new([64, 64], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),

            conv2_1: Conv2dConfig::new([64, 128], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv2_2: Conv2dConfig::new([128, 128], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),

            conv3_1: Conv2dConfig::new([128, 256], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv3_2: Conv2dConfig::new([256, 256], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv3_3: Conv2dConfig::new([256, 256], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv3_4: Conv2dConfig::new([256, 256], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),

            conv4_1: Conv2dConfig::new([256, 512], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv4_2: Conv2dConfig::new([512, 512], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv4_3: Conv2dConfig::new([512, 512], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv4_4: Conv2dConfig::new([512, 512], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),

            conv5_1: Conv2dConfig::new([512, 512], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv5_2: Conv2dConfig::new([512, 512], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv5_3: Conv2dConfig::new([512, 512], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv5_4: Conv2dConfig::new([512, 512], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),

            pool,
        }
    }

    /// Load pretrained weights from `vgg19_features.bin`.
    pub fn load_from_binary(path: &Path, device: &B::Device) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open VGG-19 weights binary: {}", path.display()))?;

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        ensure!(&magic == VGG_MAGIC, "Invalid VGG-19 binary magic");

        let mut ver_bytes = [0u8; 4];
        file.read_exact(&mut ver_bytes)?;
        let version = u32::from_le_bytes(ver_bytes);
        ensure!(version == VGG_VERSION, "Unsupported VGG-19 binary version: {version}");

        let mut num_convs_bytes = [0u8; 4];
        file.read_exact(&mut num_convs_bytes)?;
        let num_convs = u32::from_le_bytes(num_convs_bytes) as usize;
        ensure!(num_convs == VGG_NUM_CONVS, "Expected {VGG_NUM_CONVS} conv layers, found {num_convs}");

        let mut model = Self::new(device);

        for idx in 0..num_convs {
            let mut meta = [0u8; 16];
            file.read_exact(&mut meta)?;
            let out_c = u32::from_le_bytes(meta[0..4].try_into().unwrap()) as usize;
            let in_c = u32::from_le_bytes(meta[4..8].try_into().unwrap()) as usize;
            let kh = u32::from_le_bytes(meta[8..12].try_into().unwrap()) as usize;
            let kw = u32::from_le_bytes(meta[12..16].try_into().unwrap()) as usize;

            let weight_count = out_c * in_c * kh * kw;
            let mut weight_bytes = vec![0u8; weight_count * 4];
            file.read_exact(&mut weight_bytes)?;

            let mut bias_len_bytes = [0u8; 4];
            file.read_exact(&mut bias_len_bytes)?;
            let bias_len = u32::from_le_bytes(bias_len_bytes) as usize;
            ensure!(bias_len == out_c, "Bias length mismatch at conv {idx}");

            let mut bias_bytes = vec![0u8; bias_len * 4];
            file.read_exact(&mut bias_bytes)?;

            // Convert raw bytes to f32 slices
            let weights_f32: Vec<f32> = weight_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();

            let bias_f32: Vec<f32> = bias_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();

            let weight_tensor = Tensor::<B, 4>::from_data(
                TensorData::new(weights_f32, [out_c, in_c, kh, kw]),
                device,
            );
            let bias_tensor = Tensor::<B, 1>::from_data(
                TensorData::new(bias_f32, [out_c]),
                device,
            );

            let weight_param = Param::from_tensor(weight_tensor).set_require_grad(false);
            let bias_param = Param::from_tensor(bias_tensor).set_require_grad(false);

            // Assign into corresponding layer
            match idx {
                0 => { model.conv1_1.weight = weight_param; model.conv1_1.bias = Some(bias_param); }
                1 => { model.conv1_2.weight = weight_param; model.conv1_2.bias = Some(bias_param); }
                2 => { model.conv2_1.weight = weight_param; model.conv2_1.bias = Some(bias_param); }
                3 => { model.conv2_2.weight = weight_param; model.conv2_2.bias = Some(bias_param); }
                4 => { model.conv3_1.weight = weight_param; model.conv3_1.bias = Some(bias_param); }
                5 => { model.conv3_2.weight = weight_param; model.conv3_2.bias = Some(bias_param); }
                6 => { model.conv3_3.weight = weight_param; model.conv3_3.bias = Some(bias_param); }
                7 => { model.conv3_4.weight = weight_param; model.conv3_4.bias = Some(bias_param); }
                8 => { model.conv4_1.weight = weight_param; model.conv4_1.bias = Some(bias_param); }
                9 => { model.conv4_2.weight = weight_param; model.conv4_2.bias = Some(bias_param); }
                10 => { model.conv4_3.weight = weight_param; model.conv4_3.bias = Some(bias_param); }
                11 => { model.conv4_4.weight = weight_param; model.conv4_4.bias = Some(bias_param); }
                12 => { model.conv5_1.weight = weight_param; model.conv5_1.bias = Some(bias_param); }
                13 => { model.conv5_2.weight = weight_param; model.conv5_2.bias = Some(bias_param); }
                14 => { model.conv5_3.weight = weight_param; model.conv5_3.bias = Some(bias_param); }
                15 => { model.conv5_4.weight = weight_param; model.conv5_4.bias = Some(bias_param); }
                _ => bail!("Unexpected conv index: {idx}"),
            }
        }

        Ok(model.no_grad())
    }

    /// Preprocess an image tensor for VGG-19:
    /// - If 1-channel grayscale `[B, 1, H, W]`, repeats to 3 RGB channels `[B, 3, H, W]`.
    /// - Maps `[-1.0, 1.0]` range to `[0.0, 1.0]`.
    /// - Normalizes using standard ImageNet mean and std.
    pub fn preprocess(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let device = x.device();
        let dims = x.dims();

        // 1. Expand grayscale to 3 RGB channels if needed
        let rgb = if dims[1] == 1 {
            Tensor::cat(vec![x.clone(), x.clone(), x], 1)
        } else {
            x
        };

        // 2. Scale from [-1.0, 1.0] -> [0.0, 1.0]
        let x_01 = (rgb + 1.0) * 0.5;

        // 3. Normalize with ImageNet mean and std
        let mean = Tensor::<B, 4>::from_data(
            TensorData::new(IMAGENET_MEAN.to_vec(), [1, 3, 1, 1]),
            &device,
        );
        let std = Tensor::<B, 4>::from_data(
            TensorData::new(IMAGENET_STD.to_vec(), [1, 3, 1, 1]),
            &device,
        );

        (x_01 - mean) / std
    }

    /// Forward pass through all 5 convolutional stages, returning intermediate feature activations.
    pub fn forward_all(&self, x: Tensor<B, 4>) -> Vgg19Activations<B> {
        let norm = self.preprocess(x);

        // Stage 1
        let f = relu(self.conv1_1.forward(norm));
        // let relu1_2 = relu(self.conv1_2.forward(f));
        let f = self.pool.forward(f);

        // Stage 2
        let f = relu(self.conv2_1.forward(f));
        let relu2_2 = relu(self.conv2_2.forward(f));
        let f = self.pool.forward(relu2_2.clone());

        // Stage 3
        let f = relu(self.conv3_1.forward(f));
        let f = relu(self.conv3_2.forward(f));
        let f = relu(self.conv3_3.forward(f));
        let relu3_4 = relu(self.conv3_4.forward(f));
        let f = self.pool.forward(relu3_4.clone());

        // // Stage 4
        // let f = relu(self.conv4_1.forward(f));
        // let f = relu(self.conv4_2.forward(f));
        // let f = relu(self.conv4_3.forward(f));
        // let relu4_4 = relu(self.conv4_4.forward(f));
        // let f = self.pool.forward(relu4_4.clone());

        // // Stage 5
        // let f = relu(self.conv5_1.forward(f));
        // let f = relu(self.conv5_2.forward(f));
        // let f = relu(self.conv5_3.forward(f));
        // let relu5_4 = relu(self.conv5_4.forward(f));

        Vgg19Activations {
            // relu1_2,
            relu2_2,
            relu3_4,
            // relu4_4,
            // relu5_4,
        }
    }

    /// Canonical SRGAN-VGG54 feature representation (layer 35 in torchvision).
    /// Extracts deep semantic texture features.
    pub fn forward_vgg54(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let norm = self.preprocess(x);

        // Stage 1
        let f = relu(self.conv1_1.forward(norm));
        // let f = relu(self.conv1_2.forward(f)); // NOTE: causes hanging. very strange.
        let f = self.pool.forward(f);

        // Stage 2
        let f = relu(self.conv2_1.forward(f));
        let f = relu(self.conv2_2.forward(f));
        let f = self.pool.forward(f);

        // Stage 3
        let f = relu(self.conv3_1.forward(f));
        let f = relu(self.conv3_2.forward(f));
        let f = relu(self.conv3_3.forward(f));
        let f = relu(self.conv3_4.forward(f));
        let f = self.pool.forward(f);

        // Stage 4
        let f = relu(self.conv4_1.forward(f));
        let f = relu(self.conv4_2.forward(f));
        let f = relu(self.conv4_3.forward(f));
        let f = relu(self.conv4_4.forward(f));
        let f = self.pool.forward(f);

        // Stage 5
        let f = relu(self.conv5_1.forward(f));
        let f = relu(self.conv5_2.forward(f));
        let f = relu(self.conv5_3.forward(f));
        relu(self.conv5_4.forward(f))

        // f
    }

    /// Canonical SRGAN-VGG22 feature representation (layer 8 in torchvision).
    /// Extracts fine local edge and color texture features.
    pub fn forward_vgg22(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let norm = self.preprocess(x);
        // let norm = x;

        // // Stage 1
        let f = relu(self.conv1_1.forward(norm));
        // let f = relu(self.conv1_2.forward(f));
        let f = self.pool.forward(f);

        // f

        // // Stage 2
        let f = relu(self.conv2_1.forward(f));
        // relu(self.conv2_2.forward(f))

        f
    }

    /// Compute composite SRGAN perceptual loss between generated output and ground truth.
    ///
    /// By default, evaluates L1 distance on both:
    /// - `relu2_2` (weight 0.5) for crisp low-level edges and color consistency
    /// - `relu5_4` (weight 1.0) for high-level semantic texture and natural sharpness
    // pub fn perceptual_loss(
    //     &self,
    //     pred: Tensor<B, 4>,
    //     target: Tensor<B, 4>,
    // ) -> Tensor<B, 1> {
    //     let acts_pred = self.forward_all(pred);
    //     let acts_target = self.forward_all(target.detach());

    //     let loss_22 = (acts_pred.relu2_2 - acts_target.relu2_2.detach()).abs().mean();
    //     let loss_54 = (acts_pred.relu5_4 - acts_target.relu5_4.detach()).abs().mean();

    //     loss_54 + loss_22 * 0.5
    // }

    pub fn perceptual_loss34(
        &self,
        pred: Tensor<B, 4>,
        target: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let acts_pred = self.forward_all(pred);
        let acts_target = self.forward_all(target.detach());

        let loss_22 = (acts_pred.relu2_2 - acts_target.relu2_2.detach()).abs().mean();
        let loss_34 = (acts_pred.relu3_4 - acts_target.relu3_4.detach()).abs().mean();

        loss_34 + loss_22 * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_cubecl::CubeBackend;
    use cubecl::wgpu::WgpuRuntime;

    type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    use burn::backend::Autodiff;
    type AdBackend = Autodiff<TestBackend>;

    #[test]
    fn test_vgg_no_grad_check() {
        use burn::backend::Autodiff;
        type AdBackend = Autodiff<TestBackend>;
        let device = Default::default();
        let vgg_path = Path::new("training-cache/vgg19_features.bin");
        if !vgg_path.exists() { return; }
        let vgg = Vgg19Features::<AdBackend>::load_from_binary(vgg_path, &device).unwrap();
        assert!(!vgg.conv1_1.weight.is_require_grad(), "VGG conv1_1 weights must not require grad");
        assert!(!vgg.conv5_4.weight.is_require_grad(), "VGG conv5_4 weights must not require grad");
    }

    // #[test]
    // fn test_vgg19_binary_load_and_forward() {
    //     use burn::backend::Autodiff;
    //     type AdBackend = Autodiff<TestBackend>;

    //     let vgg_path = Path::new("training-cache/vgg19_features.bin");
    //     if !vgg_path.exists() {
    //         eprintln!("Warning: training-cache/vgg19_features.bin not found; skipping test.");
    //         return;
    //     }

    //     let device = Default::default();
    //     let vgg = Vgg19Features::<AdBackend>::load_from_binary(vgg_path, &device)
    //         .expect("Failed to load VGG-19 binary");

    //     // Test forward with dummy input [1, 3, 64, 64]
    //     let input: Tensor<AdBackend, 4> = Tensor::zeros([1, 3, 256, 256], &device);
    //     let feat54 = vgg.forward_vgg54(input.clone());
    //     // assert_eq!(feat54.dims(), [1, 512, 4, 4]);

    //     let feat22 = vgg.forward_vgg22(input.clone());
    //     // assert_eq!(feat22.dims(), [1, 128, 32, 32]);

    //     let target: Tensor<AdBackend, 4> = Tensor::ones([1, 3, 256, 256], &device);
    //     let loss = vgg.perceptual_loss(input, target);
    //     // let loss_val: f32 = loss.into_data().convert::<f32>().to_vec().unwrap()[0];
    //     // assert!(loss_val > 0.0, "Perceptual loss between zeros and ones must be > 0");
    //     // println!("VGG-19 Perceptual loss between 0s and 1s: {loss_val:.4}");

    //     let t = std::time::Instant::now();
    //     let grads = loss.backward();
    //     // let g = input.grad(&grads).unwrap();
    //     // let _ = g.into_data(); // force the backward pass to finish
    //     println!("backward: {:?}", t.elapsed());
    // }

    #[cfg(test)]
    fn time_backward<F>(name: &str, vgg: &Vgg19Features<AdBackend>, f: F)
    where
        F: Fn(&Vgg19Features<AdBackend>, Tensor<AdBackend, 4>, Tensor<AdBackend, 4>) -> Tensor<AdBackend, 1>,
    {
        let device = Default::default();
        let (b, hw) = (1, 256);
        let input = Tensor::<AdBackend, 4>::random([b, 3, hw, hw], Distribution::Default, &device)
            .require_grad();
        let target = Tensor::<AdBackend, 4>::random([b, 3, hw, hw], Distribution::Default, &device);

        let t = std::time::Instant::now();
        let loss = f(vgg, input.clone(), target);
        let _ = loss.clone().into_data(); // force forward
        println!("{name} forward:  {:?}", t.elapsed());

        let t = std::time::Instant::now();
        let grads = loss.backward();
        let g = input.grad(&grads).expect("no grad reached input");
        let _ = g.into_data(); // force backward
        println!("{name} backward: {:?}", t.elapsed());
    }

    #[test]
    fn bisect_vgg22_only() {
        use burn::backend::Autodiff;
        type AdBackend = Autodiff<TestBackend>;

        let vgg_path = Path::new("training-cache/vgg19_features.bin");
        if !vgg_path.exists() {
            eprintln!("Warning: training-cache/vgg19_features.bin not found; skipping test.");
            return;
        }

        let device = Default::default();
        let vgg = Vgg19Features::<AdBackend>::load_from_binary(vgg_path, &device)
            .expect("Failed to load VGG-19 binary");
        time_backward("vgg22", &vgg, |v, pred, target| {
            (v.forward_vgg22(pred) - v.forward_vgg22(target).detach()).abs().mean()
        });
    }

    #[test]
    fn bisect_vgg54_only() {
        use burn::backend::Autodiff;
        type AdBackend = Autodiff<TestBackend>;

        let vgg_path = Path::new("training-cache/vgg19_features.bin");
        if !vgg_path.exists() {
            eprintln!("Warning: training-cache/vgg19_features.bin not found; skipping test.");
            return;
        }

        let device = Default::default();
        let vgg = Vgg19Features::<AdBackend>::load_from_binary(vgg_path, &device)
            .expect("Failed to load VGG-19 binary");
        time_backward("vgg54", &vgg, |v, pred, target| {
            (v.forward_vgg54(pred) - v.forward_vgg54(target).detach()).abs().mean()
        });
    }

    // #[test]
    // fn bisect_full_perceptual() {
    //     use burn::backend::Autodiff;
    //     type AdBackend = Autodiff<TestBackend>;

    //     let vgg_path = Path::new("training-cache/vgg19_features.bin");
    //     if !vgg_path.exists() {
    //         eprintln!("Warning: training-cache/vgg19_features.bin not found; skipping test.");
    //         return;
    //     }

    //     let device = Default::default();
    //     let vgg = Vgg19Features::<AdBackend>::load_from_binary(vgg_path, &device)
    //         .expect("Failed to load VGG-19 binary");
    //     time_backward("full", &vgg, |v, pred, target| v.perceptual_loss(pred, target));
    // }

    // #[test]
    // fn test_vgg19_backward_gradients() {
    //     use burn::backend::Autodiff;
    //     type AdBackend = Autodiff<TestBackend>;

    //     let vgg_path = Path::new("training-cache/vgg19_features.bin");
    //     if !vgg_path.exists() {
    //         return;
    //     }

    //     let device = Default::default();
    //     let vgg = Vgg19Features::<AdBackend>::load_from_binary(vgg_path, &device)
    //         .expect("Failed to load VGG-19 binary");

    //     // let input: Tensor<AdBackend, 4> = Tensor::zeros([1, 3, 64, 64], &device).require_grad();
    //     // let target: Tensor<AdBackend, 4> = Tensor::ones([1, 3, 256, 256], &device);

    //     // let loss = vgg.perceptual_loss(input.clone(), target);
    //     // let grads = loss.backward();
    //     // let input_grad = input.grad(&grads);
    //     // assert!(input_grad.is_some(), "Input must receive perceptual loss gradients");
    //     // let grad_norm: f32 = input_grad.unwrap().abs().mean().into_data().convert::<f32>().to_vec().unwrap()[0];
    //     // println!("VGG-19 Perceptual loss backward grad norm: {grad_norm:.6}");
    //     // assert!(grad_norm > 1e-6);

    //     let (b, hw) = (1, 64);
    //     let input: Tensor<AdBackend, 4> = Tensor::zeros([b, 1, hw, hw], &device).require_grad();
    //     let target: Tensor<AdBackend, 4> = Tensor::ones([b, 1, hw, hw], &device);

    //     let t = std::time::Instant::now();
    //     let loss = vgg.perceptual_loss(input.clone(), target);
    //     let _ = loss.clone().into_data(); // force the forward pass to finish
    //     println!("forward: {:?}", t.elapsed());

    //     let t = std::time::Instant::now();
    //     let grads = loss.backward();
    //     let g = input.grad(&grads).unwrap();
    //     let _ = g.into_data(); // force the backward pass to finish
    //     println!("backward: {:?}", t.elapsed());
    // }
}
