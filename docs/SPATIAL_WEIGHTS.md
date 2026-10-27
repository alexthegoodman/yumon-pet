# User Story: Multi-Scale Spatial & Frequency Loss for Micro-Portrait Generation

Here is a comprehensive Agile User Story and Technical Specification tailored for your Burn pipeline and multi-resolution curriculum.

---

**Title:** Implement Resolution-Adaptive Frequency & ROI-Weighted Losses in Burn for Tokenizer and MoE AR Training

**As a** Machine Learning Engineer building an edge-targeted MoE M-RoPE autoregressive image generator in Burn,

**I want** to replace our uniform reconstruction and cross-entropy losses with resolution-adaptive 2D frequency losses and spatially weighted feature masks (starting at 16x16 up to 64x64),

**So that** our discrete tokenizer and small MoE transformer prioritize high-frequency textures and crisp facial/eye structures (like single-pixel pupils and catchlights) without increasing inference latency or parameter count.

---

### Context & Engineering Constraints

* **Framework:** Rust (`burn` crate, targeting backend-agnostic `Autodiff<B>` execution).
* **Resolution Curriculum:** Progressive training from `16x16` $\rightarrow$ `32x32` $\rightarrow$ `64x64`.
* **Micro-Patch Dynamics:** At `16x16`, an entire eye occupies 1–4 pixels. Standard pretrained face networks (like ArcFace, which expects 112x112) cannot be run directly on a 2x2 pixel eye crop without extreme interpolation blur. At 16x16, "eye quality" is defined by **local contrast, exact pixel placement, and high-frequency transitions**.
* **Burn FFT Workaround:** Because patches are tiny ($P \in \{2, 3, 4, 8\}$), a 2D DFT can be formulated directly in pure Burn tensor math using `conv2d` with fixed basis kernels (Haar wavelets / DCT / DFT bases) or matrix multiplication $F = W \cdot X \cdot W^T$, ensuring zero external C/CUDA wrapper dependencies and seamless `wgpu`/`tch`/` candle` support.

---

### Resolution Scaling Matrix

| Image Resolution | Token Patch Size ($P \times P$) | Token Grid Size | Primary Eye / Face Signal | Frequency / Texture Mechanism |
| --- | --- | --- | --- | --- |
| **16x16** | `2x2` (or `3x3` w/ pad) | `8x8` | Spatially weighted L1 + Local Contrast (Sobel/Laplacian) on 2x2 eye coords | 2x2 Haar Wavelet / DCT basis via `conv2d` |
| **32x32** | `4x4` | `8x8` or `16x16` | Spatially weighted L1 + Bicubic upscaled ROI perceptual features | 4x4 Patch 2D DFT via precomputed basis `matmul` |
| **64x64** | `4x4` or `8x8` | `16x16` | Full InsightTok ROI-Aligned Feature Loss + Spatial CE Mask | 8x8 Patch 2D DFT with Focal Amplitude Weighting |

---

### Acceptance Criteria (AC)

#### AC 1: Offline Dataset Pre-Processing & Spatial Mask Generation

* [ ] **Landmark-to-Grid Projection:** The data pipeline ingests high-res source images, detects facial/eye landmarks (prior to downsampling to 16x16 / 32x32 / 64x64), and projects the coordinates onto the target low-res pixel grid and token grid. *(Crucial: Detecting eyes on a 16x16 image directly will fail; landmarks must be extracted from the source image before downsampling).*
* [ ] **Normalized Area Weighting:** Generate a spatial weight tensor `Mask<B, 4>` of shape `[B, 1, H, W]` where eye/face bounding boxes receive inverse-area normalized weights (e.g., `w_eye = 10.0`, `w_face = 3.0`, `w_bg = 1.0`) so small faces produce consistent gradient magnitudes.

Note: always save visual image samples of any masks or processed data for confirmation.

#### AC 2: Backend-Agnostic Micro-Patch Frequency Loss in Burn

* [ ] **Matrix-Based 2D DFT/DCT Module:** Implement a `PatchFrequencyLoss<B: Backend>` struct in Burn that takes dynamic patch sizes (`2x2`, `3x3`, `4x4`, `8x8`).
* [ ] **Pure Tensor Execution:** Instead of relying on backend-specific FFT bindings, initialize non-trainable `Param<Tensor<B, 4>>` convolution filters representing the Real and Imaginary 2D DFT (or DCT/Haar) basis functions for size $P \times P$ with `stride = P`.
* [ ] **Focal Amplitude Penalty:** Compute the log-magnitude spectrum in `f32` precision, apply the detached focal weight $\vert{}A_{\text{real}} - A_{\text{pred}}\vert{}^\alpha$, and mask out the DC (`[0, 0]`) component so L1 handles flat color and the frequency loss exclusively drives crisp textures.

#### AC 3: Spatially Weighted VQ-Tokenizer Training

* [ ] **Pixel & Gradient Alignment:** Multiply pixel-wise L1 reconstruction loss and 3x3 Sobel/Laplacian gradient differences by the `eye_face_mask`.
* [ ] **Codebook Health Metrics:** Log codebook perplexity and utilization per epoch. Ensure high-frequency loss weighting increases codebook allocation to eye/texture patches without dropping overall codebook utilization below 80%.

#### AC 4: Autoregressive MoE & M-RoPE Integration

* [ ] **Token-Space Cross-Entropy Weighting:** Downsample `eye_face_mask` via `avg_pool2d` or `max_pool2d` to match the AR token sequence length `[B, SeqLen]`. Multiply per-token Cross-Entropy loss by the pooled mask before reduction.
* [ ] **MoE Router Telemetry:** Track expert routing distribution for tokens where `mask_weight > 1.0` to verify that specific experts naturally specialize in high-frequency / facial tokens.

---

### Technical Implementation Tasks (Sub-Tickets)

1. **`[Data]` Source-to-Micro Downsampler:** Write a Rust pre-processing CLI (or Python data-prep script outputting `.safetensors` / Burn `BinFileRecorder` format) that extracts eye bounding boxes at native resolution and saves `(image_16x16, image_32x32, image_64x64, mask_16x16, mask_32x32, mask_64x64)`.
2. **`[Tokenizer]` `MicroFrequencyLoss` Burn Module:** Implement the `conv2d` DFT/DCT basis generator for $P \in \{2, 3, 4, 8\}$ in Burn. Add unit tests verifying that a blurry 2x2 eye patch yields $>5\times$ higher loss than a sharp 2x2 eye patch with identical average luminance.
3. **`[AR-MoE]` Weighted CE & Router Hook:** Update the AR forward pass to accept `spatial_weights: Tensor<B, 2>` and apply it to the unreduced `CrossEntropyLoss` output.

---

### Definition of Done (DoD)

* All new loss modules compile and pass gradient-check tests on Burn's `NdArray` (CPU) and `Wgpu` / `Tch` (GPU) backends.
* At `16x16` resolution, decoded VQ reconstructions preserve single-pixel eye contrast (pupil/sclera separation) rather than averaging the eye into a flat 2x2 skin-colored square.
* Zero regression in inference latency or memory footprint on target low-performance hardware (all changes are strictly confined to training losses and data loading).