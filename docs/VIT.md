# User Story: Standalone 64x64 Generative Vision Transformer (DiT)

**Title:** Implement an Independent Pixel-Space Diffusion Vision Transformer for 64x64 Image Generation

**Repository:** `alexthegoodman/yumon-pet`

**Component:** Vision Generation Module (Standalone Model + Shared Core Primitives)

---

### Story

**As the** developer of `yumon-pet`,

**I want to** build and train a standalone, small-scale Vision Transformer (Diffusion Transformer / DiT) within the existing repository that reuses core primitives like Flash Attention,

**So that** I can generate novel 64x64 images from my custom dataset without duplicating low-level attention kernels or introducing heavy external U-Net/VAE dependencies.

---

### Architectural Overview & Scope

This model will live as an independent module alongside the existing `yumon-pet` language model. It will operate directly in pixel space on $64 \times 64$ RGB images using a pure-Transformer isotropic backbone trained via **Rectified Flow (Flow Matching)**.

* **Shared / Reused Components:**
* **Flash Attention:** Parameterized with a `causal: bool` flag so the language model uses `causal: true` and the Vision Transformer uses `causal: false` (full bidirectional self-attention).
* **Normalization & MLP Primitives:** Existing `RMSNorm` / `LayerNorm` and Feed-Forward (`GELU` / `SwiGLU`) implementations.


* **New Vision-Specific Components:**
* **Pure-Linear Patchify & Unpatchify:** Tensor reshape/permute + `Linear(3 * P * P -> d_model)` stem and inverse `Linear(d_model -> 3 * P * P)` output head (configurable patch size $P = 4$ for 256 tokens or $P = 2$ for 1024 tokens).
* **2D Positional Embeddings:** 2D Sinusoidal or 2D Axial RoPE encoding row ($Y$) and column ($X$) grid coordinates.
* **Timestep Embedder:** Continuous sinusoidal frequency encoder + 2-layer `SiLU` MLP mapping $t \in [0, 1]$ to `[B, d_model]`.
* **AdaLN-Zero DiT Block:** Parameterless normalization modulated by a zero-initialized `Linear(d_model -> 6 * d_model)` projection ($\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2$) plus `QK-Norm` prior to Flash Attention.
* **Rectified Flow Trainer & EMA:** Linear interpolation training step ($x_t = (1 - t)x_0 + t z_1$), MSE velocity loss ($\mathcal{L} = \Vert{}v_\theta(x_t, t) - (z_1 - x_0)\Vert{}^2$), Exponential Moving Average (EMA) shadow weights, and an Euler ODE inference sampler.



---

### Implementation Tasks

1. **Refactor Shared Primitives**
* [ ] Update the existing Flash Attention module to accept a `causal: bool` toggle, bypassing diagonal/upper-triangular causal masking when `false`.
* [ ] Expose `RMSNorm` / `LayerNorm` with an option to disable learnable affine parameters (`elementwise_affine = false`) for use in AdaLN-Zero.


2. **Build Vision Tokenization & Positional Encoding**
* [ ] Implement `Patchify`: reshape `[B, 3, 64, 64]` into `[B, (64/P)^2, 3 * P * P]` and project to `[B, N, d_model]`.
* [ ] Implement `Unpatchify`: project `[B, N, d_model]` to `[B, N, 3 * P * P]` and reshape/permute back to `[B, 3, 64, 64]`.
* [ ] Implement 2D Positional Embeddings ($X$ and $Y$ grid coordinates) and `TimestepEmbedder` ($t \in [0, 1] \rightarrow \text{Sinusoidal} \rightarrow \text{Linear} \rightarrow \text{SiLU} \rightarrow \text{Linear}$).


3. **Assemble the DiT Backbone**
* [ ] Implement `DiTBlock` combining AdaLN-Zero modulation, `QK-Norm` (normalizing $Q$ and $K$ per head before attention), bidirectional Flash Attention, and the MLP block.
* [ ] Ensure zero-initialization on the AdaLN-Zero $6 \times d_{\text{model}}$ modulation weights/biases and the final output projection layer so the initial network acts as an identity pass.
* [ ] Wrap blocks into a standalone top-level `VisionDiT` struct with its own hyperparameter config (`Micro` / `Tiny` presets), completely decoupled from the LLM state.


4. **Training & Inference Pipeline**
* [ ] Implement the 64x64 image dataset loader with `[-1.0, 1.0]` pixel normalization.
* [ ] Implement the Rectified Flow training loop with `AdamW` (weight decay excluded on biases/norms/pos-embeds), gradient clipping (`1.0`), and EMA weight tracking ($\beta_{\text{EMA}} = 0.999$).
* [ ] Build a standalone CLI command or binary target to run 30-step Euler sampling from EMA weights and export generated $64 \times 64$ PNGs.



---

### Acceptance Criteria (Definition of Done)

* **Non-Regression on LLM:** Existing `yumon-pet` language model unit tests, training, and inference run identically with `causal: true` on the shared Flash Attention module.
* **Shape & Identity Verification:** A forward pass with a batch of `[B, 3, 64, 64]` tensors and timesteps `[B]` returns a `[B, 3, 64, 64]` velocity tensor; at initialization (step 0), the final output tensor is all zeros due to AdaLN-Zero initialization.
* **Single-Batch Overfit Test:** The model successfully overfits a single batch of 4–8 images from the 64x64 dataset (MSE loss drops steadily and the 30-step Euler sampler reconstructs the training images cleanly).
* **Full Dataset Training & Generation:** Training runs stably without `NaN` attention spikes, saves periodic EMA checkpoints, and outputs coherent $64 \times 64$ PNG samples via its own CLI entry point.

### Notes

- Dataset is a portion of DiffusionDb in the `/data/` folder. See `/data/DiffusionDb/images-64/` and you will find PNG images and JSON manifests.
- This will be a distinct `Architecture` on the `RunConfig`.