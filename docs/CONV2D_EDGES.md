Integrating **Conv2D** into a multi-modal, MoE + FlashAttention autoregressive pixel generator will significantly improve edge continuity, structural alignment, and high-frequency detail—provided you adapt the convolution to your architecture's causality rules.

### Why Pure Transformers Struggle with Pixel Edges

When you flatten a 2D canvas into a 1D sequence for FlashAttention (typically in raster-scan order, row by row), you break the natural 2D spatial neighborhood.

* **The Row-Wrap Discontinuity:** A pixel at coordinate $(x, y)$ is immediately preceded by $(x-1, y)$ in the 1D sequence, making horizontal edges easy to track. However, its vertical neighbor $(x, y-1)$ is $W$ tokens back in the sequence (where $W$ is the image width).
* **Attention Dilution:** Standard 1D or even 2D Rotary Position Embeddings (RoPE) give the Transformer a hint about spatial proximity, but Softmax attention naturally smooths out sharp local gradients over long contexts—especially when competing with multi-modal prompt tokens.
* **MoE Routing Jitter:** In a Mixture-of-Experts model, adjacent pixels along an edge might get routed to different experts. If Expert A handles $(x, y-1)$ and Expert B handles $(x, y)$, tiny discrepancies in their weights can cause "staircasing" or jagged artifacts along diagonal and vertical edges.

A `Conv2D` layer hardcodes **2D translation equivariance** and forces the model to compute local spatial derivatives (which is literally how classical edge detectors like Sobel or Laplacian filters work).

---

### 3 Ways to Integrate Conv2D into Your Architecture

Depending on how your canvas and pixel generation are set up, there are three distinct places to put `Conv2D`, ranked from easiest to most impactful:

| Integration Point | Type of Conv2D | Computational Cost | Primary Benefit |
| --- | --- | --- | --- |
| **1. Canvas Scene Encoder** | Standard `Conv2D` (Bidirectional) | Very Low (runs once per canvas update) | Extracts sharp multi-scale edges from the input scene so cross-attention has crisp spatial anchors. |
| **2. Pre-MoE Token Mixer** | **Masked (Causal) `Conv2D**` | Low (Depthwise $3\times3$ or $5\times5$) | Fixes MoE routing jitter and bridges the vertical $(x, y-1)$ row gap before attention. |
| **3. Shared "Spatial Expert" inside MoE** | **Masked Depthwise `Conv2D**` | Negligible | Gives the router a dedicated local-edge processor alongside standard MLP experts. |

---

#### 1. On the Input Canvas Scene (Standard Conv2D)

Because your model accepts an existing canvas scene as condition/input, those tokens do **not** need to be causally masked against themselves.

* **How it works:** Before tokenizing or projecting the input canvas into the Transformer backbone, pass it through a lightweight ConvNeXt block or a multi-scale residual `Conv2D` stem (e.g., $3\times3$ depthwise separable convolutions).
* **Why it helps edges:** If you rely purely on patch embeddings (like a standard ViT) for the canvas input, patch boundaries destroy sub-patch edge continuity. Adding overlapping `Conv2D` layers before the FlashAttention layers ensures every canvas token already carries explicit local gradient (edge orientation) features from its 8-pixel neighborhood.

#### 2. In the Autoregressive Backbone (Masked / Causal Conv2D)

If you apply a standard `Conv2D` directly to the autoregressive pixel stream during training, **the model will cheat**. A standard $3\times3$ kernel centered at $(x, y)$ looks at $(x+1, y)$ and $(x, y+1)$—which are future pixels. During inference, those pixels don't exist yet, and generation will collapse.

To put `Conv2D` inside the autoregressive loop, you must use a **Causal (PixelCNN-style) Masked Conv2D**:

* For a $3\times3$ kernel centered at $(x, y)$, zero out the weights for the bottom row $(y+1)$ and the right neighbor $(x+1, y)$.
* This leaves 4 active neighbors: top-left $(x-1, y-1)$, top $(x, y-1)$, top-right $(x+1, y-1)$, and left $(x-1, y)$, plus the current hidden state $(x, y)$.
* **Implementation trick (Shift-and-Conv):** Instead of multiplying by a binary mask at every step, you can implement a causal $3\times3$ convolution by combining a $1 \times 3$ convolution on the row above $(y-1)$ with a $1 \times 2$ causal convolution on the current row $(y)$.

#### 3. Inside the MoE Layer (Depthwise Conv2D Shared Expert)

In modern MoE architectures (like DeepSeek-MoE), using a **always-active shared expert** alongside routed experts stabilizes training.

* You can make the shared expert (or a residual branch parallel to the MoE router) a **Masked Depthwise Separable `Conv2D**` applied only to the generated image tokens (skipping text/prompt tokens).
* Because depthwise convolutions operate per-channel with tiny parameter counts, they add almost zero FLOPs, prevent expert boundary artifacts along edges, and let the routed MLP experts focus on complex semantic/multi-modal mapping while the `Conv2D` handles crisp local geometry.

---

### How to Handle the Multi-Modal Sequence Split

Because your sequence mixes 1D prompt/status tokens with 2D pixel tokens:

1. **In the Attention Layer:** Keep all tokens (prompt + canvas + generated pixels) flattened in 1D so **FlashAttention** runs at full hardware efficiency across the entire multi-modal context.
2. **In the Conv2D Branch:** Slice out only the 2D spatial tokens, reshape them back to $(B, C, H, W)$, apply the Masked Depthwise `Conv2D`, flatten them back to $(B, H \times W, C)$, and add them back via a residual connection before the MoE router or FFN.
3. **Inference KV-Cache:** Just like attention uses a KV-cache during pixel-by-pixel decoding, a $3\times3$ Causal `Conv2D` only requires caching the **last 1 row plus 1 pixel** ($W + 1$ states) in a rolling buffer, keeping single-pixel decoding $O(1)$.
