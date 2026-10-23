# Multimodal Pixel-Aware Vision & Art Generation Architecture

This document describes how Yumon Pet generates 64×64 pixel art end-to-end from text prompts. It explains the multiscale coarse-to-fine progression, data representation, sequence packing, coordinate querying, and loss calculation.

---

## 1. High-Level Concept

Yumon is a tabletop electronic pet with an autoregressive Mixture-of-Experts (MoE) brain. Rather than relying on a diffusion U-Net or a traditional GAN, Yumon creates art **interactively**:
1. Yumon observes the current canvas through a 2D patch encoder (256 visual tokens).
2. It receives the user's text prompt and a coordinate query $(x, y)$.
3. Its transformer attention heads look at the text prompt, attend across the visual canvas, and predict the exact 8-bit truecolor palette value for that coordinate.
4. The canvas updates, Yumon re-glances at its work, and paints the next region.

```mermaid
flowchart TD
    subgraph Input ["1. Multimodal Input Stream"]
        Canvas["64x64 RGB Canvas"] --> PatchEnc["2D Linear Patch Encoder<br/>(16x16 grid of 4x4 patches + 2D coord embeddings)"]
        PatchEnc --> VisTokens["256 Visual Tokens"]
        Prompt["Text Prompt<br/>('a cute friendly dragon...')"] --> BPE["Kilo Talk Tokenizer"]
        BPE --> TextTokens["Text Prompt Tokens"]
        Coord["Coordinate Query (x, y)"] --> CoordTokens["Coord Tokens &lt;C_x&gt;, &lt;C_y&gt;"]
    end

    VisTokens --> UnifiedSeq["Unified Sequence<br/>[256 Visual Tokens | BOS, Prompt, &lt;CANVAS_START&gt;, &lt;CANVAS_END&gt;, Coord_X, Coord_Y]"]
    TextTokens --> UnifiedSeq
    CoordTokens --> UnifiedSeq

    UnifiedSeq --> MoE["Sparse Mixture-of-Experts Brain<br/>(FlashAttention + Top-k Expert Routing)"]
    MoE --> QuerySlice["Extract Logits at Final Query Position (Coord_Y)"]
    QuerySlice --> PalettePred["Predict Next Token: 8-Bit Color &lt;COL_0..255&gt;"]

    PalettePred --> CanvasUpdate["Update 64x64 Canvas Pixels"]
    CanvasUpdate -.->|"Re-glance every N pixels"| Canvas
```

---

## 2. Canvas & Token Representation

### 8-Bit Truecolor Palette (256 Colors)
To balance rich artistic expressiveness with a compact vocabulary, Yumon uses a bijective 8-bit truecolor palette (3 bits Red, 3 bits Green, 2 bits Blue):
- **Red ($3$ bits)**: $8$ levels ($0..7 \implies 0, 36, 73, 109, 146, 182, 219, 255$)
- **Green ($3$ bits)**: $8$ levels ($0..7 \implies 0, 36, 73, 109, 146, 182, 219, 255$)
- **Blue ($2$ bits)**: $4$ levels ($0..3 \implies 0, 85, 170, 255$)

$$\text{Palette Index} = (R_3 \ll 5) \mid (G_3 \ll 2) \mid B_2$$

This preserves skin tones, skies, atmospheric lighting, and subtle shadows with zero loss of distinct colors across 256 discrete tokens.

### Dedicated Vocabulary Space (`vocab_size = 4428`)
The vocabulary extends the 4,096-token Kilo Talk language vocabulary:

| ID Range | Token Name | Purpose |
| :--- | :--- | :--- |
| `0..4095` | Kilo Talk Text Tokens | BPE text vocabulary for English conversation & prompts |
| `4096` | `<CANVAS_START>` | Delimiter indicating the canvas state is being queried |
| `4097` | `<CANVAS_END>` | Closes the canvas context tag |
| `4098` | `<PAINT>` | Reserved paint action tag |
| `4099` | `<DONE>` | Reserved completion tag |
| `4100..4163` | `<C_0>` .. `<C_63>` | 64 discrete coordinate indices for $x$ and $y$ |
| `4164..4419` | `<COL_0>` .. `<COL_255>` | 256 discrete 8-bit palette colors |
| `4420..4427` | `<RAD_1>` .. `<RAD_8>` | Brush radii |

---

## 3. The 2D Coordinate-Aware Vision Tokenizer

Traditional convolutional neural networks downsample images using pooling (`MaxPool2d`, `AdaptiveAvgPool2d`), creating *translation invariance*. That destroys the exact 2D position of edges and pixels.

Yumon uses a **2D Coordinate-Aware Linear Patch Encoder** ([`src/vision/patch_encoder.rs`](file:///D:/projects/common/yumon-pet/src/vision/patch_encoder.rs)):
1. **Patch Grid**: The $64 \times 64$ canvas is partitioned into a $16 \times 16$ grid of non-overlapping $4 \times 4$ pixel patches ($256$ visual tokens).
2. **Zero-Pooling Projection**: Each $4 \times 4 \times 3$ patch (48 floating-point values) is projected via a `Conv2d` (kernel size 4, stride 4) directly into `embed_dim` (e.g., 64).
3. **2D Coordinate Embeddings**: To inform the transformer of where each patch is located on the 2D plane, two learned embedding tables (`row_embed` for $y \in [0..15]$ and `col_embed` for $x \in [0..15]$) are added:
   $$\mathbf{z}_{r, c} = \text{RMSNorm}\left(\mathbf{W}_{\text{patch}} \mathbf{p}_{r, c} + \mathbf{E}_{\text{row}}(r) + \mathbf{E}_{\text{col}}(c)\right)$$
4. The resulting 256 visual vectors are prepended to the text sequence.

---

## 4. Why Coarse-to-Fine Multiscale Progression?

### The "Fax Machine" Failure
If an autoregressive model is trained to paint scanlines from top to bottom (pixel $(0,0)$, $(0,1)$, $(0,2)$...):
- When predicting $(0,0)$, the canvas is blank. The model has no idea what the bottom or middle of the scene will look like.
- It is forced to invent the scene from scratch with no global grounding.
- Any tiny mistake on line 2 cascades and corrupts the entire remaining image.

### The 3-Pass Solution
Instead of raster scanlines, Yumon paints in **three multiscale passes**:

```mermaid
flowchart LR
    P0["Blank Canvas<br/>(Light Gray 240,240,240)"] -->|Pass 1| P1["Pass 1: Composition<br/>(64 blocks of 8x8)<br/>Broad layout & skies"]
    P1 -->|Pass 2| P2["Pass 2: Shapes & Silhouettes<br/>(256 patches of 4x4)<br/>Aligns with visual tokens"]
    P2 -->|Pass 3| P3["Pass 3: Fine Pixel Art<br/>(4,096 native pixels)<br/>Highlights, eyes, edges"]
```

| Pass | Visual Canvas Input | Query Coordinates $(x, y)$ | Target Color | Granularity | Queries |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pass 1: Composition** | Blank canvas | Block centers: $(bx \cdot 8 + 4, by \cdot 8 + 4)$ | Average $8 \times 8$ block color | $8 \times 8$ blocks | 64 |
| **Pass 2: Shapes** | Pass 1 canvas | Patch centers: $(px \cdot 4 + 2, py \cdot 4 + 2)$ | Average $4 \times 4$ patch color | $4 \times 4$ patches | 256 |
| **Pass 3: Details** | Pass 2 canvas | Native pixels: $(x, y) \in [0..63] \times [0..63]$ | Native $64 \times 64$ ground-truth pixel | $1 \times 1$ pixel | 4,096 |

- In **Pass 1**, Yumon establishes global color balance (e.g. blue sky on top, green grass below, brown mass in center).
- In **Pass 2**, Yumon sees the $8 \times 8$ blocks and carves out silhouettes, character faces, and objects.
- In **Pass 3**, Yumon sees the sharpened shapes and renders fine pixel lighting, borders, and textures.

---

## 5. Training Dataset Generation

The dataset generator ([`src/bin/cache_diffusion_kilo.rs`](file:///D:/projects/common/yumon-pet/src/bin/cache_diffusion_kilo.rs)) ingests 1,000 DiffusionDB images and synthesizes **48 multiscale samples per image** (48,000 samples total, ~570 MB binary cache):

1. **Composition Samples (Pass 1 - 16 samples/image)**:
   - Canvas: Blank.
   - Coordinate: One of the 64 block centers.
   - Action: `[COORD_START + x, COORD_START + y, COLOR_START + comp_color]`.

2. **Shape Samples (Pass 2 - 16 samples/image)**:
   - Canvas: Pre-rendered $8 \times 8$ composition canvas.
   - Coordinate: One of the 256 patch centers.
   - Action: `[COORD_START + x, COORD_START + y, COLOR_START + shape_color]`.

3. **Fine Detail Samples (Pass 3a - 8 samples/image)**:
   - Canvas: Pre-rendered $16 \times 16$ shape canvas.
   - Coordinate: A native pixel coordinate $(x, y)$.
   - Action: `[COORD_START + x, COORD_START + y, COLOR_START + pixel_color]`.

4. **Refining Samples (Pass 3b - 8 samples/image)**:
   - Canvas: Half-painted mixed canvas (top half finished pixels, bottom half shapes).
   - Coordinate: A pixel coordinate from the unpainted area.
   - Action: `[COORD_START + x, COORD_START + y, COLOR_START + pixel_color]`.

---

## 6. Sequence Structure & Query Alignment

### Unified Transformer Input Sequence
In both training and inference, the sequence fed into `YumonMoeBrain` is constructed identically:

```text
[Position 0..255]     : 256 Visual Patch Tokens (Canvas Image)
[Position 256]        : BOS_TOKEN
[Position 257..256+P] : Prompt Tokens (P tokens from English prompt)
[Position 256+P+1]    : <CANVAS_START> (Token 4096)
[Position 256+P+2]    : <CANVAS_END>   (Token 4097)
[Position 256+P+3]    : Coordinate X   (<C_x>, Token 4100 + x)
[Position 256+P+4]    : Coordinate Y   (<C_y>, Token 4100 + y)  <--- QUERY POSITION
```

### Why Position Alignment Matters
Because causal self-attention masks all subsequent positions, token representations up to `<C_y>` depend only on:
1. All 256 visual canvas patches (bidirectionally visible or causally prior).
2. The user's text prompt.
3. The query coordinates $(x, y)$.

At sequence index:
$$pos = 256 + 4 + \text{prompt\_len}$$
The model's linear language head produces a logit vector across `ART_VOCAB_SIZE` ($4,428$). In next-token prediction, the token that immediately follows `<C_y>` is the target palette color `<COL_c>`.

---

## 7. Loss Calculation & Optimization

In [`src/bin/train_art_moe.rs`](file:///D:/projects/common/yumon-pet/src/bin/train_art_moe.rs):

### Direct Coordinate Logit Slicing
Rather than padding the sequence with `PAD_TOKEN` and masking, the trainer directly slices the logit vector at the $(x, y)$ coordinate position for each item $b$ in the batch:

```rust
let mut logit_slices = Vec::with_capacity(bs);
for (b, s) in chunk.iter().enumerate() {
    let pos = 256 + 4 + s.prompt_tokens.len(); // Coordinate Y position
    let slice = logits.clone().slice([b..b + 1, pos..pos + 1, 0..cfg.vocab_size]);
    logit_slices.push(slice);
}
let active_logits = Tensor::cat(logit_slices, 0).reshape([bs, cfg.vocab_size]);
```

### Unpadded Cross-Entropy Loss
The loss is computed directly between `active_logits` ($B \times 4428$) and `target_labels` ($B$):

$$\mathcal{L}_{\text{CE}} = -\frac{1}{B} \sum_{b=1}^B \ln P(\text{target\_color}_b)$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{aux}}$$

- **Zero Gradient Dilution**: The minibatch gradient scale is $\frac{1}{B}$, ensuring 100% full-strength weight updates from AdamW.
- **MoE Router Balance**: $\mathcal{L}_{\text{aux}}$ regularizes top-$k$ expert load balancing without dominating the color gradients.
- **Live Validation Metric**: Top-1 exact color match accuracy:
  $$\text{Color Acc} = \frac{\sum_{b=1}^B \mathbf{1}[\operatorname{argmax}(\mathbf{z}_b) = \mathbf{y}_b]}{B} \times 100\%$$

---

## 8. Interactive Studio Inference Loop

When running [`src/bin/yumon_paint.rs`](file:///D:/projects/common/yumon-pet/src/bin/yumon_paint.rs):

1. **Pass 1 (Composition)**:
   - Loops over $8 \times 8$ block centers (64 queries).
   - Samples best color with `--temperature` (or greedy argmax with `temperature = 0.0`).
   - Paints $8 \times 8$ rectangles on the in-memory canvas.
   - Re-encodes canvas into `canvas_tensor`.
   - Prints 24-bit ANSI truecolor preview to the terminal.

2. **Pass 2 (Shapes & Silhouettes)**:
   - Loops over $16 \times 16$ patch centers (256 queries).
   - Paints $4 \times 4$ patches.
   - Re-encodes canvas into `canvas_tensor`.
   - Prints ANSI terminal preview.

3. **Pass 3 (Fine Pixel Art)**:
   - Loops over all $64 \times 64 = 4,096$ pixels.
   - Every `--glance-interval` pixels (default: every 32 pixels), Yumon updates `canvas_tensor` so it actively sees the strokes it has just made.
   - Prints ANSI row-by-row animation to the terminal.

4. **Output Generation**:
   - Saves `yumon_artwork_64.png` (raw $64 \times 64$ RGB PNG).
   - Saves `yumon_artwork.png` (crisp nearest-neighbor $4\times$ upscaled $256 \times 256$ PNG).

---

## 9. Key Source Files Reference

- Vision Patch Encoder: [`src/vision/patch_encoder.rs`](file:///D:/projects/common/yumon-pet/src/vision/patch_encoder.rs)
- Palette & Multiscale Action Generator: [`src/vision/paint_action.rs`](file:///D:/projects/common/yumon-pet/src/vision/paint_action.rs)
- Multimodal MoE Brain: [`src/brain/moe_model.rs`](file:///D:/projects/common/yumon-pet/src/brain/moe_model.rs)
- Model Trainer: [`src/bin/train_art_moe.rs`](file:///D:/projects/common/yumon-pet/src/bin/train_art_moe.rs)
- Interactive Painting Studio: [`src/bin/yumon_paint.rs`](file:///D:/projects/common/yumon-pet/src/bin/yumon_paint.rs)
- Training Config: [`configs/yumon-art-moe.json`](file:///D:/projects/common/yumon-pet/configs/yumon-art-moe.json)
