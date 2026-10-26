//! Full Zhang et al. (ECCV 2016) "Colorful Image Colorization" subsystem.
//!
//! Takes a 1-channel grayscale Lightness ($L \in [0, 100]$) image at 256x256 (from Pass 5 Upscaler)
//! and predicts the chromaticity ($a^*, b^* \in [-128, 127]$) channels to yield a vibrant,
//! photo-realistic 256x256 RGB image.
//!
//! Features:
//! - CIE $L^*a^*b^*$ color space separation: preserves 100% of the sharpness and edges from Pass 5.
//! - Multinomial classification over $Q = 313$ in-gamut color cluster centers (`COLOR_BINS_AB`).
//! - Class-rebalancing loss to counteract natural desaturation bias and reward vibrant hues.
//! - Temperature-annealed softmax ($T = 0.38$) at inference for rich, smooth color fields.

use burn::{
    module::{Ignored, Module},
    nn::{
        conv::{Conv2d, Conv2dConfig},
        PaddingConfig2d,
    },
    prelude::*,
    tensor::{
        activation::{log_softmax, relu, softmax},
        Tensor,
    },
};

/// Total number of discrete color cluster centers in the CIE $a^*b^*$ color gamut.
pub const NUM_COLOR_BINS: usize = 313;

/// Grid step size in Lab space from Zhang et al. (2016).
pub const COLOR_GRID_STEP: f32 = 10.0;

/// Default inference temperature for annealed-mean softmax ($T = 0.20$).
pub const DEFAULT_TEMPERATURE: f32 = 0.20;

/// Default chroma saturation multiplier to counteract expectation contraction.
pub const DEFAULT_CHROMA_BOOST: f32 = 1.8;

/// Default input resolution (256x256).
pub const INPUT_SIZE: usize = 256;

/// Default target chroma grid resolution (64x64).
pub const TARGET_GRID: usize = 64;

/// Official 313 color cluster centers in CIE $a^*b^*$ coordinates from Richard Zhang's `pts_in_hull.npy`.
pub const COLOR_BINS_AB: [[f32; 2]; NUM_COLOR_BINS] = [
    [-90.0, 50.0],  // bin 0
    [-90.0, 60.0],  // bin 1
    [-90.0, 70.0],  // bin 2
    [-90.0, 80.0],  // bin 3
    [-90.0, 90.0],  // bin 4
    [-80.0, 20.0],  // bin 5
    [-80.0, 30.0],  // bin 6
    [-80.0, 40.0],  // bin 7
    [-80.0, 50.0],  // bin 8
    [-80.0, 60.0],  // bin 9
    [-80.0, 70.0],  // bin 10
    [-80.0, 80.0],  // bin 11
    [-80.0, 90.0],  // bin 12
    [-70.0, 0.0],   // bin 13
    [-70.0, 10.0],  // bin 14
    [-70.0, 20.0],  // bin 15
    [-70.0, 30.0],  // bin 16
    [-70.0, 40.0],  // bin 17
    [-70.0, 50.0],  // bin 18
    [-70.0, 60.0],  // bin 19
    [-70.0, 70.0],  // bin 20
    [-70.0, 80.0],  // bin 21
    [-70.0, 90.0],  // bin 22
    [-60.0, -20.0], // bin 23
    [-60.0, -10.0], // bin 24
    [-60.0, 0.0],   // bin 25
    [-60.0, 10.0],  // bin 26
    [-60.0, 20.0],  // bin 27
    [-60.0, 30.0],  // bin 28
    [-60.0, 40.0],  // bin 29
    [-60.0, 50.0],  // bin 30
    [-60.0, 60.0],  // bin 31
    [-60.0, 70.0],  // bin 32
    [-60.0, 80.0],  // bin 33
    [-60.0, 90.0],  // bin 34
    [-50.0, -30.0], // bin 35
    [-50.0, -20.0], // bin 36
    [-50.0, -10.0], // bin 37
    [-50.0, 0.0],   // bin 38
    [-50.0, 10.0],  // bin 39
    [-50.0, 20.0],  // bin 40
    [-50.0, 30.0],  // bin 41
    [-50.0, 40.0],  // bin 42
    [-50.0, 50.0],  // bin 43
    [-50.0, 60.0],  // bin 44
    [-50.0, 70.0],  // bin 45
    [-50.0, 80.0],  // bin 46
    [-50.0, 90.0],  // bin 47
    [-50.0, 100.0], // bin 48
    [-40.0, -40.0], // bin 49
    [-40.0, -30.0], // bin 50
    [-40.0, -20.0], // bin 51
    [-40.0, -10.0], // bin 52
    [-40.0, 0.0],   // bin 53
    [-40.0, 10.0],  // bin 54
    [-40.0, 20.0],  // bin 55
    [-40.0, 30.0],  // bin 56
    [-40.0, 40.0],  // bin 57
    [-40.0, 50.0],  // bin 58
    [-40.0, 60.0],  // bin 59
    [-40.0, 70.0],  // bin 60
    [-40.0, 80.0],  // bin 61
    [-40.0, 90.0],  // bin 62
    [-40.0, 100.0], // bin 63
    [-30.0, -50.0], // bin 64
    [-30.0, -40.0], // bin 65
    [-30.0, -30.0], // bin 66
    [-30.0, -20.0], // bin 67
    [-30.0, -10.0], // bin 68
    [-30.0, 0.0],   // bin 69
    [-30.0, 10.0],  // bin 70
    [-30.0, 20.0],  // bin 71
    [-30.0, 30.0],  // bin 72
    [-30.0, 40.0],  // bin 73
    [-30.0, 50.0],  // bin 74
    [-30.0, 60.0],  // bin 75
    [-30.0, 70.0],  // bin 76
    [-30.0, 80.0],  // bin 77
    [-30.0, 90.0],  // bin 78
    [-30.0, 100.0], // bin 79
    [-20.0, -70.0], // bin 80
    [-20.0, -60.0], // bin 81
    [-20.0, -50.0], // bin 82
    [-20.0, -40.0], // bin 83
    [-20.0, -30.0], // bin 84
    [-20.0, -20.0], // bin 85
    [-20.0, -10.0], // bin 86
    [-20.0, 0.0],   // bin 87
    [-20.0, 10.0],  // bin 88
    [-20.0, 20.0],  // bin 89
    [-20.0, 30.0],  // bin 90
    [-20.0, 40.0],  // bin 91
    [-20.0, 50.0],  // bin 92
    [-20.0, 60.0],  // bin 93
    [-20.0, 70.0],  // bin 94
    [-20.0, 80.0],  // bin 95
    [-20.0, 90.0],  // bin 96
    [-10.0, -80.0], // bin 97
    [-10.0, -70.0], // bin 98
    [-10.0, -60.0], // bin 99
    [-10.0, -50.0], // bin 100
    [-10.0, -40.0], // bin 101
    [-10.0, -30.0], // bin 102
    [-10.0, -20.0], // bin 103
    [-10.0, -10.0], // bin 104
    [-10.0, 0.0],   // bin 105
    [-10.0, 10.0],  // bin 106
    [-10.0, 20.0],  // bin 107
    [-10.0, 30.0],  // bin 108
    [-10.0, 40.0],  // bin 109
    [-10.0, 50.0],  // bin 110
    [-10.0, 60.0],  // bin 111
    [-10.0, 70.0],  // bin 112
    [-10.0, 80.0],  // bin 113
    [-10.0, 90.0],  // bin 114
    [0.0, -90.0],   // bin 115
    [0.0, -80.0],   // bin 116
    [0.0, -70.0],   // bin 117
    [0.0, -60.0],   // bin 118
    [0.0, -50.0],   // bin 119
    [0.0, -40.0],   // bin 120
    [0.0, -30.0],   // bin 121
    [0.0, -20.0],   // bin 122
    [0.0, -10.0],   // bin 123
    [0.0, 0.0],     // bin 124
    [0.0, 10.0],    // bin 125
    [0.0, 20.0],    // bin 126
    [0.0, 30.0],    // bin 127
    [0.0, 40.0],    // bin 128
    [0.0, 50.0],    // bin 129
    [0.0, 60.0],    // bin 130
    [0.0, 70.0],    // bin 131
    [0.0, 80.0],    // bin 132
    [0.0, 90.0],    // bin 133
    [10.0, -110.0], // bin 134
    [10.0, -100.0], // bin 135
    [10.0, -90.0],  // bin 136
    [10.0, -80.0],  // bin 137
    [10.0, -70.0],  // bin 138
    [10.0, -60.0],  // bin 139
    [10.0, -50.0],  // bin 140
    [10.0, -40.0],  // bin 141
    [10.0, -30.0],  // bin 142
    [10.0, -20.0],  // bin 143
    [10.0, -10.0],  // bin 144
    [10.0, 0.0],    // bin 145
    [10.0, 10.0],   // bin 146
    [10.0, 20.0],   // bin 147
    [10.0, 30.0],   // bin 148
    [10.0, 40.0],   // bin 149
    [10.0, 50.0],   // bin 150
    [10.0, 60.0],   // bin 151
    [10.0, 70.0],   // bin 152
    [10.0, 80.0],   // bin 153
    [10.0, 90.0],   // bin 154
    [20.0, -110.0], // bin 155
    [20.0, -100.0], // bin 156
    [20.0, -90.0],  // bin 157
    [20.0, -80.0],  // bin 158
    [20.0, -70.0],  // bin 159
    [20.0, -60.0],  // bin 160
    [20.0, -50.0],  // bin 161
    [20.0, -40.0],  // bin 162
    [20.0, -30.0],  // bin 163
    [20.0, -20.0],  // bin 164
    [20.0, -10.0],  // bin 165
    [20.0, 0.0],    // bin 166
    [20.0, 10.0],   // bin 167
    [20.0, 20.0],   // bin 168
    [20.0, 30.0],   // bin 169
    [20.0, 40.0],   // bin 170
    [20.0, 50.0],   // bin 171
    [20.0, 60.0],   // bin 172
    [20.0, 70.0],   // bin 173
    [20.0, 80.0],   // bin 174
    [20.0, 90.0],   // bin 175
    [30.0, -110.0], // bin 176
    [30.0, -100.0], // bin 177
    [30.0, -90.0],  // bin 178
    [30.0, -80.0],  // bin 179
    [30.0, -70.0],  // bin 180
    [30.0, -60.0],  // bin 181
    [30.0, -50.0],  // bin 182
    [30.0, -40.0],  // bin 183
    [30.0, -30.0],  // bin 184
    [30.0, -20.0],  // bin 185
    [30.0, -10.0],  // bin 186
    [30.0, 0.0],    // bin 187
    [30.0, 10.0],   // bin 188
    [30.0, 20.0],   // bin 189
    [30.0, 30.0],   // bin 190
    [30.0, 40.0],   // bin 191
    [30.0, 50.0],   // bin 192
    [30.0, 60.0],   // bin 193
    [30.0, 70.0],   // bin 194
    [30.0, 80.0],   // bin 195
    [30.0, 90.0],   // bin 196
    [40.0, -110.0], // bin 197
    [40.0, -100.0], // bin 198
    [40.0, -90.0],  // bin 199
    [40.0, -80.0],  // bin 200
    [40.0, -70.0],  // bin 201
    [40.0, -60.0],  // bin 202
    [40.0, -50.0],  // bin 203
    [40.0, -40.0],  // bin 204
    [40.0, -30.0],  // bin 205
    [40.0, -20.0],  // bin 206
    [40.0, -10.0],  // bin 207
    [40.0, 0.0],    // bin 208
    [40.0, 10.0],   // bin 209
    [40.0, 20.0],   // bin 210
    [40.0, 30.0],   // bin 211
    [40.0, 40.0],   // bin 212
    [40.0, 50.0],   // bin 213
    [40.0, 60.0],   // bin 214
    [40.0, 70.0],   // bin 215
    [40.0, 80.0],   // bin 216
    [40.0, 90.0],   // bin 217
    [50.0, -110.0], // bin 218
    [50.0, -100.0], // bin 219
    [50.0, -90.0],  // bin 220
    [50.0, -80.0],  // bin 221
    [50.0, -70.0],  // bin 222
    [50.0, -60.0],  // bin 223
    [50.0, -50.0],  // bin 224
    [50.0, -40.0],  // bin 225
    [50.0, -30.0],  // bin 226
    [50.0, -20.0],  // bin 227
    [50.0, -10.0],  // bin 228
    [50.0, 0.0],    // bin 229
    [50.0, 10.0],   // bin 230
    [50.0, 20.0],   // bin 231
    [50.0, 30.0],   // bin 232
    [50.0, 40.0],   // bin 233
    [50.0, 50.0],   // bin 234
    [50.0, 60.0],   // bin 235
    [50.0, 70.0],   // bin 236
    [50.0, 80.0],   // bin 237
    [50.0, 90.0],   // bin 238
    [60.0, -110.0], // bin 239
    [60.0, -100.0], // bin 240
    [60.0, -90.0],  // bin 241
    [60.0, -80.0],  // bin 242
    [60.0, -70.0],  // bin 243
    [60.0, -60.0],  // bin 244
    [60.0, -50.0],  // bin 245
    [60.0, -40.0],  // bin 246
    [60.0, -30.0],  // bin 247
    [60.0, -20.0],  // bin 248
    [60.0, -10.0],  // bin 249
    [60.0, 0.0],    // bin 250
    [60.0, 10.0],   // bin 251
    [60.0, 20.0],   // bin 252
    [60.0, 30.0],   // bin 253
    [60.0, 40.0],   // bin 254
    [60.0, 50.0],   // bin 255
    [60.0, 60.0],   // bin 256
    [60.0, 70.0],   // bin 257
    [60.0, 80.0],   // bin 258
    [60.0, 90.0],   // bin 259
    [70.0, -110.0], // bin 260
    [70.0, -100.0], // bin 261
    [70.0, -90.0],  // bin 262
    [70.0, -80.0],  // bin 263
    [70.0, -70.0],  // bin 264
    [70.0, -60.0],  // bin 265
    [70.0, -50.0],  // bin 266
    [70.0, -40.0],  // bin 267
    [70.0, -30.0],  // bin 268
    [70.0, -20.0],  // bin 269
    [70.0, -10.0],  // bin 270
    [70.0, 0.0],    // bin 271
    [70.0, 10.0],   // bin 272
    [70.0, 20.0],   // bin 273
    [70.0, 30.0],   // bin 274
    [70.0, 40.0],   // bin 275
    [70.0, 50.0],   // bin 276
    [70.0, 60.0],   // bin 277
    [70.0, 70.0],   // bin 278
    [70.0, 80.0],   // bin 279
    [80.0, -110.0], // bin 280
    [80.0, -100.0], // bin 281
    [80.0, -90.0],  // bin 282
    [80.0, -80.0],  // bin 283
    [80.0, -70.0],  // bin 284
    [80.0, -60.0],  // bin 285
    [80.0, -50.0],  // bin 286
    [80.0, -40.0],  // bin 287
    [80.0, -30.0],  // bin 288
    [80.0, -20.0],  // bin 289
    [80.0, -10.0],  // bin 290
    [80.0, 0.0],    // bin 291
    [80.0, 10.0],   // bin 292
    [80.0, 20.0],   // bin 293
    [80.0, 30.0],   // bin 294
    [80.0, 40.0],   // bin 295
    [80.0, 50.0],   // bin 296
    [80.0, 60.0],   // bin 297
    [80.0, 70.0],   // bin 298
    [80.0, 80.0],   // bin 299
    [90.0, -110.0], // bin 300
    [90.0, -100.0], // bin 301
    [90.0, -90.0],  // bin 302
    [90.0, -80.0],  // bin 303
    [90.0, -70.0],  // bin 304
    [90.0, -60.0],  // bin 305
    [90.0, -50.0],  // bin 306
    [90.0, -40.0],  // bin 307
    [90.0, -30.0],  // bin 308
    [100.0, -40.0], // bin 309
    [100.0, -30.0], // bin 310
    [100.0, -20.0], // bin 311
    [100.0, -10.0], // bin 312
];

/// Precomputed O(1) Lookup Table mapping (a, b) integers in [-128..127] to nearest bin index [0..312].
pub struct ColorizerLut {
    table: Box<[u16; 256 * 256]>,
}

impl ColorizerLut {
    pub fn new() -> Self {
        let mut table = vec![0u16; 256 * 256].into_boxed_slice();
        let pts = &COLOR_BINS_AB;

        for a_idx in 0..256 {
            let a = (a_idx as f32) - 128.0;
            for b_idx in 0..256 {
                let b = (b_idx as f32) - 128.0;

                let mut best_dist = f32::MAX;
                let mut best_q = 0u16;

                for (q, p) in pts.iter().enumerate() {
                    let d = (a - p[0]).powi(2) + (b - p[1]).powi(2);
                    if d < best_dist {
                        best_dist = d;
                        best_q = q as u16;
                    }
                }
                table[a_idx * 256 + b_idx] = best_q;
            }
        }

        Self {
            table: table.try_into().unwrap(),
        }
    }

    #[inline(always)]
    pub fn lookup(&self, a: f32, b: f32) -> usize {
        let a_idx = ((a + 128.0).round().clamp(0.0, 255.0)) as usize;
        let b_idx = ((b + 128.0).round().clamp(0.0, 255.0)) as usize;
        self.table[a_idx * 256 + b_idx] as usize
    }
}

impl Default for ColorizerLut {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Color Space Conversion (sRGB <-> CIE Lab D65) ──────────────────────────

#[inline(always)]
fn to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline(always)]
fn from_linear(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[inline(always)]
fn lab_f(t: f32) -> f32 {
    let delta: f32 = 6.0 / 29.0;
    if t > delta.powi(3) {
        t.cbrt()
    } else {
        t / (3.0 * delta.powi(2)) + 4.0 / 29.0
    }
}

#[inline(always)]
fn lab_inv_f(t: f32) -> f32 {
    let delta: f32 = 6.0 / 29.0;
    if t > delta {
        t.powi(3)
    } else {
        3.0 * delta.powi(2) * (t - 4.0 / 29.0)
    }
}

/// Convert sRGB [0..255] to CIE Lab D65: L in [0..100], a in [-128..127], b in [-128..127].
pub fn srgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r_lin = to_linear(r as f32 / 255.0);
    let g_lin = to_linear(g as f32 / 255.0);
    let b_lin = to_linear(b as f32 / 255.0);

    let x = (r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375) / 0.95047;
    let y = (r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750) / 1.00000;
    let z = (r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041) / 1.08883;

    let fx = lab_f(x);
    let fy = lab_f(y);
    let fz = lab_f(z);

    let l = 116.0 * fy - 16.0;
    let a_val = 500.0 * (fx - fy);
    let b_val = 200.0 * (fy - fz);

    (l, a_val, b_val)
}

/// Convert CIE Lab D65 back to sRGB [0..255].
pub fn lab_to_srgb(l: f32, a: f32, b: f32) -> (u8, u8, u8) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;

    let x = 0.95047 * lab_inv_f(fx);
    let y = 1.00000 * lab_inv_f(fy);
    let z = 1.08883 * lab_inv_f(fz);

    let r_lin =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let b_lin =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

    let r = (from_linear(r_lin.clamp(0.0, 1.0)) * 255.0).round().clamp(0.0, 255.0) as u8;
    let g = (from_linear(g_lin.clamp(0.0, 1.0)) * 255.0).round().clamp(0.0, 255.0) as u8;
    let b = (from_linear(b_lin.clamp(0.0, 1.0)) * 255.0).round().clamp(0.0, 255.0) as u8;

    (r, g, b)
}

// ─── Neural Network Architecture ──────────────────────────────────────────

/// Residual block with two 3x3 convolutions and 0.2 residual scaling.
#[derive(Module, Debug)]
pub struct ColorizerResBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> ColorizerResBlock<B> {
    pub fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([channels, channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([channels, channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let res = relu(self.conv1.forward(x.clone()));
        x + self.conv2.forward(res) * 0.2
    }
}

/// Configuration for the Full Zhang 2016 Colorizer.
#[derive(Config, Debug)]
pub struct ColorizerModelConfig {
    #[config(default = 256)]
    pub scales_from: usize,

    #[config(default = 64)]
    pub channels: usize,

    #[config(default = 5)]
    pub num_res_blocks: usize,
}

impl ColorizerModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ColorizerModel<B> {
        let c = self.channels;

        // If input is 256x256, downsample 4x to 64x64 via two strided convolutions:
        let (down1, down2) = if self.scales_from >= 256 {
            let d1 = Conv2dConfig::new([1, c / 2], [4, 4])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device);
            let d2 = Conv2dConfig::new([c / 2, c], [4, 4])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device);
            (Some(d1), Some(d2))
        } else if self.scales_from == 128 {
            let d1 = Conv2dConfig::new([1, c], [4, 4])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device);
            (Some(d1), None)
        } else {
            // Already at 64x64
            (None, None)
        };

        let stem = if self.scales_from <= 64 {
            Some(
                Conv2dConfig::new([1, c], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
            )
        } else {
            None
        };

        let res_blocks = (0..self.num_res_blocks)
            .map(|_| ColorizerResBlock::new(c, device))
            .collect();

        // Refinement before multinomial classification head
        let head_conv = Conv2dConfig::new([c, c], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        // Final 1x1 conv predicting logits over all 313 color bins
        let out_conv = Conv2dConfig::new([c, NUM_COLOR_BINS], [1, 1])
            .init(device);

        ColorizerModel {
            down1,
            down2,
            stem,
            res_blocks,
            head_conv,
            out_conv,
            config: Ignored(self.clone()),
        }
    }
}

/// The Full Zhang 2016 Colorizer Model.
#[derive(Module, Debug)]
pub struct ColorizerModel<B: Backend> {
    down1: Option<Conv2d<B>>,
    down2: Option<Conv2d<B>>,
    stem: Option<Conv2d<B>>,
    res_blocks: Vec<ColorizerResBlock<B>>,
    head_conv: Conv2d<B>,
    out_conv: Conv2d<B>,
    pub config: Ignored<ColorizerModelConfig>,
}

impl<B: Backend> ColorizerModel<B> {
    /// Forward pass:
    /// Input:  `[batch, 1, H, W]` (Grayscale Lightness normalized in [-1.0, 1.0])
    /// Output: `[batch, 313, 64, 64]` (Unnormalized logits over the 313 color cluster bins)
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut f = x;

        if let Some(ref d1) = self.down1 {
            f = relu(d1.forward(f));
        }
        if let Some(ref d2) = self.down2 {
            f = relu(d2.forward(f));
        }
        if let Some(ref stem) = self.stem {
            f = relu(stem.forward(f));
        }

        for block in &self.res_blocks {
            f = block.forward(f);
        }

        f = relu(self.head_conv.forward(f));
        self.out_conv.forward(f)
    }

    /// Predict continuous chromaticity (a, b) at 64x64 using temperature-annealed softmax.
    ///
    /// At inference, taking standard expectation ($\mathbb{E}$) yields dull averages, while argmax causes speckling.
    /// Annealed-mean softmax with temperature $T = 0.38$ sharpens the distribution towards vibrant modes while
    /// producing spatially coherent color gradients.
    pub fn predict_ab_annealed(
        &self,
        x: Tensor<B, 4>,
        temperature: f32,
        chroma_boost: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let logits = self.forward(x);
        let [b, q, h, w] = logits.dims();
        assert_eq!(b, 1, "Inference annealed prediction currently supports single image");
        assert_eq!(q, NUM_COLOR_BINS);

        // Softmax with temperature scaling: Z / T
        let scaled_logits = logits / temperature;
        let probs = softmax(scaled_logits, 1);
        let prob_data: Vec<f32> = probs.into_data().convert::<f32>().to_vec().unwrap();

        let spatial_pixels = h * w;
        let mut out_a = vec![0.0f32; spatial_pixels];
        let mut out_b = vec![0.0f32; spatial_pixels];

        let pts = &COLOR_BINS_AB;

        for p_idx in 0..spatial_pixels {
            let mut sum_a = 0.0f32;
            let mut sum_b = 0.0f32;

            for bin in 0..NUM_COLOR_BINS {
                let prob = prob_data[bin * spatial_pixels + p_idx];
                sum_a += prob * pts[bin][0];
                sum_b += prob * pts[bin][1];
            }

            out_a[p_idx] = (sum_a * chroma_boost).clamp(-128.0, 127.0);
            out_b[p_idx] = (sum_b * chroma_boost).clamp(-128.0, 127.0);
        }

        (out_a, out_b)
    }
}

/// Bilinearly upsample a planar 64x64 float buffer to 256x256.
pub fn bilinear_upsample_64_to_256(src: &[f32], src_size: usize, dst_size: usize) -> Vec<f32> {
    assert_eq!(src.len(), src_size * src_size);
    let mut dst = vec![0.0f32; dst_size * dst_size];
    let scale = (src_size as f32) / (dst_size as f32);

    for y in 0..dst_size {
        let src_y = (y as f32 + 0.5) * scale - 0.5;
        let y0 = (src_y.floor() as isize).clamp(0, (src_size - 1) as isize) as usize;
        let y1 = (y0 + 1).min(src_size - 1);
        let wy = (src_y - y0 as f32).clamp(0.0, 1.0);

        for x in 0..dst_size {
            let src_x = (x as f32 + 0.5) * scale - 0.5;
            let x0 = (src_x.floor() as isize).clamp(0, (src_size - 1) as isize) as usize;
            let x1 = (x0 + 1).min(src_size - 1);
            let wx = (src_x - x0 as f32).clamp(0.0, 1.0);

            let v00 = src[y0 * src_size + x0];
            let v01 = src[y0 * src_size + x1];
            let v10 = src[y1 * src_size + x0];
            let v11 = src[y1 * src_size + x1];

            let top = v00 * (1.0 - wx) + v01 * wx;
            let bot = v10 * (1.0 - wx) + v11 * wx;
            dst[y * dst_size + x] = top * (1.0 - wy) + bot * wy;
        }
    }

    dst
}

/// Full colorization: fuse pristine 256x256 grayscale Lightness (L) with predicted (a, b) chroma to produce an RGB image.
pub fn colorize_fused_rgb(
    l_bytes: &[u8],
    pred_a_64: &[f32],
    pred_b_64: &[f32],
    out_size: usize,
) -> image::RgbImage {
    assert_eq!(l_bytes.len(), out_size * out_size);
    let up_a = bilinear_upsample_64_to_256(pred_a_64, 64, out_size);
    let up_b = bilinear_upsample_64_to_256(pred_b_64, 64, out_size);

    let mut img = image::RgbImage::new(out_size as u32, out_size as u32);

    for y in 0..out_size {
        for x in 0..out_size {
            let idx = y * out_size + x;
            // Grayscale byte [0..255] maps linearly to L in [0..100]
            let l_val = (l_bytes[idx] as f32 / 255.0) * 100.0;
            let a_val = up_a[idx];
            let b_val = up_b[idx];

            let (r, g, b) = lab_to_srgb(l_val, a_val, b_val);
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    img
}

/// Rebalanced multinomial cross-entropy loss from Zhang et al. (2016).
///
/// Logits:  `[B, 313, 64, 64]`
/// Targets: `[B, 4096]` (Int tensor of class bin indices in 0..312)
/// Weights: `[313]` (Class rebalancing multiplier for each bin)
pub fn rebalanced_cross_entropy<B: Backend>(
    logits: Tensor<B, 4>,
    targets: Tensor<B, 2, Int>,
    weights: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let [b, q, h, w] = logits.dims();
    let spatial_len = h * w;

    // Reshape logits to [B, spatial_len, 313]
    let flat_logits = logits
        .swap_dims(1, 2)
        .swap_dims(2, 3)
        .reshape([b, spatial_len, q]);

    // Log-softmax over the 313 color bin classes
    let log_probs = log_softmax(flat_logits, 2);

    // Negative log-likelihood gathered at target indices: [B, spatial_len]
    let nll = log_probs
        .gather(2, targets.clone().reshape([b, spatial_len, 1]))
        .reshape([b, spatial_len])
        .neg();

    // Gather class rebalancing weights for each target pixel
    let flat_targets = targets.reshape([b * spatial_len]);
    let target_weights = weights.select(0, flat_targets).reshape([b, spatial_len]);

    // Weighted cross entropy properly normalized by total batch weight
    let total_weight = target_weights.clone().sum().clamp_min(1.0);
    (nll * target_weights).sum() / total_weight
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

    #[test]
    fn test_loss_computation() {
        let device = Default::default();
        let b = 2;
        let q = NUM_COLOR_BINS;
        let h = 64;
        let w = 64;
        let spatial_len = h * w;

        let mut flat_logits = vec![0.0f32; b * q * spatial_len];
        for batch in 0..b {
            for s in 0..spatial_len {
                flat_logits[batch * (q * spatial_len) + 5 * spatial_len + s] = 10.0;
            }
        }
        let logits: Tensor<B, 4> = Tensor::from_floats(
            TensorData::new(flat_logits, [b, q, h, w]),
            &device,
        );

        let targets: Tensor<B, 2, Int> = Tensor::from_ints(
            TensorData::new(vec![5i32; b * spatial_len], [b, spatial_len]),
            &device,
        );

        let weights: Tensor<B, 1> = Tensor::from_floats(
            TensorData::new(vec![1.0f32; q], [q]),
            &device,
        );

        let loss = rebalanced_cross_entropy(logits.clone(), targets.clone(), weights);
        let loss_val: f32 = loss.into_data().convert::<f32>().to_vec().unwrap()[0];
        println!("Loss when predicting target: {loss_val}");

        let wrong_targets: Tensor<B, 2, Int> = Tensor::from_ints(
            TensorData::new(vec![0i32; b * spatial_len], [b, spatial_len]),
            &device,
        );
        let wrong_loss = rebalanced_cross_entropy(
            logits,
            wrong_targets,
            Tensor::from_floats(TensorData::new(vec![1.0f32; q], [q]), &device),
        );
        let wrong_loss_val: f32 = wrong_loss.into_data().convert::<f32>().to_vec().unwrap()[0];
        println!("Loss when predicting WRONG target: {wrong_loss_val}");

        assert!(loss_val < 0.1, "Loss when correct should be tiny, was {loss_val}");
        assert!(wrong_loss_val > 5.0, "Loss when wrong should be large, was {wrong_loss_val}");
    }

    #[test]
    fn test_4d_tensor_layout() {
        let device = Default::default();
        let data = vec![
            10.0, 11.0, 12.0, 13.0,
            20.0, 21.0, 22.0, 23.0,
        ];
        let t: Tensor<B, 4> = Tensor::from_floats(TensorData::new(data.clone(), [1, 2, 2, 2]), &device);
        let vec_out: Vec<f32> = t.into_data().convert::<f32>().to_vec().unwrap();
        println!("vec_out: {:?}", vec_out);
        assert_eq!(data, vec_out);
    }

    #[test]
    fn test_inspect_saved_model() {
        let device = Default::default();
        let config = ColorizerModelConfig::new()
            .with_scales_from(256)
            .with_channels(64)
            .with_num_res_blocks(5);
        let model: ColorizerModel<B> = config.init(&device);
        let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::default();
        let path = std::path::PathBuf::from("checkpoints/yumon-colorizer-256/model");
        let model = match model.load_file(&path, &recorder, &device) {
            Ok(m) => m,
            Err(e) => {
                println!("Failed to load: {e}");
                return;
            }
        };

        let input = Tensor::<B, 4>::zeros([1, 1, 256, 256], &device);
        let logits = model.forward(input);
        let [b, q, h, w] = logits.dims();
        println!("Logits dims: {:?}", [b, q, h, w]);

        let logit_data: Vec<f32> = logits.into_data().convert::<f32>().to_vec().unwrap();
        let min_l = logit_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_l = logit_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_l = logit_data.iter().sum::<f32>() / logit_data.len() as f32;
        println!("Logits min: {min_l}, max: {max_l}, mean: {mean_l}");

        let p_idx = 32 * 64 + 32;
        let mut best_bin = 0;
        let mut best_val = f32::NEG_INFINITY;
        for bin in 0..q {
            let val = logit_data[bin * 4096 + p_idx];
            if val > best_val {
                best_val = val;
                best_bin = bin;
            }
        }
        println!("Predicted bin at (32,32): {best_bin} with logit {best_val}");
    }

    #[test]
    fn test_image_iter_order() {
        let img = image::RgbImage::new(3, 2);
        let mut coords = Vec::new();
        for (x, y, _) in img.enumerate_pixels() {
            coords.push((x, y));
        }
        println!("Coords: {:?}", coords);
        assert_eq!(coords[0], (0, 0));
        assert_eq!(coords[1], (1, 0));
        assert_eq!(coords[2], (2, 0));
        assert_eq!(coords[3], (0, 1));
        assert_eq!(coords[4], (1, 1));
        assert_eq!(coords[5], (2, 1));
    }

    #[test]
    fn test_sample0_predictions() {
        let device = Default::default();
        let config = ColorizerModelConfig::new()
            .with_scales_from(256)
            .with_channels(64)
            .with_num_res_blocks(5);
        let model: ColorizerModel<B> = config.init(&device);
        let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::default();
        let path = std::path::PathBuf::from("checkpoints/yumon-colorizer-256/model");
        let model = model.load_file(&path, &recorder, &device).unwrap();

        let mut f = std::fs::File::open("training-cache/colorizer_256.bin").unwrap();
        use std::io::Read;
        let mut header = [0u8; 1284];
        f.read_exact(&mut header).unwrap();
        let mut sample0 = vec![0u8; 73728];
        f.read_exact(&mut sample0).unwrap();
        let l_bytes = &sample0[..65536];

        let flat_in: Vec<f32> = l_bytes.iter().map(|&b| (b as f32 / 255.0) * 2.0 - 1.0).collect();
        let in_tensor = Tensor::<B, 4>::from_floats(
            TensorData::new(flat_in, [1, 1, 256, 256]),
            &device,
        );

        let (pred_a, pred_b) = model.predict_ab_annealed(in_tensor.clone(), 0.38, 1.0);

        let max_a = pred_a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_b = pred_b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let mean_a = pred_a.iter().map(|v| v.abs()).sum::<f32>() / pred_a.len() as f32;
        let mean_b = pred_b.iter().map(|v| v.abs()).sum::<f32>() / pred_b.len() as f32;

        println!("T=0.38 -> max |a|: {max_a:.2}, max |b|: {max_b:.2}, mean |a|: {mean_a:.2}, mean |b|: {mean_b:.2}");

        let (pred_a_low, pred_b_low) = model.predict_ab_annealed(in_tensor.clone(), 0.15, DEFAULT_CHROMA_BOOST);
        let max_a_l = pred_a_low.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_b_l = pred_b_low.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let mean_a_l = pred_a_low.iter().map(|v| v.abs()).sum::<f32>() / pred_a_low.len() as f32;
        let mean_b_l = pred_b_low.iter().map(|v| v.abs()).sum::<f32>() / pred_b_low.len() as f32;

        println!("T=0.15 -> max |a|: {max_a_l:.2}, max |b|: {max_b_l:.2}, mean |a|: {mean_a_l:.2}, mean |b|: {mean_b_l:.2}");

        let pts = &COLOR_BINS_AB;
        let tgt_bytes = &sample0[65536..];
        let mut gt_a = vec![0.0f32; 4096];
        let mut gt_b = vec![0.0f32; 4096];
        for i in 0..4096 {
            let bin = u16::from_le_bytes([tgt_bytes[i*2], tgt_bytes[i*2+1]]) as usize;
            gt_a[i] = pts[bin][0];
            gt_b[i] = pts[bin][1];
        }
        let gt_max_a = gt_a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let gt_max_b = gt_b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let gt_mean_a = gt_a.iter().map(|v| v.abs()).sum::<f32>() / gt_a.len() as f32;
        let gt_mean_b = gt_b.iter().map(|v| v.abs()).sum::<f32>() / gt_b.len() as f32;

        println!("Ground Truth -> max |a|: {gt_max_a:.2}, max |b|: {gt_max_b:.2}, mean |a|: {gt_mean_a:.2}, mean |b|: {gt_mean_b:.2}");
    }
}
