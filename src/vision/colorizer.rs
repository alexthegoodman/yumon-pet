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
    module::{Ignored, Module, Param},
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        Initializer, Linear, LinearConfig,
        PaddingConfig2d,
    },
    prelude::*,
    tensor::{
        activation::{log_softmax, relu, softmax},
        Tensor, TensorData,
    },
};

/// Total number of discrete color cluster centers in the CIE $a^*b^*$ color gamut.
pub const NUM_COLOR_BINS: usize = 313;

/// Grid step size in Lab space from Zhang et al. (2016).
pub const COLOR_GRID_STEP: f32 = 10.0;

/// Default inference temperature for annealed-mean softmax ($T = 0.20$).
pub const DEFAULT_TEMPERATURE: f32 = 0.38;

/// Default chroma saturation multiplier to counteract expectation contraction.
pub const DEFAULT_CHROMA_BOOST: f32 = 1.0;

/// Default input resolution (256x256).
pub const INPUT_SIZE: usize = 256;

/// Default target chroma grid resolution (64x64).
pub const TARGET_GRID: usize = 64;

/// Official 313 color cluster centers in CIE $a^*b^*$ coordinates from Richard Zhang's `pts_in_hull.npy`.
pub const COLOR_BINS_AB: [[f32; 2]; NUM_COLOR_BINS] = [
    [-90.0, 50.0], // bin 0
    [-90.0, 60.0], // bin 1
    [-90.0, 70.0], // bin 2
    [-90.0, 80.0], // bin 3
    [-90.0, 90.0], // bin 4
    [-80.0, 20.0], // bin 5
    [-80.0, 30.0], // bin 6
    [-80.0, 40.0], // bin 7
    [-80.0, 50.0], // bin 8
    [-80.0, 60.0], // bin 9
    [-80.0, 70.0], // bin 10
    [-80.0, 80.0], // bin 11
    [-80.0, 90.0], // bin 12
    [-70.0, 0.0], // bin 13
    [-70.0, 10.0], // bin 14
    [-70.0, 20.0], // bin 15
    [-70.0, 30.0], // bin 16
    [-70.0, 40.0], // bin 17
    [-70.0, 50.0], // bin 18
    [-70.0, 60.0], // bin 19
    [-70.0, 70.0], // bin 20
    [-70.0, 80.0], // bin 21
    [-70.0, 90.0], // bin 22
    [-60.0, -20.0], // bin 23
    [-60.0, -10.0], // bin 24
    [-60.0, 0.0], // bin 25
    [-60.0, 10.0], // bin 26
    [-60.0, 20.0], // bin 27
    [-60.0, 30.0], // bin 28
    [-60.0, 40.0], // bin 29
    [-60.0, 50.0], // bin 30
    [-60.0, 60.0], // bin 31
    [-60.0, 70.0], // bin 32
    [-60.0, 80.0], // bin 33
    [-60.0, 90.0], // bin 34
    [-50.0, -30.0], // bin 35
    [-50.0, -20.0], // bin 36
    [-50.0, -10.0], // bin 37
    [-50.0, 0.0], // bin 38
    [-50.0, 10.0], // bin 39
    [-50.0, 20.0], // bin 40
    [-50.0, 30.0], // bin 41
    [-50.0, 40.0], // bin 42
    [-50.0, 50.0], // bin 43
    [-50.0, 60.0], // bin 44
    [-50.0, 70.0], // bin 45
    [-50.0, 80.0], // bin 46
    [-50.0, 90.0], // bin 47
    [-50.0, 100.0], // bin 48
    [-40.0, -40.0], // bin 49
    [-40.0, -30.0], // bin 50
    [-40.0, -20.0], // bin 51
    [-40.0, -10.0], // bin 52
    [-40.0, 0.0], // bin 53
    [-40.0, 10.0], // bin 54
    [-40.0, 20.0], // bin 55
    [-40.0, 30.0], // bin 56
    [-40.0, 40.0], // bin 57
    [-40.0, 50.0], // bin 58
    [-40.0, 60.0], // bin 59
    [-40.0, 70.0], // bin 60
    [-40.0, 80.0], // bin 61
    [-40.0, 90.0], // bin 62
    [-40.0, 100.0], // bin 63
    [-30.0, -50.0], // bin 64
    [-30.0, -40.0], // bin 65
    [-30.0, -30.0], // bin 66
    [-30.0, -20.0], // bin 67
    [-30.0, -10.0], // bin 68
    [-30.0, 0.0], // bin 69
    [-30.0, 10.0], // bin 70
    [-30.0, 20.0], // bin 71
    [-30.0, 30.0], // bin 72
    [-30.0, 40.0], // bin 73
    [-30.0, 50.0], // bin 74
    [-30.0, 60.0], // bin 75
    [-30.0, 70.0], // bin 76
    [-30.0, 80.0], // bin 77
    [-30.0, 90.0], // bin 78
    [-30.0, 100.0], // bin 79
    [-20.0, -50.0], // bin 80
    [-20.0, -40.0], // bin 81
    [-20.0, -30.0], // bin 82
    [-20.0, -20.0], // bin 83
    [-20.0, -10.0], // bin 84
    [-20.0, 0.0], // bin 85
    [-20.0, 10.0], // bin 86
    [-20.0, 20.0], // bin 87
    [-20.0, 30.0], // bin 88
    [-20.0, 40.0], // bin 89
    [-20.0, 50.0], // bin 90
    [-20.0, 60.0], // bin 91
    [-20.0, 70.0], // bin 92
    [-20.0, 80.0], // bin 93
    [-20.0, 90.0], // bin 94
    [-20.0, 100.0], // bin 95
    [-10.0, -60.0], // bin 96
    [-10.0, -50.0], // bin 97
    [-10.0, -40.0], // bin 98
    [-10.0, -30.0], // bin 99
    [-10.0, -20.0], // bin 100
    [-10.0, -10.0], // bin 101
    [-10.0, 0.0], // bin 102
    [-10.0, 10.0], // bin 103
    [-10.0, 20.0], // bin 104
    [-10.0, 30.0], // bin 105
    [-10.0, 40.0], // bin 106
    [-10.0, 50.0], // bin 107
    [-10.0, 60.0], // bin 108
    [-10.0, 70.0], // bin 109
    [-10.0, 80.0], // bin 110
    [-10.0, 90.0], // bin 111
    [-10.0, 100.0], // bin 112
    [0.0, -70.0], // bin 113
    [0.0, -60.0], // bin 114
    [0.0, -50.0], // bin 115
    [0.0, -40.0], // bin 116
    [0.0, -30.0], // bin 117
    [0.0, -20.0], // bin 118
    [0.0, -10.0], // bin 119
    [0.0, 0.0], // bin 120
    [0.0, 10.0], // bin 121
    [0.0, 20.0], // bin 122
    [0.0, 30.0], // bin 123
    [0.0, 40.0], // bin 124
    [0.0, 50.0], // bin 125
    [0.0, 60.0], // bin 126
    [0.0, 70.0], // bin 127
    [0.0, 80.0], // bin 128
    [0.0, 90.0], // bin 129
    [0.0, 100.0], // bin 130
    [10.0, -80.0], // bin 131
    [10.0, -70.0], // bin 132
    [10.0, -60.0], // bin 133
    [10.0, -50.0], // bin 134
    [10.0, -40.0], // bin 135
    [10.0, -30.0], // bin 136
    [10.0, -20.0], // bin 137
    [10.0, -10.0], // bin 138
    [10.0, 0.0], // bin 139
    [10.0, 10.0], // bin 140
    [10.0, 20.0], // bin 141
    [10.0, 30.0], // bin 142
    [10.0, 40.0], // bin 143
    [10.0, 50.0], // bin 144
    [10.0, 60.0], // bin 145
    [10.0, 70.0], // bin 146
    [10.0, 80.0], // bin 147
    [10.0, 90.0], // bin 148
    [20.0, -80.0], // bin 149
    [20.0, -70.0], // bin 150
    [20.0, -60.0], // bin 151
    [20.0, -50.0], // bin 152
    [20.0, -40.0], // bin 153
    [20.0, -30.0], // bin 154
    [20.0, -20.0], // bin 155
    [20.0, -10.0], // bin 156
    [20.0, 0.0], // bin 157
    [20.0, 10.0], // bin 158
    [20.0, 20.0], // bin 159
    [20.0, 30.0], // bin 160
    [20.0, 40.0], // bin 161
    [20.0, 50.0], // bin 162
    [20.0, 60.0], // bin 163
    [20.0, 70.0], // bin 164
    [20.0, 80.0], // bin 165
    [20.0, 90.0], // bin 166
    [30.0, -90.0], // bin 167
    [30.0, -80.0], // bin 168
    [30.0, -70.0], // bin 169
    [30.0, -60.0], // bin 170
    [30.0, -50.0], // bin 171
    [30.0, -40.0], // bin 172
    [30.0, -30.0], // bin 173
    [30.0, -20.0], // bin 174
    [30.0, -10.0], // bin 175
    [30.0, 0.0], // bin 176
    [30.0, 10.0], // bin 177
    [30.0, 20.0], // bin 178
    [30.0, 30.0], // bin 179
    [30.0, 40.0], // bin 180
    [30.0, 50.0], // bin 181
    [30.0, 60.0], // bin 182
    [30.0, 70.0], // bin 183
    [30.0, 80.0], // bin 184
    [30.0, 90.0], // bin 185
    [40.0, -100.0], // bin 186
    [40.0, -90.0], // bin 187
    [40.0, -80.0], // bin 188
    [40.0, -70.0], // bin 189
    [40.0, -60.0], // bin 190
    [40.0, -50.0], // bin 191
    [40.0, -40.0], // bin 192
    [40.0, -30.0], // bin 193
    [40.0, -20.0], // bin 194
    [40.0, -10.0], // bin 195
    [40.0, 0.0], // bin 196
    [40.0, 10.0], // bin 197
    [40.0, 20.0], // bin 198
    [40.0, 30.0], // bin 199
    [40.0, 40.0], // bin 200
    [40.0, 50.0], // bin 201
    [40.0, 60.0], // bin 202
    [40.0, 70.0], // bin 203
    [40.0, 80.0], // bin 204
    [40.0, 90.0], // bin 205
    [50.0, -100.0], // bin 206
    [50.0, -90.0], // bin 207
    [50.0, -80.0], // bin 208
    [50.0, -70.0], // bin 209
    [50.0, -60.0], // bin 210
    [50.0, -50.0], // bin 211
    [50.0, -40.0], // bin 212
    [50.0, -30.0], // bin 213
    [50.0, -20.0], // bin 214
    [50.0, -10.0], // bin 215
    [50.0, 0.0], // bin 216
    [50.0, 10.0], // bin 217
    [50.0, 20.0], // bin 218
    [50.0, 30.0], // bin 219
    [50.0, 40.0], // bin 220
    [50.0, 50.0], // bin 221
    [50.0, 60.0], // bin 222
    [50.0, 70.0], // bin 223
    [50.0, 80.0], // bin 224
    [60.0, -110.0], // bin 225
    [60.0, -100.0], // bin 226
    [60.0, -90.0], // bin 227
    [60.0, -80.0], // bin 228
    [60.0, -70.0], // bin 229
    [60.0, -60.0], // bin 230
    [60.0, -50.0], // bin 231
    [60.0, -40.0], // bin 232
    [60.0, -30.0], // bin 233
    [60.0, -20.0], // bin 234
    [60.0, -10.0], // bin 235
    [60.0, 0.0], // bin 236
    [60.0, 10.0], // bin 237
    [60.0, 20.0], // bin 238
    [60.0, 30.0], // bin 239
    [60.0, 40.0], // bin 240
    [60.0, 50.0], // bin 241
    [60.0, 60.0], // bin 242
    [60.0, 70.0], // bin 243
    [60.0, 80.0], // bin 244
    [70.0, -110.0], // bin 245
    [70.0, -100.0], // bin 246
    [70.0, -90.0], // bin 247
    [70.0, -80.0], // bin 248
    [70.0, -70.0], // bin 249
    [70.0, -60.0], // bin 250
    [70.0, -50.0], // bin 251
    [70.0, -40.0], // bin 252
    [70.0, -30.0], // bin 253
    [70.0, -20.0], // bin 254
    [70.0, -10.0], // bin 255
    [70.0, 0.0], // bin 256
    [70.0, 10.0], // bin 257
    [70.0, 20.0], // bin 258
    [70.0, 30.0], // bin 259
    [70.0, 40.0], // bin 260
    [70.0, 50.0], // bin 261
    [70.0, 60.0], // bin 262
    [70.0, 70.0], // bin 263
    [70.0, 80.0], // bin 264
    [80.0, -110.0], // bin 265
    [80.0, -100.0], // bin 266
    [80.0, -90.0], // bin 267
    [80.0, -80.0], // bin 268
    [80.0, -70.0], // bin 269
    [80.0, -60.0], // bin 270
    [80.0, -50.0], // bin 271
    [80.0, -40.0], // bin 272
    [80.0, -30.0], // bin 273
    [80.0, -20.0], // bin 274
    [80.0, -10.0], // bin 275
    [80.0, 0.0], // bin 276
    [80.0, 10.0], // bin 277
    [80.0, 20.0], // bin 278
    [80.0, 30.0], // bin 279
    [80.0, 40.0], // bin 280
    [80.0, 50.0], // bin 281
    [80.0, 60.0], // bin 282
    [80.0, 70.0], // bin 283
    [90.0, -110.0], // bin 284
    [90.0, -100.0], // bin 285
    [90.0, -90.0], // bin 286
    [90.0, -80.0], // bin 287
    [90.0, -70.0], // bin 288
    [90.0, -60.0], // bin 289
    [90.0, -50.0], // bin 290
    [90.0, -40.0], // bin 291
    [90.0, -30.0], // bin 292
    [90.0, -20.0], // bin 293
    [90.0, -10.0], // bin 294
    [90.0, 0.0], // bin 295
    [90.0, 10.0], // bin 296
    [90.0, 20.0], // bin 297
    [90.0, 30.0], // bin 298
    [90.0, 40.0], // bin 299
    [90.0, 50.0], // bin 300
    [90.0, 60.0], // bin 301
    [90.0, 70.0], // bin 302
    [100.0, -90.0], // bin 303
    [100.0, -80.0], // bin 304
    [100.0, -70.0], // bin 305
    [100.0, -60.0], // bin 306
    [100.0, -50.0], // bin 307
    [100.0, -40.0], // bin 308
    [100.0, -30.0], // bin 309
    [100.0, -20.0], // bin 310
    [100.0, -10.0], // bin 311
    [100.0, 0.0], // bin 312
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

/// Empirical ImageNet prior distribution p(q) over 313 in-gamut color bins from Zhang et al. (2016).
pub const OFFICIAL_PRIOR_PROBS: [f32; NUM_COLOR_BINS] = [
    4.8863707747e-08, // bin 0
    1.3851789925e-07, // bin 1
    5.7657532963e-07, // bin 2
    2.6393382123e-06, // bin 3
    9.5031242780e-07, // bin 4
    3.5669146618e-08, // bin 5
    2.5370121821e-07, // bin 6
    7.5253128553e-07, // bin 7
    1.9617667000e-06, // bin 8
    3.0148059201e-06, // bin 9
    4.8864458370e-06, // bin 10
    5.6405340251e-06, // bin 11
    1.2942927344e-06, // bin 12
    9.3667982131e-08, // bin 13
    6.3539144708e-07, // bin 14
    2.2464141508e-06, // bin 15
    5.2555358040e-06, // bin 16
    8.7660741422e-06, // bin 17
    1.3409165456e-05, // bin 18
    1.8277922913e-05, // bin 19
    1.6072496429e-05, // bin 20
    7.2220233500e-06, // bin 21
    1.1914289047e-06, // bin 22
    4.1888778153e-07, // bin 23
    1.8878667101e-06, // bin 24
    5.5625712200e-06, // bin 25
    1.3943089012e-05, // bin 26
    2.4079202602e-05, // bin 27
    3.4740368256e-05, // bin 28
    4.8228069593e-05, // bin 29
    6.8446653875e-05, // bin 30
    7.2276376839e-05, // bin 31
    4.2962515738e-05, // bin 32
    1.6485100269e-05, // bin 33
    3.0256843434e-06, // bin 34
    7.7380516600e-07, // bin 35
    1.6446115065e-05, // bin 36
    4.3874108625e-05, // bin 37
    5.8734062799e-05, // bin 38
    8.7231631517e-05, // bin 39
    1.1877063181e-04, // bin 40
    1.6708896103e-04, // bin 41
    2.6325249387e-04, // bin 42
    3.3851845263e-04, // bin 43
    2.6034702955e-04, // bin 44
    1.2697504458e-04, // bin 45
    4.0896175449e-05, // bin 46
    7.2281756353e-06, // bin 47
    1.9969387284e-07, // bin 48
    8.9209522189e-07, // bin 49
    2.1768106293e-05, // bin 50
    1.1251304913e-04, // bin 51
    2.1637953763e-04, // bin 52
    2.6603057687e-04, // bin 53
    3.3227612515e-04, // bin 54
    5.0588613099e-04, // bin 55
    9.3073664352e-04, // bin 56
    1.3746057427e-03, // bin 57
    1.2239982554e-03, // bin 58
    6.8494196668e-04, // bin 59
    2.6633175002e-04, // bin 60
    7.8126915909e-05, // bin 61
    1.5124833395e-05, // bin 62
    6.9190901500e-07, // bin 63
    7.8444240721e-07, // bin 64
    2.6044292351e-05, // bin 65
    1.7322590351e-04, // bin 66
    4.2618323148e-04, // bin 67
    6.9435143822e-04, // bin 68
    9.1010660749e-04, // bin 69
    1.3639643455e-03, // bin 70
    2.6561395667e-03, // bin 71
    4.2854946772e-03, // bin 72
    4.1220795164e-03, // bin 73
    2.4615193363e-03, // bin 74
    1.0569683480e-03, // bin 75
    3.6527998147e-04, // bin 76
    1.1669796003e-04, // bin 77
    3.9337946491e-05, // bin 78
    5.5155688258e-06, // bin 79
    2.3173647413e-05, // bin 80
    2.1886435961e-04, // bin 81
    7.0024952197e-04, // bin 82
    1.3998257111e-03, // bin 83
    2.4676962673e-03, // bin 84
    4.1254400350e-03, // bin 85
    6.6595268433e-03, // bin 86
    9.4814535077e-03, // bin 87
    9.4715196142e-03, // bin 88
    6.0142344112e-03, // bin 89
    2.7363266005e-03, // bin 90
    1.0593163895e-03, // bin 91
    4.2481314778e-04, // bin 92
    2.0660897049e-04, // bin 93
    1.1251665965e-04, // bin 94
    1.8234379820e-05, // bin 95
    1.2980620738e-05, // bin 96
    1.9845964794e-04, // bin 97
    1.0234851993e-03, // bin 98
    3.0212100536e-03, // bin 99
    6.8347323734e-03, // bin 100
    1.9154820690e-02, // bin 101
    4.9108907012e-02, // bin 102
    3.7343616435e-02, // bin 103
    2.1541026271e-02, // bin 104
    1.1957624878e-02, // bin 105
    5.3749602023e-03, // bin 106
    2.2204508311e-03, // bin 107
    1.0022836851e-03, // bin 108
    5.3452216418e-04, // bin 109
    2.9887577233e-04, // bin 110
    1.1331255319e-04, // bin 111
    7.1443113838e-06, // bin 112
    5.0989109739e-06, // bin 113
    1.1701696763e-04, // bin 114
    7.9160877599e-04, // bin 115
    2.8516033126e-03, // bin 116
    6.6310600318e-03, // bin 117
    1.5282381400e-02, // bin 118
    6.4202922178e-02, // bin 119
    2.0123401136e-01, // bin 120
    1.0829850924e-01, // bin 121
    4.1507638341e-02, // bin 122
    1.6551475893e-02, // bin 123
    6.1702533765e-03, // bin 124
    2.4903778692e-03, // bin 125
    1.1721089920e-03, // bin 126
    6.2621370802e-04, // bin 127
    3.1961592914e-04, // bin 128
    7.2517444365e-05, // bin 129
    1.5689404911e-06, // bin 130
    1.3685372060e-06, // bin 131
    4.8093222371e-05, // bin 132
    3.5480824042e-04, // bin 133
    1.1985967341e-03, // bin 134
    2.4265588600e-03, // bin 135
    3.7785252625e-03, // bin 136
    6.6116848446e-03, // bin 137
    1.9560146439e-02, // bin 138
    5.3108429978e-02, // bin 139
    4.8359484234e-02, // bin 140
    3.2731244957e-02, // bin 141
    1.7650565434e-02, // bin 142
    7.5748316206e-03, // bin 143
    3.0766248871e-03, // bin 144
    1.3164294317e-03, // bin 145
    6.0891974483e-04, // bin 146
    2.5108509510e-04, // bin 147
    2.9883528931e-05, // bin 148
    1.6299051135e-05, // bin 149
    1.5067354956e-04, // bin 150
    4.4874316477e-04, // bin 151
    8.1125908993e-04, // bin 152
    1.0288900844e-03, // bin 153
    1.0708981263e-03, // bin 154
    1.1956905758e-03, // bin 155
    2.1915721098e-03, // bin 156
    5.4147040624e-03, // bin 157
    1.0513671789e-02, // bin 158
    1.2240769775e-02, // bin 159
    9.8372000955e-03, // bin 160
    5.7559641755e-03, // bin 161
    2.7884107054e-03, // bin 162
    1.2290707607e-03, // bin 163
    5.1876712920e-04, // bin 164
    1.6017383751e-04, // bin 165
    8.7330852024e-06, // bin 166
    3.6774120509e-06, // bin 167
    6.0186228985e-05, // bin 168
    1.9501417189e-04, // bin 169
    3.2902751138e-04, // bin 170
    3.9210896150e-04, // bin 171
    3.3646720231e-04, // bin 172
    2.8758073727e-04, // bin 173
    3.5586498154e-04, // bin 174
    6.0002503836e-04, // bin 175
    1.3030295670e-03, // bin 176
    2.9426484777e-03, // bin 177
    4.3551007516e-03, // bin 178
    4.2325310597e-03, // bin 179
    3.1058469788e-03, // bin 180
    1.8508245974e-03, // bin 181
    9.2759074394e-04, // bin 182
    3.9652177971e-04, // bin 183
    8.9029729897e-05, // bin 184
    2.1696851892e-06, // bin 185
    5.7128617462e-07, // bin 186
    2.0362583501e-05, // bin 187
    9.3823396550e-05, // bin 188
    1.5340057213e-04, // bin 189
    1.7328958194e-04, // bin 190
    1.3987396172e-04, // bin 191
    1.0541354920e-04, // bin 192
    1.2405359997e-04, // bin 193
    1.6873385425e-04, // bin 194
    2.7270080572e-04, // bin 195
    4.9487192411e-04, // bin 196
    9.9326691115e-04, // bin 197
    1.8447476583e-03, // bin 198
    2.3181411138e-03, // bin 199
    1.8082609417e-03, // bin 200
    1.1588441192e-03, // bin 201
    6.2956658578e-04, // bin 202
    2.6736759181e-04, // bin 203
    3.9811007531e-05, // bin 204
    4.2198563872e-07, // bin 205
    4.9844336959e-06, // bin 206
    4.4112686493e-05, // bin 207
    8.2753403059e-05, // bin 208
    7.9399330111e-05, // bin 209
    5.4345107332e-05, // bin 210
    4.1244828875e-05, // bin 211
    4.8651839248e-05, // bin 212
    6.6812036672e-05, // bin 213
    9.2308607828e-05, // bin 214
    1.4897956002e-04, // bin 215
    2.4439293973e-04, // bin 216
    4.3272214429e-04, // bin 217
    8.1805393651e-04, // bin 218
    1.3919068255e-03, // bin 219
    1.5471231517e-03, // bin 220
    9.4595873868e-04, // bin 221
    4.4719296811e-04, // bin 222
    1.6897661876e-04, // bin 223
    1.5442962277e-05, // bin 224
    8.5477346657e-07, // bin 225
    1.6792760109e-05, // bin 226
    4.3084665899e-05, // bin 227
    3.9019168117e-05, // bin 228
    2.0858331173e-05, // bin 229
    1.5273487685e-05, // bin 230
    1.7686113416e-05, // bin 231
    2.5990175313e-05, // bin 232
    3.6372534708e-05, // bin 233
    5.4060590837e-05, // bin 234
    8.4976427594e-05, // bin 235
    1.2596453521e-04, // bin 236
    1.9437726636e-04, // bin 237
    3.5492255209e-04, // bin 238
    6.6003612857e-04, // bin 239
    1.0244340094e-03, // bin 240
    9.7327788230e-04, // bin 241
    4.2081802533e-04, // bin 242
    1.0954374229e-04, // bin 243
    5.6273341680e-06, // bin 244
    5.8447861577e-06, // bin 245
    2.4636848590e-05, // bin 246
    1.7884281746e-05, // bin 247
    7.6380278412e-06, // bin 248
    5.2752555203e-06, // bin 249
    5.8700419414e-06, // bin 250
    9.4484795244e-06, // bin 251
    1.4146463271e-05, // bin 252
    1.8974230722e-05, // bin 253
    2.8331754973e-05, // bin 254
    4.1785819890e-05, // bin 255
    5.5343161170e-05, // bin 256
    7.3289729343e-05, // bin 257
    1.1744556019e-04, // bin 258
    2.1751016064e-04, // bin 259
    3.7345979305e-04, // bin 260
    5.2329000386e-04, // bin 261
    3.9474306338e-04, // bin 262
    9.1096148138e-05, // bin 263
    2.3617176618e-06, // bin 264
    7.4368502679e-06, // bin 265
    9.7934989830e-06, // bin 266
    3.5609264627e-06, // bin 267
    2.2353162599e-06, // bin 268
    2.2270245581e-06, // bin 269
    2.8910121388e-06, // bin 270
    5.4278258841e-06, // bin 271
    6.5786654855e-06, // bin 272
    8.0894771294e-06, // bin 273
    1.0783273103e-05, // bin 274
    1.4315098924e-05, // bin 275
    1.6177370730e-05, // bin 276
    1.7617238693e-05, // bin 277
    2.2242031768e-05, // bin 278
    3.4304382134e-05, // bin 279
    5.6733849939e-05, // bin 280
    9.2410690239e-05, // bin 281
    1.3405434266e-04, // bin 282
    5.9185829582e-05, // bin 283
    3.7875506600e-07, // bin 284
    4.5824286187e-07, // bin 285
    4.8146570129e-07, // bin 286
    6.9272289464e-07, // bin 287
    1.0198476268e-06, // bin 288
    2.3483049608e-06, // bin 289
    2.9254767964e-06, // bin 290
    2.8335363100e-06, // bin 291
    2.4896257134e-06, // bin 292
    2.0816893034e-06, // bin 293
    2.1228095538e-06, // bin 294
    1.6643559637e-06, // bin 295
    1.2891309500e-06, // bin 296
    1.1141332002e-06, // bin 297
    1.2292729125e-06, // bin 298
    1.5969706143e-06, // bin 299
    2.4648962685e-06, // bin 300
    5.0724564192e-06, // bin 301
    3.6597350249e-06, // bin 302
    6.0714447539e-09, // bin 303
    3.6555956542e-08, // bin 304
    2.9015751049e-07, // bin 305
    1.4043046237e-06, // bin 306
    6.0051156443e-07, // bin 307
    2.6490093993e-07, // bin 308
    1.2090215483e-07, // bin 309
    4.7852691004e-08, // bin 310
    2.8815293158e-08, // bin 311
    1.4503950353e-08, // bin 312
];


/// Compute Zhang et al. (2016) class rebalancing weights w(q) from the official empirical ImageNet prior.
pub fn official_class_weights(lambda: f32) -> [f32; NUM_COLOR_BINS] {
    let q = NUM_COLOR_BINS as f32;
    let mut raw_w = [0.0f32; NUM_COLOR_BINS];
    let mut expected_w = 0.0f32;

    for i in 0..NUM_COLOR_BINS {
        let p = OFFICIAL_PRIOR_PROBS[i];
        let smoothed = (1.0 - lambda) * p + (lambda / q);
        let w = 1.0 / smoothed;
        raw_w[i] = w;
        expected_w += p * w;
    }

    let norm = 1.0 / expected_w.max(1e-8);
    let mut final_w = [0.0f32; NUM_COLOR_BINS];
    for i in 0..NUM_COLOR_BINS {
        final_w[i] = raw_w[i] * norm;
    }
    final_w
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
        logits_to_ab_annealed(self.forward(x), temperature, chroma_boost)
    }
}

/// Convert raw 313-bin output logits [1, 313, 64, 64] to continuous (a, b) planes
/// using temperature-annealed expectation.
pub fn logits_to_ab_annealed<B: Backend>(
    logits: Tensor<B, 4>,
    temperature: f32,
    chroma_boost: f32,
) -> (Vec<f32>, Vec<f32>) {
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

/// Common trait implemented by both CNN and Vision Transformer colorizer architectures.
pub trait Colorizer<B: Backend> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4>;
    fn predict_ab_annealed(&self, x: Tensor<B, 4>, temperature: f32, chroma_boost: f32) -> (Vec<f32>, Vec<f32>);
}

impl<B: Backend> Colorizer<B> for ColorizerModel<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(x)
    }

    fn predict_ab_annealed(&self, x: Tensor<B, 4>, temperature: f32, chroma_boost: f32) -> (Vec<f32>, Vec<f32>) {
        self.predict_ab_annealed(x, temperature, chroma_boost)
    }
}

// ─── Non-Autoregressive Vision Transformer (ViT) Architecture ─────────────

#[inline]
fn silu_act<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * burn::tensor::activation::sigmoid(x)
}

/// RMSNorm for Vision Transformer tokens.
#[derive(Config, Debug)]
pub struct ViTRMSNormConfig {
    pub size: usize,
    #[config(default = 1e-6)]
    pub eps: f64,
}

impl ViTRMSNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ViTRMSNorm<B> {
        let weight = burn::module::Param::from_tensor(Tensor::ones([self.size], device));
        ViTRMSNorm {
            weight,
            eps: self.eps,
        }
    }
}

#[derive(Module, Debug)]
pub struct ViTRMSNorm<B: Backend> {
    weight: burn::module::Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> ViTRMSNorm<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let rms = (x.clone().powf_scalar(2.0).mean_dim(D - 1) + self.eps).sqrt();
        (x / rms) * self.weight.val().unsqueeze()
    }
}

/// SwiGLU Feed-Forward Network for Vision Transformer.
#[derive(Config, Debug)]
pub struct ViTMLPConfig {
    pub d_model: usize,
    pub d_hidden: usize,
}

impl ViTMLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ViTMLP<B> {
        ViTMLP {
            w1: LinearConfig::new(self.d_model, self.d_hidden)
                .with_initializer(Initializer::KaimingUniform {
                    gain: 1.0 / f64::sqrt(3.0),
                    fan_out_only: false,
                })
                .with_bias(false)
                .init(device),
            w2: LinearConfig::new(self.d_hidden, self.d_model)
                .with_initializer(Initializer::KaimingUniform {
                    gain: 1.0 / f64::sqrt(3.0),
                    fan_out_only: false,
                })
                .with_bias(false)
                .init(device),
            w3: LinearConfig::new(self.d_model, self.d_hidden)
                .with_initializer(Initializer::KaimingUniform {
                    gain: 1.0 / f64::sqrt(3.0),
                    fan_out_only: false,
                })
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct ViTMLP<B: Backend> {
    w1: Linear<B>,
    w2: Linear<B>,
    w3: Linear<B>,
}

impl<B: Backend> ViTMLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x1 = self.w1.forward(x.clone());
        let silu_x1 = silu_act(x1);
        let x3 = self.w3.forward(x);
        self.w2.forward(silu_x1 * x3)
    }
}

/// Bidirectional Multi-Head Self-Attention for Vision Patches.
#[derive(Config, Debug)]
pub struct ViTSelfAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
}

impl ViTSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ViTSelfAttention<B> {
        assert_eq!(self.d_model % self.n_heads, 0, "d_model must be divisible by n_heads");
        let head_dim = self.d_model / self.n_heads;
        ViTSelfAttention {
            q: LinearConfig::new(self.d_model, self.d_model)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .with_bias(false)
                .init(device),
            k: LinearConfig::new(self.d_model, self.d_model)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .with_bias(false)
                .init(device),
            v: LinearConfig::new(self.d_model, self.d_model)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .with_bias(false)
                .init(device),
            o: LinearConfig::new(self.d_model, self.d_model)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .with_bias(false)
                .init(device),
            n_heads: self.n_heads,
            head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct ViTSelfAttention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    n_heads: usize,
    head_dim: usize,
}

impl<B: Backend> ViTSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        let scale = 1.0 / (self.head_dim as f64).sqrt();

        let reshape = |t: Tensor<B, 3>| {
            t.reshape([batch, seq, self.n_heads, self.head_dim])
                .swap_dims(1, 2)
        };

        let q = reshape(self.q.forward(x.clone())) * scale;
        let k = reshape(self.k.forward(x.clone()));
        let v = reshape(self.v.forward(x));

        let qk = q.matmul(k.transpose());
        let w = burn::tensor::activation::softmax(qk, 3);
        let out = w.matmul(v)
            .swap_dims(1, 2)
            .flatten(2, 3);

        self.o.forward(out)
    }
}

/// Vision Transformer Block with Pre-RMSNorm and Residuals.
#[derive(Config, Debug)]
pub struct ViTBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_hidden: usize,
}

impl ViTBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ViTBlock<B> {
        ViTBlock {
            attn_norm: ViTRMSNormConfig::new(self.d_model).init(device),
            attn: ViTSelfAttentionConfig::new(self.d_model, self.n_heads).init(device),
            mlp_norm: ViTRMSNormConfig::new(self.d_model).init(device),
            mlp: ViTMLPConfig::new(self.d_model, self.d_hidden).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct ViTBlock<B: Backend> {
    attn_norm: ViTRMSNorm<B>,
    attn: ViTSelfAttention<B>,
    mlp_norm: ViTRMSNorm<B>,
    mlp: ViTMLP<B>,
}

impl<B: Backend> ViTBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let norm_x = self.attn_norm.forward(x.clone());
        let x = x + self.attn.forward(norm_x);

        let norm_x = self.mlp_norm.forward(x.clone());
        x + self.mlp.forward(norm_x)
    }
}

/// Precomputed 2D Sine-Cosine Positional Embedding for a HxW grid.
pub fn get_2d_sincos_pos_embed(grid_size: usize, d_model: usize) -> Vec<f32> {
    assert_eq!(d_model % 2, 0, "d_model must be even for 2D sincos pos embed");
    let d_feat = d_model / 2;
    let num_tokens = grid_size * grid_size;
    let mut embed = vec![0.0f32; num_tokens * d_model];

    for y in 0..grid_size {
        for x in 0..grid_size {
            let token_idx = y * grid_size + x;
            let offset = token_idx * d_model;

            // 1D sincos for y
            for i in 0..(d_feat / 2) {
                let omega = 1.0 / (10000.0f64.powf((2 * i) as f64 / d_feat as f64) as f32);
                let val_sin = (y as f32 * omega).sin();
                let val_cos = (y as f32 * omega).cos();
                embed[offset + 2 * i] = val_sin;
                embed[offset + 2 * i + 1] = val_cos;
            }

            // 1D sincos for x
            for i in 0..(d_feat / 2) {
                let omega = 1.0 / (10000.0f64.powf((2 * i) as f64 / d_feat as f64) as f32);
                let val_sin = (x as f32 * omega).sin();
                let val_cos = (x as f32 * omega).cos();
                embed[offset + d_feat + 2 * i] = val_sin;
                embed[offset + d_feat + 2 * i + 1] = val_cos;
            }
        }
    }
    embed
}

/// Configuration for the Non-Autoregressive Vision Transformer (ViT) Colorizer.
#[derive(Config, Debug)]
pub struct TransformerColorizerConfig {
    #[config(default = 256)]
    pub d_model: usize,

    #[config(default = 8)]
    pub n_heads: usize,

    #[config(default = 6)]
    pub num_layers: usize,

    #[config(default = 680)]
    pub d_hidden: usize,

    #[config(default = 64)]
    pub stem_c1: usize,

    #[config(default = 128)]
    pub stem_c2: usize,
}

impl TransformerColorizerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerColorizerModel<B> {
        let c1 = self.stem_c1;
        let c2 = self.stem_c2;
        let d = self.d_model;

        // Stage 1: 256x256 (1 ch) -> 128x128 (c1 ch)
        let stem_conv1 = Conv2dConfig::new([1, c1], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Stage 2: 128x128 (c1 ch) -> 64x64 (c2 ch) -- provides high-resolution edge skip connection
        let stem_conv2 = Conv2dConfig::new([c1, c2], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Stage 3: 64x64 (c2 ch) -> 32x32 (d_model ch)
        let stem_conv3 = Conv2dConfig::new([c2, d], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // 2D Positional Embeddings for 32x32 grid (1024 tokens)
        let pos_vec = get_2d_sincos_pos_embed(32, d);
        let pos_tensor = Tensor::from_floats(
            TensorData::new(pos_vec, [1, 1024, d]),
            device,
        );
        let pos_embed = Param::from_tensor(pos_tensor);

        // Transformer blocks
        let block_cfg = ViTBlockConfig::new(d, self.n_heads, self.d_hidden);
        let blocks = (0..self.num_layers)
            .map(|_| block_cfg.init(device))
            .collect();

        // Decoder: Upsample 32x32 (d_model) -> 64x64 (c2)
        let up_conv = ConvTranspose2dConfig::new([d, c2], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .init(device);

        // Refinement after high-res skip fusion: 64x64 (c2 -> c2)
        let refine_conv = Conv2dConfig::new([c2, c2], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        // Final output projection: 64x64 (c2 -> 313)
        let out_conv = Conv2dConfig::new([c2, NUM_COLOR_BINS], [1, 1])
            .init(device);

        TransformerColorizerModel {
            stem_conv1,
            stem_conv2,
            stem_conv3,
            pos_embed,
            blocks,
            up_conv,
            refine_conv,
            out_conv,
            config: Ignored(self.clone()),
        }
    }
}

/// The Non-Autoregressive Vision Transformer (ViT) Colorizer Model.
///
/// Combines a multi-scale convolutional stem with bidirectional self-attention
/// across 1,024 spatial patches to provide complete, global semantic understanding,
/// and fuses early high-resolution features to preserve razor-sharp luminance edges.
#[derive(Module, Debug)]
pub struct TransformerColorizerModel<B: Backend> {
    stem_conv1: Conv2d<B>,
    stem_conv2: Conv2d<B>,
    stem_conv3: Conv2d<B>,
    pos_embed: Param<Tensor<B, 3>>,
    blocks: Vec<ViTBlock<B>>,
    up_conv: ConvTranspose2d<B>,
    refine_conv: Conv2d<B>,
    out_conv: Conv2d<B>,
    pub config: Ignored<TransformerColorizerConfig>,
}

impl<B: Backend> TransformerColorizerModel<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, _, _, _] = x.dims();

        // 1. Multi-scale early conv stem
        let h1 = silu_act(self.stem_conv1.forward(x));
        let f64 = silu_act(self.stem_conv2.forward(h1));
        let h3 = silu_act(self.stem_conv3.forward(f64.clone()));

        // 2. Token sequence: [B, d, 32, 32] -> [B, 1024, d]
        let [_, d, h, w] = h3.dims();
        let seq_len = h * w;
        let mut tokens = h3
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .reshape([b, seq_len, d]);

        // Add 2D Positional Embeddings
        tokens = tokens + self.pos_embed.val();

        // 3. Global bidirectional self-attention transformer blocks
        for block in &self.blocks {
            tokens = block.forward(tokens);
        }

        // 4. Reshape tokens back to spatial grid: [B, 1024, d] -> [B, d, 32, 32]
        let tokens_2d = tokens
            .reshape([b, h, w, d])
            .swap_dims(2, 3)
            .swap_dims(1, 2);

        // 5. Decoder: Transposed conv upsampling to 64x64
        let up = silu_act(self.up_conv.forward(tokens_2d));

        // 6. High-resolution edge skip fusion: blend semantic tokens with crisp grayscale edges
        let fused = up + f64;

        // 7. Refinement & output projection
        let refined = silu_act(self.refine_conv.forward(fused));
        self.out_conv.forward(refined)
    }

    pub fn predict_ab_annealed(
        &self,
        x: Tensor<B, 4>,
        temperature: f32,
        chroma_boost: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        logits_to_ab_annealed(self.forward(x), temperature, chroma_boost)
    }
}

impl<B: Backend> Colorizer<B> for TransformerColorizerModel<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(x)
    }

    fn predict_ab_annealed(&self, x: Tensor<B, 4>, temperature: f32, chroma_boost: f32) -> (Vec<f32>, Vec<f32>) {
        self.predict_ab_annealed(x, temperature, chroma_boost)
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
        let model = match model.load_file(&path, &recorder, &device) {
            Ok(m) => m,
            Err(e) => {
                println!("Skipping test_sample0_predictions: {e}");
                return;
            }
        };

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

    #[test]
    fn test_transformer_colorizer_forward() {
        let device = Default::default();
        let config = TransformerColorizerConfig::new()
            .with_d_model(128)
            .with_n_heads(4)
            .with_num_layers(2)
            .with_d_hidden(256)
            .with_stem_c1(32)
            .with_stem_c2(64);

        let model: TransformerColorizerModel<B> = config.init(&device);
        let in_tensor = Tensor::<B, 4>::zeros([1, 1, 256, 256], &device);
        let logits = model.forward(in_tensor);
        assert_eq!(logits.dims(), [1, NUM_COLOR_BINS, 64, 64]);
    }
}
