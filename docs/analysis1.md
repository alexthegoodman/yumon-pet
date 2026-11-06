==================================================================
 PASS 1: MODEL WEIGHT FORENSICS
   Model File:         checkpoints/yumon-base-moe\code_256h_16l_8a_512ff_256len_b2_Moe_e8_k2_f32_tied\model.bin
   Shape:              vocab 4096 | embed 256 | ff 512 | 8 routed experts | 16 layers | 8 heads
   Run dir:            checkpoints/yumon-base-moe\code_256h_16l_8a_512ff_256len_b2_Moe_e8_k2_f32_tied
   model.bin:          212.2 MiB, written 45h ago
   Metadata:           epochs_trained=0 final_loss=4.7389755 val_loss=4.571622 tied=true kv_heads=-
   Optimizer:          AdamW (no optimizer section)
==================================================================

=== PARAMETERS ===
  500 tensors, 55615746 floats (212.2 MiB)

=== HEALTH BY GROUP ===
  matrices: COLLAPSED = rank-1 share > max(25%, 3x a random matrix of that shape), RANDOM-LIKE = s1 < 1.1x the MP edge for its RMS
  norm gains start at exactly 1.0, so UNTOUCHED means no update ever reached them
  embedding          1 tensors: 1 COLLAPSED | mean srank 1.5, s1/MP 10.62, rank1 68.8%, rms 0.0512
  token_head         2 tensors: 2 -
  attention         64 tensors: 64 learned | mean srank 9.7, s1/MP 2.69, rank1 11.6%, rms 0.0277
  experts          384 tensors: 384 learned | mean srank 22.3, s1/MP 2.08, rank1 5.1%, rms 0.0236
  router            16 tensors: 14 COLLAPSED, 2 learned | mean srank 1.8, s1/MP 1.81, rank1 56.8%, rms 0.0326
  norm gains        33 tensors: 33 ok

=== ISSUES ===
  [!] embedding.weight: 53% of its energy is the mean row (every token shares one direction; s1/s2 6.4)

=== MOST ANISOTROPIC MATRICES (highest rank-1 share) ===
  embedding.weight                               4096x256 | rms   0.0512 | s1  43.462 s1/s2  6.39 | srank    1.45 | rank1  68.8% | s1/MP 10.62 | mean 52.6% | COLLAPSED
  blocks.5.moe.router.weight                        256x8 | rms   0.0319 | s1   1.129 s1/s2  2.91 | srank    1.64 | rank1  61.1% | s1/MP  1.88 | mean  0.2% | COLLAPSED
  blocks.8.moe.router.weight                        256x8 | rms   0.0319 | s1   1.129 s1/s2  3.04 | srank    1.64 | rank1  61.1% | s1/MP  1.88 | mean  0.2% | COLLAPSED
  blocks.4.moe.router.weight                        256x8 | rms   0.0320 | s1   1.130 s1/s2  2.92 | srank    1.64 | rank1  60.8% | s1/MP  1.87 | mean  0.2% | COLLAPSED
  blocks.9.moe.router.weight                        256x8 | rms   0.0339 | s1   1.191 s1/s2  2.75 | srank    1.66 | rank1  60.4% | s1/MP  1.87 | mean  0.4% | COLLAPSED
  blocks.6.moe.router.weight                        256x8 | rms   0.0320 | s1   1.126 s1/s2  2.81 | srank    1.66 | rank1  60.4% | s1/MP  1.87 | mean  0.1% | COLLAPSED
  blocks.3.moe.router.weight                        256x8 | rms   0.0323 | s1   1.135 s1/s2  2.84 | srank    1.66 | rank1  60.2% | s1/MP  1.86 | mean  0.4% | COLLAPSED
  blocks.7.moe.router.weight                        256x8 | rms   0.0324 | s1   1.136 s1/s2  2.94 | srank    1.67 | rank1  59.9% | s1/MP  1.86 | mean  0.3% | COLLAPSED
  blocks.2.moe.router.weight                        256x8 | rms   0.0308 | s1   1.071 s1/s2  2.76 | srank    1.69 | rank1  59.0% | s1/MP  1.85 | mean  0.1% | COLLAPSED
  blocks.10.moe.router.weight                       256x8 | rms   0.0346 | s1   1.199 s1/s2  2.71 | srank    1.71 | rank1  58.6% | s1/MP  1.84 | mean  0.1% | COLLAPSED
  blocks.14.moe.router.weight                       256x8 | rms   0.0351 | s1   1.195 s1/s2  2.37 | srank    1.77 | rank1  56.6% | s1/MP  1.81 | mean  0.2% | COLLAPSED
  blocks.11.moe.router.weight                       256x8 | rms   0.0344 | s1   1.171 s1/s2  2.49 | srank    1.77 | rank1  56.5% | s1/MP  1.81 | mean  0.2% | COLLAPSED

=== CLOSEST TO RANDOM INIT (lowest s1/MP) ===
  blocks.4.moe.experts.0.w3.weight                256x512 | rms   0.0208 | s1   1.004 s1/s2  1.06 | srank   56.09 | rank1   1.8% | s1/MP  1.25 | mean  0.4% | learned
  blocks.4.moe.experts.0.w1.weight                256x512 | rms   0.0209 | s1   1.035 s1/s2  1.07 | srank   53.29 | rank1   1.9% | s1/MP  1.28 | mean  0.4% | learned
  blocks.5.moe.experts.5.w3.weight                256x512 | rms   0.0209 | s1   1.065 s1/s2  1.04 | srank   50.28 | rank1   2.0% | s1/MP  1.32 | mean  0.4% | learned
  blocks.5.moe.experts.7.w1.weight                256x512 | rms   0.0214 | s1   1.153 s1/s2  1.05 | srank   45.10 | rank1   2.2% | s1/MP  1.40 | mean  0.4% | learned
  blocks.5.moe.experts.3.w1.weight                256x512 | rms   0.0214 | s1   1.179 s1/s2  1.08 | srank   43.37 | rank1   2.3% | s1/MP  1.42 | mean  0.4% | learned
  blocks.5.moe.experts.5.w1.weight                256x512 | rms   0.0209 | s1   1.153 s1/s2  1.13 | srank   43.23 | rank1   2.3% | s1/MP  1.43 | mean  0.4% | learned
  blocks.2.moe.experts.5.w3.weight                256x512 | rms   0.0212 | s1   1.169 s1/s2  1.16 | srank   43.22 | rank1   2.3% | s1/MP  1.43 | mean  0.4% | learned
  blocks.5.moe.experts.3.w3.weight                256x512 | rms   0.0213 | s1   1.191 s1/s2  1.19 | srank   41.93 | rank1   2.4% | s1/MP  1.45 | mean  0.4% | learned
  blocks.6.moe.experts.6.w3.weight                256x512 | rms   0.0214 | s1   1.201 s1/s2  1.15 | srank   41.57 | rank1   2.4% | s1/MP  1.45 | mean  0.4% | learned
  blocks.3.moe.experts.3.w3.weight                256x512 | rms   0.0212 | s1   1.203 s1/s2  1.09 | srank   40.69 | rank1   2.5% | s1/MP  1.47 | mean  0.3% | learned
  blocks.5.moe.experts.7.w3.weight                256x512 | rms   0.0212 | s1   1.206 s1/s2  1.18 | srank   40.36 | rank1   2.5% | s1/MP  1.48 | mean  0.4% | learned
  blocks.3.moe.experts.3.w1.weight                256x512 | rms   0.0213 | s1   1.220 s1/s2  1.13 | srank   40.00 | rank1   2.5% | s1/MP  1.48 | mean  0.4% | learned

=== PER-LAYER SUMMARY ===
  attn: mean stable rank / mean s1-over-MP-edge of q,k,v,o (MP ~1.0 = random init)
  experts: L2 of each routed expert's weights (min-max), pairwise cosine (mean / max), mean s1/MP
  router: per-expert column L2 (min-max), bias range; gains: mean RMSNorm gain per norm in the block
  starved: experts with mean s1/MP < 1.25; collapsed: experts with most matrices COLLAPSED
  layer |  srank s1/MP |       expert L2  cos mean/max s1/MP |    router col bias rng |     gains | starved    collapsed
      0 |   22.2  1.70 |   14.61-20.69   -0.000/0.005   2.69 |  0.420-0.496        - | 0.92 1.04 | -          -
      1 |   11.2  2.40 |   13.37-15.62    0.001/0.005   2.40 |  0.436-0.482        - | 0.97 0.98 | -          -
      2 |    9.4  2.62 |   13.12-14.73    0.002/0.006   1.92 |  0.469-0.518        - | 0.97 0.93 | -          -
      3 |   10.0  2.65 |   13.48-14.21    0.002/0.007   1.78 |  0.493-0.542        - | 0.97 0.91 | -          -
      4 |   11.0  2.49 |   13.22-14.21    0.003/0.006   1.78 |  0.489-0.585        - | 0.97 0.91 | -          -
      5 |   12.7  2.33 |   13.30-14.44    0.002/0.005   1.83 |  0.481-0.565        - | 0.96 0.91 | -          -
      6 |    8.7  2.77 |   13.56-14.59    0.003/0.006   1.83 |  0.467-0.555        - | 0.99 0.91 | -          -
      7 |    7.8  2.89 |   13.81-14.68    0.003/0.006   1.87 |  0.484-0.585        - | 0.98 0.91 | -          -
      8 |    8.3  2.80 |   13.76-14.90    0.003/0.008   1.93 |  0.469-0.557        - | 0.98 0.91 | -          -
      9 |    8.5  2.78 |   13.96-15.43    0.003/0.007   1.90 |  0.498-0.593        - | 0.98 0.90 | -          -
     10 |    7.8  2.88 |   14.11-16.14    0.003/0.010   2.14 |  0.515-0.634        - | 0.99 0.92 | -          -
     11 |    8.9  2.70 |   14.43-16.30    0.003/0.007   2.12 |  0.508-0.617        - | 0.98 0.93 | -          -
     12 |    7.3  2.97 |   14.31-16.16    0.003/0.007   2.18 |  0.514-0.575        - | 0.99 0.94 | -          -
     13 |    7.4  2.97 |   14.10-16.22    0.003/0.009   2.26 |  0.489-0.591        - | 1.00 0.94 | -          -
     14 |    7.1  3.01 |   14.47-17.82    0.002/0.006   2.26 |  0.519-0.599        - | 0.99 0.96 | -          -
     15 |    7.0  3.05 |   14.43-18.45    0.004/0.013   2.45 |  0.504-0.639        - | 0.98 0.98 | -          -

=== EMBEDDING ROWS (embedding.weight) ===
  row L2:            p1 0.433  p10 0.507  p50 0.733  p90 1.114  p99 1.481  max 1.551
  cos(row, mean):    p10 0.376  p50 0.791  p90 0.918   (near 0 = isotropic; near 1 = every token points the same way)
  mean-row L2:       0.594
