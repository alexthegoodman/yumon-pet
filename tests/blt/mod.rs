use super::*;
use burn_cubecl::CubeBackend;
use cubecl::wgpu::WgpuRuntime;

type Gpu = CubeBackend<WgpuRuntime, f32, i32, u32>;

fn yumon_bpe() -> TokenizerKind {
    TokenizerKind::Bpe(crate::brain::bpe::BpeTokenizer::load("yumon_bpe").unwrap())
}

/// Deterministic pseudo-random byte rows (BOS first).
fn random_row(len: usize, seed: u64) -> Vec<u32> {
    let mut state = seed | 1;
    let alphabet = b"ab cd_(x): fn\n12.";
    std::iter::once(BOS).chain((1..len).map(|_| {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        byte_id(alphabet[(state % alphabet.len() as u64) as usize])
    })).collect()
}

/// Next-byte "entropy" that depends only on bytes 0..=q.
fn fake_entropy(bytes: &[u32]) -> Vec<f32> {
    let mut sum = 0u32;
    bytes.iter().map(|&b| { sum = sum.wrapping_mul(31).wrapping_add(b); (sum % 7) as f32 / 2.0 }).collect()
}

fn tiny_config() -> BltModelConfig {
    BltModelConfig::new(32, 2, 64, 1, 2, 48, 2, 96, 2, vec![3, 4], 64, 2).with_dropout(0.0)
}

fn rows_of(inputs: &[Vec<u32>]) -> Vec<ByteRow> {
    inputs.iter().map(|input| {
        let mut targets: Vec<u32> = input[1..].to_vec();
        targets.push(EOS);
        ByteRow { input: input.clone(), targets, tokens: 0, dropped: 0 }
    }).collect()
}

#[test]
fn token_bytes_are_the_decoded_text() {
    let tokenizer = yumon_bpe();
    let table = TokenBytes::new(&tokenizer).unwrap();
    for text in ["pub fn main() {\n    let x = vec![1, 2];\n}", "héllo wörld ✓ 日本 \"q\": [1]", "  \t\r\n  "] {
        let ids = tokenizer.encode(text);
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| table.bytes(id).iter().map(|&b| (b - BYTE_OFFSET) as u8)).collect();
        assert_eq!(String::from_utf8(bytes).unwrap(), tokenizer.decode(&ids), "{text:?}");
    }
    assert_eq!(table.bytes(BOS_TOKEN), &[BOS]);
    assert!(table.bytes(PAD_TOKEN).is_empty());
}

#[test]
fn rows_follow_token_supervision_and_cut_at_token_boundaries() {
    let (a, b, c) = (byte_id(b'a'), byte_id(b'b'), byte_id(b'c'));
    let mut table = vec![Vec::new(); 6];
    table[BOS_TOKEN] = vec![BOS];
    table[EOS_TOKEN] = vec![EOS];
    table[4] = vec![a, b];
    table[5] = vec![c];
    let table = TokenBytes { table };
    let pad = PAD_TOKEN as i32;
    // Prompt token 4 unsupervised, then 5, 4, EOS supervised.
    let ids = [BOS_TOKEN as i32, 4, 5, 4, pad];
    let targets = [pad, 5, 4, EOS_TOKEN as i32, pad];
    let row = table.row(&ids, &targets, 64);
    assert_eq!(row.input, vec![BOS, a, b, c, a, b]);
    assert_eq!(row.targets, vec![PAD, PAD, c, a, b, EOS]);
    assert_eq!((row.tokens, row.dropped), (3, 0));
    // Four bytes: "ab" after BOS a b c does not fit, so 4 and EOS are dropped whole.
    let row = table.row(&ids, &targets, 4);
    assert_eq!(row.input, vec![BOS, a, b]);
    assert_eq!(row.targets, vec![PAD, PAD, c]);
    assert_eq!((row.tokens, row.dropped), (1, 2));
}

#[test]
fn space_patches_split_after_words() {
    let mut bytes = vec![BOS];
    bytes.extend(b"fn foo(x: u32)".iter().map(|&b| byte_id(b)));
    let starts = Patching::Space { max_len: 16 }.starts(&bytes, None);
    let (enc, _, count) = patch_ids(&starts);
    let mut patches = vec![String::new(); count];
    for (p, &b) in bytes.iter().enumerate().skip(1) {
        patches[enc[p]].push((b - BYTE_OFFSET) as u8 as char);
    }
    assert_eq!(patches, ["", "fn ", "foo(", "x:", " u32)"]);
    let starts = Patching::Space { max_len: 2 }.starts(&bytes, None);
    assert!(patch_ids(&starts).2 > count, "max_len splits long patches");
}

#[test]
fn patch_starts_and_reads_are_causal() {
    let modes = [
        Patching::Static { size: 3 },
        Patching::Space { max_len: 5 },
        Patching::Entropy { model: PathBuf::new(), threshold: 1.6, max_len: 6 },
    ];
    for mode in &modes {
        for seed in 1..40 {
            let row = random_row(60, seed);
            let full = mode.starts(&row, Some(&fake_entropy(&row)));
            for k in 1..row.len() {
                let prefix = &row[..k];
                let part = mode.starts(prefix, Some(&fake_entropy(prefix)));
                assert_eq!(part, full[..=k], "{mode:?}: starts through {k} depend on later bytes");
            }
            let (enc, dec, count) = patch_ids(&full);
            assert_eq!(enc[0], 0);
            assert!(enc[1] == 1 && count == enc[row.len() - 1] + 1);
            for p in 0..row.len() {
                assert!(dec[p] <= enc[p]);
                // Every byte of the patch read at p lies at or before p.
                assert!(enc.iter().enumerate().all(|(q, &e)| e != dec[p] || q <= p), "{mode:?}: position {p} reads ahead");
                // ...and it is the last such patch.
                if p + 1 < row.len() { assert_eq!(dec[p] + 1, enc[p + 1]); }
            }
        }
    }
}

#[test]
fn blt_logits_do_not_see_later_bytes() {
    let device = Default::default();
    let config = tiny_config();
    let model = config.init::<Gpu>(&device);
    let patching = Patching::Space { max_len: 4 };
    let a = random_row(48, 7);
    for k in [1, 9, 20, 40] {
        let mut b = a.clone();
        for (i, byte) in b.iter_mut().enumerate().skip(k + 1) { *byte = byte_id(b'z' - (i % 5) as u8); }
        let logits: Vec<Vec<f32>> = [&a, &b].iter().map(|row| {
            let rows = rows_of(&[(*row).clone()]);
            let starts: Vec<_> = rows.iter().map(|r| patching.starts(&r.input, None)).collect();
            let batch = make_batch::<Gpu>(&rows, Some((&starts, &config)), &device);
            model.forward(&batch).narrow(1, 0, k + 1).into_data().convert::<f32>().to_vec().unwrap()
        }).collect();
        let worst = logits[0].iter().zip(&logits[1]).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-4, "positions 0..={k} changed by {worst} when later bytes changed");
    }
}

#[test]
fn blt_batch_rows_match_single_rows() {
    let device = Default::default();
    let config = tiny_config();
    let model = config.init::<Gpu>(&device);
    let patching = Patching::Static { size: 3 };
    let rows = rows_of(&[random_row(30, 3), random_row(17, 4), random_row(5, 5)]);
    let starts: Vec<_> = rows.iter().map(|r| patching.starts(&r.input, None)).collect();
    let together: Vec<f32> = model.forward(&make_batch::<Gpu>(&rows, Some((&starts, &config)), &device))
        .into_data().convert::<f32>().to_vec().unwrap();
    for (r, row) in rows.iter().enumerate() {
        let one = make_batch::<Gpu>(std::slice::from_ref(row), Some((&starts[r..=r], &config)), &device);
        let alone: Vec<f32> = model.forward(&one).into_data().convert::<f32>().to_vec().unwrap();
        let off = r * 30 * BYTE_VOCAB;
        let worst = alone.iter().zip(&together[off..off + alone.len()]).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-4, "row {r}: padding changed logits by {worst}");
    }
}

#[test]
fn blt_trains_and_round_trips_a_checkpoint() {
    let device = Default::default();
    let config = BltModelConfig::new(64, 4, 128, 1, 2, 64, 4, 128, 2, vec![3, 4], 256, 2).with_dropout(0.0);
    let mut model = config.init::<TrainBackend>(&device);
    let patching = Patching::Space { max_len: 6 };
    let mut row = vec![BOS];
    row.extend(b"fn main() { println!(\"hello, world\"); }".iter().map(|&b| byte_id(b)));
    // 12 copies: 480 byte rows and ~100 patch rows, clear of the small-row matmul bug.
    let rows = rows_of(&vec![row; 12]);
    let starts: Vec<_> = rows.iter().map(|r| patching.starts(&r.input, None)).collect();
    let mut optimizer = AdamWConfig::new().init::<TrainBackend, BltModel<TrainBackend>>();
    let mut losses = Vec::new();
    for _ in 0..150 {
        let batch = make_batch::<TrainBackend>(&rows, Some((&starts, &config)), &device);
        let [n, seq] = batch.bytes.dims();
        let logits = model.forward(&batch).reshape([n * seq, BYTE_VOCAB]);
        let loss = crate::brain::train::masked_token_ce(logits, batch.targets.clone(), batch.supervised, 0.0);
        losses.push(loss.clone().inner().into_data().convert::<f32>().to_vec::<f32>().unwrap()[0]);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(3e-3, model, grads);
    }
    assert!(losses[0] > 4.0 && *losses.last().unwrap() < 0.05, "loss {} -> {}", losses[0], losses.last().unwrap());
    // Generation rebuilds patches as it goes; a memorized row must come back out.
    let patcher = Patcher::<Gpu> { patching: patching.clone(), entropy: None };
    let text = generate(&model.valid(), Some(&patcher), Some(&config), "fn main() { pr", 40, 64, 1, &device);
    assert_eq!(text, "intln!(\"hello, world\"); }");

    let dir = std::env::temp_dir().join(format!("yumon-blt-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    save(&model, Some(&optimizer), &Progress::default(), &dir).unwrap();
    assert!(dir.join("optimizer.bin").exists());
    let loaded = config.init::<TrainBackend>(&device)
        .load_file(dir.join("model"), &BinFileRecorder::<FullPrecisionSettings>::new(), &device).unwrap();
    let batch = make_batch::<TrainBackend>(&rows, Some((&starts, &config)), &device);
    let (x, y): (Vec<f32>, Vec<f32>) = (model.forward(&batch).into_data().convert::<f32>().to_vec().unwrap(),
        loaded.forward(&batch).into_data().convert::<f32>().to_vec().unwrap());
    assert_eq!(x, y);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn entropy_patcher_reads_a_byte_lm_causally() {
    let device = Default::default();
    let lm = YumonDecBrainConfig::new(BYTE_VOCAB, TrainingStage::Language).with_embed_dim(32).with_hidden_units(32)
        .with_n_layers(1).with_attn_heads(2).with_ff_dim(64).with_max_seq_len(64).with_dropout_rate(0.0);
    let dir = std::env::temp_dir().join(format!("yumon-blt-entropy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(BYTE_LM_FILE), serde_json::to_vec(&json!({"model": lm})).unwrap()).unwrap();
    lm.init::<Gpu>(&device).save_file(dir.join("model"), &BinFileRecorder::<FullPrecisionSettings>::new()).unwrap();
    // A fresh model's entropy sits near ln 260 = 5.56; pick a threshold inside its spread.
    let row = rows_of(&[random_row(50, 11)]);
    let probe = Patcher::<Gpu>::new(&Patching::Entropy { model: dir.clone(), threshold: 0.0, max_len: 50 }, 50, &device).unwrap();
    let mut h = Patcher::entropies(probe.entropy.as_ref().unwrap(), &row, &device).remove(0);
    h.sort_by(f32::total_cmp);
    let patcher = Patcher::<Gpu>::new(&Patching::Entropy { model: dir.clone(), threshold: (h[h.len() / 2] + h[h.len() / 2 + 1]) / 2.0, max_len: 50 }, 50, &device).unwrap();
    let full = patcher.starts(&row, &device).remove(0);
    let inner = full[2..50].iter().filter(|&&s| s).count();
    assert!(inner > 5 && inner < 45, "median threshold should start some patches, got {inner}");
    for k in [3, 17, 33, 49] {
        let prefix = rows_of(&[row[0].input[..k].to_vec()]);
        let part = patcher.starts(&prefix, &device).remove(0);
        assert_eq!(part, full[..=k], "entropy starts through {k} depend on later bytes");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Burn 0.20 / CubeCL 0.9 on WGPU (Intel UHD 770): the input gradient of a matmul,
/// dX = dY W^T, comes back wrong (max error ~3) for some small row counts, e.g.
/// [8, 128] x [128, 260] and [40, 32] x [32, 260]; which shapes fail depends on what
/// autotune picked earlier. Every shape with 400+ rows was exact. Forward and dW were
/// exact. This is why the BLT training test uses 12 rows.
#[test]
#[ignore = "repro of a WGPU backend bug, not a BLT check"]
fn wgpu_matmul_input_gradient() {
    let device = Default::default();
    let mut state = 7u64;
    let mut rnd = |n: usize| -> Vec<f32> { (0..n).map(|_| { state ^= state << 13; state ^= state >> 7; state ^= state << 17; ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5 }).collect() };
    let mut wrong = Vec::new();
    let shapes = [(8, 128, 260), (40, 32, 260), (64, 128, 260), (160, 32, 260), (400, 32, 260), (1024, 128, 260)];
    for &(m, k, n) in &shapes {
        let (x, w, g) = (rnd(m * k), rnd(k * n), rnd(m * n));
        let xt = Tensor::<TrainBackend, 2>::from_data(TensorData::new(x.clone(), [m, k]), &device).require_grad();
        let wt = Tensor::<TrainBackend, 2>::from_data(TensorData::new(w.clone(), [k, n]), &device).require_grad();
        let gt = Tensor::<TrainBackend, 2>::from_data(TensorData::new(g.clone(), [m, n]), &device);
        let y = xt.clone().matmul(wt.clone());
        let grads = (y.clone() * gt).sum().backward();
        let dx: Vec<f32> = xt.grad(&grads).unwrap().into_data().convert::<f32>().to_vec().unwrap();
        let mut ex = 0f32;
        for i in 0..m { for t in 0..k {
            let r: f32 = (0..n).map(|j| g[i * n + j] * w[t * n + j]).sum();
            ex = ex.max((r - dx[i * k + t]).abs());
        } }
        eprintln!("m {m} k {k} n {n}: dX max err {ex:.2e}");
        if ex > 1e-3 { wrong.push((m, k, n)); }
    }
    assert!(wrong.is_empty(), "wrong input gradients at {wrong:?}");
}
