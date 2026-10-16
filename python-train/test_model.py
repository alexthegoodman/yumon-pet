import os
import torch
import torch.nn.functional as F
from model import YumonPetMoeBrain, ModelConfig, SparseMoE, CausalSelfAttention
from dataloader import load_samples_cache, YumonPetDataset, trim_batch_collate, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


def test_sparse_moe():
    print("Testing SparseMoE component...")
    dim = 64
    ff_dim = 128
    num_experts = 4
    top_k = 2
    N = 16

    moe = SparseMoE(dim, ff_dim, num_experts, top_k)
    x = torch.randn(N, dim, requires_grad=True)
    active_mask = torch.ones(N, dtype=torch.bool)
    active_mask[12:] = False  # some padding

    out, balance_loss, z_loss = moe(x, active_mask=active_mask)
    assert out.shape == (N, dim), f"Expected shape ({N}, {dim}), got {out.shape}"
    assert balance_loss.numel() == 1 and balance_loss > 0, "Invalid balance_loss"
    assert z_loss.numel() == 1 and z_loss >= 0, "Invalid z_loss"

    # Backward gradient flow test
    loss = out.sum() + balance_loss + z_loss
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any(), "MoE gradient backward failed"
    print("  SparseMoE forward and backward passed.")


def test_full_model_forward():
    print("Testing full YumonPetMoeBrain forward and backward...")
    config = ModelConfig(
        vocab_size=1000,
        embed_dim=64,
        n_layers=2,
        attn_heads=4,
        ff_dim=128,
        max_seq_len=32,
        num_experts=4,
        top_k=1,
    )
    model = YumonPetMoeBrain(config)

    batch_size = 8
    seq_len = 32
    tokens = torch.randint(1, 1000, (batch_size, seq_len))
    tokens[:, 24:] = PAD_TOKEN  # simulate padding

    logits, aux_loss = model(tokens)
    assert logits.shape == (batch_size, seq_len, 1000), f"Logits shape mismatch: {logits.shape}"
    assert aux_loss.numel() == 1, "aux_loss is not scalar"

    # Test backward
    dummy_targets = torch.randint(1, 1000, (batch_size, seq_len))
    dummy_targets[:, 12:] = PAD_TOKEN
    loss = F.cross_entropy(logits.view(-1, 1000), dummy_targets.view(-1), ignore_index=PAD_TOKEN) + aux_loss
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
    print("  Full model forward, backward, and autograd passed.")


def test_generation():
    print("Testing autoregressive generation with KV caching...")
    config = ModelConfig(
        vocab_size=500,
        embed_dim=64,
        n_layers=2,
        attn_heads=4,
        ff_dim=128,
        max_seq_len=32,
        num_experts=4,
        top_k=1,
    )
    model = YumonPetMoeBrain(config)
    model.eval()

    prompt = torch.tensor([[BOS_TOKEN, 10, 20, 30]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=8, temperature=0.8, top_k=20)
    assert generated.shape[0] == 1, f"Batch dim mismatch: {generated.shape}"
    assert generated.shape[1] > prompt.shape[1], "Did not generate new tokens"

    # Verify cached generation is identical to un-cached full forward pass
    cached_greedy = model.generate(prompt, max_new_tokens=5, temperature=0.0)
    cur_tokens = list(prompt[0].tolist())
    for _ in range(5):
        with torch.no_grad():
            logits, _ = model(torch.tensor([cur_tokens], dtype=torch.long))
            nxt = torch.argmax(logits[0, -1, :]).item()
            cur_tokens.append(nxt)
            if nxt == EOS_TOKEN:
                break
    assert cached_greedy[0].tolist() == cur_tokens, "KV cached generation diverged from full forward pass!"
    print(f"  Generated sequence length {generated.shape[1]} from prompt length {prompt.shape[1]}. Cache parity verified.")


def test_samples_cache_loading():
    print("Testing samples.bin loader and dataset...")
    # Check default path relative to repo or current folder
    sample_paths = [
        "../training-cache/samples.bin",
        "../../yumon-pet/training-cache/samples.bin",
        "yumon-pet/training-cache/samples.bin",
    ]
    path = next((p for p in sample_paths if os.path.exists(p)), None)
    if not path:
        print("  samples.bin not found at standard paths, skipping file read test.")
        return

    meta, samples = load_samples_cache(path, max_samples=20)
    assert len(samples) == 20, f"Expected 20 samples, got {len(samples)}"
    print(f"  Successfully loaded 20 test samples from {path}.")

    dataset = YumonPetDataset(samples, max_seq_len=64)
    tokens, targets = dataset[0]
    assert tokens.shape == (64,), f"Tokens shape: {tokens.shape}"
    assert targets.shape == (64,), f"Targets shape: {targets.shape}"

    # Test collate with dynamic trimming
    batch = [dataset[i] for i in range(4)]
    b_tokens, b_targets = trim_batch_collate(batch)
    assert b_tokens.shape[0] == 4, f"Batch size mismatch: {b_tokens.shape}"
    assert b_tokens.shape[1] <= 64, f"Trimmed length exceeded max: {b_tokens.shape}"
    print(f"  Dataset and batch trimming verified (trimmed to length {b_tokens.shape[1]}).")


if __name__ == "__main__":
    print("=== Running Yumon Pet PyTorch Unit Tests ===")
    test_sparse_moe()
    test_full_model_forward()
    test_generation()
    test_samples_cache_loading()
    print("All tests passed successfully!")
