import os
import torch
import torch.nn.functional as F
from model import YumonPetMoeBrain, ModelConfig, SparseMoE, CausalSelfAttention
from dataloader import load_samples_cache, YumonPetDataset, trim_batch_collate, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from tokenizers import Tokenizer
from generate import (
    ChatMessage,
    fit_chat_prompt,
    language_prompt,
    extract_memories,
    chat_prompt,
    generate_reply,
)



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



def test_memories_and_chat_prompt():
    print("Testing memory formatting and chat prompt parity with yumon_rss.rs...")
    tok_path = "tokenizer.json" if os.path.exists("tokenizer.json") else "../yumon_bpe/tokenizer.json"
    if not os.path.exists(tok_path):
        print(f"  Tokenizer not found at {tok_path}, skipping chat prompt test.")
        return

    tokenizer = Tokenizer.from_file(tok_path)

    # 1. Parity test with Rust chat_memories_match_training_and_exclude_failures_and_pending_messages
    history = [
        ChatMessage("user", "My favorite color is blue."),
        ChatMessage("yumon", "I like blue too."),
        ChatMessage("user", "A failed question."),
        ChatMessage("error", "Could not reply: GPU unavailable"),
        ChatMessage("user", "Remember it?"),
        ChatMessage("yumon", "Yes, blue."),
        ChatMessage("user", "What color?"),
    ]

    p1 = chat_prompt("What color?", history, 256, tokenizer)
    assert p1 == (
        "Human: My favorite color is blue.\n"
        "Yumon: I like blue too.\n"
        "Human: Remember it?\n"
        "Yumon: Yes, blue.\n"
        "Human: What color?"
    ), f"Prompt mismatch: {p1}"

    assert chat_prompt("What color?", [], 256, tokenizer) == "What color?"
    assert chat_prompt("What color?", history[2:4], 256, tokenizer) == "What color?"

    space_len = len(tokenizer.encode(" ", add_special_tokens=False).ids)
    p1_len = len(tokenizer.encode(p1, add_special_tokens=False).ids)
    assert 1 + p1_len + space_len + 64 <= 256

    # 2. Parity test with Rust chat_budget_keeps_recent_complete_turns_and_reserves_reply_space
    repeat_history = [
        ChatMessage("user", "An old memory with many words. " * 100),
        ChatMessage("yumon", "An old answer."),
        ChatMessage("user", "My favorite color is blue."),
        ChatMessage("yumon", "I like blue too."),
    ]
    p2 = chat_prompt("What color?", repeat_history, 256, tokenizer)
    assert p2 == (
        "Human: My favorite color is blue.\n"
        "Yumon: I like blue too.\n"
        "Human: What color?"
    )

    long_msg = "Hello \U0001f43e friend! " * 200
    for context in [32, 256]:
        prompt = chat_prompt(long_msg, repeat_history, context, tokenizer)
        assert long_msg.startswith(prompt)
        reply_tokens = max(8, min(64, context // 4))
        p_len = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
        assert 1 + p_len + space_len + reply_tokens <= context

    try:
        chat_prompt("Hello", repeat_history, 8, tokenizer)
        assert False, "Context 8 should raise error"
    except ValueError:
        pass

    # 3. Parity test with Rust chat_prompt_preserves_direct_messages_and_trims_unicode
    assert fit_chat_prompt("  Hello Yumon!  ", 32, len) == "Hello Yumon!"
    assert fit_chat_prompt("Hello \U0001f43e friend", 7, len) == "Hello \U0001f43e"
    try:
        fit_chat_prompt("Hello", 0, len)
        assert False
    except ValueError:
        pass
    try:
        fit_chat_prompt(" \n ", 20, len)
        assert False
    except ValueError:
        pass

    # 4. Test generate_reply with history using small test model
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=64,
        n_layers=2,
        attn_heads=4,
        ff_dim=128,
        max_seq_len=256,
        num_experts=4,
        top_k=1,
    )
    device = torch.device("cpu")
    model = YumonPetMoeBrain(config).to(device)
    model.eval()

    reply = generate_reply(
        model,
        tokenizer,
        "What color?",
        device,
        max_new_tokens=8,
        history=history,
        context=256,
    )
    assert isinstance(reply, str)
    print("  Memory prompt formatting and reply generation passed.")


def test_trainer_mid_epoch_checkpoint():
    import tempfile
    import shutil
    from train import Trainer, TrainingConfig

    print("Testing Trainer mid-epoch evaluation and best model checkpointing...")
    sample_paths = [
        "../training-cache/samples.bin",
        "../../yumon-pet/training-cache/samples.bin",
        "training-cache/samples.bin",
    ]
    path = next((p for p in sample_paths if os.path.exists(p)), None)
    if not path:
        print("  samples.bin not found, skipping trainer test.")
        return

    tmp_dir = tempfile.mkdtemp(prefix="yumon_test_ckpt_")
    try:
        config = TrainingConfig(
            data_path=path,
            checkpoint_dir=tmp_dir,
            embed_dim=32,
            n_layers=1,
            attn_heads=2,
            ff_dim=64,
            num_experts=2,
            top_k=1,
            max_seq_len=32,
            max_samples=24,
            batch_size=4,
            epochs=1,
            eval_every=2,
            generate_samples=False,
            precision="fp32",
        )
        trainer = Trainer(config)
        trainer.train()

        best_model_path = os.path.join(tmp_dir, "best_model.pt")
        epoch_1_path = os.path.join(tmp_dir, "checkpoint_epoch_1.pt")
        assert os.path.exists(best_model_path), "best_model.pt was not saved"
        assert os.path.exists(epoch_1_path), "checkpoint_epoch_1.pt was not saved"

        ckpt = torch.load(best_model_path, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "best_val_loss" in ckpt
        assert ckpt["best_val_loss"] < float("inf")
        print(f"  Trainer mid-epoch checkpoint verified (best val loss: {ckpt['best_val_loss']:.4f}).")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=== Running Yumon Pet PyTorch Unit Tests ===")
    test_sparse_moe()
    test_full_model_forward()
    test_generation()
    test_samples_cache_loading()
    test_memories_and_chat_prompt()
    test_trainer_mid_epoch_checkpoint()
    print("All tests passed successfully!")

