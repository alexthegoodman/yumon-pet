import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tokenizers import Tokenizer
from tqdm import tqdm

from model import YumonPetMoeBrain, ModelConfig
from dataloader import get_dataloaders, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class TrainingConfig:
    # Paths & environment
    data_path: str = field(
        default_factory=lambda: os.environ.get(
            "YUMON_SAMPLE_CACHE",
            "../training-cache/samples.bin"
            if os.path.exists("../training-cache/samples.bin")
            else "training-cache/samples.bin",
        )
    )
    tokenizer_path: str = field(
        default_factory=lambda: os.environ.get(
            "TOKENIZER_PATH",
            "../yumon_bpe/tokenizer-16k-2.json"
            if os.path.exists("../yumon_bpe/tokenizer-16k-2.json")
            else "yumon_bpe/tokenizer-16k-2.json",
        )
    )
    checkpoint_dir: str = field(
        default_factory=lambda: os.environ.get(
            "CHECKPOINT_DIR",
            "/workspace/checkpoints"
            if os.path.exists("/workspace")
            else "checkpoints",
        )
    )

    # Model architecture
    vocab_size: int = 16384
    embed_dim: int = 1024
    n_layers: int = 24
    attn_heads: int = 16
    ff_dim: int = 4096
    max_seq_len: int = 256
    num_experts: int = 8
    top_k: int = 2
    dropout: float = 0.05
    aux_loss_weight: float = 0.01
    z_loss_weight: float = 0.001
    kv_heads: Optional[int] = None

    # Optimization
    epochs: int = 15
    batch_size: int = 16
    first_lr: float = 1e-4
    last_lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    precision: str = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32"
    seed: int = 4815162342

    # Dataset & Validation
    max_samples: Optional[int] = None
    val_fraction: float = 0.01
    log_every: int = 50
    generate_samples: bool = True
    sample_prompts: List[str] = field(
        default_factory=lambda: [
            "Tell me about yourself.",
            "Can you go home?",
            "What should we build today?",
            "This is a wonderful day.",
            "What is the universe?",
        ]
    )

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            n_layers=self.n_layers,
            attn_heads=self.attn_heads,
            ff_dim=self.ff_dim,
            max_seq_len=self.max_seq_len,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dropout=self.dropout,
            aux_loss_weight=self.aux_loss_weight,
            z_loss_weight=self.z_loss_weight,
            kv_heads=self.kv_heads,
        )


def compute_top3_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = PAD_TOKEN) -> float:
    """Compute top-3 token prediction accuracy over active target positions."""
    mask = (targets != ignore_index)
    if not mask.any():
        return 0.0
    active_logits = logits[mask]
    active_targets = targets[mask]
    k = min(3, active_logits.shape[-1])
    top3_pred = torch.topk(active_logits, k=k, dim=-1).indices
    correct = (top3_pred == active_targets.unsqueeze(-1)).any(dim=-1).sum().item()
    return correct / active_targets.numel()


class Trainer:
    def __init__(self, config: TrainingConfig, resume_path: Optional[str] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Load Tokenizer if available
        self.tokenizer = None
        if os.path.exists(config.tokenizer_path):
            self.tokenizer = Tokenizer.from_file(config.tokenizer_path)
            self.config.vocab_size = self.tokenizer.get_vocab_size()
            print(f"Loaded tokenizer from {config.tokenizer_path} (vocab size: {self.config.vocab_size})")

        # Load Data
        print(f"Preparing datasets from {config.data_path}...")
        self.train_loader, self.val_loader, self.metadata = get_dataloaders(
            cache_path=config.data_path,
            tokenizer_path=config.tokenizer_path if os.path.exists(config.tokenizer_path) else None,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
            val_fraction=config.val_fraction,
            max_samples=config.max_samples,
            seed=config.seed,
        )

        # Init Model
        model_config = self.config.to_model_config()
        self.model = YumonPetMoeBrain(model_config).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Initialized YumonPetMoeBrain: {total_params / 1e6:.2f}M parameters ({trainable_params / 1e6:.2f}M trainable).")

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.first_lr,
            weight_decay=config.weight_decay,
            eps=1e-7,
        )

        # Mixed precision setup
        self.use_amp = (self.device.type == "cuda" and config.precision in ["bf16", "fp16"])
        self.amp_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and config.precision == "fp16"))

        # State tracking
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        if resume_path:
            self.load_checkpoint(resume_path)

        # Save config
        config_path = os.path.join(config.checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def load_checkpoint(self, path: str):
        print(f"Resuming checkpoint from {path}...")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {self.start_epoch}, step {self.global_step}, best val loss {self.best_val_loss:.4f}")

    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(state, best_path)
            print(f"[BEST] Saved new best model to {best_path}")

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        total_batches = len(self.train_loader)
        total_steps = self.config.epochs * total_batches

        epoch_loss = 0.0
        epoch_lm_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_top3 = 0.0
        batches_processed = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
        for batch_idx, (tokens, targets) in enumerate(pbar):
            # Dynamic linear learning rate schedule: first_lr -> last_lr
            step_progress = self.global_step / max(1, total_steps)
            current_lr = self.config.first_lr * (1.0 - step_progress) + self.config.last_lr * step_progress
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

            tokens = tokens.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                logits, aux_loss = self.model(tokens)
                B, S, V = logits.shape

                # Language modeling loss on non-pad targets
                lm_loss = F.cross_entropy(
                    logits.reshape(B * S, V),
                    targets.reshape(B * S),
                    ignore_index=PAD_TOKEN,
                )
                total_loss = lm_loss + aux_loss

            if self.scaler.is_enabled():
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            top3 = compute_top3_accuracy(logits.detach(), targets)

            epoch_loss += total_loss.item()
            epoch_lm_loss += lm_loss.item()
            epoch_aux_loss += aux_loss.item()
            epoch_top3 += top3
            batches_processed += 1
            self.global_step += 1

            if (batch_idx + 1) % self.config.log_every == 0 or (batch_idx + 1) == total_batches:
                pbar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "lm": f"{lm_loss.item():.4f}",
                    "aux": f"{aux_loss.item():.4f}",
                    "top3": f"{top3:.1%}",
                    "lr": f"{current_lr:.2e}",
                })

        mean_loss = epoch_loss / max(1, batches_processed)
        mean_lm = epoch_lm_loss / max(1, batches_processed)
        mean_top3 = epoch_top3 / max(1, batches_processed)
        return mean_loss, mean_lm, mean_top3

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_top3 = 0.0
        batches = 0

        for tokens, targets in self.val_loader:
            tokens = tokens.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                logits, aux_loss = self.model(tokens)
                B, S, V = logits.shape
                lm_loss = F.cross_entropy(
                    logits.reshape(B * S, V),
                    targets.reshape(B * S),
                    ignore_index=PAD_TOKEN,
                )
                loss = lm_loss + aux_loss

            total_loss += loss.item()
            total_top3 += compute_top3_accuracy(logits, targets)
            batches += 1

        val_loss = total_loss / max(1, batches)
        val_top3 = total_top3 / max(1, batches)
        perplexity = float(torch.exp(torch.tensor(val_loss)).item())
        return val_loss, val_top3, perplexity

    def sample_generation(self):
        if not self.tokenizer:
            return

        print("\n" + "=" * 50)
        print("Generating sample model replies:")
        for prompt in self.config.sample_prompts:
            encoded = self.tokenizer.encode(prompt).ids
            prompt_tensor = torch.tensor([encoded], dtype=torch.long, device=self.device)
            generated_tokens = self.model.generate(
                prompt_tensor,
                max_new_tokens=48,
                temperature=0.8,
                top_k=40,
                eos_token_id=EOS_TOKEN,
            )
            out_ids = generated_tokens[0].tolist()
            reply = self.tokenizer.decode(out_ids)
            print(f"  Prompt: \"{prompt}\"")
            print(f"  Yumon:  \"{reply}\"\n")
        print("=" * 50 + "\n")

    def train(self):
        print(f"\nStarting training for {self.config.epochs} epochs on device: {self.device}")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start = time.time()
            train_loss, train_lm, train_top3 = self.train_epoch(epoch)
            val_loss, val_top3, perplexity = self.evaluate()
            epoch_time = time.time() - epoch_start

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            print(
                f"\nEpoch {epoch + 1}/{self.config.epochs} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f} (Top3: {train_top3:.1%}) | "
                f"Val Loss: {val_loss:.4f} (PPL: {perplexity:.2f}, Top3: {val_top3:.1%})"
                + (" [BEST]" if is_best else "")
            )

            # Save epoch checkpoint
            ckpt_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            self.save_checkpoint(ckpt_path, epoch=epoch, is_best=is_best)

            # Sample text generation
            if self.config.generate_samples:
                self.sample_generation()

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time / 60:.1f} minutes. Best Val Loss: {self.best_val_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Yumon Pet Sparse MoE Model (PyTorch)")
    parser.add_argument("--data-path", type=str, default=None, help="Path to samples.bin")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to tokenizer.json")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per step")
    parser.add_argument("--embed-dim", type=int, default=None, help="Transformer embedding dimension")
    parser.add_argument("--n-layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--attn-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--ff-dim", type=int, default=None, help="SwiGLU expert hidden dimension")
    parser.add_argument("--num-experts", type=int, default=None, help="Number of experts per MoE layer")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k experts routed per token")
    parser.add_argument("--first-lr", type=float, default=None, help="Starting learning rate")
    parser.add_argument("--last-lr", type=float, default=None, help="Ending learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Max sequence length")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap dataset sample count for quick runs")
    parser.add_argument("--precision", type=str, default=None, choices=["bf16", "fp16", "fp32"], help="Precision")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--no-samples", action="store_true", help="Disable text sample generation during training")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig()

    if args.data_path is not None:
        config.data_path = args.data_path
    if args.tokenizer_path is not None:
        config.tokenizer_path = args.tokenizer_path
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.embed_dim is not None:
        config.embed_dim = args.embed_dim
    if args.n_layers is not None:
        config.n_layers = args.n_layers
    if args.attn_heads is not None:
        config.attn_heads = args.attn_heads
    if args.ff_dim is not None:
        config.ff_dim = args.ff_dim
    if args.num_experts is not None:
        config.num_experts = args.num_experts
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.first_lr is not None:
        config.first_lr = args.first_lr
    if args.last_lr is not None:
        config.last_lr = args.last_lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.max_seq_len is not None:
        config.max_seq_len = args.max_seq_len
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.precision is not None:
        config.precision = args.precision
    if args.no_samples:
        config.generate_samples = False

    trainer = Trainer(config, resume_path=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
