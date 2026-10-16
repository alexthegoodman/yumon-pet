import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Hyperparameters for Yumon Pet Sparse MoE Model."""
    vocab_size: int = 16384
    embed_dim: int = 256
    n_layers: int = 12
    attn_heads: int = 8
    ff_dim: int = 1024
    max_seq_len: int = 256
    num_experts: int = 4
    top_k: int = 1
    dropout: float = 0.05
    aux_loss_weight: float = 0.01
    z_loss_weight: float = 0.001
    kv_heads: Optional[int] = None
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "n_layers": self.n_layers,
            "attn_heads": self.attn_heads,
            "ff_dim": self.ff_dim,
            "max_seq_len": self.max_seq_len,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "dropout": self.dropout,
            "aux_loss_weight": self.aux_loss_weight,
            "z_loss_weight": self.z_loss_weight,
            "kv_heads": self.kv_heads,
            "rope_base": self.rope_base,
            "rms_norm_eps": self.rms_norm_eps,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(max(seq_len, self.cos_cached.size(0) * 2))
        return (
            self.cos_cached[:seq_len].to(device=device),
            self.sin_cached[:seq_len].to(device=device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [batch, heads, seq, head_dim]
    seq_len = q.shape[2]
    cos = cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(1)  # [1, 1, seq, head_dim]
    sin = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: w2(silu(w1(x)) * w3(x))."""
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, in_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SparseMoE(nn.Module):
    """Dropless Sparse Mixture of Experts matching Burn's formulation.
    Tokens are sparsely dispatched to their top-k selected SwiGLU experts.
    Computes load balancing auxiliary loss and router z-loss.
    """
    def __init__(self, dim: int, ff_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim, ff_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor, active_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [N, dim] where N = batch_size * seq_len
        active_mask: [N] bool tensor, True for non-padding tokens
        Returns:
            output: [N, dim]
            balance_loss: scalar tensor
            z_loss: scalar tensor
        """
        N, D = x.shape
        device = x.device

        # Router logits in fp32 for numerical stability
        router_logits = self.router(x).float()  # [N, num_experts]
        probabilities = F.softmax(router_logits, dim=-1)  # [N, num_experts]

        # Top-k selection
        _, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)  # [N, top_k]

        # Filter to active tokens for auxiliary losses
        if active_mask is not None and active_mask.any():
            active_logits = router_logits[active_mask]
            active_probs = probabilities[active_mask]
            active_indices = topk_indices[active_mask]
            num_active = active_logits.shape[0]
        else:
            active_logits = router_logits
            active_probs = probabilities
            active_indices = topk_indices
            num_active = N

        # Switch-style auxiliary balance loss:
        # fractions f_i = tokens_routed_to_i / (total_tokens * top_k)
        # balance = num_experts * sum(f_i * mean(p_i))
        counts = torch.bincount(active_indices.reshape(-1), minlength=self.num_experts).float()
        fractions = counts / max(1.0, float(num_active * self.top_k))
        mean_probs = active_probs.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(fractions * mean_probs)

        # Router z-loss: penalizes large router logits to prevent instability
        # log_z = logits - log_softmax(logits) = logsumexp(logits)
        log_z = torch.logsumexp(active_logits, dim=-1)
        z_loss = torch.mean(log_z ** 2)

        # Sparse dispatch: accumulate expert outputs on device
        output = torch.zeros((N, D), dtype=x.dtype, device=device)

        # When active_mask is given, only dispatch active tokens to save computation
        dispatch_indices = active_mask.nonzero(as_tuple=True)[0] if active_mask is not None else None

        for expert_id in range(self.num_experts):
            # Find tokens where topk_indices contains expert_id
            mask = (topk_indices == expert_id)  # [N, top_k]
            if not mask.any():
                continue

            # Selected token rows in x
            token_rows = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            if dispatch_indices is not None:
                # Intersect with active tokens
                token_rows = token_rows[active_mask[token_rows]]

            if token_rows.numel() == 0:
                continue

            # Full-softmax gating preserves language-loss router gradients
            gates = probabilities[token_rows, expert_id].unsqueeze(-1).to(x.dtype)
            expert_in = x[token_rows]
            expert_out = self.experts[expert_id](expert_in)

            # Scatter-add to output
            output.index_add_(0, token_rows, expert_out * gates)

        return output, balance_loss, z_loss


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with built-in Flash Attention & RoPE."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.attn_heads = config.attn_heads
        self.kv_heads = config.kv_heads or config.attn_heads
        assert self.embed_dim % self.attn_heads == 0, "embed_dim must be divisible by attn_heads"
        self.head_dim = self.embed_dim // self.attn_heads
        assert self.attn_heads % self.kv_heads == 0, "attn_heads must be divisible by kv_heads"
        self.num_kv_groups = self.attn_heads // self.kv_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.attn_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin, offset=offset)

        # Update KV cache for incremental generation
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        elif use_cache:
            new_cache = (k, v)
        else:
            new_cache = None

        # Repeat KV if using Grouped-Query Attention (GQA)
        if self.num_kv_groups > 1:
            k_att = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_att = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_att = k
            v_att = v

        # PyTorch Builtin Flash Attention (scaled_dot_product_attention)
        # When kv_cache is active, query length is S, key length is total_seq
        is_causal = (S > 1 and kv_cache is None)
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k_att, v_att,
            is_causal=is_causal,
            dropout_p=dropout_p,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.o_proj(attn_out), new_cache


class MoeBlock(nn.Module):
    """Transformer block with RMSNorm, Causal Flash Attention, and Sparse MoE."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.moe = SparseMoE(
            dim=config.embed_dim,
            ff_dim=config.ff_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, D = x.shape

        # Attention + residual
        norm_x = self.attn_norm(x)
        attn_out, new_cache = self.attn(
            norm_x, cos, sin, kv_cache=kv_cache, offset=offset, use_cache=use_cache
        )
        x = x + attn_out

        # MoE + residual
        norm_ffn = self.ffn_norm(x).view(B * S, D)
        moe_out, balance_loss, z_loss = self.moe(norm_ffn, active_mask=active_mask)
        x = x + moe_out.view(B, S, D)

        return x, balance_loss, z_loss, new_cache


class YumonPetMoeBrain(nn.Module):
    """Yumon Pet Sparse MoE Language Model.
    Direct PyTorch counterpart to YumonMoeBrain in Rust/Burn.
    Features:
      - Genuine dropless Sparse Mixture of Experts with SwiGLU
      - Built-in PyTorch Flash Attention (is_causal SDPA)
      - Rotary Position Embeddings (RoPE)
      - Pre-RMSNorm Transformer Decoders
      - Switch load balancing auxiliary loss & router z-loss
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(
            head_dim=config.embed_dim // config.attn_heads,
            max_seq_len=max(config.max_seq_len, 2048),
            base=config.rope_base,
        )
        self.blocks = nn.ModuleList([MoeBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.token_head = nn.Linear(config.embed_dim, config.vocab_size, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: [batch_size, seq_len] int64 tensor
        active_mask: optional [batch_size * seq_len] bool tensor (True for non-pad)
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            aux_loss: scalar tensor (balance_loss * aux_weight + z_loss * z_weight)
        """
        B, S = tokens.shape
        device = tokens.device

        # If active_mask not supplied, determine it from tokens != pad_token_id
        if active_mask is None:
            active_mask = (tokens != self.config.pad_token_id).view(-1)

        cos, sin = self.rope(S, device)
        x = self.dropout(self.embedding(tokens))

        total_balance_loss = torch.tensor(0.0, device=device)
        total_z_loss = torch.tensor(0.0, device=device)

        for block in self.blocks:
            x, b_loss, z_loss, _ = block(x, cos, sin, active_mask=active_mask)
            total_balance_loss = total_balance_loss + b_loss
            total_z_loss = total_z_loss + z_loss

        num_layers = len(self.blocks)
        aux_loss = (
            (total_balance_loss / num_layers) * self.config.aux_loss_weight
            + (total_z_loss / num_layers) * self.config.z_loss_weight
        )

        logits = self.token_head(self.norm(x))
        return logits, aux_loss

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation with KV caching.
        prompt_tokens: [1, prompt_len] int64 tensor
        Returns generated tokens: [1, total_len]
        """
        self.eval()
        device = prompt_tokens.device
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        generated = prompt_tokens.clone()
        kv_caches = [None] * len(self.blocks)

        # Prime with initial prompt
        B, S = prompt_tokens.shape
        cos, sin = self.rope(self.config.max_seq_len, device)

        x = self.embedding(prompt_tokens)
        for i, block in enumerate(self.blocks):
            x, _, _, kv_caches[i] = block(
                x, cos, sin, active_mask=None, kv_cache=None, offset=0, use_cache=True
            )

        logits = self.token_head(self.norm(x[:, -1:, :]))

        for step in range(max_new_tokens):
            if generated.shape[1] >= self.config.max_seq_len:
                break

            if temperature > 0:
                scaled_logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == eos_id:
                break

            # Forward single step with cached KV
            cur_offset = generated.shape[1] - 1
            x = self.embedding(next_token)
            for i, block in enumerate(self.blocks):
                x, _, _, kv_caches[i] = block(
                    x, cos, sin, active_mask=None, kv_cache=kv_caches[i], offset=cur_offset, use_cache=True
                )
            logits = self.token_head(self.norm(x))

        return generated
