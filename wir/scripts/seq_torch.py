"""Run compiled WIR 0.3 sequence weights in PyTorch and compare with the Rust reference.

Run from yumon-pet/wir after `cargo build --release`, using the python-train env:

    uv run --project ../python-train python scripts/seq_torch.py [examples ...]

For each example, `wir seq` writes the weights JSON and a probe of random sequences with
the program's outputs (exact hard attention) and the Rust f64 network's outputs. The
transformer is rebuilt as a torch.nn.Module (embeddings, per block attention heads then
ReLU MLP, unembed; no norms, no biases, scores without 1/sqrt(d_head)) and run in float64
and float32, batched with padding and a key mask.
"""

import json
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "target", "release", "wir.exe" if os.name == "nt" else "wir")


class SeqTransformer(torch.nn.Module):
    def __init__(self, w, dtype):
        super().__init__()
        t = lambda m: torch.nn.Parameter(torch.tensor(m, dtype=dtype), requires_grad=False)
        self.vocab_lo, self.causal = w["vocab"][0], w["causal"]
        self.tok_embed, self.pos_embed, self.unembed = t(w["tok_embed"]), t(w["pos_embed"]), t(w["unembed"])
        self.blocks = []
        for i, b in enumerate(w["blocks"]):
            heads = [tuple(t(h[k]) for k in ("wq", "wk", "wv", "wo")) for h in b["heads"]]
            for j, h in enumerate(heads):
                for k, p in zip(("wq", "wk", "wv", "wo"), h):
                    self.register_parameter(f"b{i}_h{j}_{k}", p)
            w_in = t(b["w_in"]) if b["w_in"] else None
            w_out = t(b["w_out"]) if b["w_in"] else None
            if w_in is not None:
                self.register_parameter(f"b{i}_w_in", w_in)
                self.register_parameter(f"b{i}_w_out", w_out)
            self.blocks.append((heads, w_in, w_out))

    def forward(self, tokens, lengths):
        # tokens [B, L] (padded), lengths [B]; position 0 is BOS.
        bsz, length = tokens.shape
        ids = torch.cat([torch.zeros(bsz, 1, dtype=torch.long), tokens - self.vocab_lo + 1], dim=1)
        x = self.tok_embed[ids] + self.pos_embed[: length + 1].unsqueeze(0)
        keys = torch.arange(length + 1)
        mask = keys.unsqueeze(0) <= lengths.unsqueeze(1)  # [B, L+1] real keys and BOS
        mask = mask.unsqueeze(1).expand(bsz, length + 1, length + 1)
        if self.causal:
            mask = mask & (keys.unsqueeze(0) <= keys.unsqueeze(1)).unsqueeze(0)
        for heads, w_in, w_out in self.blocks:
            add = torch.zeros_like(x)
            for wq, wk, wv, wo in heads:
                s = (x @ wq.T) @ (x @ wk.T).transpose(1, 2)
                a = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
                add = add + (a @ (x @ wv.T)) @ wo.T
            x = x + add
            if w_in is not None and w_in.shape[0] > 0:
                x = x + torch.relu(x @ w_in.T) @ w_out.T
        return x[:, 1:] @ self.unembed.T


def run(name, count=20000):
    src = os.path.join(ROOT, "examples", f"{name}.wir")
    with tempfile.TemporaryDirectory() as d:
        wpath, ppath = os.path.join(d, "w.json"), os.path.join(d, "probe.json")
        r = subprocess.run([EXE, "seq", src, "-o", wpath, "--probe", f"{count}:{ppath}"], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"wir seq {name} failed:\n{r.stderr}")
        w, probe = json.load(open(wpath)), json.load(open(ppath))
    seqs = probe["sequences"]
    length = max(len(s) for s in seqs)
    tokens = torch.tensor([s + [w["vocab"][0]] * (length - len(s)) for s in seqs])
    lengths = torch.tensor([len(s) for s in seqs])
    print(f"{name}: {len(seqs)} sequences, bound {probe['bound']}")
    for dtype in (torch.float64, torch.float32):
        with torch.no_grad():
            out = SeqTransformer(w, dtype)(tokens, lengths).double()
        worst_prog = worst_rust = 0.0
        wrong = 0
        for i, s in enumerate(seqs):
            got = out[i, : len(s)]
            prog = torch.tensor(probe["program"][i], dtype=torch.float64)
            rust = torch.tensor(probe["network"][i], dtype=torch.float64)
            worst_prog = max(worst_prog, (got - prog).abs().max().item())
            worst_rust = max(worst_rust, (got - rust).abs().max().item())
            wrong += int((got.round() != prog.round()).any().item())
        print(f"  {str(dtype):14} worst |torch - program| {worst_prog:.1e}, |torch - rust f64| {worst_rust:.1e}, rounded wrong {wrong}")
        if wrong:
            sys.exit(f"{name}: {wrong} sequences round wrong in {dtype}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    for name in sys.argv[1:] or ["reverse", "hist", "sort", "prefix_hist"]:
        run(name)
