"""Fit one-hidden-layer ReLU networks to a decoder's SwiGLU MLPs for WIR.

Reads `layer{l}_x.npy` / `layer{l}_y.npy` written by `wir_decoder record` (MLP input
and output on real tokens) and writes `layer{l}.npz` (W1 b1 W2 b2, nn.Linear layout,
float64, plain np.savez so `wir decompile` reads it).

    uv run --project python-train python scripts/wir_fit_mlp.py tmp/wir-sunrise --hidden 1024

Reported per layer, on 2% held-out rows: relative error |y_hat - y| / |y| (RMS over
rows) for the ReLU fit and, as a floor for comparison, the best affine map.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch


def rel_err(pred: torch.Tensor, y: torch.Tensor) -> float:
    return math.sqrt(((pred - y) ** 2).sum().item() / (y**2).sum().item())


def fit_layer(x: np.ndarray, y: np.ndarray, hidden: int, epochs: int, batch: int, lr: float, seed: int):
    torch.manual_seed(seed)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    n = x.shape[0]
    perm = torch.randperm(n)
    val, tr = perm[: n // 50], perm[n // 50 :]
    xt, yt, xv, yv = x[tr], y[tr], x[val], y[val]

    # Affine baseline (least squares with a bias column).
    ones = torch.ones(xt.shape[0], 1)
    sol = torch.linalg.lstsq(torch.cat([xt, ones], 1).double(), yt.double()).solution.float()
    lin = rel_err(torch.cat([xv, torch.ones(xv.shape[0], 1)], 1) @ sol, yv)

    d = x.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, d))
    # Start from the affine solution: the readout bias carries the mean, the hidden
    # layer then only has to learn the residual.
    with torch.no_grad():
        net[2].bias.copy_(yt.mean(0))
        net[2].weight.mul_(0.1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps = epochs * math.ceil(xt.shape[0] / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for ep in range(epochs):
        order = torch.randperm(xt.shape[0])
        for i in range(0, xt.shape[0], batch):
            idx = order[i : i + batch]
            loss = ((net(xt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            with torch.no_grad():
                print(f"    epoch {ep + 1}/{epochs}: val rel err {rel_err(net(xv), yv):.4f} ({time.time() - t0:.0f}s)", flush=True)
    with torch.no_grad():
        relu_err = rel_err(net(xv), yv)
        dead = (torch.relu(xv @ net[0].weight.T + net[0].bias) > 0).sum(0).eq(0).sum().item()
    return net, lin, relu_err, dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--layers", type=str, default=None, help="comma list, default all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", type=str, default="")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    layers = sorted(int(p.stem[5:-2]) for p in args.dir.glob("layer*_x.npy"))
    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
    print(f"{'layer':>5} {'|y|/|x|':>8} {'affine':>8} {'relu':>8} {'dead':>6}")
    for l in layers:
        x = np.load(args.dir / f"layer{l}_x.npy")
        y = np.load(args.dir / f"layer{l}_y.npy")
        print(f"layer {l}: {x.shape[0]} rows, hidden {args.hidden}", flush=True)
        net, lin, err, dead = fit_layer(x, y, args.hidden, args.epochs, args.batch, args.lr, args.seed + l)
        scale = math.sqrt((y**2).sum() / (x**2).sum())
        print(f"{l:>5} {scale:>8.4f} {lin:>8.4f} {err:>8.4f} {dead:>6}", flush=True)
        w = {k: t.detach().double().numpy() for k, t in
             zip(["W1", "b1", "W2", "b2"], [net[0].weight, net[0].bias, net[2].weight, net[2].bias])}
        np.savez(args.dir / f"layer{l}{args.suffix}.npz", **w)


if __name__ == "__main__":
    main()
