"""Distil a sunrise SwiGLU MLP into a smaller gate program (function space, not weights).

    uv run --project ../python-train python scripts/distill.py LAYER --gates 32 --init top --out l0.json

Fits on tmp/wir-approx/layer{L}_probe.json (`wir_decoder record --eval 1024 --offset 3
--probe 40000 --out tmp/wir-approx`, or --train), reports relative error on
tmp/wir-sunrise/layer{L}_probe.json (other samples) and writes a wir weights JSON for
`wir decompile --codebook g4`. --lin adds a low-rank linear term (exported as gates
with W1 = 0), --frame r puts the gates in an r-dim frame; neither beat plain gates.
"""
import json, sys, os, time, argparse
import numpy as np, torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tmp")
CACHE = os.path.join(ROOT, "wir-approx", "npz")
os.makedirs(CACHE, exist_ok=True)


def load(path):
    key = os.path.join(CACHE, os.path.abspath(path).replace(os.sep, "_").replace("/", "_").replace(":", "") + ".npz")
    if os.path.exists(key):
        z = np.load(key); return z["x"], z["y"]
    d = json.load(open(path))
    x, y = np.asarray(d["inputs"], np.float32), np.asarray(d["outputs"], np.float32)
    np.savez(key, x=x, y=y); return x, y


def weights(layer):
    d = json.load(open(f"{ROOT}/wir-approx/layer{layer}_swiglu.json"))
    return [torch.tensor(d[k], dtype=torch.float32) for k in ("W1", "W3", "W2")]


def rel(yhat, y):
    return float(((yhat - y).pow(2).sum() / y.pow(2).sum()).sqrt())


class Mlp(torch.nn.Module):
    """y = W2 (silu(W1 x + b1) * (W3 x + b3)) + L x + b2, L optionally of rank r."""
    def __init__(s, d, h, lin_rank, frame=0):
        super().__init__()
        s.frame = frame
        if frame:
            # gates read an r-dim frame P x and write an r-dim frame mapped out by Q
            s.P = torch.nn.Parameter(torch.randn(frame, d) * d ** -0.5)
            s.Q = torch.nn.Parameter(torch.randn(d, frame) * frame ** -0.5)
            d = frame
        s.W1 = torch.nn.Parameter(torch.randn(h, d) * d ** -0.5)
        s.b1 = torch.nn.Parameter(torch.zeros(h))
        s.W3 = torch.nn.Parameter(torch.randn(h, d) * d ** -0.5)
        s.b3 = torch.nn.Parameter(torch.zeros(h))
        s.W2 = torch.nn.Parameter(torch.zeros(d, h))
        s.b2 = torch.nn.Parameter(torch.zeros(s.Q.shape[0] if frame else d))
        s.r = lin_rank
        if lin_rank:
            s.U = torch.nn.Parameter(torch.zeros(d, lin_rank))
            s.V = torch.nn.Parameter(torch.randn(lin_rank, d) * d ** -0.5)

    def forward(s, x):
        if s.frame:
            z = x @ s.P.T
            y = (F.silu(z @ s.W1.T + s.b1) * (z @ s.W3.T + s.b3)) @ s.W2.T
            return y @ s.Q.T + s.b2
        y = (F.silu(x @ s.W1.T + s.b1) * (x @ s.W3.T + s.b3)) @ s.W2.T + s.b2
        if s.r:
            y = y + (x @ s.V.T) @ s.U.T
        return y

    def params(s):
        n = sum(p.numel() for p in s.parameters())
        return n

    def export(s):
        """Linear rank r as r extra gates with W1 = 0, b1 = 1 (silu(1) constant)."""
        W1, b1, W3, b3, W2 = (t.detach() for t in (s.W1, s.b1, s.W3, s.b3, s.W2))
        if s.frame:
            P, Q = s.P.detach(), s.Q.detach()
            W1, W3, W2 = W1 @ P, W3 @ P, Q @ W2
        if s.r:
            c = F.silu(torch.tensor(1.0))
            W1 = torch.cat([W1, torch.zeros(s.r, W1.shape[1])])
            b1 = torch.cat([b1, torch.ones(s.r)])
            W3 = torch.cat([W3, s.V.detach()])
            b3 = torch.cat([b3, torch.zeros(s.r)])
            W2 = torch.cat([W2, s.U.detach() / c], 1)
        return dict(W1=W1, b1=b1, W3=W3, b3=b3, W2=W2, b2=s.b2.detach())


def fit(model, x, y, steps, lr, bs=4096):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(0)
    for i in range(steps):
        idx = torch.randint(0, len(x), (bs,), generator=g)
        loss = F.mse_loss(model(x[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", type=int)
    ap.add_argument("--gates", type=int, default=32)
    ap.add_argument("--lin", type=int, default=0, help="rank of the linear term")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--init", choices=["random", "top"], default="random",
                    help="top: start gates from the original's most-used units")
    ap.add_argument("--out")
    ap.add_argument("--train", help="probe to fit on (default tmp/wir-approx)")
    a = ap.parse_args()
    torch.manual_seed(0)
    x, y = (torch.tensor(t) for t in load(a.train or f"{ROOT}/wir-approx/layer{a.layer}_probe.json"))
    xt, yt = (torch.tensor(t) for t in load(f"{ROOT}/wir-sunrise/layer{a.layer}_probe.json"))
    d = x.shape[1]
    m = Mlp(d, a.gates, a.lin, a.frame)
    if a.init == "top" and a.gates and not a.frame:
        W1, W3, W2 = weights(a.layer)
        act = F.silu(x @ W1.T) * (x @ W3.T)
        score = (act.pow(2).mean(0) * W2.pow(2).sum(0))
        top = score.argsort(descending=True)[: a.gates]
        with torch.no_grad():
            m.W1.copy_(W1[top]); m.W3.copy_(W3[top]); m.W2.copy_(W2[:, top])
    # closed-form linear baseline for reference
    X1 = torch.cat([x, torch.ones(len(x), 1)], 1)
    A = torch.linalg.lstsq(X1, y).solution
    lin_err = rel(torch.cat([xt, torch.ones(len(xt), 1)], 1) @ A, yt)
    t0 = time.time()
    fit(m, x, y, a.steps, a.lr)
    with torch.no_grad():
        tr, te = rel(m(x), y), rel(m(xt), yt)
    print(f"layer {a.layer} gates {a.gates} frame {a.frame} lin {a.lin} init {a.init}: params {m.params()} "
          f"train {tr:.3f} test {te:.3f} (full-linear lstsq test {lin_err:.3f}) {time.time()-t0:.0f}s")
    if a.out:
        w = {k: v.tolist() for k, v in m.export().items()}
        json.dump({"format": "wir-weights", "input": "x", **w}, open(a.out, "w"))


if __name__ == "__main__":
    main()
