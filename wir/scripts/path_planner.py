"""Grid pathfinding networks in PyTorch: the WIR-compiled step and learned planners.

Run from yumon-pet/wir after `cargo build --release`, using the python-train env:

    uv run --project ../python-train python scripts/path_planner.py check FILE.safetensors
    uv run --project ../python-train python scripts/path_planner.py train --out DIR --mode step|unroll
        [--hidden 6,5,2] [--init FILE.safetensors] [--steps 3000] [--seed 0]
    uv run --project ../python-train python scripts/path_planner.py eval FILE.safetensors [...]

`check` loads a stencil .safetensors written by `wir compile` (nn.Sequential of
Conv2d with per-channel pad values in the metadata) and compares one step with the
rule and settled fields with BFS. `train` fits the same Conv2d shape to the
4-neighbour, unit-cost rule from scratch, either one step at a time (`step`: random
fields in, the rule's output as the target) or end to end (`unroll`: start from the
ceiling everywhere, run K steps, only the final field is compared with BFS). The
result is saved as a stencil .safetensors that `wir decompile` reads. `eval` scores
any of them on held-out grids, including grids larger than the training ones.

Everything here is independent of the Rust simulator: mazes, BFS and the rule are
re-implemented in NumPy.
"""

import argparse
import collections
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = ["d", "wall", "goal"]

# ---------------------------------------------------------------- safetensors


def read_safetensors(path):
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + n])
    meta = header.pop("__metadata__", {})
    out = {}
    for k, v in header.items():
        dt = {"F64": np.float64, "F32": np.float32}[v["dtype"]]
        a, b = v["data_offsets"]
        out[k] = np.frombuffer(raw[8 + n + a : 8 + n + b], dtype=dt).reshape(v["shape"]).astype(np.float64)
    return out, meta


def write_safetensors(path, tensors, meta):
    header, blobs, off = {}, [], 0
    for k, a in tensors.items():
        b = np.ascontiguousarray(a, dtype=np.float64).tobytes()
        header[k] = {"dtype": "F64", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = meta
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


# ---------------------------------------------------------------- model


class StencilNet(nn.Module):
    """Conv2d(k x k) -> ReLU -> 1x1 ... -> 1x1, run on raw values. Inside, d is divided
    by the ceiling and the output multiplied by it, so learned weights train at unit
    scale; `export` folds both scales into the weights."""

    def __init__(self, hidden, ceiling, k=3, channels=3):
        super().__init__()
        sizes = [channels] + list(hidden) + [1]
        self.convs = nn.ModuleList(
            nn.Conv2d(a, b, k if i == 0 else 1, dtype=torch.float64) for i, (a, b) in enumerate(zip(sizes, sizes[1:]))
        )
        self.k, self.ceiling = k, int(ceiling)
        self.register_buffer("pads", torch.tensor([float(ceiling), 1.0, 0.0][:channels], dtype=torch.float64))
        self.register_buffer("scale", torch.tensor([1.0 / ceiling, 1.0, 1.0][:channels], dtype=torch.float64))

    def forward(self, x):
        r = self.k // 2
        p = self.pads.view(1, -1, 1, 1)
        x = F.pad(x - p, (r, r, r, r)) + p  # each channel padded with its own value
        x = x * self.scale.view(1, -1, 1, 1)
        for i, c in enumerate(self.convs):
            x = c(x)
            if i + 1 < len(self.convs):
                x = F.relu(x)
        return x * self.ceiling

    def step(self, d, wall, goal):
        return self(torch.stack([d, wall, goal], 1))[:, 0]

    @torch.no_grad()
    def export(self, path):
        tensors = {}
        n = len(self.convs)
        for i, c in enumerate(self.convs):
            w, b = c.weight.detach().clone(), c.bias.detach().clone()
            if i == 0:
                w = w * self.scale.view(1, -1, 1, 1)
            if i + 1 == n:
                w, b = w * self.ceiling, b * self.ceiling
            tensors[f"{2 * i}.weight"], tensors[f"{2 * i}.bias"] = w.numpy(), b.numpy()
        meta = {
            "wir.kind": "stencil",
            "wir.inputs": json.dumps(CHANNELS),
            "wir.pads": json.dumps(self.pads.tolist()),
            "wir.outputs": json.dumps(["d2"]),
        }
        write_safetensors(path, tensors, meta)

    @classmethod
    def load(cls, path):
        t, meta = read_safetensors(path)
        pads = json.loads(meta["wir.pads"])
        ws = [t[f"{2 * i}.weight"] for i in range(len(t) // 2)]
        bs = [t[f"{2 * i}.bias"] for i in range(len(t) // 2)]
        net = cls([w.shape[0] for w in ws[:-1]], pads[0], k=ws[0].shape[-1], channels=ws[0].shape[1])
        with torch.no_grad():
            for i, c in enumerate(net.convs):
                w, b = torch.tensor(ws[i]), torch.tensor(bs[i])
                if i == 0:
                    w = w / net.scale.view(1, -1, 1, 1)
                if i + 1 == len(ws):
                    w, b = w / net.ceiling, b / net.ceiling
                c.weight.copy_(w)
                c.bias.copy_(b)
        return net


# ---------------------------------------------------------------- reference


def rule(d, wall, goal, m):
    """One step of the 4-neighbour unit-cost rule (NumPy, batched [B, H, W])."""
    p = np.pad(d, ((0, 0), (1, 1), (1, 1)), constant_values=m)
    best = np.minimum(np.minimum(p[:, :-2, 1:-1], p[:, 2:, 1:-1]), np.minimum(p[:, 1:-1, :-2], p[:, 1:-1, 2:]))
    out = np.minimum(best + 1, m)
    out = np.where(wall == 1, m, out)
    return np.where(goal == 1, 0, out)


def bfs(wall, goal, m):
    h, w = wall.shape
    dist = np.full((h, w), m, dtype=np.int64)
    q = collections.deque()
    for r, c in zip(*np.nonzero(goal)):
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y, x = r + dy, c + dx
            if 0 <= y < h and 0 <= x < w and not wall[y, x] and not goal[y, x] and dist[y, x] > dist[r, c] + 1:
                dist[y, x] = dist[r, c] + 1
                q.append((y, x))
    return np.minimum(dist, m)


def random_grid(h, w, rng):
    """Open field with scattered walls, or a carved maze with a few extra openings."""
    if rng.random() < 0.5:
        wall = (rng.random((h, w)) < rng.random() * 0.45).astype(np.int64)
    else:
        wall = np.ones((h, w), dtype=np.int64)
        wall[0, 0] = 0
        stack = [(0, 0)]
        while stack:
            r, c = stack[-1]
            opts = [(r + dy, c + dx) for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)) if 0 <= r + dy < h and 0 <= c + dx < w and wall[r + dy, c + dx]]
            if not opts:
                stack.pop()
                continue
            y, x = opts[rng.integers(len(opts))]
            wall[(r + y) // 2, (c + x) // 2] = 0
            wall[y, x] = 0
            stack.append((y, x))
        for _ in range(h * w // 16):
            wall[rng.integers(h), rng.integers(w)] = 0
    goal = np.zeros((h, w), dtype=np.int64)
    for _ in range(1 + rng.integers(2)):
        goal[rng.integers(h), rng.integers(w)] = 1
    return wall, goal


def batch(n, h, w, m, rng):
    walls, goals = zip(*(random_grid(h, w, rng) for _ in range(n)))
    wall, goal = np.stack(walls), np.stack(goals)
    target = np.stack([bfs(a, b, m) for a, b in zip(wall, goal)])
    return wall, goal, target


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def settle(net, wall, goal, m, limit, at=None):
    """Run the network from the ceiling everywhere until it stops moving (or `limit`).
    With `at`, also return the field after exactly `at` steps."""
    wall_t, goal_t = torch.tensor(wall, dtype=torch.float64), torch.tensor(goal, dtype=torch.float64)
    d = torch.full(wall_t.shape, float(m), dtype=torch.float64)
    snap = None
    for k in range(1, max(limit, at or 0) + 1):
        nd = net.step(d, wall_t, goal_t)
        delta = (nd - d).abs().max().item()
        d = nd
        if k == at:
            snap = d.numpy().copy()
        if delta < 1e-6 and k >= (at or 0):
            break
    if at is not None and snap is None:
        snap = d.numpy().copy()
    out = (d.numpy(), min(k, limit) if delta >= 1e-6 else k, delta)
    return out + (snap,) if at is not None else out


def greedy_optimal(field, wall, goal, target, m):
    """Share of reachable start cells from which stepping to the lowest neighbour of the
    network's field reaches a goal in exactly the BFS distance."""
    h, w = wall.shape
    ok = total = 0
    for r in range(h):
        for c in range(w):
            t = target[r, c]
            if wall[r, c] or goal[r, c] or t >= m:
                continue
            total += 1
            y, x = r, c
            for _ in range(int(t)):
                # A goal can sit on a wall cell; it is still a goal (BFS and the rule agree).
                nbrs = [(y + dy, x + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)) if 0 <= y + dy < h and 0 <= x + dx < w and (not wall[y + dy, x + dx] or goal[y + dy, x + dx])]
                if not nbrs:
                    break
                y, x = min(nbrs, key=lambda p: field[p])
            ok += bool(goal[y, x])
    return ok, total


def evaluate(net, m, sizes=((8, 8), (16, 16), (32, 32)), grids=100, seed=12345):
    rng = np.random.default_rng(seed)
    rows = []
    for h, w in sizes:
        wall, goal, target = batch(grids, h, w, m, rng)
        # From the ceiling the exact rule settles within m + 1 steps; 2m + 10 leaves room.
        field, steps, delta, snap = settle(net, wall, goal, m, limit=2 * m + 10, at=48)
        open_ = wall == 0

        def score(f):
            rounded = np.rint(f)
            ok = total = 0
            for i in range(grids):
                a, b = greedy_optimal(f[i], wall[i], goal[i], target[i], m)
                ok, total = ok + a, total + b
            return float((rounded == target)[open_].mean()), float(np.mean([(rounded[i] == target[i]).all() for i in range(grids)])), ok / max(total, 1)

        cell, grid_exact, paths = score(field)
        cell48, _, paths48 = score(snap)
        rows.append(
            {
                "size": f"{h}x{w}",
                "cells_exact": cell,
                "grids_exact": grid_exact,
                "paths_optimal": paths,
                "settle_steps": steps,
                "last_change": delta,
                "max_abs_err": float(np.abs(field - target)[open_].max()),
                "cells_exact_at48": cell48,
                "paths_optimal_at48": paths48,
            }
        )
    return rows


def step_error(net, m, n=2000, seed=777):
    """One-step accuracy on random fields (the rule's own input distribution)."""
    rng = np.random.default_rng(seed)
    wall, goal, _ = batch(n, 8, 8, m, rng)
    d = rng.integers(0, m + 1, size=wall.shape)
    want = rule(d, wall, goal, m)
    with torch.no_grad():
        got = net.step(*(torch.tensor(a, dtype=torch.float64) for a in (d, wall, goal))).numpy()
    return float((np.rint(got) == want).mean()), float(np.abs(got - want).max())


# ---------------------------------------------------------------- training


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    m = args.ceiling
    if args.init:
        net = StencilNet.load(args.init)
    else:
        net = StencilNet([int(x) for x in args.hidden.split(",")], m)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    pool = [batch(256, args.size, args.size, m, rng) for _ in range(args.pool)]
    t0 = time.time()
    log = []
    for it in range(1, args.steps + 1):
        wall, goal, target = pool[rng.integers(len(pool))]
        idx = rng.choice(len(wall), args.batch, replace=False)
        wall_t, goal_t = (torch.tensor(a[idx], dtype=torch.float64) for a in (wall, goal))
        if args.mode == "step":
            # Random fields, plus fields part-way to the answer, so the step sees what iteration produces.
            d = rng.integers(0, m + 1, size=(args.batch, args.size, args.size))
            mix = rng.random(args.batch) < 0.5
            d[mix] = np.minimum(target[idx][mix] + rng.integers(0, 3, size=d[mix].shape) * (rng.random(d[mix].shape) < 0.3), m)
            want = torch.tensor(rule(d, wall[idx], goal[idx], m), dtype=torch.float64)
            got = net.step(torch.tensor(d, dtype=torch.float64), wall_t, goal_t)
        else:
            d = torch.full(wall_t.shape, float(m), dtype=torch.float64)
            for _ in range(args.unroll):
                d = net.step(d, wall_t, goal_t)
            got, want = d, torch.tensor(target[idx], dtype=torch.float64)
        loss = (((got - want) / m) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % max(1, args.steps // 20) == 0 or it == 1:
            log.append({"step": it, "loss": loss.item(), "seconds": time.time() - t0})
            print(f"step {it}: loss {loss.item():.3e} ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(args.out, exist_ok=True)
    net.export(os.path.join(args.out, "planner.safetensors"))
    result = {
        "args": vars(args),
        "train_seconds": time.time() - t0,
        "log": log,
        "step_exact": step_error(net, m),
        "eval": evaluate(net, m),
    }
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=2)
    print(json.dumps({"step_exact": result["step_exact"], "eval": result["eval"]}, indent=2))


def analyze(path):
    """First-layer structure. Each weight is scaled by its input's range (d spans the
    ceiling, bits span 1) so channels compare. `unread` is the share on taps the
    4-neighbour rule never reads (the centre and diagonal distances, any neighbour's
    wall or goal); `orth` is the share of distance weight on the 4 orthogonal
    neighbours; `difference` is the share of distance-reading units whose distance
    weights cancel (|sum| < 5% of sum |w|), the signature of relu(a - b)."""
    t, meta = read_safetensors(path)
    w0 = t["0.weight"]
    m = json.loads(meta["wir.pads"])[0]
    u, c, k, _ = w0.shape
    r = k // 2
    span = np.array([m, 1.0, 1.0][:c]).reshape(1, c, 1, 1)
    a = np.abs(w0) * span
    read = np.zeros_like(a, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        read[:, 0, r + dy, r + dx] = True
    read[:, 1:, r, r] = True
    orth = read[:, 0]
    d = w0[:, 0]
    dmass = np.abs(d).sum(axis=(1, 2))
    live = dmass * m > 1e-6 * a.sum(axis=(1, 2, 3)).max()
    diff = (np.abs(d.sum(axis=(1, 2))) < 0.05 * dmass)[live]
    params = sum(v.size for v in t.values())
    return {
        "units": int(u),
        "params": int(params),
        "unread": float(a[~read].sum() / a.sum()),
        "orth": float((np.abs(d) * orth).sum() / np.abs(d).sum()),
        "difference": float(diff.mean()) if diff.size else 0.0,
        "distance_units": int(live.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("files", nargs="+")
    c = sub.add_parser("check")
    c.add_argument("file")
    e = sub.add_parser("eval")
    e.add_argument("files", nargs="+")
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--mode", choices=["step", "unroll"], default="step")
    t.add_argument("--hidden", default="6,5,2")
    t.add_argument("--init")
    t.add_argument("--ceiling", type=int, default=64)
    t.add_argument("--size", type=int, default=8)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--pool", type=int, default=16)
    t.add_argument("--unroll", type=int, default=48)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(6)
    if args.cmd == "train":
        return train(args)
    if args.cmd == "analyze":
        for f in args.files:
            print(f, json.dumps(analyze(f)))
        return 0
    if args.cmd == "check":
        net = StencilNet.load(args.file)
        m = net.ceiling
        exact, worst = step_error(net, m)
        print(f"one step on random fields: {exact:.6f} of cells exact, max |diff| {worst:.2e}")
        rng = np.random.default_rng(1)
        wall, goal, target = batch(200, 12, 12, m, rng)
        field, steps, _ = settle(net, wall, goal, m, 2 * m + 10)
        bad = int((np.abs(field - target) > 1e-6).any(axis=(1, 2)).sum())
        print(f"200 random 12x12 grids settled in {steps} steps; {bad} differ from BFS")
        return 1 if bad or worst > 1e-6 else 0
    for f in args.files:
        net = StencilNet.load(f)
        print(f, json.dumps({"step_exact": step_error(net, net.ceiling), "eval": evaluate(net, net.ceiling)}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
