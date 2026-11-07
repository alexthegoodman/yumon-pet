"""Cross-check wir weight files against NumPy and PyTorch.

Run from yumon-pet/wir after `cargo build`, using the python-train env:

    uv run --project ../python-train python scripts/interop.py

1. Compiles examples/xor.wir to .npz and .safetensors with the wir CLI, loads
   them with NumPy and into torch.nn.Sequential(Linear, ReLU, Linear), and
   checks the XOR truth table.
2. Writes a random network with np.savez, decompiles it with the CLI,
   recompiles the text, and checks the function on 10,000 points.
3. Checks that np.savez_compressed is rejected with a clear message.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "target", "debug", "wir.exe" if os.name == "nt" else "wir")


def wir(*args, check=True):
    r = subprocess.run([EXE, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"wir {' '.join(args)} failed:\n{r.stderr}")
    return r


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        body = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        dtype = {"F64": np.float64, "F32": np.float32}[info["dtype"]]
        out[name] = torch.from_numpy(np.frombuffer(body[a:b], dtype=dtype).reshape(info["shape"]).copy())
    return out, header.get("__metadata__", {})


def numpy_forward(z, x):
    return np.maximum(x @ z["W1"].T + z["b1"], 0) @ z["W2"].T + z["b2"]


def main():
    tmp = tempfile.mkdtemp(prefix="wir-interop-")
    xor_pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    xor_out = np.array([[0], [1], [1], [0]], dtype=np.float64)

    # 1. wir -> NumPy / PyTorch
    npz = os.path.join(tmp, "xor.npz")
    st = os.path.join(tmp, "xor.safetensors")
    wir("compile", os.path.join(ROOT, "examples", "xor.wir"), "-o", npz, "--quiet")
    wir("compile", os.path.join(ROOT, "examples", "xor.wir"), "-o", st, "--quiet")
    z = np.load(npz)
    meta = json.loads(bytes(z["wir_meta"]).decode())
    assert np.array_equal(numpy_forward(z, xor_pts), xor_out), "numpy forward"
    print(f"numpy {np.__version__}: xor.npz keys {sorted(z.files)}, meta {meta}, truth table ok")

    state, st_meta = load_safetensors(st)
    for dtype in (torch.float64, torch.float32):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1)).to(dtype)
        model.load_state_dict(state)
        with torch.no_grad():
            y = model(torch.tensor(xor_pts, dtype=dtype))
        assert torch.equal(y, torch.tensor(xor_out, dtype=dtype)), f"torch forward {dtype}"
    print(f"torch {torch.__version__}: xor.safetensors loads into nn.Sequential (f64 and f32), meta {st_meta}, truth table ok")

    # 2. NumPy -> wir -> NumPy
    rng = np.random.default_rng(0)
    worst = 0.0
    for d, h in [(1, 9), (2, 16), (3, 12)]:
        net = {
            "W1": rng.uniform(-2, 2, (h, d)),
            "b1": rng.uniform(-2, 2, h),
            "W2": rng.uniform(-2, 2, (1, h)),
            "b2": rng.uniform(-1, 1, 1),
        }
        src = os.path.join(tmp, f"rand{d}.npz")
        np.savez(src, **net)
        text = os.path.join(tmp, f"rand{d}.wir")
        back = os.path.join(tmp, f"rand{d}_back.npz")
        wir("decompile", src, "--exact", "-o", text)
        wir("compile", text, "-o", back, "--quiet")
        x = rng.uniform(-3, 3, (10_000, d))
        diff = np.abs(numpy_forward(net, x) - numpy_forward(np.load(back), x)).max()
        worst = max(worst, diff)
        assert diff < 1e-9, f"D={d}: {diff}"
    print(f"np.savez random nets (D=1,2,3) -> decompile --exact -> compile: worst |diff| on 10,000 points = {worst:.3e}")

    # 3. Compressed npz is refused clearly.
    comp = os.path.join(tmp, "compressed.npz")
    np.savez_compressed(comp, **net)
    r = wir("decompile", comp, check=False)
    assert r.returncode != 0 and "np.savez_compressed" in r.stderr, r.stderr
    print(f"np.savez_compressed rejected: {r.stderr.strip()}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
