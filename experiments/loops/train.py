"""train.py — entraîne et compare des transformers à mémoire égale ou à calcul égal.

  python3 experiments/loops/train.py --name deep4 --blocks 4 --loops 1 --steps 800
  python3 experiments/loops/train.py --sweep            # toutes les configs
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from model import Model

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "experiments" / "loops" / "data" / "input.txt"
OUT = ROOT / "results" / "loops"


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.99, eps=1e-8, wd=0.0):
        self.p = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = [np.zeros_like(x.d) for x in params]
        self.v = [np.zeros_like(x.d) for x in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.p):
            if p.g is None:
                continue
            g = p.g
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.d -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)
            if self.wd:
                p.d -= (self.lr * self.wd * p.d).astype(np.float32)
            p.g = None


def load(path, ctx):
    text = Path(path).read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    return data[:n], data[n:], "".join(chars)


def batches(data, ctx, batch, rng, steps):
    for _ in range(steps):
        ix = rng.integers(0, len(data) - ctx - 1, size=batch)
        x = np.stack([data[i:i + ctx] for i in ix])
        y = np.stack([data[i + 1:i + ctx + 1] for i in ix])
        yield x, y


def evaluate(model, data, ctx, batch, rng, n_batch=8):
    tot = 0.0
    for _ in range(n_batch):
        ix = rng.integers(0, len(data) - ctx - 1, size=batch)
        x = np.stack([data[i:i + ctx] for i in ix])
        y = np.stack([data[i + 1:i + ctx + 1] for i in ix])
        _, loss = model.forward(x, y)
        tot += float(loss.d)
        model.zero_grad()
    return tot / n_batch


def run(name, blocks, loops, steps, d=64, ctx=96, batch=12, lr=3e-3, seed=0, log_every=50):
    rng = np.random.default_rng(seed)
    train, val, chars = load(DATA, ctx)
    model = Model(len(chars), d=d, ctx=ctx, n_block=blocks, loops=loops, seed=seed)
    n_par = model.n_params()
    opt = Adam(model.params(), lr=lr, wd=0.01)
    hist, t0 = [], time.time()
    gen = batches(train, ctx, batch, rng, steps)
    for step, (x, y) in enumerate(gen, 1):
        _, loss = model.forward(x, y)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            vl = evaluate(model, val, ctx, batch, rng, 4)
            hist.append({"step": step, "train": float(loss.d), "val": vl})
            el = time.time() - t0
            print(f"  {name:10s} step {step:4d}  train {loss.d:.3f}  val {vl:.3f}  "
                  f"{el:5.0f}s  {step*batch*ctx/el:6.0f} tok/s", flush=True)
    final = evaluate(model, val, ctx, batch, rng, 10)
    idx = np.array([[next(iter(range(len(chars))))]], dtype=np.int64)
    idx = np.array([[0]], dtype=np.int64)
    sample = Model.generate(model, idx, 400, temperature=0.8, rng=np.random.default_rng(1))
    text_out = "".join(chars[i] for i in sample[0])
    rec = {
        "name": name, "blocks": blocks, "loops": loops, "d": d, "ctx": ctx,
        "batch": batch, "steps": steps, "lr": lr, "seed": seed,
        "params": n_par,
        "block_applications": blocks * loops,
        "val_loss": final,
        "val_bits_per_char": final / math.log(2),
        "seconds": round(time.time() - t0, 1),
        "hist": hist,
        "sample": text_out,
        "date": "2026-10-19",
        "hardware": "Xeon 2.60GHz, 2 vCPU, 4 Go, NumPy 2.4.6, float32",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    (OUT / f"{name}.txt").write_text(
        f"{name} : {n_par/1e3:.1f} k paramètres, {blocks} blocs × {loops} boucles = "
        f"{blocks*loops} applications\nval loss {final:.4f} ({final/math.log(2):.3f} bits/car)\n\n"
        + text_out + "\n")
    print(f"{name:10s} FINAL val {final:.4f} ({final/math.log(2):.3f} bpc) "
          f"{n_par/1e3:.0f}k params {rec['seconds']:.0f}s", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    if args.sweep:
        cfgs = [
            ("deep1", 1, 1),
            ("deep2", 2, 1),
            ("loop2", 1, 2),
            ("deep4", 4, 1),
            ("loop4", 1, 4),
            ("loop2x2", 2, 2),
            ("loop8", 1, 8),
        ]
        summary = []
        for name, b, l in cfgs:
            summary.append(run(name, b, l, args.steps, d=args.d, ctx=args.ctx,
                               batch=args.batch, lr=args.lr, seed=args.seed))
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\n{'config':10s} {'params':>8s} {'applis':>7s} {'val':>7s} {'bpc':>6s}")
        for r in summary:
            print(f"{r['name']:10s} {r['params']:8d} {r['block_applications']:7d} "
                  f"{r['val_loss']:7.3f} {r['val_bits_per_char']:6.3f}")
        return

    run(args.name or "run", args.blocks, args.loops, args.steps, d=args.d,
        ctx=args.ctx, batch=args.batch, lr=args.lr, seed=args.seed)


if __name__ == "__main__":
    main()
