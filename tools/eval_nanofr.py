#!/usr/bin/env python3
"""Perplexité de validation de NanoFR + bits par caractère.

    python3 tools/eval_nanofr.py data/nanofr_ckpt.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nanofr_train import Config, forward, rope_tables  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHARS_PER_TOKEN = 16.02e6 / 4.21e6  # mesuré sur le corpus d'entraînement


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "nanofr_ckpt.npz")
    ck = np.load(path)
    P = {k: ck[k] for k in ck.files if k != "step"}
    n_layers = 1 + max(int(k.split(".")[0][1:]) for k in P if k.startswith("l"))
    d = P["tok_embd"].shape[1]
    cfg = Config(vocab=P["tok_embd"].shape[0], d=d, n_layers=n_layers,
                 n_heads=4, n_kv=2, ffn=P["l0.w_gate"].shape[1], ctx=256)
    cos, sin = rope_tables(cfg.ctx, cfg.head_dim)

    total, n = 0.0, 0
    for split in ("val", "train"):
        arr = np.load(ROOT / "data" / "corpus" / f"{split}.npy").astype(np.int64)
        arr = arr[: (len(arr) // cfg.ctx) * cfg.ctx]
        rng = np.random.default_rng(0)
        starts = rng.choice(np.arange(0, len(arr) - cfg.ctx, cfg.ctx),
                            size=min(200, len(arr) // cfg.ctx), replace=False)
        losses = []
        for s in starts:
            s = int(s)
            idx = arr[None, s:s + cfg.ctx]
            tgt = arr[None, s + 1:s + 1 + cfg.ctx]
            logits = forward(P, cfg, idx, cos, sin)
            m = logits.max(-1, keepdims=True)
            logp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
            losses.append(-logp[0, np.arange(cfg.ctx), tgt[0]].mean())
        v = float(np.mean(losses))
        print(f"{split:5s} : loss {v:.4f} nats/token | ppl {np.exp(v):9.1f} | "
              f"{v/np.log(2)/CHARS_PER_TOKEN:.3f} bits/caractère")
        total += v
        n += 1
    print(f"étape du checkpoint : {int(ck['step'])}")


if __name__ == "__main__":
    main()
