"""model.py — un transformer minimal, configurable en profondeur et en boucles.

Deux leviers de « capacité » :
  - profondeur empilée : n_block blocs distincts, chacun ses paramètres ;
  - profondeur bouclée : les mêmes blocs réutilisés k fois.

La question mesurée : à calcul égal, combien de mémoire coûte la profondeur ?
"""
from __future__ import annotations

import numpy as np

from autograd import T


def init(rng, shape, scale=0.02):
    return T(rng.normal(0, scale, size=shape).astype(np.float32), req=True)


class Model:
    def __init__(self, vocab, d=64, ctx=96, n_block=2, loops=1, n_head=4, seed=0):
        self.vocab, self.d, self.ctx = vocab, d, ctx
        self.n_block, self.loops, self.n_head = n_block, loops, n_head
        self.dh = d // n_head
        rng = np.random.default_rng(seed)
        self.emb = init(rng, (vocab, d), 0.02)
        # encodage positionnel sinusoïdal fixe : aucun paramètre, pas de tranche
        pe = np.zeros((ctx, d), np.float32)
        pos = np.arange(ctx)[:, None]
        div = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)[:, : d // 2] if d % 2 == 0 else np.cos(pos * div)[:, : d // 2]
        self.pe = pe
        self.blocks = []
        for _ in range(n_block):
            b = {
                "ln1": init(rng, (d,), 0.0), "ln1b": T(np.zeros(d, np.float32), req=True),
                "wq": init(rng, (d, d), 0.02), "bq": T(np.zeros(d, np.float32), req=True),
                "wk": init(rng, (d, d), 0.02), "bk": T(np.zeros(d, np.float32), req=True),
                "wv": init(rng, (d, d), 0.02), "bv": T(np.zeros(d, np.float32), req=True),
                "wo": init(rng, (d, d), 0.02), "bo": T(np.zeros(d, np.float32), req=True),
                "ln2": init(rng, (d,), 0.0), "ln2b": T(np.zeros(d, np.float32), req=True),
                "w1": init(rng, (d, 4 * d), 0.02), "b1": T(np.zeros(4 * d, np.float32), req=True),
                "w2": init(rng, (4 * d, d), 0.02), "b2": T(np.zeros(d, np.float32), req=True),
                "lnf": init(rng, (d,), 0.0), "lnfb": T(np.zeros(d, np.float32), req=True),
            }
            self.blocks.append(b)
        self.loop_id = init(rng, (d,), 0.01)
        self.head = init(rng, (d, vocab), 0.02)

    # --- paramètres --------------------------------------------------------
    def params(self):
        ps = [self.emb, self.loop_id, self.head]
        for b in self.blocks:
            ps.extend(b.values())
        return ps

    def n_params(self):
        return int(sum(p.d.size for p in self.params()))

    def zero_grad(self):
        for p in self.params():
            p.g = None

    # --- briques -----------------------------------------------------------
    def _ln(self, x, g, b):
        return x.layernorm() * g + b

    def _block(self, x, b, mask):
        B, Tn, d = x.shape
        h = self._ln(x, b["ln1"], b["ln1b"])
        q = (h.matmul(b["wq"]) + b["bq"]).reshape(B, Tn, self.n_head, self.dh).permute((0, 2, 1, 3))
        k = (h.matmul(b["wk"]) + b["bk"]).reshape(B, Tn, self.n_head, self.dh).permute((0, 2, 1, 3))
        v = (h.matmul(b["wv"]) + b["bv"]).reshape(B, Tn, self.n_head, self.dh).permute((0, 2, 1, 3))
        att = q.matmul(k.transpose()) * np.float32(1.0 / np.sqrt(self.dh))
        att = (att + mask).softmax(-1)
        o = att.matmul(v).permute((0, 2, 1, 3)).reshape(B, Tn, d)
        x = x + o.matmul(b["wo"]) + b["bo"]
        h2 = self._ln(x, b["ln2"], b["ln2b"])
        mlp = ((h2.matmul(b["w1"]) + b["b1"]).gelu()).matmul(b["w2"]) + b["b2"]
        return x + mlp

    def forward(self, idx, targets=None):
        B, Tn = idx.shape
        x = self.emb.embedding(idx) + T(self.pe[:Tn].reshape(1, Tn, self.d))
        mask = np.triu(np.full((Tn, Tn), -1e9, np.float32), k=1)
        for it in range(self.loops):
            for b in self.blocks:
                x = self._block(x, b, mask)
            if self.loops > 1:
                x = x + self.loop_id.reshape(1, 1, self.d) * np.float32((it + 1) / self.loops)
        lb = self.blocks[-1]
        x = self._ln(x, lb["lnf"], lb["lnfb"])
        logits = x.reshape(B * Tn, self.d).matmul(self.head)
        loss = None
        if targets is not None:
            logits = logits.reshape(B * Tn, self.vocab)
            flat = targets.reshape(-1)
            n = flat.shape[0]
            # cross-entropy stable
            logits = logits + T(np.zeros((1, 1), np.float32))
            mx = T(logits.d.max(-1, keepdims=True))
            sh = logits - mx
            lse = sh.exp().sum(-1).log()
            logp = sh - lse.reshape(n, 1)
            # logp[i, y_i] : sélection par masque
            onehot = np.zeros((n, self.vocab), np.float32)
            onehot[np.arange(n), flat] = 1.0
            loss = (logp * T(onehot)).sum() * np.float32(-1.0 / n)
        return logits, loss

    @staticmethod
    def generate(model, idx, n, temperature=0.8, rng=None):
        rng = rng or np.random.default_rng(0)
        out = idx.copy()
        for _ in range(n):
            ctx = out[:, -model.ctx:]
            logits, _ = model.forward(ctx)
            logits = logits.d[-1]
            logits = logits / temperature
            p = np.exp(logits - logits.max())
            p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
            out = np.concatenate([out, np.array([[nxt]], dtype=out.dtype)], axis=1)
        return out
