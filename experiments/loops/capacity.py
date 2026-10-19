"""capacity.py — combien de faits tient dans un paramètre ?

Banc de mémorisation, indépendant du texte : on tire M associations
(clé aléatoire dans R^24 -> valeur parmi 32) et on demande au réseau de les
restituer.  Aucune structure : réussir = stocker, pas comprendre.

Trois substrats, tous au même nombre de multiplications près :

  deep1   1 couche cachée                 h = 256
  loop2   la même couche appliquée 2 fois  (paramètres inchangés, calcul x2)
  deep2   2 couches distinctes             (paramètres +h², calcul x2)

Ce que l'on veut savoir :
  * combien de faits par paramètre (donc par octet, donc par Mo) ;
  * une boucle ajoute-t-elle de la capacité *sans* ajouter de mémoire ?
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from autograd import T
from train import Adam

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "loops"

D_IN = 24
H = 256
V = 32          # taille du vocabulaire de sortie (5 bits par fait)
BATCH = 256


def he(rng, shape, fan_in):
    return T((rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32), req=True)


class MemNet:
    """Petit MLP de mémorisation. `loops` applications d'une même couche."""

    def __init__(self, loops=1, deep=False, seed=0, hidden=H):
        rng = np.random.default_rng(seed)
        self.loops = loops
        self.hidden = hidden
        self.params = []
        self.W1, self.b1 = he(rng, (D_IN, hidden), D_IN), T(np.zeros(hidden, np.float32), req=True)
        self.params += [self.W1, self.b1]
        n_mid = loops if deep else 1          # couches distinctes, ou une seule réutilisée
        self.mid = []
        for _ in range(n_mid):
            W = he(rng, (hidden, hidden), hidden)
            b = T(np.zeros(hidden, np.float32), req=True)
            self.mid.append((W, b))
            self.params += [W, b]
        self.W3, self.b3 = he(rng, (hidden, V), hidden), T(np.zeros(V, np.float32), req=True)
        self.params += [self.W3, self.b3]
        self.apps = 1 + loops                 # applications de couche au total

    def n_params(self):
        return sum(int(p.d.size) for p in self.params)

    def forward(self, x, y):
        h = (x.matmul(self.W1) + self.b1).gelu()
        for i in range(self.loops):
            W, b = self.mid[i] if len(self.mid) > 1 else self.mid[0]
            h = (h.matmul(W) + b).gelu()
        logits = h.matmul(self.W3) + self.b3
        # log-softmax stable, avec la valeur cible récupérée par produit one-hot
        m = logits.d.max(axis=-1, keepdims=True)
        shifted = logits + T((-m).astype(np.float32))
        lse = shifted.exp().sum(axis=-1, keepdims=True).log()
        onehot = np.zeros((y.shape[0], V), np.float32)
        onehot[np.arange(y.shape[0]), y] = 1.0
        ly = (shifted * T(onehot)).sum(axis=-1, keepdims=True)
        loss = (lse - ly).mean()
        return loss

    def zero_grad(self):
        for p in self.params:
            p.g = None

    def predict(self, x):
        h = (x.matmul(self.W1) + self.b1).gelu()
        for i in range(self.loops):
            W, b = self.mid[i] if len(self.mid) > 1 else self.mid[0]
            h = (h.matmul(W) + b).gelu()
        logits = h.matmul(self.W3) + self.b3
        return logits.d.argmax(-1)


def make_facts(n, seed, d=D_IN, v=V):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, v, size=n).astype(np.int64)
    return x, y


def run_one(kind, n_facts, steps=1500, lr=3e-3, seed=0, verbose=False):
    if kind == "deep1":
        model, loops, deep = MemNet(loops=1), 1, False
    elif kind == "loop2":
        model, loops, deep = MemNet(loops=2, deep=False), 2, False
    elif kind == "loop4":
        model, loops, deep = MemNet(loops=4, deep=False), 4, False
    elif kind == "deep2":
        model, loops, deep = MemNet(loops=2, deep=True), 2, True
    elif kind == "deep4":
        model, loops, deep = MemNet(loops=4, deep=True), 4, True
    elif kind == "wide1":
        # contrôle : mêmes paramètres que deep2 (~146 k) mais une seule couche
        # large, pour séparer « plus de paramètres » de « plus de profondeur ».
        model, loops, deep = MemNet(loops=1, hidden=384), 1, False
    else:
        raise ValueError(kind)

    x, y = make_facts(n_facts, seed=1)
    x_test, y_test = make_facts(512, seed=2)          # clés jamais vues
    rng = np.random.default_rng(seed)
    opt = Adam(model.params, lr=lr, b1=0.9, b2=0.99, wd=0.0)
    t0 = time.time()
    for step in range(steps):
        ix = rng.integers(0, n_facts, size=min(BATCH, n_facts))
        xb, yb = T(x[ix]), y[ix]
        loss = model.forward(xb, yb)
        model.zero_grad()
        loss.backward()
        opt.step()
        if verbose and step % 250 == 0:
            print(f"    {kind} M={n_facts} step {step} loss {float(loss.d):.4f}", flush=True)

    train_acc = float((model.predict(T(x)) == y).mean())
    test_acc = float((model.predict(T(x_test)) == y_test).mean())
    return {
        "kind": kind,
        "facts": n_facts,
        "params": model.n_params(),
        "applications": model.apps,
        "train_acc": round(train_acc, 4),
        "test_acc_unseen_keys": round(test_acc, 4),
        "chance": round(1.0 / V, 4),
        "bits_per_fact": round(float(np.log2(V)), 3),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--kinds", default="deep1,loop2,deep2")
    ap.add_argument("--facts", default="512,1024,2048")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="capacity")
    args = ap.parse_args()

    kinds = args.kinds.split(",")
    facts = [int(f) for f in args.facts.split(",")]
    rows = []
    for kind in kinds:
        for m in facts:
            r = run_one(kind, m, steps=args.steps, seed=args.seed, verbose=True)
            rows.append(r)
            print(f"  {kind:6s} M={m:6d}  params {r['params']:7d}  "
                  f"train {r['train_acc']*100:5.1f}%  test {r['test_acc_unseen_keys']*100:4.1f}%  "
                  f"{r['seconds']:5.1f}s", flush=True)

    print("\nFaits restitués (>=90 % en train) par 1 000 paramètres :")
    for kind in kinds:
        ok = [r for r in rows if r["kind"] == kind and r["train_acc"] >= 0.90]
        if not ok:
            print(f"  {kind:6s} : < {min(facts)} faits")
            continue
        best = max(ok, key=lambda r: r["facts"])
        per_k = best["facts"] / best["params"] * 1000
        bits_per_param = per_k * best["bits_per_fact"] / 1000
        print(f"  {kind:6s} : {best['facts']} faits / {best['params']} params "
              f"=> {per_k:.2f} faits par 1k params = {bits_per_param:.3f} bit/param")

    path = OUT / f"{args.out}.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"\nécrit {path}")


if __name__ == "__main__":
    main()
