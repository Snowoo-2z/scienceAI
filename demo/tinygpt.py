#!/usr/bin/env python3
"""
tinygpt.py — un vrai transformer, écrit à la main, entraîné ici.

Ce n'est pas une bibliothèque : c'est un GPT complet (embeddings, attention
multi-têtes causale, LayerNorm, MLP, tête liée), avec sa passe arrière écrite
à la main, AdamW, cosine schedule, clipping, évaluation et génération.
Aucune dépendance sauf numpy.

Il est petit parce que la machine est petite (2 vCPU, 3 Go). Il est réel :
entraîné sur 1,8 million de mots de littérature française du domaine public
(Les Misérables, Monte-Cristo, Notre-Dame de Paris, Madame Bovary, Candide…).

Commandes
---------
  python3 demo/tinygpt.py gradcheck          # vérifie la passe arrière
  python3 demo/tinygpt.py train              # entraîne et sauvegarde
  python3 demo/tinygpt.py sample "il était"  # génère du texte
  python3 demo/tinygpt.py info               # paramètres du modèle

Auteur : KAIROS · 2026-10-19.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ---------------------------------------------------------------------------
# 1. Tokenisation
# ---------------------------------------------------------------------------

class Tokenizer:
    """Tokenizer mots (fréquence) ou caractères, avec détokenisation propre."""

    def __init__(self, mode: str = "word", vocab_size: int = 12000):
        self.mode = mode
        self.vocab_size = vocab_size
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}

    @staticmethod
    def _split(text: str, mode: str) -> List[str]:
        t = unicodedata.normalize("NFC", text).lower()
        if mode == "char":
            return list(t)
        # mots, ponctuation et espaces significatifs
        return re.findall(r"[a-zà-ÿ0-9]+|[^\sa-zà-ÿ0-9]", t)

    def train(self, text: str) -> None:
        toks = self._split(text, self.mode)
        from collections import Counter
        counts = Counter(toks)
        specials = ["<pad>", "<unk>"]
        common = [w for w, _ in counts.most_common(self.vocab_size - len(specials))]
        self.itos = specials + common
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text: str) -> List[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(t, unk) for t in self._split(text, self.mode)]

    def decode(self, ids: List[int]) -> str:
        toks = [self.itos[i] for i in ids if 0 <= i < len(self.itos)]
        if self.mode == "char":
            return "".join(toks)
        CLOSERS = set(".,;:!?»)]")
        TIGHT = set("-'’")
        OPENERS = set("([")
        out = ""
        tight_after = False
        cap_next = True
        for t in toks:
            if t in CLOSERS:
                out = out.rstrip() + t
                tight_after = False
                cap_next = t in ".!?"
            elif t in TIGHT:
                out = out.rstrip() + t
                tight_after = True
            elif t in OPENERS:
                out += (" " if out else "") + t
                tight_after = True
            elif t == "«":
                out += (" " if out else "") + t
                tight_after = False
            else:
                word = t
                if cap_next and word[:1].isalpha():
                    word = word[0].upper() + word[1:]
                if tight_after or not out:
                    out += word
                else:
                    out += " " + word
                tight_after = False
                cap_next = False
        return out.replace("--", " — ").replace("  ", " ").strip()

    def state(self) -> Dict:
        return {"mode": self.mode, "vocab_size": self.vocab_size,
                "itos": self.itos}

    @classmethod
    def from_state(cls, st: Dict) -> "Tokenizer":
        t = cls(st["mode"], st["vocab_size"])
        t.itos = st["itos"]
        t.stoi = {w: i for i, w in enumerate(t.itos)}
        return t


# ---------------------------------------------------------------------------
# 2. Le modèle
# ---------------------------------------------------------------------------

def init_params(vocab: int, d: int, n_layer: int, n_head: int,
                block: int, seed: int = 1337) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    P: Dict[str, np.ndarray] = {}
    P["tok_emb"] = rng.normal(0, 0.02, (vocab, d)).astype(np.float32)
    P["pos_emb"] = rng.normal(0, 0.01, (block, d)).astype(np.float32)
    for l in range(n_layer):
        P[f"ln1_g_{l}"] = np.ones(d, np.float32)
        P[f"ln1_b_{l}"] = np.zeros(d, np.float32)
        P[f"Wqkv_{l}"] = rng.normal(0, 0.02, (d, 3 * d)).astype(np.float32)
        P[f"bqkv_{l}"] = np.zeros(3 * d, np.float32)
        P[f"Wo_{l}"] = rng.normal(0, 0.02 / math.sqrt(2), (d, d)).astype(np.float32)
        P[f"bo_{l}"] = np.zeros(d, np.float32)
        P[f"ln2_g_{l}"] = np.ones(d, np.float32)
        P[f"ln2_b_{l}"] = np.zeros(d, np.float32)
        P[f"W1_{l}"] = rng.normal(0, 0.02, (d, 4 * d)).astype(np.float32)
        P[f"b1_{l}"] = np.zeros(4 * d, np.float32)
        P[f"W2_{l}"] = rng.normal(0, 0.02 / math.sqrt(2), (4 * d, d)).astype(np.float32)
        P[f"b2_{l}"] = np.zeros(d, np.float32)
    P["lnf_g"] = np.ones(d, np.float32)
    P["lnf_b"] = np.zeros(d, np.float32)
    P["_meta"] = np.array([vocab, d, n_layer, n_head, block], dtype=np.int64)
    return P


def n_params(P: Dict[str, np.ndarray]) -> int:
    return sum(v.size for k, v in P.items() if not k.startswith("_"))


def layernorm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xn = (x - mu) * rstd
    return xn * g + b, (xn, rstd)


def layernorm_backward(dy, g, cache):
    xn, rstd = cache
    dg = (dy * xn).reshape(-1, xn.shape[-1]).sum(0)
    db = dy.reshape(-1, xn.shape[-1]).sum(0)
    dxn = dy * g
    n = xn.shape[-1]
    dx = (rstd / n) * (n * dxn
                       - dxn.sum(-1, keepdims=True)
                       - xn * (dxn * xn).sum(-1, keepdims=True))
    return dx, dg, db


def gelu(x: np.ndarray) -> np.ndarray:
    # x*x*x plutôt que x**3 : pow est 20 fois plus lent sur float32
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def gelu_backward(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    t = np.tanh(0.7978845608 * (x + 0.044715 * x * x * x))
    dt = (1 - t * t) * 0.7978845608 * (1 + 3 * 0.044715 * x * x)
    return dy * (0.5 * (1 + t) + 0.5 * x * dt)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def forward(P, idx, targets=None, train=False, dropout=0.0, rng=None):
    V, d, L, nh, block = [int(v) for v in P["_meta"]]
    B, T = idx.shape
    hd = d // nh
    caches = []

    x = P["tok_emb"][idx] + P["pos_emb"][:T]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    for l in range(L):
        xn1, ln1c = layernorm(x, P[f"ln1_g_{l}"], P[f"ln1_b_{l}"])
        qkv = xn1 @ P[f"Wqkv_{l}"] + P[f"bqkv_{l}"]
        q, k, v = np.split(qkv, 3, axis=-1)
        # (B,T,d) -> (B,nh,T,hd)
        q = q.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        att_logits = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hd)
        att_logits = np.where(mask, -1e9, att_logits)
        att = softmax(att_logits, axis=-1)
        y = att @ v                                    # (B,nh,T,hd)
        yc = y.transpose(0, 2, 1, 3).reshape(B, T, d)
        o = yc @ P[f"Wo_{l}"] + P[f"bo_{l}"]
        x = x + o

        xn2, ln2c = layernorm(x, P[f"ln2_g_{l}"], P[f"ln2_b_{l}"])
        h1 = xn2 @ P[f"W1_{l}"] + P[f"b1_{l}"]
        a1 = gelu(h1)
        h2 = a1 @ P[f"W2_{l}"] + P[f"b2_{l}"]
        x = x + h2

        caches.append((xn1, ln1c, q, k, v, att, yc, xn2, ln2c, h1, a1))

    xf, lnfc = layernorm(x, P["lnf_g"], P["lnf_b"])
    logits = xf @ P["tok_emb"].T                      # (B,T,V)

    loss = None
    if targets is not None:
        logits_f = logits.reshape(-1, V)
        tgt = targets.reshape(-1)
        m = logits_f.max(axis=1, keepdims=True)
        e = np.exp(logits_f - m)
        s = e.sum(axis=1, keepdims=True)
        probs = e / s
        rows = np.arange(tgt.size)
        loss = float(-np.log(probs[rows, tgt] + 1e-12).mean())
        dlogits = probs
        dlogits[rows, tgt] -= 1.0
        dlogits /= tgt.size
        dlogits = dlogits.reshape(B, T, V)
    else:
        dlogits = None

    return logits, loss, (caches, xf, lnfc, idx, logits, target_dummy(targets)) if targets is not None else (caches, xf, lnfc, idx)


def target_dummy(t):
    return t


def backward(P, cache, dlogits) -> Dict[str, np.ndarray]:
    V, d, L, nh, block = [int(v) for v in P["_meta"]]
    B, T = dlogits.shape[0], dlogits.shape[1]
    hd = d // nh
    caches, xf, lnfc, idx = cache[:4]
    grads: Dict[str, np.ndarray] = {}

    # tête liée : logits = xf @ tok_emb.T
    grads["tok_emb"] = (dlogits.reshape(-1, V).T @ xf.reshape(-1, d)).astype(np.float32)
    dxf = (dlogits.reshape(-1, V) @ P["tok_emb"]).reshape(B, T, d)
    dx, grads["lnf_g"], grads["lnf_b"] = layernorm_backward(dxf, P["lnf_g"], lnfc)

    for l in reversed(range(L)):
        xn1, ln1c, q, k, v, att, yc, xn2, ln2c, h1, a1 = caches[l]

        # --- MLP ---
        grads[f"W2_{l}"] = (a1.reshape(-1, 4 * d).T @ dx.reshape(-1, d)).astype(np.float32)
        grads[f"b2_{l}"] = dx.reshape(-1, d).sum(0).astype(np.float32)
        da1 = (dx.reshape(-1, d) @ P[f"W2_{l}"].T).reshape(B, T, 4 * d)
        dh1 = gelu_backward(h1, da1)
        grads[f"W1_{l}"] = (xn2.reshape(-1, d).T @ dh1.reshape(-1, 4 * d)).astype(np.float32)
        grads[f"b1_{l}"] = dh1.reshape(-1, 4 * d).sum(0).astype(np.float32)
        dxn2 = (dh1.reshape(-1, 4 * d) @ P[f"W1_{l}"].T).reshape(B, T, d)
        dln2, grads[f"ln2_g_{l}"], grads[f"ln2_b_{l}"] = layernorm_backward(dxn2, P[f"ln2_g_{l}"], ln2c)
        dx = dx + dln2

        # --- attention ---
        grads[f"Wo_{l}"] = (yc.reshape(-1, d).T @ dx.reshape(-1, d)).astype(np.float32)
        grads[f"bo_{l}"] = dx.reshape(-1, d).sum(0).astype(np.float32)
        dyc = (dx.reshape(-1, d) @ P[f"Wo_{l}"].T).reshape(B, T, d)
        dy = dyc.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)   # (B,nh,T,hd)

        datt = dy @ v.transpose(0, 1, 3, 2)                     # (B,nh,T,T)
        dv = att.transpose(0, 1, 3, 2) @ dy                     # (B,nh,T,hd)
        dlogits_a = att * (datt - (datt * att).sum(-1, keepdims=True))
        dq = (dlogits_a @ k) / math.sqrt(hd)
        dk = (dlogits_a.transpose(0, 1, 3, 2) @ q) / math.sqrt(hd)

        dq_ = dq.transpose(0, 2, 1, 3).reshape(B, T, d)
        dk_ = dk.transpose(0, 2, 1, 3).reshape(B, T, d)
        dv_ = dv.transpose(0, 2, 1, 3).reshape(B, T, d)
        dqkv = np.concatenate([dq_, dk_, dv_], axis=-1)
        grads[f"Wqkv_{l}"] = (xn1.reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)).astype(np.float32)
        grads[f"bqkv_{l}"] = dqkv.reshape(-1, 3 * d).sum(0).astype(np.float32)
        dxn1 = (dqkv.reshape(-1, 3 * d) @ P[f"Wqkv_{l}"].T).reshape(B, T, d)
        dln1, grads[f"ln1_g_{l}"], grads[f"ln1_b_{l}"] = layernorm_backward(dxn1, P[f"ln1_g_{l}"], ln1c)
        dx = dx + dln1

    # embeddings
    dtok = np.zeros_like(P["tok_emb"])
    np.add.at(dtok, idx.reshape(-1), dx.reshape(-1, d))
    grads["tok_emb"] = grads["tok_emb"] + dtok
    dpos = np.zeros_like(P["pos_emb"])
    dpos[:T] = dx.sum(0)
    grads["pos_emb"] = dpos
    return grads


# ---------------------------------------------------------------------------
# 3. Données
# ---------------------------------------------------------------------------

def load_ids(tok: Tokenizer, path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    ids = tok.encode(text)
    return np.array(ids, dtype=np.int32)


def batches(ids: np.ndarray, block: int, batch: int, rng: np.random.Generator,
            steps: int):
    for _ in range(steps):
        ix = rng.integers(0, len(ids) - block - 1, size=batch)
        x = np.stack([ids[i:i + block] for i in ix])
        y = np.stack([ids[i + 1:i + block + 1] for i in ix])
        yield x, y


# ---------------------------------------------------------------------------
# 4. Entraînement
# ---------------------------------------------------------------------------

def train(args) -> int:
    t_start = time.time()
    corpus = Path(args.corpus)
    text = corpus.read_text(encoding="utf-8")
    tok = Tokenizer(args.level, args.vocab)
    tok.train(text)
    ids = load_ids(tok, corpus)
    n_val = max(20000, len(ids) // 50)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    print(f"corpus : {len(text):,} caractères, {len(ids):,} tokens "
          f"(vocabulaire {len(tok.itos):,})".replace(",", " "))
    print(f"train : {len(train_ids):,} · val : {len(val_ids):,}".replace(",", " "))

    P = init_params(len(tok.itos), args.d, args.layers, args.heads, args.block, args.seed)
    print(f"modèle : {n_params(P) / 1e6:.2f} M paramètres · "
          f"d={args.d} · {args.layers} couches · {args.heads} têtes · bloc {args.block}")

    m = {k: np.zeros_like(v) for k, v in P.items() if not k.startswith("_")}
    vv = {k: np.zeros_like(v) for k, v in P.items() if not k.startswith("_")}
    b1, b2, eps = 0.9, 0.95, 1e-8
    lr_max, lr_min, warmup = args.lr, args.lr * 0.05, args.warmup
    rng = np.random.default_rng(args.seed + 1)

    log: List[Tuple[int, float, float]] = []
    best_val = 1e9
    step = 0
    def do_eval(step, final=False):
        nonlocal best_val, run_loss, t0
        avg = run_loss / args.log_every
        vl = evaluate(P, val_ids, args.block, args.batch, rng, iters=20)
        elapsed = time.time() - t0
        tps = args.log_every * args.batch * args.block / elapsed if elapsed else 0
        print(f"  step {step:6d} · perte train {avg:.4f} · perte val {vl:.4f} · "
              f"ppl {math.exp(min(vl, 20)):8.2f} · {tps:7.0f} tok/s · {elapsed:5.1f} s",
              flush=True)
        log.append((step, avg, vl))
        run_loss = 0.0
        t0 = time.time()
        if final or vl < best_val:
            best_val = min(best_val, vl)
            save(P, tok, args.ckpt, {"step": step, "val": vl, "train": avg,
                                     "params": n_params(P), "level": args.level,
                                     "corpus_chars": len(text),
                                     "d": args.d, "layers": args.layers,
                                     "heads": args.heads, "block": args.block,
                                     "vocab": len(tok.itos)})
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text(json.dumps(
            {"config": vars(args), "log": log, "params": n_params(P),
             "vocab": len(tok.itos), "tokens": int(len(ids))}, indent=2))

    data_iter = batches(train_ids, args.block, args.batch, rng, args.steps)
    t0 = time.time()
    run_loss = 0.0
    for x, y in data_iter:
        lr = (lr_max * (step + 1) / warmup if step < warmup else
              lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))))
        logits, loss, cache = forward(P, x, y)
        grads = backward(P, cache, _dlogits_from(P, logits, y))
        gnorm = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))
        if args.clip > 0 and gnorm > args.clip:
            k = args.clip / (gnorm + 1e-12)
            for g in grads.values():
                g *= k
        t_ = step + 1
        for name, g in grads.items():
            m[name] = b1 * m[name] + (1 - b1) * g
            vv[name] = b2 * vv[name] + (1 - b2) * (g * g)
            mh = m[name] / (1 - b1 ** t_)
            vh = vv[name] / (1 - b2 ** t_)
            P[name] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
            if args.wd > 0 and P[name].ndim == 2:
                P[name] -= (lr * args.wd * P[name]).astype(np.float32)
        run_loss += loss
        step += 1
        if step % args.log_every == 0:
            do_eval(step)
    do_eval(step, final=True)

    print(f"\nterminé en {(time.time() - t_start) / 60:.1f} min · "
          f"meilleure perte val {best_val:.4f} · ppl {math.exp(min(best_val, 20)):.2f}")
    return 0


def _dlogits_from(P, logits, y):
    V = logits.shape[-1]
    lf = logits.reshape(-1, V)
    tgt = y.reshape(-1)
    m = lf.max(axis=1, keepdims=True)
    e = np.exp(lf - m)
    probs = e / e.sum(axis=1, keepdims=True)
    d = probs
    d[np.arange(tgt.size), tgt] -= 1.0
    d /= tgt.size
    return d.reshape(logits.shape)


def evaluate(P, ids, block, batch, rng, iters=20) -> float:
    V, d, L, nh, blk = [int(v) for v in P["_meta"]]
    losses = []
    for x, y in batches(ids, block, batch, rng, iters):
        _, loss, _ = forward(P, x, y)
        losses.append(loss)
    return float(np.mean(losses))


def save(P, tok: Tokenizer, path: str, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in P.items() if not k.startswith("_")}
    np.savez_compressed(path, **arrays)
    Path(path + ".json").write_text(json.dumps(
        {**meta, "tokenizer": tok.state()}, ensure_ascii=False))


def load(path: str = "models/tinygpt.npz") -> Tuple[Dict[str, np.ndarray], Tokenizer, dict]:
    z = np.load(path)
    P = {k: z[k] for k in z.files}
    meta = json.loads(Path(path + ".json").read_text(encoding="utf-8"))
    tok = Tokenizer.from_state(meta["tokenizer"])
    P["_meta"] = np.array([len(tok.itos), meta["d"], meta["layers"],
                           meta["heads"], meta["block"]], dtype=np.int64)
    return P, tok, meta


# ---------------------------------------------------------------------------
# 5. Génération
# ---------------------------------------------------------------------------

def generate(P, tok: Tokenizer, prompt: str, n_tokens: int = 120,
             temperature: float = 0.85, top_k: int = 40,
             seed: Optional[int] = None) -> str:
    V, d, L, nh, block = [int(v) for v in P["_meta"]]
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt)
    if not ids:
        ids = [tok.stoi.get("<unk>", 0)]
    ctx = ids[-block:]
    out = list(ctx)
    for _ in range(n_tokens):
        x = np.array([out[-block:]], dtype=np.int64)
        logits, _, _ = forward(P, x)
        logit = logits[0, -1] / max(1e-6, temperature)
        # on interdit le token inconnu : mieux vaut un mot rare qu'un trou
        logit[tok.stoi.get("<unk>", 0)] = -1e9
        logit[tok.stoi.get("<pad>", 0)] = -1e9
        if top_k and top_k < V:
            kth = np.partition(logit, -top_k)[-top_k]
            logit = np.where(logit < kth, -1e9, logit)
        p = softmax(logit)
        nxt = int(rng.choice(V, p=p / p.sum()))
        out.append(nxt)
        if nxt == tok.stoi.get("<unk>") and len(out) > len(ctx) + 8:
            break
    return tok.decode(out)


# ---------------------------------------------------------------------------
# 6. Vérification du gradient
# ---------------------------------------------------------------------------

def gradcheck() -> int:
    rng = np.random.default_rng(0)
    V, d, L, nh, block = 11, 8, 2, 2, 6
    P = init_params(V, d, L, nh, block, seed=3)
    for k in P:
        if not k.startswith("_"):
            # float64 pour la vérification : en float32 l'écart de perte
            # (1e-9) est noyé dans le bruit de la somme
            P[k] = (P[k].astype(np.float64)
                    + rng.normal(0, 0.05, P[k].shape))
    x = rng.integers(0, V, (2, block))
    y = rng.integers(0, V, (2, block))
    logits, loss, cache = forward(P, x, y)
    grads = backward(P, cache, _dlogits_from(P, logits, y))

    print(f"  perte de référence : {loss:.6f}")
    ok = True
    names = ["tok_emb", "pos_emb", "ln1_g_0", "Wqkv_0", "Wo_0", "W1_0", "W2_1",
             "lnf_g", "b1_0"]
    eps = 1e-6
    for name in names:
        arr = P[name]
        # on choisit l'élément de plus fort gradient : les éléments quasi nuls
        # donnent des écarts relatifs trompeurs
        flat = np.argsort(-np.abs(grads[name].reshape(-1)))[:20]
        j = int(rng.integers(0, len(flat)))
        idxs = np.unravel_index(int(flat[j]), arr.shape)
        old = float(arr[idxs])
        arr[idxs] = old + eps
        _, lp, _ = forward(P, x, y)
        arr[idxs] = old - eps
        _, lm, _ = forward(P, x, y)
        arr[idxs] = old
        num = (lp - lm) / (2 * eps)
        ana = float(grads[name][idxs])
        rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
        flag = "OK" if rel < 1e-4 else "ÉCHEC"
        if rel >= 1e-4:
            ok = False
        print(f"  {name:10} numérique {num:+.6f} · analytique {ana:+.6f} · "
              f"écart relatif {rel:.2e} {flag}")
    print("  " + ("PASSE" if ok else "ÉCHEC"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("train")
    s.add_argument("--corpus", default="data/corpus-fr.txt")
    s.add_argument("--level", default="word", choices=["word", "char"])
    s.add_argument("--vocab", type=int, default=12000)
    s.add_argument("--d", type=int, default=224)
    s.add_argument("--layers", type=int, default=4)
    s.add_argument("--heads", type=int, default=4)
    s.add_argument("--block", type=int, default=128)
    s.add_argument("--batch", type=int, default=24)
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--lr", type=float, default=6e-4)
    s.add_argument("--warmup", type=int, default=80)
    s.add_argument("--clip", type=float, default=1.0)
    s.add_argument("--wd", type=float, default=0.05)
    s.add_argument("--log-every", type=int, default=50)
    s.add_argument("--seed", type=int, default=1337)
    s.add_argument("--ckpt", default="models/tinygpt.npz")
    s.add_argument("--log", default="results/tinygpt-train.json")

    g = sub.add_parser("sample")
    g.add_argument("prompt", nargs="?", default="il était une fois")
    g.add_argument("-n", type=int, default=120)
    g.add_argument("-t", type=float, default=0.85)
    g.add_argument("-k", type=int, default=40)
    g.add_argument("--ckpt", default="models/tinygpt.npz")
    g.add_argument("--seed", type=int, default=None)

    i = sub.add_parser("info")
    i.add_argument("--ckpt", default="models/tinygpt.npz")

    sub.add_parser("gradcheck")
    args = ap.parse_args()

    if args.cmd == "gradcheck":
        return gradcheck()
    if args.cmd == "info":
        P, tok, meta = load(args.ckpt)
        print(json.dumps({k: v for k, v in meta.items() if k != "tokenizer"},
                         indent=2, ensure_ascii=False))
        print(f"  vocabulaire : {len(tok.itos)} · mode {tok.mode}")
        print(f"  paramètres  : {n_params(P) / 1e6:.3f} M")
        print(f"  taille      : {Path(args.ckpt).stat().st_size / 1e6:.1f} Mo")
        return 0
    if args.cmd == "sample":
        P, tok, _ = load(args.ckpt)
        if args.seed is not None:
            np.random.seed(args.seed)
        print(generate(P, tok, args.prompt, args.n, args.t, args.k, args.seed))
        return 0
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
