#!/usr/bin/env python3
"""NanoFR : petit transformer « architecture LLaMA » écrit en numpy, entraîné
sur CPU sous 1 Go de RAM.

Aucune dépendance à PyTorch : forward + backward à la main, AdamW, RoPE
split-half, RMSNorm, SwiGLU, GQA, embeddings partagés (weight tying).

Le modèle produit est exportable tel quel en GGUF (voir tools/export_gguf.py)
et donc exécutable par llama.cpp.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ----------------------------------------------------------------------------
# briques
# ----------------------------------------------------------------------------
def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    inv = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return x * inv * w


def rmsnorm_backward(dy, x, w, inv, eps=1e-5):
    dw = (dy * x * inv).sum(axis=(0, 1))
    dxhat = dy * w
    dx = inv * (dxhat - (dxhat * x).mean(-1, keepdims=True) * inv * inv * x)
    return dx, dw


def silu(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return x * s, s


def rope_tables(seq_len: int, head_dim: int, base: float = 10000.0):
    half = head_dim // 2
    inv_freq = base ** (-np.arange(half, dtype=np.float64) / half)
    pos = np.arange(seq_len, dtype=np.float64)
    ang = np.outer(pos, inv_freq)          # (T, half)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x, cos, sin):
    """x: (B, T, H, D) -> rotation split-half (layout llama.cpp NORM)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def apply_rope_backward(dy, cos, sin):
    half = dy.shape[-1] // 2
    d1, d2 = dy[..., :half], dy[..., half:]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return np.concatenate([d1 * c + d2 * s, -d1 * s + d2 * c], axis=-1)


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


# ----------------------------------------------------------------------------
# modèle
# ----------------------------------------------------------------------------
class Config:
    def __init__(self, vocab=8192, d=256, n_layers=6, n_heads=4, n_kv=2,
                 ffn=704, ctx=256, rope_base=10000.0, eps=1e-5):
        assert d % n_heads == 0
        self.vocab, self.d, self.n_layers = vocab, d, n_layers
        self.n_heads, self.n_kv = n_heads, n_kv
        self.head_dim = d // n_heads
        self.ffn, self.ctx, self.rope_base, self.eps = ffn, ctx, rope_base, eps


def init_params(cfg: Config, seed=0):
    rng = np.random.default_rng(seed)
    d, f = cfg.d, cfg.ffn
    P = {}
    scale = 0.02
    P["tok_embd"] = (rng.standard_normal((cfg.vocab, d)) * scale).astype(np.float32)
    for i in range(cfg.n_layers):
        p = f"l{i}."
        P[p + "attn_norm"] = np.ones(d, dtype=np.float32)
        P[p + "wq"] = (rng.standard_normal((d, cfg.n_heads * cfg.head_dim)) / math.sqrt(d)).astype(np.float32)
        P[p + "wk"] = (rng.standard_normal((d, cfg.n_kv * cfg.head_dim)) / math.sqrt(d)).astype(np.float32)
        P[p + "wv"] = (rng.standard_normal((d, cfg.n_kv * cfg.head_dim)) / math.sqrt(d)).astype(np.float32)
        P[p + "wo"] = (rng.standard_normal((cfg.n_heads * cfg.head_dim, d)) / math.sqrt(d * 1.0)).astype(np.float32)
        P[p + "ffn_norm"] = np.ones(d, dtype=np.float32)
        P[p + "w_gate"] = (rng.standard_normal((d, f)) / math.sqrt(d)).astype(np.float32)
        P[p + "w_up"] = (rng.standard_normal((d, f)) / math.sqrt(d)).astype(np.float32)
        P[p + "w_down"] = (rng.standard_normal((f, d)) / math.sqrt(f)).astype(np.float32)
        # sortie de l'attention : petite init pour un départ stable
        P[p + "wo"] *= 1.0 / math.sqrt(2 * cfg.n_layers)
    P["out_norm"] = np.ones(d, dtype=np.float32)
    return P


def forward(P, cfg: Config, idx: np.ndarray, cos, sin, keep=False):
    """idx: (B, T) int -> logits (B, T, V)"""
    B, T = idx.shape
    x = P["tok_embd"][idx]                      # (B,T,d)
    cache = {"idx": idx, "x0": x, "cos": cos, "sin": sin, "layers": []}
    H, KV, D = cfg.n_heads, cfg.n_kv, cfg.head_dim
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    for i in range(cfg.n_layers):
        p = f"l{i}."
        x_in = x
        h = rmsnorm(x, P[p + "attn_norm"], cfg.eps)
        inv = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + cfg.eps)
        q = (h @ P[p + "wq"]).reshape(B, T, H, D)
        k = (h @ P[p + "wk"]).reshape(B, T, KV, D)
        v = (h @ P[p + "wv"]).reshape(B, T, KV, D)
        qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = H // KV
        kk = np.repeat(kr, rep, axis=2)
        vv = np.repeat(v, rep, axis=2)
        # scores (B,H,T,T)
        qh = qr.transpose(0, 2, 1, 3)
        kh = kk.transpose(0, 2, 3, 1)
        att = (qh @ kh) / math.sqrt(D)
        att = np.where(mask[None, None], -np.inf, att)
        probs = softmax(att, axis=-1)
        vh = vv.transpose(0, 2, 1, 3)
        ctx = (probs @ vh).transpose(0, 2, 1, 3).reshape(B, T, H * D)
        o = ctx @ P[p + "wo"]
        x = x + o
        x_mid = x
        h2 = rmsnorm(x, P[p + "ffn_norm"], cfg.eps)
        inv2 = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + cfg.eps)
        g = h2 @ P[p + "w_gate"]
        u = h2 @ P[p + "w_up"]
        sg, sig = silu(g)
        ff = (sg * u) @ P[p + "w_down"]
        x = x + ff
        if keep:
            cache["layers"].append(dict(h=h, inv=inv, x_in=x_in, q=qr, k=kr, v=v,
                                        probs=probs, ctx=ctx, h2=h2, inv2=inv2,
                                        x_mid=x_mid, g=g, u=u, sig=sig, sg=sg))
    x_final = x
    xn = rmsnorm(x, P["out_norm"], cfg.eps)
    invn = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + cfg.eps)
    logits = xn @ P["tok_embd"].T
    cache.update(x=x_final, xn=xn, invn=invn, logits=logits)
    return (logits, cache) if keep else logits


def loss_and_grads(P, cfg: Config, idx, targets, cos, sin):
    B, T = idx.shape
    logits, c = forward(P, cfg, idx, cos, sin, keep=True)
    V = cfg.vocab
    lg = logits.reshape(-1, V)
    tg = targets.reshape(-1)
    m = lg.max(axis=-1, keepdims=True)
    e = np.exp(lg - m)
    Z = e.sum(axis=-1, keepdims=True)
    logp = lg - m - np.log(Z)
    n = tg.size
    loss = -logp[np.arange(n), tg].mean()
    dlogits = e / Z
    dlogits[np.arange(n), tg] -= 1.0
    dlogits /= n
    dlogits = dlogits.reshape(B, T, V).astype(np.float32)

    G = {k: np.zeros_like(v) for k, v in P.items()}
    # sortie
    xn2 = c["xn"].reshape(-1, cfg.d)
    G["tok_embd"] += dlogits.reshape(-1, V).T @ xn2
    dx = dlogits @ P["tok_embd"]                       # (B,T,d)
    G["out_norm"] = (dx * c["x"] * c["invn"]).sum(axis=(0, 1))
    dxhat = dx * P["out_norm"]
    dx = c["invn"] * (dxhat - (dxhat * c["x"]).mean(-1, keepdims=True) * c["invn"] ** 2 * c["x"])

    for i in reversed(range(cfg.n_layers)):
        p = f"l{i}."
        L = c["layers"][i]
        # --- FFN ---
        d_ff = dx
        G[p + "w_down"] += (L["sg"] * L["u"]).reshape(-1, cfg.ffn).T @ d_ff.reshape(-1, cfg.d)
        d_sgu = d_ff @ P[p + "w_down"].T                  # (B,T,ffn)
        d_g = d_sgu * L["u"] * (L["sig"] * (1 + L["g"] * (1 - L["sig"])))
        d_u = d_sgu * L["sg"]
        G[p + "w_gate"] += L["h2"].reshape(-1, cfg.d).T @ d_g.reshape(-1, cfg.ffn)
        G[p + "w_up"] += L["h2"].reshape(-1, cfg.d).T @ d_u.reshape(-1, cfg.ffn)
        dh2 = d_g @ P[p + "w_gate"].T + d_u @ P[p + "w_up"].T
        G[p + "ffn_norm"] = (dh2 * L["x_mid"] * L["inv2"]).sum(axis=(0, 1))
        dxh = dh2 * P[p + "ffn_norm"]
        dx_ffn = L["inv2"] * (dxh - (dxh * L["x_mid"]).mean(-1, keepdims=True) * L["inv2"] ** 2 * L["x_mid"])
        dx = dx + dx_ffn
        # --- attention ---
        d_o = dx
        G[p + "wo"] += L["ctx"].reshape(-1, cfg.d).T @ d_o.reshape(-1, cfg.d)
        d_ctx = d_o @ P[p + "wo"].T                        # (B,T,H*D)
        d_ctx = d_ctx.reshape(B, T, cfg.n_heads, cfg.head_dim)
        d_probs = d_ctx.transpose(0, 2, 1, 3) @ np.repeat(L["v"], cfg.n_heads // cfg.n_kv, axis=2).transpose(0, 2, 3, 1)
        d_v = (L["probs"].transpose(0, 1, 3, 2) @ d_ctx.transpose(0, 2, 1, 3))
        d_v = d_v.transpose(0, 2, 1, 3)
        # softmax backward
        dp = d_probs
        ds = L["probs"] * (dp - (dp * L["probs"]).sum(-1, keepdims=True))
        ds /= math.sqrt(cfg.head_dim)
        d_q = (ds @ np.repeat(L["k"], cfg.n_heads // cfg.n_kv, axis=2).transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        d_k = (ds.transpose(0, 1, 3, 2) @ L["q"].transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        # GQA : sommer les gradients des têtes groupées
        if cfg.n_heads != cfg.n_kv:
            rep = cfg.n_heads // cfg.n_kv
            d_k = d_k.reshape(B, T, cfg.n_kv, rep, cfg.head_dim).sum(axis=3)
            d_v = d_v.reshape(B, T, cfg.n_kv, rep, cfg.head_dim).sum(axis=3)
        d_q = apply_rope_backward(d_q, cos, sin)
        d_k = apply_rope_backward(d_k, cos, sin)
        d_q = d_q.reshape(B, T, cfg.d)
        d_k = d_k.reshape(B, T, cfg.n_kv * cfg.head_dim)
        d_v = d_v.reshape(B, T, cfg.n_kv * cfg.head_dim)
        G[p + "wq"] += L["h"].reshape(-1, cfg.d).T @ d_q.reshape(-1, cfg.d)
        G[p + "wk"] += L["h"].reshape(-1, cfg.d).T @ d_k.reshape(-1, cfg.n_kv * cfg.head_dim)
        G[p + "wv"] += L["h"].reshape(-1, cfg.d).T @ d_v.reshape(-1, cfg.n_kv * cfg.head_dim)
        dh = d_q @ P[p + "wq"].T + d_k @ P[p + "wk"].T + d_v @ P[p + "wv"].T
        G[p + "attn_norm"] = (dh * L["x_in"] * L["inv"]).sum(axis=(0, 1))
        dxh = dh * P[p + "attn_norm"]
        dx_att = L["inv"] * (dxh - (dxh * L["x_in"]).mean(-1, keepdims=True) * L["inv"] ** 2 * L["x_in"])
        dx = dx + dx_att
    # gradient des embeddings d'entrée (poids partagés)
    np.add.at(G["tok_embd"], idx.reshape(-1), dx.reshape(-1, cfg.d))
    return loss, G


# ----------------------------------------------------------------------------
# AdamW
# ----------------------------------------------------------------------------
class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, wd=0.01):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, P, G, lr):
        self.t += 1
        b1, b2 = self.b1, self.b2
        for k in P:
            g = G[k]
            if k.endswith("_norm"):
                g = g  # pas de weight decay sur les norms
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            upd = mhat / (np.sqrt(vhat) + self.eps)
            if not k.endswith("_norm"):
                upd = upd + self.wd * P[k]
            P[k] -= (lr * upd).astype(np.float32)


# ----------------------------------------------------------------------------
# génération
# ----------------------------------------------------------------------------
def generate(P, cfg, cos, sin, prompt_ids, n_new=60, temp=0.85, top_k=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = list(prompt_ids)
    out = []
    for _ in range(n_new):
        ctx = ids[-cfg.ctx:]
        x = np.array([ctx], dtype=np.int64)
        logits = forward(P, cfg, x, cos[:len(ctx)], sin[:len(ctx)])
        z = logits[0, -1].astype(np.float64) / temp
        if top_k:
            kth = np.partition(z, -top_k)[-top_k]
            z[z < kth] = -np.inf
        p = np.exp(z - z.max())
        p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        ids.append(nxt)
        out.append(nxt)
        if nxt == 2:  # </s>
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=704)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--sample-every", type=int, default=200)
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="arrêt après N secondes (0 = pas de limite)")
    args = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(DATA / "tokenizer_nanofr.model"))
    train = np.load(DATA / "corpus" / "train.npy", mmap_mode="r")
    val = np.load(DATA / "corpus" / "val.npy", mmap_mode="r")
    train = np.asarray(train)
    val = np.asarray(val)

    cfg = Config(vocab=sp.get_piece_size(), d=args.d, n_layers=args.layers,
                 n_heads=args.heads, n_kv=args.kv, ffn=args.ffn, ctx=args.ctx)
    P = init_params(cfg, seed=1234)
    n_params = sum(v.size for v in P.values())
    print(f"paramètres : {n_params/1e6:.2f} M  (d={cfg.d}, L={cfg.n_layers}, "
          f"heads={cfg.n_heads}/{cfg.n_kv}, ffn={cfg.ffn}, ctx={cfg.ctx})")
    print(f"corpus     : {len(train)/1e6:.2f} M tokens train, {len(val)/1e6:.2f} M val")

    cos, sin = rope_tables(cfg.ctx, cfg.head_dim, cfg.rope_base)
    opt = AdamW(P)
    rng = np.random.default_rng(7)
    t0 = time.time()
    log = []
    ctx = cfg.ctx

    for step in range(1, args.steps + 1):
        frac = step / args.steps
        warm = min(1.0, step / 60.0)
        lr = args.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))
        G = None
        tot_loss = 0.0
        for _ in range(args.accum):
            starts = rng.integers(0, len(train) - ctx - 1, size=args.batch)
            idx = np.stack([train[s:s + ctx] for s in starts]).astype(np.int64)
            tgt = np.stack([train[s + 1:s + 1 + ctx] for s in starts]).astype(np.int64)
            loss, Gb = loss_and_grads(P, cfg, idx, tgt, cos, sin)
            tot_loss += loss
            if G is None:
                G = Gb
            else:
                for k in G:
                    G[k] += Gb[k]
        if args.accum > 1:
            for k in G:
                G[k] /= args.accum
        # clipping global
        gn = math.sqrt(sum(float((g * g).sum()) for g in G.values()))
        if gn > 1.0:
            for k in G:
                G[k] *= 1.0 / gn
        opt.step(P, G, lr)

        if step % 10 == 0 or step == 1:
            dt = time.time() - t0
            tok = step * args.batch * ctx
            print(f"step {step:5d}  loss {tot_loss/args.accum:6.3f}  lr {lr:.2e}  "
                  f"|g| {gn:5.2f}  {tok/dt/1e3:6.1f} k tok/s  {dt/60:5.1f} min", flush=True)
            log.append({"step": step, "loss": float(tot_loss / args.accum), "gn": float(gn), "t": float(dt)})

        if step % args.sample_every == 0 or step == args.steps:
            prompt = "Il était une fois"
            ids = [1] + sp.encode(prompt)
            out = generate(P, cfg, cos, sin, ids, n_new=60, seed=step)
            print("   > " + prompt + sp.decode(out).replace("\n", " "), flush=True)
            np.savez(DATA / "nanofr_ckpt.npz", step=step, **P)
            (DATA / "nanofr_log.json").write_text(json.dumps(log, indent=1))

        if args.time_budget and time.time() - t0 > args.time_budget:
            print("budget temps atteint", flush=True)
            break

    np.savez(DATA / "nanofr_ckpt.npz", step=step, **P)
    (DATA / "nanofr_log.json").write_text(json.dumps(log, indent=1))
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"checkpoint -> {DATA / 'nanofr_ckpt.npz'}  (pic RSS {rss:.0f} Mo)")


if __name__ == "__main__":
    main()
