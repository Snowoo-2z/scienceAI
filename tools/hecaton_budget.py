#!/usr/bin/env python3
"""
hecaton_budget.py — calculateur de budget pour le régime « ~100 Go de RAM ».

Auteur : KAIROS · branche arena/c9070fe4-scienceai · 2026-10-19
Aucune dépendance (stdlib pure), testé Python 3.9+.

Ce que fait l'outil
-------------------
Étant donné une spécification de modèle et une plateforme matérielle, calcule :

  * la décomposition exacte de la mémoire (poids par groupe, KV, état récurrent,
    runtime, buffers) ;
  * les octets lus par token en décodage batch=1 (le vrai goulot du régime) ;
  * le débit roofline : min(bande passante / octets, calcul / FLOPs) ;
  * l'énergie par token et par réponse utile ;
  * le coût d'entraînement (6ND) et le coût de distillation ;
  * la loi de budget, c'est-à-dire ce qui tient dans N Go selon les bits.

Sous-commandes
--------------
  show    <config>   détail complet d'une configuration
  table              tableau comparatif des configurations connues
  sweep              balayage « ce qui tient dans X Go » (bits, top-k, experts)
  law                loi de budget 100 Mo -> 100 Go (le pont entre les branches)
  hw                 le même modèle sur plusieurs plateformes
  train              coût d'entraînement / distillation
  svg                figures (budget mémoire, roofline) en SVG autonome

Conventions
-----------
  * 1 Go = 10^9 octets, 1 GiB = 2^30 octets. Tout est calculé en octets puis
    affiché en Go décimaux (sauf mention « GiB »).
  * « bits effectifs » = bits réellement stockés, échelles et métadonnées
    incluses (approximation : +5 % par rapport aux bits nominaux).
  * Le débit est un **plafond roofline** : aucun kernel réel n'atteint 100 %.
    Le facteur d'efficacité (0,65 par défaut sur x86) est là pour ça.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Iterable, List, Optional, Tuple

GO = 1e9
GIB = float(1 << 30)


# ---------------------------------------------------------------------------
# 1. Spécification d'un modèle
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Description architecturale suffisante pour un budget de mémoire exact.

    Les unités sont des nombres de paramètres ; la conversion en octets se fait
    avec ``bits_*``. Les bits sont *effectifs* (packing + échelles incluses).
    """

    name: str = "hecaton-300a6"
    # --- socle ---
    n_layers: int = 64
    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4          # GQA
    d_head: int = 128
    vocab: int = 131072
    tied_embeddings: bool = False

    # --- attention ---
    attn_kind: str = "gqa"        # "gqa" | "mla"
    frac_attn: float = 0.25       # part des couches avec attention pleine
    mla_dc: int = 512             # latence compressée (MLA)
    mla_drope: int = 64           # queue RoPE

    # --- MoE ---
    n_experts: int = 512
    top_k: int = 10
    d_expert: int = 1024
    d_shared: int = 1024
    moe_every: float = 1.0        # 1.0 = toutes les couches sont MoE
    router_rank: int = 0          # 0 = routeur dense d x E
    d_ffn: int = 5632             # FFN dense pour les couches non-MoE

    # --- couches mémoire (product key) ---
    n_mem_layers: int = 6
    n_slots: int = 2_000_000
    d_mem: int = 384
    top_slots: int = 32

    # --- précision (bits effectifs) ---
    bits_experts: float = 1.7     # ternaire packé
    bits_shared: float = 4.0
    bits_attn: float = 4.0
    bits_embed: float = 6.0
    bits_mem: float = 4.0
    bits_kv: float = 4.0
    bits_state: float = 16.0

    # --- KV / contexte ---
    ctx: int = 32768
    sparse_attn_window: int = 0   # 0 = attention dense sur tout le contexte
    kv_read_frac: float = 1.0     # 1.0 = tout le KV est relu à chaque token

    # --- divers ---
    runtime_bytes: float = 1.5 * GO
    scratch_bytes: float = 0.5 * GO
    notes: str = ""

    # -- construction --
    @classmethod
    def from_json(cls, path: str) -> "ModelSpec":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise SystemExit(f"{path}: clés inconnues {sorted(unknown)}")
        return cls(**raw)

    def replace(self, **kw) -> "ModelSpec":
        d = asdict(self)
        d.update(kw)
        return ModelSpec(**d)

    # -- géométrie --
    @property
    def n_moe_layers(self) -> int:
        return round(self.n_layers * self.moe_every)

    @property
    def n_attn_layers(self) -> int:
        return max(1, round(self.n_layers * self.frac_attn))

    @property
    def n_linear_layers(self) -> int:
        return self.n_layers - self.n_attn_layers

    @property
    def d_qk(self) -> int:
        return self.mla_dc + self.mla_drope


# ---------------------------------------------------------------------------
# 2. Paramètres par groupe
# ---------------------------------------------------------------------------

def params_breakdown(s: ModelSpec) -> Dict[str, float]:
    """Nombre de paramètres par groupe fonctionnel."""
    d = s.d_model
    p: Dict[str, float] = {}

    # MoE : SwiGLU = 3 matrices (gate, up, down)
    p["experts"] = s.n_moe_layers * s.n_experts * 3 * d * s.d_expert
    p["shared_expert"] = s.n_moe_layers * 3 * d * s.d_shared
    if s.router_rank:
        p["router"] = s.n_moe_layers * (d * s.router_rank + s.router_rank * s.n_experts)
    else:
        p["router"] = s.n_moe_layers * d * s.n_experts

    # FFN dense sur les couches non-MoE
    n_dense = s.n_layers - s.n_moe_layers
    p["ffn_dense"] = n_dense * 3 * d * s.d_ffn

    # Attention pleine
    if s.attn_kind == "gqa":
        per_layer = 2 * d * d + 2 * d * s.n_kv_heads * s.d_head
    elif s.attn_kind == "mla":
        per_layer = (
            d * d                                   # q_proj
            + d * s.d_qk                            # kv_down (latent)
            + s.d_qk * s.n_heads * (s.d_head + s.mla_drope)  # kv_up
            + d * d                                 # o_proj
        )
    else:
        raise ValueError(f"attn_kind inconnu : {s.attn_kind}")
    p["attention"] = s.n_attn_layers * per_layer

    # Couches linéaires (Gated DeltaNet : q,k,v,o + decay + gates)
    # Approximation : 4 projections d*d + 2 petits vecteurs.
    p["linear_attn"] = s.n_linear_layers * (5 * d * d)

    # Couches mémoire : valeurs N*d + clés produit sqrt(N)*d
    if s.n_mem_layers:
        slots = s.n_slots
        p["memory_layers"] = s.n_mem_layers * (slots * s.d_mem + math.isqrt(slots) * s.d_mem)
    else:
        p["memory_layers"] = 0.0

    # Embeddings (tied : une seule matrice, mais on la compte une fois)
    p["embeddings"] = s.vocab * d

    # Normes et biais : ~0,1 % du total, on les ignore explicitement
    p["norms_biases"] = 0.001 * sum(p.values())
    return p


BITS_OF = {
    "experts": "bits_experts",
    "shared_expert": "bits_shared",
    "router": None,           # routeur gardé en 8 bits
    "ffn_dense": "bits_attn",
    "attention": "bits_attn",
    "linear_attn": "bits_attn",
    "memory_layers": "bits_mem",
    "embeddings": "bits_embed",
    "norms_biases": None,
}


def params_with_bits(s: ModelSpec) -> Dict[str, Tuple[float, float]]:
    """{groupe: (paramètres, bits effectifs)}"""
    out = {}
    for group, n in params_breakdown(s).items():
        attr = BITS_OF.get(group)
        bits = getattr(s, attr) if attr else 8.0
        out[group] = (n, bits)
    return out


def weights_bytes(s: ModelSpec) -> Dict[str, float]:
    return {g: n * b / 8.0 for g, (n, b) in params_with_bits(s).items()}


# ---------------------------------------------------------------------------
# 3. KV cache et état récurrent
# ---------------------------------------------------------------------------

def kv_bytes_per_token(s: ModelSpec) -> float:
    """Octets de KV par token et par couche d'attention pleine."""
    if s.attn_kind == "gqa":
        n_values = 2 * s.n_kv_heads * s.d_head
    else:
        n_values = s.d_qk
    return n_values * s.bits_kv / 8.0


def kv_cache_bytes(s: ModelSpec, ctx: Optional[int] = None) -> float:
    ctx = ctx if ctx is not None else s.ctx
    return s.n_attn_layers * ctx * kv_bytes_per_token(s)


def kv_read_bytes(s: ModelSpec, ctx: Optional[int] = None) -> float:
    """Ce qu'on relit réellement par token : tout le cache, ou la fenêtre."""
    ctx = ctx if ctx is not None else s.ctx
    if s.sparse_attn_window:
        effective = min(ctx, s.sparse_attn_window)
    else:
        effective = ctx
    return s.n_attn_layers * effective * kv_bytes_per_token(s) * s.kv_read_frac


def recurrent_state_bytes(s: ModelSpec) -> float:
    """État constant des couches linéaires (DeltaNet) — ne grandit pas avec ctx."""
    # état par tête : d_head x d_head, en bits_state
    per_layer = s.n_heads * s.d_head * s.d_head
    return s.n_linear_layers * per_layer * s.bits_state / 8.0


# ---------------------------------------------------------------------------
# 4. Budget mémoire complet
# ---------------------------------------------------------------------------

def memory_budget(s: ModelSpec, ctx: Optional[int] = None) -> Dict[str, float]:
    ctx = ctx if ctx is not None else s.ctx
    w = weights_bytes(s)
    total_weights = sum(w.values())
    kv = kv_cache_bytes(s, ctx)
    state = recurrent_state_bytes(s)
    used = total_weights + kv + state + s.runtime_bytes + s.scratch_bytes
    return {
        "poids_experts": w["experts"],
        "poids_partages": w["shared_expert"],
        "poids_attention": w["attention"] + w["linear_attn"],
        "poids_memoire": w["memory_layers"],
        "poids_embeddings": w["embeddings"],
        "poids_routeur": w["router"],
        "poids_denses": w["ffn_dense"],
        "poids_divers": w["norms_biases"],
        "poids_total": total_weights,
        "kv_cache": kv,
        "etat_recurrent": state,
        "runtime": s.runtime_bytes,
        "scratch": s.scratch_bytes,
        "total": used,
    }


def n_params_total(s: ModelSpec) -> float:
    return sum(params_breakdown(s).values())


def n_params_active(s: ModelSpec) -> float:
    """Paramètres lus pour un token (hors KV, hors embedding lookup)."""
    d = s.d_model
    expert_one = 3 * d * s.d_expert
    return (
        s.n_moe_layers * s.top_k * expert_one
        + s.n_moe_layers * 3 * d * s.d_shared
        + s.n_moe_layers * d * s.n_experts            # routeur
        + (s.n_layers - s.n_moe_layers) * 3 * d * s.d_ffn
        + params_breakdown(s)["attention"]
        + params_breakdown(s)["linear_attn"]
        + s.n_mem_layers * s.top_slots * s.d_mem
        + s.vocab * d                                 # lm_head relu à chaque token
    )


# ---------------------------------------------------------------------------
# 5. Débit et énergie
# ---------------------------------------------------------------------------

@dataclass
class Hardware:
    name: str
    bw_gbs: float          # bande passante crête, Go/s
    gops: float            # calcul utile, GOP/s (int8/ternaire)
    watts: float           # puissance au mur en charge, W
    ram_gb: float          # RAM maximale, Go
    price_eur: float
    efficiency: float = 0.65   # fraction de la bande passante réellement atteinte


HARDWARE: Dict[str, Hardware] = {
    "ddr5-2ch":   Hardware("PC DDR5-5600 2 canaux", 96, 300, 55, 128, 1200, 0.62),
    "strix-halo": Hardware("Ryzen AI Max+ 395 (128 Go)", 256, 800, 90, 128, 3500, 0.65),
    "m4-max":     Hardware("Mac Studio M4 Max (128 Go)", 546, 1200, 65, 128, 4000, 0.72),
    "m3-ultra":   Hardware("Mac Studio M3 Ultra (192 Go)", 819, 1800, 90, 192, 5500, 0.72),
    "dual-epyc":  Hardware("Bi-EPYC 12 canaux (768 Go)", 460, 1200, 400, 768, 15000, 0.60),
    "rtx-5090":   Hardware("RTX 5090 + 128 Go DDR5 (hybride)", 110, 3000, 450, 128, 4000, 0.60),
}


def read_bytes_per_token(s: ModelSpec, ctx: Optional[int] = None) -> Dict[str, float]:
    """Octets lus en DRAM pour produire un token (batch = 1)."""
    ctx = ctx if ctx is not None else s.ctx
    d = s.d_model
    expert_one = 3 * d * s.d_expert
    active_experts = s.n_moe_layers * s.top_k * expert_one * s.bits_experts / 8
    shared = s.n_moe_layers * 3 * d * s.d_shared * s.bits_shared / 8
    attn = (params_breakdown(s)["attention"] * s.bits_attn / 8)
    lin = (params_breakdown(s)["linear_attn"] * s.bits_attn / 8)
    mem = s.n_mem_layers * s.top_slots * s.d_mem * s.bits_mem / 8
    lm_head = s.vocab * d * s.bits_embed / 8
    if s.router_rank:
        router = s.n_moe_layers * (d * s.router_rank + s.router_rank * s.n_experts)
    else:
        router = s.n_moe_layers * d * s.n_experts  # 8 bits
    kv = kv_read_bytes(s, ctx)
    dense_ffn = (s.n_layers - s.n_moe_layers) * 3 * d * s.d_ffn * s.bits_attn / 8
    total = active_experts + shared + attn + lin + mem + lm_head + router + kv + dense_ffn
    return {
        "experts_actifs": active_experts,
        "expert_partage": shared,
        "attention": attn,
        "attention_lineaire": lin,
        "couches_memoire": mem,
        "lm_head": lm_head,
        "routeur": router,
        "ffn_dense": dense_ffn,
        "kv_relu": kv,
        "total": total,
    }


def flops_per_token(s: ModelSpec, ctx: Optional[int] = None) -> float:
    """FLOPs pour un token en décodage (2 x paramètres actifs + attention)."""
    ctx = ctx if ctx is not None else s.ctx
    attn_ctx = min(ctx, s.sparse_attn_window) if s.sparse_attn_window else ctx
    # QK^T + AV : 4 * d * n_tokens par couche d'attention
    attn_flops = s.n_attn_layers * 4 * s.d_model * attn_ctx
    return 2 * n_params_active(s) + attn_flops


@dataclass
class Perf:
    bytes_per_token: float
    flops_per_token: float
    tok_s_memory: float
    tok_s_compute: float
    tok_s: float
    bottleneck: str
    ms_per_token: float
    j_per_token: float
    j_per_token_static: float
    j_per_token_total: float
    tok_s_batch8: float
    j_per_token_batch8: float
    prefill_s_1k: float


def evaluate(s: ModelSpec, hw: Hardware, ctx: Optional[int] = None,
             e_dram_j_per_gb: float = 0.08, e_op_pj: float = 0.10,
             batch: int = 1) -> Perf:
    ctx = ctx if ctx is not None else s.ctx
    rb = read_bytes_per_token(s, ctx)
    fp = flops_per_token(s, ctx)
    bw_eff = hw.bw_gbs * GO * hw.efficiency       # octets/s
    gops_eff = hw.gops * GO                        # op/s

    tok_mem = bw_eff / rb["total"]
    # en batch > 1, les poids sont lus une fois pour `batch` tokens
    tok_comp = gops_eff / fp
    tok = min(tok_mem, tok_comp)
    bottleneck = "memoire" if tok_mem < tok_comp else "calcul"

    j_dram = rb["total"] * e_dram_j_per_gb / GO
    j_comp = fp * e_op_pj * 1e-12
    j_static = hw.watts / tok
    return Perf(
        bytes_per_token=rb["total"],
        flops_per_token=fp,
        tok_s_memory=tok_mem,
        tok_s_compute=tok_comp,
        tok_s=tok,
        bottleneck=bottleneck,
        ms_per_token=1000.0 / tok,
        j_per_token=j_dram + j_comp,
        j_per_token_static=j_static,
        j_per_token_total=j_dram + j_comp + j_static,
        tok_s_batch8=min(tok_mem * 8, tok_comp),
        j_per_token_batch8=(j_dram + j_comp) + hw.watts / min(tok_mem * 8, tok_comp),
        prefill_s_1k=(2 * n_params_active(s) * 1000) / gops_eff,
    )


# ---------------------------------------------------------------------------
# 6. Entraînement
# ---------------------------------------------------------------------------

@dataclass
class TrainCost:
    tokens: float
    flops: float
    gpu_hours: float
    cost_eur: float
    days_on_1_gpu: float


def train_cost(s: ModelSpec, tokens: float, gpu_tflops: float = 990,
               mfu: float = 0.40, eur_per_hour: float = 2.50) -> TrainCost:
    flops = 6.0 * n_params_active(s) * tokens
    eff = gpu_tflops * 1e12 * mfu
    hours = flops / eff / 3600.0
    return TrainCost(tokens, flops, hours, hours * eur_per_hour, hours / 24.0)


def distillation_cost(n_traces: float, teacher_cost_per_mtok: float = 0.5,
                      student: Optional[ModelSpec] = None,
                      epochs: int = 1, gpu_tflops: float = 990, mfu: float = 0.40,
                      eur_per_hour: float = 2.50) -> Dict[str, float]:
    data_eur = (n_traces / 1e6) * teacher_cost_per_mtok
    out = {"traces": n_traces, "cout_generation_eur": data_eur}
    if student is not None:
        tc = train_cost(student, n_traces * epochs, gpu_tflops, mfu, eur_per_hour)
        out.update({"flops_etudiant": tc.flops, "heures_gpu": tc.gpu_hours,
                    "cout_etudiant_eur": tc.cost_eur,
                    "total_eur": data_eur + tc.cost_eur})
    return out


# ---------------------------------------------------------------------------
# 7. Catalogue
# ---------------------------------------------------------------------------

def catalog() -> Dict[str, ModelSpec]:
    return {
        # ---- cible 64 Go : même lecture par token, moins de capacité ----
        "hecaton-215a6": ModelSpec(
            name="HECATON-215A6",
            n_layers=64, d_model=2048, n_heads=16, n_kv_heads=4, d_head=128,
            vocab=131072, attn_kind="mla", frac_attn=0.25,
            n_experts=512, top_k=10, d_expert=1024, d_shared=1024, router_rank=256,
            n_mem_layers=6, n_slots=2_000_000, d_mem=384, top_slots=32,
            bits_experts=1.7, bits_shared=4.0, bits_attn=4.0,
            bits_embed=6.0, bits_mem=4.0, bits_kv=4.0,
            ctx=32768, notes="tient en 64 Go ; 2,07 Go lus/token",
        ),
        # ---- cible 96 Go : +75 % de capacité à lecture constante ----
        "hecaton-420a6": ModelSpec(
            name="HECATON-420A6",
            n_layers=64, d_model=2048, n_heads=16, n_kv_heads=4, d_head=128,
            vocab=131072, attn_kind="mla", frac_attn=0.25,
            n_experts=896, top_k=10, d_expert=1024, d_shared=1024, router_rank=256,
            n_mem_layers=8, n_slots=3_000_000, d_mem=384, top_slots=32,
            bits_experts=1.7, bits_shared=4.0, bits_attn=4.0,
            bits_embed=6.0, bits_mem=4.0, bits_kv=4.0,
            ctx=32768, notes="cible 96 Go : même top-k, 1,75x plus d'experts",
        ),
        # ---- variante économe ----
        "hecaton-95a3": ModelSpec(
            name="HECATON-95A3",
            n_layers=48, d_model=1792, n_heads=14, n_kv_heads=4, d_head=128,
            vocab=65536, attn_kind="mla", frac_attn=0.25,
            n_experts=384, top_k=8, d_expert=896, d_shared=896,
            n_mem_layers=4, n_slots=1_000_000, d_mem=320, top_slots=24,
            bits_experts=1.7, bits_shared=4.0, bits_attn=4.0,
            bits_embed=6.0, bits_mem=4.0, bits_kv=4.0,
            ctx=32768, notes="variante 32 Go, pour station de travail modeste",
        ),
        # ---- pont vers l'autre branche (100 Mo) ----
        "nanofr-75m": ModelSpec(
            name="NanoFR-75M (autre branche)",
            n_layers=20, d_model=512, n_heads=8, n_kv_heads=2, d_head=64,
            vocab=32768, attn_kind="gqa", frac_attn=1.0,
            n_experts=1, top_k=1, d_expert=1408, d_shared=0, d_ffn=1408,
            moe_every=0.0, n_mem_layers=0, n_slots=0, d_mem=0, top_slots=0,
            bits_experts=4.5, bits_shared=4.5, bits_attn=4.5,
            bits_embed=8.0, bits_mem=4.0, bits_kv=8.0,
            ctx=2048, runtime_bytes=26e6, scratch_bytes=2e6,
            notes="référence de la branche 100 Mo — sert de point bas à la loi",
        ),
        # ---- références réelles ----
        "qwen3-next-80b": ModelSpec(
            name="Qwen3-Next-80B-A3B (référence)",
            n_layers=48, d_model=2048, n_heads=16, n_kv_heads=2, d_head=128,
            vocab=151936, attn_kind="gqa", frac_attn=0.25,
            n_experts=512, top_k=10, d_expert=512, d_shared=512,
            n_mem_layers=0, n_slots=0, d_mem=0, top_slots=0,
            bits_experts=4.5, bits_shared=4.5, bits_attn=4.5,
            bits_embed=8.0, bits_mem=4.0, bits_kv=8.0,
            ctx=32768, notes="80 B total / 3 B actifs, publié en BF16",
        ),
        "gpt-oss-120b": ModelSpec(
            name="gpt-oss-120b (référence)",
            n_layers=36, d_model=2880, n_heads=64, n_kv_heads=8, d_head=64,
            vocab=201088, attn_kind="gqa", frac_attn=1.0,
            n_experts=128, top_k=4, d_expert=2880, d_shared=0,
            n_mem_layers=0, n_slots=0, d_mem=0, top_slots=0,
            bits_experts=4.25, bits_shared=4.25, bits_attn=4.25,
            bits_embed=8.0, bits_mem=4.0, bits_kv=8.0,
            ctx=32768, notes="MXFP4 natif, 5,1 B actifs",
        ),
    }


# ---------------------------------------------------------------------------
# 8. Affichage
# ---------------------------------------------------------------------------

def _gb(x: float, nd: int = 2) -> str:
    return f"{x / GO:,.{nd}f} Go".replace(",", " ")


def _fmt_params(n: float) -> str:
    if n >= 1e9:
        return f"{n/1e9:,.1f} B".replace(",", " ")
    if n >= 1e6:
        return f"{n/1e6:,.1f} M".replace(",", " ")
    return f"{n/1e3:,.0f} k".replace(",", " ")


def report(s: ModelSpec, hw: Optional[Hardware] = None, ctx: Optional[int] = None) -> str:
    ctx = ctx if ctx is not None else s.ctx
    mb = memory_budget(s, ctx)
    rb = read_bytes_per_token(s, ctx)
    p = params_breakdown(s)
    lines: List[str] = []
    w = lines.append

    w("=" * 78)
    w(f"  {s.name}")
    w("=" * 78)
    w(f"  {s.n_layers} couches · d={s.d_model} · {s.n_heads} têtes / "
      f"{s.n_kv_heads} KV · {s.attn_kind.upper()} · contexte {ctx}")
    if s.n_moe_layers:
        w(f"  MoE : {s.n_experts} experts, top-{s.top_k}, d_expert={s.d_expert}, "
          f"{s.n_moe_layers} couches")
        kind = "rang faible" if s.router_rank else "dense"
        w(f"  (routeur {kind} : {params_breakdown(s)['router']/1e6:.0f} M paramètres, "
          f"relus à chaque token)")
    w(f"  Précision : experts {s.bits_experts} bit · attention {s.bits_attn} bit · "
      f"embeddings {s.bits_embed} bit · KV {s.bits_kv} bit")
    w("")

    total_p = n_params_total(s)
    active_p = n_params_active(s)
    w("PARAMÈTRES")
    w(f"  total ............. {_fmt_params(total_p)}")
    w(f"  actifs / token .... {_fmt_params(active_p)}  "
      f"({100*active_p/total_p:.2f} % du total)")
    if s.n_moe_layers:
        w(f"  experts lus ....... {s.top_k}/{s.n_experts} par couche")
    w("")

    w(f"MÉMOIRE (ctx {ctx})")
    for k, label in [
        ("poids_experts", "experts routés"),
        ("poids_partages", "expert partagé"),
        ("poids_attention", "attention (pleine + linéaire)"),
        ("poids_memoire", "couches mémoire"),
        ("poids_embeddings", "embeddings"),
        ("poids_routeur", "routeur"),
        ("poids_denses", "FFN dense"),
        ("poids_divers", "normes / biais"),
        ("poids_total", "  → poids totaux"),
        ("kv_cache", "KV cache"),
        ("etat_recurrent", "état récurrent"),
        ("runtime", "runtime"),
        ("scratch", "buffers / scratch"),
    ]:
        w(f"  {label:<28} {_gb(mb[k]):>12}")
    w(f"  {'TOTAL':<28} {_gb(mb['total']):>12}")
    w("")

    w("LECTURE PAR TOKEN (décodage, batch = 1)")
    for k, label in [
        ("experts_actifs", "experts actifs"),
        ("expert_partage", "expert partagé"),
        ("attention", "attention"),
        ("attention_lineaire", "attention linéaire"),
        ("couches_memoire", "couches mémoire"),
        ("lm_head", "tête de sortie"),
        ("routeur", "routeur"),
        ("kv_relu", "KV relu"),
        ("total", "  → TOTAL"),
    ]:
        val = rb[k]
        w(f"  {label:<28} {_gb(val):>12}  {100*val/rb['total']:5.1f} %")
    w("")

    if hw is None:
        return "\n".join(lines)

    perf = evaluate(s, hw, ctx)
    w(f"DÉBIT — {hw.name}")
    w(f"  bande passante utile .... {hw.bw_gbs*hw.efficiency:,.0f} Go/s "
      f"(crête {hw.bw_gbs:.0f}, rendement {hw.efficiency:.0%})")
    w(f"  plafond mémoire ......... {perf.tok_s_memory:,.1f} tok/s")
    w(f"  plafond calcul .......... {perf.tok_s_compute:,.1f} tok/s")
    w(f"  → retenu ................ {perf.tok_s:,.1f} tok/s  "
      f"(goulot : {perf.bottleneck})")
    w(f"  latence ................. {perf.ms_per_token:.1f} ms/token")
    w(f"  prefill 1 000 tokens .... {perf.prefill_s_1k:.1f} s")
    w("")
    w("ÉNERGIE")
    w(f"  DRAM + calcul ........... {perf.j_per_token*1000:.2f} mJ/token")
    w(f"  plateforme (statique) ... {perf.j_per_token_static*1000:.2f} mJ/token")
    w(f"  → total ................. {perf.j_per_token_total:.3f} J/token")
    w(f"  part du modèle .......... {100*perf.j_per_token/perf.j_per_token_total:.2f} %")
    w(f"  réponse (400 tokens) .... {perf.j_per_token_total*400:.1f} J "
      f"= {perf.j_per_token_total*400/3600:.4f} Wh")
    w(f"  batch 8 ................. {perf.tok_s_batch8:,.0f} tok/s, "
      f"{perf.j_per_token_batch8:.3f} J/token")
    w("")
    w("TENUE EN MÉMOIRE")
    margin = hw.ram_gb * GO - mb["total"]
    verdict = "TIENT" if margin >= 0 else "DÉBORDE"
    w(f"  {verdict} — {_gb(mb['total'])} sur {hw.ram_gb:.0f} Go, "
      f"marge {_gb(margin)}")
    return "\n".join(lines)


def table(specs: Dict[str, ModelSpec], hw: Hardware, ctx: int = 32768) -> str:
    rows = []
    header = (f"{'modèle':<26} {'total':>9} {'actifs':>9} {'bits':>5} {'poids':>9} "
              f"{'RAM':>9} {'Go/tok':>8} {'tok/s':>8} {'J/tok':>7} {'tient':>6}")
    rows.append(header)
    rows.append("-" * len(header))
    for s in specs.values():
        s = s.replace(ctx=ctx)
        mb = memory_budget(s, ctx)
        perf = evaluate(s, hw, ctx)
        ok = "oui" if mb["total"] <= hw.ram_gb * GO else "NON"
        rows.append(
            f"{s.name[:26]:<26} {_fmt_params(n_params_total(s)):>9} "
            f"{_fmt_params(n_params_active(s)):>9} {s.bits_experts:>5.1f} "
            f"{_gb(mb['poids_total'],1):>9} "
            f"{_gb(mb['total'],1):>9} {perf.bytes_per_token/GO:>8.2f} "
            f"{perf.tok_s:>8.1f} {perf.j_per_token_total:>7.2f} {ok:>6}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# 9. Loi de budget : le pont 100 Mo <-> 100 Go
# ---------------------------------------------------------------------------

def law_table(hw: Hardware, budgets_gb: Iterable[float] = (0.1, 1, 8, 32, 96, 192),
              bits_list: Iterable[float] = (1.7, 2.5, 4.0, 8.0),
              top_k: int = 10) -> str:
    """Pour chaque budget mémoire, quel modèle tient et à quel débit.

    Principe : on garde la forme HECATON (MoE + MLA + 1 attention sur 4) et on
    fait varier *la seule variable qui achète de la capacité sans coûter de
    bande passante* : le nombre d'experts. La profondeur et d_model sont bornés
    à des valeurs crédibles (>= 32 couches, d_model >= 1024) pour que la ligne
    « la plus grosse » reste un modèle défendable, pas un artefact.
    """
    rows = [f"{'budget':>7} {'bits':>4} {'total':>8} {'actifs':>8} "
            f"{'Go/tok':>7} {'tok/s':>7} {'J/tok':>6} {'Λ':>6} "
            f"{'experts':>8} {'géométrie':>11}"]
    rows.append("-" * len(rows[0]))
    for budget in budgets_gb:
        for bits in bits_list:
            best = None
            for n_layers in (12, 16, 24, 32, 40, 48, 64, 80, 96):
                for d_model in (256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072):
                    d_expert = max(256, d_model // 2)
                    for n_experts in (128, 256, 512, 768, 1024, 1536, 2048):
                        s = ModelSpec(
                            name="law", n_layers=n_layers, d_model=d_model,
                            n_heads=max(8, d_model // 128),
                            n_kv_heads=max(2, d_model // 512), d_head=128,
                            vocab=min(131072, max(32768, d_model * 48)),
                            attn_kind="mla", frac_attn=0.25,
                            n_experts=n_experts, top_k=min(top_k, n_experts),
                            d_expert=d_expert, d_shared=d_expert,
                            router_rank=min(256, d_model),
                            n_mem_layers=0, n_slots=0, d_mem=0, top_slots=0,
                            bits_experts=bits, bits_shared=min(bits * 2, 8),
                            bits_attn=4.0, bits_embed=min(bits * 3, 8),
                            bits_kv=4.0, ctx=8192,
                            runtime_bytes=min(1.5 * GO, 0.25 * budget * GO),
                            scratch_bytes=min(0.5 * GO, 0.10 * budget * GO),
                        )
                        mb = memory_budget(s, 8192)
                        if mb["total"] > budget * GO:
                            continue
                        score = n_params_total(s)
                        if best is None or score > best[0]:
                            best = (score, s)
            if best is None:
                continue
            _, s = best
            perf = evaluate(s, hw, 8192)
            lam = (budget * GO) / perf.bytes_per_token
            rows.append(
                f"{budget:>6.2f}G {bits:>4.1f} {_fmt_params(n_params_total(s)):>8} "
                f"{_fmt_params(n_params_active(s)):>8} "
                f"{perf.bytes_per_token/GO:>7.2f} {perf.tok_s:>7.1f} "
                f"{perf.j_per_token_total:>6.2f} {lam:>6.1f} "
                f"{s.n_experts:>8} {f'{s.n_layers}x{s.d_model}':>11}"
            )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# 10. SVG
# ---------------------------------------------------------------------------

def svg_memory(s: ModelSpec, path: str, ctx: Optional[int] = None) -> None:
    ctx = ctx if ctx is not None else s.ctx
    mb = memory_budget(s, ctx)
    parts = [
        ("experts routés", mb["poids_experts"], "#1f4e79"),
        ("expert partagé", mb["poids_partages"], "#2e75b6"),
        ("attention", mb["poids_attention"], "#9dc3e6"),
        ("couches mémoire", mb["poids_memoire"], "#c55a11"),
        ("embeddings", mb["poids_embeddings"], "#f4b183"),
        ("routeur", mb["poids_routeur"], "#a6a6a6"),
        ("KV cache", mb["kv_cache"], "#548235"),
        ("runtime + buffers", mb["runtime"] + mb["scratch"], "#7f7f7f"),
    ]
    total = mb["total"]
    W, H = 980, 310
    bx, by, bw, bh = 90, 90, 820, 74
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="24" y="36" font-size="19" font-weight="700" fill="#111">'
           f'{s.name} — budget mémoire à {ctx:,} tokens de contexte</text>'.replace(",", " "),
           f'<text x="24" y="58" font-size="13" fill="#555">'
           f'{_fmt_params(n_params_total(s))} paramètres · '
           f'{_gb(total)} au total · {s.bits_experts} bit sur les experts</text>']
    x = bx
    for label, value, color in parts:
        if value <= 0:
            continue
        w = max(1.0, bw * value / total)
        out.append(f'<rect x="{x:.1f}" y="{by}" width="{w:.1f}" height="{bh}" '
                   f'fill="{color}"><title>{label} : {_gb(value)}</title></rect>')
        x += w
    out.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="none" '
               f'stroke="#333" stroke-width="1"/>')
    # légende en deux colonnes
    y = by + bh + 30
    for i, (label, value, color) in enumerate(parts):
        col, row = i % 2, i // 2
        lx = bx + col * 420
        ly = y + row * 22
        out.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{color}"/>')
        out.append(f'<text x="{lx+18}" y="{ly}" font-size="12.5" fill="#222">'
                   f'{label} — {_gb(value)} ({100*value/total:.1f} %)</text>')
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))


def svg_roofline(specs: Dict[str, ModelSpec], hw: Hardware, path: str,
                 ctx: int = 32768) -> None:
    """Débit en fonction de la bande passante, une droite par modèle."""
    W, H = 900, 520
    ml, mr, mt, mb_ = 80, 210, 60, 70
    pw, ph = W - ml - mr, H - mt - mb_
    bw_max = 900.0
    v_max = max(evaluate(s.replace(ctx=ctx), hw, ctx).tok_s for s in specs.values())
    v_max = max(10.0, v_max * 1.25)

    def sx(bw: float) -> float:
        return ml + pw * bw / bw_max

    def sy(v: float) -> float:
        return mt + ph * (1 - v / v_max)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>',
           f'<text x="24" y="32" font-size="18" font-weight="700">'
           f'Débit roofline — {ctx:,} tokens de contexte</text>'.replace(",", " "),
           f'<text x="24" y="50" font-size="12.5" fill="#555">'
           f'plafond théorique = bande passante utile / octets lus par token</text>']
    # grille
    for v in range(0, int(v_max) + 1, max(1, int(v_max // 6))):
        y = sy(v)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                   f'stroke="#e6e6e6"/>')
        out.append(f'<text x="{ml-10}" y="{y+4:.1f}" font-size="11" fill="#666" '
                   f'text-anchor="end">{v}</text>')
    for bw in range(0, int(bw_max) + 1, 150):
        x = sx(bw)
        out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" '
                   f'stroke="#f2f2f2"/>')
        out.append(f'<text x="{x:.1f}" y="{mt+ph+22}" font-size="11" fill="#666" '
                   f'text-anchor="middle">{bw}</text>')
    out.append(f'<text x="{ml+pw/2}" y="{H-22}" font-size="12.5" fill="#333" '
               f'text-anchor="middle">bande passante utile (Go/s)</text>')
    out.append(f'<text x="20" y="{mt+ph/2}" font-size="12.5" fill="#333" '
               f'transform="rotate(-90 20 {mt+ph/2})" text-anchor="middle">'
               f'tokens/s (décodage batch 1)</text>')
    out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>')
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>')

    colors = ["#1f4e79", "#c55a11", "#548235", "#7030a0", "#bf9000", "#2e75b6"]
    for i, (key, s) in enumerate(specs.items()):
        s = s.replace(ctx=ctx)
        rb = read_bytes_per_token(s, ctx)["total"]
        color = colors[i % len(colors)]
        x1, y1 = sx(0), sy(0)
        x2, y2 = sx(bw_max), sy(bw_max * GO / rb)
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                   f'stroke="{color}" stroke-width="2.2"/>')
        # marqueur de la plateforme actuelle
        px, py = sx(hw.bw_gbs * hw.efficiency), sy(min(hw.bw_gbs*hw.efficiency*GO/rb, v_max))
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>')
        ly = mt + 20 + i * 22
        out.append(f'<rect x="{ml+pw+18}" y="{ly-10}" width="12" height="12" fill="{color}"/>')
        out.append(f'<text x="{ml+pw+36}" y="{ly}" font-size="11.5" fill="#222">'
                   f'{s.name[:26]}</text>')
    out.append(f'<text x="{ml+pw+18}" y="{mt+20+len(specs)*22+18}" font-size="10.5" '
               f'fill="#777">point = {hw.name}</text>')
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------

def _load(spec_or_path: str) -> ModelSpec:
    if os.path.exists(spec_or_path):
        return ModelSpec.from_json(spec_or_path)
    cat = catalog()
    if spec_or_path in cat:
        return cat[spec_or_path]
    raise SystemExit(f"configuration inconnue : {spec_or_path}\n"
                     f"disponibles : {', '.join(cat)}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="détail d'une configuration")
    p_show.add_argument("config")
    p_show.add_argument("--hw", default="strix-halo", choices=list(HARDWARE))
    p_show.add_argument("--ctx", type=int, default=None)
    p_show.add_argument("--json", action="store_true")

    p_tab = sub.add_parser("table", help="tableau comparatif")
    p_tab.add_argument("--hw", default="strix-halo", choices=list(HARDWARE))
    p_tab.add_argument("--ctx", type=int, default=32768)
    p_tab.add_argument("--configs", nargs="*", default=None)

    p_sw = sub.add_parser("sweep", help="ce qui tient dans N Go")
    p_sw.add_argument("--hw", default="strix-halo", choices=list(HARDWARE))
    p_sw.add_argument("--budgets", nargs="*", type=float, default=[64, 96, 128, 192, 256])
    p_sw.add_argument("--bits", nargs="*", type=float, default=[1.7, 2.5, 4.0])
    p_sw.add_argument("--ctx", type=int, default=32768)

    p_law = sub.add_parser("law", help="loi de budget 100 Mo -> 100 Go")
    p_law.add_argument("--hw", default="strix-halo", choices=list(HARDWARE))
    p_law.add_argument("--budgets", nargs="*", type=float,
                       default=[0.1, 0.5, 1, 8, 32, 96, 192])
    p_law.add_argument("--bits", nargs="*", type=float, default=[1.7, 2.5, 4.0, 8.0])

    p_hw = sub.add_parser("hw", help="le même modèle sur plusieurs plateformes")
    p_hw.add_argument("config")
    p_hw.add_argument("--ctx", type=int, default=None)

    p_tr = sub.add_parser("train", help="coût d'entraînement / distillation")
    p_tr.add_argument("config")
    p_tr.add_argument("--tokens", type=float, default=1e12)
    p_tr.add_argument("--traces", type=float, default=0.0,
                      help="tokens de traces de distillation")
    p_tr.add_argument("--teacher-cost", type=float, default=0.5,
                      help="€ par million de tokens générés par l'enseignant")
    p_tr.add_argument("--gpu-tflops", type=float, default=990)
    p_tr.add_argument("--mfu", type=float, default=0.40)
    p_tr.add_argument("--eur-per-hour", type=float, default=2.50)

    p_sens = sub.add_parser("sensitivity",
                            help="débit selon le noyau et la bande passante")
    p_sens.add_argument("config")
    p_sens.add_argument("--gops", nargs="*", type=float,
                        default=[100, 200, 400, 800, 1600, 3200])
    p_sens.add_argument("--bw", nargs="*", type=float,
                        default=[60, 90, 166, 256, 400, 550])

    p_svg = sub.add_parser("svg", help="figures SVG")
    p_svg.add_argument("--out", default="figures")
    p_svg.add_argument("--ctx", type=int, default=32768)
    p_svg.add_argument("--hw", default="strix-halo", choices=list(HARDWARE))

    args = ap.parse_args(argv)

    if args.cmd == "show":
        s = _load(args.config)
        if args.ctx is not None:
            s = s.replace(ctx=args.ctx)
        hw = HARDWARE[args.hw]
        if args.json:
            mb = memory_budget(s, s.ctx)
            pf = evaluate(s, hw, s.ctx)
            print(json.dumps({"spec": asdict(s), "memory": mb,
                              "read": read_bytes_per_token(s, s.ctx),
                              "perf": asdict(pf)}, indent=2))
        else:
            print(report(s, hw))
        return 0

    if args.cmd == "table":
        hw = HARDWARE[args.hw]
        cat = catalog()
        if args.configs:
            cat = {k: v for k, v in cat.items() if k in args.configs}
        print(f"Plateforme : {hw.name} · {hw.bw_gbs} Go/s · {hw.ram_gb} Go · "
              f"{hw.price_eur:.0f} €\n")
        print(table(cat, hw, args.ctx))
        return 0

    if args.cmd == "sweep":
        hw = HARDWARE[args.hw]
        print(f"Plateforme : {hw.name}\n")
        print("Ce qui tient dans un budget donné (MoE 512 experts, top-10, MLA)\n")
        print(law_table(hw, args.budgets, args.bits[:1], top_k=10))
        print()
        print("Effet des bits sur la capacité à 96 Go :\n")
        print(law_table(hw, [96], args.bits, top_k=10))
        return 0

    if args.cmd == "law":
        hw = HARDWARE[args.hw]
        print("=" * 78)
        print("  Loi de budget — de 100 Mo à 192 Go")
        print("=" * 78)
        print(f"  Plateforme de référence : {hw.name}")
        print("  Hypothèses : MoE 512 experts, top-10, MLA, attention sur 1/4")
        print("  des couches, KV 4 bits, runtime borné par le budget.")
        print()
        print(law_table(hw, args.budgets, args.bits))
        print()
        print("Lecture : 'Λ' = budget / octets lus par token = multiplicateur")
        print("de débit. C'est le vrai paramètre de conception, pas le nombre")
        print("de paramètres.")
        print()
        print("ATTENTION : sous ~1 Go, les coûts fixes (tokenizer, sampling,")
        print("latence de lancement) et la qualité plafonnent bien avant la")
        print("roofline. Les lignes 0,5-1 Go sont une borne haute théorique,")
        print("pas un débit atteignable. Voir la branche 100 Mo pour ce régime.")
        return 0

    if args.cmd == "hw":
        s = _load(args.config)
        if args.ctx is not None:
            s = s.replace(ctx=args.ctx)
        print(f"{s.name} — contexte {s.ctx}\n")
        head = (f"{'plateforme':<30} {'RAM':>6} {'BW':>6} {'tok/s':>8} "
                f"{'goulot':>8} {'J/tok':>7} {'tient':>7}")
        print(head)
        print("-" * len(head))
        for hw in HARDWARE.values():
            mb = memory_budget(s, s.ctx)
            pf = evaluate(s, hw, s.ctx)
            ok = "oui" if mb["total"] <= hw.ram_gb * GO else "NON"
            print(f"{hw.name[:30]:<30} {hw.ram_gb:>5.0f}G {hw.bw_gbs:>5.0f}G "
                  f"{pf.tok_s:>8.1f} {pf.bottleneck:>8} "
                  f"{pf.j_per_token_total:>7.3f} {ok:>7}")
        return 0

    if args.cmd == "sensitivity":
        s = _load(args.config)
        gops_list = args.gops
        bw_list = args.bw
        print(f"{s.name} — débit (tok/s) selon noyau et bande passante")
        print(f"contexte {s.ctx} · {s.bits_experts} bit · "
              f"{_fmt_params(n_params_active(s))} actifs · "
              f"{read_bytes_per_token(s, s.ctx)['total']/GO:.2f} Go/token\n")
        print(" " * 18 + "".join(f"{bw:>8.0f}" for bw in bw_list) + "   Go/s utiles")
        for gops in gops_list:
            row = f"noyau {gops:>5.0f} GOP/s  "
            for bw in bw_list:
                hw = Hardware("x", bw, gops, 60, 128, 0, 1.0)
                row += f"{evaluate(s, hw, s.ctx).tok_s:>8.1f}"
            print(row)
        print()
        print("Seuil mémoire-borné : GOPS >= 16 * BW / bits.")
        print(f"Ici : {16*bw_list[-1]/s.bits_experts:.0f} GOP/s pour être "
              f"limité par la mémoire à {bw_list[-1]:.0f} Go/s.")
        return 0

    if args.cmd == "train":
        s = _load(args.config)
        tc = train_cost(s, args.tokens, args.gpu_tflops, args.mfu, args.eur_per_hour)
        print(f"{s.name} — {_fmt_params(n_params_active(s))} actifs\n")
        print(f"Pré-entraînement sur {args.tokens:.3g} tokens")
        print(f"  FLOPs ............ {tc.flops:.3e}")
        print(f"  heures GPU ....... {tc.gpu_hours:,.0f}".replace(",", " "))
        print(f"  jours sur 1 GPU .. {tc.days_on_1_gpu:,.1f}")
        print(f"  coût ............. {tc.cost_eur:,.0f} €".replace(",", " "))
        if args.traces:
            d = distillation_cost(args.traces, args.teacher_cost, s,
                                  gpu_tflops=args.gpu_tflops, mfu=args.mfu,
                                  eur_per_hour=args.eur_per_hour)
            print(f"\nDistillation sur {args.traces:.3g} tokens de traces")
            print(f"  génération ....... {d['cout_generation_eur']:,.0f} €".replace(",", " "))
            print(f"  entraînement ..... {d['cout_etudiant_eur']:,.0f} €".replace(",", " "))
            print(f"  TOTAL ............ {d['total_eur']:,.0f} €".replace(",", " "))
        return 0

    if args.cmd == "svg":
        os.makedirs(args.out, exist_ok=True)
        cat = catalog()
        main_spec = cat["hecaton-420a6"]
        p1 = os.path.join(args.out, "hecaton-memoire.svg")
        p2 = os.path.join(args.out, "hecaton-roofline.svg")
        svg_memory(main_spec, p1, args.ctx)
        svg_roofline({"hecaton": main_spec, "qwen3-next": cat["qwen3-next-80b"],
                      "gpt-oss": cat["gpt-oss-120b"], "nanofr": cat["nanofr-75m"]},
                     HARDWARE[args.hw], p2, args.ctx)
        print(f"écrit : {p1}")
        print(f"écrit : {p2}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
