#!/usr/bin/env python3
"""
llm_budget.py — Calculateur de budget pour LLM "ultra-léger".

Répond à trois questions, avec des formules explicites plutôt que des ordres de grandeur :
  1. Mémoire d'inférence (poids + KV cache + runtime) → tient-on dans 100 Mo de RSS ?
  2. Énergie par token (modèle mémoire-bandwidth) → combien de watts à N tok/s ?
  3. Mémoire et temps d'entraînement → tient-on dans 1 Go, et en combien de jours ?

Toutes les hypothèses sont paramétrables en ligne de commande. Aucune dépendance.

Usage :
    python tools/llm_budget.py presets
    python tools/llm_budget.py show nano60
    python tools/llm_budget.py table
    python tools/llm_budget.py train nano60 --tokens 2e9 --gpu-tflops 100
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, fields

# --------------------------------------------------------------------------------------
# Constantes physiques et empiriques
# --------------------------------------------------------------------------------------

PJ_PER_BYTE_DRAM = 80.0      # Horowitz 2014 : 640 pJ pour 32 bits lus en DRAM → 80 pJ/octet
PJ_PER_MAC_FP16 = 1.5        # ~1.1 pJ (mul) + 0.4 pJ (add), 45 nm
PJ_PER_MAC_INT8 = 0.2        # ordre de grandeur : l'addition int8 domine le ternaire
BYTES_PER_PARAM_TRAIN = {
    # octets par paramètre, pour un entraînement Adam "classique" (poids + grad + m + v)
    "adam-fp32": 16.0,       # 4 + 4 + 4 + 4
    "adam-8bit": 10.0,       # 4 (master) + 4 (grad) + 2 (m,v 8-bit)
    "adam-bf16": 6.0,        # 2 + 2 + 2  (instable en pratique, mais borne basse)
}


@dataclass
class Arch:
    """Description d'une architecture decoder-only de type Llama."""

    name: str
    layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    ffn_dim: int
    vocab: int
    tied_embeddings: bool = True
    ctx: int = 2048
    weight_bpw: float = 4.5      # bits par poids effectifs (4.5 ≈ Q4_K_M, 1.58 = ternaire)
    kv_bytes: float = 1.0        # octets par élément de KV cache (1.0 = q8_0, 2.0 = f16)
    runtime_mb: float = 20.0     # RSS du moteur hors poids et hors KV (llama.cpp & co)
    activation_mb: float = 6.0   # buffers de calcul / scratch
    head_dim_explicit: float = 0.0   # 0 = d_model / n_heads
    local_layers: int = 0            # couches à fenêtre glissante (Gemma, Mistral...)
    local_window: int = 0            # taille de la fenêtre
    file_mb: float = 0.0             # taille GGUF MESURÉE sur disque (0 = estimer par bpw)

    # ---- paramètres -------------------------------------------------------------
    @property
    def head_dim(self) -> float:
        return self.head_dim_explicit or (self.d_model / self.n_heads)

    @property
    def global_layers(self) -> int:
        return self.layers - self.local_layers

    @property
    def embed_params(self) -> float:
        p = self.vocab * self.d_model
        return p if self.tied_embeddings else 2 * p

    @property
    def attn_params_per_layer(self) -> float:
        d, hd = self.d_model, self.head_dim
        q = d * self.n_heads * hd          # = d*d
        k = d * self.n_kv_heads * hd
        v = d * self.n_kv_heads * hd
        o = self.n_heads * hd * d          # = d*d
        return q + k + v + o

    @property
    def mlp_params_per_layer(self) -> float:
        # SwiGLU : gate (d→f), up (d→f), down (f→d)
        return 3.0 * self.d_model * self.ffn_dim

    @property
    def norm_params_per_layer(self) -> float:
        return 2.0 * self.d_model

    @property
    def params(self) -> float:
        per_layer = (
            self.attn_params_per_layer
            + self.mlp_params_per_layer
            + self.norm_params_per_layer
        )
        return self.embed_params + self.layers * per_layer

    @property
    def non_embed_params(self) -> float:
        return self.params - self.embed_params

    # ---- mémoire ----------------------------------------------------------------
    @property
    def weight_bytes(self) -> float:
        if self.file_mb:
            return self.file_mb * 1e6      # mesuré : inclut les embeddings haute précision
        return self.params * self.weight_bpw / 8.0

    @property
    def measured(self) -> bool:
        return self.file_mb > 0

    @property
    def kv_bytes_total(self) -> float:
        """KV total. Les couches locales ne stockent que min(ctx, fenêtre)."""
        per_layer_token = 2.0 * self.n_kv_heads * self.head_dim * self.kv_bytes
        window = min(self.ctx, self.local_window) if self.local_layers else 0
        units = self.global_layers * self.ctx + self.local_layers * window
        return per_layer_token * units

    @property
    def kv_bytes_per_token(self) -> float:
        return self.kv_bytes_total / self.ctx

    @property
    def rss_bytes(self) -> float:
        return (
            self.weight_bytes
            + self.kv_bytes_total
            + (self.runtime_mb + self.activation_mb) * 1e6
        )

    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / 1e6

    # ---- énergie ----------------------------------------------------------------
    @property
    def energy_per_token_j(self) -> float:
        """Borne inférieure : lire tous les poids une fois + faire les MAC.

        En décodage batch=1, le coût dominant est la lecture des poids depuis la DRAM.
        C'est le modèle qui explique les mesures publiées (~0,17 J/token pour ~0,6 Md
        de paramètres en bf16 sur CPU Intel).
        """
        e_mem = self.weight_bytes * PJ_PER_BYTE_DRAM * 1e-12
        e_mac_pj = PJ_PER_MAC_INT8 if self.weight_bpw < 2.0 else PJ_PER_MAC_FP16
        e_compute = 2.0 * self.non_embed_params * e_mac_pj * 1e-12
        return e_mem + e_compute

    def watts_at(self, tok_s: float) -> float:
        """Puissance incrémentale attribuable au modèle (DRAM + MAC)."""
        return self.energy_per_token_j * tok_s

    def wall_energy_per_token(self, device_watts: float, tok_s: float) -> float:
        """Énergie réellement payée au mur : la plateforme tourne de toute façon.

        À petite échelle, E/token ≈ P_plateforme / débit. C'est le « race-to-idle » :
        ce n'est pas le modèle qui coûte, c'est le temps qu'il met à répondre.
        """
        return device_watts / tok_s

    # ---- entraînement -----------------------------------------------------------
    def train_flops(self, tokens: float) -> float:
        return 6.0 * self.params * tokens

    def train_memory_bytes(self, scheme: str = "adam-8bit", activation_factor: float = 1.0) -> float:
        base = self.params * BYTES_PER_PARAM_TRAIN[scheme]
        # activations ~ quelques dizaines d'octets/paramètre selon checkpointing
        return base * activation_factor

    def train_days(self, tokens: float, tflops: float, mfu: float = 0.30) -> float:
        flops = self.train_flops(tokens)
        seconds = flops / (tflops * 1e12 * mfu)
        return seconds / 86400.0

    def train_cost_usd(self, tokens: float, tflops: float, mfu: float, usd_per_hour: float) -> float:
        flops = self.train_flops(tokens)
        gpu_hours = flops / (tflops * 1e12 * mfu) / 3600.0
        return gpu_hours * usd_per_hour


# --------------------------------------------------------------------------------------
# Presets : configurations réelles ou proposées
# --------------------------------------------------------------------------------------

PRESETS: dict[str, Arch] = {
    # Référence : SmolLM2-135M (30 L, 576 d, 9 têtes, 3 têtes KV, FFN 1536, vocab 49152)
    "smollm2-135m": Arch("SmolLM2-135M", 30, 576, 9, 3, 1536, 49152, True, 2048, 4.5, 1.0),
    "smollm2-135m-q2": Arch("SmolLM2-135M Q2", 30, 576, 9, 3, 1536, 49152, True, 2048, 2.6, 1.0),
    # Référence : MobileLLM-R1-140M (15 L, 576 d, 9 têtes, 3 KV, FFN 2048, vocab 128k)
    "mobilellm-r1-140m": Arch("MobileLLM-R1-140M", 15, 576, 9, 3, 2048, 128_000, True, 2048, 4.5, 1.0),
    # Candidats "100 Mo" (dimensionnés pour tenir sous 100 Mo de RSS)
    "nano60": Arch("NanoFR-60M", 20, 448, 7, 2, 1280, 32_768, True, 2048, 4.5, 1.0),
    "nano75": Arch("NanoFR-75M", 20, 512, 8, 2, 1408, 32_768, True, 2048, 4.5, 1.0),
    "nano85": Arch("NanoFR-85M", 24, 512, 8, 2, 1408, 32_768, True, 2048, 4.5, 1.0),
    "nano125": Arch("NanoFR-125M", 30, 576, 9, 3, 1536, 32_768, True, 2048, 4.5, 1.0),
    # --- Régime 200 Mo : généraliste + déploiement web ---
    # file_mb = taille GGUF mesurée (unsloth / DevQuasar). 0 = estimation par bpw.
    "smollm2-135m-q4": Arch("SmolLM2-135M Q4_K_M", 30, 576, 9, 3, 1536, 49_152, True,
                            ctx=2048, weight_bpw=4.5, kv_bytes=1.0, file_mb=105.0),
    "smollm2-135m-q4-4k": Arch("SmolLM2-135M Q4_K_M (4k)", 30, 576, 9, 3, 1536, 49_152, True,
                               ctx=4096, weight_bpw=4.5, kv_bytes=1.0, file_mb=105.0),
    "smollm2-135m-q5": Arch("SmolLM2-135M Q5_K_M", 30, 576, 9, 3, 1536, 49_152, True,
                            ctx=2048, weight_bpw=5.5, kv_bytes=1.0, file_mb=112.0),
    "smollm2-135m-q8": Arch("SmolLM2-135M Q8_0", 30, 576, 9, 3, 1536, 49_152, True,
                            ctx=2048, weight_bpw=8.5, kv_bytes=1.0, file_mb=145.0),
    "smollm2-135m-q2": Arch("SmolLM2-135M Q2_K", 30, 576, 9, 3, 1536, 49_152, True,
                            ctx=2048, weight_bpw=2.6, kv_bytes=1.0, file_mb=88.2),
    "mobilellm-r1-140m": Arch("MobileLLM-R1-140M Q4_K_M", 15, 576, 9, 3, 1536, 128_256, True,
                              ctx=2048, weight_bpw=4.5, kv_bytes=1.0, file_mb=132.0),
    "mobilellm-r1-140m-4k": Arch("MobileLLM-R1-140M Q4_K_M (4k)", 15, 576, 9, 3, 1536, 128_256, True,
                                 ctx=4096, weight_bpw=4.5, kv_bytes=1.0, file_mb=132.0),
    "gemma3-270m-q4km-2k": Arch("Gemma3-270M Q4_K_M (2k)", 18, 640, 4, 1, 2048, 262_144, True,
                                ctx=2048, weight_bpw=4.5, kv_bytes=1.0,
                                head_dim_explicit=256, local_layers=15, local_window=512,
                                file_mb=253.0),
    "gemma3-270m-q4km-8k": Arch("Gemma3-270M Q4_K_M (8k)", 18, 640, 4, 1, 2048, 262_144, True,
                                ctx=8192, weight_bpw=4.5, kv_bytes=1.0,
                                head_dim_explicit=256, local_layers=15, local_window=512,
                                file_mb=253.0),
    "gemma3-270m-vocab32k-8k": Arch("Gemma3-270M vocab 32k [calc.]", 18, 640, 4, 1, 2048, 32_768, True,
                                    ctx=8192, weight_bpw=4.5, kv_bytes=1.0,
                                    head_dim_explicit=256, local_layers=15, local_window=512,
                                    file_mb=106.2),
    "smollm2-360m-q2": Arch("SmolLM2-360M Q2_K", 32, 960, 15, 5, 2560, 49_152, True,
                            ctx=2048, weight_bpw=2.6, kv_bytes=1.0, file_mb=219.0),
    "smollm2-360m-q4": Arch("SmolLM2-360M Q4_K_M", 32, 960, 15, 5, 2560, 49_152, True,
                            ctx=2048, weight_bpw=4.5, kv_bytes=1.0, file_mb=271.0),
    # Piste ternaire (BitNet b1.58)
    "ternary300": Arch("Ternaire-270M", 28, 896, 14, 4, 2432, 32_768, True, 2048, 1.58, 1.0),
    "ternary400": Arch("Ternaire-400M", 32, 1024, 16, 4, 2816, 32_768, True, 2048, 1.58, 1.0),
    "ternary400-8k": Arch("Ternaire-400M (8k)", 32, 1024, 16, 4, 2816, 32_768, True, 8192, 1.58, 1.0),
}


# --------------------------------------------------------------------------------------
# Rendu
# --------------------------------------------------------------------------------------

def fmt_bytes(n: float) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f} Go"
    if n >= 1e6:
        return f"{n/1e6:.1f} Mo"
    if n >= 1e3:
        return f"{n/1e3:.1f} Ko"
    return f"{n:.0f} o"


def show(a: Arch) -> None:
    w, kv, rt = a.weight_bytes, a.kv_bytes_total, (a.runtime_mb + a.activation_mb) * 1e6
    print(f"\n=== {a.name} ===")
    print(f"  Params            : {a.params/1e6:8.1f} M  (dont embeddings {a.embed_params/1e6:6.1f} M)")
    print(f"  Non-embed         : {a.non_embed_params/1e6:8.1f} M")
    print(f"  Bits/poids        : {a.weight_bpw:.2f}   KV: {a.kv_bytes:.1f} o/élément   ctx: {a.ctx}")
    print(f"  --- Mémoire d'inférence ---")
    print(f"  Poids             : {fmt_bytes(w):>9}  ({100*w/a.rss_bytes:4.1f} %)")
    print(f"  KV cache          : {fmt_bytes(kv):>9}  ({100*kv/a.rss_bytes:4.1f} %)  "
          f"[{fmt_bytes(a.kv_bytes_per_token)}/token]")
    print(f"  Runtime + scratch : {fmt_bytes(rt):>9}  ({100*rt/a.rss_bytes:4.1f} %)")
    print(f"  TOTAL RSS estimé  : {fmt_bytes(a.rss_bytes):>9}   "
          f"{'✅ < 100 Mo' if a.rss_mb <= 100 else ('✅ < 200 Mo' if a.rss_mb <= 200 else '❌ > 200 Mo')}")
    print(f"  --- Énergie (borne basse mémoire) ---")
    print(f"  E/token           : {a.energy_per_token_j*1e3:8.2f} mJ")
    for tps in (5, 10, 20, 50):
        print(f"    à {tps:2d} tok/s   : {a.watts_at(tps)*1e3:7.1f} mW")
    print()


def table(names: list[str]) -> None:
    head = (
        f"{'Modèle':<33}{'Params':>8}{'Poids':>10}{'KV':>9}{'RSS':>8}"
        f"{'E/tok':>10}{'mW@10':>8}{'≤200Mo':>9}"
    )
    print(head)
    print("-" * len(head))
    for n in names:
        a = PRESETS[n]
        mark = "*" if a.measured else " "
        print(
            f"{a.name:<32}{mark}{a.params/1e6:>7.0f}M"
            f"{fmt_bytes(a.weight_bytes):>9}{fmt_bytes(a.kv_bytes_total):>9}"
            f"{a.rss_mb:>7.0f}Mo{a.energy_per_token_j*1e3:>9.2f}mJ"
            f"{a.watts_at(10)*1e3:>7.0f}  {'OUI' if a.rss_mb <= 200 else 'non':>7}"
        )
    print("\nNote : * = taille GGUF mesurée ; sinon estimation bits/poids (optimiste).")
    print("       E/token = borne basse DRAM+MAC ; mW@10 = puissance incrémentale à 10 tok/s.")
    print("       L'énergie murale vaut ~P_plateforme / débit (voir 'wall').")


def train_report(a: Arch, tokens: float, tflops: float, mfu: float, usd_h: float) -> None:
    print(f"\n=== Entraînement : {a.name} sur {tokens/1e9:.2f} Md tokens ===")
    print(f"  FLOPs             : {a.train_flops(tokens):.2e}")
    for scheme in ("adam-fp32", "adam-8bit", "adam-bf16"):
        mem = a.train_memory_bytes(scheme)
        print(f"  Mémoire {scheme:<10}: {fmt_bytes(mem):>10} "
              f"{'✅ < 1 Go' if mem <= 1e9 else '❌ > 1 Go'}")
    days = a.train_days(tokens, tflops, mfu)
    print(f"  Temps @ {tflops:.0f} TFLOPS, MFU {mfu:.0%} : {days:.2f} jours "
          f"({days*24:.1f} h)")
    print(f"  Coût  @ {usd_h:.2f} $/h          : {a.train_cost_usd(tokens, tflops, mfu, usd_h):.2f} $")
    # Chinchilla
    chin = 20 * a.params
    print(f"  Chinchilla (20 tok/param)      : {chin/1e9:.2f} Md tokens")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("presets", help="liste les presets")
    s = sub.add_parser("show", help="détail d'un preset")
    s.add_argument("name")
    t = sub.add_parser("table", help="tableau comparatif")
    t.add_argument("names", nargs="*", default=None)
    tr = sub.add_parser("train", help="budget d'entraînement")
    tr.add_argument("name")
    tr.add_argument("--tokens", type=float, default=2e9)
    tr.add_argument("--gpu-tflops", type=float, default=100.0, help="TFLOPS effectifs crête")
    tr.add_argument("--mfu", type=float, default=0.30)
    tr.add_argument("--usd-per-hour", type=float, default=0.60)
    sv = sub.add_parser("serve", help="dimensionner un serveur web (slots, RSS, énergie)")
    sv.add_argument("name")
    sv.add_argument("--parallel", type=int, default=4)
    sv.add_argument("--ctx-per-slot", type=int, default=2048)
    sv.add_argument("--tok-s-single", type=float, default=15.0)
    sv.add_argument("--batching-gain", type=float, default=2.5,
                    help="gain agrégé du continuous batching (décodage memory-bound)")
    sv.add_argument("--device-watts", type=float, default=15.0)
    sv.add_argument("--requests-per-day", type=int, default=1000)
    sv.add_argument("--tokens-per-response", type=int, default=200)
    w = sub.add_parser("wall", help="énergie murale : P_plateforme / débit")
    w.add_argument("name")
    w.add_argument("--device-watts", type=float, default=15.0)
    w.add_argument("--tok-s", type=float, nargs="+", default=[2, 5, 10, 20, 50])
    c = sub.add_parser("json", help="export JSON d'un preset")
    c.add_argument("name")

    args = p.parse_args()

    if args.cmd == "presets":
        for k, a in PRESETS.items():
            print(f"{k:<20} {a.name}")
    elif args.cmd == "show":
        show(PRESETS[args.name])
    elif args.cmd == "table":
        table(args.names or list(PRESETS))
    elif args.cmd == "train":
        train_report(PRESETS[args.name], args.tokens, args.gpu_tflops, args.mfu, args.usd_per_hour)
    elif args.cmd == "serve":
        import dataclasses
        total_ctx = args.parallel * args.ctx_per_slot
        a = dataclasses.replace(PRESETS[args.name], ctx=total_ctx)
        print(f"\n=== Serveur web : {a.name} ===")
        print(f"  Slots {args.parallel} × {args.ctx_per_slot} tokens = contexte total {total_ctx}")
        print(f"  Poids ................ {fmt_bytes(a.weight_bytes)}")
        print(f"  KV total (partagé) ... {fmt_bytes(a.kv_bytes_total)}")
        print(f"  Runtime + scratch .... {fmt_bytes((a.runtime_mb + a.activation_mb)*1e6)}")
        print(f"  RSS par process ...... {a.rss_mb:.0f} Mo  "
              f"{'OK < 200 Mo' if a.rss_mb <= 200 else 'HORS BUDGET'}")
        aggregate = args.tok_s_single * args.batching_gain
        per_user = aggregate / args.parallel
        print(f"  Débit agrégé estimé .. {aggregate:.0f} tok/s "
              f"({args.tok_s_single:.0f} mono-flux × {args.batching_gain:.1f}) "
              f"→ ~{per_user:.1f} tok/s par utilisateur à {args.parallel} slots [estim.]")
        sec_per_req = args.tokens_per_response / max(per_user, 0.1)
        print(f"  Latence réponse ...... ~{sec_per_req:.0f} s pour {args.tokens_per_response} tokens")
        energy_wh = args.device_watts * sec_per_req * args.requests_per_day / 3600
        print(f"  Énergie .............. {energy_wh:.0f} Wh/jour pour {args.requests_per_day} requêtes")
        print(f"  Coût annuel .......... {energy_wh*365/1000*0.25:.0f} € à 0,25 €/kWh")
        print()
    elif args.cmd == "wall":
        a = PRESETS[args.name]
        print(f"\n=== {a.name} : énergie réellement payée au mur ===")
        print(f"  Hypothèse : plateforme à {args.device_watts:.0f} W pendant la génération.")
        print(f"  {'débit':>10}  {'E_murale':>12}  {'dont DRAM':>12}  {'part modèle':>12}")
        for tps in args.tok_s:
            wall = a.wall_energy_per_token(args.device_watts, tps)
            print(f"  {tps:>7.1f} t/s  {wall:>10.3f} J  {a.energy_per_token_j:>10.4f} J"
                  f"  {100*a.energy_per_token_j/wall:>10.1f} %")
        print()
    elif args.cmd == "json":
        a = PRESETS[args.name]
        d = asdict(a)
        d.update(
            params=a.params,
            weight_bytes=a.weight_bytes,
            kv_bytes_total=a.kv_bytes_total,
            rss_mb=a.rss_mb,
            energy_per_token_j=a.energy_per_token_j,
        )
        print(json.dumps(d, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
