#!/usr/bin/env python3
"""working_set.py — budget d'un LLM sous contrainte de « working set ».

L'idée : la variable de conception n'est pas le nombre de paramètres, mais le
nombre d'octets qui doivent être résidents en RAM pour produire un token.
Le reste peut vivre sur disque (bibliothèque de cartouches, index, experts
froids). Ce calculateur chiffre cette hiérarchie.

Conventions
-----------
- Les tailles sont en Mo décimaux (1 Mo = 10^6 octets), comme dans le rapport
  de référence.
- bits = bits effectifs par poids, échelles et métadonnées incluses.
  4.5 bits ≈ Q4 avec échelles ; 4.0 ≈ QAT q4_0 ; 1.58 ≈ ternaire BitNet.
- ``core`` : paramètres résidents du cœur (le « bibliothécaire »).
- ``cart`` : paramètres actifs d'une cartouche de compétence.
- ``library`` : paramètres totaux de la bibliothèque sur disque (hors cœur).
- ``load`` : octets lus sur disque pour charger la/les cartouche(s) d'une requête.
- ``gops`` : débit de calcul utile en milliards d'opérations/s.
- ``watts`` : puissance de la plateforme, hors veille.

Aucune dépendance : Python 3.9+ et sa bibliothèque standard.

Usage :
    python3 tools/working_set.py table
    python3 tools/working_set.py show alex-60m-lib300m
    python3 tools/working_set.py law
    python3 tools/working_set.py csv > results/working-set-configs.csv
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

MB = 1_000_000


@dataclass
class Config:
    name: str
    core_m: float          # paramètres du cœur, en millions
    cart_m: float          # paramètres actifs d'une cartouche, en millions
    library_m: float       # paramètres de la bibliothèque sur disque, en millions
    core_bits: float = 4.5
    lib_bits: float = 4.5
    kv_mb: float = 10.0
    runtime_mb: float = 20.0
    cache_mb: float = 5.0
    load_mb: float = 0.0   # octets lus par requête (0 = tout est résident)
    disk_bw_mbs: float = 1000.0
    gops: float = 20.0
    watts: float = 15.0
    answer_tokens: int = 200
    note: str = ""

    # ---- octets ----
    @property
    def core_mb(self) -> float:
        return self.core_m * self.core_bits / 8.0

    @property
    def cart_mb(self) -> float:
        return self.cart_m * self.lib_bits / 8.0

    @property
    def active_mb(self) -> float:
        return self.core_mb + self.cart_mb

    @property
    def library_mb(self) -> float:
        return self.core_mb + self.library_m * self.lib_bits / 8.0

    @property
    def rss_mb(self) -> float:
        return self.active_mb + self.kv_mb + self.runtime_mb + self.cache_mb

    @property
    def load_ms(self) -> float:
        if self.load_mb <= 0 or self.disk_bw_mbs <= 0:
            return 0.0
        return self.load_mb / self.disk_bw_mbs * 1000.0

    # ---- débit ----
    @property
    def compute_tok_s(self) -> float:
        """Plafond roofline : 2 FLOPs par paramètre actif et par token."""
        if self.core_m + self.cart_m <= 0:
            return 0.0
        return self.gops * 1000.0 / (2.0 * (self.core_m + self.cart_m))

    @property
    def query_latency_s(self) -> float:
        return self.load_ms / 1000.0 + self.answer_tokens / self.compute_tok_s

    @property
    def effective_tok_s(self) -> float:
        if self.query_latency_s <= 0:
            return 0.0
        return self.answer_tokens / self.query_latency_s

    @property
    def j_per_token(self) -> float:
        if self.effective_tok_s <= 0:
            return float("inf")
        return self.watts / self.effective_tok_s

    @property
    def j_per_answer(self) -> float:
        return self.j_per_token * self.answer_tokens

    @property
    def kwh_per_1k_answers(self) -> float:
        return self.j_per_answer * 1000.0 / 3.6e6

    @property
    def eur_per_1k_answers(self) -> float:
        return self.kwh_per_1k_answers * 0.25

    @property
    def verdict(self) -> str:
        if self.rss_mb <= 100:
            return "OK <=100"
        if self.rss_mb <= 200:
            return "OK <=200"
        return "HORS BUDGET"


PRESETS: List[Config] = [
    # Baseline denses (le produit actuel et ses concurrents)
    Config("dense-60m-q4", 60, 0, 60, core_bits=4.5, lib_bits=4.5,
           kv_mb=10, runtime_mb=20, cache_mb=5, gops=20, watts=15,
           note="plancher from-scratch sous 1 Go"),
    Config("dense-120m-q4", 120, 0, 120, core_bits=4.5, lib_bits=4.5,
           kv_mb=10, runtime_mb=20, cache_mb=2, gops=20, watts=15,
           note="le dense le plus gros qui tient vers 100 Mo"),
    Config("gemma-270m-qat-8k", 268, 0, 268, core_bits=4.0, lib_bits=4.0,
           kv_mb=16.5, runtime_mb=26, cache_mb=0, gops=40, watts=15,
           note="baseline produit de l'onglet 12 (RSS 176,5 Mo)"),

    # ALEXANDRIE : cœur + cartouches
    Config("alex-60m", 60, 0.5, 0.5, core_bits=4.5, lib_bits=4.5,
           kv_mb=10, runtime_mb=20, cache_mb=5, load_mb=0.28,
           gops=20, watts=15,
           note="cœur seul + une mini-cartouche résidente"),
    Config("alex-60m-lib300m", 60, 2.0, 300, core_bits=4.5, lib_bits=2.0,
           kv_mb=10, runtime_mb=20, cache_mb=5, load_mb=0.5,
           gops=20, watts=15,
           note="bibliothèque ternaire 300 M, 2 M actifs"),
    Config("alex-60m-lib1g", 60, 2.0, 1000, core_bits=4.5, lib_bits=2.0,
           kv_mb=10, runtime_mb=20, cache_mb=5, load_mb=0.5,
           gops=20, watts=15,
           note="bibliothèque 1 Md de paramètres ternaires"),
    Config("alex-60m-lib10g", 60, 2.0, 10000, core_bits=4.5, lib_bits=2.0,
           kv_mb=10, runtime_mb=20, cache_mb=5, load_mb=0.5,
           gops=20, watts=15,
           note="10 Md de capacité, 2,5 Go sur disque, 2 M actifs"),
    Config("alex-100m-lib1g", 100, 4.0, 1000, core_bits=4.5, lib_bits=2.0,
           kv_mb=10, runtime_mb=20, cache_mb=5, load_mb=1.0,
           gops=20, watts=15,
           note="cœur 100 M + cartouches de 4 M"),
    Config("alex-60m-lib1g-slowdisk", 60, 2.0, 1000, core_bits=4.5, lib_bits=2.0,
           kv_mb=10, runtime_mb=20, cache_mb=5, load_mb=0.5,
           disk_bw_mbs=100, gops=20, watts=15,
           note="même chose sur un disque à 100 Mo/s"),
]


def fmt(x: float, nd: int = 1) -> str:
    if x == float("inf"):
        return "inf"
    return f"{x:.{nd}f}"


def row(c: Config) -> str:
    return (f"{c.name:26s} {fmt(c.rss_mb):>7s} {fmt(c.active_mb):>7s} "
            f"{fmt(c.library_mb):>8s} {fmt(c.load_ms, 1):>6s} "
            f"{fmt(c.compute_tok_s, 1):>7s} {fmt(c.effective_tok_s, 1):>7s} "
            f"{fmt(c.j_per_answer, 0):>7s} {c.verdict}")


def table(configs: List[Config]) -> str:
    head = (f"{'config':26s} {'RSS':>7s} {'actif':>7s} {'biblio':>8s} "
            f"{'load':>6s} {'tok/s':>7s} {'eff.':>7s} {'J/rép':>7s} verdict")
    lines = [head, "-" * len(head)]
    for c in configs:
        lines.append(row(c))
    lines.append("")
    lines.append("RSS = cœur + cartouche + KV + runtime + cache (Mo décimaux).")
    lines.append("actif = octets de poids résidents et calculés par token.")
    lines.append("biblio = octets totaux sur disque (cœur + bibliothèque).")
    lines.append("eff. = débit sur une réponse de 200 tokens, chargement inclus.")
    lines.append("J/rép = énergie murale par réponse de 200 tokens (plateforme seule).")
    return "\n".join(lines)


def law() -> str:
    """Le seuil calcul/mémoire et les points de bascule utiles."""
    out = []
    out.append("LOI DU WORKING SET")
    out.append("=" * 72)
    out.append("RSS  ~= N_resident * b/8 + KV + runtime + cache")
    out.append("tok/s ~= min( GOPS / (2 * N_active), BW_disque / octets_lus_par_token )")
    out.append("")
    out.append("Seuil calcul/mémoire (comme KAIROS, généralisé) :")
    out.append("  on est calcul-borné si GOPS < 16 * BW_disque / bits")
    out.append("")
    for bits in (1.58, 2.0, 4.0, 4.5, 8.0):
        for bw in (100, 500, 1000, 3000):
            need = 16.0 * bw / bits
            out.append(f"  {bits:4.2f} bit, {bw:4d} Mo/s -> il faut {need:8.0f} GOPS "
                       f"pour être mémoire-borné")
        out.append("")
    out.append("Lecture : à 2 bits et 1 Go/s, il faut 8 000 GOPS pour que le disque")
    out.append("devienne le goulot. Un cœur de 60 M actifs coûte 0,12 GFLOP/token :")
    out.append("à 20 GOPS, il tourne à 166 tok/s, très au-dessus d'un disque.")
    out.append("")
    out.append("Conséquence : le chargement d'une cartouche par REQUÊTE (et non par")
    out.append("token) est le régime viable. Le disque paie une fois, le calcul paie")
    out.append("à chaque token.")
    return "\n".join(out)


def show(name: str) -> str:
    for c in PRESETS:
        if c.name == name:
            d = asdict(c)
            lines = [f"{c.name}", "=" * len(c.name)]
            for k, v in d.items():
                lines.append(f"{k:14s} : {v}")
            lines.append("")
            lines.append(f"core_mb        : {fmt(c.core_mb, 2)}")
            lines.append(f"cart_mb        : {fmt(c.cart_mb, 2)}")
            lines.append(f"active_mb      : {fmt(c.active_mb, 2)}")
            lines.append(f"library_mb     : {fmt(c.library_mb, 2)}")
            lines.append(f"rss_mb         : {fmt(c.rss_mb, 2)}")
            lines.append(f"load_ms        : {fmt(c.load_ms, 2)}")
            lines.append(f"compute_tok_s  : {fmt(c.compute_tok_s, 1)}")
            lines.append(f"effective_tok_s: {fmt(c.effective_tok_s, 1)}")
            lines.append(f"j_per_token    : {fmt(c.j_per_token, 3)}")
            lines.append(f"j_per_answer   : {fmt(c.j_per_answer, 1)}")
            lines.append(f"eur/1k answers : {fmt(c.eur_per_1k_answers, 4)}")
            lines.append(f"verdict        : {c.verdict}")
            return "\n".join(lines)
    return f"config inconnue: {name}\nconnues: " + ", ".join(c.name for c in PRESETS)


def csv() -> str:
    keys = list(asdict(PRESETS[0]).keys()) + [
        "core_mb", "cart_mb", "active_mb", "library_mb", "rss_mb",
        "load_ms", "compute_tok_s", "effective_tok_s", "j_per_token",
        "j_per_answer", "eur_per_1k_answers", "verdict",
    ]
    lines = [",".join(keys)]
    for c in PRESETS:
        d = asdict(c)
        d.update({
            "core_mb": c.core_mb, "cart_mb": c.cart_mb,
            "active_mb": c.active_mb, "library_mb": c.library_mb,
            "rss_mb": c.rss_mb, "load_ms": c.load_ms,
            "compute_tok_s": c.compute_tok_s,
            "effective_tok_s": c.effective_tok_s,
            "j_per_token": c.j_per_token, "j_per_answer": c.j_per_answer,
            "eur_per_1k_answers": c.eur_per_1k_answers,
            "verdict": c.verdict,
        })
        lines.append(",".join(str(round(d[k], 4)) if isinstance(d[k], float) else str(d[k])
                              for k in keys))
    return "\n".join(lines)


def main() -> None:
    import sys
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
    elif args[0] == "table":
        print(table(PRESETS))
    elif args[0] == "law":
        print(law())
    elif args[0] == "show" and len(args) > 1:
        print(show(args[1]))
    elif args[0] == "csv":
        print(csv())
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
