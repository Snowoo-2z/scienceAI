#!/usr/bin/env python3
"""substrate_budget.py — répartir un budget de RAM entre plusieurs substrats.

Auteur : MNÉMOSYNE · 2026-10-19 · stdlib pure.

La question n'est pas « combien de paramètres », mais « combien de Mo pour
chaque substrat » : cœur dense, cache de compétences-code, cartouches LoRA,
mémoire plastique, index de retrieval.

Le modèle de couverture est celui de KAIROS (tools/zipf_check.py) :
couverture(top n) = H_n(alpha) / H_N(alpha), N = 10 000 compétences.
Les temps de chargement sont les médianes mesurées par tools/wsprobe.c
(2026-10-19, Xeon 2,60 GHz, 2 vCPU, ext4, cache évincé).

Usage :
  python3 tools/substrate_budget.py table
  python3 tools/substrate_budget.py coverage --alpha 1.0
  python3 tools/substrate_budget.py compare
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Médianes wsprobe, cache évincé, pread buffered (results/wsprobe-sandbox-2g.txt)
# taille en Kio -> ms
LOAD_MS = {256: 0.34, 512: 0.72, 1024: 1.01, 2048: 1.86,
           4096: 3.07, 8192: 5.75}
# O_DIRECT (results/wsprobe-sandbox-direct.txt), plus favorable
LOAD_MS_DIRECT = {256: 0.16, 512: 0.22, 1024: 0.34, 2048: 0.61,
                  4096: 0.95, 8192: 2.45}


def harmonic(n: int, alpha: float) -> float:
    if n <= 0:
        return 0.0
    if abs(alpha - 1.0) < 1e-12:
        return math.log(n) + 0.5772156649 + 1.0 / (2 * n) if n > 100 else sum(1 / k for k in range(1, n + 1))
    if n > 20000:
        return (sum(k ** -alpha for k in range(1, 2001))
                + ((2001 ** (1 - alpha)) - 1) / (alpha - 1)
                + 0.5 * (n ** -alpha + 2001 ** -alpha))
    return sum(k ** -alpha for k in range(1, n + 1))


def coverage(n_cached: int, alpha: float, n_labels: int = 10000) -> float:
    if n_cached >= n_labels:
        return 1.0
    return harmonic(n_cached, alpha) / harmonic(n_labels, alpha)


def load_ms(kib: float, direct: bool = False) -> float:
    """Interpolation log-linéaire des médianes mesurées; <256 Kio = linéaire."""
    table = LOAD_MS_DIRECT if direct else LOAD_MS
    keys = sorted(table)
    if kib <= keys[0]:
        return table[keys[0]] * kib / keys[0]
    if kib >= keys[-1]:
        return table[keys[-1]] * kib / keys[-1]
    for a, b in zip(keys, keys[1:]):
        if a <= kib <= b:
            t = (math.log2(kib) - math.log2(a)) / (math.log2(b) - math.log2(a))
            return table[a] + t * (table[b] - table[a])
    raise AssertionError


@dataclass
class Alloc:
    name: str
    core_m: float          # paramètres du cœur (millions)
    core_bits: float
    kv_mb: float
    router_mb: float
    plastic_mb: float
    skill_mb: float        # cache de compétences-code
    lora_mb: float         # cache de cartouches LoRA
    retrieval_mb: float
    runtime_mb: float
    skill_kb: float = 5.0  # taille moyenne d'une compétence-code
    lora_kb: float = 2048.0
    gops: float = 20.0
    watts: float = 15.0
    answer_tokens: int = 200

    @property
    def core_mb(self) -> float:
        return self.core_m * self.core_bits / 8.0

    @property
    def rss_mb(self) -> float:
        return (self.core_mb + self.kv_mb + self.router_mb + self.plastic_mb
                + self.skill_mb + self.lora_mb + self.retrieval_mb + self.runtime_mb)

    @property
    def n_skills(self) -> int:
        return int(self.skill_mb * 1024 / self.skill_kb)

    @property
    def n_loras(self) -> int:
        return int(self.lora_mb * 1024 / self.lora_kb)

    @property
    def tok_s(self) -> float:
        if self.core_m <= 0:
            return 0.0
        return self.gops * 1e3 / (2.0 * self.core_m)

    def query_time_s(self, alpha: float, use_skills: bool = True,
                     use_lora: bool = True) -> float:
        gen = self.answer_tokens / self.tok_s if self.tok_s else float("inf")
        load = 0.0
        if use_skills and self.n_skills:
            load += load_ms(self.skill_kb, direct=True) / 1000.0
        if use_lora and self.n_loras:
            # une cartouche par requête, pas par token
            p = min(1.0, coverage(self.n_loras, alpha))
            load += p * load_ms(self.lora_kb, direct=True) / 1000.0
        if self.retrieval_mb:
            load += load_ms(3.0, direct=True) / 1000.0
        return gen + load

    def joules(self, alpha: float) -> float:
        return self.watts * self.query_time_s(alpha)


def configs() -> list[Alloc]:
    return [
        Alloc("dense-120m", 120, 4.0, 10.0, 0, 0, 0, 0, 0, 26),
        Alloc("core60-only", 60, 4.0, 10.0, 3, 0, 0, 0, 0, 5),
        Alloc("core60+skills4.1k", 60, 4.0, 10.0, 3, 0, 20, 0, 0, 5),
        Alloc("core60+skills3.7k", 60, 4.0, 10.0, 3, 0, 20, 0, 0, 5, skill_kb=5.5),
        Alloc("core60+skills3.7k+ret8", 60, 4.0, 10.0, 3, 0, 20, 0, 8, 5, skill_kb=5.5),
        Alloc("core60+lora10", 60, 4.0, 10.0, 3, 0, 0, 20, 0, 5),
        Alloc("core60+plastic20", 60, 4.0, 10.0, 3, 20, 0, 0, 0, 5),
        Alloc("core60+mix", 60, 4.0, 10.0, 3, 8, 10, 6, 4, 5, skill_kb=5.5),
    ]


def cmd_table(args) -> None:
    print(f"{'config':30s} {'RSS':>6s} {'cœur':>6s} {'skills':>7s} {'LoRA':>5s} "
          f"{'couverture':>10s} {'tok/s':>6s} {'J/rép':>6s} {'charge':>7s}")
    print("-" * 96)
    for c in configs():
        cov = coverage(c.n_skills + c.n_loras, args.alpha) if (c.n_skills or c.n_loras) else 0.0
        load = 0.0
        if c.n_skills:
            load += load_ms(c.skill_kb, direct=True)
        if c.n_loras:
            load += coverage(c.n_loras, args.alpha) * load_ms(c.lora_kb, direct=True)
        print(f"{c.name:30s} {c.rss_mb:6.1f} {c.core_mb:6.1f} {c.n_skills:7d} "
              f"{c.n_loras:5d} {100*cov:9.1f}% {c.tok_s:6.1f} {c.joules(args.alpha):6.0f} "
              f"{load:6.2f}ms")


def cmd_coverage(args) -> None:
    print(f"Couverture du top-n, Zipf alpha={args.alpha}, N=10 000")
    print(f"{'n':>6s} {'couverture':>10s} {'code 5,5 ko':>12s} {'LoRA 2 Mio':>11s}")
    for n in (10, 30, 100, 300, 1000, 2000, 3646, 5000, 7141, 10000):
        c = coverage(n, args.alpha)
        print(f"{n:6d} {100*c:9.1f}% {n*5.5/1024:11.1f} Mo {n*2/1024:10.1f} Mo")


def cmd_compare(args) -> None:
    """Le contraste central : streaming dense vs chargement modulaire."""
    print("Comparaison des deux régimes de chargement")
    print("Hypothèses : disque 1 Go/s effectif, 4 bits/poids, réponse 200 tokens.")
    print(f"{'régime':38s} {'octets':>12s} {'unité':>8s} {'ms':>10s} {'ms/token amorti':>16s}")
    print("-" * 90)
    cases = [
        ("dense 7B 4-bit, tout relu", 3.5e9, "par token"),
        ("dense 270M 4-bit, tout relu", 1.35e8, "par token"),
        ("dense 60M 4-bit, tout relu", 3.0e7, "par token"),
        ("MoE 3B, 300M actifs 4-bit", 1.5e8, "par token"),
        ("compétence-code 5,5 ko", 5.5e3, "par requête"),
        ("cartouche LoRA 2 Mio", 2.1e6, "par requête"),
    ]
    for name, b, unit in cases:
        ms = b / 1e9 * 1000
        if unit == "par requête":
            print(f"{name:38s} {b:12.0f} {unit:>8s} {ms:10.3f} {ms/200:16.5f}")
        else:
            print(f"{name:38s} {b:12.0f} {unit:>8s} {ms:10.3f} {ms:16.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["table", "coverage", "compare"])
    p.add_argument("--alpha", type=float, default=1.0)
    args = p.parse_args()
    {"table": cmd_table, "coverage": cmd_coverage, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
