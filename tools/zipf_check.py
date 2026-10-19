#!/usr/bin/env python3
"""
zipf_check.py — mesurer les deux paramètres qui décident du Programme II.

Auteur : KAIROS · branche arena/c9070fe4-scienceai · 2026-10-19
Aucune dépendance (stdlib pure).

Pourquoi cet outil
------------------
`tools/skill_econ.py` montre que le rendement d'une bibliothèque de
compétences dépend de deux nombres, et de deux seulement :

  1. l'exposant de Zipf `alpha` de la demande de compétences ;
  2. le nombre moyen de décompositions alternatives par requête.

Si `alpha < 0,8`, la longue traîne est trop lourde : la couverture
s'effondre et l'avantage du tri-système tombe à 1,5x. Si plusieurs
décompositions indépendantes existent, la couverture monte en flèche.

Ces deux nombres se mesurent sur des logs d'usage réels, pour zéro euro.
Cet outil fait le calcul ; il ne remplace pas les logs.

Entrée
------
Un fichier JSONL, une requête par ligne :

  {"label": "traduire", "alternatives": ["traduire", "reformuler"]}
  {"label": "resumer"}

`label` est la compétence réellement nécessaire. `alternatives` est
optionnel : la liste des compétences qui auraient aussi résolu la requête.
Si le fichier est du texte brut, chaque ligne est un label.

Sortie
------
  * nombre de requêtes, de labels distincts ;
  * alpha estimé par maximum de vraisemblance (loi de puissance tronquée) ;
  * couverture empirique et prédite par le modèle ;
  * nombre de compétences nécessaires pour couvrir 90 / 95 / 99 % ;
  * statistiques de décomposition et couverture effective.

Usage
-----
  python3 tools/zipf_check.py logs.jsonl
  python3 tools/zipf_check.py --demo --alpha 1.0 --labels 10000 --requests 200000
  python3 tools/zipf_check.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1. Estimation de l'exposant de Zipf
# ---------------------------------------------------------------------------

def harmonic(n: int, alpha: float) -> float:
    """H_n(alpha) = somme_{k=1..n} k^-alpha, sans boucle quand c'est utile."""
    if n <= 0:
        return 0.0
    if abs(alpha - 1.0) < 1e-12:
        # approximation harmonique, suffisante pour n > 100
        if n > 100:
            return math.log(n) + 0.5772156649 + 1.0 / (2 * n)
        return sum(1.0 / k for k in range(1, n + 1))
    if n > 20000:
        # Euler-Maclaurin : somme + intégrale + correction
        return (sum(k ** -alpha for k in range(1, 2001))
                + ((2001 ** (1 - alpha)) - 1) / (alpha - 1)
                + 0.5 * (n ** -alpha + 2001 ** -alpha))
    return sum(k ** -alpha for k in range(1, n + 1))


def zipf_loglik(counts_sorted: Sequence[int], alpha: float, n_labels: int) -> float:
    """Log-vraisemblance d'un échantillon pour une Zipf tronquée à n_labels."""
    total = sum(counts_sorted)
    if total == 0:
        return 0.0
    h = harmonic(n_labels, alpha)
    if h <= 0 or not math.isfinite(h):
        return -1e18
    # les rangs vont de 1 à n_labels
    s = 0.0
    for rank, c in enumerate(counts_sorted, start=1):
        if c:
            s += c * (-alpha * math.log(rank))
    return s - total * math.log(h)


def estimate_alpha(counts_sorted: Sequence[int], n_labels: int,
                   lo: float = 0.05, hi: float = 3.0) -> Tuple[float, float]:
    """Recherche ternaire sur la log-vraisemblance (concave en alpha)."""
    for _ in range(80):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if zipf_loglik(counts_sorted, m1, n_labels) < zipf_loglik(counts_sorted, m2, n_labels):
            lo = m1
        else:
            hi = m2
    a = (lo + hi) / 2.0
    return a, zipf_loglik(counts_sorted, a, n_labels)


def predicted_coverage(alpha: float, n_labels: int, n_skills: int) -> float:
    if n_skills >= n_labels:
        return 1.0
    return harmonic(n_skills, alpha) / harmonic(n_labels, alpha)


def required_skills(alpha: float, n_labels: int, target: float) -> int:
    """Plus petit n tel que H_n/H_M >= target (recherche dichotomique)."""
    lo, hi = 1, n_labels
    while lo < hi:
        mid = (lo + hi) // 2
        if predicted_coverage(alpha, n_labels, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# 2. Lecture des logs
# ---------------------------------------------------------------------------

def load_logs(path: str) -> Tuple[Counter, List[int], int]:
    """Retourne (comptes, nombres d'alternatives, requêtes totales)."""
    counts: Counter = Counter()
    n_alternatives: List[int] = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    counts[line] += 1
                    n_alternatives.append(1)
                    continue
                label = str(obj.get("label") or obj.get("skill") or obj.get("task") or "")
                alts = obj.get("alternatives") or []
                if isinstance(alts, str):
                    alts = [a.strip() for a in alts.split("|") if a.strip()]
                if label:
                    counts[label] += 1
                n_alternatives.append(max(1, len(alts)))
            else:
                counts[line] += 1
                n_alternatives.append(1)
    return counts, n_alternatives, total


# ---------------------------------------------------------------------------
# 3. Analyse et rapport
# ---------------------------------------------------------------------------

def analyse(counts: Counter, n_alternatives: Sequence[int], total: int) -> Dict[str, float]:
    if total <= 0:
        raise ValueError("aucune requête")
    ordered = sorted(counts.values(), reverse=True)
    n_labels = len(ordered)
    alpha, ll = estimate_alpha(ordered, n_labels)

    n_skill_counts = {n: sum(ordered[:n]) / total for n in (10, 100, 1000, 10000, 100000)
                      if n <= n_labels}

    mean_alts = sum(n_alternatives) / len(n_alternatives)
    # couverture effective si les alternatives sont indépendantes :
    # 1 - (1 - c)^k, avec c = couverture d'une seule tentative
    base = min(1.0, sum(ordered[:min(1000000, n_labels)]) / total)
    eff_90 = 1.0 - (1.0 - base) ** mean_alts

    return {
        "total": total,
        "n_labels": n_labels,
        "alpha": alpha,
        "loglik": ll,
        "mean_alternatives": mean_alts,
        "distinct_ratio": n_labels / total,
        "coverage_top": n_skill_counts,
        "skills_90": required_skills(alpha, n_labels, 0.90),
        "skills_95": required_skills(alpha, n_labels, 0.95),
        "skills_99": required_skills(alpha, n_labels, 0.99),
        "coverage_effective_1m": eff_90,
    }


def report(r: Dict[str, float], label: str = "") -> None:
    print("=" * 72)
    print("  MESURE DE LA DEMANDE — " + (label or "logs"))
    print("=" * 72)
    print(f"  requêtes ............. {r['total']:,}".replace(",", " "))
    print(f"  labels distincts ..... {r['n_labels']:,}".replace(",", " ").replace(".", " "))
    print(f"  alpha (Zipf, MLE) .... {r['alpha']:.3f}")
    print(f"  alternatives/requête . {r['mean_alternatives']:.2f}")
    print()
    print("  Couverture réelle par le top-n (mesurée sur ces logs) :")
    for n, c in sorted(r["coverage_top"].items()):
        print(f"    top {n:>7,} ....... {100 * c:6.2f} %".replace(",", " "))
    print()
    print("  Nombre de compétences nécessaires (modèle de Zipf ajusté) :")
    print(f"    90 % ............ {r['skills_90']:,}".replace(",", " "))
    print(f"    95 % ............ {r['skills_95']:,}".replace(",", " "))
    print(f"    99 % ............ {r['skills_99']:,}".replace(",", " "))
    print()
    print("  Lecture :")
    if r["alpha"] < 0.8:
        print("    alpha < 0,8 -> longue traîne trop lourde ; la bibliothèque")
        print("    ne couvrira pas assez. Le tri-système perd l'essentiel de")
        print("    son avantage (1,5x au lieu de 20x). PIVOTER.")
    elif r["alpha"] < 1.2:
        print("    alpha entre 0,8 et 1,2 -> zone de travail. L'avantage dépend")
        print("    fortement des décompositions alternatives ; les mesurer.")
    else:
        print("    alpha > 1,2 -> tête de distribution très concentrée ; la")
        print("    bibliothèque est très rentable. Aller à E-Ω1.")
    print()
    if r["mean_alternatives"] > 1.5:
        print(f"    {r['mean_alternatives']:.2f} alternatives par requête : la couverture")
        print("    effective dépasse la couverture stricte. Vérifier l'indépendance")
        print("    des alternatives avant de s'appuyer dessus.")
    else:
        print("    Peu ou pas d'alternatives : la couverture stricte est la bonne")
        print("    hypothèse, et elle est plus fragile.")
    print()


# ---------------------------------------------------------------------------
# 4. Démonstration et auto-test
# ---------------------------------------------------------------------------

def generate_zipf(alpha: float, n_labels: int, n_requests: int,
                  seed: int = 42) -> List[str]:
    """Échantillonne une Zipf tronquée par transformée inverse approchée."""
    rng = random.Random(seed)
    weights = [k ** -alpha for k in range(1, n_labels + 1)]
    cdf = []
    acc = 0.0
    for w in weights:
        acc += w
        cdf.append(acc)
    total_w = acc
    out = []
    import bisect
    for _ in range(n_requests):
        x = rng.random() * total_w
        i = bisect.bisect_left(cdf, x)
        out.append("skill_%06d" % (i + 1))
    return out


def demo(alpha: float, n_labels: int, n_requests: int, seed: int = 42) -> Dict[str, float]:
    requests = generate_zipf(alpha, n_labels, n_requests, seed)
    counts = Counter(requests)
    alts = [1] * len(requests)
    return analyse(counts, alts, len(requests))


def self_test() -> int:
    ok = True
    for true_alpha in (0.7, 1.0, 1.3):
        r = demo(true_alpha, 2000, 20000)
        err = abs(r["alpha"] - true_alpha)
        status = "OK" if err < 0.08 else "ÉCHEC"
        if err >= 0.08:
            ok = False
        print(f"  alpha réel {true_alpha:.1f} -> estimé {r['alpha']:.3f} "
              f"(erreur {err:.3f}) {status}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="fichier JSONL ou texte")
    ap.add_argument("--demo", action="store_true", help="générer des logs synthétiques")
    ap.add_argument("--alpha", type=float, default=1.0, help="exposant des logs de démo")
    ap.add_argument("--labels", type=int, default=10000, help="labels distincts (démo)")
    ap.add_argument("--requests", type=int, default=200000, help="requêtes (démo)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true", help="sortie JSON")
    args = ap.parse_args(argv)

    if args.self_test:
        print("Auto-test de l'estimateur :")
        return self_test()

    if args.demo:
        r = demo(args.alpha, args.labels, args.requests, args.seed)
        if args.json:
            print(json.dumps(r, indent=2, ensure_ascii=False))
        else:
            report(r, f"démo Zipf alpha={args.alpha}, {args.labels} labels")
        return 0

    if not args.path:
        ap.error("fournir un fichier, ou --demo / --self-test")
    counts, alts, total = load_logs(args.path)
    r = analyse(counts, alts, total)
    if args.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        report(r, args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
