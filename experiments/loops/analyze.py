"""analyze.py — tableaux de comparaison boucles vs couches.

Lit results/loops/*.json (écrits par train.py) et imprime, en texte ou en
markdown, les trois lectures qui nous intéressent :

  1. à calcul égal (même nombre d'applications de bloc) ;
  2. à mémoire égale (même nombre de paramètres) ;
  3. rendement : ce que coûte chaque point de val gagné.

  python3 experiments/loops/analyze.py                 # texte
  python3 experiments/loops/analyze.py --md            # markdown
  python3 experiments/loops/analyze.py --md --partial  # tolère un sweep inachevé
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DIR = ROOT / "results" / "loops"
ORDER = ["deep1", "deep2", "deep4", "loop2", "loop4", "loop2x2", "loop8"]


def load():
    data = []
    for p in sorted(DIR.glob("*.json")):
        if p.name in ("summary.json", "capacity.json", "capacity-smoke.json"):
            continue
        obj = json.loads(p.read_text())
        if isinstance(obj, dict) and obj.get("name", "smoke") != "smoke":
            data.append(obj)
    return data


def fmt_pairs(data, pairs):
    by = {r["name"]: r for r in data}
    out = []
    for a, b in pairs:
        if a not in by or b not in by:
            continue
        ra, rb = by[a], by[b]
        if ra["block_applications"] != rb["block_applications"]:
            continue
        out.append((a, b, ra, rb))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()

    data = load()
    if not data:
        print("aucun résultat")
        return
    by = {r["name"]: r for r in data}
    rows = [by[n] for n in ORDER if n in by]

    if args.md:
        print("| configuration | blocs | boucles | applications | paramètres | val | bits/car | secondes |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['name']} | {r['blocks']} | {r['loops']} | {r['block_applications']} | "
                  f"{r['params']:,} | {r['val_loss']:.4f} | {r['val_bits_per_char']:.3f} | {r['seconds']:.0f} |".replace(",", " "))
        print()
        print("**À calcul égal** (même nombre d'applications de bloc) :")
        print()
        print("| comparaison | paramètres | val | écart |")
        print("|---|---|---|---|")
        for a, b, ra, rb in fmt_pairs(data, [("deep2", "loop2"), ("deep4", "loop4"),
                                             ("deep4", "loop2x2"), ("deep2", "loop4")]):
            print(f"| {a} vs {b} | {ra['params']:,} / {rb['params']:,} | "
                  f"{ra['val_loss']:.4f} / {rb['val_loss']:.4f} | "
                  f"{ra['val_loss'] - rb['val_loss']:+.4f} |".replace(",", " "))
        print()
        print("**À mémoire égale** (même nombre de paramètres) :")
        print()
        print("| comparaison | applications | val | écart |")
        print("|---|---|---|---|")
        for a, b in [("deep1", "loop2"), ("deep1", "loop4"), ("deep1", "loop8")]:
            if a in by and b in by and abs(by[a]["params"] - by[b]["params"]) < 2000:
                ra, rb = by[a], by[b]
                print(f"| {a} vs {b} | {ra['block_applications']} / {rb['block_applications']} | "
                      f"{ra['val_loss']:.4f} / {rb['val_loss']:.4f} | "
                      f"{ra['val_loss'] - rb['val_loss']:+.4f} |")
        print()
        return

    print("Configurations (char-level Shakespeare, d=64, ctx=96, 3 000 pas, même lr)")
    print(f"{'config':9s} {'blocs':>5s} {'boucles':>7s} {'applis':>6s} {'params':>7s} "
          f"{'val':>6s} {'bits/car':>8s} {'s':>5s}")
    print("-" * 66)
    for r in rows:
        print(f"{r['name']:9s} {r['blocks']:5d} {r['loops']:7d} {r['block_applications']:6d} "
              f"{r['params']:7d} {r['val_loss']:6.3f} {r['val_bits_per_char']:8.3f} {r['seconds']:5.0f}")

    print("\n1) À CALCUL ÉGAL (même nombre d'applications de bloc)")
    n = 0
    for a, b, ra, rb in fmt_pairs(data, [("deep2", "loop2"), ("deep4", "loop4"),
                                         ("deep4", "loop2x2"), ("deep2", "loop4")]):
        n += 1
        print(f"  {a:6s} ({ra['params']:6d} p) val {ra['val_loss']:.4f}   vs   "
              f"{b:6s} ({rb['params']:6d} p) val {rb['val_loss']:.4f}   "
              f"écart {ra['val_loss'] - rb['val_loss']:+.4f}")
    if not n:
        print("  (aucune paire complète pour l'instant)")
    print("  écart > 0 : la version bouclée gagne à calcul égal")

    print("\n2) À MÉMOIRE ÉGALE (même nombre de paramètres)")
    n = 0
    for a, b in [("deep1", "loop2"), ("deep1", "loop4"), ("deep1", "loop8")]:
        if a in by and b in by and abs(by[a]["params"] - by[b]["params"]) < 2000:
            n += 1
            ra, rb = by[a], by[b]
            print(f"  {a:6s} appli {ra['block_applications']} val {ra['val_loss']:.4f}   vs   "
                  f"{b:6s} appli {rb['block_applications']} val {rb['val_loss']:.4f}   "
                  f"écart {ra['val_loss'] - rb['val_loss']:+.4f}")
    if not n:
        print("  (aucune paire complète pour l'instant)")
    print("  écart > 0 : plus de boucles = meilleur, à mémoire égale")

    base = by.get("deep1", rows[0])["val_loss"]
    print("\n3) RENDEMENT par rapport à deep1")
    for r in rows:
        if r["name"] == "deep1":
            continue
        dp = (r["params"] - by["deep1"]["params"]) / 1e5 if "deep1" in by else 0
        print(f"  {r['name']:8s} Δval {base - r['val_loss']:+.4f}  Δparams {dp:+.2f}e5  "
              f"applications {r['block_applications']}  {r['seconds']:.0f}s")


if __name__ == "__main__":
    main()
