#!/usr/bin/env python3
"""
route_bench.py — le moissonnage suffit-il ? Non. Voici la mesure.

On passe les 51 questions du banc d'évaluation dans le routeur lexical du
moissonneur, sans aucune compétence écrite à la main. On compte trois choses :

  * couvertes et justes ;
  * couvertes et FAUSSES (le pire cas : une preuve pour une mauvaise réponse) ;
  * non couvertes.

C'est la mesure honnête du goulot restant : trouver la bonne fonction et
remplir ses arguments.

Auteur : KAIROS · 2026-10-19.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harvest  # noqa: E402


def main() -> int:
    cards, st = harvest.harvest()
    items = json.loads((HERE / "eval_questions.json").read_text(encoding="utf-8"))
    justes, faux, non_couverts, hors = [], [], [], []
    for it in items:
        res, fn, proof = harvest.route_and_call(cards, it["q"])
        if res is None:
            non_couverts.append((it["q"], proof))
            continue
        if it["kind"] != "competence" or not it["expect"]:
            hors.append((it["q"], res, fn))
            continue
        norm = lambda s: s.lower().replace(",", ".")
        if norm(it["expect"]) in norm(res):
            justes.append((it["q"], res, fn))
        else:
            faux.append((it["q"], res, fn, it["expect"]))

    print("=" * 74)
    print("  ROUTAGE SUR LA BIBLIOTHÈQUE MOISSONNÉE SEULE (aucune main humaine)")
    print("=" * 74)
    print(f"  fiches disponibles ....... {st['testees_ok']} (sur {st['fiches']})")
    print(f"  questions ................ {len(items)}")
    print(f"  justes ................... {len(justes)}")
    print(f"  FAUSSES (réponse + preuve) {len(faux)}")
    print(f"  non couvertes ............ {len(non_couverts)}")
    print(f"  hors domaine, répondu .... {len(hors)}")
    print()
    print("  Justes :")
    for q, r, fn in justes:
        print(f"    {q} -> {r}  ({fn})")
    if faux:
        print("\n  Fausses — le mode d'échec qui compte :")
        for q, r, fn, exp in faux:
            print(f"    {q} -> {r}  ({fn})   attendu {exp}")
    if hors:
        print("\n  Hors domaine mais répondu quand même :")
        for q, r, fn in hors:
            print(f"    {q} -> {r}  ({fn})")
    print()
    print(f"  Précision quand le routeur répond : "
          f"{len(justes)}/{len(justes) + len(faux) + len(hors)}")
    print("  Lecture : le moissonnage supprime le coût d'écriture, pas le coût")
    print("  de compréhension. Un routeur sans sémantique produit des réponses")
    print("  fausses avec une preuve — exactement ce que le vérificateur doit")
    print("  interdire.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
