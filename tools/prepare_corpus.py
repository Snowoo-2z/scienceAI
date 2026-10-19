#!/usr/bin/env python3
"""Construit le corpus d'entraînement de NanoFR à partir de sources libres.

Sources :
  * isosceles (CC-BY 4.0)  : Maupassant FR + Poe/Baudelaire FR  -> ../corpus_src/data
  * ELTeC-fra  (CC-BY 4.0) : 100 romans français 1840-1920      -> ../eltec/plain1

Sortie : data/corpus/train.txt, data/corpus/val.txt (+ statistiques)
"""
from __future__ import annotations

import argparse
import random
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT.parent / "corpus_src" / "data"
ELTEC = ROOT.parent / "eltec" / "plain1"
OUT = ROOT / "data" / "corpus"


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    lines = []
    for line in text.splitlines():
        line = " ".join(line.split())
        lines.append(line)
    text = "\n".join(lines)
    # au plus une ligne vide consécutive
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def isosceles_docs() -> list[str]:
    docs = []
    if not CORPUS.exists():
        return docs
    for p in sorted(CORPUS.rglob("*.txt")):
        if "/fr/" not in str(p):
            continue
        t = clean(p.read_text(encoding="utf-8", errors="replace"))
        if len(t) > 2000:
            docs.append(t)
    return docs


def eltec_docs(limit_chars: int) -> list[str]:
    docs = []
    if not ELTEC.exists():
        return docs
    files = sorted(ELTEC.glob("*.txt"))
    total = 0
    for p in files:
        t = clean(p.read_text(encoding="utf-8", errors="replace"))
        if len(t) > 2000:
            docs.append(t)
            total += len(t)
            if total >= limit_chars:
                break
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eltec-chars", type=int, default=9_000_000,
                    help="budget de caractères pris dans ELTeC-fra")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    iso = isosceles_docs()
    elt = eltec_docs(args.eltec_chars)
    docs = iso + elt
    rng = random.Random(args.seed)
    rng.shuffle(docs)

    n_val = max(2, int(0.02 * len(docs)))
    val, train = docs[:n_val], docs[n_val:]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.txt").write_text("\n\n".join(train), encoding="utf-8")
    (OUT / "val.txt").write_text("\n\n".join(val), encoding="utf-8")

    ct = sum(len(d) for d in train)
    cv = sum(len(d) for d in val)
    print(f"documents   : {len(docs)} (isosceles={len(iso)}, eltec={len(elt)})")
    print(f"train       : {len(train)} docs, {ct/1e6:.2f} M caractères")
    print(f"validation  : {len(val)} docs, {cv/1e6:.2f} M caractères")
    print(f"écrit dans  : {OUT}")


if __name__ == "__main__":
    main()
