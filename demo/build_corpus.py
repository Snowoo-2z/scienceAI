#!/usr/bin/env python3
"""
build_corpus.py — constituer un corpus de français réel depuis GitHub.

Source : les livres du domaine public de Project Gutenberg, miroir GITenberg
déposé sur GitHub (donc accessible par codeload, sans HuggingFace).

Aucune dépendance (stdlib). Les livres sont téléchargés en tar.gz, le texte
est extrait et nettoyé de l'en-tête/pied Gutenberg.

Usage :
    python3 demo/build_corpus.py --out data/corpus-fr.txt
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

API = "https://api.github.com/search/repositories?q=%d+org:GITenberg"
RAW = "https://codeload.github.com/GITenberg/%s/tar.gz/HEAD"

# Livres français de Project Gutenberg (identifiants du catalogue).
# Les identifiants consécutifs correspondent souvent aux tomes d'un même roman.
IDS = [
    17489, 17490, 17491, 17492, 17493,   # Les Misérables, tomes I-V
    17989, 17990, 17991, 17992, 17993, 17994,  # Monte-Cristo, tomes I-VI
    13951, 13952,                        # Les Trois Mousquetaires
    19657,                               # Notre-Dame de Paris
    14155,                               # Madame Bovary
    8558,                                # L'Assommoir
    4650,                                # Candide
    5097,                                # Vingt mille lieues sous les mers
    10770,                               # Boule de Suif
    14500,                               # Une vie
]

FR_STOP = {"le", "la", "les", "de", "des", "du", "et", "un", "une", "que",
           "qui", "dans", "pour", "pas", "sur", "avec", "il", "elle", "nous",
           "vous", "ce", "se", "au", "aux", "ne", "plus", "son", "sa", "ses"}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": "scienceAI-corpus/1.0",
        "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def find_repo(book_id: int) -> str | None:
    """La recherche passe par `gh api` : l'API GitHub refuse les appels
    anonymes depuis cette sandbox, mais la CLI est authentifiée."""
    import subprocess
    try:
        out = subprocess.run(
            ["gh", "api", API % book_id],
            capture_output=True, text=True, timeout=45, check=True).stdout
        d = json.loads(out)
    except Exception:  # noqa: BLE001
        return None
    for item in d.get("items", []):
        name = item["full_name"]
        if name.lower().endswith("_%d" % book_id):
            return name
    return None


def clean_gutenberg(text: str) -> str:
    m = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text,
                  re.IGNORECASE | re.DOTALL)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE)
    if m:
        text = text[:m.start()]
    return text


def is_french(text: str, sample: int = 20000) -> bool:
    words = re.findall(r"[a-zà-ÿ']+", text[:sample].lower())
    if len(words) < 500:
        return False
    hits = sum(1 for w in words if w in FR_STOP)
    return hits / len(words) > 0.15


def extract_txt(blob: bytes) -> str | None:
    try:
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
    except Exception:  # noqa: BLE001
        return None
    best = None
    for member in tf.getmembers():
        if not member.isfile() or not member.name.lower().endswith(".txt"):
            continue
        try:
            data = tf.extractfile(member).read()
        except Exception:  # noqa: BLE001
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="ignore")
        text = clean_gutenberg(text)
        if best is None or len(text) > len(best[0]):
            best = (text, member.name)
    return best[0] if best else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus-fr.txt")
    ap.add_argument("--ids", nargs="*", type=int, default=IDS)
    ap.add_argument("--min-kb", type=float, default=40.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts, log = [], []
    total = 0
    for i, bid in enumerate(args.ids, 1):
        t0 = time.time()
        repo = find_repo(bid)
        if not repo:
            log.append((bid, "introuvable", 0, 0.0))
            continue
        try:
            blob = _get(RAW % repo.split("/")[-1], timeout=120)
        except Exception as e:  # noqa: BLE001
            log.append((bid, f"téléchargement : {e}", 0, time.time() - t0))
            continue
        text = extract_txt(blob)
        if not text:
            log.append((bid, "pas de .txt", 0, time.time() - t0))
            continue
        fr = is_french(text)
        kb = len(text.encode("utf-8")) / 1000
        if kb < args.min_kb:
            log.append((bid, "trop court", kb, time.time() - t0))
            continue
        if not fr:
            log.append((bid, "pas français", kb, time.time() - t0))
            continue
        parts.append(text)
        total += len(text)
        log.append((bid, repo.split("/")[-1], kb, time.time() - t0))
        print(f"  [{i:2d}/{len(args.ids)}] {bid} {repo.split('/')[-1][:52]:52} "
              f"{kb:8.0f} Ko  {time.time() - t0:5.1f} s", flush=True)

    corpus = "\n\n".join(parts)
    out.write_text(corpus, encoding="utf-8")
    print()
    print(f"  livres retenus ... {len(parts)} / {len(args.ids)}")
    print(f"  caractères ....... {total:,}".replace(",", " "))
    print(f"  mots ............. {len(re.findall(r'[A-Za-zÀ-ÿ]+', corpus)):,}".replace(",", " "))
    print(f"  fichier .......... {out} ({out.stat().st_size / 1e6:.1f} Mo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
