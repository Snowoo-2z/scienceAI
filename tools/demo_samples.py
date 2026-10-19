#!/usr/bin/env python3
"""Échantillonne NanoFR sur une série de débuts de phrases (via llama-server).

    python3 tools/demo_samples.py --port 8081 --n 60 --seed 1

Produit un tableau Markdown prêt à coller dans la documentation.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

PROMPTS = [
    "Il était une fois, au bord de la Seine,",
    "Le vieux pêcheur regarda la mer et",
    "Elle entra dans la salle, et",
    "La nuit tombait sur le village ;",
    "— Mon ami, dit-il,",
    "Ce matin-là, le soleil",
]


def complete(port: int, prompt: str, n: int, seed: int) -> dict:
    payload = json.dumps({
        "prompt": prompt, "n_predict": n, "temperature": 0.8, "top_k": 40,
        "top_p": 0.95, "repeat_penalty": 1.1, "seed": seed, "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    print("| Début | Continuation (brute) | tok/s |")
    print("|---|---|---:|")
    for i, p in enumerate(PROMPTS):
        d = complete(args.port, p, args.n, args.seed + i)
        txt = d["content"].replace("\n", " ").replace("|", "/").strip()
        tps = d["timings"]["predicted_per_second"]
        print(f"| {p} | {txt} | {tps:.0f} |")


if __name__ == "__main__":
    main()
