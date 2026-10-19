"""stream_probe.py — combien de tokens par seconde si les poids vivent sur disque ?

Question : un modèle dense de 132 Mo (MobileLLM-R1-140M Q4_K_M) tient-il dans
100 Mo de RAM si on ne garde aucun poids résident et qu'on relit le fichier à
chaque token ?  Réponse = 132 Mo / débit réel du disque.

Ce que fait ce script, sans modèle ni GPU :
  1. écrit un fichier de la taille demandée (une seule fois) ;
  2. simule T tokens : pour chaque token, lit le fichier couche par couche
     (18 couches pour MobileLLM-R1-140M), puis évince les pages lues avec
     fadvise(DONTNEED) pour ne pas mesurer le page cache ;
  3. rapporte Mo/token, tokens/s, débit et RSS du processus.

Trois régimes comparés :
  cache   lecture bufferisée, page cache laissé en place   (runtime naïf)
  évincé  lecture bufferisée + DONTNEED par couche         (RAM bornée)
  direct  O_DIRECT + DONTNEED                              (RAM bornée, sans copie)

Limite honnête : le disque de cette sandbox n'est pas celui d'un petit PC.
Le chiffre vaut comme borne de la *méthode*, pas comme promesse produit.

  python3 tools/stream_probe.py --size-mb 132 --layers 18 --tokens 20
"""
from __future__ import annotations

import argparse
import mmap
import os
import resource
import time

BLOCK = 8 * 1024 * 1024


def make_file(path: str, size_mb: int) -> None:
    if os.path.exists(path) and os.path.getsize(path) >= size_mb * 1024 * 1024:
        return
    block = os.urandom(BLOCK)
    with open(path, "wb", buffering=0) as f:
        written = 0
        while written < size_mb * 1024 * 1024:
            n = min(len(block), size_mb * 1024 * 1024 - written)
            f.write(block[:n])
            written += n


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run(path: str, size_mb: int, layers: int, tokens: int, mode: str) -> dict:
    size = size_mb * 1024 * 1024
    per_layer = (size // layers) & ~4095          # multiple de 4 Kio
    flags = os.O_RDONLY | (os.O_DIRECT if mode == "direct" else 0)
    fd = os.open(path, flags)
    # tampon alloué une fois, aligné sur une page : ni allocation ni copie
    buf = mmap.mmap(-1, per_layer) if mode == "direct" else bytearray(per_layer)
    t0 = time.perf_counter()
    total = 0
    for _tok in range(tokens):
        for layer in range(layers):
            off = layer * per_layer
            if mode == "direct":
                n = os.preadv(fd, [buf], off)
            else:
                n = len(os.pread(fd, per_layer, off))
            total += n
            if mode != "cache":
                os.posix_fadvise(fd, off, per_layer, os.POSIX_FADV_DONTNEED)
    dt = time.perf_counter() - t0
    os.close(fd)
    return {
        "mode": mode,
        "size_mb": size_mb,
        "tokens": tokens,
        "seconds": round(dt, 3),
        "gb_per_s": round(total / dt / 1e9, 2),
        "tokens_per_s": round(tokens / dt, 2),
        "max_rss_mb": round(rss_mb(), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/tmp/stream-probe.bin")
    ap.add_argument("--size-mb", type=int, default=132)
    ap.add_argument("--layers", type=int, default=18)
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--modes", default="cache,evince,direct")
    ap.add_argument("--keep-file", action="store_true")
    args = ap.parse_args()

    make_file(args.path, args.size_mb)
    modes = [m.replace("evince", "evince") for m in args.modes.split(",")]
    modes = ["evince" if m == "evince" else m for m in modes]

    rows = [run(args.path, args.size_mb, args.layers, args.tokens, m) for m in modes]

    hdr = f"{'régime':8s} {'s':>7s} {'GB/s':>6s} {'tok/s':>7s} {'RSS max':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['mode']:8s} {r['seconds']:7.2f} {r['gb_per_s']:6.2f} "
              f"{r['tokens_per_s']:7.2f} {r['max_rss_mb']:7.1f} Mo")

    print(f"\nFichier de {args.size_mb} Mo relu intégralement à chaque token,")
    print(f"{args.layers} lectures par token, DONTNEED après chaque lecture.")
    if not args.keep_file:
        os.unlink(args.path)


if __name__ == "__main__":
    main()
