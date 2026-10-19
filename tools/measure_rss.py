#!/usr/bin/env python3
"""Mesure le pic de RSS et la durée d'une commande (E0).

    python3 tools/measure_rss.py -- llama-cli -m modele.gguf -p "..." -n 32

ru_maxrss (RUSAGE_CHILDREN) donne le pic de RSS de tous les enfants : c'est
la mémoire réellement touchée par le processus, hors page cache.
"""
from __future__ import annotations

import resource
import subprocess
import sys
import time


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] != "--":
        print(__doc__)
        return 2
    cmd = sys.argv[2:]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    dt = time.time() - t0
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    rss = ru.ru_maxrss / 1024.0
    print(f"commande   : {' '.join(cmd[:3])} ...")
    print(f"durée      : {dt:.2f} s")
    print(f"pic RSS    : {rss:.1f} Mo")
    print(f"user/sys   : {ru.ru_utime:.2f} / {ru.ru_stime:.2f} s")
    print(f"code retour: {proc.returncode}")
    if proc.stdout:
        print("--- stdout (fin) ---")
        print(proc.stdout[-1500:])
    if proc.returncode != 0 and proc.stderr:
        print("--- stderr (fin) ---")
        print(proc.stderr[-1500:])
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
