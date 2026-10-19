"""sparse_probe.py — et si le modèle était creux ?

Le banc `stream_probe.py` a montré qu'un modèle dense plus gros que la RAM se
relit en entier à chaque token (132 Mo/token, 2,4–3,1 tok/s). Celui-ci mesure
le seul régime qui échappe à cette loi : n'aller chercher sur disque que les
morceaux réellement activés.

Modèle simulé : N experts de S Mo chacun, k experts activés par token.
  - routage uniforme  : chaque expert a la même chance (pire cas, aucun cache)
  - routage Zipf      : quelques experts chauds, comme dans un vrai MoE
Mesures : octets lus par token, tok/s, effet d'un cache LRU borné.

  python3 tools/sparse_probe.py --experts 64 --expert-mb 2 --top-k 2 --tokens 200
"""
from __future__ import annotations

import argparse
import os
import resource
import time
from collections import OrderedDict

BLOCK = 1 << 20


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_files(directory: str, n: int, size_mb: int) -> list[str]:
    os.makedirs(directory, exist_ok=True)
    paths = []
    block = os.urandom(BLOCK)
    for i in range(n):
        p = os.path.join(directory, f"expert-{i:03d}.bin")
        paths.append(p)
        if os.path.exists(p) and os.path.getsize(p) == size_mb * BLOCK:
            continue
        with open(p, "wb", buffering=0) as f:
            for _ in range(size_mb):
                f.write(block)
    return paths


def routing(n: int, tokens: int, k: int, kind: str, seed: int = 0):
    """Renvoie, pour chaque token, la liste des experts activés (sans doublon)."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(tokens):
        if kind == "zipf":
            weights = [1.0 / (i + 1) for i in range(n)]
            picks = set()
            while len(picks) < k:
                picks.add(rng.choices(range(n), weights=weights)[0])
            out.append(sorted(picks))
        else:
            out.append(sorted(rng.sample(range(n), k)))
    return out


class LRU:
    def __init__(self, capacity_bytes: int):
        self.cap = capacity_bytes
        self.cur = 0
        self.d: OrderedDict[str, int] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def touch(self, key: str, size: int) -> bool:
        if key in self.d:
            self.d.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        while self.cur + size > self.cap and self.d:
            _, sz = self.d.popitem(last=False)
            self.cur -= sz
            self.evictions += 1
        self.d[key] = size
        self.cur += size
        return False


def simulate(paths: list[str], size_mb: int, plan, cache_mb: int):
    size = size_mb * BLOCK
    fds = [os.open(p, os.O_RDONLY) for p in paths]
    cache = LRU(cache_mb * BLOCK) if cache_mb > 0 else None
    total = 0
    t0 = time.perf_counter()
    for experts in plan:
        for e in experts:
            hit = cache.touch(paths[e], size) if cache else False
            if hit:
                continue
            total += len(os.pread(fds[e], size, 0))
            os.posix_fadvise(fds[e], 0, size, os.POSIX_FADV_DONTNEED)
    dt = time.perf_counter() - t0
    for fd in fds:
        os.close(fd)
    return {
        "seconds": round(dt, 3),
        "tokens_per_s": round(len(plan) / dt, 2),
        "mb_per_token": round(total / len(plan) / BLOCK, 2),
        "gb_read": round(total / 1e9, 3),
        "cache_hit": None if not cache else round(cache.hits / max(1, cache.hits + cache.misses), 3),
        "rss_mb": round(rss_mb(), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/sparse-probe")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--expert-mb", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--cache-mb", type=int, default=24)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    paths = make_files(args.dir, args.experts, args.expert_mb)
    dense_mb = args.experts * args.expert_mb
    print(f"{args.experts} experts × {args.expert_mb} Mo = {dense_mb} Mo "
          f"(même volume qu'un modèle dense), top-{args.top_k} activé par token\n")

    rows = []
    for kind in ("uniforme", "zipf"):
        plan = routing(args.experts, args.tokens, args.top_k, "zipf" if kind == "zipf" else "uniform")
        for cache_mb in (0, args.cache_mb):
            runs = [simulate(paths, args.expert_mb, plan, cache_mb)
                    for _ in range(args.repeat)]
            speeds = sorted(r["tokens_per_s"] for r in runs)
            med = runs[len(runs) // 2] if len(runs) % 2 else runs[len(runs) // 2]
            rows.append({
                "routing": kind,
                "cache_mb": cache_mb,
                "mb_per_token": med["mb_per_token"],
                "median_tok_s": speeds[len(speeds) // 2],
                "min_tok_s": speeds[0],
                "max_tok_s": speeds[-1],
                "cache_hit": med["cache_hit"],
                "rss_mb": med["rss_mb"],
            })

    hdr = f"{'routage':9s} {'cache':>7s} {'Mo/token':>9s} {'tok/s médian':>13s} {'min-max':>13s} {'hits':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        hit = "-" if r["cache_hit"] is None else f"{r['cache_hit']*100:.0f} %"
        print(f"{r['routing']:9s} {r['cache_mb']:5d} Mo {r['mb_per_token']:9.2f} "
              f"{r['median_tok_s']:13.1f} {r['min_tok_s']:6.1f}-{r['max_tok_s']:<6.1f} {hit:>6s}")
    print(f"({args.repeat} passes par configuration, médiane)")

    print(f"\nRappel du banc dense (même disque) : 2,4–3,1 tok/s pour 132 Mo/token.")
    if not args.keep:
        for p in paths:
            os.unlink(p)
        os.rmdir(args.dir)


if __name__ == "__main__":
    main()
