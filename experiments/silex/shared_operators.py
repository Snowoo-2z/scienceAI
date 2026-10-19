"""Mechanical toy, NOT a language model or a new algorithm. Python stdlib only."""
import argparse
from array import array
import json
import math
import platform
import random
import resource
import sys
import time


def factors(d, rank, seed):
    rng = random.Random(seed)
    scale = 1 / math.sqrt(d)
    return (array('d', (rng.uniform(-scale, scale) for _ in range(d * rank))),
            array('d', (rng.uniform(-scale, scale) for _ in range(rank * d))))


def code(operator, rank):
    # Deterministic bounded code: independent of the number of logical operators.
    # It is NOT learned and contains no independently learned knowledge.
    return [math.sin((operator + 1) * (j + 1) * 0.61803398875)
            for j in range(rank)]


def apply(x, u, v, rank, operator):
    """W_i x = U diag(code(i)) V x, without allocating W_i."""
    d = len(x)
    c = code(operator, rank)
    z = [c[j] * sum(v[j * d + k] * x[k] for k in range(d))
         for j in range(rank)]
    return [sum(u[k * rank + j] * z[j] for j in range(rank))
            for k in range(d)]


def dense_reference(x, u, v, rank, operator):
    d = len(x)
    c = code(operator, rank)
    w = [[sum(u[a * rank + j] * c[j] * v[j * d + b]
              for j in range(rank)) for b in range(d)] for a in range(d)]
    return [sum(w[a][b] * x[b] for b in range(d)) for a in range(d)]


def self_test():
    for d, rank in [(3, 1), (8, 3), (16, 4)]:
        u, v = factors(d, rank, 42)
        x = [math.cos(k) for k in range(d)]
        for op in (0, 1, 10000):
            a = apply(x, u, v, rank, op)
            b = dense_reference(x, u, v, rank, op)
            assert max(abs(p - q) for p, q in zip(a, b)) < 1e-12
    print('9 factorized/dense equivalence checks passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if min(args.steps, args.width, args.rank) <= 0:
        parser.error('steps, width and rank must be positive')
    start = time.perf_counter()
    u, v = factors(args.width, args.rank, args.seed)
    x = [math.cos(k) for k in range(args.width)]
    for operator in range(args.steps):
        y = apply(x, u, v, args.rank, operator)
        x = [math.tanh(a + b) for a, b in zip(x, y)]
    elapsed = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
    print(json.dumps({
        'status': 'mechanical_toy_not_language_model',
        'date': '2026-10-19', 'model': 'shared_low_rank_operators',
        'quantization': 'none_float64', 'context': 'one_synthetic_vector',
        'backend': 'python_stdlib_' + platform.python_version(),
        'hardware': platform.machine(), 'platform': platform.platform(),
        'seed': args.seed, 'width': args.width, 'rank': args.rank,
        'steps': args.steps,
        'factor_payload_bytes': (len(u) + len(v)) * u.itemsize,
        'hypothetical_independent_dense_bytes': args.steps * args.width ** 2 * 8,
        'process_peak_rss_bytes': rss_bytes,
        'setup_and_execution_seconds': elapsed, 'output_checksum': sum(x),
        'quality': None, 'energy_joules': None,
        'warning': 'Logical matrix entries are correlated, not independent parameters.'
    }, indent=2))


if __name__ == '__main__':
    main()
