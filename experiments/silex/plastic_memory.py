"""S3a: synthetic online associative recall. No language training or novelty claim."""
import argparse
import json
import platform
import resource
import sys
import time
import numpy as np


class Memory:
    def __init__(self, method, width, classes, beta=1.0):
        self.method = method
        self.beta = beta
        self.classes = classes
        self.cursor = 0
        if method == 'fifo_q8':
            self.slots = width * classes * 4 // (width + 1)
            self.keys = np.zeros((self.slots, width), dtype=np.int8)
            self.values = np.zeros(self.slots, dtype=np.uint8)
        elif method == 'fifo':
            self.slots = width * classes // (width + classes)
            self.keys = np.zeros((self.slots, width), dtype=np.float32)
            self.values = np.zeros((self.slots, classes), dtype=np.float32)
        else:
            self.matrix = np.zeros((width, classes), dtype=np.float32)

    @property
    def payload(self):
        if self.method.startswith('fifo'):
            return self.keys.nbytes + self.values.nbytes
        return self.matrix.nbytes

    def read(self, key):
        if self.method == 'fifo_q8':
            scores = (self.keys @ key) / np.maximum(np.linalg.norm(self.keys, axis=1), 1e-12)
            label = self.values[np.argmax(scores)]
            return np.eye(self.classes, dtype=np.float32)[label]
        if self.method == 'fifo':
            return self.values[np.argmax(self.keys @ key)]
        return key @ self.matrix

    def write(self, key, value):
        if self.method.startswith('fifo'):
            if self.method == 'fifo_q8':
                key = np.rint(key * 127).astype(np.int8)
                value = np.argmax(value)
            # Exact matching for update; retrieval itself is nearest dot product.
            matches = np.flatnonzero(np.all(self.keys == key, axis=1))
            if len(matches):
                slot = matches[0]
            else:
                slot = self.cursor
                self.cursor = (self.cursor + 1) % self.slots
            self.keys[slot] = key
            self.values[slot] = value
        else:
            residual = value if self.method == 'hebb' else value - self.read(key)
            self.matrix += (self.beta if self.method == 'delta' else 1.0) * np.outer(key, residual)


def trial(method, count, seed, noise=0.0, width=64, classes=16, beta=1.0):
    rng = np.random.default_rng(seed)
    keys = rng.normal(size=(count, width)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    labels = rng.integers(classes, size=count)
    eye = np.eye(classes, dtype=np.float32)
    memory = Memory(method, width, classes, beta)
    start = time.perf_counter()
    for key, label in zip(keys, labels):
        memory.write(key, eye[label])
    query = keys + noise * rng.normal(size=keys.shape).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)

    def correct():
        return np.array([np.argmax(memory.read(k)) == label
                         for k, label in zip(query, labels)])

    before = correct()
    # Update a random half, not merely the newest or oldest items.
    changed = rng.permutation(count)[:count // 2]
    labels[changed] = (labels[changed] + rng.integers(1, classes, len(changed))) % classes
    for index in changed:
        memory.write(keys[index], eye[labels[index]])
    after = correct()
    unchanged = np.ones(count, dtype=bool)
    unchanged[changed] = False
    return dict(method=method, count=count, seed=seed, noise=noise,
                payload_bytes=memory.payload, beta=beta, before=float(before.mean()),
                after=float(after.mean()), changed=float(after[changed].mean()),
                unchanged=float(after[unchanged].mean()),
                seconds=time.perf_counter() - start)


def self_test():
    eye = np.eye(4, dtype=np.float32)
    for method in ('delta', 'hebb'):
        m = Memory(method, 4, 4)
        for i in range(4):
            m.write(eye[i], eye[i])
        np.testing.assert_allclose(m.read(eye[2]), eye[2])
    m = Memory('delta', 4, 4)
    m.write(eye[0], eye[1])
    m.write(eye[0], eye[3])
    np.testing.assert_allclose(m.read(eye[0]), eye[3])
    m = Memory('fifo', 4, 4)
    assert m.slots == 2 and m.payload <= 4 * 4 * 4
    m.write(eye[0], eye[1])
    m.write(eye[0], eye[3])
    assert m.cursor == 1
    np.testing.assert_allclose(m.read(eye[0]), eye[3])
    m.write(eye[1], eye[2])
    m.write(eye[2], eye[0])
    assert not np.any(np.all(m.keys == eye[0], axis=1))
    assert trial('delta', 16, 42)['after'] == trial('delta', 16, 42)['after']
    m = Memory('fifo_q8', 64, 16)
    assert m.slots == 63 and m.payload == 4095
    key = np.eye(64, dtype=np.float32)[0]
    value = np.eye(16, dtype=np.float32)[7]
    m.write(key, value)
    np.testing.assert_array_equal(m.read(key), value)
    m.write(key, np.eye(16, dtype=np.float32)[9])
    assert m.cursor == 1 and np.argmax(m.read(key)) == 9
    print('orthogonal recall, overwrite, FIFO budget/update/eviction, reproducibility: PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not 0 < args.beta <= 1:
        parser.error('beta must be in (0, 1]')
    if args.noise < 0:
        parser.error('noise must be nonnegative')
    rows = [trial(method, n, seed, args.noise, beta=args.beta)
            for n in (16, 64, 128, 256)
            for seed in range(args.seed_start, args.seed_start + 5)
            for method in ('hebb', 'delta', 'fifo', 'fifo_q8')]
    summary = []
    for n in (16, 64, 128, 256):
        for method in ('hebb', 'delta', 'fifo', 'fifo_q8'):
            selected = [r for r in rows if r['count'] == n and r['method'] == method]
            item = dict(count=n, method=method, payload_bytes=selected[0]['payload_bytes'])
            for metric in ('before', 'after', 'changed', 'unchanged', 'seconds'):
                data = [r[metric] for r in selected]
                item[metric + '_mean'] = float(np.mean(data))
                item[metric + '_seed_std'] = float(np.std(data, ddof=1))
            summary.append(item)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(date='2026-10-19', model='synthetic_associative_memory',
                         quantization='none_float32', context='16_to_256_key_value_pairs',
                         backend='numpy_' + np.__version__, python=platform.python_version(),
                         hardware=platform.machine(), platform=platform.platform(),
                         width=64, classes=16, seeds=list(range(args.seed_start, args.seed_start + 5)), noise=args.noise, beta=args.beta,
                         process_peak_rss_bytes=rss if sys.platform == 'darwin' else rss * 1024,
                         energy_joules=None, language_quality=None, summary=summary, trials=rows), indent=2))


if __name__ == '__main__':
    main()
