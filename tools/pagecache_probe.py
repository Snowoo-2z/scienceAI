#!/usr/bin/env python3
"""pagecache_probe.py — mesurer le page cache invisible dans un budget RAM.

Question : quand on lit une bibliothèque de 2 Go, combien de RAM le noyau
consacre-t-il au page cache, alors que le RSS du processus reste petit ?
Et que coûte `posix_fadvise(DONTNEED)` pour rendre ces pages au système ?

Usage :
  python3 tools/pagecache_probe.py --file /tmp/wsprobe2.bin --mode buffered
  python3 tools/pagecache_probe.py --file /tmp/wsprobe2.bin --mode dontneed
"""
from __future__ import annotations

import argparse
import os
import time

CGROUP = "/sys/fs/cgroup/user"


def read_int(path: str) -> int:
    with open(path) as f:
        return int(f.read().strip())


def mem() -> tuple[int, int, int]:
    """(memory.current, VmRSS, file) en octets."""
    cur = read_int(f"{CGROUP}/memory.current")
    rss = 0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
                break
    file_bytes = 0
    with open(f"{CGROUP}/memory.stat") as f:
        for line in f:
            if line.startswith("file "):
                file_bytes = int(line.split()[1])
                break
    return cur, rss, file_bytes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--mode", choices=["buffered", "dontneed"], default="buffered")
    p.add_argument("--chunk-mb", type=int, default=8)
    args = p.parse_args()

    fd = os.open(args.file, os.O_RDONLY)
    size = os.fstat(fd).st_size
    chunk = args.chunk_mb << 20
    buf = bytearray(chunk)
    before = mem()
    t0 = time.monotonic()
    done = 0
    next_report = 256 << 20
    print(f"# mode={args.mode} file={args.file} size={size/2**30:.2f} GiB chunk={args.chunk_mb} MiB")
    print(f"# start memory.current={before[0]/2**20:.1f} MiB VmRSS={before[1]/2**20:.1f} MiB file={before[2]/2**20:.1f} MiB")
    while done < size:
        data = os.pread(fd, chunk, done)
        n = len(data)
        if n <= 0:
            break
        done += n
        if args.mode == "dontneed":
            os.posix_fadvise(fd, done - n, n, os.POSIX_FADV_DONTNEED)
        if done >= next_report or done == size:
            cur, rss, fileb = mem()
            dt = time.monotonic() - t0
            print(f"read {done/2**20:7.0f} MiB  {done/2**20/dt:7.1f} MiB/s  "
                  f"memory.current={cur/2**20:7.1f}  VmRSS={rss/2**20:6.1f}  file={fileb/2**20:7.1f} MiB")
            next_report += 256 << 20
    dt = time.monotonic() - t0
    after = mem()
    print(f"# end   memory.current={after[0]/2**20:.1f} MiB VmRSS={after[1]/2**20:.1f} MiB file={after[2]/2**20:.1f} MiB")
    print(f"# total {size/2**20:.0f} MiB in {dt:.2f} s = {size/2**20/dt:.1f} MiB/s")
    os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
    os.close(fd)


if __name__ == "__main__":
    main()
