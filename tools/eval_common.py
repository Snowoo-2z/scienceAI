#!/usr/bin/env python3
"""eval_common.py — évalue demo_assistant sur demo/eval-questions.json.

  python3 tools/eval_common.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import demo_assistant as eng  # noqa: E402


def norm(s: str) -> str:
    s = s.lower().replace("\u00a0", " ").replace("\u202f", " ")
    s = re.sub(r"(?<=\d)[ \u00a0](?=\d)", "", s)  # 3 646 -> 3646
    s = s.replace(",", ".")
    return s


def main() -> None:
    docs, vecs, idf = eng.build_index()
    data = json.loads((ROOT / "demo" / "eval-questions.json").read_text(encoding="utf-8"))
    ok = 0
    t0 = time.perf_counter()
    lines = []
    for item in data["questions"]:
        top = eng.rerank(item["q"], docs, vecs, idf)
        skill = eng.skill_answer(item["q"], top)
        if item["type"] == "answerable":
            blob = norm((skill[0] + " " + skill[3]) if skill else "")
            good = skill is not None and any(norm(x) in blob for x in item["expect"])
        else:
            good = skill is None
        ok += good
        lines.append(f"{'OK  ' if good else 'FAIL'} {item['id']} {item['type']:9s} {item['q']}")
    dt = (time.perf_counter() - t0) * 1000
    report = "\n".join(lines) + f"\n\n{ok}/{len(data['questions'])} = {100*ok//len(data['questions'])} %" \
             f"   ({dt:.0f} ms, {dt/len(data['questions']):.1f} ms/question)\n"
    print(report)
    (ROOT / "results" / "eval-common.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
