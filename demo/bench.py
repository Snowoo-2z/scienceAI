#!/usr/bin/env python3
"""
bench.py — évaluation du tri-système sur des questions tenues à l'écart.

Les questions de `eval_questions.json` ne sont PAS celles des tests internes
des compétences : le système n'a jamais vu ces formulations. On mesure donc
trois choses :

  1. la justesse des réponses de type « compétence » (sous-chaîne attendue) ;
  2. le taux d'aveu correct sur les questions hors domaine ;
  3. le coût : latence et énergie estimée par réponse.

Sortie : results/tri-system-eval.txt (+ JSON).

Auteur : KAIROS · 2026-10-19 · stdlib pure.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tri_system import TriSystem, norm  # noqa: E402


def run(verbose: bool = True):
    ts = TriSystem()
    items = json.loads((HERE / "eval_questions.json").read_text(encoding="utf-8"))
    rows = []
    t0 = time.perf_counter()
    for it in items:
        a = ts.answer(it["q"])
        ok = None
        if it["kind"] == "any":
            ok = None
        elif it["kind"] == "aveu":
            ok = (a.kind == "aveu")
        elif it["kind"] == "competence":
            ok = (a.kind == "competence" and norm(it["expect"]) in norm(a.text))
        elif it["kind"] == "memoire":
            ok = (a.kind == "memoire" and norm(it["expect"]) in norm(a.text))
        rows.append({"q": it["q"], "expected_kind": it["kind"], "expect": it["expect"],
                     "kind": a.kind, "text": a.text, "proof": a.proof,
                     "ms": a.ms, "mj": a.energy_mj, "ok": ok,
                     "source": a.source})
        if verbose:
            mark = {True: "OK ", False: "ÉCHEC", None: "—"}[ok]
            print(f"{mark:5} [{a.kind:10}] {it['q']}")
            print(f"        -> {a.text[:110]}")
            if ok is False:
                print(f"        attendu : {it['kind']} / {it['expect']!r}")
    wall = time.perf_counter() - t0

    comp = [r for r in rows if r["ok"] is True]
    scored = [r for r in rows if r["ok"] is not None]
    ood = [r for r in rows if r["expected_kind"] == "aveu"]
    inb = [r for r in rows if r["expected_kind"] in ("competence", "memoire")]

    summary = {
        "questions": len(rows),
        "justes": len(comp),
        "scorees": len(scored),
        "exactitude": len(comp) / len(scored) if scored else 0.0,
        "in_domain_ok": sum(1 for r in inb if r["ok"]),
        "in_domain": len(inb),
        "ood_aveu_ok": sum(1 for r in ood if r["ok"]),
        "ood": len(ood),
        "ms_median": statistics.median(r["ms"] for r in rows),
        "ms_p95": sorted(r["ms"] for r in rows)[max(0, int(0.95 * len(rows)) - 1)],
        "mj_total": sum(r["mj"] for r in rows),
        "mj_median": statistics.median(r["mj"] for r in rows),
        "wall_s": wall,
        "metrics": ts.metrics(),
        "rows": rows,
    }
    return summary


def main() -> int:
    s = run()
    out = HERE.parent / "results" / "tri-system-eval.txt"
    out.parent.mkdir(exist_ok=True)
    lines = []
    lines.append("=" * 78)
    lines.append("  TRI-SYSTÈME — ÉVALUATION SUR QUESTIONS TENUES À L'ÉCART")
    lines.append("=" * 78)
    lines.append(f"  compétences vérifiées ... {s['metrics']['skills']}/"
                 f"{s['metrics']['skills_total']} "
                 f"({s['metrics']['tests_passed']}/{s['metrics']['tests']} tests internes)")
    lines.append(f"  questions ............... {s['questions']}")
    lines.append(f"  exactitude .............. {100 * s['exactitude']:.1f} % "
                 f"({s['justes']}/{s['scorees']})")
    lines.append(f"  dans le domaine ......... {s['in_domain_ok']}/{s['in_domain']}")
    lines.append(f"  hors domaine (aveu) ..... {s['ood_aveu_ok']}/{s['ood']}")
    lines.append(f"  latence médiane ......... {s['ms_median']:.2f} ms")
    lines.append(f"  latence p95 ............. {s['ms_p95']:.2f} ms")
    lines.append(f"  énergie estimée ......... {s['mj_total']:.2f} mJ pour "
                 f"{s['questions']} questions")
    lines.append(f"  RSS ..................... {s['metrics']['rss_mb']} Mo")
    lines.append("")
    lines.append("  Détail :")
    for r in s["rows"]:
        mark = "OK " if r["ok"] else ("ÉCHEC" if r["ok"] is False else "—")
        lines.append(f"  {mark:5} [{r['kind']:10}] {r['q']}")
        lines.append(f"        -> {r['text'][:110]}")
    txt = "\n".join(lines) + "\n"
    out.write_text(txt, encoding="utf-8")
    print(txt)
    (out.parent / "tri-system-eval.json").write_text(
        json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print(f"écrit : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
