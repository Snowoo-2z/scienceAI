#!/usr/bin/env python3
"""
plot_loss.py — courbe d'apprentissage en SVG autonome (stdlib).

Lit results/tinygpt-train.json et écrit figures/tinygpt-courbe.svg.

Auteur : KAIROS · 2026-10-19.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def polyline(points, x0, x1, y0, y1, lo, hi):
    out = []
    for step, val in points:
        x = x0 + (x1 - x0) * (step / max(1, points[-1][0]))
        y = y1 - (y1 - y0) * ((val - lo) / max(1e-9, hi - lo))
        out.append(f"{x:.1f},{y:.1f}")
    return " ".join(out)


def main() -> int:
    src = ROOT / "results" / "tinygpt-train.json"
    out = ROOT / "figures" / "tinygpt-courbe.svg"
    d = json.loads(src.read_text(encoding="utf-8"))
    # la dernière entrée est une évaluation finale sans perte d'entraînement
    log = [(int(s), float(t), float(v)) for s, t, v in d["log"] if t > 0]
    cfg = d["config"]
    width, height = 880, 420
    x0, x1, y0, y1 = 80, 700, 60, 340

    vals = [t for _, t, _ in log] + [v for _, _, v in log]
    lo = 0.0
    hi = max(vals) * 1.05
    train_pts = [(s, t) for s, t, _ in log]
    val_pts = [(s, v) for s, _, v in log]

    ticks_y = 6
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif,system-ui,sans-serif">',
             f'<rect width="{width}" height="{height}" fill="#fff"/>',
             f'<text x="24" y="28" font-size="17" font-weight="700">'
             f'Courbe d\'apprentissage — GPT {cfg.get("layers")} couches, '
             f'{d["params"]/1e6:.2f} M paramètres</text>',
             f'<text x="24" y="46" font-size="12" fill="#555">'
             f'{d["tokens"]:,} tokens de français · vocabulaire {d["vocab"]:,} · '
             f'perte par token (entropie croisée)</text>'.replace(",", " ")]

    for i in range(ticks_y + 1):
        val = lo + (hi - lo) * i / ticks_y
        y = y1 - (y1 - y0) * i / ticks_y
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" font-size="10.5" fill="#666" '
                     f'text-anchor="end">{val:.1f}</text>')
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#999"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#999"/>')

    parts.append(f'<polyline fill="none" stroke="#58a6ff" stroke-width="2.2" '
                 f'points="{polyline(train_pts, x0, x1, y0, y1, lo, hi)}"/>')
    parts.append(f'<polyline fill="none" stroke="#f0883e" stroke-width="2.2" '
                 f'points="{polyline(val_pts, x0, x1, y0, y1, lo, hi)}"/>')

    last = log[-1]
    for label, pts, color in (("perte d'entraînement", train_pts, "#58a6ff"),
                              ("perte de validation", val_pts, "#f0883e")):
        x = x0 + (x1 - x0) * (pts[-1][0] / max(1, pts[-1][0]))
        y = y1 - (y1 - y0) * ((pts[-1][1] - lo) / max(1e-9, hi - lo))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    parts.append(f'<g font-size="12">'
                 f'<rect x="720" y="70" width="12" height="3" fill="#58a6ff"/>'
                 f'<text x="740" y="75" fill="#333">train</text>'
                 f'<rect x="720" y="92" width="12" height="3" fill="#f0883e"/>'
                 f'<text x="740" y="97" fill="#333">validation</text></g>')

    steps = log[-1][0]
    parts.append(f'<text x="{x0}" y="{y1+26}" font-size="11" fill="#666">0</text>')
    parts.append(f'<text x="{x1}" y="{y1+26}" font-size="11" fill="#666" '
                 f'text-anchor="end">{steps} pas</text>')
    parts.append(f'<text x="{x0}" y="{y1+46}" font-size="12" fill="#333">'
                 f'dernière perte train {last[1]:.3f} · validation {last[2]:.3f} · '
                 f'perplexité {math.exp(min(last[2], 20)):.1f}</text>')
    parts.append('</svg>')

    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"écrit : {out} ({out.stat().st_size} octets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
