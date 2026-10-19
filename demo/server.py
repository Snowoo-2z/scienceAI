#!/usr/bin/env python3
"""server.py — interface de chat pour la démo « mode compétences ».

Sert une page HTML et une route /ask. La réponse vient de tools/demo_assistant.py
(retrieval TF-IDF + skills vérifiées) : aucun modèle de langue, aucun réseau.

  python3 demo/server.py --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "tools"))
import demo_assistant as eng  # noqa: E402

DOCS, VECTORS, IDF = eng.build_index()

PAGE = (ROOT / "index.html").read_text(encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/stats":
            stats = {
                "passages": len(DOCS),
                "fichiers": len({d[0] for d in DOCS}),
                "skills": 12,
            }
            self._send(200, json.dumps(stats).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/ask":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n).decode("utf-8", "replace")
        try:
            payload = json.loads(raw or "{}")
            q = str(payload.get("q", "")).strip()
        except json.JSONDecodeError:
            q = parse_qs(raw).get("q", [""])[0].strip()
        if not q:
            self._send(400, b'{"error":"question vide"}', "application/json")
            return
        top = eng.rerank(q, DOCS, VECTORS, IDF)
        skill = eng.skill_answer(q, top)
        if skill:
            ans, name, src, proof = skill
            body = {"type": "skill", "skill": name, "answer": ans, "source": src, "proof": proof}
        else:
            best = top[0] if top else (0.0, "?", "", False)
            body = {
                "type": "fallback",
                "answer": "Je n'ai pas de compétence vérifiée pour cette question. "
                          "Voici le passage le plus proche du corpus, sans garantie.",
                "source": best[1],
                "proof": best[2],
                "score": round(best[0], 3),
            }
        self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json")

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}  ({len(DOCS)} passages indexés)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
