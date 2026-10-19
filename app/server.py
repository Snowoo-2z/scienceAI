#!/usr/bin/env python3
"""Petite application web pour NanoFR.

Elle sert une page (continuer une phrase) et relaie les requêtes vers
llama-server, qui exécute le modèle sur le même mini-PC.

    python3 app/server.py --port 8000 --backend http://127.0.0.1:8081
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = "http://127.0.0.1:8081"


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            body = json.dumps({"ok": True, "rss_mb": round(rss_mb(), 1)}).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/complete":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"json"}', "application/json")
            return

        prompt = str(req.get("prompt", ""))[:600]
        n_predict = int(req.get("n_predict", 70))
        payload = json.dumps({
            "prompt": prompt,
            "n_predict": max(8, min(n_predict, 160)),
            "temperature": float(req.get("temperature", 0.8)),
            "top_k": 40,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "stop": ["\n\n"],
            "stream": False,
        }).encode()
        try:
            r = urllib.request.Request(BACKEND + "/completion", data=payload,
                                       headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=120) as resp:
                data = json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001
            self._send(502, json.dumps({"error": str(exc)}).encode(), "application/json")
            return

        out = {
            "text": data.get("content", ""),
            "tokens": data.get("tokens_predicted", 0),
            "ms": round(data.get("timings", {}).get("predicted_ms", 0.0), 1),
            "tps": round(data.get("timings", {}).get("predicted_per_second", 0.0), 1),
        }
        self._send(200, json.dumps(out).encode(), "application/json")

    def log_message(self, fmt: str, *args) -> None:  # silence
        pass


def main() -> None:
    global BACKEND
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", default=BACKEND)
    args = ap.parse_args()
    BACKEND = args.backend
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"http://0.0.0.0:{args.port}  ->  {BACKEND}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
