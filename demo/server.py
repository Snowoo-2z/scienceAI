#!/usr/bin/env python3
"""
server.py — interface web du tri-système (stdlib uniquement).

Lancement :
    python3 demo/server.py            # http://localhost:8000
    PORT=8080 python3 demo/server.py

Aucune dépendance, aucun asset externe : la page est autonome et interroge
la même origine (`/api/ask`), donc elle fonctionne derrière un proxy.

Auteur : KAIROS · 2026-10-19.
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tri_system import TriSystem  # noqa: E402

TS = TriSystem()
STARTED = time.time()

PAGE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tri-système — Programme II</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --line:#30363d; --txt:#e6edf3;
    --dim:#8b949e; --accent:#58a6ff; --ok:#3fb950; --warn:#d29922; --bad:#f85149;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { padding:22px 24px 14px; border-bottom:1px solid var(--line); }
  h1 { margin:0 0 6px; font-size:19px; font-weight:700; letter-spacing:-.01em; }
  .sub { color:var(--dim); font-size:13px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px;
           background:#12261a; color:var(--ok); border:1px solid #1f4429;
           font-size:12px; margin-right:6px; }
  main { max-width:880px; margin:0 auto; padding:20px 16px 96px; }
  .examples { display:flex; flex-wrap:wrap; gap:8px; margin:4px 0 20px; }
  .chip { background:var(--panel); border:1px solid var(--line); color:var(--dim);
          padding:6px 11px; border-radius:999px; font-size:12.5px; cursor:pointer; }
  .chip:hover { border-color:var(--accent); color:var(--txt); }
  .turn { margin:0 0 18px; }
  .q { font-weight:600; margin:0 0 8px; }
  .q::before { content:"vous · "; color:var(--dim); font-weight:400; font-size:12px; }
  .a { background:var(--panel); border:1px solid var(--line); border-radius:10px;
       padding:12px 14px; white-space:pre-wrap; }
  .meta { margin-top:8px; display:flex; flex-wrap:wrap; gap:6px; font-size:11.5px;
          color:var(--dim); }
  .tag { padding:2px 7px; border-radius:6px; border:1px solid var(--line); }
  .t-competence { color:var(--ok); border-color:#1f4429; background:#0f1f14; }
  .t-memoire { color:var(--accent); border-color:#1f3040; background:#0f1a24; }
  .t-aveu { color:var(--warn); border-color:#3d2f0c; background:#1f1a0a; }
  .proof { color:var(--dim); font-size:11.5px; margin-top:6px;
           font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  form { position:fixed; bottom:0; left:0; right:0; background:rgba(13,17,23,.94);
         border-top:1px solid var(--line); padding:12px 16px; backdrop-filter:blur(6px); }
  .bar { max-width:880px; margin:0 auto; display:flex; gap:10px; }
  input[type=text] { flex:1; background:var(--panel); border:1px solid var(--line);
                     color:var(--txt); border-radius:8px; padding:12px 14px; font-size:15px; }
  input[type=text]:focus { outline:none; border-color:var(--accent); }
  button { background:var(--accent); color:#06121f; border:0; border-radius:8px;
           padding:0 20px; font-weight:700; font-size:14px; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  footer { color:var(--dim); font-size:12px; text-align:center; padding:6px 16px 14px; }
</style>
</head>
<body>
<header>
  <h1>Tri-système — le Programme II, en miniature</h1>
  <div class="sub">
    <span class="badge" id="b-skills">…</span>
    <span class="badge" id="b-tests">…</span>
    Pas un transformer : un routeur, une bibliothèque de compétences vérifiées,
    une mémoire documentaire, et l'aveu quand il ne sait pas.
  </div>
</header>
<main>
  <div class="examples" id="examples"></div>
  <div id="turn"></div>
</main>
<form id="f">
  <div class="bar">
    <input id="q" type="text" autocomplete="off"
           placeholder="Posez une question : calcul, conversion, date, capitale, ou sur le projet…">
    <button id="send" type="submit">Demander</button>
  </div>
</form>
<footer id="foot">Chargement des métriques…</footer>
<script>
const EXAMPLES = [
  "Combien font 17 × 23 ?",
  "Combien font 15 % de 240 ?",
  "Convertir 10 km en miles",
  "Quel jour de la semaine le 14 juillet 2026 ?",
  "Quelle est la capitale de l'Australie ?",
  "1987 en chiffres romains",
  "Quel est le budget mémoire du Programme II ?",
  "Qui est Silex ?",
  "Quelle est la recette du gratin dauphinois ?"
];
const ex = document.getElementById('examples');
EXAMPLES.forEach(t => {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'chip'; b.textContent = t;
  b.onclick = () => { document.getElementById('q').value = t; ask(t); };
  ex.appendChild(b);
});
const turn = document.getElementById('turn');
let busy = false;

async function ask(q) {
  if (busy || !q.trim()) return;
  busy = true; document.getElementById('send').disabled = true;
  const div = document.createElement('div'); div.className = 'turn';
  const qd = document.createElement('div'); qd.className = 'q'; qd.textContent = q;
  const ad = document.createElement('div'); ad.className = 'a'; ad.textContent = '…';
  div.appendChild(qd); div.appendChild(ad); turn.appendChild(div);
  ad.scrollIntoView({behavior:'smooth', block:'center'});
  try {
    const r = await fetch('api/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({q})
    });
    const d = await r.json();
    ad.textContent = d.text;
    const meta = document.createElement('div'); meta.className = 'meta';
    const labels = {competence:'compétence vérifiée', memoire:'mémoire documentaire',
                    aveu:'aveu'};
    const cls = {competence:'t-competence', memoire:'t-memoire', aveu:'t-aveu'};
    meta.innerHTML =
      `<span class="tag ${cls[d.kind]}">${labels[d.kind]}</span>` +
      `<span class="tag">${d.source}</span>` +
      `<span class="tag">${d.ms.toFixed(2)} ms</span>` +
      `<span class="tag">${d.energy_mj.toFixed(4)} mJ</span>` +
      `<span class="tag">vérifié : ${d.verified ? 'oui' : 'non'}</span>`;
    ad.appendChild(meta);
    if (d.proof) {
      const p = document.createElement('div'); p.className = 'proof';
      p.textContent = 'preuve : ' + d.proof; ad.appendChild(p);
    }
    refresh();
  } catch (e) {
    ad.textContent = "Erreur réseau : " + e;
  }
  busy = false; document.getElementById('send').disabled = false;
}

document.getElementById('f').onsubmit = (ev) => {
  ev.preventDefault();
  const inp = document.getElementById('q');
  const q = inp.value; inp.value = '';
  ask(q);
};

async function refresh() {
  try {
    const d = await (await fetch('api/metrics')).json();
    document.getElementById('b-skills').textContent =
      d.skills + '/' + d.skills_total + ' compétences actives';
    document.getElementById('b-tests').textContent =
      d.tests_passed + '/' + d.tests + ' tests internes';
    document.getElementById('foot').textContent =
      'RSS ' + d.rss_mb + ' Mo · ' + d.chunks + ' paragraphes indexés · ' +
      d.counters.competence + ' compétences, ' + d.counters.memoire +
      ' mémoires, ' + d.counters.aveu + ' aveux · up ' +
      d.uptime_s.toFixed(0) + ' s · hypothèse énergie : 15 W de plateforme';
  } catch (e) { /* silencieux */ }
}
refresh(); setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "TriSystem/1.0"

    def log_message(self, fmt, *args):  # silence
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/metrics":
            m = TS.metrics()
            m["uptime_s"] = time.time() - STARTED
            self._send(200, json.dumps(m, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/health":
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0] != "/api/ask":
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            q = str(payload.get("q", ""))[:1000]
        except Exception:  # noqa: BLE001
            self._send(400, b'{"error":"bad request"}', "application/json")
            return
        ans = TS.answer(q)
        body = json.dumps(ans.as_dict(), ensure_ascii=False).encode("utf-8")
        self._send(200, body, "application/json; charset=utf-8")


def main() -> int:
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    m = TS.metrics()
    print(f"tri-système : {m['skills']}/{m['skills_total']} compétences, "
          f"{m['tests_passed']}/{m['tests']} tests, RSS {m['rss_mb']} Mo")
    print(f"écoute sur http://0.0.0.0:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
