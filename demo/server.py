#!/usr/bin/env python3
"""
server.py — interface web du projet (stdlib + numpy optionnel).

Deux systèmes sur la même page :

  * le tri-système : compétences vérifiées, mémoire documentaire, aveu ;
  * le modèle entraîné : un GPT de 4,2 M de paramètres, écrit à la main
    (demo/tinygpt.py) et entraîné ici sur 10 Mo de littérature française.

Lancement :
    python3 demo/server.py            # http://localhost:8000
    PORT=8080 python3 demo/server.py

Le second panneau nécessite numpy et un checkpoint `models/tinygpt.npz`.
S'ils manquent, la page fonctionne quand même et l'indique.

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
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from tri_system import TriSystem  # noqa: E402

TS = TriSystem()
STARTED = time.time()

# --- modèle de langage (optionnel) -----------------------------------------
MODEL = None
MODEL_ERR = ""
try:
    import numpy as np  # noqa: F401
    import tinygpt

    CKPT = ROOT / "models" / "tinygpt.npz"
    if CKPT.exists():
        P_M, TOK_M, META_M = tinygpt.load(str(CKPT))
        MODEL = {"P": P_M, "tok": TOK_M, "meta": META_M,
                 "params": tinygpt.n_params(P_M),
                 "size": CKPT.stat().st_size}
    else:
        MODEL_ERR = "checkpoint absent (lancer demo/tinygpt.py train)"
except Exception as e:  # noqa: BLE001
    MODEL_ERR = f"{type(e).__name__}: {e}"


PAGE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scienceAI — tri-système et modèle entraîné</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --txt:#e6edf3;
          --dim:#8b949e; --accent:#58a6ff; --ok:#3fb950; --warn:#d29922; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { padding:20px 24px 12px; border-bottom:1px solid var(--line); }
  h1 { margin:0 0 6px; font-size:19px; font-weight:700; letter-spacing:-.01em; }
  .sub { color:var(--dim); font-size:13px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px;
           background:#12261a; color:var(--ok); border:1px solid #1f4429;
           font-size:12px; margin-right:6px; }
  .tabs { display:flex; gap:4px; margin-top:14px; }
  .tab { padding:8px 16px; border-radius:8px 8px 0 0; border:1px solid transparent;
         color:var(--dim); cursor:pointer; font-size:14px; font-weight:600; }
  .tab.on { background:var(--panel); border-color:var(--line); color:var(--txt); }
  main { max-width:900px; margin:0 auto; padding:18px 16px 110px; }
  .panel { display:none; }
  .panel.on { display:block; }
  .examples { display:flex; flex-wrap:wrap; gap:8px; margin:2px 0 18px; }
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
  .bar { max-width:900px; margin:0 auto; display:flex; gap:10px; }
  input[type=text] { flex:1; background:var(--panel); border:1px solid var(--line);
                     color:var(--txt); border-radius:8px; padding:12px 14px; font-size:15px; }
  input[type=text]:focus { outline:none; border-color:var(--accent); }
  button.go { background:var(--accent); color:#06121f; border:0; border-radius:8px;
              padding:0 20px; font-weight:700; font-size:14px; cursor:pointer; }
  button.go:disabled { opacity:.5; cursor:default; }
  .opts { max-width:900px; margin:0 auto 8px; display:flex; gap:14px; align-items:center;
          color:var(--dim); font-size:12px; }
  .opts input[type=range] { width:110px; }
  footer { color:var(--dim); font-size:12px; text-align:center; padding:6px 16px 14px; }
  .gen { white-space:pre-wrap; background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:14px; min-height:70px; font-size:15.5px; }
  .note { color:var(--dim); font-size:12.5px; margin:10px 0 0; }
</style>
</head>
<body>
<header>
  <h1>scienceAI — deux façons de répondre</h1>
  <div class="sub">
    <span class="badge" id="b-skills">…</span>
    <span class="badge" id="b-model">…</span>
  </div>
  <div class="tabs">
    <div class="tab on" data-p="skills">Compétences vérifiées</div>
    <div class="tab" data-p="model">Modèle entraîné</div>
  </div>
</header>
<main>
  <div class="panel on" id="p-skills">
    <div class="examples" id="examples"></div>
    <div class="note">
      20 compétences exécutables, 46 tests passés au chargement, une mémoire
      documentaire sur les textes du projet, et l'aveu quand rien ne s'applique.
    </div>
    <div id="turn" style="margin-top:16px"></div>
  </div>
  <div class="panel" id="p-model">
    <div class="note" id="model-info" style="margin-bottom:12px">…</div>
    <div class="examples" id="gexamples"></div>
    <div class="gen" id="gen">La génération apparaîtra ici.</div>
    <div class="note">
      C'est un vrai transformer : 4 couches, 4 têtes d'attention causale,
      écrit et entraîné ici, sans bibliothèque de deep learning. Il a lu
      10 Mo de romans français du domaine public. Il ne « sait » rien :
      il prédit le mot suivant. Sa perte et ses limites sont dans
      <code>docs/11-LE-MODELE.md</code>.
    </div>
  </div>
</main>
<form id="f">
  <div class="opts" id="opts" style="display:none">
    <label>longueur <input id="n" type="range" min="40" max="240" value="120" step="10">
      <span id="nv">120</span></label>
    <label>température <input id="t" type="range" min="0.4" max="1.2" value="0.85" step="0.05">
      <span id="tv">0.85</span></label>
  </div>
  <div class="bar">
    <input id="q" type="text" autocomplete="off" placeholder="Posez une question…">
    <button id="send" class="go" type="submit">Envoyer</button>
  </div>
</form>
<footer id="foot">Chargement…</footer>
<script>
const EXAMPLES = [
  "Combien font 17 × 23 ?", "Combien font 15 % de 240 ?",
  "Convertir 10 km en miles", "Quel jour de la semaine le 14 juillet 2026 ?",
  "Quelle est la capitale de l'Australie ?", "1987 en chiffres romains",
  "Quel est le budget mémoire du Programme II ?", "Qui est Silex ?",
  "Quelle est la recette du gratin dauphinois ?"
];
const GEXAMPLES = ["Il était une fois", "Le comte de Monte-Cristo",
                   "Madame Bovary", "Dans la ville de Paris,", "Non, dit-il,"];
const ex = document.getElementById('examples');
EXAMPLES.forEach(t => {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'chip'; b.textContent = t;
  b.onclick = () => { document.getElementById('q').value = t; askSkills(t); };
  ex.appendChild(b);
});
const gex = document.getElementById('gexamples');
GEXAMPLES.forEach(t => {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'chip'; b.textContent = t;
  b.onclick = () => { document.getElementById('q').value = t; askModel(t); };
  gex.appendChild(b);
});
const turn = document.getElementById('turn');
let mode = 'skills', busy = false;

document.querySelectorAll('.tab').forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('on'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('on'));
    tab.classList.add('on');
    mode = tab.dataset.p;
    document.getElementById('p-' + mode).classList.add('on');
    document.getElementById('opts').style.display = mode === 'model' ? 'flex' : 'none';
    document.getElementById('q').placeholder = mode === 'model'
      ? "Écrivez un début de phrase…" : "Posez une question…";
  };
});

function addTurn(q) {
  const div = document.createElement('div'); div.className = 'turn';
  const qd = document.createElement('div'); qd.className = 'q'; qd.textContent = q;
  const ad = document.createElement('div'); ad.className = 'a'; ad.textContent = '…';
  div.appendChild(qd); div.appendChild(ad); turn.appendChild(div);
  ad.scrollIntoView({behavior:'smooth', block:'center'});
  return ad;
}

async function askSkills(q) {
  if (busy || !q.trim()) return;
  busy = true; document.getElementById('send').disabled = true;
  const ad = addTurn(q);
  try {
    const r = await fetch('api/ask', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({q})});
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
  } catch (e) { ad.textContent = "Erreur réseau : " + e; }
  busy = false; document.getElementById('send').disabled = false;
}

async function askModel(q) {
  if (busy || !q.trim()) return;
  busy = true; document.getElementById('send').disabled = true;
  const out = document.getElementById('gen');
  out.textContent = q + " ▌";
  try {
    const n = +document.getElementById('n').value;
    const t = +document.getElementById('t').value;
    const r = await fetch('api/generate', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({prompt: q, n: n, t: t, k: 40})});
    const d = await r.json();
    if (d.error) { out.textContent = "Le modèle n'est pas disponible : " + d.error; }
    else {
      out.textContent = d.text;
      const meta = document.createElement('div'); meta.className = 'meta';
      meta.innerHTML = `<span class="tag">${d.tokens} tokens</span>` +
        `<span class="tag">${d.ms.toFixed(0)} ms</span>` +
        `<span class="tag">${d.tok_s.toFixed(0)} tok/s</span>` +
        `<span class="tag">${(d.params/1e6).toFixed(2)} M paramètres</span>` +
        `<span class="tag">perte val ${d.val_loss.toFixed(3)}</span>`;
      out.appendChild(meta);
    }
  } catch (e) { out.textContent = "Erreur : " + e; }
  busy = false; document.getElementById('send').disabled = false;
}

document.getElementById('f').onsubmit = (ev) => {
  ev.preventDefault();
  const inp = document.getElementById('q');
  const q = inp.value; inp.value = '';
  (mode === 'model' ? askModel : askSkills)(q);
};
document.getElementById('n').oninput = e => document.getElementById('nv').textContent = e.target.value;
document.getElementById('t').oninput = e => document.getElementById('tv').textContent = e.target.value;

async function refresh() {
  try {
    const d = await (await fetch('api/metrics')).json();
    document.getElementById('b-skills').textContent =
      d.skills + '/' + d.skills_total + ' compétences · ' +
      d.tests_passed + '/' + d.tests + ' tests';
    let m = 'modèle indisponible';
    if (d.model && d.model.ok) {
      m = 'modèle ' + (d.model.params / 1e6).toFixed(2) + ' M params · ppl ' +
          d.model.ppl.toFixed(1);
    }
    document.getElementById('b-model').textContent = m;
    document.getElementById('foot').textContent =
      'tri-système : ' + d.skills + ' compétences, ' + d.chunks +
      ' paragraphes indexés, RSS ' + d.rss_mb + ' Mo · up ' +
      d.uptime_s.toFixed(0) + ' s';
    if (d.model && d.model.ok) {
      document.getElementById('model-info').textContent =
        'GPT ' + d.model.layers + ' couches · d=' + d.model.d + ' · vocab ' +
        d.model.vocab + ' · ' + (d.model.params / 1e6).toFixed(2) +
        ' M paramètres · ' + d.model.size_mb.toFixed(1) + ' Mo sur disque · ' +
        'entraîné ' + d.model.tokens_seen.toLocaleString('fr-FR') +
        ' tokens · perte val ' + d.model.val_loss.toFixed(3) +
        ' (perplexité ' + d.model.ppl.toFixed(1) + ')';
    } else if (d.model) {
      document.getElementById('model-info').textContent = 'Modèle indisponible : ' + d.model.error;
    }
  } catch (e) { /* silencieux */ }
}
refresh(); setInterval(refresh, 4000);
</script>
</body>
</html>
"""


def model_metrics() -> dict:
    if MODEL is None:
        return {"ok": False, "error": MODEL_ERR}
    meta = MODEL["meta"]
    return {
        "ok": True, "params": MODEL["params"], "size_mb": MODEL["size"] / 1e6,
        "d": meta.get("d"), "layers": meta.get("layers"), "heads": meta.get("heads"),
        "vocab": len(MODEL["tok"].itos), "step": meta.get("step"),
        "val_loss": meta.get("val", 0.0),
        "ppl": float(__import__("math").exp(min(meta.get("val", 0.0), 20))),
        "tokens_seen": meta.get("step", 0) * 24 * 128,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "scienceAI/2.0"

    def log_message(self, fmt, *args):
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
            m["model"] = model_metrics()
            self._send(200, json.dumps(m, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/health":
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception:  # noqa: BLE001
            self._send(400, b'{"error":"bad request"}', "application/json")
            return

        if path == "/api/ask":
            q = str(payload.get("q", ""))[:1000]
            ans = TS.answer(q)
            self._send(200, json.dumps(ans.as_dict(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return

        if path == "/api/generate":
            if MODEL is None:
                self._send(200, json.dumps({"error": MODEL_ERR},
                                           ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            prompt = str(payload.get("prompt", ""))[:400]
            nt = max(8, min(400, int(payload.get("n", 120))))
            temp = float(payload.get("t", 0.85))
            k = int(payload.get("k", 40))
            t0 = time.time()
            text = tinygpt.generate(MODEL["P"], MODEL["tok"], prompt,
                                    n_tokens=nt, temperature=temp, top_k=k)
            dt = time.time() - t0
            body = {"text": text, "tokens": nt, "ms": dt * 1000,
                    "tok_s": nt / dt if dt else 0,
                    "params": MODEL["params"],
                    "val_loss": MODEL["meta"].get("val", 0.0)}
            self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return

        self._send(404, b"not found", "text/plain")

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.end_headers()


def main() -> int:
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    m = TS.metrics()
    print(f"tri-système : {m['skills']}/{m['skills_total']} compétences, "
          f"{m['tests_passed']}/{m['tests']} tests, RSS {m['rss_mb']} Mo")
    if MODEL:
        print(f"modèle : {MODEL['params'] / 1e6:.2f} M paramètres, "
              f"perte val {MODEL['meta'].get('val'):.4f}")
    else:
        print(f"modèle indisponible : {MODEL_ERR}")
    print(f"écoute sur http://0.0.0.0:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
