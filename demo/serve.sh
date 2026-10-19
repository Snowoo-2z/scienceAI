#!/usr/bin/env bash
# serve.sh — démarre la démo (compétences vérifiées + modèle entraîné).
#
# Répare l'environnement si besoin : crée .venv et y installe numpy
# (nécessaire uniquement au panneau « Modèle entraîné »).
#
#   ./demo/serve.sh            # port 8000
#   PORT=8080 ./demo/serve.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
VENV="$ROOT/.venv"

if [ ! -x "$VENV/bin/python" ]; then
  echo "[serve] création de l'environnement ($VENV)"
  python3 -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c "import numpy" >/dev/null 2>&1; then
  echo "[serve] installation de numpy"
  "$VENV/bin/pip" install --quiet --disable-pip-version-check numpy
fi

if [ ! -f "$ROOT/models/tinygpt.npz" ]; then
  echo "[serve] ATTENTION : models/tinygpt.npz absent — le panneau modèle sera désactivé"
  echo "[serve]            pour l'entraîner : .venv/bin/python demo/tinygpt.py train"
fi

export PORT
exec "$VENV/bin/python" demo/server.py
