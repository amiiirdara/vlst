#!/usr/bin/env bash
# Create repo .venv, install modeling deps + ipykernel, register a Jupyter kernel "vlst".
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f requirements.txt ]]; then
  echo "requirements.txt not found in $REPO_ROOT" >&2
  exit 1
fi

# Cursor’s integrated terminal can put a python3 on PATH whose sys.executable is the
# AppImage; venv/pip then break. Prefer the distro interpreter unless PYTHON is set.
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x /usr/bin/python3 ]]; then
  PY=/usr/bin/python3
else
  PY=python3
fi
_pyexe="$("$PY" -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
if [[ "$_pyexe" == *.AppImage ]]; then
  echo "error: $PY runs as $_pyexe — not suitable for venv." >&2
  echo "Run:  PYTHON=/usr/bin/python3 bash scripts/setup_notebook_venv.sh" >&2
  exit 1
fi

if [[ -f .venv/pyvenv.cfg ]] && grep -q '\.AppImage' .venv/pyvenv.cfg; then
  echo "error: .venv was created with an AppImage/Cursor Python. Fix: rm -rf .venv  then re-run this script." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name=vlst --display-name="Python (vlst .venv)"

echo
echo "Done. In Cursor/VS Code: Select Kernel → Python (vlst .venv), or Python: Select Interpreter → .venv/bin/python"
