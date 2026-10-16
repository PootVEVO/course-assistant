#!/bin/bash
# One-click live run for macOS/Linux (double-click in Finder, or: bash scripts/run_live.command)
# Installs into a virtualenv OUTSIDE the repo (~/.venvs/course-assistant), then runs:
#   offline tests -> live service tests -> reranker on/off evaluation -> README screenshots
# Everything is logged to results/live_run_log.txt
cd "$(dirname "$0")/.." || exit 1
mkdir -p results
exec > >(tee results/live_run_log.txt) 2>&1
set -x
PY=""
for c in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then PY=$(command -v "$c"); break; fi
done
if [ -z "$PY" ]; then echo "ERROR: need Python 3.10+ (brew install python@3.12 or python.org)"; read -p "Press Enter to close"; exit 1; fi
"$PY" --version
VENV="$HOME/.venvs/course-assistant"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r requirements-dev.txt
"$VENV/bin/python" -m playwright install chromium
[ -f .env ] || { echo "ERROR: create .env from .env.example first"; read -p "Press Enter to close"; exit 1; }
curl -s -m 10 -o /dev/null -w "class server reachable: HTTP %{http_code}\n" http://dobolyi.com:9001/v1/models
"$VENV/bin/python" -m pytest 2>&1 | tail -3
"$VENV/bin/python" -m pytest -m live -v 2>&1 | tee results/live_tests.txt | tail -15
"$VENV/bin/python" -m eval.run_eval --materials course_files
"$VENV/bin/python" scripts/capture_screenshots.py --data-dir eval/.data
set +x
echo "=== LIVE RUN FINISHED ==="
read -p "Done - press Enter to close this window"
