#!/usr/bin/env bash
# Lokalny gatekeeping — odpowiednik GitHub Actions gatekeeper.yml
# Uruchom: bash scripts/run_checks.sh
# Exit 1 przy pierwszym błędzie.

set -euo pipefail

echo "==> [1/4] Ruff — linter"
ruff check .

echo "==> [2/4] Ruff — formatter"
ruff format --check .

echo "==> [3/4] Mypy — typowanie statyczne"
mypy src/

echo "==> [4/4] Pytest — testy i coverage"
# Exit code 5 = brak testów do uruchomienia (akceptowalne przed Epikiem 2).
# Każdy inny niezerowy kod = błąd testu lub coverage.
pytest || { code=$?; [ "$code" -eq 5 ] || exit "$code"; }

echo ""
echo "Gatekeeping: OK. Wszystkie etapy przeszły pomyślnie."
