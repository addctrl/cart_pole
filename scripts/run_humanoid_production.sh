#!/usr/bin/env bash
# Wrapper uruchamiający produkcyjny trening Humanoida 30M kroków.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

cd "$repo_root"
source .venv/bin/activate
mkdir -p logs

log_file="logs/humanoid_production_30m_$(date +%Y%m%d_%H%M%S).log"
echo "[RUN] Humanoid production 30M start"
echo "[RUN] log => $log_file"

PYTHONUNBUFFERED=1 python -u -m src.humanoid_production | tee -a "$log_file"
