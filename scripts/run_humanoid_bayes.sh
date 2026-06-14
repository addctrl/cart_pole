#!/usr/bin/env bash
# Wrapper do eksperymentu bayesowskiego Humanoid-v5.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

cd "$repo_root"
source .venv/bin/activate
mkdir -p logs
log_file="logs/humanoid_bayes_512x512_$(date +%Y%m%d_%H%M%S).log"

echo "[RUN] Humanoid bayes start"
echo "[RUN] logs => $log_file"

PYTHONUNBUFFERED=1 python -u -m src.humanoid_bayes \
	--trials 50 \
	--timesteps 1000000 \
	--startup-trials 5 \
	--pruner-warmup-steps 100000 \
	--report-interval-timesteps 100000 \
	--eval-episodes 20 \
	--stability-penalty 0.1 \
	--results-csv data/humanoid_bayes_results_512x512.csv \
	| tee -a "$log_file"