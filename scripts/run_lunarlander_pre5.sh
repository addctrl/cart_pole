#!/usr/bin/env bash
# Wrapper do eksperymentu pre5 dla LunarLander-v3.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

cd "$repo_root"
source .venv/bin/activate
python -m src.lunarlander_bayes --trials 40 --timesteps 300000 --startup-trials 8 --pruner-warmup-steps 50000 --report-interval-timesteps 50000 --eval-episodes 20 --stability-penalty 0.1 --results-csv data/lunarlander_bayes_results.csv