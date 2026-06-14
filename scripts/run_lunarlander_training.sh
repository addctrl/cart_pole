#!/usr/bin/env bash
# Wrapper do treningu LunarLander-v3 z macierzy CSV.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

cd "$repo_root"
source .venv/bin/activate
python -m src.training --csv data/lunarlander_experiments.csv