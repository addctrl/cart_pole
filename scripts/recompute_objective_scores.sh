#!/usr/bin/env bash
# Przelicza objective_score dla wskazanych CSV.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
penalty="${1:-0.1}"
shift $(( $# > 0 ? 1 : 0 )) || true

cd "$repo_root"
source .venv/bin/activate

if (( $# > 0 )); then
  python -m src.objective_score_csv --csv-paths "$@" --stability-penalty "$penalty" --in-place
else
  python -m src.objective_score_csv \
    --csv-paths data/experiments.csv data/lunarlander_experiments.csv data/humanoid_bayes_results.csv \
    --stability-penalty "$penalty" \
    --in-place
fi
