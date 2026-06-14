#!/usr/bin/env bash
# Eksportuje skalary TensorBoard z wielu runow do CSV.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
logdir="${1:-logs/tensorboard}"
output_csv="${2:-data/tensorboard_scalars_pivot.csv}"
shift $(( $# > 1 ? 2 : $# )) || true

cd "$repo_root"
source .venv/bin/activate

if (( $# > 0 )); then
  python -m src.tensorboard_export \
    --logdir "$logdir" \
    --output-csv "$output_csv" \
    --long-output-csv data/tensorboard_scalars_long.csv \
    --tags "$@"
else
  python -m src.tensorboard_export \
    --logdir "$logdir" \
    --output-csv "$output_csv" \
    --long-output-csv data/tensorboard_scalars_long.csv
fi
