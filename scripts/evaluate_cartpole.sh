#!/usr/bin/env bash
# Wrapper do ewaluacji najlepszego modelu CartPole.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
model_path="${1:-}"
episodes="${2:-10}"

cd "$repo_root"

if [[ -z "$model_path" ]]; then
	source .venv/bin/activate
	model_path="$(python - <<'PY'
import csv
from pathlib import Path


def parse_float(value: str | None) -> float | None:
	if value is None:
		return None
	clean = value.strip()
	if not clean:
		return None
	return float(clean)


results_path = Path("data/experiments.csv")
rows = list(csv.DictReader(results_path.open(encoding="utf-8", newline="")))
candidates: list[dict[str, str]] = []
for row in rows:
	if row.get("mean_reward", "").strip():
		candidates.append(row)

if not candidates:
	raise SystemExit("Brak ukonczonych eksperymentow w data/experiments.csv")

best = max(
	candidates,
	key=lambda row: (
		parse_float(row.get("objective_score")) if parse_float(row.get("objective_score")) is not None else float("-inf"),
		parse_float(row.get("mean_reward")) if parse_float(row.get("mean_reward")) is not None else float("-inf"),
		-(parse_float(row.get("std_reward")) if parse_float(row.get("std_reward")) is not None else float("inf")),
		-(parse_float(row.get("training_time_s")) if parse_float(row.get("training_time_s")) is not None else float("inf")),
	),
)
print(f"models/{best['experiment_id']}.zip")
PY
)"
fi

export SDL_VIDEODRIVER=cocoa
source .venv/bin/activate
python -m src.evaluate --model-path "$model_path" --env-id CartPole-v1 --episodes "$episodes"
