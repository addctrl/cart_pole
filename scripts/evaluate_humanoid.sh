#!/usr/bin/env bash
# Wrapper do ewaluacji najlepszego modelu Humanoida.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
model_path="${1:-}"
episodes="${7:-10}"

cd "$repo_root"

if [[ -z "$model_path" ]]; then
	source .venv/bin/activate
	model_path="$(python - <<'PY'
import csv
from pathlib import Path

results_path = Path("data/humanoid_bayes_results.csv")
rows = list(csv.DictReader(results_path.open(encoding="utf-8", newline="")))
completed = [row for row in rows if row.get("status") == "completed" and row.get("objective_score")]
if not completed:
	raise SystemExit("Brak ukończonych prób w data/humanoid_bayes_results.csv")

best = max(
	completed,
	key=lambda row: (
		float(row["objective_score"]),
		float(row["mean_reward"]),
		-float(row["std_reward"]),
		-float(row["training_time_s"]),
	),
)
print(f"models/{best['experiment_id']}.zip")
PY
)"
fi

export SDL_VIDEODRIVER=cocoa
source .venv/bin/activate
python -m src.evaluate --model-path "$model_path" --env-id Humanoid-v5 --episodes "$episodes"