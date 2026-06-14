#!/usr/bin/env bash
# Wrapper do ewaluacji ostatniego produkcyjnego modelu Humanoida.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
episodes="${1:-5}"

cd "$repo_root"
export SDL_VIDEODRIVER=cocoa
source .venv/bin/activate

python -m src.evaluate_humanoid_production --episodes "$episodes"
