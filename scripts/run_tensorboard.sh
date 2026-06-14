#!/usr/bin/env bash
# Wrapper do uruchomienia TensorBoard dla logów PPO.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

cd "$repo_root"
source .venv/bin/activate
tensorboard --logdir=./logs/tensorboard/ --port=6006