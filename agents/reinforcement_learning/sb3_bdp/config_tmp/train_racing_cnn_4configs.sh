#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"

CONFIGS=(
  "agents/reinforcement_learning/sb3_bdp/config_tmp/config_racing_TRPO_cnn_train.yaml"
  "agents/reinforcement_learning/sb3_bdp/config_tmp/config_racing_PPO_cnn_train.yaml"
  "agents/reinforcement_learning/sb3_bdp/config_tmp/config_racing_TRPO_builtin_cnn_train.yaml"
  "agents/reinforcement_learning/sb3_bdp/config_tmp/config_racing_PPO_builtin_cnn_train.yaml"
)

for config_file in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Training with ${config_file}"
  echo "============================================================"
  python3 run_BDPL_sb3.py --config_file="${config_file}"
done
