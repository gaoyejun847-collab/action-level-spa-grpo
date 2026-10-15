#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export VLLM_USE_V1=1
MODEL="${SPA_MODEL_PATH:-$ROOT/models/Qwen3-4B-Instruct-2507}"
if [[ ! -f "$MODEL/config.json" ]]; then
    echo "Missing model config: $MODEL/config.json" >&2
    exit 1
fi
exec "$ROOT/.venv/bin/python" -u -m spa_next.training.launch \
    "actor_rollout_ref.model.path=$MODEL" "$@"
