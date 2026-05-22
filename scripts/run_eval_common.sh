#!/usr/bin/env bash
set -euo pipefail

TASK_ID="${1:?Usage: run_eval_common.sh <task-id> [extra main_pipeline args...]}"
shift

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-rl-project}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/cfg/main_pipeline.yaml}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/setup/environment.yml}"
REQ_FILE="${REQ_FILE:-$ROOT_DIR/setup/requirements.txt}"
INSTALL_ENV="${INSTALL_ENV:-1}"

cd "$ROOT_DIR"

echo "[eval] Repository: $ROOT_DIR"
echo "[eval] Task: $TASK_ID"
echo "[eval] Config: $CONFIG_PATH"
echo "[eval] Environment file: $ENV_FILE"

if [[ "$INSTALL_ENV" != "0" ]]; then
    if command -v micromamba >/dev/null 2>&1; then
        if micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
            echo "[eval] Updating Micromamba environment: $ENV_NAME"
            micromamba env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
        else
            echo "[eval] Creating Micromamba environment from $ENV_FILE"
            micromamba env create -n "$ENV_NAME" -f "$ENV_FILE"
        fi
        RUNNER=(micromamba run -n "$ENV_NAME" python)
    elif command -v conda >/dev/null 2>&1; then
        if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
            echo "[eval] Updating Conda environment: $ENV_NAME"
            conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
        else
            echo "[eval] Creating Conda environment from $ENV_FILE"
            conda env create -n "$ENV_NAME" -f "$ENV_FILE"
        fi
        RUNNER=(conda run -n "$ENV_NAME" python)
    else
        echo "[eval] Micromamba/Conda not found; creating local .venv with $REQ_FILE"
        python3 -m venv "$ROOT_DIR/.venv"
        "$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
        "$ROOT_DIR/.venv/bin/python" -m pip install -r "$REQ_FILE"
        RUNNER=("$ROOT_DIR/.venv/bin/python")
    fi
else
    echo "[eval] Skipping environment install/update because INSTALL_ENV=0"
    if command -v micromamba >/dev/null 2>&1; then
        RUNNER=(micromamba run -n "$ENV_NAME" python)
    elif command -v conda >/dev/null 2>&1; then
        RUNNER=(conda run -n "$ENV_NAME" python)
    elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
        RUNNER=("$ROOT_DIR/.venv/bin/python")
    else
        RUNNER=(python3)
    fi
fi

echo "[eval] Running main pipeline"
exec "${RUNNER[@]}" "$ROOT_DIR/main_pipeline.py" --config "$CONFIG_PATH" --task "$TASK_ID" "$@"
