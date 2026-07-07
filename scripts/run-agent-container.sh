#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?RUN_ID is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${SWE_BENCH_SUBSET:=verified}"
: "${SWE_BENCH_SPLIT:=test}"
: "${MINI_SWE_MODEL:=nebius/moonshotai/Kimi-K2.6}"
: "${SWE_BENCH_WORKERS:=5}"
: "${MINI_SWE_STEP_LIMIT:=30}"
: "${SWE_BENCH_TASK_SLICE:=}"
: "${MINI_SWE_CONFIG:=}"
: "${MSWEA_COST_TRACKING:=ignore_errors}"
if [[ -z "${MSWEA_GLOBAL_COST_LIMIT:-}" ]]; then
    unset MSWEA_GLOBAL_COST_LIMIT
fi

PROJECT_ROOT="${PROJECT_ROOT:-/mlops-assignment}"
AGENT_DIR="$RUN_DIR/run-agent"
TRAJECTORIES_DIR="$AGENT_DIR/trajectories"
LOG_PATH="$AGENT_DIR/mini-swe-agent.log"
COMMAND_PATH="$AGENT_DIR/command.json"
STABLE_PREDS="$AGENT_DIR/preds.json"

mkdir -p "$TRAJECTORIES_DIR"
cd "$PROJECT_ROOT"

command=(
    uv run mini-extra swebench
    --subset "$SWE_BENCH_SUBSET"
    --split "$SWE_BENCH_SPLIT"
    --model "$MINI_SWE_MODEL"
    --workers "$SWE_BENCH_WORKERS"
    -o "$TRAJECTORIES_DIR"
    --config swebench.yaml
)

if [[ -n "$SWE_BENCH_TASK_SLICE" ]]; then
    command+=(--slice "$SWE_BENCH_TASK_SLICE")
fi

if [[ -n "$MINI_SWE_CONFIG" ]]; then
    command+=(--config "$MINI_SWE_CONFIG")
fi

command+=(--config "agent.step_limit=$MINI_SWE_STEP_LIMIT")

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '$ ' > "$LOG_PATH"
printf '%q ' "${command[@]}" >> "$LOG_PATH"
printf '\n\n' >> "$LOG_PATH"

set +e
"${command[@]}" >> "$LOG_PATH" 2>&1
returncode=$?
set -e

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python - "$COMMAND_PATH" "$PROJECT_ROOT" "$LOG_PATH" "$started_at" "$finished_at" "$returncode" "${command[@]}" <<'PY'
import json
import sys

metadata_path, cwd, log_path, started_at, finished_at, returncode, *command = sys.argv[1:]
with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "command": command,
            "cwd": cwd,
            "started_at": started_at,
            "finished_at": finished_at,
            "log_path": log_path,
            "returncode": int(returncode),
            "executor": "DockerOperator",
        },
        file,
        indent=2,
        sort_keys=True,
    )
    file.write("\n")
PY

if [[ "$returncode" -ne 0 ]]; then
    exit "$returncode"
fi

PRODUCED_PREDS="$TRAJECTORIES_DIR/preds.json"
if [[ ! -f "$PRODUCED_PREDS" ]]; then
    echo "mini-swe-agent did not produce $PRODUCED_PREDS" >&2
    exit 1
fi
cp "$PRODUCED_PREDS" "$STABLE_PREDS"

python - "$STABLE_PREDS" "$TRAJECTORIES_DIR" "$AGENT_DIR/instances.json" "$LOG_PATH" <<'PY'
import json
import sys
from pathlib import Path

preds_path = Path(sys.argv[1])
trajectories_dir = Path(sys.argv[2])
instances_path = Path(sys.argv[3])
log_path = Path(sys.argv[4])

predictions = json.loads(preds_path.read_text())
if not predictions:
    raise SystemExit(f"mini-swe-agent produced no predictions in {preds_path}")

missing = [
    instance_id
    for instance_id in predictions
    if not (trajectories_dir / instance_id / f"{instance_id}.traj.json").exists()
]
if missing:
    raise SystemExit(
        "mini-swe-agent did not produce trajectory files for "
        f"{len(missing)} instance(s): {', '.join(sorted(missing))}. Check {log_path}"
    )

if all(not prediction.get("model_patch") for prediction in predictions.values()):
    raise SystemExit(f"mini-swe-agent produced only empty patches. Check {log_path}")

instances_path.write_text(
    json.dumps(
        {
            "count": len(predictions),
            "instance_ids": sorted(predictions),
            "source": str(preds_path),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
