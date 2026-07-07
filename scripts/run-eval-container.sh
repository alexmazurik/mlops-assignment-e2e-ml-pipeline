#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?RUN_ID is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${SWE_BENCH_DATASET:=princeton-nlp/SWE-bench_Verified}"
: "${SWE_BENCH_SPLIT:=test}"
: "${SWE_BENCH_WORKERS:=5}"

PROJECT_ROOT="${PROJECT_ROOT:-/mlops-assignment}"
PREDS_PATH="${SWE_BENCH_PREDS:-$RUN_DIR/run-agent/preds.json}"
EVAL_DIR="$RUN_DIR/run-eval"
LOG_PATH="$EVAL_DIR/swe-bench-eval.log"
COMMAND_PATH="$EVAL_DIR/command.json"
REPORTS_DIR="$EVAL_DIR/reports"

mkdir -p "$EVAL_DIR/logs" "$REPORTS_DIR"
cd "$EVAL_DIR"

command=(
    uv run python -m swebench.harness.run_evaluation
    --dataset_name "$SWE_BENCH_DATASET"
    --split "$SWE_BENCH_SPLIT"
    --predictions_path "$PREDS_PATH"
    --max_workers "$SWE_BENCH_WORKERS"
    --run_id "$RUN_ID"
)

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '$ ' > "$LOG_PATH"
printf '%q ' "${command[@]}" >> "$LOG_PATH"
printf '\n\n' >> "$LOG_PATH"

set +e
"${command[@]}" >> "$LOG_PATH" 2>&1
returncode=$?
set -e

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python - "$COMMAND_PATH" "$EVAL_DIR" "$LOG_PATH" "$started_at" "$finished_at" "$returncode" "${command[@]}" <<'PY'
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

python - "$EVAL_DIR" "$REPORTS_DIR" <<'PY'
import json
import shutil
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
reports_dir = Path(sys.argv[2])

for report_path in (eval_dir / "logs").glob("run_evaluation/**/*.json"):
    if report_path.name != "report.json":
        continue
    report = json.loads(report_path.read_text())
    for instance_id in report:
        shutil.copy2(report_path, reports_dir / f"{instance_id}.report.json")
PY
