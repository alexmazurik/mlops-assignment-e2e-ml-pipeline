set -euo pipefail

: "${SWE_BENCH_DATASET:=princeton-nlp/SWE-bench_Verified}"
: "${SWE_BENCH_PREDS:=trajectories/preds.json}"
: "${SWE_BENCH_WORKERS:=5}"
: "${SWE_BENCH_RUN_ID:=test}"

python -m swebench.harness.run_evaluation \
    --dataset_name "$SWE_BENCH_DATASET" \
    --predictions_path "$SWE_BENCH_PREDS" \
    --max_workers "$SWE_BENCH_WORKERS" \
    --run_id "$SWE_BENCH_RUN_ID"
