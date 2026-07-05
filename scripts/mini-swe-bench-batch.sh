set -euo pipefail

: "${SWE_BENCH_SUBSET:=verified}"
: "${SWE_BENCH_SPLIT:=test}"
: "${MINI_SWE_MODEL:=nebius/moonshotai/Kimi-K2.6}"
: "${SWE_BENCH_TASK_SLICE:=0:3}"
: "${MINI_SWE_CONFIG:=}"
: "${SWE_BENCH_WORKERS:=5}"
: "${MINI_SWE_OUTPUT:=trajectories}"
: "${MSWEA_COST_TRACKING:=ignore_errors}"

command=(
    mini-extra swebench
    --subset "$SWE_BENCH_SUBSET"
    --split "$SWE_BENCH_SPLIT"
    --model "$MINI_SWE_MODEL"
    --workers "$SWE_BENCH_WORKERS"
    -o "$MINI_SWE_OUTPUT"
)

if [[ -n "$SWE_BENCH_TASK_SLICE" ]]; then
    command+=(--slice "$SWE_BENCH_TASK_SLICE")
fi

if [[ -n "$MINI_SWE_CONFIG" ]]; then
    command+=(--config "$MINI_SWE_CONFIG")
fi

MSWEA_COST_TRACKING="$MSWEA_COST_TRACKING" "${command[@]}"
