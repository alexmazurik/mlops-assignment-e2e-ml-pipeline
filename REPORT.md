# SWE-bench Agent Evaluation Pipeline

## Architecture

The main DAG is `dags/evaluate_agent.py` with four tasks:

1. `prepare_run`: reads Airflow params, creates `runs/<run-id>/`, and writes `config.json`.
2. `run_agent`: runs `uv run mini-extra swebench` with the selected SWE-bench split, subset, model, workers, task slice, and cost limit. Outputs are written under `run-agent/`.
3. `run_eval`: runs `uv run python -m swebench.harness.run_evaluation` against `run-agent/preds.json`. Logs and reports are written under `run-eval/`.
4. `summarize_and_log`: parses SWE-bench `report.json` files, writes `metrics.json` and `manifest.json`, and logs params/metrics plus the artifact path to MLflow.

The DAG defaults to a small `verified`/`test` run with `task_slice=0:3`, which is the intended smoke-test size. Larger runs can be started from the Airflow trigger form by changing the same params.

## Airflow Parameters

Required:

- `split`: SWE-bench split, for example `test`.
- `subset`: mini-swe-agent subset, for example `verified` or `lite`.
- `workers`: parallel workers for agent and evaluation steps.

Useful optional params:

- `model`: model name passed to mini-swe-agent.
- `task_slice`: mini-swe-agent slice syntax, for example `0:3`; set empty for the full subset.
- `run_id`: stable run directory name. If empty, the Airflow run id is sanitized and used.
- `cost_limit`: mini-swe-agent global cost limit, exported as `MSWEA_GLOBAL_COST_LIMIT`.
- `dataset_name`: override the SWE-bench dataset name. By default it is derived from `subset`.
- `mini_swe_config`: optional path to a mini-swe-agent SWE-bench config.
- `mlflow_tracking_uri`: MLflow server URL. In compose this is `http://mlflow:5000`.

## Artifact Layout

Every DAG run writes:

```text
runs/<run-id>/
  config.json
  run-agent/
    command.json
    instances.json
    mini-swe-agent.log
    preds.json
    trajectories/
  run-eval/
    command.json
    logs/
    reports/
    swe-bench-eval.log
  metrics.json
  manifest.json
```

`manifest.json` is the entrypoint for reconstructing a run. It records the local artifact URI, config path, command metadata, prediction path, trajectory directory, evaluation logs, report directory, metrics path, and MLflow logging status.

## Running Locally

Install dependencies and add secrets:

```bash
cp .env.example .env
# edit .env and set NEBIUS_API_KEY
uv sync
```

Start standalone Airflow:

```bash
bash run-airflow-standalone.sh
```

Open Airflow at `http://localhost:8080`, log in with `admin` / `admin`, and trigger the `evaluate-agent` DAG.

## Running With Compose

The compose setup starts both MLflow and Airflow:

```bash
cp .env.example .env
# edit .env and set NEBIUS_API_KEY
docker compose up --build
```

Then open:

- Airflow: `http://localhost:8080`
- MLflow: `http://localhost:5000`

Compose passes `MLFLOW_TRACKING_URI=http://mlflow:5000` to Airflow, so completed DAG runs are logged to the `swe-bench-agent-evals` MLflow experiment.

## Completed Smoke Run

I ran a real one-instance smoke test using the DAG helper functions with:

- `run_id`: `codex-smoke-20260705`
- `subset`: `verified`
- `split`: `test`
- `task_slice`: `0:1`
- `workers`: `1`
- `model`: `nebius/moonshotai/Kimi-K2.6`

The run artifacts are in `runs/codex-smoke-20260705/`. The result was:

- submitted instances: `1`
- evaluated instances: `1`
- resolved instances: `1`
- resolve rate: `1.0`
- resolved id: `astropy__astropy-12907`

The important files are:

- `runs/codex-smoke-20260705/config.json`
- `runs/codex-smoke-20260705/run-agent/preds.json`
- `runs/codex-smoke-20260705/run-agent/trajectories/`
- `runs/codex-smoke-20260705/run-eval/reports/astropy__astropy-12907.report.json`
- `runs/codex-smoke-20260705/metrics.json`
- `runs/codex-smoke-20260705/manifest.json`

For this direct smoke run, MLflow logging was skipped intentionally so a local MLflow server was not required. The Airflow DAG logs to MLflow when `MLFLOW_TRACKING_URI` points to a reachable server, as in the compose setup.

## Bundled Sample Summary

The bundled sample artifacts in `sample/` summarize correctly with the DAG metrics collector:

- submitted instances: `3`
- evaluated instances: `3`
- resolved instances: `1`
- resolve rate: `0.3333333333333333`
- resolved id: `astropy__astropy-12907`

This was validated with:

```bash
python3 - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("evaluate_agent", "dags/evaluate_agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.collect_metrics("sample", "sample/trajectories/preds.json"))
PY
```

## Remote Storage

The current implementation writes durable local folders under `runs/<run-id>/` and logs that path to MLflow. For long-term object storage, sync the completed run folder after `summarize_and_log`, for example:

```bash
aws s3 sync runs/<run-id>/ "$S3_ARTIFACT_URI/<run-id>/"
```

The manifest format already has a single `artifact_uri` field, so replacing the local path with an S3 URI is the intended next production step.
