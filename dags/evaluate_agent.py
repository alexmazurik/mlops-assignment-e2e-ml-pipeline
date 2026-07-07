from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from airflow.sdk import Param, dag, get_current_context, task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_MODEL = "nebius/moonshotai/Kimi-K2.6"
DEFAULT_EXPERIMENT_NAME = "swe-bench-agent-evals"
DEFAULT_MLFLOW_TRACKING_URI = "http://localhost:5018"
DEFAULT_STEP_LIMIT = 30
DEFAULT_S3_ARTIFACT_URI = "s3://mlops-runs/swe-bench-runs/"
DEFAULT_MINIO_BROWSER_URI = "http://localhost:9001"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _sanitized_run_id(value: str | None) -> str:
    base = value or f"manual-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-._")
    return base or f"run-{int(time.time())}"


def _dataset_name(subset: str) -> str:
    normalized = subset.lower()
    if normalized == "verified":
        return "princeton-nlp/SWE-bench_Verified"
    if normalized == "lite":
        return "princeton-nlp/SWE-bench_Lite"
    return "princeton-nlp/SWE-bench"


def build_run_config(params: dict[str, Any], airflow_run_id: str | None = None) -> dict[str, Any]:
    env = {**_load_dotenv(PROJECT_ROOT / ".env"), **os.environ}
    run_id_param = str(params.get("run_id") or "").strip()
    run_id = _sanitized_run_id(run_id_param or airflow_run_id)
    subset = str(params.get("subset") or "verified")
    split = str(params.get("split") or "test")
    model = str(params.get("model") or DEFAULT_MODEL)
    task_slice = str(params.get("task_slice") or "").strip()
    workers = int(params.get("workers") or 1)
    step_limit = int(params.get("step_limit") or DEFAULT_STEP_LIMIT)
    cost_limit_raw = params.get("cost_limit", 0)
    cost_limit = None if cost_limit_raw in (None, "") else float(cost_limit_raw)
    mini_swe_config = str(params.get("mini_swe_config") or "").strip()
    s3_artifact_uri = str(params.get("s3_artifact_uri") or env.get("S3_ARTIFACT_URI") or "").strip()
    s3_endpoint_url = str(
        params.get("s3_endpoint_url")
        or env.get("AWS_ENDPOINT_URL_S3")
        or env.get("S3_ENDPOINT_URL")
        or ""
    ).strip()
    minio_browser_uri = str(
        params.get("minio_browser_uri")
        or env.get("MINIO_BROWSER_URI")
        or DEFAULT_MINIO_BROWSER_URI
    ).strip()

    return {
        "run_id": run_id,
        "created_at": _utc_now(),
        "project_root": str(PROJECT_ROOT),
        "split": split,
        "subset": subset,
        "dataset_name": str(params.get("dataset_name") or _dataset_name(subset)),
        "workers": workers,
        "step_limit": step_limit,
        "model": model,
        "task_slice": task_slice,
        "cost_limit": cost_limit,
        "mini_swe_config": mini_swe_config,
        "s3_artifact_uri": s3_artifact_uri,
        "s3_endpoint_url": s3_endpoint_url,
        "minio_browser_uri": minio_browser_uri,
        "mlflow_tracking_uri": str(
            params.get("mlflow_tracking_uri")
            or env.get("MLFLOW_TRACKING_URI")
            or DEFAULT_MLFLOW_TRACKING_URI
        ),
        "mlflow_experiment_name": str(
            params.get("mlflow_experiment_name")
            or env.get("MLFLOW_EXPERIMENT_NAME")
            or DEFAULT_EXPERIMENT_NAME
        ),
    }


def prepare_run_dir(run_config: dict[str, Any]) -> str:
    run_dir = RUNS_ROOT / run_config["run_id"]
    (run_dir / "run-agent" / "trajectories").mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval" / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval" / "reports").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", run_config)
    return str(run_dir)


def _command_env(run_config: dict[str, Any] | None = None) -> dict[str, str]:
    env = {**os.environ, **_load_dotenv(PROJECT_ROOT / ".env")}
    env.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    if run_config and run_config.get("cost_limit") is not None:
        env["MSWEA_GLOBAL_COST_LIMIT"] = str(run_config["cost_limit"])
    return env


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    metadata_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    started_at = _utc_now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "command": command,
        "cwd": str(cwd),
        "started_at": started_at,
        "log_path": str(log_path),
    }
    with log_path.open("w") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=log_file,
            check=False,
        )
    metadata.update({"finished_at": _utc_now(), "returncode": completed.returncode})
    _write_json(metadata_path, metadata)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def run_agent_batch(run_config: dict[str, Any], run_dir: str) -> str:
    run_path = Path(run_dir)
    trajectories_dir = run_path / "run-agent" / "trajectories"
    command = [
        "uv",
        "run",
        "mini-extra",
        "swebench",
        "--subset",
        run_config["subset"],
        "--split",
        run_config["split"],
        "--model",
        run_config["model"],
        "--workers",
        str(run_config["workers"]),
        "-o",
        str(trajectories_dir),
    ]
    if run_config.get("task_slice"):
        command.extend(["--slice", str(run_config["task_slice"])])
    command.extend(["--config", "swebench.yaml"])
    if run_config.get("mini_swe_config"):
        command.extend(["--config", str(run_config["mini_swe_config"])])
    command.extend(["--config", f"agent.step_limit={run_config['step_limit']}"])

    _run_command(
        command,
        cwd=PROJECT_ROOT,
        env=_command_env(run_config),
        log_path=run_path / "run-agent" / "mini-swe-agent.log",
        metadata_path=run_path / "run-agent" / "command.json",
    )

    produced_preds = trajectories_dir / "preds.json"
    stable_preds = run_path / "run-agent" / "preds.json"
    if not produced_preds.exists():
        raise FileNotFoundError(f"mini-swe-agent did not produce {produced_preds}")
    shutil.copy2(produced_preds, stable_preds)
    predictions = _read_json(stable_preds)
    if not predictions:
        raise RuntimeError(f"mini-swe-agent produced no predictions in {stable_preds}")

    missing_trajectories = [
        instance_id
        for instance_id in predictions
        if not (trajectories_dir / instance_id / f"{instance_id}.traj.json").exists()
    ]
    if missing_trajectories:
        raise RuntimeError(
            "mini-swe-agent did not produce trajectory files for "
            f"{len(missing_trajectories)} instance(s): {', '.join(sorted(missing_trajectories))}. "
            f"Check {run_path / 'run-agent' / 'mini-swe-agent.log'}"
        )

    if all(not prediction.get("model_patch") for prediction in predictions.values()):
        raise RuntimeError(
            "mini-swe-agent produced only empty patches. "
            f"Check {run_path / 'run-agent' / 'mini-swe-agent.log'}"
        )

    _write_json(
        run_path / "run-agent" / "instances.json",
        {
            "count": len(predictions),
            "instance_ids": sorted(predictions),
            "source": str(stable_preds),
        },
    )
    return str(stable_preds)


def run_swebench_eval(run_config: dict[str, Any], preds_path: str, run_dir: str) -> str:
    run_path = Path(run_dir)
    eval_dir = run_path / "run-eval"
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        run_config["dataset_name"],
        "--split",
        run_config["split"],
        "--predictions_path",
        str(Path(preds_path).resolve()),
        "--max_workers",
        str(run_config["workers"]),
        "--run_id",
        run_config["run_id"],
    ]
    _run_command(
        command,
        cwd=eval_dir,
        env=_command_env(run_config),
        log_path=eval_dir / "swe-bench-eval.log",
        metadata_path=eval_dir / "command.json",
    )

    reports_dir = eval_dir / "reports"
    for report_path in (eval_dir / "logs").glob("run_evaluation/**/*.json"):
        if report_path.name != "report.json":
            continue
        report = _read_json(report_path)
        for instance_id in report:
            shutil.copy2(report_path, reports_dir / f"{instance_id}.report.json")
    return str(eval_dir)


def collect_metrics(eval_dir: str, preds_path: str) -> dict[str, Any]:
    eval_path = Path(eval_dir)
    reports_dir = eval_path / "reports"
    predictions = _read_json(Path(preds_path)) if Path(preds_path).exists() else {}
    report_files = sorted(reports_dir.glob("*.report.json"))
    if not report_files:
        report_files = sorted((eval_path / "logs").glob("run_evaluation/**/report.json"))

    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    patch_applied_ids: list[str] = []
    empty_patch_ids: list[str] = []
    errored_ids: list[str] = []

    for report_file in report_files:
        report = _read_json(report_file)
        for instance_id, payload in report.items():
            if payload.get("resolved"):
                resolved_ids.append(instance_id)
            else:
                unresolved_ids.append(instance_id)
            if payload.get("patch_successfully_applied"):
                patch_applied_ids.append(instance_id)
            if payload.get("patch_is_None") or not payload.get("patch_exists", True):
                empty_patch_ids.append(instance_id)
            if payload.get("error") or payload.get("patch_successfully_applied") is False:
                errored_ids.append(instance_id)

    submitted_ids = set(predictions)
    evaluated_ids = set(resolved_ids) | set(unresolved_ids)
    missing_report_ids = sorted(submitted_ids - evaluated_ids)
    submitted_instances = len(submitted_ids)
    evaluated_instances = len(evaluated_ids)
    resolved_instances = len(set(resolved_ids))

    return {
        "submitted_instances": submitted_instances,
        "evaluated_instances": evaluated_instances,
        "resolved_instances": resolved_instances,
        "unresolved_instances": len(set(unresolved_ids)),
        "missing_report_instances": len(missing_report_ids),
        "patch_applied_instances": len(set(patch_applied_ids)),
        "empty_patch_instances": len(set(empty_patch_ids)),
        "error_instances": len(set(errored_ids)),
        "resolve_rate": resolved_instances / evaluated_instances if evaluated_instances else 0.0,
        "resolved_ids": sorted(set(resolved_ids)),
        "unresolved_ids": sorted(set(unresolved_ids)),
        "missing_report_ids": missing_report_ids,
    }


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3://bucket/prefix URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _s3_client(run_config: dict[str, Any]):
    import boto3
    from botocore.config import Config

    env = {**_load_dotenv(PROJECT_ROOT / ".env"), **os.environ}
    endpoint_url = (
        run_config.get("s3_endpoint_url")
        or env.get("AWS_ENDPOINT_URL_S3")
        or env.get("S3_ENDPOINT_URL")
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        region_name=env.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=env.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY"),
        config=Config(s3={"addressing_style": "path"}),
    )


def _ensure_s3_bucket(client: Any, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)


def _upload_file_to_s3(run_config: dict[str, Any], local_path: Path, artifact_uri: str) -> str:
    bucket, key = _parse_s3_uri(artifact_uri)
    client = _s3_client(run_config)
    _ensure_s3_bucket(client, bucket)
    client.upload_file(str(local_path), bucket, key)
    return artifact_uri


def upload_run_artifacts(run_config: dict[str, Any], run_dir: str) -> str | None:
    base_uri = str(run_config.get("s3_artifact_uri") or "").strip().rstrip("/")
    if not base_uri:
        return None

    run_path = Path(run_dir)
    bucket, prefix = _parse_s3_uri(base_uri)
    run_prefix = "/".join(part for part in [prefix, run_config["run_id"]] if part)
    client = _s3_client(run_config)
    _ensure_s3_bucket(client, bucket)

    for path in sorted(run_path.rglob("*")):
        if not path.is_file():
            continue
        key = f"{run_prefix}/{path.relative_to(run_path).as_posix()}"
        client.upload_file(str(path), bucket, key)

    return f"s3://{bucket}/{run_prefix}/"


def minio_artifact_uri(run_config: dict[str, Any], artifact_uri: str) -> str | None:
    if not artifact_uri.startswith("s3://"):
        return None
    bucket, prefix = _parse_s3_uri(artifact_uri)
    browser_uri = str(run_config.get("minio_browser_uri") or DEFAULT_MINIO_BROWSER_URI).rstrip("/")
    encoded_prefix = urllib.parse.quote(prefix, safe="")
    return f"{browser_uri}/browser/{bucket}/{encoded_prefix}"


def _mlflow_http_request(tracking_uri: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = tracking_uri.rstrip("/") + endpoint
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def _mlflow_http_get(tracking_uri: str, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{tracking_uri.rstrip('/')}{endpoint}?{query}"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def _mlflow_get_or_create_experiment(tracking_uri: str, experiment_name: str) -> str:
    try:
        response = _mlflow_http_get(
            tracking_uri,
            "/api/2.0/mlflow/experiments/get-by-name",
            {"experiment_name": experiment_name},
        )
        experiment = response.get("experiment") or {}
        if experiment.get("experiment_id"):
            return str(experiment["experiment_id"])
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise

    response = _mlflow_http_request(
        tracking_uri,
        "/api/2.0/mlflow/experiments/create",
        {"name": experiment_name},
    )
    return str(response["experiment_id"])


def log_mlflow_run(
    run_config: dict[str, Any],
    metrics: dict[str, Any],
    artifact_uri: str,
    minio_uri: str | None = None,
) -> dict[str, Any]:
    env = {**_load_dotenv(PROJECT_ROOT / ".env"), **os.environ}
    tracking_uri = run_config.get("mlflow_tracking_uri") or env.get("MLFLOW_TRACKING_URI") or DEFAULT_MLFLOW_TRACKING_URI
    if not tracking_uri:
        return {"status": "skipped", "reason": "MLFLOW_TRACKING_URI is not configured"}
    if not tracking_uri.startswith(("http://", "https://")):
        return {
            "status": "skipped",
            "reason": "Only HTTP(S) MLflow tracking URIs are supported without the mlflow package",
            "tracking_uri": tracking_uri,
        }

    timestamp_ms = int(time.time() * 1000)
    experiment_id = _mlflow_get_or_create_experiment(
        tracking_uri,
        str(run_config.get("mlflow_experiment_name") or DEFAULT_EXPERIMENT_NAME),
    )
    run = _mlflow_http_request(
        tracking_uri,
        "/api/2.0/mlflow/runs/create",
        {
            "experiment_id": experiment_id,
            "start_time": timestamp_ms,
            "tags": [
                {"key": "mlflow.runName", "value": run_config["run_id"]},
                {"key": "artifact_uri", "value": artifact_uri},
                {"key": "minio_artifact_uri", "value": minio_uri or ""},
            ],
        },
    )
    mlflow_run_id = run["run"]["info"]["run_id"]
    params = [
        {"key": key, "value": str(value)}
        for key, value in run_config.items()
        if isinstance(value, str | int | float | bool) or value is None
    ]
    params.append({"key": "artifact_uri", "value": artifact_uri})
    if minio_uri:
        params.append({"key": "minio_artifact_uri", "value": minio_uri})
    numeric_metrics = [
        {"key": key, "value": float(value), "timestamp": timestamp_ms, "step": 0}
        for key, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    _mlflow_http_request(
        tracking_uri,
        "/api/2.0/mlflow/runs/log-batch",
        {"run_id": mlflow_run_id, "params": params, "metrics": numeric_metrics},
    )
    _mlflow_http_request(
        tracking_uri,
        "/api/2.0/mlflow/runs/update",
        {"run_id": mlflow_run_id, "status": "FINISHED", "end_time": int(time.time() * 1000)},
    )
    return {
        "status": "logged",
        "tracking_uri": tracking_uri,
        "experiment_id": experiment_id,
        "mlflow_run_id": mlflow_run_id,
    }


def write_manifest(
    run_config: dict[str, Any],
    run_dir: str,
    preds_path: str,
    eval_dir: str,
    metrics: dict[str, Any],
    mlflow_status: dict[str, Any],
    artifact_uri: str,
    minio_uri: str | None,
) -> str:
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.json"
    _write_json(metrics_path, metrics)
    manifest = {
        "run_id": run_config["run_id"],
        "created_at": run_config["created_at"],
        "updated_at": _utc_now(),
        "artifact_uri": artifact_uri,
        "minio_artifact_uri": minio_uri,
        "local_artifact_uri": str(run_path.resolve()),
        "config": "config.json",
        "run_agent": {
            "predictions": str(Path(preds_path).relative_to(run_path)),
            "instances": "run-agent/instances.json",
            "trajectories": "run-agent/trajectories",
            "log": "run-agent/mini-swe-agent.log",
            "command": "run-agent/command.json",
        },
        "run_eval": {
            "directory": str(Path(eval_dir).relative_to(run_path)),
            "logs": "run-eval/logs",
            "reports": "run-eval/reports",
            "log": "run-eval/swe-bench-eval.log",
            "command": "run-eval/command.json",
        },
        "metrics": "metrics.json",
        "mlflow": mlflow_status,
    }
    manifest_path = run_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return str(manifest_path)


@dag(
    dag_id="evaluate-agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "split": Param("test", type="string"),
        "subset": Param("verified", type="string"),
        "workers": Param(5, type="integer", minimum=1),
        "step_limit": Param(DEFAULT_STEP_LIMIT, type="integer", minimum=1),
        "model": Param(DEFAULT_MODEL, type="string"),
        "task_slice": Param("0:3", type=["string", "null"]),
        "run_id": Param("", type=["string", "null"]),
        "cost_limit": Param(0.0, type=["number", "null"]),
        "dataset_name": Param("", type=["string", "null"]),
        "mini_swe_config": Param("", type=["string", "null"]),
        "s3_artifact_uri": Param(DEFAULT_S3_ARTIFACT_URI, type=["string", "null"]),
        "s3_endpoint_url": Param("", type=["string", "null"]),
        "minio_browser_uri": Param(DEFAULT_MINIO_BROWSER_URI, type=["string", "null"]),
        "mlflow_tracking_uri": Param("", type=["string", "null"]),
        "mlflow_experiment_name": Param(DEFAULT_EXPERIMENT_NAME, type="string"),
    },
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["swe-bench", "mini-swe-agent", "mlops"],
)
def evaluate_agent_dag() -> None:
    @task
    def prepare_run() -> dict[str, Any]:
        context = get_current_context()
        assert 'params' in context, "Airflow context missing 'params'"
        assert 'dag_run' in context, "Airflow context missing 'dag_run'"
        run_config = build_run_config(context["params"], context["dag_run"].run_id)
        run_dir = prepare_run_dir(run_config)
        run_config["run_dir"] = run_dir
        return run_config

    @task(execution_timeout=timedelta(hours=8))
    def run_agent(run_config: dict[str, Any]) -> str:
        return run_agent_batch(run_config, run_config["run_dir"])

    @task(execution_timeout=timedelta(hours=8))
    def run_eval(run_config: dict[str, Any], preds_path: str) -> str:
        return run_swebench_eval(run_config, preds_path, run_config["run_dir"])

    @task
    def summarize_and_log(run_config: dict[str, Any], preds_path: str, eval_dir: str) -> str:
        metrics = collect_metrics(eval_dir, preds_path)
        run_path = Path(run_config["run_dir"])
        _write_json(run_path / "metrics.json", metrics)
        artifact_uri = upload_run_artifacts(run_config, run_config["run_dir"]) or str(run_path.resolve())
        minio_uri = minio_artifact_uri(run_config, artifact_uri)
        mlflow_status = log_mlflow_run(run_config, metrics, artifact_uri, minio_uri)
        manifest_path = write_manifest(
            run_config,
            run_config["run_dir"],
            preds_path,
            eval_dir,
            metrics,
            mlflow_status,
            artifact_uri,
            minio_uri,
        )
        if artifact_uri.startswith("s3://"):
            manifest_uri = artifact_uri.rstrip("/") + "/manifest.json"
            _upload_file_to_s3(run_config, Path(manifest_path), manifest_uri)
        return manifest_path

    config = cast(dict[str, Any], prepare_run())
    predictions = cast(str, run_agent(config))
    evaluation = cast(str, run_eval(config, predictions))
    summarize_and_log(config, predictions, evaluation)


evaluate_agent = evaluate_agent_dag()
