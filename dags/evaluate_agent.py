from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from airflow.sdk import Param, dag, get_current_context, task
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_MODEL = "nebius/moonshotai/Kimi-K2.6"
DEFAULT_EXPERIMENT_NAME = "swe-bench-agent-evals"
DEFAULT_MLFLOW_TRACKING_URI = "http://localhost:5018"
DEFAULT_STEP_LIMIT = 30
DEFAULT_S3_ARTIFACT_URI = "s3://mlops-runs/swe-bench-runs/"
DEFAULT_MINIO_BROWSER_URI = "http://localhost:9001"
DEFAULT_EXECUTION_IMAGE = "my_fork-airflow:latest"
DEFAULT_AGENT_TIMEOUT_HOURS = 8
DEFAULT_EVAL_TIMEOUT_HOURS = 8
DEFAULT_SUMMARY_TIMEOUT_MINUTES = 20
NETWORK_RETRY_ATTEMPTS = 3
NETWORK_RETRY_DELAY_SECONDS = 5

T = TypeVar("T")


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


def _with_network_retries(operation: Callable[[], T]) -> T:
    last_error: Exception | None = None
    for attempt in range(1, NETWORK_RETRY_ATTEMPTS + 1):
        try:
            return operation()
        except urllib.error.HTTPError as error:
            if error.code < 500:
                raise
            last_error = error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
        if attempt < NETWORK_RETRY_ATTEMPTS:
            time.sleep(NETWORK_RETRY_DELAY_SECONDS * attempt)
    assert last_error is not None
    raise last_error


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


def _execution_image() -> str:
    return os.environ.get("EXECUTION_IMAGE", DEFAULT_EXECUTION_IMAGE)


def _docker_network_mode() -> str | None:
    value = os.environ.get("DOCKER_OPERATOR_NETWORK_MODE", "").strip()
    return value or None


def _docker_mounts() -> list[Mount]:
    host_project_root = os.environ.get("HOST_PROJECT_ROOT", str(PROJECT_ROOT))
    host_docker_socket = os.environ.get("HOST_DOCKER_SOCKET", "/var/run/docker.sock")
    return [
        Mount(source=host_project_root, target="/mlops-assignment", type="bind"),
        Mount(source=host_docker_socket, target="/var/run/docker.sock", type="bind"),
    ]


def _xcom_config(key: str) -> str:
    return "{{ ti.xcom_pull(task_ids='prepare_run')['" + key + "'] }}"


def _container_environment() -> dict[str, str]:
    return {
        "PROJECT_ROOT": "/mlops-assignment",
        "RUN_ID": _xcom_config("run_id"),
        "RUN_DIR": _xcom_config("run_dir"),
        "SWE_BENCH_SUBSET": _xcom_config("subset"),
        "SWE_BENCH_SPLIT": _xcom_config("split"),
        "SWE_BENCH_DATASET": _xcom_config("dataset_name"),
        "SWE_BENCH_WORKERS": _xcom_config("workers"),
        "SWE_BENCH_TASK_SLICE": _xcom_config("task_slice"),
        "MINI_SWE_MODEL": _xcom_config("model"),
        "MINI_SWE_STEP_LIMIT": _xcom_config("step_limit"),
        "MINI_SWE_CONFIG": _xcom_config("mini_swe_config"),
        "SWE_BENCH_PREDS": "{{ ti.xcom_pull(task_ids='prepare_run')['run_dir'] }}/run-agent/preds.json",
        "MSWEA_COST_TRACKING": os.environ.get("MSWEA_COST_TRACKING", "ignore_errors"),
        "MSWEA_GLOBAL_COST_LIMIT": (
            "{{ '' if ti.xcom_pull(task_ids='prepare_run')['cost_limit'] is none "
            "else ti.xcom_pull(task_ids='prepare_run')['cost_limit'] }}"
        ),
        "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
        "SCRIPT_EXT": "sh",
    }


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
        config=Config(
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": NETWORK_RETRY_ATTEMPTS, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
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
    _with_network_retries(lambda: client.upload_file(str(local_path), bucket, key))
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
        _with_network_retries(lambda path=path, key=key: client.upload_file(str(path), bucket, key))

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
    def send() -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8") or "{}")

    return _with_network_retries(send)


def _mlflow_http_get(tracking_uri: str, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{tracking_uri.rstrip('/')}{endpoint}?{query}"
    request = urllib.request.Request(url, method="GET")
    def send() -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8") or "{}")

    return _with_network_retries(send)


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
    @task(retries=1, retry_delay=timedelta(minutes=2), execution_timeout=timedelta(minutes=5))
    def prepare_run() -> dict[str, Any]:
        context = get_current_context()
        assert 'params' in context, "Airflow context missing 'params'"
        assert 'dag_run' in context, "Airflow context missing 'dag_run'"
        run_config = build_run_config(context["params"], context["dag_run"].run_id)
        run_dir = prepare_run_dir(run_config)
        run_config["run_dir"] = run_dir
        return run_config

    @task(retries=2, retry_delay=timedelta(minutes=2), execution_timeout=timedelta(minutes=DEFAULT_SUMMARY_TIMEOUT_MINUTES))
    def summarize_and_log(run_config: dict[str, Any]) -> str:
        preds_path = str(Path(run_config["run_dir"]) / "run-agent" / "preds.json")
        eval_dir = str(Path(run_config["run_dir"]) / "run-eval")
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

    run_agent = DockerOperator(
        task_id="run_agent",
        image=_execution_image(),
        command=["bash", "-c", "exec ./scripts/run-agent-container.${SCRIPT_EXT}"],
        working_dir="/mlops-assignment",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
        mounts=_docker_mounts(),
        network_mode=_docker_network_mode(),
        environment=_container_environment(),
        retries=2,
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(hours=DEFAULT_AGENT_TIMEOUT_HOURS),
    )

    run_eval = DockerOperator(
        task_id="run_eval",
        image=_execution_image(),
        command=["bash", "-c", "exec ./scripts/run-eval-container.${SCRIPT_EXT}"],
        working_dir="/mlops-assignment",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
        mounts=_docker_mounts(),
        network_mode=_docker_network_mode(),
        environment=_container_environment(),
        retries=1,
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(hours=DEFAULT_EVAL_TIMEOUT_HOURS),
    )

    summary = summarize_and_log(config)
    config >> run_agent >> run_eval >> summary


evaluate_agent = evaluate_agent_dag()
