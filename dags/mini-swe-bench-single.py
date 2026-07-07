from datetime import datetime
import os
from pathlib import Path

from airflow.decorators import dag
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dag(
    dag_id="mini-swe-bench-single",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
)
def my_dag():
    DockerOperator(
        task_id="run_single_instance",
        image=os.environ.get("EXECUTION_IMAGE", "swe-bench-agent-airflow:latest"),
        command=[
            "uv",
            "run",
            "mini-extra",
            "swebench-single",
            "--subset",
            "verified",
            "--split",
            "test",
            "--model",
            "nebius/moonshotai/Kimi-K2.6",
            "--yolo",
            "--cost-limit",
            "0",
            "-i",
            "sympy__sympy-15599",
            "-o",
            "trajectory.json",
        ],
        working_dir="/mlops-assignment",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source=os.environ.get("HOST_PROJECT_ROOT", str(PROJECT_ROOT)),
                target="/mlops-assignment",
                type="bind",
            ),
        ],
        network_mode=os.environ.get("DOCKER_OPERATOR_NETWORK_MODE") or None,
        environment={
            "MSWEA_COST_TRACKING": "ignore_errors",
            "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
        },
    )


my_dag()
