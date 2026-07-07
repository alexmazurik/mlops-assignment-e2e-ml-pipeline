set -euo pipefail

: "${AIRFLOW_HOME:=$HOME/airflow}"
: "${AIRFLOW__CORE__DAGS_FOLDER:=$(pwd)/dags}"
: "${AIRFLOW__CORE__LOAD_EXAMPLES:=false}"
export AIRFLOW_HOME
export AIRFLOW__CORE__DAGS_FOLDER
export AIRFLOW__CORE__LOAD_EXAMPLES

mkdir -p "$AIRFLOW_HOME"

echo '{"admin": "admin"}' > "$AIRFLOW_HOME/simple_auth_manager_passwords.json.generated"

uv tool run \
    --python 3.12 \
    --with boto3 \
    --with apache-airflow-providers-docker \
    apache-airflow standalone
