from airflow import DAG
from airflow.operators.bash import BashOperator

from datetime import datetime

default_args = {
    "owner": "Patryk",
    "start_date": datetime(2025, 1, 1)
}

with DAG(
    dag_id="retrain_model_pipeline",
    default_args=default_args,
    schedule="@daily",
    catchup=False
) as dag:

    retrain_task = BashOperator(
        task_id="retrain_model",
        bash_command="python LAB08/retrain.py"
    )

    retrain_task