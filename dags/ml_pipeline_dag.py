from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import os
import yaml

# Загрузка конфигурации
with open('/opt/airflow/config/config.yaml') as f:
    config = yaml.safe_load(f)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    's3_conn_id': 'minio_conn'
}

def run_etl_script(script_name, **kwargs):
    import subprocess
    result = subprocess.run(
        ['python', f'/opt/airflow/etl/{script_name}.py'], 
        capture_output=True, 
        text=True
    )
    if result.returncode != 0:
        error_msg = f"Script {script_name} failed: {result.stderr}"
        raise Exception(error_msg)
    return result.stdout

with DAG(
    'breast_cancer_pipeline',
    default_args=default_args,
    description='Automated ML Pipeline for Breast Cancer Diagnosis',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'medical'],
) as dag:

    download_task = PythonOperator(
        task_id='download_data',
        python_callable=run_etl_script,
        op_kwargs={'script_name': 'download_data'},
        on_failure_callback=lambda context: print("Download failed!"),
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=run_etl_script,
        op_kwargs={'script_name': 'preprocess_data'},
        retries=2,
        on_failure_callback=lambda context: print("Preprocessing failed!"),
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=run_etl_script,
        op_kwargs={'script_name': 'train_model'},
        execution_timeout=timedelta(minutes=30),
        on_failure_callback=lambda context: print("Training failed!"),
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=run_etl_script,
        op_kwargs={'script_name': 'evaluate_model'},
        on_failure_callback=lambda context: print("Evaluation failed!"),
    )

    download_task >> preprocess_task >> train_task >> evaluate_task