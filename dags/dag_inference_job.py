
from datetime import datetime

from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.utils.dates import days_ago

from airflow.contrib.hooks.databricks_hook import DatabricksHook

def get_job_id_by_name(job_name: str, databricks_conn_id: str) -> str:
    list_endpoint = ('GET', 'api/2.0/jobs/list')
    hook = DatabricksHook(databricks_conn_id=databricks_conn_id)
    response_payload = hook._do_api_call(list_endpoint, {})
    all_jobs = response_payload.get("jobs", [])
    matching_jobs = [j for j in all_jobs if j["settings"]["name"] == job_name]

    if not matching_jobs:
        raise Exception(f"Job with name {job_name} not found")

    if len(matching_jobs) > 1:
        raise Exception(f"Job with name {job_name} is duplicated. Please make job name unique in Databricks UI.")

    job_id = matching_jobs[0]["job_id"]
    return job_id



JOB_ID = get_job_id_by_name("cd-infer-job", 'databricks_default')

default_args = {
  'owner': 'airflow'
}

with DAG(
    dag_id='iris-inference-job',
    schedule_interval='*/10 * * * *', #'@daily',
    start_date=datetime(2021, 1, 1),
    tags=['example'],
    catchup=False,
) as dag:

  opr_run_now_job_1 = DatabricksRunNowOperator(
    task_id = 'job_1',
    databricks_conn_id = 'databricks_default',
    job_id = JOB_ID
  )

