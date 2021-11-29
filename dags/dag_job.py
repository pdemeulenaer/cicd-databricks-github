# #
# # Licensed to the Apache Software Foundation (ASF) under one
# # or more contributor license agreements.  See the NOTICE file
# # distributed with this work for additional information
# # regarding copyright ownership.  The ASF licenses this file
# # to you under the Apache License, Version 2.0 (the
# # "License"); you may not use this file except in compliance
# # with the License.  You may obtain a copy of the License at
# #
# #   http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing,
# # software distributed under the License is distributed on an
# # "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# # KIND, either express or implied.  See the License for the
# # specific language governing permissions and limitations
# # under the License.
# """
# This is an example DAG which uses the DatabricksSubmitRunOperator.
# In this example, we create two tasks which execute sequentially.
# The first task is to run a notebook at the workspace path "/test"
# and the second task is to run a JAR uploaded to DBFS. Both,
# tasks use new clusters.
# Because we have set a downstream dependency on the notebook task,
# the spark jar task will NOT run until the notebook task completes
# successfully.
# The definition of a successful run is if the run has a result_state of "SUCCESS".
# For more information about the state of a run refer to
# https://docs.databricks.com/api/latest/jobs.html#runstate
# """

# from datetime import datetime

# from airflow import DAG
# from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator

# from airflow.contrib.hooks.databricks_hook import DatabricksHook
# from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator

# def get_job_id_by_name(job_name: str, databricks_conn_id: str) -> str:
#     list_endpoint = ('GET', 'api/2.0/jobs/list')
#     hook = DatabricksHook(databricks_conn_id=databricks_conn_id)
#     response_payload = hook._do_api_call(list_endpoint, {})
#     all_jobs = response_payload.get("jobs", [])
#     matching_jobs = [j for j in all_jobs if j["settings"]["name"] == job_name]

#     if not matching_jobs:
#         raise Exception(f"Job with name {job_name} not found")

#     if len(matching_jobs) > 1:
#         raise Exception(f"Job with name {job_name} is duplicated. Please make job name unique in Databricks UI.")

#     job_id = matching_jobs[0]["job_id"]
#     return job_id



# job_id = get_job_id_by_name("cicd-databricks-github-sample", "some-databricks-conn-id")
# operator = DatabricksRunNowOperator(
#     job_id=job_id,
#     # add your arguments
# )    

# with DAG(
#     dag_id='pmeu_test_dag',
#     schedule_interval='@daily',
#     start_date=datetime(2021, 1, 1),
#     tags=['example'],
#     catchup=False,
# ) as dag:
#     new_cluster = {
# #        9.1 LTS (includes Apache Spark 3.1.2, Scala 2.12)
#         'spark_version': '9.1.x-cpu-ml-scala2.12',
#         'node_type_id': 'i3.xlarge',
#         'aws_attributes': {'availability': 'ON_DEMAND'},
#         'num_workers': 2,
#     }

#     notebook_task_params = {
#         'new_cluster': new_cluster,
#         'notebook_task': {
#             'notebook_path': '/Users/pmeu@danskebank.lt/test',
#         },
#     }
#     # [START howto_operator_databricks_json]
#     # Example of using the JSON parameter to initialize the operator.
#     notebook_task = DatabricksSubmitRunOperator(task_id='notebook_task', json=notebook_task_params)
#     # [END howto_operator_databricks_json]

#     # [START howto_operator_databricks_named]
#     # Example of using the named parameters of DatabricksSubmitRunOperator
#     # to initialize the operator.
#     #spark_jar_task = DatabricksSubmitRunOperator(
#     #    task_id='spark_jar_task',
#     #    new_cluster=new_cluster,
#     #    spark_jar_task={'main_class_name': 'com.example.ProcessData'},
#     #    libraries=[{'jar': 'dbfs:/lib/etl-0.1.jar'}],
#     #)
#     # [END howto_operator_databricks_named]
#     notebook_task 

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


JOB_ID = get_job_id_by_name("cicd-databricks-github-sample", 'databricks_default')

default_args = {
  'owner': 'airflow'
}

with DAG('pmeu_test_job_example',
  start_date = days_ago(2),
  schedule_interval = None,
  default_args = default_args
  ) as dag:

  opr_run_now = DatabricksRunNowOperator(
    task_id = 'run_now',
    databricks_conn_id = 'databricks_default',
    job_id = JOB_ID
  )