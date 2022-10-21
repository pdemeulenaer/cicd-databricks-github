[![CI pipeline](https://github.com/pdemeulenaer/cicd-databricks-github/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/pdemeulenaer/cicd-databricks-github/actions/workflows/ci.yml)
[![CD to Staging](https://github.com/pdemeulenaer/cicd-databricks-github/actions/workflows/cd_staging.yml/badge.svg)](https://github.com/pdemeulenaer/cicd-databricks-github/actions/workflows/cd_staging.yml)
[![CD to Prod](https://github.com/pdemeulenaer/cicd-databricks-github/actions/workflows/cd_prod.yml/badge.svg?branch=main)](https://github.com/pdemeulenaer/cicd-databricks-github/actions/workflows/cd_prod.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=pdemeulenaer_cicd-databricks-github&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=pdemeulenaer_cicd-databricks-github)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=pdemeulenaer_cicd-databricks-github&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=pdemeulenaer_cicd-databricks-github)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=pdemeulenaer_cicd-databricks-github&metric=coverage)](https://sonarcloud.io/summary/new_code?id=pdemeulenaer_cicd-databricks-github)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=pdemeulenaer_cicd-databricks-github&metric=duplicated_lines_density)](https://sonarcloud.io/summary/new_code?id=pdemeulenaer_cicd-databricks-github)

# cicd-databricks-github

This repo is built on top of the cicd-templates from DatabricksLabs (Documentation [here](https://dbx.readthedocs.io/en/latest/templates/python_basic.html), originally in this Github [repo](https://github.com/databrickslabs/cicd-templates), generated via cookiecutter using an Azure platform context with Github Actions as CICD engine. 

It is adapted to flesh out the adoption of a git flow using Databricks. We take the example of a git flow where:

* Model coding and training by data scientists happen in feature branches (which are meant to be short-lived)

* The common "code-base" is the development branch, where each feature branch are pull-requested

* The release of specific versions happen in dedicated release branches

* Main branch is there for book-keeping, displaying the last version deployed

The idea is to trigger the CI during every PR from feature branches to the development branch. By running all the tests, the CI makes sure that the code landing on the development branch is healthy. Then, for each specific version of the code, the model can be deployed to a release branch in a PR, which will trigger the CD phase. 


## The CI/CD procedure:

Use case: to flesh out the use of the CI/CD, we use a simple random forest classifier (using scikit-learn) of the Iris dataset. Before running the CI/CD (or any experiments), the Train-Validation and Test datasets have to be generated out of the Iris dataset. This is done using the notebook /notebooks/datasets_creation.py. The information around the file structure and the model definition (parameters) is to be found in the /conf/model.json config file. 

CI:

* Unit tests (dummy from original repo, hence so far unrelated to the use case)

* Deploy & trigger training job. Training job made of 2 tasks:

  - task 1 (step_data_prep.py): data preparation task: Iris dataset is downloaded and split into train and test. Both are saved (test kept for validation step)
  - task 2 (step_training.py): training RF model (aka the "CI experiment"). Experiment saved to MLflow
  - task 3 (step_compare_performance.py): comparison of model performance to all experiments logged during the feature branch; Validate as a custom tag in MLflow for the CI experiment

Based on the CI experiment tag, reviewer will know if the PR is to be merged (also looking at results of unit tests, code analysis if such exists,...)  

CD:

* Deploy & trigger integration tests (dummy from original repo, hence so far unrelated to the use case)

* Validation job: run the scoring function on the test dataset

* Deploy the scheduled batch inference (here the scoring function is applied on "unseen" data) on either:

    - Databricks Jobs: using the "dbx deploy" approach, we can deploy the package as a Databricks Job, visible in the Jobs UI, and using the native Databricks Jobs Scheduling

    - AWS Managed Airflow: the idea is to delegate the scheduling of the Job to Airflow via a dag file, that is deployed to an S3 "dags/" bucket


## TODO list

* [Done] Build the TRAIN and TEST datasets BEFORE the data preparation task. It should be there even before the CI takes place.

* [Done] Store data path and model configs in json, and read them from each task file 

* [Done] Look for a way to trigger jobs from Airflow (follow the example from the Databricks documentation: https://docs.databricks.com/dev-tools/data-pipelines.html)

* [Done] Link the Github folder to a Databricks repo

* [Done] Add a folder to contain development notebooks

* [Done] Finish the inference on test dataset; reading the model from MLflow. In the simplistic case, we will use the best artifact among ALL experiments tracked in mlflow (later, we need to capture the latest mlflow experiment built during the CI execution). So that is a strong simplification so far.

* [TODO] update the code to dbx v0.7.6

* [TODO] Update the deployment file following Jobs API 2.1 instead of 2.0 convention

* [TODO] Update the databricks local configuration using Jobs API 2.1 instead of 2.0. To do this, run configure like
databricks jobs configure --version=2.1 (that will reconfigure ONLY the current environment in the .databrickcfg file)

* (optional) Get the name of the branch of the last commit (if in feature branch) as a tag in MLflow experiment. For the CI experiment, if passes tests, set the tag as "development"

* (optional) Add step in the CI the captures all experiments of the branch that is PR'ed into the development branch, and compare the performance of the new experiment, produced during the training that happens within the CI, as a safeguard (the performance should not deviate "too much")

* (optional) In the CD, load from MLflow the artifact from the experiment produced during the latest CI, and use it to perform validation test and schedule run in shadow mode

* (optional) Automate the copy of the dag file to S3 bucket

* (optional) Use a pool for clusters to keep them alive between tasks within a job. Then indicate in each task the id of the pool cluster, instead of the configuration of a new cluster for each task. This will speed up drastically execution of multi-task jobs

* (optional) Put all functions into a utils.py module that we can refer to in any file. 

* (optional) use Sonar and pylint within the CI part


## Original Readme (generated via cookiecutter)

This is a sample project for Databricks, generated via cookiecutter.

While using this project, you need Python 3.X and `pip` or `conda` for package management.

## Installing project requirements

```bash
pip install -r unit-requirements.txt
```

## Install project package in a developer mode

```bash
pip install -e .
```

## Testing

For local unit testing, please use `pytest`:
```
pytest tests/unit --cov
```

For an integration test on interactive cluster, use the following command:
```
dbx execute --cluster-name=<name of interactive cluster> --job=cicd-databricks-github-sample-integration-test
```

For a test on an automated job cluster, deploy the job files and then launch:
```
dbx deploy --jobs=cicd-databricks-github-sample-integration-test --files-only
dbx launch --job=cicd-databricks-github-sample-integration-test --as-run-submit --trace
```

## Interactive execution and development

1. `dbx` expects that cluster for interactive execution supports `%pip` and `%conda` magic [commands](https://docs.databricks.com/libraries/notebooks-python-libraries.html).
2. Please configure your job in `conf/deployment.json` file. 
2. To execute the code interactively, provide either `--cluster-id` or `--cluster-name`.
```bash
dbx execute \
    --cluster-name="<some-cluster-name>" \
    --job=job-name
```

Multiple users also can use the same cluster for development. Libraries will be isolated per each execution context.

## Preparing deployment file

Next step would be to configure your deployment objects. To make this process easy and flexible, we're using JSON for configuration.

By default, deployment configuration is stored in `conf/deployment.json`.

## Deployment for Run Submit API

To deploy only the files and not to override the job definitions, do the following:

```bash
dbx deploy --files-only
```

To launch the file-based deployment:
```
dbx launch --as-run-submit --trace
```

This type of deployment is handy for working in different branches, not to affect the main job definition.

## Deployment for Run Now API

To deploy files and update the job definitions:

```bash
dbx deploy
```

To launch the file-based deployment:
```
dbx launch --job=<job-name>
```

This type of deployment shall be mainly used from the CI pipeline in automated way during new release.


## CICD pipeline settings

Please set the following secrets or environment variables for your CI provider:
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`

## Testing and releasing via CI pipeline

- To trigger the CI pipeline, simply push your code to the repository. If CI provider is correctly set, it shall trigger the general testing pipeline
- To trigger the release pipeline, get the current version from the `cicd_databricks_github/__init__.py` file and tag the current code version:
```
git tag -a v<your-project-version> -m "Release tag for version <your-project-version>"
git push origin --tags
```

