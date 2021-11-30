# Databricks notebook source
import pandas as pd
import numpy as np
import mlflow
import json

# Import of Sklearn packages
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# COMMAND ----------

experiment='/Shared/simple-rf-sklearn/simple-rf-sklearn_experiment'
mlflow.set_experiment(experiment) 

# COMMAND ----------

# Load model from MLflow experiment

# Initialize client
client = mlflow.tracking.MlflowClient()

# Get experiment and runs 
exp  = client.get_experiment_by_name(experiment)
runs = mlflow.search_runs([exp.experiment_id], "", order_by=["metrics.Accuracy DESC"], max_results=1)
# best_run = runs[0]
best_run_id = runs["run_id"][0]

model_path = "runs:/{0}/model".format(best_run_id)
model = mlflow.pyfunc.load_model(model_path)
model


# COMMAND ----------


