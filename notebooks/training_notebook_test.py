# Databricks notebook source
# Explanations

# COMMAND ----------

# Imports

import pandas as pd
import numpy as np
import mlflow
import json

from pyspark.sql.functions import *

from databricks import feature_store
from databricks.feature_store import FeatureLookup

from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import lightgbm as lgb
import mlflow.lightgbm
import mlflow
import mlflow.sklearn #mlflow.lightgbm
from mlflow.models.signature import infer_signature

from sklearn.model_selection import GridSearchCV

# COMMAND ----------

# import sys
# import os
 
# # In the command below, replace <username> with your Databricks user name.
# sys.path.append(os.path.abspath('/Workspace/Repos/pdemeulenaer@outlook.com/cicd-databricks-github'))

# COMMAND ----------

# MAGIC %sh pwd

# COMMAND ----------

with open("../conf/model.json") as f:
    conf = json.load(f)
conf

# COMMAND ----------

# config_json = '''
# {
#     "data": {
#         "output_format": "delta",
#         "data_path": "dbfs:/dbx/tmp/cicd_databricks_github/data/",
#         "output_path":"dbfs:/dbx/tmp/cicd_databricks_github/output_data/",
#         "train_val_dataset":"train_val_iris_dataset",
#         "train_dataset":"train_iris_dataset",
#         "val_dataset":"val_iris_dataset",
#         "test_dataset":"test_iris_dataset",
#         "inference_dataset":"full_iris_dataset",
#         "scored_inference_dataset":"scored_full_iris_dataset"
#     },    
#     "model": {
#         "model_name": "IrisClassificationRF",
#         "experiment_name": "/Shared/simple-rf-sklearn/simple-rf-sklearn_experiment",
#         "hyperparameters_grid": {
#             "max_depth": [3,10],
#             "n_estimators": [30,50],
#             "max_features": ["auto"],
#             "criterion": ["gini","entropy"]      
#         },
#         "hyperparameters_fixed": {
#             "class_weight": "balanced",
#             "bootstrap": "True",
#             "random_state": "21"        
#         },
#         "minimal_threshold": "0.8"
#     }
# }
# '''

# conf = json.loads(config_json)


# Read config file and necessary config fields
model_conf = conf["model"]
# self.logger.info("model configs: {0}".format(model_conf))  
data_path = conf["data"]["data_path"]
train_val_dataset = conf["data"]["train_val_dataset"]
train_dataset = conf["data"]["train_dataset"]
val_dataset = conf["data"]["val_dataset"]   
experiment = conf["model"]["experiment_name"] 
output_path = conf["data"]["output_path"]

model_conf

# COMMAND ----------

# Load the raw data and associated label tables

raw_data = spark.table("iris_data.raw_data")
labels = spark.table("iris_data.labels")

# display(raw_data)
# display(labels)

# Joining raw_data and labels
raw_data_with_labels = raw_data.join(labels, ['Id','hour'])

display(raw_data_with_labels)

# Here I need to SELECT the data and labels until last LARGE time step (e.g. day or week let's say)
# Hence we will remove the last large timestep of the data
# max_hour = raw_data_with_labels.select("hour").rdd.max()[0]
max_date = raw_data_with_labels.select("date").rdd.max()[0]
# print(max_hour,max_date)
print(max_date)
# raw_data_with_labels = raw_data_with_labels.withColumn("filter_out", when((col("hour")==max_hour) & (col("date")==max_date),"1").otherwise(0)) # don't take last hour of last day of data
raw_data_with_labels = raw_data_with_labels.withColumn("filter_out", when(col("date")==max_date,"1").otherwise(0)) # don't take last day of data
raw_data_with_labels = raw_data_with_labels.filter("filter_out==0").drop("filter_out")

display(raw_data_with_labels)

# COMMAND ----------

# Build the training dataset

# Initialize the Feature Store client
fs = feature_store.FeatureStoreClient()

scaled_features_table = "feature_store_iris_example.scaled_features"

scaled_feature_lookups = [
    FeatureLookup( 
      table_name = scaled_features_table,
      feature_names = ["sl_norm","sw_norm","pl_norm","pw_norm"],
      lookup_key = ["Id","hour"],
    ),
]

exclude_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Id', 'hour','date']

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
training_set = fs.create_training_set(
  raw_data_with_labels,
  feature_lookups = scaled_feature_lookups,
  label = "target",
  exclude_columns = exclude_columns
)

# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()
display(training_df)

# COMMAND ----------

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run()
run = mlflow.active_run()
print("Active run_id: {}".format(run.info.run_id))

features_and_label = training_df.columns

# Collect data into a Pandas array for training
data = training_df.toPandas()[features_and_label]

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["target"], axis=1)
X_test = test.drop(["target"], axis=1)
y_train = train.target
y_test = test.target

# # with mlflow.start_run() as run:  
mlflow.sklearn.autolog()
base_estimator = RandomForestClassifier(oob_score = True,
                                        random_state=21,
                                        n_jobs=-1)   

CV_rfc = GridSearchCV(estimator=base_estimator, 
                      param_grid=model_conf['hyperparameters_grid'],
                      cv=5)

CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)
print(CV_rfc.best_score_)
print(CV_rfc.best_estimator_)

model = CV_rfc.best_estimator_

# Accuracy and Confusion Matrix
y_val_pred = model.predict(X_test) 
accuracy = accuracy_score(y_test, y_val_pred)
print(accuracy)

# COMMAND ----------

# Register the model to MLflow MR as well as FS MR

fs.log_model(
  model,
  artifact_path="iris_model_packaged",
  flavor=mlflow.sklearn,
  training_set=training_set,
  registered_model_name="iris_model_packaged"
) 

# COMMAND ----------


