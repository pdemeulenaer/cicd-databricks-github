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

from matplotlib import pyplot as plt

# COMMAND ----------

# Load model from MLflow experiment

experiment='/Shared/simple-rf-sklearn/simple-rf-sklearn_experiment'
mlflow.set_experiment(experiment) 

# Initialize client
client = mlflow.tracking.MlflowClient()

# Get experiment and runs 
exp  = client.get_experiment_by_name(experiment)
runs = mlflow.search_runs([exp.experiment_id], "", order_by=["metrics.Accuracy DESC"], max_results=1)
# best_run = runs[0]
best_run_id = runs["run_id"][0]
print(best_run_id)

model_path = "runs:/{0}/model".format(best_run_id)
model = mlflow.pyfunc.load_model(model_path)
model


# COMMAND ----------

test_df = spark.read.format("delta").load("dbfs:/dbx/tmp/cicd_databricks_github/data/{0}".format('test_iris_dataset'))
test_df.show(3)

# COMMAND ----------

# Feature selection
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'label'   

test_pd = test_df.toPandas()
x_test = test_pd[feature_cols].values
y_test = test_pd[target].values

# COMMAND ----------

y_test_pred = model.predict(pd.DataFrame(x_test)) 
y_test_pred

# COMMAND ----------

# Accuracy and Confusion Matrix
accuracy = accuracy_score(y_test, y_test_pred)
print('Accuracy = ',accuracy)
print('Confusion matrix:')
Classes = ['setosa','versicolor','virginica']
C = confusion_matrix(y_test, y_test_pred)
C_normalized = C / C.astype(np.float).sum()        
C_normalized_pd = pd.DataFrame(C_normalized,columns=Classes,index=Classes)
print(C_normalized_pd)   

# Figure plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(C,cmap='Blues')
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + Classes)
ax.set_yticklabels([''] + Classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# fig.savefig(output_path+'confusion_matrix_iris.png')

# COMMAND ----------

# with mlflow.start_run(best_run_id) as run:
#     mlflow.log_figure(fig, "test_confusion_matrix.png")
# artifact_uri = mlflow.get_artifact_uri(best_run_id)
# artifact_uri
# # MISSING LINK !!!

# mlflow.log_figure(fig, "test_confusion_matrix.png")
# mlflow.end_run()

# mlflow.set_tag("tag_test","test")
mlflow.log_figure(fig, "test_confusion_matrix.png")

# COMMAND ----------


