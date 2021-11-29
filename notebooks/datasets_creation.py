# Databricks notebook source
# Imports

import pandas as pd
import numpy as np
# import mlflow
import json

#Import of SKLEARN packages
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# COMMAND ----------

# Loading of dataset
iris = load_iris()                  #The Iris dataset is available through the scikit-learn API
idx = list(range(len(iris.target)))
np.random.shuffle(idx)              #We shuffle it (important if we want to split in train and test sets)
X = iris.data[idx]
y = iris.target[idx]

# Load data in Pandas dataFrame
data_pd = pd.DataFrame(data=np.column_stack((X,y)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
data_pd.loc[data_pd['label']==0,'species'] = 'setosa'
data_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
data_pd.loc[data_pd['label']==2,'species'] = 'virginica'
data_pd.head()

# COMMAND ----------

df = spark.createDataFrame(data_pd)
df.write.format("delta").mode("overwrite").save("dbfs:/dbx/tmp/cicd_databricks_github/data/{0}".format('full_iris_dataset'))
df.show(5)

# COMMAND ----------

# Feature selection
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target       = 'label'   

X = data_pd[feature_cols].values
y = data_pd[target].values

# Creation of train and test datasets
x_train_val, x_test, y_train_val, y_test = train_test_split(X,y,train_size=0.7, stratify=y) #stratify=y ensures that the same proportion of labels are in both train and test sets! 

# Save train dataset
train_val_pd = pd.DataFrame(data=np.column_stack((x_train_val,y_train_val)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
train_val_pd.loc[train_val_pd['label']==0,'species'] = 'setosa'
train_val_pd.loc[train_val_pd['label']==1,'species'] = 'versicolor'
train_val_pd.loc[train_val_pd['label']==2,'species'] = 'virginica'
train_val_df = spark.createDataFrame(train_val_pd)
train_val_df.write.format("delta").mode("overwrite").save("dbfs:/dbx/tmp/cicd_databricks_github/data/{0}".format('train_val_iris_dataset'))

# Save test dataset
test_pd = pd.DataFrame(data=np.column_stack((x_test,y_test)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
test_pd.loc[test_pd['label']==0,'species'] = 'setosa'
test_pd.loc[test_pd['label']==1,'species'] = 'versicolor'
test_pd.loc[test_pd['label']==2,'species'] = 'virginica'
test_df = spark.createDataFrame(test_pd)
test_df.write.format("delta").mode("overwrite").save("dbfs:/dbx/tmp/cicd_databricks_github/data/{0}".format('test_iris_dataset'))

# COMMAND ----------

df = spark.read.format("delta").load("dbfs:/dbx/tmp/cicd_databricks_github/data/{0}".format('train_val_iris_dataset'))
df.show(3)

# COMMAND ----------

import boto3
from botocore.client import Config
ACCESS_KEY = 'YOUR_ACCESS_KEY'
SECRET_KEY = 'YOUR_SECRET_KEY'
AWS_BUCKET_NAME = "mwaa-environment-public-network-environmentbucket-6qum5wuwc0ed" #"BUCKET_NAME"

s3 = boto3.resource('s3', aws_access_key_id = ACCESS_KEY, aws_secret_access_key = SECRET_KEY, config = Config(signature_version = 's3v4') )

# s3.meta.client.upload_file( '/dbfs/FileStore/filename.csv', AWS_BUCKET_NAME, "filename.csv")

s3.meta.client.upload_file( '/dbfs/Shared/dbx/projects/cicd_databricks_github/59051c72079840d7baeb7596b5966eb5/artifacts/cicd_databricks_github/jobs/sample/entrypoint.py', AWS_BUCKET_NAME, "entrypoint.py")



# COMMAND ----------

databricks fs ls  dbfs:/Shared/dbx/projects/cicd_databricks_github/59051c72079840d7baeb7596b5966eb5/artifacts/cicd_databricks_github/jobs/sample

# COMMAND ----------

databricks fs -h

# COMMAND ----------


