# Databricks notebook source
# PLAN FOR THE DATA ENGINEERING PART

# I want to have a scheduled process that builds daily (or any "small" period) the features for the UNSEEN dataset

# As well, I want a scheduled process that builds weekly (or monthly, or any "large" period) the features for the TRAINING dataset

# What this notebook does?

# - It builds the tables "iris_data.raw_data" and "iris_data.labels"

# - It build the feature store table "feature_store_iris_example.scaled_features"

# COMMAND ----------

# Packages import

import numpy as np
import pandas as pd
import random
import uuid
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql.functions import *
from databricks import feature_store
from cicd_databricks_github import module

# COMMAND ----------

def scaled_features_fn(df):
    """
    Computes the scaled_features feature group.
    To restrict features to a time range, pass in ts_column, start_date, and/or end_date as kwargs.
    """

    pdf = df.toPandas()
    id = pdf['Id']
    date = pdf['date']
    hour = pdf['hour']
#     timestamp = pdf['timestamp']
#     unix_ts = pdf['unix_ts']
#     target = pdf['target']
    pdf.drop('Id', axis=1, inplace=True)
    pdf.drop('date', axis=1, inplace=True)
    pdf.drop('hour', axis=1, inplace=True)
#     pdf.drop('timestamp', axis=1, inplace=True)
#     pdf.drop('unix_ts', axis=1, inplace=True)

#     columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'Id'] #list(pdf.columns)

    scaler = MinMaxScaler()
    scaler.fit(pdf)
    pdf_norm=pdf.to_numpy()    #scaler.transform(pdf)
    columns=['sl_norm','sw_norm','pl_norm','pw_norm']
    pdf_norm = pd.DataFrame(data=pdf_norm, columns=columns)
    pdf_norm['sl_norm'] = pdf_norm['sl_norm'] * 2
    pdf_norm['sw_norm'] = pdf_norm['sw_norm'] * 2
    pdf_norm['pl_norm'] = pdf_norm['pl_norm'] * 2
    pdf_norm['pw_norm'] = pdf_norm['pw_norm'] * 2
    pdf_norm['Id'] = id
    pdf_norm['date'] = date
#     pdf_norm['timestamp'] = timestamp
#     pdf_norm['unix_ts'] = unix_ts
    pdf_norm['hour'] = hour
    
    return spark.createDataFrame(pdf_norm)

# COMMAND ----------

# Creation of a new data batch

# Initialize the dataframe
iris = load_iris()
iris_generated_all = pd.DataFrame(columns = iris.feature_names)

# Generate 50 sample randomly out of each target class
for target_class in [0,1,2]:
  iris_generated = module.iris_data_generator(target_class=str(target_class),n_samples=50)
  iris_generated_all = pd.concat([iris_generated_all, iris_generated], axis=0, ignore_index=True)

data_batch = spark.createDataFrame(iris_generated_all)
data_batch = data_batch.withColumnRenamed('sepal length (cm)','sepal_length')
data_batch = data_batch.withColumnRenamed('sepal width (cm)','sepal_width',)
data_batch = data_batch.withColumnRenamed('petal length (cm)','petal_length')
data_batch = data_batch.withColumnRenamed('petal width (cm)','petal_width',)

# data_batch = data_batch.withColumn('Id', monotonically_increasing_id())
# data_batch = data_batch.withColumn('month',lit('202203'))
data_batch = data_batch.withColumn('date',current_date())
data_batch = data_batch.withColumn('hour',hour(current_timestamp()))
# data_batch = data_batch.withColumn("timestamp",lit(current_timestamp()))
# data_batch = data_batch.withColumn("unix_ts",lit(unix_timestamp('timestamp')))

data_batch = data_batch.withColumn("Id",expr("uuid()"))

# display(data_batch)
data_batch.show(3) # IMPORTANT AS THIS ENFORCES THE COMPUTATION (e.g. forces the lazy computation to happen now)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS iris_data;

# COMMAND ----------

# %sql
# DROP TABLE iris_data.raw_data;
# DROP TABLE iris_data.labels;

# COMMAND ----------

raw_data_batch = data_batch.drop('target')
label_batch = data_batch.select('Id','hour','target')
display(raw_data_batch)
display(label_batch)

raw_data_batch.write.format("delta").mode("append").saveAsTable("iris_data.raw_data")
label_batch.write.format("delta").mode("append").saveAsTable("iris_data.labels")

# COMMAND ----------

# one = spark.table("iris_data.raw_data")
# two = spark.table("iris_data.labels")
# display(two)

# COMMAND ----------

# %sql
# DROP TABLE iris_data.raw_data;

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "5")

# Initialize the Feature Store client
fs = feature_store.FeatureStoreClient()

# Creation of the features
features_df = scaled_features_fn(raw_data_batch)
display(features_df)

# Feature store table name
fs_table = "feature_store_iris_example.scaled_features"

# If the table does not exists, create it
if not spark.catalog._jcatalog.tableExists(fs_table): 
  print('Created feature table: ', fs_table)
  fs.create_table(
      name=fs_table,
      primary_keys=["Id","hour"],
      df=features_df,
      partition_columns="date",
      description="Iris scaled Features",
  )
else:
  # Update the feature store table
  print('Updated feature table: ', fs_table)
  fs.write_table(
    name=fs_table,
    df=features_df,
    mode="merge",
  )

# COMMAND ----------

# %sql
# SELECT * FROM feature_store_iris_example.scaled_features VERSION AS OF 2

# COMMAND ----------

# %sql
# DROP TABLE feature_store_iris_example.scaled_features

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY feature_store_iris_example.scaled_features

# COMMAND ----------


