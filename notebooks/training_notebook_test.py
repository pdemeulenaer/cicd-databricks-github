# Databricks notebook source
# Explanations

# COMMAND ----------

# from cicd_databricks_github.common import Job

import pandas as pd
import numpy as np
import mlflow
import json

from pyspark.sql.functions import *

# Import of Feature Store
from databricks import feature_store
from databricks.feature_store import FeatureLookup

# Import of Sklearn packages
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Import MLflow 
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn #mlflow.lightgbm
from mlflow.models.signature import infer_signature

# Import matplotlib packages
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import pylab
from pylab import *
import matplotlib.cm as cm
import matplotlib.mlab as mlab

# class SampleJob(Job):

# Custom function
def train(**kwargs):

    with open("../conf/model.json") as f:
        conf = json.load(f)

    # Read config file and necessary config fields
    model_conf = conf["model"]

    print(model_conf)   
    data_path = conf["data"]["data_path"]
    train_val_dataset = conf["data"]["train_val_dataset"]
    train_dataset = conf["data"]["train_dataset"]
    val_dataset = conf["data"]["val_dataset"]   
    experiment = conf["model"]["experiment_name"] 
    output_path = conf["data"]["output_path"]
    
    # Configuration of direct connection to Azure Blob storage (no mount needed)
    environment = "dev"  # ATTENTION !!!!
    blob_name = conf['workspace'][environment]['data-lake']
    account_name = conf['workspace'][environment]['azure-storage-account-name']
    storage_key = dbutils.secrets.get(scope = conf['workspace'][environment]['storage-secret-scope'], 
                                      key = conf['workspace'][environment]['storage-secret-scope-key'])
    spark.conf.set("fs.azure.account.key."+account_name+".blob.core.windows.net", storage_key)
    cwd = "wasbs://"+blob_name+"@"+account_name+".blob.core.windows.net/"
    
    # Define the MLFlow experiment location
    mlflow.set_experiment(experiment)    

    print()
    print("-----------------------------------")
    print("         Model Training            ")
    print("-----------------------------------")
    print()

    # ==============================
    # 1.0 Data Loading
    # ==============================
    # try:
    
    # Load the raw data and associated label tables
    raw_data = spark.read.format('delta').load(cwd + 'raw_data')
    labels = spark.read.format('delta').load(cwd + 'labels')
    
    # Joining raw_data and labels
    raw_data_with_labels = raw_data.join(labels, ['Id','hour'])
    display(raw_data_with_labels)
    
    # Selection of the data and labels until last LARGE time step (e.g. day or week let's say)
    # Hence we will remove the last large timestep of the data
    # max_hour = raw_data_with_labels.select("hour").rdd.max()[0]
    max_date = raw_data_with_labels.select("date").rdd.max()[0]
    print(max_date)
    # raw_data_with_labels = raw_data_with_labels.withColumn("filter_out", when((col("hour")==max_hour) & (col("date")==max_date),"1").otherwise(0)) # don't take last hour of last day of data
    raw_data_with_labels = raw_data_with_labels.withColumn("filter_out", when(col("date")==max_date,"1").otherwise(0)) # don't take last day of data
    raw_data_with_labels = raw_data_with_labels.filter("filter_out==0").drop("filter_out")
    display(raw_data_with_labels)
    
#     logger.info("Step 1.0 completed: Loaded historical raw data and labels")   
      
    # except Exception as e:
    #     print("Errored on 1.0: data loading")
    #     print("Exception Trace: {0}".format(e))
    #     # print(traceback.format_exc())
    #     raise e  
    
    # ==================================
    # 1.1 Building the training dataset
    # ==================================
    # try:
    
    # Initialize the Feature Store client
    fs = feature_store.FeatureStoreClient()

    # Declaration of the Feature Store
    scaled_features_table = "feature_store_iris_example.scaled_features"

    # Declaration of the features, in a "feature lookup" object
    scaled_feature_lookups = [
        FeatureLookup( 
          table_name = scaled_features_table,
          feature_names = ["sl_norm","sw_norm","pl_norm","pw_norm"],
          lookup_key = ["Id","hour"],
        ),
    ]

    # Create the training dataset (includes the raw input data merged with corresponding features from feature table)
    exclude_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Id', 'hour','date'] # should I exclude the 'Id', 'hour','date'? 
    training_set = fs.create_training_set(
      raw_data_with_labels,
      feature_lookups = scaled_feature_lookups,
      label = "target",
      exclude_columns = exclude_columns
    )

    # Load the training dataset into a dataframe which can be passed into model training algo
    training_df = training_set.load_df()
    display(training_df)      

#     logger.info("Step 1.1 completed: Loaded features from the Feature Store")   
      
    # except Exception as e:
    #     print("Errored on 1.1: loading features from the feature store")
    #     print("Exception Trace: {0}".format(e))
    #     # print(traceback.format_exc())
    #     raise e    

    # ADD A TASK AROUND BUILDING AND SAVING TRAIN-TEST (SPLIT)
    # ==================================
    # 1.2 Create Train-Test data
    # ==================================
    # try:
    
    training_df = training_set.load_df()
    display(training_df)
    
    features_and_label = training_df.columns

    # Collect data into a Pandas array for training
    data_pd = training_df.toPandas()[features_and_label]

    train, test = train_test_split(data_pd, train_size=0.7, random_state=123)   #, stratify=y not working now
    x_train = train.drop(["target"], axis=1)
    x_test = test.drop(["target"], axis=1)
    y_train = train.target
    y_test = test.target
    
    # Save train dataset
    train_pd = pd.DataFrame(data=np.column_stack((x_train,y_train)), columns=[features_and_label])
    train_df = spark.createDataFrame(train_pd)
    train_df.write.option("header", "true").format("delta").mode("overwrite").save(cwd+"train_iris_dataset")
    
    # Save test dataset
    test_pd = pd.DataFrame(data=np.column_stack((x_test,y_test)), columns=[features_and_label])
    test_df = spark.createDataFrame(test_pd)
    test_df.write.option("header", "true").format("delta").mode("overwrite").save(cwd+"test_iris_dataset") 
      
    # except Exception as e:
    #     print("Errored on 1.2: create and save the train and test data")
    #     print("Exception Trace: {0}".format(e))
    #     # print(traceback.format_exc())
    #     raise e  

    # ========================================
    # 1.3 Model training
    # ========================================
    # try:
    
    with mlflow.start_run() as run:    
        mlflow.sklearn.autolog()
        print("Active run_id: {}".format(run.info.run_id))
        logger.info("Active run_id: {}".format(run.info.run_id))

        # Model definition


        base_estimator = RandomForestClassifier(oob_score = True,
                                                random_state=21,
                                                n_jobs=-1)   

        CV_rfc = GridSearchCV(estimator=base_estimator, 
                              param_grid=model_conf['hyperparameters_grid'],
                              cv=5)

        CV_rfc.fit(x_train, y_train)
        print(CV_rfc.best_params_)
        print(CV_rfc.best_score_)
        print(CV_rfc.best_estimator_)

        model = CV_rfc.best_estimator_
        
        # Inference on validation dataset
        y_val_pred = model.predict(x_val)    

        # Accuracy and Confusion Matrix
        accuracy = accuracy_score(y_val, y_val_pred)
        print('Accuracy = ',accuracy)
        print('Confusion matrix:')
        Classes = ['setosa','versicolor','virginica']
        C = confusion_matrix(y_val, y_val_pred)
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
        plt.savefig("confusion_matrix.png")
        
        # Log the model within the MLflow run

        # Tracking performance metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_figure(fig, "confusion_matrix.png")
        mlflow.set_tag("type", "CI run")   

        # Log the model (not registering in DEV !!!)
        # mlflow.sklearn.log_model(model, "model") #, registered_model_name="sklearn-rf")   
        
        # Register the model to MLflow MR as well as FS MR (not registering in DEV !!!!l!!!)
        fs.log_model(
          model,
          artifact_path="iris_model_packaged",
          flavor=mlflow.sklearn,
          training_set=training_set,
          registered_model_name="iris_model_packaged"
        ) 

#     logger.info("Step 1.3 completed: model training and saved to MLFlow")                

    # except Exception as e:
    #     print("Errored on step 1.3: model training")
    #     print("Exception Trace: {0}".format(e))
    #     print(traceback.format_exc())
    #     raise e                  

        

if __name__ == "__main__":
    train()

# COMMAND ----------



# COMMAND ----------



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

#EXAMPLE OF DIRECT CONNECTION TO BLOB STORAGE (no mount needed)
environment = "dev"
blob_name = conf['workspace'][environment]['data-lake']
account_name = conf['workspace'][environment]['azure-storage-account-name']
storage_key = dbutils.secrets.get(scope = conf['workspace'][environment]['storage-secret-scope'], 
                                  key = conf['workspace'][environment]['storage-secret-scope-key'])
spark.conf.set("fs.azure.account.key."+account_name+".blob.core.windows.net", storage_key)
cwd = "wasbs://"+blob_name+"@"+account_name+".blob.core.windows.net/"

# COMMAND ----------

# Load the raw data and associated label tables

# raw_data = spark.table("iris_data.raw_data")
# labels = spark.table("iris_data.labels")
raw_data = spark.read.format('delta').load(cwd + 'raw_data')
labels = spark.read.format('delta').load(cwd + 'labels')

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

exclude_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Id', 'hour','date'] # should I exclude the 'Id', 'hour','date'? 

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

# Actual Model training

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

print('t')

# COMMAND ----------


