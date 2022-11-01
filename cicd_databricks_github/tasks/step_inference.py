from cicd_databricks_github.common import Task
from cicd_databricks_github import module

# General packages
import pandas as pd
import numpy as np
import json
from pyspark.sql.functions import *

# Databricks
import mlflow
from databricks import feature_store
from mlflow.tracking import MlflowClient

# Monitoring
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, ClassificationPerformanceProfileSection
from evidently.pipeline.column_mapping import ColumnMapping


class InferenceTask(Task):

    # def __init__(self):
    #     self.workspace = self.detect_workspace()

    # Custom function
    def _inference(self, **kwargs):
        """
        Model inference function
        """

        self.logger.info("Launching INFERENCE")

        listing = self.dbutils.fs.ls("dbfs:/")

        for l in listing:
            self.logger.info(f"DBFS directory: {l}")     

        # Read config file and necessary config fields
        model_conf = self.conf["model"]
        self.logger.info("model configs: {0}".format(model_conf))
        print(model_conf)   
        data_path = self.conf["data"]["data_path"]         
        inference_dataset = self.conf["data"]["inference_dataset"] 
        scored_inference_dataset = self.conf["data"]["scored_inference_dataset"] 
        output_path = self.conf["data"]["output_path"]
        model_name = self.conf["model"]["model_name"] 
        experiment = self.conf["model"]["experiment_name"] 
        registry_uri = self.conf['workspace'][self.workspace]['registry-uri']
        tracking_uri = self.conf['workspace'][self.workspace]['tracking-uri']       

        # Configuration of direct connection to Azure Blob storage (no mount needed)
        # Workspace should be one of "dev", "staging", "prod"
        # workspace = "dev"  # This is dynamically changed depending on workspace !!!!
        # workspace = self.detect_workspace() # done at the Job class level: self.workspace
        blob_name = self.conf['workspace'][self.workspace]['data-lake']
        account_name = self.conf['workspace'][self.workspace]['azure-storage-account-name']
        storage_key = dbutils.secrets.get(scope = self.conf['workspace'][self.workspace]['storage-secret-scope'], 
                                          key = self.conf['workspace'][self.workspace]['storage-secret-scope-key'])
        spark.conf.set("fs.azure.account.key."+account_name+".blob.core.windows.net", storage_key)
        cwd = "wasbs://"+blob_name+"@"+account_name+".blob.core.windows.net/"

        # Define the centralized registry
        # registry_uri = f'databricks://connection-to-data-workspace:data-workspace'
        mlflow.set_tracking_uri(tracking_uri) # BUG: is this working here?
        mlflow.set_registry_uri(registry_uri) # BUG: is this working here?
        
        # Define the MLFlow experiment location
        mlflow.set_experiment(experiment)    # note: the experiment will STILL be recorded to local MLflow instance!      

        # try:
        print()
        print("-----------------------------------")
        print("         Model Inference           ")
        print("-----------------------------------")
        print()

        # ==============================
        # 1.0 Data Loading
        # ==============================

        # data_df = self.spark.read.format("delta").load(data_path+inference_dataset)
        # data_pd = data_df.toPandas()

        # Load the raw data and associated label tables
        raw_data = spark.read.format('delta').load(cwd + 'raw_data')
        # labels = spark.read.format('delta').load(cwd + 'labels') # TODO: REMOVE LABELS HERE, WE ARE NOT USING THEM DURING THE INFERENCE !!!!
        
        # # Joining raw_data and labels  # TODO: REMOVE LABELS HERE, WE ARE NOT USING THEM DURING THE INFERENCE !!!!
        # raw_data_with_labels = raw_data.join(labels, ['Id','hour'])
        # display(raw_data_with_labels)
        
        # # Selection of the data and labels FROM last LARGE time step (e.g. day or week let's say)
        # # Hence we will TAKE the last large timestep of the data
        # # max_hour = raw_data_with_labels.select("hour").rdd.max()[0]
        # max_date = raw_data_with_labels.select("date").rdd.max()[0]
        # print(max_date)
        # # raw_data_with_labels = raw_data_with_labels.withColumn("filter_out", when((col("hour")==max_hour) & (col("date")==max_date),"1").otherwise(0)) # don't take last hour of last day of data
        # raw_data_with_labels = raw_data_with_labels.withColumn("filter_out", when(col("date")==max_date,"1").otherwise(0)) # don't take last day of data
        # raw_data_with_labels = raw_data_with_labels.filter("filter_out==1").drop("filter_out")
        # display(raw_data_with_labels)
        # raw_data = raw_data_with_labels.drop("target")

        max_date = raw_data.select("date").rdd.max()[0]
        print(max_date)
        raw_data = raw_data.withColumn("filter_out", when(col("date")==max_date,"1").otherwise(0)) # don't take last day of data
        raw_data = raw_data.filter("filter_out==1").drop("filter_out")
        display(raw_data)

        # print("Step 1.0 completed: Loaded Iris dataset in Pandas")
        self.logger.info("Step 1.0 completed: Loaded data to be inferred")   
          
        # except Exception as e:
        #     print("Errored on 1.0: data loading")
        #     print("Exception Trace: {0}".format(e))
        #     # print(traceback.format_exc())
        #     raise e    


        # try:
        # ========================================
        # 1.1 Model inference
        # ========================================
        
        # # Load model from MLflow experiment

        # # Initialize client
        # client = mlflow.tracking.MlflowClient()
        
        # # Extract the model in Staging mode from Model Registry
        # for mv in client.search_model_versions("name='{0}'".format(model_name)):
        #     if dict(mv)['current_stage'] == "Staging":
        #         model_dict = dict(mv)
        #         break   

        # print('Model extracted run_id: ', model_dict['run_id'])
        # print('Model extracted version number: ', model_dict['version'])
        # print('Model extracted stage: ', model_dict['current_stage']) 
        # print('Model path: ', model_dict['source'])            

        # # De-serialize the model
        # # mlflow_path = model_dict['source'] 
        # # model = mlflow.pyfunc.load_model(mlflow_path) # Load model as a PyFuncModel.                              
        # run_id = model_dict['run_id']
        # model_path = "runs:/{0}/model".format(run_id)
        # print('Model path from run_id: ', model_path)   
        # model = mlflow.pyfunc.load_model(model_path)

        # # Prediction
        # y_data_pred = model.predict(pd.DataFrame(x_data)) 

        # # Save scored inference dataset
        # data_scored_pd = pd.DataFrame(data=np.column_stack((x_data,y_data_pred)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label_scored'])
        # data_scored_pd.loc[data_scored_pd['label_scored']==0,'species'] = 'setosa'
        # data_scored_pd.loc[data_scored_pd['label_scored']==1,'species'] = 'versicolor'
        # data_scored_pd.loc[data_scored_pd['label_scored']==2,'species'] = 'virginica'
        # data_scored_df = spark.createDataFrame(data_scored_pd)
        # data_scored_df.write.format("delta").mode("overwrite").save(output_path+scored_inference_dataset)      

        # Initialize the Feature Store client
        fs = feature_store.FeatureStoreClient(feature_store_uri=registry_uri, model_registry_uri=registry_uri)

        # Get the model URI
        latest_model = module.get_latest_model_version(model_name,registry_uri)
        latest_model_version = int(latest_model.version)
        model_uri = f"models:/" + model_name + f"/{latest_model_version}"

        # Call score_batch to get the predictions from the model
        df_with_predictions = fs.score_batch(model_uri, raw_data)
        display(df_with_predictions)    

        # Write scored data
        df_with_predictions.write.format("delta").mode("overwrite").save(cwd+scored_inference_dataset)                  

        # print("Step 1.1 completed: model inference")  
        self.logger.info("Step 1.1 completed: model inference")                

        # except Exception as e:
        #     print("Errored on step 1.1: model inference")
        #     print("Exception Trace: {0}".format(e))
        #     print(traceback.format_exc())
        #     raise e   y


        # try:
        # ========================================
        # 1.2 Data monitoring
        # ========================================           

        # Extract the right version of the training dataset (as logged in MLflow)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
        run = client.get_run(latest_model.run_id)
        train_dataset_version = run.data.tags['train_dataset_version']
        train_dataset_path = run.data.tags['train_dataset_path']
        # test_dataset_version = run.data.tags['test_dataset_version']
        # fs_table_version = run.data.tags['fs_table_version']
        train_dataset = spark.read.format("delta").option("versionAsOf", train_dataset_version).load(train_dataset_path)
        train_dataset_pd = train_dataset.toPandas()

        train_dataset_pd.drop('target', inplace=True, axis=1)
        
        # Data drift calculation
        data_columns = ColumnMapping()
        data_columns.numerical_features = train_dataset_pd.columns #['sl_norm', 'sw_norm', 'pl_norm', 'pw_norm']
        data_drift_profile = Profile(sections=[DataDriftProfileSection()])
        df_with_predictions_pd = df_with_predictions.toPandas()
        print(train_dataset_pd.columns)
        print(df_with_predictions_pd.columns)
        data_drift_profile.calculate(train_dataset_pd, df_with_predictions_pd, column_mapping=data_columns) 
        data_drift_profile_dict = json.loads(data_drift_profile.json())
        print(data_drift_profile.json())
        print(data_drift_profile_dict['data_drift'])
        
        # Save the data monitoring to data lake 
        data_monitor_json = json.dumps(data_drift_profile_dict['data_drift'])
        data_monitor_df = spark.read.json(sc.parallelize([data_monitor_json]))
        display(data_monitor_df)
        data_monitor_df.write.option("header", "true").format("delta").mode("overwrite").save(cwd+"data_monitoring")

        self.logger.info("Step 1.2 completed: data monitoring")  

        # except Exception as e:
        #     print("Errored on step 1.2: data monitoring")
        #     print("Exception Trace: {0}".format(e))
        #     print(traceback.format_exc())
        #     raise e        


        # try:
        # ========================================
        # 1.3 Performance monitoring  (Here assumption of no delayed outcome!)
        # ========================================            

        # Extract the right version of the training dataset (as logged in MLflow)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
        run = client.get_run(latest_model.run_id)
        train_dataset_version = run.data.tags['train_dataset_version']
        train_dataset_path = run.data.tags['train_dataset_path']
        # test_dataset_version = run.data.tags['test_dataset_version']
        # fs_table_version = run.data.tags['fs_table_version']
        train_dataset = spark.read.format("delta").option("versionAsOf", train_dataset_version).load(train_dataset_path)
        train_dataset_pd = train_dataset.toPandas()

        # Load the target labels of the unseen data (the ones we tried to infer in step 1.1). Here is the assumption of no delayed outcome... 
        labels = spark.read.format('delta').load(cwd + 'labels')
        df_with_predictions = df_with_predictions.join(labels, ['Id','hour'])
        
        # Performance drift calculation
        data_columns = ColumnMapping()
        data_columns.target = 'target'
        data_columns.prediction = 'prediction'
        data_columns.numerical_features = train_dataset_pd.columns #['sl_norm', 'sw_norm', 'pl_norm', 'pw_norm']

        performance_drift_profile = Profile(sections=[ClassificationPerformanceProfileSection()])
        df_with_predictions_pd = df_with_predictions.toPandas()
        print(train_dataset_pd.columns)
        print(df_with_predictions_pd.columns)
        print(train_dataset_pd.head())
        print(df_with_predictions_pd.head())
        performance_drift_profile.calculate(train_dataset_pd, df_with_predictions_pd, column_mapping=data_columns) 
        performance_drift_profile_dict = json.loads(performance_drift_profile.json())
        print(performance_drift_profile.json())
        print(performance_drift_profile_dict)
        
        # Save the data monitoring to data lake 
        performance_monitor_json = json.dumps(performance_drift_profile_dict)
        performance_monitor_df = spark.read.json(sc.parallelize([performance_monitor_json]))
        print(performance_monitor_df)
        performance_monitor_df.write.option("header", "true").format("delta").mode("overwrite").save(cwd+"performance_monitoring")

        self.logger.info("Step 1.3 completed: performance monitoring")  

        # except Exception as e:
        #     print("Errored on step 1.2: data monitoring")
        #     print("Exception Trace: {0}".format(e))
        #     print(traceback.format_exc())
        #     raise e           

    def launch(self):
        self.logger.info("Launching inference task")
        self._inference()
        self.logger.info("Inference task finished!")  

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = InferenceTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
