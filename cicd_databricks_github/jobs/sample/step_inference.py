from cicd_databricks_github.common import Job
from cicd_databricks_github import module

import pandas as pd
import numpy as np
import mlflow
import json
from pyspark.sql.functions import *

from databricks import feature_store


class SampleJob(Job):

    # Custom function
    def inference(self, **kwargs):

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

        # Configuration of direct connection to Azure Blob storage (no mount needed)
        environment = "dev"  # TODO: needs to be dynamically changed depending on platform !!!!
        blob_name = self.conf['workspace'][environment]['data-lake']
        account_name = self.conf['workspace'][environment]['azure-storage-account-name']
        storage_key = dbutils.secrets.get(scope = self.conf['workspace'][environment]['storage-secret-scope'], 
                                          key = self.conf['workspace'][environment]['storage-secret-scope-key'])
        spark.conf.set("fs.azure.account.key."+account_name+".blob.core.windows.net", storage_key)
        cwd = "wasbs://"+blob_name+"@"+account_name+".blob.core.windows.net/"

        # Define the centralized registry
        registry_uri = f'databricks://connection-to-data-workspace:data-workspace'
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

        # # Feature selection
        # feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  

        # x_data = data_pd[feature_cols].values


        # Load the raw data and associated label tables
        raw_data = spark.read.format('delta').load(cwd + 'raw_data')
        labels = spark.read.format('delta').load(cwd + 'labels')
        
        # Joining raw_data and labels
        raw_data_with_labels = raw_data.join(labels, ['Id','hour'])
        display(raw_data_with_labels)
        
        # Selection of the data and labels FROM last LARGE time step (e.g. day or week let's say)
        # Hence we will TAKE the last large timestep of the data
        # max_hour = raw_data_with_labels.select("hour").rdd.max()[0]
        max_date = raw_data_with_labels.select("date").rdd.max()[0]
        print(max_date)
        # raw_data_with_labels = raw_data_with_labels.withColumn("filter_out", when((col("hour")==max_hour) & (col("date")==max_date),"1").otherwise(0)) # don't take last hour of last day of data
        raw_data_with_labels = raw_data_with_labels.withColumn("filter_out", when(col("date")==max_date,"1").otherwise(0)) # don't take last day of data
        raw_data_with_labels = raw_data_with_labels.filter("filter_out==1").drop("filter_out")
        display(raw_data_with_labels)
        raw_data = raw_data_with_labels.drop("target")
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
        latest_model_version = module.get_latest_model_version(model_name)
        model_uri = f"models:/"+model_name+"/{latest_model_version}"

        # Call score_batch to get the predictions from the model
        df_with_predictions = fs.score_batch(model_uri, raw_data)
        display(df_with_predictions)    

        # Write scored data
        df_with_predictions.write.format("delta").mode("overwrite").save(cwd+scored_inference_dataset)                  

        # print("Step 1.1 completed: model inference")  
        self.logger.info("Step 1.1 completed: model inference")                

        # except Exception as e:
        #     print("Errored on step 1.1: model training")
        #     print("Exception Trace: {0}".format(e))
        #     print(traceback.format_exc())
        #     raise e                  


    def launch(self):
        self.logger.info("Launching sample job")

        listing = self.dbutils.fs.ls("dbfs:/")

        for l in listing:
            self.logger.info(f"DBFS directory: {l}")

        df = self.spark.range(0, 1000)

        df.write.format(self.conf["output_format"]).mode("overwrite").save(
            self.conf["output_path"]
        )

        self.logger.info("Sample job finished!")
        

if __name__ == "__main__":
    job = SampleJob()
    job.inference()
