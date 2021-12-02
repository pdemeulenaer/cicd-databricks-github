from cicd_databricks_github.common import Job

import pandas as pd
import numpy as np
import mlflow
import json

# Import of Sklearn packages
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


class SampleJob(Job):

    # Custom function
    def train(self, **kwargs):

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

        # Define the MLFlow experiment location
        mlflow.set_experiment(experiment)       

        # try:
        print()
        print("-----------------------------------")
        print("         Model Inference           ")
        print("-----------------------------------")
        print()

        # ==============================
        # 1.0 Data Loading
        # ==============================

        data_df = self.spark.read.format("delta").load(data_path+inference_dataset)
        data_pd = data_df.toPandas()

        # Feature selection
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  

        x_data = data_pd[feature_cols].values

        # print("Step 1.0 completed: Loaded Iris dataset in Pandas")   
        self.logger.info("Step 1.0 completed: Loaded Iris dataset in Pandas")   
          
        # except Exception as e:
        #     print("Errored on 1.0: data loading")
        #     print("Exception Trace: {0}".format(e))
        #     # print(traceback.format_exc())
        #     raise e    

        # try:
        # ========================================
        # 1.1 Model inference
        # ========================================
        
        # Load model from MLflow experiment

        # Initialize client
        client = mlflow.tracking.MlflowClient()
        
        # Extract the model in Staging mode from Model Registry
        for mv in client.search_model_versions("name='{0}'".format(model_name)):
            if dict(mv)['current_stage'] == "Staging":
                model_dict = dict(mv)
                break   

        print('Model extracted run_id: ', model_dict['run_id'])
        print('Model extracted version number: ', model_dict['version'])
        print('Model extracted stage: ', model_dict['current_stage']) 
        print('Model path: ', model_dict['source'])            

        # De-serialize the model
        # mlflow_path = model_dict['source'] 
        # model = mlflow.pyfunc.load_model(mlflow_path) # Load model as a PyFuncModel.                              
        run_id = model_dict['run_id']
        model_path = "runs:/{0}/model".format(run_id)
        print('Model path from run_id: ', model_path)   
        model = mlflow.pyfunc.load_model(model_path)

        # Prediction
        y_data_pred = model.predict(pd.DataFrame(x_data)) 

        # Save scored inference dataset
        data_scored_pd = pd.DataFrame(data=np.column_stack((x_data,y_data_pred)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label_scored'])
        data_scored_pd.loc[data_scored_pd['label']==0,'species'] = 'setosa'
        data_scored_pd.loc[data_scored_pd['label']==1,'species'] = 'versicolor'
        data_scored_pd.loc[data_scored_pd['label']==2,'species'] = 'virginica'
        data_scored_df = spark.createDataFrame(data_scored_pd)
        data_scored_df.write.format("delta").mode("overwrite").save(output_path+scored_inference_dataset)                           

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
    job.train()
