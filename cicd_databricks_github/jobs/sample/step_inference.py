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
        train_val_dataset = self.conf["data"]["train_val_dataset"]
        train_dataset = self.conf["data"]["train_dataset"]
        val_dataset = self.conf["data"]["val_dataset"]  
        test_dataset = self.conf["data"]["test_dataset"]         
        experiment = self.conf["model"]["experiment_name"] 
        output_path = self.conf["data"]["output_path"]
        minimal_threshold = self.conf["model"]["minimal_threshold"] 

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

        test_df = self.spark.read.format("delta").load(data_path+test_dataset) #"dbfs:/dbx/tmp/test/{0}".format('test_data_sklearn_rf'))
        test_pd = test_df.toPandas()

        # Feature selection
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        target = 'label'   

        x_test = test_pd[feature_cols].values
        y_test = test_pd[target].values

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
        
        # Get experiment and runs 
        exp  = client.get_experiment_by_name(experiment)
        query = "tags.type = 'CI' and metrics.accuracy >= {0}".format(minimal_threshold)
        runs = mlflow.search_runs([exp.experiment_id], filter_string=query, order_by=["metrics.accuracy DESC"], max_results=1)
        best_run_id = runs["run_id"][0]

        model_path = "runs:/{0}/model".format(best_run_id)
        model = mlflow.pyfunc.load_model(model_path)
        y_test_pred = model.predict(pd.DataFrame(x_test))                    

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
