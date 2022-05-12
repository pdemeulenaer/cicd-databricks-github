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

# Import matplotlib packages
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import pylab
from pylab import *
import matplotlib.cm as cm
import matplotlib.mlab as mlab


class SampleJob(Job):

    # Custom function
    def validate(self, **kwargs):

        self.logger.info("Launching VALIDATION")

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
        model_name = self.conf["model"]["model_name"]            
        experiment = self.conf["model"]["experiment_name"] 
        output_path = self.conf["data"]["output_path"]
        minimal_threshold = self.conf["model"]["minimal_threshold"] 

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
        mlflow.set_experiment(experiment)       

        # try:
        print()
        print("-----------------------------------")
        print("         Model Validation          ")
        print("-----------------------------------")
        print()

        # ==============================
        # 1.0 Data Loading
        # ==============================

        # test_df = self.spark.read.format("delta").load(data_path+test_dataset) #"dbfs:/dbx/tmp/test/{0}".format('test_data_sklearn_rf'))
        test_df = self.spark.read.format('delta').load(cwd+"test_iris_dataset")
        test_pd = test_df.toPandas()

        # Feature selection
        feature_cols = ["sl_norm","sw_norm","pl_norm","pw_norm"] #['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        target = 'target'   

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
        # 1.1 Load model from MLflow
        # ========================================
        
        # Load model from MLflow experiment
        # Conditions:
        # - model accuracy should be higher than pre-defined threshold (defined in model.json)
        # - it should have the tag.type = CI, meaning it was trained during a CI run

        # Initialize client
        # client = mlflow.tracking.MlflowClient()
        client = mlflow.tracking.MlflowClient(tracking_uri=None, registry_uri=registry_uri)
        model_names = [m.name for m in client.list_registered_models()] # if m.name.startswith(prefix)]
        print(model_names)
        
        # # Get experiment and runs # NOT CONSIDERED SINCE ONLY LOCAL MODEL
        # exp  = client.get_experiment_by_name(experiment)
        # query = "tags.type = 'CI run' and metrics.accuracy >= {0}".format(minimal_threshold)
        # runs = mlflow.search_runs([exp.experiment_id], filter_string=query, order_by=["metrics.accuracy DESC"], max_results=1)
        # best_run_id = runs["run_id"][0]
        # model_path = "runs:/{0}/model".format(best_run_id)
        # model = mlflow.pyfunc.load_model(model_path)

        # model = mlflow.pyfunc.load_model(f'models://{scope}:{key}@databricks/{model3_name}/Staging')
        model = mlflow.pyfunc.load_model(f'models://connection-to-data-workspace:data-workspace@databricks/'+model_conf['model_name']+'/None')  
        # model = mlflow.pyfunc.load_model(model_path)

        version = client.get_latest_versions(model_conf['model_name'], ['None'])[0].version
        model_uri = client.get_model_version_download_uri(model_conf['model_name'], version)
        artifact_uri = client.get_model_version_download_uri(name, mv.version)
        run_id = artifact_uri.lstrip("runs:/")
        run_id = run_id.lstrip("/"+model_conf['model_name'])
        print(model_uri, version, artifact_uri, run_id)
                                
        # print("Step 1.1 completed: load model from MLflow")  
        self.logger.info("Step 1.1 completed: load model from MLflow")                

        # except Exception as e:
        #     print("Errored on step 1.1: model training")
        #     print("Exception Trace: {0}".format(e))
        #     print(traceback.format_exc())
        #     raise e      


        # try:
        # ========================================
        # 1.2 Model validation (and registration to MLflow model registry)
        # ========================================

        # # Note: Sparkish way of loading the model and doing the inference on a spark df. Put here for comparison
        # logged_model = f'models://connection-to-data-workspace:data-workspace@databricks/'+model_conf['model_name']+'/None'
        # loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

        # # Predict on a Spark DataFrame.
        # columns = ['sl_norm', 'sw_norm', 'pl_norm', 'pw_norm'] #list(test_df.columns)
        # test_df = test_df.withColumn('predictions', loaded_model(*columns))
        
        
        # Derive accuracy on TEST dataset
        y_test_pred = model.predict(test_pd[feature_cols]) 

        # Accuracy and Confusion Matrix
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print('TEST accuracy = ',test_accuracy)
        print('TEST Confusion matrix:')
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
        plt.savefig("confusion_matrix_TEST.png")    

        
        with mlflow.start_run(run_id) as run:

            # Tracking performance metrics on TEST dataset   
            mlflow.log_metric("accuracy_TEST", test_accuracy)
            mlflow.log_figure(fig, "confusion_matrix_TEST.png")  

            # IF we pass the validation, we REGISTER the model and push to Staging 
            print(f"Minimal accuracy threshold: {minimal_threshold:5.2f}")          
            if test_accuracy >= minimal_threshold: 
                mlflow.set_tag("validation", "passed")
                # model_uri = "runs:/{}/model".format(best_run_id)
                # mv = mlflow.register_model(model_uri, model_name)
                client.transition_model_version_stage(name=model_name, version=version, stage="Staging")

            else: 
                mlflow.set_tag("validation", "failed")
                        
        # print("Step 1.2 completed: model inference")  
        self.logger.info("Step 1.1 completed: model validation")                

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
    job.validate()
