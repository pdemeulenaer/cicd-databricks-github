from cicd_databricks_github.common import Task

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


class TransitionToProdTask(Task):

    # Custom function
    def _transition_to_prod(self, **kwargs):
        """
        Model transition to prod function
        """

        self.logger.info("Launching Transition to Prod")

        listing = self.dbutils.fs.ls("dbfs:/")

        for l in listing:
            self.logger.info(f"DBFS directory: {l}")        

        # Read config file and necessary config fields
        model_conf = self.conf["model"]
        self.logger.info("model configs: {0}".format(model_conf))
        print(model_conf)    
        model_name = self.conf["model"]["model_name"] 
        experiment = self.conf["model"]["experiment_name"] 
        registry_uri = self.conf['workspace'][self.workspace]['registry-uri']
        tracking_uri = self.conf['workspace'][self.workspace]['tracking-uri']
        output_path = self.conf["data"]["output_path"]

       # Define the centralized registry
        # registry_uri = f'databricks://connection-to-data-workspace:data-workspace'
        mlflow.set_tracking_uri(tracking_uri) # BUG: is this working here?
        mlflow.set_registry_uri(registry_uri) # BUG: is this working here?
        
        # Define the MLFlow experiment location
        mlflow.set_experiment(experiment)    # note: the experiment will STILL be recorded to local MLflow instance!      

        # try:
        # ========================================
        # 1.0 Model transition to prod
        # ========================================
        
        # Initialize client
        # client = mlflow.tracking.MlflowClient()
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
        model_names = [m.name for m in client.list_registered_models()]
        print(model_names)

        # Extracting model information
        mv = client.get_latest_versions(model_name, ['Staging'])[0]
        version = mv.version
        run_id = mv.run_id
        artifact_uri = client.get_model_version_download_uri(model_name, version)
        print(version, artifact_uri, run_id)

        # Model transition to prod
        client.transition_model_version_stage(name=model_name, version=version, stage="Production")
                                
        # print("Step 1.0 completed: model transition to prod")  
        self.logger.info("Step 1.0 completed: model transition to prod")                

        # except Exception as e:
        #     print("Errored on step 1.0: model transition to prod")
        #     print("Exception Trace: {0}".format(e))
        #     print(traceback.format_exc())
        #     raise e                          

    def launch(self):
        self.logger.info("Launching transition to prod task")
        self._transition_to_prod()
        self.logger.info("Transition to prod task finished!")  

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = TransitionToProdTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
