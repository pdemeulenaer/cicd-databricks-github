from cicd_databricks_github.common import Job

import pandas as pd
import numpy as np
import mlflow
import json

#Import of SKLEARN packages
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


class SampleJob(Job):

    # Custom function
    def data_prep(self, **kwargs):

        self.logger.info("Launching DATA PREP job")

        listing = self.dbutils.fs.ls("dbfs:/")

        for l in listing:
            self.logger.info(f"DBFS directory: {l}")        

        # Define the MLFlow experiment location
        mlflow.set_experiment("/Shared/simple-rf-sklearn/simple-rf-sklearn_experiment")
        
        config_json = self.conf["model"]
        # config_json = '''{
        #     "hyperparameters": {
        #         "max_depth": "20",
        #         "n_estimators": "100",
        #         "max_features": "auto",
        #         "criterion": "gini",
        #         "class_weight": "balanced",
        #         "bootstrap": "True",
        #         "random_state": "21"        
        #     }
        # }'''
        self.logger.info("model configs: {0}".format(config_json))
        print(config_json)

        model_conf = json.loads(config_json)   
        self.logger.info("model configs: {0}".format(model_conf))
        print(model_conf)             

        # try:
        print()
        print("-----------------------------------")
        print("        Data preparation           ")
        print("-----------------------------------")
        print()

        # ==============================
        # 1.0 Data Loading
        # ==============================

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
        
        # Feature selection
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        target       = 'label'   
        
        X = data_pd[feature_cols].values
        y = data_pd[target].values

        # Creation of train and test datasets
        x_train, x_val, y_train, y_val = train_test_split(X,y,train_size=0.7, stratify=y) #stratify=y ensures that the same proportion of labels are in both train and test sets! 

        # Save train dataset
        train_pd = pd.DataFrame(data=np.column_stack((x_train,y_train)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
        train_pd.loc[data_pd['label']==0,'species'] = 'setosa'
        train_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
        train_pd.loc[data_pd['label']==2,'species'] = 'virginica'
        train_df = self.spark.createDataFrame(train_pd)
        train_df.write.format("delta").mode("overwrite").save("dbfs:/dbx/tmp/test/{0}".format('train_data_sklearn_rf'))

        # Save test dataset
        val_pd = pd.DataFrame(data=np.column_stack((x_val,y_val)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
        val_pd.loc[data_pd['label']==0,'species'] = 'setosa'
        val_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
        val_pd.loc[data_pd['label']==2,'species'] = 'virginica'
        test_df = self.spark.createDataFrame(val_pd)
        test_df.write.format("delta").mode("overwrite").save("dbfs:/dbx/tmp/test/{0}".format('val_data_sklearn_rf'))

        # print("Step 1.0 completed: Loaded Iris dataset in Pandas")  
        self.logger.info("Step 1.0 completed: Loaded Iris dataset in Pandas")    

        # except Exception as e:
        #     print("Errored on 1.0: data loading")
        #     print("Exception Trace: {0}".format(e))
        #     # print(traceback.format_exc())
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
    job.data_prep()
