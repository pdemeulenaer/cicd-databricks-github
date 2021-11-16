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
    def train(self, **kwargs):

        self.logger.info("Launching TRAIN job")

        listing = self.dbutils.fs.ls("dbfs:/")

        for l in listing:
            self.logger.info(f"DBFS directory: {l}")        

        # Define the MLFlow experiment location
        mlflow.set_experiment("/Shared/simple-rf-sklearn/simple-rf-sklearn_experiment")
        
        config_json = '''{
            "hyperparameters": {
                "max_depth": "20",
                "n_estimators": "100",
                "max_features": "auto",
                "criterion": "gini",
                "class_weight": "balanced",
                "bootstrap": "True",
                "random_state": "21"        
            }
        }'''

        model_conf = json.loads(config_json)        

        # try:
        print()
        print("-----------------------------------")
        print("         Model Training            ")
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
        x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.7, stratify=y) #stratify=y ensures that the same proportion of labels are in both train and test sets! 
        
        # Save test dataset
        test_pd = pd.DataFrame(data=np.column_stack((x_test,y_test)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
        test_pd.loc[data_pd['label']==0,'species'] = 'setosa'
        test_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
        test_pd.loc[data_pd['label']==2,'species'] = 'virginica'
        test_df = self.spark.createDataFrame(test_pd)
        test_df.write.format("delta").mode("overwrite").save("dbfs:/dbx/tmp/test/{0}".format('test_data_sklearn_rf'))

        print("Step 1.0 completed: Loaded Iris dataset in Pandas")      

        # except Exception as e:
        #     print("Errored on 1.0: data loading")
        #     print("Exception Trace: {0}".format(e))
        #     # print(traceback.format_exc())
        #     raise e    

        # try:
        # ========================================
        # 1.1 Model training
        # ========================================
        
        with mlflow.start_run() as run:          

            # Model definition
            max_depth = int(model_conf['hyperparameters']['max_depth'])
            n_estimators = int(model_conf['hyperparameters']['n_estimators'])
            max_features = model_conf['hyperparameters']['max_features']
            criterion = model_conf['hyperparameters']['criterion']
            class_weight = model_conf['hyperparameters']['class_weight']
            bootstrap = bool(model_conf['hyperparameters']['bootstrap'])
            clf = RandomForestClassifier(max_depth=max_depth,
                                    n_estimators=n_estimators,
                                    max_features=max_features,
                                    criterion=criterion,
                                    class_weight=class_weight,
                                    bootstrap=bootstrap,
                                    random_state=21,
                                    n_jobs=-1)          
            
            # Fit of the model on the training set
            model = clf.fit(x_train, y_train) 
            
            # Log the model within the MLflow run
            mlflow.log_param("max_depth", str(max_depth))
            mlflow.log_param("n_estimators", str(n_estimators))  
            mlflow.log_param("max_features", str(max_features))             
            mlflow.log_param("criterion", str(criterion))  
            mlflow.log_param("class_weight", str(class_weight))  
            mlflow.log_param("bootstrap", str(bootstrap))  
            mlflow.log_param("max_features", str(max_features)) 
            mlflow.sklearn.log_model(model, 
                                "model",
                                registered_model_name="sklearn-rf")                        

        print("Step 1.1 completed: model training and saved to MLFlow")                  

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
    # job.launch()
    job.train()
