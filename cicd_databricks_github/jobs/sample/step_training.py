# Model training code

from cicd_databricks_github.common import Job
from cicd_databricks_github import module

# General packages
import pandas as pd
import numpy as np
import mlflow
import json
from pyspark.sql.functions import *

# Import matplotlib packages
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import pylab
from pylab import *
import matplotlib.cm as cm
import matplotlib.mlab as mlab

# Sklearn packages
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Databricks packages
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn #mlflow.lightgbm
from mlflow.models.signature import infer_signature
from mlflow.tracking.artifact_utils import get_artifact_uri
from databricks import feature_store
from databricks.feature_store import FeatureLookup
# from delta.tables import DeltaTable


class SampleJob(Job):

    # Custom function
    def train(self, **kwargs):
        """
        Model training function
        """

        self.logger.info("Launching TRAINING")

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
        experiment = self.conf["model"]["experiment_name"] 
        output_path = self.conf["data"]["output_path"]
        
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
        
        self.logger.info("Step 1.0 completed: Loaded historical raw data and labels")   
          
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
        # fs = feature_store.FeatureStoreClient(feature_store_uri=registry_uri)
        fs = feature_store.FeatureStoreClient(feature_store_uri=registry_uri, model_registry_uri=registry_uri)

        # Declaration of the Feature Store
        fs_table = "feature_store_iris_prod.scaled_features"

        # Declaration of the features, in a "feature lookup" object
        scaled_feature_lookups = [
            FeatureLookup( 
              table_name = fs_table,
              feature_names = ["sl_norm","sw_norm","pl_norm","pw_norm"],
              lookup_key = ["Id","hour"],
            ),
        ]

        # Create the training dataset (includes the raw input data merged with corresponding features from feature table)
        exclude_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Id', 'hour','date'] # BUG should I exclude the 'Id', 'hour','date'? 
        training_set = fs.create_training_set(
          raw_data_with_labels,
          feature_lookups = scaled_feature_lookups,
          label = "target",
          exclude_columns = exclude_columns
        )

        # Load the training dataset into a dataframe which can be passed into model training algo
        training_df = training_set.load_df()
        display(training_df)

#         train_df = self.spark.read.format("delta").load(data_path+train_dataset) #"dbfs:/dbx/tmp/test/{0}".format('train_data_sklearn_rf'))
#         train_pd = train_df.toPandas()

#         val_df = self.spark.read.format("delta").load(data_path+val_dataset) #"dbfs:/dbx/tmp/test/{0}".format('train_data_sklearn_rf'))
#         val_pd = val_df.toPandas()        

#         # Feature selection
#         feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#         target = 'label'   

#         x_train = train_pd[feature_cols].values
#         y_train = train_pd[target].values

#         x_val = val_pd[feature_cols].values
#         y_val = val_pd[target].values        
 
        self.logger.info("Step 1.1 completed: Loaded features from the Feature Store")   
          
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
        
        # Collect data into a Pandas array for training
        features_and_label = training_df.columns
        data_pd = training_df.toPandas()[features_and_label]

        train, test = train_test_split(data_pd, train_size=0.7, random_state=123)   # BUG: stratify=y not working now
        x_train = train.drop(["target"], axis=1)
        x_test = test.drop(["target"], axis=1)
        y_train = train.target
        y_test = test.target
        
        # Save train dataset
        train_pd = pd.DataFrame(data=np.column_stack((x_train,y_train)), columns=features_and_label)
        train_df = spark.createDataFrame(train_pd)
        train_df.write.option("header", "true").format("delta").mode("overwrite").save(cwd+"train_iris_dataset")
        
        # Save test dataset
        test_pd = pd.DataFrame(data=np.column_stack((x_test,y_test)), columns=features_and_label)
        test_df = spark.createDataFrame(test_pd)
        test_df.write.option("header", "true").format("delta").mode("overwrite").save(cwd+"test_iris_dataset")

#         train_df = self.spark.read.format("delta").load(data_path+train_dataset) #"dbfs:/dbx/tmp/test/{0}".format('train_data_sklearn_rf'))
#         train_pd = train_df.toPandas()

#         val_df = self.spark.read.format("delta").load(data_path+val_dataset) #"dbfs:/dbx/tmp/test/{0}".format('train_data_sklearn_rf'))
#         val_pd = val_df.toPandas()        

#         # Feature selection
#         feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#         target = 'label'   

#         x_train = train_pd[feature_cols].values
#         y_train = train_pd[target].values

#         x_val = val_pd[feature_cols].values
#         y_val = val_pd[target].values        
 
        self.logger.info("Step 1.2 completed: Create and save the train and test data")   
          
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
            self.logger.info("Active run_id: {}".format(run.info.run_id))

            # Model definition
#             max_depth = int(model_conf['hyperparameters']['max_depth'])
#             n_estimators = int(model_conf['hyperparameters']['n_estimators'])
#             max_features = model_conf['hyperparameters']['max_features']
#             criterion = model_conf['hyperparameters']['criterion']
#             class_weight = model_conf['hyperparameters']['class_weight']
#             bootstrap = bool(model_conf['hyperparameters']['bootstrap'])
#             clf = RandomForestClassifier(max_depth=max_depth,
#                                     n_estimators=n_estimators,
#                                     max_features=max_features,
#                                     criterion=criterion,
#                                     class_weight=class_weight,
#                                     bootstrap=bootstrap,
#                                     random_state=21,
#                                     n_jobs=-1)          
            
#             # Fit of the model on the training set
#             model = clf.fit(x_train, y_train) 

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

            signature = infer_signature(x_train, model.predict(x_train))
            
            # # Inference on validation dataset
            # y_val_pred = model.predict(x_val)    

            # # Accuracy and Confusion Matrix
            # accuracy = accuracy_score(y_val, y_val_pred)
            # print('Accuracy = ',accuracy)
            # print('Confusion matrix:')
            # Classes = ['setosa','versicolor','virginica']
            # C = confusion_matrix(y_val, y_val_pred)
            # C_normalized = C / C.astype(np.float).sum()        
            # C_normalized_pd = pd.DataFrame(C_normalized,columns=Classes,index=Classes)
            # print(C_normalized_pd)   

            # # Figure plot
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # cax = ax.matshow(C,cmap='Blues')
            # plt.title('Confusion matrix of the classifier')
            # fig.colorbar(cax)
            # ax.set_xticklabels([''] + Classes)
            # ax.set_yticklabels([''] + Classes)
            # plt.xlabel('Predicted')
            # plt.ylabel('True')
            # plt.savefig("confusion_matrix.png")
            
            # # Tracking performance metrics
            # mlflow.log_metric("accuracy", accuracy)
            # mlflow.log_figure(fig, "confusion_matrix.png")
            mlflow.set_tag("type", "CI run")  

            # Tracking the data
            # train_dataset_version = self.get_delta_version(cwd+"train_iris_dataset")
            # test_dataset_version = self.get_delta_version(cwd+"test_iris_dataset")
            train_dataset_version = module.get_delta_version(cwd+"train_iris_dataset")
            test_dataset_version = module.get_delta_version(cwd+"test_iris_dataset")
            # fs_table_version = self.get_table_version(fs_table)
            fs_table_version = module.get_table_version(fs_table)
            mlflow.set_tag("train_dataset_version", train_dataset_version)
            mlflow.set_tag("test_dataset_version", test_dataset_version)
            mlflow.set_tag("fs_table_version", fs_table_version)
            mlflow.set_tag("train_dataset_path", cwd+"train_iris_dataset")
            mlflow.set_tag("test_dataset_path", cwd+"test_iris_dataset")
            mlflow.set_tag("raw_data_path", cwd + 'raw_data')
            mlflow.set_tag("raw_labels_path", cwd + 'labels')            

            # Log the model (not registering in DEV !!!)
            # mlflow.sklearn.log_model(model, "model") #, registered_model_name="sklearn-rf")   
            
            input_example = {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
            
            # Register the model to MLflow MR as well as FS MR (should not register in DEV !!!!!!)
            fs.log_model(
            model,
            artifact_path=model_conf['model_name'],
            flavor=mlflow.sklearn,
            training_set=training_set,
            # registered_model_name=model_conf['model_name'],
            )
            
            # Register the model to the CENTRALIZED MLflow MR
            mlflow.set_registry_uri(registry_uri)
            print(mlflow.get_registry_uri())
            mlflow.sklearn.log_model(model, 
                                    model_conf['model_name'],
                                    registered_model_name=model_conf['model_name'],
                                    signature=signature,
                                    input_example=input_example)           

            self.logger.info("Step 1.3 completed: model training and saved to MLFlow")                

        # except Exception as e:
        #     print("Errored on step 1.3: model training")
        #     print("Exception Trace: {0}".format(e))
        #     print(traceback.format_exc())
        #     raise e                  


    # def get_delta_version(self,delta_path):
    #     """
    #     Function to get the most recent version of a Delta table give the path to the Delta table
        
    #     :param delta_path: (str) path to Delta table
    #     :return: Delta version (int)
    #     """
    #     # DeltaTable is the main class for programmatically interacting with Delta tables
    #     delta_table = DeltaTable.forPath(spark, delta_path)
    #     # Get the information of the latest commits on this table as a Spark DataFrame. 
    #     # The information is in reverse chronological order.
    #     delta_table_history = delta_table.history() 
        
    #     # Retrieve the lastest Delta version - this is the version loaded when reading from delta_path
    #     delta_version = delta_table_history.first()["version"]
        
    #     return delta_version


    # def get_table_version(self,table):
    #     """
    #     Function to get the most recent version of a Delta table (present in Hive metastore) given the path to the Delta table
        
    #     :param table: (str) Delta table name
    #     :return: Delta version (int)
    #     """
    #     delta_version = spark.sql(f"SELECT MAX(version) as maxval FROM (DESCRIBE HISTORY {table})").first()[0]
    #     return delta_version

        
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



