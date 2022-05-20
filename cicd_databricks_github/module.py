# SOURCES: 
# https://www.stats.ox.ac.uk/~sejdinov/teaching/dmml17/Mixtures.html
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn import cluster, datasets
from sklearn.mixture import BayesianGaussianMixture
from mlflow.tracking import MlflowClient
from delta.tables import DeltaTable


def iris_data_generator(target_class='all',n_samples=10):
  '''
  This function is meant to generate random samples from a PDF fitted on Iris dataset using Bayesian GMM
  Input:
    - target_class: the desired target class to be generated. Options:
      - '0': for class 0
      - '1': for class 1
      - '2': for class 2
      - 'all': for a random mix of all classes (not available yet)
    - n_samples: the desired number of samples generated
  Output:
    - final_data_generated: the dataframe containing the generated samples (including the target label)
  '''

  # Loading the iris dataset
  iris = datasets.load_iris()
  iris_df = pd.DataFrame(iris.data,columns = iris.feature_names)
  iris_df['target'] = iris.target

  # Initialize the output dataframe
  final_data_generated = pd.DataFrame(columns = iris.feature_names)

  # Selecting the desired target class
  if target_class=='0': weights_target_class=[1,0,0]
  elif target_class=='1': weights_target_class=[0,1,0]
  elif target_class=='2': weights_target_class=[0,0,1]
  else: weights_target_class=[1./3.,1./3.,1./3.]
  
  # Now we need to generate samples for each of the 3 classes
  samples_per_class = random.choices([0,1,2], weights=weights_target_class, k=n_samples)

  # Target class id and counts per target class:
  class_id, counts_per_class = np.unique(samples_per_class, return_counts=True)

  # Looping on the 3 target classes
  for j,one_class_id in enumerate(class_id):

    # Extract the data of a given target class
    subset_df = iris_df[iris_df['target']==one_class_id]
    subset_df.drop('target', axis=1, inplace=True)

    # Fit the Bayesian GMM on the data
    n_components = 10 # Number of Gaussian components in the GMM model
    gmm = BayesianGaussianMixture(n_components=n_components,
                                  covariance_type='full', 
                                  # tol=0.00001, 
                                  # reg_covar=1e-06, 
                                  max_iter=20, 
                                  random_state=0, 
                                  n_init=10,
                                  # weight_concentration_prior=0.1
                                  )

    gmm.fit(subset_df.to_numpy()) 

    means = gmm.means_
    cov = gmm.covariances_
    weights = gmm.weights_

    # Compute the number of samples for each component of the GMM PDF
    # Indeed the GMM pdf is made of multiple Gaussian components.
    # So we sample each component respecting its own weight
    # The "counts" list is a list of the number of samples for each component
    component_samples = random.choices(population=np.arange(n_components),  # list to pick from
                                      weights=weights,  # weights of the population, in order
                                      k=counts_per_class[j]  # amount of samples to draw
                                      )
    # print(component_samples)

    component_id, counts_per_component = np.unique(component_samples, return_counts=True)
    # print(component_id, counts_per_component)

    # Generate the samples for each GMM components following the counts
    data_gen = np.random.multivariate_normal(means[component_id[0],:],cov[component_id[0]],counts_per_component[0]) 
    for i in range(1,len(component_id)):
      data_new = np.random.multivariate_normal(means[component_id[i],:],cov[component_id[i]],counts_per_component[i]) 
      data_gen = np.vstack((data_gen,data_new)) 
      del data_new 

    data_generated_per_class = pd.DataFrame(data_gen,columns = iris.feature_names)
    data_generated_per_class['target'] = one_class_id

    final_data_generated = pd.concat([final_data_generated, data_generated_per_class], axis=0, ignore_index=True)

  return final_data_generated


# def detect_workspace(spark):
#     """
#     This function detects in which workspace the code is running. It uses secrets stored in Azure KeyVault (following this tutorial: https://microsoft-bitools.blogspot.com/2020/02/use-azure-key-vault-for-azure-databricks.html)
    
#     :returns environment: (Str) the environment detected (either "dev", "staging", or "prod"). "Null" is returned in case of no workspace detection.
#     """
#     if dbutils.secrets.get(scope = "connection-to-datalakeblobstorage", key = "dev") == spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId"): 
#       environment = 'dev'
#     elif dbutils.secrets.get(scope = "connection-to-datalakeblobstorage", key = "staging") == spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId"):
#       environment = 'staging'
#     elif dbutils.secrets.get(scope = "connection-to-datalakeblobstorage", key = "prod") == spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId"):
#       environment = 'prod'
#     else:
#       print('NO WORKSPACE FOUND !!! ERROR')
#       environment = 'Null'
      
#     return environment


def get_latest_model_version(model_name,registry_uri):
    '''
    This function identifies the latest version of a model registered in the Model Registry
    :param model_name: (str) model name saved in Model Registry
    :param registry_uri: (str) uri of the Model Registry
    :return latest_model: (object) The last mlflow model registered to the Model Registry. Its parameters can be accessed such as:
    
    - latest_model.creation_timestamp
    - latest_model.current_stage
    - latest_model.description
    - latest_model.last_updated_timestamp
    - latest_model.name
    - latest_model.run_id
    - latest_model.run_link
    - latest_model.source
    - latest_model.status
    - latest_model.status_message
    - latest_model.tags
    - latest_model.user_id
    - latest_model.version
    '''
    latest_version = 1
    mlflow_client = MlflowClient(registry_uri=registry_uri)
    
    # Find last model version
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
        
    # Extract model of last version
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        if latest_version == int(mv.version):
            latest_model = mv
        
    return latest_model  


def get_delta_version(spark,delta_path):
    """
    Function to get the most recent version of a Delta table give the path to the Delta table
    
    :param delta_path: (str) path to Delta table
    :return delta_version: (int) Delta version
    """
    # DeltaTable is the main class for programmatically interacting with Delta tables
    delta_table = DeltaTable.forPath(spark, delta_path)
    # Get the information of the latest commits on this table as a Spark DataFrame. 
    # The information is in reverse chronological order.
    delta_table_history = delta_table.history() 
    
    # Retrieve the lastest Delta version - this is the version loaded when reading from delta_path
    delta_version = delta_table_history.first()["version"]
    
    return delta_version


def get_table_version(spark,table):
    """
    Function to get the most recent version of a Delta table (present in Hive metastore) given the path to the Delta table
    
    :param table: (str) Delta table name
    :return delta_version: (int) Delta table version
    """
    delta_version = spark.sql(f"SELECT MAX(version) as maxval FROM (DESCRIBE HISTORY {table})").first()[0]
    return delta_version


# def dummy_function(a):
#   '''
#   This function is doing nothing
#   '''
#   return print(a)
