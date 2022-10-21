import json
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from logging import Logger
from typing import Dict, Any

from pyspark.sql import SparkSession
import sys


# abstract class for jobs
class Job(ABC):

    def __init__(self, spark=None, init_conf=None):
        self.spark = self._prepare_spark(spark)
        self.logger = self._prepare_logger()
        self.dbutils = self.get_dbutils()
        if init_conf:
            self.conf = init_conf
        else:
            self.conf = self._provide_config()
        self._log_conf()
        self.workspace = self.detect_workspace()

    @staticmethod
    def _prepare_spark(spark) -> SparkSession:
        if not spark:
            return SparkSession.builder.getOrCreate()
        else:
            return spark

    @staticmethod
    def _get_dbutils(spark: SparkSession):
        try:
            from pyspark.dbutils import DBUtils # noqa
            if "dbutils" not in locals():
                utils = DBUtils(spark)
                return utils
            else:
                return locals().get("dbutils")
        except ImportError:
            return None

    def get_dbutils(self):
        utils = self._get_dbutils(self.spark)

        if not utils:
            self.logger.warn("No DBUtils defined in the runtime")
        else:
            self.logger.info("DBUtils class initialized")

        return utils

    def _provide_config(self):
        self.logger.info("Reading configuration from --conf-file job option")
        conf_file = self._get_conf_file()
        if not conf_file:
            self.logger.info(
                "No conf file was provided, setting configuration to empty dict."
                "Please override configuration in subclass init method"
            )
            return {}
        else:
            self.logger.info(
                f"Conf file was provided, reading configuration from {conf_file}"
            )
            return self._read_config(conf_file)

    @staticmethod
    def _get_conf_file():
        p = ArgumentParser()
        p.add_argument("--conf-file", required=False, type=str)
        namespace = p.parse_known_args(sys.argv[1:])[0]
        return namespace.conf_file

    def _read_config(self, conf_file) -> Dict[str, Any]:
        raw_content = "".join(
            self.spark.read.format("text").load(conf_file).toPandas()["value"].tolist()
        )
        config = json.loads(raw_content)
        return config

    def _prepare_logger(self) -> Logger:
        log4j_logger = self.spark._jvm.org.apache.log4j # noqa
        return log4j_logger.LogManager.getLogger(self.__class__.__name__)

    def _log_conf(self):
        # log parameters
        self.logger.info("Launching job with configuration parameters:")
        for key, item in self.conf.items():
            self.logger.info("\t Parameter: %-30s with value => %-30s" % (key, item))

    def detect_workspace(self):
        """
        This function detects in which workspace the code is running. It uses secrets stored in Azure KeyVault (following this tutorial: https://microsoft-bitools.blogspot.com/2020/02/use-azure-key-vault-for-azure-databricks.html)
        
        :returns environment: (Str) the environment detected (either "dev", "staging", or "prod"). "Null" is returned in case of no workspace detection.
        """
        if self.dbutils.secrets.get(scope = "connection-to-datalakeblobstorage", key = "dev") == self.spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId"): 
            environment = 'dev'
        elif self.dbutils.secrets.get(scope = "connection-to-datalakeblobstorage", key = "staging") == self.spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId"):
            environment = 'staging'
        elif self.dbutils.secrets.get(scope = "connection-to-datalakeblobstorage", key = "prod") == self.spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId"):
            environment = 'prod'
        else:
            print('NO WORKSPACE FOUND !!! ERROR')
            environment = 'Null'
        
        return environment

    @abstractmethod
    def launch(self):
        """
        Main method of the job.
        :return:
        """
        pass