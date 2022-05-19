from setuptools import find_packages, setup
from cicd_databricks_github import __version__

setup(
    name="cicd_databricks_github",
    packages=find_packages(exclude=["tests", "tests.*", "dags", "notebooks"]),
    setup_requires=["wheel","scikit-learn<1.1.0","evidently"],
    version=__version__,
    description="Databricks Labs CICD Templates Sample Project",
    author="Philippe de Meulenaer",
)
