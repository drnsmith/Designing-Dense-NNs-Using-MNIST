from setuptools import setup, find_packages

setup(
    name="DesigningDenseNNs",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorflow",
        "matplotlib",
        "scikit-learn"
    ],
)
