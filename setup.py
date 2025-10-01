from setuptools import setup, find_packages

setup(
    name="stock_prediction_system",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "pandas", 
        "numpy",
        "scikit-learn",
        "matplotlib",
        "yfinance",
        "confluent-kafka",
        "absl-py",
    ],
    python_requires=">=3.8",
)