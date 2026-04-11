from setuptools import setup, find_packages

setup(
    name="automlease",
    version="0.3.0",
    author="Vikash Singh Rajput",
    author_email="vickybanna3327@gmail.com",
    description="Automatic Machine Learning for beginners — train ML models in 3 lines!",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vickybanna3327-byte/automlease",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
        "rich",
        "xgboost",
        "shap",
        "streamlit",
    ],
)