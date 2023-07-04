# Full AutoML with PyCaret and Streamlit

This repository contains code for a full-blown AutoML pipeline using PyCaret and Streamlit. The pipeline automatically selects the best machine learning algorithm for a given dataset, and then deploys the model to a Streamlit app.

## Overview

The pipeline consists of the following steps:

1. **Data loading and cleaning:** The data is loaded from a CSV file and cleaned using PyCaret's `setup()` function.
2. **Feature engineering:** The features are engineered using PyCaret's `feature_engineering()` function.
3. **Model selection:** The best machine learning algorithm is selected using PyCaret's `create_model()` function.
4. **Model training:** The model is trained using PyCaret's `fit()` function.
5. **Model evaluation:** The model is evaluated using PyCaret's `evaluate()` function.
6. **Model deployment:** The model is deployed to a Streamlit app using PyCaret's `deploy()` function.

## Features

* The pipeline can be used to train models for a variety of supervised learning tasks, including classification and regression.
* The pipeline can be used to train models on a variety of datasets, including both categorical and numerical features.
* The pipeline is easy to use and can be used by beginners and experienced users alike.

## Benefits

* The pipeline can save you time and effort by automating the machine learning process.
* The pipeline can help you to improve the accuracy of your models.
* The pipeline can help you to deploy your models to production quickly and easily.

## Usage

To use the pipeline, you first need to install the necessary Python libraries. You can then clone the repository and run the `main.py` script. The script will guide you through the process of training a model and deploying it to a Streamlit app.

## Example

The following example shows how to use the pipeline to train a model for a classification task.

```python
import pycaret

# Load the dataset
dataset = pd.read_csv('data.csv')

# Create the pipeline
pipeline = pycaret.setup(dataset, target='target')

# Select the best model
model = pipeline.create_model('catboost')

# Train the model
model.fit()

# Evaluate the model
model.evaluate()

# Deploy the model
model.deploy()
