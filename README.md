# SentryML Risk Engine

This project is a simple end‑to‑end machine learning system that predicts customer churn and monitors how the model behaves after it is deployed.

The goal of this project was not only to train a model but also to understand what happens after a model is running in production. In real systems the model is only one part of the workflow. Logging predictions, checking data drift, and monitoring model health are also important.

This project is an attempt to recreate that kind of real‑world setup in a simple way.


## What the project does

The system predicts the probability that a customer will churn based on their service usage and billing information. The prediction is exposed through a FastAPI service so it can be called like a normal API.

Each prediction is stored in a database. This makes it possible to track how the model behaves over time and analyze prediction patterns.

The project also includes a basic monitoring layer. The system compares new incoming data with statistics from the original training data to check if the data distribution changes too much.


## Main components


### Machine Learning model

The churn prediction model is trained using XGBoost.  
Some simple feature engineering is applied before prediction, such as calculating average charge per month.


### Prediction API

A FastAPI application exposes the model through a `/predict` endpoint.

The API receives customer information and returns a churn probability. FastAPI also provides an interactive interface where the API can be tested directly.


### Prediction logging

Every prediction request is written to a database.  
This allows the system to track how often the model predicts churn and how confident those predictions are.


### Drift monitoring

The project includes a small drift monitoring system.  
Statistics from the original training data are stored as a baseline. New data can be compared with this baseline to detect large changes.

Drift reports are saved so that model behaviour can be reviewed later.


### Model health checks

The API includes several endpoints that show basic monitoring information such as prediction counts, churn prediction rate, and drift status.


### Containerization

The project runs inside a Docker container so the environment stays consistent across machines.


### CI pipeline

GitHub Actions is used to run tests and verify that the project builds correctly whenever changes are pushed to the repository.


## Tech Stack

Python  
FastAPI  
Pandas  
NumPy  
Scikit‑learn  
XGBoost  
SQLite  
SQLAlchemy  
Docker  
GitHub Actions


## Example API Request

Once the API server is running, you can send a request to the `/predict` endpoint to get a churn prediction.

Example request:

```json
{
  "gender": "Female",
  "seniorcitizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 12,
  "phoneservice": "Yes",
  "multiplelines": "No",
  "internetservice": "DSL",
  "contract": "Month-to-month",
  "paymentmethod": "Electronic check",
  "monthlycharges": 70.5,
  "totalcharges": 845.5
}
```

You can also test the API directly using the FastAPI interactive documentation at:

```
http://localhost:8000/docs
```

This page allows you to try the `/predict` endpoint directly from the browser.


## Example API Response

A typical response from the API looks like this:

```json
{
  "customerid": "c12345",
  "churn_probability": 0.37,
  "churn_prediction": 0,
  "model_version": "xgb_v4",
  "threshold": 0.5
}
```

This response shows the predicted churn probability, the final churn decision, and the model version used for the prediction.


## Monitoring Endpoints

The API also includes several endpoints that help monitor the model after deployment.

Examples:

```
/health
/model-health
/prediction-stats
/metrics
```

These endpoints provide information about prediction activity, drift monitoring, and overall system health.


## How the System Works (Simple Overview)

1. A request is sent to the FastAPI `/predict` endpoint.
2. The API prepares the input data and performs feature engineering.
3. The trained XGBoost model generates a churn probability.
4. The prediction is logged in the database.
5. Monitoring endpoints track prediction behaviour and potential data drift.


## Possible Improvements

This project focuses on the core components of an end‑to‑end ML system.  
There are several directions where the system could be extended further:

 Deploy the API to a cloud environment such as AWS or GCP so it can be accessed publicly.

 Add monitoring dashboards to visualize metrics such as prediction volume, churn rate, and drift statistics.

 Introduce automated retraining so the model can periodically update when new data becomes available.
 
 Experiment with more advanced drift detection techniques for deeper monitoring of model behaviour.

These improvements would extend the system, but the current version already demonstrates the full lifecycle of a machine learning service: model training, API deployment, prediction logging, and monitoring.