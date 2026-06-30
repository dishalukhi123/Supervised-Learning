# Credit Card Fraud Detection - Summary Report

## Project Overview

This project detects fraudulent credit card transactions using Machine
Learning. The objective is to identify fraudulent transactions while
minimizing false positives and maximizing recall.

## Dataset

-   Dataset: Credit Card Fraud Detection
-   Features: Time, Amount, V1--V28 (anonymized), Class
-   Target:
    -   **0** = Legitimate Transaction
    -   **1** = Fraudulent Transaction

## Workflow

1.  Data Loading
2.  Data Cleaning
3.  Exploratory Data Analysis (EDA)
4.  Feature Scaling
5.  Train/Test Split
6.  Model Training
7.  Threshold Optimization
8.  Model Evaluation
9.  Streamlit Dashboard Deployment

## Models Evaluated

-   Logistic Regression
-   Decision Tree
-   Random Forest
-   XGBoost (optional)

## Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   F1-Score
-   ROC-AUC
-   PR-AUC
-   Confusion Matrix

## Dashboard Features

-   Upload CSV file
-   Dataset preview
-   Interactive EDA
-   Fraud prediction
-   Probability scores
-   Threshold slider
-   Confusion matrix
-   Correlation heatmap
-   Business cost analysis
-   Download prediction results

## Business Impact

The solution helps financial institutions: - Detect fraud earlier -
Reduce financial losses - Improve customer trust - Support real-time
transaction monitoring

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Plotly
-   Streamlit
-   Joblib

## Conclusion

The project demonstrates an end-to-end machine learning workflow for
fraud detection, from data preprocessing to deployment in an interactive
Streamlit dashboard.
