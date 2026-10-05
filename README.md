# Customer Churn Prediction using Machine Learning

## Overview

This project uses machine learning to predict whether a customer is likely to churn based on customer information such as tenure, contract type, internet service, monthly charges, payment method, and other service-related features.

The project includes data preprocessing, exploratory data analysis (EDA), categorical encoding, model training, model comparison, evaluation, threshold tuning, and a saved prediction model.

The final solution uses **Scaled Logistic Regression** with a **0.4 classification threshold** to identify customers who may be at risk of churn.

---

## Problem Statement

Customer churn is a major challenge for subscription-based businesses. When customers stop using a service, the business loses revenue and may also lose opportunities to retain those customers.

The goal of this project is to build a machine learning model that can:

- Predict whether a customer is likely to churn.
- Estimate the customer's probability of churn.
- Identify factors associated with higher or lower churn risk.
- Help businesses identify potentially at-risk customers early.

---

## Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains customer demographic information, account details, subscribed services, billing information, and the target variable `Churn`.

### Target Variable

- `Churn = Yes` → Customer left the service
- `Churn = No` → Customer stayed with the service

### Main Feature Categories

- Customer information: gender, senior citizen status, partner, dependents
- Account information: tenure and contract type
- Services: internet service, online security, online backup, technical support, streaming services
- Charges: monthly charges and total charges
- Payment: payment method and paperless billing

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Google Colab
- Jupyter Notebook

---

## Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
Model Evaluation
   ↓
Threshold Tuning
   ↓
Final Prediction System
