# Customer Churn Prediction using Machine Learning

## Overview

This project uses machine learning to predict whether a customer is likely to churn based on customer information such as tenure, contract type, internet service, monthly charges, payment method, and other service-related features.

The project includes data preprocessing, exploratory data analysis (EDA), categorical encoding, model training, model comparison, evaluation, threshold tuning, and a saved prediction model.

The final solution uses **Scaled Logistic Regression** with a **0.4 classification threshold** to identify customers who may be at risk of churn.

---

## Problem Statement

Customer churn is a major challenge for subscription-based businesses. When customers stop using a service, the business loses revenue and may also lose opportunities to retain those customers.

The goal of this project is to build a machine learning model that can:

* Predict whether a customer is likely to churn.
* Estimate the customer's probability of churn.
* Identify factors associated with higher or lower churn risk.
* Help businesses identify potentially at-risk customers early.

---

## Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains customer demographic information, account details, subscribed services, billing information, and the target variable `Churn`.

### Target Variable

* `Churn = Yes` → Customer left the service
* `Churn = No` → Customer stayed with the service

### Main Feature Categories

* Customer information: gender, senior citizen status, partner, dependents
* Account information: tenure and contract type
* Services: internet service, online security, online backup, technical support, streaming services
* Charges: monthly charges and total charges
* Payment: payment method and paperless billing

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Google Colab
* Jupyter Notebook

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
```

---

## Data Preprocessing

The following preprocessing steps were performed:

* Checked for missing values
* Checked for duplicate records
* Removed the `customerID` column
* Separated features (`X`) and target (`y`)
* Converted categorical variables using one-hot encoding
* Converted the target variable into binary values:

  * `No = 0`
  * `Yes = 1`
* Split the data into 80% training and 20% testing sets
* Applied `StandardScaler` to numerical features

---

## Models Used

Two classification algorithms were tested:

### 1. Logistic Regression

Used as the main baseline classification model.

### 2. Random Forest

Used as a second model to compare performance with Logistic Regression.

### Model Comparison

| Model                      | Accuracy | Precision | Recall | F1 Score |
| -------------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression        |   78.82% |    62.26% | 51.60% |   56.43% |
| Random Forest              |   77.61% |    60.97% | 43.85% |   51.01% |
| Scaled Logistic Regression |   78.82% |    62.18% | 51.87% |   56.56% |

Scaled Logistic Regression performed slightly better than the other tested models for identifying churners.

---

## Threshold Tuning

The default classification threshold of `0.5` was changed to `0.4`.

This was done because identifying customers who may churn is important for the business.

### Results at Threshold = 0.4

* Accuracy: approximately **78%**
* Precision: **58%**
* Recall: **65%**
* F1 Score: **61%**
* ROC-AUC: **0.832**

Lowering the threshold increased churn recall from approximately **52% to 65%**, allowing the model to identify more potential churners.

---

## Feature Analysis

Logistic Regression coefficients were analyzed to understand which features were associated with churn.

Some features with stronger positive coefficients included:

* Total Charges
* Fiber Optic Internet Service
* Streaming Movies
* Streaming TV
* Multiple Lines
* Paperless Billing
* Electronic Check Payment

Some features with negative coefficients included:

* Tenure
* One-year contract
* Two-year contract
* Online Security
* Tech Support
* Dependents

These coefficients represent model associations and should not be interpreted as direct causes of churn.

---

## Prediction System

The trained model and scaler were saved using Joblib:

```text
churn_model.pkl
scaler.pkl
```

The prediction system:

1. Takes customer data as input.
2. Applies the saved scaler.
3. Calculates the probability of churn.
4. Uses a threshold of `0.4`.
5. Returns either:

   * `Customer may churn`
   * `Customer likely to stay`

Example:

```text
Churn Probability: 85.56%
Prediction: Customer may churn
```

---

## Project Files

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction_ML.ipynb
├── churn_model.pkl
├── scaler.pkl
└── README.md
```

---

## How to Run

### 1. Open the notebook

Open:

```text
Customer_Churn_Prediction_ML.ipynb
```

using Google Colab or Jupyter Notebook.

### 2. Install required libraries

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

### 3. Run the notebook

Run the cells in order to:

* Load the dataset
* Preprocess the data
* Train the models
* Evaluate the models
* Generate predictions

### 4. Load the saved model

```python
import joblib

loaded_model = joblib.load("churn_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
```

---

## Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Testing additional algorithms such as XGBoost
* Handling class imbalance
* Building a Streamlit web application
* Adding a customer-friendly prediction form
* Deploying the model using a cloud platform
* Monitoring model performance after deployment

---

## Key Learning Outcomes

Through this project, I learned:

* Exploratory Data Analysis
* Data cleaning and preprocessing
* Categorical encoding
* Feature scaling
* Train-test splitting
* Classification algorithms
* Model evaluation
* Confusion matrix
* Precision, recall and F1-score
* ROC-AUC
* Classification threshold tuning
* Model interpretation
* Saving and loading ML models using Joblib
* Building a basic prediction pipeline
