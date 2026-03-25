# Netflix Churn Prediction

## Overview
This project builds a machine learning pipeline to predict customer churn for a Netflix-style subscription service. The objective is to identify users likely to cancel their subscriptions and enable proactive retention strategies.

---

## Business Problem
Customer churn directly impacts revenue in subscription-based platforms. By predicting churn, businesses can:
- Target high-risk users with retention offers  
- Reduce customer acquisition costs  
- Increase customer lifetime value (CLV)  

---

## Solution
Developed an end-to-end supervised learning pipeline using customer behavioral and demographic data.

Key components:
- Data preprocessing and feature engineering  
- Exploratory data analysis (EDA) to identify churn patterns  
- Model training and evaluation across multiple algorithms  
- Performance optimization using hyperparameter tuning  

---

## Dataset Features
Typical features used in the model include:
- Account information (tenure, subscription type)  
- Usage behavior (watch time, login frequency)  
- Customer engagement metrics  
- Billing and payment details  

---

## Models Used
- Logistic Regression (baseline)  
- Random Forest  
- Gradient Boosting / XGBoost  

---

## Results
- Achieved strong predictive performance using ensemble methods  
- Improved model accuracy and recall compared to baseline logistic regression  
- Identified key churn drivers such as low engagement and short tenure  

---

## Tech Stack
- Python  
- pandas, NumPy  
- scikit-learn  
- Matplotlib / Seaborn  

---

## Future Improvements
- Incorporate time-series features to capture user behavior trends  
- Deploy model as an API for real-time predictions  
- Integrate with business dashboards (e.g., Power BI) for stakeholder insights  

---

## Author
Brendan Lauterborn  
