📌 Telecom Customer Churn Prediction & Cost-Sensitive Retention Engine
🚀 Project Overview

This project develops an end-to-end machine learning system to:

Predict customer churn in a telecom company

Identify high-risk customers

Explain churn drivers using SHAP

Optimize retention strategies using cost-sensitive ROI modeling

Unlike basic churn projects that stop at prediction, this system integrates business logic to maximize net profit through selective customer targeting.

🎯 Business Problem

Customer churn significantly impacts telecom revenue.

The goal of this project is to:

Predict which customers are likely to churn.

Understand the key drivers of churn.

Design a cost-aware retention strategy.

Maximize net benefit by targeting only profitable customers.

📊 Dataset

The dataset contains ~7,000 telecom customers with:

Demographics (gender, senior citizen, dependents)

Service usage (internet, streaming, tech support)

Contract details

Monthly and total charges

Target variable: Churn (Yes/No)

🔎 Exploratory Data Analysis (01_eda.ipynb)

Key Insights:

Customers with month-to-month contracts show higher churn.

Fiber optic users demonstrate higher churn rates.

Short tenure customers are more likely to churn.

Higher monthly charges correlate with churn probability.

Class imbalance exists (~26% churn rate).

These findings guided feature engineering and model selection.

🤖 Modeling & Evaluation (02_modeling.ipynb)
Models Implemented

Logistic Regression (baseline)

Logistic Regression with class weighting

XGBoost (final selected model)

Handling Class Imbalance

Used class_weight='balanced' for Logistic Regression

Used scale_pos_weight for XGBoost

Performed threshold tuning for recall optimization

Final Model: XGBoost
Metric	Value
ROC-AUC	~0.84
Recall	~74%
Precision	~51%
F1 Score	~0.61

XGBoost was selected for its balanced precision-recall performance and strong discrimination ability.

🔍 Explainability (SHAP Analysis)

Using SHAP:

Tenure reduces churn risk significantly.

Month-to-month contracts increase churn probability.

Fiber optic service increases churn likelihood.

High total charges indicate more stable customers.

Tech support reduces churn risk.

This ensures the model is interpretable and aligned with business logic.

💰 Cost-Sensitive Retention Engine (03_business_retention.ipynb)

Beyond prediction, a business optimization layer was built.

Steps Implemented:

Calculate churn probability for each customer.

Estimate expected revenue loss.

Assign retention strategies (premium, standard, downgrade).

Model retention cost.

Estimate success probability.

Compute expected revenue saved.

Calculate net benefit.

Target only customers with positive net ROI.

📈 Final Business Impact

Profitable customers identified: 310

Projected Net Profit: ₹168,047

Blanket retention strategy → Negative ROI

Optimized selective targeting → Positive ROI
