# Hospital-Readmission-Prediction-Using-Machine-Learning


## Overview
This project applies machine learning techniques to predict hospital readmission risk using patient health data. It demonstrates how data-driven models can support healthcare decision-making and improve patient outcomes.

## Research Aim
The aim of this project is to explore how machine learning can be used to identify patients at higher risk of hospital readmission based on health-related indicators.

## Objectives
- Predict the likelihood of patient readmission
- Apply machine learning techniques to healthcare data
- Evaluate model performance using classification metrics
- Demonstrate the value of predictive analytics in healthcare

## Dataset
The dataset used in this project contains patient health indicators such as:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

For demonstration purposes, a simple readmission target variable was created from patient glucose levels.

## Methodology
The project followed a structured machine learning workflow:

### 1. Data Loading
The dataset was imported into Python using Pandas.

### 2. Data Preparation
A target variable called `Readmitted` was created to simulate readmission risk.

### 3. Train-Test Split
The data was split into training and testing sets for model development and evaluation.

### 4. Model Training
A Random Forest Classifier was used to train the prediction model.

### 5. Evaluation
The model was evaluated using:
- Accuracy score
- Classification report
- Confusion matrix
- Feature importance analysis

## Tools and Technologies
- Python
- Pandas
- NumPy
- Scikit-learn

## Results
The model produced a prediction output for hospital readmission risk and identified important patient features contributing to the classification.

## Healthcare Relevance
Hospital readmission is an important issue in healthcare because it affects both patient outcomes and healthcare costs. Predictive models can help identify high-risk patients early and support better interventions, planning, and decision-making.

## Conclusion
This project demonstrates how machine learning can be applied to healthcare-related data to predict hospital readmission risk. It provides a useful foundation for future work in health data analytics, predictive modelling, and health informatics.

## Future Work
Future improvements could include:
- Using real hospital readmission datasets
- Testing additional machine learning models such as XGBoost
- Improving feature engineering
- Adding data visualisations and dashboards

## Repository Structure
- `readmission_prediction_model.py` — main machine learning script
- `README.md` — project documentation
