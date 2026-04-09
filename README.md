Hospital Readmission Prediction Using Machine Learning

Overview

Hospital readmissions are a major challenge in healthcare, often indicating gaps in patient care, discharge planning, or follow-up support. This project builds a machine learning model to predict whether a patient is likely to be readmitted within 30 days using real hospital data.

The goal is to support early intervention, improve patient outcomes, and reduce healthcare costs.


Dataset

This project uses the Diabetes 130-US hospitals dataset (1999–2008), which contains over 100,000 patient records across 130 hospitals.

Key Features:
	•	Patient demographics (age, gender, race)
	•	Admission details
	•	Diagnosis information
	•	Medication data
	•	Number of procedures and visits

Target Variable:
	•	readmitted
	•	<30 → readmitted within 30 days (High Risk)
	•	>30 or NO → not readmitted within 30 days (Low Risk)


Objective

To build a classification model that predicts 30-day hospital readmission risk, enabling healthcare providers to identify high-risk patients and prioritise care interventions.

 Technologies Used
	•	Python
	•	Pandas & NumPy
	•	Scikit-learn
	•	Matplotlib


 Methodology

1. Data Preprocessing
	•	Replaced missing values (?) with NaN
	•	Handled missing data using imputation
	•	Dropped irrelevant columns (IDs, high-missing fields)
	•	Encoded categorical variables using One-Hot Encoding



2. Feature Engineering
	•	Converted target into binary:
	•	1 → readmitted within 30 days
	•	0 → otherwise



3. Model Development

Two models were built and compared:
	•	Logistic Regression (baseline, interpretable)
	•	Random Forest Classifier (non-linear, higher performance)



4. Handling Class Imbalance
	•	Used class_weight="balanced" to improve detection of high-risk patients



5. Evaluation Metrics
	•	ROC-AUC Score
	•	Precision & Recall
	•	Confusion Matrix
	•	Precision-Recall Curve


6. Threshold Tuning

Adjusted decision threshold from 0.50 → 0.30 to improve recall, ensuring more high-risk patients are identified.


Results
	•	Random Forest outperformed Logistic Regression in capturing complex patterns
	•	Threshold tuning significantly improved recall for high-risk patients
	•	Key predictors of readmission were identified using feature importance


🔍 Key Insights
	•	Patients with higher hospital utilisation and certain diagnoses are more likely to be readmitted
	•	Model performance improves when focusing on recall rather than accuracy
	•	Early identification of high-risk patients can support better discharge planning

