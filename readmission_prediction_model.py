import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Create a simple target variable for demonstration
# Patients with higher glucose are flagged as higher readmission risk
df["Readmitted"] = np.where(df["Glucose"] > 120, 1, 0)

# Separate features and target
X = df.drop("Readmitted", axis=1)
y = df["Readmitted"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print("Hospital Readmission Prediction Model")
print("=" * 40)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(matrix)

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop Feature Importances:")
print(feature_importance)
