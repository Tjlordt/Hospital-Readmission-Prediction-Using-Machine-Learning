
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)

# 1. Load the dataset
df = pd.read_csv("diabetic_data.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 2. Basic cleaning
df = df.replace("?", np.nan)

# 3. Create binary target
df["target_30d_readmit"] = (df["readmitted"] == "<30").astype(int)

# 4. Drop unnecessary columns
drop_cols = [
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "weight",
    "payer_code",
    "medical_specialty"
]
df_model = df.drop(columns=drop_cols)

# 5. Split features and target
X = df_model.drop(columns=["target_30d_readmit"])
y = df_model["target_30d_readmit"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Preprocessing pipelines
numeric_transformer_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_transformer_rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))
])

preprocessor_lr = ColumnTransformer(transformers=[
    ("num", numeric_transformer_lr, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

preprocessor_rf = ColumnTransformer(transformers=[
    ("num", numeric_transformer_rf, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# 8. Models
log_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_lr),
    ("model", LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="liblinear"
    ))
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_rf),
    ("model", RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# 9. Train models
print("\nTraining Logistic Regression...")
log_reg_pipeline.fit(X_train, y_train)

print("Training Random Forest...")
rf_pipeline.fit(X_train, y_train)

# 10. Evaluation function
def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"{model_name} | Threshold = {threshold:.2f}")
    print(f"{'='*60}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return y_proba, y_pred

# 11. Default threshold evaluation
log_proba, log_pred = evaluate_model(
    log_reg_pipeline, X_test, y_test,
    "Logistic Regression", threshold=0.50
)

rf_proba, rf_pred = evaluate_model(
    rf_pipeline, X_test, y_test,
    "Random Forest", threshold=0.50
)

# 12. Threshold tuning for recall
precision, recall, thresholds = precision_recall_curve(y_test, rf_proba)

threshold_table = pd.DataFrame({
    "threshold": np.append(thresholds, 1.0),
    "precision": precision,
    "recall": recall
})

print("\nThreshold candidates:")
print(threshold_table.head(10))

chosen_threshold = 0.30

rf_proba_tuned, rf_pred_tuned = evaluate_model(
    rf_pipeline, X_test, y_test,
    "Random Forest (Recall-Optimised)", threshold=chosen_threshold
)

# 13. ROC Curve
log_fpr, log_tpr, _ = roc_curve(y_test, log_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)

plt.figure(figsize=(8, 6))
plt.plot(log_fpr, log_tpr, label="Logistic Regression")
plt.plot(rf_fpr, rf_tpr, label="Random Forest")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# 14. Feature importance
ohe = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
encoded_cat_names = ohe.get_feature_names_out(categorical_cols)

all_feature_names = numeric_cols + list(encoded_cat_names)

importances = rf_pipeline.named_steps["model"].feature_importances_

feature_importance = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 20 Important Features:")
print(feature_importance.head(20))

plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15).sort_values("importance")
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

# 15. Save scoring output
results = X_test.copy()
results["actual_readmitted_30d"] = y_test.values
results["rf_probability"] = rf_proba
results["rf_prediction_0_5"] = rf_pred
results["rf_prediction_0_3"] = rf_pred_tuned

results.to_csv("readmission_predictions_output.csv", index=False)
print("\nSaved scoring file: readmission_predictions_output.csv")
