# models/random_forest_model.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_utils import load_data, preprocess, cross_validate_model

import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE  # type: ignore


# 0) Ensure results folder exists
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1) Load & optionally reduce data for speed
X_train, X_test, y_train, y_test = load_data()
if len(X_train) > 100_000:
    X_train = X_train.sample(n=100_000, random_state=42)
    y_train = y_train.loc[X_train.index]

# 2) Preprocess (TF-IDF + scaling)
X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

# 3) Balance training set with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# 4) Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,  # Reduced CV folds for speed
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train_smote, y_train_smote)

# Save best parameters
with open("results/best_params_random_forest.json", "w") as f:
    json.dump(grid_search.best_params_, f, indent=4)

# 5) Train best model
model = grid_search.best_estimator_
model.fit(X_train_smote, y_train_smote)

# 🔁 Save model
joblib.dump(model, "models/random_forest_model.pkl")

# 6) Predict & evaluate
y_pred = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, probs)

# 📊 Save classification report and confusion matrix
report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv("results/random_forest_classification_report.csv")

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=["Actual Neg", "Actual Pos"], columns=["Pred Neg", "Pred Pos"])\
  .to_csv("results/random_forest_confusion_matrix.csv")

# 📉 Confusion Matrix Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot(cmap='Blues')
plt.title("Random Forest - Confusion Matrix")
plt.savefig("results/random_forest_confusion_matrix.png")
plt.close()

# 📈 ROC Curve Plot
roc_disp = RocCurveDisplay.from_predictions(y_test, probs)
roc_disp.plot()
plt.title("Random Forest - ROC Curve")
plt.savefig("results/random_forest_roc_curve.png")
plt.close()

# 7) Save ROC-AUC for comparison
pd.DataFrame([["RandomForest", roc_auc]],
             columns=["Model", "ROC_AUC"]) \
  .to_csv("results/roc_auc_comparison.csv", mode='a', header=not os.path.exists("results/roc_auc_comparison.csv"), index=False)

# 8) Cross-validation on original imbalanced data
cv_results = cross_validate_model(model, X_train_scaled, y_train)
print("Random Forest Accuracy (CV):", round(cv_results['accuracy_mean'], 4), "±", round(cv_results['accuracy_std'], 4))
print("Random Forest F1 Score (CV):", round(cv_results['f1_mean'], 4), "±", round(cv_results['f1_std'], 4))

# 9) Save all metrics for comparison
pd.DataFrame([["RandomForest", accuracy, precision, recall, f1]],
             columns=["Model", "Accuracy", "Precision", "Recall", "F1"]) \
  .to_csv("results/model_comparison.csv", mode='a', header=not os.path.exists("results/model_comparison.csv"), index=False)

# ✅ Done
print(f"✅ Random Forest done — Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
