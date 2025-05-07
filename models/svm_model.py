import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_utils import load_data, preprocess, cross_validate_model

import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE  # type: ignore

# 0) Ensure folders exist
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1) Load & split data
X_train, X_test, y_train, y_test = load_data()

# 2) Preprocess (TF-IDF + scaling)
X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

# 3) Downsample for speed, then apply SMOTE
X_train_sub, _, y_train_sub, _ = train_test_split(X_train_scaled, y_train, train_size=0.1, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_sub, y_train_sub)

# 4) Dimensionality reduction with SVD
svd = TruncatedSVD(n_components=50, random_state=42)
X_train_reduced = svd.fit_transform(X_train_smote)
X_test_reduced = svd.transform(X_test_scaled)

# 5) Train LinearSVC
model = LinearSVC(dual=False, max_iter=1000, random_state=42)
model.fit(X_train_reduced, y_train_smote)

# 🔁 Save model
joblib.dump(model, "models/linear_svc_model.pkl")

# 6) Predict & evaluate
y_pred = model.predict(X_test_reduced)

# For ROC-AUC, get decision function instead of predict_proba
try:
    probs = model.decision_function(X_test_reduced)
    roc_auc = roc_auc_score(y_test, probs)
except:
    probs = np.zeros_like(y_pred)
    roc_auc = 0.0

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# 📊 Save classification report and confusion matrix
report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv("results/linear_svc_classification_report.csv")

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=["Actual Neg", "Actual Pos"], columns=["Pred Neg", "Pred Pos"]) \
  .to_csv("results/linear_svc_confusion_matrix.csv")

# 📉 Confusion Matrix Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot(cmap='Purples')
plt.title("LinearSVC - Confusion Matrix")
plt.savefig("results/linear_svc_confusion_matrix.png")
plt.close()

# 📈 ROC Curve Plot (if available)
if roc_auc > 0:
    roc_disp = RocCurveDisplay.from_predictions(y_test, probs)
    roc_disp.plot()
    plt.title("LinearSVC - ROC Curve")
    plt.savefig("results/linear_svc_roc_curve.png")
    plt.close()

# 7) Save ROC-AUC for comparison
pd.DataFrame([["LinearSVC", roc_auc]], columns=["Model", "ROC_AUC"]) \
  .to_csv("results/roc_auc_comparison.csv", mode='a', header=not os.path.exists("results/roc_auc_comparison.csv"), index=False)

# 8) Cross-validation on original imbalanced data using pipeline
pipeline = make_pipeline(
    TruncatedSVD(n_components=50, random_state=42),
    LinearSVC(dual=False, max_iter=1000, random_state=42)
)

cv_results = cross_validate_model(pipeline, X_train_scaled, y_train)


print("LinearSVC Accuracy (CV):", round(cv_results['accuracy_mean'], 4), "±", round(cv_results['accuracy_std'], 4))
print("LinearSVC F1 Score (CV):", round(cv_results['f1_mean'], 4), "±", round(cv_results['f1_std'], 4))

# 9) Save all metrics for comparison
pd.DataFrame([["LinearSVC", accuracy, precision, recall, f1]], 
             columns=["Model", "Accuracy", "Precision", "Recall", "F1"]) \
  .to_csv("results/model_comparison.csv", mode='a', header=not os.path.exists("results/model_comparison.csv"), index=False)

# ✅ Done
print(f"✅ LinearSVC done — Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
