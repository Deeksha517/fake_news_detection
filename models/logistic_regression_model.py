import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_utils import load_data, preprocess, cross_validate_model

import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE  # type: ignore


# 0) Ensure results folders exist
os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# 1) CLI Argument for full training
parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Run full training on entire dataset")
args = parser.parse_args()

# 2) Load & split data
X_train, X_test, y_train, y_test = load_data()

# 3) Sample only 20% of training data (unless --full is passed)
if not args.full:
    X_train = X_train.sample(frac=0.2, random_state=42)
    y_train = y_train.loc[X_train.index]

# 4) Preprocess (TF-IDF + scaling)
X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

# 5) Apply TruncatedSVD for dimensionality reduction
svd = TruncatedSVD(n_components=50, random_state=42)  # Reduce features to 50 dimensions
X_train_scaled = svd.fit_transform(X_train_scaled)
X_test_scaled = svd.transform(X_test_scaled)

# 5) Balance the training set with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# 6) Train Logistic Regression
start = time.time()
model = LogisticRegression(solver='saga', max_iter=2000, random_state=42)
model.fit(X_train_smote, y_train_smote)
print(f"✅ Training completed in {time.time() - start:.2f} seconds")

# 7) Predict & evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# 8) 5-fold Cross-validation on original (imbalanced) training set
cv_results = cross_validate_model(model, X_train_scaled, y_train)
print("Logistic Regression Accuracy (CV):", round(cv_results['accuracy_mean'], 4), "±", round(cv_results['accuracy_std'], 4))
print("Logistic Regression F1 Score (CV):", round(cv_results['f1_mean'], 4), "±", round(cv_results['f1_std'], 4))

# 9) ROC-AUC Score
probs = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, probs)
print(f"✅ Logistic Regression ROC-AUC: {roc_auc:.3f}")

# 10) Save Accuracy + F1 + ROC-AUC + Precision + Recall (overwrite old result)
pd.DataFrame([["LogisticRegression", accuracy, f1, roc_auc, precision, recall]],
             columns=["Model", "Accuracy", "F1", "ROC_AUC", "Precision", "Recall"]) \
  .to_csv("results/model_comparison.csv", index=False)

# 11) Save Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("results/plots/confusion_matrix_logreg.png")
plt.close()

# 12) Save ROC Curve
roc_display = RocCurveDisplay.from_predictions(y_test, probs)
roc_display.plot()
plt.title("ROC Curve - Logistic Regression")
plt.savefig("results/plots/roc_curve_logreg.png")
plt.close()

print(f"✅ Logistic Regression done — Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
