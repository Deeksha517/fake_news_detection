# models/xgboost_model.py

import os
import sys
# Make sure shared_utils is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from shared_utils import load_data, preprocess

# Create output folders
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- CONFIG ---
SAMPLE_FRAC   = 0.05
N_ESTIMATORS  = 100
MAX_DEPTH     = 3
LEARNING_RATE = 0.1
EARLY_STOP    = 5

# 1) Load and sample dataset
X_train, X_test, y_train, y_test = load_data()
X_train, _, y_train, _ = train_test_split(
    X_train, y_train,
    train_size=SAMPLE_FRAC,
    stratify=y_train,
    random_state=42
)

# 2) Preprocess
X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

# 3) Train/Validation split
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled, y_train,
    test_size=0.2,
    random_state=42
)

# 4) Model training (move early_stopping_rounds to constructor)
model = XGBClassifier(
    tree_method='hist',
    grow_policy='lossguide',
    n_jobs=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=EARLY_STOP,
    random_state=42
)

print("⏳ Training XGBoost...")
model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_train_final, y_train_final), (X_val, y_val)],
    verbose=False
)

# Save model
joblib.dump(model, "models/xgboost_model.pkl")

# 5) Evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)

# Save classification report and confusion matrix
pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose() \
  .to_csv("results/xgboost_classification_report.csv")

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=["Actual Neg","Actual Pos"], columns=["Pred Neg","Pred Pos"]) \
  .to_csv("results/xgboost_confusion_matrix.csv")

# Confusion matrix plot
ConfusionMatrixDisplay(cm, display_labels=["Fake","Real"]).plot(cmap='Blues')
plt.title("XGBoost - Confusion Matrix")
plt.savefig("results/xgboost_confusion_matrix.png")
plt.close()

# ROC curve plot
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("XGBoost - ROC Curve")
plt.savefig("results/xgboost_roc_curve.png")
plt.close()

# --- Additional Visualizations ---

# Feature Importance
plt.figure(figsize=(10,6))
plot_importance(model, max_num_features=10, importance_type='gain')
plt.title("XGBoost - Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("results/xgboost_feature_importance.png")
plt.close()

# Metrics Bar Plot
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": roc_auc
}
plt.figure(figsize=(8,5))
plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.ylim(0,1)
for i, v in enumerate(metrics.values()):
    plt.text(i, v+0.01, f"{v:.2f}", ha='center')
plt.title("XGBoost Performance Metrics")
plt.tight_layout()
plt.savefig("results/xgboost_metrics_barplot.png")
plt.close()

# Learning Curve (Log Loss)
evals_result = model.evals_result()
if 'validation_0' in evals_result and 'validation_1' in evals_result:
    train_loss = evals_result['validation_0']['logloss']
    val_loss   = evals_result['validation_1']['logloss']
    epochs     = range(len(train_loss))
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label='Train LogLoss')
    plt.plot(epochs, val_loss,   label='Validation LogLoss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss During Training')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/xgboost_learning_curve.png")
    plt.close()

# Save metrics
metrics_df = pd.DataFrame([{
    "Model": "XGBoost",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC-AUC": roc_auc
}])
metrics_df.to_csv("results/model_metrics.csv", index=False)
metrics_df.drop(columns=["ROC-AUC"]).to_csv(
    "results/model_comparison.csv",
    mode='a',
    header=not os.path.exists("results/model_comparison.csv"),
    index=False
)
pd.DataFrame([["XGBoost", roc_auc]], columns=["Model","ROC_AUC"]).to_csv(
    "results/roc_auc_comparison.csv",
    mode='a',
    header=not os.path.exists("results/roc_auc_comparison.csv"),
    index=False
)

print(f"✅ XGBoost Done — Acc: {accuracy:.3f}, Prec: {precision:.3f}, "
      f"Rec: {recall:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
