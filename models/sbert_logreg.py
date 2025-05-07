# models/sbert_logreg.py

import os
import sys

# ─── Make sure shared_utils is on the path ──────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_utils import load_data, cross_validate_model

import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE

# 0) Prepare output dirs
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1) Load & split your cleaned data
X_train, X_test, y_train, y_test = load_data()

# 2) (Optional) sample 20% for quick experiments
X_train, _, y_train, _ = train_test_split(
    X_train, y_train, train_size=0.2, stratify=y_train, random_state=42
)

# 3) Load SBERT and embed
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Encoding training texts…")
X_train_emb = embedder.encode(X_train.tolist(), show_progress_bar=True)
print("Encoding test texts…")
X_test_emb  = embedder.encode(X_test.tolist(),  show_progress_bar=True)

# 4) Balance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_emb, y_train)

# 5) Train Logistic Regression
start = time.time()
clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
clf.fit(X_train_res, y_train_res)
print(f"Training time: {time.time() - start:.1f}s")

# 6) Cross-validation on the SBERT embeddings
cv = cross_validate_model(clf, X_train_emb, y_train)
print(f"CV Acc: {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
print(f"CV F1 : {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}")

# 7) Evaluate on hold-out
y_pred = clf.predict(X_test_emb)
y_prob = clf.predict_proba(X_test_emb)[:,1]

acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred)
rec   = recall_score(y_test, y_pred)
f1    = f1_score(y_test, y_pred)
roc   = roc_auc_score(y_test, y_prob)

print("\nTest set performance:")
print(classification_report(y_test, y_pred, target_names=['Fake','Real']))
print(f"ROC-AUC: {roc:.4f}")

# 8) Save the model
import joblib
joblib.dump(clf, "models/sbert_logreg.pkl")

# 9) Write summary to CSV
pd.DataFrame([{
    "Model": "SBERT+LogReg",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "ROC_AUC": roc
}]).to_csv("results/model_comparison.csv", index=False)

print("✅ SBERT + Logistic Regression complete. Metrics saved to results/model_comparison.csv")
