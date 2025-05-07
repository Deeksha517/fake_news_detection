import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

def load_data(path='dataset/cleaned_fakenews_dataset.csv'):
    """
    Load and split the dataset into training and testing sets.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    
    df = pd.read_csv(path)
    df = df.dropna(subset=['clean_title'])  # Drop rows with missing text
    X = df['clean_title']
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocess(X_train, X_test, max_features=2000):
    """
    TF-IDF vectorization and standard scaling (without mean subtraction).
    Returns transformed features and fitted vectorizer/scaler.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_tfidf)
    X_test_scaled = scaler.transform(X_test_tfidf)

    return X_train_scaled, X_test_scaled  # optionally also return vectorizer, scaler


def cross_validate_model(model, X, y, cv=5):
    """
    Perform k-fold cross-validation with accuracy, F1, precision, and recall scores.
    """
    model_copy = clone(model)

    acc_scores = cross_val_score(model_copy, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    f1_scores = cross_val_score(model_copy, X, y, cv=cv, scoring='f1', n_jobs=-1)
    precision_scores = cross_val_score(model_copy, X, y, cv=cv, scoring='precision', n_jobs=-1)
    recall_scores = cross_val_score(model_copy, X, y, cv=cv, scoring='recall', n_jobs=-1)

    return {
        "accuracy_mean": np.mean(acc_scores),
        "accuracy_std": np.std(acc_scores),
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "precision_mean": np.mean(precision_scores),
        "precision_std": np.std(precision_scores),
        "recall_mean": np.mean(recall_scores),
        "recall_std": np.std(recall_scores),
    }
