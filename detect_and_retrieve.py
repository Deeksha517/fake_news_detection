import os
import re
import string
import joblib
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 1) Text cleaning (reuse your function)
STOPWORDS = set(stopwords.words('english'))
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in STOPWORDS]
    return ' '.join(tokens)

# 2) Load the cleaned dataset
df = pd.read_csv('dataset/cleaned_fakenews_dataset.csv')
df = df.dropna(subset=['clean_title'])

# 3) Load (or train) the SBERT+LogReg classifier
MODEL_PATH = "models/sbert_logreg.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

print("🔸 Loading classifier...")
clf: LogisticRegression = joblib.load(MODEL_PATH)

# 4) Prepare SBERT embedder
print(f"🔸 Loading SBERT model ({EMBED_MODEL})...")
embedder = SentenceTransformer(EMBED_MODEL)

# 5) Build FAISS index over real articles
real_df = df[df['label'] == 1].reset_index(drop=True)
real_texts = real_df['clean_title'].tolist()
print(f"🔸 Embedding {len(real_texts)} real headlines for retrieval…")

# Batch-encode & normalize
real_embs = embedder.encode(
    real_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
faiss.normalize_L2(real_embs)

# Create index
index = faiss.IndexFlatIP(real_embs.shape[1])
index.add(real_embs)

def check_headline(headline: str, top_k: int = 3):
    """
    1) Cleans & embeds the headline.
    2) Classifies Fake (0) vs Real (1).
    3) If Fake, retrieves top_k similar real headlines.
    """
    cleaned = clean_text(headline)
    emb = embedder.encode([cleaned], convert_to_numpy=True)
    pred = clf.predict(emb)[0]
    prob = clf.predict_proba(emb)[0, pred]
    label = "Real" if pred == 1 else "Fake"
    print(f"\n📰 Input: {headline}")
    print(f"➡️  Classified as: {label} (conf={prob:.2f})")

    if pred == 0:
        faiss.normalize_L2(emb)
        D, I = index.search(emb, top_k)
        print(f"\n🔍 Top {top_k} “real” suggestions:")
        for rank, idx in enumerate(I[0], start=1):
            print(f" {rank}. {real_texts[idx]}")

if __name__ == "__main__":
    print("\nEnter a headline (or 'exit'):")
    while True:
        text = input(">> ").strip()
        if text.lower() in {"exit","quit"}:
            break
        if not text:
            continue
        check_headline(text, top_k=3)
