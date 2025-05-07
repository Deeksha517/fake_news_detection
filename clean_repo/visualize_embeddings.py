import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# 1. Load cleaned dataset
df = pd.read_csv("dataset/cleaned_fakenews_dataset.csv")
df = df.dropna(subset=["clean_title"])
df = df.sample(1000, random_state=42)

# 2. Generate SBERT embeddings
print("🔸 Loading SBERT model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print(f"🔸 Encoding {len(df)} headlines...")
embeddings = embedder.encode(df["clean_title"].tolist(), show_progress_bar=True)

# 3. Dimensionality reduction
use_tsne = False  # 🔁 Set to True for t-SNE

if use_tsne:
    print("🔸 Reducing dimensions with t-SNE...")
    reducer = TSNE(n_components=2, perplexity=40, n_iter=500, random_state=42)
    method = "tsne"
else:
    print("🔸 Reducing dimensions with PCA...")
    reducer = PCA(n_components=2)
    method = "pca"

reduced_embeddings = reducer.fit_transform(embeddings)

# 4. Plotting
print("🔸 Plotting and saving...")
plt.figure(figsize=(10, 6))

colors = df["label"].map({0: "red", 1: "blue"})
labels = df["label"].map({0: "Fake", 1: "Real"})

for label_value, label_name in [(0, "Fake"), (1, "Real")]:
    indices = df["label"] == label_value
    plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1],
                c=colors[indices],
                label=label_name,
                alpha=0.6, s=30)

plt.title("SBERT Embeddings Visualization (Fake vs Real)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 5. Save to results/ folder
os.makedirs("results", exist_ok=True)
plot_path = f"results/sbert_visualization_{method}.png"
plt.savefig(plot_path)
print(f"✅ Plot saved at: {plot_path}")

plt.show()
