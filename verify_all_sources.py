import pandas as pd
import re
import string
from nltk.corpus import stopwords  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
import nltk  # type: ignore

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

# Text cleaning function (same as in the preprocessing)
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in STOPWORDS]
    return ' '.join(tokens)

# Dataset paths
datasets = {
    'politifact_fake.csv': 0,
    'politifact_real.csv': 1,
    'gossipcop_fake.csv': 0,
    'gossipcop_real.csv': 1
}

total_expected_rows = 0

print("🔍 Counting expected rows after exploding tweet_ids...")

# Iterate over datasets and calculate expected row count
for file, label in datasets.items():
    path = f'dataset/{file}'
    df = pd.read_csv(path)
    
    # Clean the tweet_ids and count rows after splitting
    df['tweet_ids'] = df['tweet_ids'].astype(str).str.strip()
    df = df[df['tweet_ids'].str.len() > 0]
    df['num_tweets'] = df['tweet_ids'].apply(lambda x: len(x.split()))
    file_total = df['num_tweets'].sum()
    total_expected_rows += file_total
    print(f"{file}: {file_total} tweet ID rows")

print(f"\n📦 Total expected rows after cleaning: {total_expected_rows}")

# Load the final cleaned dataset
cleaned_df = pd.read_csv('C:/Users/deeks/Desktop/cleaned_fakenews_dataset.csv')
print(f"✅ Rows in cleaned dataset: {len(cleaned_df)}")

# Check if the cleaned dataset matches the expected row count
if len(cleaned_df) == total_expected_rows:
    print("\n🎉 All tweet ID rows included correctly!")
else:
    print("\n⚠️ Mismatch! Check for missing or malformed tweet_ids.")
