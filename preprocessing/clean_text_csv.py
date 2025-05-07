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

# Text cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in STOPWORDS]
    return ' '.join(tokens)

# Extract timestamp from tweet ID using Twitter snowflake formula
def extract_timestamp(tweet_id):
    try:
        tweet_id = int(tweet_id)
        timestamp_ms = (tweet_id >> 22) + 1288834974657
        return pd.to_datetime(timestamp_ms, unit='ms')
    except:
        return pd.NaT

# Updated function to load and clean each dataset
def load_and_clean_csv(path, label):
    df = pd.read_csv(path)

    # Expand multiple tweet IDs into individual rows
    df['tweet_ids'] = df['tweet_ids'].astype(str)
    df['tweet_ids'] = df['tweet_ids'].str.strip()
    df = df[df['tweet_ids'].str.len() > 0]  # Remove rows with no tweet_ids

    rows = []

    for _, row in df.iterrows():
        tweet_ids = row['tweet_ids'].split()
        for tid in tweet_ids:
            rows.append({
                'id': row['id'],
                'tweetid': tid,
                'timestamp': extract_timestamp(tid),
                'tweetsource': row['news_url'],
                'title': row['title'],
                'clean_title': clean_text(row['title']),
                'label': label
            })

    return pd.DataFrame(rows)
