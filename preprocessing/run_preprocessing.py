from preprocessing.clean_text_csv import load_and_clean_csv
import pandas as pd
import os

# Define dataset file names and their corresponding labels
datasets = {
    'politifact_fake.csv': 0,
    'politifact_real.csv': 1,
    'gossipcop_fake.csv': 0,
    'gossipcop_real.csv': 1
}

# Folder where your raw datasets are stored
dataset_dir = 'dataset'

# List to store processed DataFrames
cleaned_dfs = []

# Process each dataset
for file, label in datasets.items():
    file_path = os.path.join(dataset_dir, file)
    print(f"🔄 Processing {file}...")

    # Load and clean the dataset
    df = load_and_clean_csv(file_path, label)

    # Print progress after processing each dataset
    print(f"✅ Finished processing {file}. Rows: {len(df)}")
    cleaned_dfs.append(df)

# Combine all datasets, shuffle, and save
all_data = pd.concat(cleaned_dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)

# Save cleaned data to a CSV file
output_path = 'C:/Users/deeks/Desktop/cleaned_fakenews_dataset.csv'
all_data.to_csv(output_path, index=False)

# Final output message
print(f"🎉 Cleaned dataset saved to: {output_path}")
