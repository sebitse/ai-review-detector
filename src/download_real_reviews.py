from pathlib import Path
import pandas as pd
from datasets import load_dataset

# Configuration
DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_CONFIG = "raw_review_All_Beauty"
SPLIT_NAME = "full"
MAX_SAMPLES = 3000
RANDOM_STATE = 42

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "real_reviews.csv"

def main() -> None:
    # Ensure the target directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # load the dataset from Hugging Face
    print(f"Loading dataset: {DATASET_NAME} / {DATASET_CONFIG}...")

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT_NAME, trust_remote_code=True)

    print(f"Successfully loaded {len(ds)} rows.")


    df = ds.to_pandas()

    # The Amazon 2023 dataset often uses 'text', but we check for common variations
    possible_text_cols = ["text", "review_text", "review_body", "content"]
    text_col = next((c for c in possible_text_cols if c in df.columns), None)

    # clean and transform data
    print(f"Cleaning data using column: '{text_col}'...")
    
    # select only the text column and rename it for consistency
    df = df[[text_col]].copy().rename(columns={text_col: "text"})

    # filter out nulls, convert to string, and remove whitespace
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()

    # keep only reviews with meaningful length (more than 20 characters)
    df = df[df["text"].str.len() > 20]

    # Remove duplicate entries
    df = df.drop_duplicates(subset=["text"])

    # sampling
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)
    
    # add metadata labels
    df = df.reset_index(drop=True)
    df["label"] = 0               # 0 represents 'real' in many classification tasks
    df["source"] = "amazon_real"

    # export to CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Saved {len(df)} rows to: {OUTPUT_FILE}")
    print(df.head())

if __name__ == "__main__":
    main()
    
