from pathlib import Path
import pandas as pd


REAL_FILE = Path("data/raw/real_reviews.csv")
AI_FILE = Path("data/raw/ai_reviews.csv")
OUTPUT_FILE = Path("data/raw/combined_reviews.csv")

# load the datasets into memory
df_real = pd.read_csv(REAL_FILE)
df_ai = pd.read_csv(AI_FILE)

# feature selection
df_real = df_real[["text", "label", "source"]].copy()
df_ai = df_ai[["text", "label", "source"]].copy()

df = pd.concat([df_real, df_ai], ignore_index=True)

# data cleaning & validation
df = df.dropna(subset=["text"])

df["text"] = df["text"].astype(str).str.strip()

df = df[df["text"].str.len() > 20]

df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# final export
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"Combined dataset saved to {OUTPUT_FILE}")
print(df["label"].value_counts())

