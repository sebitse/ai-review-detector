import pandas as pd
import random
import time
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm 


# Configuration
INPUT_FILE = Path("data/raw/real_reviews.csv")
OUTPUT_FILE = Path("data/raw/ai_reviews.csv")

N_SAMPLES = 1500
RANDOM_STATE = 42

generator = pipeline("text-generation", model="google/gemma-2-2b-it", device_map="auto")

# Variations of prompts to ensure the AI doesn't get repetitive
PROMPTS = [
    "Rewrite this review naturally: {review_text}",
    "Write a realistic product review based on this: {review_text}",
    "Create a human-like e-commerce review using this context: {review_text}"
]

def generate_ai_review(real_text: str) -> str:
    """Sends a prompt to the LLM and returns the cleaned result."""
    prompt_template = random.choice(PROMPTS)
    full_prompt = prompt_template.format(review_text=real_text)
    
    output = generator(
        full_prompt, 
        max_new_tokens=120, 
        do_sample=True, 
        temperature=0.8
    )
    
    # extract only the newly generated text (exclude the prompt)
    raw_text = output[0]["generated_text"][len(full_prompt):]
    
    # clean up
    return " ".join(raw_text.split()).strip()


def main():
    random.seed(RANDOM_STATE)

    if not INPUT_FILE.exists():
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # load and prepare the real reviews
    df_real = pd.read_csv(INPUT_FILE).dropna(subset=["text"])
    df_real = df_real[df_real["text"].str.len() > 20]
    
    # select samples
    num_to_sample = min(len(df_real), N_SAMPLES)
    samples = df_real.sample(n=num_to_sample, random_state=RANDOM_STATE)

    print(f"Starting generation for {num_to_sample} reviews...")
    ai_results = []

    # generation Loop
    for _, row in tqdm(samples.iterrows(), total=num_to_sample):
        try:
            ai_text = generate_ai_review(row["text"])
            
            if len(ai_text) > 20:
                ai_results.append({
                    "text": ai_text,
                    "label": 1,           # 1 represents 'AI'
                    "source": "gemma_2",
                    "original_reference": row["text"]
                })
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Skipping a row due to error: {e}")

    # save
    df_ai = pd.DataFrame(ai_results).drop_duplicates(subset=["text"])
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_ai.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Done! Saved {len(df_ai)} AI reviews to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
