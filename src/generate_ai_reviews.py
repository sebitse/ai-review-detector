import pandas as pd
import random
import time
import re
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm
import torch

# Configuration
INPUT_FILE = Path("data/raw/real_reviews.csv")
OUTPUT_FILE = Path("data/raw/ai_reviews.csv")
REJECTED_FILE = Path("data/raw/rejected_ai_reviews.csv")

N_SAMPLES = 3000
RANDOM_STATE = 42
SAVE_EVERY = 10
SLEEP_SECONDS = 0.1
MAX_RETRIES_PER_REVIEW = 3

MODEL_NAME = "google/gemma-2-2b-it"


LOCAL_MODEL_PATH = r"../models/gemma-2-2b-it"

generator = pipeline(
    "text-generation",
    model=LOCAL_MODEL_PATH,
    tokenizer=LOCAL_MODEL_PATH,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.bfloat16}
)

generator.model.generation_config.max_length = None
generator.model.generation_config.max_new_tokens = 110

PROMPT_TEMPLATE = (
    "Write a short and natural customer review in English for an online product.\n"
    "Make it sound like a real person wrote it quickly.\n"
    "Do not make it overly polished or too formal.\n"
    "Return only the review text.\n"
    "Do not include titles, labels, notes, explanations, markdown, placeholders, quotation marks, or bullet points.\n"
    "Do not write words such as Review, Final review, Critique, Product Name, Brand Name, Reference, Overall, or Rating.\n"
    "It must contain 2 to 4 sentences, include at least one positive aspect and one mild negative aspect, "
    "and sound like something a real customer would post online.\n"
    "Do not copy the reference text.\n"
    "Output only the review itself.\n\n"
    "Reference review:\n{review_text}"
)

BAD_PATTERNS = [
    r"\*\*",
    r"\[.*?\]",
    r"\breview\b",
    r"\bfinal review\b",
    r"\bcritique\b",
    r"\brevised\b",
    r"\boriginal\b",
    r"\breference\b",
    r"\bproduct name\b",
    r"\bbrand name\b",
    r"\byour name\b",
    r"\bhere is\b",
    r"\bhere'?s\b",
    r"\bplease provide\b",
    r"\bwrite a\b",
    r"\bthis is a review\b",
    r"\brating:\b",
    r"\boverall:\b",
    r"\bdate:\b",
    r"\bsummary:\b",
    r"\bexplanation:\b",
    r"<br\s*/?>",
]

BAD_STARTS = (
    "and ", "but ", "so ", "or ", "across ", "because ", "then ", "also ",
    ".", ",", ";", ":", "-", '"', "'"
)

BAD_ENDS = (
    " and", " but", " or", " because", " although", " however",
    "more", "less", "better", "worse", "with", "for", "to", "of", "the", "a", "an"
)


# Cleaning and validation
def clean_generated_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())

    # strip surrounding quotes/spaces
    text = text.strip(' "\'')

    prefixes = [
        "Review:",
        "Final review:",
        "Revised review:",
        "Here is the review:",
        "Here's the review:",
        "This is a review.",
        "Customer review:",
        "Review text:",
    ]

    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip(' "\':-')

    # remove weird leading punctuation repeatedly
    while text and text[0] in '.,"\'-:;':
        text = text[1:].strip()

    return text


def count_sentences(text: str) -> int:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def looks_like_english_basic(text: str) -> bool:
    non_ascii_count = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii_count <= 5


def is_truncated(text: str) -> bool:
    t = text.strip()
    tl = t.lower()

    if not t:
        return True

    # must end like a normal review
    if t[-1] not in ".!?":
        return True

    for bad_end in BAD_ENDS:
        if tl.endswith(bad_end):
            return True

    return False


def is_too_generic(text: str) -> bool:
    tl = text.lower()

    generic_phrases = [
        "this product is amazing",
        "i would definitely recommend it",
        "great value for the money",
        "works exactly as advertised",
        "i am very happy with this product",
    ]

    hits = sum(1 for phrase in generic_phrases if phrase in tl)
    return hits >= 2


def is_valid_ai_review(text: str) -> bool:
    t = text.strip()
    tl = t.lower()

    if len(t) < 80 or len(t) > 700:
        return False

    if tl.startswith(BAD_STARTS):
        return False

    for pattern in BAD_PATTERNS:
        if re.search(pattern, tl):
            return False

    if not looks_like_english_basic(t):
        return False

    if is_truncated(t):
        return False

    num_sentences = count_sentences(t)
    if num_sentences < 2 or num_sentences > 6:
        return False

    words = re.findall(r"\b\w+\b", tl)
    if len(words) < 20:
        return False

    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.45:
        return False

    if is_too_generic(t):
        return False

    return True


# Generation
def build_prompt(real_text: str) -> str:
    return PROMPT_TEMPLATE.format(review_text=real_text)


def generate_one_candidate(real_text: str) -> str:
    prompt = build_prompt(real_text)

    output = generator(
        prompt,
        max_new_tokens=110,
        do_sample=True,
        temperature=0.65,
        top_p=0.9,
        return_full_text=False
    )

    raw_text = output[0]["generated_text"]
    return clean_generated_text(raw_text)


def generate_ai_review(real_text: str) -> tuple[str | None, str | None]:
    """
    Returns:
        (valid_text, rejected_text)
    """
    for _ in range(MAX_RETRIES_PER_REVIEW):
        candidate = generate_one_candidate(real_text)

        if is_valid_ai_review(candidate):
            return candidate, None

    return None, candidate


# Saving
def save_progress(results: list[dict], output_file: Path) -> None:
    if not results:
        return

    df_ai = pd.DataFrame(results).drop_duplicates(subset=["text"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_ai.to_csv(output_file, index=False, encoding="utf-8")


def save_rejected(rejected: list[dict], rejected_file: Path) -> None:
    if not rejected:
        return

    df_rej = pd.DataFrame(rejected)
    rejected_file.parent.mkdir(parents=True, exist_ok=True)
    df_rej.to_csv(rejected_file, index=False, encoding="utf-8")


# Main
def main():
    random.seed(RANDOM_STATE)

    if not INPUT_FILE.exists():
        print(f"Error: Could not find {INPUT_FILE}")
        return

    df_real = pd.read_csv(INPUT_FILE).dropna(subset=["text"]).copy()
    df_real["text"] = df_real["text"].astype(str).str.strip()
    df_real = df_real[df_real["text"].str.len() > 20]
    df_real = df_real.drop_duplicates(subset=["text"]).reset_index(drop=True)

    num_to_sample = min(len(df_real), N_SAMPLES)
    samples = df_real.sample(n=num_to_sample, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Starting generation for up to {num_to_sample} source reviews...")

    ai_results = []
    rejected_results = []
    existing_texts = set()

    if OUTPUT_FILE.exists():
        existing_df = pd.read_csv(OUTPUT_FILE)
        if "text" in existing_df.columns:
            ai_results = existing_df.to_dict(orient="records")
            existing_texts = set(existing_df["text"].astype(str).tolist())
            print(f"Loaded existing progress: {len(ai_results)} valid reviews already saved.")

    generated_count = len(ai_results)

    for _, row in tqdm(samples.iterrows(), total=num_to_sample):
        if generated_count >= N_SAMPLES:
            break

        real_text = row["text"]

        try:
            ai_text, rejected_text = generate_ai_review(real_text)

            if ai_text is not None and ai_text not in existing_texts:
                ai_results.append({
                    "text": ai_text,
                    "label": 1,
                    "source": "gemma_2",
                    "original_reference": real_text
                })
                existing_texts.add(ai_text)
                generated_count += 1

                if generated_count % SAVE_EVERY == 0:
                    save_progress(ai_results, OUTPUT_FILE)
                    save_rejected(rejected_results, REJECTED_FILE)
                    print(f"Saved progress at {generated_count} valid generated reviews.")

            elif rejected_text is not None:
                rejected_results.append({
                    "rejected_text": rejected_text,
                    "source": "gemma_2",
                    "original_reference": real_text
                })

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"Skipping a row due to error: {e}")

    save_progress(ai_results, OUTPUT_FILE)
    save_rejected(rejected_results, REJECTED_FILE)

    final_count = len(pd.DataFrame(ai_results).drop_duplicates(subset=["text"]))
    print(f"Done! Saved {final_count} valid AI reviews to {OUTPUT_FILE}")
    print(f"Rejected samples saved to {REJECTED_FILE}")


if __name__ == "__main__":
    main()



