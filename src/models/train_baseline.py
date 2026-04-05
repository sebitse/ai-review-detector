import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


INPUT_FILE = "data/raw/combined_reviews.csv"
RANDOM_STATE = 42


def extract_surface_features(text: str) -> dict:
    text = str(text).strip()

    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    num_chars = len(text)
    num_words = len(words)
    num_sentences = len(sentences)

    avg_word_length = (
        sum(len(w) for w in words) / num_words if num_words > 0 else 0.0
    )

    avg_sentence_length = (
        num_words / num_sentences if num_sentences > 0 else 0.0
    )

    unique_words = len(set(w.lower() for w in words))
    type_token_ratio = (
        unique_words / num_words if num_words > 0 else 0.0
    )

    num_exclamations = text.count("!")
    num_questions = text.count("?")
    num_digits = sum(ch.isdigit() for ch in text)

    return {
        "num_chars": num_chars,
        "num_words": num_words,
        "num_sentences": num_sentences,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "unique_words": unique_words,
        "type_token_ratio": type_token_ratio,
        "num_exclamations": num_exclamations,
        "num_questions": num_questions,
        "num_digits": num_digits,
    }


def main():
    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)

    feature_rows = df["text"].apply(extract_surface_features)
    X = pd.DataFrame(feature_rows.tolist())
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Surface Features Baseline - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", ascending=False)

    print("\nFeature importance (Logistic Regression coefficients):\n")
    print(coef_df)


if __name__ == "__main__":
    main()