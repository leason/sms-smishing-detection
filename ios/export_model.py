#!/usr/bin/env python3
"""Export simplified model weights for iOS app.

Trains a TF-IDF (5k features) + Logistic Regression model on the combined
dataset and exports vocabulary, IDF weights, and coefficients as JSON.
"""

import json
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Ensure we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.paths import PROCESSED_MANUAL, PROCESSED_SYNTHETIC

def main():
    manual = pd.read_csv(PROCESSED_MANUAL)
    synthetic = pd.read_csv(PROCESSED_SYNTHETIC)

    manual_train, manual_test = train_test_split(
        manual, test_size=0.20, stratify=manual["label"], random_state=1
    )
    combined_train = pd.concat([manual_train, synthetic], ignore_index=True)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95,
            sublinear_tf=True, max_features=5000
        )),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=1
        ))
    ])
    model.fit(combined_train["text_clean"], combined_train["label"])

    # Evaluate
    y_pred = model.predict(manual_test["text_clean"])
    print("=== iOS Model Performance ===")
    print(classification_report(manual_test["label"], y_pred, digits=4))

    # Export weights
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    weights = {
        "model_info": {
            "description": "Simplified TF-IDF + LogisticRegression for iOS SMS classification",
            "max_features": 5000,
            "ngram_range": [1, 2],
            "classes": clf.classes_.tolist(),
            "training_data": "combined (manual + synthetic)",
        },
        "vocabulary": {word: int(idx) for word, idx in tfidf.vocabulary_.items()},
        "idf": tfidf.idf_.tolist(),
        "coefficients": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
    }

    output_path = os.path.join(os.path.dirname(__file__), "SMSShield", "Shared", "SMSFilterModel.json")
    with open(output_path, "w") as f:
        json.dump(weights, f)

    size = os.path.getsize(output_path)
    print(f"Exported to {output_path}: {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
