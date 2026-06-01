"""Feature engineering: TF-IDF + structured indicator features."""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

INDICATOR_COLS = ["has_url", "has_email", "has_phone"]


def build_feature_transformer() -> ColumnTransformer:
    """Build a ColumnTransformer that combines TF-IDF on text_clean
    with passthrough indicator features."""
    return ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words=None,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
                "text_clean",
            ),
            ("indicators", "passthrough", INDICATOR_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(classifier) -> Pipeline:
    """Wrap feature extraction + classifier into a single sklearn Pipeline."""
    return Pipeline(
        [
            ("features", build_feature_transformer()),
            ("clf", classifier),
        ]
    )
