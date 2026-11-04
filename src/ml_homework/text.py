"""Helpers for experiments with preprocessed text."""

from collections.abc import Collection, Iterable

import nltk
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

SENTIMENT_NEGATION_WORDS: frozenset[str] = frozenset({"no", "nor", "not"})
_NEGATION_CONTRACTIONS: frozenset[str] = frozenset({"n't", "n’t"})


def ensure_nltk_resources() -> None:
    """Download the tokenization and stopword data only when absent locally."""
    for resource, location in (
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ):
        try:
            nltk.data.find(location)
        except LookupError:
            nltk.download(resource, raise_on_error=True)


def filter_sentiment_tokens(
    tokens: Iterable[str], stop_words: Collection[str]
) -> list[str]:
    """Keep useful words and normalize contracted negation to ``not``."""
    filtered = []
    for token in tokens:
        word = "not" if token in _NEGATION_CONTRACTIONS else token
        if word.isalpha() and len(word) >= 2 and word not in stop_words:
            filtered.append(word)
    return filtered


def count_vocabulary_coverage(
    texts: pd.Series, max_features_values: Iterable[int]
) -> pd.DataFrame:
    """Measure token coverage for candidate CountVectorizer vocabulary sizes.

    ``texts`` contains whitespace-separated tokens after preprocessing.
    The result is descriptive; select model parameters using training data only.
    """
    all_counts = CountVectorizer(analyzer=str.split).fit_transform(texts)
    total_tokens = int(all_counts.sum())
    rows: list[dict[str, int | float]] = []

    for max_features in max_features_values:
        counts = CountVectorizer(
            max_features=max_features, analyzer=str.split
        ).fit_transform(texts)
        rows.append(
            {
                "max_features": max_features,
                "vocabulary": counts.shape[1],
                "coverage_pct": 100 * int(counts.sum()) / total_tokens,
            }
        )

    return pd.DataFrame(rows, columns=["max_features", "vocabulary", "coverage_pct"])


def fit_count_vectorizer(
    train_texts: pd.Series, test_texts: pd.Series, *, max_features: int
) -> tuple[CountVectorizer, csr_matrix, csr_matrix]:
    """Fit a BoW vocabulary on train only, then transform held-out text."""
    vectorizer = CountVectorizer(max_features=max_features, analyzer=str.split)
    train_counts = vectorizer.fit_transform(train_texts)
    test_counts = vectorizer.transform(test_texts)
    return vectorizer, train_counts, test_counts
