import nltk
import pandas as pd
import pytest

from ml_homework.text import (
    count_vocabulary_coverage,
    ensure_nltk_resources,
    filter_sentiment_tokens,
    fit_count_vectorizer,
)


def test_ensure_nltk_resources_downloads_only_missing_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked: list[str] = []
    downloaded: list[tuple[str, bool]] = []

    def fake_find(location: str) -> object:
        checked.append(location)
        if location == "tokenizers/punkt_tab":
            raise LookupError(location)
        return object()

    def fake_download(resource: str, *, raise_on_error: bool) -> bool:
        downloaded.append((resource, raise_on_error))
        return True

    monkeypatch.setattr(nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk, "download", fake_download)

    ensure_nltk_resources()

    assert checked == [
        "corpora/stopwords",
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
    ]
    assert downloaded == [("punkt_tab", True)]


def test_filter_sentiment_tokens_preserves_negation_and_removes_noise() -> None:
    tokens = ["i", "do", "n't", "love", "!", "a", "not", "good", "n’t", "123", "bad2"]
    stop_words = frozenset({"i", "do", "love", "a"})

    result = filter_sentiment_tokens(tokens, stop_words)

    assert result == ["not", "not", "good", "not"]
    assert tokens[2] == "n't"


def test_count_vocabulary_coverage_counts_preprocessed_tokens() -> None:
    texts = pd.Series(["good good bad", "good neutral", "bad neutral neutral"])

    result = count_vocabulary_coverage(texts, [2, 3])

    assert result["max_features"].tolist() == [2, 3]
    assert result["vocabulary"].tolist() == [2, 3]
    assert result["coverage_pct"].tolist() == pytest.approx([75.0, 100.0])
    assert texts.tolist() == [
        "good good bad",
        "good neutral",
        "bad neutral neutral",
    ]


def test_fit_count_vectorizer_excludes_held_out_words_from_vocabulary() -> None:
    train_texts = pd.Series(["good good", "bad"])
    test_texts = pd.Series(["good unseen"])

    vectorizer, train_counts, test_counts = fit_count_vectorizer(
        train_texts, test_texts, max_features=2
    )

    assert vectorizer.get_feature_names_out().tolist() == ["bad", "good"]
    assert train_counts.toarray().tolist() == [[0, 2], [1, 0]]
    assert test_counts.toarray().tolist() == [[0, 1]]
