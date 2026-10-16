"""Shared test helpers. pytest loads this file by itself and gives fixtures to tests by name."""

from pathlib import Path

import pytest

from marginalia.index import ingest_book, open_collection

VOCAB = ["trend", "risk", "candle"]


class FakeEmbedder:
    """Counts vocabulary words: fast and predictable, no model to load.

    Every test text contains at least one vocabulary word, otherwise its
    vector would be all zeros and cosine would be undefined.
    """

    def _vector(self, text: str) -> list[float]:
        words = text.lower().replace(".", " ").split()
        return [float(words.count(word)) for word in VOCAB]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


# A preface and three sections: 4 chunks in total.
BOOK = """A short book about trend and risk.

# Trends

A trend is your friend until the trend ends.

# Risk

Never risk more than you can afford to risk.

# Candles

A hammer candle is a reversal candle.
"""


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def collection(tmp_path: Path):
    """An empty Chroma collection in a fresh temporary folder."""
    return open_collection(tmp_path)


@pytest.fixture
def book_collection(collection, embedder):
    """The collection with BOOK ingested as book "b"."""
    ingest_book(collection, "b", BOOK, embedder)
    return collection
