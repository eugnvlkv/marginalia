from pathlib import Path

from marginalia.index import ingest_book, open_collection, search

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


BOOK = """A short book about trend and risk.

# Trends

A trend is your friend until the trend ends.

# Risk

Never risk more than you can afford to risk.

# Candles

A hammer candle is a reversal candle.
"""


def test_search_finds_the_closest_chunk(tmp_path: Path):
    collection = open_collection(tmp_path)
    ingest_book(collection, "b", BOOK, FakeEmbedder())

    hits = search(collection, FakeEmbedder(), "risk", k=1)

    assert [h.heading_path for h in hits] == [("Risk",)]
    assert hits[0].text == "Never risk more than you can afford to risk."


def test_hits_are_sorted_by_score_and_limited_to_k(tmp_path: Path):
    collection = open_collection(tmp_path)
    ingest_book(collection, "b", BOOK, FakeEmbedder())

    hits = search(collection, FakeEmbedder(), "trend", k=2)

    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert hits[0].heading_path == ("Trends",)


def test_empty_heading_path_survives_the_round_trip(tmp_path: Path):
    # The preface has heading_path () — Chroma cannot store empty lists, we store a string.
    collection = open_collection(tmp_path)
    ingest_book(collection, "b", BOOK, FakeEmbedder())

    hits = search(collection, FakeEmbedder(), "trend", k=4)

    assert () in [h.heading_path for h in hits]


def test_reingest_replaces_old_chunks(tmp_path: Path):
    collection = open_collection(tmp_path)
    ingest_book(collection, "b", BOOK, FakeEmbedder())

    ingest_book(collection, "b", "# Risk\n\nOnly risk is left.\n", FakeEmbedder())

    assert collection.count() == 1


def test_search_can_filter_by_book(tmp_path: Path):
    collection = open_collection(tmp_path)
    ingest_book(collection, "one", "# A\n\nA trend in book one.\n", FakeEmbedder())
    ingest_book(collection, "two", "# A\n\nA trend in book two.\n", FakeEmbedder())

    hits = search(collection, FakeEmbedder(), "trend", k=5, book_id="two")

    assert [h.book_id for h in hits] == ["two"]
