"""Text → vector.

The model sits behind the small `Embedder` interface, so it can be swapped for
another model (or for a fast fake in tests) without touching the index code.
"""

from typing import Protocol

E5_SMALL = "intfloat/multilingual-e5-small"


class Embedder(Protocol):
    """Anything with these two methods is an Embedder (like a TS interface)."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class E5Embedder:
    """multilingual-e5 was trained with "passage: " / "query: " prefixes, so we add them."""

    def __init__(self, model_name: str = E5_SMALL) -> None:
        # Imported here, not at the top: loading torch takes seconds, and tests
        # that use a fake embedder should not pay for it.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            [f"passage: {text}" for text in texts],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(f"query: {text}", normalize_embeddings=True)
        return vector.tolist()
