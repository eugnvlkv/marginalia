"""Chroma index over book chunks: load a book in, search it."""

import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb

from marginalia.chunker import chunk_markdown
from marginalia.embeddings import Embedder
from marginalia.models import Chunk

COLLECTION = "books"
PATH_SEP = " › "


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    book_id: str
    heading_path: tuple[str, ...]
    text: str
    score: float  # cosine similarity: higher is closer


def open_collection(path: Path) -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(path),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    # embedding_function=None: we pass our own vectors, so Chroma must not load its default model.
    return client.get_or_create_collection(
        COLLECTION,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def passage_text(chunk: Chunk) -> str:
    """What the model sees: the heading path (the chapter title carries meaning) and the text."""
    path = PATH_SEP.join(chunk.heading_path)
    return f"{path}\n\n{chunk.text}" if path else chunk.text


def ingest_book(collection: chromadb.Collection, book_id: str, markdown: str, embedder: Embedder) -> int:
    """Replace everything stored for this book with fresh chunks. Returns the number of chunks."""
    chunks = chunk_markdown(markdown, book_id)
    # upsert updates by id but never removes: if the book now has fewer chunks,
    # the old tail would stay. So we drop the book first.
    collection.delete(where={"book_id": book_id})
    collection.upsert(
        ids=[c.id for c in chunks],
        embeddings=embedder.embed_passages([passage_text(c) for c in chunks]),
        documents=[c.text for c in chunks],
        # Chroma rejects empty lists in metadata, so the path is stored as one string.
        metadatas=[
            {"book_id": c.book_id, "heading_path": PATH_SEP.join(c.heading_path), "position": c.position}
            for c in chunks
        ],
    )
    return len(chunks)


def search(
    collection: chromadb.Collection,
    embedder: Embedder,
    query: str,
    k: int = 5,
    book_id: str | None = None,
) -> list[Hit]:
    result = collection.query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=k,
        where={"book_id": book_id} if book_id else None,
    )
    # query() takes a list of questions, so every field is a list of lists: [0] is our only question.
    hits = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        path = meta["heading_path"]
        hits.append(
            Hit(
                chunk_id=chunk_id,
                book_id=meta["book_id"],
                heading_path=tuple(path.split(PATH_SEP)) if path else (),
                text=text,
                score=1 - distance,  # Chroma returns cosine distance = 1 - similarity
            )
        )
    return hits


if __name__ == "__main__":
    from marginalia.embeddings import E5Embedder

    collection = open_collection(Path("data/chroma"))
    embedder = E5Embedder()
    command, arg = sys.argv[1], sys.argv[2]

    if command == "ingest":
        md = Path(arg)
        count = ingest_book(collection, md.stem, md.read_text(encoding="utf-8"), embedder)
        print(f"{md} -> {count} chunks in '{COLLECTION}'")
    elif command == "search":
        for hit in search(collection, embedder, arg):
            print(f"{hit.score:.3f}  {hit.chunk_id:15}  {PATH_SEP.join(hit.heading_path)}")
    else:
        sys.exit(f"unknown command: {command} (use ingest or search)")
