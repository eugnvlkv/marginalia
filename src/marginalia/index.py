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


def parse_path(stored: str) -> tuple[str, ...]:
    """The heading path is stored as one " › "-joined string; this turns it back into a tuple."""
    return tuple(stored.split(PATH_SEP)) if stored else ()


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
        hits.append(
            Hit(
                chunk_id=chunk_id,
                book_id=meta["book_id"],
                heading_path=parse_path(meta["heading_path"]),
                text=text,
                score=1 - distance,  # Chroma returns cosine distance = 1 - similarity
            )
        )
    return hits


def chunk_paths(collection: chromadb.Collection, chunk_ids: list[str]) -> dict[str, tuple[str, tuple[str, ...]]]:
    """chunk id → (book_id, heading_path) for the ids that exist; unknown ids are simply absent."""
    if not chunk_ids:
        return {}
    result = collection.get(ids=list(chunk_ids), include=["metadatas"])
    return {
        chunk_id: (meta["book_id"], parse_path(meta["heading_path"]))
        for chunk_id, meta in zip(result["ids"], result["metadatas"])
    }


@dataclass(frozen=True)
class Section:
    book_id: str
    heading_path: tuple[str, ...]
    text: str
    chunk_ids: tuple[str, ...]


def get_section(
    collection: chromadb.Collection,
    book_id: str,
    heading_path: tuple[str, ...],
) -> Section | None:
    """The whole section back from its chunks: search finds a small piece, the agent reads all of it.

    Only chunks with exactly this path; subsections are separate sections.
    """
    # No vectors here: a plain metadata filter. Two conditions need $and in Chroma.
    result = collection.get(
        where={"$and": [{"book_id": book_id}, {"heading_path": PATH_SEP.join(heading_path)}]},
        include=["documents", "metadatas"],
    )
    if not result["ids"]:
        return None

    # Chroma does not promise any order. Tuples sort by their first element, so this is book order.
    rows = sorted(
        (meta["position"], chunk_id, text)
        for chunk_id, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
    )
    return Section(
        book_id=book_id,
        heading_path=heading_path,
        text="\n\n".join(text for _, _, text in rows),
        chunk_ids=tuple(chunk_id for _, chunk_id, _ in rows),
    )


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
