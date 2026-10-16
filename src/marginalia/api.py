"""HTTP API over the book index: GET /search for crypto-analyst.

Run from the project root:
    HF_HUB_OFFLINE=1 uv run uvicorn marginalia.api:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import chromadb
from fastapi import Depends, FastAPI, Query, Request
from pydantic import BaseModel

from marginalia.embeddings import Embedder
from marginalia.index import open_collection, search

CHROMA_PATH = Path("data/chroma")


class SearchHit(BaseModel):
    chunk_id: str
    book_id: str
    heading_path: list[str]
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: the model takes seconds to load, a request must take milliseconds.
    from marginalia.embeddings import E5Embedder

    app.state.collection = open_collection(CHROMA_PATH)
    app.state.embedder = E5Embedder()
    yield  # the server works here; code after yield would run at shutdown


app = FastAPI(title="marginalia", lifespan=lifespan)


# Endpoints get these through Depends, so tests can swap them for a fake embedder and a temp DB.
def get_collection(request: Request) -> chromadb.Collection:
    return request.app.state.collection


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


@app.get("/health")
def health(
    collection: Annotated[chromadb.Collection, Depends(get_collection)],
) -> dict[str, str | int]:
    return {"status": "ok", "chunks": collection.count()}


@app.get("/search")
def search_books(
    q: Annotated[str, Query(min_length=1, description="Question in any language")],
    collection: Annotated[chromadb.Collection, Depends(get_collection)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    k: Annotated[int, Query(ge=1, le=20, description="How many chunks to return")] = 5,
    book: Annotated[str | None, Query(description="Search only this book_id")] = None,
) -> SearchResponse:
    hits = search(collection, embedder, q, k=k, book_id=book)
    return SearchResponse(
        query=q,
        hits=[
            SearchHit(
                chunk_id=h.chunk_id,
                book_id=h.book_id,
                heading_path=list(h.heading_path),
                text=h.text,
                score=h.score,
            )
            for h in hits
        ],
    )
