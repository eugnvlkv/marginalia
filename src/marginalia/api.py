"""HTTP API over the book index for crypto-analyst: GET /search, GET /section, POST /ask.

Run from the project root (--env-file loads ANTHROPIC_API_KEY for /ask):
    HF_HUB_OFFLINE=1 uv run --env-file .env uvicorn marginalia.api:app --reload
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import anthropic
import chromadb
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from marginalia.agent import build_agent, run_agent, tool_calls
from marginalia.answer import cost_usd
from marginalia.embeddings import Embedder
from marginalia.index import chunk_paths, get_section, open_collection, search

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


class SectionResponse(BaseModel):
    book_id: str
    heading_path: list[str]
    text: str
    chunk_ids: list[str]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="Question in any language")


class Source(BaseModel):
    chunk_id: str
    book_id: str
    heading_path: list[str]


class ToolCall(BaseModel):
    name: str
    input: dict


class AskResponse(BaseModel):
    question: str
    answer: str
    not_in_books: bool
    sources: list[Source]
    tool_calls: list[ToolCall]
    model_calls: int
    cost_usd: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: the model takes seconds to load, a request must take milliseconds.
    from marginalia.embeddings import E5Embedder

    app.state.collection = open_collection(CHROMA_PATH)
    app.state.embedder = E5Embedder()
    # /ask needs Claude. Without a key the server still starts: /search and /section keep working.
    app.state.agent = (
        build_agent(anthropic.Anthropic(), app.state.collection, app.state.embedder)
        if os.environ.get("ANTHROPIC_API_KEY")
        else None
    )
    yield  # the server works here; code after yield would run at shutdown


app = FastAPI(title="marginalia", lifespan=lifespan)


# Endpoints get these through Depends, so tests can swap them for a fake embedder and a temp DB.
def get_collection(request: Request) -> chromadb.Collection:
    return request.app.state.collection


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_agent(request: Request) -> CompiledStateGraph:
    agent = getattr(request.app.state, "agent", None)  # None if lifespan found no key (or did not run)
    if agent is None:
        raise HTTPException(status_code=503, detail="Claude is not configured: start the server with --env-file .env")
    return agent


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


@app.get("/section")
def read_section(
    book: Annotated[str, Query(description="book_id from a search hit")],
    collection: Annotated[chromadb.Collection, Depends(get_collection)],
    path: Annotated[list[str] | None, Query(description="heading_path from a search hit, one ?path= per title")] = None,
) -> SectionResponse:
    heading_path = tuple(path or ())
    found = get_section(collection, book, heading_path)
    if found is None:
        # Echo what arrived: a wrong path is much easier to spot this way.
        raise HTTPException(
            status_code=404,
            detail=f"section not found: book={book!r}, path={list(heading_path)!r}",
        )
    return SectionResponse(
        book_id=found.book_id,
        heading_path=list(found.heading_path),
        text=found.text,
        chunk_ids=list(found.chunk_ids),
    )


@app.post("/ask")
def ask(
    body: AskRequest,
    collection: Annotated[chromadb.Collection, Depends(get_collection)],
    agent: Annotated[CompiledStateGraph, Depends(get_agent)],
) -> AskResponse:
    try:
        state = run_agent(agent, body.question)
    except RuntimeError as error:  # no answer within the step limit, or no structured answer
        raise HTTPException(status_code=502, detail=str(error)) from None
    except anthropic.APIError as error:  # Claude API: rate limit, overload, bad key ...
        raise HTTPException(status_code=502, detail=f"Claude API error: {error}") from None

    answer = state["answer"]
    known = chunk_paths(collection, answer.sources)  # ids the model cited that really exist
    return AskResponse(
        question=body.question,
        answer=answer.answer,
        not_in_books=answer.not_in_books,
        sources=[
            Source(chunk_id=chunk_id, book_id=known[chunk_id][0], heading_path=list(known[chunk_id][1]))
            for chunk_id in answer.sources
            if chunk_id in known
        ],
        tool_calls=[ToolCall(name=name, input=args) for name, args in tool_calls(state)],
        model_calls=len(state["usage"]),
        cost_usd=round(sum(cost_usd(u) for u in state["usage"]), 6),
    )
