import pytest
from fakes import ScriptedClient, final, response, tool_use
from fastapi.testclient import TestClient

from marginalia.agent import build_agent
from marginalia.answer import Answer
from marginalia.api import app, get_agent, get_collection, get_embedder


@pytest.fixture
def client(book_collection, embedder):
    # Swap the real model and DB for the fake embedder and a temp DB with BOOK (see conftest.py).
    # Without `with TestClient(app)` the startup code (lifespan) does not run, so e5 is never loaded.
    app.dependency_overrides[get_collection] = lambda: book_collection
    app.dependency_overrides[get_embedder] = lambda: embedder
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_reports_chunk_count(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks": 4}


def test_search_returns_hits_as_json(client):
    response = client.get("/search", params={"q": "risk", "k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "risk"
    assert [hit["heading_path"] for hit in body["hits"]] == [["Risk"]]
    assert body["hits"][0]["book_id"] == "b"


def test_search_rejects_bad_parameters(client):
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "risk", "k": 0}).status_code == 422
    assert client.get("/search").status_code == 422


def test_search_filters_by_book(client):
    response = client.get("/search", params={"q": "trend", "book": "other"})

    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_section_returns_the_section_text(client):
    response = client.get("/section", params={"book": "b", "path": ["Risk"]})

    assert response.status_code == 200
    assert response.json() == {
        "book_id": "b",
        "heading_path": ["Risk"],
        "text": "Never risk more than you can afford to risk.",
        "chunk_ids": ["b:2"],
    }


def test_section_without_path_is_the_preface(client):
    response = client.get("/section", params={"book": "b"})

    assert response.json()["text"] == "A short book about trend and risk."


def test_unknown_section_is_404(client):
    assert client.get("/section", params={"book": "b", "path": ["Nope"]}).status_code == 404


def use_agent(book_collection, embedder, responses):
    """Make /ask run the real agent graph, but with a scripted fake Claude."""
    app.dependency_overrides[get_agent] = lambda: build_agent(ScriptedClient(responses), book_collection, embedder)


def test_ask_returns_answer_with_sources_trace_and_cost(client, book_collection, embedder):
    answer = Answer(answer="Не больше 2%.", sources=["b:2", "made:up"], not_in_books=False)
    use_agent(book_collection, embedder, [response([tool_use("t1", "search_docs", {"query": "risk"})], "tool_use"), final(answer)])

    response_ = client.post("/ask", json={"question": "сколько рисковать?"})

    assert response_.status_code == 200
    assert response_.json() == {
        "question": "сколько рисковать?",
        "answer": "Не больше 2%.",
        "not_in_books": False,
        "sources": [{"chunk_id": "b:2", "book_id": "b", "heading_path": ["Risk"]}],  # made:up is dropped
        "tool_calls": [{"name": "search_docs", "input": {"query": "risk"}}],
        "model_calls": 2,
        "cost_usd": 0.0003,  # 2 calls x (100 in x $1 + 10 out x $5) / 1M
    }


def test_ask_validates_the_question(client, book_collection, embedder):
    # Dependencies are resolved too, so give /ask an agent; it is never called because validation fails first.
    use_agent(book_collection, embedder, [])

    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_ask_reports_an_agent_that_never_answers_as_502(client, book_collection, embedder):
    endless = [response([tool_use(f"t{i}", "search_docs", {"query": "risk"})], "tool_use") for i in range(10)]
    use_agent(book_collection, embedder, endless)

    response_ = client.post("/ask", json={"question": "q"})

    assert response_.status_code == 502
    assert "did not answer" in response_.json()["detail"]


def test_ask_without_claude_is_503(client):
    # No agent in app.state: what lifespan leaves when ANTHROPIC_API_KEY is missing.
    response_ = client.post("/ask", json={"question": "q"})

    assert response_.status_code == 503
    assert "--env-file .env" in response_.json()["detail"]

