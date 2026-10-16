import pytest
from fastapi.testclient import TestClient

from marginalia.api import app, get_collection, get_embedder


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
