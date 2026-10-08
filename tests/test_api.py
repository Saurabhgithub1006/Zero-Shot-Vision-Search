import pytest
from fastapi.testclient import TestClient

import api
from src.search import SearchService
from tests.conftest import FakeDataset, FakeEmbedder, FakeIndexer


@pytest.fixture
def service(labeled_records):
    return SearchService(FakeEmbedder(), FakeIndexer(labeled_records), FakeDataset(labeled_records, name="a2d2"))


@pytest.fixture
def client(service):
    api.app.dependency_overrides[api.get_service] = lambda: service
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "dataset": "a2d2", "images": 3}


def test_labels(client):
    assert client.get("/labels").json() == ["Bicycle", "Car", "Pedestrian"]


def test_search_returns_ranked_hits(client, labeled_records):
    resp = client.post("/search", json={"query": "a cyclist", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "a cyclist"
    assert body["count"] == 2
    first = body["results"][0]
    assert first["id"] == labeled_records[0].id
    assert first["labels"] == ["Bicycle", "Car"]
    assert first["image_url"] == f"/images/{labeled_records[0].id}"


def test_search_passes_label_filter(client, service):
    client.post("/search", json={"query": "x", "labels": ["Car"]})
    assert service.indexer.calls[-1]["filter"] == {"$and": [{"labels": {"$in": ["Car"]}}]}


@pytest.mark.parametrize("payload", [
    {},
    {"query": ""},
    {"query": "   "},
    {"query": "x", "top_k": 0},
    {"query": "x", "top_k": 51},
    {"query": "x" * 301},
])
def test_search_rejects_invalid_input(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_search_embedding_failure_is_500(client, service):
    service.embedder.vector = None
    assert client.post("/search", json={"query": "x"}).status_code == 500


def test_image_served_by_id(client, labeled_records, tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(api, "project_path", lambda *parts: str(img))
    resp = client.get(f"/images/{labeled_records[0].id}")
    assert resp.status_code == 200
    assert resp.content == b"\x89PNG fake"


def test_image_unknown_id_is_404(client):
    assert client.get("/images/does-not-exist").status_code == 404


def test_image_missing_file_is_404(client, labeled_records, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "project_path", lambda *parts: str(tmp_path / "gone.png"))
    assert client.get(f"/images/{labeled_records[0].id}").status_code == 404


def test_path_traversal_is_not_possible(client):
    assert client.get("/images/..%2F..%2F.env").status_code == 404
