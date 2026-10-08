from types import SimpleNamespace

import pytest

from src.search import SearchService, label_filter
from tests.conftest import FakeDataset, FakeEmbedder, FakeIndexer, FakeRanker


def make_service(records, captions=False, ranker=None, embedder=None):
    return SearchService(
        embedder or FakeEmbedder(), FakeIndexer(records),
        FakeDataset(records, has_captions=captions, has_labels=not captions),
        ranker=ranker, candidate_k=50,
    )


def test_returns_vector_order_with_metadata(labeled_records):
    service = make_service(labeled_records)
    results = service.search("a cyclist", top_k=2)
    assert [r.id for r in results] == [labeled_records[0].id, labeled_records[1].id]
    assert results[0].labels == ["Bicycle", "Car"]
    assert results[0].score == results[0].vector_score == pytest.approx(0.9)


def test_ranker_is_skipped_for_uncaptioned_datasets(labeled_records):
    service = make_service(labeled_records, ranker=FakeRanker())
    assert service.ranker is None
    service.search("x", top_k=2)
    assert service.indexer.calls[0]["top_k"] == 2


def test_ranker_reorders_captioned_results(captioned_records):
    service = make_service(captioned_records, captions=True, ranker=FakeRanker())
    results = service.search("a cat", top_k=2)
    assert service.indexer.calls[0]["top_k"] == 50  # fetches a wider candidate pool
    assert [r.caption for r in results] == ["a cat", "a dog"]
    assert results[0].score == 10.0
    assert results[0].vector_score == pytest.approx(0.8)


def test_label_filter_requires_all_labels(labeled_records):
    assert label_filter([]) is None
    assert label_filter(["Car", "Bicycle"]) == {"$and": [{"labels": {"$in": ["Car"]}}, {"labels": {"$in": ["Bicycle"]}}]}
    service = make_service(labeled_records)
    service.search("x", labels=["Car"])
    assert service.indexer.calls[0]["filter"] == {"$and": [{"labels": {"$in": ["Car"]}}]}


@pytest.mark.parametrize("query", ["", "   ", None])
def test_blank_query_is_rejected(labeled_records, query):
    with pytest.raises(ValueError):
        make_service(labeled_records).search(query)


def test_embedding_failure_raises(labeled_records):
    service = make_service(labeled_records, embedder=FakeEmbedder(vector=None))
    with pytest.raises(RuntimeError):
        service.search("x")


def test_object_shaped_pinecone_response(labeled_records):
    record = labeled_records[0]
    match = SimpleNamespace(id=record.id, score=0.5, metadata={"path": record.path, "filename": "a.png", "labels": ["Car"]})
    service = make_service(labeled_records)
    service.indexer.search = lambda *a, **k: SimpleNamespace(matches=[match])
    results = service.search("x")
    assert results[0].id == record.id and results[0].labels == ["Car"]


def test_empty_index_returns_no_results(labeled_records):
    service = make_service(labeled_records)
    service.indexer.search = lambda *a, **k: {"matches": []}
    assert service.search("x") == []


def test_result_carries_source_url(labeled_records):
    labeled_records[0].extra["source_url"] = "https://example.org/a.png"
    results = make_service(labeled_records).search("x", top_k=1)
    assert results[0].source_url == "https://example.org/a.png"
