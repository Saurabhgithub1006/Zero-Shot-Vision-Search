import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.base import Dataset, ImageRecord


class FakeDataset(Dataset):
    def __init__(self, records, has_captions=False, has_labels=True, name="fake"):
        self._records = records
        self.has_captions = has_captions
        self.has_labels = has_labels
        self.name = name

    def records(self):
        return self._records


class FakeEmbedder:
    def __init__(self, vector=(0.1, 0.2)):
        self.vector = list(vector) if vector is not None else None
        self.queries = []

    def get_text_embedding(self, text):
        self.queries.append(text)
        return self.vector


class FakeIndexer:
    """Returns the given records as matches with descending scores, dict-shaped like Pinecone."""

    def __init__(self, records):
        self.records = records
        self.calls = []

    def search(self, vector, top_k=5, metadata_filter=None):
        self.calls.append({"top_k": top_k, "filter": metadata_filter})
        matches = [
            {"id": r.id, "score": 0.9 - i * 0.1,
             "metadata": {"path": r.path, "filename": r.filename, "labels": r.labels}}
            for i, r in enumerate(self.records[:top_k])
        ]
        return {"matches": matches}


class FakeRanker:
    """Reverses the candidate order so tests can see that re-ranking happened."""

    def rank(self, query, candidates, top_k=12):
        ranked = list(reversed(candidates))
        for i, c in enumerate(ranked):
            c["score"] = 10.0 - i
        return ranked[:top_k]


@pytest.fixture
def labeled_records():
    return [
        ImageRecord(path="assets/a2d2/camera/s1/a.png", labels=["Bicycle", "Car"], extra={"scene": "s1"}),
        ImageRecord(path="assets/a2d2/camera/s1/b.png", labels=["Car"], extra={"scene": "s1"}),
        ImageRecord(path="assets/a2d2/camera/s2/c.png", labels=["Pedestrian"], extra={"scene": "s2"}),
    ]


@pytest.fixture
def captioned_records():
    return [
        ImageRecord(path="assets/image-dataset/x.jpg", caption="a dog"),
        ImageRecord(path="assets/image-dataset/y.jpg", caption="a cat"),
    ]
