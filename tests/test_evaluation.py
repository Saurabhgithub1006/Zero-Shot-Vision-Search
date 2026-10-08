import pytest

from src.datasets.base import ImageRecord
from src.evaluation import base_rate, caption_metrics, precision_at_k, rank_of


def test_rank_of():
    assert rank_of("b", ["a", "b", "c"]) == 2
    assert rank_of("z", ["a"]) is None


def test_caption_metrics():
    m = caption_metrics([1, 3, None, 10], sample_size=4)
    assert m["Recall@1"] == 0.25
    assert m["Recall@5"] == 0.5
    assert m["Recall@10"] == 0.75
    assert m["MRR"] == pytest.approx((1 + 1 / 3 + 1 / 10) / 4)


def test_caption_metrics_empty():
    assert caption_metrics([], 0) == {}


def test_precision_at_k():
    labels = [["Car", "Bicycle"], ["Car"], ["Bicycle"], ["Truck"]]
    assert precision_at_k(labels, "Bicycle", 2) == 0.5
    assert precision_at_k(labels, "Bicycle", 10) == 0.5  # fewer results than k: divide by what came back
    assert precision_at_k([], "Car", 10) == 0.0


def test_base_rate():
    records = [ImageRecord(path="a", labels=["Car"]), ImageRecord(path="b", labels=[])]
    assert base_rate(records, "Car") == 0.5
    assert base_rate([], "Car") == 0.0
