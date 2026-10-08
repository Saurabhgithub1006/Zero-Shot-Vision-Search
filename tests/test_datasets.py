import json

import numpy as np
import pytest
from PIL import Image

from src.datasets import get_dataset
from src.datasets.a2d2 import (
    A2D2Dataset, classes_in_label, label_classes_from_file, load_color_map, write_manifest,
)
from src.datasets.unsplash import UnsplashDataset

COLOR_MAP = {0xFF0000: "Car", 0xC80000: "Car", 0xB65906: "Bicycle", 0x87CEFF: "Sky"}


def test_color_map_groups_numbered_variants(tmp_path):
    path = tmp_path / "class_list.json"
    path.write_text(json.dumps({"#ff0000": "Car 1", "#c80000": "Car 2", "#87ceff": "Sky", "#9f79ee": "Traffic guide obj."}))
    cmap = load_color_map(str(path))
    assert cmap[0xFF0000] == cmap[0xC80000] == "Car"
    assert cmap[0x87CEFF] == "Sky"
    assert cmap[0x9F79EE] == "Traffic guide obj."


def test_classes_in_label_sums_variants_and_drops_tiny_regions():
    label = np.zeros((100, 100, 3), dtype=np.uint8)
    label[:30, :] = (0x87, 0xCE, 0xFF)       # sky: 3000 px
    label[30:40, :] = (0xFF, 0x00, 0x00)     # car 1: 1000 px
    label[40:45, :] = (0xC8, 0x00, 0x00)     # car 2: 500 px -> car total 1500
    label[50, :10] = (0xB6, 0x59, 0x06)      # bicycle: 10 px, below threshold
    assert classes_in_label(label, COLOR_MAP, min_pixels=1500) == ["Car", "Sky"]


def test_unknown_colors_are_ignored():
    label = np.full((50, 50, 3), 7, dtype=np.uint8)
    assert classes_in_label(label, COLOR_MAP, min_pixels=1) == []


def test_label_file_roundtrip(tmp_path):
    label = np.zeros((40, 40, 3), dtype=np.uint8)
    label[:] = (0xFF, 0x00, 0x00)
    path = tmp_path / "label.png"
    Image.fromarray(label).save(path)
    assert label_classes_from_file(str(path), COLOR_MAP, min_pixels=100) == ["Car"]


def test_a2d2_manifest_to_records(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest([
        {"scene": "s1", "image_path": "assets/a2d2/camera/s1/a.png", "labels": ["Car", "Sky"]},
        {"scene": "s2", "image_path": "assets/a2d2/camera/s2/b.png", "labels": []},
    ], str(manifest))
    records = A2D2Dataset(str(manifest)).records()
    assert [r.labels for r in records] == [["Car", "Sky"], []]
    assert records[0].extra["scene"] == "s1"
    assert records[0].filename == "a.png"
    assert records[0].extra["source_url"] == (
        "https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com"
        "/camera_lidar_semantic/s1/camera/cam_front_center/a.png"
    )


def test_a2d2_missing_manifest_gives_no_records(tmp_path):
    assert A2D2Dataset(str(tmp_path / "missing.csv")).records() == []


def test_unsplash_attaches_captions(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for name in ("p1.jpg", "p2.png", "notes.txt"):
        (img_dir / name).write_bytes(b"x")
    tsv = tmp_path / "photos.csv000"
    tsv.write_text("photo_id\tphoto_description\tai_description\np1\thuman\tai text\np2\thuman only\t\n", encoding="utf-8")

    records = UnsplashDataset(str(img_dir), str(tsv)).records()
    captions = {r.filename: r.caption for r in records}
    assert captions == {"p1.jpg": "ai text", "p2.png": "human only"}


def test_registry():
    assert get_dataset("a2d2").name == "a2d2"
    assert get_dataset("unsplash").has_captions
    with pytest.raises(ValueError):
        get_dataset("nope")
