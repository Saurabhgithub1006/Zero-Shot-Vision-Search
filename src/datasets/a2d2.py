# A2D2 (Audi Autonomous Driving Dataset, CC BY-ND 4.0) adapter.
# Front-center camera frames; ground-truth classes are derived from the semantic label masks.
import csv
import json
import os
import re

import numpy as np
from PIL import Image

from src.config import project_path
from src.datasets.base import Dataset, make_record

DATA_DIR = project_path("assets", "a2d2")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")
CLASS_LIST_PATH = os.path.join(DATA_DIR, "class_list.json")

# Official public bucket: images can be shown from the source instead of being re-hosted
SOURCE_BASE_URL = "https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic"

# Ignore tiny label fragments so a class only counts when it is actually visible
MIN_CLASS_PIXELS = 1500


def load_color_map(class_list_path=CLASS_LIST_PATH):
    """Packed RGB int -> class group ("Car 1".."Car 4" all become "Car")."""
    with open(class_list_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(hex_color[1:], 16): re.sub(r"\s+\d+$", "", name) for hex_color, name in raw.items()}


def classes_in_label(label_rgb, color_map, min_pixels=MIN_CLASS_PIXELS):
    """Sorted class groups covering at least min_pixels in an HxWx3 label mask."""
    rgb = label_rgb.astype(np.int32)
    packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    colors, counts = np.unique(packed, return_counts=True)

    totals = {}
    for color, count in zip(colors.tolist(), counts.tolist()):
        name = color_map.get(color)
        if name:
            totals[name] = totals.get(name, 0) + count
    return sorted(name for name, total in totals.items() if total >= min_pixels)


def label_classes_from_file(label_path, color_map, min_pixels=MIN_CLASS_PIXELS):
    with Image.open(label_path) as img:
        return classes_in_label(np.asarray(img.convert("RGB")), color_map, min_pixels)


def source_url(scene, filename):
    return f"{SOURCE_BASE_URL}/{scene}/camera/cam_front_center/{filename}"


def write_manifest(rows, manifest_path=MANIFEST_PATH):
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "image_path", "labels"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "labels": ";".join(row["labels"])})


class A2D2Dataset(Dataset):
    name = "a2d2"
    has_labels = True

    def __init__(self, manifest_path=MANIFEST_PATH):
        self.manifest_path = manifest_path
        self._records = None

    def records(self):
        if self._records is None:
            self._records = []
            if os.path.exists(self.manifest_path):
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        labels = [l for l in row["labels"].split(";") if l]
                        filename = row["image_path"].rsplit("/", 1)[-1]
                        extra = {"scene": row["scene"], "source_url": source_url(row["scene"], filename)}
                        self._records.append(make_record(row["image_path"], labels=labels, extra=extra))
        return self._records
