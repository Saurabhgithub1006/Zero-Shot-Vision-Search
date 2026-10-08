# Unsplash Lite adapter (the upstream dataset): images on disk + captions from the TSV.
import csv
import os

from src.config import project_path
from src.datasets.base import Dataset, make_record
from src.utils import get_image_paths

IMAGE_DIR = project_path("assets", "image-dataset")
CSV_PATH = project_path("assets", "unsplash-research-dataset-lite-latest", "photos.csv000")


def load_captions(csv_path):
    """photo_id -> best available description."""
    captions = {}
    if not os.path.exists(csv_path):
        return captions
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            desc = row.get("ai_description") or row.get("photo_description")
            if desc:
                captions[row["photo_id"]] = desc
    return captions


class UnsplashDataset(Dataset):
    name = "unsplash"
    has_captions = True

    def __init__(self, image_dir=IMAGE_DIR, csv_path=CSV_PATH):
        self.image_dir = image_dir
        self.csv_path = csv_path
        self._records = None

    def records(self):
        if self._records is None:
            captions = load_captions(self.csv_path)
            self._records = []
            for path in sorted(get_image_paths(self.image_dir)):
                photo_id = os.path.splitext(os.path.basename(path))[0]
                self._records.append(make_record(path, caption=captions.get(photo_id)))
        return self._records
