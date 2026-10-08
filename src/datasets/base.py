from dataclasses import dataclass, field
from typing import Optional

from src.ids import image_id, to_rel_posix


@dataclass
class ImageRecord:
    path: str                      # project-relative, forward slashes
    caption: Optional[str] = None  # free text used by the re-ranker, if the dataset has one
    labels: list = field(default_factory=list)  # ground-truth classes, if the dataset has them
    extra: dict = field(default_factory=dict)   # dataset-specific metadata (scene, frame...)

    @property
    def id(self):
        return image_id(self.path)

    @property
    def filename(self):
        return self.path.rsplit("/", 1)[-1]


class Dataset:
    """Adapter interface: every dataset exposes its images as ImageRecords."""

    name = "base"
    has_captions = False
    has_labels = False

    def records(self):
        raise NotImplementedError

    def by_id(self):
        return {r.id: r for r in self.records()}


def make_record(path, **kwargs):
    return ImageRecord(path=to_rel_posix(path), **kwargs)
