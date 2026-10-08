import hashlib
import os

from src.config import PROJECT_ROOT
from src.ids import image_id, to_rel_posix


def test_rel_posix_uses_forward_slashes():
    assert to_rel_posix(os.path.join("assets", "image-dataset", "x.jpg")) == "assets/image-dataset/x.jpg"


def test_absolute_path_is_made_project_relative():
    abs_path = os.path.join(PROJECT_ROOT, "assets", "a2d2", "f.png")
    assert to_rel_posix(abs_path) == "assets/a2d2/f.png"


def test_id_is_identical_for_every_path_spelling():
    # Regression: ingest hashed 'assets\\image-dataset\\x.jpg' while evaluation hashed
    # 'assets/image-dataset\\x.jpg' on Windows, so no evaluation target ever matched.
    spellings = [
        os.path.join("assets", "image-dataset", "x.jpg"),
        os.path.join("assets/image-dataset", "x.jpg"),
        "assets/image-dataset/x.jpg",
        os.path.join(PROJECT_ROOT, "scripts", "..", "assets", "image-dataset", "x.jpg"),
    ]
    assert len({image_id(p) for p in spellings}) == 1


def test_path_outside_project_root_does_not_crash(tmp_path):
    outside = os.path.join(str(tmp_path), "img.jpg")
    assert to_rel_posix(outside).endswith("/img.jpg")
    assert image_id(outside) == image_id(outside)


def test_id_matches_upstream_scheme_on_posix_paths():
    # Existing POSIX-built indexes keep their ids
    assert image_id("assets/image-dataset/x.jpg") == hashlib.md5(b"assets/image-dataset/x.jpg").hexdigest()
